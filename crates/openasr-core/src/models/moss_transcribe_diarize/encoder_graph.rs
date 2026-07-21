//! MOSS-Transcribe-Diarize's Whisper-Medium-style audio encoder: conv1
//! (kernel 3, stride 1, pad 1) -> GELU -> conv2 (kernel 3, stride 2, pad 1)
//! -> GELU -> + fixed positional embedding -> 24 standard pre-norm
//! Transformer encoder blocks (`crate::nn::encoder::transformer_layer`, the
//! same "Whisper / Qwen-audio encoder shape" primitive `qwen::audio_encoder`
//! already uses) -> final LayerNorm. Non-causal (full bidirectional
//! self-attention over all 1500 positions, an all-zero additive mask) --
//! this is standard upstream `transformers.WhisperEncoder`, verified against
//! the checkpoint's tensor names and shapes (`package_import`'s module doc).
//!
//! Whisper's own `k_proj` carries no bias (only q/v do); `transformer_layer`
//! requires a bias tensor for every projection, so this module supplies an
//! all-zero one for K -- adding zero is an exact no-op, not an approximation.
//!
//! This always runs a fixed-size 30s chunk (3000 mel frames -> 1500 encoder
//! output frames, `max_source_positions`); audio longer than 30s is chunked
//! by the executor (see `executor.rs`), which trims each chunk's output to
//! its own valid frame count before concatenating -- mirrors
//! `MossTranscribeDiarizeModel.get_audio_features`'s
//! `whisper_features[chunk_idx:chunk_idx+1, :token_len*4]` truncation.
//!
//! Deliberately skips the WEIGHTS-usage resident-arena optimization every
//! other family's encoder uses for GPU offload (see `qwen::audio_encoder`'s
//! module doc): correctness first, performance tuning is explicitly
//! out-of-scope for this stage (see `mod.rs`'s stage-status note). Every
//! weight is a genuine per-call graph input, which is correct on every
//! backend, just not the fastest.

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgufTensorDataReadError, GgufTensorDataReader,
};
use crate::nn::conv::{
    Conv1dParams, ConvActivation, ConvBlockSteps, apply_conv_1d_bias_activation,
};
use crate::nn::encoder::{
    TransformerEncoderConfig, TransformerEncoderLayerWeights, transformer_layer,
};

use super::tensor_names::{
    ENC_CONV1_BIAS, ENC_CONV1_WEIGHT, ENC_CONV2_BIAS, ENC_CONV2_WEIGHT, ENC_OUT_NORM_BIAS,
    ENC_OUT_NORM_WEIGHT, ENC_POS_EMBD_WEIGHT, moss_encoder_layer_tensor_names,
};

/// `nn.LayerNorm`'s default epsilon -- verified against upstream
/// `transformers.models.whisper.modeling_whisper.WhisperEncoderLayer`, which
/// never overrides it (same value every Whisper size uses).
const MOSS_ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const CONV_KERNEL_SIZE: usize = 3;
const CONV1_STRIDE: usize = 1;
const CONV2_STRIDE: usize = 2;
const CONV_PADDING: usize = 1;
const CONV_DILATION: usize = 1;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MossEncoderConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_mels: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Error)]
pub(crate) enum MossEncoderError {
    #[error("moss-transcribe-diarize encoder tensor read failed: {0}")]
    TensorRead(#[from] GgufTensorDataReadError),
    #[error("moss-transcribe-diarize encoder graph build failed at '{step}': {source}")]
    GraphBuild {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("moss-transcribe-diarize encoder graph execution failed: {reason}")]
    GraphExecution { reason: String },
    #[error(
        "moss-transcribe-diarize encoder mel input length {found} does not match expected {expected} for {n_mels} mels x {frames} frames"
    )]
    InvalidMelInputLength {
        found: usize,
        expected: usize,
        n_mels: usize,
        frames: usize,
    },
}

fn map_graph_error(step: &'static str, source: GgmlCpuGraphError) -> MossEncoderError {
    MossEncoderError::GraphBuild { step, source }
}

struct MossEncoderLayerWeights {
    attn_norm_weight: Vec<f32>,
    attn_norm_bias: Vec<f32>,
    attn_q_weight: Vec<f32>,
    attn_q_bias: Vec<f32>,
    attn_k_weight: Vec<f32>,
    attn_v_weight: Vec<f32>,
    attn_v_bias: Vec<f32>,
    attn_out_weight: Vec<f32>,
    attn_out_bias: Vec<f32>,
    ffn_norm_weight: Vec<f32>,
    ffn_norm_bias: Vec<f32>,
    ffn_up_weight: Vec<f32>,
    ffn_up_bias: Vec<f32>,
    ffn_down_weight: Vec<f32>,
    ffn_down_bias: Vec<f32>,
}

pub(crate) struct MossEncoderWeights {
    conv1_weight_f16_bits: Vec<u16>,
    conv1_bias: Vec<f32>,
    conv2_weight_f16_bits: Vec<u16>,
    conv2_bias: Vec<f32>,
    /// `[max_source_positions, d_model]` row-major (position-major,
    /// dim-minor -- matches HF `embed_positions.weight`'s own layout, so no
    /// transpose is needed before uploading as a `[d_model, positions]`
    /// (ne0=d_model) ggml tensor).
    pos_embd: Vec<f32>,
    layers: Vec<MossEncoderLayerWeights>,
    out_norm_weight: Vec<f32>,
    out_norm_bias: Vec<f32>,
}

pub(crate) fn load_moss_encoder_weights_from_reader(
    reader: &GgufTensorDataReader,
    config: MossEncoderConfig,
) -> Result<MossEncoderWeights, MossEncoderError> {
    let conv1_weight_f16_bits = reader.host_tensor_f16_bits_copy_by_name(
        ENC_CONV1_WEIGHT,
        &[
            CONV_KERNEL_SIZE as u64,
            config.n_mels as u64,
            config.d_model as u64,
        ],
    )?;
    let conv1_bias = reader
        .host_tensor_f32_copy_dequantized_by_name(ENC_CONV1_BIAS, &[config.d_model as u64])?;
    let conv2_weight_f16_bits = reader.host_tensor_f16_bits_copy_by_name(
        ENC_CONV2_WEIGHT,
        &[
            CONV_KERNEL_SIZE as u64,
            config.d_model as u64,
            config.d_model as u64,
        ],
    )?;
    let conv2_bias = reader
        .host_tensor_f32_copy_dequantized_by_name(ENC_CONV2_BIAS, &[config.d_model as u64])?;
    let pos_embd = reader.host_tensor_f32_copy_dequantized_by_name(
        ENC_POS_EMBD_WEIGHT,
        &[config.max_source_positions as u64, config.d_model as u64],
    )?;
    let out_norm_weight = reader
        .host_tensor_f32_copy_dequantized_by_name(ENC_OUT_NORM_WEIGHT, &[config.d_model as u64])?;
    let out_norm_bias = reader
        .host_tensor_f32_copy_dequantized_by_name(ENC_OUT_NORM_BIAS, &[config.d_model as u64])?;

    let d = config.d_model as u64;
    let mut layers = Vec::with_capacity(config.n_layers);
    for layer_idx in 0..config.n_layers {
        let names = moss_encoder_layer_tensor_names(layer_idx);
        layers.push(MossEncoderLayerWeights {
            attn_norm_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_norm_weight, &[d])?,
            attn_norm_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_norm_bias, &[d])?,
            attn_q_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_q_weight, &[d, d])?,
            attn_q_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_q_bias, &[d])?,
            attn_k_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_k_weight, &[d, d])?,
            attn_v_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_v_weight, &[d, d])?,
            attn_v_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_v_bias, &[d])?,
            attn_out_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_out_weight, &[d, d])?,
            attn_out_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_out_bias, &[d])?,
            ffn_norm_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_norm_weight, &[d])?,
            ffn_norm_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_norm_bias, &[d])?,
            ffn_up_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_up_weight, &[d, 4 * d])?,
            ffn_up_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_up_bias, &[4 * d])?,
            ffn_down_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_down_weight, &[4 * d, d])?,
            ffn_down_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_down_bias, &[d])?,
        });
    }

    Ok(MossEncoderWeights {
        conv1_weight_f16_bits,
        conv1_bias,
        conv2_weight_f16_bits,
        conv2_bias,
        pos_embd,
        layers,
        out_norm_weight,
        out_norm_bias,
    })
}

/// Run one 30s-chunk forward pass: `mel` is `[n_mels, mel_frames]` row-major
/// (mel-major, frame-minor -- matches
/// `whisper::whisper_log_mel_spectrogram_16khz_mono_v0`'s own output layout,
/// so it uploads with zero reshaping). Always produces exactly
/// `max_source_positions` output frames (a full un-trimmed 30s chunk); the
/// caller trims to the chunk's own valid length (see `executor.rs`).
///
/// Returns frame-major rows: `[frame][d_model]`, `max_source_positions *
/// d_model` values.
pub(crate) fn run_moss_encoder_chunk(
    runner: &mut GgmlCpuGraphRunner,
    weights: &MossEncoderWeights,
    config: MossEncoderConfig,
    mel: &[f32],
    mel_frames: usize,
) -> Result<Vec<f32>, MossEncoderError> {
    let expected_mel_len = config.n_mels * mel_frames;
    if mel.len() != expected_mel_len {
        return Err(MossEncoderError::InvalidMelInputLength {
            found: mel.len(),
            expected: expected_mel_len,
            n_mels: config.n_mels,
            frames: mel_frames,
        });
    }
    if config.n_heads == 0 || !config.d_model.is_multiple_of(config.n_heads) {
        return Err(MossEncoderError::GraphExecution {
            reason: format!(
                "d_model {} is not a multiple of n_heads {}",
                config.d_model, config.n_heads
            ),
        });
    }
    let head_dim = config.d_model / config.n_heads;
    let output_frames = config.max_source_positions;
    let ffn_dim = 4 * config.d_model;

    let mut graph = runner.start_graph();

    let mel_tensor = graph
        .new_tensor_2d_f32(mel_frames, config.n_mels, "moss_enc_mel")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(mel)", source))?;
    graph
        .set_input(mel_tensor)
        .map_err(|source| map_graph_error("ggml_set_input(mel)", source))?;

    let conv1_w = graph
        .new_tensor_3d_f16(
            CONV_KERNEL_SIZE,
            config.n_mels,
            config.d_model,
            "moss_enc_conv1_w",
        )
        .map_err(|source| map_graph_error("ggml_new_tensor_3d_f16(conv1_w)", source))?;
    graph
        .set_input(conv1_w)
        .map_err(|source| map_graph_error("ggml_set_input(conv1_w)", source))?;
    let conv1_b = graph
        .new_tensor_2d_f32(1, config.d_model, "moss_enc_conv1_b")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(conv1_b)", source))?;
    graph
        .set_input(conv1_b)
        .map_err(|source| map_graph_error("ggml_set_input(conv1_b)", source))?;
    let conv2_w = graph
        .new_tensor_3d_f16(
            CONV_KERNEL_SIZE,
            config.d_model,
            config.d_model,
            "moss_enc_conv2_w",
        )
        .map_err(|source| map_graph_error("ggml_new_tensor_3d_f16(conv2_w)", source))?;
    graph
        .set_input(conv2_w)
        .map_err(|source| map_graph_error("ggml_set_input(conv2_w)", source))?;
    let conv2_b = graph
        .new_tensor_2d_f32(1, config.d_model, "moss_enc_conv2_b")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(conv2_b)", source))?;
    graph
        .set_input(conv2_b)
        .map_err(|source| map_graph_error("ggml_set_input(conv2_b)", source))?;
    let pos_embd = graph
        .new_tensor_2d_f32(config.d_model, output_frames, "moss_enc_pos_embd")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(pos_embd)", source))?;
    graph
        .set_input(pos_embd)
        .map_err(|source| map_graph_error("ggml_set_input(pos_embd)", source))?;
    let mask = graph
        .new_tensor_2d_f32(output_frames, output_frames, "moss_enc_mask")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(mask)", source))?;
    graph
        .set_input(mask)
        .map_err(|source| map_graph_error("ggml_set_input(mask)", source))?;

    let conv1 = apply_conv_1d_bias_activation(
        &graph,
        conv1_w,
        mel_tensor,
        conv1_b,
        Conv1dParams {
            stride: CONV1_STRIDE,
            padding: CONV_PADDING,
            dilation: CONV_DILATION,
        },
        ConvActivation::Gelu,
        ConvBlockSteps {
            conv: "ggml_conv_1d(conv1)",
            bias: "ggml_add(conv1_bias)",
            activation: "ggml_gelu(conv1)",
        },
        map_graph_error,
    )?;
    let conv2 = apply_conv_1d_bias_activation(
        &graph,
        conv2_w,
        conv1,
        conv2_b,
        Conv1dParams {
            stride: CONV2_STRIDE,
            padding: CONV_PADDING,
            dilation: CONV_DILATION,
        },
        ConvActivation::Gelu,
        ConvBlockSteps {
            conv: "ggml_conv_1d(conv2)",
            bias: "ggml_add(conv2_bias)",
            activation: "ggml_gelu(conv2)",
        },
        map_graph_error,
    )?;
    let conv2 = graph
        .permute(conv2, 1, 0, 2, 3)
        .map_err(|source| map_graph_error("ggml_permute(conv2)", source))?;
    let conv2 = graph
        .cont(conv2)
        .map_err(|source| map_graph_error("ggml_cont(conv2)", source))?;
    let mut state = graph
        .add(conv2, pos_embd)
        .map_err(|source| map_graph_error("ggml_add(pos_embd)", source))?;

    let mut layer_uploads: Vec<(crate::ggml_runtime::GgmlCpuTensor<'_>, &[f32], &'static str)> =
        Vec::new();
    let zero_k_bias = vec![0.0_f32; config.d_model];

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let scope = format!("moss_enc_l{layer_idx}");
        let attn_norm_weight = graph
            .new_tensor_1d_f32(config.d_model, "attn_norm_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_norm_w)", source))?;
        let attn_norm_bias = graph
            .new_tensor_1d_f32(config.d_model, "attn_norm_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_norm_b)", source))?;
        let attn_q_weight = graph
            .new_tensor_2d_f32(config.d_model, config.d_model, "attn_q_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(attn_q_w)", source))?;
        let attn_q_bias = graph
            .new_tensor_1d_f32(config.d_model, "attn_q_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_q_b)", source))?;
        let attn_k_weight = graph
            .new_tensor_2d_f32(config.d_model, config.d_model, "attn_k_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(attn_k_w)", source))?;
        let attn_k_bias = graph
            .new_tensor_1d_f32(config.d_model, "attn_k_b_zero")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_k_b_zero)", source))?;
        let attn_v_weight = graph
            .new_tensor_2d_f32(config.d_model, config.d_model, "attn_v_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(attn_v_w)", source))?;
        let attn_v_bias = graph
            .new_tensor_1d_f32(config.d_model, "attn_v_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_v_b)", source))?;
        let attn_out_weight = graph
            .new_tensor_2d_f32(config.d_model, config.d_model, "attn_out_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(attn_out_w)", source))?;
        let attn_out_bias = graph
            .new_tensor_1d_f32(config.d_model, "attn_out_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(attn_out_b)", source))?;
        let ffn_norm_weight = graph
            .new_tensor_1d_f32(config.d_model, "ffn_norm_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(ffn_norm_w)", source))?;
        let ffn_norm_bias = graph
            .new_tensor_1d_f32(config.d_model, "ffn_norm_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(ffn_norm_b)", source))?;
        let ffn_up_weight = graph
            .new_tensor_2d_f32(config.d_model, ffn_dim, "ffn_up_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(ffn_up_w)", source))?;
        let ffn_up_bias = graph
            .new_tensor_1d_f32(ffn_dim, "ffn_up_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(ffn_up_b)", source))?;
        let ffn_down_weight = graph
            .new_tensor_2d_f32(ffn_dim, config.d_model, "ffn_down_w")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(ffn_down_w)", source))?;
        let ffn_down_bias = graph
            .new_tensor_1d_f32(config.d_model, "ffn_down_b")
            .map_err(|source| map_graph_error("ggml_new_tensor_1d(ffn_down_b)", source))?;

        for tensor in [
            attn_norm_weight,
            attn_norm_bias,
            attn_q_weight,
            attn_q_bias,
            attn_k_weight,
            attn_k_bias,
            attn_v_weight,
            attn_v_bias,
            attn_out_weight,
            attn_out_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
        ] {
            graph.set_input(tensor).map_err(|source| {
                map_graph_error("ggml_set_input(encoder_layer_weight)", source)
            })?;
        }

        layer_uploads.push((
            attn_norm_weight,
            &layer_weights.attn_norm_weight,
            "attn_norm_w",
        ));
        layer_uploads.push((attn_norm_bias, &layer_weights.attn_norm_bias, "attn_norm_b"));
        layer_uploads.push((attn_q_weight, &layer_weights.attn_q_weight, "attn_q_w"));
        layer_uploads.push((attn_q_bias, &layer_weights.attn_q_bias, "attn_q_b"));
        layer_uploads.push((attn_k_weight, &layer_weights.attn_k_weight, "attn_k_w"));
        layer_uploads.push((attn_k_bias, &zero_k_bias, "attn_k_b_zero"));
        layer_uploads.push((attn_v_weight, &layer_weights.attn_v_weight, "attn_v_w"));
        layer_uploads.push((attn_v_bias, &layer_weights.attn_v_bias, "attn_v_b"));
        layer_uploads.push((
            attn_out_weight,
            &layer_weights.attn_out_weight,
            "attn_out_w",
        ));
        layer_uploads.push((attn_out_bias, &layer_weights.attn_out_bias, "attn_out_b"));
        layer_uploads.push((
            ffn_norm_weight,
            &layer_weights.ffn_norm_weight,
            "ffn_norm_w",
        ));
        layer_uploads.push((ffn_norm_bias, &layer_weights.ffn_norm_bias, "ffn_norm_b"));
        layer_uploads.push((ffn_up_weight, &layer_weights.ffn_up_weight, "ffn_up_w"));
        layer_uploads.push((ffn_up_bias, &layer_weights.ffn_up_bias, "ffn_up_b"));
        layer_uploads.push((
            ffn_down_weight,
            &layer_weights.ffn_down_weight,
            "ffn_down_w",
        ));
        layer_uploads.push((ffn_down_bias, &layer_weights.ffn_down_bias, "ffn_down_b"));

        state = transformer_layer(
            &mut graph,
            state,
            mask,
            TransformerEncoderConfig {
                head_dim,
                attention_heads: config.n_heads,
                token_count: output_frames,
                layer_norm_epsilon: MOSS_ENCODER_LAYER_NORM_EPSILON,
                ffn_activation: crate::nn::ffn::FeedForwardActivation::Gelu,
                use_flash_attention: false,
            },
            TransformerEncoderLayerWeights {
                attn_norm_weight,
                attn_norm_bias,
                attn_q_weight,
                attn_q_bias,
                attn_k_weight,
                attn_k_bias,
                attn_v_weight,
                attn_v_bias,
                attn_out_weight,
                attn_out_bias,
                ffn_norm_weight,
                ffn_norm_bias,
                ffn_up_weight,
                ffn_up_bias,
                ffn_down_weight,
                ffn_down_bias,
            },
            map_graph_error,
        )?;
        let _ = scope;
    }

    let out_norm_weight = graph
        .new_tensor_1d_f32(config.d_model, "out_norm_w")
        .map_err(|source| map_graph_error("ggml_new_tensor_1d(out_norm_w)", source))?;
    graph
        .set_input(out_norm_weight)
        .map_err(|source| map_graph_error("ggml_set_input(out_norm_w)", source))?;
    let out_norm_bias = graph
        .new_tensor_1d_f32(config.d_model, "out_norm_b")
        .map_err(|source| map_graph_error("ggml_new_tensor_1d(out_norm_b)", source))?;
    graph
        .set_input(out_norm_bias)
        .map_err(|source| map_graph_error("ggml_set_input(out_norm_b)", source))?;

    state = crate::nn::norm::apply_affine_layer_norm(
        &graph,
        state,
        MOSS_ENCODER_LAYER_NORM_EPSILON,
        out_norm_weight,
        out_norm_bias,
        crate::nn::norm::AffineLayerNormSteps {
            norm: "ggml_norm(out_norm)",
            scale: "out_norm",
            bias: "out_norm",
        },
        map_graph_error,
    )?;

    graph
        .set_output(state)
        .map_err(|source| map_graph_error("ggml_set_output(encoder_out)", source))?;

    graph
        .set_f32_slice(mel_tensor, mel, "moss_enc_mel")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload mel features: {source}"),
        })?;
    graph
        .set_f16_bits_slice(conv1_w, &weights.conv1_weight_f16_bits, "moss_enc_conv1_w")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload conv1 weight: {source}"),
        })?;
    graph
        .set_f32_slice(conv1_b, &weights.conv1_bias, "moss_enc_conv1_b")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload conv1 bias: {source}"),
        })?;
    graph
        .set_f16_bits_slice(conv2_w, &weights.conv2_weight_f16_bits, "moss_enc_conv2_w")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload conv2 weight: {source}"),
        })?;
    graph
        .set_f32_slice(conv2_b, &weights.conv2_bias, "moss_enc_conv2_b")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload conv2 bias: {source}"),
        })?;
    graph
        .set_f32_slice(pos_embd, &weights.pos_embd, "moss_enc_pos_embd")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload positional embedding: {source}"),
        })?;
    let mask_zeros = vec![0.0_f32; output_frames * output_frames];
    graph
        .set_f32_slice(mask, &mask_zeros, "moss_enc_mask")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload attention mask: {source}"),
        })?;
    graph
        .set_f32_slice(out_norm_weight, &weights.out_norm_weight, "out_norm_w")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload out_norm weight: {source}"),
        })?;
    graph
        .set_f32_slice(out_norm_bias, &weights.out_norm_bias, "out_norm_b")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload out_norm bias: {source}"),
        })?;
    for (tensor, values, label) in layer_uploads {
        graph
            .set_f32_slice(tensor, values, label)
            .map_err(|source| MossEncoderError::GraphExecution {
                reason: format!("could not upload encoder layer weight '{label}': {source}"),
            })?;
    }

    let values = graph
        .compute_output_f32(state, output_frames * config.d_model)
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("encoder graph compute failed: {source}"),
        })?;
    Ok(values)
}
