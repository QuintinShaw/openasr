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
//! Weights live resident across chunks instead of being re-declared as
//! per-call graph inputs, mirroring `qwen::audio_encoder`'s
//! `Qwen3AsrAudioEncoderRuntime` pattern: the six 2D projection weights per
//! layer (`attn_{q,k,v,out}`, `ffn_{up,down}` -- by far the largest tensors,
//! ~48 MiB/layer x 24 layers of host f32 if dequantized) bind zero-copy from
//! the mmap'd pack's native f16 storage (see `load_moss_encoder_weights_from_reader`
//! and `loaded_or_arena_2d` below), never touching host memory; everything
//! else small (conv stem, every 1D norm/bias, the fixed positional embedding,
//! the final LayerNorm) lives in a WEIGHTS-usage static-tensor arena uploaded
//! once per `encode()` call. Only the per-chunk mel features and the (always
//! all-zero, but genuinely re-uploaded per call for op-order clarity)
//! attention mask stay real graph inputs.

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedWeightContext,
    GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReadError, GgufTensorDataReader,
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

/// A 2D projection weight's GGUF tensor name, with `values` always empty:
/// this encoder never dequantizes these six per-layer weights to host f32
/// (see `load_moss_encoder_weights_from_reader`), it only records the name so
/// `loaded_or_arena_2d` can bind the mmap'd pack's native tensor zero-copy at
/// `encode()` time. `values` exists (rather than dropping the field) so
/// `loaded_or_arena_2d` has the same fail-closed shape as
/// `qwen::audio_encoder`'s `F32Tensor`/`loaded_or_arena_2d`: an empty slice
/// with no `loaded` bind is a hard error, never a silently-empty upload.
struct MossProjectionTensor {
    name: String,
    values: Vec<f32>,
}

struct MossEncoderLayerWeights {
    attn_norm_weight: Vec<f32>,
    attn_norm_bias: Vec<f32>,
    attn_q_weight: MossProjectionTensor,
    attn_q_bias: Vec<f32>,
    attn_k_weight: MossProjectionTensor,
    attn_v_weight: MossProjectionTensor,
    attn_v_bias: Vec<f32>,
    attn_out_weight: MossProjectionTensor,
    attn_out_bias: Vec<f32>,
    ffn_norm_weight: Vec<f32>,
    ffn_norm_bias: Vec<f32>,
    ffn_up_weight: MossProjectionTensor,
    ffn_up_bias: Vec<f32>,
    ffn_down_weight: MossProjectionTensor,
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
            // The four attention projections and the two FFN projections are
            // never dequantized here -- `encode()` binds them zero-copy from
            // the mmap'd pack via `loaded_or_arena_2d` (goals 7+8 Step 1b,
            // mirrored from `qwen::audio_encoder`). Only the name is kept.
            attn_q_weight: MossProjectionTensor {
                name: names.attn_q_weight,
                values: Vec::new(),
            },
            attn_q_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_q_bias, &[d])?,
            attn_k_weight: MossProjectionTensor {
                name: names.attn_k_weight,
                values: Vec::new(),
            },
            attn_v_weight: MossProjectionTensor {
                name: names.attn_v_weight,
                values: Vec::new(),
            },
            attn_v_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_v_bias, &[d])?,
            attn_out_weight: MossProjectionTensor {
                name: names.attn_out_weight,
                values: Vec::new(),
            },
            attn_out_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.attn_out_bias, &[d])?,
            ffn_norm_weight: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_norm_weight, &[d])?,
            ffn_norm_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_norm_bias, &[d])?,
            ffn_up_weight: MossProjectionTensor {
                name: names.ffn_up_weight,
                values: Vec::new(),
            },
            ffn_up_bias: reader
                .host_tensor_f32_copy_dequantized_by_name(&names.ffn_up_bias, &[4 * d])?,
            ffn_down_weight: MossProjectionTensor {
                name: names.ffn_down_weight,
                values: Vec::new(),
            },
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

/// Owns the encoder's graph runner, the optional zero-copy weight context,
/// and the fully-loaded [`MossEncoderWeights`]; all three are expensive to
/// rebuild per chunk (the loaded-weight context mmaps the whole pack, and
/// loading the weights reads every small tensor off disk), so callers build
/// one runtime (typically via the thread-local resident cache in
/// `executor.rs`, mirroring `firered_aed::executor`'s
/// `FireRedEncoderGraphRuntime`) and call [`Self::encode`] once per 30s
/// chunk. Mirrors `qwen::audio_encoder::Qwen3AsrAudioEncoderRuntime`.
pub(crate) struct MossEncoderRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: Option<GgmlLoadedWeightContext>,
    weights: MossEncoderWeights,
}

impl MossEncoderRuntime {
    pub(crate) fn new(
        runtime_path: &Path,
        config: MossEncoderConfig,
    ) -> Result<Self, MossEncoderError> {
        let graph_config = super::graph_config::moss_td_encoder_graph_config();
        let runner = GgmlCpuGraphRunner::new(graph_config)
            .map_err(|source| map_graph_error("runner_init", source))?;
        let loaded = runner.load_gguf_weight_context(runtime_path).ok();
        let reader = GgufTensorDataReader::from_path(runtime_path)?;
        let weights = load_moss_encoder_weights_from_reader(&reader, config)?;
        Ok(Self {
            runner,
            loaded,
            weights,
        })
    }

    /// Run one 30s-chunk forward pass: `mel` is `[n_mels, mel_frames]`
    /// row-major (mel-major, frame-minor -- matches
    /// `whisper::whisper_log_mel_spectrogram_16khz_mono_v0`'s own output
    /// layout, so it uploads with zero reshaping). Always produces exactly
    /// `max_source_positions` output frames (a full un-trimmed 30s chunk);
    /// the caller trims to the chunk's own valid length (see `executor.rs`).
    ///
    /// Returns frame-major rows: `[frame][d_model]`, `max_source_positions *
    /// d_model` values.
    pub(crate) fn encode(
        &mut self,
        config: MossEncoderConfig,
        mel: &[f32],
        mel_frames: usize,
    ) -> Result<Vec<f32>, MossEncoderError> {
        let weights = &self.weights;
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

        let loaded = self.loaded.as_ref();

        // Resident encoder weights (conv stem, every 1D norm/bias, the fixed
        // positional embedding, the final LayerNorm) live in a WEIGHTS-usage
        // arena buffer instead of per-call graph-input leaves; the six 2D
        // projection weights per layer bind zero-copy from the mmap'd pack
        // when bound. Only the mel input and the (per-call, always all-zero)
        // attention mask stay genuine graph inputs. Mirrors
        // `qwen::audio_encoder`'s `build_qwen_audio_resident_weights`.
        let mut arena = self
            .runner
            .start_static_tensor_arena(moss_encoder_arena_context_bytes(weights))
            .map_err(|source| map_graph_error("static_tensor_arena", source))?;
        let resident = build_moss_encoder_resident_weights(&mut arena, weights, config, loaded)?;

        let mut graph = self.runner.start_graph();

        let mel_tensor = graph
            .new_tensor_2d_f32(mel_frames, config.n_mels, "moss_enc_mel")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(mel)", source))?;
        graph
            .set_input(mel_tensor)
            .map_err(|source| map_graph_error("ggml_set_input(mel)", source))?;
        let mask = graph
            .new_tensor_2d_f32(output_frames, output_frames, "moss_enc_mask")
            .map_err(|source| map_graph_error("ggml_new_tensor_2d(mask)", source))?;
        graph
            .set_input(mask)
            .map_err(|source| map_graph_error("ggml_set_input(mask)", source))?;

        let conv1 = apply_conv_1d_bias_activation(
            &graph,
            resident.conv1_weight,
            mel_tensor,
            resident.conv1_bias,
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
            resident.conv2_weight,
            conv1,
            resident.conv2_bias,
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
            .add(conv2, resident.pos_embd)
            .map_err(|source| map_graph_error("ggml_add(pos_embd)", source))?;

        for tensors in &resident.layers {
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
                    use_flash_attention: true,
                },
                TransformerEncoderLayerWeights {
                    attn_norm_weight: tensors.attn_norm_weight,
                    attn_norm_bias: tensors.attn_norm_bias,
                    attn_q_weight: tensors.attn_q_weight,
                    attn_q_bias: tensors.attn_q_bias,
                    attn_k_weight: tensors.attn_k_weight,
                    attn_k_bias: tensors.attn_k_bias,
                    attn_v_weight: tensors.attn_v_weight,
                    attn_v_bias: tensors.attn_v_bias,
                    attn_out_weight: tensors.attn_out_weight,
                    attn_out_bias: tensors.attn_out_bias,
                    ffn_norm_weight: tensors.ffn_norm_weight,
                    ffn_norm_bias: tensors.ffn_norm_bias,
                    ffn_up_weight: tensors.ffn_up_weight,
                    ffn_up_bias: tensors.ffn_up_bias,
                    ffn_down_weight: tensors.ffn_down_weight,
                    ffn_down_bias: tensors.ffn_down_bias,
                },
                map_graph_error,
            )?;
        }

        state = crate::nn::norm::apply_affine_layer_norm(
            &graph,
            state,
            MOSS_ENCODER_LAYER_NORM_EPSILON,
            resident.out_norm_weight,
            resident.out_norm_bias,
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

        // Peak-RSS lever: allocate the compute graph via the scheduler's
        // gallocr (liveness-based buffer REUSE) before uploading inputs, so
        // the per-layer intermediates collapse to the working-set peak
        // instead of each getting its own buffer.
        graph
            .prepare_outputs_for_upload(&[state])
            .map_err(|source| map_graph_error("ggml_prepare_outputs(encoder_out)", source))?;

        // Only the genuine per-call inputs are uploaded here; every weight
        // already resides in the arena's WEIGHTS-usage buffer (uploaded once
        // in `build_moss_encoder_resident_weights`) or is bound zero-copy.
        graph
            .set_f32_slice(mel_tensor, mel, "moss_enc_mel")
            .map_err(|source| MossEncoderError::GraphExecution {
                reason: format!("could not upload mel features: {source}"),
            })?;
        let mask_zeros = vec![0.0_f32; output_frames * output_frames];
        graph
            .set_f32_slice(mask, &mask_zeros, "moss_enc_mask")
            .map_err(|source| MossEncoderError::GraphExecution {
                reason: format!("could not upload attention mask: {source}"),
            })?;

        let values = graph
            .compute_output_f32(state, output_frames * config.d_model)
            .map_err(|source| MossEncoderError::GraphExecution {
                reason: format!("encoder graph compute failed: {source}"),
            })?;
        Ok(values)
    }
}

/// Test-only twin of [`MossEncoderRuntime::encode`] for the CPU-vs-Metal
/// numeric-divergence bisection (see `parity_tests` below): builds the
/// identical graph but taps every layer's final output (post-`transformer_layer`,
/// pre-final-norm) plus the subsample stem's output and the post-final-norm
/// `encoder_out`, so a caller can run this once per backend and diff layer by
/// layer to find the first one that decorrelates. Mirrors firered-aed's
/// `encode_with_layer_taps` (`models/firered_aed/encoder_graph.rs`).
#[cfg(test)]
pub(crate) fn encode_with_layer_taps(
    runner: &mut GgmlCpuGraphRunner,
    loaded: Option<&GgmlLoadedWeightContext>,
    weights: &MossEncoderWeights,
    config: MossEncoderConfig,
    mel: &[f32],
    mel_frames: usize,
) -> Result<MossEncoderTapDump, MossEncoderError> {
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

    let mut arena = runner
        .start_static_tensor_arena(moss_encoder_arena_context_bytes(weights))
        .map_err(|source| map_graph_error("static_tensor_arena", source))?;
    let resident = build_moss_encoder_resident_weights(&mut arena, weights, config, loaded)?;

    let mut graph = runner.start_graph();

    let mel_tensor = graph
        .new_tensor_2d_f32(mel_frames, config.n_mels, "moss_enc_mel")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(mel)", source))?;
    graph
        .set_input(mel_tensor)
        .map_err(|source| map_graph_error("ggml_set_input(mel)", source))?;
    let mask = graph
        .new_tensor_2d_f32(output_frames, output_frames, "moss_enc_mask")
        .map_err(|source| map_graph_error("ggml_new_tensor_2d(mask)", source))?;
    graph
        .set_input(mask)
        .map_err(|source| map_graph_error("ggml_set_input(mask)", source))?;

    let conv1 = apply_conv_1d_bias_activation(
        &graph,
        resident.conv1_weight,
        mel_tensor,
        resident.conv1_bias,
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
        resident.conv2_weight,
        conv1,
        resident.conv2_bias,
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
        .add(conv2, resident.pos_embd)
        .map_err(|source| map_graph_error("ggml_add(pos_embd)", source))?;
    let subsample_out = state;

    let mut layer_outputs = Vec::with_capacity(resident.layers.len());
    for tensors in &resident.layers {
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
                use_flash_attention: true,
            },
            TransformerEncoderLayerWeights {
                attn_norm_weight: tensors.attn_norm_weight,
                attn_norm_bias: tensors.attn_norm_bias,
                attn_q_weight: tensors.attn_q_weight,
                attn_q_bias: tensors.attn_q_bias,
                attn_k_weight: tensors.attn_k_weight,
                attn_k_bias: tensors.attn_k_bias,
                attn_v_weight: tensors.attn_v_weight,
                attn_v_bias: tensors.attn_v_bias,
                attn_out_weight: tensors.attn_out_weight,
                attn_out_bias: tensors.attn_out_bias,
                ffn_norm_weight: tensors.ffn_norm_weight,
                ffn_norm_bias: tensors.ffn_norm_bias,
                ffn_up_weight: tensors.ffn_up_weight,
                ffn_up_bias: tensors.ffn_up_bias,
                ffn_down_weight: tensors.ffn_down_weight,
                ffn_down_bias: tensors.ffn_down_bias,
            },
            map_graph_error,
        )?;
        layer_outputs.push(state);
    }

    let encoder_out = crate::nn::norm::apply_affine_layer_norm(
        &graph,
        state,
        MOSS_ENCODER_LAYER_NORM_EPSILON,
        resident.out_norm_weight,
        resident.out_norm_bias,
        crate::nn::norm::AffineLayerNormSteps {
            norm: "ggml_norm(out_norm)",
            scale: "out_norm",
            bias: "out_norm",
        },
        map_graph_error,
    )?;

    let mut all_outputs = vec![subsample_out];
    all_outputs.extend(layer_outputs.iter().copied());
    all_outputs.push(encoder_out);
    // Every tap must be marked `ggml_set_output` -- see firered-aed's
    // `encode_with_layer_taps` module doc for why (gallocr buffer-reuse
    // otherwise silently recycles a tap's buffer for a later tensor).
    for &tensor in &all_outputs {
        graph
            .set_output(tensor)
            .map_err(|source| map_graph_error("ggml_set_output(encoder_tap)", source))?;
    }

    graph
        .prepare_outputs_for_upload(&all_outputs)
        .map_err(|source| map_graph_error("ggml_prepare_outputs(encoder_taps)", source))?;

    graph
        .set_f32_slice(mel_tensor, mel, "moss_enc_mel")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload mel features: {source}"),
        })?;
    let mask_zeros = vec![0.0_f32; output_frames * output_frames];
    graph
        .set_f32_slice(mask, &mask_zeros, "moss_enc_mask")
        .map_err(|source| MossEncoderError::GraphExecution {
            reason: format!("could not upload attention mask: {source}"),
        })?;

    let expected_len = output_frames * config.d_model;
    let requests: Vec<(crate::ggml_runtime::GgmlCpuTensor<'_>, usize)> =
        all_outputs.iter().map(|&t| (t, expected_len)).collect();
    let mut computed = graph.compute_outputs_f32(&requests).map_err(|source| {
        MossEncoderError::GraphExecution {
            reason: format!("encoder taps graph compute failed: {source}"),
        }
    })?;

    let mut iter = computed.drain(..);
    let subsample_rows = iter.next().expect("subsample_out present");
    let layer_rows: Vec<Vec<f32>> = (0..layer_outputs.len())
        .map(|_| iter.next().expect("layer output present"))
        .collect();
    let encoder_out_rows = iter.next().expect("encoder_out present");

    Ok(MossEncoderTapDump {
        d_model: config.d_model,
        subsample_rows,
        layer_rows,
        encoder_out_rows,
    })
}

/// Test-only bisection dump: subsample-stem output, every layer's final
/// output (post-`transformer_layer`, pre-final-norm), and the final
/// post-final-norm `encoder_out` -- all row-major `[frame][d_model]` f32.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MossEncoderTapDump {
    pub d_model: usize,
    pub subsample_rows: Vec<f32>,
    /// `layer_rows[i]` is transformer layer `i`'s final output (0-indexed).
    pub layer_rows: Vec<Vec<f32>>,
    pub encoder_out_rows: Vec<f32>,
}

/// Resident encoder graph tensors for one transformer layer, all living in
/// the arena's WEIGHTS-usage backend buffer (the six 2D projections either
/// bound zero-copy from the mmap'd pack or, when unbound, arena-f32-uploaded
/// as a defensive fallback -- see `loaded_or_arena_2d`). Mirrors
/// `qwen::audio_encoder`'s `AudioLayerGraphTensors`.
#[derive(Clone, Copy)]
struct MossEncoderLayerGraphTensors<'a> {
    attn_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_q_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_q_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_k_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_k_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_v_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_v_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_out_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    attn_out_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_up_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_up_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_down_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    ffn_down_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
}

/// Resident tensors shared by every chunk's forward pass. Only the mel input
/// and the attention mask (built in [`MossEncoderRuntime::encode`]) are
/// genuine per-call graph inputs.
struct MossEncoderResidentTensors<'a> {
    conv1_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv1_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv2_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    conv2_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    pos_embd: crate::ggml_runtime::GgmlCpuTensor<'a>,
    out_norm_weight: crate::ggml_runtime::GgmlCpuTensor<'a>,
    out_norm_bias: crate::ggml_runtime::GgmlCpuTensor<'a>,
    layers: Vec<MossEncoderLayerGraphTensors<'a>>,
}

/// Upper bound on the arena's metadata context: the fixed conv-stem/pos-embd/
/// out-norm tensors plus the worst-case per-layer tensor count (10 x 1D
/// norm/bias that always land in the arena + up to 6 x 2D projection when the
/// pack does not bind them for a zero-copy read). Over-counting only sizes the
/// (cheap) tensor-overhead context; the real weight bytes land in a
/// separately sized backend buffer. Mirrors
/// `qwen::audio_encoder::qwen_audio_encoder_arena_context_bytes`.
fn moss_encoder_arena_context_bytes(weights: &MossEncoderWeights) -> usize {
    const FIXED_TENSORS: usize = 8;
    const MAX_TENSORS_PER_LAYER: usize = 16;
    let count =
        FIXED_TENSORS.saturating_add(MAX_TENSORS_PER_LAYER.saturating_mul(weights.layers.len()));
    GgmlCpuGraphConfig::metadata_context_bytes(count)
}

/// Collects `(arena handle, host slice, label)` uploads while every arena
/// tensor is allocated, then flushes them once. Allocation MUST precede the
/// arena's first upload (the first `set_*_slice` freezes further creation),
/// so callers allocate every tensor first and call [`Self::upload`] last.
/// Mirrors `qwen::audio_encoder`'s `QwenAudioArenaBuilder`.
struct MossEncoderArenaBuilder<'w> {
    f32_uploads: Vec<(GgmlStaticTensor, &'w [f32], &'static str)>,
    f16_uploads: Vec<(GgmlStaticTensor, &'w [u16], &'static str)>,
}

impl<'w> MossEncoderArenaBuilder<'w> {
    fn new() -> Self {
        Self {
            f32_uploads: Vec::new(),
            f16_uploads: Vec::new(),
        }
    }

    /// A 1D norm/bias, or the fixed positional embedding treated as one flat
    /// f32 slice: always an f32 arena tensor.
    fn arena_1d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        values: &'w [f32],
        step: &'static str,
    ) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, MossEncoderError> {
        let handle = arena
            .new_tensor_1d_f32(values.len(), step)
            .map_err(|source| map_graph_error(step, source))?;
        self.f32_uploads.push((handle, values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// The fixed positional embedding: a 2D f32 arena tensor (constant across
    /// every chunk, unlike qwen's per-call-derived positional embedding).
    fn arena_2d_f32<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        values: &'w [f32],
        ne0: usize,
        ne1: usize,
        step: &'static str,
    ) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, MossEncoderError> {
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, step)
            .map_err(|source| map_graph_error(step, source))?;
        self.f32_uploads.push((handle, values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// A rank-3 conv kernel: always an f16 arena tensor (the conv stem's own
    /// native storage type; unlike the 2D projections, small enough that
    /// zero-copy binding buys nothing worth the extra code path).
    fn arena_3d_f16<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        values: &'w [u16],
        ne0: usize,
        ne1: usize,
        ne2: usize,
        step: &'static str,
    ) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, MossEncoderError> {
        let handle = arena
            .new_tensor_3d_f16(ne0, ne1, ne2, step)
            .map_err(|source| map_graph_error(step, source))?;
        self.f16_uploads.push((handle, values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// A 2D attention/FFN projection weight: bound zero-copy from the mmap'd
    /// pack (`loaded`, native f16) when present. `load_moss_encoder_weights_from_reader`
    /// never materializes a host f32 copy for these, so an unbound tensor
    /// with an empty `values` means the pack lacks it -- better to error than
    /// bind an empty buffer. Mirrors `qwen::audio_encoder`'s
    /// `loaded_or_arena_2d`.
    fn loaded_or_arena_2d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        loaded: Option<&GgmlLoadedWeightContext>,
        tensor: &'w MossProjectionTensor,
        ne0: usize,
        ne1: usize,
        step: &'static str,
    ) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, MossEncoderError> {
        if let Some(loaded_tensor) = loaded.and_then(|context| context.tensor(&tensor.name)) {
            return Ok(loaded_tensor.as_graph_tensor());
        }
        if tensor.values.is_empty() {
            return Err(MossEncoderError::GraphExecution {
                reason: format!(
                    "encoder weight '{}' is neither bound zero-copy nor f32-materialized",
                    tensor.name
                ),
            });
        }
        let handle = arena
            .new_tensor_2d_f32(ne0, ne1, step)
            .map_err(|source| map_graph_error(step, source))?;
        self.f32_uploads.push((handle, &tensor.values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// Flush every collected upload into the arena's backend buffer. The
    /// first upload allocates the buffer and freezes further tensor
    /// creation, so this runs after all allocation.
    fn upload(self, arena: &mut GgmlStaticTensorArena) -> Result<(), MossEncoderError> {
        for (handle, values, step) in self.f32_uploads {
            arena
                .set_f32_slice(handle, values, step)
                .map_err(|source| MossEncoderError::GraphExecution {
                    reason: format!("could not upload arena weight '{step}': {source}"),
                })?;
        }
        for (handle, values, step) in self.f16_uploads {
            arena
                .set_f16_bits_slice(handle, values, step)
                .map_err(|source| MossEncoderError::GraphExecution {
                    reason: format!("could not upload arena weight '{step}': {source}"),
                })?;
        }
        Ok(())
    }
}

/// Allocate every resident encoder weight in the arena and upload it once,
/// returning the graph-tensor handles the forward graph references. Mirrors
/// `qwen::audio_encoder`'s `build_qwen_audio_resident_weights`.
fn build_moss_encoder_resident_weights<'a>(
    arena: &mut GgmlStaticTensorArena,
    weights: &MossEncoderWeights,
    config: MossEncoderConfig,
    loaded: Option<&GgmlLoadedWeightContext>,
) -> Result<MossEncoderResidentTensors<'a>, MossEncoderError> {
    let mut builder = MossEncoderArenaBuilder::new();
    let ffn_dim = 4 * config.d_model;

    let conv1_weight = builder.arena_3d_f16(
        arena,
        &weights.conv1_weight_f16_bits,
        CONV_KERNEL_SIZE,
        config.n_mels,
        config.d_model,
        "moss_enc_conv1_w",
    )?;
    // Conv bias tensors are `[1, d_model]` (ne0=1, ne1=d_model), NOT a flat 1D
    // `[d_model]`: `ggml_add`'s broadcast against the conv output (ne0=out_len,
    // ne1=d_model) needs ne0=1 to repeat across `out_len`, and ne1=d_model to
    // match channel-for-channel -- a 1D tensor would instead broadcast across
    // channels, corrupting the result. Preserves the original per-call-input
    // shape exactly (this is a shape-only change, not a math change).
    let conv1_bias = builder.arena_2d_f32(
        arena,
        &weights.conv1_bias,
        1,
        config.d_model,
        "moss_enc_conv1_b",
    )?;
    let conv2_weight = builder.arena_3d_f16(
        arena,
        &weights.conv2_weight_f16_bits,
        CONV_KERNEL_SIZE,
        config.d_model,
        config.d_model,
        "moss_enc_conv2_w",
    )?;
    let conv2_bias = builder.arena_2d_f32(
        arena,
        &weights.conv2_bias,
        1,
        config.d_model,
        "moss_enc_conv2_b",
    )?;
    let pos_embd = builder.arena_2d_f32(
        arena,
        &weights.pos_embd,
        config.d_model,
        config.max_source_positions,
        "moss_enc_pos_embd",
    )?;
    let out_norm_weight = builder.arena_1d(arena, &weights.out_norm_weight, "out_norm_w")?;
    let out_norm_bias = builder.arena_1d(arena, &weights.out_norm_bias, "out_norm_b")?;

    // Whisper's own `k_proj` carries no bias; a fresh all-zero arena tensor
    // per layer is the exact no-op the module doc describes, just resident
    // instead of a per-call graph input.
    let zero_k_bias = vec![0.0_f32; config.d_model];
    let mut layers = Vec::with_capacity(weights.layers.len());
    for layer in &weights.layers {
        layers.push(MossEncoderLayerGraphTensors {
            attn_norm_weight: builder.arena_1d(arena, &layer.attn_norm_weight, "attn_norm_w")?,
            attn_norm_bias: builder.arena_1d(arena, &layer.attn_norm_bias, "attn_norm_b")?,
            attn_q_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.attn_q_weight,
                config.d_model,
                config.d_model,
                "attn_q_w",
            )?,
            attn_q_bias: builder.arena_1d(arena, &layer.attn_q_bias, "attn_q_b")?,
            attn_k_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.attn_k_weight,
                config.d_model,
                config.d_model,
                "attn_k_w",
            )?,
            attn_k_bias: builder.arena_1d(arena, &zero_k_bias, "attn_k_b_zero")?,
            attn_v_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.attn_v_weight,
                config.d_model,
                config.d_model,
                "attn_v_w",
            )?,
            attn_v_bias: builder.arena_1d(arena, &layer.attn_v_bias, "attn_v_b")?,
            attn_out_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.attn_out_weight,
                config.d_model,
                config.d_model,
                "attn_out_w",
            )?,
            attn_out_bias: builder.arena_1d(arena, &layer.attn_out_bias, "attn_out_b")?,
            ffn_norm_weight: builder.arena_1d(arena, &layer.ffn_norm_weight, "ffn_norm_w")?,
            ffn_norm_bias: builder.arena_1d(arena, &layer.ffn_norm_bias, "ffn_norm_b")?,
            ffn_up_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.ffn_up_weight,
                config.d_model,
                ffn_dim,
                "ffn_up_w",
            )?,
            ffn_up_bias: builder.arena_1d(arena, &layer.ffn_up_bias, "ffn_up_b")?,
            ffn_down_weight: builder.loaded_or_arena_2d(
                arena,
                loaded,
                &layer.ffn_down_weight,
                ffn_dim,
                config.d_model,
                "ffn_down_w",
            )?,
            ffn_down_bias: builder.arena_1d(arena, &layer.ffn_down_bias, "ffn_down_b")?,
        });
    }

    builder.upload(arena)?;

    Ok(MossEncoderResidentTensors {
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        pos_embd,
        out_norm_weight,
        out_norm_bias,
        layers,
    })
}

#[cfg(test)]
mod parity_tests {
    //! CPU-vs-Metal numeric-divergence bisection for the "encoder decorrelates
    //! on Metal" defect (`arch/mod.rs`'s `MOSS_TD_GGML_ARCHITECTURE_ID`
    //! `auto_gpu_policy: ExceptMetal` doc comment, defect 1 of 2). Dev-only:
    //! needs the real ~1.8 GB `moss-transcribe-diarize-fp16.oasr` pack (never
    //! committed) and a real Metal device, so every test here is `#[ignore]`d
    //! and points at the pack through an env var rather than a fixed path.
    //!
    //! Methodology mirrors firered-aed's CPU-vs-PyTorch bisection
    //! (`models/firered_aed/encoder_graph.rs`'s `parity_tests` module doc):
    //! run the identical graph through [`encode_with_layer_taps`] once per
    //! backend on the same input, then compare every tap with a
    //! mean-per-frame cosine similarity (mirrors
    //! `xasr_zipformer::encoder_graph`'s `xasr_mean_frame_cosine_similarity` --
    //! a single degraded frame pulls this metric down without a spread-evenly
    //! algorithmic bug being able to hide behind one bad frame dominating a
    //! whole-tensor max-diff).
    use super::*;
    use crate::ggml_runtime::{
        GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgufTensorDataReader, read_gguf_metadata,
    };
    use crate::models::moss_transcribe_diarize::runtime_contract::parse_encoder_metadata;
    use crate::models::whisper::whisper_log_mel_spectrogram_16khz_mono_v0;

    const REAL_PACK_ENV: &str = "OPENASR_MOSS_TD_ENCODER_REAL_PACK";
    /// Optional override for the harness input wav (defaults to
    /// `fixtures/jfk.wav`); lets the same bisection run on longer/harder
    /// audio without touching the harness.
    const REAL_WAV_ENV: &str = "OPENASR_MOSS_TD_ENCODER_REAL_WAV";
    /// Optional 16 kHz-second offset of the 30s chunk window cut from the
    /// harness wav (defaults to 0): the encoder is a fixed 30s-chunk graph,
    /// so "long audio" coverage means different 30s windows of a long file.
    const CHUNK_OFFSET_ENV: &str = "OPENASR_MOSS_TD_ENCODER_CHUNK_OFFSET_SECS";
    /// `WhisperFeatureExtractor`'s target frame count for one 30s chunk
    /// (mirrors `executor.rs`'s `MEL_TARGET_FRAMES`; not exported from there
    /// since that module keeps it private).
    const MEL_TARGET_FRAMES: usize = 3000;

    fn real_pack_path() -> std::path::PathBuf {
        std::env::var_os(REAL_PACK_ENV)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{REAL_PACK_ENV} must point to a moss-transcribe-diarize .oasr pack")
            })
    }

    fn dev_wav_path() -> std::path::PathBuf {
        match std::env::var_os(REAL_WAV_ENV) {
            Some(raw) if !raw.is_empty() => std::path::PathBuf::from(raw),
            _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav"),
        }
    }

    /// Cosine similarity between two equal-length vectors, treated as a
    /// single flattened point in R^n.
    fn cosine_similarity(actual: &[f32], expected: &[f32]) -> f64 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "cosine_similarity length mismatch"
        );
        let dot: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm_a: f64 = actual.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let norm_b: f64 = expected.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return if norm_a == norm_b { 1.0 } else { 0.0 };
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Mean cosine similarity across every `dim`-wide row (mirrors
    /// `xasr_zipformer::encoder_graph`'s `xasr_mean_frame_cosine_similarity`).
    fn mean_frame_cosine_similarity(actual: &[f32], expected: &[f32], dim: usize) -> f64 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "mean_frame_cosine_similarity length mismatch"
        );
        assert_eq!(actual.len() % dim, 0, "tap length not a multiple of dim");
        let frames = actual.len() / dim;
        assert!(frames > 0, "tap has no frames");
        let sum: f64 = (0..frames)
            .map(|f| {
                let lo = f * dim;
                let hi = lo + dim;
                cosine_similarity(&actual[lo..hi], &expected[lo..hi])
            })
            .sum();
        sum / frames as f64
    }

    fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Runs the tapped encoder graph once on the requested backend against
    /// the real pack + a real 30s chunk of `fixtures/jfk.wav`.
    fn run_tapped_encoder(backend: GgmlCpuGraphBackend) -> MossEncoderTapDump {
        let pack_path = real_pack_path();
        let metadata_view = read_gguf_metadata(&pack_path).expect("read gguf metadata");
        let encoder_metadata =
            parse_encoder_metadata(&metadata_view).expect("parse moss encoder metadata");
        let config = MossEncoderConfig {
            n_layers: encoder_metadata.n_layers,
            d_model: encoder_metadata.d_model,
            n_heads: encoder_metadata.n_heads,
            n_mels: encoder_metadata.n_mels,
            max_source_positions: encoder_metadata.max_source_positions,
        };

        let wav_path = dev_wav_path();
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path.clone(),
            "moss encoder metal parity test",
            "moss encoder metal parity test",
        )
        .expect("load harness wav");
        let offset_secs: usize = std::env::var(CHUNK_OFFSET_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
            .unwrap_or(0);
        let start = (offset_secs * 16_000).min(samples.len());
        let chunk = &samples[start..samples.len().min(start + 480_000)];
        eprintln!(
            "moss encoder harness input: {} (chunk offset {offset_secs}s, {} samples)",
            wav_path.display(),
            chunk.len()
        );
        let mel =
            whisper_log_mel_spectrogram_16khz_mono_v0(chunk, config.n_mels, MEL_TARGET_FRAMES)
                .expect("compute mel");

        let reader = GgufTensorDataReader::from_path(&pack_path).expect("open tensor reader");
        let weights =
            load_moss_encoder_weights_from_reader(&reader, config).expect("load encoder weights");

        // Route through the same `moss_td_encoder_graph_config()` the real
        // executor uses (not a bare `GgmlCpuGraphConfig::default()` with only
        // `.backend` overwritten): `configure_model_runtime_graph_config`
        // only forces `n_threads=1`/`use_scheduler=true` when the config's
        // backend is ALREADY `Metal` at build time (see
        // `models/graph_runtime_config.rs`), so overwriting `.backend` on an
        // already-built CPU-default config silently keeps CPU's
        // thread/scheduler pairing under a Metal backend tag -- not the real
        // production Metal path. Driving backend selection through the
        // `OPENASR_GGML_BACKEND` env var before the config is built (exactly
        // how `resolve_runtime_backend` is meant to be steered) is the only
        // way to get the correctly-paired settings for each backend. Safe
        // here only because this manual harness runs single-threaded
        // (`--test-threads=1`, enforced by the `#[ignore]` message).
        let env_value = match backend {
            GgmlCpuGraphBackend::Cpu => "cpu",
            GgmlCpuGraphBackend::Metal => "metal",
            GgmlCpuGraphBackend::Gpu => "gpu",
        };
        // SAFETY: this harness is documented `--test-threads=1`-only (see the
        // `#[ignore]` messages on its two callers), so no concurrent test
        // observes this process-global env var mid-mutation.
        unsafe {
            std::env::set_var(GgmlCpuGraphConfig::BACKEND_ENV, env_value);
        }
        // The family's `ExceptMetal` gate (`graph_config.rs`) downgrades an
        // Auto-resolved Metal -- including one steered by `BACKEND_ENV` above
        // -- to CPU, so env-var steering alone can never reach the Metal leg.
        // Install an explicit `Accelerated` request override for that leg:
        // the documented production path the gate always honors
        // (`resolve_family_runtime_backend` doc comment -- an explicit
        // per-request preference wins over any Auto-mode gate), i.e. exactly
        // what an `execution_target=accelerated` request runs in production.
        let _accelerated_override = (backend == GgmlCpuGraphBackend::Metal).then(|| {
            crate::ggml_runtime::install_request_backend_override(Some(
                crate::ggml_runtime::RequestBackendPreference::Accelerated,
            ))
        });
        let mut graph_config =
            crate::models::moss_transcribe_diarize::graph_config::moss_td_encoder_graph_config();
        unsafe {
            std::env::remove_var(GgmlCpuGraphConfig::BACKEND_ENV);
        }
        assert_eq!(
            graph_config.backend,
            backend,
            "moss_td_encoder_graph_config did not resolve the requested backend from {}",
            GgmlCpuGraphConfig::BACKEND_ENV
        );
        graph_config.graph_size = graph_config.graph_size.max(16_384);
        graph_config.context_bytes =
            graph_config
                .context_bytes
                .max(GgmlCpuGraphConfig::metadata_context_bytes(
                    graph_config.graph_size,
                ));

        let mut runner = GgmlCpuGraphRunner::new(graph_config).expect("build graph runner");
        let loaded = runner
            .load_gguf_weight_context(&pack_path)
            .expect("load gguf weight context");

        encode_with_layer_taps(
            &mut runner,
            Some(&loaded),
            &weights,
            config,
            mel.data(),
            MEL_TARGET_FRAMES,
        )
        .expect("encode_with_layer_taps")
    }

    /// Layer-by-layer CPU-vs-Metal bisection: prints a cosine/max-abs-diff
    /// table for the subsample stem + every transformer layer + the final
    /// `encoder_out`, so the first layer that decorrelates is visible by eye.
    /// `--nocapture` only, not a pass/fail gate by itself -- see
    /// `metal_encoder_matches_cpu_reference` below for the actual assertion.
    #[test]
    #[ignore = "manual real-pack Metal bisection harness: set OPENASR_MOSS_TD_ENCODER_REAL_PACK to the real moss-transcribe-diarize .oasr pack; requires a Metal device"]
    fn dump_cpu_vs_metal_layer_cosine_table() {
        let cpu = run_tapped_encoder(GgmlCpuGraphBackend::Cpu);
        let metal = run_tapped_encoder(GgmlCpuGraphBackend::Metal);

        assert_eq!(cpu.d_model, metal.d_model);
        assert_eq!(cpu.layer_rows.len(), metal.layer_rows.len());
        let dim = cpu.d_model;

        let subsample_cosine =
            mean_frame_cosine_similarity(&metal.subsample_rows, &cpu.subsample_rows, dim);
        eprintln!(
            "moss encoder cpu-vs-metal subsample: cosine={subsample_cosine:.6} max_abs_diff={:.6}",
            max_abs_diff(&metal.subsample_rows, &cpu.subsample_rows)
        );

        for (idx, (metal_rows, cpu_rows)) in metal
            .layer_rows
            .iter()
            .zip(cpu.layer_rows.iter())
            .enumerate()
        {
            let cosine = mean_frame_cosine_similarity(metal_rows, cpu_rows, dim);
            eprintln!(
                "moss encoder cpu-vs-metal layer[{idx:02}]: cosine={cosine:.6} max_abs_diff={:.6}",
                max_abs_diff(metal_rows, cpu_rows)
            );
        }

        let final_cosine =
            mean_frame_cosine_similarity(&metal.encoder_out_rows, &cpu.encoder_out_rows, dim);
        eprintln!(
            "moss encoder cpu-vs-metal encoder_out: cosine={final_cosine:.6} max_abs_diff={:.6}",
            max_abs_diff(&metal.encoder_out_rows, &cpu.encoder_out_rows)
        );
    }

    /// The actual regression gate this defect's fix must satisfy (see
    /// `arch/mod.rs`'s `ExceptMetal` doc comment): once fixed, flip
    /// `auto_gpu_policy` back to `AllBackends` and un-ignore this test.
    ///
    /// Status: the Metal leg is now reachable (previously it silently
    /// downgraded to CPU under the family's own `ExceptMetal` gate even when
    /// driven via `BACKEND_ENV`, so it never actually ran on Metal -- fixed
    /// by installing an explicit `Accelerated` request override, see this
    /// module's `run_tapped_encoder`) and currently PASSES on this host: on
    /// `fixtures/jfk.wav`'s 30s chunk, `encoder_out` cosine = 0.999542
    /// (threshold 0.999); the per-layer breakdown from
    /// `dump_cpu_vs_metal_layer_cosine_table` shows layer 7 onward carrying
    /// a large but cosine-preserving `max_abs_diff` (a few late-layer
    /// activations grow to a magnitude in the single digits, then a shared
    /// scale factor keeps direction -- hence cosine, not max-abs-diff, is
    /// the pass/fail metric here) and layer 23 is the sharpest single-layer
    /// drop (cosine 0.999128) before recovering slightly at `encoder_out`.
    ///
    /// This does NOT flip `auto_gpu_policy` to `AllBackends` on its own --
    /// that is a separate, explicitly user-gated decision (this pass is one
    /// data point: one host, one 30s clip; the `en_zh_mixed`-clip e2e smoke
    /// in `executor.rs` found a small but real divergence at the decoded-
    /// text level on a different clip, so do not read this test alone as
    /// "Metal is safe to default to").
    ///
    /// Kept `#[ignore]` regardless of the above: it needs a private,
    /// uncommitted dev-only `.oasr` pack (`real_pack_path` panics without
    /// `OPENASR_MOSS_TD_ENCODER_REAL_PACK` set) and a real Metal device,
    /// neither of which CI has -- there is no small synthetic fixture this
    /// could run against instead (the encoder graph needs real converted
    /// checkpoint weights, not a hand-built tensor). Run locally with:
    ///
    /// ```text
    /// OPENASR_MOSS_TD_ENCODER_REAL_PACK=/path/to/moss-transcribe-diarize-fp16.oasr \
    ///   cargo test -p openasr-core --lib \
    ///   moss_transcribe_diarize::encoder_graph::parity_tests::metal_encoder_matches_cpu_reference \
    ///   -- --ignored --nocapture
    /// ```
    ///
    /// Add `OPENASR_MOSS_TD_ENCODER_REAL_WAV=/path/to/clip.wav` (and
    /// optionally `OPENASR_MOSS_TD_ENCODER_CHUNK_OFFSET_SECS=<secs>`) to
    /// retest against a different clip or a later 30s window of a long one.
    #[test]
    #[ignore = "requires a private dev-only moss-transcribe-diarize .oasr pack via \
                OPENASR_MOSS_TD_ENCODER_REAL_PACK and a real Metal device -- see the doc \
                comment above for current pass/fail status and the local run command"]
    fn metal_encoder_matches_cpu_reference() {
        let cpu = run_tapped_encoder(GgmlCpuGraphBackend::Cpu);
        let metal = run_tapped_encoder(GgmlCpuGraphBackend::Metal);
        let dim = cpu.d_model;
        let cosine =
            mean_frame_cosine_similarity(&metal.encoder_out_rows, &cpu.encoder_out_rows, dim);
        assert!(
            cosine > 0.999,
            "moss encoder_out cpu-vs-metal cosine too low: {cosine:.6} (see arch/mod.rs's \
             ExceptMetal doc comment; run dump_cpu_vs_metal_layer_cosine_table for the per-layer \
             breakdown)"
        );
    }
}
