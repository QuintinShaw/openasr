use std::path::Path;
use std::time::Instant;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena, env_var_truthy,
};
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation,
    reshape_bias_4d as nn_reshape_bias_4d,
};
use crate::nn::encoder::{
    TransformerEncoderConfig, TransformerEncoderLayerWeights, transformer_layer,
};
use crate::nn::ffn::FeedForwardActivation;
use crate::{GgufTensorDataReadError, GgufTensorDataReader, GgufTensorMetadata};

use super::frontend::Qwen3AsrMelFeatures;
use super::graph_config::qwen_encoder_graph_config;
use super::runtime_contract::Qwen3AsrExecutionMetadata;

/// Env flag to emit a per-chunk audio-encoder timing split. `setup_us` covers
/// graph build/upload; `compute_us` covers the GPU graph compute. Used to
/// confirm whether the longform encoder bottleneck is upload/setup or compute.
const QWEN3_ENCODER_PROFILE_ENV: &str = "OPENASR_QWEN_ENCODER_PROFILE";

/// Opt-out escape hatch for `transformer_layer`'s flash-attention branch in
/// the qwen audio encoder self-attention, mirrored from the Whisper encoder's
/// `OPENASR_WHISPER_GGML_DISABLE_ENCODER_FLASH_ATTN` convention
/// (`models/whisper/execution_policy.rs`): fused softmax/scale on a trusted
/// backend (Metal/CPU), no materialized `[seq, seq]` scores tensor, default
/// on.
const OPENASR_QWEN_GGML_DISABLE_AUDIO_ENCODER_FLASH_ATTN: &str =
    "OPENASR_QWEN_GGML_DISABLE_AUDIO_ENCODER_FLASH_ATTN";

fn qwen_audio_encoder_flash_attention_enabled() -> bool {
    std::env::var_os(OPENASR_QWEN_GGML_DISABLE_AUDIO_ENCODER_FLASH_ATTN).is_none()
}
use super::tensor_names::{
    AUDIO_CONV_OUT_BIAS, AUDIO_CONV_OUT_WEIGHT, AUDIO_CONV1_BIAS, AUDIO_CONV1_WEIGHT,
    AUDIO_CONV2_BIAS, AUDIO_CONV2_WEIGHT, AUDIO_CONV3_BIAS, AUDIO_CONV3_WEIGHT, AUDIO_LN_POST_BIAS,
    AUDIO_LN_POST_WEIGHT, AUDIO_PROJ1_BIAS, AUDIO_PROJ1_WEIGHT, AUDIO_PROJ2_BIAS,
    AUDIO_PROJ2_WEIGHT, audio_layer_tensor_names,
};

const QWEN3_AUDIO_CHUNK_FRAMES: usize = 100;
const QWEN3_AUDIO_CONV_KERNEL: usize = 3;
const QWEN3_AUDIO_CONV_STRIDE: usize = 2;
const QWEN3_AUDIO_CONV_PADDING: usize = 1;
const QWEN3_AUDIO_CONV_DILATION: usize = 1;
const QWEN3_AUDIO_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const QWEN3_AUDIO_GRAPH_CONTEXT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrAudioEncoderOutput {
    pub row_count: usize,
    pub rows: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrAudioEncoderError {
    #[error("qwen3-asr audio encoder could not read GGUF tensor '{tensor_name}': {source}")]
    TensorRead {
        tensor_name: String,
        #[source]
        source: Box<GgufTensorDataReadError>,
    },
    #[error("qwen3-asr audio encoder tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: String,
        shape: String,
        reason: String,
    },
    #[error("qwen3-asr audio encoder mel features are invalid: {reason}")]
    InvalidMelFeatures { reason: String },
    #[error("qwen3-asr audio encoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("qwen3-asr audio encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("qwen3-asr audio encoder shape overflowed")]
    ShapeOverflow,
}

#[derive(Debug, Clone)]
struct F32Tensor {
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AudioLayerWeights {
    attn_norm_weight: F32Tensor,
    attn_norm_bias: F32Tensor,
    attn_q_weight: F32Tensor,
    attn_q_bias: F32Tensor,
    attn_k_weight: F32Tensor,
    attn_k_bias: F32Tensor,
    attn_v_weight: F32Tensor,
    attn_v_bias: F32Tensor,
    attn_out_weight: F32Tensor,
    attn_out_bias: F32Tensor,
    ffn_norm_weight: F32Tensor,
    ffn_norm_bias: F32Tensor,
    ffn_up_weight: F32Tensor,
    ffn_up_bias: F32Tensor,
    ffn_down_weight: F32Tensor,
    ffn_down_bias: F32Tensor,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrAudioEncoderWeights {
    conv1_weight: F32Tensor,
    conv1_bias: F32Tensor,
    conv2_weight: F32Tensor,
    conv2_bias: F32Tensor,
    conv3_weight: F32Tensor,
    conv3_bias: F32Tensor,
    conv_out_weight: F32Tensor,
    conv_out_bias: Option<F32Tensor>,
    ln_post_weight: F32Tensor,
    ln_post_bias: F32Tensor,
    proj1_weight: F32Tensor,
    proj1_bias: F32Tensor,
    proj2_weight: F32Tensor,
    proj2_bias: F32Tensor,
    layers: Vec<AudioLayerWeights>,
    conv_channels: usize,
    conv_out_freq_bins: usize,
}

impl Qwen3AsrAudioEncoderWeights {
    /// The number of transformer-encoder layers materialized from the GGUF — the
    /// count the composer walks. Cross-checked against the block-stack
    /// descriptor's `qwen3-asr.audio.n_layers` at executor construction (P4 S5d).
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[cfg(test)]
    pub(crate) fn zero_copy_audio_projection_payloads_dropped_for_test(&self) -> bool {
        self.conv_out_weight.values.is_empty()
            && self.proj1_weight.values.is_empty()
            && self.proj2_weight.values.is_empty()
            && self.layers.iter().all(|layer| {
                layer.attn_q_weight.values.is_empty()
                    && layer.attn_k_weight.values.is_empty()
                    && layer.attn_v_weight.values.is_empty()
                    && layer.attn_out_weight.values.is_empty()
                    && layer.ffn_up_weight.values.is_empty()
                    && layer.ffn_down_weight.values.is_empty()
            })
    }
}

pub(crate) struct Qwen3AsrAudioEncoderRuntime {
    runner: GgmlCpuGraphRunner,
    loaded: Option<GgmlLoadedWeightContext>,
}

impl Qwen3AsrAudioEncoderRuntime {
    pub(crate) fn new(runtime_path: Option<&Path>) -> Result<Self, Qwen3AsrAudioEncoderError> {
        // See `qwen_encoder_graph_config` for the `EncoderPrelude` threading
        // tier rationale.
        let mut config = qwen_encoder_graph_config();
        config.context_bytes = QWEN3_AUDIO_GRAPH_CONTEXT_BYTES;
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            Qwen3AsrAudioEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        // goals 7+8 Step 1: bind the encoder's 2D projection weights zero-copy from
        // the mmap'd pack (native q8/f16) instead of dequantizing them to f32. The
        // loader (1b) does not materialize f32 for these — `loaded` is the only
        // source. `None` (no path) only happens off the production executor path.
        let loaded = runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        Ok(Self { runner, loaded })
    }

    pub(crate) fn encode(
        &mut self,
        weights: &Qwen3AsrAudioEncoderWeights,
        metadata: Qwen3AsrExecutionMetadata,
        mel_features: &Qwen3AsrMelFeatures,
    ) -> Result<Qwen3AsrAudioEncoderOutput, Qwen3AsrAudioEncoderError> {
        validate_mel_features(metadata, mel_features)?;
        let profile_started = env_var_truthy(QWEN3_ENCODER_PROFILE_ENV).then(Instant::now);
        let chunked_mel = pack_mel_into_chunked_layout(mel_features)?;
        let positional = build_audio_positional_embedding(
            metadata.audio_d_model,
            chunked_mel.chunk_output_frames,
        )?;
        let mask_len = chunked_mel
            .row_count
            .checked_mul(chunked_mel.row_count)
            .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
        let mask = vec![0.0_f32; mask_len];

        let loaded = self.loaded.as_ref();

        // Resident encoder weights (conv stem, every 1D norm/bias, and any 2D
        // projection the runtime pack did not materialize for a zero-copy bind)
        // live in a WEIGHTS-usage arena buffer instead of per-call graph-input
        // leaves. ggml's multi-backend scheduler only offloads an op to the
        // accelerator when its weight `src` is in a WEIGHTS-usage buffer, so the
        // conv stem and its norms used to pin the encoder front to the CPU even
        // under an explicit Metal backend (the 2D matmul projections already bind
        // zero-copy from the mmap'd pack, `loaded`). Only the mel input and the
        // per-chunk-derived positional/mask tensors stay genuine graph inputs.
        // Mirrors the sibling dolphin/cohere/xasr encoders; the uploaded byte
        // content and op order are unchanged, so the output stays golden.
        let mut arena = self
            .runner
            .start_static_tensor_arena(qwen_audio_encoder_arena_context_bytes(weights))
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;
        let resident = build_qwen_audio_resident_weights(&mut arena, weights, loaded)?;

        let mut graph = self.runner.start_graph();

        encode_qwen3_audio_embeddings_with_graph(
            &mut graph,
            &resident,
            weights,
            metadata,
            &chunked_mel,
            &positional,
            &mask,
            profile_started,
        )
    }
}

fn encode_qwen3_audio_embeddings_with_graph<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    resident: &QwenAudioResidentTensors<'a>,
    weights: &Qwen3AsrAudioEncoderWeights,
    metadata: Qwen3AsrExecutionMetadata,
    chunked_mel: &ChunkedMelInput,
    positional: &[f32],
    mask: &[f32],
    profile_started: Option<Instant>,
) -> Result<Qwen3AsrAudioEncoderOutput, Qwen3AsrAudioEncoderError> {
    // Per-call graph inputs: the mel features plus the per-chunk-derived
    // positional embedding and attention mask. Every weight is already resident
    // in the arena's WEIGHTS-usage buffer (see
    // `build_qwen_audio_resident_weights`), so weights carry no `set_input` and
    // are referenced through `resident`.
    let mel = graph
        .new_tensor_4d_f32(
            chunked_mel.chunk_frames,
            chunked_mel.n_mels,
            1,
            chunked_mel.num_chunks,
            "qwen_audio_mel",
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_new_tensor_4d(mel)",
            source,
        })?;
    let positional_tensor = graph
        .new_tensor_3d_f32(
            metadata.audio_d_model,
            chunked_mel.chunk_output_frames,
            1,
            "audio_positional",
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_new_tensor_3d(positional)",
            source,
        })?;
    let mask_tensor = graph
        .new_tensor_2d_f32(
            chunked_mel.row_count,
            chunked_mel.row_count,
            "audio_attention_mask",
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_new_tensor_2d(mask)",
            source,
        })?;
    for tensor in [mel, positional_tensor, mask_tensor] {
        graph
            .set_input(tensor)
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
                step: "ggml_set_input(audio_input)",
                source,
            })?;
    }

    let conv1_weight = resident.conv1_weight;
    let conv1_bias = resident.conv1_bias;
    let conv2_weight = resident.conv2_weight;
    let conv2_bias = resident.conv2_bias;
    let conv3_weight = resident.conv3_weight;
    let conv3_bias = resident.conv3_bias;
    let conv_out_weight = resident.conv_out_weight;
    let conv_out_bias = resident.conv_out_bias;
    let ln_post_weight = resident.ln_post_weight;
    let ln_post_bias = resident.ln_post_bias;
    let proj1_weight = resident.proj1_weight;
    let proj1_bias = resident.proj1_bias;
    let proj2_weight = resident.proj2_weight;
    let proj2_bias = resident.proj2_bias;

    let map_graph_error =
        |step, source| Qwen3AsrAudioEncoderError::GraphBuildFailed { step, source };
    let conv_params = Conv2dParams {
        stride_x: QWEN3_AUDIO_CONV_STRIDE,
        stride_y: QWEN3_AUDIO_CONV_STRIDE,
        padding_x: QWEN3_AUDIO_CONV_PADDING,
        padding_y: QWEN3_AUDIO_CONV_PADDING,
        dilation_x: QWEN3_AUDIO_CONV_DILATION,
        dilation_y: QWEN3_AUDIO_CONV_DILATION,
    };
    let conv1_bias_4d = nn_reshape_bias_4d(
        graph,
        conv1_bias,
        weights.conv1_bias.values.len(),
        "ggml_reshape_4d(conv_bias)",
        map_graph_error,
    )?;
    let conv2_bias_4d = nn_reshape_bias_4d(
        graph,
        conv2_bias,
        weights.conv2_bias.values.len(),
        "ggml_reshape_4d(conv_bias)",
        map_graph_error,
    )?;
    let conv3_bias_4d = nn_reshape_bias_4d(
        graph,
        conv3_bias,
        weights.conv3_bias.values.len(),
        "ggml_reshape_4d(conv_bias)",
        map_graph_error,
    )?;

    let mut state = apply_conv_2d_bias_activation(
        graph,
        conv1_weight,
        mel,
        conv1_bias_4d,
        conv_params,
        ConvActivation::GeluErf,
        ConvBlockSteps {
            conv: "ggml_conv_2d(conv1)",
            bias: "ggml_add(conv1_bias)",
            activation: "ggml_gelu(conv1)",
        },
        map_graph_error,
    )?;
    state = apply_conv_2d_bias_activation(
        graph,
        conv2_weight,
        state,
        conv2_bias_4d,
        conv_params,
        ConvActivation::GeluErf,
        ConvBlockSteps {
            conv: "ggml_conv_2d(conv2)",
            bias: "ggml_add(conv2_bias)",
            activation: "ggml_gelu(conv2)",
        },
        map_graph_error,
    )?;
    state = apply_conv_2d_bias_activation(
        graph,
        conv3_weight,
        state,
        conv3_bias_4d,
        conv_params,
        ConvActivation::GeluErf,
        ConvBlockSteps {
            conv: "ggml_conv_2d(conv3)",
            bias: "ggml_add(conv3_bias)",
            activation: "ggml_gelu(conv3)",
        },
        map_graph_error,
    )?;

    state = graph.permute(state, 2, 0, 1, 3).map_err(|source| {
        Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_permute(conv_front)",
            source,
        }
    })?;
    state = graph
        .cont(state)
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_cont(conv_front)",
            source,
        })?;
    state = graph
        .reshape_3d(
            state,
            weights.conv_channels * weights.conv_out_freq_bins,
            chunked_mel.chunk_output_frames,
            chunked_mel.num_chunks,
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_reshape_3d(conv_front)",
            source,
        })?;
    state = graph.mul_mat(conv_out_weight, state).map_err(|source| {
        Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_mul_mat(conv_out)",
            source,
        }
    })?;
    if let Some(conv_out_bias) = conv_out_bias {
        state = graph.add(state, conv_out_bias).map_err(|source| {
            Qwen3AsrAudioEncoderError::GraphBuildFailed {
                step: "ggml_add(conv_out_bias)",
                source,
            }
        })?;
    }

    state = graph.add(state, positional_tensor).map_err(|source| {
        Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_add(audio_positional)",
            source,
        }
    })?;
    state = graph
        .reshape_2d(state, metadata.audio_d_model, chunked_mel.row_count)
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_reshape_2d(audio_sequence)",
            source,
        })?;

    let head_dim = metadata.audio_d_model / metadata.audio_heads;
    state = compose_transformer_encoder_layer_stack(
        graph,
        state,
        &resident.layers,
        head_dim,
        metadata.audio_heads,
        chunked_mel.row_count,
        mask_tensor,
        qwen_audio_encoder_flash_attention_enabled(),
    )?;

    state = graph
        .norm(state, QWEN3_AUDIO_LAYER_NORM_EPSILON)
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_norm(audio_ln_post)",
            source,
        })?;
    state = graph.mul(state, ln_post_weight).map_err(|source| {
        Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_mul(audio_ln_post_weight)",
            source,
        }
    })?;
    state = graph.add(state, ln_post_bias).map_err(|source| {
        Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_add(audio_ln_post_bias)",
            source,
        }
    })?;
    state = graph
        .add(
            graph.mul_mat(proj1_weight, state).map_err(|source| {
                Qwen3AsrAudioEncoderError::GraphBuildFailed {
                    step: "ggml_mul_mat(audio_proj1)",
                    source,
                }
            })?,
            proj1_bias,
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_add(audio_proj1_bias)",
            source,
        })?;
    state =
        graph
            .gelu_erf(state)
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
                step: "ggml_gelu(audio_proj1)",
                source,
            })?;
    state = graph
        .add(
            graph.mul_mat(proj2_weight, state).map_err(|source| {
                Qwen3AsrAudioEncoderError::GraphBuildFailed {
                    step: "ggml_mul_mat(audio_proj2)",
                    source,
                }
            })?,
            proj2_bias,
        )
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_add(audio_proj2_bias)",
            source,
        })?;
    graph
        .set_output(state)
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_set_output(audio_encoder_out)",
            source,
        })?;

    // Peak-RSS lever: allocate the audio-encoder compute graph via the scheduler's
    // gallocr (liveness-based buffer REUSE) before uploading inputs/weights, so the
    // per-layer intermediates collapse to the working-set peak instead of each
    // getting its own buffer (alloc_ctx_tensors). The full graph is built above.
    graph
        .prepare_outputs_for_upload(&[state])
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed {
            step: "ggml_prepare_outputs(audio_encoder_out)",
            source,
        })?;

    // Only the genuine per-call inputs are uploaded here; every weight already
    // resides in the arena's WEIGHTS-usage buffer (uploaded once in
    // `build_qwen_audio_resident_weights`).
    graph
        .set_f32_slice(mel, &chunked_mel.values, "qwen_audio_mel")
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
            reason: format!("could not upload mel features: {source}"),
        })?;
    graph
        .set_f32_slice(positional_tensor, positional, "audio_positional")
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
            reason: format!("could not upload positional embedding: {source}"),
        })?;
    graph
        .set_f32_slice(mask_tensor, mask, "audio_attention_mask")
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
            reason: format!("could not upload attention mask: {source}"),
        })?;

    let compute_started = profile_started.map(|_| Instant::now());
    let values = graph
        .compute_output_f32(state, metadata.llm_d_model * chunked_mel.row_count)
        .map_err(|source| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
            reason: format!("audio encoder graph compute failed: {source}"),
        })?;
    if let (Some(started), Some(compute_started)) = (profile_started, compute_started) {
        let total_us = started.elapsed().as_micros();
        let compute_us = compute_started.elapsed().as_micros();
        eprintln!(
            "openasr_qwen_encoder_profile: rows={} setup_us={} compute_us={}",
            chunked_mel.row_count,
            total_us.saturating_sub(compute_us),
            compute_us,
        );
    }
    Ok(Qwen3AsrAudioEncoderOutput {
        row_count: chunked_mel.row_count,
        rows: values,
    })
}

#[derive(Debug, Clone)]
struct ChunkedMelInput {
    chunk_frames: usize,
    chunk_output_frames: usize,
    num_chunks: usize,
    n_mels: usize,
    row_count: usize,
    values: Vec<f32>,
}

#[derive(Clone, Copy)]
struct AudioLayerGraphTensors<'a> {
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

fn validate_mel_features(
    metadata: Qwen3AsrExecutionMetadata,
    mel_features: &Qwen3AsrMelFeatures,
) -> Result<(), Qwen3AsrAudioEncoderError> {
    if mel_features.n_mels != metadata.n_mels {
        return Err(Qwen3AsrAudioEncoderError::InvalidMelFeatures {
            reason: format!(
                "n_mels mismatch: got {}, expected {}",
                mel_features.n_mels, metadata.n_mels
            ),
        });
    }
    if mel_features.n_frames == 0 {
        return Err(Qwen3AsrAudioEncoderError::InvalidMelFeatures {
            reason: "mel frame count must be > 0".to_string(),
        });
    }
    let expected_len = mel_features
        .n_mels
        .checked_mul(mel_features.n_frames)
        .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
    if mel_features.data.len() != expected_len {
        return Err(Qwen3AsrAudioEncoderError::InvalidMelFeatures {
            reason: format!(
                "mel value count mismatch: got {}, expected {}",
                mel_features.data.len(),
                expected_len
            ),
        });
    }
    if mel_features.data.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrAudioEncoderError::InvalidMelFeatures {
            reason: "mel features contain non-finite values".to_string(),
        });
    }
    Ok(())
}

fn pack_mel_into_chunked_layout(
    mel_features: &Qwen3AsrMelFeatures,
) -> Result<ChunkedMelInput, Qwen3AsrAudioEncoderError> {
    let num_chunks = mel_features.n_frames.div_ceil(QWEN3_AUDIO_CHUNK_FRAMES);
    let chunk_output_frames = conv_out_len(conv_out_len(conv_out_len(QWEN3_AUDIO_CHUNK_FRAMES)));
    let row_count = num_chunks
        .checked_mul(chunk_output_frames)
        .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
    let value_len = QWEN3_AUDIO_CHUNK_FRAMES
        .checked_mul(mel_features.n_mels)
        .and_then(|value| value.checked_mul(num_chunks))
        .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
    let mut values = vec![0.0_f32; value_len];
    for chunk_idx in 0..num_chunks {
        let frame_start = chunk_idx * QWEN3_AUDIO_CHUNK_FRAMES;
        let frame_end = (frame_start + QWEN3_AUDIO_CHUNK_FRAMES).min(mel_features.n_frames);
        let chunk_len = frame_end.saturating_sub(frame_start);
        for mel_idx in 0..mel_features.n_mels {
            for t in 0..chunk_len {
                let src = mel_idx
                    .checked_mul(mel_features.n_frames)
                    .and_then(|value| value.checked_add(frame_start + t))
                    .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
                let dst = t
                    .checked_add(
                        QWEN3_AUDIO_CHUNK_FRAMES
                            .checked_mul(
                                mel_idx
                                    .checked_add(mel_features.n_mels * chunk_idx)
                                    .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?,
                            )
                            .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?,
                    )
                    .ok_or(Qwen3AsrAudioEncoderError::ShapeOverflow)?;
                values[dst] = mel_features.data[src];
            }
        }
    }
    Ok(ChunkedMelInput {
        chunk_frames: QWEN3_AUDIO_CHUNK_FRAMES,
        chunk_output_frames,
        num_chunks,
        n_mels: mel_features.n_mels,
        row_count,
        values,
    })
}

fn conv_out_len(input: usize) -> usize {
    (input + 2 * QWEN3_AUDIO_CONV_PADDING - QWEN3_AUDIO_CONV_KERNEL) / QWEN3_AUDIO_CONV_STRIDE + 1
}

fn build_audio_positional_embedding(
    d_model: usize,
    positions: usize,
) -> Result<Vec<f32>, Qwen3AsrAudioEncoderError> {
    if d_model == 0 || positions == 0 || !d_model.is_multiple_of(2) {
        return Err(Qwen3AsrAudioEncoderError::InvalidMelFeatures {
            reason: format!("audio positional embedding requires even d_model > 0, got {d_model}"),
        });
    }
    let half = d_model / 2;
    let log_inc = 10000.0_f32.ln() / (half.saturating_sub(1).max(1) as f32);
    let mut inv_t = Vec::with_capacity(half);
    for index in 0..half {
        inv_t.push((-log_inc * index as f32).exp());
    }
    let mut values = vec![0.0_f32; d_model * positions];
    for position in 0..positions {
        let row = &mut values[position * d_model..(position + 1) * d_model];
        for index in 0..half {
            let angle = position as f32 * inv_t[index];
            row[index] = angle.sin();
            row[half + index] = angle.cos();
        }
    }
    Ok(values)
}

pub(crate) fn load_qwen3_audio_encoder_weights_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Qwen3AsrAudioEncoderWeights, Qwen3AsrAudioEncoderError> {
    let index = reader.tensor_index();
    let conv1_weight = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV1_WEIGHT)?)?;
    let conv1_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV1_BIAS)?)?;
    let conv2_weight = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV2_WEIGHT)?)?;
    let conv2_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV2_BIAS)?)?;
    let conv3_weight = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV3_WEIGHT)?)?;
    let conv3_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_CONV3_BIAS)?)?;
    let conv_out_weight = load_tensor_meta_only(require_tensor(index, AUDIO_CONV_OUT_WEIGHT)?);
    let conv_out_bias = index
        .get(AUDIO_CONV_OUT_BIAS)
        .map(|tensor| load_tensor_f32(reader, tensor))
        .transpose()?;
    let ln_post_weight = load_tensor_f32(reader, require_tensor(index, AUDIO_LN_POST_WEIGHT)?)?;
    let ln_post_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_LN_POST_BIAS)?)?;
    let proj1_weight = load_tensor_meta_only(require_tensor(index, AUDIO_PROJ1_WEIGHT)?);
    let proj1_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_PROJ1_BIAS)?)?;
    let proj2_weight = load_tensor_meta_only(require_tensor(index, AUDIO_PROJ2_WEIGHT)?);
    let proj2_bias = load_tensor_f32(reader, require_tensor(index, AUDIO_PROJ2_BIAS)?)?;

    validate_tensor_rank(&conv1_weight, 4, "expected rank-4 conv2d kernel")?;
    validate_tensor_rank(&conv2_weight, 4, "expected rank-4 conv2d kernel")?;
    validate_tensor_rank(&conv3_weight, 4, "expected rank-4 conv2d kernel")?;
    validate_vector_len(&conv1_bias, conv1_weight.dims[3] as usize)?;
    validate_vector_len(&conv2_bias, conv2_weight.dims[3] as usize)?;
    validate_vector_len(&conv3_bias, conv3_weight.dims[3] as usize)?;

    let conv_channels = usize::try_from(conv3_weight.dims[3]).map_err(|_| {
        Qwen3AsrAudioEncoderError::InvalidTensorShape {
            tensor_name: conv3_weight.name.clone(),
            shape: render_shape(&conv3_weight.dims),
            reason: "conv output channels exceed usize".to_string(),
        }
    })?;
    let conv_out_freq_bins = conv_out_len(conv_out_len(conv_out_len(metadata.n_mels)));
    validate_matrix_shape(
        &conv_out_weight,
        conv_channels * conv_out_freq_bins,
        metadata.audio_d_model,
    )?;
    if let Some(conv_out_bias) = conv_out_bias.as_ref() {
        validate_vector_len(conv_out_bias, metadata.audio_d_model)?;
    }
    validate_vector_len(&ln_post_weight, metadata.audio_d_model)?;
    validate_vector_len(&ln_post_bias, metadata.audio_d_model)?;
    validate_matrix_shape(
        &proj1_weight,
        metadata.audio_d_model,
        metadata.audio_d_model,
    )?;
    validate_vector_len(&proj1_bias, metadata.audio_d_model)?;
    validate_matrix_shape(&proj2_weight, metadata.audio_d_model, metadata.llm_d_model)?;
    validate_vector_len(&proj2_bias, metadata.llm_d_model)?;

    let hidden = metadata.audio_d_model;
    let mut layers = Vec::with_capacity(metadata.audio_layers);
    for layer_idx in 0..metadata.audio_layers {
        let names = audio_layer_tensor_names(layer_idx);
        let attn_norm_weight =
            load_tensor_f32(reader, require_tensor(index, &names.attn_norm_weight)?)?;
        let attn_norm_bias =
            load_tensor_f32(reader, require_tensor(index, &names.attn_norm_bias)?)?;
        // 2D projections: metadata-only (Step 1b) — bound zero-copy in encode,
        // never materialized to f32. 1D biases/norms stay f32 (tiny).
        let attn_q_weight = load_tensor_meta_only(require_tensor(index, &names.attn_q_weight)?);
        let attn_q_bias = load_tensor_f32(reader, require_tensor(index, &names.attn_q_bias)?)?;
        let attn_k_weight = load_tensor_meta_only(require_tensor(index, &names.attn_k_weight)?);
        let attn_k_bias = load_tensor_f32(reader, require_tensor(index, &names.attn_k_bias)?)?;
        let attn_v_weight = load_tensor_meta_only(require_tensor(index, &names.attn_v_weight)?);
        let attn_v_bias = load_tensor_f32(reader, require_tensor(index, &names.attn_v_bias)?)?;
        let attn_out_weight = load_tensor_meta_only(require_tensor(index, &names.attn_out_weight)?);
        let attn_out_bias = load_tensor_f32(reader, require_tensor(index, &names.attn_out_bias)?)?;
        let ffn_norm_weight =
            load_tensor_f32(reader, require_tensor(index, &names.ffn_norm_weight)?)?;
        let ffn_norm_bias = load_tensor_f32(reader, require_tensor(index, &names.ffn_norm_bias)?)?;
        let ffn_up_weight = load_tensor_meta_only(require_tensor(index, &names.ffn_up_weight)?);
        let ffn_up_bias = load_tensor_f32(reader, require_tensor(index, &names.ffn_up_bias)?)?;
        let ffn_down_weight = load_tensor_meta_only(require_tensor(index, &names.ffn_down_weight)?);
        let ffn_down_bias = load_tensor_f32(reader, require_tensor(index, &names.ffn_down_bias)?)?;

        validate_vector_len(&attn_norm_weight, hidden)?;
        validate_vector_len(&attn_norm_bias, hidden)?;
        validate_matrix_shape(&attn_q_weight, hidden, hidden)?;
        validate_vector_len(&attn_q_bias, hidden)?;
        validate_matrix_shape(&attn_k_weight, hidden, hidden)?;
        validate_vector_len(&attn_k_bias, hidden)?;
        validate_matrix_shape(&attn_v_weight, hidden, hidden)?;
        validate_vector_len(&attn_v_bias, hidden)?;
        validate_matrix_shape(&attn_out_weight, hidden, hidden)?;
        validate_vector_len(&attn_out_bias, hidden)?;
        let ffn_hidden = usize::try_from(ffn_up_weight.dims[1]).map_err(|_| {
            Qwen3AsrAudioEncoderError::InvalidTensorShape {
                tensor_name: ffn_up_weight.name.clone(),
                shape: render_shape(&ffn_up_weight.dims),
                reason: "ffn_up output width exceeds usize".to_string(),
            }
        })?;
        validate_vector_len(&ffn_norm_weight, hidden)?;
        validate_vector_len(&ffn_norm_bias, hidden)?;
        validate_matrix_shape(&ffn_up_weight, hidden, ffn_hidden)?;
        validate_vector_len(&ffn_up_bias, ffn_hidden)?;
        validate_matrix_shape(&ffn_down_weight, ffn_hidden, hidden)?;
        validate_vector_len(&ffn_down_bias, hidden)?;

        layers.push(AudioLayerWeights {
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
        });
    }

    Ok(Qwen3AsrAudioEncoderWeights {
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        conv3_weight,
        conv3_bias,
        conv_out_weight,
        conv_out_bias,
        ln_post_weight,
        ln_post_bias,
        proj1_weight,
        proj1_bias,
        proj2_weight,
        proj2_bias,
        layers,
        conv_channels,
        conv_out_freq_bins,
    })
}

/// Walk the qwen audio transformer encoder layer stack, emitting one
/// `nn::encoder::transformer_layer` block per resident layer and chaining
/// `state` through them. This is the TransformerEncoderLayer-stage assembly
/// (P4 data-driven): the layer COUNT comes from `layers` (built from the
/// `qwen3-asr.audio.n_layers` hparam) and the block KIND is `transformer_layer`
/// (the descriptor's `TransformerEncoderLayer`). The encoder mirror of the
/// LlmDecoder composer (S2); the emitted op sequence is unchanged.
#[allow(clippy::too_many_arguments)]
fn compose_transformer_encoder_layer_stack<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    mut state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    layers: &[AudioLayerGraphTensors<'a>],
    head_dim: usize,
    attention_heads: usize,
    token_count: usize,
    mask: crate::ggml_runtime::GgmlCpuTensor<'a>,
    use_flash_attention: bool,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, Qwen3AsrAudioEncoderError> {
    for tensors in layers {
        state = run_audio_encoder_layer(
            graph,
            state,
            head_dim,
            attention_heads,
            token_count,
            mask,
            use_flash_attention,
            tensors,
        )?;
    }
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn run_audio_encoder_layer<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: crate::ggml_runtime::GgmlCpuTensor<'a>,
    head_dim: usize,
    attention_heads: usize,
    token_count: usize,
    mask: crate::ggml_runtime::GgmlCpuTensor<'a>,
    use_flash_attention: bool,
    tensors: &AudioLayerGraphTensors<'a>,
) -> Result<crate::ggml_runtime::GgmlCpuTensor<'a>, Qwen3AsrAudioEncoderError> {
    transformer_layer(
        graph,
        state,
        mask,
        TransformerEncoderConfig {
            head_dim,
            attention_heads,
            token_count,
            layer_norm_epsilon: QWEN3_AUDIO_LAYER_NORM_EPSILON,
            ffn_activation: FeedForwardActivation::GeluErf,
            use_flash_attention,
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
        |step, source| Qwen3AsrAudioEncoderError::GraphBuildFailed { step, source },
    )
}

/// Resident audio-encoder weight tensors, all living in the arena's
/// WEIGHTS-usage backend buffer (the 2D projections either bound zero-copy from
/// the mmap'd pack or, when the pack did not materialize them, f32-uploaded into
/// the same arena). Only the per-call mel/positional/mask leaves are graph
/// inputs; every field here is offload-eligible for the ggml scheduler.
struct QwenAudioResidentTensors<'a> {
    conv1_weight: GgmlCpuTensor<'a>,
    conv1_bias: GgmlCpuTensor<'a>,
    conv2_weight: GgmlCpuTensor<'a>,
    conv2_bias: GgmlCpuTensor<'a>,
    conv3_weight: GgmlCpuTensor<'a>,
    conv3_bias: GgmlCpuTensor<'a>,
    conv_out_weight: GgmlCpuTensor<'a>,
    conv_out_bias: Option<GgmlCpuTensor<'a>>,
    ln_post_weight: GgmlCpuTensor<'a>,
    ln_post_bias: GgmlCpuTensor<'a>,
    proj1_weight: GgmlCpuTensor<'a>,
    proj1_bias: GgmlCpuTensor<'a>,
    proj2_weight: GgmlCpuTensor<'a>,
    proj2_bias: GgmlCpuTensor<'a>,
    layers: Vec<AudioLayerGraphTensors<'a>>,
}

/// Upper bound on the arena's metadata context: the fixed conv stem / output
/// head tensors plus the worst-case per-layer tensor count (10 x 1D norm/bias
/// that always land in the arena + up to 6 x 2D projection when the pack did not
/// materialize them for a zero-copy bind). Over-counting only sizes the (cheap)
/// tensor-overhead context; the real weight bytes land in a separately sized
/// backend buffer.
fn qwen_audio_encoder_arena_context_bytes(weights: &Qwen3AsrAudioEncoderWeights) -> usize {
    const FIXED_TENSORS: usize = 16;
    const MAX_TENSORS_PER_LAYER: usize = 16;
    let count =
        FIXED_TENSORS.saturating_add(MAX_TENSORS_PER_LAYER.saturating_mul(weights.layers.len()));
    GgmlCpuGraphConfig::metadata_context_bytes(count)
}

/// Collects `(arena handle, host f32 slice, label)` uploads while every arena
/// tensor is allocated, then flushes them once. Allocation MUST precede the
/// arena's first upload (the first `set_f32_slice` freezes further creation),
/// so callers allocate every tensor first and call [`Self::upload`] last. This
/// is the qwen audio-encoder analogue of dolphin's `WeightBuilder`.
struct QwenAudioArenaBuilder<'w> {
    uploads: Vec<(GgmlStaticTensor, &'w [f32], &'static str)>,
}

impl<'w> QwenAudioArenaBuilder<'w> {
    fn new() -> Self {
        Self {
            uploads: Vec::new(),
        }
    }

    /// A 1D norm/bias: always an f32 arena tensor.
    fn arena_1d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        tensor: &'w F32Tensor,
        step: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, Qwen3AsrAudioEncoderError> {
        if tensor.dims.len() != 1 {
            return Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
                tensor_name: tensor.name.clone(),
                shape: render_shape(&tensor.dims),
                reason: "expected rank-1 tensor".to_string(),
            });
        }
        let handle = arena
            .new_tensor_1d_f32(tensor.dims[0] as usize, step)
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed { step, source })?;
        self.uploads.push((handle, &tensor.values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// A rank-4 conv kernel: always an f32 arena tensor.
    fn arena_4d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        tensor: &'w F32Tensor,
        step: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, Qwen3AsrAudioEncoderError> {
        if tensor.dims.len() != 4 {
            return Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
                tensor_name: tensor.name.clone(),
                shape: render_shape(&tensor.dims),
                reason: "expected rank-4 tensor".to_string(),
            });
        }
        let handle = arena
            .new_tensor_4d_f32(
                tensor.dims[0] as usize,
                tensor.dims[1] as usize,
                tensor.dims[2] as usize,
                tensor.dims[3] as usize,
                step,
            )
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed { step, source })?;
        self.uploads.push((handle, &tensor.values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// A 2D projection: bound zero-copy from the mmap'd pack (`loaded`, native
    /// q8/f16) when present; otherwise f32-uploaded into the arena. Fails closed
    /// if the weight is neither bound nor f32-materialized (1b drops the host
    /// f32 for bound weights, so an unbound+empty one means the pack lacks it --
    /// better to error than bind an empty buffer). The audio-encoder analogue of
    /// cohere's `loaded_or_static`.
    fn loaded_or_arena_2d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        loaded: Option<&GgmlLoadedWeightContext>,
        tensor: &'w F32Tensor,
        step: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, Qwen3AsrAudioEncoderError> {
        if let Some(loaded_tensor) = loaded.and_then(|context| context.tensor(&tensor.name)) {
            return Ok(loaded_tensor.as_graph_tensor());
        }
        if tensor.values.is_empty() {
            return Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
                tensor_name: tensor.name.clone(),
                shape: render_shape(&tensor.dims),
                reason: "2D weight is neither bound zero-copy nor f32-materialized".to_string(),
            });
        }
        if tensor.dims.len() != 2 {
            return Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
                tensor_name: tensor.name.clone(),
                shape: render_shape(&tensor.dims),
                reason: "expected rank-2 tensor".to_string(),
            });
        }
        let handle = arena
            .new_tensor_2d_f32(tensor.dims[0] as usize, tensor.dims[1] as usize, step)
            .map_err(|source| Qwen3AsrAudioEncoderError::GraphBuildFailed { step, source })?;
        self.uploads.push((handle, &tensor.values, step));
        Ok(arena.graph_tensor(handle))
    }

    /// Flush every collected f32 upload into the arena's backend buffer. The
    /// first upload allocates the buffer and freezes further tensor creation, so
    /// this runs after all allocation.
    fn upload(self, arena: &mut GgmlStaticTensorArena) -> Result<(), Qwen3AsrAudioEncoderError> {
        for (handle, values, step) in self.uploads {
            arena
                .set_f32_slice(handle, values, step)
                .map_err(|source| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
                    reason: format!("could not upload arena weight '{step}': {source}"),
                })?;
        }
        Ok(())
    }
}

/// Allocate every resident encoder weight in the arena and upload it once,
/// returning the graph-tensor handles the forward graph references. The conv
/// stem and all 1D norms/biases f32-upload into the WEIGHTS-usage arena buffer;
/// the 2D projections bind zero-copy from the mmap'd pack when present (only
/// falling back to an arena upload when the pack did not materialize them). This
/// is what lets the ggml scheduler offload the encoder front instead of pinning
/// it to the CPU. Mirrors dolphin's `WeightBuilder` two-phase (alloc-all, then
/// upload-once) build.
fn build_qwen_audio_resident_weights<'a>(
    arena: &mut GgmlStaticTensorArena,
    weights: &Qwen3AsrAudioEncoderWeights,
    loaded: Option<&GgmlLoadedWeightContext>,
) -> Result<QwenAudioResidentTensors<'a>, Qwen3AsrAudioEncoderError> {
    let mut builder = QwenAudioArenaBuilder::new();

    let conv1_weight = builder.arena_4d(arena, &weights.conv1_weight, "conv1_weight")?;
    let conv1_bias = builder.arena_1d(arena, &weights.conv1_bias, "conv1_bias")?;
    let conv2_weight = builder.arena_4d(arena, &weights.conv2_weight, "conv2_weight")?;
    let conv2_bias = builder.arena_1d(arena, &weights.conv2_bias, "conv2_bias")?;
    let conv3_weight = builder.arena_4d(arena, &weights.conv3_weight, "conv3_weight")?;
    let conv3_bias = builder.arena_1d(arena, &weights.conv3_bias, "conv3_bias")?;
    let conv_out_weight =
        builder.loaded_or_arena_2d(arena, loaded, &weights.conv_out_weight, "conv_out_weight")?;
    let conv_out_bias = weights
        .conv_out_bias
        .as_ref()
        .map(|tensor| builder.arena_1d(arena, tensor, "conv_out_bias"))
        .transpose()?;
    let ln_post_weight =
        builder.arena_1d(arena, &weights.ln_post_weight, "audio_ln_post_weight")?;
    let ln_post_bias = builder.arena_1d(arena, &weights.ln_post_bias, "audio_ln_post_bias")?;
    let proj1_weight =
        builder.loaded_or_arena_2d(arena, loaded, &weights.proj1_weight, "audio_proj1_weight")?;
    let proj1_bias = builder.arena_1d(arena, &weights.proj1_bias, "audio_proj1_bias")?;
    let proj2_weight =
        builder.loaded_or_arena_2d(arena, loaded, &weights.proj2_weight, "audio_proj2_weight")?;
    let proj2_bias = builder.arena_1d(arena, &weights.proj2_bias, "audio_proj2_bias")?;

    let mut layers = Vec::with_capacity(weights.layers.len());
    for layer in &weights.layers {
        layers.push(build_audio_layer_tensors(
            arena,
            &mut builder,
            loaded,
            layer,
        )?);
    }

    builder.upload(arena)?;

    Ok(QwenAudioResidentTensors {
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        conv3_weight,
        conv3_bias,
        conv_out_weight,
        conv_out_bias,
        ln_post_weight,
        ln_post_bias,
        proj1_weight,
        proj1_bias,
        proj2_weight,
        proj2_bias,
        layers,
    })
}

/// Build one audio layer's arena-resident graph tensors. The 2D projection
/// weights (`attn_{q,k,v,out}`, `ffn_{up,down}`) bind zero-copy from `loaded`
/// (mmap'd pack, native q8/f16) when present; everything else (norms/biases, and
/// any unmaterialized projection) f32-uploads into the arena. All handles are
/// collected into `builder` and uploaded once by the caller.
fn build_audio_layer_tensors<'a, 'w>(
    arena: &GgmlStaticTensorArena,
    builder: &mut QwenAudioArenaBuilder<'w>,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &'w AudioLayerWeights,
) -> Result<AudioLayerGraphTensors<'a>, Qwen3AsrAudioEncoderError> {
    Ok(AudioLayerGraphTensors {
        attn_norm_weight: builder.arena_1d(
            arena,
            &layer.attn_norm_weight,
            "audio_attn_norm_weight",
        )?,
        attn_norm_bias: builder.arena_1d(arena, &layer.attn_norm_bias, "audio_attn_norm_bias")?,
        attn_q_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.attn_q_weight,
            "audio_attn_q_weight",
        )?,
        attn_q_bias: builder.arena_1d(arena, &layer.attn_q_bias, "audio_attn_q_bias")?,
        attn_k_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.attn_k_weight,
            "audio_attn_k_weight",
        )?,
        attn_k_bias: builder.arena_1d(arena, &layer.attn_k_bias, "audio_attn_k_bias")?,
        attn_v_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.attn_v_weight,
            "audio_attn_v_weight",
        )?,
        attn_v_bias: builder.arena_1d(arena, &layer.attn_v_bias, "audio_attn_v_bias")?,
        attn_out_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.attn_out_weight,
            "audio_attn_out_weight",
        )?,
        attn_out_bias: builder.arena_1d(arena, &layer.attn_out_bias, "audio_attn_out_bias")?,
        ffn_norm_weight: builder.arena_1d(
            arena,
            &layer.ffn_norm_weight,
            "audio_ffn_norm_weight",
        )?,
        ffn_norm_bias: builder.arena_1d(arena, &layer.ffn_norm_bias, "audio_ffn_norm_bias")?,
        ffn_up_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.ffn_up_weight,
            "audio_ffn_up_weight",
        )?,
        ffn_up_bias: builder.arena_1d(arena, &layer.ffn_up_bias, "audio_ffn_up_bias")?,
        ffn_down_weight: builder.loaded_or_arena_2d(
            arena,
            loaded,
            &layer.ffn_down_weight,
            "audio_ffn_down_weight",
        )?,
        ffn_down_bias: builder.arena_1d(arena, &layer.ffn_down_bias, "audio_ffn_down_bias")?,
    })
}

/// Build a metadata-only `F32Tensor` (name + dims, NO f32 values) for a weight
/// that `encode_…` binds zero-copy from the mmap'd pack (goals 7+8 Step 1b).
/// Skips the host f32 dequant entirely — the ~1.2 GB resident-memory win. The
/// dims (from the tensor index) still satisfy the shape validators; an empty
/// `values` is the signal `loaded_or_arena_2d` fails closed on if it isn't bound.
fn load_tensor_meta_only(tensor: &GgufTensorMetadata) -> F32Tensor {
    F32Tensor {
        name: tensor.name.clone(),
        dims: tensor.dims.clone(),
        values: Vec::new(),
    }
}

fn load_tensor_f32(
    reader: &GgufTensorDataReader,
    tensor: &GgufTensorMetadata,
) -> Result<F32Tensor, Qwen3AsrAudioEncoderError> {
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(&tensor.name, &tensor.dims)
        .map_err(|source| Qwen3AsrAudioEncoderError::TensorRead {
            tensor_name: tensor.name.clone(),
            source: Box::new(source),
        })?;
    Ok(F32Tensor {
        name: tensor.name.clone(),
        dims: tensor.dims.clone(),
        values,
    })
}

fn require_tensor<'a>(
    index: &'a crate::GgufTensorIndex,
    name: &str,
) -> Result<&'a GgufTensorMetadata, Qwen3AsrAudioEncoderError> {
    index
        .get(name)
        .ok_or_else(|| Qwen3AsrAudioEncoderError::InvalidTensorShape {
            tensor_name: name.to_string(),
            shape: "[]".to_string(),
            reason: "required tensor is missing".to_string(),
        })
}

fn validate_tensor_rank(
    tensor: &F32Tensor,
    expected_rank: usize,
    reason: &str,
) -> Result<(), Qwen3AsrAudioEncoderError> {
    if tensor.dims.len() == expected_rank {
        return Ok(());
    }
    Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
        tensor_name: tensor.name.clone(),
        shape: render_shape(&tensor.dims),
        reason: reason.to_string(),
    })
}

fn validate_vector_len(
    tensor: &F32Tensor,
    expected_len: usize,
) -> Result<(), Qwen3AsrAudioEncoderError> {
    if tensor.dims == [expected_len as u64] {
        return Ok(());
    }
    Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
        tensor_name: tensor.name.clone(),
        shape: render_shape(&tensor.dims),
        reason: format!("expected [{}]", expected_len),
    })
}

fn validate_matrix_shape(
    tensor: &F32Tensor,
    expected_ne0: usize,
    expected_ne1: usize,
) -> Result<(), Qwen3AsrAudioEncoderError> {
    if tensor.dims == [expected_ne0 as u64, expected_ne1 as u64] {
        return Ok(());
    }
    Err(Qwen3AsrAudioEncoderError::InvalidTensorShape {
        tensor_name: tensor.name.clone(),
        shape: render_shape(&tensor.dims),
        reason: format!("expected [{} x {}]", expected_ne0, expected_ne1),
    })
}

fn render_shape(dims: &[u64]) -> String {
    format!(
        "[{}]",
        dims.iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(" x ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_mel_layout_pads_trailing_partial_chunk_with_zeroes() {
        let mel = Qwen3AsrMelFeatures {
            n_mels: 2,
            n_frames: 3,
            data: vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        };
        let packed = pack_mel_into_chunked_layout(&mel).expect("pack");
        assert_eq!(packed.num_chunks, 1);
        assert_eq!(packed.row_count, 13);
        assert_eq!(packed.values[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(
            packed.values[QWEN3_AUDIO_CHUNK_FRAMES..QWEN3_AUDIO_CHUNK_FRAMES + 3],
            [4.0, 5.0, 6.0]
        );
        assert!(
            packed.values[3..QWEN3_AUDIO_CHUNK_FRAMES]
                .iter()
                .all(|v| *v == 0.0)
        );
    }

    #[test]
    fn positional_embedding_matches_expected_shape() {
        let values = build_audio_positional_embedding(8, 13).expect("positional");
        assert_eq!(values.len(), 8 * 13);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[4], 1.0);
    }

    /// Stage 2 bisection gate: run the ggml audio encoder (24 layers/1024/16
    /// heads, reused unmodified from qwen3-asr -- everything here is
    /// metadata-driven) against the Qwen3-ForcedAligner-0.6B checkpoint's real
    /// weights, fed the exact mel input the Python reference
    /// (`thinker.get_audio_features`) consumed for `fixtures/jfk.wav`, and
    /// compare row-for-row against the Python reference's `last_hidden_state`.
    /// Dev-machine only (needs the Stage 0 HF download + Stage 0 reference
    /// dump); skips cleanly elsewhere.
    #[test]
    fn forced_aligner_audio_encoder_matches_python_reference_for_jfk() {
        use std::path::PathBuf;

        use super::super::forced_aligner_import::{
            Qwen3ForcedAlignerLocalSourceImportRequest,
            convert_local_qwen_forced_aligner_source_to_runtime_pack,
        };
        use super::super::package_import::Qwen3AsrRuntimeQuantizationMode as ForcedAlignerQuantMode;
        use crate::ggml_runtime::GgufTensorDataReader;
        use crate::models::qwen::runtime_contract::Qwen3AsrExecutionMetadata;

        let source_root = match crate::testing::external_test_fixture_path(
            "OPENASR_QWEN_FORCED_ALIGNER_SOURCE",
            "Qwen forced-aligner source checkpoint directory",
        ) {
            Ok(path) => path,
            Err(skip) => {
                eprintln!("skipping: {skip}");
                return;
            }
        };
        let ref_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tmp/forced-aligner-ref/fixtures");
        let mel_path = ref_dir.join("audio_mel_input.f32le");
        let output_path = ref_dir.join("audio_encoder_output.f32le");
        if !source_root.exists() || !mel_path.exists() || !output_path.exists() {
            eprintln!(
                "skipping: {} / {} not present (Stage 0/2 dev-machine reference artifacts)",
                source_root.display(),
                mel_path.display()
            );
            return;
        }

        let pack_dir = std::env::temp_dir().join("openasr-forced-aligner-stage2-test");
        let _ = std::fs::create_dir_all(&pack_dir);
        let pack_path = pack_dir.join("qwen3-forced-aligner-0.6b-fp16.oasr");
        let _ = std::fs::remove_file(&pack_path);
        let request = Qwen3ForcedAlignerLocalSourceImportRequest {
            source_root,
            output_root: pack_path.clone(),
            package_id: "qwen3-forced-aligner-0.6b".to_string(),
            package_variant: Some("fp16".to_string()),
            source_name: "Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            source_revision: "test".to_string(),
            license_name: "Apache-2.0".to_string(),
            license_source: "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B".to_string(),
            quantization: ForcedAlignerQuantMode::Fp16,
        };
        convert_local_qwen_forced_aligner_source_to_runtime_pack(&request)
            .expect("forced-aligner conversion must succeed");

        let metadata = Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 24,
            audio_d_model: 1024,
            audio_heads: 16,
            llm_layers: 28,
            llm_d_model: 1024,
            llm_heads: 16,
            llm_kv_heads: 8,
            llm_head_dim: 128,
            vocab_size: 152_064,
            llm_max_positions: 8_192,
            audio_start_token_id: 151_669,
            audio_end_token_id: 151_670,
            audio_pad_token_id: 151_676,
            eos_token_id: 151_645,
            pad_token_id: 151_643,
        };

        let reader = GgufTensorDataReader::from_path(&pack_path).expect("gguf reader");
        let weights = load_qwen3_audio_encoder_weights_from_reader(&reader, metadata)
            .expect("audio encoder weights");
        assert_eq!(weights.layer_count(), 24);

        let mel_bytes = std::fs::read(&mel_path).expect("read reference mel");
        let mel_values: Vec<f32> = mel_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(mel_values.len(), 128 * 1100);
        let mel_features = Qwen3AsrMelFeatures {
            n_mels: 128,
            n_frames: 1100,
            data: mel_values,
        };

        let mut runtime = Qwen3AsrAudioEncoderRuntime::new(Some(&pack_path)).expect("runtime");
        let output = runtime
            .encode(&weights, metadata, &mel_features)
            .expect("encode");

        let ref_bytes = std::fs::read(&output_path).expect("read reference output");
        let ref_values: Vec<f32> = ref_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(ref_values.len(), 143 * 1024);
        assert_eq!(
            output.row_count, 143,
            "ggml chunked row count must match the Python reference's audio-feature row count"
        );
        assert_eq!(output.rows.len(), ref_values.len());

        let mut max_abs_diff = 0.0_f32;
        let mut sum_abs_diff = 0.0_f64;
        for (a, b) in output.rows.iter().zip(ref_values.iter()) {
            let diff = (a - b).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff as f64;
        }
        let mean_abs_diff = sum_abs_diff / output.rows.len() as f64;
        eprintln!(
            "forced_aligner_audio_encoder_matches_python_reference_for_jfk: max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff}"
        );
        // fp16-quantized 2D weights (Python ran fp32) accumulated over 24
        // encoder layers. Observed parity is fp16-rounding-level tight
        // (max_abs_diff ~0.006, mean_abs_diff ~0.0002); bound with headroom so
        // the test still catches wiring bugs (wrong shapes/permutes/layer
        // count/head count) without being brittle to harmless rounding drift.
        assert!(
            max_abs_diff < 0.1,
            "audio encoder output diverges from Python reference: max_abs_diff={max_abs_diff}"
        );
        assert!(
            mean_abs_diff < 0.01,
            "audio encoder output diverges from Python reference on average: mean_abs_diff={mean_abs_diff}"
        );
    }

    const QWEN_AUDIO_ENCODER_REAL_PACK_ENV: &str = "OPENASR_QWEN_AUDIO_ENCODER_REAL_PACK";

    /// Long-audio degradation guard for the flash-attention self-attention
    /// path: run the real audio encoder near the family's
    /// `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` (30s) chunk cap -- this test does
    /// NOT change that cap, it only probes flash's numerical behavior at the
    /// edge of it -- and confirm the output stays finite and bounded over the
    /// full self-attention sequence length. Dev-machine-only (needs a real
    /// qwen3-asr `.oasr` pack); skips via `#[ignore]` like the other
    /// `*_REAL_PACK` harnesses in this family (see `llm_transformer.rs`).
    #[test]
    #[ignore = "manual real-pack harness: set OPENASR_QWEN_AUDIO_ENCODER_REAL_PACK to a qwen3-asr .oasr model pack"]
    fn qwen_audio_encoder_flash_attention_stays_finite_near_chunk_cap() {
        use std::path::PathBuf;

        use crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;
        use crate::{read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path};

        let pack_path = std::env::var_os(QWEN_AUDIO_ENCODER_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "{QWEN_AUDIO_ENCODER_REAL_PACK_ENV} must point to a qwen3-asr .oasr model pack"
                )
            });

        let runtime_source =
            validate_ggml_runtime_source_path(&pack_path).expect("valid qwen3-asr runtime source");
        let raw_metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .expect("read qwen3-asr runtime metadata");
        let metadata =
            super::super::runtime_contract::parse_qwen3_execution_metadata(&raw_metadata)
                .expect("parse qwen3-asr metadata");

        let reader = GgufTensorDataReader::from_path(&pack_path).expect("gguf reader");
        let weights = load_qwen3_audio_encoder_weights_from_reader(&reader, metadata)
            .expect("audio encoder weights");

        // Just under the 30s cap this test deliberately does not touch.
        let target_seconds = DEFAULT_ENCODER_SAFE_CHUNK_SECONDS - 2.0;
        let frames_per_second = metadata.sample_rate_hz as usize / metadata.hop_length;
        let n_frames = (frames_per_second as f32 * target_seconds) as usize;
        // Deterministic pseudo-mel data (not real speech): this test targets
        // NUMERICAL degeneration (NaN/Inf/unbounded growth) in the flash
        // self-attention over a long sequence, which real-speech content
        // would not exercise any more directly than a dense synthetic signal.
        let data: Vec<f32> = (0..metadata.n_mels * n_frames)
            .map(|index| ((index % 97) as f32 - 48.0) / 97.0)
            .collect();
        let mel_features = Qwen3AsrMelFeatures {
            n_mels: metadata.n_mels,
            n_frames,
            data,
        };

        let mut runtime = Qwen3AsrAudioEncoderRuntime::new(Some(&pack_path)).expect("runtime");
        let output = runtime
            .encode(&weights, metadata, &mel_features)
            .expect("encode near the chunk cap");

        assert!(!output.rows.is_empty());
        assert_eq!(output.rows.len(), output.row_count * metadata.llm_d_model);
        let mut max_abs = 0.0_f32;
        for &value in &output.rows {
            assert!(
                value.is_finite(),
                "flash-attention encoder output must stay finite near the chunk cap"
            );
            max_abs = max_abs.max(value.abs());
        }
        eprintln!(
            "qwen_audio_encoder_flash_attention_stays_finite_near_chunk_cap: n_frames={n_frames} row_count={} max_abs={max_abs}",
            output.row_count
        );
        assert!(
            max_abs < 1.0e4,
            "flash-attention encoder output magnitude blew up near the chunk cap: max_abs={max_abs}"
        );
    }
}
