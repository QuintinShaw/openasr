//! sensevoice encoder graph: [prompt + LFR features] (host-prepared, already
//! scaled by sqrt(d_model) with the sinusoidal PE added) -> SAN-M blocks
//! (`enc.blk.0` at 560-dim input, then 512-dim blocks) -> `enc.after_norm` ->
//! `tp.blk.*` -> `tp.norm` -> CTC head -> `[vocab, frames]` logits.
//!
//! The per-layer math is `nn::encoder::sanm_fsmn_encoder_layer`; this module
//! owns the weight residency (arena norms/biases/FSMN kernels + zero-copy bound
//! quantized linears, the parakeet pattern) and the stage sequencing.

#![allow(dead_code)]

use std::path::Path;

use crate::ggml_runtime::{
    ArenaAllocError, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedWeightContext,
    GgmlStaticTensor, GgmlStaticTensorArena, WeightSlot,
    alloc_static_f16 as arena_alloc_static_f16, alloc_static_f32 as arena_alloc_static_f32,
    bind_loaded as arena_bind_loaded, upload_static_f16 as arena_upload_static_f16,
    upload_static_f32 as arena_upload_static_f32,
};
use crate::nn::encoder::{SanMFsmnBlockConfig, SanMFsmnBlockWeights, sanm_fsmn_encoder_layer};
use crate::nn::half::f32_to_f16_bits;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::encoder_weights::{NamedTensor, SenseVoiceEncoderWeights, SenseVoiceLayerWeights};
use super::graph_config::sensevoice_encoder_graph_config;
use super::runtime_contract::SenseVoiceExecutionMetadata;

const SENSEVOICE_ENCODER_GRAPH_CONTEXT_BYTES: usize = 768 * 1024 * 1024;
/// FunASR LayerNorm epsilon (torch LayerNorm eps=1e-12 in EncoderLayerSANM).
const ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-12;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SenseVoiceEncoderError {
    #[error("sensevoice encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("sensevoice encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("sensevoice encoder shape error: {reason}")]
    Shape { reason: String },
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> SenseVoiceEncoderError {
    move |source| SenseVoiceEncoderError::GraphBuildFailed { step, source }
}

/// Encoder input: the host-prepared `[feature_dim, n_frames]` matrix
/// (feature-fastest): 4 prompt embeddings + LFR+CMVN features, already scaled
/// by `sqrt(d_model)` and with the 560-dim sinusoidal PE added.
#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceEncoderInput {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub feature_dim: usize,
}

/// Encoder output: per-frame CTC logits, `logits[frame * vocab_size + v]`.
#[derive(Debug, Clone)]
pub(crate) struct SenseVoiceEncoderOutput {
    pub frame_count: usize,
    pub vocab_size: usize,
    pub logits: Vec<f32>,
}

// `WeightSlot` (imported above from `ggml_runtime`): arena tensor or
// zero-copy bound to the mmap'd pack (native f16/q8_0/q4_k — the
// keep-quantized seam). Shared with parakeet-ctc/parakeet-tdt/wav2vec2-ctc —
// see `ggml_runtime::arena_weight_pipeline` — since it is pure residency
// plumbing with no sensevoice-specific semantics.

fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, SenseVoiceEncoderError> {
    arena_bind_loaded(loaded, name)
        .map(WeightSlot::Loaded)
        .map_err(|reason| SenseVoiceEncoderError::Shape { reason })
}

fn alloc_static(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, SenseVoiceEncoderError> {
    arena_alloc_static_f32(arena, &weight.dims, weight.values.len(), step, false).map_err(|e| {
        match e {
            ArenaAllocError::Graph(source) => {
                SenseVoiceEncoderError::GraphBuildFailed { step, source }
            }
            ArenaAllocError::UnsupportedRank(dims) => SenseVoiceEncoderError::Shape {
                reason: format!("tensor '{}' has unsupported rank {:?}", weight.name, dims),
            },
        }
    })
}

fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, SenseVoiceEncoderError> {
    arena_alloc_static_f16(arena, &weight.dims, step, false).map_err(|e| match e {
        ArenaAllocError::Graph(source) => SenseVoiceEncoderError::GraphBuildFailed { step, source },
        ArenaAllocError::UnsupportedRank(dims) => SenseVoiceEncoderError::Shape {
            reason: format!("f16 fsmn kernel '{}' rank {:?}", weight.name, dims),
        },
    })
}

fn upload_static(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), SenseVoiceEncoderError> {
    arena_upload_static_f32(arena, tensor, &weight.values, step)
        .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), SenseVoiceEncoderError> {
    arena_upload_static_f16(arena, tensor, &weight.values, step, f32_to_f16_bits)
        .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed { step, source })
}

/// Per-layer handles: bound linears (`attn.qkv/out`, `ffn.up/down`) +
/// arena norms/biases + the f16 FSMN kernel.
struct LayerArena {
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    attn_qkv_weight: WeightSlot,
    attn_qkv_bias: GgmlStaticTensor,
    attn_out_weight: WeightSlot,
    attn_out_bias: GgmlStaticTensor,
    attn_fsmn_weight: GgmlStaticTensor,
    ffn_norm_weight: GgmlStaticTensor,
    ffn_norm_bias: GgmlStaticTensor,
    ffn_up_weight: WeightSlot,
    ffn_up_bias: GgmlStaticTensor,
    ffn_down_weight: WeightSlot,
    ffn_down_bias: GgmlStaticTensor,
    /// The block's input width (560 for `enc.blk.0`, `d_model` elsewhere),
    /// read from the attn norm weight length at load.
    input_dim: usize,
}

pub(crate) struct SenseVoiceEncoderGraph {
    metadata: SenseVoiceExecutionMetadata,
    runner: GgmlCpuGraphRunner,
    // `loaded_weights` owns the mmap-backed buffer the `Loaded` slots alias
    // (drop-order note mirrors parakeet/cohere/qwen).
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    enc_layers: Vec<LayerArena>,
    tp_layers: Vec<LayerArena>,
    enc_after_norm_weight: GgmlStaticTensor,
    enc_after_norm_bias: GgmlStaticTensor,
    tp_norm_weight: GgmlStaticTensor,
    tp_norm_bias: GgmlStaticTensor,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
}

impl SenseVoiceEncoderGraph {
    pub(crate) fn new(
        weights: &SenseVoiceEncoderWeights,
        metadata: SenseVoiceExecutionMetadata,
        runtime_path: Option<&Path>,
    ) -> Result<Self, SenseVoiceEncoderError> {
        let mut config = sensevoice_encoder_graph_config();
        config.context_bytes = SENSEVOICE_ENCODER_GRAPH_CONTEXT_BYTES;
        let total_layers = weights.enc_layers.len() + weights.tp_layers.len();
        config.graph_size = config.graph_size.max(total_layers * 128 + 2048);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            SenseVoiceEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(SENSEVOICE_ENCODER_GRAPH_CONTEXT_BYTES)
            .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;

        // ----- declare all arena tensors first (first upload freezes) -----
        let mut enc_handles = Vec::with_capacity(weights.enc_layers.len());
        for layer in &weights.enc_layers {
            enc_handles.push(alloc_layer(&arena, loaded, layer)?);
        }
        let mut tp_handles = Vec::with_capacity(weights.tp_layers.len());
        for layer in &weights.tp_layers {
            tp_handles.push(alloc_layer(&arena, loaded, layer)?);
        }
        let enc_after_norm_weight_t =
            alloc_static(&arena, &weights.enc_after_norm_weight, "after_norm_w")?;
        let enc_after_norm_bias_t =
            alloc_static(&arena, &weights.enc_after_norm_bias, "after_norm_b")?;
        let tp_norm_weight_t = alloc_static(&arena, &weights.tp_norm_weight, "tp_norm_w")?;
        let tp_norm_bias_t = alloc_static(&arena, &weights.tp_norm_bias, "tp_norm_b")?;
        let ctc_head_weight_slot = bind_loaded(loaded, &weights.ctc_head_weight.name)?;
        let ctc_head_bias_t = alloc_static(&arena, &weights.ctc_head_bias, "ctc_head_b")?;

        // ----- upload all arena values -----
        for (layer, handles) in weights.enc_layers.iter().zip(&enc_handles) {
            upload_layer(&mut arena, layer, handles)?;
        }
        for (layer, handles) in weights.tp_layers.iter().zip(&tp_handles) {
            upload_layer(&mut arena, layer, handles)?;
        }
        upload_static(
            &mut arena,
            enc_after_norm_weight_t,
            &weights.enc_after_norm_weight,
            "after_norm_w",
        )?;
        upload_static(
            &mut arena,
            enc_after_norm_bias_t,
            &weights.enc_after_norm_bias,
            "after_norm_b",
        )?;
        upload_static(
            &mut arena,
            tp_norm_weight_t,
            &weights.tp_norm_weight,
            "tp_norm_w",
        )?;
        upload_static(
            &mut arena,
            tp_norm_bias_t,
            &weights.tp_norm_bias,
            "tp_norm_b",
        )?;
        upload_static(
            &mut arena,
            ctc_head_bias_t,
            &weights.ctc_head_bias,
            "ctc_head_b",
        )?;

        Ok(Self {
            metadata,
            runner,
            loaded_weights,
            arena,
            enc_layers: enc_handles,
            tp_layers: tp_handles,
            enc_after_norm_weight: enc_after_norm_weight_t,
            enc_after_norm_bias: enc_after_norm_bias_t,
            tp_norm_weight: tp_norm_weight_t,
            tp_norm_bias: tp_norm_bias_t,
            ctc_head_weight: ctc_head_weight_slot,
            ctc_head_bias: ctc_head_bias_t,
        })
    }

    pub(crate) fn encode(
        &mut self,
        input: &SenseVoiceEncoderInput,
    ) -> Result<SenseVoiceEncoderOutput, SenseVoiceEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.d_model;
        let frames = input.n_frames;
        if input.feature_dim != metadata.feature_dim
            || input.data.len() != frames * metadata.feature_dim
        {
            return Err(SenseVoiceEncoderError::Shape {
                reason: format!(
                    "encoder input {}x{} does not match feature dim {}",
                    frames, input.feature_dim, metadata.feature_dim
                ),
            });
        }

        let mut graph = self.runner.start_graph();
        let input_t = graph
            .new_tensor_2d_f32(metadata.feature_dim, frames, "sensevoice_input")
            .map_err(bf("new_input"))?;
        graph.set_input(input_t).map_err(bf("set_input"))?;

        let map = |step, source| SenseVoiceEncoderError::GraphBuildFailed { step, source };
        let mut state = input_t;
        for handles in self.enc_layers.iter() {
            state = sanm_fsmn_encoder_layer(
                &mut graph,
                state,
                SanMFsmnBlockConfig {
                    d_model,
                    input_dim: handles.input_dim,
                    attention_heads: metadata.n_heads,
                    head_dim: metadata.head_dim,
                    frame_count: frames,
                    fsmn_kernel: metadata.fsmn_kernel,
                    layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
                },
                sanm_weights(&self.arena, handles),
                map,
            )?;
        }
        state = apply_affine_layer_norm(
            &graph,
            state,
            ENCODER_LAYER_NORM_EPSILON,
            self.arena.graph_tensor(self.enc_after_norm_weight),
            self.arena.graph_tensor(self.enc_after_norm_bias),
            AffineLayerNormSteps {
                norm: "ggml_norm(layer_norm)",
                scale: "enc_after_norm",
                bias: "enc_after_norm",
            },
            map,
        )?;
        for handles in self.tp_layers.iter() {
            state = sanm_fsmn_encoder_layer(
                &mut graph,
                state,
                SanMFsmnBlockConfig {
                    d_model,
                    input_dim: handles.input_dim,
                    attention_heads: metadata.n_heads,
                    head_dim: metadata.head_dim,
                    frame_count: frames,
                    fsmn_kernel: metadata.fsmn_kernel,
                    layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
                },
                sanm_weights(&self.arena, handles),
                map,
            )?;
        }
        state = apply_affine_layer_norm(
            &graph,
            state,
            ENCODER_LAYER_NORM_EPSILON,
            self.arena.graph_tensor(self.tp_norm_weight),
            self.arena.graph_tensor(self.tp_norm_bias),
            AffineLayerNormSteps {
                norm: "ggml_norm(layer_norm)",
                scale: "tp_norm",
                bias: "tp_norm",
            },
            map,
        )?;

        // ----- CTC head -----
        let head = graph
            .reshape_2d(
                self.ctc_head_weight.graph(&self.arena),
                d_model,
                metadata.vocab_size,
            )
            .map_err(bf("ctc_head_reshape"))?;
        let mut logits = graph.mul_mat(head, state).map_err(bf("ctc_head_matmul"))?;
        logits = graph
            .add(logits, self.arena.graph_tensor(self.ctc_head_bias))
            .map_err(bf("ctc_head_bias"))?;
        graph.set_output(logits).map_err(bf("set_output_logits"))?;

        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(input_t, &input.data, "upload_input")
            .map_err(bf("upload_input"))?;

        let want = metadata.vocab_size.checked_mul(frames).ok_or_else(|| {
            SenseVoiceEncoderError::Shape {
                reason: "logits overflow".into(),
            }
        })?;
        let logits = graph.compute_output_f32(logits, want).map_err(|error| {
            SenseVoiceEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(SenseVoiceEncoderOutput {
            frame_count: frames,
            vocab_size: metadata.vocab_size,
            logits,
        })
    }
}

fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &SenseVoiceLayerWeights,
) -> Result<LayerArena, SenseVoiceEncoderError> {
    Ok(LayerArena {
        input_dim: layer.attn_norm_weight.values.len(),
        attn_norm_weight: alloc_static(arena, &layer.attn_norm_weight, "attn_norm_w")?,
        attn_norm_bias: alloc_static(arena, &layer.attn_norm_bias, "attn_norm_b")?,
        attn_qkv_weight: bind_loaded(loaded, &layer.attn_qkv_weight.name)?,
        attn_qkv_bias: alloc_static(arena, &layer.attn_qkv_bias, "attn_qkv_b")?,
        attn_out_weight: bind_loaded(loaded, &layer.attn_out_weight.name)?,
        attn_out_bias: alloc_static(arena, &layer.attn_out_bias, "attn_out_b")?,
        attn_fsmn_weight: alloc_static_f16(arena, &layer.attn_fsmn_weight, "attn_fsmn_w")?,
        ffn_norm_weight: alloc_static(arena, &layer.ffn_norm_weight, "ffn_norm_w")?,
        ffn_norm_bias: alloc_static(arena, &layer.ffn_norm_bias, "ffn_norm_b")?,
        ffn_up_weight: bind_loaded(loaded, &layer.ffn_up_weight.name)?,
        ffn_up_bias: alloc_static(arena, &layer.ffn_up_bias, "ffn_up_b")?,
        ffn_down_weight: bind_loaded(loaded, &layer.ffn_down_weight.name)?,
        ffn_down_bias: alloc_static(arena, &layer.ffn_down_bias, "ffn_down_b")?,
    })
}

fn upload_layer(
    arena: &mut GgmlStaticTensorArena,
    layer: &SenseVoiceLayerWeights,
    h: &LayerArena,
) -> Result<(), SenseVoiceEncoderError> {
    upload_static_f16(
        arena,
        h.attn_fsmn_weight,
        &layer.attn_fsmn_weight,
        "attn_fsmn_w",
    )?;
    let pairs: [(GgmlStaticTensor, &NamedTensor); 8] = [
        (h.attn_norm_weight, &layer.attn_norm_weight),
        (h.attn_norm_bias, &layer.attn_norm_bias),
        (h.attn_qkv_bias, &layer.attn_qkv_bias),
        (h.attn_out_bias, &layer.attn_out_bias),
        (h.ffn_norm_weight, &layer.ffn_norm_weight),
        (h.ffn_norm_bias, &layer.ffn_norm_bias),
        (h.ffn_up_bias, &layer.ffn_up_bias),
        (h.ffn_down_bias, &layer.ffn_down_bias),
    ];
    for (tensor, weight) in pairs {
        upload_static(arena, tensor, weight, "layer_weight")?;
    }
    Ok(())
}

fn sanm_weights<'a>(arena: &'a GgmlStaticTensorArena, h: &LayerArena) -> SanMFsmnBlockWeights<'a> {
    let g = |t: GgmlStaticTensor| arena.graph_tensor(t);
    let b = |slot: WeightSlot| slot.graph(arena);
    SanMFsmnBlockWeights {
        attn_norm_weight: g(h.attn_norm_weight),
        attn_norm_bias: g(h.attn_norm_bias),
        attn_qkv_weight: b(h.attn_qkv_weight),
        attn_qkv_bias: g(h.attn_qkv_bias),
        attn_out_weight: b(h.attn_out_weight),
        attn_out_bias: g(h.attn_out_bias),
        attn_fsmn_weight: g(h.attn_fsmn_weight),
        ffn_norm_weight: g(h.ffn_norm_weight),
        ffn_norm_bias: g(h.ffn_norm_bias),
        ffn_up_weight: b(h.ffn_up_weight),
        ffn_up_bias: g(h.ffn_up_bias),
        ffn_down_weight: b(h.ffn_down_weight),
        ffn_down_bias: g(h.ffn_down_bias),
    }
}

/// Build the encoder input matrix from the 4 prompt-embedding rows + the
/// LFR+CMVN features: `x = concat(prompt, lfr) * sqrt(d_model) + sinusoidal_pe`
/// (FunASR `SinusoidalPositionEncoder`: positions start at 1, first half sin,
/// second half cos, inverse timescales `exp(-i * ln(10000) / (depth/2 - 1))`).
pub(crate) fn build_sensevoice_encoder_input(
    prompt_rows: &[&[f32]],
    lfr_features: &[f32],
    feature_dim: usize,
    d_model: usize,
) -> Result<SenseVoiceEncoderInput, SenseVoiceEncoderError> {
    if feature_dim == 0 || !lfr_features.len().is_multiple_of(feature_dim) {
        return Err(SenseVoiceEncoderError::Shape {
            reason: format!(
                "lfr feature length {} is not a multiple of feature dim {feature_dim}",
                lfr_features.len()
            ),
        });
    }
    for row in prompt_rows {
        if row.len() != feature_dim {
            return Err(SenseVoiceEncoderError::Shape {
                reason: format!(
                    "prompt row has {} values, expected feature dim {feature_dim}",
                    row.len()
                ),
            });
        }
    }
    let lfr_frames = lfr_features.len() / feature_dim;
    let n_frames = prompt_rows.len() + lfr_frames;
    let scale = (d_model as f32).sqrt();

    let mut data = Vec::with_capacity(n_frames * feature_dim);
    for row in prompt_rows {
        data.extend_from_slice(row);
    }
    data.extend_from_slice(lfr_features);
    for value in &mut data {
        *value *= scale;
    }

    // Sinusoidal PE over the concatenated sequence.
    let half = feature_dim / 2;
    if half < 2 {
        return Err(SenseVoiceEncoderError::Shape {
            reason: format!("feature dim {feature_dim} too small for sinusoidal PE"),
        });
    }
    let log_timescale_increment = (10_000.0f64).ln() / (half as f64 - 1.0);
    let inv_timescales: Vec<f64> = (0..half)
        .map(|i| (-(i as f64) * log_timescale_increment).exp())
        .collect();
    for frame in 0..n_frames {
        let position = (frame + 1) as f64;
        let row = &mut data[frame * feature_dim..(frame + 1) * feature_dim];
        for (i, &inv) in inv_timescales.iter().enumerate() {
            let scaled = position * inv;
            row[i] += scaled.sin() as f32;
            row[half + i] += scaled.cos() as f32;
        }
    }

    Ok(SenseVoiceEncoderInput {
        data,
        n_frames,
        feature_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;
    use crate::models::sensevoice::encoder_weights::load_sensevoice_encoder_weights;
    use crate::models::sensevoice::runtime_contract::parse_sensevoice_execution_metadata;

    /// Offline bring-up parity vs the PyTorch reference (ref.py oracle).
    /// Requires SENSEVOICE_BRINGUP_DIR with `ref_lfr_zh.bin` ([94,560] f32 LFR+CMVN
    /// features) + `ref_logits_zh.bin` ([98,25055] f32) and SENSEVOICE_PACK
    /// pointing at the fp16 .oasr pack. Asserts the greedy argmax sequence is
    /// IDENTICAL to the reference and reports the logit max-abs-error.
    #[test]
    #[ignore = "requires SENSEVOICE_BRINGUP_DIR + SENSEVOICE_PACK with local oracle refs"]
    fn encoder_graph_matches_pytorch_reference_logits() {
        let dir = std::path::PathBuf::from(
            std::env::var("SENSEVOICE_BRINGUP_DIR").expect("SENSEVOICE_BRINGUP_DIR"),
        );
        let pack =
            std::path::PathBuf::from(std::env::var("SENSEVOICE_PACK").expect("SENSEVOICE_PACK"));
        let read_f32 = |name: &str| -> Vec<f32> {
            std::fs::read(dir.join(name))
                .unwrap_or_else(|e| panic!("read {name}: {e}"))
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let lfr = read_f32("ref_lfr_zh.bin");
        let ref_logits = read_f32("ref_logits_zh.bin");

        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_sensevoice_execution_metadata(&gguf_metadata).expect("contract");
        let weights = load_sensevoice_encoder_weights(&reader, &metadata).expect("weights");

        // zh prompt: [lang=3, event=1, emotion=2, textnorm(woitn)=15].
        let embed = &weights.prompt_embed;
        let dim = metadata.feature_dim;
        let row = |i: usize| &embed.values[i * dim..(i + 1) * dim];
        let prompt = [row(3), row(1), row(2), row(15)];
        let input =
            build_sensevoice_encoder_input(&prompt, &lfr, dim, metadata.d_model).expect("input");
        assert_eq!(input.n_frames, ref_logits.len() / metadata.vocab_size);

        let mut graph =
            SenseVoiceEncoderGraph::new(&weights, metadata, Some(pack.as_path())).expect("graph");
        let out = graph.encode(&input).expect("encode");
        assert_eq!(out.logits.len(), ref_logits.len());

        let mut max_err = 0.0f32;
        for (a, b) in out.logits.iter().zip(&ref_logits) {
            max_err = max_err.max((a - b).abs());
        }
        // Greedy argmax parity (the decode-relevant signal).
        let vocab = metadata.vocab_size;
        let mut mismatches = 0usize;
        for frame in 0..out.frame_count {
            let argmax = |v: &[f32]| -> usize {
                let mut best = 0usize;
                for (i, &x) in v.iter().enumerate() {
                    if x > v[best] {
                        best = i;
                    }
                }
                best
            };
            let ours = argmax(&out.logits[frame * vocab..(frame + 1) * vocab]);
            let refs = argmax(&ref_logits[frame * vocab..(frame + 1) * vocab]);
            if ours != refs {
                mismatches += 1;
                eprintln!("frame {frame}: ours {ours} vs ref {refs}");
            }
        }
        eprintln!(
            "sensevoice graph parity: logits max-abs-err = {max_err:e}, argmax mismatches = {mismatches}/{}",
            out.frame_count
        );
        assert_eq!(mismatches, 0, "greedy argmax must match the reference");
    }
}
