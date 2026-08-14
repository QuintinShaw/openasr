//! Fun-ASR-Nano SAN-M encoder graph: `build_sensevoice_encoder_input` (host-
//! prepared `[560, frames]` scaled-by-sqrt(d_model) + sinusoidal PE, NO prompt
//! rows) -> 50 SAN-M `enc.blk.*` (block 0 at 560-dim input) -> `enc.after_norm`
//! -> 20 `tp.blk.*` -> `tp.norm` -> the hidden state `[512, frames]` the adaptor
//! consumes. This is a trimmed copy of `sensevoice::encoder_graph`: the per-
//! layer math is the shared `nn::encoder::sanm_fsmn_encoder_layer` primitive,
//! but Fun-ASR-Nano's encoder uses LayerNorm eps = 1e-5 (not SenseVoice's
//! 1e-12), takes NO SenseVoice CTC-multitask prompt rows, and stops at
//! `tp.norm` (no CTC head) -- it outputs hidden states, not `[vocab, frames]`
//! CTC logits.

#![allow(dead_code)]

use crate::ggml_runtime::{
    ArenaAllocError, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReadError,
    GgufTensorDataReader, WeightSlot, alloc_static_f16 as arena_alloc_static_f16,
    alloc_static_f32 as arena_alloc_static_f32, bind_loaded as arena_bind_loaded,
    upload_static_f16 as arena_upload_static_f16, upload_static_f32 as arena_upload_static_f32,
};
use crate::nn::encoder::{
    SanMFsmnBlockConfig, SanMFsmnBlockWeights, sanm_fsmn_encoder_layer,
    sanm_fsmn_graph_node_capacity,
};
use crate::nn::half::f32_to_f16_bits;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use crate::models::sensevoice::graph_config::sensevoice_encoder_graph_config;

use super::runtime_contract::{FUNASR_NANO_ENCODER_LAYER_NORM_EPSILON, FunasrNanoEncoderMetadata};

const SANM_ARENA_TENSORS_PER_LAYER: usize = 9;
const FUNASR_NANO_FIXED_ARENA_TENSORS: usize = 4;
const FUNASR_NANO_FIXED_GRAPH_NODES: usize = 6;
const FUNASR_NANO_FIXED_GRAPH_LEAFS: usize = 5;

#[derive(Debug, thiserror::Error)]
pub(crate) enum FunasrNanoEncoderError {
    #[error("funasr-nano encoder graph build failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error("funasr-nano encoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("funasr-nano encoder weight read failed: {0}")]
    WeightRead(#[from] GgufTensorDataReadError),
    #[error("funasr-nano encoder shape error: {reason}")]
    Shape { reason: String },
    #[error("funasr-nano encoder tensor '{name}' is not part of the runtime tensor contract")]
    NotInContract { name: String },
}

fn bf(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> FunasrNanoEncoderError {
    move |source| FunasrNanoEncoderError::GraphBuildFailed { step, source }
}

/// Encoder output: per-frame hidden state, row-major `[frame][d_model]`
/// (`rows[frame * d_model + h]`) -- the token-major layout the adaptor graph
/// consumes directly.
#[derive(Debug, Clone)]
pub(crate) struct FunasrNanoEncoderOutput {
    pub frame_count: usize,
    pub d_model: usize,
    pub rows: Vec<f32>,
}

/// A host weight: stored dims (from the GGUF index) + dequantized f32 values.
#[derive(Debug, Clone)]
struct NamedTensor {
    name: String,
    dims: Vec<usize>,
    values: Vec<f32>,
}

/// One SAN-M block's weights (`enc.blk.{i}.*` or `tp.blk.{i}.*`).
#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm_weight: NamedTensor,
    attn_norm_bias: NamedTensor,
    attn_qkv_weight: NamedTensor,
    attn_qkv_bias: NamedTensor,
    attn_out_weight: NamedTensor,
    attn_out_bias: NamedTensor,
    attn_fsmn_weight: NamedTensor,
    ffn_norm_weight: NamedTensor,
    ffn_norm_bias: NamedTensor,
    ffn_up_weight: NamedTensor,
    ffn_up_bias: NamedTensor,
    ffn_down_weight: NamedTensor,
    ffn_down_bias: NamedTensor,
}

struct EncoderWeights {
    enc_layers: Vec<LayerWeights>,
    tp_layers: Vec<LayerWeights>,
    enc_after_norm_weight: NamedTensor,
    enc_after_norm_bias: NamedTensor,
    tp_norm_weight: NamedTensor,
    tp_norm_bias: NamedTensor,
}

fn load_tensor_meta(
    reader: &GgufTensorDataReader,
    guard: &crate::models::tensor_binding::TensorReadGuard,
    name: &str,
) -> Result<(String, Vec<usize>, Vec<u64>), FunasrNanoEncoderError> {
    if !guard.contains(name) {
        return Err(FunasrNanoEncoderError::NotInContract {
            name: name.to_string(),
        });
    }
    let tensor = reader.tensor_index().get(name).ok_or_else(|| {
        FunasrNanoEncoderError::WeightRead(GgufTensorDataReadError::TensorNotFound {
            path: reader.tensor_index().path().to_path_buf(),
            tensor_name: name.to_string(),
        })
    })?;
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    Ok((name.to_string(), dims, tensor.dims.clone()))
}

/// Host-f32 materialization for tensors that legitimately stay f32/f16 in the
/// keep-quantized contract: 1-D norms/biases and the FSMN conv kernel.
fn load_named(
    reader: &GgufTensorDataReader,
    guard: &crate::models::tensor_binding::TensorReadGuard,
    name: &str,
) -> Result<NamedTensor, FunasrNanoEncoderError> {
    let (name, dims, shape_u64) = load_tensor_meta(reader, guard, name)?;
    let values = reader.host_tensor_f32_copy_dequantized_by_name(&name, &shape_u64)?;
    Ok(NamedTensor { name, dims, values })
}

/// Metadata-only load for rank-2 `mul_mat` weights bound zero-copy from the
/// mmap'd pack. Must NOT dequant to host f32 -- that is the load-time-dequant
/// pitfall K1 forbids for bulk weights.
fn load_named_bound(
    reader: &GgufTensorDataReader,
    guard: &crate::models::tensor_binding::TensorReadGuard,
    name: &str,
) -> Result<NamedTensor, FunasrNanoEncoderError> {
    let (name, dims, _) = load_tensor_meta(reader, guard, name)?;
    Ok(NamedTensor {
        name,
        dims,
        values: Vec::new(),
    })
}

fn load_layer(
    reader: &GgufTensorDataReader,
    guard: &crate::models::tensor_binding::TensorReadGuard,
    scope: &str,
    layer: usize,
) -> Result<LayerWeights, FunasrNanoEncoderError> {
    let n = |suffix: &str| format!("{scope}.{layer}.{suffix}");
    Ok(LayerWeights {
        attn_norm_weight: load_named(reader, guard, &n("attn.norm.weight"))?,
        attn_norm_bias: load_named(reader, guard, &n("attn.norm.bias"))?,
        attn_qkv_weight: load_named_bound(reader, guard, &n("attn.qkv.weight"))?,
        attn_qkv_bias: load_named(reader, guard, &n("attn.qkv.bias"))?,
        attn_out_weight: load_named_bound(reader, guard, &n("attn.out.weight"))?,
        attn_out_bias: load_named(reader, guard, &n("attn.out.bias"))?,
        attn_fsmn_weight: load_named(reader, guard, &n("attn.fsmn.weight"))?,
        ffn_norm_weight: load_named(reader, guard, &n("ffn.norm.weight"))?,
        ffn_norm_bias: load_named(reader, guard, &n("ffn.norm.bias"))?,
        ffn_up_weight: load_named_bound(reader, guard, &n("ffn.up.weight"))?,
        ffn_up_bias: load_named(reader, guard, &n("ffn.up.bias"))?,
        ffn_down_weight: load_named_bound(reader, guard, &n("ffn.down.weight"))?,
        ffn_down_bias: load_named(reader, guard, &n("ffn.down.bias"))?,
    })
}

fn load_encoder_weights(
    reader: &GgufTensorDataReader,
    guard: &crate::models::tensor_binding::TensorReadGuard,
    metadata: &FunasrNanoEncoderMetadata,
) -> Result<EncoderWeights, FunasrNanoEncoderError> {
    use super::tensor_names::{
        ENC_AFTER_NORM_BIAS, ENC_AFTER_NORM_WEIGHT, TP_NORM_BIAS, TP_NORM_WEIGHT,
    };
    let mut enc_layers = Vec::with_capacity(metadata.n_layers);
    for layer in 0..metadata.n_layers {
        enc_layers.push(load_layer(reader, guard, "enc.blk", layer)?);
    }
    let mut tp_layers = Vec::with_capacity(metadata.tp_blocks);
    for layer in 0..metadata.tp_blocks {
        tp_layers.push(load_layer(reader, guard, "tp.blk", layer)?);
    }
    Ok(EncoderWeights {
        enc_layers,
        tp_layers,
        enc_after_norm_weight: load_named(reader, guard, ENC_AFTER_NORM_WEIGHT)?,
        enc_after_norm_bias: load_named(reader, guard, ENC_AFTER_NORM_BIAS)?,
        tp_norm_weight: load_named(reader, guard, TP_NORM_WEIGHT)?,
        tp_norm_bias: load_named(reader, guard, TP_NORM_BIAS)?,
    })
}

fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, FunasrNanoEncoderError> {
    arena_bind_loaded(loaded, name)
        .map(WeightSlot::Loaded)
        .map_err(|reason| FunasrNanoEncoderError::Shape { reason })
}

fn alloc_static(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, FunasrNanoEncoderError> {
    arena_alloc_static_f32(arena, &weight.dims, weight.values.len(), step, false).map_err(|e| {
        match e {
            ArenaAllocError::Graph(source) => {
                FunasrNanoEncoderError::GraphBuildFailed { step, source }
            }
            ArenaAllocError::UnsupportedRank(dims) => FunasrNanoEncoderError::Shape {
                reason: format!("tensor '{}' has unsupported rank {:?}", weight.name, dims),
            },
        }
    })
}

fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, FunasrNanoEncoderError> {
    arena_alloc_static_f16(arena, &weight.dims, step, false).map_err(|e| match e {
        ArenaAllocError::Graph(source) => FunasrNanoEncoderError::GraphBuildFailed { step, source },
        ArenaAllocError::UnsupportedRank(dims) => FunasrNanoEncoderError::Shape {
            reason: format!("f16 fsmn kernel '{}' rank {:?}", weight.name, dims),
        },
    })
}

fn upload_static(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), FunasrNanoEncoderError> {
    arena_upload_static_f32(arena, tensor, &weight.values, step)
        .map_err(|source| FunasrNanoEncoderError::GraphBuildFailed { step, source })
}

fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), FunasrNanoEncoderError> {
    arena_upload_static_f16(arena, tensor, &weight.values, step, f32_to_f16_bits)
        .map_err(|source| FunasrNanoEncoderError::GraphBuildFailed { step, source })
}

/// Per-layer handles: bound linears (`attn.qkv/out`, `ffn.up/down`) + arena
/// norms/biases + the f16 FSMN kernel.
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
    input_dim: usize,
}

pub(crate) struct FunasrNanoEncoderGraph {
    metadata: FunasrNanoEncoderMetadata,
    runner: GgmlCpuGraphRunner,
    loaded_weights: Option<GgmlLoadedWeightContext>,
    arena: GgmlStaticTensorArena,
    enc_layers: Vec<LayerArena>,
    tp_layers: Vec<LayerArena>,
    enc_after_norm_weight: GgmlStaticTensor,
    enc_after_norm_bias: GgmlStaticTensor,
    tp_norm_weight: GgmlStaticTensor,
    tp_norm_bias: GgmlStaticTensor,
}

impl FunasrNanoEncoderGraph {
    pub(crate) fn graph_lane(&self) -> (crate::ggml_runtime::GgmlCpuGraphBackend, bool) {
        (self.runner.backend_kind(), self.runner.uses_scheduler())
    }

    pub(crate) fn loaded_weight_binding_identity(
        &self,
    ) -> Option<crate::ggml_runtime::GgmlLoadedWeightBindingIdentity> {
        self.loaded_weights
            .as_ref()
            .map(|loaded| self.runner.loaded_weight_binding_identity(loaded))
    }

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: FunasrNanoEncoderMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FunasrNanoEncoderError> {
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| FunasrNanoEncoderError::Shape {
                    reason: error.to_string(),
                })?;
        let guard = super::runtime_contract::funasr_nano_encoder_read_guard(&metadata);
        let weights = load_encoder_weights(&reader, &guard, &metadata)?;

        let total_layers = weights.enc_layers.len() + weights.tp_layers.len();
        let mut config = sensevoice_encoder_graph_config(backend);
        let graph_capacity = sanm_fsmn_graph_node_capacity(
            total_layers,
            FUNASR_NANO_FIXED_GRAPH_NODES,
            FUNASR_NANO_FIXED_GRAPH_LEAFS,
            config.graph_size,
        );
        config.set_graph_node_capacity(graph_capacity);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            FunasrNanoEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights = Some(
            runner
                .load_gguf_weight_context_from_preflight(preflight)
                .map_err(bf("loaded_weight_context"))?,
        );
        let loaded = loaded_weights.as_ref();
        let arena_tensor_capacity = total_layers
            .saturating_mul(SANM_ARENA_TENSORS_PER_LAYER)
            .saturating_add(FUNASR_NANO_FIXED_ARENA_TENSORS);
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
                arena_tensor_capacity,
            ))
            .map_err(|source| FunasrNanoEncoderError::GraphBuildFailed {
                step: "static_tensor_arena",
                source,
            })?;

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
        })
    }

    /// Encode a full utterance. `input` is the host-prepared `[feature_dim,
    /// frames]` matrix (scaled by sqrt(d_model) with the sinusoidal PE added,
    /// no prompt rows) produced by `sensevoice::encoder_graph::
    /// build_sensevoice_encoder_input`.
    pub(crate) fn encode(
        &mut self,
        input_data: &[f32],
        n_frames: usize,
        feature_dim: usize,
    ) -> Result<FunasrNanoEncoderOutput, FunasrNanoEncoderError> {
        let metadata = self.metadata;
        let d_model = metadata.d_model;
        if feature_dim != metadata.feature_dim || input_data.len() != n_frames * feature_dim {
            return Err(FunasrNanoEncoderError::Shape {
                reason: format!(
                    "encoder input {n_frames}x{feature_dim} does not match feature dim {}",
                    metadata.feature_dim
                ),
            });
        }
        let eps = FUNASR_NANO_ENCODER_LAYER_NORM_EPSILON;

        let mut graph = self.runner.start_graph();
        let input_t = graph
            .new_tensor_2d_f32(feature_dim, n_frames, "funasr_nano_input")
            .map_err(bf("new_input"))?;
        graph.set_input(input_t).map_err(bf("set_input"))?;

        let map = |step, source| FunasrNanoEncoderError::GraphBuildFailed { step, source };
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
                    frame_count: n_frames,
                    fsmn_kernel: metadata.fsmn_kernel,
                    layer_norm_epsilon: eps,
                    use_flash_attention: false,
                },
                sanm_weights(&self.arena, handles),
                map,
            )?;
        }
        state = apply_affine_layer_norm(
            &graph,
            state,
            eps,
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
                    frame_count: n_frames,
                    fsmn_kernel: metadata.fsmn_kernel,
                    layer_norm_epsilon: eps,
                    use_flash_attention: false,
                },
                sanm_weights(&self.arena, handles),
                map,
            )?;
        }
        state = apply_affine_layer_norm(
            &graph,
            state,
            eps,
            self.arena.graph_tensor(self.tp_norm_weight),
            self.arena.graph_tensor(self.tp_norm_bias),
            AffineLayerNormSteps {
                norm: "ggml_norm(layer_norm)",
                scale: "tp_norm",
                bias: "tp_norm",
            },
            map,
        )?;

        graph.set_output(state).map_err(bf("set_output_hidden"))?;
        graph
            .prepare_outputs_for_upload(&[state])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(input_t, input_data, "upload_input")
            .map_err(bf("upload_input"))?;

        let want = d_model
            .checked_mul(n_frames)
            .ok_or_else(|| FunasrNanoEncoderError::Shape {
                reason: "hidden state overflow".into(),
            })?;
        let rows = graph.compute_output_f32(state, want).map_err(|error| {
            FunasrNanoEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(FunasrNanoEncoderOutput {
            frame_count: n_frames,
            d_model,
            rows,
        })
    }

    pub(crate) fn release_transient_compute_memory(
        &mut self,
    ) -> Result<(), FunasrNanoEncoderError> {
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(|source| FunasrNanoEncoderError::GraphBuildFailed {
                step: "release_transient_scheduler_working_set",
                source,
            })
    }
}

fn alloc_layer(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &LayerWeights,
) -> Result<LayerArena, FunasrNanoEncoderError> {
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
    layer: &LayerWeights,
    h: &LayerArena,
) -> Result<(), FunasrNanoEncoderError> {
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

#[cfg(test)]
mod trace_tests {
    use super::*;
    use crate::models::funasr_nano::runtime_contract::{
        funasr_nano_encoder_read_guard, funasr_nano_encoder_tensor_descriptors,
    };
    use crate::models::tensor_binding::{
        assert_trace_matches_descriptor_set, project_fixture_tensors,
    };
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    fn tiny_encoder_metadata() -> FunasrNanoEncoderMetadata {
        FunasrNanoEncoderMetadata {
            n_layers: 1,
            tp_blocks: 1,
            d_model: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            fsmn_kernel: 5,
            feature_dim: 28,
        }
    }

    /// The equivalence evidence the count-plus-sampling pin used to fake: run
    /// the REAL encoder weight loader (both SAN-M scopes plus the tail norms)
    /// over a synthetic pack projected from the encoder-half contract itself,
    /// with the tensor index's access trace enabled, and assert the traced
    /// read set equals the encoder-half descriptor set name for name and
    /// shape for shape. Any drift -- the loader reading a tensor the contract
    /// does not list, a descriptor no loader reads, or a read violating the
    /// descriptor's shape -- fails here. Also exercises the read guard: every
    /// read is contract-listed.
    ///
    /// Encoder half only. Adaptor and decoder halves have their own certificates
    /// (`adapter_graph::trace_tests`, `llm_transformer::trace_tests`); do not
    /// read this as a whole-family access-trace claim.
    #[test]
    fn encoder_loader_read_trace_equals_the_contract_descriptors() {
        let metadata = tiny_encoder_metadata();
        let descriptors = funasr_nano_encoder_tensor_descriptors(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("funasr-nano-encoder-trace.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in project_fixture_tensors(&descriptors) {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write trace pack");

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        let guard = funasr_nano_encoder_read_guard(&metadata);
        load_encoder_weights(&reader, &guard, &metadata).expect("full encoder load");

        assert_trace_matches_descriptor_set(&reader.tensor_index().access_trace(), &descriptors);
    }

    /// The read guard fails closed on any tensor the contract does not
    /// enumerate, so a loader/name drift cannot read off-contract.
    #[test]
    fn encoder_read_guard_rejects_off_contract_tensors() {
        let metadata = tiny_encoder_metadata();
        let guard = funasr_nano_encoder_read_guard(&metadata);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("funasr-nano-encoder-guard.oasr");
        let spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new())
            .with_tensor_shape("off.contract.weight", vec![2, 2]);
        write_tiny_gguf_runtime_source(&path, &spec).expect("write pack");
        let reader = GgufTensorDataReader::from_path(&path).expect("reader");

        let error = load_named(&reader, &guard, "off.contract.weight")
            .expect_err("off-contract reads must fail closed");
        assert!(
            matches!(error, FunasrNanoEncoderError::NotInContract { ref name } if name == "off.contract.weight"),
            "unexpected error: {error}"
        );
    }
}
