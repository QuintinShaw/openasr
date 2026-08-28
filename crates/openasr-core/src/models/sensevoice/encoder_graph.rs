//! sensevoice encoder graph: device prompt lookup + LFR features -> scale +
//! sinusoidal PE -> SAN-M blocks
//! (`enc.blk.0` at 560-dim input, then 512-dim blocks) -> `enc.after_norm` ->
//! `tp.blk.*` -> `tp.norm` -> CTC head -> `[vocab, frames]` logits.
//!
//! The per-layer math is `nn::encoder::sanm_fsmn_encoder_layer`; this module
//! owns the weight residency (arena norms/biases/FSMN kernels + zero-copy bound
//! quantized linears, the parakeet pattern) and the stage sequencing.

#![allow(dead_code)]

use crate::ggml_runtime::{
    ArenaAllocError, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlDecodeReuseMode, GgmlGraphShapeKey, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgmlSameShapePersistentGraph, GgmlSelectionEvidenceRef,
    GgmlStaticTensor, GgmlStaticTensorArena, GgufRuntimeSourcePreflight, WeightSlot,
    alloc_static_f16 as arena_alloc_static_f16, alloc_static_f32 as arena_alloc_static_f32,
    bind_loaded as arena_bind_loaded, upload_static_f16 as arena_upload_static_f16,
    upload_static_f32 as arena_upload_static_f32,
};
use crate::models::runtime_memory::{checked_sum, element_bytes};
use crate::models::system_memory_owner::{SystemMemoryCapacity, SystemMemoryOwnerError};
use crate::nn::encoder::{
    SanMFsmnBlockConfig, SanMFsmnBlockWeights, sanm_fsmn_encoder_layer,
    sanm_fsmn_graph_node_capacity,
};
use crate::nn::half::f32_to_f16_bits;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::encoder_weights::{NamedTensor, SenseVoiceEncoderWeights, SenseVoiceLayerWeights};
use super::graph_config::{
    sensevoice_encoder_graph_config, sensevoice_sanm_flash_attention_for_current_request,
};
use super::runtime_contract::SenseVoiceExecutionMetadata;

const SANM_ARENA_TENSORS_PER_LAYER: usize = 9;
const SENSEVOICE_FIXED_ARENA_TENSORS: usize = 6;
const SENSEVOICE_FIXED_GRAPH_NODES: usize = 19;
const SENSEVOICE_FIXED_GRAPH_LEAFS: usize = 11;
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
    pub frame_compute: Option<Vec<GgmlSelectionEvidenceRef>>,
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

/// Count only the two handle vectors retained by a built SenseVoice graph.
/// Native ggml buffers are admitted in their backend memory domain instead.
pub(crate) fn quoted_graph_retained_bytes(
    enc_layer_capacity: usize,
    tp_layer_capacity: usize,
) -> Result<u64, SystemMemoryOwnerError> {
    checked_sum(
        [
            element_bytes::<LayerArena>(
                enc_layer_capacity,
                "sensevoice",
                "graph encoder layer handles",
            )?,
            element_bytes::<LayerArena>(
                tp_layer_capacity,
                "sensevoice",
                "graph transcription layer handles",
            )?,
        ],
        "sensevoice",
        "graph retained bytes",
    )
}

struct SenseVoiceReuseTensors {
    lfr: GgmlCpuTensor<'static>,
    prompt: GgmlCpuTensor<'static>,
    position: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
}

pub(crate) struct SenseVoiceEncoderGraph {
    metadata: SenseVoiceExecutionMetadata,
    use_flash_attention: bool,
    reuse_enabled: bool,
    same_shape: GgmlSameShapePersistentGraph,
    reuse_tensors: Option<SenseVoiceReuseTensors>,
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
    prompt_embedding: GgmlLoadedTensor,
    prompt_embedding_rows: usize,
    positional_inv_timescales: GgmlStaticTensor,
}

impl SenseVoiceEncoderGraph {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = SystemMemoryCapacity::default();
        bytes.add_vec(&self.enc_layers, "sensevoice graph encoder layer handles")?;
        bytes.add_vec(
            &self.tp_layers,
            "sensevoice graph transcription layer handles",
        )?;
        Ok(bytes.finish())
    }

    pub(crate) fn new(
        weights: &SenseVoiceEncoderWeights,
        metadata: SenseVoiceExecutionMetadata,
        runtime_preflight: &GgufRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        _reuse_mode: GgmlDecodeReuseMode,
    ) -> Result<Self, SenseVoiceEncoderError> {
        let total_layers = weights.enc_layers.len() + weights.tp_layers.len();
        let mut config = sensevoice_encoder_graph_config(backend);
        let use_flash_attention = sensevoice_sanm_flash_attention_for_current_request(&config);
        let graph_capacity = sanm_fsmn_graph_node_capacity(
            total_layers,
            SENSEVOICE_FIXED_GRAPH_NODES,
            SENSEVOICE_FIXED_GRAPH_LEAFS,
            config.graph_size,
        );
        config.set_graph_node_capacity(graph_capacity);
        let runner = GgmlCpuGraphRunner::new(config).map_err(|source| {
            SenseVoiceEncoderError::GraphBuildFailed {
                step: "runner_init",
                source,
            }
        })?;
        let loaded_weights = Some(
            runner
                .load_gguf_weight_context_from_preflight(runtime_preflight)
                .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed {
                    step: "load_gguf_weight_context",
                    source,
                })?,
        );
        let loaded = loaded_weights.as_ref();
        let prompt_embedding = loaded
            .and_then(|loaded| loaded.tensor("embed.prompt.weight"))
            .ok_or_else(|| SenseVoiceEncoderError::Shape {
                reason: "missing device-bound SenseVoice prompt embedding".to_string(),
            })?;
        let prompt_embedding_rows = weights.prompt_embed.values.len() / metadata.feature_dim;
        let positional_half = metadata.feature_dim / 2;
        if positional_half < 2 || !metadata.feature_dim.is_multiple_of(2) {
            return Err(SenseVoiceEncoderError::Shape {
                reason: format!(
                    "SenseVoice feature dim {} cannot form sinusoidal pairs",
                    metadata.feature_dim
                ),
            });
        }
        let arena_tensor_capacity = total_layers
            .saturating_mul(SANM_ARENA_TENSORS_PER_LAYER)
            .saturating_add(SENSEVOICE_FIXED_ARENA_TENSORS);
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(
                arena_tensor_capacity,
            ))
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
        let positional_inv_timescales_t = arena
            .new_tensor_2d_f32(1, positional_half, "sensevoice_positional_inv_timescales")
            .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed {
                step: "positional_inv_timescales_alloc",
                source,
            })?;

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
        let log_timescale_increment = (10_000.0f64).ln() / (positional_half as f64 - 1.0);
        let positional_inv_timescales = (0..positional_half)
            .map(|index| (-(index as f64) * log_timescale_increment).exp() as f32)
            .collect::<Vec<_>>();
        arena
            .set_f32_slice(
                positional_inv_timescales_t,
                &positional_inv_timescales,
                "sensevoice_positional_inv_timescales",
            )
            .map_err(|source| SenseVoiceEncoderError::GraphBuildFailed {
                step: "positional_inv_timescales_upload",
                source,
            })?;

        Ok(Self {
            metadata,
            use_flash_attention,
            reuse_enabled: crate::ggml_runtime::encoder_same_shape_reuse_is_enabled(),
            same_shape: GgmlSameShapePersistentGraph::default(),
            reuse_tensors: None,
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
            prompt_embedding,
            prompt_embedding_rows,
            positional_inv_timescales: positional_inv_timescales_t,
        })
    }

    pub(crate) fn backend(&self) -> crate::ggml_runtime::GgmlCpuGraphBackend {
        self.runner.backend_kind()
    }

    pub(crate) fn encode(
        &mut self,
        input: &SenseVoiceEncoderInput,
    ) -> Result<SenseVoiceEncoderOutput, SenseVoiceEncoderError> {
        let metadata = self.metadata;
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

        let logits = compose_sensevoice_encoder_logits(
            &mut graph,
            input_t,
            frames,
            metadata,
            &self.arena,
            &self.enc_layers,
            &self.tp_layers,
            self.enc_after_norm_weight,
            self.enc_after_norm_bias,
            self.tp_norm_weight,
            self.tp_norm_bias,
            self.ctc_head_weight,
            self.ctc_head_bias,
            self.use_flash_attention,
        )?;
        graph.set_output(logits).map_err(bf("set_output_logits"))?;

        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(input_t, &input.data, "upload_input")
            .map_err(bf("upload_input"))?;

        read_sensevoice_encoder_logits(&mut graph, logits, metadata.vocab_size, frames)
    }

    /// Build the four prompt rows, input scaling, and sinusoidal position
    /// encoding inside the selected backend. The caller supplies only ordinary
    /// LFR+CMVN frontend features and prompt token indices. SenseVoice always
    /// retains complete per-frame logits so host CTC semantics remain intact.
    pub(crate) fn encode_lfr_with_prompt(
        &mut self,
        lfr_features: &[f32],
        prompt_indices: &[usize],
    ) -> Result<SenseVoiceEncoderOutput, SenseVoiceEncoderError> {
        let metadata = self.metadata;
        if lfr_features.is_empty() || !lfr_features.len().is_multiple_of(metadata.feature_dim) {
            return Err(SenseVoiceEncoderError::Shape {
                reason: format!(
                    "SenseVoice LFR payload {} is not a positive multiple of {}",
                    lfr_features.len(),
                    metadata.feature_dim
                ),
            });
        }
        let prompt_ids = prompt_indices
            .iter()
            .map(|&index| {
                if index >= self.prompt_embedding_rows {
                    return Err(SenseVoiceEncoderError::Shape {
                        reason: format!(
                            "SenseVoice prompt index {index} exceeds {} rows",
                            self.prompt_embedding_rows
                        ),
                    });
                }
                i32::try_from(index).map_err(|_| SenseVoiceEncoderError::Shape {
                    reason: format!("SenseVoice prompt index {index} exceeds i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lfr_frames = lfr_features.len() / metadata.feature_dim;
        let frames = prompt_ids.len().checked_add(lfr_frames).ok_or_else(|| {
            SenseVoiceEncoderError::Shape {
                reason: "SenseVoice encoder frame count overflowed".to_string(),
            }
        })?;
        let positions = (1..=frames)
            .map(|position| position as f32)
            .collect::<Vec<_>>();

        if self.reuse_enabled {
            let shape = GgmlGraphShapeKey::new(lfr_frames, prompt_ids.len(), frames, 0);
            let built = {
                let (graph, reused) = self
                    .same_shape
                    .builder_for_shape(&mut self.runner, shape)
                    .map_err(bf("same_shape_session"))?;
                if reused {
                    None
                } else {
                    let (lfr, prompt, position, logits) = compose_lfr_graph(
                        graph,
                        self.metadata,
                        self.prompt_embedding,
                        self.positional_inv_timescales,
                        &self.arena,
                        &self.enc_layers,
                        &self.tp_layers,
                        self.enc_after_norm_weight,
                        self.enc_after_norm_bias,
                        self.tp_norm_weight,
                        self.tp_norm_bias,
                        self.ctc_head_weight,
                        self.ctc_head_bias,
                        self.use_flash_attention,
                        lfr_frames,
                        prompt_ids.len(),
                        frames,
                    )?;
                    graph.set_output(logits).map_err(bf("set_encoder_output"))?;
                    graph
                        .prepare_outputs_for_upload(&[logits])
                        .map_err(bf("prepare_outputs"))?;
                    Some(SenseVoiceReuseTensors {
                        lfr,
                        prompt,
                        position,
                        logits,
                    })
                }
            };
            if let Some(tensors) = built {
                self.reuse_tensors = Some(tensors);
            }
            let graph = self
                .same_shape
                .builder_for_shape(&mut self.runner, shape)
                .map_err(bf("same_shape_session"))?
                .0;
            let tensors = self
                .reuse_tensors
                .as_ref()
                .expect("reuse tensors installed with the session");
            graph
                .set_f32_slice(tensors.lfr, lfr_features, "sensevoice_lfr")
                .map_err(bf("upload_lfr"))?;
            graph
                .set_i32_slice(tensors.prompt, &prompt_ids, "sensevoice_prompt_ids")
                .map_err(bf("upload_prompt_ids"))?;
            graph
                .set_f32_slice(tensors.position, &positions, "sensevoice_positions")
                .map_err(bf("upload_positions"))?;
            return read_sensevoice_encoder_logits(
                graph,
                tensors.logits,
                metadata.vocab_size,
                frames,
            );
        }

        let mut graph = self.runner.start_graph();
        let (lfr, prompt, position, logits) = compose_lfr_graph(
            &mut graph,
            self.metadata,
            self.prompt_embedding,
            self.positional_inv_timescales,
            &self.arena,
            &self.enc_layers,
            &self.tp_layers,
            self.enc_after_norm_weight,
            self.enc_after_norm_bias,
            self.tp_norm_weight,
            self.tp_norm_bias,
            self.ctc_head_weight,
            self.ctc_head_bias,
            self.use_flash_attention,
            lfr_frames,
            prompt_ids.len(),
            frames,
        )?;
        graph.set_output(logits).map_err(bf("set_encoder_output"))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(lfr, lfr_features, "sensevoice_lfr")
            .map_err(bf("upload_lfr"))?;
        graph
            .set_i32_slice(prompt, &prompt_ids, "sensevoice_prompt_ids")
            .map_err(bf("upload_prompt_ids"))?;
        graph
            .set_f32_slice(position, &positions, "sensevoice_positions")
            .map_err(bf("upload_positions"))?;
        read_sensevoice_encoder_logits(&mut graph, logits, metadata.vocab_size, frames)
    }

    /// Same encoder graph as [`Self::encode_lfr_with_prompt`], but each logit
    /// row is visited from a reused host buffer so CTC greedy never materializes
    /// the full `[vocab, frames]` matrix while backend compute buffers are live.
    pub(crate) fn encode_lfr_with_prompt_for_each_frame<F>(
        &mut self,
        lfr_features: &[f32],
        prompt_indices: &[usize],
        mut visit: F,
    ) -> Result<Option<Vec<GgmlSelectionEvidenceRef>>, SenseVoiceEncoderError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), SenseVoiceEncoderError>,
    {
        self.encode_lfr_prompt_rows(lfr_features, prompt_indices, &mut visit)
    }

    fn encode_lfr_prompt_rows<F>(
        &mut self,
        lfr_features: &[f32],
        prompt_indices: &[usize],
        visit: &mut F,
    ) -> Result<Option<Vec<GgmlSelectionEvidenceRef>>, SenseVoiceEncoderError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), SenseVoiceEncoderError>,
    {
        let metadata = self.metadata;
        if lfr_features.is_empty() || !lfr_features.len().is_multiple_of(metadata.feature_dim) {
            return Err(SenseVoiceEncoderError::Shape {
                reason: format!(
                    "SenseVoice LFR payload {} is not a positive multiple of {}",
                    lfr_features.len(),
                    metadata.feature_dim
                ),
            });
        }
        let prompt_ids = prompt_indices
            .iter()
            .map(|&index| {
                if index >= self.prompt_embedding_rows {
                    return Err(SenseVoiceEncoderError::Shape {
                        reason: format!(
                            "SenseVoice prompt index {index} exceeds {} rows",
                            self.prompt_embedding_rows
                        ),
                    });
                }
                i32::try_from(index).map_err(|_| SenseVoiceEncoderError::Shape {
                    reason: format!("SenseVoice prompt index {index} exceeds i32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lfr_frames = lfr_features.len() / metadata.feature_dim;
        let frames = prompt_ids.len().checked_add(lfr_frames).ok_or_else(|| {
            SenseVoiceEncoderError::Shape {
                reason: "SenseVoice encoder frame count overflowed".to_string(),
            }
        })?;
        let positions = (1..=frames)
            .map(|position| position as f32)
            .collect::<Vec<_>>();

        if self.reuse_enabled {
            let shape = GgmlGraphShapeKey::new(lfr_frames, prompt_ids.len(), frames, 0);
            let built = {
                let (graph, reused) = self
                    .same_shape
                    .builder_for_shape(&mut self.runner, shape)
                    .map_err(bf("same_shape_session"))?;
                if reused {
                    None
                } else {
                    let (lfr, prompt, position, logits) = compose_lfr_graph(
                        graph,
                        self.metadata,
                        self.prompt_embedding,
                        self.positional_inv_timescales,
                        &self.arena,
                        &self.enc_layers,
                        &self.tp_layers,
                        self.enc_after_norm_weight,
                        self.enc_after_norm_bias,
                        self.tp_norm_weight,
                        self.tp_norm_bias,
                        self.ctc_head_weight,
                        self.ctc_head_bias,
                        self.use_flash_attention,
                        lfr_frames,
                        prompt_ids.len(),
                        frames,
                    )?;
                    graph.set_output(logits).map_err(bf("set_encoder_output"))?;
                    graph
                        .prepare_outputs_for_upload(&[logits])
                        .map_err(bf("prepare_outputs"))?;
                    Some(SenseVoiceReuseTensors {
                        lfr,
                        prompt,
                        position,
                        logits,
                    })
                }
            };
            if let Some(tensors) = built {
                self.reuse_tensors = Some(tensors);
            }
            let graph = self
                .same_shape
                .builder_for_shape(&mut self.runner, shape)
                .map_err(bf("same_shape_session"))?
                .0;
            let tensors = self
                .reuse_tensors
                .as_ref()
                .expect("reuse tensors installed with the session");
            graph
                .set_f32_slice(tensors.lfr, lfr_features, "sensevoice_lfr")
                .map_err(bf("upload_lfr"))?;
            graph
                .set_i32_slice(tensors.prompt, &prompt_ids, "sensevoice_prompt_ids")
                .map_err(bf("upload_prompt_ids"))?;
            graph
                .set_f32_slice(tensors.position, &positions, "sensevoice_positions")
                .map_err(bf("upload_positions"))?;
            let logits = tensors.logits;
            let mut visit_error = None;
            let evidence = graph
                .compute_output_f32_rows_for_each(
                    logits,
                    metadata.vocab_size,
                    frames,
                    |index, row| {
                        if let Err(error) = visit(index, row) {
                            visit_error = Some(error);
                            return Err(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "sensevoice frame visitor failed",
                            });
                        }
                        Ok(())
                    },
                )
                .map_err(|error| {
                    visit_error.take().unwrap_or_else(|| {
                        SenseVoiceEncoderError::GraphExecutionFailed {
                            reason: error.to_string(),
                        }
                    })
                })?;
            if let Some(error) = visit_error {
                return Err(error);
            }
            return Ok(evidence);
        }

        let mut graph = self.runner.start_graph();
        let (lfr, prompt, position, logits) = compose_lfr_graph(
            &mut graph,
            self.metadata,
            self.prompt_embedding,
            self.positional_inv_timescales,
            &self.arena,
            &self.enc_layers,
            &self.tp_layers,
            self.enc_after_norm_weight,
            self.enc_after_norm_bias,
            self.tp_norm_weight,
            self.tp_norm_bias,
            self.ctc_head_weight,
            self.ctc_head_bias,
            self.use_flash_attention,
            lfr_frames,
            prompt_ids.len(),
            frames,
        )?;
        graph.set_output(logits).map_err(bf("set_encoder_output"))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(bf("prepare_outputs"))?;
        graph
            .set_f32_slice(lfr, lfr_features, "sensevoice_lfr")
            .map_err(bf("upload_lfr"))?;
        graph
            .set_i32_slice(prompt, &prompt_ids, "sensevoice_prompt_ids")
            .map_err(bf("upload_prompt_ids"))?;
        graph
            .set_f32_slice(position, &positions, "sensevoice_positions")
            .map_err(bf("upload_positions"))?;
        let mut visit_error = None;
        let evidence = graph
            .compute_output_f32_rows_for_each(logits, metadata.vocab_size, frames, |index, row| {
                if let Err(error) = visit(index, row) {
                    visit_error = Some(error);
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "sensevoice frame visitor failed",
                    });
                }
                Ok(())
            })
            .map_err(|error| {
                visit_error
                    .take()
                    .unwrap_or_else(|| SenseVoiceEncoderError::GraphExecutionFailed {
                        reason: error.to_string(),
                    })
            })?;
        if let Some(error) = visit_error {
            return Err(error);
        }
        Ok(evidence)
    }

    pub(crate) fn release_transient_compute_memory(
        &mut self,
    ) -> Result<(), SenseVoiceEncoderError> {
        if self.reuse_enabled && self.same_shape.has_session() {
            return Ok(());
        }
        self.runner
            .release_transient_scheduler_working_set()
            .map_err(bf("release_transient_scheduler_working_set"))
    }
}

#[allow(clippy::too_many_arguments)]
fn compose_lfr_graph<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    metadata: SenseVoiceExecutionMetadata,
    prompt_embedding: GgmlLoadedTensor,
    positional_inv_timescales: GgmlStaticTensor,
    arena: &GgmlStaticTensorArena,
    enc_layers: &[LayerArena],
    tp_layers: &[LayerArena],
    enc_after_norm_weight: GgmlStaticTensor,
    enc_after_norm_bias: GgmlStaticTensor,
    tp_norm_weight: GgmlStaticTensor,
    tp_norm_bias: GgmlStaticTensor,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
    use_flash_attention: bool,
    lfr_frames: usize,
    prompt_len: usize,
    frames: usize,
) -> Result<
    (
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
    ),
    SenseVoiceEncoderError,
> {
    let lfr = graph
        .new_tensor_2d_f32(metadata.feature_dim, lfr_frames, "sensevoice_lfr")
        .map_err(bf("new_lfr"))?;
    let prompt = graph
        .new_tensor_1d_i32(prompt_len, "sensevoice_prompt_ids")
        .map_err(bf("new_prompt_ids"))?;
    let position = graph
        .new_tensor_2d_f32(1, frames, "sensevoice_positions")
        .map_err(bf("new_positions"))?;
    graph.set_input(lfr).map_err(bf("set_lfr_input"))?;
    graph.set_input(prompt).map_err(bf("set_prompt_input"))?;
    graph
        .set_input(position)
        .map_err(bf("set_position_input"))?;

    let prompt_rows = graph
        .get_rows(prompt_embedding.as_graph_tensor(), prompt)
        .map_err(bf("prompt_get_rows"))?;
    let combined = graph
        .concat(prompt_rows, lfr, 1)
        .map_err(bf("prompt_lfr_concat"))?;
    let scaled = graph
        .scale(combined, (metadata.d_model as f32).sqrt())
        .map_err(bf("input_scale"))?;
    let angles = graph
        .mul_mat(arena.graph_tensor(positional_inv_timescales), position)
        .map_err(bf("positional_outer_product"))?;
    let sin = graph.sin(angles).map_err(bf("positional_sin"))?;
    let cos = graph.cos(angles).map_err(bf("positional_cos"))?;
    let positional = graph.concat(sin, cos, 0).map_err(bf("positional_concat"))?;
    let state = graph
        .add(scaled, positional)
        .map_err(bf("positional_add"))?;
    let logits = compose_sensevoice_encoder_logits(
        graph,
        state,
        frames,
        metadata,
        arena,
        enc_layers,
        tp_layers,
        enc_after_norm_weight,
        enc_after_norm_bias,
        tp_norm_weight,
        tp_norm_bias,
        ctc_head_weight,
        ctc_head_bias,
        use_flash_attention,
    )?;
    Ok((lfr, prompt, position, logits))
}

fn read_sensevoice_encoder_logits<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    logits: GgmlCpuTensor<'a>,
    vocab_size: usize,
    frames: usize,
) -> Result<SenseVoiceEncoderOutput, SenseVoiceEncoderError> {
    if vocab_size.checked_mul(frames).is_none() {
        return Err(SenseVoiceEncoderError::Shape {
            reason: "SenseVoice logits size overflowed".to_string(),
        });
    }
    let output = graph
        .compute_output_f32_rows_with_evidence(logits, vocab_size, frames)
        .map_err(|error| SenseVoiceEncoderError::GraphExecutionFailed {
            reason: error.to_string(),
        })?;
    let (logits, frame_compute) = output.into_parts();
    Ok(SenseVoiceEncoderOutput {
        frame_count: frames,
        vocab_size,
        logits,
        frame_compute,
    })
}

#[allow(clippy::too_many_arguments)]
fn compose_sensevoice_encoder_logits<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    mut state: GgmlCpuTensor<'a>,
    frames: usize,
    metadata: SenseVoiceExecutionMetadata,
    arena: &GgmlStaticTensorArena,
    enc_layers: &[LayerArena],
    tp_layers: &[LayerArena],
    enc_after_norm_weight: GgmlStaticTensor,
    enc_after_norm_bias: GgmlStaticTensor,
    tp_norm_weight: GgmlStaticTensor,
    tp_norm_bias: GgmlStaticTensor,
    ctc_head_weight: WeightSlot,
    ctc_head_bias: GgmlStaticTensor,
    use_flash_attention: bool,
) -> Result<GgmlCpuTensor<'a>, SenseVoiceEncoderError> {
    let map = |step, source| SenseVoiceEncoderError::GraphBuildFailed { step, source };
    for handles in enc_layers {
        state = sanm_fsmn_encoder_layer(
            graph,
            state,
            SanMFsmnBlockConfig {
                d_model: metadata.d_model,
                input_dim: handles.input_dim,
                attention_heads: metadata.n_heads,
                head_dim: metadata.head_dim,
                frame_count: frames,
                fsmn_kernel: metadata.fsmn_kernel,
                layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
                use_flash_attention,
            },
            sanm_weights(arena, handles),
            map,
        )?;
    }
    state = apply_affine_layer_norm(
        graph,
        state,
        ENCODER_LAYER_NORM_EPSILON,
        arena.graph_tensor(enc_after_norm_weight),
        arena.graph_tensor(enc_after_norm_bias),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "enc_after_norm",
            bias: "enc_after_norm",
        },
        map,
    )?;
    for handles in tp_layers {
        state = sanm_fsmn_encoder_layer(
            graph,
            state,
            SanMFsmnBlockConfig {
                d_model: metadata.d_model,
                input_dim: handles.input_dim,
                attention_heads: metadata.n_heads,
                head_dim: metadata.head_dim,
                frame_count: frames,
                fsmn_kernel: metadata.fsmn_kernel,
                layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
                use_flash_attention,
            },
            sanm_weights(arena, handles),
            map,
        )?;
    }
    state = apply_affine_layer_norm(
        graph,
        state,
        ENCODER_LAYER_NORM_EPSILON,
        arena.graph_tensor(tp_norm_weight),
        arena.graph_tensor(tp_norm_bias),
        AffineLayerNormSteps {
            norm: "ggml_norm(layer_norm)",
            scale: "tp_norm",
            bias: "tp_norm",
        },
        map,
    )?;
    let head = graph
        .reshape_2d(
            ctc_head_weight.graph(arena),
            metadata.d_model,
            metadata.vocab_size,
        )
        .map_err(bf("ctc_head_reshape"))?;
    let logits = graph.mul_mat(head, state).map_err(bf("ctc_head_matmul"))?;
    graph
        .add(logits, arena.graph_tensor(ctc_head_bias))
        .map_err(bf("ctc_head_bias"))
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

fn sanm_weights<'a>(arena: &GgmlStaticTensorArena, h: &LayerArena) -> SanMFsmnBlockWeights<'a> {
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
    use crate::ggml_runtime::{
        build_runtime_tensor_reader_from_preflight, load_runtime_source_metadata_and_tensor_index,
    };
    use crate::models::sensevoice::encoder_weights::load_sensevoice_encoder_weights;
    use crate::models::sensevoice::runtime_contract::parse_sensevoice_execution_metadata;

    #[test]
    fn encoder_same_shape_reuse_is_not_gated_on_decode_reusable_graph() {
        const SRC: &str = include_str!("encoder_graph.rs");
        assert!(
            crate::ggml_runtime::encoder_same_shape_reuse_is_enabled(),
            "encoder same-shape reuse must stay on when decode evidence is FreshGraph"
        );
        let forbidden = format!(
            "reuse_enabled: reuse_mode == {}::ReusableGraph",
            "GgmlDecodeReuseMode"
        );
        assert!(
            !SRC.contains(&forbidden),
            "SenseVoice encoder same-shape must not bind to decode capture evidence"
        );
        assert!(
            SRC.contains("encoder_same_shape_reuse_is_enabled()"),
            "SenseVoice encoder must take same-shape reuse from the shared helper"
        );
    }

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

        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let gguf_metadata = preflight.metadata.as_ref();
        let metadata = parse_sensevoice_execution_metadata(gguf_metadata).expect("contract");
        let weights = load_sensevoice_encoder_weights(&reader, &metadata).expect("weights");

        // zh prompt: [lang=3, event=1, emotion=2, textnorm(woitn)=15].
        let embed = &weights.prompt_embed;
        let dim = metadata.feature_dim;
        let row = |i: usize| &embed.values[i * dim..(i + 1) * dim];
        let prompt = [row(3), row(1), row(2), row(15)];
        let input =
            build_sensevoice_encoder_input(&prompt, &lfr, dim, metadata.d_model).expect("input");
        assert_eq!(input.n_frames, ref_logits.len() / metadata.vocab_size);

        let mut graph = SenseVoiceEncoderGraph::new(
            &weights,
            metadata,
            &preflight,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph,
        )
        .expect("graph");
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
