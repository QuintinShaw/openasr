//! firered-aed Transformer decoder ggml graph (Stage 3).
//!
//! Faithfully reproduces `fireredasr/models/module/transformer_decoder.py`: a
//! standard ESPnet/WeNet-lineage pre-norm Transformer decoder --
//! `decoder.tgt_word_emb` scaled by `sqrt(d_model)` plus the baked absolute
//! sinusoidal `decoder.positional_encoding.pe` -> 16 x DecoderLayer -> final
//! affine LayerNorm (`decoder.layer_norm_out`) -> untied `decoder.tgt_word_prj`
//! (bias-free). Each DecoderLayer is pre-norm: `norm -> causal self-attn ->
//! residual`, `norm -> cross-attn on the encoder output -> residual`, `norm ->
//! GELU FFN -> residual`. `self_attn.w_ks` and `cross_attn.w_ks` are upstream
//! bias-free linears (see [`super::decoder_weights`]); this graph supplies one
//! shared zero bias tensor for both.
//!
//! Built on the shared incremental seq2seq decoder block
//! ([`crate::nn::decoder::seq2seq_layer`]): pre-norm causal self-attention
//! with an f16 KV cache, pre-norm cross-attention over cross-KV precomputed
//! once from the encoder output, and a GELU feed-forward. On the
//! single-backend GPU path (`nn::decoder::reusable_decode_graph_supported`:
//! GPU-class backend, scheduler off -- the Metal default, see
//! [`super::graph_config`]) the single-token incremental step runs through a
//! build-once/re-run [`Seq2SeqReusableDecodeGraph`] (fixed-span self-KV via
//! `set_rows` + an externally-uploaded attention mask, the cohere/moonshine
//! pattern), eliminating the per-token graph rebuild; prefill and every CPU
//! (or scheduler-on) step keep the rebuild-per-step path, which stays the
//! correctness baseline (CPU direct execution mis-recomputes reused graphs
//! with in-place KV writes -- a ggml-level limit, not a firered one).

#![allow(dead_code)]

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedWeightBindingIdentity, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufRuntimeSourcePreflight,
};
use crate::models::decode_policy_component_registry::{
    BuiltinSeq2SeqDecodePolicyConfigInput, run_builtin_seq2seq_decode_policy,
};
use crate::models::device_greedy_token::{
    DeviceGreedyStepOutputMode, first_max_argmax_reverse_indices,
    first_max_token_id_from_reversed_argmax,
};
use crate::models::seq2seq_decoder_state::Seq2SeqDecoderState;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    Seq2SeqGreedyDecodeStopReason,
};
use crate::nn::decoder::{
    CrossKvHandle, SelfKvHandle, Seq2SeqLayerConfig, Seq2SeqLayerWeights,
    Seq2SeqReusableDecodeGraph, build_causal_mask_f16_bits, build_fixed_kv_attention_mask_bits,
    reusable_decode_graph_supported, reusable_decode_graph_supported_for_runner, seq2seq_layer,
};
use crate::nn::ffn::FeedForwardActivation;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::decoder_weights::{FireRedDecoderWeights, FireRedDecoderWeightsError};
use super::graph_config::firered_decoder_graph_config;
use super::runtime_contract::FireRedAedExecutionMetadata;

const FIRERED_DECODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
/// Static tensors created directly in the decoder's arena (not through
/// `start_graph`): one shared zero-bias vector plus, per decoder layer, a
/// cross-K/cross-V pair and a self-K/self-V pair.
const FIRERED_DECODER_ARENA_TENSORS_PER_LAYER: usize = 4;
const FIRERED_DECODER_ARENA_FIXED_TENSORS: usize = 1;

#[derive(Debug, Error)]
pub(crate) enum FireRedDecoderError {
    #[error("firered-aed decoder weights: {0}")]
    Weights(#[from] FireRedDecoderWeightsError),
    #[error("firered-aed decoder input is invalid: {reason}")]
    InvalidInput { reason: String },
    #[error("firered-aed decoder graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("firered-aed decoder graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("firered-aed decoder shape overflowed")]
    ShapeOverflow,
}

fn map_err(step: &'static str, source: GgmlCpuGraphError) -> FireRedDecoderError {
    FireRedDecoderError::GraphBuildFailed { step, source }
}

/// Byte capacity for the decoder's static-tensor arena context. Like the
/// graph context above, this is a `no_alloc` metadata pool: `start_static_tensor_arena`
/// only needs room for the `ggml_tensor` struct + name of each tensor created
/// in it (one shared zero-bias vector, plus a cross-K/V and self-K/V pair per
/// decoder layer); the real cross-KV/self-KV bytes are allocated afterwards
/// into their own backend buffer sized from the tensors' actual shapes
/// (`ggml_backend_alloc_ctx_tensors`), independent of this context's size.
/// Previously hardcoded to a flat 256 MiB regardless of layer count.
fn firered_decoder_arena_context_bytes(
    decoder_n_layers: usize,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
) -> usize {
    let tensor_count = FIRERED_DECODER_ARENA_FIXED_TENSORS
        .saturating_add(usize::from(
            greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1,
        ))
        .saturating_add(FIRERED_DECODER_ARENA_TENSORS_PER_LAYER.saturating_mul(decoder_n_layers));
    GgmlCpuGraphConfig::metadata_context_bytes(tensor_count)
}

#[derive(Clone, Copy)]
struct FireRedDecoderCrossCacheLayer {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
}

#[derive(Clone, Copy)]
struct FireRedDecoderSelfKvLayer {
    key: GgmlStaticTensor,
    value: GgmlStaticTensor,
}

/// Owns the decoder's mmap'd weight context plus its persistent cross/self KV
/// arena. Construction uses the planner's stable resident spans; each call
/// activates a logical shape inside those spans and never reallocates them.
pub(crate) struct FireRedDecoderGraphRuntime {
    // `reuse` holds raw pointers into `runner`, `arena`, and the weight
    // context, so it must be declared first and dropped first (same drop-order
    // contract as cohere's decoder runtime).
    reuse: Option<Seq2SeqReusableDecodeGraph>,
    runner: GgmlCpuGraphRunner,
    _loaded: GgmlLoadedWeightContext,
    weights: FireRedDecoderWeights,
    metadata: FireRedAedExecutionMetadata,
    /// The `no_alloc` metadata context size used for `runner`'s own graph
    /// context; reused verbatim for `start_persistent_graph_session` in
    /// [`Self::build_reusable_decode_graph`] so it does not have to be
    /// recomputed from a hardcoded constant.
    persistent_graph_context_bytes: usize,
    arena: GgmlStaticTensorArena,
    /// Shared zero bias for the two bias-free K projections (self-attn and
    /// cross-attn `w_ks`), length `d_model`.
    zero_bias: GgmlStaticTensor,
    /// Reverse vocabulary rows used to preserve the shared driver's first-max
    /// tie semantics when a direct CUDA/Vulkan graph returns only top-1.
    argmax_reverse_indices: Option<GgmlStaticTensor>,
    cross_layers: Vec<FireRedDecoderCrossCacheLayer>,
    self_kv_layers: Vec<FireRedDecoderSelfKvLayer>,
    decoder_state: Seq2SeqDecoderState,
    /// Stable planner-reserved column count of every cross-KV tensor.
    cross_capacity_frames: usize,
    /// The CURRENT utterance's actual encoder frame count -- always
    /// `<= cross_capacity_frames` -- set by
    /// [`Self::populate_cross_attention_cache`] and read back by
    /// [`Self::compute_step_logits`]'s cross-attention view. `0` before the
    /// first populate call.
    cross_frame_count: usize,
    /// The cross-frame-count [`Self::reuse`]'s persistent graph was last
    /// built for (0 if never built). `compute_reused_incremental_step_logits`
    /// compares this against the current [`Self::cross_frame_count`] and
    /// rebuilds the (cheap, metadata-only) reusable graph whenever a
    /// differently-sized chunk swaps in, since
    /// [`Self::build_reusable_decode_graph`] bakes the cross-attention view's
    /// frame count into the persistent graph's topology at build time.
    reuse_cross_frame_count: usize,
    cached_positions: usize,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
}

/// The static-tensor arena plus everything allocated directly in it
/// (cross-KV/self-KV caches, shared zero-bias) -- everything about a
/// [`FireRedDecoderGraphRuntime`] that depends on the stable resident spans.
/// Kept separate from `runner`/`_loaded`/`weights`, which borrow the mmap'd
/// GGUF context independently of decoder-state capacity.
struct FireRedDecoderArenaState {
    arena: GgmlStaticTensorArena,
    zero_bias: GgmlStaticTensor,
    argmax_reverse_indices: Option<GgmlStaticTensor>,
    cross_layers: Vec<FireRedDecoderCrossCacheLayer>,
    self_kv_layers: Vec<FireRedDecoderSelfKvLayer>,
}

fn build_firered_decoder_arena_state(
    runner: &GgmlCpuGraphRunner,
    metadata: &FireRedAedExecutionMetadata,
    self_kv_capacity_positions: usize,
    cross_capacity_frames: usize,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
) -> Result<FireRedDecoderArenaState, FireRedDecoderError> {
    let arena = runner
        .start_static_tensor_arena(firered_decoder_arena_context_bytes(
            metadata.decoder_n_layers,
            greedy_step_output_mode,
        ))
        .map_err(|source| map_err("static_tensor_arena", source))?;
    let zero_bias = arena
        .new_tensor_1d_f32(metadata.d_model, "firered_dec_zero_bias")
        .map_err(|source| map_err("zero_bias_alloc", source))?;
    let argmax_reverse_indices =
        if greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            Some(
                arena
                    .new_tensor_1d_i32(metadata.vocab_size, "firered_dec_argmax_reverse_indices")
                    .map_err(|source| map_err("argmax_reverse_indices_alloc", source))?,
            )
        } else {
            None
        };
    let mut cross_layers = Vec::with_capacity(metadata.decoder_n_layers);
    let mut self_kv_layers = Vec::with_capacity(metadata.decoder_n_layers);
    for _ in 0..metadata.decoder_n_layers {
        cross_layers.push(FireRedDecoderCrossCacheLayer {
            key: arena
                .new_tensor_2d_f32(
                    metadata.d_model,
                    cross_capacity_frames,
                    "firered_dec_cross_k",
                )
                .map_err(|source| map_err("cross_k_alloc", source))?,
            value: arena
                .new_tensor_2d_f32(
                    metadata.d_model,
                    cross_capacity_frames,
                    "firered_dec_cross_v",
                )
                .map_err(|source| map_err("cross_v_alloc", source))?,
        });
        self_kv_layers.push(FireRedDecoderSelfKvLayer {
            key: arena
                .new_tensor_3d_f16(
                    metadata.head_dim,
                    self_kv_capacity_positions,
                    metadata.n_heads,
                    "firered_dec_self_k",
                )
                .map_err(|source| map_err("self_k_alloc", source))?,
            value: arena
                .new_tensor_3d_f16(
                    metadata.head_dim,
                    self_kv_capacity_positions,
                    metadata.n_heads,
                    "firered_dec_self_v",
                )
                .map_err(|source| map_err("self_v_alloc", source))?,
        });
    }
    let mut arena = arena;
    arena
        .set_f32_slice(
            zero_bias,
            &vec![0.0f32; metadata.d_model],
            "firered_dec_zero_bias",
        )
        .map_err(|source| map_err("zero_bias_upload", source))?;
    if let Some(reverse_indices) = argmax_reverse_indices {
        arena
            .set_i32_slice(
                reverse_indices,
                &first_max_argmax_reverse_indices(metadata.vocab_size)
                    .map_err(|source| map_err("argmax_reverse_indices", source))?,
                "firered_dec_argmax_reverse_indices",
            )
            .map_err(|source| map_err("argmax_reverse_indices_upload", source))?;
    }

    // Zero-fill the persistent self-KV tensors so the fixed-span reusable
    // decode graph's masked (not-yet-written) rows never feed uninitialized
    // f16 bit patterns (potentially NaN) into flash attention: the kernel
    // skips fully `-inf`-masked KV blocks, but the partially masked boundary
    // block is still computed lane-by-lane, and `NaN + (-inf)` poisons the
    // softmax. Same convention as `allocate_zeroed_llm_resident_kv_arena`
    // (the all-zero f16 bit pattern is 0.0).
    //
    // Gated to runners where the reusable decode graph can actually activate
    // (`reusable_decode_graph_supported_for_runner` is a pure function of the
    // runner's backend + scheduler config, fixed for the runner's lifetime):
    // the rebuild-per-step path views only written rows, so on CPU /
    // scheduler-on runners no unwritten row is ever read and the fill is pure
    // waste -- worse than waste, actually, because touching every byte of the
    // full planner-reserved cache commits all of its pages up front where the
    // untouched malloc'd CPU arena
    // pages would otherwise stay uncommitted until the decode actually wrote
    // them.
    if reusable_decode_graph_supported_for_runner(runner) {
        let self_kv_tensor_bytes = metadata
            .head_dim
            .checked_mul(self_kv_capacity_positions)
            .and_then(|value| value.checked_mul(metadata.n_heads))
            .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        let self_kv_zero = vec![0_u8; self_kv_tensor_bytes];
        for self_kv in &self_kv_layers {
            arena
                .set_bytes_slice(self_kv.key, &self_kv_zero, "firered_dec_self_k")
                .map_err(|source| map_err("self_k_zero_fill", source))?;
            arena
                .set_bytes_slice(self_kv.value, &self_kv_zero, "firered_dec_self_v")
                .map_err(|source| map_err("self_v_zero_fill", source))?;
        }
    }

    Ok(FireRedDecoderArenaState {
        arena,
        zero_bias,
        argmax_reverse_indices,
        cross_layers,
        self_kv_layers,
    })
}

impl FireRedDecoderGraphRuntime {
    /// Provisional SystemMemory quote for the family-owned Rust containers in
    /// one resident decoder. Self-attention and cross-attention capacities are
    /// kept as independent multiplicands; adding their position axes would
    /// describe no allocation the runtime actually owns.
    pub(crate) fn system_memory_quote(
        metadata: FireRedAedExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
        pack_content_id: &str,
    ) -> Result<crate::models::system_memory_owner::SystemMemoryAllocationQuote, String> {
        decoder_state
            .validate()
            .map_err(|error| error.to_string())?;
        let retained = Self::quoted_retained_system_memory_bytes(metadata)?;
        let config = firered_decoder_graph_config(backend);
        let transient = firered_decoder_construction_transient_system_memory_bytes(
            metadata,
            decoder_state,
            reusable_decode_graph_supported(config.backend, config.use_scheduler),
            greedy_step_output_mode,
        )?;
        let peak = retained.checked_add(transient).ok_or_else(|| {
            "firered-aed decoder SystemMemory construction peak overflowed".to_string()
        })?;
        let capacity = decoder_state.resident_capacity();
        crate::models::system_memory_owner::SystemMemoryAllocationQuote::new(
            format!(
                "firered-aed-decoder-runtime:{pack_content_id}:self={}:cross={}",
                capacity.self_attention_positions, capacity.cross_attention_positions
            ),
            peak,
            retained,
        )
        .map_err(|error| error.to_string())
    }

    fn quoted_retained_system_memory_bytes(
        metadata: FireRedAedExecutionMetadata,
    ) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            FireRedDecoderWeights::quoted_retained_system_memory_bytes(metadata.decoder_n_layers)?,
            "firered-aed decoder weight handles",
        )?;
        bytes.add(
            quoted_vec_capacity_bytes::<FireRedDecoderCrossCacheLayer>(
                metadata.decoder_n_layers,
                "firered-aed cross-KV handles",
            )?,
            "firered-aed cross-KV handles",
        )?;
        bytes.add(
            quoted_vec_capacity_bytes::<FireRedDecoderSelfKvLayer>(
                metadata.decoder_n_layers,
                "firered-aed self-KV handles",
            )?,
            "firered-aed self-KV handles",
        )?;
        Ok(bytes.finish())
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.weights.retained_system_memory_bytes()?,
            "firered-aed decoder weight handles",
        )?;
        bytes.add_vec(&self.cross_layers, "firered-aed cross-KV handles")?;
        bytes.add_vec(&self.self_kv_layers, "firered-aed self-KV handles")?;
        Ok(bytes.finish())
    }

    pub(crate) fn graph_lane(&self) -> (crate::ggml_runtime::GgmlCpuGraphBackend, bool) {
        (self.runner.backend_kind(), self.runner.uses_scheduler())
    }

    pub(crate) fn loaded_weight_binding_identity(&self) -> GgmlLoadedWeightBindingIdentity {
        self.runner.loaded_weight_binding_identity(&self._loaded)
    }

    /// Construction staging has already been released when the actor owner is
    /// published. Recompute its exact requested capacity from the runtime's
    /// resolved backend and resident self-KV shape for post-build validation.
    pub(crate) fn construction_transient_system_memory_bytes(&self) -> Result<u64, String> {
        firered_decoder_construction_transient_system_memory_bytes(
            self.metadata,
            self.decoder_state,
            reusable_decode_graph_supported_for_runner(&self.runner),
            self.greedy_step_output_mode,
        )
    }

    pub(crate) fn new(
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FireRedDecoderError> {
        Self::new_with_greedy_step_output_mode(
            preflight,
            metadata,
            decoder_state,
            backend,
            DeviceGreedyStepOutputMode::FullLogits,
        )
    }

    pub(crate) fn new_with_greedy_step_output_mode(
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        decoder_state: Seq2SeqDecoderState,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Self, FireRedDecoderError> {
        decoder_state
            .validate()
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        decoder_state
            .self_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::SelfAttentionKv,
                metadata.decoder_pe_len,
            )
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        decoder_state
            .cross_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::CrossAttentionKv,
                metadata.encoder_max_frames(),
            )
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        let cross_capacity_frames = decoder_state.cross_attention.resident_positions;
        let config = firered_decoder_graph_config(backend);
        let persistent_graph_context_bytes = config.context_bytes;
        let runner =
            GgmlCpuGraphRunner::new(config).map_err(|source| map_err("runner_init", source))?;
        if greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1
            && !reusable_decode_graph_supported_for_runner(&runner)
        {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "device top-1 requires a direct GPU-class decoder runner".to_string(),
            });
        }
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let weights = FireRedDecoderWeights::load(&loaded, metadata.decoder_n_layers)?;
        let arena_state = build_firered_decoder_arena_state(
            &runner,
            &metadata,
            decoder_state.self_attention.resident_positions,
            cross_capacity_frames,
            greedy_step_output_mode,
        )?;

        Ok(Self {
            reuse: None,
            runner,
            _loaded: loaded,
            weights,
            metadata,
            persistent_graph_context_bytes,
            arena: arena_state.arena,
            zero_bias: arena_state.zero_bias,
            argmax_reverse_indices: arena_state.argmax_reverse_indices,
            cross_layers: arena_state.cross_layers,
            self_kv_layers: arena_state.self_kv_layers,
            decoder_state,
            cross_capacity_frames,
            cross_frame_count: decoder_state.cross_attention.logical_positions,
            reuse_cross_frame_count: 0,
            cached_positions: 0,
            greedy_step_output_mode,
        })
    }

    pub(crate) fn activate_decoder_state(
        &mut self,
        decoder_state: Seq2SeqDecoderState,
    ) -> Result<(), FireRedDecoderError> {
        decoder_state
            .validate()
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        if decoder_state.self_attention.resident_positions
            != self.decoder_state.self_attention.resident_positions
            || decoder_state.cross_attention.resident_positions
                != self.decoder_state.cross_attention.resident_positions
        {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "firered-aed cached decoder resident capacity mismatch: cached self/cross={}/{}, requested={}/{}",
                    self.decoder_state.self_attention.resident_positions,
                    self.decoder_state.cross_attention.resident_positions,
                    decoder_state.self_attention.resident_positions,
                    decoder_state.cross_attention.resident_positions,
                ),
            });
        }
        decoder_state
            .self_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::SelfAttentionKv,
                self.metadata.decoder_pe_len,
            )
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        decoder_state
            .cross_attention
            .validate_runtime_ceiling(
                crate::capacity::topology::StateKind::CrossAttentionKv,
                self.metadata.encoder_max_frames(),
            )
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        if decoder_state.self_attention.logical_positions
            != self.decoder_state.self_attention.logical_positions
            || decoder_state.cross_attention.logical_positions
                != self.decoder_state.cross_attention.logical_positions
        {
            self.reuse = None;
            self.reuse_cross_frame_count = 0;
        }
        self.decoder_state = decoder_state;
        self.cross_frame_count = decoder_state.cross_attention.logical_positions;
        Ok(())
    }

    /// Precompute cross-attention K/V for every layer from the encoder output
    /// and write them into the persistent cross-KV cache. Must be called once
    /// before the first [`Self::compute_step_logits`]. `frame_count` must
    /// equal the planner's logical cross shape and fit the stable resident
    /// span; mismatches fail closed before any write.
    pub(crate) fn populate_cross_attention_cache(
        &mut self,
        encoder_rows: &[f32],
        frame_count: usize,
    ) -> Result<(), FireRedDecoderError> {
        self.decoder_state
            .cross_attention
            .validate_exact_shape(
                crate::capacity::topology::StateKind::CrossAttentionKv,
                frame_count,
            )
            .map_err(|error| FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            })?;
        if frame_count > self.cross_capacity_frames {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "firered-aed logical cross shape {frame_count} exceeds resident capacity {}",
                    self.cross_capacity_frames
                ),
            });
        }
        let d_model = self.metadata.d_model;
        let expected = frame_count
            .checked_mul(d_model)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        if encoder_rows.len() != expected {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "encoder rows length mismatch: got {}, expected {expected}",
                    encoder_rows.len()
                ),
            });
        }
        self.cached_positions = 0;

        let mut graph = self.runner.start_graph();
        let encoder_tensor = graph
            .new_tensor_2d_f32(d_model, frame_count, "firered_dec_encoder_rows")
            .map_err(|source| map_err("encoder_rows_alloc", source))?;
        graph
            .set_input(encoder_tensor)
            .map_err(|source| map_err("encoder_rows_input", source))?;

        let zero_bias_tensor = self.arena.graph_tensor(self.zero_bias);
        // Row stride for a view into the (capacity-sized) cross-KV arena
        // tensors: `frame_count` (this utterance's actual encoder frame
        // count) may be smaller than the tensor's allocated column count
        // (`cross_capacity_frames`), so every write below targets a
        // contiguous-prefix VIEW of exactly `frame_count` columns rather than
        // the full capacity-sized tensor -- the trailing (never populated)
        // columns are simply never read, since `compute_step_logits` also
        // views only `self.cross_frame_count` columns for cross-attention.
        let cross_row_stride = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        let mut last_value_rows = None;
        for (layer, cross) in self.weights.layers.iter().zip(&self.cross_layers) {
            let key_rows = apply_linear_with_bias(
                &mut graph,
                encoder_tensor,
                layer.cross_attn_k_weight.as_graph_tensor(),
                zero_bias_tensor,
                "cross_cache_k",
            )?;
            let key_target_full = self.arena.graph_tensor(cross.key);
            let key_target = graph
                .view_2d(key_target_full, d_model, frame_count, cross_row_stride, 0)
                .map_err(|source| map_err("cross_cache_k_view", source))?;
            let write_key = graph
                .cpy(key_rows, key_target)
                .map_err(|source| map_err("cross_cache_k_write", source))?;
            graph
                .add_side_effect_root(write_key)
                .map_err(|source| map_err("cross_cache_k_root", source))?;

            let value_rows = apply_linear_with_bias(
                &mut graph,
                encoder_tensor,
                layer.cross_attn_v_weight.as_graph_tensor(),
                layer.cross_attn_v_bias.as_graph_tensor(),
                "cross_cache_v",
            )?;
            let value_target_full = self.arena.graph_tensor(cross.value);
            let value_target = graph
                .view_2d(value_target_full, d_model, frame_count, cross_row_stride, 0)
                .map_err(|source| map_err("cross_cache_v_view", source))?;
            let write_value = graph
                .cpy(value_rows, value_target)
                .map_err(|source| map_err("cross_cache_v_write", source))?;
            graph
                .add_side_effect_root(write_value)
                .map_err(|source| map_err("cross_cache_v_root", source))?;
            last_value_rows = Some(value_rows);
        }
        let output_root = last_value_rows.ok_or(FireRedDecoderError::InvalidInput {
            reason: "decoder must have at least one layer".to_string(),
        })?;
        graph
            .set_output(output_root)
            .map_err(|source| map_err("cross_cache_set_output", source))?;
        // Allocate the cross-KV precompute graph (side-effect cpy writes plus the
        // output root) through the scheduler's gallocr for liveness-based buffer
        // reuse before uploading the encoder rows -- same ordering as the encoder
        // forward and the sibling cohere/moonshine decoders.
        graph
            .prepare_outputs_for_upload(&[output_root])
            .map_err(|source| map_err("cross_cache_prepare_outputs", source))?;
        graph
            .set_f32_slice(encoder_tensor, encoder_rows, "firered_dec_encoder_rows")
            .map_err(|source| map_err("encoder_rows_upload", source))?;
        let expected_len = frame_count
            .checked_mul(d_model)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        graph
            .compute_output_f32(output_root, expected_len)
            .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        self.cross_frame_count = frame_count;
        Ok(())
    }

    /// Compute logits for the next token given the full token prefix so far
    /// (prompt + already-generated tokens). Incremental: after the first call
    /// (which may prefill more than one token), every subsequent call must
    /// append exactly one new token. On the single-backend GPU path a
    /// single-token incremental step runs through the build-once/re-run
    /// reusable decode graph ([`Self::compute_reused_incremental_step_logits`]);
    /// everywhere else (prefill, CPU, scheduler-on) it rebuilds a fresh graph.
    pub(crate) fn compute_step_logits(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        let output = self.compute_step_output_impl(
            decoder_tokens,
            true,
            DeviceGreedyStepOutputMode::FullLogits,
        )?;
        debug_assert!(output.greedy_token_hint.is_none());
        Ok(output.logits)
    }

    /// Test-only bypass of the reusable-graph dispatch: always rebuild a
    /// fresh graph for this step, so the reused-vs-rebuilt byte-identity test
    /// can drive both paths on the same backend against the same inputs.
    #[cfg(test)]
    pub(crate) fn compute_step_logits_forcing_fresh_graph(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        let output = self.compute_step_output_impl(
            decoder_tokens,
            false,
            DeviceGreedyStepOutputMode::FullLogits,
        )?;
        debug_assert!(output.greedy_token_hint.is_none());
        Ok(output.logits)
    }

    /// Whether this step went (or will go) through the reusable decode graph
    /// -- exposed so tests can assert the reuse path is actually live rather
    /// than silently falling back to the rebuild path.
    #[cfg(test)]
    pub(crate) fn has_active_reuse_graph(&self) -> bool {
        self.reuse.is_some()
    }

    fn compute_step_output(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, FireRedDecoderError> {
        self.compute_step_output_impl(decoder_tokens, true, self.greedy_step_output_mode)
    }

    fn compute_step_output_impl(
        &mut self,
        decoder_tokens: &[u32],
        allow_reuse: bool,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, FireRedDecoderError> {
        let total_prefix_tokens = decoder_tokens.len();
        if total_prefix_tokens == 0 {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "decoder token_count must be > 0".to_string(),
            });
        }
        let logical_max_positions = self.decoder_state.self_attention.logical_positions;
        if total_prefix_tokens > logical_max_positions {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "decoder token_count {total_prefix_tokens} exceeds max context {}",
                    logical_max_positions
                ),
            });
        }
        let position_offset = self.cached_positions;
        let single_token;
        let decode_tokens: &[u32] = if position_offset == 0 {
            decoder_tokens
        } else {
            if total_prefix_tokens != position_offset.saturating_add(1) {
                return Err(FireRedDecoderError::InvalidInput {
                    reason: format!(
                        "incremental decoder prefix mismatch: got {total_prefix_tokens} tokens, \
                         expected {position_offset} cached + 1"
                    ),
                });
            }
            single_token = [*decoder_tokens.last().expect("checked non-empty above")];
            &single_token
        };
        let token_count = decode_tokens.len();
        let total_token_count = position_offset
            .checked_add(token_count)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        if allow_reuse
            && position_offset > 0
            && token_count == 1
            && self.supports_reusable_decode_graph()
        {
            return self.compute_reused_incremental_step_output(
                decode_tokens[0],
                position_offset,
                output_mode,
            );
        }
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;

        let mut graph = self.runner.start_graph();
        let token_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "firered_dec_tokens")
            .map_err(|source| map_err("tokens_alloc", source))?;
        graph
            .set_input(token_ids_tensor)
            .map_err(|source| map_err("tokens_input", source))?;
        let position_ids_tensor = graph
            .new_tensor_1d_i32(token_count, "firered_dec_positions")
            .map_err(|source| map_err("positions_alloc", source))?;
        graph
            .set_input(position_ids_tensor)
            .map_err(|source| map_err("positions_input", source))?;

        let self_attention_mask = if token_count > 1 {
            let mask = graph
                .new_tensor_3d_f16(token_count, token_count, 1, "firered_dec_self_mask")
                .map_err(|source| map_err("self_mask_alloc", source))?;
            graph
                .set_input(mask)
                .map_err(|source| map_err("self_mask_input", source))?;
            Some(mask)
        } else {
            None
        };

        let token_ids_i32 = tokens_as_i32(decode_tokens)?;
        let position_ids_i32 = position_ids_i32_with_offset(position_offset, token_count)?;

        let token_state = graph
            .get_rows(
                self.weights.token_embedding.as_graph_tensor(),
                token_ids_tensor,
            )
            .map_err(|source| map_err("embed_get_rows", source))?;
        let scaled_token_state = graph
            .scale(token_state, (d_model as f32).sqrt())
            .map_err(|source| map_err("embed_xscale", source))?;
        let position_state = graph
            .get_rows(
                self.weights.positional_encoding.as_graph_tensor(),
                position_ids_tensor,
            )
            .map_err(|source| map_err("position_get_rows", source))?;
        let mut state = graph
            .add(scaled_token_state, position_state)
            .map_err(|source| map_err("embed_add_pos", source))?;

        let zero_bias_tensor = self.arena.graph_tensor(self.zero_bias);
        // Deferred input uploads (mirrors cohere's decoder): every graph-input
        // write is queued and applied AFTER `prepare_outputs_for_upload`, so no
        // upload triggers an independent backend-buffer allocation mid-build and
        // the scheduler's gallocr owns the whole graph's tensor allocation. For
        // firered this queue always stays empty (the shared top-level causal mask
        // means `seq2seq_layer` never emits a per-layer `deferred_self_mask`), but
        // queuing keeps the ordering invariant robust and matches the sibling.
        let mut deferred_self_masks = Vec::new();
        for (layer, (cross, self_kv)) in self
            .weights
            .layers
            .iter()
            .zip(self.cross_layers.iter().zip(&self.self_kv_layers))
        {
            let config = Seq2SeqLayerConfig {
                hidden: d_model,
                attention_heads: heads,
                head_dim,
                token_count,
                n_seq: 1,
                total_token_count,
                position_offset,
                layer_norm_epsilon: FIRERED_DECODER_LAYER_NORM_EPSILON,
                ffn_activation: FeedForwardActivation::Gelu,
                self_kv_max_positions: self.decoder_state.self_attention.resident_positions,
                cross_frame_count: self.cross_frame_count,
                cross_kv_max_positions: self.decoder_state.cross_attention.resident_positions,
                cross_hidden_size: d_model,
            };
            let weights = Seq2SeqLayerWeights {
                self_attn_norm_weight: layer.self_attn_norm_weight.as_graph_tensor(),
                self_attn_norm_bias: layer.self_attn_norm_bias.as_graph_tensor(),
                self_attn_q_weight: layer.self_attn_q_weight.as_graph_tensor(),
                self_attn_q_bias: layer.self_attn_q_bias.as_graph_tensor(),
                self_attn_k_weight: layer.self_attn_k_weight.as_graph_tensor(),
                self_attn_k_bias: zero_bias_tensor,
                self_attn_v_weight: layer.self_attn_v_weight.as_graph_tensor(),
                self_attn_v_bias: layer.self_attn_v_bias.as_graph_tensor(),
                self_attn_o_weight: layer.self_attn_out_weight.as_graph_tensor(),
                self_attn_o_bias: layer.self_attn_out_bias.as_graph_tensor(),
                cross_attn_norm_weight: layer.cross_attn_norm_weight.as_graph_tensor(),
                cross_attn_norm_bias: layer.cross_attn_norm_bias.as_graph_tensor(),
                cross_attn_q_weight: layer.cross_attn_q_weight.as_graph_tensor(),
                cross_attn_q_bias: layer.cross_attn_q_bias.as_graph_tensor(),
                cross_attn_o_weight: layer.cross_attn_out_weight.as_graph_tensor(),
                cross_attn_o_bias: layer.cross_attn_out_bias.as_graph_tensor(),
                ffn_norm_weight: layer.ffn_norm_weight.as_graph_tensor(),
                ffn_norm_bias: layer.ffn_norm_bias.as_graph_tensor(),
                ffn_up_weight: layer.ffn_up_weight.as_graph_tensor(),
                ffn_up_bias: layer.ffn_up_bias.as_graph_tensor(),
                ffn_down_weight: layer.ffn_down_weight.as_graph_tensor(),
                ffn_down_bias: layer.ffn_down_bias.as_graph_tensor(),
            };
            let self_kv_handle = SelfKvHandle {
                key: self.arena.graph_tensor(self_kv.key),
                value: self.arena.graph_tensor(self_kv.value),
                row_indices: None,
                attention_mask: self_attention_mask,
            };
            let cross_kv_handle = CrossKvHandle {
                key: self.arena.graph_tensor(cross.key),
                value: self.arena.graph_tensor(cross.value),
            };
            let block = seq2seq_layer(
                &mut graph,
                state,
                config,
                weights,
                self_kv_handle,
                cross_kv_handle,
                map_err,
            )?;
            if let Some(deferred) = block.deferred_self_mask {
                deferred_self_masks.push(deferred);
            }
            state = block.output;
        }

        state = apply_affine_layer_norm(
            &graph,
            state,
            FIRERED_DECODER_LAYER_NORM_EPSILON,
            self.weights.out_norm_weight.as_graph_tensor(),
            self.weights.out_norm_bias.as_graph_tensor(),
            AffineLayerNormSteps {
                norm: "decoder_out_norm",
                scale: "decoder_out_norm",
                bias: "decoder_out_norm",
            },
            map_err,
        )?;
        let last_state = view_last_token_state(&graph, state, d_model, token_count)?;
        let logits = graph
            .mul_mat(self.weights.out_proj_weight.as_graph_tensor(), last_state)
            .map_err(|source| map_err("output_proj", source))?;
        let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            let reverse_indices =
                self.argmax_reverse_indices
                    .ok_or(FireRedDecoderError::InvalidInput {
                        reason: "device top-1 reverse indices are unavailable".to_string(),
                    })?;
            Some(
                graph
                    .top1_argmax_first_max_reversed(
                        logits,
                        self.arena.graph_tensor(reverse_indices),
                    )
                    .map_err(|source| map_err("output_top1", source))?,
            )
        } else {
            None
        };
        let output_root = top1.unwrap_or(logits);
        graph
            .set_output(output_root)
            .map_err(|source| map_err("set_output", source))?;
        // Allocate the decode graph through the scheduler's gallocr for
        // liveness-based buffer reuse before uploading inputs (mirrors the
        // cohere/moonshine decoders); the queued uploads below then write into the
        // already-allocated input tensors instead of forcing an independent
        // allocation.
        graph
            .prepare_outputs_for_upload(&[output_root])
            .map_err(|source| map_err("prepare_outputs", source))?;

        graph
            .set_i32_slice(token_ids_tensor, &token_ids_i32, "firered_dec_tokens")
            .map_err(|source| map_err("tokens_upload", source))?;
        graph
            .set_i32_slice(
                position_ids_tensor,
                &position_ids_i32,
                "firered_dec_positions",
            )
            .map_err(|source| map_err("positions_upload", source))?;
        if let Some(mask) = self_attention_mask {
            let bits = build_causal_mask_f16_bits(token_count, "firered_dec_self_mask", map_err)?;
            graph
                .set_f16_bits_slice(mask, &bits, "firered_dec_self_mask")
                .map_err(|source| map_err("self_mask_upload", source))?;
        }
        for (mask_tensor, bits) in deferred_self_masks {
            graph
                .set_f16_bits_slice(mask_tensor, &bits, "firered_dec_layer_self_mask")
                .map_err(|source| map_err("layer_self_mask_upload", source))?;
        }

        let output = match top1 {
            Some(top1) => {
                let reversed_token_id = graph
                    .compute_output_i32(top1, 1)
                    .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                        reason: error.to_string(),
                    })?
                    .into_iter()
                    .next()
                    .ok_or_else(|| FireRedDecoderError::GraphExecutionFailed {
                        reason: "device top-1 returned no token id".to_string(),
                    })?;
                Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(map_reversed_top1_token(
                        reversed_token_id,
                        self.metadata.vocab_size,
                    )?),
                }
            }
            None => Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: graph
                    .compute_output_f32(logits, self.metadata.vocab_size)
                    .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                        reason: error.to_string(),
                    })?,
                greedy_token_hint: None,
            },
        };
        self.cached_positions = total_token_count;
        Ok(output)
    }

    /// Reused decode graphs with in-place resident KV are only correct on the
    /// single-backend GPU path (see `nn::decoder::reusable_decode_graph_supported`);
    /// everywhere else the rebuild-per-step path above stays authoritative.
    fn supports_reusable_decode_graph(&self) -> bool {
        reusable_decode_graph_supported_for_runner(&self.runner)
    }

    /// Single-token incremental step through the build-once/re-run persistent
    /// graph: refresh the token/row/position inputs and the fixed-span
    /// attention mask, then recompute -- no graph construction, no
    /// reallocation (the cohere/moonshine reuse pattern). Must produce
    /// byte-identical logits to the rebuild path for the same step; the
    /// masked (`-inf`) tail of the fixed logical self-KV view
    /// contributes exactly zero attention weight, and the underlying
    /// arithmetic per valid position is unchanged.
    fn compute_reused_incremental_step_output(
        &mut self,
        token_id: u32,
        position: usize,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, FireRedDecoderError> {
        let max_positions = self.decoder_state.self_attention.logical_positions;
        if position >= max_positions {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!("decoder position {position} exceeds max context {max_positions}"),
            });
        }
        let token_id_i32 =
            i32::try_from(token_id).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("token id {token_id} does not fit i32"),
            })?;
        let position_i32 =
            i32::try_from(position).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("decoder position {position} does not fit i32"),
            })?;
        let total_tokens = position
            .checked_add(1)
            .ok_or(FireRedDecoderError::ShapeOverflow)?;
        // A cross-frame-count change also forces a rebuild:
        // `build_reusable_decode_graph` bakes the current `cross_frame_count`
        // into the persistent graph's cross-attention view topology, so a
        // graph built for a different (earlier) chunk's frame count would
        // silently attend over the wrong span for this one.
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.is_poisoned()
                    || reuse.max_positions != max_positions
                    || reuse.n_seq != 1
                    || self.reuse_cross_frame_count != self.cross_frame_count
                    || reuse.top1.is_some()
                        != (output_mode == DeviceGreedyStepOutputMode::DeviceTop1)
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph(output_mode)?;
        }

        let reuse = self
            .reuse
            .as_mut()
            .expect("firered reusable decode graph built above");
        let token_tensor = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let attention_mask = reuse.attention_mask;
        let logits = reuse.logits;
        let top1 = reuse.top1;
        let graph = reuse.builder();

        graph
            .set_i32_slice(token_tensor, &[token_id_i32], "firered_reuse_token")
            .map_err(|source| map_err("reuse_token_upload", source))?;
        graph
            .set_i32_slice(row_index, &[position_i32], "firered_reuse_row")
            .map_err(|source| map_err("reuse_row_upload", source))?;
        graph
            .set_i32_slice(position_tensor, &[position_i32], "firered_reuse_position")
            .map_err(|source| map_err("reuse_position_upload", source))?;
        let mask_bits = build_fixed_kv_attention_mask_bits(max_positions, total_tokens)
            .map_err(|source| map_err("reuse_self_mask", source))?;
        graph
            .set_f16_bits_slice(attention_mask, &mask_bits, "firered_reuse_self_mask")
            .map_err(|source| map_err("reuse_self_mask_upload", source))?;

        let output = match top1 {
            Some(top1) => {
                let reversed_token_id = graph
                    .compute_output_i32(top1, 1)
                    .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                        reason: error.to_string(),
                    })?
                    .into_iter()
                    .next()
                    .ok_or_else(|| FireRedDecoderError::GraphExecutionFailed {
                        reason: "reused device top-1 returned no token id".to_string(),
                    })?;
                Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(map_reversed_top1_token(
                        reversed_token_id,
                        self.metadata.vocab_size,
                    )?),
                }
            }
            None => Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: graph
                    .compute_output_f32(logits, self.metadata.vocab_size)
                    .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                        reason: error.to_string(),
                    })?,
                greedy_token_hint: None,
            },
        };
        self.cached_positions = total_tokens;
        Ok(output)
    }

    /// Build the persistent single-token decode graph: the same op sequence as
    /// one `token_count == 1` step of the rebuild path, except the self-KV
    /// write goes through `set_rows` on a runtime row-index input (so the
    /// write slot can move per step without rebuilding) and self-attention
    /// reads the full fixed logical self-KV span under an externally-uploaded
    /// `-inf` tail mask (so the graph shape is constant across steps). The
    /// current `cross_frame_count` is baked into the cross-attention views;
    /// `compute_reused_incremental_step_logits` rebuilds on mismatch.
    fn build_reusable_decode_graph(
        &mut self,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<(), FireRedDecoderError> {
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;
        let max_positions = self.decoder_state.self_attention.logical_positions;
        let resident_self_positions = self.decoder_state.self_attention.resident_positions;
        let resident_cross_positions = self.decoder_state.cross_attention.resident_positions;
        let cross_frame_count = self.cross_frame_count;

        let mut session = self
            .runner
            .start_persistent_graph_session(self.persistent_graph_context_bytes)
            .map_err(|source| map_err("reuse_session", source))?;
        let graph = session.builder();
        let token_id = graph
            .new_tensor_1d_i32(1, "firered_reuse_token")
            .map_err(|source| map_err("reuse_token_alloc", source))?;
        let row_index = graph
            .new_tensor_1d_i32(1, "firered_reuse_row")
            .map_err(|source| map_err("reuse_row_alloc", source))?;
        let position = graph
            .new_tensor_1d_i32(1, "firered_reuse_position")
            .map_err(|source| map_err("reuse_position_alloc", source))?;
        let attention_mask = graph
            .new_tensor_3d_f16(max_positions, 1, 1, "firered_reuse_self_mask")
            .map_err(|source| map_err("reuse_self_mask_alloc", source))?;
        graph
            .set_input(token_id)
            .map_err(|source| map_err("reuse_token_input", source))?;
        graph
            .set_input(row_index)
            .map_err(|source| map_err("reuse_row_input", source))?;
        graph
            .set_input(position)
            .map_err(|source| map_err("reuse_position_input", source))?;
        graph
            .set_input(attention_mask)
            .map_err(|source| map_err("reuse_self_mask_input", source))?;

        let token_state = graph
            .get_rows(self.weights.token_embedding.as_graph_tensor(), token_id)
            .map_err(|source| map_err("reuse_embed_get_rows", source))?;
        let scaled_token_state = graph
            .scale(token_state, (d_model as f32).sqrt())
            .map_err(|source| map_err("reuse_embed_xscale", source))?;
        let position_state = graph
            .get_rows(self.weights.positional_encoding.as_graph_tensor(), position)
            .map_err(|source| map_err("reuse_position_get_rows", source))?;
        let mut state = graph
            .add(scaled_token_state, position_state)
            .map_err(|source| map_err("reuse_embed_add_pos", source))?;

        let zero_bias_tensor = self.arena.graph_tensor(self.zero_bias);
        for (layer, (cross, self_kv)) in self
            .weights
            .layers
            .iter()
            .zip(self.cross_layers.iter().zip(&self.self_kv_layers))
        {
            let config = Seq2SeqLayerConfig {
                hidden: d_model,
                attention_heads: heads,
                head_dim,
                token_count: 1,
                n_seq: 1,
                // Fixed span: attend over the whole self-KV capacity; the
                // per-step mask upload marks positions past the current
                // prefix as `-inf`.
                total_token_count: max_positions,
                position_offset: 0,
                layer_norm_epsilon: FIRERED_DECODER_LAYER_NORM_EPSILON,
                ffn_activation: FeedForwardActivation::Gelu,
                self_kv_max_positions: resident_self_positions,
                cross_frame_count,
                cross_kv_max_positions: resident_cross_positions,
                cross_hidden_size: d_model,
            };
            let weights = Seq2SeqLayerWeights {
                self_attn_norm_weight: layer.self_attn_norm_weight.as_graph_tensor(),
                self_attn_norm_bias: layer.self_attn_norm_bias.as_graph_tensor(),
                self_attn_q_weight: layer.self_attn_q_weight.as_graph_tensor(),
                self_attn_q_bias: layer.self_attn_q_bias.as_graph_tensor(),
                self_attn_k_weight: layer.self_attn_k_weight.as_graph_tensor(),
                self_attn_k_bias: zero_bias_tensor,
                self_attn_v_weight: layer.self_attn_v_weight.as_graph_tensor(),
                self_attn_v_bias: layer.self_attn_v_bias.as_graph_tensor(),
                self_attn_o_weight: layer.self_attn_out_weight.as_graph_tensor(),
                self_attn_o_bias: layer.self_attn_out_bias.as_graph_tensor(),
                cross_attn_norm_weight: layer.cross_attn_norm_weight.as_graph_tensor(),
                cross_attn_norm_bias: layer.cross_attn_norm_bias.as_graph_tensor(),
                cross_attn_q_weight: layer.cross_attn_q_weight.as_graph_tensor(),
                cross_attn_q_bias: layer.cross_attn_q_bias.as_graph_tensor(),
                cross_attn_o_weight: layer.cross_attn_out_weight.as_graph_tensor(),
                cross_attn_o_bias: layer.cross_attn_out_bias.as_graph_tensor(),
                ffn_norm_weight: layer.ffn_norm_weight.as_graph_tensor(),
                ffn_norm_bias: layer.ffn_norm_bias.as_graph_tensor(),
                ffn_up_weight: layer.ffn_up_weight.as_graph_tensor(),
                ffn_up_bias: layer.ffn_up_bias.as_graph_tensor(),
                ffn_down_weight: layer.ffn_down_weight.as_graph_tensor(),
                ffn_down_bias: layer.ffn_down_bias.as_graph_tensor(),
            };
            let self_kv_handle = SelfKvHandle {
                key: self.arena.graph_tensor(self_kv.key),
                value: self.arena.graph_tensor(self_kv.value),
                row_indices: Some(row_index),
                attention_mask: Some(attention_mask),
            };
            let cross_kv_handle = CrossKvHandle {
                key: self.arena.graph_tensor(cross.key),
                value: self.arena.graph_tensor(cross.value),
            };
            let block = seq2seq_layer(
                graph,
                state,
                config,
                weights,
                self_kv_handle,
                cross_kv_handle,
                map_err,
            )?;
            debug_assert!(
                block.deferred_self_mask.is_none(),
                "single-token reuse steps never emit a deferred per-layer mask"
            );
            state = block.output;
        }

        state = apply_affine_layer_norm(
            graph,
            state,
            FIRERED_DECODER_LAYER_NORM_EPSILON,
            self.weights.out_norm_weight.as_graph_tensor(),
            self.weights.out_norm_bias.as_graph_tensor(),
            AffineLayerNormSteps {
                norm: "reuse_decoder_out_norm",
                scale: "reuse_decoder_out_norm",
                bias: "reuse_decoder_out_norm",
            },
            map_err,
        )?;
        let last_state = view_last_token_state(graph, state, d_model, 1)?;
        let logits = graph
            .mul_mat(self.weights.out_proj_weight.as_graph_tensor(), last_state)
            .map_err(|source| map_err("reuse_output_proj", source))?;
        let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            let reverse_indices =
                self.argmax_reverse_indices
                    .ok_or(FireRedDecoderError::InvalidInput {
                        reason: "device top-1 reverse indices are unavailable".to_string(),
                    })?;
            Some(
                graph
                    .top1_argmax_first_max_reversed(
                        logits,
                        self.arena.graph_tensor(reverse_indices),
                    )
                    .map_err(|source| map_err("reuse_output_top1", source))?,
            )
        } else {
            None
        };
        let output_root = top1.unwrap_or(logits);
        graph
            .set_output(output_root)
            .map_err(|source| map_err("reuse_set_output", source))?;
        graph
            .prepare_outputs_for_upload(&[output_root])
            .map_err(|source| map_err("reuse_prepare_outputs", source))?;

        self.reuse = Some(
            Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena_and_optional_top1(
                session,
                max_positions,
                1,
                token_id,
                row_index,
                position,
                attention_mask,
                logits,
                top1,
            ),
        );
        self.reuse_cross_frame_count = cross_frame_count;
        Ok(())
    }
}

fn quoted_vec_capacity_bytes<T>(elements: usize, label: &str) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| format!("{label} quote overflowed"))?;
    u64::try_from(bytes).map_err(|_| format!("{label} quote does not fit u64"))
}

fn firered_decoder_construction_transient_system_memory_bytes(
    metadata: FireRedAedExecutionMetadata,
    decoder_state: Seq2SeqDecoderState,
    reusable_graph_supported: bool,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
) -> Result<u64, String> {
    let zero_bias_bytes = metadata
        .d_model
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| "firered-aed zero-bias staging byte count overflowed".to_string())?;
    let reverse_indices_bytes = if greedy_step_output_mode == DeviceGreedyStepOutputMode::DeviceTop1
    {
        quoted_vec_capacity_bytes::<i32>(
            metadata.vocab_size,
            "firered-aed argmax reverse-index staging",
        )?
    } else {
        0
    };
    let base_transient = zero_bias_bytes.max(reverse_indices_bytes);
    if !reusable_graph_supported {
        return Ok(base_transient);
    }
    let self_kv_zero_bytes = metadata
        .head_dim
        .checked_mul(decoder_state.self_attention.resident_positions)
        .and_then(|value| value.checked_mul(metadata.n_heads))
        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| "firered-aed self-KV zero-fill staging byte count overflowed".to_string())?;
    Ok(base_transient.max(self_kv_zero_bytes))
}

fn map_reversed_top1_token(
    reversed_token_id: i32,
    vocab_size: usize,
) -> Result<u32, FireRedDecoderError> {
    let token_id = first_max_token_id_from_reversed_argmax(reversed_token_id, vocab_size).map_err(
        |error| FireRedDecoderError::GraphExecutionFailed {
            reason: error.to_string(),
        },
    )?;
    u32::try_from(token_id).map_err(|_| FireRedDecoderError::GraphExecutionFailed {
        reason: format!("device top-1 token id {token_id} does not fit u32"),
    })
}

/// firered-aed decodes through the shared seq2seq greedy driver: every step
/// recomputes logits for the full `<sos> ++ generated` prefix (the incremental
/// KV cache inside [`Self::compute_step_logits`] makes this cheap after the
/// prefill). Exact direct CUDA/Vulkan requests return only a first-max top-1
/// hint; every other route returns full logits and keeps the host argmax path.
impl Seq2SeqGreedyDecodeStepExecutor for FireRedDecoderGraphRuntime {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        let prefix: Vec<u32> = input
            .initial_prompt_tokens
            .iter()
            .copied()
            .chain(input.generated_tokens.iter().copied())
            .collect();
        self.compute_step_output(&prefix).map_err(|error| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            }
        })
    }
}

fn apply_linear_with_bias<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    input: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    step: &'static str,
) -> Result<GgmlCpuTensor<'a>, FireRedDecoderError> {
    let projected = graph
        .mul_mat(weight, input)
        .map_err(|source| map_err(step, source))?;
    graph
        .add(projected, bias)
        .map_err(|source| map_err(step, source))
}

fn view_last_token_state<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    hidden: usize,
    prefix_len: usize,
) -> Result<GgmlCpuTensor<'a>, FireRedDecoderError> {
    // No `ggml_cont` needed: `state` is the output of the final affine
    // layer_norm (an `ggml_add` of scale and bias), which is always a freshly
    // allocated contiguous tensor, so `ggml_view_2d` can slice it directly.
    let row_stride = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FireRedDecoderError::ShapeOverflow)?;
    let offset = prefix_len
        .checked_sub(1)
        .and_then(|index| index.checked_mul(row_stride))
        .ok_or(FireRedDecoderError::ShapeOverflow)?;
    graph
        .view_2d(state, hidden, 1, row_stride, offset)
        .map_err(|source| map_err("last_token_view", source))
}

fn tokens_as_i32(tokens: &[u32]) -> Result<Vec<i32>, FireRedDecoderError> {
    tokens
        .iter()
        .map(|&token| {
            i32::try_from(token).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("token id {token} does not fit i32"),
            })
        })
        .collect()
}

fn position_ids_i32_with_offset(
    position_offset: usize,
    token_count: usize,
) -> Result<Vec<i32>, FireRedDecoderError> {
    (0..token_count)
        .map(|index| {
            let position = position_offset
                .checked_add(index)
                .ok_or(FireRedDecoderError::ShapeOverflow)?;
            i32::try_from(position).map_err(|_| FireRedDecoderError::InvalidInput {
                reason: format!("decoder position {position} does not fit i32"),
            })
        })
        .collect()
}

/// Greedy-decode result: the detokenized text and the raw generated ids
/// (excluding the leading `<sos>` prompt token, excluding the trailing
/// `<eos>`).
#[derive(Debug, Clone)]
pub(crate) struct FireRedAedGreedyDecodeOutput {
    pub text: String,
    pub generated_tokens: Vec<u32>,
    /// How the shared driver ended this decode, carried to the executor so a
    /// cut-short transcript is not returned as a complete one.
    pub stop_reason: Seq2SeqGreedyDecodeStopReason,
}

/// Run the full attention-based greedy decode for one utterance against an
/// already-built (and possibly cached/reused across transcriptions)
/// [`FireRedDecoderGraphRuntime`]. Resets the runtime's cross-KV cache and
/// incremental self-KV position for this utterance via
/// [`FireRedDecoderGraphRuntime::populate_cross_attention_cache`] before
/// decoding, then autoregresses from `<sos>` through the shared seq2seq greedy
/// driver (`run_builtin_seq2seq_decode_policy` -> `run_seq2seq_greedy_decode_loop_v0`)
/// under the firered decode-policy descriptor. Routing through the shared driver
/// (rather than a hand-written argmax loop) is what gives firered the degenerate
/// n-gram-repeat guard for free (issue #60): firered declares no phrase bias,
/// no suppression and no extra stop tokens, so the policy config is a plain
/// `<sos>`-prompted greedy decode to `<eos>`.
pub(crate) fn run_firered_aed_decoder_greedy_with_runtime(
    runtime: &mut FireRedDecoderGraphRuntime,
    metadata: FireRedAedExecutionMetadata,
    encoder_rows: &[f32],
    encoder_frame_count: usize,
    decode_text: impl Fn(&[u32]) -> Result<String, String>,
    control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
) -> Result<FireRedAedGreedyDecodeOutput, FireRedDecoderError> {
    runtime.populate_cross_attention_cache(encoder_rows, encoder_frame_count)?;

    let decode_budget = super::decode_budget::firered_aed_decode_budget(
        encoder_frame_count,
        metadata.decoder_pe_len,
    )
    .map_err(|error| FireRedDecoderError::InvalidInput {
        reason: error.to_string(),
    })?;
    runtime
        .decoder_state
        .self_attention
        .validate_exact_shape(
            crate::capacity::topology::StateKind::SelfAttentionKv,
            decode_budget.self_kv_positions,
        )
        .map_err(|error| FireRedDecoderError::InvalidInput {
            reason: error.to_string(),
        })?;
    let max_generated_tokens = decode_budget.max_generated_tokens;
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: vec![metadata.sos_token_id],
        eot_token_id: metadata.eos_token_id,
        vocab_size: metadata.vocab_size,
        max_generated_tokens,
    };
    let decode_text_token_ids = |token_ids: &[u32]| {
        decode_text(token_ids)
            .map_err(|reason| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason })
    };
    let decode = match run_builtin_seq2seq_decode_policy::<Seq2SeqGreedyDecodeError>(
        crate::arch::FIRERED_AED_DECODE_POLICY_ID,
        &config,
        // firered has no special tokens and no phrase bias (supports_phrase_bias
        // is false), so the unit token source with `phrase_bias: None` never
        // needs to encode anything.
        &(),
        None,
        runtime,
        &decode_text_token_ids,
        |error| error,
        |error| error,
        |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
            reason: error.to_string(),
        },
        control,
        decode_work_progress,
    ) {
        Ok(output) => output,
        // Budget exhausted before `<eos>`: keep the generated prefix and
        // detokenize it, matching the pre-unification behavior (return the
        // partial transcript rather than erroring out).
        Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens, ..
        }) => {
            let text = decode_text(&generated_tokens).map_err(|reason| {
                FireRedDecoderError::InvalidInput {
                    reason: format!("tokenizer decode failed: {reason}"),
                }
            })?;
            Seq2SeqGreedyDecodeResult {
                text,
                generated_tokens,
                generated_probabilities: Vec::new(),
                stop_reason: Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
            }
        }
        // Preserve the stable cancel marker so native/server boundaries can
        // rewrite to `BackendError::TranscriptionCanceled`.
        Err(Seq2SeqGreedyDecodeError::Canceled) => {
            return Err(FireRedDecoderError::InvalidInput {
                reason: Seq2SeqGreedyDecodeError::Canceled.to_string(),
            });
        }
        Err(error) => {
            return Err(FireRedDecoderError::InvalidInput {
                reason: error.to_string(),
            });
        }
    };
    Ok(FireRedAedGreedyDecodeOutput {
        text: decode.text,
        generated_tokens: decode.generated_tokens,
        stop_reason: decode.stop_reason,
    })
}
