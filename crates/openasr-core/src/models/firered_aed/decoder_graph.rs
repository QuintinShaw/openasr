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

use crate::GgmlRuntimeSource;
use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena,
};
use crate::models::decode_policy_component_registry::{
    BuiltinSeq2SeqDecodePolicyConfigInput, run_builtin_seq2seq_decode_policy,
};
use crate::models::decode_token_history::context_window_budget;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    Seq2SeqGreedyDecodeStopReason,
};
use crate::nn::decoder::{
    CrossKvHandle, SelfKvHandle, Seq2SeqLayerConfig, Seq2SeqLayerWeights,
    Seq2SeqReusableDecodeGraph, build_causal_mask_f16_bits, build_fixed_kv_attention_mask_bits,
    reusable_decode_graph_supported_for_runner, seq2seq_layer,
};
use crate::nn::ffn::FeedForwardActivation;
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::decoder_weights::{FireRedDecoderWeights, FireRedDecoderWeightsError};
use super::encoder_graph::predicted_encoder_time_frames;
use super::frontend::{FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES, SAMPLE_RATE_HZ};
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
fn firered_decoder_arena_context_bytes(decoder_n_layers: usize) -> usize {
    let tensor_count = FIRERED_DECODER_ARENA_FIXED_TENSORS
        .saturating_add(FIRERED_DECODER_ARENA_TENSORS_PER_LAYER.saturating_mul(decoder_n_layers));
    GgmlCpuGraphConfig::metadata_context_bytes(tensor_count)
}

/// Extra headroom (beyond the architecture's declared safe chunk length)
/// folded into the cross-KV capacity below, purely to reduce how often a
/// chunk-boundary rounding in an upstream caller (VAD snapping, sample-count
/// truncation) triggers [`FireRedDecoderGraphRuntime::grow_cross_capacity`] --
/// never load-bearing for correctness: a chunk past this capacity still
/// grows the cache to fit rather than silently truncating, overrunning the
/// arena, or being rejected.
const FIRERED_DECODER_CROSS_CAPACITY_MARGIN_SECONDS: f32 = 2.0;

/// Predict the mel (pre-subsampling) fbank frame count for `duration_seconds`
/// of 16 kHz audio under this family's `snip_edges=true` framing
/// (`1 + (samples - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES`; see
/// [`super::frontend`]).
fn firered_mel_frames_for_seconds(duration_seconds: f32) -> usize {
    let samples = (duration_seconds.max(0.0) * SAMPLE_RATE_HZ as f32) as usize;
    let Some(usable) = samples.checked_sub(FRAME_LENGTH_SAMPLES) else {
        return 1;
    };
    1 + usable / FRAME_SHIFT_SAMPLES
}

/// Cross-attention KV cache capacity, in post-subsampling encoder frames: the
/// decoder's cross-KV cache is now allocated ONCE per pack at this size
/// (issue #68's `GlobalQuadratic` chunk ceiling +
/// [`FIRERED_DECODER_CROSS_CAPACITY_MARGIN_SECONDS`] margin) instead of
/// exactly matching each utterance's actual encoder frame count. Every VAD /
/// longform chunk for this architecture -- which the shared longform safety
/// policy (`apply_encoder_attention_span_longform_safety_policy`) already caps
/// at `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` -- then reuses the SAME decoder
/// runtime (thread-local cache keyed only by pack path + backend, no more
/// per-frame-count fan-out) instead of rebuilding the whole GGUF weight
/// context and cross-KV arena per differing chunk length. Clamped to the
/// pack's own PE-table ceiling (`encoder_max_frames`) so this never allocates
/// past what any single call could legally present (the executor's
/// `AudioWindowTooLong` preflight already rejects anything larger before it
/// reaches the decoder). Reserve-once sizing follows the same convention as
/// `references/transcribe.cpp@b6a6aca`'s `causal_lm.cpp` KV-cache init
/// (`kv_init`: reserve at a known max, never reallocate per step) -- see
/// `GgmlCpuStepBufferPool`'s doc in `ggml_runtime/cpu_graph.rs` for the
/// sibling citation and `ACKNOWLEDGMENTS.md`.
pub(crate) fn firered_decoder_cross_capacity_frames(
    metadata: &FireRedAedExecutionMetadata,
) -> usize {
    let chunk_seconds = crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS
        + FIRERED_DECODER_CROSS_CAPACITY_MARGIN_SECONDS;
    let mel_frames = firered_mel_frames_for_seconds(chunk_seconds);
    let predicted = predicted_encoder_time_frames(mel_frames)
        .map(|frames| frames.max(1))
        .unwrap_or(1);
    predicted.min(metadata.encoder_max_frames().max(1))
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

/// Owns the decoder's mmap'd weight context plus the persistent runtime state
/// (cross-KV cache, incremental self-KV cache). The cross-KV cache arena
/// starts, at construction, sized to this pack's
/// [`firered_decoder_cross_capacity_frames`] capacity -- NOT to any one
/// utterance's actual encoder frame count -- so the same runtime is reusable
/// across every chunk a VAD/longform run presents (each capped at the shared
/// longform safety ceiling by the caller), not just chunks that happen to
/// share an exact frame count. A chunk past that capacity (a caller that
/// legitimately bypasses longform, e.g. `segment_mode=off`) grows it in place
/// ([`Self::grow_cross_capacity`]) rather than rejecting a request the pack's
/// PE-table ceiling would otherwise support.
/// [`Self::populate_cross_attention_cache`] views only the first
/// `cross_frame_count` (actual, current-utterance) columns of the current
/// capacity on every call.
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
    cross_layers: Vec<FireRedDecoderCrossCacheLayer>,
    self_kv_layers: Vec<FireRedDecoderSelfKvLayer>,
    /// Allocated column count of every `cross_layers[i].{key,value}` tensor
    /// (see [`firered_decoder_cross_capacity_frames`]). Grows (never shrinks)
    /// via [`Self::grow_cross_capacity`] when a chunk exceeds it.
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
}

/// The static-tensor arena plus everything allocated directly in it
/// (cross-KV/self-KV caches, shared zero-bias) -- everything about a
/// [`FireRedDecoderGraphRuntime`] that depends on `cross_capacity_frames` and
/// must be rebuilt wholesale when that capacity grows
/// ([`FireRedDecoderGraphRuntime::grow_cross_capacity`]). Kept separate from
/// `runner`/`_loaded`/`weights` (independent of cross-KV capacity: `weights`
/// borrows the mmap'd GGUF context directly and never needs rebuilding).
struct FireRedDecoderArenaState {
    arena: GgmlStaticTensorArena,
    zero_bias: GgmlStaticTensor,
    cross_layers: Vec<FireRedDecoderCrossCacheLayer>,
    self_kv_layers: Vec<FireRedDecoderSelfKvLayer>,
}

fn build_firered_decoder_arena_state(
    runner: &GgmlCpuGraphRunner,
    metadata: &FireRedAedExecutionMetadata,
    cross_capacity_frames: usize,
) -> Result<FireRedDecoderArenaState, FireRedDecoderError> {
    let arena = runner
        .start_static_tensor_arena(firered_decoder_arena_context_bytes(
            metadata.decoder_n_layers,
        ))
        .map_err(|source| map_err("static_tensor_arena", source))?;
    let zero_bias = arena
        .new_tensor_1d_f32(metadata.d_model, "firered_dec_zero_bias")
        .map_err(|source| map_err("zero_bias_alloc", source))?;
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
                    metadata.decoder_pe_len,
                    metadata.n_heads,
                    "firered_dec_self_k",
                )
                .map_err(|source| map_err("self_k_alloc", source))?,
            value: arena
                .new_tensor_3d_f16(
                    metadata.head_dim,
                    metadata.decoder_pe_len,
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
    // full `decoder_pe_len`-span cache commits all of its pages up front
    // (hundreds of MB of peak RSS) where the untouched malloc'd CPU arena
    // pages would otherwise stay uncommitted until the decode actually wrote
    // them.
    if reusable_decode_graph_supported_for_runner(runner) {
        let self_kv_tensor_bytes = metadata
            .head_dim
            .checked_mul(metadata.decoder_pe_len)
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
        cross_layers,
        self_kv_layers,
    })
}

impl FireRedDecoderGraphRuntime {
    pub(crate) fn new(
        runtime_source: &GgmlRuntimeSource,
        metadata: FireRedAedExecutionMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FireRedDecoderError> {
        let cross_capacity_frames = firered_decoder_cross_capacity_frames(&metadata);
        let config = firered_decoder_graph_config(backend);
        let persistent_graph_context_bytes = config.context_bytes;
        let runner =
            GgmlCpuGraphRunner::new(config).map_err(|source| map_err("runner_init", source))?;
        let loaded = runner
            .load_gguf_weight_context(runtime_source)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let weights = FireRedDecoderWeights::load(&loaded, metadata.decoder_n_layers)?;
        let arena_state =
            build_firered_decoder_arena_state(&runner, &metadata, cross_capacity_frames)?;

        Ok(Self {
            reuse: None,
            runner,
            _loaded: loaded,
            weights,
            metadata,
            persistent_graph_context_bytes,
            arena: arena_state.arena,
            zero_bias: arena_state.zero_bias,
            cross_layers: arena_state.cross_layers,
            self_kv_layers: arena_state.self_kv_layers,
            cross_capacity_frames,
            cross_frame_count: 0,
            reuse_cross_frame_count: 0,
            cached_positions: 0,
        })
    }

    /// Grow-to-fit: rebuilds the cross-KV (and, incidentally, self-KV -- its
    /// content is about to be invalidated by the caller's `cached_positions =
    /// 0` reset anyway) arena at `max(needed_frames, 2x current capacity)`,
    /// clamped to this pack's PE-table ceiling. Only reachable when a caller
    /// presents a chunk past `cross_capacity_frames` -- normal VAD/longform
    /// chunks never do, since the shared longform safety policy already caps
    /// every chunk at (or under) the capacity this runtime was built with;
    /// this exists for callers that legitimately bypass longform with a
    /// larger single window (e.g. `segment_mode=off`), which the executor's
    /// own `AudioWindowTooLong` preflight allows up to the PE-table ceiling.
    /// Cheap relative to the original miss-every-chunk bug this module fixes:
    /// `weights`/`_loaded` (the mmap'd GGUF context) are untouched, only the
    /// arena's own metadata pool + backend buffer are rebuilt.
    fn grow_cross_capacity(&mut self, needed_frames: usize) -> Result<(), FireRedDecoderError> {
        let max_frames = self.metadata.encoder_max_frames().max(1);
        if needed_frames > max_frames {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "encoder frame count {needed_frames} exceeds this pack's positional-encoding \
                     capacity of {max_frames} frames; this should already have been rejected by \
                     the executor's AudioWindowTooLong preflight before reaching the decoder"
                ),
            });
        }
        let new_capacity = needed_frames
            .max(self.cross_capacity_frames.saturating_mul(2))
            .min(max_frames);
        let arena_state =
            build_firered_decoder_arena_state(&self.runner, &self.metadata, new_capacity)?;
        // The persistent reuse graph holds raw pointers into the OLD arena's
        // cross-KV/self-KV tensors, which this call is about to free -- the
        // `reuse_cross_frame_count` mismatch-triggered rebuild alone is not
        // enough (that logic assumes the arena itself is stable), so drop it
        // unconditionally and let the next reusable step rebuild against the
        // new arena instead of dereferencing freed memory.
        self.reuse = None;
        self.reuse_cross_frame_count = 0;
        self.arena = arena_state.arena;
        self.zero_bias = arena_state.zero_bias;
        self.cross_layers = arena_state.cross_layers;
        self.self_kv_layers = arena_state.self_kv_layers;
        self.cross_capacity_frames = new_capacity;
        Ok(())
    }

    /// Precompute cross-attention K/V for every layer from the encoder output
    /// and write them into the persistent cross-KV cache. Must be called once
    /// before the first [`Self::compute_step_logits`]. `frame_count` is this
    /// utterance's ACTUAL encoder frame count -- normally `<=`
    /// [`Self::cross_capacity_frames`], but when it isn't (a caller past the
    /// architecture's chunk-cap, e.g. `segment_mode=off`), this grows the
    /// cross-KV cache to fit first ([`Self::grow_cross_capacity`]) rather
    /// than rejecting a request the pack's own PE-table ceiling would
    /// otherwise support.
    pub(crate) fn populate_cross_attention_cache(
        &mut self,
        encoder_rows: &[f32],
        frame_count: usize,
    ) -> Result<(), FireRedDecoderError> {
        if frame_count == 0 {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "cross_frame_count must be > 0".to_string(),
            });
        }
        if frame_count > self.cross_capacity_frames {
            self.grow_cross_capacity(frame_count)?;
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
        self.compute_step_logits_impl(decoder_tokens, true)
    }

    /// Test-only bypass of the reusable-graph dispatch: always rebuild a
    /// fresh graph for this step, so the reused-vs-rebuilt byte-identity test
    /// can drive both paths on the same backend against the same inputs.
    #[cfg(test)]
    pub(crate) fn compute_step_logits_forcing_fresh_graph(
        &mut self,
        decoder_tokens: &[u32],
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        self.compute_step_logits_impl(decoder_tokens, false)
    }

    /// Whether this step went (or will go) through the reusable decode graph
    /// -- exposed so tests can assert the reuse path is actually live rather
    /// than silently falling back to the rebuild path.
    #[cfg(test)]
    pub(crate) fn has_active_reuse_graph(&self) -> bool {
        self.reuse.is_some()
    }

    fn compute_step_logits_impl(
        &mut self,
        decoder_tokens: &[u32],
        allow_reuse: bool,
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        let total_prefix_tokens = decoder_tokens.len();
        if total_prefix_tokens == 0 {
            return Err(FireRedDecoderError::InvalidInput {
                reason: "decoder token_count must be > 0".to_string(),
            });
        }
        if total_prefix_tokens > self.metadata.decoder_pe_len {
            return Err(FireRedDecoderError::InvalidInput {
                reason: format!(
                    "decoder token_count {total_prefix_tokens} exceeds max context {}",
                    self.metadata.decoder_pe_len
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
            return self.compute_reused_incremental_step_logits(decode_tokens[0], position_offset);
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
                self_kv_max_positions: self.metadata.decoder_pe_len,
                cross_frame_count: self.cross_frame_count,
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
        graph
            .set_output(logits)
            .map_err(|source| map_err("set_output", source))?;
        // Allocate the decode graph through the scheduler's gallocr for
        // liveness-based buffer reuse before uploading inputs (mirrors the
        // cohere/moonshine decoders); the queued uploads below then write into the
        // already-allocated input tensors instead of forcing an independent
        // allocation.
        graph
            .prepare_outputs_for_upload(&[logits])
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

        let output = graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
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
    /// masked (`-inf`) tail of the fixed `decoder_pe_len`-span self-KV view
    /// contributes exactly zero attention weight, and the underlying
    /// arithmetic per valid position is unchanged.
    fn compute_reused_incremental_step_logits(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, FireRedDecoderError> {
        let max_positions = self.metadata.decoder_pe_len;
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
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
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

        let output = graph
            .compute_output_f32(logits, self.metadata.vocab_size)
            .map_err(|error| FireRedDecoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        self.cached_positions = total_tokens;
        Ok(output)
    }

    /// Build the persistent single-token decode graph: the same op sequence as
    /// one `token_count == 1` step of the rebuild path, except the self-KV
    /// write goes through `set_rows` on a runtime row-index input (so the
    /// write slot can move per step without rebuilding) and self-attention
    /// reads the full fixed `decoder_pe_len` span under an externally-uploaded
    /// `-inf` tail mask (so the graph shape is constant across steps). The
    /// current `cross_frame_count` is baked into the cross-attention views;
    /// `compute_reused_incremental_step_logits` rebuilds on mismatch.
    fn build_reusable_decode_graph(&mut self) -> Result<(), FireRedDecoderError> {
        let d_model = self.metadata.d_model;
        let heads = self.metadata.n_heads;
        let head_dim = self.metadata.head_dim;
        let max_positions = self.metadata.decoder_pe_len;
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
                self_kv_max_positions: max_positions,
                cross_frame_count,
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
        graph
            .set_output(logits)
            .map_err(|source| map_err("reuse_set_output", source))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(|source| map_err("reuse_prepare_outputs", source))?;

        self.reuse = Some(Seq2SeqReusableDecodeGraph::new_with_borrowed_kv_arena(
            session,
            max_positions,
            1,
            token_id,
            row_index,
            position,
            attention_mask,
            logits,
        ));
        self.reuse_cross_frame_count = cross_frame_count;
        Ok(())
    }
}

/// firered-aed decodes through the shared seq2seq greedy driver: every step
/// recomputes logits for the full `<sos> ++ generated` prefix (the incremental
/// KV cache inside [`Self::compute_step_logits`] makes this cheap after the
/// prefill). `greedy_token_hint: None` -- firered has no device-side argmax, so
/// the shared loop owns the host argmax.
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
        let logits = self.compute_step_logits(&prefix).map_err(|error| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
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
) -> Result<FireRedAedGreedyDecodeOutput, FireRedDecoderError> {
    runtime.populate_cross_attention_cache(encoder_rows, encoder_frame_count)?;

    let max_generated_tokens = context_window_budget(metadata.decoder_pe_len, 1).unwrap_or(0);
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
