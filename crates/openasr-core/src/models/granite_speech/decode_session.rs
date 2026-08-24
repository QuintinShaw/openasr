//! Granite Speech incremental KV-cache decode session.
//!
//! The one-shot `decoder_graph::prefill_logits*` path recomputes the *entire*
//! token prefix from scratch on every decode step (see `decode_executor`'s
//! historical doc): step N re-runs a full 40-layer forward over
//! `prompt ++ generated[..N]`, so a full transcription is `O(n^2)` in decoded
//! length -- for the 2B Granite dense decoder that is ~430x realtime, which
//! makes full-length WER / peer-gate runs infeasible.
//!
//! This session removes that quadratic by giving Granite the same incremental
//! KV cache every other autoregressive family here already has (qwen's
//! `Qwen3AsrLayerKvCacheState`, firered-llm, cohere, ...): the prompt is
//! prefilled once, each layer's post-RoPE K/V is persisted, and every
//! subsequent step computes Q/K/V for **only the new token**, appends its K/V
//! to the cache, and attends the single new query against the full cached
//! history. Per-step compute drops from `O(prefix)` to `O(1)` projection/MLP
//! plus an `O(prefix)` attention dot-product -- total decode `O(n)` projection/
//! MLP work plus the attention's inherent `O(n^2)` score work. Host K/V is
//! retained token-major at fixed capacity, so an incremental step appends one
//! row and never flattens or recopies the existing history.
//!
//! Bit-exactness (the hard requirement): every op here is byte-for-byte the one
//! the one-shot recompute runs. Prefill and decode share
//! `decoder_graph::granite_pre_attention` / `granite_post_attention` verbatim,
//! so a cached K/V equals the K/V a full recompute would produce at that
//! position (a causal decoder's position-`j` representation is independent of
//! any later token, and this runs CPU-only, where ggml `mul_mat` computes each
//! output element via a fixed-order `vec_dot` regardless of batch width). The
//! cache stores **f32** (never f16) so no rounding is introduced. The only
//! attention difference is prefill's additive causal mask (masked keys underflow
//! to exactly `0.0` in `soft_max_ext`) versus decode attending a history that
//! simply omits those never-contributing keys -- the surviving softmax terms,
//! their max, and their sum are identical, so the last-position logits match to
//! the bit. This is proven in-repo by
//! `granite_incremental_decode_matches_full_recompute_bit_exact`.
//!
//! Weights are held for the session's whole lifetime -- either uploaded once
//! into a persistent f32 `GraniteDecoderWeightArena` (the `new` path, used by
//! the synthetic bit-exact test and any host-`HashMap` provider) or bound
//! zero-copy, keep-quantized, from the mmap'd `.oasr` pack via
//! `GraniteDecoderLoadedWeights` (the `new_keep_quantized` path the runtime
//! executor uses, so a 2B decoder stays ~its packed size resident instead of a
//! ~8 GB f32 dequant + upload). Only the tiny per-step inputs (one embedding,
//! one position, the K/V history views) live in the reset-per-step graph
//! context.
//!
//! ## Metal reuse path (device-resident KV + build-once decode graph)
//!
//! The description above is the CPU / scheduler-on path: it keeps the K/V in
//! host `Vec<f32>` buffers and rebuilds the 40-layer step graph every token,
//! re-uploading the whole history each step. When the immutable planner
//! authorizes `GgmlDecodeReuseMode::ReusableGraph`, that host round-trip is
//! the dominant decode cost, so this module also provides the resident path
//! taken there:
//!
//! - The K/V lives in a device-resident fixed-span arena (`resident_kv`,
//!   `[head_dim, resident_capacity, kv_heads]` f32 per layer). `prefill` seeds
//!   rows `0..n_tokens` in place from the prefill graph (`set_rows`), never
//!   copying K/V to the host; each `decode_step` writes only the one new row.
//! - A single persistent single-token decode graph is built once
//!   (`build_reusable_decode_graph`) and re-run per step -- no 40-layer rebuild.
//!   It reads the full fixed span under an externally uploaded additive `-inf`
//!   tail mask, so its shape is constant across steps.
//!
//! Both mirror the firered-aed / qwen resident-decode machinery while keeping
//! Granite's forked attention numerics. This path is NOT bit-identical to the
//! CPU reference (ggml's Metal reduction order differs from the CPU
//! `vec_dot` -- by design); the bit-exact gate
//! `granite_incremental_decode_matches_full_recompute_bit_exact` covers the CPU
//! path it does not touch, and end-to-end transcript equality validates the
//! Metal path.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlDecodeReuseMode, GgmlKvElementType, GgmlLoadedTensor, GgmlLoadedWeightContext,
    GgmlPersistentGraphSession, GgmlRopeExtParams,
};
use crate::models::device_greedy_token::{DeviceGreedyStepOutputMode, device_top1_token_id};
use crate::models::mapped_token_embedding::MappedTokenEmbeddingDeviceSpec;
use crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeStepLogitsOutput;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote, SystemMemoryOwner,
};
use crate::nn::decoder::{
    LlmKvCacheSpec, LlmResidentKvArena, allocate_zeroed_llm_resident_kv_arena,
    build_causal_mask_f16_bits, build_fixed_kv_attention_mask_bits, last_token_hidden_view,
    reusable_decode_graph_supported,
};

/// Exact per-invocation KV bound plus the stable session-envelope reservation
/// used by Granite's reusable device graph.
///
/// The host path consumes only `logical_positions`. The GPU reuse path owns an
/// arena and graph whose fixed span is `resident_positions`; varying legal
/// chunk lengths therefore never change their physical shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeechKvCacheCapacity {
    logical_positions: usize,
    resident_positions: usize,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum GraniteSpeechKvCacheCapacityError {
    #[error("granite-speech request carries no planned persistent decoder state")]
    DecoderStateNotPlanned,
    #[error("granite-speech decoder state axis is invalid: {source}")]
    InvalidStateAxis {
        #[source]
        source: crate::capacity::topology::TopologyError,
    },
    #[error("granite-speech logical KV span must be positive")]
    ZeroLogicalPositions,
    #[error(
        "granite-speech resident KV span {resident_positions} does not cover logical span {logical_positions}"
    )]
    ResidentDoesNotCoverLogical {
        logical_positions: usize,
        resident_positions: usize,
    },
    #[error(
        "granite-speech runtime measured {measured_positions} KV positions, but the planner proved {planned_positions}"
    )]
    LogicalPositionMismatch {
        planned_positions: usize,
        measured_positions: usize,
    },
    #[error("granite-speech prompt-plus-generation position arithmetic overflowed")]
    LogicalPositionOverflow,
    #[error(
        "granite-speech resident KV span {resident_positions} exceeds decoder hard cap {hard_cap}"
    )]
    HardCapExceeded {
        resident_positions: usize,
        hard_cap: usize,
    },
}

impl GraniteSpeechKvCacheCapacity {
    pub(crate) fn from_decoder_state(
        state: &crate::models::ggml_asr_executor::GgmlAsrDecoderState,
    ) -> Result<Self, GraniteSpeechKvCacheCapacityError> {
        let crate::models::ggml_asr_executor::GgmlAsrDecoderState::Planned(plan) = state else {
            return Err(GraniteSpeechKvCacheCapacityError::DecoderStateNotPlanned);
        };
        let axis = plan
            .position_axis(
                super::capacity::GRANITE_SPEECH_SELF_KV_STATE_ID,
                crate::capacity::topology::StateKind::SelfAttentionKv,
            )
            .map_err(|source| GraniteSpeechKvCacheCapacityError::InvalidStateAxis { source })?;
        Self::new(axis.logical_positions, axis.resident_positions)
    }

    pub(crate) fn new(
        logical_positions: usize,
        resident_positions: usize,
    ) -> Result<Self, GraniteSpeechKvCacheCapacityError> {
        if logical_positions == 0 {
            return Err(GraniteSpeechKvCacheCapacityError::ZeroLogicalPositions);
        }
        if resident_positions < logical_positions {
            return Err(
                GraniteSpeechKvCacheCapacityError::ResidentDoesNotCoverLogical {
                    logical_positions,
                    resident_positions,
                },
            );
        }
        Ok(Self {
            logical_positions,
            resident_positions,
        })
    }

    pub(crate) const fn logical_positions(self) -> usize {
        self.logical_positions
    }

    pub(crate) const fn resident_positions(self) -> usize {
        self.resident_positions
    }

    pub(crate) fn validate_measured_logical_positions(
        self,
        measured_positions: usize,
    ) -> Result<Self, GraniteSpeechKvCacheCapacityError> {
        if measured_positions != self.logical_positions {
            return Err(GraniteSpeechKvCacheCapacityError::LogicalPositionMismatch {
                planned_positions: self.logical_positions,
                measured_positions,
            });
        }
        Ok(self)
    }

    pub(crate) fn validate_hard_cap(
        self,
        hard_cap: usize,
    ) -> Result<Self, GraniteSpeechKvCacheCapacityError> {
        if self.resident_positions > hard_cap {
            return Err(GraniteSpeechKvCacheCapacityError::HardCapExceeded {
                resident_positions: self.resident_positions,
                hard_cap,
            });
        }
        Ok(self)
    }
}

use super::decoder_graph::{
    GraniteDecoderLoadedWeights, GraniteDecoderWeightArena, GraniteDecoderWeights,
    GraniteSpeechDecoderConfig, GraniteSpeechDecoderError, embed_token_row, granite_post_attention,
    granite_pre_attention, linear, rms_norm, weight_in_major,
};

/// f32 resident KV element type for the Metal reuse path. f32 (not f16) keeps
/// the cached K/V rounding-free, so the only numerical difference from the CPU
/// growing-KV reference is ggml's backend reduction order (by design; the
/// bit-exact gate covers the CPU path). Both host + resident are f32.
const GRANITE_RESIDENT_KV_SPEC: LlmKvCacheSpec = LlmKvCacheSpec {
    host: GgmlKvElementType::F32,
    resident: GgmlKvElementType::F32,
};

const GRANITE_DECODE_GRAPH_SIZE: usize = 32_768;

fn granite_decode_graph_context_bytes() -> usize {
    GgmlCpuGraphConfig::metadata_context_bytes(GRANITE_DECODE_GRAPH_SIZE)
}

pub(crate) fn decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    GgmlCpuGraphConfig {
        context_bytes: granite_decode_graph_context_bytes(),
        graph_size: GRANITE_DECODE_GRAPH_SIZE,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        // Scheduler off on the single-backend GPU path so the in-place
        // resident-KV reuse graph is legal. CPU keeps the scheduler and the
        // growing-KV host decode path.
        use_scheduler: !backend.is_gpu_class(),
    }
}

const fn resident_flash_attention_enabled(backend: GgmlCpuGraphBackend) -> bool {
    // The resident path has a measured, transcript-equivalent win on Metal.
    // Keep the generic CUDA/HIP/Vulkan lane on its established naive path
    // until each backend's wide-prefill flash kernel has its own correctness
    // and performance evidence.
    matches!(backend, GgmlCpuGraphBackend::Metal)
}

/// Additive self-attention mask for the fixed-span reuse graph: `0.0` for every
/// key column `<= position` (the prompt + generated-so-far + the just-written
/// new token) and `f32::MIN` for the never-yet-written / future tail. Added
/// after the `attention_multiplier` scale in `soft_max_ext`, exactly as the
/// growing path's causal mask, so masked columns underflow to `0.0` and the
/// surviving softmax terms match the growing-KV attention over `position + 1`
/// keys.
fn fixed_span_tail_mask(max_positions: usize, position: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; max_positions];
    for slot in mask.iter_mut().skip(position + 1) {
        *slot = f32::MIN;
    }
    mask
}

fn map_ggml(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GraniteSpeechDecoderError + Copy {
    move |source| GraniteSpeechDecoderError::Ggml { stage, source }
}

const GRANITE_HOST_KV_RESOURCE_ID: &str = "granite-speech.decoder.host-self-kv";

/// One allocation transaction for the complete CPU-path KV owner.
///
/// Each layer is a fixed-length token-major `[position, kv_head, head_dim]`
/// array. The prefill graph emits that layout directly and every incremental
/// graph writes exactly one row, so no history-sized staging allocation is
/// ever needed after this owner commits.
#[derive(Debug)]
struct GraniteHostKvState {
    k_history: Vec<Vec<f32>>,
    v_history: Vec<Vec<f32>>,
}

impl GraniteHostKvState {
    fn try_allocate(
        num_layers: usize,
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<SystemMemoryOwner<Self>, GraniteSpeechDecoderError> {
        let quoted_bytes =
            granite_host_kv_quoted_bytes(num_layers, max_positions, kv_heads, head_dim)?;
        let quote = SystemMemoryAllocationQuote::new(
            GRANITE_HOST_KV_RESOURCE_ID,
            quoted_bytes,
            quoted_bytes,
        )
        .map_err(|error| GraniteSpeechDecoderError::Shape {
            reason: error.to_string(),
        })?;
        SystemMemoryOwner::try_allocate(quote, || {
            let state =
                Self::try_allocate_unadmitted(num_layers, max_positions, kv_heads, head_dim)?;
            let actual_bytes = state.actual_capacity_bytes()?;
            Ok(SystemMemoryAllocationOutcome::new(
                state,
                actual_bytes,
                actual_bytes,
            ))
        })
        .map_err(|error| GraniteSpeechDecoderError::Shape {
            reason: error.to_string(),
        })
    }

    fn try_allocate_unadmitted(
        num_layers: usize,
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, String> {
        let row_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "granite host KV row width overflowed".to_string())?;
        let layer_len = max_positions
            .checked_mul(row_width)
            .ok_or_else(|| "granite host KV layer length overflowed".to_string())?;
        let allocate_layers = || -> Result<Vec<Vec<f32>>, String> {
            let mut layers = Vec::new();
            layers.try_reserve_exact(num_layers).map_err(|error| {
                format!("granite host KV layer table allocation failed: {error}")
            })?;
            for _ in 0..num_layers {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(layer_len)
                    .map_err(|error| format!("granite host KV layer allocation failed: {error}"))?;
                values.resize(layer_len, 0.0);
                layers.push(values);
            }
            Ok(layers)
        };
        Ok(Self {
            k_history: allocate_layers()?,
            v_history: allocate_layers()?,
        })
    }

    fn actual_capacity_bytes(&self) -> Result<u64, String> {
        fn nested_capacity_bytes(values: &Vec<Vec<f32>>) -> Result<u64, String> {
            let table = values
                .capacity()
                .checked_mul(std::mem::size_of::<Vec<f32>>())
                .ok_or_else(|| "granite host KV table capacity overflowed".to_string())?;
            let payload = values.iter().try_fold(0usize, |total, values| {
                let bytes = values
                    .capacity()
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| "granite host KV payload capacity overflowed".to_string())?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| "granite host KV payload sum overflowed".to_string())
            })?;
            u64::try_from(
                table
                    .checked_add(payload)
                    .ok_or_else(|| "granite host KV capacity sum overflowed".to_string())?,
            )
            .map_err(|_| "granite host KV capacity exceeds u64".to_string())
        }
        nested_capacity_bytes(&self.k_history)?
            .checked_add(nested_capacity_bytes(&self.v_history)?)
            .ok_or_else(|| "granite host K/V capacity sum overflowed".to_string())
    }
}

fn granite_host_kv_quoted_bytes(
    num_layers: usize,
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<u64, GraniteSpeechDecoderError> {
    let bytes = num_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(max_positions))
        .and_then(|value| value.checked_mul(kv_heads))
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|payload| {
            num_layers
                .checked_mul(2)
                .and_then(|tables| tables.checked_mul(std::mem::size_of::<Vec<f32>>()))
                .and_then(|tables| payload.checked_add(tables))
        })
        .ok_or_else(|| GraniteSpeechDecoderError::Shape {
            reason: "granite host KV quoted bytes overflowed".to_string(),
        })?;
    u64::try_from(bytes).map_err(|_| GraniteSpeechDecoderError::Shape {
        reason: "granite host KV quoted bytes exceed u64".to_string(),
    })
}

/// A prefilled, persistent Granite decoder ready to emit one incremental
/// single-token step at a time. Construct with [`new`](Self::new), seed with
/// [`prefill`](Self::prefill), then call [`decode_step`](Self::decode_step) once
/// per generated token.
pub(crate) struct GraniteSpeechDecodeSession {
    /// Build-once/re-run single-token decode graph for the Metal reuse path
    /// (`None` until the first reused step builds it, and on CPU / scheduler-on
    /// runners where the growing-KV host path stays authoritative). Declared
    /// FIRST so it drops first: its persistent graph holds raw pointers into
    /// `runner`, `weights`/`_loaded`, and the `resident_kv` arena, all declared
    /// below and therefore dropped after it.
    reuse: Option<GraniteReusableDecodeGraph>,
    greedy_step_output_mode: DeviceGreedyStepOutputMode,
    reuse_mode: GgmlDecodeReuseMode,
    config: GraniteSpeechDecoderConfig,
    runner: GgmlCpuGraphRunner,
    weights: GraniteDecoderWeights,
    /// Canonical token-major embedding matrix bound inside `_loaded`. Direct
    /// GPU decode consumes token ids through `get_rows`; CPU and synthetic
    /// f32-arena sessions keep using their existing host embedding provider.
    device_token_embedding: Option<GraniteDeviceTokenEmbedding>,
    /// Kept alive so the keep-quantized `weights`' zero-copy handles (raw
    /// pointers into this context's mmap-backed backend buffer) stay valid for
    /// the session's lifetime. `None` on the f32-arena path (the arena owns its
    /// own storage inside `weights`). Declared after `weights` so `weights`
    /// drops first.
    ///
    /// The session holds NO borrow of the weight provider: `new`/`new_keep_quantized`
    /// consult it transiently (arena upload / not at all), and `decode_step`
    /// takes the provider as a call argument. That keeps the whole session
    /// owned (no lifetime), so a keep-quantized instance -- runner plus the
    /// mmap'd loaded context plus its zero-copy bound weights -- can live in
    /// the cross-request resident cache (`executor::GraniteSpeechPreparedRuntime`).
    _loaded: Option<GgmlLoadedWeightContext>,
    /// Transactionally admitted token-major CPU-path K/V. `None` for the
    /// resident-only GPU path. The owner drops its Vec payloads before its
    /// SystemMemory lease, including when a candidate prefill fails.
    host_kv: Option<SystemMemoryOwner<GraniteHostKvState>>,
    seq_len: usize,
    prefilled: bool,
    /// Device-resident, per-layer fixed-span `[head_dim, resident_capacity,
    /// kv_heads]` f32 K/V arena for the Metal reuse path. Seeded once per
    /// request by `prefill` (rows `0..n_tokens` via `set_rows`) and extended one
    /// row per `decode_step`, so the K/V history never round-trips to the host.
    /// `None` on CPU / scheduler-on runners (which keep `k_history`/`v_history`).
    /// Declared after `reuse` so the reuse graph (which points into this arena)
    /// drops first.
    resident_kv: Option<LlmResidentKvArena>,
    /// Allocated column count (`max_positions`) of every `resident_kv` tensor.
    /// The fixed span the reuse graph attends over and bakes into its topology;
    /// exactly equal to the planner's stable session-envelope reserve. `0`
    /// before the first resident allocation.
    resident_capacity: usize,
    /// Exact prompt-plus-generation bound for the active invocation. This is
    /// request-scoped even when the resident arena/graph survives in the
    /// cross-request cache.
    logical_capacity: usize,
}

/// Build-once/re-run persistent single-token Granite decode graph plus its
/// per-step runtime inputs. The op sequence is exactly one `n_tokens == 1` step
/// of the growing-KV path, except the new token's K/V is written into the
/// resident arena via `set_rows` on a runtime `row_index` input and
/// self-attention reads the full fixed `resident_capacity` span under an
/// externally uploaded additive `-inf` tail mask, so the graph shape stays
/// constant across steps. The owning [`GraniteSpeechDecodeSession`] declares
/// this object before its resident arena and runner, so `session` drops before
/// every allocation its graph tensors point into.
struct GraniteReusableDecodeGraph {
    session: GgmlPersistentGraphSession,
    /// The fixed self-KV span this persistent graph was built for; a mismatch
    /// against the session's current `resident_capacity` forces a rebuild.
    max_positions: usize,
    use_flash_attention: bool,
    input_kind: GraniteReusableDecodeInputKind,
    output_mode: DeviceGreedyStepOutputMode,
    embed: Option<GgmlCpuTensor<'static>>,
    token_id: Option<GgmlCpuTensor<'static>>,
    row_index: GgmlCpuTensor<'static>,
    position: GgmlCpuTensor<'static>,
    mask: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
    top1: Option<GgmlCpuTensor<'static>>,
}

#[derive(Clone, Copy)]
struct GraniteDeviceTokenEmbedding {
    tensor: GgmlLoadedTensor,
    vocab_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraniteReusableDecodeInputKind {
    Embedding,
    TokenId,
}

enum GraniteReusableDecodeInput<'a> {
    Embedding(&'a [f32]),
    TokenId(i32),
}

impl GraniteReusableDecodeInput<'_> {
    fn kind(&self) -> GraniteReusableDecodeInputKind {
        match self {
            Self::Embedding(_) => GraniteReusableDecodeInputKind::Embedding,
            Self::TokenId(_) => GraniteReusableDecodeInputKind::TokenId,
        }
    }
}

fn map_device_top1_token(
    token_id: i32,
    vocab_size: usize,
) -> Result<u32, GraniteSpeechDecoderError> {
    device_top1_token_id(token_id, vocab_size).map_err(map_ggml("device_top1_map_token"))
}

impl GraniteSpeechDecodeSession {
    pub(crate) fn quoted_retained_system_memory_bytes(
        config: &GraniteSpeechDecoderConfig,
    ) -> Result<u64, String> {
        GraniteDecoderLoadedWeights::quoted_retained_system_memory_bytes(config.num_layers)
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        self.weights.retained_system_memory_bytes()
    }

    pub(crate) fn quoted_construction_transient_system_memory_bytes(
        _config: &GraniteSpeechDecoderConfig,
        _output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<u64, String> {
        Ok(0)
    }

    pub(crate) fn construction_transient_system_memory_bytes(&self) -> Result<u64, String> {
        Self::quoted_construction_transient_system_memory_bytes(
            &self.config,
            self.greedy_step_output_mode,
        )
    }

    /// Build the runner and upload every decoder weight once. No prefill yet.
    /// `provider` is consulted transiently here (to fill the f32 arena) and is
    /// not retained; `decode_step` takes it again per call for the token embed.
    pub(crate) fn new(
        config: GraniteSpeechDecoderConfig,
        provider: &HashMap<String, Vec<f32>>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let graph_config = decoder_graph_config(backend);
        let runner =
            GgmlCpuGraphRunner::new(graph_config).map_err(map_ggml("session_runner_init"))?;
        let weights = GraniteDecoderWeightArena::load(&runner, &config, provider)?;
        Self::assemble(
            config,
            runner,
            GraniteDecoderWeights::Arena(Box::new(weights)),
            None,
            None,
            DeviceGreedyStepOutputMode::FullLogits,
            GgmlDecodeReuseMode::FreshGraph,
        )
    }

    /// Keep-quantized session: bind every decoder weight zero-copy from the
    /// mmap'd `.oasr` pack (native q8_0/q4_k/f16/f32) instead of dequantizing the
    /// whole 2-B decoder to an f32 host copy + arena upload. The projection/norm/
    /// lm_head weights come from the pack; the token-embedding rows are supplied
    /// by the `provider` passed to `decode_step`. The loaded weight context is
    /// loaded context and this session's runner use the same thread-cached
    /// backend/device, and both are held for the session's whole lifetime.
    pub(crate) fn new_keep_quantized_from_preflight(
        config: GraniteSpeechDecoderConfig,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'_>>,
        backend: GgmlCpuGraphBackend,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
        reuse_mode: GgmlDecodeReuseMode,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let graph_config = decoder_graph_config(backend);
        let runner =
            GgmlCpuGraphRunner::new(graph_config).map_err(map_ggml("session_runner_init"))?;
        let loaded = runner
            .load_gguf_weight_context_from_preflight(preflight)
            .map_err(map_ggml("session_load_gguf_weight_context"))?;
        let weights = GraniteDecoderLoadedWeights::load(&loaded, &config)?;
        let device_token_embedding = token_embedding
            .map(|spec| {
                if spec.d_model != config.hidden_size || spec.vocab_size != config.vocab_size {
                    return Err(GraniteSpeechDecoderError::Shape {
                        reason: format!(
                            "granite device token embedding is {}x{}, expected {}x{}",
                            spec.d_model, spec.vocab_size, config.hidden_size, config.vocab_size
                        ),
                    });
                }
                Ok(GraniteDeviceTokenEmbedding {
                    tensor: loaded.tensor(spec.tensor_name).ok_or_else(|| {
                        GraniteSpeechDecoderError::MissingWeight {
                            name: spec.tensor_name.to_string(),
                        }
                    })?,
                    vocab_size: spec.vocab_size,
                })
            })
            .transpose()?;
        if reusable_decode_graph_supported(reuse_mode) && device_token_embedding.is_none() {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "reusable Granite decode requires a canonical token-major embedding tensor"
                    .to_string(),
            });
        }
        Self::assemble(
            config,
            runner,
            GraniteDecoderWeights::Loaded(weights),
            device_token_embedding,
            Some(loaded),
            greedy_step_output_mode,
            reuse_mode,
        )
    }

    fn assemble(
        config: GraniteSpeechDecoderConfig,
        runner: GgmlCpuGraphRunner,
        weights: GraniteDecoderWeights,
        device_token_embedding: Option<GraniteDeviceTokenEmbedding>,
        loaded: Option<GgmlLoadedWeightContext>,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
        reuse_mode: GgmlDecodeReuseMode,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        Ok(Self {
            reuse: None,
            greedy_step_output_mode,
            reuse_mode,
            config,
            runner,
            weights,
            device_token_embedding,
            _loaded: loaded,
            host_kv: None,
            seq_len: 0,
            prefilled: false,
            resident_kv: None,
            resident_capacity: 0,
            logical_capacity: 0,
        })
    }

    /// Whether the immutable planner authorized in-place resident-KV reuse.
    /// Unproven lanes, including GPU FullDevice, stay on the growing-KV host path.
    fn reuse_supported(&self) -> bool {
        reusable_decode_graph_supported(self.reuse_mode)
    }

    /// Reset this decode's request-visible state before the prepared runtime
    /// re-enters the cross-request cache. On the CPU / scheduler path the grown
    /// host K/V vectors are released. On the GPU reuse path the fixed-capacity
    /// resident K/V arena and persistent graph deliberately remain allocated;
    /// the next prefill overwrites every newly visible prompt row, each decode
    /// step overwrites its own row, and the fixed-span mask keeps all remaining
    /// stale rows invisible. In both cases clearing `seq_len` and `prefilled`
    /// makes a new prefill mandatory before another step can execute.
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        self.host_kv = None;
        self.seq_len = 0;
        self.prefilled = false;
        self.logical_capacity = 0;
    }

    pub(crate) fn is_prefilled(&self) -> bool {
        self.prefilled
    }

    pub(crate) fn config(&self) -> &GraniteSpeechDecoderConfig {
        &self.config
    }

    /// Number of tokens (prompt + generated) whose K/V is currently cached.
    pub(crate) fn cached_positions(&self) -> usize {
        self.seq_len
    }

    /// Prefill the whole prompt once: run a single causal forward over
    /// `embeddings` (`[n_tokens, hidden_size]`, row-major, pre-`embedding_multiplier`),
    /// persist every layer's post-RoPE K/V, and return the logits row for the
    /// token immediately following the prompt (i.e. the first generated token's
    /// distribution). Same op sequence as `decoder_graph::prefill_logits_from_embeddings`.
    pub(crate) fn prefill(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        capacity: GraniteSpeechKvCacheCapacity,
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        if self.prefilled {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite decode session already prefilled".to_string(),
            });
        }
        if n_tokens == 0 {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "prefill n_tokens must be non-zero".to_string(),
            });
        }
        if embeddings.len() != n_tokens * self.config.hidden_size {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "prefill embeddings has {} values, expected {n_tokens}x{}",
                    embeddings.len(),
                    self.config.hidden_size
                ),
            });
        }
        if n_tokens > capacity.logical_positions() {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite prefill requires {n_tokens} KV positions, exceeding logical span {}",
                    capacity.logical_positions()
                ),
            });
        }

        if self.reuse_supported() {
            // Metal reuse path: seed the resident KV arena directly from the
            // prefill graph (rows `0..n_tokens` via `set_rows`), no host copy.
            self.ensure_resident_arena(capacity.resident_positions())?;
            let output = run_prefill_graph_seeding_resident(
                &mut self.runner,
                &self.weights,
                &self.config,
                self.resident_kv
                    .as_ref()
                    .expect("resident arena allocated above"),
                GraniteResidentPrefillInput::Embeddings(embeddings),
                n_tokens,
                DeviceGreedyStepOutputMode::FullLogits,
            )?;
            debug_assert!(output.greedy_token_hint.is_none());
            self.seq_len = n_tokens;
            self.prefilled = true;
            self.logical_capacity = capacity.logical_positions();
            return Ok(output.logits);
        }

        // Allocate and commit the entire CPU-path owner before graph
        // admission. The graph then writes token-major taps directly into
        // this storage; no provisional SystemMemory gate is held while ggml
        // performs its own native-memory transaction.
        let mut host_kv = GraniteHostKvState::try_allocate(
            self.config.num_layers,
            capacity.logical_positions(),
            self.config.num_kv_heads,
            self.config.head_dim,
        )?;
        let last_logits = run_prefill_graph(
            &mut self.runner,
            &self.weights,
            &self.config,
            embeddings,
            n_tokens,
            &mut host_kv,
        )?;
        self.host_kv = Some(host_kv);
        self.seq_len = n_tokens;
        self.prefilled = true;
        self.logical_capacity = capacity.logical_positions();
        Ok(last_logits)
    }

    /// GPU-only first-prompt path: gather canonical token rows from the
    /// pack-bound embedding matrix and replace audio placeholders inside the
    /// same resident prefill graph. CPU/scheduler callers receive `None` and
    /// retain the existing host-gather fallback.
    pub(crate) fn prefill_token_ids_with_audio(
        &mut self,
        token_ids: &[u32],
        audio_rows: &[f32],
        audio_positions: &[usize],
        capacity: GraniteSpeechKvCacheCapacity,
    ) -> Result<Option<Seq2SeqGreedyDecodeStepLogitsOutput>, GraniteSpeechDecoderError> {
        if !self.reuse_supported() {
            return Ok(None);
        }
        if self.prefilled {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite decode session already prefilled".to_string(),
            });
        }
        let n_tokens = token_ids.len();
        if n_tokens == 0 || n_tokens > capacity.logical_positions() {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite device prompt token span is invalid".to_string(),
            });
        }
        let embedding =
            self.device_token_embedding
                .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                    reason: "granite device prompt requires a bound token embedding".to_string(),
                })?;
        if !audio_rows.len().is_multiple_of(self.config.hidden_size) {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite device prompt audio row width mismatch".to_string(),
            });
        }
        let audio_count = audio_rows.len() / self.config.hidden_size;
        if audio_positions.len() != audio_count
            || audio_positions.iter().any(|&position| position >= n_tokens)
            || audio_positions
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite device prompt audio positions are invalid".to_string(),
            });
        }
        if token_ids.iter().enumerate().any(|(position, &token_id)| {
            audio_positions.binary_search(&position).is_err()
                && token_id as usize >= embedding.vocab_size
        }) {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite device prompt token id exceeds vocabulary".to_string(),
            });
        }
        self.ensure_resident_arena(capacity.resident_positions())?;
        let output = run_prefill_graph_seeding_resident(
            &mut self.runner,
            &self.weights,
            &self.config,
            self.resident_kv
                .as_ref()
                .expect("resident arena allocated above"),
            GraniteResidentPrefillInput::TokenIds {
                embedding,
                token_ids,
                audio_rows,
                audio_positions,
            },
            n_tokens,
            self.greedy_step_output_mode,
        )?;
        self.seq_len = n_tokens;
        self.prefilled = true;
        self.logical_capacity = capacity.logical_positions();
        Ok(Some(output))
    }

    /// Run one incremental decode step for `new_token_id` (the position it
    /// occupies is the current cache length), append its K/V, and return the
    /// logits row for the NEXT token.
    pub(crate) fn decode_step(
        &mut self,
        new_token_id: u32,
        provider: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        self.ensure_can_decode_step()?;
        let embed_row = embed_token_row(&self.config, provider, new_token_id)?.to_vec();
        self.decode_step_from_embedding_unchecked(&embed_row)
    }

    /// Advance one token from a caller-supplied embedding row. Production
    /// uses this seam with the mmap-backed shared token table, while the
    /// host-`HashMap` test path above remains an exact numerical oracle.
    pub(crate) fn decode_step_from_embedding(
        &mut self,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        self.ensure_can_decode_step()?;
        if embed_row.len() != self.config.hidden_size {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite decode embedding row has {} values, expected {}",
                    embed_row.len(),
                    self.config.hidden_size
                ),
            });
        }
        self.decode_step_from_embedding_unchecked(embed_row)
    }

    /// Advance a direct-GPU session from a token id without materializing its
    /// embedding row on the host. `None` means this session intentionally uses
    /// the CPU/scheduler path and the caller should retain its existing host
    /// gather fallback.
    pub(crate) fn decode_step_from_token_id(
        &mut self,
        token_id: u32,
    ) -> Result<Option<Seq2SeqGreedyDecodeStepLogitsOutput>, GraniteSpeechDecoderError> {
        if !self.reuse_supported() || self.device_token_embedding.is_none() {
            return Ok(None);
        }
        self.ensure_can_decode_step()?;
        let embedding = self
            .device_token_embedding
            .expect("checked Granite device token embedding above");
        let token_index =
            usize::try_from(token_id).map_err(|_| GraniteSpeechDecoderError::Shape {
                reason: format!("granite token id {token_id} does not fit usize"),
            })?;
        if token_index >= embedding.vocab_size {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite token id {token_id} exceeds vocabulary {}",
                    embedding.vocab_size
                ),
            });
        }
        let token_id = i32::try_from(token_id).map_err(|_| GraniteSpeechDecoderError::Shape {
            reason: format!("granite token id {token_id} does not fit i32"),
        })?;
        self.decode_step_reused(
            GraniteReusableDecodeInput::TokenId(token_id),
            self.greedy_step_output_mode,
        )
        .map(Some)
    }

    fn ensure_can_decode_step(&self) -> Result<(), GraniteSpeechDecoderError> {
        if !self.prefilled {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite decode session must be prefilled before decode_step".to_string(),
            });
        }
        if self.seq_len >= self.logical_capacity {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite decode position {} exceeds logical KV span {}",
                    self.seq_len, self.logical_capacity
                ),
            });
        }
        Ok(())
    }

    fn decode_step_from_embedding_unchecked(
        &mut self,
        embed_row: &[f32],
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        if self.reuse_supported() {
            // Metal reuse path: one persistent single-token step against the
            // resident KV arena (write the new row via `set_rows`, attend the
            // fixed span). No graph rebuild, no host K/V round-trip.
            let output = self.decode_step_reused(
                GraniteReusableDecodeInput::Embedding(embed_row),
                DeviceGreedyStepOutputMode::FullLogits,
            )?;
            debug_assert!(output.greedy_token_hint.is_none());
            return Ok(output.logits);
        }

        let seq_len = self.seq_len;
        let host_kv = self
            .host_kv
            .as_mut()
            .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                reason: "granite CPU decode path has no admitted host KV owner".to_string(),
            })?;
        let logits = run_decode_step_graph(
            &mut self.runner,
            &self.weights,
            &self.config,
            embed_row,
            seq_len,
            host_kv,
        )?;
        self.seq_len += 1;
        Ok(logits)
    }

    /// Ensure the resident KV arena has exactly the planner's stable envelope
    /// span. A different envelope invalidates the graph and arena together;
    /// changing only the current invocation's logical span never reaches this
    /// branch and therefore never rebuilds either object.
    fn ensure_resident_arena(&mut self, required: usize) -> Result<(), GraniteSpeechDecoderError> {
        if self.resident_kv.is_some() && self.resident_capacity == required {
            return Ok(());
        }
        // Drop the reuse graph before freeing the old arena it points into.
        self.reuse = None;
        self.resident_kv = None;
        let num_layers = self.config.num_layers;
        let arena = allocate_zeroed_llm_resident_kv_arena(
            &self.runner,
            num_layers,
            self.config.head_dim,
            required,
            self.config.num_kv_heads,
            1,
            "granite_resident_kv",
            GRANITE_RESIDENT_KV_SPEC,
        )
        .map_err(map_ggml("resident_kv_arena_alloc"))?;
        self.resident_kv = Some(arena);
        self.resident_capacity = required;
        Ok(())
    }

    /// One incremental single-token step on the Metal reuse path: build the
    /// persistent graph on first use (or after a capacity change / poison),
    /// refresh the per-step token / row-index / position / tail-mask inputs, and
    /// re-run -- no graph construction, no reallocation, and the new K/V is
    /// written in place into the resident arena (`set_rows`) instead of round-
    /// tripping to the host. The new token occupies row `self.seq_len`.
    fn decode_step_reused(
        &mut self,
        input: GraniteReusableDecodeInput<'_>,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, GraniteSpeechDecoderError> {
        let position = self.seq_len;
        let max_positions = self.resident_capacity;
        if position >= max_positions {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite decode position {position} exceeds resident KV span {max_positions}"
                ),
            });
        }
        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.session.is_poisoned()
                    || reuse.max_positions != max_positions
                    || reuse.input_kind != input.kind()
                    || reuse.output_mode != output_mode
            })
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph(input.kind(), output_mode)?;
        }

        let vocab_size = self.config.vocab_size;
        let position_i32 =
            i32::try_from(position).map_err(|_| GraniteSpeechDecoderError::Shape {
                reason: format!("granite decode position {position} does not fit i32"),
            })?;
        let use_flash_attention = self
            .reuse
            .as_ref()
            .expect("granite reusable decode graph built above")
            .use_flash_attention;
        let mask_values =
            (!use_flash_attention).then(|| fixed_span_tail_mask(max_positions, position));
        let mask_bits = if use_flash_attention {
            Some(
                build_fixed_kv_attention_mask_bits(max_positions, position + 1)
                    .map_err(map_ggml("reuse_build_flash_mask"))?,
            )
        } else {
            None
        };

        let reuse = self
            .reuse
            .as_mut()
            .expect("granite reusable decode graph built above");
        let embed = reuse.embed;
        let token_id = reuse.token_id;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let mask = reuse.mask;
        let logits = reuse.logits;
        let top1 = reuse.top1;
        let graph = reuse.session.builder();

        match input {
            GraniteReusableDecodeInput::Embedding(values) => graph
                .set_f32_slice(
                    embed.expect("embedding-mode Granite reuse input"),
                    values,
                    "granite_reuse_embed",
                )
                .map_err(map_ggml("reuse_upload_embed"))?,
            GraniteReusableDecodeInput::TokenId(value) => graph
                .set_i32_slice(
                    token_id.expect("token-mode Granite reuse input"),
                    &[value],
                    "granite_reuse_token_id",
                )
                .map_err(map_ggml("reuse_upload_token_id"))?,
        }
        graph
            .set_i32_slice(row_index, &[position_i32], "granite_reuse_row")
            .map_err(map_ggml("reuse_upload_row"))?;
        graph
            .set_i32_slice(position_tensor, &[position_i32], "granite_reuse_position")
            .map_err(map_ggml("reuse_upload_position"))?;
        if let Some(mask_bits) = mask_bits {
            graph
                .set_f16_bits_slice(mask, &mask_bits, "granite_reuse_mask")
                .map_err(map_ggml("reuse_upload_flash_mask"))?;
        } else {
            graph
                .set_f32_slice(
                    mask,
                    mask_values
                        .as_deref()
                        .expect("naive Granite reuse mask values"),
                    "granite_reuse_mask",
                )
                .map_err(map_ggml("reuse_upload_mask"))?;
        }
        let output = match top1 {
            Some(top1) => {
                let token_id = graph
                    .compute_output_i32(top1, 1)
                    .map_err(map_ggml("reuse_compute_top1"))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                        reason: "granite reused device top-1 returned no token id".to_string(),
                    })?;
                Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(map_device_top1_token(token_id, vocab_size)?),
                }
            }
            None => Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: graph
                    .compute_output_f32(logits, vocab_size)
                    .map_err(map_ggml("reuse_compute"))?,
                greedy_token_hint: None,
            },
        };
        self.seq_len = position + 1;
        Ok(output)
    }

    /// Build the persistent single-token decode graph against the current
    /// resident arena and `resident_capacity` span. Mirrors firered-aed's
    /// `build_reusable_decode_graph`, but keeps Granite's forked
    /// `granite_pre_attention` / `granite_post_attention` numerics (all four
    /// Granite scalars: `attention_multiplier`, `residual_multiplier`,
    /// `embedding_multiplier`, `logits_scaling`). The new token's K/V is written
    /// into the arena via `set_rows(arena, k/v, row_index)`; self-attention then
    /// reads the whole fixed span, with the per-step additive `-inf` tail mask
    /// zeroing every not-yet-written (or masked-future) column so the result is
    /// numerically the growing-KV attention over exactly `position + 1` keys.
    fn build_reusable_decode_graph(
        &mut self,
        input_kind: GraniteReusableDecodeInputKind,
        output_mode: DeviceGreedyStepOutputMode,
    ) -> Result<(), GraniteSpeechDecoderError> {
        let config = self.config;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let max_positions = self.resident_capacity;
        let use_flash_attention = resident_flash_attention_enabled(self.runner.backend_kind());
        let resident_kv = self
            .resident_kv
            .as_ref()
            .expect("resident arena present before building reuse graph");
        // Fixed RoPE params: `qwen_neox` uses `ext_factor = 0`, so `n_ctx_orig`
        // never enters the rotation -- baking `max_positions` here is bit-for-bit
        // identical to the growing path's per-step `seq_len + 1`.
        let rope = GgmlRopeExtParams::qwen_neox(head_dim, max_positions, config.rope_theta)
            .map_err(map_ggml("reuse_rope_params"))?;

        // Snapshot the resident arena's graph tensors as `'static` for the
        // persistent session (the arena outlives the reuse graph by field order).
        let kv_tensors: Vec<(GgmlCpuTensor<'static>, GgmlCpuTensor<'static>)> =
            resident_kv.graph_tensors();

        // The persistent reuse graph has the same node capacity as the session
        // runner; derive its metadata backing from that shared contract.
        let mut session = self
            .runner
            .start_persistent_graph_session(granite_decode_graph_context_bytes())
            .map_err(map_ggml("reuse_session_start"))?;
        let graph = session.builder();

        let (embed, token_id, hidden_input) = match input_kind {
            GraniteReusableDecodeInputKind::Embedding => {
                let embed = graph
                    .new_tensor_2d_f32(hidden_size, 1, "granite_reuse_embed")
                    .map_err(map_ggml("reuse_embed_alloc"))?;
                graph
                    .set_input(embed)
                    .map_err(map_ggml("reuse_embed_input"))?;
                (Some(embed), None, embed)
            }
            GraniteReusableDecodeInputKind::TokenId => {
                let token_id = graph
                    .new_tensor_1d_i32(1, "granite_reuse_token_id")
                    .map_err(map_ggml("reuse_token_id_alloc"))?;
                graph
                    .set_input(token_id)
                    .map_err(map_ggml("reuse_token_id_input"))?;
                let embedding = self.device_token_embedding.ok_or_else(|| {
                    GraniteSpeechDecoderError::Shape {
                        reason: "Granite token-id graph has no device embedding binding"
                            .to_string(),
                    }
                })?;
                let hidden = graph
                    .get_rows(embedding.tensor.as_graph_tensor(), token_id)
                    .map_err(map_ggml("reuse_token_embedding_lookup"))?;
                (None, Some(token_id), hidden)
            }
        };
        let row_index = graph
            .new_tensor_1d_i32(1, "granite_reuse_row")
            .map_err(map_ggml("reuse_row_alloc"))?;
        let position = graph
            .new_tensor_1d_i32(1, "granite_reuse_position")
            .map_err(map_ggml("reuse_position_alloc"))?;
        let mask = if use_flash_attention {
            graph
                .new_tensor_2d_f16(max_positions, 1, "granite_reuse_mask")
                .map_err(map_ggml("reuse_flash_mask_alloc"))?
        } else {
            graph
                .new_tensor_2d_f32(max_positions, 1, "granite_reuse_mask")
                .map_err(map_ggml("reuse_mask_alloc"))?
        };
        graph
            .set_input(row_index)
            .map_err(map_ggml("reuse_row_input"))?;
        graph
            .set_input(position)
            .map_err(map_ggml("reuse_position_input"))?;
        graph
            .set_input(mask)
            .map_err(map_ggml("reuse_mask_input"))?;

        let mut hidden = graph
            .scale(hidden_input, config.embedding_multiplier)
            .map_err(map_ggml("reuse_embed_scale"))?;

        for (index, (arena_k, arena_v)) in kv_tensors.iter().copied().enumerate() {
            let layer_weights = self.weights.layer_weights(index);
            let pre =
                granite_pre_attention(graph, hidden, position, &layer_weights, &config, 1, rope)?;
            // Write this token's K/V into row `row_index` of the resident arena;
            // the returned handles are the full `[head_dim, max_positions,
            // kv_heads]` span (with the new row now live) that attention reads.
            let k_full = graph
                .set_kv_rows(arena_k, pre.k_perm, row_index)
                .map_err(map_ggml("reuse_k_set_rows"))?;
            let v_full = graph
                .set_kv_rows(arena_v, pre.v_perm, row_index)
                .map_err(map_ggml("reuse_v_set_rows"))?;
            let attended = if use_flash_attention {
                graph
                    .flash_attn_ext(
                        pre.q_perm,
                        k_full,
                        v_full,
                        Some(mask),
                        config.attention_multiplier,
                        0.0,
                        0.0,
                    )
                    .map_err(map_ggml("reuse_flash_attn"))?
            } else {
                let scores = graph
                    .mul_mat(k_full, pre.q_perm)
                    .map_err(map_ggml("reuse_scores"))?;
                let probs = graph
                    .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
                    .map_err(map_ggml("reuse_softmax"))?;
                let v_t = graph
                    .cont(
                        graph
                            .transpose(v_full)
                            .map_err(map_ggml("reuse_v_transpose"))?,
                    )
                    .map_err(map_ggml("reuse_v_cont"))?;
                graph
                    .mul_mat(v_t, probs)
                    .map_err(map_ggml("reuse_attended"))?
            };
            hidden = granite_post_attention(
                graph,
                hidden,
                attended,
                &layer_weights,
                &config,
                1,
                use_flash_attention,
            )?;
        }

        let hidden_out = rms_norm(
            graph,
            hidden,
            config.rms_norm_eps,
            self.weights.final_norm_weight(),
        )?;
        let lm_head_w = weight_in_major(
            graph,
            self.weights.lm_head_weight(),
            config.hidden_size,
            config.vocab_size,
            "reuse_lm_head_reshape",
        )?;
        let logits_raw = linear(graph, lm_head_w, hidden_out, "reuse_lm_head")?;
        let logits = graph
            .scale(logits_raw, 1.0 / config.logits_scaling)
            .map_err(map_ggml("reuse_logits_scale"))?;
        let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
            Some(
                graph
                    .top1_argmax_first_max(logits)
                    .map_err(map_ggml("reuse_output_top1"))?,
            )
        } else {
            None
        };
        let output_root = top1.unwrap_or(logits);
        graph
            .set_output(output_root)
            .map_err(map_ggml("reuse_set_output"))?;
        graph
            .prepare_outputs_for_upload(&[output_root])
            .map_err(map_ggml("reuse_prepare_outputs"))?;

        self.reuse = Some(GraniteReusableDecodeGraph {
            session,
            max_positions,
            use_flash_attention,
            input_kind,
            output_mode,
            embed,
            token_id,
            row_index,
            position,
            mask,
            logits,
            top1,
        });
        Ok(())
    }
}

/// One-shot causal prefill that ALSO taps every layer's post-RoPE K/V. Returns
/// the last-position logits row and writes each tap directly into the
/// transactionally admitted token-major host owner.
fn run_prefill_graph(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    embeddings: &[f32],
    n_tokens: usize,
    host_kv: &mut GraniteHostKvState,
) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let kv_heads = config.num_kv_heads;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let kv_width = kv_heads * head_dim;

    let mut graph = runner.start_graph();

    let embed_tensor = graph
        .new_tensor_2d_f32(hidden_size, n_tokens, "granite_session_prefill_embeds")
        .map_err(map_ggml("session_prefill_input_alloc"))?;
    let positions = graph
        .new_tensor_1d_i32(n_tokens, "granite_session_prefill_positions")
        .map_err(map_ggml("session_prefill_positions_alloc"))?;
    let mask = graph
        .new_tensor_2d_f32(n_tokens, n_tokens, "granite_session_prefill_mask")
        .map_err(map_ggml("session_prefill_mask_alloc"))?;

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(map_ggml("session_prefill_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, n_tokens, config.rope_theta)
        .map_err(map_ggml("session_prefill_rope_params"))?;

    let mut kv_taps = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            n_tokens,
            rope,
        )?;
        // Convert the graph tap itself to contiguous token-major
        // `[head_dim, kv_heads, position]`. The admitted host owner can then
        // be the readback target directly; no Rust staging Vec or transpose
        // survives outside ggml's already-admitted graph workspace.
        let k_token_major = graph
            .cont(
                graph
                    .permute(pre.k_perm, 0, 2, 1, 3)
                    .map_err(map_ggml("session_prefill_k_token_permute"))?,
            )
            .map_err(map_ggml("session_prefill_k_token_contiguous"))?;
        let v_token_major = graph
            .cont(
                graph
                    .permute(pre.v_perm, 0, 2, 1, 3)
                    .map_err(map_ggml("session_prefill_v_token_permute"))?,
            )
            .map_err(map_ggml("session_prefill_v_token_contiguous"))?;
        kv_taps.push((k_token_major, v_token_major));

        let scores = graph
            .mul_mat(pre.k_perm, pre.q_perm)
            .map_err(map_ggml("session_prefill_scores"))?;
        let probs = graph
            .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
            .map_err(map_ggml("session_prefill_softmax"))?;
        let v_t = graph
            .cont(
                graph
                    .transpose(pre.v_perm)
                    .map_err(map_ggml("session_prefill_v_transpose"))?,
            )
            .map_err(map_ggml("session_prefill_v_cont"))?;
        let attended = graph
            .mul_mat(v_t, probs)
            .map_err(map_ggml("session_prefill_attended"))?;
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            n_tokens,
            false,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "lm_head_reshape",
    )?;
    let logits_input = last_token_hidden_view(&graph, hidden_out, hidden_size, n_tokens)
        .map_err(map_ggml("session_prefill_last_hidden"))?;
    let logits_raw = linear(&graph, lm_head_w, logits_input, "lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("session_prefill_logits_scale"))?;

    graph
        .set_output(logits)
        .map_err(map_ggml("session_prefill_set_output_logits"))?;
    for (k_tap, v_tap) in &kv_taps {
        graph
            .set_output(*k_tap)
            .map_err(map_ggml("session_prefill_set_output_k"))?;
        graph
            .set_output(*v_tap)
            .map_err(map_ggml("session_prefill_set_output_v"))?;
    }
    graph
        .set_input(embed_tensor)
        .map_err(map_ggml("session_prefill_mark_input_embeds"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("session_prefill_mark_input_positions"))?;
    graph
        .set_input(mask)
        .map_err(map_ggml("session_prefill_mark_input_mask"))?;

    let mut outputs: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    outputs.push(logits);
    for (k_tap, v_tap) in &kv_taps {
        outputs.push(*k_tap);
        outputs.push(*v_tap);
    }
    graph
        .prepare_outputs_for_upload(&outputs)
        .map_err(map_ggml("session_prefill_prepare_outputs"))?;

    graph
        .set_f32_slice(embed_tensor, embeddings, "granite_session_prefill_embeds")
        .map_err(map_ggml("session_prefill_upload_embeds"))?;
    let position_ids: Vec<i32> = (0..n_tokens as i32).collect();
    graph
        .set_i32_slice(
            positions,
            &position_ids,
            "granite_session_prefill_positions",
        )
        .map_err(map_ggml("session_prefill_upload_positions"))?;
    let mask_values = super::decoder_graph::causal_mask(n_tokens);
    graph
        .set_f32_slice(mask, &mask_values, "granite_session_prefill_mask")
        .map_err(map_ggml("session_prefill_upload_mask"))?;

    if host_kv.k_history.len() != config.num_layers || host_kv.v_history.len() != config.num_layers
    {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: "granite admitted host KV layer count does not match decoder".to_string(),
        });
    }
    let written_len =
        n_tokens
            .checked_mul(kv_width)
            .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                reason: "granite prefill host KV written length overflowed".to_string(),
            })?;
    let mut last_logits = Vec::new();
    last_logits.try_reserve_exact(vocab_size).map_err(|error| {
        crate::models::native_execution_services::record_current_execution_candidate_failure(
            crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                "granite_prefill_logits_allocate",
                error.to_string(),
            ),
        );
        GraniteSpeechDecoderError::Shape {
            reason: format!("granite prefill logits allocation failed: {error}"),
        }
    })?;
    last_logits.resize(vocab_size, 0.0);

    let mut destinations: Vec<(GgmlCpuTensor<'_>, &mut [f32])> =
        Vec::with_capacity(1 + kv_taps.len() * 2);
    destinations.push((logits, last_logits.as_mut_slice()));
    for ((k_tap, v_tap), (k_history, v_history)) in kv_taps.iter().zip(
        host_kv
            .k_history
            .iter_mut()
            .zip(host_kv.v_history.iter_mut()),
    ) {
        if k_history.len() < written_len || v_history.len() < written_len {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite admitted host KV storage is smaller than prefill".to_string(),
            });
        }
        destinations.push((*k_tap, &mut k_history[..written_len]));
        destinations.push((*v_tap, &mut v_history[..written_len]));
    }
    graph
        .compute_outputs_into_f32(destinations.as_mut_slice())
        .map_err(map_ggml("session_prefill_compute"))?;
    Ok(last_logits)
}

/// One-shot causal prefill that ALSO seeds the device-resident KV arena
/// (`set_rows` writing rows `0..n_tokens` of every layer's K/V in place), used
/// on the Metal reuse path. Same batched causal forward as `run_prefill_graph`
/// (byte-for-byte the same attention op sequence), but the per-layer K/V is
/// written straight into the resident arena instead of tapped back to the host,
/// so prefill also avoids the O(n) K/V readback. Returns only the last-position
/// logits row.
enum GraniteResidentPrefillInput<'a> {
    Embeddings(&'a [f32]),
    TokenIds {
        embedding: GraniteDeviceTokenEmbedding,
        token_ids: &'a [u32],
        audio_rows: &'a [f32],
        audio_positions: &'a [usize],
    },
}

enum GraniteResidentPrefillUpload<'a> {
    Embeddings {
        tensor: GgmlCpuTensor<'a>,
        values: &'a [f32],
    },
    TokenIds {
        tensor: GgmlCpuTensor<'a>,
        values: Vec<i32>,
        audio: Option<(GgmlCpuTensor<'a>, &'a [f32], GgmlCpuTensor<'a>, Vec<i32>)>,
    },
}

fn run_prefill_graph_seeding_resident(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    resident_kv: &LlmResidentKvArena,
    input: GraniteResidentPrefillInput<'_>,
    n_tokens: usize,
    output_mode: DeviceGreedyStepOutputMode,
) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let use_flash_attention = resident_flash_attention_enabled(runner.backend_kind());

    let mut graph = runner.start_graph();

    let (prompt_rows, prompt_upload) = match input {
        GraniteResidentPrefillInput::Embeddings(values) => {
            let tensor = graph
                .new_tensor_2d_f32(hidden_size, n_tokens, "granite_seed_prefill_embeds")
                .map_err(map_ggml("seed_prefill_input_alloc"))?;
            graph
                .set_input(tensor)
                .map_err(map_ggml("seed_prefill_mark_input_embeds"))?;
            (
                tensor,
                GraniteResidentPrefillUpload::Embeddings { tensor, values },
            )
        }
        GraniteResidentPrefillInput::TokenIds {
            embedding,
            token_ids,
            audio_rows,
            audio_positions,
        } => {
            let token_tensor = graph
                .new_tensor_1d_i32(n_tokens, "granite_seed_prefill_token_ids")
                .map_err(map_ggml("seed_prefill_token_ids_alloc"))?;
            graph
                .set_input(token_tensor)
                .map_err(map_ggml("seed_prefill_mark_input_token_ids"))?;
            // Granite's audio placeholder may be outside the decoder vocab.
            // Match HF's masked-scatter contract: gather row zero at every
            // audio slot, then overwrite those rows with projector output.
            let token_values = token_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(position, token_id)| {
                    let token_id = if audio_positions.binary_search(&position).is_ok() {
                        0
                    } else {
                        token_id
                    };
                    i32::try_from(token_id).map_err(|_| GraniteSpeechDecoderError::Shape {
                        reason: "granite device prompt token id exceeds i32".to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let token_rows = graph
                .get_rows(embedding.tensor.as_graph_tensor(), token_tensor)
                .map_err(map_ggml("seed_prefill_token_lookup"))?;
            if audio_positions.is_empty() {
                (
                    token_rows,
                    GraniteResidentPrefillUpload::TokenIds {
                        tensor: token_tensor,
                        values: token_values,
                        audio: None,
                    },
                )
            } else {
                let audio_count = audio_positions.len();
                let audio_tensor = graph
                    .new_tensor_2d_f32(hidden_size, audio_count, "granite_seed_prefill_audio_rows")
                    .map_err(map_ggml("seed_prefill_audio_rows_alloc"))?;
                let audio_indices = graph
                    .new_tensor_1d_i32(audio_count, "granite_seed_prefill_audio_positions")
                    .map_err(map_ggml("seed_prefill_audio_positions_alloc"))?;
                graph
                    .set_input(audio_tensor)
                    .map_err(map_ggml("seed_prefill_mark_input_audio_rows"))?;
                graph
                    .set_input(audio_indices)
                    .map_err(map_ggml("seed_prefill_mark_input_audio_positions"))?;
                let audio_index_values = audio_positions
                    .iter()
                    .copied()
                    .map(|position| {
                        i32::try_from(position).map_err(|_| GraniteSpeechDecoderError::Shape {
                            reason: "granite device prompt audio position exceeds i32".to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let prompt_rows = graph
                    .set_rows(token_rows, audio_tensor, audio_indices)
                    .map_err(map_ggml("seed_prefill_audio_splice"))?;
                (
                    prompt_rows,
                    GraniteResidentPrefillUpload::TokenIds {
                        tensor: token_tensor,
                        values: token_values,
                        audio: Some((audio_tensor, audio_rows, audio_indices, audio_index_values)),
                    },
                )
            }
        }
    };
    let positions = graph
        .new_tensor_1d_i32(n_tokens, "granite_seed_prefill_positions")
        .map_err(map_ggml("seed_prefill_positions_alloc"))?;
    let mask = if use_flash_attention {
        graph
            .new_tensor_2d_f16(n_tokens, n_tokens, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_flash_mask_alloc"))?
    } else {
        graph
            .new_tensor_2d_f32(n_tokens, n_tokens, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_mask_alloc"))?
    };
    let seed_indices = graph
        .new_tensor_1d_i32(n_tokens, "granite_seed_prefill_rows")
        .map_err(map_ggml("seed_prefill_rows_alloc"))?;

    let kv_tensors = resident_kv.graph_tensors();

    let mut hidden = graph
        .scale(prompt_rows, config.embedding_multiplier)
        .map_err(map_ggml("seed_prefill_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, n_tokens, config.rope_theta)
        .map_err(map_ggml("seed_prefill_rope_params"))?;

    for (index, (arena_k, arena_v)) in kv_tensors.iter().copied().enumerate() {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            n_tokens,
            rope,
        )?;
        // Seed rows 0..n_tokens of the resident arena (side effects), then run
        // the ordinary batched causal attention over this prompt's own K/V.
        let k_seed = graph
            .set_rows(arena_k, pre.k_perm, seed_indices)
            .map_err(map_ggml("seed_prefill_k_set_rows"))?;
        graph
            .add_kv_write_root(k_seed)
            .map_err(map_ggml("seed_prefill_k_root"))?;
        let v_seed = graph
            .set_rows(arena_v, pre.v_perm, seed_indices)
            .map_err(map_ggml("seed_prefill_v_set_rows"))?;
        graph
            .add_kv_write_root(v_seed)
            .map_err(map_ggml("seed_prefill_v_root"))?;

        let attended = if use_flash_attention {
            graph
                .flash_attn_ext(
                    pre.q_perm,
                    pre.k_perm,
                    pre.v_perm,
                    Some(mask),
                    config.attention_multiplier,
                    0.0,
                    0.0,
                )
                .map_err(map_ggml("seed_prefill_flash_attn"))?
        } else {
            let scores = graph
                .mul_mat(pre.k_perm, pre.q_perm)
                .map_err(map_ggml("seed_prefill_scores"))?;
            let probs = graph
                .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
                .map_err(map_ggml("seed_prefill_softmax"))?;
            let v_t = graph
                .cont(
                    graph
                        .transpose(pre.v_perm)
                        .map_err(map_ggml("seed_prefill_v_transpose"))?,
                )
                .map_err(map_ggml("seed_prefill_v_cont"))?;
            graph
                .mul_mat(v_t, probs)
                .map_err(map_ggml("seed_prefill_attended"))?
        };
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            n_tokens,
            use_flash_attention,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "seed_lm_head_reshape",
    )?;
    let logits_input = last_token_hidden_view(&graph, hidden_out, hidden_size, n_tokens)
        .map_err(map_ggml("seed_prefill_last_hidden"))?;
    let logits_raw = linear(&graph, lm_head_w, logits_input, "seed_lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("seed_prefill_logits_scale"))?;

    let top1 = if output_mode == DeviceGreedyStepOutputMode::DeviceTop1 {
        Some(
            graph
                .top1_argmax_first_max(logits)
                .map_err(map_ggml("seed_prefill_output_top1"))?,
        )
    } else {
        None
    };
    let output_root = top1.unwrap_or(logits);

    graph
        .set_output(output_root)
        .map_err(map_ggml("seed_prefill_set_output_logits"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("seed_prefill_mark_input_positions"))?;
    graph
        .set_input(mask)
        .map_err(map_ggml("seed_prefill_mark_input_mask"))?;
    graph
        .set_input(seed_indices)
        .map_err(map_ggml("seed_prefill_mark_input_rows"))?;

    graph
        .prepare_outputs_for_upload(&[output_root])
        .map_err(map_ggml("seed_prefill_prepare_outputs"))?;
    match prompt_upload {
        GraniteResidentPrefillUpload::Embeddings { tensor, values } => graph
            .set_f32_slice(tensor, values, "granite_seed_prefill_embeds")
            .map_err(map_ggml("seed_prefill_upload_embeds"))?,
        GraniteResidentPrefillUpload::TokenIds {
            tensor,
            values,
            audio,
        } => {
            graph
                .set_i32_slice(tensor, &values, "granite_seed_prefill_token_ids")
                .map_err(map_ggml("seed_prefill_upload_token_ids"))?;
            if let Some((audio_tensor, audio_rows, audio_indices, audio_index_values)) = audio {
                graph
                    .set_f32_slice(audio_tensor, audio_rows, "granite_seed_prefill_audio_rows")
                    .map_err(map_ggml("seed_prefill_upload_audio_rows"))?;
                graph
                    .set_i32_slice(
                        audio_indices,
                        &audio_index_values,
                        "granite_seed_prefill_audio_positions",
                    )
                    .map_err(map_ggml("seed_prefill_upload_audio_positions"))?;
            }
        }
    }
    let position_ids: Vec<i32> = (0..n_tokens as i32).collect();
    graph
        .set_i32_slice(positions, &position_ids, "granite_seed_prefill_positions")
        .map_err(map_ggml("seed_prefill_upload_positions"))?;
    if use_flash_attention {
        let mask_bits = build_causal_mask_f16_bits(
            n_tokens,
            "granite_seed_prefill_flash_mask",
            |stage, source| GraniteSpeechDecoderError::Ggml { stage, source },
        )?;
        graph
            .set_f16_bits_slice(mask, &mask_bits, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_upload_flash_mask"))?;
    } else {
        let mask_values = super::decoder_graph::causal_mask(n_tokens);
        graph
            .set_f32_slice(mask, &mask_values, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_upload_mask"))?;
    }
    // Rows 0..n_tokens: the prompt's K/V seeds the arena's leading span.
    graph
        .set_i32_slice(seed_indices, &position_ids, "granite_seed_prefill_rows")
        .map_err(map_ggml("seed_prefill_upload_rows"))?;
    match top1 {
        Some(top1) => {
            let token_id = graph
                .compute_output_i32(top1, 1)
                .map_err(map_ggml("seed_prefill_compute_top1"))?
                .into_iter()
                .next()
                .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                    reason: "granite prefill device top-1 returned no token id".to_string(),
                })?;
            Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(map_device_top1_token(token_id, vocab_size)?),
            })
        }
        None => Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits: graph
                .compute_output_f32(logits, vocab_size)
                .map_err(map_ggml("seed_prefill_compute"))?,
            greedy_token_hint: None,
        }),
    }
}

/// One incremental single-token step. Each admitted history layer is a
/// contiguous token-major
/// `[seq_len, kv_heads, head_dim]` prefixes. The graph uploads them as
/// `[head_dim, kv_heads, seq_len]` and permutes/contiguates it inside the graph
/// workspace to the attention layout `[head_dim, seq_len, kv_heads]`. The new
/// token's K/V is read directly into row `seq_len` of the admitted owner.
#[allow(clippy::too_many_arguments)]
fn run_decode_step_graph(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    embed_row: &[f32],
    seq_len: usize,
    host_kv: &mut GraniteHostKvState,
) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let kv_heads = config.num_kv_heads;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let kv_width = kv_heads * head_dim;
    let new_position = seq_len; // 0-based position of the new token.
    if host_kv.k_history.len() != config.num_layers || host_kv.v_history.len() != config.num_layers
    {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: "granite admitted host KV layer count does not match decoder".to_string(),
        });
    }
    let history_len =
        seq_len
            .checked_mul(kv_width)
            .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                reason: "granite step host KV history length overflowed".to_string(),
            })?;
    let row_end =
        history_len
            .checked_add(kv_width)
            .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                reason: "granite step host KV row end overflowed".to_string(),
            })?;
    if host_kv
        .k_history
        .iter()
        .chain(host_kv.v_history.iter())
        .any(|history| history.len() < row_end)
    {
        return Err(GraniteSpeechDecoderError::Shape {
            reason: "granite admitted host KV storage is smaller than decode step".to_string(),
        });
    }

    let mut graph = runner.start_graph();

    let embed_tensor = graph
        .new_tensor_2d_f32(hidden_size, 1, "granite_session_step_embed")
        .map_err(map_ggml("session_step_input_alloc"))?;
    let positions = graph
        .new_tensor_1d_i32(1, "granite_session_step_position")
        .map_err(map_ggml("session_step_position_alloc"))?;

    // Per-layer token-major K/V history input tensors
    // (`[head_dim, kv_heads, seq_len]`). This matches append order in the host
    // arena, so no full-history host flatten/copy is needed per token.
    let mut k_hist_tensors = Vec::with_capacity(config.num_layers);
    let mut v_hist_tensors = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        k_hist_tensors.push(
            graph
                .new_tensor_3d_f32(head_dim, kv_heads, seq_len, "granite_session_step_k_hist")
                .map_err(map_ggml("session_step_k_hist_alloc"))?,
        );
        v_hist_tensors.push(
            graph
                .new_tensor_3d_f32(head_dim, kv_heads, seq_len, "granite_session_step_v_hist")
                .map_err(map_ggml("session_step_v_hist_alloc"))?,
        );
    }

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(map_ggml("session_step_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, seq_len + 1, config.rope_theta)
        .map_err(map_ggml("session_step_rope_params"))?;

    let mut kv_taps = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            1,
            rope,
        )?;
        kv_taps.push((pre.k_perm, pre.v_perm));

        // Restore the attention layout `[head_dim, seq_len, kv_heads]` and
        // make it contiguous for concat. The copy now lives in ggml's quoted
        // graph workspace instead of an untracked Rust Vec allocation.
        let k_history = graph
            .cont(
                graph
                    .permute(k_hist_tensors[index], 0, 2, 1, 3)
                    .map_err(map_ggml("session_step_k_history_permute"))?,
            )
            .map_err(map_ggml("session_step_k_history_contiguous"))?;
        let v_history = graph
            .cont(
                graph
                    .permute(v_hist_tensors[index], 0, 2, 1, 3)
                    .map_err(map_ggml("session_step_v_history_permute"))?,
            )
            .map_err(map_ggml("session_step_v_history_contiguous"))?;

        // Attend the single new query against `history ++ new`.
        let k_full = graph
            .concat(k_history, pre.k_perm, 1)
            .map_err(map_ggml("session_step_k_concat"))?;
        let v_full = graph
            .concat(v_history, pre.v_perm, 1)
            .map_err(map_ggml("session_step_v_concat"))?;
        let scores = graph
            .mul_mat(k_full, pre.q_perm)
            .map_err(map_ggml("session_step_scores"))?;
        // No mask: every cached key precedes the new query, so all are valid
        // (prefill's masked keys would contribute exactly 0.0 and are simply
        // absent here -- bit-identical, see module doc).
        let probs = graph
            .soft_max_ext(scores, None, config.attention_multiplier, 0.0)
            .map_err(map_ggml("session_step_softmax"))?;
        let v_t = graph
            .cont(
                graph
                    .transpose(v_full)
                    .map_err(map_ggml("session_step_v_transpose"))?,
            )
            .map_err(map_ggml("session_step_v_cont"))?;
        let attended = graph
            .mul_mat(v_t, probs)
            .map_err(map_ggml("session_step_attended"))?;
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            1,
            false,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "lm_head_reshape",
    )?;
    let logits_raw = linear(&graph, lm_head_w, hidden_out, "lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("session_step_logits_scale"))?;

    graph
        .set_output(logits)
        .map_err(map_ggml("session_step_set_output_logits"))?;
    for (k_tap, v_tap) in &kv_taps {
        graph
            .set_output(*k_tap)
            .map_err(map_ggml("session_step_set_output_k"))?;
        graph
            .set_output(*v_tap)
            .map_err(map_ggml("session_step_set_output_v"))?;
    }
    graph
        .set_input(embed_tensor)
        .map_err(map_ggml("session_step_mark_input_embed"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("session_step_mark_input_position"))?;
    for index in 0..config.num_layers {
        graph
            .set_input(k_hist_tensors[index])
            .map_err(map_ggml("session_step_mark_input_k_hist"))?;
        graph
            .set_input(v_hist_tensors[index])
            .map_err(map_ggml("session_step_mark_input_v_hist"))?;
    }

    let mut outputs: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    outputs.push(logits);
    for (k_tap, v_tap) in &kv_taps {
        outputs.push(*k_tap);
        outputs.push(*v_tap);
    }
    graph
        .prepare_outputs_for_upload(&outputs)
        .map_err(map_ggml("session_step_prepare_outputs"))?;

    graph
        .set_f32_slice(embed_tensor, embed_row, "granite_session_step_embed")
        .map_err(map_ggml("session_step_upload_embed"))?;
    let new_position_i32 =
        i32::try_from(new_position).map_err(|_| GraniteSpeechDecoderError::Shape {
            reason: format!("granite decode position {new_position} does not fit i32"),
        })?;
    graph
        .set_i32_slice(
            positions,
            &[new_position_i32],
            "granite_session_step_position",
        )
        .map_err(map_ggml("session_step_upload_position"))?;
    for index in 0..config.num_layers {
        graph
            .set_f32_slice(
                k_hist_tensors[index],
                &host_kv.k_history[index][..history_len],
                "granite_session_step_k_hist",
            )
            .map_err(map_ggml("session_step_upload_k_hist"))?;
        graph
            .set_f32_slice(
                v_hist_tensors[index],
                &host_kv.v_history[index][..history_len],
                "granite_session_step_v_hist",
            )
            .map_err(map_ggml("session_step_upload_v_hist"))?;
    }

    let mut logits_row = Vec::new();
    logits_row.try_reserve_exact(vocab_size).map_err(|error| {
        crate::models::native_execution_services::record_current_execution_candidate_failure(
            crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                "granite_step_logits_allocate",
                error.to_string(),
            ),
        );
        GraniteSpeechDecoderError::Shape {
            reason: format!("granite step logits allocation failed: {error}"),
        }
    })?;
    logits_row.resize(vocab_size, 0.0);
    let mut destinations: Vec<(GgmlCpuTensor<'_>, &mut [f32])> =
        Vec::with_capacity(1 + kv_taps.len() * 2);
    destinations.push((logits, logits_row.as_mut_slice()));
    for ((k_tap, v_tap), (k_history, v_history)) in kv_taps.iter().zip(
        host_kv
            .k_history
            .iter_mut()
            .zip(host_kv.v_history.iter_mut()),
    ) {
        destinations.push((*k_tap, &mut k_history[history_len..row_end]));
        destinations.push((*v_tap, &mut v_history[history_len..row_end]));
    }
    graph
        .compute_outputs_into_f32(destinations.as_mut_slice())
        .map_err(map_ggml("session_step_compute"))?;
    Ok(logits_row)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::granite_speech::decoder_graph::prefill_logits;

    /// Deterministic pseudo-random f32 generator (xorshift64*, no `rand` dep),
    /// values scaled into `[-amp, amp)` so a two-layer forward stays finite.
    fn deterministic_weights(seed: u64, len: usize, amp: f32) -> Vec<f32> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0;
            out.push((unit * 2.0 - 1.0) * amp);
        }
        out
    }

    /// A tiny Granite decoder config exercising every scaling scalar plus GQA
    /// (4 query / 2 KV heads) and an even (RoPE-NEOX) head dim.
    fn tiny_config() -> GraniteSpeechDecoderConfig {
        GraniteSpeechDecoderConfig {
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            vocab_size: 48,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10000.0,
            attention_multiplier: 0.0078125,
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
        }
    }

    #[test]
    fn device_top1_quote_has_no_reverse_index_construction_staging() {
        let config = tiny_config();
        assert_eq!(
            GraniteSpeechDecodeSession::quoted_construction_transient_system_memory_bytes(
                &config,
                DeviceGreedyStepOutputMode::FullLogits,
            )
            .expect("full-logits construction quote"),
            0
        );
        assert_eq!(
            GraniteSpeechDecodeSession::quoted_construction_transient_system_memory_bytes(
                &config,
                DeviceGreedyStepOutputMode::DeviceTop1,
            )
            .expect("device top-1 construction quote"),
            0
        );
    }

    fn build_tiny_weights(config: &GraniteSpeechDecoderConfig) -> HashMap<String, Vec<f32>> {
        let d = config.hidden_size;
        let q_width = config.num_heads * config.head_dim;
        let kv_width = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;
        let mut weights = HashMap::new();
        let mut seed = 1u64;
        let next = |len: usize, amp: f32, seed: &mut u64| {
            *seed = seed.wrapping_add(0x1000);
            deterministic_weights(*seed, len, amp)
        };
        for layer in 0..config.num_layers {
            let p = |suffix: &str| format!("language_model.model.layers.{layer}.{suffix}");
            // Norm weights near 1.0 (RMSNorm scale); projections small.
            weights.insert(
                p("input_layernorm.weight"),
                next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
            );
            weights.insert(
                p("self_attn.q_proj.weight"),
                next(d * q_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.k_proj.weight"),
                next(d * kv_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.v_proj.weight"),
                next(d * kv_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.o_proj.weight"),
                next(q_width * d, 0.05, &mut seed),
            );
            weights.insert(
                p("post_attention_layernorm.weight"),
                next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
            );
            weights.insert(p("mlp.gate_proj.weight"), next(d * inter, 0.05, &mut seed));
            weights.insert(p("mlp.up_proj.weight"), next(d * inter, 0.05, &mut seed));
            weights.insert(p("mlp.down_proj.weight"), next(inter * d, 0.05, &mut seed));
        }
        weights.insert(
            "language_model.model.norm.weight".to_string(),
            next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
        );
        weights.insert(
            "language_model.lm_head.weight".to_string(),
            next(d * config.vocab_size, 0.05, &mut seed),
        );
        weights.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            next(config.vocab_size * d, 0.1, &mut seed),
        );
        weights
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            })
            .0 as u32
    }

    /// Full-recompute reference last-position logits over `token_ids`, using the
    /// one-shot `prefill_logits` (the path this session replaces).
    fn recompute_last_logits(
        config: &GraniteSpeechDecoderConfig,
        weights: &HashMap<String, Vec<f32>>,
        token_ids: &[u32],
    ) -> Vec<f32> {
        let out = prefill_logits(config, weights, token_ids, GgmlCpuGraphBackend::Cpu)
            .expect("full recompute prefill");
        let last_start = (out.n_tokens - 1) * out.vocab_size;
        out.logits[last_start..last_start + out.vocab_size].to_vec()
    }

    fn assert_bit_identical(step: usize, incremental: &[f32], recompute: &[f32]) {
        assert_eq!(
            incremental.len(),
            recompute.len(),
            "step {step}: logits width mismatch"
        );
        for (i, (a, b)) in incremental.iter().zip(recompute.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step}: logit[{i}] differs (incremental {a} vs recompute {b})"
            );
        }
    }

    #[test]
    fn fixed_span_tail_mask_exposes_only_cached_prefix() {
        assert_eq!(
            fixed_span_tail_mask(4, 0),
            vec![0.0, f32::MIN, f32::MIN, f32::MIN]
        );
        assert_eq!(fixed_span_tail_mask(4, 2), vec![0.0, 0.0, 0.0, f32::MIN]);
        assert_eq!(fixed_span_tail_mask(4, 3), vec![0.0; 4]);
    }

    #[test]
    fn granite_capacity_keeps_runtime_bound_and_stable_reserve_distinct() {
        let capacity = GraniteSpeechKvCacheCapacity::new(640, 960).expect("capacity");
        assert_eq!(capacity.logical_positions(), 640);
        assert_eq!(capacity.resident_positions(), 960);
        assert_eq!(
            capacity
                .validate_measured_logical_positions(639)
                .expect_err("runtime/planner drift must fail closed"),
            GraniteSpeechKvCacheCapacityError::LogicalPositionMismatch {
                planned_positions: 640,
                measured_positions: 639,
            }
        );
        assert_eq!(
            capacity
                .validate_hard_cap(959)
                .expect_err("reserve above hard cap must fail"),
            GraniteSpeechKvCacheCapacityError::HardCapExceeded {
                resident_positions: 960,
                hard_cap: 959,
            }
        );
    }

    #[test]
    fn granite_capacity_requires_planned_decoder_state() {
        assert_eq!(
            GraniteSpeechKvCacheCapacity::from_decoder_state(
                &crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            )
            .expect_err("runtime must not invent a capacity fallback"),
            GraniteSpeechKvCacheCapacityError::DecoderStateNotPlanned,
        );
    }

    /// The load-bearing correctness gate: the incremental KV-cache session must
    /// reproduce the one-shot full recompute's logits BIT-FOR-BIT at every step
    /// (which also forces identical greedy token choices). Runs entirely on
    /// synthetic weights, so it needs no external checkpoint and runs in CI.
    #[test]
    fn granite_incremental_decode_matches_full_recompute_bit_exact() {
        let config = tiny_config();
        let weights = build_tiny_weights(&config);
        let prompt: Vec<u32> = vec![1, 7, 3, 42, 5, 9];

        // Prompt embeddings (raw, pre-`embedding_multiplier`) for the session.
        let mut prompt_embeddings = Vec::with_capacity(prompt.len() * config.hidden_size);
        for &id in &prompt {
            prompt_embeddings
                .extend_from_slice(embed_token_row(&config, &weights, id).expect("embed prompt"));
        }

        let mut session =
            GraniteSpeechDecodeSession::new(config, &weights, GgmlCpuGraphBackend::Cpu)
                .expect("session");
        let logical_positions = prompt.len() + 8;
        let capacity = GraniteSpeechKvCacheCapacity::new(logical_positions, logical_positions + 4)
            .expect("test capacity");

        // Step 0: prefill logits vs full recompute over the prompt alone.
        let inc0 = session
            .prefill(&prompt_embeddings, prompt.len(), capacity)
            .expect("prefill");
        assert!(
            session.resident_kv.is_none(),
            "CPU must not allocate resident KV"
        );
        let host_kv = session.host_kv.as_ref().expect("CPU host KV owner");
        let k_allocations: Vec<(*const f32, usize)> = host_kv
            .k_history
            .iter()
            .map(|history| (history.as_ptr(), history.capacity()))
            .collect();
        let v_allocations: Vec<(*const f32, usize)> = host_kv
            .v_history
            .iter()
            .map(|history| (history.as_ptr(), history.capacity()))
            .collect();
        assert_eq!(session.logical_capacity, logical_positions);
        let ref0 = recompute_last_logits(&config, &weights, &prompt);
        assert_bit_identical(0, &inc0, &ref0);
        assert_eq!(session.cached_positions(), prompt.len());

        // Greedy-decode a handful of steps, comparing each step's incremental
        // logits against a fresh full recompute over prompt ++ generated.
        let mut generated: Vec<u32> = Vec::new();
        let mut next_logits = inc0;
        for step in 1..=8usize {
            let token = argmax(&next_logits);
            generated.push(token);

            let inc = session
                .decode_step(token, &weights)
                .expect("incremental decode step");

            let mut sequence = prompt.clone();
            sequence.extend_from_slice(&generated);
            let reference = recompute_last_logits(&config, &weights, &sequence);

            assert_bit_identical(step, &inc, &reference);
            assert_eq!(session.cached_positions(), prompt.len() + generated.len());
            let host_kv = session.host_kv.as_ref().expect("CPU host KV owner");
            assert_eq!(
                host_kv
                    .k_history
                    .iter()
                    .map(|history| (history.as_ptr(), history.capacity()))
                    .collect::<Vec<_>>(),
                k_allocations,
                "K history must never realloc"
            );
            assert_eq!(
                host_kv
                    .v_history
                    .iter()
                    .map(|history| (history.as_ptr(), history.capacity()))
                    .collect::<Vec<_>>(),
                v_allocations,
                "V history must never realloc"
            );
            next_logits = inc;
        }
    }

    /// Not a gate -- a manual demonstration that the incremental session is
    /// `O(1)` per step (flat decode time as the prefix grows) while the old
    /// recompute-the-whole-prefix path is `O(prefix)` per step (i.e. `O(n^2)`
    /// over a full decode). Uses a mid-sized synthetic config so ggml compute,
    /// not graph construction, dominates. Run with:
    /// `cargo test -p openasr-core --lib granite_incremental_decode_is_linear_not_quadratic -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf demonstration (synthetic weights), not a correctness gate"]
    fn granite_incremental_decode_is_linear_not_quadratic() {
        use std::time::Instant;

        let config = GraniteSpeechDecoderConfig {
            hidden_size: 1024,
            num_layers: 6,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 2816,
            vocab_size: 2048,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10000.0,
            attention_multiplier: 0.0078125,
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
        };
        let weights = build_tiny_weights(&config);
        let prompt: Vec<u32> = (0..32u32).collect();

        let mut prompt_embeddings = Vec::with_capacity(prompt.len() * config.hidden_size);
        for &id in &prompt {
            prompt_embeddings
                .extend_from_slice(embed_token_row(&config, &weights, id).expect("embed prompt"));
        }

        // Incremental: prefill once, then time individual steps as the cache grows.
        let mut session =
            GraniteSpeechDecodeSession::new(config, &weights, GgmlCpuGraphBackend::Cpu)
                .expect("session");
        let logical_positions = prompt.len() + 96;
        let capacity = GraniteSpeechKvCacheCapacity::new(logical_positions, logical_positions + 16)
            .expect("test capacity");
        let mut logits = session
            .prefill(&prompt_embeddings, prompt.len(), capacity)
            .expect("prefill");
        let mut incremental_first = None;
        let mut incremental_last = None;
        let mut incremental_total = std::time::Duration::ZERO;
        let steps = 96usize;
        for step in 0..steps {
            let token = argmax(&logits) % config.vocab_size as u32;
            let start = Instant::now();
            logits = session
                .decode_step(token, &weights)
                .expect("incremental step");
            let elapsed = start.elapsed();
            incremental_total += elapsed;
            if step == 0 {
                incremental_first = Some(elapsed);
            }
            if step == steps - 1 {
                incremental_last = Some(elapsed);
            }
        }

        // Recompute: time a full-prefix forward at growing prefix lengths (the
        // work the old executor did on EVERY step).
        let mut recompute_samples = Vec::new();
        for extra in [0usize, 32, 64, 96] {
            let mut sequence = prompt.clone();
            sequence.extend((0..extra as u32).map(|i| i % config.vocab_size as u32));
            let start = Instant::now();
            let _ = prefill_logits(&config, &weights, &sequence, GgmlCpuGraphBackend::Cpu)
                .expect("recompute");
            recompute_samples.push((sequence.len(), start.elapsed()));
        }

        let inc_first = incremental_first.unwrap();
        let inc_last = incremental_last.unwrap();
        println!("== granite incremental-vs-recompute scaling ==");
        println!(
            "incremental: {steps} steps, first-step {inc_first:?}, last-step (prefix {}) {inc_last:?}, avg {:?}",
            prompt.len() + steps - 1,
            incremental_total / steps as u32
        );
        for (len, dur) in &recompute_samples {
            println!("recompute full forward: prefix {len:>4} tokens -> {dur:?}");
        }
        // The incremental last step attends a ~4x longer prefix than the first
        // yet stays within a small constant factor (projections/MLP are O(1));
        // the recompute grows roughly linearly with prefix length.
        println!(
            "incremental last/first ratio: {:.2}x (flat == O(1) per step)",
            inc_last.as_secs_f64() / inc_first.as_secs_f64().max(1e-9)
        );
    }
}
