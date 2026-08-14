//! moss-transcribe-diarize dedicated executor: chunked Whisper-Medium encoder
//! (30s windows) -> valid-prefix trim + [`adaptor_graph`] per chunk ->
//! concatenate final adaptor rows. Accelerated lanes fuse trim, 4x merge, and
//! VQAdaptor into the encoder ggml graph; CPU keeps the original scalar adaptor
//! as its exact numerical oracle. Per-chunk adaptor execution is equivalent to
//! adapting the concatenated sequence because every kept chunk length is a
//! multiple of the merge size. The result feeds the ChatML+audio-span
//! prompt ([`decode_prompt`] + [`prompt_embedding`]'s sparse splice, since
//! digit time-anchor tokens interrupt the `<|audio_pad|>` run) -> Qwen3-0.6B
//! [`llm_decoder`] prefill/decode, driven through the ONE shared greedy
//! decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a [`Seq2SeqGreedyDecodeStepExecutor`]
//! impl below -- never a hand-rolled argmax loop (this repo's
//! `model-integration-shared-driver` invariant, see `AGENTS.md`).
//!
//! File-transcribe only: no streaming/realtime session (this family's
//! architecture always needs the full audio to compute time-anchor markers
//! ahead of the prompt, so there is no meaningful "partial" mode yet).

#![allow(dead_code)]

use std::sync::Arc;

use thiserror::Error;

use crate::api::backend::Transcription;
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlNativeGqaCapability, RequestBackendPreference,
    request_backend_override,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::current_execution_placement;
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::prepared_runtime_cache::{
    PreparedRuntimeCache, PreparedRuntimeHandle, PreparedRuntimeQuoteContext,
};
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrKvCacheCapacity, Qwen3AsrKvCacheCapacityError,
    Qwen3AsrPromptTokenInput, qwen_llm_effective_native_gqa_capability,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    Seq2SeqGreedyDecodeStopReason,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryCapacity, SystemMemoryOwner,
};
use crate::models::whisper::whisper_log_mel_spectrogram_16khz_mono_v0;

use super::adaptor_graph::MossAdaptorWeights;
use super::decode_budget::moss_td_generated_token_budget as moss_td_budget_for_shape;
#[cfg(test)]
use super::decode_budget::{MOSS_TD_MAX_GENERATED_TOKENS, MOSS_TD_MIN_GENERATED_TOKENS};
use super::decode_prompt::build_moss_td_decode_prompt;
use super::encoder_graph::{MossEncoderConfig, MossEncoderRuntime};
use super::graph_config::{moss_td_encoder_graph_config, moss_td_runtime_graph_config};
use super::llm_decoder::MossTdDecoderRuntime;
use super::prepared_runtime::{
    MossTdPreparedRuntime, MossTdPreparedRuntimeError, build_moss_td_prepared_runtime,
};
use super::runtime_contract::{
    MOSS_TD_ADAPTOR_NORM_EPSILON, MossTdDecoderMetadata, moss_td_kv_cache_positions,
    moss_td_request_kv_cache_positions,
};
use super::speaker_segments::MossTdDecodeExtent;
use super::tokenizer::MossTdTokenizer;

/// `WhisperFeatureExtractor`'s `chunk_length=30` @ 16kHz (`preprocessor_config.json`,
/// verified against the real checkpoint). `pub(crate)` because the capacity
/// derivation shares the chunk quantum (see `super::capacity`).
pub(crate) const CHUNK_SAMPLES: usize = 480_000;
const MEL_TARGET_FRAMES: usize = 3000;
/// `pub(crate)` for the same reason: the capacity frontend registry states
/// the same architectural facts and is pinned equal to these constants.
pub(crate) const SAMPLE_RATE_HZ: usize = 16_000;
/// `WhisperFeatureExtractor.hop_length` (160) * the Whisper conv stem's 2x
/// stride * `audio_merge_size` -- upstream's
/// `_compute_audio_token_length`'s `stride` (`processing_moss_transcribe_diarize.py`).
pub(crate) const WHISPER_ENCODER_CONV_STRIDE: usize = 2;
pub(crate) const HOP_LENGTH: usize = 160;
/// Audio tokens per second the adaptor emits (`audio_tokens_per_second` in
/// `processing_moss_transcribe_diarize.py`, same value `decode_prompt`'s marker
/// cadence uses). Only used to render the `AudioExceedsContext` limit as an
/// approximate minutes figure; not part of any decode math. `pub(crate)` so
/// `super::capacity`'s drift guard can pin it equal to the capacity frontend
/// registry's derived rate (three copies of one fact, one pinned number).
pub(crate) const AUDIO_TOKENS_PER_SECOND_FOR_LIMIT: f32 =
    super::decode_prompt::AUDIO_TOKENS_PER_SECOND;

#[derive(Debug, Error)]
enum MossTdExecutorError {
    #[error("moss-transcribe-diarize executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("moss-transcribe-diarize runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("moss-transcribe-diarize tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("moss-transcribe-diarize requires non-empty audio")]
    EmptyAudio,
    #[error(
        "moss-transcribe-diarize one invocation contains {samples} samples, exceeding the product envelope {max_samples} samples"
    )]
    InvocationTooLong { samples: usize, max_samples: usize },
    #[error("moss-transcribe-diarize decode budget is unavailable: {reason}")]
    DecodeBudgetUnavailable { reason: String },
    #[error(
        "moss-transcribe-diarize audio is too long: its {prompt_tokens}-token audio prompt plus \
         the {generation_budget}-token decode budget needs {required_positions} positions within \
         the {kv_capacity}-position decoder context (about {max_minutes:.0} min of audio); split \
         the input into shorter files"
    )]
    AudioExceedsContext {
        prompt_tokens: usize,
        generation_budget: usize,
        required_positions: usize,
        kv_capacity: usize,
        max_minutes: f32,
    },
    #[error("moss-transcribe-diarize mel frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("moss-transcribe-diarize encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("moss-transcribe-diarize decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("moss-transcribe-diarize decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("moss-transcribe-diarize {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error("moss-transcribe-diarize prepared runtime failed: {reason}")]
    PreparedRuntimeFailed { reason: String },
    #[error("moss-transcribe-diarize prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("moss-transcribe-diarize greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

const MOSS_TD_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const MOSS_TD_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MossTdGraphRuntimeCacheProfile {
    context_bytes: usize,
    graph_size: usize,
    n_threads: Option<usize>,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    use_scheduler: bool,
}

impl From<crate::ggml_runtime::GgmlCpuGraphConfig> for MossTdGraphRuntimeCacheProfile {
    fn from(config: crate::ggml_runtime::GgmlCpuGraphConfig) -> Self {
        Self {
            context_bytes: config.context_bytes,
            graph_size: config.graph_size,
            n_threads: config.n_threads,
            backend: config.backend,
            use_scheduler: config.use_scheduler,
        }
    }
}

type MossTdEncoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    MossTdGraphRuntimeCacheProfile,
);
type MossTdDecoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    MossTdGraphRuntimeCacheProfile,
    GgmlNativeGqaCapability,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MossTdUnifiedRuntimeCacheKey {
    content: PackContentKey,
    lane: ExecutionLaneKey,
    encoder_profile: MossTdGraphRuntimeCacheProfile,
    decoder_profile: MossTdGraphRuntimeCacheProfile,
    native_gqa: GgmlNativeGqaCapability,
}

struct MossTdEncoderActorState {
    runtime: MossEncoderRuntime,
    _prepared_owner: PreparedRuntimeHandle<MossTdPreparedRuntime>,
}

struct MossTdDecoderActorState {
    runtime: MossTdDecoderRuntime,
    _prepared_owner: PreparedRuntimeHandle<MossTdPreparedRuntime>,
}

struct MossTdUnifiedActorState {
    encoder: MossEncoderRuntime,
    decoder: MossTdDecoderRuntime,
    _prepared_owner: PreparedRuntimeHandle<MossTdPreparedRuntime>,
}

type MossTdEncoderRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MossTdEncoderRuntimeCacheKey, MossTdEncoderActorState>;
type MossTdDecoderRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MossTdDecoderRuntimeCacheKey, MossTdDecoderActorState>;
type MossTdUnifiedRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MossTdUnifiedRuntimeCacheKey, MossTdUnifiedActorState>;
type MossTdEncoderRuntimeActor =
    PinnedRuntimeActorCheckout<MossTdEncoderRuntimeCacheKey, MossTdEncoderActorState>;
type MossTdDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<MossTdDecoderRuntimeCacheKey, MossTdDecoderActorState>;
type MossTdUnifiedRuntimeActor =
    PinnedRuntimeActorCheckout<MossTdUnifiedRuntimeCacheKey, MossTdUnifiedActorState>;

fn moss_td_unified_system_memory_shape(
    encoder_retained: u64,
    decoder_retained: u64,
) -> Result<(u64, u64), String> {
    let retained = encoder_retained
        .checked_add(decoder_retained)
        .ok_or_else(|| "MOSS unified runtime retained bytes overflowed".to_string())?;
    Ok((retained, retained))
}

impl MossTdUnifiedActorState {
    fn system_memory_quote(
        prepared: &MossTdPreparedRuntime,
        content_id: &str,
        encoder_config: GgmlCpuGraphConfig,
        decoder_config: GgmlCpuGraphConfig,
    ) -> Result<SystemMemoryAllocationQuote, String> {
        if encoder_config.backend != GgmlCpuGraphBackend::Gpu
            || encoder_config.use_scheduler
            || decoder_config.backend != GgmlCpuGraphBackend::Gpu
            || decoder_config.use_scheduler
        {
            return Err("MOSS unified runtime quote requires direct GPU stages".to_string());
        }
        // The accelerated encoder owns only typed handles into the prepared
        // runtime and the shared loaded context. Its CPU adaptor Vecs do not
        // exist on this lane, so its actor-local SystemMemory retention is 0.
        let encoder_retained = 0;
        let decoder_retained =
            MossTdDecoderRuntime::quoted_resident_system_memory_bytes(&prepared.decoder_plan)?;
        let (peak, retained) =
            moss_td_unified_system_memory_shape(encoder_retained, decoder_retained)?;
        SystemMemoryAllocationQuote::new(
            format!("moss-td-unified-runtime:{content_id}"),
            peak,
            retained,
        )
        .map_err(|error| error.to_string())
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = SystemMemoryCapacity::default();
        bytes.add(
            self.encoder.retained_host_system_memory_bytes()?,
            "MOSS unified encoder runtime",
        )?;
        bytes.add(
            self.decoder.resident_system_memory_bytes()?,
            "MOSS unified decoder runtime",
        )?;
        Ok(bytes.finish())
    }
}

fn moss_td_unified_runtime_configs(
    allow_unified_runtime: bool,
    backend: GgmlCpuGraphBackend,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> Option<(GgmlCpuGraphConfig, GgmlCpuGraphConfig)> {
    let encoder = moss_td_encoder_graph_config(backend);
    let decoder = moss_td_runtime_graph_config(backend);
    moss_td_unified_runtime_enabled(
        allow_unified_runtime,
        backend,
        backend_preference,
        placement,
        encoder,
        decoder,
    )
    .then_some((encoder, decoder))
}

fn moss_td_unified_runtime_enabled(
    allow_unified_runtime: bool,
    backend: GgmlCpuGraphBackend,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
    encoder: GgmlCpuGraphConfig,
    decoder: GgmlCpuGraphConfig,
) -> bool {
    allow_unified_runtime
        && backend == GgmlCpuGraphBackend::Gpu
        && placement == Some(ExecutionPlacement::FullDevice)
        && matches!(
            backend_preference,
            Some(RequestBackendPreference::Exact(route))
                if route.addressability.is_exactly_addressable()
                    && matches!(route.provider, ExecutionProvider::Cuda | ExecutionProvider::Vulkan)
        )
        && encoder.backend == GgmlCpuGraphBackend::Gpu
        && !encoder.use_scheduler
        && decoder.backend == GgmlCpuGraphBackend::Gpu
        && !decoder.use_scheduler
}

fn moss_td_native_gqa_candidate(
    backend: GgmlCpuGraphBackend,
    preference: Option<&RequestBackendPreference>,
    resolved: GgmlNativeGqaCapability,
) -> GgmlNativeGqaCapability {
    match backend {
        GgmlCpuGraphBackend::Cpu | GgmlCpuGraphBackend::Metal => resolved,
        GgmlCpuGraphBackend::Gpu => match preference {
            Some(RequestBackendPreference::Exact(route))
                if route.provider == ExecutionProvider::Vulkan =>
            {
                resolved
            }
            Some(RequestBackendPreference::CpuOnly)
            | Some(RequestBackendPreference::Accelerated)
            | Some(RequestBackendPreference::Exact(_))
            | None => GgmlNativeGqaCapability::Unsupported,
        },
    }
}

#[derive(Clone)]
pub(crate) struct MossTdGgmlExecutor {
    prepared_runtimes: Arc<PreparedRuntimeCache<MossTdPreparedRuntime>>,
    encoder_runtimes: Arc<MossTdEncoderRuntimePool>,
    decoder_runtimes: Arc<MossTdDecoderRuntimePool>,
    unified_runtimes: Arc<MossTdUnifiedRuntimePool>,
}

impl std::fmt::Debug for MossTdGgmlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTdGgmlExecutor").finish_non_exhaustive()
    }
}

impl Default for MossTdGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            MOSS_TD_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            MOSS_TD_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            prepared_runtimes: Arc::new(PreparedRuntimeCache::default()),
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moss-td-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moss-td-decoder-owner",
                limits,
            )),
            unified_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moss-td-unified-owner",
                limits,
            )),
        }
    }
}

const MOSS_TD_EXECUTOR_ID: &str = crate::arch::MOSS_TD_EXECUTOR_COMPONENT_ID;
const MOSS_TD_STREAMING_EXECUTOR_ID: &str =
    "moss-transcribe-diarize-ggml-snapshot-streaming-executor-v1";

struct MossTdGreedyStepExecutor<'a> {
    decoder: &'a mut MossTdDecoderRuntime,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    prompt_input: Option<Qwen3AsrPromptTokenInput>,
    cache_prompt_tokens: usize,
    /// Explicit cancel/pause/resume control for this decode -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    control: std::sync::Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for MossTdGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if let Some(prompt_input) = self.prompt_input.take() {
            self.cache_prompt_tokens = prompt_input.token_ids.len();
            let prefill = self
                .decoder
                .prefill_token_ids_with_audio(
                    &prompt_input.token_ids,
                    &prompt_input.audio_rows,
                    &prompt_input.audio_positions,
                    &mut self.layer_kv_caches,
                    self.kv_capacity,
                    &self.control,
                )
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: prefill.logits,
                greedy_token_hint: prefill.greedy_token_hint,
            });
        }
        let last_token = input.generated_tokens.last().copied().ok_or_else(|| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "moss-transcribe-diarize generated token history is unexpectedly empty"
                    .to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "moss-transcribe-diarize decode cache position underflowed".to_string(),
            })?;
        if let Some(token_id) = self
            .decoder
            .decode_step_reused_top1(
                last_token,
                cache_position,
                &self.layer_kv_caches,
                self.kv_capacity,
            )
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?
        {
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(token_id),
            });
        }
        let logits = self
            .decoder
            .decode_step(
                last_token,
                cache_position,
                &mut self.layer_kv_caches,
                self.kv_capacity,
            )
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

/// Upstream `_compute_audio_token_length`'s per-chunk audio-token count: how
/// many post-merge adaptor tokens one Whisper-encoder chunk of `chunk_samples`
/// raw 16kHz samples produces, given `token_stride` (`hop_length` * the
/// Whisper conv stem's 2x stride * the adaptor's merge size). Pure integer
/// arithmetic with no model-pack dependency -- factored out of the encode
/// loop below so the slice-planning math can be pinned by a weight-free unit
/// test (`moss_td_chunk_frame_math_tests`) that runs in every default
/// `cargo nextest run`, unlike the family's real end-to-end `golden_diff_*`
/// tests, which need a local real fp16 pack and stay `#[ignore]`d because
/// weight-bearing fixtures are not part of weight-free CI (the same artifact
/// policy every other builtin family's CI golden coverage works around -- see
/// e.g. firered-aed's weight-free frontend golden).
pub(crate) fn moss_td_chunk_token_length(chunk_samples: usize, token_stride: usize) -> usize {
    (chunk_samples - 1) / token_stride.max(1) + 1
}

/// This chunk's post-merge encoder frames actually kept: `token_length` audio
/// tokens each span `merge_size` pre-merge encoder frames, capped at the
/// encoder's `max_source_positions` (a full un-trimmed 30s chunk can never
/// legitimately need more than that many frames kept).
pub(crate) fn moss_td_chunk_keep_frames(
    token_length: usize,
    merge_size: usize,
    max_source_positions: usize,
) -> usize {
    (token_length * merge_size).min(max_source_positions)
}

/// Upstream's `time_merge` truncation: the total kept frames across every
/// chunk, rounded down to the nearest full `merge_size` group. In practice
/// every chunk's `moss_td_chunk_keep_frames` result is already a multiple of
/// `merge_size` (either `token_length * merge_size` directly, or the
/// `max_source_positions` cap, which is itself merge-size-aligned for every
/// real checkpoint), so summing them keeps the running total aligned too --
/// this is a no-op guard against that invariant, not a silent frame drop.
pub(crate) fn moss_td_aligned_frame_count(total_frames: usize, merge_size: usize) -> usize {
    let merge_size = merge_size.max(1);
    (total_frames / merge_size) * merge_size
}

/// Derive this request's decode budget: audio-proportional, clamped by both
/// the checkpoint's 4096-token runaway backstop and whatever decoder context
/// this request's own prompt left unused.
///
/// The context clamp is what makes the generous rate above safe to state. The
/// KV cache is allocated for exactly `prompt + budget` and the executor
/// rejects a request whose total does not fit, so an allowance the context
/// cannot serve is not a bigger budget -- it is a refused request. Clamping
/// here makes the budget "as much as this context can still serve, up to the
/// backstop": the largest honest answer available, and never a promise the
/// cache cannot keep.
fn moss_td_generated_token_budget(
    sample_count: usize,
    prompt_tokens: usize,
    kv_capacity: usize,
) -> Result<usize, MossTdExecutorError> {
    moss_td_budget_for_shape(sample_count, SAMPLE_RATE_HZ, prompt_tokens, kv_capacity).map_err(
        |error| MossTdExecutorError::DecodeBudgetUnavailable {
            reason: error.to_string(),
        },
    )
}

/// Weight-free, always-on coverage for the executor's chunk/slice-planning
/// arithmetic: pure integer math with no model pack involved, so (unlike the
/// family's `golden_diff_*` end-to-end tests below, which need a local real
/// fp16 pack and stay `#[ignore]`d outside weight-free CI) these run in every
/// default `cargo nextest run --workspace`. Constants are pinned against the real
/// checkpoint's shape (`runtime_contract::tests::parses_adaptor_metadata_matching_real_checkpoint`'s
/// `merge_size == 4`, `package_import`'s `audio_merge_size: 4`, and
/// `parses_encoder_metadata_matching_real_checkpoint`'s
/// `max_source_positions == 1500` -- the standard Whisper-Medium 30s ->
/// 1500-frame shape).
#[cfg(test)]
mod moss_td_chunk_frame_math_tests {
    use super::*;

    const MERGE_SIZE: usize = 4;
    const MAX_SOURCE_POSITIONS: usize = 1500;
    const TOKEN_STRIDE: usize = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * MERGE_SIZE;

    fn budget_for(window_seconds: f32) -> usize {
        let samples = (window_seconds * SAMPLE_RATE_HZ as f32) as usize;
        // These tests exercise the audio-proportional rule itself. Exact
        // prompt+budget coverage is tested by the family topology module.
        moss_td_generated_token_budget(
            samples,
            0,
            crate::models::moss_transcribe_diarize::runtime_contract::MOSS_TD_MAX_KV_CACHE_POSITIONS,
        )
        .expect("budget")
    }

    /// The whole point of the declared slice window: at the family's maximum
    /// slice length, the audio prompt plus this call's generation budget must
    /// still fit inside the decoder's KV context, or the executor fails the
    /// request closed instead of decoding it. Pins the arithmetic that ties
    /// `OpenAsrLongformSliceShape::ScopedSlices` on the moss architecture
    /// descriptor to the budget rule -- the two are a pair, and widening the
    /// window alone silently eats the headroom.
    #[test]
    fn product_envelope_is_30s_target_and_60s_maximum() {
        let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds,
            target_seconds,
            max_seconds,
        } = crate::arch::longform_slice_shape_for_model_architecture(
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
        )
        else {
            panic!("moss-transcribe-diarize must declare ScopedSlices");
        };
        assert_eq!(
            target_seconds,
            crate::arch::MOSS_TD_TARGET_INVOCATION_SECONDS as f32
        );
        assert_eq!(
            max_seconds,
            crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as f32
        );
        assert_eq!(integral_seconds, max_seconds);
        assert_eq!(budget_for(30.0), 818);
        assert_eq!(budget_for(60.0), 1_508);
    }

    /// A short clip keeps a small, audio-proportional budget: reserving the
    /// full backstop for ten seconds of speech would size its persistent Metal
    /// reuse graph for a transcript that cannot exist.
    #[test]
    fn a_short_clip_keeps_a_small_proportional_budget() {
        let budget = budget_for(11.0);
        assert!(
            budget < MOSS_TD_MAX_GENERATED_TOKENS / 4,
            "an 11s clip must not reserve the runaway backstop, got {budget}"
        );
        assert!(budget >= MOSS_TD_MIN_GENERATED_TOKENS);
    }

    /// The budget never outruns the context: a prompt that has already eaten
    /// most of the decoder leaves only what is left, so the executor's
    /// fail-closed capacity check cannot be handed an impossible request.
    #[test]
    fn the_budget_never_exceeds_the_context_the_prompt_left() {
        let kv_capacity = 4_096;
        let prompt_tokens = 3_900;
        let budget =
            moss_td_generated_token_budget(600 * SAMPLE_RATE_HZ, prompt_tokens, kv_capacity)
                .expect("budget");
        assert!(
            prompt_tokens + budget <= kv_capacity,
            "budget {budget} on top of prompt {prompt_tokens} overruns capacity {kv_capacity}"
        );
    }

    #[test]
    fn token_stride_matches_the_real_checkpoints_merge_size() {
        assert_eq!(TOKEN_STRIDE, 1_280);
    }

    #[test]
    fn short_clip_single_partial_chunk_keeps_the_expected_frame_count() {
        // A ~10s clip (jfk.wav-shaped): one partial 30s chunk, well under
        // `CHUNK_SAMPLES`, never hits the `max_source_positions` cap.
        let chunk_samples = 160_000; // 10s @ 16kHz
        let token_length = moss_td_chunk_token_length(chunk_samples, TOKEN_STRIDE);
        assert_eq!(token_length, 125);
        let keep_frames = moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        assert_eq!(keep_frames, 500);
    }

    #[test]
    fn full_chunk_saturates_max_source_positions_without_truncating() {
        // A full un-trimmed 30s chunk (`CHUNK_SAMPLES`) always keeps exactly
        // `max_source_positions` frames -- the encoder always outputs that
        // many for a full chunk, so the `.min()` cap lands exactly on it
        // rather than truncating away real content.
        let token_length = moss_td_chunk_token_length(CHUNK_SAMPLES, TOKEN_STRIDE);
        assert_eq!(token_length, 375);
        let keep_frames = moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        assert_eq!(keep_frames, MAX_SOURCE_POSITIONS);
    }

    #[test]
    fn multi_chunk_long_file_sums_every_chunks_kept_frames() {
        // A ~76s file (longform-shaped, like the other builtin families'
        // committed `fixtures/longform_en_zh.wav` golden): splits into three
        // `CHUNK_SAMPLES`-bounded chunks -- two full 30s chunks plus a ~16s
        // tail -- exercising the same multi-chunk accumulation the
        // executor's real encode loop runs across every chunk of a longform
        // request, all the way through the final merge-size-alignment
        // truncation, without needing a real pack/weights.
        let chunk_lens = [CHUNK_SAMPLES, CHUNK_SAMPLES, 256_000];
        let mut total_frames = 0usize;
        for &chunk_samples in &chunk_lens {
            let token_length = moss_td_chunk_token_length(chunk_samples, TOKEN_STRIDE);
            total_frames +=
                moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        }
        assert_eq!(total_frames, 1_500 + 1_500 + 800);
        // Every chunk's kept-frame count is already a multiple of
        // `MERGE_SIZE`, so the running total across all three chunks stays
        // aligned and the final truncation is a no-op (see
        // `moss_td_aligned_frame_count`'s doc comment).
        assert_eq!(
            moss_td_aligned_frame_count(total_frames, MERGE_SIZE),
            total_frames
        );
    }

    #[test]
    fn aligned_frame_count_truncates_a_synthetic_misaligned_total() {
        // Real per-chunk totals are always already merge-size-aligned (see
        // the test above), so this never fires in production -- but the
        // truncation function itself must still behave correctly as
        // defense-in-depth if that invariant is ever violated by a future
        // change.
        assert_eq!(moss_td_aligned_frame_count(3_803, MERGE_SIZE), 3_800);
        assert_eq!(moss_td_aligned_frame_count(3_800, MERGE_SIZE), 3_800);
    }

    #[test]
    fn decode_budget_scales_to_the_real_moss_golden_lengths() {
        // The reference goldens emit 71 tokens for JFK (11s), 76 for
        // the mixed clip (13s), and 920 for the three-minute AISHELL-4 clip.
        // The two short clips stay on the proportional floor (no fixed
        // 4096-token Metal reuse-graph reservation for a few seconds of
        // speech); the historical three-minute direct-call fixture still
        // demonstrates the independent runaway backstop. Product slicing now
        // keeps legal invocations at or below 60s.
        assert_eq!(budget_for(11.0), 381);
        assert_eq!(budget_for(13.0), 427);
        assert_eq!(budget_for(180.0), MOSS_TD_MAX_GENERATED_TOKENS);
        // Every one of them still clears the golden's real token count with
        // room to spare.
        for (window_seconds, golden_tokens) in [(11.0_f32, 71), (13.0, 76), (180.0, 920)] {
            assert!(budget_for(window_seconds) > golden_tokens);
        }
    }
}

/// Encodes and adapts every 30s chunk of `samples` against the checked-out
/// resident encoder runtime for this pack+backend. Each chunk's valid encoder
/// prefix is time-merged inside the same ggml graph, so only final adaptor rows
/// cross back to the host.
fn encode_moss_td_chunks_with_runtime(
    runtime: &mut MossEncoderRuntime,
    encoder_config: MossEncoderConfig,
    merge_size: usize,
    adaptor_input_dim: usize,
    llm_dim: usize,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), MossTdExecutorError> {
    // Upstream `_compute_audio_token_length`'s stride: hop_length * the
    // Whisper conv stem's 2x stride * audio_merge_size.
    let token_stride = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * merge_size;
    let mut concatenated_rows: Vec<f32> = Vec::new();
    let mut total_tokens = 0usize;
    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let mel = whisper_log_mel_spectrogram_16khz_mono_v0(
            chunk,
            encoder_config.n_mels,
            MEL_TARGET_FRAMES,
        )
        .map_err(|error| MossTdExecutorError::FrontendFailed {
            reason: error.to_string(),
        })?;
        let token_length = moss_td_chunk_token_length(chunk.len(), token_stride);
        let keep_frames = moss_td_chunk_keep_frames(
            token_length,
            merge_size,
            encoder_config.max_source_positions,
        );
        let adaptor_rows = runtime
            .encode_and_adapt(
                encoder_config,
                mel.data(),
                MEL_TARGET_FRAMES,
                keep_frames,
                merge_size,
                adaptor_input_dim,
                llm_dim,
                MOSS_TD_ADAPTOR_NORM_EPSILON,
            )
            .map_err(|error| MossTdExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;
        let expected_values = token_length.checked_mul(llm_dim).ok_or_else(|| {
            MossTdExecutorError::EncoderFailed {
                reason: "adaptor output length overflowed".to_string(),
            }
        })?;
        if adaptor_rows.len() != expected_values {
            return Err(MossTdExecutorError::EncoderFailed {
                reason: format!(
                    "adaptor output length {} != expected {expected_values}",
                    adaptor_rows.len()
                ),
            });
        }
        concatenated_rows.extend_from_slice(&adaptor_rows);
        total_tokens = total_tokens.checked_add(token_length).ok_or_else(|| {
            MossTdExecutorError::EncoderFailed {
                reason: "adaptor token count overflowed".to_string(),
            }
        })?;
    }
    Ok((concatenated_rows, total_tokens))
}

fn encode_moss_td_chunks_with_cached_runtime(
    actor: &MossTdEncoderRuntimeActor,
    encoder_config: MossEncoderConfig,
    merge_size: usize,
    adaptor_input_dim: usize,
    llm_dim: usize,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), MossTdExecutorError> {
    let samples = samples.to_vec();
    actor
        .call_mut_fallible(move |state| {
            let encode_result = encode_moss_td_chunks_with_runtime(
                &mut state.runtime,
                encoder_config,
                merge_size,
                adaptor_input_dim,
                llm_dim,
                &samples,
            );
            let release_result =
                state
                    .runtime
                    .release_transient_compute_memory()
                    .map_err(|error| MossTdExecutorError::EncoderFailed {
                        reason: error.to_string(),
                    });
            match (encode_result, release_result) {
                (Ok(output), Ok(())) => Ok(output),
                (Err(error), _) => Err(error),
                (Ok(_), Err(error)) => Err(error),
            }
        })
        .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
            stage: "encoder",
            reason: error.to_string(),
        })?
}

/// One decode's text plus how the shared driver ended it. The stop reason is
/// what keeps `speaker_segments` from closing a cut-short decode's final
/// segment at the end of the clip (see [`MossTdDecodeExtent`]) and what the
/// executor lifts into the transcript's truncation signal.
struct MossTdDecodeOutput {
    text: String,
    stop_reason: Seq2SeqGreedyDecodeStopReason,
}

/// Runs the ChatML+audio-splice prompt embedding through the cached, resident
/// decoder runtime for this pack+backend: prefill, then the shared greedy
/// decode driver through to `<|im_end|>` (or the fail-closed token budget),
/// returning the trimmed decode text. Mirrors `firered_aed::executor`'s
/// `decode_with_cached_runtime`: the runtime (loaded weights + the Qwen
/// decode graph's reuse machinery) stays resident across calls. Every
/// per-utterance host KV cache is allocated fresh at the exact logical span;
/// the device arena/graph remains at the stable session-envelope reserve and
/// is logically reset by the next prefill. `release_session_scoped_buffers`
/// below drops poisoned resident state and CPU-only scratch before reuse.
#[allow(clippy::too_many_arguments)]
fn run_moss_td_decoder_with_runtime(
    decoder: &mut MossTdDecoderRuntime,
    decoder_metadata: MossTdDecoderMetadata,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    max_generated_tokens: usize,
    decode_prompt_token_ids: &[u32],
    audio_pad_positions: &[usize],
    audio_rows: &[f32],
    tokenizer: MossTdTokenizer,
    control: Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
) -> Result<MossTdDecodeOutput, MossTdExecutorError> {
    if std::env::var_os("OPENASR_MOSS_TD_PROFILE").is_some() {
        eprintln!(
            "OPENASR_MOSS_TD_PROFILE decoder_backend={} decoder_reuse_supported={} native_gqa={}",
            decoder.backend_label(),
            decoder.supports_graph_reuse(),
            decoder.uses_native_gqa()
        );
    }

    let prompt_input = Qwen3AsrPromptTokenInput {
        token_ids: decode_prompt_token_ids.to_vec(),
        audio_rows: audio_rows.to_vec(),
        audio_positions: audio_pad_positions.to_vec(),
    };

    let layer_kv_caches = decoder
        .new_kv_caches(kv_capacity)
        .map_err(|reason| MossTdExecutorError::DecoderFailed { reason })?;
    let mut step_executor = MossTdGreedyStepExecutor {
        decoder,
        layer_kv_caches,
        kv_capacity,
        prompt_input: Some(prompt_input),
        cache_prompt_tokens: 0,
        control: Arc::clone(&control),
    };
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: decode_prompt_token_ids.to_vec(),
        eot_token_id: tokenizer.im_end_token_id,
        vocab_size: decoder_metadata.vocab_size,
        max_generated_tokens,
    };
    let result = run_builtin_seq2seq_decode_policy(
        crate::arch::MOSS_TD_DECODE_POLICY_ID,
        &config,
        &tokenizer,
        None,
        &mut step_executor,
        &|token_ids: &[u32]| {
            tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                    reason: error.to_string(),
                }
            })
        },
        |error: Seq2SeqGreedyDecodeError| error,
        |error: Seq2SeqGreedyDecodeError| error,
        map_registry_error,
        &control,
        decode_work_progress.as_ref(),
    );
    // Release this request's per-token grow-to-fit host buffer before the
    // runtime goes back into the cache -- unconditionally, on both success
    // and failure, so a failed decode never leaves session-scoped memory in
    // the cached runtime.
    step_executor.decoder.release_session_scoped_buffers();
    let result = match result {
        Ok(result) => result,
        // Budget exhausted before `<|im_end|>`: preserve the generated prefix
        // and mark it truncated instead of discarding useful transcription.
        Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens, ..
        }) => {
            let text = tokenizer
                .decode_text_token_ids(&generated_tokens)
                .map_err(|error| MossTdExecutorError::GreedyDecodeFailed {
                    reason: format!(
                        "tokenizer decode of the budget-exhausted prefix failed: {error}"
                    ),
                })?;
            Seq2SeqGreedyDecodeResult {
                text,
                generated_tokens,
                generated_probabilities: Vec::new(),
                stop_reason: Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
            }
        }
        Err(error) => {
            return Err(MossTdExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            });
        }
    };
    Ok(MossTdDecodeOutput {
        text: result.text.trim().to_string(),
        stop_reason: result.stop_reason,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_moss_td_decoder_with_cached_runtime(
    actor: &MossTdDecoderRuntimeActor,
    decoder_metadata: MossTdDecoderMetadata,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    max_generated_tokens: usize,
    decode_prompt_token_ids: &[u32],
    audio_pad_positions: &[usize],
    audio_rows: &[f32],
    tokenizer: MossTdTokenizer,
    control: Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
) -> Result<MossTdDecodeOutput, MossTdExecutorError> {
    let decode_prompt_token_ids = decode_prompt_token_ids.to_vec();
    let audio_pad_positions = audio_pad_positions.to_vec();
    let audio_rows = audio_rows.to_vec();
    actor
        .call_mut_fallible(move |state| {
            run_moss_td_decoder_with_runtime(
                &mut state.runtime,
                decoder_metadata,
                kv_capacity,
                max_generated_tokens,
                &decode_prompt_token_ids,
                &audio_pad_positions,
                &audio_rows,
                tokenizer,
                control,
                decode_work_progress,
            )
        })
        .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
            stage: "decoder",
            reason: error.to_string(),
        })?
}

impl MossTdGgmlExecutor {
    fn map_actor_error(stage: &'static str, error: PinnedRuntimeActorError) -> MossTdExecutorError {
        MossTdExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn prepared_runtime_for_preflight(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<PreparedRuntimeHandle<MossTdPreparedRuntime>, MossTdExecutorError> {
        self.prepared_runtimes.get_or_try_insert_with(
            &preflight.runtime_source,
            PreparedRuntimeQuoteContext {
                model_architecture: crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
                metadata: &preflight.metadata,
                tensor_index: &preflight.tensor_index,
                backend,
            },
            || {
                build_moss_td_prepared_runtime(preflight, backend).map_err(
                    |error: MossTdPreparedRuntimeError| {
                        MossTdExecutorError::PreparedRuntimeFailed {
                            reason: error.to_string(),
                        }
                    },
                )
            },
            || MossTdExecutorError::PreparedRuntimeFailed {
                reason: "prepared-runtime cache lock poisoned".to_string(),
            },
            |error| MossTdExecutorError::RuntimeOwnershipFailed {
                stage: "prepared",
                reason: error.to_string(),
            },
        )
    }

    fn checkout_encoder_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MossTdPreparedRuntime>,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<MossTdEncoderRuntimeActor, MossTdExecutorError> {
        let graph_config = moss_td_encoder_graph_config(backend);
        let encoder_backend = graph_config.backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(encoder_backend),
            graph_config.into(),
        );
        let preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let host_adaptor = if encoder_backend
                    == crate::ggml_runtime::GgmlCpuGraphBackend::Cpu
                {
                    let reader =
                        crate::ggml_runtime::build_runtime_tensor_reader_from_preflight(&preflight)
                            .map_err(|error| MossTdExecutorError::EncoderFailed {
                                reason: format!(
                                    "could not open CPU adaptor tensor reader: {error}"
                                ),
                            })?;
                    let quote = MossAdaptorWeights::system_memory_quote(
                        &preflight.tensor_index,
                        &content_id,
                    )
                    .map_err(|error| {
                        MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "encoder",
                            reason: error.to_string(),
                        }
                    })?;
                    Some((reader, quote))
                } else {
                    None
                };
                let retained = host_adaptor
                    .as_ref()
                    .map_or(0, |(_, quote)| quote.retained_bytes);
                Ok((retained, (preflight, prepared, host_adaptor)))
            },
            move |(preflight, prepared, host_adaptor)| {
                let build_runtime = |host_adaptor_reader| {
                    let runtime = MossEncoderRuntime::new_with_prepared_weights_from_preflight(
                        &preflight,
                        Arc::clone(&prepared.encoder_weights),
                        host_adaptor_reader,
                        prepared.encoder_metadata.d_model,
                        prepared.adaptor_metadata.merge_size,
                        prepared.decoder_metadata.d_model,
                        MOSS_TD_ADAPTOR_NORM_EPSILON,
                        graph_config,
                    )
                    .map_err(|error| MossTdExecutorError::EncoderFailed {
                        reason: format!("could not initialize encoder runtime: {error}"),
                    })?;
                    Ok(MossTdEncoderActorState {
                        runtime,
                        _prepared_owner: prepared,
                    })
                };

                let Some((reader, quote)) = host_adaptor else {
                    return build_runtime(None).map(SystemMemoryOwner::without_allocation);
                };
                match SystemMemoryOwner::try_allocate_transaction(quote, || {
                    let state = build_runtime(Some(&reader))?;
                    let retained =
                        state
                            .runtime
                            .retained_host_system_memory_bytes()
                            .map_err(|reason| MossTdExecutorError::RuntimeOwnershipFailed {
                                stage: "encoder",
                                reason,
                            })?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        state, retained, retained,
                    ))
                }) {
                    Ok(owner) => Ok(owner),
                    Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        Err(MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "encoder",
                            reason: error.to_string(),
                        })
                    }
                }
            },
            |error| Self::map_actor_error("encoder", error),
        )
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MossTdPreparedRuntime>,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        native_gqa: GgmlNativeGqaCapability,
    ) -> Result<MossTdDecoderRuntimeActor, MossTdExecutorError> {
        let graph_config = moss_td_runtime_graph_config(backend);
        let effective_backend = graph_config.backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(effective_backend),
            graph_config.into(),
            native_gqa,
        );
        let preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let retained = MossTdDecoderRuntime::quoted_resident_system_memory_bytes(
                    &prepared.decoder_plan,
                )
                .map_err(|reason| MossTdExecutorError::RuntimeOwnershipFailed {
                    stage: "decoder",
                    reason,
                })?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("moss-td-decoder-runtime:{content_id}"),
                    retained,
                    retained,
                )
                .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
                    stage: "decoder",
                    reason: error.to_string(),
                })?;
                Ok((retained, (preflight, prepared, quote)))
            },
            move |(preflight, prepared, quote)| match SystemMemoryOwner::try_allocate_transaction(
                quote,
                || {
                    let runtime = MossTdDecoderRuntime::new_with_prepared_state_from_preflight(
                        &preflight,
                        prepared.decoder_metadata,
                        Arc::clone(&prepared.decoder_plan),
                        Arc::clone(&prepared.logits_head),
                        Arc::clone(&prepared.token_embedding),
                        graph_config,
                        native_gqa,
                    )
                    .map_err(|error| MossTdExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    })?;
                    let retained = runtime.resident_system_memory_bytes().map_err(|reason| {
                        MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason,
                        }
                    })?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        MossTdDecoderActorState {
                            runtime,
                            _prepared_owner: prepared,
                        },
                        retained,
                        retained,
                    ))
                },
            ) {
                Ok(owner) => Ok(owner),
                Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(MossTdExecutorError::RuntimeOwnershipFailed {
                        stage: "decoder",
                        reason: error.to_string(),
                    })
                }
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn checkout_unified_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MossTdPreparedRuntime>,
        backend: GgmlCpuGraphBackend,
        encoder_config: GgmlCpuGraphConfig,
        decoder_config: GgmlCpuGraphConfig,
        native_gqa: GgmlNativeGqaCapability,
    ) -> Result<MossTdUnifiedRuntimeActor, MossTdExecutorError> {
        let key = MossTdUnifiedRuntimeCacheKey {
            content: PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane: current_execution_lane_key(backend),
            encoder_profile: encoder_config.into(),
            decoder_profile: decoder_config.into(),
            native_gqa,
        };
        let preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.unified_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let quote = MossTdUnifiedActorState::system_memory_quote(
                    &prepared,
                    &content_id,
                    encoder_config,
                    decoder_config,
                )
                .map_err(|reason| MossTdExecutorError::RuntimeOwnershipFailed {
                    stage: "unified-runtime",
                    reason,
                })?;
                Ok((quote.retained_bytes, (preflight, prepared, quote)))
            },
            move |(preflight, prepared, quote)| {
                match SystemMemoryOwner::try_allocate_transaction(quote, || {
                    // Build both graph stages on this owner thread while the
                    // encoder's pack binding is live. The shared loaded-weight
                    // cache can then upgrade the same Rc for the decoder rather
                    // than upload the complete pack a second time.
                    let encoder = MossEncoderRuntime::new_with_prepared_weights_from_preflight(
                        &preflight,
                        Arc::clone(&prepared.encoder_weights),
                        None,
                        prepared.encoder_metadata.d_model,
                        prepared.adaptor_metadata.merge_size,
                        prepared.decoder_metadata.d_model,
                        MOSS_TD_ADAPTOR_NORM_EPSILON,
                        encoder_config,
                    )
                    .map_err(|error| MossTdExecutorError::EncoderFailed {
                        reason: format!("could not initialize unified encoder runtime: {error}"),
                    })?;
                    let decoder = MossTdDecoderRuntime::new_with_prepared_state_from_preflight(
                        &preflight,
                        prepared.decoder_metadata,
                        Arc::clone(&prepared.decoder_plan),
                        Arc::clone(&prepared.logits_head),
                        Arc::clone(&prepared.token_embedding),
                        decoder_config,
                        native_gqa,
                    )
                    .map_err(|error| MossTdExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    })?;
                    let expected_lane = (GgmlCpuGraphBackend::Gpu, false);
                    let decoder_lanes = decoder.graph_lanes();
                    if encoder.graph_lane() != expected_lane
                        || decoder_lanes.0 != expected_lane
                        || decoder_lanes.1 != Some(expected_lane)
                    {
                        return Err(MossTdExecutorError::RuntimeContractViolation {
                            reason: "unified MOSS runtime requires direct GPU encoder, decoder, and logits lanes"
                                .to_string(),
                        });
                    }
                    if decoder.loaded_weight_binding_identity()
                        != Some(encoder.loaded_weight_binding_identity())
                    {
                        return Err(MossTdExecutorError::RuntimeContractViolation {
                            reason: "unified MOSS runtime did not coalesce its pack-wide weight binding"
                                .to_string(),
                        });
                    }
                    let state = MossTdUnifiedActorState {
                        encoder,
                        decoder,
                        _prepared_owner: prepared,
                    };
                    let retained = state.retained_system_memory_bytes().map_err(|reason| {
                        MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "unified-runtime",
                            reason,
                        }
                    })?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        state, retained, retained,
                    ))
                }) {
                    Ok(owner) => Ok(owner),
                    Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        Err(MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "unified-runtime",
                            reason: error.to_string(),
                        })
                    }
                }
            },
            |error| Self::map_actor_error("unified-runtime", error),
        )
    }

    fn clear_runtime_owners(&self) {
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.unified_runtimes.clear();
        self.prepared_runtimes.clear();
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.unified_runtimes
            .evict_where(|key| key.content.pack_content_id == pack_content_id);
        self.prepared_runtimes.evict_content_id(pack_content_id);
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, MossTdExecutorError> {
        self.execute_inner_with_runtime_mode(request, true)
    }

    fn execute_inner_with_runtime_mode(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        allow_unified_runtime: bool,
    ) -> Result<GgmlAsrExecutionResult, MossTdExecutorError> {
        let expected_adapter = crate::arch::MOSS_TD_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(MossTdExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request.runtime_source_preflight();

        let backend = request.resolved_runtime.backend();
        let prepared = self.prepared_runtime_for_preflight(preflight, backend)?;
        let encoder_metadata = prepared.encoder_metadata;
        let adaptor_metadata = prepared.adaptor_metadata;
        let decoder_metadata = prepared.decoder_metadata;
        let tokenizer = prepared.tokenizer.clone();
        let backend_preference = request_backend_override();
        let native_gqa = qwen_llm_effective_native_gqa_capability(moss_td_native_gqa_candidate(
            backend,
            backend_preference.as_ref(),
            request.resolved_runtime.native_gqa_capability(),
        ));
        let unified_configs = moss_td_unified_runtime_configs(
            allow_unified_runtime,
            backend,
            backend_preference.as_ref(),
            current_execution_placement(),
        );
        let unified_runtime = unified_configs
            .map(|(encoder_config, decoder_config)| {
                self.checkout_unified_runtime(
                    preflight,
                    Arc::clone(&prepared),
                    backend,
                    encoder_config,
                    decoder_config,
                    native_gqa,
                )
            })
            .transpose()?;

        let samples = &request.prepared_audio.samples_f32;
        if samples.is_empty() {
            return Err(MossTdExecutorError::EmptyAudio);
        }
        let max_samples = SAMPLE_RATE_HZ
            .checked_mul(crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as usize)
            .ok_or_else(|| MossTdExecutorError::RuntimeContractViolation {
                reason: "MOSS-TD product invocation sample ceiling overflowed".to_string(),
            })?;
        if samples.len() > max_samples {
            return Err(MossTdExecutorError::InvocationTooLong {
                samples: samples.len(),
                max_samples,
            });
        }
        // Derived from THIS call's buffer, never from a request-level "whole
        // recording" duration. Under longform slicing that buffer is one slice,
        // and this value is what `speaker_segments` clamps a truncated decode's
        // final segment to -- so a slice that ends without a stop token can only
        // ever blanket the rest of its own slice, not the rest of the recording.
        let audio_duration_seconds = samples.len() as f32 / SAMPLE_RATE_HZ as f32;

        let encoder_config = MossEncoderConfig {
            n_layers: encoder_metadata.n_layers,
            d_model: encoder_metadata.d_model,
            n_heads: encoder_metadata.n_heads,
            n_mels: encoder_metadata.n_mels,
            max_source_positions: encoder_metadata.max_source_positions,
        };
        let (audio_rows, audio_token_count) = match unified_runtime.as_ref() {
            Some(actor) => {
                let samples = samples.to_vec();
                actor
                    .call_mut_fallible(move |state| {
                        let encode_result = encode_moss_td_chunks_with_runtime(
                            &mut state.encoder,
                            encoder_config,
                            adaptor_metadata.merge_size,
                            adaptor_metadata.input_dim,
                            decoder_metadata.d_model,
                            &samples,
                        );
                        let release_result = state
                            .encoder
                            .release_transient_compute_memory()
                            .map_err(|error| MossTdExecutorError::EncoderFailed {
                                reason: error.to_string(),
                            });
                        match (encode_result, release_result) {
                            (Ok(output), Ok(())) => Ok(output),
                            (Err(error), _) => Err(error),
                            (Ok(_), Err(error)) => Err(error),
                        }
                    })
                    .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
                        stage: "unified-encoder",
                        reason: error.to_string(),
                    })??
            }
            None => {
                let encoder_actor =
                    self.checkout_encoder_runtime(preflight, Arc::clone(&prepared), backend)?;
                encode_moss_td_chunks_with_cached_runtime(
                    &encoder_actor,
                    encoder_config,
                    adaptor_metadata.merge_size,
                    adaptor_metadata.input_dim,
                    decoder_metadata.d_model,
                    samples,
                )?
            }
        };

        let decode_prompt =
            build_moss_td_decode_prompt(&tokenizer, audio_token_count).map_err(|error| {
                MossTdExecutorError::DecodePromptFailed {
                    reason: error.to_string(),
                }
            })?;

        // Fail closed up front when this call's prompt plus the configured
        // decode budget cannot fit the decoder's KV context. The shared native
        // slicer keeps ordinary requests well inside it (the family declares
        // its own slice window via `OpenAsrLongformSliceShape::ScopedSlices`),
        // so this is the backstop for a caller that bypasses longform slicing
        // entirely. The request-sized cache must reserve every possible decode
        // position; clamping an over-limit request would defer the failure to a
        // cryptic KV write mid-generation.
        let kv_capacity = moss_td_kv_cache_positions(decoder_metadata.max_positions);
        // Sized once the prompt is known, so the budget can claim the decoder
        // context the prompt did not need (see `moss_td_generated_token_budget`).
        let max_generated_tokens = moss_td_generated_token_budget(
            samples.len(),
            decode_prompt.token_ids.len(),
            kv_capacity,
        )?;
        let semantic_required_positions = decode_prompt
            .token_ids
            .len()
            .checked_add(max_generated_tokens)
            .ok_or_else(|| MossTdExecutorError::DecodeBudgetUnavailable {
                reason: "prompt plus generation position count overflowed".to_string(),
            })?;
        let request_kv_cache_positions = moss_td_request_kv_cache_positions(
            decoder_metadata.max_positions,
            decode_prompt.token_ids.len(),
            max_generated_tokens,
        )
        .ok_or_else(|| MossTdExecutorError::AudioExceedsContext {
            prompt_tokens: decode_prompt.token_ids.len(),
            generation_budget: max_generated_tokens,
            required_positions: semantic_required_positions,
            kv_capacity,
            max_minutes: (kv_capacity.saturating_sub(max_generated_tokens) as f32
                / AUDIO_TOKENS_PER_SECOND_FOR_LIMIT
                / 60.0)
                .max(0.0),
        })?;
        let kv_capacity_plan = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::MOSS_TD_SELF_KV_STATE_ID,
        )
        .and_then(|capacity| {
            capacity.validate_measured_logical_positions(request_kv_cache_positions)
        })
        .map_err(|source| MossTdExecutorError::DecoderStateCapacity { source })?;
        if kv_capacity_plan.resident_positions() > kv_capacity {
            return Err(MossTdExecutorError::RuntimeContractViolation {
                reason: format!(
                    "planned resident KV span {} exceeds the validated MOSS cap {kv_capacity}",
                    kv_capacity_plan.resident_positions()
                ),
            });
        }

        let control = Arc::clone(&request.execution_context.control);
        let decode_work_progress = request
            .execution_context
            .decode_work_progress_observer()
            .cloned();
        let decoded = match unified_runtime.as_ref() {
            Some(actor) => {
                let token_ids = decode_prompt.token_ids;
                let audio_positions = decode_prompt.audio_pad_positions;
                actor
                    .call_mut_fallible(move |state| {
                        run_moss_td_decoder_with_runtime(
                            &mut state.decoder,
                            decoder_metadata,
                            kv_capacity_plan,
                            max_generated_tokens,
                            &token_ids,
                            &audio_positions,
                            &audio_rows,
                            tokenizer,
                            control,
                            decode_work_progress,
                        )
                    })
                    .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
                        stage: "unified-decoder",
                        reason: error.to_string(),
                    })??
            }
            None => {
                let decoder_actor =
                    self.checkout_decoder_runtime(preflight, prepared, backend, native_gqa)?;
                run_moss_td_decoder_with_cached_runtime(
                    &decoder_actor,
                    decoder_metadata,
                    kv_capacity_plan,
                    max_generated_tokens,
                    &decode_prompt.token_ids,
                    &decode_prompt.audio_pad_positions,
                    &audio_rows,
                    tokenizer,
                    control,
                    decode_work_progress,
                )?
            }
        };
        // Normalize the model's own inline `[start][end][SNN]` markup into the
        // engine's shared segment representation. The decode prompt is fixed,
        // so the markers are written whether or not the request asked for
        // speakers: stripping them from the transcript is this layer's job, and
        // `in_decoder_speakers` decides only whether the recording-local
        // `SPEAKER_NN` labels survive. See `speaker_segments`'s module doc for
        // the grammar, the fail-closed policy, and the degrade shape.
        let normalized = super::speaker_segments::normalize_moss_td_decode(
            &decoded.text,
            MossTdDecodeExtent {
                audio_duration_seconds,
                truncated: decoded.stop_reason.is_truncated(),
            },
            request.request_options.in_decoder_speakers,
        );
        // moss-td is the one family with decoder-emitted timestamps, so it can
        // name the point the transcript stops describing the audio instead of
        // only reporting that it does.
        let decode_truncation = decoded
            .stop_reason
            .into_decode_truncation(normalized.truncated_at_seconds);
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            segments: normalized.segments,
            text: normalized.text,
            longform: None,
            language: None,
            ..Default::default()
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
            decode_truncation,
        })
    }
}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

impl GgmlAsrViewExecutor for MossTdGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        MossTdGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        MOSS_TD_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_moss_td_decoder_state,
                super::capacity::MOSS_TD_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_owners();
    }
}

impl MossTdGgmlExecutor {
    fn execute_streaming_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner_with_runtime_mode(request, false)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: MOSS_TD_STREAMING_EXECUTOR_ID,
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

/// Not a true incremental streaming session -- this family's architecture
/// needs the full audio up front to place its numeric time-anchor markers
/// (see `decode_prompt`'s module doc), so there is no meaningful "partial"
/// mode yet (matches the top-of-file doc's "file-transcribe only" note).
/// Still registers a buffered snapshot-streaming session (mirrors
/// `firered_llm`'s identical precedent: a family with no real partial path
/// still needs SOME streaming executor, or the builtin dispatch's
/// fail-fast completeness gate rejects the whole registry at startup) so a
/// live-caption request degrades to "one final result at end of audio"
/// instead of silently falling back to a broken cadence.
impl GgmlAsrStreamingExecutor for MossTdGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MOSS_TD_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            MOSS_TD_STREAMING_EXECUTOR_ID,
            crate::arch::MOSS_TD_GGML_ADAPTER_ID,
            "moss-transcribe-diarize",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            MossTdGgmlExecutor::execute_streaming_view,
        )
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_owners();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::arch::builtin_adapter_descriptor;
    use crate::device::execution_route::{
        DeviceAddressability, PhysicalResourceKey, ResolvedExecutionRoute, RouteDeviceKind,
    };
    use crate::ggml_runtime::install_request_backend_override;
    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudioView};

    use crate::api::backend::Segment;

    use super::super::speaker_segments::parse_moss_td_speaker_segments;
    use super::*;

    #[test]
    fn graph_runtime_cache_profile_separates_scheduler_and_threading() {
        let base = crate::ggml_runtime::GgmlCpuGraphConfig {
            backend: crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            use_scheduler: false,
            n_threads: Some(2),
            ..crate::ggml_runtime::GgmlCpuGraphConfig::conservative_default()
        };
        let direct: MossTdGraphRuntimeCacheProfile = base.into();
        let scheduled: MossTdGraphRuntimeCacheProfile = crate::ggml_runtime::GgmlCpuGraphConfig {
            use_scheduler: true,
            ..base
        }
        .into();
        let more_threads: MossTdGraphRuntimeCacheProfile =
            crate::ggml_runtime::GgmlCpuGraphConfig {
                n_threads: Some(4),
                ..base
            }
            .into();

        assert_ne!(direct, scheduled);
        assert_ne!(direct, more_threads);
    }

    fn exactly_addressable_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: RouteDeviceKind::Accelerated,
            addressability: DeviceAddressability::ExactlyAddressable {
                physical_key: PhysicalResourceKey::new("0000:01:00.0")
                    .expect("synthetic physical key"),
            },
        })
    }

    fn direct_gpu_config() -> GgmlCpuGraphConfig {
        GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Gpu,
            use_scheduler: false,
            ..GgmlCpuGraphConfig::conservative_default()
        }
    }

    #[test]
    fn native_gqa_candidate_is_vulkan_only_on_discrete_gpu() {
        let validated = GgmlNativeGqaCapability::Validated;
        for (provider, expected) in [
            (
                ExecutionProvider::Cuda,
                GgmlNativeGqaCapability::Unsupported,
            ),
            (ExecutionProvider::Vulkan, validated),
            (ExecutionProvider::Hip, GgmlNativeGqaCapability::Unsupported),
            (
                ExecutionProvider::Accelerator,
                GgmlNativeGqaCapability::Unsupported,
            ),
            (
                ExecutionProvider::Unknown,
                GgmlNativeGqaCapability::Unsupported,
            ),
        ] {
            let preference = exactly_addressable_preference(provider);
            assert_eq!(
                moss_td_native_gqa_candidate(
                    GgmlCpuGraphBackend::Gpu,
                    Some(&preference),
                    validated,
                ),
                expected,
            );
        }
        assert_eq!(
            moss_td_native_gqa_candidate(GgmlCpuGraphBackend::Cpu, None, validated),
            validated,
        );
        assert_eq!(
            moss_td_native_gqa_candidate(GgmlCpuGraphBackend::Metal, None, validated),
            validated,
        );
    }

    #[test]
    fn unified_runtime_is_ordinary_exact_direct_cuda_or_vulkan_full_device_only() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let preference = exactly_addressable_preference(provider);
            assert!(moss_td_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu_config(),
                direct_gpu_config(),
            ));
            assert!(!moss_td_unified_runtime_enabled(
                false,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu_config(),
                direct_gpu_config(),
            ));
            assert!(!moss_td_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::Hybrid),
                direct_gpu_config(),
                direct_gpu_config(),
            ));
            assert!(!moss_td_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                GgmlCpuGraphConfig {
                    use_scheduler: true,
                    ..direct_gpu_config()
                },
                direct_gpu_config(),
            ));
            assert!(!moss_td_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu_config(),
                GgmlCpuGraphConfig {
                    backend: GgmlCpuGraphBackend::Cpu,
                    ..direct_gpu_config()
                },
            ));
        }

        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Hip,
            ExecutionProvider::Accelerator,
            ExecutionProvider::Unknown,
        ] {
            let preference = exactly_addressable_preference(provider);
            assert!(!moss_td_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                direct_gpu_config(),
                direct_gpu_config(),
            ));
        }
    }

    #[test]
    fn unified_system_memory_shape_is_checked_and_phase_exact() {
        assert_eq!(moss_td_unified_system_memory_shape(17, 29), Ok((46, 46)));
        assert!(moss_td_unified_system_memory_shape(u64::MAX, 1).is_err());
    }

    /// Real converted local pack (fp16), not committed. It is a weight-bearing
    /// fixture supplied for opt-in tests, which remain outside weight-free CI.
    fn dev_pack_path() -> Option<PathBuf> {
        crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_PACK",
            "MOSS Transcribe Diarize .oasr pack",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}"))
        .ok()
    }

    fn dev_sample_path(name: &str) -> PathBuf {
        match crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_SAMPLES",
            "MOSS Transcribe Diarize sample directory",
        ) {
            Ok(path) => path.join(name),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                PathBuf::new()
            }
        }
    }

    // The `GOLDEN_*_TEXT` constants below are the raw *reference decode* -- the
    // tagged string the model itself produces, compared against the HF fp32
    // reference. They are deliberately NOT what the executor returns: the
    // family's inline markup is an internal transport for speaker structure and
    // is normalized away before anything else sees it (see
    // `speaker_segments`'s module doc), so the executor's flat text is the
    // markup-free projection of these, obtained through the same normalizer.
    // Keeping the goldens in reference form is what lets a decode regression
    // (different words, shifted anchors, a lost speaker change) still show up
    // here instead of being hidden by the stripping.
    //
    // Pinned to the real-pack CPU baseline decode (backend forced to CPU below).
    // The encoder binds its 2D projection weights zero-copy as native f16 and
    // runs flash attention (see `encoder_graph`), so this decode path is f16
    // weights + flash, NOT the f32-naive path -- do not assert flash == naive or
    // f16 == f32 bit-for-bit. What IS asserted, matching the reference-platform
    // golden policy: the transcript is text-level identical to the HF fp32
    // reference (`tmp/moss-td/golden/*.json`'s `text`), including speaker labels,
    // and every emitted time anchor is within 0.05s of it. In practice jfk and
    // the 3-minute aishell clip come out byte-for-byte equal to the HF golden
    // (time anchors included); en_zh_mixed matches the HF text exactly with two
    // anchors shifted by 0.02s ([2.34]->[2.32], [4.94]->[4.96]), the f16+flash
    // numeric delta.
    const GOLDEN_JFK_TEXT: &str = concat!(
        "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S01] ask not what your ",
        "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
    );

    // Code-switch coverage: `en_zh_mixed.wav` mixes English then Mandarin in a
    // single utterance, exercising both tokenizer/decode paths plus a second
    // speaker label (`[S02]`) in one prefill+decode. Text identical to the HF
    // golden `en_zh_mixed.json`'s `text`; two time anchors sit 0.02s off (see the
    // pinning note above).
    const GOLDEN_EN_ZH_MIXED_TEXT: &str = concat!(
        "[0.27][S01]And so, my fellow Americans,[2.32][3.21][S01]ask not.",
        "[4.44][4.96][S02]今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去伊加新[12.88]",
    );

    /// The flat transcript a caller receives for a given reference decode: the
    /// same words with the family's markup normalized away.
    fn normalized_golden_text(reference_decode: &str, audio_duration_seconds: f32) -> String {
        super::super::speaker_segments::normalize_moss_td_decode(
            reference_decode,
            MossTdDecodeExtent::complete(audio_duration_seconds),
            true,
        )
        .text
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<(String, std::time::Duration, f32)> {
        // Force CPU for a deterministic reference baseline. The family policy
        // is `AutoGpuPolicy::AllBackends`; the accelerated variants below cover
        // the explicit Metal path separately rather than mixing backends into
        // the CPU golden.
        transcribe_with_dev_pack_backend(wav_path, GgmlAsrBackendPreference::CpuOnly).map(
            |(text, _, elapsed, audio_duration_seconds)| (text, elapsed, audio_duration_seconds),
        )
    }

    /// Same real-pack e2e path as [`transcribe_with_dev_pack`], but lets the
    /// caller pick the backend preference -- used by the `_accelerated`
    /// variants below to drive an explicit `execution_target=accelerated`
    /// request end to end (encoder AND decode), the same override an
    /// `Accelerated` request installs in production (see
    /// `GgmlAsrBackendPreference::request_backend_override`'s doc and
    /// `graph_config.rs`'s note that an explicit request wins over Auto under
    /// `AutoGpuPolicy::AllBackends`).
    fn transcribe_with_dev_pack_backend(
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, Vec<Segment>, std::time::Duration, f32)> {
        let executor = MossTdGgmlExecutor::default();
        transcribe_with_dev_pack_backend_using_executor(&executor, wav_path, backend_preference)
    }

    fn plan_test_request_decoder_state(
        executor: &MossTdGgmlExecutor,
        request: &mut GgmlAsrExecutionViewRequest<'_>,
    ) {
        let decoder_state = {
            let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput::for_offline_view_request(
                request.runtime_source_preflight(),
                &request.prepared_audio,
                &request.request_options,
                request.resolved_runtime.backend(),
            )
            .expect("build MOSS-TD decoder-state planning input");
            executor
                .decoder_state_contract(&request.selected_family)
                .expect("load MOSS-TD decoder-state contract")
                .plan(&planning_input)
                .expect("plan MOSS-TD decoder state")
        };
        request.decoder_state = decoder_state;
    }

    /// Execute one real-pack request through the caller-provided executor.
    /// Keeping the executor outside this helper lets tests prove that the
    /// owner-actor pools retain and reuse runtimes across requests, while the
    /// ordinary fixture helpers below still get an isolated default executor.
    fn transcribe_with_dev_pack_backend_using_executor(
        executor: &MossTdGgmlExecutor,
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, Vec<Segment>, std::time::Duration, f32)> {
        let pack_path = dev_pack_path()?;
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return None;
        }
        // `backend_preference` alone is inert on a direct `execute()` (it is
        // only consulted via the thread-local override -- see
        // `GgmlAsrExecutionViewRequest::backend_preference`'s doc), so install the
        // override explicitly rather than relying on the ambient backend.
        // Hold the RAII guard for the whole decode: it restores the previous
        // thread-local override on drop at the end of this function.
        let _backend_override_guard =
            install_request_backend_override(backend_preference.request_backend_override());
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
        );

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("moss runtime must pass preflight");
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            // The goldens pin the reference decode including its speaker
            // structure, so ask for it -- with Voice ID off the normalizer
            // drops the labels by design (see `speaker_segments`).
            request_options: crate::models::ggml_asr_executor::GgmlAsrExecutionOptions {
                in_decoder_speakers: true,
                ..Default::default()
            },
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_test_request_decoder_state(executor, &mut request);

        let started_at = Instant::now();
        let result = executor.execute_view(&request).expect("moss-td transcribe");
        let elapsed = started_at.elapsed();
        Some((
            result.transcription.text,
            result.transcription.segments,
            elapsed,
            audio_duration_seconds,
        ))
    }

    /// Same real-pack e2e path as [`transcribe_with_dev_pack`], but returns the
    /// full [`Segment`] list instead of only the flat text -- used to check
    /// that the real decode's speaker/time-anchor markup round-trips through
    /// `speaker_segments::parse_moss_td_speaker_segments` (as wired into the
    /// executor) into the same structure the golden `[Sxx]`/`[t]` tags encode.
    fn transcribe_with_dev_pack_segments(wav_path: PathBuf) -> Option<Vec<Segment>> {
        let pack_path = dev_pack_path()?;
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return None;
        }
        let _backend_override_guard = install_request_backend_override(
            GgmlAsrBackendPreference::CpuOnly.request_backend_override(),
        );
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            GgmlAsrBackendPreference::CpuOnly.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
        );
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("moss runtime must pass preflight");
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: crate::models::ggml_asr_executor::GgmlAsrExecutionOptions {
                in_decoder_speakers: true,
                ..Default::default()
            },
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let executor = MossTdGgmlExecutor::default();
        plan_test_request_decoder_state(&executor, &mut request);
        let result = executor.execute_view(&request).expect("moss-td transcribe");
        Some(result.transcription.segments)
    }

    /// Three-layer comparison for the accelerated e2e smoke tests, run over the
    /// normalized segments rather than the raw tagged string (the family's
    /// markup never leaves its normalizer, see `speaker_segments`): (1) segment
    /// count, alphanumeric text, and speaker labels match exactly; (2) raw text
    /// differs by at most one character per segment, bounding punctuation
    /// drift; (3) each time-anchor edge stays within `tolerance_secs`.
    ///
    /// Rationale for tolerating (2) rather than requiring (1)'s strictness
    /// there too: this repo's own firered-aed encoder parity investigation
    /// (`firered_aed::encoder_graph::parity_tests`, see its `dump_...`
    /// harness doc comment) already concluded that cross-backend/cross-
    /// implementation fp32 bit-identical output is not a goal this runtime
    /// has ever held anywhere -- ggml's vs another implementation's non-
    /// bit-identical fp32 reduction order routinely produces small absolute
    /// diffs at numerically delicate positions without either side being
    /// wrong. Time anchors here are exactly such a floating-point-derived
    /// value (not a token id), and the measured 0.02s CPU-vs-accelerated
    /// divergence on `en_zh_mixed.wav` lands the accelerated run on the same
    /// values as the HF fp32 reference (see that test's comment) -- i.e.
    /// both sides are plausible fp32 outcomes, not a defect on either one.
    fn assert_segments_match_accelerated_quality_envelope(
        actual: &[Segment],
        golden_reference_decode: &str,
        audio_duration_seconds: f32,
        tolerance_secs: f32,
    ) {
        let golden = super::super::speaker_segments::parse_moss_td_speaker_segments(
            golden_reference_decode,
            MossTdDecodeExtent::complete(audio_duration_seconds),
        )
        .expect("the golden reference decode parses");
        assert_eq!(
            actual.len(),
            golden.len(),
            "segment count diverged from the CPU golden"
        );
        for (index, (actual_segment, golden_segment)) in actual.iter().zip(&golden).enumerate() {
            let semantic_text = |text: &str| {
                text.chars()
                    .filter(|character| character.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            };
            assert_eq!(
                semantic_text(&actual_segment.text),
                semantic_text(&golden_segment.text),
                "segment[{index}] word/character content diverged from the CPU golden"
            );
            let actual_chars = actual_segment.text.chars().collect::<Vec<_>>();
            let golden_chars = golden_segment.text.chars().collect::<Vec<_>>();
            let text_edits = crate::metrics::wer::levenshtein(&actual_chars, &golden_chars);
            assert!(
                text_edits <= 1,
                "segment[{index}] raw text differs by {text_edits} characters; at most one \
                 punctuation edit is allowed"
            );
            assert_eq!(
                actual_segment.speaker, golden_segment.speaker,
                "segment[{index}] speaker label diverged from the CPU golden"
            );
            for (edge, actual_time, golden_time) in [
                ("start", actual_segment.start, golden_segment.start),
                ("end", actual_segment.end, golden_segment.end),
            ] {
                let diff = (actual_time - golden_time).abs();
                assert!(
                    diff <= tolerance_secs,
                    "segment[{index}].{edge} exceeds tolerance: actual={actual_time} \
                     golden={golden_time} diff={diff:.4}s (tolerance={tolerance_secs}s)"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize .oasr pack and jfk.wav; runs the CPU host-prefill path outside weight-free CI"]
    fn voice_id_disabled_real_jfk_request_prefills_without_prior_host_history() {
        // A server request with `diarize=false` reaches this native MOSS
        // executor before any optional diarization or Voice ID post-processing.
        // Keep this a real-pack smoke so an empty Q8_0 host KV prefix cannot
        // regress into a cache-count error behind the HTTP boundary.
        let Some((text, _, _)) = transcribe_with_dev_pack(dev_sample_path("jfk.wav")) else {
            return;
        };
        assert!(
            !text.trim().is_empty(),
            "MOSS must return a transcript before optional Voice ID processing"
        );
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; runs the CPU reference baseline outside weight-free CI"]
    fn golden_diff_end_to_end_transcribe_jfk_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(dev_sample_path("jfk.wav"))
        else {
            return;
        };
        eprintln!(
            "moss-td e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, normalized_golden_text(GOLDEN_JFK_TEXT, 10.59));
    }

    /// Pins the resident owner pool's two contracts: (1) a second
    /// `execute()` through the same executor reuses both owner-actor runtimes
    /// rather than creating another cache entry, and (2) reuse changes nothing
    /// observable: the second call's transcript is byte-for-byte identical to
    /// the first (and to `GOLDEN_JFK_TEXT`).
    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; runs the CPU reference baseline outside weight-free CI"]
    fn resident_runtime_cache_hits_on_a_second_transcribe_call_for_the_same_pack() {
        let executor = MossTdGgmlExecutor::default();
        let Some((first_text, _, _, _)) = transcribe_with_dev_pack_backend_using_executor(
            &executor,
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };
        assert_eq!(first_text, normalized_golden_text(GOLDEN_JFK_TEXT, 10.59));
        assert_eq!(
            executor.encoder_runtimes.usage_for_test().0,
            1,
            "first call must retain one idle encoder owner"
        );
        assert_eq!(
            executor.decoder_runtimes.usage_for_test().0,
            1,
            "first call must retain one idle decoder owner"
        );

        let Some((second_text, _, _, _)) = transcribe_with_dev_pack_backend_using_executor(
            &executor,
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };
        assert_eq!(
            second_text, first_text,
            "reusing the owner-actor runtimes must not change the decode"
        );
        assert_eq!(
            executor.encoder_runtimes.usage_for_test().0,
            1,
            "second call must reuse the same idle encoder owner"
        );
        assert_eq!(
            executor.decoder_runtimes.usage_for_test().0,
            1,
            "second call must reuse the same idle decoder owner"
        );
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; runs the CPU reference baseline outside weight-free CI"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(dev_sample_path("en_zh_mixed.wav"))
        else {
            return;
        };
        eprintln!(
            "moss-td e2e [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, normalized_golden_text(GOLDEN_EN_ZH_MIXED_TEXT, 12.88));
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; runs the CPU reference baseline outside weight-free CI"]
    fn golden_diff_end_to_end_transcribe_jfk_wav_speaker_segments() {
        let Some(segments) = transcribe_with_dev_pack_segments(dev_sample_path("jfk.wav")) else {
            return;
        };
        // Same three speaker turns the flat-text golden's `[Sxx]`/`[t]` tags
        // encode (see `golden_diff_end_to_end_transcribe_jfk_wav` and
        // `GOLDEN_JFK_TEXT`) -- this asserts the executor's real-pack
        // decode round-trips through `speaker_segments` into that same
        // structure, not just that the flat string matches.
        let expected =
            parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, MossTdDecodeExtent::complete(10.59))
                .expect("golden text itself must parse");
        assert_eq!(segments, expected);
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; runs the CPU reference baseline outside weight-free CI"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav_speaker_segments() {
        let Some(segments) = transcribe_with_dev_pack_segments(dev_sample_path("en_zh_mixed.wav"))
        else {
            return;
        };
        let expected = parse_moss_td_speaker_segments(
            GOLDEN_EN_ZH_MIXED_TEXT,
            MossTdDecodeExtent::complete(12.88),
        )
        .expect("golden text itself must parse");
        assert_eq!(segments, expected);
    }

    /// Snapshot of the shape `speaker_segments` produces for the two golden
    /// transcripts pinned above, independent of any real-pack decode -- pins
    /// the exact segment count/speaker-label/start/end/text tuple this PR's
    /// parser derives from the reference HF text, so a future edit to the
    /// grammar (e.g. changing how a back-to-back closing/opening anchor pair
    /// is split) shows up as a diff here even without a local real pack.
    #[test]
    fn snapshot_jfk_and_en_zh_mixed_golden_speaker_segments() {
        let jfk =
            parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, MossTdDecodeExtent::complete(10.59))
                .expect("jfk parses");
        let jfk_snapshot: Vec<(&str, f32, f32, &str)> = jfk
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            jfk_snapshot,
            vec![
                ("SPEAKER_01", 0.28, 2.32, "And so, my fellow Americans,"),
                (
                    "SPEAKER_01",
                    3.22,
                    7.71,
                    "ask not what your country can do for you,"
                ),
                (
                    "SPEAKER_01",
                    8.12,
                    10.59,
                    "ask what you can do for your country."
                ),
            ]
        );

        let en_zh_mixed = parse_moss_td_speaker_segments(
            GOLDEN_EN_ZH_MIXED_TEXT,
            MossTdDecodeExtent::complete(12.88),
        )
        .expect("parses");
        let en_zh_mixed_snapshot: Vec<(&str, f32, f32, &str)> = en_zh_mixed
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            en_zh_mixed_snapshot,
            vec![
                ("SPEAKER_01", 0.27, 2.32, "And so, my fellow Americans,"),
                ("SPEAKER_01", 3.21, 4.44, "ask not."),
                (
                    "SPEAKER_02",
                    4.96,
                    12.88,
                    "今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去伊加新"
                ),
            ]
        );
    }

    /// Synthetic (not a real decode) multi-chunk-duration transcript: every
    /// anchor above sits well inside `executor.rs`'s first 30s encoder chunk
    /// (`CHUNK_SAMPLES`), so `snapshot_jfk_and_en_zh_mixed_golden_speaker_segments`
    /// never exercises `speaker_segments` against text spanning more than one
    /// chunk's worth of audio duration. This transcript's anchors straddle
    /// two `CHUNK_SAMPLES` boundaries (30s and 60s) across three speaker
    /// turns and a language switch, covering the shape a real multi-chunk
    /// longform decode would produce -- text parsing itself is chunk-count-
    /// agnostic (it runs once over the final concatenated decode, same as
    /// for a single-chunk clip), so this is a scale/structure regression
    /// check on the parser, not a claim that this exact text was ever
    /// decoded from real audio.
    const SYNTHETIC_MULTI_CHUNK_TEXT: &str = concat!(
        "[0.50][S01] Good morning everyone, let's get started.[29.80][31.20][S01] ",
        "First, a quick recap of last week's numbers.[58.90][61.40][S02] 谢谢，我来补充一下财务方面的情况。",
        "[92.15][93.00][S01] Great, let's move to questions then.[110.75]",
    );

    #[test]
    fn synthetic_multi_chunk_duration_transcript_parses_into_structured_segments() {
        let segments = parse_moss_td_speaker_segments(
            SYNTHETIC_MULTI_CHUNK_TEXT,
            MossTdDecodeExtent::complete(110.75),
        )
        .expect("synthetic multi-chunk transcript parses");
        let snapshot: Vec<(&str, f32, f32, &str)> = segments
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            snapshot,
            vec![
                (
                    "SPEAKER_01",
                    0.50,
                    29.80,
                    "Good morning everyone, let's get started."
                ),
                (
                    "SPEAKER_01",
                    31.20,
                    58.90,
                    "First, a quick recap of last week's numbers."
                ),
                (
                    "SPEAKER_02",
                    61.40,
                    92.15,
                    "谢谢，我来补充一下财务方面的情况。"
                ),
                (
                    "SPEAKER_01",
                    93.00,
                    110.75,
                    "Great, let's move to questions then."
                ),
            ]
        );
    }

    /// Time anchors are floating-point-derived (see
    /// `assert_segments_match_accelerated_quality_envelope`'s doc
    /// comment for why exact cross-backend anchor equality is not the
    /// right bar). Moving the adaptor MLP onto Metal measured a maximum 0.07s
    /// shift on these clips; 0.08s admits that bounded delta while still
    /// rejecting a structurally different time token.
    const ACCELERATED_ANCHOR_TOLERANCE_SECS: f32 = 0.08;

    // Explicit `execution_target=accelerated` e2e smoke: an explicit
    // `Accelerated` request installs the same thread-local override
    // `graph_config.rs` documents as always winning over this family's
    // `AutoGpuPolicy::AllBackends`, so Auto may resolve the encoder to Metal
    // when available and an explicit accelerated request selects it directly
    // (see `encoder_graph_config_honors_explicit_accelerated_request` in
    // `graph_config.rs`). Decode also runs on Metal under
    // Auto today (the shared qwen decode path is `AllBackends`, and #180
    // fixed its reuse-path graph so Metal decode reuses its graph), so this
    // is the full accelerated-request path: Metal encoder + Metal decode,
    // diffed against the same CPU golden the two tests above pin, via
    // `assert_segments_match_accelerated_quality_envelope` (strict on semantic
    // text and speaker labels; bounded to one punctuation edit and 0.08s per
    // anchor edge).
    //
    // With the adaptor moved from the host scalar loop into this same Metal
    // graph, jfk.wav keeps every word/punctuation/speaker label; its largest
    // time-token edge shift is 0.07s.
    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; drives an explicit accelerated request \
                (Metal encoder + Metal decode) and needs a Metal device; remains outside \
                weight-free CI"]
    fn golden_diff_end_to_end_transcribe_jfk_wav_accelerated() {
        let Some((_, segments, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_segments_match_accelerated_quality_envelope(
            &segments,
            GOLDEN_JFK_TEXT,
            10.59,
            ACCELERATED_ANCHOR_TOLERANCE_SECS,
        );
    }

    // Measured after moving the adaptor MLP onto Metal: every alphanumeric
    // character and speaker label remains identical; the second segment drops
    // one trailing period, one edge shifts 0.02s, and the final edge shifts
    // 0.03s. The bounded envelope above records those exact acceptable deltas
    // instead of weakening the CPU/HF official golden.
    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and \
                tmp/moss-td/samples/*.wav; drives an explicit accelerated request \
                (Metal encoder + Metal decode) and needs a Metal device; remains outside \
                weight-free CI"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav_accelerated() {
        let Some((_, segments, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("en_zh_mixed.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_segments_match_accelerated_quality_envelope(
            &segments,
            GOLDEN_EN_ZH_MIXED_TEXT,
            12.88,
            ACCELERATED_ANCHOR_TOLERANCE_SECS,
        );
    }

    #[test]
    #[ignore = "requires a local real moss-transcribe-diarize-fp16.oasr pack and the 3-minute AISHELL-4 fixture; drives an explicit accelerated request outside weight-free CI"]
    fn accelerated_aishell4_three_minute_smoke_completes_with_structured_transcript() {
        let Some((text, segments, _, _)) = transcribe_with_dev_pack_backend(
            dev_sample_path("aishell4_multispeaker_3min.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        assert!(
            !text.trim().is_empty(),
            "accelerated AISHELL-4 decode must emit a non-empty transcript"
        );
        let Ok(golden_root) = crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_GOLDEN",
            "MOSS Transcribe Diarize development golden directory",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}")) else {
            return;
        };
        let golden_path = golden_root.join("aishell4_multispeaker_3min.json");
        if !golden_path.exists() {
            eprintln!("skipping: {} not present", golden_path.display());
            return;
        }
        let golden: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&golden_path).expect("read AISHELL-4 development golden"),
        )
        .expect("parse AISHELL-4 development golden");
        // The pinned reference text is the raw tagged decode; what a caller
        // gets is its markup-free projection (see `speaker_segments`).
        assert_eq!(
            text,
            normalized_golden_text(
                golden["text"].as_str().expect("AISHELL-4 golden text"),
                180.0
            ),
            "accelerated AISHELL-4 transcript must match the pinned reference text"
        );
        assert!(
            !segments.is_empty(),
            "AISHELL-4 must emit structured segments"
        );
        assert!(
            segments.iter().all(|segment| {
                segment.speaker.is_some()
                    && segment.start.is_finite()
                    && segment.end.is_finite()
                    && segment.start <= segment.end
                    && !segment.text.trim().is_empty()
            }),
            "AISHELL-4 segments must retain speaker labels and valid time ranges"
        );
        assert!(
            segments
                .windows(2)
                .all(|pair| pair[0].start <= pair[1].start),
            "AISHELL-4 segment starts must be ordered"
        );
        assert!(
            !text.contains("[S"),
            "the family's speaker markup must never reach the caller"
        );
    }
}
