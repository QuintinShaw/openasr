use std::sync::{Arc, OnceLock};
use std::time::Instant;

use thiserror::Error;

use super::Qwen3AsrTokenizer;
use super::audio_encoder::{
    Qwen3AsrAudioEncoderError, Qwen3AsrAudioEncoderRuntime, Qwen3AsrAudioEncoderWeights,
};
use super::batched_decode::{
    Qwen3AsrServeBatchConfig, Qwen3AsrServeBatchEngineRegistry, Qwen3AsrServeBatchJob,
    Qwen3AsrServeBatchPromptInput, shutdown_qwen_serve_batch_engines, submit_qwen_serve_batch_job,
};
#[cfg(test)]
use super::decode_budget::QWEN3_DECODE_MIN_GENERATED_TOKENS;
use super::decode_budget::qwen3_generated_token_budget as qwen3_budget_for_shape;
use super::decode_prompt::{Qwen3AsrDecodePromptError, build_qwen3_decode_prompt};
use super::frontend::{
    Qwen3AsrMelFeatures, Qwen3AsrMelFrontendError, Qwen3AsrMelFrontendPlan,
    qwen3_mel_features_from_prepared_audio,
};
use super::graph_config::{
    qwen_decoder_graph_config, qwen_encoder_graph_config, qwen_runtime_graph_config,
};
use super::greedy_decode::{Qwen3AsrGreedyDecodeError, run_qwen3_greedy_decode_loop};
use super::kv_cache::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrKvCacheCapacityError,
};
use super::llm_prefill::build_qwen3_llm_prefill_input;
use super::llm_transformer::{
    Qwen3AsrLlmWholeDecoderGraphExecutor, QwenQkvExecutionMode, QwenWholeDecoderPlan,
    qwen_llm_effective_native_gqa_capability,
};
use super::logits_head::{Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime};
use super::lora::{qwen_adapter_cache_fingerprint, resolve_qwen_lora_adapter};
#[cfg(test)]
use super::prepared_runtime::Qwen3AsrPreparedRuntime;
use super::prepared_runtime::Qwen3AsrPreparedRuntimeError;
use super::prompt_embedding::{
    Qwen3AsrPromptTokenInput, build_qwen3_prompt_embeddings_with_audio_positions,
};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::hparams::{QWEN3_AUDIO_LAYERS_KEY, QWEN3_LLM_LAYERS_KEY};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{
    OpenAsrArchitectureRegistry, OpenAsrBlockStackStrategy, QWEN3_ASR_GGML_ARCHITECTURE_ID,
};
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphError, GgmlNativeGqaCapability, RequestBackendPreference,
    ResolvedFamilyRuntimeInput, env_toggle_with_raw, env_var_truthy, request_backend_override,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinSeq2SeqDecodePolicyConfigInput, build_builtin_seq2seq_decode_policy_config,
    resolve_builtin_decode_policy,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::lora_adapter::{
    ResolvedLoraAdapterCache, ResolvedLoraAdapterHandle, resolved_lora_adapter,
};
use crate::models::native_execution_services::{
    ExecutionLaneKey, current_execution_lane_key, current_execution_placement,
};
use crate::models::prepared_runtime_cache::PreparedRuntimeHandle;
use crate::models::runtime_cache_coordinator::{PackContentKey, canonical_runtime_cache_path};
use crate::models::runtime_prepared_registry::{
    BuiltinPreparedRuntime, BuiltinPreparedRuntimeCache, BuiltinPreparedRuntimeRegistryError,
    PreparedRuntimeLookup,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeStopReason,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryCapacity, SystemMemoryOwner,
};
use crate::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
    GgmlAsrViewExecutor, NativeAsrSession, QWEN3_ASR_GGML_ADAPTER_ID, Segment, Transcription,
};

#[cfg(test)]
use super::runtime_contract::parse_qwen3_execution_metadata;
use crate::GgufRuntimeSourcePreflight;
#[cfg(test)]
use crate::models::runtime_prepared_registry::build_builtin_prepared_runtime;

const QWEN3_EXECUTOR_ID: &str = crate::arch::QWEN3_ASR_EXECUTOR_COMPONENT_ID;
const QWEN3_STREAMING_EXECUTOR_ID: &str = "qwen3-asr-ggml-snapshot-streaming-executor-v1";
const QWEN3_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const QWEN3_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;
const QWEN3_DECODE_PROFILE_ENV: &str = "OPENASR_QWEN_DECODE_PROFILE";
const QWEN3_GPU_SPLIT_LOADED_QKV_ENV: &str = "OPENASR_QWEN_GPU_SPLIT_LOADED_QKV";

type Qwen3AsrAudioEncoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
type Qwen3AsrDecoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    String,
    GgmlNativeGqaCapability,
    QwenQkvExecutionMode,
);

/// Immutable identity of one ordinary Qwen model owner. Request-sized KV is
/// deliberately absent: it is created inside one actor call and dropped before
/// checkout return, while the pack, physical lane and adapter stay resident.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Qwen3AsrRuntimeOwnerCacheKey {
    content: PackContentKey,
    lane: ExecutionLaneKey,
    native_gqa: GgmlNativeGqaCapability,
    qkv_execution_mode: QwenQkvExecutionMode,
}

struct Qwen3AsrAudioEncoderRuntimeActorState {
    runtime: Qwen3AsrAudioEncoderRuntime,
    _prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
}

impl std::fmt::Debug for Qwen3AsrAudioEncoderRuntimeActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrAudioEncoderRuntimeActorState")
            .finish_non_exhaustive()
    }
}

struct Qwen3AsrDecoderRuntimeActorState {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Arc<Qwen3AsrLlmLogitsHead>,
    logits_head_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    _prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
}

impl std::fmt::Debug for Qwen3AsrDecoderRuntimeActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrDecoderRuntimeActorState")
            .finish_non_exhaustive()
    }
}

/// All ordinary Qwen native stages live and die on this one owner thread.
/// GPU-class runners therefore resolve one thread-local raw backend and their
/// pack-wide weight loads coalesce into one Rc binding. Mutable KV/session
/// state remains request-owned and never enters this object.
struct Qwen3AsrRuntimeOwnerState {
    audio_encoder: Qwen3AsrAudioEncoderRuntime,
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Arc<Qwen3AsrLlmLogitsHead>,
    logits_head_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    _prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
}

impl std::fmt::Debug for Qwen3AsrRuntimeOwnerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrRuntimeOwnerState")
            .finish_non_exhaustive()
    }
}

impl Qwen3AsrRuntimeOwnerState {
    fn quoted_retained_system_memory_bytes(layer_count: usize) -> Result<u64, String> {
        Qwen3AsrLlmWholeDecoderGraphExecutor::quoted_retained_system_memory_bytes(layer_count)
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        // The decoder layer-handle Vec is the only uniquely owned Rust heap in
        // this actor. Encoder/decoder native contexts and arenas admit their
        // own allocations at construction; the loaded pack binding is one Rc
        // owner; prepared/logits assets are Arc clones already charged by the
        // prepared-runtime owner. Keep quote and outcome on this same exact
        // actor-owned boundary rather than double-charging shared/native data.
        let mut bytes = SystemMemoryCapacity::default();
        bytes.add(
            self.whole_decoder.retained_system_memory_bytes()?,
            "qwen unified runtime decoder handles",
        )?;
        Ok(bytes.finish())
    }
}

type Qwen3AsrAudioEncoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    Qwen3AsrAudioEncoderRuntimeCacheKey,
    Qwen3AsrAudioEncoderRuntimeActorState,
>;
type Qwen3AsrDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    Qwen3AsrDecoderRuntimeCacheKey,
    Qwen3AsrDecoderRuntimeActorState,
>;
type Qwen3AsrRuntimeOwnerPool =
    AdmittedPinnedRuntimeActorCheckoutPool<Qwen3AsrRuntimeOwnerCacheKey, Qwen3AsrRuntimeOwnerState>;
type Qwen3AsrAudioEncoderRuntimeActor = PinnedRuntimeActorCheckout<
    Qwen3AsrAudioEncoderRuntimeCacheKey,
    Qwen3AsrAudioEncoderRuntimeActorState,
>;
type Qwen3AsrDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<Qwen3AsrDecoderRuntimeCacheKey, Qwen3AsrDecoderRuntimeActorState>;
type Qwen3AsrRuntimeOwnerActor =
    PinnedRuntimeActorCheckout<Qwen3AsrRuntimeOwnerCacheKey, Qwen3AsrRuntimeOwnerState>;

struct Qwen3AsrDecoderRuntimeView<'a> {
    whole_decoder: &'a mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: &'a Arc<Qwen3AsrLlmLogitsHead>,
    logits_head_runtime: &'a mut Qwen3AsrLlmLogitsHeadRuntime,
}

enum Qwen3AsrDecoderActorCheckout {
    UnifiedGpu(Qwen3AsrRuntimeOwnerActor),
    Split(Qwen3AsrDecoderRuntimeActor),
}

impl Qwen3AsrDecoderActorCheckout {
    fn call_mut_fallible<O, E, F>(
        &self,
        operation: F,
    ) -> Result<Result<O, E>, PinnedRuntimeActorError>
    where
        O: Send + 'static,
        E: Send + 'static,
        F: FnOnce(&mut Qwen3AsrDecoderRuntimeView<'_>) -> Result<O, E> + Send + 'static,
    {
        match self {
            Self::UnifiedGpu(actor) => actor.call_mut_fallible(move |state| {
                operation(&mut Qwen3AsrDecoderRuntimeView {
                    whole_decoder: &mut state.whole_decoder,
                    logits_head: &state.logits_head,
                    logits_head_runtime: &mut state.logits_head_runtime,
                })
            }),
            Self::Split(actor) => actor.call_mut(move |state| {
                operation(&mut Qwen3AsrDecoderRuntimeView {
                    whole_decoder: &mut state.whole_decoder,
                    logits_head: &state.logits_head,
                    logits_head_runtime: &mut state.logits_head_runtime,
                })
            }),
        }
    }
}

fn qwen_unified_runtime_owner_enabled(
    resolved_runtime: ResolvedFamilyRuntimeInput,
    native_gqa: GgmlNativeGqaCapability,
    native_logits_runtime: bool,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> bool {
    let backend = resolved_runtime.backend();
    if backend != GgmlCpuGraphBackend::Gpu
        || !native_gqa.is_validated()
        || !native_logits_runtime
        || placement != Some(ExecutionPlacement::FullDevice)
        || !matches!(
            backend_preference,
            Some(RequestBackendPreference::Exact(route))
                if route.addressability.is_exactly_addressable()
                    && matches!(route.provider, ExecutionProvider::Cuda | ExecutionProvider::Vulkan)
        )
    {
        return false;
    }
    let encoder = qwen_encoder_graph_config(backend);
    let decoder = qwen_decoder_graph_config(backend);
    let logits = qwen_decoder_graph_config(backend);
    [encoder, decoder, logits].into_iter().all(|config| {
        config.backend == backend && config.backend.is_gpu_class() && !config.use_scheduler
    })
}

fn qwen_qkv_execution_mode(
    unified_runtime_enabled: bool,
    split_loaded_requested: bool,
) -> QwenQkvExecutionMode {
    if unified_runtime_enabled && split_loaded_requested {
        QwenQkvExecutionMode::SplitLoaded
    } else {
        QwenQkvExecutionMode::FusedArena
    }
}

fn qwen_split_loaded_qkv_enabled_with_env(raw: Option<&str>) -> bool {
    env_toggle_with_raw(None, raw, true)
}

fn validate_unified_runtime_owner_state(
    state: &Qwen3AsrRuntimeOwnerState,
    expected_backend: GgmlCpuGraphBackend,
) -> Result<(), Qwen3AsrGgmlExecutorError> {
    let expected_lane = (expected_backend, false);
    let encoder_lane = state.audio_encoder.graph_lane();
    let decoder_lane = state.whole_decoder.graph_lane();
    let logits_lane = state.logits_head_runtime.graph_lane();
    if !expected_backend.is_gpu_class()
        || encoder_lane != expected_lane
        || decoder_lane != expected_lane
        || logits_lane != Some(expected_lane)
    {
        return Err(Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
            reason: format!(
                "unified runtime requires one direct GPU lane: expected={expected_lane:?}, encoder={encoder_lane:?}, decoder={decoder_lane:?}, logits={logits_lane:?}"
            ),
        });
    }
    let encoder_binding = state.audio_encoder.loaded_weight_binding_identity();
    let decoder_binding = state.whole_decoder.loaded_weight_binding_identity();
    if encoder_binding.is_none() || encoder_binding != decoder_binding {
        return Err(Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
            reason: format!(
                "unified runtime failed to coalesce its pack-wide loaded-weight binding: encoder={encoder_binding:?}, decoder={decoder_binding:?}"
            ),
        });
    }
    Ok(())
}

fn qwen_decode_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_var_truthy(QWEN3_DECODE_PROFILE_ENV))
}

fn qwen_decode_profile_start() -> Option<Instant> {
    qwen_decode_profile_enabled().then(Instant::now)
}

fn qwen_decode_profile_log(stage: &str, started_at: Instant) {
    eprintln!(
        "openasr_qwen_decode_profile: stage={} total_us={}",
        stage,
        started_at.elapsed().as_micros()
    );
}

fn qwen_decode_profile_log_opt(stage: &str, started_at: Option<Instant>) {
    if let Some(started_at) = started_at {
        qwen_decode_profile_log(stage, started_at);
    }
}

fn qwen_decode_profile_log_prefill_chunk(
    position_offset: usize,
    chunk_len: usize,
    started_at: Option<Instant>,
) {
    if let Some(started_at) = started_at {
        eprintln!(
            "openasr_qwen_decode_profile: stage=prefill_chunk position_offset={} chunk_len={} total_us={}",
            position_offset,
            chunk_len,
            started_at.elapsed().as_micros()
        );
    }
}

#[derive(Debug, Error)]
enum Qwen3AsrGgmlExecutorError {
    #[error("qwen3-asr ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("qwen3-asr runtime contract check failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("qwen3-asr runtime metadata read failed: {reason}")]
    RuntimeMetadataReadFailed { reason: String },
    #[error("qwen3-asr decode prompt construction failed: {reason}")]
    DecodePromptConstructionFailed { reason: String },
    #[error("qwen3-asr mel frontend failed: {reason}")]
    MelFrontendFailed { reason: String },
    #[error("qwen3-asr audio encoder failed: {reason}")]
    AudioEncoderFailed { reason: String },
    #[error("qwen3-asr token embedding prefill failed: {reason}")]
    TokenEmbeddingPrefillFailed { reason: String },
    #[error("qwen3-asr prompt embedding assembly failed: {reason}")]
    PromptEmbeddingAssemblyFailed { reason: String },
    #[error("qwen3-asr greedy decode loop failed: {reason}")]
    GreedyDecodeFailed { reason: String },
    #[error("qwen3-asr decode token budget is unavailable: {reason}")]
    DecodeBudgetUnavailable { reason: String },
    #[error("qwen3-asr decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("qwen3-asr llm logits head failed: {reason}")]
    LlmLogitsHeadFailed { reason: String },
    #[error("qwen3-asr llm transformer decode step failed: {reason}")]
    LlmTransformerDecodeStepFailed { reason: String },
    #[error("qwen3-asr {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error(
        "qwen3-asr ggml executor currently supports only {expected_sample_rate_hz}Hz mono input, got sample_rate={sample_rate_hz} channels={channels}"
    )]
    UnsupportedInputShape {
        expected_sample_rate_hz: u32,
        sample_rate_hz: u32,
        channels: u16,
    },
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrGgmlExecutor {
    runtime_cache_by_path: BuiltinPreparedRuntimeCache,
    audio_encoder_runtimes: Arc<Qwen3AsrAudioEncoderRuntimePool>,
    decoder_runtimes: Arc<Qwen3AsrDecoderRuntimePool>,
    unified_gpu_runtimes: Arc<Qwen3AsrRuntimeOwnerPool>,
    serve_batch_engines: Qwen3AsrServeBatchEngineRegistry,
    lora_adapters: ResolvedLoraAdapterCache,
}

impl Default for Qwen3AsrGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            QWEN3_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            QWEN3_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            runtime_cache_by_path: BuiltinPreparedRuntimeCache::default(),
            audio_encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-qwen3-audio-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-qwen3-decoder-owner",
                limits,
            )),
            unified_gpu_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-qwen3-unified-gpu-owner",
                limits,
            )),
            serve_batch_engines: Qwen3AsrServeBatchEngineRegistry::default(),
            lora_adapters: ResolvedLoraAdapterCache::default(),
        }
    }
}

impl Qwen3AsrGgmlExecutor {
    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> Qwen3AsrGgmlExecutorError {
        Qwen3AsrGgmlExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn checkout_audio_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Qwen3AsrAudioEncoderRuntimeActor, Qwen3AsrGgmlExecutorError> {
        let encoder_backend = qwen_encoder_graph_config(backend).backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(encoder_backend),
        );
        let preflight = preflight.clone();
        self.audio_encoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared_runtime_owner))),
            move |(preflight, prepared_runtime_owner)| {
                let runtime = Qwen3AsrAudioEncoderRuntime::new_from_preflight(&preflight, backend)
                    .map_err(|error| Qwen3AsrGgmlExecutorError::AudioEncoderFailed {
                        reason: error.to_string(),
                    })?;
                Ok(SystemMemoryOwner::without_allocation(
                    Qwen3AsrAudioEncoderRuntimeActorState {
                        runtime,
                        _prepared_runtime_owner: prepared_runtime_owner,
                    },
                ))
            },
            |error| Self::map_actor_error("audio-encoder", error),
        )
    }

    fn encode_with_owned_audio_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        mel_features: Qwen3AsrMelFeatures,
        backend: GgmlCpuGraphBackend,
    ) -> Result<super::audio_encoder::Qwen3AsrAudioEncoderOutput, Qwen3AsrGgmlExecutorError> {
        let actor =
            self.checkout_audio_encoder_runtime(preflight, prepared_runtime_owner, backend)?;
        actor
            .call_mut(move |state| {
                let prepared_runtime = state
                    ._prepared_runtime_owner
                    .as_ref()
                    .as_qwen3_asr()
                    .ok_or_else(|| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
                        reason: "audio actor received a non-qwen prepared runtime".to_string(),
                    })?;
                let encode_result = state.runtime.encode(
                    &prepared_runtime.audio_encoder_weights,
                    prepared_runtime.metadata,
                    &mel_features,
                );
                let release_result = state.runtime.release_transient_compute_memory();
                match (encode_result, release_result) {
                    (Ok(output), Ok(())) => Ok(output),
                    (Err(error), _) => Err(error),
                    (Ok(_), Err(error)) => Err(error),
                }
            })
            .map_err(|error| Self::map_actor_error("audio-encoder", error))?
            .map_err(map_audio_encoder_error)
    }

    fn encode_with_unified_gpu_runtime(
        &self,
        actor: &Qwen3AsrRuntimeOwnerActor,
        mel_features: Qwen3AsrMelFeatures,
    ) -> Result<super::audio_encoder::Qwen3AsrAudioEncoderOutput, Qwen3AsrGgmlExecutorError> {
        actor
            .call_mut_fallible(move |state| {
                let prepared_runtime = state
                    ._prepared_runtime_owner
                    .as_ref()
                    .as_qwen3_asr()
                    .ok_or_else(|| Qwen3AsrAudioEncoderError::GraphExecutionFailed {
                        reason: "unified actor received a non-qwen prepared runtime".to_string(),
                    })?;
                let encode_result = state.audio_encoder.encode(
                    &prepared_runtime.audio_encoder_weights,
                    prepared_runtime.metadata,
                    &mel_features,
                );
                let release_result = state.audio_encoder.release_transient_compute_memory();
                match (encode_result, release_result) {
                    (Ok(output), Ok(())) => Ok(output),
                    (Err(error), _) => Err(error),
                    (Ok(_), Err(error)) => Err(error),
                }
            })
            .map_err(|error| Self::map_actor_error("unified-audio-encoder", error))?
            .map_err(map_audio_encoder_error)
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        adapter: Option<ResolvedLoraAdapterHandle>,
        resolved_runtime: ResolvedFamilyRuntimeInput,
        native_gqa: GgmlNativeGqaCapability,
        qkv_execution_mode: QwenQkvExecutionMode,
    ) -> Result<Qwen3AsrDecoderRuntimeActor, Qwen3AsrGgmlExecutorError> {
        let backend = resolved_runtime.backend();
        let decoder_backend = qwen_runtime_graph_config(backend).backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(decoder_backend),
            qwen_adapter_cache_fingerprint(adapter.as_ref().map(resolved_lora_adapter)),
            native_gqa,
            qkv_execution_mode,
        );
        let preflight = preflight.clone();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared_runtime_owner, adapter))),
            move |(preflight, prepared_runtime_owner, adapter)| {
                let prepared_runtime =
                    prepared_runtime_owner
                        .as_ref()
                        .as_qwen3_asr()
                        .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                            reason: "decoder actor received a non-qwen prepared runtime"
                                .to_string(),
                        })?;
                let whole_decoder =
                    Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_lora_for_qwen(
                        &prepared_runtime.decoder_plan,
                        &preflight,
                        prepared_runtime.logits_head.fused_top1_spec(),
                        prepared_runtime.token_embedding_table.device_graph_spec(),
                        adapter.as_ref().map(resolved_lora_adapter),
                        backend,
                        native_gqa,
                        qkv_execution_mode,
                    )
                    .map_err(|error| {
                        Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                            reason: format!("qwen3-asr whole-decoder graph init failed: {error}"),
                        }
                    })?;
                let logits_head_runtime = prepared_runtime
                    .logits_head
                    .new_runtime(decoder_backend)
                    .map_err(|error| Qwen3AsrGgmlExecutorError::LlmLogitsHeadFailed {
                        reason: error.to_string(),
                    })?;
                Ok(SystemMemoryOwner::without_allocation(
                    Qwen3AsrDecoderRuntimeActorState {
                        whole_decoder,
                        logits_head: Arc::clone(&prepared_runtime.logits_head),
                        logits_head_runtime,
                        _prepared_runtime_owner: prepared_runtime_owner,
                    },
                ))
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn checkout_unified_gpu_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        resolved_runtime: ResolvedFamilyRuntimeInput,
        native_gqa: GgmlNativeGqaCapability,
        qkv_execution_mode: QwenQkvExecutionMode,
    ) -> Result<Qwen3AsrRuntimeOwnerActor, Qwen3AsrGgmlExecutorError> {
        let backend = resolved_runtime.backend();
        let decoder_backend = qwen_runtime_graph_config(backend).backend;
        let key = Qwen3AsrRuntimeOwnerCacheKey {
            content: PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane: current_execution_lane_key(decoder_backend),
            native_gqa,
            qkv_execution_mode,
        };
        let preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.unified_gpu_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let prepared_runtime = prepared_runtime_owner
                    .as_ref()
                    .as_qwen3_asr()
                    .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                        reason: "unified actor received a non-qwen prepared runtime".to_string(),
                    })?;
                let retained = Qwen3AsrRuntimeOwnerState::quoted_retained_system_memory_bytes(
                    prepared_runtime.decoder_plan.layer_count(),
                )
                .map_err(|reason| Qwen3AsrGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "unified-runtime",
                    reason,
                })?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("qwen-unified-gpu-runtime:{content_id}"),
                    retained,
                    retained,
                )
                .map_err(|error| Qwen3AsrGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "unified-runtime",
                    reason: error.to_string(),
                })?;
                Ok((
                    retained,
                    (preflight, prepared_runtime_owner, quote),
                ))
            },
            move |(preflight, prepared_runtime_owner, quote)| {
                match SystemMemoryOwner::try_allocate_transaction(quote, || {
                    let prepared_runtime = prepared_runtime_owner
                        .as_ref()
                        .as_qwen3_asr()
                        .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                            reason: "unified actor received a non-qwen prepared runtime"
                                .to_string(),
                        })?;
                    // Keep the encoder's loaded context alive while constructing
                    // the decoder and logits runners. On CUDA/Vulkan these three
                    // runners resolve one cached raw backend on this owner thread,
                    // so the decoder load upgrades the encoder's weak Rc entry.
                    let audio_encoder = Qwen3AsrAudioEncoderRuntime::new_from_preflight(
                        &preflight, backend,
                    )
                    .map_err(|error| Qwen3AsrGgmlExecutorError::AudioEncoderFailed {
                        reason: error.to_string(),
                    })?;
                    let whole_decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_lora_for_qwen(
                        &prepared_runtime.decoder_plan,
                        &preflight,
                        prepared_runtime.logits_head.fused_top1_spec(),
                        prepared_runtime.token_embedding_table.device_graph_spec(),
                        None,
                        backend,
                        native_gqa,
                        qkv_execution_mode,
                    )
                    .map_err(|error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                        reason: format!("qwen3-asr whole-decoder graph init failed: {error}"),
                    })?;
                    let logits_head_runtime = prepared_runtime
                        .logits_head
                        .new_runtime(decoder_backend)
                        .map_err(|error| Qwen3AsrGgmlExecutorError::LlmLogitsHeadFailed {
                            reason: error.to_string(),
                        })?;
                    let state = Qwen3AsrRuntimeOwnerState {
                        audio_encoder,
                        whole_decoder,
                        logits_head: Arc::clone(&prepared_runtime.logits_head),
                        logits_head_runtime,
                        _prepared_runtime_owner: prepared_runtime_owner,
                    };
                    validate_unified_runtime_owner_state(&state, decoder_backend)?;
                    let retained = state.retained_system_memory_bytes().map_err(|reason| {
                        Qwen3AsrGgmlExecutorError::RuntimeOwnershipFailed {
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
                        Err(Qwen3AsrGgmlExecutorError::RuntimeOwnershipFailed {
                            stage: "unified-runtime",
                            reason: error.to_string(),
                        })
                    }
                }
            },
            |error| Self::map_actor_error("unified-runtime", error),
        )
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        if request.selected_family.adapter_id != QWEN3_ASR_GGML_ADAPTER_ID {
            return Err(Qwen3AsrGgmlExecutorError::AdapterMismatch {
                expected: QWEN3_ASR_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let profile_started_at = qwen_decode_profile_start();
        let preflight_started_at = qwen_decode_profile_start();
        let preflight = request.runtime_source_preflight();
        qwen_decode_profile_log_opt("runtime_preflight", preflight_started_at);
        let prepared_runtime_started_at = qwen_decode_profile_start();
        let result = self
            .runtime_cache_by_path
            .prepared_runtime_for_preflight(
                PreparedRuntimeLookup {
                    model_architecture: request.selected_family.model_architecture,
                    preflight,
                    backend: request.resolved_runtime.backend(),
                },
                map_prepared_runtime_registry_error,
                qwen_runtime_cache_slot_unavailable,
            )
            .and_then(|prepared_runtime_owner| {
                self.execute_with_prepared_runtime(
                    request,
                    preflight,
                    &prepared_runtime_owner,
                    skip_serve_batch,
                )
            });
        qwen_decode_profile_log_opt("prepared_runtime_and_execute", prepared_runtime_started_at);
        qwen_decode_profile_log_opt("execute_inner_total", profile_started_at);
        result
    }

    fn execute_with_prepared_runtime(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_runtime_owner: &PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        let prepared_runtime = prepared_runtime_owner
            .as_ref()
            .as_qwen3_asr()
            .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!(
                    "prepared runtime registry returned non-qwen runtime for architecture '{}'",
                    request.selected_family.model_architecture
                ),
            })?;
        let adapter = resolve_qwen_lora_adapter(
            &self.lora_adapters,
            request.request_options.adapter_path.as_deref(),
            preflight,
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!("qwen3-asr lora adapter rejected: {error}"),
            },
        )?;
        self.execute_with_runtime_assets(
            request,
            preflight,
            prepared_runtime.metadata,
            prepared_runtime.tokenizer.as_ref(),
            &prepared_runtime.mel_frontend_plan,
            &prepared_runtime.audio_encoder_weights,
            prepared_runtime.token_embedding_table.clone(),
            prepared_runtime.decoder_plan.clone(),
            Arc::clone(prepared_runtime_owner),
            adapter,
            skip_serve_batch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with_runtime_assets(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: Qwen3AsrExecutionMetadata,
        tokenizer: Option<&Qwen3AsrTokenizer>,
        mel_frontend_plan: &Qwen3AsrMelFrontendPlan,
        audio_encoder_weights: &Qwen3AsrAudioEncoderWeights,
        token_embedding_table: Arc<
            crate::models::mapped_token_embedding::MappedTokenEmbeddingTable,
        >,
        decoder_plan: Arc<QwenWholeDecoderPlan>,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        adapter: Option<ResolvedLoraAdapterHandle>,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        let profile_started_at = qwen_decode_profile_start();
        let validate_shape_started_at = qwen_decode_profile_start();
        self.validate_prepared_audio_shape(metadata, &request.prepared_audio)?;
        qwen_decode_profile_log_opt("validate_prepared_audio_shape", validate_shape_started_at);
        let mel_started_at = qwen_decode_profile_start();
        let mel_features =
            qwen3_mel_features_from_prepared_audio(&request.prepared_audio, mel_frontend_plan)
                .map_err(map_mel_frontend_error)?;
        qwen_decode_profile_log_opt("mel_frontend", mel_started_at);
        let result = self.decode_with_runtime_assets(
            request,
            preflight,
            metadata,
            tokenizer,
            token_embedding_table,
            audio_encoder_weights,
            &mel_features,
            decoder_plan,
            prepared_runtime_owner,
            adapter,
            skip_serve_batch,
        );
        qwen_decode_profile_log_opt("execute_with_runtime_assets_total", profile_started_at);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_runtime_assets(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: Qwen3AsrExecutionMetadata,
        tokenizer: Option<&Qwen3AsrTokenizer>,
        token_embedding_table: Arc<
            crate::models::mapped_token_embedding::MappedTokenEmbeddingTable,
        >,
        audio_encoder_weights: &Qwen3AsrAudioEncoderWeights,
        mel_features: &Qwen3AsrMelFeatures,
        decoder_plan: Arc<QwenWholeDecoderPlan>,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        adapter: Option<ResolvedLoraAdapterHandle>,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        let profile_started_at = qwen_decode_profile_start();
        let runtime_source = &preflight.runtime_source;
        // Resolved once by whoever built this request, carried as an
        // explicit field: every cache key / job field below reads this same
        // local instead of each independently re-deriving it from a
        // thread-local override + env.
        let backend = request.resolved_runtime.backend();
        let backend_preference = request_backend_override();
        let placement = current_execution_placement();
        let native_gqa = qwen_llm_effective_native_gqa_capability(
            request.resolved_runtime.native_gqa_capability(),
        );
        // The serve-batch owner loads the pack itself; it needs the path, not
        // the content id.
        let runtime_cache_path = canonical_runtime_cache_path(runtime_source.path());
        let serve_batch_graph_config = super::graph_config::qwen_runtime_graph_config(backend);
        // Serve-batch remains an independent owner topology in this change.
        // In particular, do not build and retain an otherwise-idle unified
        // decoder merely to run its encoder before handing decode to the batch
        // engine. Adapter-bearing, streaming, CPU and scheduler-backed requests
        // retain their established split-owner path as well.
        let adapter_active = adapter.is_some();
        let serve_batch_config = Qwen3AsrServeBatchConfig::from_policy(
            request.request_options.serve_batch,
        )
        .filter(|_| {
            !skip_serve_batch
                && !adapter_active
                && serve_batch_graph_config.backend.is_gpu_class()
                && !serve_batch_graph_config.use_scheduler
                && native_gqa.is_validated()
        });
        let native_logits_runtime = prepared_runtime_owner
            .as_ref()
            .as_qwen3_asr()
            .is_some_and(|prepared| prepared.logits_head.supports_native_runtime());
        let unified_runtime_enabled = qwen_unified_runtime_owner_enabled(
            request.resolved_runtime,
            native_gqa,
            native_logits_runtime,
            backend_preference.as_ref(),
            placement,
        ) && !adapter_active
            && !skip_serve_batch
            && serve_batch_config.is_none();
        let qkv_execution_mode = qwen_qkv_execution_mode(
            unified_runtime_enabled,
            qwen_split_loaded_qkv_enabled_with_env(
                std::env::var(QWEN3_GPU_SPLIT_LOADED_QKV_ENV)
                    .ok()
                    .as_deref(),
            ),
        );
        let unified_gpu_runtime = if unified_runtime_enabled {
            Some(self.checkout_unified_gpu_runtime(
                preflight,
                Arc::clone(&prepared_runtime_owner),
                request.resolved_runtime,
                native_gqa,
                qkv_execution_mode,
            )?)
        } else {
            None
        };
        let audio_encoder_started_at = qwen_decode_profile_start();
        let audio_embeddings = match unified_gpu_runtime.as_ref() {
            Some(actor) => self.encode_with_unified_gpu_runtime(actor, mel_features.clone())?,
            None => self.encode_with_owned_audio_encoder_runtime(
                preflight,
                Arc::clone(&prepared_runtime_owner),
                mel_features.clone(),
                backend,
            )?,
        };
        qwen_decode_profile_log_opt("audio_encoder_actor", audio_encoder_started_at);
        let decode_prompt_started_at = qwen_decode_profile_start();
        let decode_prompt = build_qwen3_decode_prompt(
            metadata,
            tokenizer,
            audio_embeddings.row_count,
            &request.request_options,
        )
        .map_err(map_decode_prompt_error)?;
        qwen_decode_profile_log_opt("decode_prompt", decode_prompt_started_at);
        let audio_pad_end = decode_prompt
            .audio_pad_start_index
            .checked_add(decode_prompt.audio_pad_count)
            .ok_or_else(
                || Qwen3AsrGgmlExecutorError::PromptEmbeddingAssemblyFailed {
                    reason: "audio pad position overflowed".to_string(),
                },
            )?;
        let prompt_token_input = Qwen3AsrPromptTokenInput {
            token_ids: decode_prompt.token_ids.clone(),
            audio_rows: audio_embeddings.rows,
            audio_positions: (decode_prompt.audio_pad_start_index..audio_pad_end).collect(),
        };
        if decoder_plan.layer_count() == 0 {
            return Err(Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: "qwen3-asr runtime exposes zero llm layers; at least 1 is required"
                    .to_string(),
            });
        }
        let token_budget_started_at = qwen_decode_profile_start();
        let max_generated_tokens = qwen3_generated_token_budget(
            &request.prepared_audio,
            decode_prompt.token_ids.len(),
            metadata,
        )?;
        let measured_positions =
            crate::capacity::topology::causal_prefix_positions_with_context_cap(
                super::capacity::QWEN3_SELF_KV_STATE_ID,
                decode_prompt.token_ids.len(),
                max_generated_tokens,
                metadata.llm_max_positions,
            )
            .map_err(|error| Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
                reason: error.to_string(),
            })?;
        let kv_capacity = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::QWEN3_SELF_KV_STATE_ID,
        )
        .and_then(|capacity| capacity.validate_measured_logical_positions(measured_positions))
        .map_err(|source| Qwen3AsrGgmlExecutorError::DecoderStateCapacity { source })?;
        qwen_decode_profile_log_opt("decode_token_budget", token_budget_started_at);
        let decode_config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer
                .map(|tokenizer| tokenizer.eos_token_id)
                .unwrap_or(metadata.eos_token_id),
            vocab_size: metadata.vocab_size,
            max_generated_tokens,
        };
        let token_source: &dyn crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource =
            tokenizer
                .map(|tokenizer| tokenizer as _)
                .unwrap_or(&metadata);
        let validate_stacks_started_at = qwen_decode_profile_start();
        self.validate_materialized_block_stacks(
            metadata,
            audio_encoder_weights.layer_count(),
            decoder_plan.layer_count(),
        )?;
        qwen_decode_profile_log_opt(
            "validate_materialized_block_stacks",
            validate_stacks_started_at,
        );
        if let Some(serve_batch_config) = serve_batch_config {
            let serve_batch_started_at = qwen_decode_profile_start();
            let decode_policy = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
                .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                })?;
            let seq2seq_decode_config = build_builtin_seq2seq_decode_policy_config(
                decode_policy,
                &decode_config,
                token_source,
                request.request_options.phrase_bias.as_ref(),
            )
            .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?;
            let result = submit_qwen_serve_batch_job(
                &self.serve_batch_engines,
                serve_batch_config,
                Qwen3AsrServeBatchJob {
                    runtime_cache_path,
                    runtime_source_preflight: preflight.clone(),
                    build_identity:
                        crate::models::ggml_asr_executor::serve_batch_build_identity_for_request(
                            &request.request_options,
                            "qwen",
                            backend,
                            runtime_source,
                        ),
                    resolved_runtime: request.resolved_runtime,
                    native_gqa,
                    metadata,
                    prepared_assets:
                        super::batched_decode::Qwen3AsrServeBatchPreparedAssets::Admitted(
                            prepared_runtime_owner,
                        ),
                    prompt_input: Qwen3AsrServeBatchPromptInput::TokenIds(prompt_token_input),
                    kv_capacity,
                    decode_config: seq2seq_decode_config,
                    text_postprocess_kind: decode_policy.seq2seq_text_postprocess_kind,
                    word_timestamps: request.request_options.word_timestamps,
                    audio_duration_seconds: audio_duration_seconds(&request.prepared_audio),
                    // Owner-thread prefill cannot see this thread's binding;
                    // carry the same explicit `Arc` so chunk-boundary polls
                    // observe cancel regardless of which thread runs them.
                    execution_context: Arc::clone(&request.execution_context),
                },
            )
            .map_err(|error| match error.unavailable_retryable() {
                Some(retryable) => Qwen3AsrGgmlExecutorError::ServeBatchUnavailable {
                    reason: error.to_string(),
                    retryable,
                },
                None => Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                },
            });
            qwen_decode_profile_log_opt("serve_batch_submit", serve_batch_started_at);
            return result;
        }
        let decoder_backend = qwen_runtime_graph_config(backend).backend;
        let whole_decoder_started_at = qwen_decode_profile_start();
        let decoder_actor = match unified_gpu_runtime {
            Some(actor) => Qwen3AsrDecoderActorCheckout::UnifiedGpu(actor),
            None => Qwen3AsrDecoderActorCheckout::Split(self.checkout_decoder_runtime(
                preflight,
                Arc::clone(&prepared_runtime_owner),
                adapter,
                request.resolved_runtime,
                native_gqa,
                qkv_execution_mode,
            )?),
        };
        qwen_decode_profile_log_opt("decoder_actor_checkout", whole_decoder_started_at);

        // The host KV owner is request-scoped, while the whole decoder and its
        // logits runtime stay resident in the pinned owner actor. This mode is
        // derived from the same resolved lane policy used by the decoder
        // constructor, so a resident graph never aliases a materialized host
        // path (and vice versa).
        let kv_cache_started_at = qwen_decode_profile_start();
        let decoder_config = qwen_runtime_graph_config(backend);
        let host_mode = if decoder_config.backend.is_gpu_class() && !decoder_config.use_scheduler {
            Qwen3AsrHostKvMode::ResidentOnly
        } else {
            Qwen3AsrHostKvMode::Materialized
        };
        let kv_cache_spec = super::llm_transformer::resolve_qwen_family_production_kv_cache_policy(
            decoder_backend,
            metadata.llm_head_dim,
        )
        .to_spec();
        let layer_kv_caches = Qwen3AsrHostKvCacheOwner::try_new(
            "qwen3-asr.decoder.self-kv.host",
            metadata.llm_layers,
            kv_capacity,
            metadata.llm_kv_heads,
            metadata.llm_head_dim,
            kv_cache_spec.host,
            host_mode,
        )
        .map_err(|reason| Qwen3AsrGgmlExecutorError::RuntimeContractViolation { reason })?;
        qwen_decode_profile_log_opt("layer_kv_cache_alloc", kv_cache_started_at);
        let tokenizer_for_actor = tokenizer.cloned();
        let decode_config_for_actor = decode_config.clone();
        let phrase_bias_for_actor = request.request_options.phrase_bias.clone();
        let control_for_actor = Arc::clone(&request.execution_context.control);
        let decode_work_progress_for_actor = request
            .execution_context
            .decode_work_progress_observer()
            .cloned();
        let unstable_decode_text_for_actor = request
            .execution_context
            .unstable_decode_text_observer()
            .cloned();
        let fused_top1_hint_allowed = qwen_fused_top1_hint_allowed(
            request.request_options.word_timestamps,
            request
                .request_options
                .word_timestamps_forced_for_diarization,
            request.request_options.phrase_bias.is_some(),
        );
        let greedy_decode_started_at = qwen_decode_profile_start();
        let decode_result = decoder_actor
            .call_mut_fallible(move |state| {
                let token_source_for_actor: &dyn crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource = tokenizer_for_actor
                    .as_ref()
                    .map(|tokenizer| tokenizer as _)
                    .unwrap_or(&metadata);
                let decode_text_token_ids_for_actor = |token_ids: &[u32]| {
                    if let Some(tokenizer) = tokenizer_for_actor.as_ref() {
                        return tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                            Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
                                reason: error.to_string(),
                            }
                        });
                    }
                    Ok(token_ids
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(" "))
                };
                let logits_head = Arc::clone(state.logits_head);
                let mut step_executor = Qwen3AsrPrefillOnlyGreedyStepExecutor {
                    metadata,
                    prompt_input: Qwen3AsrRuntimePromptInput::TokenIds(prompt_token_input),
                    logits_head: logits_head.as_ref(),
                    logits_head_runtime: state.logits_head_runtime,
                    token_embedding_table,
                    layer_kv_caches,
                    kv_capacity,
                    whole_decoder: state.whole_decoder,
                    cache_prompt_tokens: 1,
                    consumed_prefill_step: false,
                    fused_top1_hint_allowed,
                    control: Arc::clone(&control_for_actor),
                };
                let result = run_qwen3_greedy_decode_loop(
                    &decode_config_for_actor,
                    token_source_for_actor,
                    phrase_bias_for_actor.as_ref(),
                    &mut step_executor,
                    &decode_text_token_ids_for_actor,
                    &control_for_actor,
                    decode_work_progress_for_actor.as_ref(),
                    unstable_decode_text_for_actor.as_ref(),
                );
                // A failed compute may poison the reusable graph. Always
                // release session-scoped buffers before the actor goes idle.
                state.whole_decoder.release_session_scoped_buffers();
                result
            })
            .map_err(|error| Self::map_actor_error("decoder", error))?;
        let decode_text_token_ids = |token_ids: &[u32]| {
            if let Some(tokenizer) = tokenizer {
                return tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                });
            }
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "))
        };
        qwen_decode_profile_log_opt("greedy_decode_loop", greedy_decode_started_at);
        // Hitting the token budget without EOT degrades to the generated prefix
        // (mirrors cohere/moonshine) rather than failing the call — so a no-EOT
        // partial cannot kill a live streaming session. The FINAL re-decodes the
        // whole buffer the same way, so it stays consistent with offline `execute()`.
        let postprocess_started_at = qwen_decode_profile_start();
        let (text, generated_tokens, generated_probabilities, stop_reason) = match decode_result {
            Ok(output) => (
                output.text.trim().to_string(),
                output.generated_tokens,
                output.generated_probabilities,
                output.stop_reason,
            ),
            Err(Qwen3AsrGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                generated_tokens,
                generated_probabilities,
                ..
            }) => {
                let text = decode_text_token_ids(&generated_tokens)
                    .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                        reason: error.to_string(),
                    })?
                    .trim()
                    .to_string();
                (
                    text,
                    generated_tokens,
                    generated_probabilities,
                    // Salvaging the prefix is not the same as completing the
                    // decode; keep the shortfall on the record.
                    Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
                )
            }
            // Preserve typed cancel identity through the stringified executor
            // boundary via the stable "canceled by transcription control" marker.
            Err(Qwen3AsrGreedyDecodeError::Canceled) => {
                return Err(Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: Qwen3AsrGreedyDecodeError::Canceled.to_string(),
                });
            }
            Err(error) => {
                return Err(Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                });
            }
        };
        qwen_decode_profile_log_opt("decode_text_postprocess", postprocess_started_at);
        let audio_duration_seconds = audio_duration_seconds(&request.prepared_audio);
        let word_timestamps_started_at = qwen_decode_profile_start();
        let words = if request.request_options.word_timestamps {
            let decode_policy = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
                .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                })?;
            seq2seq_word_timestamps_from_generated_tokens(
                &generated_tokens,
                &generated_probabilities,
                0.0,
                audio_duration_seconds,
                decode_policy.seq2seq_text_postprocess_kind,
                &decode_text_token_ids,
            )
            .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?
        } else {
            Vec::new()
        };
        qwen_decode_profile_log_opt("word_timestamps", word_timestamps_started_at);
        let segments = if words.is_empty() || text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: audio_duration_seconds,
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words,
            }]
        };
        let result = Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text,
                segments,
                longform: None,
                language: None,
                ..Default::default()
            },
            carry_context: None,
            // qwen3-asr emits no intra-decode timestamps -- its one segment
            // spans the whole buffer -- so there is no honest second to anchor
            // the cut to. See `DecodeTruncation::transcript_covers_up_to_seconds`.
            decode_truncation: stop_reason.into_decode_truncation(None),
        });
        qwen_decode_profile_log_opt("decode_with_runtime_assets_total", profile_started_at);
        result
    }

    fn validate_prepared_audio_shape(
        &self,
        metadata: Qwen3AsrExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudioView,
    ) -> Result<(), Qwen3AsrGgmlExecutorError> {
        if prepared_audio.sample_rate_hz != metadata.sample_rate_hz || prepared_audio.channels != 1
        {
            return Err(Qwen3AsrGgmlExecutorError::UnsupportedInputShape {
                expected_sample_rate_hz: metadata.sample_rate_hz,
                sample_rate_hz: prepared_audio.sample_rate_hz,
                channels: prepared_audio.channels,
            });
        }
        Ok(())
    }

    fn validate_materialized_block_stacks(
        &self,
        metadata: Qwen3AsrExecutionMetadata,
        audio_layer_count: usize,
        llm_layer_count: usize,
    ) -> Result<(), Qwen3AsrGgmlExecutorError> {
        let qwen_descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID);
        let qwen_block_stack = qwen_descriptor.as_ref().and_then(|descriptor| {
            match &descriptor.topology_contract.block_stack {
                OpenAsrBlockStackStrategy::Shared(stack) => Some(stack),
                OpenAsrBlockStackStrategy::ArchitectureGraph { .. } => None,
            }
        });
        let layer_resolver = Qwen3AsrLayerCountResolver {
            audio_layers: metadata.audio_layers,
            llm_layers: metadata.llm_layers,
        };
        validate_stage_against_descriptor(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            qwen_block_stack,
            OpenAsrStageRole::Encoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                tensor_name_scope: "audio.blk",
                family_layer_count: audio_layer_count,
            },
            &layer_resolver,
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!(
                    "qwen3-asr audio-encoder block-stack descriptor mismatch: {error:?}"
                ),
            },
        )?;
        validate_stage_against_descriptor(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            qwen_block_stack,
            OpenAsrStageRole::Decoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                tensor_name_scope: "blk",
                family_layer_count: llm_layer_count,
            },
            &layer_resolver,
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!("qwen3-asr decoder block-stack descriptor mismatch: {error:?}"),
            },
        )?;
        Ok(())
    }

    /// Evicts exactly `pack_content_id`'s cached prepared runtime, releasing
    /// resident state left over from a since-replaced pack without touching
    /// any other content identity. Reached through
    /// [`crate::NativeExecutionServices::evict_prepared_runtime_content_id`].
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.audio_encoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.unified_gpu_runtimes
            .evict_where(|key| key.content.pack_content_id == pack_content_id);
        self.lora_adapters.evict_base_content_id(pack_content_id);
        self.runtime_cache_by_path.evict_content_id(pack_content_id);
        shutdown_qwen_serve_batch_engines(&self.serve_batch_engines);
    }

    #[cfg(test)]
    fn build_prepared_runtime(
        &self,
        model_architecture: &str,
        preflight: &GgufRuntimeSourcePreflight,
    ) -> Result<Qwen3AsrPreparedRuntime, Qwen3AsrGgmlExecutorError> {
        build_builtin_prepared_runtime(PreparedRuntimeLookup {
            model_architecture,
            preflight,
            backend: GgmlCpuGraphBackend::Cpu,
        })
        .map_err(map_prepared_runtime_registry_error)?
            .into_qwen3_asr()
            .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!(
                    "prepared runtime registry returned non-qwen runtime for architecture '{model_architecture}'"
                ),
            })
    }
}

// Covers both a genuinely poisoned slot mutex and a build attempt that
// panicked and was caught (mutex stays unpoisoned, slot stays empty,
// retryable) -- see `PreparedRuntimeCache::get_or_try_insert_with`. Either way
// the cache could not deliver a prepared runtime for this attempt; the
// caller's next request retries clean.
fn qwen_runtime_cache_slot_unavailable() -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed {
        reason:
            "qwen runtime cache slot unavailable (poisoned lock or a caught build panic); retry"
                .to_string(),
    }
}

fn qwen3_generated_token_budget(
    prepared_audio: &GgmlAsrPreparedAudioView,
    prompt_tokens: usize,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<usize, Qwen3AsrGgmlExecutorError> {
    let sample_rate = usize::try_from(prepared_audio.sample_rate_hz).map_err(|_| {
        Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
            reason: format!(
                "sample_rate_hz={} does not fit usize",
                prepared_audio.sample_rate_hz
            ),
        }
    })?;
    qwen3_budget_for_shape(
        prepared_audio.samples_f32.len(),
        sample_rate,
        prompt_tokens,
        metadata.llm_max_positions,
    )
    .map_err(|error| Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
        reason: error.to_string(),
    })
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudioView) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

fn map_decode_prompt_error(error: Qwen3AsrDecodePromptError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::DecodePromptConstructionFailed {
        reason: error.to_string(),
    }
}

fn map_mel_frontend_error(error: Qwen3AsrMelFrontendError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::MelFrontendFailed {
        reason: error.to_string(),
    }
}

fn map_audio_encoder_error(error: Qwen3AsrAudioEncoderError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::AudioEncoderFailed {
        reason: error.to_string(),
    }
}

fn map_prepared_runtime_error(error: Qwen3AsrPreparedRuntimeError) -> Qwen3AsrGgmlExecutorError {
    match error {
        Qwen3AsrPreparedRuntimeError::RuntimeContractViolation { reason } => {
            Qwen3AsrGgmlExecutorError::RuntimeContractViolation { reason }
        }
        Qwen3AsrPreparedRuntimeError::RuntimeMetadataReadFailed { reason } => {
            Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::MelFrontendFailed { reason } => {
            Qwen3AsrGgmlExecutorError::MelFrontendFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::AudioEncoderFailed { reason } => {
            Qwen3AsrGgmlExecutorError::AudioEncoderFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::TokenEmbeddingPrefillFailed { reason } => {
            Qwen3AsrGgmlExecutorError::TokenEmbeddingPrefillFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::LlmLogitsHeadFailed { reason } => {
            Qwen3AsrGgmlExecutorError::LlmLogitsHeadFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::LlmTransformerDecodeStepFailed { reason } => {
            Qwen3AsrGgmlExecutorError::LlmTransformerDecodeStepFailed { reason }
        }
    }
}

fn map_prepared_runtime_registry_error(
    error: BuiltinPreparedRuntimeRegistryError,
) -> Qwen3AsrGgmlExecutorError {
    match error {
        BuiltinPreparedRuntimeRegistryError::Qwen3AsrBuild { source } => {
            map_prepared_runtime_error(source)
        }
        other => Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
            reason: other.to_string(),
        },
    }
}

/// Resolves a qwen block-stack stage's `layer_count_hparam` to the count parsed
/// from the GGUF hparams (NOT `layers.len()` — see the [`LayerCountResolver`]
/// honesty contract), so `validate_stage_against_descriptor` can cross-check the
/// materialized layer count against the descriptor's declared key. Carries both
/// stages' counts so one resolver serves the audio-encoder and LLM-decoder gates.
struct Qwen3AsrLayerCountResolver {
    audio_layers: usize,
    llm_layers: usize,
}

impl LayerCountResolver for Qwen3AsrLayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            QWEN3_AUDIO_LAYERS_KEY => Some(self.audio_layers),
            QWEN3_LLM_LAYERS_KEY => Some(self.llm_layers),
            _ => None,
        }
    }
}

/// Whether this request may take the fused device-side top-1 (hint-only) lane.
///
/// Hint-only steps report `probability: 0.0` and never materialize a host
/// logit row, so they are only taken when nothing about the request consumes
/// that row:
///
/// - USER-REQUESTED `word_timestamps` feeds `generated_probabilities` into
///   the emitted per-word confidences, so a zeroed probability would change
///   the output. The CLI transcribe path, however, force-enables
///   `word_timestamps` for EVERY non-whisper family purely to obtain word
///   anchors for cue segmentation / diarization turn-splitting and then
///   strips the words from the result (`word_timestamps_forced_for_diarization`
///   marks exactly that case). Word anchor TIMES are index-positional
///   (`seq2seq_word_timestamps_from_generated_tokens` never reads
///   probabilities for timing) and the cue splitter emits `confidence: None`,
///   so on the forced-and-stripped lane the probabilities are invisible to
///   the output and the fused lane stays byte-identical.
/// - A phrase-bias request makes the shared driver require the full row to
///   apply biases (a hint-only step would fail closed on `EmptyStepLogits`).
///
/// Both gates are decided per request; the fused head stays resident in the
/// whole-decoder arena either way so a cached decoder serves both kinds of
/// request.
fn qwen_fused_top1_hint_allowed(
    word_timestamps: bool,
    word_timestamps_forced_for_diarization: bool,
    has_phrase_bias: bool,
) -> bool {
    (!word_timestamps || word_timestamps_forced_for_diarization) && !has_phrase_bias
}

/// Prefill output for the shared greedy driver's step 0: the host logits row
/// for the first generated token, or (on the fused Metal/GPU lane) a device
/// argmax hint with no host row -- mirrors
/// `moss_transcribe_diarize::llm_decoder::MossTdPrefillOutput`.
struct Qwen3AsrPrefillStepOutput {
    logits: Vec<f32>,
    greedy_token_hint: Option<u32>,
}

enum Qwen3AsrRuntimePromptInput {
    Host(super::llm_prefill::Qwen3AsrLlmPrefillInput),
    TokenIds(Qwen3AsrPromptTokenInput),
}

impl Qwen3AsrRuntimePromptInput {
    fn token_count(&self) -> usize {
        match self {
            Self::Host(input) => input.token_count,
            Self::TokenIds(input) => input.token_ids.len(),
        }
    }
}

struct Qwen3AsrPrefillOnlyGreedyStepExecutor<'a> {
    metadata: Qwen3AsrExecutionMetadata,
    prompt_input: Qwen3AsrRuntimePromptInput,
    logits_head: &'a Qwen3AsrLlmLogitsHead,
    logits_head_runtime: &'a mut Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding_table: Arc<crate::models::mapped_token_embedding::MappedTokenEmbeddingTable>,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    whole_decoder: &'a mut Qwen3AsrLlmWholeDecoderGraphExecutor,
    cache_prompt_tokens: usize,
    consumed_prefill_step: bool,
    /// See [`qwen_fused_top1_hint_allowed`]. `true` only for requests that
    /// never consume a host logit row (no word timestamps, no phrase bias).
    fused_top1_hint_allowed: bool,
    /// Explicit cancel/pause/resume control for this decode -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    control: Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for Qwen3AsrPrefillOnlyGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<
        Seq2SeqGreedyDecodeStepLogitsOutput,
        crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError,
    > {
        if !self.consumed_prefill_step && input.step_index == 0 && input.generated_tokens.is_empty()
        {
            // Preserve typed cancel from the prefill chunk loop so the shared
            // greedy driver (and dispatch_error_to_backend) see Canceled, not a
            // generic DecoderStepFailed.
            let prefill = self.prefill_prompt_and_compute_last_logits().map_err(|error| {
                match error {
                    Qwen3AsrGreedyDecodeError::Canceled => {
                        crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::Canceled
                    }
                    other => crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                        reason: other.to_string(),
                    },
                }
            })?;
            self.consumed_prefill_step = true;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: prefill.logits,
                greedy_token_hint: prefill.greedy_token_hint,
            });
        }

        if input.generated_tokens.is_empty() {
            return Err(
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr generated token history is unexpectedly empty".to_string(),
                },
            );
        }

        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr decode cache position underflowed".to_string(),
                }
            })?;
        if let Some(token_id) = self
            .decode_step_reused_top1(input.generated_tokens, cache_position)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?
        {
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(token_id),
            });
        }
        let token_id = self
            .last_generated_token_id(input.generated_tokens)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;
        let hidden = self
            .run_llm_token_with_kv(token_id, cache_position)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;

        let logits = self
            .logits_head_runtime
            .compute_logits_for_last_hidden(self.logits_head, &hidden)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

/// Poll the request's explicit control at a host-cache prefill chunk
/// boundary. Distinct from a graph/compute failure so cancel maps to
/// [`crate::BackendError::TranscriptionCanceled`] end-to-end. Never a
/// thread-local -- see [`crate::RequestExecutionContext`].
fn ensure_prefill_chunk_not_canceled(
    control: &Arc<crate::api::backend::TranscriptionControl>,
) -> Result<(), Qwen3AsrGreedyDecodeError> {
    if control.is_canceled() {
        return Err(Qwen3AsrGreedyDecodeError::Canceled);
    }
    Ok(())
}

fn map_prefill_graph_error(error: GgmlCpuGraphError) -> Qwen3AsrGreedyDecodeError {
    match error {
        GgmlCpuGraphError::Canceled => Qwen3AsrGreedyDecodeError::Canceled,
        other => Qwen3AsrGreedyDecodeError::DecoderStepFailed {
            reason: other.to_string(),
        },
    }
}

impl Qwen3AsrPrefillOnlyGreedyStepExecutor<'_> {
    /// On the resident Metal/GPU reuse graph, return the decoder's device-side
    /// argmax for the next token directly (zero host logits materialization,
    /// zero full-vocab readback), or `None` to stay on the host logits path --
    /// mirrors `moss_transcribe_diarize::llm_decoder::decode_step_reused_top1`.
    /// Gated on [`Self::fused_top1_hint_allowed`] because qwen (unlike moss)
    /// serves requests that consume the host row (word timestamps, phrase
    /// bias); those keep the byte-identical host path.
    fn decode_step_reused_top1(
        &mut self,
        generated_tokens: &[u32],
        cache_position: usize,
    ) -> Result<Option<u32>, Qwen3AsrGreedyDecodeError> {
        if !self.fused_top1_hint_allowed
            || !self.whole_decoder.supports_device_token_embedding()
            || !self.whole_decoder.supports_fused_top1()
        {
            return Ok(None);
        }
        if self.layer_kv_caches.is_empty() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr decoder has no layer KV caches".to_string(),
            });
        }
        let token_id = self.last_generated_token_id(generated_tokens)?;
        let step = self
            .whole_decoder
            .run_token_step_reused_batched_top1(
                &[token_id],
                &[cache_position],
                1_000_000.0,
                self.kv_capacity.resident_positions(),
            )
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        Ok(Some(step.token_id))
    }

    fn prefill_prompt_and_compute_last_logits(
        &mut self,
    ) -> Result<Qwen3AsrPrefillStepOutput, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        let token_count = self.prompt_input.token_count();
        if token_count == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill token count is zero".to_string(),
            });
        }
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        if let Qwen3AsrRuntimePromptInput::TokenIds(prompt) = &self.prompt_input
            && let Some(final_hidden) = self
                .whole_decoder
                .run_token_prefill_auto_last_hidden(
                    &prompt.token_ids,
                    &prompt.audio_rows,
                    &prompt.audio_positions,
                    &self.layer_kv_caches,
                    self.kv_capacity,
                    1_000_000.0,
                    &self.control,
                )
                .map_err(map_prefill_graph_error)?
        {
            self.cache_prompt_tokens = token_count;
            qwen_decode_profile_log_opt("prefill_prompt_resident_token_ids", profile_started_at);
            if self.fused_top1_hint_allowed
                && let Some(token_id) = self
                    .whole_decoder
                    .fused_logits_top1_from_hidden(&final_hidden)
                    .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: error.to_string(),
                    })?
            {
                return Ok(Qwen3AsrPrefillStepOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_head_runtime
                .compute_logits_for_last_hidden(self.logits_head, &final_hidden)
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            return Ok(Qwen3AsrPrefillStepOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        if let Qwen3AsrRuntimePromptInput::TokenIds(prompt) = &self.prompt_input {
            let token_rows = self
                .token_embedding_table
                .gather_rows(&prompt.token_ids)
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            let embeddings = build_qwen3_prompt_embeddings_with_audio_positions(
                prompt.token_ids.len(),
                &prompt.audio_positions,
                self.token_embedding_table.d_model(),
                token_rows,
                &prompt.audio_rows,
            )
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
            self.prompt_input = Qwen3AsrRuntimePromptInput::Host(
                build_qwen3_llm_prefill_input(embeddings).map_err(|error| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: error.to_string(),
                    }
                })?,
            );
        }
        // Prefer the shared resident-KV bulk seed used by mimo/firered/moss.
        // On single-GPU backends (CUDA/Metal/HIP) this is one batched prefill
        // into the same arena `run_step_auto` will reuse for decode — avoiding
        // both the historical serial host-step storm and the host-cache/reuse
        // mix-up. Discrete-GPU wide steps fail closed to non-flash attention
        // inside the shared executor (see llm_prefill_uses_flash_attention).
        let resident_started_at = qwen_decode_profile_start();
        let prefill_input = Self::host_prefill_input(&self.prompt_input)?;
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prefill_input.token_major_embeddings,
                token_count,
                &self.layer_kv_caches,
                self.kv_capacity,
                1_000_000.0,
                &self.control,
            )
            .map_err(map_prefill_graph_error)?
        {
            self.cache_prompt_tokens = token_count;
            qwen_decode_profile_log_opt("prefill_prompt_resident_bulk", resident_started_at);
            // Fused device argmax for the first generated token too (mirrors
            // moss's prefill): only when this request never needs the host
            // row, same gate as the per-token decode steps.
            if self.fused_top1_hint_allowed
                && let Some(token_id) = self
                    .whole_decoder
                    .fused_logits_top1_from_hidden(&final_hidden)
                    .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: error.to_string(),
                    })?
            {
                qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
                return Ok(Qwen3AsrPrefillStepOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let result = self
                .logits_head_runtime
                .compute_logits_for_last_hidden(self.logits_head, &final_hidden)
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                });
            qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
            return result.map(|logits| Qwen3AsrPrefillStepOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let Some(chunk_size) = self
            .whole_decoder
            .safe_host_cache_prefill_chunk_size_for(token_count)
        else {
            // No multi-query host-cache width (typically native GQA off). Full
            // bulk host prefill is still correct under the fail-closed flash
            // policy and matches mimo/firered's non-reuse fallback — prefer it
            // over the historical serial launch storm.
            let result = self.prefill_prompt_bulk_host_and_compute_last_logits();
            qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
            return result.map(|logits| Qwen3AsrPrefillStepOutput {
                logits,
                greedy_token_hint: None,
            });
        };
        let result = self.prefill_prompt_chunked_and_compute_last_logits(chunk_size);
        qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
        result.map(|logits| Qwen3AsrPrefillStepOutput {
            logits,
            greedy_token_hint: None,
        })
    }

    fn prefill_prompt_bulk_host_and_compute_last_logits(
        &mut self,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        let prefill_input = Self::host_prefill_input(&self.prompt_input)?;
        let token_count = prefill_input.token_count;
        let step = self
            .whole_decoder
            .run_prefill(
                &prefill_input.token_major_embeddings,
                token_count,
                1_000_000.0,
            )
            .map_err(map_prefill_graph_error)?;
        qwen_decode_profile_log_prefill_chunk(0, token_count, profile_started_at);
        let result = self.write_prefill_step_outputs_and_compute_last_logits(token_count, step);
        qwen_decode_profile_log_opt("prefill_prompt_bulk_host", profile_started_at);
        result
    }

    fn prefill_prompt_chunked_and_compute_last_logits(
        &mut self,
        chunk_size: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        if chunk_size == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill chunk size is zero".to_string(),
            });
        }
        let (token_count, hidden_size) = {
            let prefill_input = Self::host_prefill_input(&self.prompt_input)?;
            (prefill_input.token_count, prefill_input.hidden_size)
        };
        if token_count <= chunk_size {
            let chunk_started_at = qwen_decode_profile_start();
            let step = {
                let prefill_input = Self::host_prefill_input(&self.prompt_input)?;
                self.whole_decoder
                    .run_prefill(
                        &prefill_input.token_major_embeddings,
                        token_count,
                        1_000_000.0,
                    )
                    .map_err(map_prefill_graph_error)?
            };
            qwen_decode_profile_log_prefill_chunk(0, token_count, chunk_started_at);
            let result = self.write_prefill_step_outputs_and_compute_last_logits(token_count, step);
            qwen_decode_profile_log_opt("prefill_prompt_chunked", profile_started_at);
            return result;
        }
        let require_even_chunks = self.whole_decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden = None;
        while position_offset < token_count {
            // L1.2 cooperative cancel between host-cache prefill chunks.
            ensure_prefill_chunk_not_canceled(&self.control)?;
            let remaining = token_count - position_offset;
            let chunk_len = if require_even_chunks {
                super::even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = position_offset.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len = chunk_len.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden width overflowed".to_string(),
                }
            })?;
            let hidden_end = hidden_start.checked_add(hidden_len).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden end overflowed".to_string(),
                }
            })?;
            let total_token_count = position_offset.checked_add(chunk_len).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk span overflowed".to_string(),
                }
            })?;
            let chunk_started_at = qwen_decode_profile_start();
            let step = {
                let prefill_input = Self::host_prefill_input(&self.prompt_input)?;
                self.whole_decoder
                    .run_prefill_chunk(
                        &prefill_input.token_major_embeddings[hidden_start..hidden_end],
                        chunk_len,
                        position_offset,
                        total_token_count,
                        &self.layer_kv_caches,
                        1_000_000.0,
                    )
                    .map_err(map_prefill_graph_error)?
            };
            qwen_decode_profile_log_prefill_chunk(position_offset, chunk_len, chunk_started_at);
            final_hidden =
                Some(self.write_prefill_chunk_outputs(position_offset, chunk_len, step)?);
            position_offset = total_token_count;
        }
        self.cache_prompt_tokens = token_count;
        let result = self
            .logits_head_runtime
            .compute_logits_for_last_hidden(
                self.logits_head,
                &final_hidden.ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill produced no final hidden state".to_string(),
                })?,
            )
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            });
        qwen_decode_profile_log_opt("prefill_prompt_chunked", profile_started_at);
        result
    }

    fn write_prefill_step_outputs_and_compute_last_logits(
        &mut self,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let final_hidden = self.write_prefill_chunk_outputs(0, token_count, step)?;
        self.cache_prompt_tokens = token_count;
        self.logits_head_runtime
            .compute_logits_for_last_hidden(self.logits_head, &final_hidden)
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })
    }

    fn write_prefill_chunk_outputs(
        &mut self,
        position_offset: usize,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        if step.layer_kv.len() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill layer-KV count mismatch".to_string(),
            });
        }
        let kv_row_width = self
            .metadata
            .llm_kv_heads
            .checked_mul(self.metadata.llm_head_dim)
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill KV row width overflowed".to_string(),
            })?;
        for token_position in 0..token_count {
            let absolute_position =
                position_offset.checked_add(token_position).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill absolute row overflowed".to_string(),
                    }
                })?;
            let row_start = token_position.checked_mul(kv_row_width).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill KV row offset overflowed".to_string(),
                }
            })?;
            let row_end = row_start.checked_add(kv_row_width).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill KV row end overflowed".to_string(),
                }
            })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill K row out of bounds".to_string(),
                    }
                })?;
                let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill V row out of bounds".to_string(),
                    }
                })?;
                self.layer_kv_caches[layer_index]
                    .write(absolute_position, key_row, value_row)
                    .map_err(|reason| Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason })?;
            }
        }
        let hidden_size = Self::host_prefill_input(&self.prompt_input)?.hidden_size;
        let final_hidden_start = token_count
            .checked_sub(1)
            .and_then(|position| position.checked_mul(hidden_size))
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final-hidden offset overflowed".to_string(),
            })?;
        let final_hidden_end = final_hidden_start.checked_add(hidden_size).ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final-hidden end overflowed".to_string(),
            }
        })?;
        let final_hidden = step
            .hidden
            .get(final_hidden_start..final_hidden_end)
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final hidden out of bounds".to_string(),
            })?
            .to_vec();
        Ok(final_hidden)
    }

    fn host_prefill_input(
        prompt_input: &Qwen3AsrRuntimePromptInput,
    ) -> Result<&super::llm_prefill::Qwen3AsrLlmPrefillInput, Qwen3AsrGreedyDecodeError> {
        match prompt_input {
            Qwen3AsrRuntimePromptInput::Host(input) => Ok(input),
            Qwen3AsrRuntimePromptInput::TokenIds(_) => {
                Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr host prefill input was not materialized".to_string(),
                })
            }
        }
    }

    fn run_llm_layers_with_kv(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        if self.metadata.llm_heads == 0 || self.metadata.llm_kv_heads == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr invalid llm head metadata: llm_heads={} llm_kv_heads={}",
                    self.metadata.llm_heads, self.metadata.llm_kv_heads
                ),
            });
        }

        let started_at = if qwen_decode_profile_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        // `run_step_auto` reuses the built decode graph across tokens only on
        // the direct single-GPU lane (`is_gpu_class && !scheduler`). CPU
        // compute and multi-backend scheduler paths rebuild the growing-KV
        // graph each token because their in-place-KV reuse path is not
        // byte-identical -- see `supports_graph_reuse`'s doc comment.
        let step = self
            .whole_decoder
            .run_step_auto(
                &hidden,
                cache_position,
                &self.layer_kv_caches,
                self.kv_capacity,
                1_000_000.0,
            )
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
            self.layer_kv_caches[layer_index]
                .write(cache_position, projected_k, projected_v)
                .map_err(|reason| Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason })?;
        }
        if let Some(started_at) = started_at {
            eprintln!(
                "openasr_qwen_decode_profile: cache_position={} layers={} total_us={} build_us={} compute_us={}",
                cache_position,
                step.layer_kv.len(),
                started_at.elapsed().as_micros(),
                step.build_micros,
                step.compute_micros,
            );
        }
        Ok(step.hidden)
    }

    fn run_llm_token_with_kv(
        &mut self,
        token_id: u32,
        cache_position: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        if self.whole_decoder.supports_device_token_embedding() {
            let started_at = if qwen_decode_profile_enabled() {
                Some(Instant::now())
            } else {
                None
            };
            let step = self
                .whole_decoder
                .run_token_step_auto(
                    token_id,
                    cache_position,
                    &self.layer_kv_caches,
                    self.kv_capacity,
                    1_000_000.0,
                )
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?
                .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen device token embedding became unavailable".to_string(),
                })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                self.layer_kv_caches[layer_index]
                    .write(cache_position, projected_k, projected_v)
                    .map_err(|reason| Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason })?;
            }
            if let Some(started_at) = started_at {
                eprintln!(
                    "openasr_qwen_decode_profile: cache_position={} layers={} total_us={} build_us={} compute_us={} input=token_ids",
                    cache_position,
                    step.layer_kv.len(),
                    started_at.elapsed().as_micros(),
                    step.build_micros,
                    step.compute_micros,
                );
            }
            return Ok(step.hidden);
        }
        let hidden = self
            .token_embedding_table
            .gather_rows(&[token_id])
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        self.run_llm_layers_with_kv(hidden, cache_position)
    }

    fn last_generated_token_id(
        &self,
        generated_tokens: &[u32],
    ) -> Result<u32, Qwen3AsrGreedyDecodeError> {
        generated_tokens.last().copied().ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr generated token history is unexpectedly empty".to_string(),
            }
        })
    }
}

impl GgmlAsrViewExecutor for Qwen3AsrGgmlExecutor {
    fn adapter_binding_strategy(
        &self,
    ) -> crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy {
        crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy::Qwen3AsrLoraV1
    }

    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        Qwen3AsrGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        QWEN3_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_qwen3_decoder_state,
                super::capacity::QWEN3_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn replan_streaming_decoder_state(
        &self,
        selected_family: &crate::GgmlFamilyAdapterDescriptor,
        input: &crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderState, GgmlAsrExecutionError> {
        if let Some(owner) = self
            .runtime_cache_by_path
            .ready_for_preflight(input.preflight)
            && let Some(prepared) = owner.as_ref().as_qwen3_asr()
        {
            let plan =
                super::capacity::plan_qwen3_decoder_state_with_prepared_runtime(input, prepared)?;
            return Ok(
                crate::models::ggml_asr_executor::GgmlAsrDecoderState::planned(
                    plan,
                    input.envelope,
                ),
            );
        }
        self.decoder_state_contract(selected_family)?
            .plan(input)
            .map_err(Into::into)
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: no token observer, batch worker allowed.
        self.execute_inner(request, false)
            .map_err(|error| qwen_execute_error_to_ggml(error, request.selected_family.adapter_id))
    }

    fn unload_idle_state(&self) {
        shutdown_qwen_serve_batch_engines(&self.serve_batch_engines);
        self.audio_encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.unified_gpu_runtimes.clear();
        self.lora_adapters.clear();
        self.runtime_cache_by_path.clear();
    }
}

impl Qwen3AsrGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| qwen_execute_error_to_ggml(error, request.selected_family.adapter_id))
    }
}

fn qwen_execute_error_to_ggml(
    error: Qwen3AsrGgmlExecutorError,
    adapter_id: &'static str,
) -> GgmlAsrExecutionError {
    match error {
        Qwen3AsrGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: QWEN3_EXECUTOR_ID,
            adapter_id,
            reason: error.to_string(),
        },
    }
}

impl GgmlAsrStreamingExecutor for Qwen3AsrGgmlExecutor {
    fn adapter_binding_strategy(
        &self,
    ) -> crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy {
        crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy::Qwen3AsrLoraV1
    }

    fn executor_id(&self) -> &'static str {
        QWEN3_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            QWEN3_STREAMING_EXECUTOR_ID,
            QWEN3_ASR_GGML_ADAPTER_ID,
            "qwen3-asr",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            Qwen3AsrGgmlExecutor::execute_streaming,
        )
    }

    fn unload_idle_state(&self) {
        shutdown_qwen_serve_batch_engines(&self.serve_batch_engines);
        self.audio_encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.unified_gpu_runtimes.clear();
        self.lora_adapters.clear();
        self.runtime_cache_by_path.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use super::super::runtime_contract::QWEN3_LLM_VOCAB_SIZE_KEY;
    use super::super::tensor_names::{
        AUDIO_CONV_OUT_BIAS, AUDIO_CONV_OUT_WEIGHT, AUDIO_CONV1_BIAS, AUDIO_CONV1_WEIGHT,
        AUDIO_CONV2_BIAS, AUDIO_CONV2_WEIGHT, AUDIO_CONV3_BIAS, AUDIO_CONV3_WEIGHT,
        AUDIO_LN_POST_BIAS, AUDIO_LN_POST_WEIGHT, AUDIO_MEL_FILTERS, AUDIO_MEL_WINDOW,
        AUDIO_PROJ1_BIAS, AUDIO_PROJ1_WEIGHT, AUDIO_PROJ2_BIAS, AUDIO_PROJ2_WEIGHT,
        OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT, TOKEN_EMBD_WEIGHT, audio_layer_tensor_names,
        llm_layer_tensor_names,
    };

    use crate::arch::builtin_adapter_descriptor;
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrExecutionViewRequest,
        GgmlAsrPreparedAudioView, LongFormOptions,
    };

    use super::*;

    fn qwen_metadata_with_llm_layers(llm_layers: usize) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        metadata.insert("qwen3-asr.sample_rate".to_string(), "16000".to_string());
        metadata.insert("qwen3-asr.n_mels".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.n_fft".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.win_length".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.hop_length".to_string(), "160".to_string());
        metadata.insert("qwen3-asr.audio.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.audio.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.audio.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.llm.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.n_kv_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.head_dim".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.llm.n_layers".to_string(), llm_layers.to_string());
        metadata.insert("qwen3-asr.llm.vocab_size".to_string(), "32".to_string());
        metadata.insert("qwen3-asr.llm.max_pos".to_string(), "256".to_string());
        metadata.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            "2".to_string(),
        );
        metadata.insert("qwen3-asr.audio_end_token_id".to_string(), "3".to_string());
        metadata.insert("qwen3-asr.audio_pad_token_id".to_string(), "4".to_string());
        metadata.insert("qwen3-asr.eos_token_id".to_string(), "0".to_string());
        metadata.insert("qwen3-asr.pad_token_id".to_string(), "6".to_string());
        metadata
    }

    fn qwen_metadata() -> BTreeMap<String, String> {
        qwen_metadata_with_llm_layers(2)
    }

    /// The fused hint-only lane must never serve a request that consumes the
    /// host logit row: USER-REQUESTED word timestamps feed per-token
    /// probabilities into the emitted word confidences, and phrase bias needs
    /// the full row for the driver to apply biases (a hint-only step would
    /// fail closed). Word timestamps that were only force-enabled for cue
    /// segmentation / diarization anchors (and are stripped from the result)
    /// never surface a probability, so that lane stays fused-eligible --
    /// otherwise the standard CLI transcribe path (which always forces them
    /// for non-whisper families) would never take the fused lane at all.
    #[test]
    fn fused_top1_hint_gate_rejects_row_consuming_requests() {
        // Plain request: fused allowed.
        assert!(qwen_fused_top1_hint_allowed(false, false, false));
        // Forced-and-stripped word anchors (the standard CLI transcribe
        // shape): probabilities are invisible, fused stays allowed.
        assert!(qwen_fused_top1_hint_allowed(true, true, false));
        // User-requested word timestamps: confidences are emitted, host row
        // required.
        assert!(!qwen_fused_top1_hint_allowed(true, false, false));
        // Phrase bias always requires the host row.
        assert!(!qwen_fused_top1_hint_allowed(false, false, true));
        assert!(!qwen_fused_top1_hint_allowed(true, true, true));
    }

    fn add_audio_layer_shapes(spec: TinyGgufFixtureSpec, layer_idx: usize) -> TinyGgufFixtureSpec {
        let names = audio_layer_tensor_names(layer_idx);
        spec.with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_norm_bias, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_bias, [16_u64])
            .with_tensor_shape(names.attn_k_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_k_bias, [16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_bias, [16_u64])
            .with_tensor_shape(names.attn_out_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_out_bias, [16_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            .with_tensor_shape(names.ffn_norm_bias, [16_u64])
            .with_tensor_shape(names.ffn_up_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_up_bias, [32_u64])
            .with_tensor_shape(names.ffn_down_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_down_bias, [16_u64])
    }

    fn add_llm_layer_shapes(spec: TinyGgufFixtureSpec, layer_idx: usize) -> TinyGgufFixtureSpec {
        let names = llm_layer_tensor_names(layer_idx);
        spec.with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_k_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_output_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_norm_weight, [8_u64])
            .with_tensor_shape(names.attn_k_norm_weight, [8_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            // ggml [in, out]: gate/up = [d_model, ffn_dim], down = [ffn_dim, d_model]
            .with_tensor_shape(names.ffn_gate_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_up_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_down_weight, [32_u64, 16_u64])
    }

    fn qwen_tensor_ready_fixture_spec_with_llm_layers(llm_layers: usize) -> TinyGgufFixtureSpec {
        let mut spec = TinyGgufFixtureSpec::new(qwen_metadata_with_llm_layers(llm_layers))
            .with_tensor_shape(AUDIO_MEL_FILTERS, [8_u64, 201_u64])
            .with_tensor_shape(AUDIO_MEL_WINDOW, [400_u64])
            .with_tensor_shape(AUDIO_CONV1_WEIGHT, [3_u64, 3_u64, 1_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV1_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV2_WEIGHT, [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV2_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV3_WEIGHT, [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV3_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV_OUT_WEIGHT, [4_u64, 16_u64])
            .with_tensor_shape(AUDIO_CONV_OUT_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_LN_POST_WEIGHT, [16_u64])
            .with_tensor_shape(AUDIO_LN_POST_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_PROJ1_WEIGHT, [16_u64, 16_u64])
            .with_tensor_shape(AUDIO_PROJ1_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_PROJ2_WEIGHT, [16_u64, 16_u64])
            .with_tensor_shape(AUDIO_PROJ2_BIAS, [16_u64])
            .with_tensor_shape(TOKEN_EMBD_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_NORM_WEIGHT, [16_u64]);
        for layer_idx in 0..2 {
            spec = add_audio_layer_shapes(spec, layer_idx);
        }
        for layer_idx in 0..llm_layers {
            spec = add_llm_layer_shapes(spec, layer_idx);
        }
        spec
    }

    fn qwen_tensor_ready_fixture_spec() -> TinyGgufFixtureSpec {
        qwen_tensor_ready_fixture_spec_with_llm_layers(2)
    }

    fn qwen_request(runtime_source_path: PathBuf) -> GgmlAsrExecutionViewRequest<'static> {
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &runtime_source_path,
            )
            .expect("qwen runtime fixture must pass preflight");
        qwen_request_from_preflight(runtime_source_preflight)
    }

    fn qwen_request_from_preflight(
        runtime_source_preflight: crate::GgufRuntimeSourcePreflight,
    ) -> GgmlAsrExecutionViewRequest<'static> {
        GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(vec![0.0; 160]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn plan_qwen_request_decoder_state(request: &mut GgmlAsrExecutionViewRequest<'_>) {
        let preflight = request.runtime_source_preflight();
        // The shared view intentionally borrows PCM, while the common offline
        // planning helper accepts the public owned DTO. This copy is test-only
        // and keeps the production execute_view path zero-copy.
        let prepared_audio = crate::models::ggml_asr_executor::GgmlAsrPreparedAudio {
            sample_rate_hz: request.prepared_audio.sample_rate_hz,
            channels: request.prepared_audio.channels,
            samples_f32: request.prepared_audio.samples_f32.to_vec(),
        };
        let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput::for_offline_request(
            preflight,
            &prepared_audio,
            &request.request_options,
            request.resolved_runtime.backend(),
        )
        .expect("build qwen decoder-state planning input");
        let plan = super::super::capacity::plan_qwen3_decoder_state(&planning_input)
            .expect("plan qwen decoder state");
        request.decoder_state =
            crate::models::ggml_asr_executor::GgmlAsrDecoderState::planned_for_test(
                plan,
                planning_input.envelope,
            );
    }

    fn exactly_addressable_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(crate::device::execution_route::ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
            addressability:
                crate::device::execution_route::DeviceAddressability::ExactlyAddressable {
                    physical_key: crate::device::execution_route::PhysicalResourceKey::new(
                        "0000:01:00.0",
                    )
                    .expect("physical key"),
                },
        })
    }

    #[test]
    fn split_loaded_qkv_is_limited_to_unified_exact_cuda_vulkan_runtime() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let preference = exactly_addressable_preference(provider);
            let resolved = ResolvedFamilyRuntimeInput::resolve(
                Some(preference.clone()),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            );
            assert!(qwen_unified_runtime_owner_enabled(
                resolved,
                GgmlNativeGqaCapability::Validated,
                true,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
            assert_eq!(
                qwen_qkv_execution_mode(true, true),
                QwenQkvExecutionMode::SplitLoaded
            );
            assert_eq!(
                qwen_qkv_execution_mode(true, false),
                QwenQkvExecutionMode::FusedArena
            );
        }

        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Hip,
            ExecutionProvider::Accelerator,
            ExecutionProvider::Unknown,
        ] {
            let preference = exactly_addressable_preference(provider);
            let resolved = ResolvedFamilyRuntimeInput::resolve(
                Some(preference.clone()),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            );
            assert!(!qwen_unified_runtime_owner_enabled(
                resolved,
                resolved.native_gqa_capability(),
                true,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
        }
        assert_eq!(
            qwen_qkv_execution_mode(false, true),
            QwenQkvExecutionMode::FusedArena
        );
    }

    #[test]
    fn qkv_execution_mode_partitions_unified_runtime_cache_identity() {
        let content = PackContentKey::new("sha256:qwen-qkv-mode-fixture");
        let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
        let fused = Qwen3AsrRuntimeOwnerCacheKey {
            content: content.clone(),
            lane: lane.clone(),
            native_gqa: GgmlNativeGqaCapability::Validated,
            qkv_execution_mode: QwenQkvExecutionMode::FusedArena,
        };
        let split = Qwen3AsrRuntimeOwnerCacheKey {
            content,
            lane,
            native_gqa: GgmlNativeGqaCapability::Validated,
            qkv_execution_mode: QwenQkvExecutionMode::SplitLoaded,
        };
        assert_ne!(fused, split);
    }

    #[test]
    fn split_loaded_qkv_defaults_on_and_allows_explicit_opt_out() {
        assert!(qwen_split_loaded_qkv_enabled_with_env(None));
        for raw in ["1", "true", "yes", "on"] {
            assert!(qwen_split_loaded_qkv_enabled_with_env(Some(raw)));
        }
        for raw in ["0", "false", "no", "off"] {
            assert!(!qwen_split_loaded_qkv_enabled_with_env(Some(raw)));
        }
        assert!(qwen_split_loaded_qkv_enabled_with_env(Some("invalid")));
    }

    #[test]
    fn decode_token_budget_scales_with_audio_and_context() {
        let metadata = parse_qwen3_execution_metadata(&qwen_metadata()).expect("metadata");
        let short_audio = GgmlAsrPreparedAudioView::mono_16khz(vec![0.0; 16_000]);
        let long_audio = GgmlAsrPreparedAudioView::mono_16khz(vec![0.0; 240_000]);

        let short_budget =
            qwen3_generated_token_budget(&short_audio, 32, metadata).expect("short budget");
        let long_budget =
            qwen3_generated_token_budget(&long_audio, 32, metadata).expect("long budget");
        let context_limited =
            qwen3_generated_token_budget(&long_audio, 240, metadata).expect("limited budget");

        assert_eq!(short_budget, QWEN3_DECODE_MIN_GENERATED_TOKENS);
        assert!(long_budget > short_budget);
        assert_eq!(context_limited, 16);
    }

    #[test]
    fn decode_token_budget_rejects_full_prompt_context() {
        let metadata = parse_qwen3_execution_metadata(&qwen_metadata()).expect("metadata");
        let audio = GgmlAsrPreparedAudioView::mono_16khz(vec![0.0; 16_000]);

        let error = qwen3_generated_token_budget(&audio, metadata.llm_max_positions, metadata)
            .expect_err("full context should fail");

        assert!(error.to_string().contains("exhausts llm_max_positions"));
    }

    #[test]
    fn qwen_executor_rejects_non_qwen_adapter() {
        let mut request = qwen_request_from_preflight(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
        );
        request.selected_family =
            builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute_view(&request)
            .expect_err("wrong adapter must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains("requires adapter"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn qwen_executor_fails_closed_when_required_metadata_missing() {
        let mut metadata = qwen_metadata();
        metadata.remove(QWEN3_LLM_VOCAB_SIZE_KEY);
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-missing-metadata.gguf");
        let fixture_spec = TinyGgufFixtureSpec::new(metadata);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);
        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute_view(&request)
            .expect_err("missing metadata must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains(QWEN3_LLM_VOCAB_SIZE_KEY), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn qwen_executor_fails_closed_when_required_tensor_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec().without_tensor(OUTPUT_NORM_WEIGHT);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);

        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute_view(&request)
            .expect_err("missing required tensor must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains("runtime contract check failed"), "{reason}");
                assert!(reason.contains(OUTPUT_NORM_WEIGHT), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    fn assert_qwen_executor_runs(runtime_path: PathBuf) {
        let mut request = qwen_request(runtime_path);
        plan_qwen_request_decoder_state(&mut request);
        let executor = Qwen3AsrGgmlExecutor::default();
        with_forced_cpu_backend_for_test(|| match executor.execute_view(&request) {
            Ok(_) => {}
            Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                if reason.contains("reached max_generated_tokens") => {}
            Err(error) => panic!("qwen executor should reach decode boundary, got {error:?}"),
        });
    }

    #[test]
    fn qwen_executor_runs_full_stack_with_base_fixture() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        assert_qwen_executor_runs(runtime_path);
    }

    #[test]
    fn qwen_executor_reuses_runtime_assets_across_repeated_runs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");

        let mut request = qwen_request(runtime_path);
        plan_qwen_request_decoder_state(&mut request);
        let executor = Qwen3AsrGgmlExecutor::default();

        with_forced_cpu_backend_for_test(|| {
            for _ in 0..2 {
                match executor.execute_view(&request) {
                    Ok(_) => {}
                    Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                        if reason.contains("reached max_generated_tokens") => {}
                    Err(error) => {
                        panic!(
                            "qwen cached runtime path should reach decode boundary, got {error:?}"
                        )
                    }
                }
            }
        });
    }

    #[test]
    fn qwen_executor_reuses_runtime_assets_for_longform_runs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");

        let mut request = qwen_request(runtime_path);
        // Keep the tiny fixture's 256-position context honest while still
        // exercising the longform request path. A production Qwen pack has a
        // much larger context ceiling; this test only needs a valid compact
        // envelope for repeated-owner reuse.
        let longform = LongFormOptions {
            chunk_seconds: 1.0,
            min_chunk_seconds: 1.0,
            max_chunk_seconds: 1.0,
            overlap_seconds: 0.0,
            padding_seconds: 0.0,
            max_context_tokens: 1,
            max_context_chars: 1,
            ..LongFormOptions::default()
        };
        request.request_options.longform = Some(longform);
        plan_qwen_request_decoder_state(&mut request);
        let executor = Qwen3AsrGgmlExecutor::default();

        with_forced_cpu_backend_for_test(|| {
            for _ in 0..2 {
                match executor.execute_view(&request) {
                    Ok(_) => {}
                    Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                        if reason.contains("reached max_generated_tokens") => {}
                    Err(error) => {
                        panic!(
                            "qwen longform cached runtime path should reach decode boundary, got {error:?}"
                        )
                    }
                }
            }
        });
    }

    #[test]
    fn qwen_prepared_runtime_builder_accepts_deeper_layer_fixtures() {
        let executor = Qwen3AsrGgmlExecutor::default();
        for llm_layers in 3..=9 {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp
                .path()
                .join(format!("qwen3-asr-0.6b-q4_k-layer{llm_layers}.gguf"));
            let fixture_spec = qwen_tensor_ready_fixture_spec_with_llm_layers(llm_layers);
            write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec)
                .expect("write gguf fixture");
            let request = qwen_request(runtime_path);
            let preflight = request.runtime_source_preflight();
            executor
                .build_prepared_runtime(request.selected_family.model_architecture, preflight)
                .expect("prepared runtime should build");
        }
    }

    #[test]
    fn qwen_prepared_runtime_drops_zero_copy_audio_projection_payloads() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);
        let preflight = request.runtime_source_preflight();
        let executor = Qwen3AsrGgmlExecutor::default();
        let prepared = executor
            .build_prepared_runtime(request.selected_family.model_architecture, preflight)
            .expect("prepared runtime should build");
        assert!(
            prepared
                .audio_encoder_weights
                .zero_copy_audio_projection_payloads_dropped_for_test()
        );
    }

    #[test]
    fn qwen_executor_rejects_non_empty_prompt_option_until_tokenization_lands() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let mut request = qwen_request(runtime_path);
        request.request_options.prompt = Some("test".to_string());

        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute_view(&request)
            .expect_err("non-empty prompt must fail closed");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(
                    reason.contains("decode prompt construction failed"),
                    "{reason}"
                );
                assert!(reason.contains("request option 'prompt'"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    /// Real-pack end-to-end AB harness for the qwen3-asr LLM decoder, mirroring
    /// firered2-llm's `firered_llm_perf_ab`: drives the actual `execute()` path
    /// (not a synthetic tiny fixture) against a real `.oasr` pack + wav clip so
    /// a maintainer can confirm the CPU decode still produces the same
    /// transcript across a change to the decode graph/buffer machinery. Never
    /// asserts a timing number (host-dependent) -- prints RTF + text so two
    /// runs (e.g. before/after a `cpu_graph.rs` change, same pack + same
    /// backend) can be diffed by eye.
    ///
    /// Env: `OPENASR_QWEN3_AB_PACK=<path to .oasr>` (required -- no committed
    /// dev pack), `OPENASR_QWEN3_AB_BACKEND=cpu|metal|auto` (default cpu, since
    /// this exists to pin the CPU decode path), `OPENASR_QWEN3_AB_CLIP=<wav
    /// path>` (default fixtures/jfk.wav).
    #[test]
    #[ignore = "real-pack AB harness: requires OPENASR_QWEN3_AB_PACK pointed at a real \
                qwen3-asr .oasr (no dev pack committed); prints the decoded text + RTF for \
                before/after comparison, does not assert a pinned golden"]
    fn qwen3_asr_real_pack_ab() {
        let Some(pack_path) = std::env::var_os("OPENASR_QWEN3_AB_PACK").map(PathBuf::from) else {
            eprintln!("skipping: OPENASR_QWEN3_AB_PACK not set");
            return;
        };
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let backend_preference = match std::env::var("OPENASR_QWEN3_AB_BACKEND").as_deref() {
            Ok("metal") | Ok("gpu") => GgmlAsrBackendPreference::Accelerated,
            Ok("auto") => GgmlAsrBackendPreference::Auto,
            _ => GgmlAsrBackendPreference::CpuOnly,
        };
        let clip = std::env::var("OPENASR_QWEN3_AB_CLIP")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
            });
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            clip,
            "qwen3-asr real-pack AB",
            "qwen3-asr real-pack AB",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;

        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                    &pack_path,
                )
                .expect("qwen runtime pack must pass preflight"),
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_qwen_request_decoder_state(&mut request);

        let executor = Qwen3AsrGgmlExecutor::default();
        let started_at = std::time::Instant::now();
        let result = executor
            .execute_view(&request)
            .expect("qwen3-asr transcribe");
        let elapsed = started_at.elapsed();
        let rtf = elapsed.as_secs_f32() / audio_duration_seconds.max(0.001);
        eprintln!(
            "QWEN3_ASR_AB backend={backend_preference:?} audio={audio_duration_seconds:.2}s \
             elapsed={elapsed:?} RTF={rtf:.3} text={}",
            result.transcription.text
        );
    }

    #[test]
    fn prefill_chunk_cancel_poll_returns_typed_canceled() {
        use std::sync::Arc;

        use crate::api::backend::TranscriptionControl;
        use crate::ggml_runtime::GgmlCpuGraphError;

        let control = Arc::new(TranscriptionControl::new());
        assert!(super::ensure_prefill_chunk_not_canceled(&control).is_ok());
        control.request_cancel();
        assert_eq!(
            super::ensure_prefill_chunk_not_canceled(&control),
            Err(super::Qwen3AsrGreedyDecodeError::Canceled)
        );
        // Graph-level cancel maps to the same typed family error.
        assert_eq!(
            super::map_prefill_graph_error(GgmlCpuGraphError::Canceled),
            super::Qwen3AsrGreedyDecodeError::Canceled
        );
        // Stable marker used by dispatch_error_to_backend.
        assert!(
            super::Qwen3AsrGreedyDecodeError::Canceled
                .to_string()
                .contains("canceled by transcription control")
        );
    }

    #[test]
    fn prefill_chunk_loop_harness_stops_between_chunks_on_cancel() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::api::backend::TranscriptionControl;

        // Lightweight stand-in for the host-cache chunk walk: each "chunk" is a
        // pure counter bump with the same cancel poll the production loop uses.
        // Cancel after the second chunk completes; the third poll must abort
        // before another chunk runs.
        let control = Arc::new(TranscriptionControl::new());
        let chunks_run = AtomicUsize::new(0);
        let token_count = 12usize;
        let chunk_size = 4usize;
        let mut position_offset = 0usize;
        let mut canceled = false;
        while position_offset < token_count {
            if let Err(error) = super::ensure_prefill_chunk_not_canceled(&control) {
                assert_eq!(error, super::Qwen3AsrGreedyDecodeError::Canceled);
                canceled = true;
                break;
            }
            let chunk_len = (token_count - position_offset).min(chunk_size);
            let seen = chunks_run.fetch_add(1, Ordering::SeqCst) + 1;
            if seen == 2 {
                control.request_cancel();
            }
            position_offset = position_offset.saturating_add(chunk_len);
        }
        assert!(canceled, "cancel must abort the harness loop");
        assert_eq!(
            chunks_run.load(Ordering::SeqCst),
            2,
            "exactly two chunks should run before the next boundary poll aborts"
        );
        assert!(
            position_offset < token_count,
            "cancel must leave residual tokens unprocessed"
        );
    }
}
