use std::sync::Arc;

#[cfg(test)]
use std::sync::{Mutex, OnceLock};

use thiserror::Error;

use super::batched_decode::{
    MoonshineServeBatchConfig, MoonshineServeBatchEngineRegistry, MoonshineServeBatchJob,
    moonshine_serve_batch_decode_config, shutdown_moonshine_serve_batch_engines,
    submit_moonshine_serve_batch_job,
};
use super::decoder_graph::{
    MoonshineDecoderGraphError, MoonshineDecoderGraphRuntime, MoonshineDecoderRuntimeInput,
    run_moonshine_decoder_short_form_with_runtime,
};
use super::encoder_graph::{MoonshineEncoderGraphRuntime, MoonshineEncoderOutput};
use super::frontend::{MoonshineFrontendError, moonshine_waveform_from_prepared_audio};
use super::graph_config::{
    MoonshineGraphConfigIdentity, moonshine_decoder_graph_config, moonshine_encoder_graph_config,
    moonshine_graph_config_identity,
};
use super::lora::{
    MoonshineLoraError, moonshine_adapter_cache_fingerprint, resolve_moonshine_lora_adapter,
};
use super::prepared_runtime::{
    MoonshinePreparedRuntime, MoonshinePreparedRuntimeError, build_moonshine_prepared_runtime,
};
use crate::MOONSHINE_GGML_ADAPTER_ID;
use crate::NativeAsrSession;
use crate::device::execution_policy::ExecutionPlacement;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlDecodeOutputContract, GgmlDecodeOutputPlan,
    GgmlDecodeReuseMode, GgufRuntimeSourcePreflight, ResolvedFamilyRuntimeInput,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::device_greedy_token::DeviceGreedyStepOutputMode;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
    GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::lora_adapter::{
    ResolvedLoraAdapterCache, ResolvedLoraAdapterHandle, resolved_lora_adapter,
};
use crate::models::native_execution_services::ExecutionLaneKey;
use crate::models::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeCache, PreparedRuntimeHandle,
    PreparedRuntimeQuoteContext, SystemMemoryMaterialization,
};
use crate::models::runtime_cache_coordinator::{PackContentKey, canonical_runtime_cache_path};
use crate::models::system_memory_owner::SystemMemoryOwner;

const MOONSHINE_EXECUTOR_ID: &str = crate::arch::MOONSHINE_EXECUTOR_COMPONENT_ID;
const MOONSHINE_STREAMING_EXECUTOR_ID: &str = "moonshine-ggml-snapshot-streaming-executor-v1";

const MOONSHINE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const MOONSHINE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

/// (pack content id, execution lane, decoder capacity, adapter fingerprint,
/// immutable output contract, output plan, and reuse mode). The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a runtime built from the old
/// bytes. The adapter fingerprint MUST stay in this key -- prepared encoder
/// graphs embed the adapter tensors, so reuse keyed only on the base pack
/// would be a correctness bug. The output topology fields keep an owner
/// built for one request proof from being reused for another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MoonshineDecodeFeatureKey {
    output_mode: DeviceGreedyStepOutputMode,
    adapter_active: bool,
    phrase_bias_active: bool,
    word_timestamps: bool,
    streaming: bool,
    serve_batch: bool,
}

type MoonshineEncoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    MoonshineGraphConfigIdentity,
    String,
);
type MoonshineDecoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    MoonshineGraphConfigIdentity,
    crate::models::seq2seq_decoder_state::Seq2SeqResidentCapacity,
    String,
    GgmlDecodeOutputContract,
    GgmlDecodeOutputPlan,
    GgmlDecodeReuseMode,
    MoonshineDecodeFeatureKey,
);

struct MoonshineEncoderActorState {
    runtime: MoonshineEncoderGraphRuntime,
    _prepared_owner: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
}

struct MoonshineDecoderActorState {
    runtime: MoonshineDecoderGraphRuntime,
    _prepared_owner: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
}

type MoonshineEncoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    MoonshineEncoderRuntimeCacheKey,
    MoonshineEncoderActorState,
>;
type MoonshineDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    MoonshineDecoderRuntimeCacheKey,
    MoonshineDecoderActorState,
>;
type MoonshineEncoderRuntimeActor =
    PinnedRuntimeActorCheckout<MoonshineEncoderRuntimeCacheKey, MoonshineEncoderActorState>;
type MoonshineDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<MoonshineDecoderRuntimeCacheKey, MoonshineDecoderActorState>;

#[cfg(test)]
static DECODER_OWNER_GRAPH_CONFIG_PROBE: OnceLock<Mutex<Option<MoonshineGraphConfigIdentity>>> =
    OnceLock::new();

#[cfg(test)]
fn record_decoder_owner_graph_config(config: GgmlCpuGraphConfig) {
    *DECODER_OWNER_GRAPH_CONFIG_PROBE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("decoder owner graph config probe lock") =
        Some(moonshine_graph_config_identity(config));
}

#[derive(Debug, Error)]
enum MoonshineGgmlExecutorError {
    #[error("moonshine ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("moonshine adapter pack rejected (fail-closed): {source}")]
    AdapterRejected {
        #[source]
        source: MoonshineLoraError,
    },
    #[error("moonshine ggml executor runtime preparation failed: {reason}")]
    PreparedRuntimeFailed { reason: String },
    #[error("moonshine ggml executor frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("moonshine ggml executor encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("moonshine ggml executor decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("moonshine ggml executor {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

#[derive(Clone)]
pub(crate) struct MoonshineGgmlExecutor {
    runtime_cache_by_path: PreparedRuntimeCache<MoonshinePreparedRuntime>,
    serve_batch_engines: MoonshineServeBatchEngineRegistry,
    encoder_runtimes: Arc<MoonshineEncoderRuntimePool>,
    decoder_runtimes: Arc<MoonshineDecoderRuntimePool>,
    lora_adapters: ResolvedLoraAdapterCache,
}

impl std::fmt::Debug for MoonshineGgmlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoonshineGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for MoonshineGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            MOONSHINE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            MOONSHINE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            runtime_cache_by_path: PreparedRuntimeCache::default(),
            serve_batch_engines: MoonshineServeBatchEngineRegistry::default(),
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moonshine-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moonshine-decoder-owner",
                limits,
            )),
            lora_adapters: ResolvedLoraAdapterCache::default(),
        }
    }
}

impl SystemMemoryMaterialization for MoonshinePreparedRuntime {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        MoonshinePreparedRuntime::retained_system_memory_bytes(self)
    }
}

impl HostNeutralPreparedRuntime for MoonshinePreparedRuntime {
    fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        MoonshinePreparedRuntime::system_memory_quote(
            context.metadata,
            context.tensor_index,
            pack_content_id,
        )
    }
}

impl MoonshineGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, MoonshineGgmlExecutorError> {
        if request.selected_family.adapter_id != MOONSHINE_GGML_ADAPTER_ID {
            return Err(MoonshineGgmlExecutorError::AdapterMismatch {
                expected: MOONSHINE_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let decoder_state =
            crate::models::seq2seq_decoder_state::Seq2SeqDecoderState::from_request_state(
                &request.decoder_state,
                super::capacity::MOONSHINE_DECODER_STATE_IDS,
            )
            .map_err(|error| MoonshineGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;

        let preflight = request.runtime_source_preflight();
        // OADP Phase 0: resolve the active adapter (request-level path, env
        // fallback — if any) against THIS base pack. Any mismatch fails the
        // whole transcription — adapters are never silently ignored.
        let adapter = resolve_moonshine_lora_adapter(
            &self.lora_adapters,
            request.request_options.adapter_path.as_deref(),
            preflight,
        )
        .map_err(|source| MoonshineGgmlExecutorError::AdapterRejected { source })?;
        let resolved_runtime = request.resolved_runtime;
        // The request boundary owns the immutable backend/output/reuse decision.
        // Every owner below consumes this value instead of reconstructing a
        // route from thread-local state or provider/backend heuristics.
        let backend = resolved_runtime.backend();
        // Snapshot each stage's exact lane once. Vulkan intentionally resolves to
        // a hybrid topology: the encoder lane stays accelerated while the
        // default decoder lane is CPU. Owners and cache keys receive these
        // identities explicitly instead of consulting ambient TLS later.
        let request_lane = request
            .execution_context
            .native_execution_lane()
            .cloned()
            .ok_or_else(|| MoonshineGgmlExecutorError::RuntimeOwnershipFailed {
                stage: "request-lane",
                reason: "candidate-resolved execution lane is missing".to_string(),
            })?;
        let encoder_config = moonshine_encoder_graph_config(backend);
        let decoder_config = moonshine_decoder_graph_config(backend, Some(request_lane.provider()));
        let encoder_execution_lane = request_lane.for_stage(
            encoder_config.backend,
            if encoder_config.backend == GgmlCpuGraphBackend::Cpu {
                ExecutionPlacement::CpuOnly
            } else {
                ExecutionPlacement::FullDevice
            },
        );
        let decoder_execution_lane = request_lane.for_stage(
            decoder_config.backend,
            if decoder_config.backend == GgmlCpuGraphBackend::Cpu {
                ExecutionPlacement::CpuOnly
            } else {
                ExecutionPlacement::FullDevice
            },
        );
        let prepared_runtime = self.prepared_runtime_for_preflight(preflight, backend)?;
        let features = moonshine_waveform_from_prepared_audio(
            &request.prepared_audio,
            prepared_runtime.metadata.sample_rate_hz,
        )
        .map_err(map_frontend_error)?;

        let encoder_output = self.encode_with_owned_runtime(
            preflight,
            Arc::clone(&prepared_runtime),
            features,
            adapter.clone(),
            backend,
            encoder_config,
            encoder_execution_lane.clone(),
        )?;

        let audio_duration = audio_duration_seconds(&request.prepared_audio);
        let serve_batch_config = MoonshineServeBatchConfig::from_policy::<
            super::batched_decode::MoonshineFamily,
        >(request.request_options.serve_batch);
        let greedy_step_output_mode =
            moonshine_greedy_step_output_mode(resolved_runtime, skip_serve_batch);
        let can_use_serve_batch = can_use_moonshine_serve_batch(
            skip_serve_batch,
            adapter.is_some(),
            decoder_config.backend,
            decoder_config.use_scheduler,
            resolved_runtime,
        );
        let feature_key = MoonshineDecodeFeatureKey {
            output_mode: greedy_step_output_mode,
            adapter_active: adapter.is_some(),
            phrase_bias_active: request.request_options.phrase_bias.is_some(),
            word_timestamps: request.request_options.word_timestamps,
            streaming: skip_serve_batch,
            serve_batch: can_use_serve_batch,
        };
        let decode =
            if let Some(serve_batch_config) = serve_batch_config.filter(|_| can_use_serve_batch) {
                let decode_config = moonshine_serve_batch_decode_config(
                    prepared_runtime.metadata,
                    decoder_state,
                    &prepared_runtime.tokenizer,
                    request.request_options.phrase_bias.as_ref(),
                )
                .map_err(|error| MoonshineGgmlExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?;
                submit_moonshine_serve_batch_job(
                    &self.serve_batch_engines,
                    serve_batch_config,
                MoonshineServeBatchJob {
                    runtime_cache_path: canonical_runtime_cache_path(
                        preflight.runtime_source.path(),
                    ),
                    runtime_preflight: preflight.clone(),
                    build_identity:
                        crate::models::ggml_asr_executor::serve_batch_build_identity_for_request(
                            &request.request_options,
                            "moonshine",
                            decoder_config.backend,
                            &preflight.runtime_source,
                        ),
                    backend: decoder_config.backend,
                    graph_config: decoder_config,
                    lane: decoder_execution_lane.clone(),
                    output_plan: resolved_runtime.output_plan(),
                    output_mode: greedy_step_output_mode,
                    reuse_mode: resolved_runtime.reuse_mode(),
                    phrase_bias_active: request.request_options.phrase_bias.is_some(),
                    uses_scheduler: decoder_config.use_scheduler,
                    prepared_runtime: Arc::clone(&prepared_runtime),
                    decoder_state,
                    // Moved (not cloned): the direct branch below also consumes
                    // its own mutually-exclusive value, so neither path needs
                    // an extra copy of the encoder output.
                    encoder_output,
                    decode_config,
                    word_timestamps: request.request_options.word_timestamps,
                    audio_duration_seconds: audio_duration,
                    execution_context: Arc::clone(&request.execution_context),
                },
            )
            .map_err(|error| match error.unavailable_retryable() {
                Some(retryable) => MoonshineGgmlExecutorError::ServeBatchUnavailable {
                    reason: error.to_string(),
                    retryable,
                },
                None => MoonshineGgmlExecutorError::DecoderFailed {
                    reason: error.to_string(),
                },
            })?
            } else {
                self.decode_with_owned_runtime(
                    preflight,
                    Arc::clone(&prepared_runtime),
                    encoder_output,
                    request.request_options.phrase_bias.clone(),
                    resolved_runtime,
                    decoder_config,
                    request.request_options.word_timestamps,
                    audio_duration,
                    adapter.clone(),
                    decoder_state,
                    decoder_execution_lane.clone(),
                    greedy_step_output_mode,
                    feature_key,
                    Arc::clone(&request.execution_context.control),
                    request
                        .execution_context
                        .decode_work_progress_observer()
                        .cloned(),
                    request
                        .execution_context
                        .unstable_decode_text_observer()
                        .cloned(),
                )?
            };

        Ok(GgmlAsrExecutionResult {
            transcription: decode.transcription,
            carry_context: None,
            // No intra-decode timestamps -- the single segment spans the whole
            // buffer -- so the cut point has no honest second to name. See
            // `DecodeTruncation::transcript_covers_up_to_seconds`.
            decode_truncation: decode.stop_reason.into_decode_truncation(None),
        })
    }

    fn prepared_runtime_for_preflight(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
    ) -> Result<PreparedRuntimeHandle<MoonshinePreparedRuntime>, MoonshineGgmlExecutorError> {
        self.runtime_cache_by_path.get_or_try_insert_with(
            &preflight.runtime_source,
            PreparedRuntimeQuoteContext {
                model_architecture: crate::MOONSHINE_GGML_ARCHITECTURE_ID,
                metadata: &preflight.metadata,
                tensor_index: &preflight.tensor_index,
                backend,
            },
            || build_moonshine_prepared_runtime(preflight).map_err(map_prepared_runtime_error),
            // Covers both a genuinely poisoned slot mutex and a build attempt
            // that panicked and was caught (mutex stays unpoisoned, slot
            // stays empty, retryable) -- see
            // `PreparedRuntimeCache::get_or_try_insert_with`. Either way the
            // cache could not deliver a prepared runtime for this attempt;
            // the caller's next request retries clean.
            || MoonshineGgmlExecutorError::PreparedRuntimeFailed {
                reason: "moonshine runtime cache slot unavailable (poisoned lock or a caught build panic); retry".to_string(),
            },
            |error| MoonshineGgmlExecutorError::PreparedRuntimeFailed {
                reason: error.to_string(),
            },
        )
    }

    /// Evicts exactly `pack_content_id`'s cached prepared runtime, releasing
    /// resident state left over from a since-replaced pack without touching
    /// any other content identity. Reached through
    /// [`crate::NativeExecutionServices::evict_prepared_runtime_content_id`].
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.lora_adapters.evict_base_content_id(pack_content_id);
        self.runtime_cache_by_path.evict_content_id(pack_content_id);
        // Engine keys contain the old build identity, but the shared registry
        // cannot safely inspect a family-private content id. Drop all idle
        // owner references so a replaced pack cannot retain decoder graphs.
        shutdown_moonshine_serve_batch_engines(&self.serve_batch_engines);
    }

    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> MoonshineGgmlExecutorError {
        MoonshineGgmlExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn checkout_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
        adapter: Option<ResolvedLoraAdapterHandle>,
        backend: GgmlCpuGraphBackend,
        graph_config: GgmlCpuGraphConfig,
        lane: ExecutionLaneKey,
    ) -> Result<MoonshineEncoderRuntimeActor, MoonshineGgmlExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane,
            moonshine_graph_config_identity(graph_config),
            moonshine_adapter_cache_fingerprint(adapter.as_ref().map(resolved_lora_adapter)),
        );
        let preflight = preflight.clone();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared, adapter))),
            move |(preflight, prepared, adapter)| {
                let runtime = MoonshineEncoderGraphRuntime::new_with_graph_config(
                    &prepared.encoder_weights,
                    prepared.metadata,
                    &preflight,
                    adapter.as_ref().map(resolved_lora_adapter),
                    backend,
                    graph_config,
                )
                .map_err(|error| MoonshineGgmlExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    MoonshineEncoderActorState {
                        runtime,
                        _prepared_owner: prepared,
                    },
                ))
            },
            |error| Self::map_actor_error("encoder", error),
        )
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
        adapter: Option<ResolvedLoraAdapterHandle>,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        resolved_runtime: ResolvedFamilyRuntimeInput,
        decoder_config: GgmlCpuGraphConfig,
        lane: ExecutionLaneKey,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
        feature_key: MoonshineDecodeFeatureKey,
    ) -> Result<MoonshineDecoderRuntimeActor, MoonshineGgmlExecutorError> {
        let backend = resolved_runtime.backend();
        let output_contract = resolved_runtime.output_contract();
        let output_plan = resolved_runtime.output_plan();
        let reuse_mode = resolved_runtime.reuse_mode();
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane,
            moonshine_graph_config_identity(decoder_config),
            decoder_state.resident_capacity(),
            moonshine_adapter_cache_fingerprint(adapter.as_ref().map(resolved_lora_adapter)),
            output_contract,
            output_plan,
            reuse_mode,
            feature_key,
        );
        let preflight = preflight.clone();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared, adapter))),
            move |(preflight, prepared, adapter)| {
                #[cfg(test)]
                record_decoder_owner_graph_config(decoder_config);
                let runtime = MoonshineDecoderGraphRuntime::new_with_greedy_step_output_mode(
                    MoonshineDecoderRuntimeInput {
                        decoder_weights: &prepared.decoder_weights,
                        metadata: prepared.metadata,
                        decoder_state,
                        backend,
                        graph_config: decoder_config,
                        reuse_mode,
                    },
                    &preflight,
                    adapter.as_ref().map(resolved_lora_adapter),
                    greedy_step_output_mode,
                )
                .map_err(|error| MoonshineGgmlExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    MoonshineDecoderActorState {
                        runtime,
                        _prepared_owner: prepared,
                    },
                ))
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn encode_with_owned_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
        features: super::frontend::MoonshineWaveformFeatures,
        adapter: Option<ResolvedLoraAdapterHandle>,
        backend: GgmlCpuGraphBackend,
        graph_config: GgmlCpuGraphConfig,
        lane: ExecutionLaneKey,
    ) -> Result<MoonshineEncoderOutput, MoonshineGgmlExecutorError> {
        let runtime = self.checkout_encoder_runtime(
            preflight,
            prepared,
            adapter,
            backend,
            graph_config,
            lane,
        )?;
        runtime
            .call_mut(move |state| {
                let encode_result = state.runtime.encode(&features);
                let release_result = state.runtime.release_transient_compute_memory();
                match (encode_result, release_result) {
                    (Ok(output), Ok(())) => Ok(output),
                    (Err(error), _) => Err(error),
                    (Ok(_), Err(error)) => Err(error),
                }
            })
            .map_err(|error| Self::map_actor_error("encoder", error))?
            .map_err(|error| MoonshineGgmlExecutorError::EncoderFailed {
                reason: error.to_string(),
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_owned_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MoonshinePreparedRuntime>,
        encoder_output: MoonshineEncoderOutput,
        phrase_bias: Option<crate::PhraseBiasConfig>,
        resolved_runtime: ResolvedFamilyRuntimeInput,
        decoder_config: GgmlCpuGraphConfig,
        word_timestamps: bool,
        audio_duration_seconds: f32,
        adapter: Option<ResolvedLoraAdapterHandle>,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        lane: ExecutionLaneKey,
        greedy_step_output_mode: DeviceGreedyStepOutputMode,
        feature_key: MoonshineDecodeFeatureKey,
        control: Arc<crate::api::backend::TranscriptionControl>,
        decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
        unstable_decode_text: Option<crate::api::backend::UnstableDecodeTextObserver>,
    ) -> Result<super::decoder_graph::MoonshineDecodeOutput, MoonshineGgmlExecutorError> {
        let tokenizer = prepared.tokenizer.clone();
        let metadata = prepared.metadata;
        let runtime = self.checkout_decoder_runtime(
            preflight,
            prepared,
            adapter,
            decoder_state,
            resolved_runtime,
            decoder_config,
            lane,
            greedy_step_output_mode,
            feature_key,
        )?;
        runtime
            .call_mut(move |state| {
                state.runtime.activate_decoder_state(decoder_state)?;
                run_moonshine_decoder_short_form_with_runtime(
                    &mut state.runtime,
                    &tokenizer,
                    metadata,
                    &encoder_output,
                    phrase_bias.as_ref(),
                    word_timestamps,
                    audio_duration_seconds,
                    &control,
                    decode_work_progress.as_ref(),
                    unstable_decode_text.as_ref(),
                )
            })
            .map_err(|error| Self::map_actor_error("decoder", error))?
            .map_err(map_decoder_error)
    }
}

/// Decide whether the moonshine decode may go through the shared serve-batch
/// worker. Dynamic adapters force the direct decode path: the serve-batch
/// worker pools runtimes per pack and would need adapter-aware job routing;
/// Phase 0 keeps that surface untouched (adapter active => always bypass).
fn can_use_moonshine_serve_batch(
    skip_serve_batch: bool,
    adapter_active: bool,
    decoder_backend: GgmlCpuGraphBackend,
    decoder_uses_scheduler: bool,
    resolved_runtime: ResolvedFamilyRuntimeInput,
) -> bool {
    !skip_serve_batch
        && !adapter_active
        && matches!(
            decoder_backend,
            GgmlCpuGraphBackend::Gpu | GgmlCpuGraphBackend::Metal
        )
        && !decoder_uses_scheduler
        && resolved_runtime.output_plan() == GgmlDecodeOutputPlan::FullLogits
        // The worker owns a persistent decode graph. Without shared evidence,
        // the request is FreshGraph and must stay on the direct executor path.
        && resolved_runtime.reuse_mode() == GgmlDecodeReuseMode::ReusableGraph
}

/// Translate the immutable request output plan into the graph-facing mode.
/// Moonshine consumes the shared planner result; request logits consumers are
/// combined once at the request boundary rather than re-OR'd here.
fn moonshine_greedy_step_output_mode(
    resolved_runtime: ResolvedFamilyRuntimeInput,
    force_full_logits: bool,
) -> DeviceGreedyStepOutputMode {
    if force_full_logits {
        return DeviceGreedyStepOutputMode::FullLogits;
    }
    crate::models::device_greedy_token::device_greedy_step_output_mode_for_resolved_runtime(
        resolved_runtime,
    )
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudioView) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

impl GgmlAsrViewExecutor for MoonshineGgmlExecutor {
    fn adapter_binding_strategy(
        &self,
    ) -> crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy {
        crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy::MoonshineLoraV1
    }

    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        MoonshineGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        MOONSHINE_EXECUTOR_ID
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
                super::capacity::plan_moonshine_decoder_state,
                super::capacity::MOONSHINE_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: batch worker allowed.
        self.execute_inner(request, false)
            .map_err(|error| moonshine_execute_error_to_ggml(self, error, request))
    }

    fn unload_idle_state(&self) {
        shutdown_moonshine_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.lora_adapters.clear();
        self.runtime_cache_by_path.clear();
    }
}

impl MoonshineGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| moonshine_execute_error_to_ggml(self, error, request))
    }
}

fn moonshine_execute_error_to_ggml(
    executor: &MoonshineGgmlExecutor,
    error: MoonshineGgmlExecutorError,
    request: &GgmlAsrExecutionViewRequest,
) -> GgmlAsrExecutionError {
    match error {
        MoonshineGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: GgmlAsrViewExecutor::executor_id(executor),
            adapter_id: request.selected_family.adapter_id,
            reason: error.to_string(),
        },
    }
}

impl GgmlAsrStreamingExecutor for MoonshineGgmlExecutor {
    fn adapter_binding_strategy(
        &self,
    ) -> crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy {
        crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy::MoonshineLoraV1
    }

    fn executor_id(&self) -> &'static str {
        MOONSHINE_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            MOONSHINE_STREAMING_EXECUTOR_ID,
            MOONSHINE_GGML_ADAPTER_ID,
            "moonshine",
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            MoonshineGgmlExecutor::execute_streaming,
        )
    }

    fn unload_idle_state(&self) {
        shutdown_moonshine_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.lora_adapters.clear();
        self.runtime_cache_by_path.clear();
    }
}

fn map_prepared_runtime_error(error: MoonshinePreparedRuntimeError) -> MoonshineGgmlExecutorError {
    MoonshineGgmlExecutorError::PreparedRuntimeFailed {
        reason: error.to_string(),
    }
}

fn map_frontend_error(error: MoonshineFrontendError) -> MoonshineGgmlExecutorError {
    MoonshineGgmlExecutorError::FrontendFailed {
        reason: error.to_string(),
    }
}

fn map_decoder_error(error: MoonshineDecoderGraphError) -> MoonshineGgmlExecutorError {
    MoonshineGgmlExecutorError::DecoderFailed {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph_config::moonshine_graph_config_identity;
    use super::super::prepared_runtime::MoonshinePreparedRuntime;
    use super::super::tokenizer::MoonshineTokenizer;
    use super::super::weights::{
        MoonshineDecoderLayerWeights, MoonshineDecoderWeights, MoonshineEncoderLayerWeights,
        MoonshineEncoderWeights, MoonshineWeight,
    };
    use super::{
        DECODER_OWNER_GRAPH_CONFIG_PROBE, ExecutionLaneKey, MoonshineDecodeFeatureKey,
        MoonshineGgmlExecutor, can_use_moonshine_serve_batch, moonshine_greedy_step_output_mode,
    };
    use crate::device::execution_policy::ExecutionPlacement;
    use crate::device::execution_route::{
        DeviceAddressability, ExecutionProvider, PhysicalResourceKey, ResolvedExecutionRoute,
        RouteDeviceKind,
    };
    use crate::ggml_runtime::{
        AutoGpuPolicy, GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlDecodeOutputContract,
        GgmlDecodeOutputPlan, GgmlDecodeReuseMode, GgufMetadataValue, GgufRuntimeSourcePreflight,
        RequestBackendPreference, ResolvedFamilyRuntimeInput, validate_ggml_runtime_source_path,
        write_gguf_file_v0,
    };
    use crate::models::device_greedy_token::DeviceGreedyStepOutputMode;
    use crate::models::seq2seq_decoder_state::{Seq2SeqDecoderState, Seq2SeqStateAxis};
    use crate::models::system_memory_owner::SystemMemoryOwner;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    fn fresh_runtime() -> ResolvedFamilyRuntimeInput {
        ResolvedFamilyRuntimeInput::resolve(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
        )
    }

    fn exact_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(ResolvedExecutionRoute {
            provider,
            stable_id: format!("{provider:?}0"),
            registry_ordinal: 0,
            kind: RouteDeviceKind::Accelerated,
            addressability: DeviceAddressability::ExactlyAddressable {
                physical_key: PhysicalResourceKey::new("0000:02:00.0")
                    .expect("synthetic PCI key is valid"),
            },
        })
    }

    fn dummy_weight(name: &str) -> MoonshineWeight {
        MoonshineWeight {
            name: name.to_string(),
            dims: vec![1],
            values: vec![0.0],
        }
    }

    fn dummy_prepared_runtime() -> MoonshinePreparedRuntime {
        let mut tokenizer_values: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
        tokenizer_values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        tokenizer_values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<pad>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ]),
        );
        let tokenizer_metadata = crate::GgufMetadata::from_values_for_test(tokenizer_values);
        let tokenizer = MoonshineTokenizer::from_gguf_metadata(&tokenizer_metadata)
            .expect("dummy tokenizer metadata");
        let metadata = super::super::runtime_contract::MoonshineExecutionMetadata {
            vocab_size: 3,
            d_model: 1,
            encoder_layers: 0,
            decoder_layers: 0,
            n_heads: 1,
            head_dim: 1,
            rotary_dim: 0,
            encoder_ffn_dim: 1,
            decoder_ffn_dim: 1,
            decoder_max_context: 1,
            bos_token_id: 1,
            eos_token_id: 2,
            sample_rate_hz: 16_000,
            rope_theta: 10_000.0,
        };
        MoonshinePreparedRuntime {
            metadata,
            tokenizer,
            encoder_weights: MoonshineEncoderWeights {
                conv1_weight: dummy_weight("enc.conv1.weight"),
                conv2_weight: dummy_weight("enc.conv2.weight"),
                conv2_bias: dummy_weight("enc.conv2.bias"),
                conv3_weight: dummy_weight("enc.conv3.weight"),
                conv3_bias: dummy_weight("enc.conv3.bias"),
                groupnorm_weight: dummy_weight("enc.groupnorm.weight"),
                groupnorm_bias: dummy_weight("enc.groupnorm.bias"),
                out_norm: dummy_weight("enc.out_norm.weight"),
                layers: Vec::<MoonshineEncoderLayerWeights>::new(),
            },
            decoder_weights: MoonshineDecoderWeights {
                embedding: MoonshineWeight {
                    name: "dec.emb.weight".to_string(),
                    dims: vec![3, 1],
                    values: vec![0.0; 3],
                },
                out_norm: dummy_weight("dec.out_norm.weight"),
                layers: Vec::<MoonshineDecoderLayerWeights>::new(),
            },
        }
    }

    fn empty_runtime_source_preflight() -> (tempfile::TempDir, GgufRuntimeSourcePreflight) {
        let directory = tempdir().expect("temporary runtime directory");
        let path = directory.path().join("moonshine-empty.gguf");
        write_gguf_file_v0(&path, &BTreeMap::new(), &[]).expect("write empty GGUF");
        let source = validate_ggml_runtime_source_path(&path).expect("validate empty GGUF");
        let preflight =
            GgufRuntimeSourcePreflight::from_runtime_source(&source).expect("preflight empty GGUF");
        (directory, preflight)
    }

    #[test]
    fn exact_cuda_and_vulkan_without_evidence_stay_gpu_full_logits_and_fresh() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let resolved = ResolvedFamilyRuntimeInput::resolve(
                Some(exact_preference(provider)),
                AutoGpuPolicy::AllBackends,
            );
            assert_eq!(resolved.backend(), GgmlCpuGraphBackend::Gpu);
            assert_eq!(resolved.output_plan(), GgmlDecodeOutputPlan::FullLogits);
            assert_eq!(resolved.reuse_mode(), GgmlDecodeReuseMode::FreshGraph);
            assert_eq!(
                moonshine_greedy_step_output_mode(resolved, false),
                DeviceGreedyStepOutputMode::FullLogits,
                "provider={provider:?}"
            );
        }
    }

    #[test]
    fn hybrid_stage_lanes_bind_gpu_encoder_and_physical_cpu_decoder() {
        let candidate = ExecutionLaneKey::unscoped_for_backend(GgmlCpuGraphBackend::Gpu);
        let encoder = candidate.for_stage(GgmlCpuGraphBackend::Gpu, ExecutionPlacement::FullDevice);
        let decoder = candidate.for_stage(GgmlCpuGraphBackend::Cpu, ExecutionPlacement::CpuOnly);
        let cpu_encoder =
            candidate.for_stage(GgmlCpuGraphBackend::Cpu, ExecutionPlacement::CpuOnly);
        assert_eq!(encoder.provider(), candidate.provider());
        assert_eq!(decoder.provider(), ExecutionProvider::Cpu);
        assert_eq!(encoder.backend(), GgmlCpuGraphBackend::Gpu);
        assert_eq!(decoder.backend(), GgmlCpuGraphBackend::Cpu);
        assert_eq!(cpu_encoder.placement(), ExecutionPlacement::CpuOnly);
        assert_eq!(cpu_encoder.provider(), ExecutionProvider::Cpu);
        assert_ne!(encoder, decoder);
    }
    #[test]
    fn streaming_context_carries_candidate_lane_without_reresolution() {
        let candidate = ExecutionLaneKey::unscoped_for_backend(GgmlCpuGraphBackend::Gpu);
        let context = crate::RequestExecutionContext::uncancellable("lane test")
            .with_native_execution_lane(candidate.clone());
        assert_eq!(context.native_execution_lane(), Some(&candidate));
    }
    #[test]
    fn checkout_owner_consumes_captured_decoder_config_after_late_overrides() {
        let captured_config = crate::test_process_env::with_test_process_env(
            [(GgmlCpuGraphConfig::THREADS_ENV, None)],
            || {
                let _request_threads =
                    crate::models::graph_runtime_config::install_request_inference_threads_override(
                        Some(2),
                    );
                super::super::graph_config::moonshine_decoder_graph_config(
                    GgmlCpuGraphBackend::Cpu,
                    None,
                )
            },
        );
        let captured_identity = moonshine_graph_config_identity(captured_config);
        *DECODER_OWNER_GRAPH_CONFIG_PROBE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .expect("decoder owner graph config probe lock") = None;

        let (_directory, preflight) = empty_runtime_source_preflight();
        let prepared = Arc::new(SystemMemoryOwner::without_allocation(
            dummy_prepared_runtime(),
        ));
        let decoder_state = Seq2SeqDecoderState {
            self_attention: Seq2SeqStateAxis {
                logical_positions: 1,
                resident_positions: 1,
                hard_position_cap: 1,
            },
            cross_attention: Seq2SeqStateAxis {
                logical_positions: 1,
                resident_positions: 1,
                hard_position_cap: 1,
            },
        };
        let lane = ExecutionLaneKey::unscoped_for_backend(GgmlCpuGraphBackend::Cpu);
        let feature_key = MoonshineDecodeFeatureKey {
            output_mode: DeviceGreedyStepOutputMode::FullLogits,
            adapter_active: false,
            phrase_bias_active: false,
            word_timestamps: false,
            streaming: true,
            serve_batch: false,
        };

        let _late_request_threads =
            crate::models::graph_runtime_config::install_request_inference_threads_override(Some(
                8,
            ));
        let result = crate::test_process_env::with_test_process_env(
            [(
                GgmlCpuGraphConfig::THREADS_ENV,
                Some(std::ffi::OsString::from("64")),
            )],
            || {
                MoonshineGgmlExecutor::default().checkout_decoder_runtime(
                    &preflight,
                    prepared,
                    None,
                    decoder_state,
                    fresh_runtime(),
                    captured_config,
                    lane,
                    DeviceGreedyStepOutputMode::FullLogits,
                    feature_key,
                )
            },
        );
        assert!(
            result.is_ok(),
            "synthetic owner runtime should build through checkout"
        );
        let observed_identity = DECODER_OWNER_GRAPH_CONFIG_PROBE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .expect("decoder owner graph config probe lock")
            .take()
            .expect("decoder owner build reached config consumer");
        assert_eq!(observed_identity, captured_identity);
    }

    #[test]
    fn graph_config_key_isolates_effective_thread_topology() {
        let mut one =
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Cpu);
        one.n_threads = Some(1);
        let mut four = one;
        four.n_threads = Some(4);
        assert_ne!(
            moonshine_graph_config_identity(one),
            moonshine_graph_config_identity(four)
        );
    }
    #[test]
    fn feature_bits_isolate_owner_cache_topology() {
        let base = MoonshineDecodeFeatureKey {
            output_mode: DeviceGreedyStepOutputMode::FullLogits,
            adapter_active: false,
            phrase_bias_active: false,
            word_timestamps: false,
            streaming: false,
            serve_batch: false,
        };
        let mut phrase = base;
        phrase.phrase_bias_active = true;
        let mut streaming = base;
        streaming.streaming = true;
        let mut batch = base;
        batch.serve_batch = true;
        assert_ne!(base, phrase);
        assert_ne!(base, streaming);
        assert_ne!(base, batch);
    }
    #[test]
    fn fresh_graph_plan_disables_serve_batch_even_on_direct_gpu_path() {
        // The shared worker owns a persistent graph, while an unproven request
        // is FreshGraph. Keep it on the direct path rather than silently
        // upgrading the topology.
        assert!(!can_use_moonshine_serve_batch(
            false,
            false,
            GgmlCpuGraphBackend::Gpu,
            false,
            fresh_runtime(),
        ));
    }

    #[test]
    fn active_adapter_forces_serve_batch_bypass() {
        // OADP Phase 0 contract: an active dynamic adapter ALWAYS bypasses the
        // shared serve-batch worker (its pooled runtimes are adapter-free),
        // even when every other condition would allow serve-batch.
        assert!(!can_use_moonshine_serve_batch(
            false,
            true,
            GgmlCpuGraphBackend::Gpu,
            false,
            fresh_runtime(),
        ));
    }

    #[test]
    fn serve_batch_bypass_for_streaming_scheduler_and_cpu() {
        // Streaming decode (skip flag), CPU-class backend, and scheduler use
        // each independently force the direct path.
        assert!(!can_use_moonshine_serve_batch(
            true,
            false,
            GgmlCpuGraphBackend::Gpu,
            false,
            fresh_runtime(),
        ));
        assert!(!can_use_moonshine_serve_batch(
            false,
            false,
            GgmlCpuGraphBackend::Cpu,
            false,
            fresh_runtime(),
        ));
        assert!(!can_use_moonshine_serve_batch(
            false,
            false,
            GgmlCpuGraphBackend::Gpu,
            true,
            fresh_runtime(),
        ));
    }

    #[test]
    fn resolved_cpu_decode_plan_uses_native_first_max_without_reuse() {
        let resolved = ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
        );
        assert_eq!(
            resolved.output_contract(),
            GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits
        );
        assert_eq!(
            resolved.output_plan(),
            GgmlDecodeOutputPlan::NativeFirstMaxToken
        );
        assert_eq!(resolved.reuse_mode(), GgmlDecodeReuseMode::FreshGraph);
        assert_eq!(
            moonshine_greedy_step_output_mode(resolved, false),
            DeviceGreedyStepOutputMode::DeviceTop1
        );
        assert_eq!(
            moonshine_greedy_step_output_mode(resolved, true),
            DeviceGreedyStepOutputMode::FullLogits
        );
    }

    #[test]
    fn request_features_keep_moonshine_on_full_logits_through_shipped_combiner() {
        use crate::MOONSHINE_GGML_ADAPTER_ID;
        use crate::ggml_runtime::GgmlDecodeLogitsConsumers;
        use crate::models::device_greedy_token::decode_logits_consumers_for_request;

        for consumers in [
            decode_logits_consumers_for_request(MOONSHINE_GGML_ADAPTER_ID, true, false, false),
            decode_logits_consumers_for_request(MOONSHINE_GGML_ADAPTER_ID, false, true, false),
            decode_logits_consumers_for_request(MOONSHINE_GGML_ADAPTER_ID, false, false, true),
            GgmlDecodeLogitsConsumers::none().with_debug_logits(true),
        ] {
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
                Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
                GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
                consumers,
            );
            assert_eq!(
                moonshine_greedy_step_output_mode(resolved, false),
                DeviceGreedyStepOutputMode::FullLogits
            );
        }
    }
}
