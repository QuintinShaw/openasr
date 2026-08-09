use std::sync::Arc;

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
use super::graph_config::{moonshine_decoder_graph_config, moonshine_encoder_graph_config};
use super::lora::{
    MoonshineLoraError, moonshine_adapter_cache_fingerprint, resolve_moonshine_lora_adapter,
};
use super::prepared_runtime::{
    MoonshinePreparedRuntime, MoonshinePreparedRuntimeError, build_moonshine_prepared_runtime,
};
use crate::MOONSHINE_GGML_ADAPTER_ID;
use crate::NativeAsrSession;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
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
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
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

/// (pack content id, execution lane, adapter fingerprint). The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a runtime built from the old
/// bytes. The adapter fingerprint MUST stay in this key -- prepared encoder
/// graphs embed the adapter tensors, so reuse keyed only on the base pack
/// would be a correctness bug.
type MoonshineEncoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, String);
type MoonshineDecoderRuntimeCacheKey = (
    PackContentKey,
    ExecutionLaneKey,
    crate::models::seq2seq_decoder_state::Seq2SeqResidentCapacity,
    String,
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
        let backend = request.resolved_runtime.backend();
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
        )?;

        let audio_duration = audio_duration_seconds(&request.prepared_audio);
        let serve_batch_config = MoonshineServeBatchConfig::from_policy::<
            super::batched_decode::MoonshineFamily,
        >(request.request_options.serve_batch);
        let decoder_config = moonshine_decoder_graph_config(backend, false);
        let can_use_serve_batch = can_use_moonshine_serve_batch(
            skip_serve_batch,
            adapter.is_some(),
            decoder_config.backend.is_gpu_class(),
            decoder_config.use_scheduler,
        );
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
                    backend,
                    request.request_options.word_timestamps,
                    audio_duration,
                    adapter.clone(),
                    decoder_state,
                    Arc::clone(&request.execution_context.control),
                    request
                        .execution_context
                        .decode_work_progress_observer()
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
    ) -> Result<MoonshineEncoderRuntimeActor, MoonshineGgmlExecutorError> {
        let encoder_backend = moonshine_encoder_graph_config(backend).backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(encoder_backend),
            moonshine_adapter_cache_fingerprint(adapter.as_ref().map(resolved_lora_adapter)),
        );
        let preflight = preflight.clone();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared, adapter))),
            move |(preflight, prepared, adapter)| {
                let runtime = MoonshineEncoderGraphRuntime::new(
                    &prepared.encoder_weights,
                    prepared.metadata,
                    &preflight,
                    adapter.as_ref().map(resolved_lora_adapter),
                    backend,
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
        backend: GgmlCpuGraphBackend,
    ) -> Result<MoonshineDecoderRuntimeActor, MoonshineGgmlExecutorError> {
        let decoder_backend = moonshine_decoder_graph_config(backend, false).backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(decoder_backend),
            decoder_state.resident_capacity(),
            moonshine_adapter_cache_fingerprint(adapter.as_ref().map(resolved_lora_adapter)),
        );
        let preflight = preflight.clone();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared, adapter))),
            move |(preflight, prepared, adapter)| {
                let runtime = MoonshineDecoderGraphRuntime::new(
                    MoonshineDecoderRuntimeInput {
                        decoder_weights: &prepared.decoder_weights,
                        metadata: prepared.metadata,
                        decoder_state,
                        backend,
                    },
                    false,
                    &preflight,
                    adapter.as_ref().map(resolved_lora_adapter),
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
    ) -> Result<MoonshineEncoderOutput, MoonshineGgmlExecutorError> {
        let runtime = self.checkout_encoder_runtime(preflight, prepared, adapter, backend)?;
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
        backend: GgmlCpuGraphBackend,
        word_timestamps: bool,
        audio_duration_seconds: f32,
        adapter: Option<ResolvedLoraAdapterHandle>,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        control: Arc<crate::api::backend::TranscriptionControl>,
        decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
    ) -> Result<super::decoder_graph::MoonshineDecodeOutput, MoonshineGgmlExecutorError> {
        let tokenizer = prepared.tokenizer.clone();
        let metadata = prepared.metadata;
        let runtime =
            self.checkout_decoder_runtime(preflight, prepared, adapter, decoder_state, backend)?;
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
    decoder_backend_is_gpu_class: bool,
    decoder_uses_scheduler: bool,
) -> bool {
    !skip_serve_batch && !adapter_active && decoder_backend_is_gpu_class && !decoder_uses_scheduler
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
    use super::can_use_moonshine_serve_batch;

    #[test]
    fn serve_batch_is_allowed_only_on_direct_gpu_path_without_adapter() {
        // The only allowed combination: offline decode, no adapter, GPU-class
        // decoder backend, no scheduler.
        assert!(can_use_moonshine_serve_batch(false, false, true, false));
    }

    #[test]
    fn active_adapter_forces_serve_batch_bypass() {
        // OADP Phase 0 contract: an active dynamic adapter ALWAYS bypasses the
        // shared serve-batch worker (its pooled runtimes are adapter-free),
        // even when every other condition would allow serve-batch.
        assert!(!can_use_moonshine_serve_batch(false, true, true, false));
    }

    #[test]
    fn serve_batch_bypass_for_streaming_scheduler_and_cpu() {
        // Streaming decode (skip flag), CPU-class backend, and scheduler use
        // each independently force the direct path.
        assert!(!can_use_moonshine_serve_batch(true, false, true, false));
        assert!(!can_use_moonshine_serve_batch(false, false, false, false));
        assert!(!can_use_moonshine_serve_batch(false, false, true, true));
    }
}
