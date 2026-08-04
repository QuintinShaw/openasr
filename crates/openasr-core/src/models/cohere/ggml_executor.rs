use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

#[cfg(test)]
use std::path::{Path, PathBuf};

use super::batched_decode::{
    CohereServeBatchConfig, CohereServeBatchConfigFromPolicy, CohereServeBatchEngineRegistry,
    CohereServeBatchJob, cohere_serve_batch_decode_config,
    cohere_serve_batch_text_postprocess_kind, shutdown_cohere_serve_batch_engines,
    submit_cohere_serve_batch_job,
};
use super::decoder_graph::{
    CohereDecoderGraphError, CohereDecoderGraphRuntime,
    run_cohere_decoder_graph_short_form_with_runtime,
};
use super::encoder_graph::{CohereTranscribeEncoderError, CohereTranscribeEncoderGraphRuntime};
use super::frontend::{
    CohereTranscribeFrontendError, CohereTranscribeMelFeatures,
    cohere_transcribe_features_from_prepared_audio,
};
use super::graph_config::{cohere_decoder_graph_config, cohere_encoder_graph_config};
use super::prepared_runtime::{CoherePreparedRuntime, CoherePreparedRuntimeError};
use crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID;
use crate::NativeAsrSession;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::hparams::{
    COHERE_TRANSCRIBE_DECODER_LAYERS_KEY, COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{
    COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, OpenAsrArchitectureRegistry, OpenAsrBlockStackStrategy,
};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufRuntimeSourcePreflight};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_token_history::build_longform_token_history_carry;
use crate::models::ggml_asr_executor::{
    GgmlAsrCarryContext, GgmlAsrExecutionError, GgmlAsrExecutionResult,
    GgmlAsrExecutionViewRequest, GgmlAsrPreparedAudioView, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::prepared_runtime_cache::PreparedRuntimeHandle;
use crate::models::runtime_cache_coordinator::{PackContentKey, canonical_runtime_cache_path};
use crate::models::runtime_prepared_registry::{
    BuiltinPreparedRuntime, BuiltinPreparedRuntimeCache, BuiltinPreparedRuntimeRegistryError,
    PreparedRuntimeLookup,
};
use crate::models::seq2seq_decoder_state::Seq2SeqResidentCapacity;
use crate::models::system_memory_owner::SystemMemoryOwner;

const COHERE_EXECUTOR_ID: &str = crate::arch::COHERE_TRANSCRIBE_EXECUTOR_COMPONENT_ID;
const COHERE_STREAMING_EXECUTOR_ID: &str = "cohere-transcribe-ggml-snapshot-streaming-executor-v1";
const COHERE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const COHERE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;
use super::prompt::{
    COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT, build_cohere_initial_prompt_token_ids,
};
const COHERE_DEBUG_TIMINGS_ENV: &str = "OPENASR_COHERE_DEBUG_TIMINGS";
const COHERE_DEBUG_ENCODER_ENV: &str = "OPENASR_COHERE_DEBUG_ENCODER";

type CohereEncoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
/// (pack content id, backend, resident self/cross spans). The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a runtime built from the old
/// bytes. Logical per-chunk shapes do not belong in this key: the runtime
/// activates them inside the planner-reserved spans without reallocating.
type CohereDecoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, Seq2SeqResidentCapacity);

/// The graph runtimes are backend-owned (`GgmlCpuGraphRunner` is not `Send`)
/// and therefore must stay on one owner thread for their whole lifetime. The
/// prepared-runtime handle is deliberately part of the actor state rather
/// than merely a build-closure capture: it keeps the host lease that backs the
/// mmap/dequantized weights alive for as long as the native arena can refer to
/// them, including while the actor is idle in the checkout pool.
struct CohereEncoderRuntimeActorState {
    runtime: CohereTranscribeEncoderGraphRuntime,
    _prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
}

impl std::fmt::Debug for CohereEncoderRuntimeActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereEncoderRuntimeActorState")
            .finish_non_exhaustive()
    }
}

struct CohereDecoderRuntimeActorState {
    runtime: CohereDecoderGraphRuntime,
    _prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
}

impl std::fmt::Debug for CohereDecoderRuntimeActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereDecoderRuntimeActorState")
            .finish_non_exhaustive()
    }
}

type CohereEncoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    CohereEncoderRuntimeCacheKey,
    CohereEncoderRuntimeActorState,
>;
type CohereDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    CohereDecoderRuntimeCacheKey,
    CohereDecoderRuntimeActorState,
>;
type CohereEncoderRuntimeActor =
    PinnedRuntimeActorCheckout<CohereEncoderRuntimeCacheKey, CohereEncoderRuntimeActorState>;
type CohereDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<CohereDecoderRuntimeCacheKey, CohereDecoderRuntimeActorState>;

// Test-only build counters, incremented from inside the two caches' `build`
// closures below -- lets a same-thread test pin "a second call against the
// same pack content id reuses the cached runtime" as a structural fact
// (build count stays put across two calls) rather than inferring cache-hit
// behavior from wall-clock timing. Mirrors
// `moss_transcribe_diarize::executor`'s `MOSS_TD_ENCODER_RUNTIME_BUILD_COUNT`.
#[cfg(test)]
thread_local! {
    static COHERE_ENCODER_RUNTIME_BUILD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static COHERE_DECODER_RUNTIME_BUILD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn cohere_runtime_build_counts_for_test() -> (usize, usize) {
    (
        COHERE_ENCODER_RUNTIME_BUILD_COUNT.with(std::cell::Cell::get),
        COHERE_DECODER_RUNTIME_BUILD_COUNT.with(std::cell::Cell::get),
    )
}

#[derive(Debug, Error)]
enum CohereTranscribeGgmlExecutorError {
    #[error("cohere-transcribe ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("cohere-transcribe ggml executor runtime preparation failed: {reason}")]
    PreparedRuntimeFailed { reason: String },
    #[error("cohere-transcribe ggml executor frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("cohere-transcribe ggml executor encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("cohere-transcribe ggml executor decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("cohere-transcribe {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

/// Resolves a cohere block-stack stage's `layer_count_hparam` to the count parsed
/// from the GGUF hparams (NOT `layers.len()` — see the [`LayerCountResolver`]
/// honesty contract), so `validate_stage_against_descriptor` can cross-check each
/// materialized stack against the descriptor's declared key.
struct CohereLayerCountResolver {
    encoder_layers: usize,
    decoder_layers: usize,
}

impl LayerCountResolver for CohereLayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY => Some(self.encoder_layers),
            COHERE_TRANSCRIBE_DECODER_LAYERS_KEY => Some(self.decoder_layers),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CohereTranscribeGgmlExecutor {
    runtime_cache_by_path: BuiltinPreparedRuntimeCache,
    encoder_runtimes: Arc<CohereEncoderRuntimePool>,
    decoder_runtimes: Arc<CohereDecoderRuntimePool>,
    serve_batch_engines: CohereServeBatchEngineRegistry,
}

impl Default for CohereTranscribeGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            COHERE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            COHERE_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            runtime_cache_by_path: BuiltinPreparedRuntimeCache::default(),
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-cohere-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-cohere-decoder-owner",
                limits,
            )),
            serve_batch_engines: CohereServeBatchEngineRegistry::default(),
        }
    }
}

impl CohereTranscribeGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, CohereTranscribeGgmlExecutorError> {
        if request.selected_family.adapter_id != COHERE_TRANSCRIBE_GGML_ADAPTER_ID {
            return Err(CohereTranscribeGgmlExecutorError::AdapterMismatch {
                expected: COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let decoder_state =
            crate::models::seq2seq_decoder_state::Seq2SeqDecoderState::from_request_state(
                &request.decoder_state,
                super::capacity::COHERE_DECODER_STATE_IDS,
            )
            .map_err(|error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;

        let preflight_start = debug_timing_start();
        let preflight = request.runtime_source_preflight();
        emit_cohere_debug_timing_if_enabled("runtime_preflight", preflight_start, None);
        let prepared_runtime_start = debug_timing_start();
        let prepared_runtime_owner = self.runtime_cache_by_path.prepared_runtime_for_preflight(
            PreparedRuntimeLookup {
                model_architecture: request.selected_family.model_architecture,
                preflight,
                backend: request.resolved_runtime.backend(),
            },
            map_prepared_runtime_registry_error,
            cohere_runtime_cache_slot_unavailable,
        )?;
        let prepared_runtime = prepared_runtime_owner
            .as_ref()
            .as_cohere_transcribe()
            .ok_or_else(|| CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                reason: format!(
                    "prepared runtime registry returned non-cohere runtime for architecture '{}'",
                    request.selected_family.model_architecture
                ),
            })?;
        emit_cohere_debug_timing_if_enabled("prepared_runtime", prepared_runtime_start, None);
        let frontend_start = debug_timing_start();
        let features = cohere_transcribe_features_from_prepared_audio(
            &request.prepared_audio,
            &prepared_runtime.frontend_plan,
        )
        .map_err(map_frontend_error)?;
        emit_cohere_debug_timing_if_enabled(
            "frontend",
            frontend_start,
            Some(format!(
                "frames={} mels={}",
                features.n_frames, features.n_mels
            )),
        );
        emit_cohere_debug_feature_preview_if_enabled(&features);
        self.decode_with_prepared_runtime(
            preflight,
            request,
            prepared_runtime,
            Arc::clone(&prepared_runtime_owner),
            features,
            decoder_state,
            skip_serve_batch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_prepared_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        request: &GgmlAsrExecutionViewRequest,
        prepared_runtime: &CoherePreparedRuntime,
        prepared_runtime_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        features: CohereTranscribeMelFeatures,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, CohereTranscribeGgmlExecutorError> {
        let runtime_source = &preflight.runtime_source;
        let runtime_path = runtime_source.path();
        // Make the block-stack descriptor load-bearing (P4 S5e/S5f): fail closed
        // unless the conformer-encoder + seq2seq-decoder stacks this runtime
        // materialized agree with the cohere descriptor's declared shape / block
        // kinds / tensor-name scopes / layer counts. A drift means the data and
        // the hand-wired composers disagree — never silently build the wrong thing.
        let cohere_descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID);
        let cohere_block_stack =
            cohere_descriptor.as_ref().and_then(|descriptor| {
                match &descriptor.topology_contract.block_stack {
                    OpenAsrBlockStackStrategy::Shared(stack) => Some(stack),
                    OpenAsrBlockStackStrategy::ArchitectureGraph { .. } => None,
                }
            });
        let layer_resolver = CohereLayerCountResolver {
            encoder_layers: prepared_runtime.metadata.encoder_layers,
            decoder_layers: prepared_runtime.metadata.decoder_layers,
        };
        validate_stage_against_descriptor(
            COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            cohere_block_stack,
            OpenAsrStageRole::Encoder,
            OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::ConformerBlock,
                tensor_name_scope: "enc.blk",
                family_layer_count: prepared_runtime.encoder_weights.layers.len(),
            },
            &layer_resolver,
        )
        .map_err(
            |error| CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                reason: format!("cohere encoder block-stack descriptor mismatch: {error:?}"),
            },
        )?;
        validate_stage_against_descriptor(
            COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            cohere_block_stack,
            OpenAsrStageRole::Decoder,
            OpenAsrOrchestrationShape::Seq2SeqEncoderDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::Seq2SeqDecoderLayer,
                tensor_name_scope: "dec.blk",
                family_layer_count: prepared_runtime.decoder_weights.layers.len(),
            },
            &layer_resolver,
        )
        .map_err(
            |error| CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                reason: format!("cohere decoder block-stack descriptor mismatch: {error:?}"),
            },
        )?;

        let prompt = prepared_runtime
            .decode_prompt(
                request.request_options.language.as_deref(),
                &request.request_options,
            )
            .map_err(|error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;
        let initial_prompt_tokens = build_cohere_initial_prompt_token_ids(
            prompt.token_ids,
            &request.request_options,
            prepared_runtime.metadata,
        )
        .map_err(|error| CohereTranscribeGgmlExecutorError::DecoderFailed {
            reason: error.to_string(),
        })?;
        let eos_token_id = prompt.eos_token_id.ok_or_else(|| {
            CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: "cohere decode prompt is missing EOS token id".to_string(),
            }
        })?;
        let encoder_start = debug_timing_start();
        let encoder_output = self
            .encode_with_owned_cohere_encoder_runtime(
                preflight,
                features,
                request.resolved_runtime.backend(),
                Arc::clone(&prepared_runtime_owner),
            )
            .map_err(map_encoder_error)?;
        emit_cohere_debug_timing_if_enabled(
            "encoder",
            encoder_start,
            Some(format!(
                "frames={} hidden={}",
                encoder_output.frame_count, encoder_output.hidden_size
            )),
        );
        emit_cohere_debug_encoder_preview_if_enabled(&encoder_output);
        let decoder_start = debug_timing_start();
        let prefer_cpu_decoder = request
            .request_options
            .prefer_cpu_decoder_for_multichunk_metal;
        let audio_duration = audio_duration_seconds(&request.prepared_audio);
        let serve_batch_config =
            CohereServeBatchConfig::from_server_policy(request.request_options.serve_batch);
        let decoder_config =
            cohere_decoder_graph_config(request.resolved_runtime.backend(), prefer_cpu_decoder);
        let can_use_serve_batch = !skip_serve_batch
            && decoder_config.backend.is_gpu_class()
            && !decoder_config.use_scheduler;
        let decode = if let Some(serve_batch_config) =
            serve_batch_config.filter(|_| can_use_serve_batch)
        {
            let decode_config = cohere_serve_batch_decode_config(
                &initial_prompt_tokens,
                prepared_runtime.metadata,
                encoder_output.frame_count,
                decoder_state,
                eos_token_id,
                &prepared_runtime.tokenizer,
                request.request_options.phrase_bias.as_ref(),
            )
            .map_err(|error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;
            submit_cohere_serve_batch_job(
                &self.serve_batch_engines,
                serve_batch_config,
                CohereServeBatchJob {
                    runtime_cache_path: canonical_runtime_cache_path(runtime_path),
                    build_identity:
                        crate::models::ggml_asr_executor::serve_batch_build_identity_for_request(
                            &request.request_options,
                            "cohere",
                            decoder_config.backend,
                            runtime_source,
                        ),
                    backend: decoder_config.backend,
                    uses_scheduler: decoder_config.use_scheduler,
                    decoder_weights: prepared_runtime.decoder_weights.clone(),
                    decoder_state,
                    tokenizer: prepared_runtime.tokenizer.clone(),
                    metadata: prepared_runtime.metadata,
                    // Moved (not cloned): this branch is the last use of
                    // `encoder_output` -- the `else` branch below only
                    // borrows it, and nothing reads it after the if/else.
                    encoder_output,
                    decode_config,
                    text_postprocess_kind: cohere_serve_batch_text_postprocess_kind().map_err(
                        |error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                            reason: error.to_string(),
                        },
                    )?,
                    word_timestamps: request.request_options.word_timestamps,
                    audio_duration_seconds: audio_duration,
                    prefer_cpu_backend: prefer_cpu_decoder,
                    execution_context: Arc::clone(&request.execution_context),
                },
            )
            .map_err(|error| match error.unavailable_retryable() {
                Some(retryable) => CohereTranscribeGgmlExecutorError::ServeBatchUnavailable {
                    reason: error.to_string(),
                    retryable,
                },
                None => CohereTranscribeGgmlExecutorError::DecoderFailed {
                    reason: error.to_string(),
                },
            })?
        } else {
            self.decode_with_owned_cohere_decoder_runtime(
                preflight.runtime_source.content_id(),
                &prepared_runtime.tokenizer,
                prepared_runtime.metadata,
                &initial_prompt_tokens,
                eos_token_id,
                encoder_output,
                decoder_state,
                request.request_options.phrase_bias.as_ref(),
                request.resolved_runtime.backend(),
                prefer_cpu_decoder,
                request.request_options.word_timestamps,
                audio_duration,
                &request.execution_context.control,
                Arc::clone(&prepared_runtime_owner),
            )
            .map_err(map_decoder_error)?
        };
        emit_cohere_debug_timing_if_enabled(
            "decoder",
            decoder_start,
            Some(format!(
                "generated_tokens={} text_len={}",
                decode.generated_tokens.len(),
                decode.transcription.text.len()
            )),
        );
        emit_cohere_debug_tokens_if_enabled(
            &prepared_runtime.tokenizer,
            &initial_prompt_tokens,
            &decode.generated_tokens,
            &decode.transcription.text,
        );
        let carry_prompt_token_ids =
            build_cohere_carry_prompt_token_ids(&request.request_options, &decode.generated_tokens);
        // cohere-transcribe's diarized mode emits inline speaker turns but no
        // audio timestamps the decode could be anchored to, and its plain mode
        // is a single whole-buffer span -- so the cut point has no honest
        // second to name. See `DecodeTruncation::transcript_covers_up_to_seconds`.
        let decode_truncation = decode.stop_reason.into_decode_truncation(None);
        Ok(GgmlAsrExecutionResult {
            transcription: decode.transcription,
            carry_context: carry_prompt_token_ids.map(|prompt_token_ids| GgmlAsrCarryContext {
                prompt_text: None,
                prompt_token_ids: Some(prompt_token_ids),
            }),
            decode_truncation,
        })
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
        shutdown_cohere_serve_batch_engines(&self.serve_batch_engines);
        self.runtime_cache_by_path.evict_content_id(pack_content_id);
    }
}

// Covers both a genuinely poisoned slot mutex and a build attempt that
// panicked and was caught (mutex stays unpoisoned, slot stays empty,
// retryable) -- see `PreparedRuntimeCache::get_or_try_insert_with`. Either way
// the cache could not deliver a prepared runtime for this attempt; the
// caller's next request retries clean.
fn cohere_runtime_cache_slot_unavailable() -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
        reason:
            "cohere runtime cache slot unavailable (poisoned lock or a caught build panic); retry"
                .to_string(),
    }
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudioView) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

impl CohereTranscribeGgmlExecutor {
    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> CohereTranscribeGgmlExecutorError {
        CohereTranscribeGgmlExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn checkout_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<CohereEncoderRuntimeActor, CohereTranscribeGgmlExecutorError> {
        let encoder_backend = cohere_encoder_graph_config(backend).backend;
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(encoder_backend),
        );
        let preflight = preflight.clone();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                #[cfg(test)]
                COHERE_ENCODER_RUNTIME_BUILD_COUNT.with(|count| count.set(count.get() + 1));
                Ok((0, (preflight, prepared_owner)))
            },
            move |(preflight, prepared_owner)| {
                let prepared_runtime =
                    prepared_owner
                        .as_ref()
                        .as_cohere_transcribe()
                        .ok_or_else(|| {
                            CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                                reason: "encoder actor received a non-cohere prepared runtime"
                                    .to_string(),
                            }
                        })?;
                let runtime = CohereTranscribeEncoderGraphRuntime::new(
                    &prepared_runtime.encoder_weights,
                    prepared_runtime.metadata,
                    &preflight,
                    backend,
                )
                .map_err(|error| {
                    CohereTranscribeGgmlExecutorError::EncoderFailed {
                        reason: error.to_string(),
                    }
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    CohereEncoderRuntimeActorState {
                        runtime,
                        _prepared_owner: prepared_owner,
                    },
                ))
            },
            |error| Self::map_actor_error("encoder", error),
        )
    }

    fn encode_with_owned_cohere_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        features: CohereTranscribeMelFeatures,
        backend: GgmlCpuGraphBackend,
        prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
    ) -> Result<super::encoder_graph::CohereTranscribeEncoderOutput, CohereTranscribeEncoderError>
    {
        let actor = self
            .checkout_encoder_runtime(preflight, prepared_owner, backend)
            .map_err(|error| CohereTranscribeEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        actor
            .call_mut(move |state| state.runtime.encode(&features))
            .map_err(|error| CohereTranscribeEncoderError::GraphExecutionFailed {
                reason: error.to_string(),
            })?
    }

    fn checkout_decoder_runtime(
        &self,
        pack_content_id: &str,
        prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        cross_hidden_size: usize,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
    ) -> Result<CohereDecoderRuntimeActor, CohereTranscribeGgmlExecutorError> {
        let decoder_backend = cohere_decoder_graph_config(backend, prefer_cpu_backend).backend;
        let key = (
            PackContentKey::new(pack_content_id),
            current_execution_lane_key(decoder_backend),
            decoder_state.resident_capacity(),
        );
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                #[cfg(test)]
                COHERE_DECODER_RUNTIME_BUILD_COUNT.with(|count| count.set(count.get() + 1));
                Ok((0, prepared_owner))
            },
            move |prepared_owner| {
                let prepared_runtime =
                    prepared_owner
                        .as_ref()
                        .as_cohere_transcribe()
                        .ok_or_else(|| {
                            CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                                reason: "decoder actor received a non-cohere prepared runtime"
                                    .to_string(),
                            }
                        })?;
                let runtime = CohereDecoderGraphRuntime::new(
                    &prepared_runtime.decoder_weights,
                    prepared_runtime.metadata,
                    decoder_state,
                    cross_hidden_size,
                    backend,
                    prefer_cpu_backend,
                )
                .map_err(|error| {
                    CohereTranscribeGgmlExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    }
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    CohereDecoderRuntimeActorState {
                        runtime,
                        _prepared_owner: prepared_owner,
                    },
                ))
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_owned_cohere_decoder_runtime(
        &self,
        pack_content_id: &str,
        tokenizer: &super::tokenizer::CohereTranscribeTokenizer,
        metadata: super::runtime_contract::CohereTranscribeExecutionMetadata,
        initial_prompt_tokens: &[u32],
        eos_token_id: u32,
        encoder_output: super::encoder_graph::CohereTranscribeEncoderOutput,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        phrase_bias: Option<&crate::PhraseBiasConfig>,
        backend: GgmlCpuGraphBackend,
        prefer_cpu_backend: bool,
        word_timestamps: bool,
        audio_duration_seconds: f32,
        control: &Arc<crate::TranscriptionControl>,
        prepared_owner: PreparedRuntimeHandle<BuiltinPreparedRuntime>,
    ) -> Result<super::decoder_graph::CohereDecoderGraphDecodeOutput, CohereDecoderGraphError> {
        let actor = self
            .checkout_decoder_runtime(
                pack_content_id,
                prepared_owner,
                decoder_state,
                encoder_output.hidden_size,
                backend,
                prefer_cpu_backend,
            )
            .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                reason: format!("decoder runtime actor checkout failed: {error}"),
            })?;
        let tokenizer = tokenizer.clone();
        let initial_prompt_tokens = initial_prompt_tokens.to_vec();
        let encoder_output = Arc::new(encoder_output);
        let phrase_bias = phrase_bias.cloned();
        let control = Arc::clone(control);
        actor
            .call_mut(move |state| {
                state.runtime.activate_decoder_state(decoder_state)?;
                run_cohere_decoder_graph_short_form_with_runtime(
                    &mut state.runtime,
                    &tokenizer,
                    metadata,
                    &initial_prompt_tokens,
                    eos_token_id,
                    &encoder_output,
                    phrase_bias.as_ref(),
                    word_timestamps,
                    audio_duration_seconds,
                    &control,
                )
            })
            .map_err(|error| CohereDecoderGraphError::GraphExecutionFailed {
                reason: format!("decoder runtime actor call failed: {error}"),
            })?
    }
}

impl GgmlAsrViewExecutor for CohereTranscribeGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        CohereTranscribeGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        COHERE_EXECUTOR_ID
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
                super::capacity::plan_cohere_decoder_state,
                super::capacity::COHERE_DECODER_STATE_STREAMS,
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
            && let Some(prepared) = owner.as_ref().as_cohere_transcribe()
        {
            let plan =
                super::capacity::plan_cohere_decoder_state_with_prepared_runtime(input, prepared)?;
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
            .map_err(|error| cohere_execute_error_to_ggml(self, error, request))
    }

    fn unload_idle_state(&self) {
        shutdown_cohere_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.runtime_cache_by_path.clear();
    }
}

impl CohereTranscribeGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| cohere_execute_error_to_ggml(self, error, request))
    }
}

fn cohere_execute_error_to_ggml(
    executor: &CohereTranscribeGgmlExecutor,
    error: CohereTranscribeGgmlExecutorError,
    request: &GgmlAsrExecutionViewRequest,
) -> GgmlAsrExecutionError {
    match error {
        CohereTranscribeGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: GgmlAsrViewExecutor::executor_id(executor),
            adapter_id: request.selected_family.adapter_id,
            reason: error.to_string(),
        },
    }
}

impl GgmlAsrStreamingExecutor for CohereTranscribeGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        COHERE_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            COHERE_STREAMING_EXECUTOR_ID,
            COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            "cohere-transcribe",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            CohereTranscribeGgmlExecutor::execute_streaming,
        )
    }

    fn unload_idle_state(&self) {
        shutdown_cohere_serve_batch_engines(&self.serve_batch_engines);
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.runtime_cache_by_path.clear();
    }
}

fn map_prepared_runtime_error(
    error: CoherePreparedRuntimeError,
) -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
        reason: error.to_string(),
    }
}

fn map_prepared_runtime_registry_error(
    error: BuiltinPreparedRuntimeRegistryError,
) -> CohereTranscribeGgmlExecutorError {
    match error {
        BuiltinPreparedRuntimeRegistryError::CohereTranscribeBuild { source } => {
            map_prepared_runtime_error(source)
        }
        other => CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
            reason: other.to_string(),
        },
    }
}

fn map_frontend_error(error: CohereTranscribeFrontendError) -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::FrontendFailed {
        reason: error.to_string(),
    }
}

fn map_encoder_error(error: CohereTranscribeEncoderError) -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::EncoderFailed {
        reason: error.to_string(),
    }
}

fn map_decoder_error(error: CohereDecoderGraphError) -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::DecoderFailed {
        reason: error.to_string(),
    }
}

fn emit_cohere_debug_tokens_if_enabled(
    tokenizer: &super::tokenizer::CohereTranscribeTokenizer,
    prompt_tokens: &[u32],
    generated_tokens: &[u32],
    decoded_text: &str,
) {
    if std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_none() {
        return;
    }
    let prompt_debug = prompt_tokens
        .iter()
        .map(|token_id| {
            format!(
                "{}:{}",
                token_id,
                tokenizer
                    .token_content_by_id(*token_id)
                    .unwrap_or("<missing>")
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let generated_debug = generated_tokens
        .iter()
        .map(|token_id| {
            format!(
                "{}:{}",
                token_id,
                tokenizer
                    .token_content_by_id(*token_id)
                    .unwrap_or("<missing>")
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    eprintln!("openasr cohere prompt tokens: {prompt_debug}");
    eprintln!("openasr cohere generated tokens: {generated_debug}");
    eprintln!("openasr cohere decoded text: {decoded_text}");
}

fn emit_cohere_debug_feature_preview_if_enabled(features: &CohereTranscribeMelFeatures) {
    if std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_none()
        || features.n_frames == 0
        || features.n_mels == 0
    {
        return;
    }
    let m0 = (0..features.n_frames.min(5))
        .map(|frame_idx| format!("{:.4}", features.data[frame_idx * features.n_mels]))
        .collect::<Vec<_>>()
        .join(", ");
    let t0 = (0..features.n_mels.min(5))
        .map(|mel_idx| format!("{:.4}", features.data[mel_idx]))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("openasr cohere mel m=0, t=0..4: [{m0}]");
    eprintln!("openasr cohere mel t=0, m=0..4: [{t0}]");
}

fn emit_cohere_debug_encoder_preview_if_enabled(
    encoder_output: &super::encoder_graph::CohereTranscribeEncoderOutput,
) {
    if std::env::var_os(COHERE_DEBUG_ENCODER_ENV).is_none()
        || encoder_output.frame_count == 0
        || encoder_output.hidden_size == 0
        || encoder_output.rows.is_empty()
    {
        return;
    }

    let first_values = encoder_output
        .rows
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    let first_frame =
        &encoder_output.rows[..encoder_output.hidden_size.min(encoder_output.rows.len())];
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for value in first_frame {
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
        sum += f64::from(*value);
    }
    let mean_value = sum / first_frame.len() as f64;
    eprintln!(
        "openasr cohere encoder: frames={} hidden={} first8=[{}] frame0_mean={:.6} frame0_min={:.6} frame0_max={:.6}",
        encoder_output.frame_count,
        encoder_output.hidden_size,
        first_values,
        mean_value,
        min_value,
        max_value
    );
}

fn debug_timing_start() -> Option<Instant> {
    std::env::var_os(COHERE_DEBUG_TIMINGS_ENV).map(|_| Instant::now())
}

fn emit_cohere_debug_timing_if_enabled(
    stage: &str,
    start: Option<Instant>,
    detail: Option<String>,
) {
    let Some(start) = start else {
        return;
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    match detail {
        Some(detail) => {
            eprintln!("openasr cohere timing: stage={stage} elapsed_ms={elapsed_ms:.2} {detail}")
        }
        None => eprintln!("openasr cohere timing: stage={stage} elapsed_ms={elapsed_ms:.2}"),
    }
}

fn build_cohere_carry_prompt_token_ids(
    request_options: &crate::GgmlAsrExecutionOptions,
    generated_tokens: &[u32],
) -> Option<Vec<u32>> {
    build_longform_token_history_carry(
        request_options.longform_prompt_carry_enabled(),
        request_options.prompt_token_ids.clone().unwrap_or_default(),
        generated_tokens,
        COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    )
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;
    use crate::api::backend::{NativeBackend, TranscriptionBackend};
    use crate::arch::builtin_adapter_descriptor;
    use crate::models::serve_batch_env::OPENASR_SERVE_BATCH_ENV;
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrPreparedAudioView,
        LongFormOptions, TranscriptionRequest,
    };

    fn sample_wav_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist")
    }

    fn plan_request_decoder_state(
        request: &mut GgmlAsrExecutionViewRequest<'_>,
        envelope_samples: Option<usize>,
    ) {
        use std::num::NonZeroU32;

        let preflight = request.runtime_source_preflight();
        let sample_rate = NonZeroU32::new(request.prepared_audio.sample_rate_hz)
            .expect("test sample rate is non-zero");
        let invocation = crate::capacity::topology::InvocationShapeInput::new(
            sample_rate,
            request.prepared_audio.samples_f32.len(),
        )
        .expect("valid cohere test invocation");
        let envelope = crate::capacity::topology::InvocationEnvelope::new(
            sample_rate,
            envelope_samples
                .unwrap_or(invocation.samples())
                .max(invocation.samples()),
        )
        .expect("valid cohere test envelope");
        let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput {
            preflight,
            invocation,
            envelope,
            request_options: &request.request_options,
            backend: request.resolved_runtime.backend(),
        };
        let plan = super::super::capacity::plan_cohere_decoder_state(&planning_input)
            .expect("plan cohere decoder state");
        request.decoder_state =
            crate::models::ggml_asr_executor::GgmlAsrDecoderState::planned_for_test(plan, envelope);
    }

    fn runtime_ready_request(runtime_path: PathBuf) -> GgmlAsrExecutionViewRequest<'static> {
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &runtime_path,
            )
            .expect("cohere runtime fixture must pass preflight");
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(
                crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                    sample_wav_fixture_path(),
                    "cohere test",
                    "cohere test",
                )
                .expect("sample wav should load"),
            ),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_request_decoder_state(&mut request, None);
        request
    }

    #[test]
    fn cohere_executor_reaches_decode_boundary_with_runtime_ready_fixture() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let executor = CohereTranscribeGgmlExecutor::default();
            let result = executor
                .execute_view(&runtime_ready_request(runtime_path))
                .expect("executor should produce a best-effort transcription");
            assert!(result.transcription.text.is_ascii() || !result.transcription.text.is_empty());
            assert!(result.transcription.segments.is_empty());
        });
    }

    #[test]
    fn cohere_fixture_decode_is_invariant_to_larger_stable_resident_envelope() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let executor = CohereTranscribeGgmlExecutor::default();
            let mut request = runtime_ready_request(runtime_path);
            plan_request_decoder_state(&mut request, Some(60 * 16_000));
            let result = executor
                .execute_view(&request)
                .expect("resident headroom must not change the active logical decode");
            assert!(result.transcription.text.is_ascii() || !result.transcription.text.is_empty());
        });
    }

    /// The family encoder/decoder TLS runtime caches key on the
    /// already-open source's content id
    /// ([`PackContentKey::for_runtime_source`]) instead of a second,
    /// weaker path-based identity. Structural proof (build counters, not
    /// timing -- see `moss_transcribe_diarize::executor`'s precedent):
    ///
    /// 1. A second `execute()` against the *same unchanged bytes* (even
    ///    through a fresh `execute()` call, which re-validates and reopens
    ///    the path into a brand new [`crate::GgmlRuntimeSource`] instance every
    ///    time -- exactly like two independent production requests) must
    ///    hit the cached encoder/decoder runtimes, not rebuild them: the
    ///    content id survives across independent opens of the same bytes.
    /// 2. Two DIFFERENT packs (distinct `model_id`, hence distinct content
    ///    and distinct content ids) are each cached under their own key:
    ///    building/using one pack's runtime must not evict or rebuild the
    ///    other's -- the healthy sibling keeps hitting its own cache slot.
    #[test]
    fn cohere_encoder_and_decoder_caches_key_on_content_id_across_independent_opens() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let path_a = temp.path().join("cohere-runtime-a.gguf");
            let path_b = temp.path().join("cohere-runtime-b.gguf");
            let spec_a = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-fixture-a");
            let spec_b = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-fixture-b");
            write_tiny_gguf_runtime_source(&path_a, &spec_a).expect("write fixture a");
            write_tiny_gguf_runtime_source(&path_b, &spec_b).expect("write fixture b");

            let executor = CohereTranscribeGgmlExecutor::default();

            // First execute() against pack A builds both caches once.
            executor
                .execute_view(&runtime_ready_request(path_a.clone()))
                .expect("pack a first execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (1, 1),
                "first execute against pack a must build the encoder and decoder runtimes exactly once"
            );

            // A second execute() against the SAME path -- a fresh
            // `GgmlRuntimeSource` open every time (no cached preflight on
            // the request) -- must still hit both caches: content id, not
            // source-instance identity, is what the key carries.
            executor
                .execute_view(&runtime_ready_request(path_a.clone()))
                .expect("pack a second execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (1, 1),
                "a second execute against unchanged pack-a bytes must reuse the cached runtimes, \
                 not rebuild them, even though the request opened a brand new GgmlRuntimeSource"
            );

            // Pack B is a genuinely different pack (different content id).
            // Building its runtimes must not disturb pack A's cached slot.
            executor
                .execute_view(&runtime_ready_request(path_b.clone()))
                .expect("pack b first execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (2, 2),
                "pack b's first execute must build its own runtimes (a distinct content id), \
                 on top of pack a's already-cached ones"
            );

            // Pack A must still be a cache hit: pack B's distinct key never
            // evicted or clobbered pack A's healthy, resident sibling entry.
            executor
                .execute_view(&runtime_ready_request(path_a))
                .expect("pack a third execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (2, 2),
                "pack a must remain a cache hit after pack b was built -- a healthy sibling \
                 pack must never be evicted or rebuilt by an unrelated pack's cache activity"
            );

            // And pack B must likewise still be resident.
            executor
                .execute_view(&runtime_ready_request(path_b))
                .expect("pack b second execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (2, 2),
                "pack b must remain a cache hit on its own subsequent execute"
            );
        });
    }

    #[test]
    fn cohere_cached_runtime_survives_a_rename_based_pack_replacement_at_its_path() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let path = temp.path().join("cohere-runtime-replace.gguf");
            let staging_old = temp.path().join("cohere-runtime-replace-old.gguf");
            let staging_new = temp.path().join("cohere-runtime-replace-new.gguf");

            let spec_old =
                TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-fixture-replace-old");
            let spec_new =
                TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-fixture-replace-new");
            write_tiny_gguf_runtime_source(&staging_old, &spec_old).expect("write old fixture");
            write_tiny_gguf_runtime_source(&staging_new, &spec_new).expect("write new fixture");
            std::fs::rename(&staging_old, &path).expect("place initial pack at path");

            // Open (and hold) the source before the pack at `path` is ever
            // replaced -- the same shape a caller with an already-cached
            // runtime built from this exact mapping would be in.
            let old_runtime_source =
                crate::validate_ggml_runtime_source_path(&path).expect("open old source");
            let old_content_id = old_runtime_source.content_id().to_string();
            let old_preflight = crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(&old_runtime_source)
                .expect("build preflight from held old source");

            let executor = CohereTranscribeGgmlExecutor::default();

            // Build the cache entry keyed on the OLD content id by handing
            // the request an explicit preflight built from the held source,
            // instead of letting it re-resolve `path` itself.
            let mut request = runtime_ready_request(path.clone());
            request.verified_pack =
                crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                    old_preflight.clone(),
                    crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                );
            executor
                .execute_view(&request)
                .expect("first execute against old pack via held preflight");
            assert_eq!(cohere_runtime_build_counts_for_test(), (1, 1));

            // Replace the pack at `path` via a RENAME, not an in-place
            // `fs::write` -- an in-place write can mutate pages a live
            // `MAP_SHARED` mapping observes and would not prove that an
            // already-held mapping is untouched by a path-level replacement.
            std::fs::rename(&staging_new, &path).expect("replace pack via rename");

            // The already-held old runtime source keeps reading its own,
            // untouched mapping: its content id must not change, and reusing
            // it (the request carries the held preflight directly, without a
            // path resolver or re-open) must still hit the OLD content id's
            // cache entry.
            assert_eq!(
                old_runtime_source.content_id(),
                old_content_id,
                "an already-open mapping's content id must not change just because the path \
                 it was opened from was later replaced"
            );
            let mut old_request = runtime_ready_request(path.clone());
            old_request.verified_pack =
                crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                    old_preflight,
                    crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                );
            executor
                .execute_view(&old_request)
                .expect("second execute reusing the held old preflight after replacement");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (1, 1),
                "an already-held runtime source must keep serving from -- and hitting the cache \
                 entry for -- its own mapping after the path it was opened from is replaced"
            );

            // A FRESH resolution of the same (now-replaced) path observes
            // the new bytes, gets a different content id, and is a genuine
            // cache miss that rebuilds -- without disturbing the old content
            // id's entry (the build count only grows by one, it is not
            // reset).
            let fresh_source =
                crate::validate_ggml_runtime_source_path(&path).expect("open replaced source");
            assert_ne!(
                fresh_source.content_id(),
                old_content_id,
                "the replaced pack must produce a different content id than the original"
            );
            let fresh_request = runtime_ready_request(path);
            executor
                .execute_view(&fresh_request)
                .expect("execute against freshly-resolved replaced pack");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (2, 2),
                "resolving the replaced path fresh must observe the new content and rebuild \
                 exactly once, on top of (not instead of) the old content id's cached entry"
            );
        });
    }

    #[test]
    fn cohere_lru_eviction_targets_the_least_recently_used_pack_and_spares_siblings() {
        with_forced_cpu_backend_for_test(|| {
            assert_eq!(
                COHERE_RUNTIME_ACTOR_MAX_IDLE_ENTRIES, 4,
                "this test drives exactly capacity + 1 distinct packs; update the pack count \
                 alongside this constant if it ever changes"
            );

            let temp = tempfile::tempdir().expect("tempdir");
            let paths: Vec<PathBuf> = (1..=5)
                .map(|index| {
                    let path = temp.path().join(format!("cohere-evict-{index}.gguf"));
                    let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(format!(
                        "cohere-fixture-evict-{index}"
                    ));
                    write_tiny_gguf_runtime_source(&path, &spec).expect("write fixture");
                    path
                })
                .collect();

            let executor = CohereTranscribeGgmlExecutor::default();

            // Fill the capacity-4 cache with packs 1..4, oldest first.
            for path in &paths[0..4] {
                executor
                    .execute_view(&runtime_ready_request(path.clone()))
                    .expect("fill cache execute");
            }
            assert_eq!(cohere_runtime_build_counts_for_test(), (4, 4));

            // Pack 5 is the 5th distinct content id: it must evict pack 1,
            // the least recently used entry (never touched since its
            // initial insert), and build its own runtimes.
            executor
                .execute_view(&runtime_ready_request(paths[4].clone()))
                .expect("fifth pack execute");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (5, 5),
                "the fifth distinct pack must build (it cannot fit alongside the other four \
                 without an eviction)"
            );

            // The three siblings that were never evicted must still hit.
            for path in &paths[1..4] {
                executor
                    .execute_view(&runtime_ready_request(path.clone()))
                    .expect("surviving sibling execute");
            }
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (5, 5),
                "packs 2-4 must remain cache hits -- evicting pack 1 for pack 5 must not \
                 collaterally evict or rebuild a healthy sibling"
            );

            // Pack 1, the evicted entry, must be a genuine miss and rebuild.
            executor
                .execute_view(&runtime_ready_request(paths[0].clone()))
                .expect("re-request evicted pack");
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (6, 6),
                "re-requesting the evicted pack must rebuild it, proving eviction actually \
                 dropped its cache entry rather than merely reordering it"
            );

            // Packs 2-4 must still be unaffected by pack 1's return (which
            // evicts pack 5, now the least recently used).
            for path in &paths[1..4] {
                executor
                    .execute_view(&runtime_ready_request(path.clone()))
                    .expect("surviving sibling execute after evicted pack returns");
            }
            assert_eq!(
                cohere_runtime_build_counts_for_test(),
                (6, 6),
                "packs 2-4 must remain cache hits after the previously-evicted pack 1 rebuilds \
                 and takes a slot back"
            );
        });
    }

    #[test]
    fn cohere_executor_serve_batch_env_keeps_cpu_path_available() {
        // Flattened into one multi-key override rather than nesting
        // `with_forced_cpu_backend_for_test` inside `with_serve_batch_env`:
        // the process env lock is not reentrant, so two nested guards on the
        // same thread would self-deadlock on the second `lock()` call.
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_GGML_BACKEND", Some(OsString::from("cpu"))),
                (OPENASR_SERVE_BATCH_ENV, Some(OsString::from("2"))),
            ],
            || {
                let temp = tempfile::tempdir().expect("tempdir");
                let runtime_path = temp.path().join("cohere-runtime.gguf");
                let spec =
                    TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
                write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

                let executor = CohereTranscribeGgmlExecutor::default();
                let result = executor
                    .execute_view(&runtime_ready_request(runtime_path))
                    .expect("CPU path should remain available when serve batch is enabled");
                assert!(
                    result.transcription.text.is_ascii() || !result.transcription.text.is_empty()
                );
            },
        );
    }

    #[test]
    fn cohere_longform_prompt_budget_truncates_history_tail() {
        let request_options = GgmlAsrExecutionOptions {
            prompt_token_ids: Some((0_u32..200_u32).collect()),
            longform: Some(LongFormOptions::default()),
            ..GgmlAsrExecutionOptions::default()
        };
        let metadata = super::super::runtime_contract::CohereTranscribeExecutionMetadata {
            vocab_size: 1024,
            encoder_layers: 2,
            encoder_d_model: 16,
            encoder_heads: 2,
            encoder_head_dim: 8,
            encoder_ffn_dim: 32,
            encoder_conv_kernel: 5,
            decoder_layers: 2,
            decoder_d_model: 16,
            decoder_heads: 2,
            decoder_head_dim: 8,
            decoder_ffn_dim: 32,
            decoder_max_context: 80,
            decoder_start_token_id: 13764,
            sample_rate_hz: 16_000,
            n_mels: 8,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
        };
        let initial =
            build_cohere_initial_prompt_token_ids(vec![100, 101, 102], &request_options, metadata)
                .expect("initial prompt should build");

        assert_eq!(initial[..3], [100, 101, 102]);
        assert_eq!(initial.len(), 67);
        assert_eq!(initial[3], 136);
        assert_eq!(initial.last().copied(), Some(199));

        let disabled = GgmlAsrExecutionOptions {
            longform: Some(LongFormOptions {
                mode: crate::LongFormMode::Off,
                ..LongFormOptions::default()
            }),
            ..request_options
        };
        let without_longform_tail =
            build_cohere_initial_prompt_token_ids(vec![100, 101, 102], &disabled, metadata)
                .expect("disabled long-form prompt should build");
        assert_eq!(without_longform_tail.len(), 79);
        assert_eq!(without_longform_tail[3], 124);
        assert_eq!(without_longform_tail.last().copied(), Some(199));
    }

    #[test]
    fn cohere_longform_carry_prompt_keeps_recent_tail() {
        let request_options = GgmlAsrExecutionOptions {
            prompt_token_ids: Some((10_u32..50_u32).collect()),
            longform: Some(LongFormOptions::default()),
            ..GgmlAsrExecutionOptions::default()
        };
        let carry = build_cohere_carry_prompt_token_ids(
            &request_options,
            &(50_u32..110_u32).collect::<Vec<_>>(),
        )
        .expect("carry tokens");

        assert_eq!(carry.len(), COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT);
        assert_eq!(carry.first().copied(), Some(46));
        assert_eq!(carry.last().copied(), Some(109));
    }

    #[test]
    fn cohere_carry_producer_honors_the_effective_carry_switch() {
        let request_options = GgmlAsrExecutionOptions {
            longform: Some(LongFormOptions {
                carry_prompt_across_slices: false,
                ..LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert_eq!(
            build_cohere_carry_prompt_token_ids(&request_options, &[1, 2, 3]),
            None
        );
    }

    #[test]
    fn cohere_executor_returns_longform_carry_context_when_requested() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let executor = CohereTranscribeGgmlExecutor::default();
            let mut request = runtime_ready_request(runtime_path);
            request.request_options.longform = Some(LongFormOptions::default());
            let result = executor
                .execute_view(&request)
                .expect("executor should produce a best-effort transcription");
            let carry = result
                .carry_context
                .and_then(|context| context.prompt_token_ids)
                .expect("longform carry tokens");
            assert!(!carry.is_empty());
            assert!(carry.len() <= COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT);
        });
    }

    #[test]
    fn cohere_executor_reuses_prepared_runtime_for_cached_preflight() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");
        let executor = CohereTranscribeGgmlExecutor::default();

        let request = runtime_ready_request(runtime_path.clone());
        let preflight = request.runtime_source_preflight();
        let runtime_a = executor
            .runtime_cache_by_path
            .prepared_runtime_for_preflight(
                PreparedRuntimeLookup {
                    model_architecture: request.selected_family.model_architecture,
                    preflight,
                    backend: request.resolved_runtime.backend(),
                },
                map_prepared_runtime_registry_error,
                cohere_runtime_cache_slot_unavailable,
            )
            .expect("prepared runtime should build");
        let runtime_b = executor
            .runtime_cache_by_path
            .prepared_runtime_for_preflight(
                PreparedRuntimeLookup {
                    model_architecture: request.selected_family.model_architecture,
                    preflight,
                    backend: request.resolved_runtime.backend(),
                },
                map_prepared_runtime_registry_error,
                cohere_runtime_cache_slot_unavailable,
            )
            .expect("prepared runtime should reuse cache");
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
    }

    /// Core-layer regression coverage for the shared-executor change: one
    /// service-owned executor scope is registered into both its offline and
    /// streaming dispatch tables. This test drives one scoped executor through
    /// its two real entry points (`execute`, the offline
    /// path, and `execute_streaming`, exactly what
    /// `GgmlAsrStreamingExecutor::start_streaming_session` calls per chunk)
    /// and asserts they resolve to the *same* `Arc<CoherePreparedRuntime>`,
    /// not just that both happen to succeed. A regression that reintroduced
    /// a second executor instance (or a second `runtime_cache_by_path`) would
    /// still let both calls decode successfully -- the host-materialized
    /// weights would just be duplicated in memory -- so pointer identity is
    /// the assertion that actually catches it.
    #[test]
    fn cohere_offline_and_streaming_entries_share_the_same_prepared_runtime() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let shared = CohereTranscribeGgmlExecutor::default();
            let request = runtime_ready_request(runtime_path);
            let preflight = request.runtime_source_preflight();

            // Cold fetch through the exact call `execute_inner` makes
            // internally for the offline path (`with_cohere_transcribe_runtime_for_preflight`
            // -> `prepared_runtime_for_preflight`), captured directly so its
            // Arc identity is observable (`execute`'s return type does not
            // expose it).
            let runtime_via_offline_style_lookup = shared
                .runtime_cache_by_path
                .prepared_runtime_for_preflight(
                    PreparedRuntimeLookup {
                        model_architecture: request.selected_family.model_architecture,
                        preflight,
                        backend: request.resolved_runtime.backend(),
                    },
                    map_prepared_runtime_registry_error,
                    cohere_runtime_cache_slot_unavailable,
                )
                .expect("offline-style lookup should build the prepared runtime");

            // Real streaming entry, real decode, through the same shared
            // singleton -- not a stand-in.
            shared
                .execute_streaming(&request)
                .expect("streaming entry should decode using the shared singleton's cache");

            // Now a pure cache hit if (and only if) `execute_streaming` above
            // reused `runtime_cache_by_path` rather than materializing its
            // own separate prepared runtime.
            let runtime_after_streaming_entry = shared
                .runtime_cache_by_path
                .prepared_runtime_for_preflight(
                    PreparedRuntimeLookup {
                        model_architecture: request.selected_family.model_architecture,
                        preflight,
                        backend: request.resolved_runtime.backend(),
                    },
                    map_prepared_runtime_registry_error,
                    cohere_runtime_cache_slot_unavailable,
                )
                .expect("post-streaming lookup should still be cached");

            assert!(
                Arc::ptr_eq(
                    &runtime_via_offline_style_lookup,
                    &runtime_after_streaming_entry
                ),
                "offline and streaming entry points on the shared executor singleton must \
                 resolve the same prepared-runtime Arc, proving the cache is actually shared"
            );
        });
    }

    #[test]
    fn decoder_cpu_preference_is_off_by_default_and_on_when_set() {
        let options = GgmlAsrExecutionOptions::default();
        assert!(!options.prefer_cpu_decoder_for_multichunk_metal);

        let mut options_with_preference = options;
        options_with_preference.prefer_cpu_decoder_for_multichunk_metal = true;
        assert!(options_with_preference.prefer_cpu_decoder_for_multichunk_metal);
    }

    #[test]
    fn cohere_executor_rejects_non_cohere_adapter() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");
        let executor = CohereTranscribeGgmlExecutor::default();
        let mut request = runtime_ready_request(runtime_path);
        request.selected_family =
            builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
        let error = executor
            .execute_view(&request)
            .expect_err("adapter mismatch must fail closed")
            .to_string();
        assert!(error.contains(COHERE_EXECUTOR_ID), "{error}");
        assert!(error.contains("requires adapter"), "{error}");
    }

    #[test]
    fn native_backend_selects_cohere_executor_after_registration() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.oasr");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let backend = NativeBackend::new(
                crate::models::native_execution_services::test_native_execution_services(),
            );
            let request =
                TranscriptionRequest::new(sample_wav_fixture_path(), "cohere-runtime-fixture")
                    .with_model_pack_path(Some(runtime_path));
            let transcription = backend
                .transcribe(request)
                .expect("cohere runtime-ready fixture should transcribe");
            assert!(transcription.text.is_ascii() || !transcription.text.is_empty());
            assert!(!transcription.segments.is_empty());
            assert!(
                transcription
                    .segments
                    .windows(2)
                    .all(|pair| pair[0].end <= pair[1].start)
            );
        });
    }

    // Pinned to the reference decode of `fixtures/jfk.wav` on the real,
    // private dev-only `cohere-transcribe-03-2026-q4_k.oasr` pack (same
    // before/after text on `origin/main` and on this cross-KV capacity
    // refactor -- verified manually; not re-asserted per CI run since the
    // pack is a non-committed dev artifact, mirroring firered-aed's
    // `GOLDEN_JFK_TEXT` convention).
    const COHERE_GOLDEN_JFK_TEXT: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    fn cohere_dev_pack_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/cohere-out/cohere-transcribe-03-2026-q4_k.oasr")
    }

    /// Structural regression test for the VAD/longform 0%-cache-hit bug this
    /// module fixes (mirrors firered-aed's committed
    /// `differently_sized_chunks_reuse_the_same_decoder_runtime_cache_slot`):
    /// two CPU-decoder chunks with different logical encoder frame counts but
    /// the same 60-second session envelope must land in the same resident
    /// decoder cache slot.
    #[test]
    #[ignore = "requires the private dev-only cohere-transcribe-03-2026-q4_k.oasr pack; see module docs"]
    fn differently_sized_chunks_reuse_the_same_decoder_runtime_cache_slot() {
        let pack_path = cohere_dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        with_forced_cpu_backend_for_test(|| {
            let executor = CohereTranscribeGgmlExecutor::default();
            let decoder_cache_len = || executor.decoder_runtimes.usage_for_test().0;
            assert_eq!(decoder_cache_len(), 0, "cache must start empty");

            let mut req_jfk = runtime_ready_request(pack_path.clone());
            req_jfk.backend_preference = GgmlAsrBackendPreference::CpuOnly;
            plan_request_decoder_state(&mut req_jfk, Some(60 * 16_000));
            let jfk = executor.execute_view(&req_jfk).expect("jfk transcribe");
            assert_eq!(jfk.transcription.text, COHERE_GOLDEN_JFK_TEXT);
            eprintln!("cohere cache slots after chunk 1: {}", decoder_cache_len());
            assert_eq!(decoder_cache_len(), 1, "first chunk must build one slot");

            let zh_samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav"),
                "cohere cache-reuse test",
                "cohere cache-reuse test",
            )
            .expect("load zh_sample.wav");
            let runtime_source_preflight =
                crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                    &pack_path,
                )
                .expect("cohere runtime fixture must pass preflight");
            let mut req_zh = GgmlAsrExecutionViewRequest {
                execution_services:
                    crate::models::native_execution_services::test_native_execution_services(),
                decoder_state:
                    crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
                verified_pack:
                    crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                        runtime_source_preflight,
                        crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    ),
                selected_family: builtin_adapter_descriptor(
                    crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                ),
                prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(zh_samples),
                request_options: Default::default(),
                backend_preference: GgmlAsrBackendPreference::CpuOnly,
                resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                    (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                    crate::ggml_runtime::AutoGpuPolicy::AllBackends,
                ),
                execution_context: std::sync::Arc::new(
                    crate::RequestExecutionContext::uncancellable("test fixture"),
                ),
            };
            plan_request_decoder_state(&mut req_zh, Some(60 * 16_000));
            // Content/language mismatch (zh_sample.wav vs cohere's English-first
            // prompt) is irrelevant here -- only decode-succeeds + cache-slot
            // count matter for this structural check.
            executor.execute_view(&req_zh).expect("zh transcribe");
            eprintln!(
                "cohere cache slots after chunk 2 (different frame count): {}",
                decoder_cache_len()
            );
            assert_eq!(
                decoder_cache_len(),
                1,
                "a differently-sized second chunk must reuse the SAME decoder cache slot, not mint a second one"
            );
        });
    }

    /// A 45-second single invocation remains legal when its declared session
    /// envelope is 60 seconds; it must fit the preplanned resident span with
    /// no allocation growth and preserve the recorded transcript.
    #[test]
    #[ignore = "requires the private dev-only cohere-transcribe-03-2026-q4_k.oasr pack; see module docs"]
    fn mode_off_single_window_fits_declared_envelope_and_matches_baseline() {
        let pack_path = cohere_dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let wav_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/mode-off-regression/longform_en_zh_45s.wav");
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return;
        }
        // Recorded on `origin/main` (pre-capacity-refactor, exact-per-call
        // cross-KV allocation) with the same pack/clip via a temporary
        // baseline test, not committed.
        const BASELINE_TEXT: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country. And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        with_forced_cpu_backend_for_test(|| {
            let executor = CohereTranscribeGgmlExecutor::default();
            let mut request = runtime_ready_request(pack_path);
            request.prepared_audio = GgmlAsrPreparedAudioView::mono_16khz(
                crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                    wav_path,
                    "cohere mode=off envelope test",
                    "cohere mode=off envelope test",
                )
                .expect("load wav fixture"),
            );
            request.backend_preference = GgmlAsrBackendPreference::CpuOnly;
            plan_request_decoder_state(&mut request, Some(60 * 16_000));
            let result = executor
                .execute_view(&request)
                .expect("mode=off single-window transcribe must succeed, not fail closed");
            assert_eq!(result.transcription.text, BASELINE_TEXT);
        });
    }
}
