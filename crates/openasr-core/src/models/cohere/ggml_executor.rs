use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use thiserror::Error;

#[cfg(test)]
use std::sync::Arc;

use super::batched_decode::{
    CohereServeBatchConfig, CohereServeBatchConfigFromEnv, CohereServeBatchJob,
    cohere_serve_batch_decode_config, cohere_serve_batch_text_postprocess_kind,
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
use crate::arch::{COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, OpenAsrArchitectureRegistry};
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_token_history::{
    build_longform_token_history_carry, trim_prompt_token_tail,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrCarryContext, GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult,
    GgmlAsrExecutor, GgmlAsrPreparedAudio, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::runtime_prepared_registry::{
    BuiltinPreparedRuntimeCache, BuiltinPreparedRuntimeRegistryError,
};
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, canonical_runtime_cache_path,
    with_thread_local_cached_mut_by_key,
};

const COHERE_EXECUTOR_ID: &str = "cohere-transcribe-ggml-executor-v1";
const COHERE_STREAMING_EXECUTOR_ID: &str = "cohere-transcribe-ggml-snapshot-streaming-executor-v1";
const COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT: usize = 64;
const COHERE_DEBUG_TIMINGS_ENV: &str = "OPENASR_COHERE_DEBUG_TIMINGS";
const COHERE_DEBUG_ENCODER_ENV: &str = "OPENASR_COHERE_DEBUG_ENCODER";

thread_local! {
    static COHERE_ENCODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<CohereEncoderRuntimeCacheKey, CohereTranscribeEncoderGraphRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
    static COHERE_DECODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<CohereDecoderRuntimeCacheKey, CohereDecoderGraphRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
}

type CohereEncoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);
type CohereDecoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend, usize, usize);

#[derive(Debug, Error)]
enum CohereTranscribeGgmlExecutorError {
    #[error("cohere-transcribe ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("cohere-transcribe ggml executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("cohere-transcribe ggml executor runtime preparation failed: {reason}")]
    PreparedRuntimeFailed { reason: String },
    #[error("cohere-transcribe ggml executor frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("cohere-transcribe ggml executor encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("cohere-transcribe ggml executor decoder failed: {reason}")]
    DecoderFailed { reason: String },
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

#[derive(Debug, Default, Clone)]
pub(crate) struct CohereTranscribeGgmlExecutor {
    runtime_cache_by_path: BuiltinPreparedRuntimeCache,
}

impl CohereTranscribeGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, CohereTranscribeGgmlExecutorError> {
        if request.selected_family.adapter_id != COHERE_TRANSCRIBE_GGML_ADAPTER_ID {
            return Err(CohereTranscribeGgmlExecutorError::AdapterMismatch {
                expected: COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let preflight_start = debug_timing_start();
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(
                |error| CohereTranscribeGgmlExecutorError::RuntimePreflightFailed {
                    reason: error.to_string(),
                },
            )?;
        emit_cohere_debug_timing_if_enabled("runtime_preflight", preflight_start, None);
        let prepared_runtime_start = debug_timing_start();
        self.runtime_cache_by_path
            .with_cohere_transcribe_runtime_for_preflight(
                request.selected_family.model_architecture,
                preflight.as_ref(),
                map_prepared_runtime_registry_error,
                cohere_runtime_cache_poisoned,
                || CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
                    reason: format!(
                        "prepared runtime registry returned non-cohere runtime for architecture '{}'",
                        request.selected_family.model_architecture
                    ),
                },
                |prepared_runtime| {
                    emit_cohere_debug_timing_if_enabled(
                        "prepared_runtime",
                        prepared_runtime_start,
                        None,
                    );
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
                        preflight.runtime_source.path(),
                        request,
                        prepared_runtime,
                        &features,
                        skip_serve_batch,
                    )
                },
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_prepared_runtime(
        &self,
        runtime_path: &Path,
        request: &GgmlAsrExecutionRequest,
        prepared_runtime: &CoherePreparedRuntime,
        features: &CohereTranscribeMelFeatures,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, CohereTranscribeGgmlExecutorError> {
        // Make the block-stack descriptor load-bearing (P4 S5e/S5f): fail closed
        // unless the conformer-encoder + seq2seq-decoder stacks this runtime
        // materialized agree with the cohere descriptor's declared shape / block
        // kinds / tensor-name scopes / layer counts. A drift means the data and
        // the hand-wired composers disagree — never silently build the wrong thing.
        let cohere_descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID);
        let cohere_block_stack = cohere_descriptor
            .as_ref()
            .and_then(|descriptor| descriptor.block_stack.as_ref());
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
        )?;
        let eos_token_id = prompt.eos_token_id.ok_or_else(|| {
            CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: "cohere decode prompt is missing EOS token id".to_string(),
            }
        })?;
        let encoder_start = debug_timing_start();
        let encoder_output =
            encode_with_cached_cohere_encoder_runtime(runtime_path, prepared_runtime, features)
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
        let serve_batch_config = CohereServeBatchConfig::from_env().map_err(|error| {
            CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = cohere_decoder_graph_config(prefer_cpu_decoder);
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
                eos_token_id,
                &prepared_runtime.tokenizer,
                request.request_options.phrase_bias.as_ref(),
            )
            .map_err(|error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;
            submit_cohere_serve_batch_job(
                serve_batch_config,
                CohereServeBatchJob {
                    runtime_cache_path: canonical_runtime_cache_path(runtime_path),
                    backend: decoder_config.backend,
                    uses_scheduler: decoder_config.use_scheduler,
                    decoder_weights: prepared_runtime.decoder_weights.clone(),
                    tokenizer: prepared_runtime.tokenizer.clone(),
                    metadata: prepared_runtime.metadata,
                    encoder_output: encoder_output.clone(),
                    decode_config,
                    text_postprocess_kind: cohere_serve_batch_text_postprocess_kind().map_err(
                        |error| CohereTranscribeGgmlExecutorError::DecoderFailed {
                            reason: error.to_string(),
                        },
                    )?,
                    word_timestamps: request.request_options.word_timestamps,
                    audio_duration_seconds: audio_duration,
                    prefer_cpu_backend: prefer_cpu_decoder,
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
            decode_with_cached_cohere_decoder_runtime(
                runtime_path,
                &prepared_runtime.decoder_weights,
                &prepared_runtime.tokenizer,
                prepared_runtime.metadata,
                &initial_prompt_tokens,
                eos_token_id,
                &encoder_output,
                request.request_options.phrase_bias.as_ref(),
                prefer_cpu_decoder,
                request.request_options.word_timestamps,
                audio_duration,
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
        Ok(GgmlAsrExecutionResult {
            transcription: decode.transcription,
            carry_context: carry_prompt_token_ids.map(|prompt_token_ids| GgmlAsrCarryContext {
                prompt_text: None,
                prompt_token_ids: Some(prompt_token_ids),
            }),
        })
    }
}

fn cohere_runtime_cache_poisoned() -> CohereTranscribeGgmlExecutorError {
    CohereTranscribeGgmlExecutorError::PreparedRuntimeFailed {
        reason: "cohere runtime cache mutex is poisoned".to_string(),
    }
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudio) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

fn encode_with_cached_cohere_encoder_runtime(
    runtime_path: &Path,
    prepared_runtime: &CoherePreparedRuntime,
    features: &CohereTranscribeMelFeatures,
) -> Result<super::encoder_graph::CohereTranscribeEncoderOutput, CohereTranscribeEncoderError> {
    let encoder_backend = cohere_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(runtime_path), encoder_backend);
    with_thread_local_cached_mut_by_key(
        &COHERE_ENCODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            CohereTranscribeEncoderGraphRuntime::new(
                &prepared_runtime.encoder_weights,
                prepared_runtime.metadata,
                Some(runtime_path),
            )
        },
        |runtime| runtime.encode(features),
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_with_cached_cohere_decoder_runtime(
    runtime_path: &Path,
    decoder_weights: &super::decoder_weights::CohereTranscribeDecoderWeights,
    tokenizer: &super::tokenizer::CohereTranscribeTokenizer,
    metadata: super::runtime_contract::CohereTranscribeExecutionMetadata,
    initial_prompt_tokens: &[u32],
    eos_token_id: u32,
    encoder_output: &super::encoder_graph::CohereTranscribeEncoderOutput,
    phrase_bias: Option<&crate::PhraseBiasConfig>,
    prefer_cpu_backend: bool,
    word_timestamps: bool,
    audio_duration_seconds: f32,
) -> Result<super::decoder_graph::CohereDecoderGraphDecodeOutput, CohereDecoderGraphError> {
    let decoder_backend = cohere_decoder_graph_config(prefer_cpu_backend).backend;
    let key = (
        canonical_runtime_cache_path(runtime_path),
        decoder_backend,
        encoder_output.frame_count,
        encoder_output.hidden_size,
    );
    with_thread_local_cached_mut_by_key(
        &COHERE_DECODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            CohereDecoderGraphRuntime::new(
                decoder_weights,
                metadata,
                encoder_output.frame_count,
                encoder_output.hidden_size,
                prefer_cpu_backend,
            )
        },
        |runtime| {
            run_cohere_decoder_graph_short_form_with_runtime(
                runtime,
                tokenizer,
                metadata,
                initial_prompt_tokens,
                eos_token_id,
                encoder_output,
                phrase_bias,
                word_timestamps,
                audio_duration_seconds,
            )
        },
    )
}

impl GgmlAsrExecutor for CohereTranscribeGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        COHERE_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: no token observer, batch worker allowed.
        self.execute_inner(request, false)
            .map_err(|error| cohere_execute_error_to_ggml(self, error, request))
    }
}

impl CohereTranscribeGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| cohere_execute_error_to_ggml(self, error, request))
    }
}

fn cohere_execute_error_to_ggml(
    executor: &CohereTranscribeGgmlExecutor,
    error: CohereTranscribeGgmlExecutorError,
    request: &GgmlAsrExecutionRequest,
) -> GgmlAsrExecutionError {
    match error {
        CohereTranscribeGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: GgmlAsrExecutor::executor_id(executor),
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

fn build_cohere_initial_prompt_token_ids(
    base_prompt_tokens: Vec<u32>,
    request_options: &crate::GgmlAsrExecutionOptions,
    metadata: super::runtime_contract::CohereTranscribeExecutionMetadata,
) -> Result<Vec<u32>, CohereTranscribeGgmlExecutorError> {
    if base_prompt_tokens.is_empty() {
        return Err(CohereTranscribeGgmlExecutorError::DecoderFailed {
            reason: "cohere decode prompt must contain at least one token".to_string(),
        });
    }

    let mut initial_prompt_tokens = base_prompt_tokens;
    let Some(mut prompt_token_ids) = request_options.prompt_token_ids.clone() else {
        return Ok(initial_prompt_tokens);
    };
    if prompt_token_ids.is_empty() {
        return Ok(initial_prompt_tokens);
    }

    let max_prompt_tokens = metadata
        .decoder_max_context
        .saturating_sub(initial_prompt_tokens.len())
        .saturating_sub(1);
    if max_prompt_tokens == 0 {
        return Err(CohereTranscribeGgmlExecutorError::DecoderFailed {
            reason: format!(
                "cohere base prompt len {} leaves no generation budget in decoder_max_context {}",
                initial_prompt_tokens.len(),
                metadata.decoder_max_context
            ),
        });
    }
    prompt_token_ids = trim_prompt_token_tail(
        prompt_token_ids,
        max_prompt_tokens,
        request_options.longform.is_some(),
        COHERE_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    );
    initial_prompt_tokens.extend(prompt_token_ids);
    Ok(initial_prompt_tokens)
}

fn build_cohere_carry_prompt_token_ids(
    request_options: &crate::GgmlAsrExecutionOptions,
    generated_tokens: &[u32],
) -> Option<Vec<u32>> {
    build_longform_token_history_carry(
        request_options.longform.is_some(),
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
    use crate::models::serve_batch_env::{OPENASR_SERVE_BATCH_ENV, with_serve_batch_env_lock};
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrPreparedAudio, LongFormOptions,
        TranscriptionRequest, cohere_transcribe_runtime_descriptor_v1,
    };

    fn sample_wav_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist")
    }

    fn runtime_ready_request(runtime_path: PathBuf) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: runtime_path,
            runtime_source_preflight: None,
            selected_family: cohere_transcribe_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(
                crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                    sample_wav_fixture_path(),
                    "cohere test",
                    "cohere test",
                )
                .expect("sample wav should load"),
            ),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        }
    }

    fn with_serve_batch_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
        with_serve_batch_env_lock(|| {
            let previous = std::env::var_os(OPENASR_SERVE_BATCH_ENV);
            set_serve_batch_env(value.map(OsString::from));
            let result = run();
            set_serve_batch_env(previous);
            result
        })
    }

    fn set_serve_batch_env(value: Option<OsString>) {
        match value {
            Some(value) => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::set_var(OPENASR_SERVE_BATCH_ENV, value);
                }
            }
            None => {
                #[expect(unsafe_code, reason = "test-only process env override")]
                unsafe {
                    std::env::remove_var(OPENASR_SERVE_BATCH_ENV);
                }
            }
        }
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
                .execute(&runtime_ready_request(runtime_path))
                .expect("executor should produce a best-effort transcription");
            assert!(result.transcription.text.is_ascii() || !result.transcription.text.is_empty());
            assert!(result.transcription.segments.is_empty());
        });
    }

    #[test]
    fn cohere_executor_serve_batch_env_keeps_cpu_path_available() {
        with_forced_cpu_backend_for_test(|| {
            with_serve_batch_env(Some("2"), || {
                let temp = tempfile::tempdir().expect("tempdir");
                let runtime_path = temp.path().join("cohere-runtime.gguf");
                let spec =
                    TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
                write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

                let executor = CohereTranscribeGgmlExecutor::default();
                let result = executor
                    .execute(&runtime_ready_request(runtime_path))
                    .expect("CPU path should remain available when serve batch is enabled");
                assert!(
                    result.transcription.text.is_ascii() || !result.transcription.text.is_empty()
                );
            });
        });
    }

    #[test]
    fn cohere_longform_prompt_budget_truncates_history_tail() {
        let request_options = GgmlAsrExecutionOptions {
            prompt_token_ids: Some((0_u32..200_u32).collect()),
            longform: Some(LongFormOptions::default()),
            ..GgmlAsrExecutionOptions::default()
        };
        let initial = build_cohere_initial_prompt_token_ids(
            vec![100, 101, 102],
            &request_options,
            super::super::runtime_contract::CohereTranscribeExecutionMetadata {
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
            },
        )
        .expect("initial prompt should build");

        assert_eq!(initial[..3], [100, 101, 102]);
        assert_eq!(initial.len(), 67);
        assert_eq!(initial[3], 136);
        assert_eq!(initial.last().copied(), Some(199));
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
                .execute(&request)
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
        let preflight = request
            .resolve_runtime_source_preflight()
            .expect("preflight should resolve")
            .into_owned();
        let runtime_a = executor
            .runtime_cache_by_path
            .prepared_runtime_for_preflight(
                request.selected_family.model_architecture,
                &preflight,
                map_prepared_runtime_registry_error,
                cohere_runtime_cache_poisoned,
            )
            .expect("prepared runtime should build");
        let runtime_b = executor
            .runtime_cache_by_path
            .prepared_runtime_for_preflight(
                request.selected_family.model_architecture,
                &preflight,
                map_prepared_runtime_registry_error,
                cohere_runtime_cache_poisoned,
            )
            .expect("prepared runtime should reuse cache");
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
    }

    #[test]
    fn decoder_cpu_preference_is_off_by_default_and_on_when_set() {
        let request = runtime_ready_request(PathBuf::from("fixtures/cohere-runtime.gguf"));
        assert!(
            !request
                .request_options
                .prefer_cpu_decoder_for_multichunk_metal
        );

        let mut request_with_preference = request;
        request_with_preference
            .request_options
            .prefer_cpu_decoder_for_multichunk_metal = true;
        assert!(
            request_with_preference
                .request_options
                .prefer_cpu_decoder_for_multichunk_metal
        );
    }

    #[test]
    fn cohere_executor_rejects_non_cohere_adapter() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");
        let executor = CohereTranscribeGgmlExecutor::default();
        let mut request = runtime_ready_request(runtime_path);
        request.selected_family = crate::whisper_runtime_descriptor_v1();
        let error = executor
            .execute(&request)
            .expect_err("adapter mismatch must fail closed")
            .to_string();
        assert!(error.contains(COHERE_EXECUTOR_ID), "{error}");
        assert!(error.contains("requires adapter"), "{error}");
    }

    #[test]
    fn native_backend_selects_cohere_executor_after_registration() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write fixture");

            let backend = NativeBackend;
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
}
