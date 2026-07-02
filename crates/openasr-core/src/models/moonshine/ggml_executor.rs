use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use super::batched_decode::{
    MoonshineServeBatchConfig, MoonshineServeBatchConfigFromEnv, MoonshineServeBatchJob,
    moonshine_serve_batch_decode_config, submit_moonshine_serve_batch_job,
};
use super::decoder_graph::{MoonshineDecoderGraphError, run_moonshine_decoder_short_form};
use super::encoder_graph::{
    MoonshineEncoderError, MoonshineEncoderGraphRuntime, MoonshineEncoderOutput,
};
use super::frontend::{MoonshineFrontendError, moonshine_waveform_from_prepared_audio};
use super::graph_config::{moonshine_decoder_graph_config, moonshine_encoder_graph_config};
use super::lora::{
    MoonshineLoraAdapter, MoonshineLoraError, moonshine_adapter_cache_fingerprint,
    resolve_moonshine_lora_adapter,
};
use super::prepared_runtime::{
    MoonshinePreparedRuntime, MoonshinePreparedRuntimeError, build_moonshine_prepared_runtime,
};
use crate::MOONSHINE_GGML_ADAPTER_ID;
use crate::NativeAsrSession;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrPreparedAudio, GgmlAsrRuntimeSourcePreflight, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::prepared_runtime_cache::PreparedRuntimeCache;
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, with_thread_local_cached_mut_by_key,
};

const MOONSHINE_EXECUTOR_ID: &str = "moonshine-ggml-executor-v1";
const MOONSHINE_STREAMING_EXECUTOR_ID: &str = "moonshine-ggml-snapshot-streaming-executor-v1";

thread_local! {
    static MOONSHINE_ENCODER_RUNTIME_BY_KEY: RefCell<HashMap<MoonshineEncoderRuntimeCacheKey, MoonshineEncoderGraphRuntime>> =
        RefCell::new(HashMap::new());
}

/// (canonical pack path, backend, adapter fingerprint). The adapter
/// fingerprint MUST stay in this key — prepared encoder graphs embed the
/// adapter tensors, so reuse keyed only on the base pack would be a
/// correctness bug.
type MoonshineEncoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend, String);

#[derive(Debug, Error)]
enum MoonshineGgmlExecutorError {
    #[error("moonshine ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("moonshine ggml executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
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
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct MoonshineGgmlExecutor {
    runtime_cache_by_path: PreparedRuntimeCache<MoonshinePreparedRuntime>,
}

impl MoonshineGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, MoonshineGgmlExecutorError> {
        if request.selected_family.adapter_id != MOONSHINE_GGML_ADAPTER_ID {
            return Err(MoonshineGgmlExecutorError::AdapterMismatch {
                expected: MOONSHINE_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| MoonshineGgmlExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;
        // OADP Phase 0: resolve the active adapter (request-level path, env
        // fallback — if any) against THIS base pack. Any mismatch fails the
        // whole transcription — adapters are never silently ignored.
        let adapter = resolve_moonshine_lora_adapter(
            request.request_options.adapter_path.as_deref(),
            preflight.as_ref(),
        )
        .map_err(|source| MoonshineGgmlExecutorError::AdapterRejected { source })?;
        let adapter_ref = adapter.as_deref();

        let prepared_runtime = self.prepared_runtime_for_preflight(preflight.as_ref())?;
        let features = moonshine_waveform_from_prepared_audio(
            &request.prepared_audio,
            prepared_runtime.metadata.sample_rate_hz,
        )
        .map_err(map_frontend_error)?;

        let encoder_output = encode_with_cached_runtime(
            preflight.runtime_source.path(),
            &prepared_runtime,
            &features,
            adapter_ref,
        )
        .map_err(map_encoder_error)?;

        let audio_duration = audio_duration_seconds(&request.prepared_audio);
        let serve_batch_config = MoonshineServeBatchConfig::from_env().map_err(|error| {
            MoonshineGgmlExecutorError::DecoderFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = moonshine_decoder_graph_config(false);
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
                    &prepared_runtime.tokenizer,
                    request.request_options.phrase_bias.as_ref(),
                )
                .map_err(|error| MoonshineGgmlExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?;
                submit_moonshine_serve_batch_job(
                    serve_batch_config,
                    MoonshineServeBatchJob {
                        runtime_cache_path: canonical_runtime_cache_path(
                            preflight.runtime_source.path(),
                        ),
                        backend: decoder_config.backend,
                        uses_scheduler: decoder_config.use_scheduler,
                        prepared_runtime: Arc::clone(&prepared_runtime),
                        encoder_output: encoder_output.clone(),
                        decode_config,
                        word_timestamps: request.request_options.word_timestamps,
                        audio_duration_seconds: audio_duration,
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
                run_moonshine_decoder_short_form(
                    &prepared_runtime.decoder_weights,
                    &prepared_runtime.tokenizer,
                    prepared_runtime.metadata,
                    &encoder_output,
                    request.request_options.phrase_bias.as_ref(),
                    false,
                    Some(preflight.runtime_source.path()),
                    request.request_options.word_timestamps,
                    audio_duration,
                    adapter_ref,
                )
                .map_err(map_decoder_error)?
            };

        Ok(GgmlAsrExecutionResult {
            transcription: decode.transcription,
            carry_context: None,
        })
    }

    fn prepared_runtime_for_preflight(
        &self,
        preflight: &GgmlAsrRuntimeSourcePreflight,
    ) -> Result<Arc<MoonshinePreparedRuntime>, MoonshineGgmlExecutorError> {
        self.runtime_cache_by_path.get_or_try_insert_with(
            preflight.runtime_source.path(),
            || build_moonshine_prepared_runtime(preflight).map_err(map_prepared_runtime_error),
            || MoonshineGgmlExecutorError::PreparedRuntimeFailed {
                reason: "moonshine runtime cache mutex is poisoned".to_string(),
            },
        )
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

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudio) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

fn encode_with_cached_runtime(
    runtime_path: &Path,
    prepared_runtime: &MoonshinePreparedRuntime,
    features: &super::frontend::MoonshineWaveformFeatures,
    adapter: Option<&MoonshineLoraAdapter>,
) -> Result<MoonshineEncoderOutput, MoonshineEncoderError> {
    let encoder_backend = moonshine_encoder_graph_config().backend;
    let key = (
        canonical_runtime_cache_path(runtime_path),
        encoder_backend,
        moonshine_adapter_cache_fingerprint(adapter),
    );
    with_thread_local_cached_mut_by_key(
        &MOONSHINE_ENCODER_RUNTIME_BY_KEY,
        key,
        || {
            MoonshineEncoderGraphRuntime::new(
                &prepared_runtime.encoder_weights,
                prepared_runtime.metadata,
                Some(runtime_path),
                adapter,
            )
        },
        |runtime| runtime.encode(features),
    )
}

impl GgmlAsrExecutor for MoonshineGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MOONSHINE_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: batch worker allowed.
        self.execute_inner(request, false)
            .map_err(|error| moonshine_execute_error_to_ggml(self, error, request))
    }
}

impl MoonshineGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| moonshine_execute_error_to_ggml(self, error, request))
    }
}

fn moonshine_execute_error_to_ggml(
    executor: &MoonshineGgmlExecutor,
    error: MoonshineGgmlExecutorError,
    request: &GgmlAsrExecutionRequest,
) -> GgmlAsrExecutionError {
    match error {
        MoonshineGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: GgmlAsrExecutor::executor_id(executor),
            adapter_id: request.selected_family.adapter_id,
            reason: error.to_string(),
        },
    }
}

impl GgmlAsrStreamingExecutor for MoonshineGgmlExecutor {
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

fn map_encoder_error(error: MoonshineEncoderError) -> MoonshineGgmlExecutorError {
    MoonshineGgmlExecutorError::EncoderFailed {
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
