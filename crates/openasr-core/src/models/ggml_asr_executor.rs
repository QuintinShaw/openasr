use std::{borrow::Cow, collections::BTreeMap, path::PathBuf, sync::Arc};

use thiserror::Error;

use crate::ggml_runtime::{RequestBackendPreference, install_request_backend_override};
use crate::models::ggml_family_registry::WHISPER_GGML_ADAPTER_ID;
use crate::models::runtime_preflight::{
    RuntimeSourceMetadataAndTensorIndexPreflightError,
    load_runtime_source_metadata_and_tensor_index,
};
use crate::{
    GgmlExecutionCapability, GgmlFamilyAdapterDescriptor, GgmlRuntimeSource, GgufMetadata,
    GgufTensorIndex, LongFormOptions, NativeAsrBackpressurePolicy, NativeAsrSession,
    PhraseBiasConfig, RealtimeAudioFormat, Transcription, TranscriptionTask,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlAsrBackendPreference {
    CpuOnly,
    /// Force the GPU-class backend (Metal on macOS). Conversion layers
    /// hard-error earlier when no GPU device exists, so this never silently
    /// downgrades.
    Accelerated,
    Auto,
}

impl GgmlAsrBackendPreference {
    /// The thread-local override `resolve_runtime_backend` consults; `Auto`
    /// installs nothing (env/global default decides).
    pub(crate) fn request_backend_override(self) -> Option<RequestBackendPreference> {
        match self {
            Self::CpuOnly => Some(RequestBackendPreference::CpuOnly),
            Self::Accelerated => Some(RequestBackendPreference::Accelerated),
            Self::Auto => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrPreparedAudio {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_f32: Vec<f32>,
}

impl GgmlAsrPreparedAudio {
    pub fn mono_16khz(samples_f32: Vec<f32>) -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrRuntimeSourcePreflight {
    pub runtime_source: GgmlRuntimeSource,
    /// `Arc`-wrapped so cloning this preflight (done once per long-form
    /// slice on the native transcribe hot path) is a refcount bump instead
    /// of a deep copy of the full GGUF metadata map (which typically
    /// embeds the whole tokenizer vocab).
    pub metadata: Arc<GgufMetadata>,
    pub tensor_index: Arc<GgufTensorIndex>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgmlAsrExecutionOptions {
    pub language: Option<String>,
    /// Speech task. Default `Transcribe` keeps the legacy byte-identical path;
    /// only whisper acts on `Translate` (other families reject it post-selection).
    pub task: TranscriptionTask,
    pub prompt: Option<String>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub phrase_bias: Option<PhraseBiasConfig>,
    pub inference_threads: Option<usize>,
    pub word_timestamps: bool,
    /// True when `word_timestamps` was forced on solely to obtain word anchors
    /// for VAD diarization (the caller did not request word timestamps). Only
    /// whisper acts on this: it keeps the decode path byte-identical to a
    /// non-diarized run (cross flash attention unchanged, no cross-attention
    /// collection) and derives anchors post hoc from the generated tokens
    /// instead of the higher-fidelity cross-attention alignment.
    pub word_timestamps_forced_for_diarization: bool,
    pub diarize: bool,
    pub longform: Option<LongFormOptions>,
    pub longform_chunk_count_hint: Option<usize>,
    /// Set from the architecture descriptor when the arch signals that multi-chunk
    /// longform on Metal should prefer the CPU decoder path.  Avoids per-executor
    /// re-derivation of this policy flag.
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    /// OADP Phase 0: request-level `.oadp` adapter pack path (CLI `--adapter`
    /// plumbs it here). `None` falls back to the server-side `OPENASR_ADAPTER`
    /// process environment variable.
    pub adapter_path: Option<PathBuf>,
}

impl GgmlAsrExecutionOptions {
    pub fn from_transcription_request(
        language: Option<String>,
        prompt: Option<String>,
        longform: Option<LongFormOptions>,
    ) -> Self {
        Self::from_transcription_request_with_phrase_bias(language, prompt, None, longform)
    }

    pub fn from_transcription_request_with_phrase_bias(
        language: Option<String>,
        prompt: Option<String>,
        phrase_bias: Option<PhraseBiasConfig>,
        longform: Option<LongFormOptions>,
    ) -> Self {
        Self {
            language,
            task: TranscriptionTask::default(),
            prompt,
            prompt_token_ids: None,
            phrase_bias,
            inference_threads: None,
            word_timestamps: false,
            word_timestamps_forced_for_diarization: false,
            diarize: false,
            longform,
            longform_chunk_count_hint: None,
            prefer_cpu_decoder_for_multichunk_metal: false,
            adapter_path: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgmlAsrCarryContext {
    pub prompt_text: Option<String>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrExecutionRequest {
    pub runtime_source_path: PathBuf,
    pub runtime_source_preflight: Option<GgmlAsrRuntimeSourcePreflight>,
    pub selected_family: GgmlFamilyAdapterDescriptor,
    pub prepared_audio: GgmlAsrPreparedAudio,
    pub request_options: GgmlAsrExecutionOptions,
    pub backend_preference: GgmlAsrBackendPreference,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrStreamingSessionConfig {
    pub audio_format: RealtimeAudioFormat,
    pub backpressure: NativeAsrBackpressurePolicy,
    pub partial_results: bool,
    pub word_timestamps: bool,
    pub min_partial_interval_ms: Option<u32>,
}

impl GgmlAsrStreamingSessionConfig {
    /// Effective partial-decode floor (ms): the client override if set, else the
    /// per-family default. Fed only to `PartialDecodeCadence`, which gates PARTIAL
    /// re-decodes (never the FINAL), so it cannot affect transcript parity.
    pub(crate) fn partial_floor_ms(&self, family_default: u32) -> u64 {
        u64::from(self.min_partial_interval_ms.unwrap_or(family_default))
    }
}

impl From<crate::NativeAsrStreamingSessionConfig> for GgmlAsrStreamingSessionConfig {
    fn from(config: crate::NativeAsrStreamingSessionConfig) -> Self {
        Self {
            audio_format: config.audio_format,
            backpressure: config.backpressure,
            partial_results: config.partial_results,
            word_timestamps: config.word_timestamps,
            min_partial_interval_ms: config.min_partial_interval_ms,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrStreamingSessionRequest {
    pub runtime_source_path: PathBuf,
    pub runtime_source_preflight: Option<GgmlAsrRuntimeSourcePreflight>,
    pub selected_family: GgmlFamilyAdapterDescriptor,
    pub request_options: GgmlAsrExecutionOptions,
    pub configured_diarize: bool,
    pub backend_preference: GgmlAsrBackendPreference,
    pub session_context: crate::NativeAsrSessionContext,
    pub session_config: GgmlAsrStreamingSessionConfig,
}

#[derive(Debug, Error)]
pub(crate) enum GgmlAsrExecutionRequestPreflightError {
    #[error(
        "runtime preflight path '{preflight_path}' does not match execution request path '{request_path}'"
    )]
    PathMismatch {
        preflight_path: String,
        request_path: String,
    },
    #[error("could not load runtime preflight from '{request_path}': {source}")]
    LoadFailed {
        request_path: String,
        source: Box<RuntimeSourceMetadataAndTensorIndexPreflightError>,
    },
}

impl GgmlAsrExecutionRequest {
    pub(crate) fn resolve_runtime_source_preflight(
        &self,
    ) -> Result<Cow<'_, GgmlAsrRuntimeSourcePreflight>, GgmlAsrExecutionRequestPreflightError> {
        if let Some(preflight) = self.runtime_source_preflight.as_ref() {
            if preflight.runtime_source.path() != self.runtime_source_path.as_path() {
                return Err(GgmlAsrExecutionRequestPreflightError::PathMismatch {
                    preflight_path: preflight.runtime_source.path().display().to_string(),
                    request_path: self.runtime_source_path.display().to_string(),
                });
            }
            return Ok(Cow::Borrowed(preflight));
        }
        let preflight = load_runtime_source_metadata_and_tensor_index(&self.runtime_source_path)
            .map_err(|source| GgmlAsrExecutionRequestPreflightError::LoadFailed {
                request_path: self.runtime_source_path.display().to_string(),
                source: Box::new(source),
            })?;
        Ok(Cow::Owned(preflight))
    }
}

impl GgmlAsrStreamingSessionRequest {
    pub(crate) fn resolve_runtime_source_preflight(
        &self,
    ) -> Result<Cow<'_, GgmlAsrRuntimeSourcePreflight>, GgmlAsrExecutionRequestPreflightError> {
        if let Some(preflight) = self.runtime_source_preflight.as_ref() {
            if preflight.runtime_source.path() != self.runtime_source_path.as_path() {
                return Err(GgmlAsrExecutionRequestPreflightError::PathMismatch {
                    preflight_path: preflight.runtime_source.path().display().to_string(),
                    request_path: self.runtime_source_path.display().to_string(),
                });
            }
            return Ok(Cow::Borrowed(preflight));
        }
        let preflight = load_runtime_source_metadata_and_tensor_index(&self.runtime_source_path)
            .map_err(|source| GgmlAsrExecutionRequestPreflightError::LoadFailed {
                request_path: self.runtime_source_path.display().to_string(),
                source: Box::new(source),
            })?;
        Ok(Cow::Owned(preflight))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrExecutionResult {
    pub transcription: Transcription,
    pub carry_context: Option<GgmlAsrCarryContext>,
}

impl GgmlAsrExecutionResult {
    pub fn into_transcription(self) -> Transcription {
        self.transcription
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GgmlAsrExecutionError {
    #[error(
        "ggml execution capability is unsupported for adapter '{adapter_id}': backend preference '{backend_preference}'"
    )]
    UnsupportedCapability {
        adapter_id: &'static str,
        backend_preference: &'static str,
    },
    #[error(
        "no ggml executor is registered for adapter '{adapter_id}' (family '{model_family}') and capability '{capability}'"
    )]
    ExecutorUnavailable {
        adapter_id: &'static str,
        model_family: &'static str,
        capability: &'static str,
    },
    #[error(
        "phrase bias / hotword boosting is unsupported for adapter '{adapter_id}' (family '{model_family}')"
    )]
    PhraseBiasUnsupported {
        adapter_id: &'static str,
        model_family: &'static str,
    },
    #[error("ggml executor '{executor_id}' failed for adapter '{adapter_id}': {reason}")]
    ExecutorFailed {
        executor_id: &'static str,
        adapter_id: &'static str,
        reason: String,
    },
    /// OADP Phase 0: an adapter is active (request `--adapter` or the
    /// server-side `OPENASR_ADAPTER` env var) but the selected family has no
    /// dynamic adapter support. Fail-closed: an adapter the user asked for is
    /// never silently ignored.
    #[error(
        "an adapter pack is active ('{adapter_path}') but model family '{model_family}' does not \
         support adapter packs (Phase 0: moonshine only); fail-closed"
    )]
    AdapterUnsupportedForFamily {
        model_family: &'static str,
        adapter_path: String,
    },
    /// A transient serve-batch failure (queue saturation / owner gone / GPU step
    /// hung) carried out of the executor so the backend can map it to a retryable
    /// HTTP status instead of a generic 500. `retryable == true` => queue full
    /// (429); `retryable == false` => owner disconnected / reply timed out (503).
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

impl GgmlAsrExecutionError {
    pub(crate) fn executor_failed(
        executor_id: &'static str,
        adapter_id: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self::ExecutorFailed {
            executor_id,
            adapter_id,
            reason: reason.into(),
        }
    }
}

pub trait GgmlAsrExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn supports_phrase_bias(&self) -> bool;
    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>;
}

pub trait GgmlAsrStreamingExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError>;
}

/// Partial-result granularity of a registered streaming executor. This is a
/// generic infrastructure property (how partials are produced), not a
/// per-model semantic: `FrameSync` executors append fixed low-latency chunks
/// and never revise already-emitted text; `Buffered` executors re-decode a
/// growing/windowed audio buffer and may revise prior partials. Only the
/// registration site (`build_builtin_ggml_streaming_execution_dispatch`)
/// knows which family is which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingPartialGranularity {
    FrameSync,
    Buffered,
}

#[derive(Default)]
pub struct GgmlAsrExecutionDispatch {
    executors_by_adapter_id: BTreeMap<&'static str, Arc<dyn GgmlAsrExecutor>>,
    executors_by_capability: BTreeMap<&'static str, Arc<dyn GgmlAsrExecutor>>,
    streaming_executors_by_adapter_id: BTreeMap<&'static str, Arc<dyn GgmlAsrStreamingExecutor>>,
    streaming_executors_by_capability: BTreeMap<&'static str, Arc<dyn GgmlAsrStreamingExecutor>>,
    streaming_partial_granularity_by_adapter_id:
        BTreeMap<&'static str, StreamingPartialGranularity>,
    streaming_partial_granularity_by_capability:
        BTreeMap<&'static str, StreamingPartialGranularity>,
}

impl GgmlAsrExecutionDispatch {
    pub fn with_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_adapter_id.insert(adapter_id, executor);
        self
    }

    pub fn with_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_capability
            .insert(capability_label(capability), executor);
        self
    }

    pub fn with_streaming_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrStreamingExecutor>,
    ) -> Self {
        self.streaming_executors_by_adapter_id
            .insert(adapter_id, executor);
        self
    }

    pub fn with_streaming_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrStreamingExecutor>,
    ) -> Self {
        self.streaming_executors_by_capability
            .insert(capability_label(capability), executor);
        self
    }

    /// Declares the partial-result granularity of the streaming executor
    /// registered for `adapter_id`. This is orthogonal to (and does not
    /// require) registering the executor itself here -- it only records the
    /// granularity fact so capability derivation can answer
    /// [`Self::is_frame_sync_for`] without touching model-family code.
    pub fn with_streaming_partial_granularity_for_adapter(
        mut self,
        adapter_id: &'static str,
        granularity: StreamingPartialGranularity,
    ) -> Self {
        self.streaming_partial_granularity_by_adapter_id
            .insert(adapter_id, granularity);
        self
    }

    /// Capability-keyed counterpart of
    /// [`Self::with_streaming_partial_granularity_for_adapter`], mirroring the
    /// adapter-id/capability duality used by the executor maps above.
    pub fn with_streaming_partial_granularity_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        granularity: StreamingPartialGranularity,
    ) -> Self {
        self.streaming_partial_granularity_by_capability
            .insert(capability_label(capability), granularity);
        self
    }

    pub fn with_whisper_non_streaming_cpu(mut self, executor: Arc<dyn GgmlAsrExecutor>) -> Self {
        self = self.with_executor_for_adapter(WHISPER_GGML_ADAPTER_ID, executor);
        self
    }

    pub fn with_native_graph_lowering_v1(mut self, executor: Arc<dyn GgmlAsrExecutor>) -> Self {
        self = self
            .with_executor_for_capability(GgmlExecutionCapability::NativeGraphLoweringV1, executor);
        self
    }

    pub fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        ensure_adapter_supported_for_family(
            &request.selected_family,
            request.request_options.adapter_path.as_deref(),
        )?;
        // Honor the request's execution preference for everything this thread
        // resolves below (graph configs, cache keys, serve-batch job
        // snapshots): the override is what makes execution_target truthful.
        let _backend_guard =
            install_request_backend_override(request.backend_preference.request_backend_override());

        if let Some(executor) = self
            .executors_by_adapter_id
            .get(request.selected_family.adapter_id)
        {
            return executor.execute(request);
        }

        if let Some(executor) = self.executors_by_capability.get(capability_label(
            request.selected_family.execution_capability,
        )) {
            return executor.execute(request);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    pub fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        ensure_adapter_supported_for_family(
            &request.selected_family,
            request.request_options.adapter_path.as_deref(),
        )?;
        if let Some(executor) = self
            .streaming_executors_by_adapter_id
            .get(request.selected_family.adapter_id)
        {
            return executor.start_streaming_session(request);
        }

        if let Some(executor) = self.streaming_executors_by_capability.get(capability_label(
            request.selected_family.execution_capability,
        )) {
            return executor.start_streaming_session(request);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    pub fn has_streaming_executor_for(&self, descriptor: &GgmlFamilyAdapterDescriptor) -> bool {
        self.streaming_executors_by_adapter_id
            .contains_key(descriptor.adapter_id)
            || self
                .streaming_executors_by_capability
                .contains_key(capability_label(descriptor.execution_capability))
    }

    /// True only when the streaming executor registered for `descriptor` was
    /// declared frame-sync at registration time. Unregistered granularity
    /// (including families with no streaming executor at all) reads as
    /// `false` -- fail closed to the buffered/no-partial-guarantee default
    /// rather than assume low-latency partials.
    pub fn is_frame_sync_for(&self, descriptor: &GgmlFamilyAdapterDescriptor) -> bool {
        matches!(
            self.streaming_partial_granularity_by_adapter_id
                .get(descriptor.adapter_id),
            Some(StreamingPartialGranularity::FrameSync)
        ) || matches!(
            self.streaming_partial_granularity_by_capability
                .get(capability_label(descriptor.execution_capability)),
            Some(StreamingPartialGranularity::FrameSync)
        )
    }
}

/// OADP Phase 0 fail-closed gate: when an adapter is active (request-level
/// adapter path, falling back to the server-side `OPENASR_ADAPTER` env var),
/// only the moonshine family may execute; the adapter is then validated
/// against the base pack inside the moonshine executor. Every other family
/// hard-errors instead of silently ignoring the adapter.
fn ensure_adapter_supported_for_family(
    selected_family: &GgmlFamilyAdapterDescriptor,
    request_adapter_path: Option<&std::path::Path>,
) -> Result<(), GgmlAsrExecutionError> {
    let Some(adapter_path) = crate::adapter_pack::active_adapter_path(request_adapter_path) else {
        return Ok(());
    };
    if selected_family.model_family == crate::models::moonshine::MOONSHINE_MODEL_FAMILY {
        return Ok(());
    }
    Err(GgmlAsrExecutionError::AdapterUnsupportedForFamily {
        model_family: selected_family.model_family,
        adapter_path: adapter_path.display().to_string(),
    })
}

const fn capability_label(capability: GgmlExecutionCapability) -> &'static str {
    match capability {
        GgmlExecutionCapability::DedicatedRuntimeExecutorV1 => "dedicated-runtime-executor-v1",
        GgmlExecutionCapability::NativeGraphLoweringV1 => "native-graph-lowering-v1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ggml_family_registry::QWEN3_ASR_GGML_ADAPTER_ID;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::{qwen3_asr_runtime_descriptor_v1, whisper_runtime_descriptor_v1};

    fn whisper_request(backend_preference: GgmlAsrBackendPreference) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: PathBuf::from("fixtures/whisper.gguf"),
            runtime_source_preflight: None,
            selected_family: whisper_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference,
        }
    }

    fn whisper_streaming_request(
        backend_preference: GgmlAsrBackendPreference,
    ) -> GgmlAsrStreamingSessionRequest {
        GgmlAsrStreamingSessionRequest {
            runtime_source_path: PathBuf::from("fixtures/whisper.gguf"),
            runtime_source_preflight: None,
            selected_family: whisper_runtime_descriptor_v1(),
            request_options: GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference,
            session_context: crate::NativeAsrSessionContext::new("rt_ggml_streaming"),
            session_config: crate::NativeAsrStreamingSessionConfig::new().into(),
        }
    }

    struct StubNativeSession {
        session_id: String,
    }

    impl crate::NativeAsrSession for StubNativeSession {
        fn session_id(&self) -> &str {
            &self.session_id
        }

        fn push_audio(
            &mut self,
            _frame: crate::RealtimeAudioFrame,
        ) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn poll_events(
            &mut self,
        ) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn cancel(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn dispatch_fails_closed_when_executor_is_not_registered() {
        let dispatch = GgmlAsrExecutionDispatch::default();
        let request = whisper_request(GgmlAsrBackendPreference::CpuOnly);

        let error = dispatch
            .execute(&request)
            .expect_err("missing executor must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: "ggml-family-whisper-runtime-v1",
                model_family: "whisper",
                capability: "dedicated-runtime-executor-v1"
            }
        ));
        assert!(
            error
                .to_string()
                .contains("no ggml executor is registered for adapter")
        );
    }

    #[test]
    fn dispatch_accepts_auto_backend_preference() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                })
            }
        }

        let request = whisper_request(GgmlAsrBackendPreference::Auto);
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_whisper_non_streaming_cpu(Arc::new(StubExecutor));
        let result = dispatch.execute(&request).expect("auto should dispatch");
        assert_eq!(result.transcription.text, "ok");
    }

    #[test]
    fn dispatch_allows_phrase_bias_to_reach_registered_executor() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "phrase-bias-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                assert!(request.request_options.phrase_bias.is_some());
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        text: "biased".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                })
            }
        }

        let mut request = whisper_request(GgmlAsrBackendPreference::Auto);
        request.request_options.phrase_bias = Some(
            crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)])
                .expect("phrase bias fixture must validate"),
        );
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_whisper_non_streaming_cpu(Arc::new(StubExecutor));

        let result = dispatch
            .execute(&request)
            .expect("registered executor receives phrase bias");

        assert_eq!(result.transcription.text, "biased");
    }

    #[test]
    fn dispatch_fails_closed_when_qwen_executor_is_not_registered() {
        let mut request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        request.selected_family = qwen3_asr_runtime_descriptor_v1();
        let dispatch = GgmlAsrExecutionDispatch::default();
        let error = dispatch
            .execute(&request)
            .expect_err("missing qwen executor must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: QWEN3_ASR_GGML_ADAPTER_ID,
                model_family: crate::QWEN3_ASR_MODEL_FAMILY,
                capability: "native-graph-lowering-v1"
            }
        ));
    }

    #[test]
    fn dispatch_fails_closed_when_adapter_is_active_for_non_moonshine_family() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "adapter-gate-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        text: "must never run".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                })
            }
        }

        // qwen (non-moonshine) family with a request-level adapter: the gate
        // must hard-error BEFORE any executor runs, even though one is
        // registered for the capability.
        let mut request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        request.selected_family = qwen3_asr_runtime_descriptor_v1();
        request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_native_graph_lowering_v1(Arc::new(StubExecutor));

        let error = dispatch
            .execute(&request)
            .expect_err("adapter on a non-moonshine family must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::AdapterUnsupportedForFamily {
                model_family: crate::QWEN3_ASR_MODEL_FAMILY,
                ..
            }
        ));
        assert!(error.to_string().contains("/tmp/fixture.oadp"));
        assert!(error.to_string().contains("fail-closed"));

        // The same adapter on the moonshine family passes the gate: with no
        // moonshine executor registered it must reach executor lookup and
        // fail with ExecutorUnavailable, NOT AdapterUnsupportedForFamily.
        let mut moonshine_request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        moonshine_request.selected_family = crate::moonshine_runtime_descriptor_v1();
        moonshine_request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let error = GgmlAsrExecutionDispatch::default()
            .execute(&moonshine_request)
            .expect_err("no moonshine executor registered");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable { .. }
        ));
    }

    #[test]
    fn streaming_dispatch_fails_closed_when_adapter_is_active_for_non_moonshine_family() {
        let mut request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);
        request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let dispatch = GgmlAsrExecutionDispatch::default();

        let error = match dispatch.start_streaming_session(&request) {
            Ok(_) => panic!("adapter on a non-moonshine family must fail closed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            GgmlAsrExecutionError::AdapterUnsupportedForFamily {
                model_family: "whisper",
                ..
            }
        ));
    }

    #[test]
    fn dispatch_falls_back_to_capability_executor() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "native-graph-lowering-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                })
            }
        }

        let mut request = whisper_request(GgmlAsrBackendPreference::Auto);
        request.selected_family = qwen3_asr_runtime_descriptor_v1();
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_native_graph_lowering_v1(Arc::new(StubExecutor));

        let result = dispatch
            .execute(&request)
            .expect("capability executor should dispatch");
        assert_eq!(result.transcription.text, "ok");
    }

    #[test]
    fn streaming_dispatch_fails_closed_when_executor_is_not_registered() {
        let dispatch = GgmlAsrExecutionDispatch::default();
        let request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);

        let error = match dispatch.start_streaming_session(&request) {
            Ok(_) => panic!("missing streaming executor must fail closed"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: "ggml-family-whisper-runtime-v1",
                model_family: "whisper",
                capability: "dedicated-runtime-executor-v1"
            }
        ));
    }

    #[test]
    fn streaming_dispatch_routes_registered_adapter_executor() {
        struct StubStreamingExecutor;
        impl GgmlAsrStreamingExecutor for StubStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "streaming-stub"
            }

            fn start_streaming_session(
                &self,
                request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                assert_eq!(request.selected_family.adapter_id, WHISPER_GGML_ADAPTER_ID);
                Ok(Box::new(StubNativeSession {
                    session_id: request.session_context.session_id.0.clone(),
                }))
            }
        }

        let request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);
        let dispatch = GgmlAsrExecutionDispatch::default().with_streaming_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(StubStreamingExecutor),
        );

        let session = dispatch
            .start_streaming_session(&request)
            .expect("registered streaming executor should dispatch");

        assert_eq!(session.session_id(), "rt_ggml_streaming");
    }

    #[test]
    fn streaming_dispatch_reports_executor_coverage() {
        struct StubStreamingExecutor;
        impl GgmlAsrStreamingExecutor for StubStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "streaming-coverage-stub"
            }

            fn start_streaming_session(
                &self,
                request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                Ok(Box::new(StubNativeSession {
                    session_id: request.session_context.session_id.0.clone(),
                }))
            }
        }

        let whisper = whisper_runtime_descriptor_v1();
        let qwen = qwen3_asr_runtime_descriptor_v1();
        let empty_dispatch = GgmlAsrExecutionDispatch::default();
        assert!(!empty_dispatch.has_streaming_executor_for(&whisper));
        assert!(!empty_dispatch.has_streaming_executor_for(&qwen));

        let adapter_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_executor_for_adapter(
                whisper.adapter_id,
                Arc::new(StubStreamingExecutor),
            );
        assert!(adapter_dispatch.has_streaming_executor_for(&whisper));
        assert!(!adapter_dispatch.has_streaming_executor_for(&qwen));

        let capability_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_executor_for_capability(
                qwen.execution_capability,
                Arc::new(StubStreamingExecutor),
            );
        assert!(capability_dispatch.has_streaming_executor_for(&qwen));
    }

    #[test]
    fn is_frame_sync_for_reports_registered_granularity_and_defaults_closed() {
        let whisper = whisper_runtime_descriptor_v1();
        let qwen = qwen3_asr_runtime_descriptor_v1();

        // No granularity registered at all: fails closed to "not frame-sync",
        // matching the treatment of an unregistered streaming executor.
        let empty_dispatch = GgmlAsrExecutionDispatch::default();
        assert!(!empty_dispatch.is_frame_sync_for(&whisper));
        assert!(!empty_dispatch.is_frame_sync_for(&qwen));

        let mixed_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_partial_granularity_for_adapter(
                whisper.adapter_id,
                StreamingPartialGranularity::FrameSync,
            )
            .with_streaming_partial_granularity_for_adapter(
                qwen.adapter_id,
                StreamingPartialGranularity::Buffered,
            );
        assert!(mixed_dispatch.is_frame_sync_for(&whisper));
        assert!(!mixed_dispatch.is_frame_sync_for(&qwen));

        let capability_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_partial_granularity_for_capability(
                qwen.execution_capability,
                StreamingPartialGranularity::FrameSync,
            );
        assert!(capability_dispatch.is_frame_sync_for(&qwen));
    }

    #[test]
    fn resolve_runtime_source_preflight_rejects_mismatched_request_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_a = temp.path().join("runtime-a.gguf");
        let runtime_b = temp.path().join("runtime-b.gguf");
        write_tiny_gguf_runtime_source(&runtime_a, &TinyGgufFixtureSpec::new(Default::default()))
            .expect("write tiny gguf");
        write_tiny_gguf_runtime_source(&runtime_b, &TinyGgufFixtureSpec::new(Default::default()))
            .expect("write tiny gguf");
        let preflight = load_runtime_source_metadata_and_tensor_index(&runtime_a)
            .expect("load preflight from runtime-a");

        let mut request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        request.runtime_source_path = runtime_b.clone();
        request.runtime_source_preflight = Some(preflight);

        let error = request
            .resolve_runtime_source_preflight()
            .expect_err("path mismatch must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionRequestPreflightError::PathMismatch { .. }
        ));
        assert!(
            error
                .to_string()
                .contains(runtime_a.display().to_string().as_str())
        );
        assert!(
            error
                .to_string()
                .contains(runtime_b.display().to_string().as_str())
        );
    }

    #[test]
    fn resolve_runtime_source_preflight_surfaces_missing_runtime_source_path() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let missing_path = temp.path().to_path_buf();
        drop(temp);

        let mut request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        request.runtime_source_path = missing_path.clone();
        request.runtime_source_preflight = None;

        let error = request
            .resolve_runtime_source_preflight()
            .expect_err("missing path must fail preflight resolution");
        assert!(matches!(
            error,
            GgmlAsrExecutionRequestPreflightError::LoadFailed { .. }
        ));
        assert!(
            error
                .to_string()
                .contains(missing_path.display().to_string().as_str())
        );
    }
}
