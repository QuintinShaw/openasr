use std::{
    borrow::Cow,
    collections::BTreeMap,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

use thiserror::Error;

use crate::api::backend::DecodeTruncation;
use crate::ggml_runtime::{RequestBackendPreference, install_request_backend_override};
use crate::models::ggml_family_registry::WHISPER_GGML_ADAPTER_ID;
use crate::models::runtime_preflight::{
    RuntimeSourceMetadataAndTensorIndexPreflightError,
    load_runtime_source_metadata_and_tensor_index,
};
use crate::{
    GgmlExecutionCapability, GgmlFamilyAdapterDescriptor, GgmlRuntimeSource, GgufMetadata,
    GgufTensorIndex, LongFormOptions, NativeAsrBackpressurePolicy, NativeAsrSession, PcmSlice,
    PhraseBiasConfig, RealtimeAudioFormat, RequestExecutionContext, Transcription,
    TranscriptionTask,
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
    /// The thread-local override the shared dispatch's backend resolution
    /// consults; `Auto` installs nothing (env/global default decides).
    pub(crate) fn request_backend_override(self) -> Option<RequestBackendPreference> {
        match self {
            Self::CpuOnly => Some(RequestBackendPreference::CpuOnly),
            Self::Accelerated => Some(RequestBackendPreference::Accelerated),
            Self::Auto => None,
        }
    }
}

/// Stable cache/engine identity for reusable native runtime state.
///
/// `pack_content_id` is a content proof, never a bare path, and is the
/// *entire* identity: two `RuntimeBuildIdentity` values with the same
/// content id, route, and options fingerprint are always interchangeable.
/// There is deliberately no invalidation generation/epoch here -- baking a
/// shared process-wide counter into this identity was an audited bug (one
/// idle unload / serve-batch owner shutdown / pack replace anywhere in the
/// process invalidated every resident identity, not just the one that
/// actually changed; see `runtime_cache_coordinator`'s module doc comment).
/// A pack replace already changes `pack_content_id` on its own; idle unload
/// and serve-batch owner shutdown now evict their own registries/caches
/// explicitly (see each family's `unload_idle_state` /
/// `shutdown_*_serve_batch_engines`) instead of relying on this identity
/// going stale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeBuildIdentity {
    /// Content identity: `sha256:<hex>` for real pack bytes, or an explicit
    /// verified/fake id supplied by tests / future coordinator bindings.
    pub pack_content_id: String,
    /// Resolved execution route (family + backend lane) that owns the reusable
    /// graph shape.
    pub route: String,
    /// Adapter/options fingerprint that changes the lowered graph without
    /// changing pack bytes (for example an active `.oadp` adapter path).
    pub options_fingerprint: String,
}

impl RuntimeBuildIdentity {
    pub fn new(
        pack_content_id: impl Into<String>,
        route: impl Into<String>,
        options_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            pack_content_id: pack_content_id.into(),
            route: route.into(),
            options_fingerprint: options_fingerprint.into(),
        }
    }

    /// Builds the effective identity for one offline request.
    ///
    /// Prefer an explicit verified/fake content id from the request when present.
    /// Otherwise use the caller-supplied content id (production always passes a
    /// content-derived id from [`crate::GgmlRuntimeSource::content_id`]).
    pub fn resolve_for_request(
        request_identity: Option<&RuntimeBuildIdentity>,
        route: impl Into<String>,
        options_fingerprint: impl Into<String>,
        content_id: impl Into<String>,
    ) -> Self {
        let route = route.into();
        let options_fingerprint = options_fingerprint.into();
        match request_identity {
            Some(identity) => Self {
                pack_content_id: identity.pack_content_id.clone(),
                route,
                options_fingerprint,
            },
            None => Self {
                pack_content_id: content_id.into(),
                route,
                options_fingerprint,
            },
        }
    }

    /// Formats a content id from a lowercase hex sha256 digest.
    pub fn content_id_from_sha256_hex(sha256_hex: &str) -> String {
        crate::models::runtime_cache_coordinator::content_id_from_sha256_hex(sha256_hex)
    }
}

/// Builds the effective serve-batch / runtime-cache identity for one request.
///
/// Always binds a content-derived pack id, taken from `runtime_source`'s
/// already-open handle (`GgmlRuntimeSource::content_id`) -- never re-derived
/// from a bare path, which would reopen a file this request already has open
/// and admitted. Explicit request identities override the content id only
/// when the caller already supplies a verified/fake binding.
pub(crate) fn serve_batch_build_identity_for_request(
    options: &GgmlAsrExecutionOptions,
    family: &str,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    runtime_source: &GgmlRuntimeSource,
) -> RuntimeBuildIdentity {
    let options_fingerprint = match options.adapter_path.as_ref() {
        Some(path) => format!("adapter={}", path.display()),
        None => "adapter=none".to_string(),
    };
    RuntimeBuildIdentity::resolve_for_request(
        options.runtime_build_identity.as_ref(),
        format!("{family}:{backend:?}"),
        options_fingerprint,
        runtime_source.content_id(),
    )
}

/// Supplies a verified runtime identity to cache-owning execution components.
/// The pack resolver owns the content proof; consumers only carry it through
/// keys and invalidate when the content id itself changes.
pub trait RuntimeBuildIdentitySource {
    fn runtime_build_identity(&self) -> Option<RuntimeBuildIdentity>;
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

    fn as_view(&self) -> GgmlAsrPreparedAudioView<'_> {
        GgmlAsrPreparedAudioView {
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples_f32: GgmlAsrSamplesView::Borrowed(&self.samples_f32),
        }
    }
}

/// Zero-copy audio view used only inside the native runtime.
///
/// The public [`GgmlAsrPreparedAudio`] remains the stable owned DTO. Native
/// long-form requests use the shared variant below so every slice and retry
/// references one immutable PCM allocation; an out-of-tree executor sees this
/// only through the dispatch's owned compatibility adapter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgmlAsrPreparedAudioView<'a> {
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
    pub(crate) samples_f32: GgmlAsrSamplesView<'a>,
}

impl GgmlAsrPreparedAudioView<'static> {
    pub(crate) fn mono_16khz(samples_f32: Vec<f32>) -> Self {
        Self::mono_16khz_shared(samples_f32.into())
    }

    pub(crate) fn mono_16khz_shared(samples_f32: PcmSlice) -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32: GgmlAsrSamplesView::Shared(samples_f32),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgmlAsrSamplesView<'a> {
    Borrowed(&'a [f32]),
    Shared(PcmSlice),
}

impl Deref for GgmlAsrSamplesView<'_> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(samples) => samples,
            Self::Shared(samples) => samples.as_slice(),
        }
    }
}

impl AsRef<[f32]> for GgmlAsrSamplesView<'_> {
    fn as_ref(&self) -> &[f32] {
        self
    }
}

#[cfg(test)]
impl GgmlAsrSamplesView<'_> {
    pub(crate) fn range(&self) -> std::ops::Range<usize> {
        match self {
            Self::Borrowed(samples) => 0..samples.len(),
            Self::Shared(samples) => samples.range(),
        }
    }

    pub(crate) fn backing_identity(&self) -> usize {
        match self {
            Self::Borrowed(samples) => samples.as_ptr() as usize,
            Self::Shared(samples) => samples.backing_identity(),
        }
    }
}

impl From<Vec<f32>> for GgmlAsrSamplesView<'static> {
    fn from(samples: Vec<f32>) -> Self {
        Self::Shared(samples.into())
    }
}

impl From<PcmSlice> for GgmlAsrSamplesView<'static> {
    fn from(samples: PcmSlice) -> Self {
        Self::Shared(samples)
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
    /// Whether this family's own decode should carry speaker structure. Set
    /// only for a family whose `arch::SpeakerSegmentationSource` is
    /// `InDecoder` and only when the request asked for Voice ID; the external
    /// VAD + speaker-embedder path never sets it, which is what keeps the two
    /// segmentation sources mutually exclusive.
    pub in_decoder_speakers: bool,
    pub longform: Option<LongFormOptions>,
    pub longform_chunk_count_hint: Option<usize>,
    /// Set from the architecture descriptor when the arch signals that multi-chunk
    /// longform on Metal should prefer the CPU decoder path. Avoids per-executor
    /// re-derivation of this policy flag.
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    /// Server-owned offline batching policy. The CLI and every non-server call
    /// retain `serial`; only the server derives this from its native-session
    /// admission limit.
    pub(crate) serve_batch: crate::models::serve_batch_env::ServeBatchPolicy,
    /// Verified cache identity for reusable native runtime state. Absent until
    /// the pack-content resolver supplies one; executors must not substitute a
    /// path-only identity.
    pub runtime_build_identity: Option<RuntimeBuildIdentity>,
    /// OADP Phase 0: request-level `.oadp` adapter pack path (CLI `--adapter`
    /// plumbs it here). `None` falls back to the server-side `OPENASR_ADAPTER`
    /// process environment variable.
    pub adapter_path: Option<PathBuf>,
}

impl RuntimeBuildIdentitySource for GgmlAsrExecutionOptions {
    fn runtime_build_identity(&self) -> Option<RuntimeBuildIdentity> {
        self.runtime_build_identity.clone()
    }
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
            in_decoder_speakers: false,
            longform,
            longform_chunk_count_hint: None,
            prefer_cpu_decoder_for_multichunk_metal: false,
            serve_batch: crate::models::serve_batch_env::ServeBatchPolicy::serial(),
            runtime_build_identity: None,
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
    /// The caller's raw execution-target choice. Still consulted by a few
    /// pre-existing, unrelated thread-local readers that install/read the
    /// override directly (the longform multichunk-metal probe, a family's
    /// own post-hoc RAM-fit check) -- but the family's own resolved backend
    /// is carried on `resolved_runtime` below, not derived from this field
    /// via any thread-local at decode time.
    pub backend_preference: GgmlAsrBackendPreference,
    /// This family's backend, already resolved from `backend_preference` and
    /// the family's own `AutoGpuPolicy` (see
    /// [`crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve`]) by
    /// whoever built this request. A required, explicit field -- not a
    /// thread-local an executor reads out of band -- so every graph-build
    /// call site an executor threads this value to (directly, or via a
    /// sub-request/job object copying it forward) observes the identical
    /// value the request was built with, including across an OS-thread
    /// boundary such as qwen's serve-batch worker.
    pub resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    /// Cancel/pause/resume control and request id for this decode, carried
    /// explicitly rather than through the (removed) thread-local
    /// transcription control. Required: a caller with nothing to cancel
    /// still passes `RequestExecutionContext::uncancellable(reason)`.
    pub execution_context: Arc<RequestExecutionContext>,
}

/// Runtime request used inside the built-in dispatch.
///
/// This is a deep internal seam: all non-audio request state keeps the same
/// shape as [`GgmlAsrExecutionRequest`], while audio may either borrow the
/// public owned DTO or retain a shared native PCM range. Keeping that choice
/// here prevents storage ownership from leaking into every model adapter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgmlAsrExecutionViewRequest<'a> {
    pub(crate) runtime_source_path: PathBuf,
    pub(crate) runtime_source_preflight: Option<GgmlAsrRuntimeSourcePreflight>,
    pub(crate) selected_family: GgmlFamilyAdapterDescriptor,
    pub(crate) prepared_audio: GgmlAsrPreparedAudioView<'a>,
    pub(crate) request_options: GgmlAsrExecutionOptions,
    pub(crate) backend_preference: GgmlAsrBackendPreference,
    pub(crate) resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    pub(crate) execution_context: Arc<RequestExecutionContext>,
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
    /// This family's backend, resolved once for the whole session by
    /// whoever built this request (see `GgmlAsrExecutionRequest::resolved_runtime`'s
    /// doc comment for why this is a required field, not a thread-local).
    /// The shared streaming drivers copy it into every per-frame
    /// `GgmlAsrExecutionRequest` they build for the life of the session.
    pub resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
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
    pub(crate) fn as_view(&self) -> GgmlAsrExecutionViewRequest<'_> {
        GgmlAsrExecutionViewRequest {
            runtime_source_path: self.runtime_source_path.clone(),
            runtime_source_preflight: self.runtime_source_preflight.clone(),
            selected_family: self.selected_family.clone(),
            prepared_audio: self.prepared_audio.as_view(),
            request_options: self.request_options.clone(),
            backend_preference: self.backend_preference,
            resolved_runtime: self.resolved_runtime,
            execution_context: Arc::clone(&self.execution_context),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn resolve_runtime_source_preflight(
        &self,
    ) -> Result<Cow<'_, GgmlAsrRuntimeSourcePreflight>, GgmlAsrExecutionRequestPreflightError> {
        resolve_runtime_source_preflight(
            &self.runtime_source_path,
            self.runtime_source_preflight.as_ref(),
        )
    }
}

impl GgmlAsrExecutionViewRequest<'_> {
    fn to_owned_request(&self) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: self.runtime_source_path.clone(),
            runtime_source_preflight: self.runtime_source_preflight.clone(),
            selected_family: self.selected_family.clone(),
            prepared_audio: GgmlAsrPreparedAudio {
                sample_rate_hz: self.prepared_audio.sample_rate_hz,
                channels: self.prepared_audio.channels,
                samples_f32: self.prepared_audio.samples_f32.to_vec(),
            },
            request_options: self.request_options.clone(),
            backend_preference: self.backend_preference,
            resolved_runtime: self.resolved_runtime,
            execution_context: Arc::clone(&self.execution_context),
        }
    }

    pub(crate) fn resolve_runtime_source_preflight(
        &self,
    ) -> Result<Cow<'_, GgmlAsrRuntimeSourcePreflight>, GgmlAsrExecutionRequestPreflightError> {
        resolve_runtime_source_preflight(
            &self.runtime_source_path,
            self.runtime_source_preflight.as_ref(),
        )
    }
}

fn resolve_runtime_source_preflight<'a>(
    runtime_source_path: &Path,
    runtime_source_preflight: Option<&'a GgmlAsrRuntimeSourcePreflight>,
) -> Result<Cow<'a, GgmlAsrRuntimeSourcePreflight>, GgmlAsrExecutionRequestPreflightError> {
    if let Some(preflight) = runtime_source_preflight {
        if preflight.runtime_source.path() != runtime_source_path {
            return Err(GgmlAsrExecutionRequestPreflightError::PathMismatch {
                preflight_path: preflight.runtime_source.path().display().to_string(),
                request_path: runtime_source_path.display().to_string(),
            });
        }
        return Ok(Cow::Borrowed(preflight));
    }
    let preflight =
        load_runtime_source_metadata_and_tensor_index(runtime_source_path).map_err(|source| {
            GgmlAsrExecutionRequestPreflightError::LoadFailed {
                request_path: runtime_source_path.display().to_string(),
                source: Box::new(source),
            }
        })?;
    Ok(Cow::Owned(preflight))
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
    /// Set when this decode stopped short of the audio it was given.
    ///
    /// A truncated decode is otherwise indistinguishable from a complete one --
    /// same shape, same success status -- so without this the caller cannot
    /// tell a transcript that covers its audio from one that gave up partway.
    /// Both the long-form loop and the single-pass path stamp it onto the
    /// returned [`Transcription`], and it is the signal a slice-level retry or
    /// degrade would key on. `None` means the decode ended on its own terms.
    ///
    /// Every seq2seq family derives this from the shared driver's stop reason
    /// via `Seq2SeqGreedyDecodeStopReason::into_decode_truncation`; CTC and
    /// transducer families never reach the greedy driver's guard and leave it
    /// `None`.
    pub decode_truncation: Option<DecodeTruncation>,
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
    /// Typed Exact/preferred device failure from graph backend init. Kept as a
    /// first-class variant so `dispatch_error_to_backend` can surface
    /// `BackendError::ExecutionDevice*` without string recovery.
    #[error(transparent)]
    ExecutionRoute(#[from] crate::device::execution_route::ExecutionRouteError),
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

    /// Preserve typed route failures from graph init; stringify everything else.
    /// Prefer this at family `GgmlCpuGraphError` boundaries so dispatch does not
    /// need Display recovery. Covered by unit tests; production call sites migrate
    /// family-by-family.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_ggml_cpu_graph_error(
        executor_id: &'static str,
        adapter_id: &'static str,
        error: crate::ggml_runtime::GgmlCpuGraphError,
    ) -> Self {
        match error {
            crate::ggml_runtime::GgmlCpuGraphError::ExecutionRoute(error) => {
                Self::ExecutionRoute(error)
            }
            other => Self::executor_failed(executor_id, adapter_id, other.to_string()),
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
    /// Drops this executor's process-lifetime cached prepared runtime(s)
    /// (mmap + materialized tensors + Metal/CPU graph context), if it caches
    /// one at all. Called by the daemon's idle-unload reaper (`idle_unload`
    /// preference); the default no-op covers executors whose only caching is
    /// per-thread. Per-thread caches are not reachable from the reaper's
    /// thread at all -- they are instead invalidated lazily through the
    /// unload generation (`thread_local_runtime_cache::bump_unload_generation`,
    /// bumped by `unload_idle_native_model_runtime_caches` after the dispatch
    /// sweep that calls this method), so they need no eviction here either.
    fn unload_idle_state(&self) {}
}

/// Required zero-copy contract for executors owned by the built-in runtime.
///
/// This trait deliberately stays crate-private. Public extensions continue to
/// implement the unchanged owned [`GgmlAsrExecutor`] contract; dispatch stores
/// those in a compatibility slot and materializes owned PCM only when an
/// internal shared view must cross that extension boundary. Built-ins cannot
/// enter the native registry without implementing this view contract.
pub(crate) trait GgmlAsrViewExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    #[cfg_attr(not(test), allow(dead_code))]
    fn supports_phrase_bias(&self) -> bool;
    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>;
    fn unload_idle_state(&self) {}
}

enum GgmlAsrExecutorSlot {
    OwnedCompatibility(Arc<dyn GgmlAsrExecutor>),
    SharedView(Arc<dyn GgmlAsrViewExecutor>),
}

impl GgmlAsrExecutorSlot {
    fn execute_owned(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => executor.execute(request),
            Self::SharedView(executor) => executor.execute_view(&request.as_view()),
        }
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => executor.execute(&request.to_owned_request()),
            Self::SharedView(executor) => executor.execute_view(request),
        }
    }

    fn unload_idle_state(&self) {
        match self {
            Self::OwnedCompatibility(executor) => executor.unload_idle_state(),
            Self::SharedView(executor) => executor.unload_idle_state(),
        }
    }
}

pub trait GgmlAsrStreamingExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError>;
    /// Streaming-side counterpart of [`GgmlAsrExecutor::unload_idle_state`].
    /// Families registered on both dispatches (offline + streaming) hold two
    /// independent executor instances with two independent caches, so both
    /// must be evicted for `idle_unload` to actually free the resident model.
    fn unload_idle_state(&self) {}
}

/// Partial-result granularity of a registered streaming executor. Declared on
/// the architecture integration descriptor and derived into this dispatch at
/// builtin registration time -- see [`crate::arch::StreamingPartialGranularity`].
pub use crate::arch::StreamingPartialGranularity;

#[derive(Default)]
pub struct GgmlAsrExecutionDispatch {
    executors_by_adapter_id: BTreeMap<&'static str, GgmlAsrExecutorSlot>,
    executors_by_capability: BTreeMap<&'static str, GgmlAsrExecutorSlot>,
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
        self.executors_by_adapter_id.insert(
            adapter_id,
            GgmlAsrExecutorSlot::OwnedCompatibility(executor),
        );
        self
    }

    pub(crate) fn with_view_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrViewExecutor>,
    ) -> Self {
        self.executors_by_adapter_id
            .insert(adapter_id, GgmlAsrExecutorSlot::SharedView(executor));
        self
    }

    pub fn with_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_capability.insert(
            capability_label(capability),
            GgmlAsrExecutorSlot::OwnedCompatibility(executor),
        );
        self
    }

    pub(crate) fn with_view_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrViewExecutor>,
    ) -> Self {
        self.executors_by_capability.insert(
            capability_label(capability),
            GgmlAsrExecutorSlot::SharedView(executor),
        );
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
        // Honor the request's execution preference for the few remaining
        // thread-local readers unrelated to backend resolution proper (the
        // longform multichunk-metal probe, a family's own post-hoc RAM-fit
        // check): this override is what makes execution_target truthful for
        // them. The family's own resolved backend is NOT computed here --
        // it already arrived as the required, explicit `request.resolved_runtime`
        // field, filled in by whoever built this request.
        let _backend_guard =
            install_request_backend_override(request.backend_preference.request_backend_override());

        if let Some(executor) = self
            .executors_by_adapter_id
            .get(request.selected_family.adapter_id)
        {
            return executor.execute_owned(request);
        }

        if let Some(executor) = self.executors_by_capability.get(capability_label(
            request.selected_family.execution_capability,
        )) {
            return executor.execute_owned(request);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    pub(crate) fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        ensure_adapter_supported_for_family(
            &request.selected_family,
            request.request_options.adapter_path.as_deref(),
        )?;
        let _backend_guard =
            install_request_backend_override(request.backend_preference.request_backend_override());

        if let Some(executor) = self
            .executors_by_adapter_id
            .get(request.selected_family.adapter_id)
        {
            return executor.execute_view(request);
        }

        if let Some(executor) = self.executors_by_capability.get(capability_label(
            request.selected_family.execution_capability,
        )) {
            return executor.execute_view(request);
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
        // Same reasoning as `execute` above: the family's resolved backend
        // is `request.resolved_runtime`, filled in by whoever built this
        // session request. The shared streaming drivers copy that value
        // into every per-frame `GgmlAsrExecutionRequest` they build for the
        // life of the session (see `build_streaming_driver`/
        // `build_ctc_streaming_driver`), so no re-resolution is needed here.
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
            self.streaming_partial_granularity_for(descriptor),
            Some(StreamingPartialGranularity::FrameSync)
        )
    }

    /// Returns the partial-result granularity registered for `descriptor`, if
    /// any. Builtin construction derives this from the architecture integration
    /// descriptor; unregistered families yield `None`.
    pub fn streaming_partial_granularity_for(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
    ) -> Option<StreamingPartialGranularity> {
        self.streaming_partial_granularity_by_adapter_id
            .get(descriptor.adapter_id)
            .copied()
            .or_else(|| {
                self.streaming_partial_granularity_by_capability
                    .get(capability_label(descriptor.execution_capability))
                    .copied()
            })
    }

    /// Idle-unload: evicts every registered executor's process-lifetime
    /// cached prepared runtime. Safe to call opportunistically (e.g. from a
    /// background reaper) -- executors with nothing resident, or whose
    /// caching is per-thread and self-managed, just no-op.
    pub fn unload_all(&self) {
        for executor in self.executors_by_adapter_id.values() {
            executor.unload_idle_state();
        }
        for executor in self.executors_by_capability.values() {
            executor.unload_idle_state();
        }
        for executor in self.streaming_executors_by_adapter_id.values() {
            executor.unload_idle_state();
        }
        for executor in self.streaming_executors_by_capability.values() {
            executor.unload_idle_state();
        }
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

    #[test]
    fn runtime_build_identity_separates_same_route_content_replacements() {
        let route = "whisper:metal:base";
        let options = "adapter=none";
        let first = RuntimeBuildIdentity::new("verified-content-a", route, options);
        let replacement = RuntimeBuildIdentity::new("verified-content-b", route, options);
        assert_ne!(
            first, replacement,
            "same path/route must not reuse replacement content"
        );
        assert_ne!(
            first,
            RuntimeBuildIdentity::new("verified-content-a", route, "adapter=/tmp/a.oadp"),
            "adapter/options fingerprint must rebuild the engine"
        );
        // Same content id/route/options must always compare equal -- there is
        // no generation/epoch field left to make an otherwise-identical
        // identity spuriously distinct (that was the audited bug).
        assert_eq!(
            first,
            RuntimeBuildIdentity::new("verified-content-a", route, options)
        );
    }

    #[test]
    fn runtime_build_identity_resolve_prefers_explicit_request_content_id() {
        let verified = RuntimeBuildIdentity::new("verified-content-a", "old", "old");
        let resolved = RuntimeBuildIdentity::resolve_for_request(
            Some(&verified),
            "whisper:gpu",
            "adapter=none",
            "sha256:should-not-win",
        );
        assert_eq!(resolved.pack_content_id, "verified-content-a");
        assert_eq!(resolved.route, "whisper:gpu");
        assert_eq!(resolved.options_fingerprint, "adapter=none");
    }

    #[test]
    fn production_pack_content_id_misses_same_path_byte_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.gguf");
        let write = |payload: &[u8]| {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(payload);
            std::fs::write(&path, bytes).expect("write pack");
        };
        let source_content_id = |path: &std::path::Path| -> String {
            crate::validate_ggml_runtime_source_path(path)
                .expect("validate runtime source")
                .content_id()
                .to_string()
        };

        write(b"content-a-bytes");
        let id_a = source_content_id(&path);
        write(b"content-b-bytes-different");
        let id_b = source_content_id(&path);
        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(
            id_a, id_b,
            "same path with different pack bytes must not share content id"
        );

        let options = GgmlAsrExecutionOptions::default();
        write(b"content-a-bytes");
        let identity_a = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate a"),
        );
        write(b"content-b-bytes-different");
        let identity_b = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate b"),
        );
        assert_eq!(identity_a.pack_content_id, id_a);
        assert_eq!(identity_b.pack_content_id, id_b);
        assert_ne!(identity_a.pack_content_id, identity_b.pack_content_id);
        assert_eq!(identity_a.route, identity_b.route);

        // Re-resolving (a fresh source, exactly like a new request) against
        // unchanged (post-rewrite) bytes must return an identity equal to
        // `identity_b` -- nothing left to bump spuriously.
        let identity_again = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate again"),
        );
        assert_eq!(identity_again, identity_b);
    }

    /// Structural proof that `execution_context` is required, not optional:
    /// this compiles only because the field's type is the concrete
    /// `Arc<RequestExecutionContext>`, not `Option<Arc<RequestExecutionContext>>`
    /// -- an `Option` field would fail to type-check against
    /// `require_concrete_execution_context`'s parameter. Never called; exists
    /// purely so `cargo check`/`clippy` re-verify the contract on every build.
    #[allow(dead_code)]
    fn require_concrete_execution_context(_: std::sync::Arc<crate::RequestExecutionContext>) {}

    #[allow(dead_code)]
    fn assert_ggml_asr_execution_request_requires_execution_context(
        request: GgmlAsrExecutionRequest,
    ) {
        let GgmlAsrExecutionRequest {
            execution_context, ..
        } = request;
        require_concrete_execution_context(execution_context);
    }

    fn whisper_request(backend_preference: GgmlAsrBackendPreference) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path: PathBuf::from("fixtures/whisper.gguf"),
            runtime_source_preflight: None,
            selected_family: whisper_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn successful_execution_result(text: &str) -> GgmlAsrExecutionResult {
        GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: text.to_string(),
                segments: Vec::new(),
                longform: None,
                language: None,
            },
            carry_context: None,
            decode_truncation: None,
        }
    }

    #[test]
    fn public_prepared_audio_retains_the_mutable_vec_contract() {
        let mut audio = GgmlAsrPreparedAudio {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32: vec![0.25],
        };
        audio.samples_f32.push(-0.5);
        assert_eq!(audio.samples_f32, vec![0.25, -0.5]);
    }

    #[test]
    fn shared_view_materializes_only_at_an_owned_extension_boundary() {
        struct OwnedExtension {
            observed: Arc<std::sync::Mutex<(usize, Vec<f32>)>>,
        }

        impl GgmlAsrExecutor for OwnedExtension {
            fn executor_id(&self) -> &'static str {
                "owned-extension"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                *self.observed.lock().unwrap() = (
                    request.prepared_audio.samples_f32.as_ptr() as usize,
                    request.prepared_audio.samples_f32.clone(),
                );
                Ok(successful_execution_result("owned"))
            }
        }

        let backing = crate::PcmBuffer::from_vec(vec![0.25, -0.5, 0.75]);
        let shared_pointer = backing.as_ptr() as usize;
        let owned = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let mut view = owned.as_view();
        view.prepared_audio = GgmlAsrPreparedAudioView::mono_16khz_shared(backing.full_slice());
        let observed = Arc::new(std::sync::Mutex::new((0, Vec::new())));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(OwnedExtension {
                observed: Arc::clone(&observed),
            }),
        );

        let result = dispatch
            .execute_view(&view)
            .expect("compatibility dispatch");
        assert_eq!(result.transcription.text, "owned");
        let (observed_pointer, observed_samples) = observed.lock().unwrap().clone();
        assert_eq!(observed_samples, backing.as_slice());
        assert_ne!(
            observed_pointer, shared_pointer,
            "the owned extension boundary must receive its own Vec"
        );
    }

    #[test]
    fn native_view_slot_preserves_pcm_for_shared_and_public_owned_requests() {
        struct ViewExecutor {
            observed_pointers: Arc<std::sync::Mutex<Vec<usize>>>,
        }

        impl GgmlAsrViewExecutor for ViewExecutor {
            fn executor_id(&self) -> &'static str {
                "view-executor"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn execute_view(
                &self,
                request: &GgmlAsrExecutionViewRequest<'_>,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                self.observed_pointers
                    .lock()
                    .unwrap()
                    .push(request.prepared_audio.samples_f32.as_ptr() as usize);
                Ok(successful_execution_result("view"))
            }
        }

        let observed_pointers = Arc::new(std::sync::Mutex::new(Vec::new()));
        let dispatch = GgmlAsrExecutionDispatch::default().with_view_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(ViewExecutor {
                observed_pointers: Arc::clone(&observed_pointers),
            }),
        );

        let backing = crate::PcmBuffer::from_vec(vec![0.1, 0.2, 0.3]);
        let shared_pointer = backing.as_ptr() as usize;
        let holder = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let mut shared_request = holder.as_view();
        shared_request.prepared_audio =
            GgmlAsrPreparedAudioView::mono_16khz_shared(backing.full_slice());
        dispatch
            .execute_view(&shared_request)
            .expect("shared view dispatch");

        let owned_request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let owned_pointer = owned_request.prepared_audio.samples_f32.as_ptr() as usize;
        dispatch.execute(&owned_request).expect("owned dispatch");

        assert_eq!(
            observed_pointers.lock().unwrap().as_slice(),
            &[shared_pointer, owned_pointer]
        );
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
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
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
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
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
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "biased".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
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
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "must never run".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
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
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
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
    fn unload_all_reaches_every_registered_executor_map() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // One stub per registration slot (offline adapter-id, offline
        // capability, streaming adapter-id, streaming capability), each
        // bumping its own counter from `unload_idle_state` -- proves
        // `unload_all` walks all four maps, not just the offline/adapter-id
        // one every other test in this file happens to exercise.
        struct CountingExecutor(Arc<AtomicUsize>);
        impl GgmlAsrExecutor for CountingExecutor {
            fn executor_id(&self) -> &'static str {
                "counting-offline-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                false
            }
            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                unreachable!("this test never executes a request")
            }
            fn unload_idle_state(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct CountingStreamingExecutor(Arc<AtomicUsize>);
        impl GgmlAsrStreamingExecutor for CountingStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "counting-streaming-stub"
            }
            fn start_streaming_session(
                &self,
                _request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                unreachable!("this test never starts a streaming session")
            }
            fn unload_idle_state(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let offline_adapter_calls = Arc::new(AtomicUsize::new(0));
        let offline_capability_calls = Arc::new(AtomicUsize::new(0));
        let streaming_adapter_calls = Arc::new(AtomicUsize::new(0));
        let streaming_capability_calls = Arc::new(AtomicUsize::new(0));

        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(
                WHISPER_GGML_ADAPTER_ID,
                Arc::new(CountingExecutor(Arc::clone(&offline_adapter_calls))),
            )
            .with_executor_for_capability(
                GgmlExecutionCapability::NativeGraphLoweringV1,
                Arc::new(CountingExecutor(Arc::clone(&offline_capability_calls))),
            )
            .with_streaming_executor_for_adapter(
                WHISPER_GGML_ADAPTER_ID,
                Arc::new(CountingStreamingExecutor(Arc::clone(
                    &streaming_adapter_calls,
                ))),
            )
            .with_streaming_executor_for_capability(
                GgmlExecutionCapability::NativeGraphLoweringV1,
                Arc::new(CountingStreamingExecutor(Arc::clone(
                    &streaming_capability_calls,
                ))),
            );

        dispatch.unload_all();

        assert_eq!(offline_adapter_calls.load(Ordering::SeqCst), 1);
        assert_eq!(offline_capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(streaming_adapter_calls.load(Ordering::SeqCst), 1);
        assert_eq!(streaming_capability_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unload_idle_state_default_is_a_no_op() {
        // Any executor that does not override `unload_idle_state` (every
        // family whose only caching is per-thread/bounded) must tolerate
        // being told to unload -- the default no-op must not panic.
        struct NoCacheExecutor;
        impl GgmlAsrExecutor for NoCacheExecutor {
            fn executor_id(&self) -> &'static str {
                "no-cache-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                false
            }
            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                unreachable!("this test never executes a request")
            }
        }

        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(WHISPER_GGML_ADAPTER_ID, Arc::new(NoCacheExecutor));
        dispatch.unload_all();
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

    #[test]
    fn from_ggml_cpu_graph_error_preserves_execution_route() {
        use crate::device::execution_route::ExecutionRouteError;
        use crate::ggml_runtime::GgmlCpuGraphError;

        let route_error = ExecutionRouteError::init_failed("provider=cuda stable_id=CUDA0");
        let mapped = GgmlAsrExecutionError::from_ggml_cpu_graph_error(
            "test-executor",
            "test-adapter",
            GgmlCpuGraphError::ExecutionRoute(route_error.clone()),
        );
        assert_eq!(mapped, GgmlAsrExecutionError::ExecutionRoute(route_error));

        let other = GgmlAsrExecutionError::from_ggml_cpu_graph_error(
            "test-executor",
            "test-adapter",
            GgmlCpuGraphError::CpuBackendUnavailable,
        );
        assert!(matches!(
            other,
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id: "test-executor",
                adapter_id: "test-adapter",
                ..
            }
        ));
    }

    /// A single `execute()` call must resolve this family's backend exactly
    /// once and hand every graph-build call site the SAME value -- not let
    /// some sites read a gated resolution and others an ungated one. The
    /// observable seam is the request's own `resolved_runtime` field (not a
    /// global/thread-local getter): a fake executor reads
    /// `_request.resolved_runtime.backend()` at multiple simulated call
    /// sites (mirroring how a real family reads it once per cache key /
    /// graph config) and records every read.
    ///
    /// The request is built on one OS thread and executed on a second,
    /// distinct OS thread -- the case a thread-local channel gets wrong
    /// (the value would either fail to cross or silently read the executing
    /// thread's own unrelated installation) but an explicit struct field
    /// gets right by construction, since it rides along with the value.
    #[test]
    fn dispatch_resolves_family_backend_once_and_consistently_across_call_sites() {
        use crate::ggml_runtime::GgmlCpuGraphBackend;
        use std::sync::Mutex;

        struct RecordingExecutor {
            observed: Arc<Mutex<Vec<GgmlCpuGraphBackend>>>,
        }
        impl GgmlAsrExecutor for RecordingExecutor {
            fn executor_id(&self) -> &'static str {
                "resolved-backend-consistency-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                // Three independent reads, standing in for three real
                // call sites within one family decode (e.g. an audio-encoder
                // cache key, a decoder cache key, and a graph-config
                // builder) -- all inside the SAME `execute()` call, all
                // reading the same explicit field on `request`.
                let mut observed = self.observed.lock().unwrap();
                for _ in 0..3 {
                    observed.push(request.resolved_runtime.backend());
                }
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        // Whisper's policy is `AllBackends` (a no-op gate), so the resolved
        // value must equal the independent generic resolution exactly --
        // host-independent equality, not a fixed backend.
        let expected = crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend;

        // Built on the submitting thread; `resolved_runtime` is materialized
        // into the request right here, before it ever crosses a thread
        // boundary.
        let request = whisper_request(GgmlAsrBackendPreference::Auto);
        let resolved_on_submitting_thread = request.resolved_runtime.backend();

        let observed = Arc::new(Mutex::new(Vec::new()));
        let dispatch = GgmlAsrExecutionDispatch::default().with_whisper_non_streaming_cpu(
            Arc::new(RecordingExecutor {
                observed: Arc::clone(&observed),
            }),
        );

        // Hand the already-resolved request to a second, distinct OS
        // thread and execute it there. If the resolved value depended on
        // any per-thread state instead of riding along on `request`, this
        // would be the boundary where it would go stale or diverge.
        std::thread::spawn(move || {
            dispatch
                .execute(&request)
                .expect("recording executor always succeeds");
        })
        .join()
        .expect("execution thread must not panic");

        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 3, "all three call sites must have run");
        assert!(
            observed.iter().all(|backend| *backend == observed[0]),
            "every call site within one request must observe the identical resolved backend, got {observed:?}"
        );
        assert_eq!(
            observed[0], resolved_on_submitting_thread,
            "the backend observed on the execution thread must be identical to the value \
             resolved on the submitting thread -- it must ride the request across the \
             thread boundary, not be re-derived from execution-thread-local state"
        );
        assert_eq!(
            observed[0], expected,
            "resolved backend must match the family's (AllBackends) generic resolution"
        );
    }

    /// A family whose descriptor declares a gated `AutoGpuPolicy`
    /// (xasr-zipformer's real `ExceptMetal`) must never observe a backend
    /// the gate forbids, even though the shared
    /// dispatch is the one doing the resolving now, not the family itself.
    /// Uses a fake executor substituted for the real xasr-zipformer one so
    /// the assertion is purely about dispatch's resolution, independent of
    /// xasr-zipformer's own graph-building code.
    #[test]
    fn dispatch_honors_gated_family_auto_policy_for_registered_architecture() {
        use crate::ggml_runtime::GgmlCpuGraphBackend;
        use std::sync::Mutex;

        struct RecordingExecutor {
            observed: Arc<Mutex<Option<GgmlCpuGraphBackend>>>,
        }
        impl GgmlAsrExecutor for RecordingExecutor {
            fn executor_id(&self) -> &'static str {
                "gated-policy-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                *self.observed.lock().unwrap() = Some(request.resolved_runtime.backend());
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let descriptor = crate::xasr_zipformer_runtime_descriptor_v1();
        let auto_gpu_policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
            descriptor.model_architecture,
        );
        assert_eq!(
            auto_gpu_policy,
            crate::ggml_runtime::AutoGpuPolicy::ExceptMetal,
            "this regression only pins something if xasr-zipformer stays ExceptMetal"
        );

        let generic_auto = crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend;
        let observed = Arc::new(Mutex::new(None));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            descriptor.adapter_id,
            Arc::new(RecordingExecutor {
                observed: Arc::clone(&observed),
            }),
        );
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: PathBuf::from("fixtures/xasr-zipformer.gguf"),
            runtime_source_preflight: None,
            selected_family: descriptor,
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::Auto,
            // The dispatch resolves against this family's OWN declared gate
            // (`ExceptMetal`, asserted above), not the generic `AllBackends`
            // policy -- an ungated resolution here would defeat the whole
            // point of this regression (it would let Auto pick Metal, which
            // `assert_ne!` below exists to catch).
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                GgmlAsrBackendPreference::Auto.request_backend_override(),
                auto_gpu_policy,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        dispatch
            .execute(&request)
            .expect("recording executor always succeeds");

        let observed = observed.lock().unwrap().expect("executor must have run");
        // The gate never lets Auto pick Metal specifically for this family --
        // this is the exact defect-A shape: an ungated read here would have
        // reported whatever the generic resolver picked, including Metal.
        assert_ne!(observed, GgmlCpuGraphBackend::Metal);
        if generic_auto == GgmlCpuGraphBackend::Metal {
            assert_eq!(observed, GgmlCpuGraphBackend::Cpu);
        } else {
            assert_eq!(observed, generic_auto);
        }
    }
}
