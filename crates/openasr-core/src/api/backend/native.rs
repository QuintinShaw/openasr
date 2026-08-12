use std::{
    collections::HashMap,
    hash::Hash,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, atomic::AtomicBool},
};

use super::{
    BackendError, BackendFeatureCapability, BackendKind, Transcription, TranscriptionBackend,
    TranscriptionBackendCapabilities, TranscriptionRequest,
};
use crate::api::native::{
    NativeAsrCapabilities, NativeAsrError, NativeAsrExecutor, NativeAsrHardwareTarget,
    NativeAsrModelAdapter, NativeAsrModelPackRef, NativeAsrOfflineRequest, NativeAsrRequestOptions,
    NativeAsrRuntimeReadiness, NativeAsrSession, NativeAsrSessionContext,
    NativeAsrStreamingSessionConfig, NativeAsrTensorLayoutRef,
};
use crate::arch::{
    GRANITE_SPEECH_GGML_ADAPTER_ID, GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
    GRANITE_SPEECH_MODEL_FAMILY, OpenAsrArchitectureRegistry, OpenAsrPhraseBiasStrategy,
    StreamingPartialGranularity,
};
use crate::device::{
    execution_policy::{
        AcceleratedDeviceConstraint, ExecutionCandidate, ExecutionIntent, ExecutionPlacement,
        ExecutionPlan, ExecutionPolicyError,
    },
    execution_route::{
        ExecutionHardwareVendor, ExecutionProvider, enumerate_compute_devices_from_ggml,
    },
};
use crate::ggml_runtime::RequestBackendPreference;
use crate::models::ggml_family_adapter::GgmlFamilyAdapterDescriptor;
use crate::models::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::realtime::RealtimeBackendCapabilities;
use crate::{
    ExecutionTarget, GgmlAsrBackendPreference, GgmlAsrExecutionError, GgmlAsrExecutionOptions,
    GgmlAsrStreamingSessionRequest, NativeExecutionServices,
};

#[path = "cue_segmentation.rs"]
mod cue_segmentation;
#[path = "native_model_id.rs"]
mod native_model_id;
#[path = "native_path.rs"]
mod native_path;
#[path = "native_transcribe.rs"]
mod native_transcribe;
#[path = "request_execution_context.rs"]
mod request_execution_context;
#[path = "transcription_control.rs"]
mod transcription_control;
#[path = "transcription_progress.rs"]
mod transcription_progress;
pub use native_model_id::{
    NativeRuntimeModelIdSource, NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError,
};
pub use native_transcribe::{
    describe_native_runtime_model_mismatch, native_runtime_model_refs_match,
    refine_existing_transcription_timeline,
};
pub use request_execution_context::RequestExecutionContext;
pub(crate) use request_execution_context::WorkProgressObserver;
pub use transcription_control::{
    GgmlAbortCallbackGuard, SliceBoundaryControl, TranscriptionControl,
};
pub use transcription_progress::{
    LegacyNativeTranscriptionProgress, NativeTranscriptionPhase, NativeTranscriptionProgress,
    ProgressBackendClass, ProgressPlan, ProgressPlanInput, ProgressReporter, ProgressSegmenterKind,
    TranscriptionStage, duration_weighted_fraction, native_transcription_progress,
    native_transcription_progress_for_id,
};

#[derive(Debug, Clone)]
pub struct NativeBackend {
    execution_services: Arc<NativeExecutionServices>,
}

#[derive(Debug, Clone)]
pub struct NativeBackendExecutor {
    execution_services: Arc<NativeExecutionServices>,
}

impl NativeBackend {
    pub fn new(execution_services: Arc<NativeExecutionServices>) -> Self {
        Self { execution_services }
    }

    pub fn execution_services(&self) -> &Arc<NativeExecutionServices> {
        &self.execution_services
    }
}

impl NativeBackendExecutor {
    pub fn new(execution_services: Arc<NativeExecutionServices>) -> Self {
        Self { execution_services }
    }

    pub fn execution_services(&self) -> &Arc<NativeExecutionServices> {
        &self.execution_services
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeRuntimeModelProjection {
    descriptor: GgmlFamilyAdapterDescriptor,
    capabilities: NativeAsrCapabilities,
    language_mode: crate::models::language::LanguageMode,
}

#[derive(Clone)]
enum NativeRuntimeModelProof {
    Verified(Arc<crate::models::pack_verifier::VerifiedPack>),
    #[cfg(test)]
    UnverifiedFixture,
}

impl std::fmt::Debug for NativeRuntimeModelProof {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verified(pack) => formatter
                .debug_struct("Verified")
                .field("route", pack.route())
                .field("path", &pack.preflight().runtime_source().path())
                .finish(),
            #[cfg(test)]
            Self::UnverifiedFixture => formatter.write_str("UnverifiedFixture"),
        }
    }
}

/// Runtime-family projection plus the verification proof for the exact open
/// pack generation from which the projection was derived.
///
/// Equality intentionally describes adapter semantics (family, capabilities,
/// language mode), as it did before proofs were attached. It never hashes or
/// compares multi-gigabyte pack contents as a side effect.
#[derive(Debug, Clone)]
pub struct NativeRuntimeModelAdapter {
    descriptor: GgmlFamilyAdapterDescriptor,
    capabilities: NativeAsrCapabilities,
    language_mode: crate::models::language::LanguageMode,
    proof: NativeRuntimeModelProof,
}

impl PartialEq for NativeRuntimeModelAdapter {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
            && self.capabilities == other.capabilities
            && self.language_mode == other.language_mode
    }
}

impl Eq for NativeRuntimeModelAdapter {}

impl NativeRuntimeModelAdapter {
    #[cfg(test)]
    fn new(
        descriptor: GgmlFamilyAdapterDescriptor,
        metadata: &crate::GgufMetadata,
        tensor_index: Option<&crate::GgufTensorIndex>,
    ) -> Self {
        let projection = NativeRuntimeModelProjection::new(descriptor, metadata, tensor_index);
        Self {
            descriptor: projection.descriptor,
            capabilities: projection.capabilities,
            language_mode: projection.language_mode,
            proof: NativeRuntimeModelProof::UnverifiedFixture,
        }
    }

    fn from_verified(
        projection: NativeRuntimeModelProjection,
        verified_pack: crate::models::pack_verifier::VerifiedPack,
    ) -> Self {
        Self {
            descriptor: projection.descriptor,
            capabilities: projection.capabilities,
            language_mode: projection.language_mode,
            proof: NativeRuntimeModelProof::Verified(Arc::new(verified_pack)),
        }
    }

    fn verified_pack(
        &self,
    ) -> Result<&Arc<crate::models::pack_verifier::VerifiedPack>, NativeAsrError> {
        match &self.proof {
            NativeRuntimeModelProof::Verified(pack) => Ok(pack),
            #[cfg(test)]
            NativeRuntimeModelProof::UnverifiedFixture => Err(NativeAsrError::SessionFailed {
                message: "test-only runtime adapter has no verified pack proof".to_string(),
            }),
        }
    }

    /// Resolve the caller-visible model id from the metadata already parsed by
    /// the pack verifier. This deliberately does not read the GGUF again: the
    /// returned identity and the adapter are projections of the same open,
    /// content-addressed generation.
    pub fn verified_runtime_model_identity(
        &self,
        explicit_model_id_fallback: Option<&str>,
    ) -> Result<NativeRuntimeModelIdentity, NativeAsrError> {
        const MODEL_ID_KEYS: [&str; 3] = ["openasr.model.id", "general.basename", "general.name"];
        let verified_pack = self.verified_pack()?;
        let preflight = verified_pack.preflight();
        let metadata = MODEL_ID_KEYS
            .iter()
            .filter_map(|key| {
                preflight
                    .metadata
                    .get_string(key)
                    .map(|value| ((*key).to_string(), value.to_string()))
            })
            .collect();
        native_model_id::resolve_native_runtime_model_identity_from_string_metadata(
            &metadata,
            preflight.runtime_source().path(),
            explicit_model_id_fallback,
        )
        .map_err(|error| NativeAsrError::SessionFailed {
            message: error.to_string(),
        })
    }

    /// Match a caller's model ref against this exact verified pack generation.
    ///
    /// The ordinary matcher intentionally remains limited to stable spelling
    /// differences (bare ids and quant aliases). Published compatibility for a
    /// pack whose metadata uses an upstream runtime id is evaluated here, where
    /// the verified content, route, and selected adapter are all available.
    pub fn verified_pack_matches_model_ref(&self, requested: &str) -> Result<bool, NativeAsrError> {
        let identity = self.verified_runtime_model_identity(None)?;
        let verified_pack = self.verified_pack()?;
        Ok(native_runtime_model_ref_matches_verified_pack(
            requested,
            &identity.model_id,
            verified_pack,
            &self.descriptor,
        ))
    }

    /// Bind a caller-visible model id to the exact verified source generation
    /// that produced this adapter. Product offline and streaming execution use
    /// this constructor so the proof crosses the executor seam with the pack
    /// reference instead of being discarded and regenerated from `root`.
    pub fn model_pack_ref(
        &self,
        id: impl Into<String>,
    ) -> Result<NativeAsrModelPackRef, NativeAsrError> {
        let verified_pack = Arc::clone(self.verified_pack()?);
        Ok(NativeAsrModelPackRef::from_verified(
            id,
            self.descriptor.model_family,
            verified_pack,
        ))
    }

    /// Whether this model's own decode carries the speaker structure. Read
    /// from the family descriptor (the single declaration); never re-derived
    /// from pack metadata or an `adapter_id` string match.
    #[cfg(test)]
    fn segments_speakers_in_decoder(&self) -> bool {
        self.descriptor.speaker_segmentation.is_in_decoder()
    }

    pub(crate) fn language_mode(&self) -> crate::models::language::LanguageMode {
        self.language_mode
    }

    /// Whether file Voice ID for this family requires the shared forced
    /// aligner in addition to external diarization. The architecture
    /// descriptor is the sole source of this capability fact.
    pub fn requires_forced_aligner_for_voice_id(&self) -> bool {
        !self.descriptor.speaker_segmentation.is_in_decoder()
            && self.descriptor.word_timestamp_source
                == crate::arch::WordTimestampSource::ForcedAligner
    }
}

const GRANITE_SPEECH_PUBLISHED_CATALOG_MODEL_ID: &str = "granite-speech-4.1-2b";
const GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID: &str = "ibm-granite/granite-speech-4.1-2b";
const GRANITE_SPEECH_PUBLISHED_FP16_CONTENT_ID: &str =
    "sha256:56ff48fd309c7219c416492ef71ff56bbf6cc836a5c6e0176f0800b3062080e0";
const GRANITE_SPEECH_PUBLISHED_Q8_CONTENT_ID: &str =
    "sha256:7368242e65f8f907bae8002a609f966a25fcb32af6575b1baae6057c48e6566c";
const GRANITE_SPEECH_PUBLISHED_Q4_CONTENT_ID: &str =
    "sha256:8092fb3209781dfee9ebd4ee2c203ab10d75597470c33d728bf3802d26289758";
#[cfg(test)]
const GRANITE_SPEECH_PUBLISHED_CONTENT_ID: &str = GRANITE_SPEECH_PUBLISHED_Q8_CONTENT_ID;

fn granite_speech_published_content_id_for_quant(quant: &str) -> Option<&'static str> {
    match crate::canonical_quant_tag(quant) {
        "fp16" => Some(GRANITE_SPEECH_PUBLISHED_FP16_CONTENT_ID),
        "q8_0" => Some(GRANITE_SPEECH_PUBLISHED_Q8_CONTENT_ID),
        "q4_k" => Some(GRANITE_SPEECH_PUBLISHED_Q4_CONTENT_ID),
        _ => None,
    }
}

/// Content-bound compatibility for the already-published Granite Speech packs.
///
/// This is deliberately separate from `native_runtime_model_refs_match`: that
/// matcher remains a generic id/quant spelling check and must not learn an
/// upstream alias. This compatibility exists only while every immutable fact
/// below still identifies one exact quant/content pair from the reviewed pack
/// generation.
pub(super) fn native_runtime_model_ref_matches_verified_pack(
    requested: &str,
    runtime_source_id: &str,
    verified_pack: &crate::models::pack_verifier::VerifiedPack,
    selected_adapter: &GgmlFamilyAdapterDescriptor,
) -> bool {
    if native_transcribe::native_runtime_model_refs_match(requested, runtime_source_id) {
        return true;
    }

    granite_speech_published_identity_compatibility_matches(
        requested,
        runtime_source_id,
        verified_pack.content_id(),
        verified_pack.proves_asr_family(
            GRANITE_SPEECH_MODEL_FAMILY,
            GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
        ),
        selected_adapter.adapter_id,
        selected_adapter.model_family,
        selected_adapter.model_architecture,
    )
}

fn granite_speech_published_identity_compatibility_matches(
    requested: &str,
    runtime_source_id: &str,
    content_id: &str,
    proven_granite_route: bool,
    adapter_id: &str,
    model_family: &str,
    model_architecture: &str,
) -> bool {
    let Ok(requested) = crate::parse_model_ref(requested.trim()) else {
        return false;
    };
    let expected_content_id = requested
        .tag
        .as_deref()
        .and_then(granite_speech_published_content_id_for_quant);
    requested.family == GRANITE_SPEECH_PUBLISHED_CATALOG_MODEL_ID
        && expected_content_id.is_some_and(|expected| content_id == expected)
        && runtime_source_id.trim() == GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID
        && proven_granite_route
        && adapter_id == GRANITE_SPEECH_GGML_ADAPTER_ID
        && model_family == GRANITE_SPEECH_MODEL_FAMILY
        && model_architecture == GRANITE_SPEECH_GGML_ARCHITECTURE_ID
}

impl NativeRuntimeModelProjection {
    fn new(
        descriptor: GgmlFamilyAdapterDescriptor,
        metadata: &crate::GgufMetadata,
        tensor_index: Option<&crate::GgufTensorIndex>,
    ) -> Self {
        let capabilities = native_runtime_streaming_capabilities_for_descriptor(&descriptor)
            .with_phrase_bias(native_runtime_descriptor_supports_phrase_bias(
                &descriptor,
                tensor_index,
            ))
            .with_timestamps(true)
            .with_in_decoder_speakers(descriptor.speaker_segmentation.is_in_decoder())
            .with_quantized_models(true)
            .with_hardware_acceleration(true);
        let language_mode = crate::models::language::resolve_language_mode(
            descriptor.language_family_hint,
            metadata,
        );
        Self {
            descriptor,
            capabilities,
            language_mode,
        }
    }
}

fn native_runtime_streaming_capabilities_for_descriptor(
    descriptor: &GgmlFamilyAdapterDescriptor,
) -> NativeAsrCapabilities {
    // Realtime cadence is descriptor/registry-driven, not pack-declared: a family
    // gets true-streaming partials iff a streaming executor is registered for its
    // adapter (`build_builtin_ggml_streaming_execution_dispatch`). Every builtin
    // ASR family registers one -- the startup completeness gate there rejects any
    // that does not -- so no real pack falls to the buffered file-per-utterance
    // path anymore. The pack no longer needs to self-declare streaming; a stale
    // declaration on an already-published pack is simply ignored.
    let Some(architecture) = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(descriptor.model_architecture)
    else {
        return NativeAsrCapabilities::native_offline();
    };
    // Partial granularity is a property of the registered streaming executor:
    // frame-sync (append-only, never revises) vs buffered (re-decodes a growing
    // window). Only xasr-zipformer is frame-sync today.
    NativeAsrCapabilities::native_true_streaming()
        .with_partial_results(true)
        .with_frame_sync_partials(matches!(
            architecture
                .execution_contract
                .streaming_partial_granularity,
            StreamingPartialGranularity::FrameSync
        ))
}

/// Phrase-bias capability for one runtime pack.
///
/// The descriptor strategy is architecture-level, while a `RequiresTensor`
/// strategy binds availability to the already-preflighted pack. Dolphin's
/// deep-biasing `context_module.*` weights are only present on some packs (the
/// multi-lingual `small`/`base` catalog tiers never trained them); reporting
/// an unconditional family-wide `true` there used to let
/// requests reach `hotword_context.rs`, which then hard-fails with a
/// `MissingWeight` error instead of a clean, pre-decode capability rejection.
fn native_runtime_descriptor_supports_phrase_bias(
    descriptor: &GgmlFamilyAdapterDescriptor,
    tensor_index: Option<&crate::GgufTensorIndex>,
) -> bool {
    match descriptor.phrase_bias {
        OpenAsrPhraseBiasStrategy::Unsupported => false,
        OpenAsrPhraseBiasStrategy::Always => true,
        OpenAsrPhraseBiasStrategy::RequiresTensor { tensor_name } => {
            tensor_index.is_some_and(|tensor_index| tensor_index.get(tensor_name).is_some())
        }
    }
}

impl NativeAsrModelAdapter for NativeRuntimeModelAdapter {
    fn adapter_id(&self) -> &'static str {
        self.descriptor.adapter_id
    }

    fn model_family(&self) -> &'static str {
        self.descriptor.model_family
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        self.capabilities.clone()
    }

    fn tensor_layout(&self) -> Option<NativeAsrTensorLayoutRef> {
        Some(NativeAsrTensorLayoutRef::new(
            self.descriptor.model_architecture,
            "gguf",
        ))
    }

    fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
        if model_pack.family != self.descriptor.model_family {
            return false;
        }
        match &self.proof {
            NativeRuntimeModelProof::Verified(pack) => {
                pack.preflight().runtime_source().path() == model_pack.root
            }
            #[cfg(test)]
            NativeRuntimeModelProof::UnverifiedFixture => true,
        }
    }

    fn start_streaming_session(
        &self,
        execution_services: Arc<NativeExecutionServices>,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        session_config.validate()?;
        if options.voice_id {
            return Err(NativeAsrError::VoiceIdUnsupportedForRealtime);
        }
        if !self.capabilities.supports_true_streaming {
            return Err(NativeAsrError::BackendDoesNotSupportTrueStreaming {
                backend: self.adapter_id().to_string(),
            });
        }
        reject_unsupported_native_phrase_bias(
            self.adapter_id(),
            self.model_family(),
            self.capabilities.supports_phrase_bias,
            options.phrase_bias.as_ref(),
        )?;
        if !self.supports_model_pack(model_pack) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "adapter '{}' for family '{}' does not support model pack '{}' ({})",
                    self.adapter_id(),
                    self.model_family(),
                    model_pack.id,
                    model_pack.family
                ),
            });
        }
        super::reject_unsupported_task_or_language(
            self.descriptor.adapter_id,
            self.language_mode,
            options.task.unwrap_or_default(),
            options.language.as_deref(),
        )
        .map_err(native_backend_error_to_asr)?;
        let request_options = native_streaming_request_options_from_session_options(&options);
        let verified_pack = self.verified_pack()?;
        if !matches!(
            verified_pack.route(),
            crate::models::pack_verifier::PackRoute::Asr {
                model_architecture,
                ..
            } if *model_architecture == self.descriptor.model_architecture
        ) {
            return Err(NativeAsrError::SessionFailed {
                message: format!(
                    "native streaming pack route {:?} does not match selected architecture '{}'",
                    verified_pack.route(),
                    self.descriptor.model_architecture
                ),
            });
        }
        let request_intent = execution_intent_from_hardware_target(target)?;
        let execution_plan = resolve_native_execution_plan_for_hardware_target(
            execution_services.as_ref(),
            &self.descriptor,
            target,
        )?;
        let streaming_punctuator =
            crate::models::firered_punc::streaming_runtime::PolicyResolvedStreamingPunctuator::prepare(
                Arc::clone(&execution_services),
                self.descriptor.model_architecture,
                self.descriptor.adapter_id,
                &request_intent,
            )
            .map_err(|error| {
                native_ggml_streaming_error_to_asr(self.descriptor.adapter_id, error)
            })?;
        let factory: Arc<dyn NativeStreamingSessionCandidateBuilder> =
            Arc::new(NativeStreamingSessionCandidateFactory {
                execution_services,
                verified_pack: Arc::clone(verified_pack),
                selected_family: self.descriptor.clone(),
                request_options,
                configured_diarize: options.voice_id,
                session_context: context,
                session_config: session_config.into(),
                auto_gpu_policy: crate::arch::family_auto_gpu_policy_for_model_architecture(
                    self.descriptor.model_architecture,
                ),
                streaming_punctuator,
            });
        PolicyResolvedNativeStreamingSession::start(factory, execution_plan)
    }
}

/// Immutable inputs needed to construct a streaming session on any one
/// semantics-equivalent execution candidate. Keeping this factory at the API
/// boundary lets session acquisition and worker-thread warm-up share the same
/// policy without teaching model-family executors about product fallback.
struct NativeStreamingSessionCandidateFactory {
    execution_services: Arc<NativeExecutionServices>,
    verified_pack: Arc<crate::models::pack_verifier::VerifiedPack>,
    selected_family: GgmlFamilyAdapterDescriptor,
    request_options: GgmlAsrExecutionOptions,
    configured_diarize: bool,
    session_context: NativeAsrSessionContext,
    session_config: crate::models::ggml_asr_executor::GgmlAsrStreamingSessionConfig,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
    streaming_punctuator: Option<
        Arc<crate::models::firered_punc::streaming_runtime::PolicyResolvedStreamingPunctuator>,
    >,
}

trait NativeStreamingSessionCandidateBuilder: Send + Sync {
    fn execution_services(&self) -> Arc<NativeExecutionServices>;

    fn initialize_auxiliary_runtimes(&self) -> Result<(), NativeAsrError> {
        Ok(())
    }

    fn build(
        &self,
        candidate: &ExecutionCandidate,
    ) -> crate::models::native_execution_services::ExecutionCandidateAttemptOutcome<
        Box<dyn NativeAsrSession>,
        NativeAsrError,
    >;
}

impl NativeStreamingSessionCandidateBuilder for NativeStreamingSessionCandidateFactory {
    fn execution_services(&self) -> Arc<NativeExecutionServices> {
        Arc::clone(&self.execution_services)
    }

    fn initialize_auxiliary_runtimes(&self) -> Result<(), NativeAsrError> {
        if let Some(punctuator) = self.streaming_punctuator.as_ref() {
            punctuator.initialize().map_err(|error| {
                native_ggml_streaming_error_to_asr(self.selected_family.adapter_id, error)
            })?;
        }
        Ok(())
    }

    fn build(
        &self,
        candidate: &ExecutionCandidate,
    ) -> crate::models::native_execution_services::ExecutionCandidateAttemptOutcome<
        Box<dyn NativeAsrSession>,
        NativeAsrError,
    > {
        crate::models::native_execution_services::run_execution_candidate_attempt(
            self.execution_services.as_ref(),
            candidate,
            || {
                let resolved_runtime =
                    resolved_runtime_for_candidate(candidate, self.auto_gpu_policy);
                let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput::for_streaming_session(
                    self.verified_pack.preflight(),
                    &self.request_options,
                    resolved_runtime.backend(),
                )
                .map_err(|error| NativeAsrError::SessionFailed {
                    message: format!("native decoder-state planning failed: {error}"),
                })?;
                let decoder_state = self
                    .execution_services
                    .streaming_dispatch()
                    .plan_decoder_state(&self.selected_family, &planning_input)
                    .map_err(|error| {
                        native_ggml_streaming_error_to_asr(self.selected_family.adapter_id, error)
                    })?;
                let request = GgmlAsrStreamingSessionRequest {
                    execution_services: Arc::clone(&self.execution_services),
                    decoder_state,
                    verified_pack: self.verified_pack.as_ref().clone(),
                    selected_family: self.selected_family.clone(),
                    request_options: self.request_options.clone(),
                    configured_diarize: self.configured_diarize,
                    backend_preference: coarse_backend_preference_for_candidate(candidate),
                    resolved_runtime,
                    final_text_processor: self
                        .streaming_punctuator
                        .as_ref()
                        .map(|punctuator| punctuator.slot()),
                    session_context: self.session_context.clone(),
                    session_config: self.session_config.clone(),
                };
                self.execution_services
                    .streaming_dispatch()
                    .start_streaming_session(&request)
                    .map_err(|error| {
                        native_ggml_streaming_error_to_asr(self.selected_family.adapter_id, error)
                    })
            },
        )
    }
}

/// A streaming session pinned to one resolved execution candidate. The wrapper
/// reinstalls that candidate on every worker-thread call. Construction and
/// warm-up may advance through typed candidate-local failures; once the first
/// audio frame is handed to the model, replay would be semantically ambiguous
/// and fallback is permanently disabled for the session.
struct PolicyResolvedNativeStreamingSession {
    factory: Arc<dyn NativeStreamingSessionCandidateBuilder>,
    execution_plan: ExecutionPlan,
    candidate_index: usize,
    session_id: String,
    session: Option<Box<dyn NativeAsrSession>>,
    terminal_error: Option<NativeAsrError>,
    cancellation_token: Option<Arc<AtomicBool>>,
    audio_started: bool,
    auxiliary_ready: bool,
}

impl PolicyResolvedNativeStreamingSession {
    fn start(
        factory: Arc<dyn NativeStreamingSessionCandidateBuilder>,
        execution_plan: ExecutionPlan,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        let (candidate_index, session) =
            Self::construct_from(factory.as_ref(), &execution_plan, 0)?;
        let session_id = session.session_id().to_string();
        Ok(Box::new(Self {
            factory,
            execution_plan,
            candidate_index,
            session_id,
            session: Some(session),
            terminal_error: None,
            cancellation_token: None,
            audio_started: false,
            auxiliary_ready: false,
        }))
    }

    fn construct_from(
        factory: &dyn NativeStreamingSessionCandidateBuilder,
        execution_plan: &ExecutionPlan,
        start_index: usize,
    ) -> Result<(usize, Box<dyn NativeAsrSession>), NativeAsrError> {
        for (candidate_index, candidate) in execution_plan
            .candidates()
            .iter()
            .enumerate()
            .skip(start_index)
        {
            let attempt = factory.build(candidate);
            match (attempt.result, attempt.candidate_failure) {
                (Ok(session), None) => return Ok((candidate_index, session)),
                (Err(error), None) => return Err(error),
                (result, Some(failure)) => {
                    let error = crate::models::native_execution_services::execution_candidate_failure_source(result)
                        .unwrap_or_else(|| candidate_success_with_failure_error("session-build", &failure));
                    if candidate_index + 1 == execution_plan.candidates().len() {
                        return Err(error);
                    }
                    log_streaming_candidate_retry("session-build", candidate, &failure);
                }
            }
        }
        Err(NativeAsrError::SessionFailed {
            message: "execution policy produced no streaming candidate attempts".to_string(),
        })
    }

    fn candidate(&self) -> &ExecutionCandidate {
        &self.execution_plan.candidates()[self.candidate_index]
    }

    fn run_current<T>(
        &mut self,
        operation: impl FnOnce(&mut dyn NativeAsrSession) -> Result<T, NativeAsrError>,
    ) -> crate::models::native_execution_services::ExecutionCandidateAttemptOutcome<T, NativeAsrError>
    {
        if let Some(error) = self.terminal_error.clone() {
            return crate::models::native_execution_services::ExecutionCandidateAttemptOutcome {
                result: Err(error),
                candidate_failure: None,
            };
        }
        let candidate = self.candidate().clone();
        let services = self.factory.execution_services();
        crate::models::native_execution_services::run_execution_candidate_attempt(
            services.as_ref(),
            &candidate,
            || {
                operation(
                    self.session
                        .as_deref_mut()
                        .expect("a non-terminal policy session owns an active candidate"),
                )
            },
        )
    }

    fn replace_with_next_candidate(
        &mut self,
        previous_error: NativeAsrError,
    ) -> Result<(), NativeAsrError> {
        let next_index = self.candidate_index.saturating_add(1);
        if next_index >= self.execution_plan.candidates().len() {
            self.terminal_error = Some(previous_error.clone());
            return Err(previous_error);
        }
        debug_assert!(
            self.session.is_none(),
            "a failed streaming candidate must be invalidated before replacement"
        );
        let (candidate_index, mut session) =
            match Self::construct_from(self.factory.as_ref(), &self.execution_plan, next_index) {
                Ok(constructed) => constructed,
                Err(error) => {
                    // The wrapper remains a valid, fail-closed object even when
                    // every later candidate failed to construct. Callers may
                    // legally inspect/cancel a session after warm_up returned an
                    // error; retaining a terminal error prevents that path from
                    // panicking on an empty session slot.
                    self.terminal_error = Some(error.clone());
                    return Err(error);
                }
            };
        debug_assert_eq!(
            session.session_id(),
            self.session_id,
            "a replacement candidate must preserve the public session identity"
        );
        if let Some(token) = self.cancellation_token.as_ref() {
            session.set_cancellation_token(Arc::clone(token));
        }
        self.candidate_index = candidate_index;
        self.session = Some(session);
        Ok(())
    }

    fn invalidate_current_candidate(&mut self) {
        if let Some(session) = self.session.take() {
            crate::models::native_execution_services::drop_execution_candidate_value_without_cache_publication(
                session,
            );
        }
    }

    fn finish_current_attempt<T>(
        &mut self,
        operation: &'static str,
        attempt: crate::models::native_execution_services::ExecutionCandidateAttemptOutcome<
            T,
            NativeAsrError,
        >,
    ) -> Result<T, NativeAsrError> {
        match (attempt.result, attempt.candidate_failure) {
            (result, None) => result,
            (result, Some(failure)) => {
                let error =
                    crate::models::native_execution_services::execution_candidate_failure_source(
                        result,
                    )
                    .unwrap_or_else(|| candidate_success_with_failure_error(operation, &failure));
                self.invalidate_current_candidate();
                self.terminal_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn ensure_auxiliary_ready(&mut self) -> Result<(), NativeAsrError> {
        if self.auxiliary_ready {
            return Ok(());
        }
        self.factory.initialize_auxiliary_runtimes()?;
        self.auxiliary_ready = true;
        Ok(())
    }

    fn ensure_auxiliary_ready_for_buffered_audio(&mut self) -> Result<(), NativeAsrError> {
        if self.audio_started {
            self.ensure_auxiliary_ready()?;
        }
        Ok(())
    }
}

impl NativeAsrSession for PolicyResolvedNativeStreamingSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn set_cancellation_token(&mut self, cancelled: Arc<AtomicBool>) {
        self.cancellation_token = Some(Arc::clone(&cancelled));
        let attempt = self.run_current(|session| {
            session.set_cancellation_token(cancelled);
            Ok::<_, NativeAsrError>(())
        });
        let _ = self.finish_current_attempt("set-cancellation-token", attempt);
    }

    fn push_audio(
        &mut self,
        frame: crate::RealtimeAudioFrame,
    ) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready()?;
        // Conservative boundary: after control enters the family session we
        // cannot prove whether a failing implementation consumed part of the
        // frame, so retry is disabled before making the call.
        self.audio_started = true;
        let attempt = self.run_current(|session| session.push_audio(frame));
        self.finish_current_attempt("push-audio", attempt)
    }

    fn poll_events(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        let attempt = self.run_current(|session| session.poll_events());
        self.finish_current_attempt("poll-events", attempt)
    }

    fn flush(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready_for_buffered_audio()?;
        let attempt = self.run_current(|session| session.flush());
        self.finish_current_attempt("flush", attempt)
    }

    fn warm_up(&mut self) -> Result<(), NativeAsrError> {
        loop {
            let candidate = self.candidate().clone();
            let attempt = self.run_current(|session| session.warm_up());
            match (attempt.result, attempt.candidate_failure) {
                (Ok(()), None) => return self.ensure_auxiliary_ready(),
                (Err(error), None) => return Err(error),
                (result, Some(failure)) => {
                    let error = crate::models::native_execution_services::execution_candidate_failure_source(result)
                        .unwrap_or_else(|| candidate_success_with_failure_error("warm-up", &failure));
                    self.invalidate_current_candidate();
                    if self.audio_started {
                        self.terminal_error = Some(error.clone());
                        return Err(error);
                    }
                    log_streaming_candidate_retry("warm-up", &candidate, &failure);
                    self.replace_with_next_candidate(error)?;
                }
            }
        }
    }

    fn finalize_utterance(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready_for_buffered_audio()?;
        let attempt = self.run_current(|session| session.finalize_utterance());
        self.finish_current_attempt("finalize-utterance", attempt)
    }

    fn split_utterance(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready_for_buffered_audio()?;
        let attempt = self.run_current(|session| session.split_utterance());
        self.finish_current_attempt("split-utterance", attempt)
    }

    fn finish(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready_for_buffered_audio()?;
        let attempt = self.run_current(|session| session.finish());
        self.finish_current_attempt("finish", attempt)
    }

    fn close(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_auxiliary_ready_for_buffered_audio()?;
        let attempt = self.run_current(|session| session.close());
        self.finish_current_attempt("close", attempt)
    }

    fn cancel(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
        // Cancellation is cleanup, never an inference boundary. It must stay
        // allocation-free even when the session was canceled before warmup or
        // the first audio frame.
        let attempt = self.run_current(|session| session.cancel());
        self.finish_current_attempt("cancel", attempt)
    }
}

fn candidate_success_with_failure_error(
    operation: &'static str,
    failure: &crate::device::execution_policy::ExecutionCandidateFailure,
) -> NativeAsrError {
    NativeAsrError::SessionFailed {
        message: format!(
            "execution candidate reported {:?} during {operation} ({}) despite returning success",
            failure.kind, failure.operation
        ),
    }
}

fn log_streaming_candidate_retry(
    operation: &'static str,
    candidate: &ExecutionCandidate,
    failure: &crate::device::execution_policy::ExecutionCandidateFailure,
) {
    crate::stage_timing::log_detail_event(
        "native_streaming",
        format_args!(
            "stage=execution_candidate event=retry operation={operation} provider={} placement={:?} failure={:?} failure_operation={}",
            candidate.device.route.provider, candidate.placement, failure.kind, failure.operation,
        ),
    );
}

fn native_streaming_request_options_from_session_options(
    options: &NativeAsrRequestOptions,
) -> GgmlAsrExecutionOptions {
    let mut request_options = GgmlAsrExecutionOptions::from_transcription_request_with_phrase_bias(
        options.language.clone(),
        options.prompt.clone(),
        options.phrase_bias.clone(),
        None,
    );
    request_options.task = options.task.unwrap_or_default();
    request_options.inference_threads = options.inference_threads.map(usize::from);
    request_options.word_timestamps = options.word_timestamps;
    // Recording-level Voice ID is rejected by `start_streaming_session` before
    // device/model execution. Keep the lower decode request speaker-free too,
    // so a future caller cannot accidentally revive model-local speaker tokens
    // by bypassing only the public capability probe.
    request_options.in_decoder_speakers = false;
    request_options
}

impl TranscriptionBackend for NativeBackend {
    fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcription, BackendError> {
        native_transcribe::run_native_transcription(request, Arc::clone(&self.execution_services))
    }
}

impl NativeAsrExecutor for NativeBackendExecutor {
    fn executor_id(&self) -> &'static str {
        "openasr-native-backend-v1"
    }

    fn capabilities(&self) -> NativeAsrCapabilities {
        NativeAsrCapabilities::native_offline()
            .with_timestamps(true)
            .with_quantized_models(true)
            .with_hardware_acceleration(true)
    }

    fn runtime_readiness(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
    ) -> NativeAsrRuntimeReadiness {
        let policy_readiness = native_runtime_policy_readiness(adapter, model_pack, target);
        if !matches!(policy_readiness, NativeAsrRuntimeReadiness::Ready) {
            return policy_readiness;
        }
        // This explicit diagnostic probe answers a caller asking whether a
        // bare path is ready. Product execution does not call it and then
        // reopen the path: the offline path verifies once inside
        // `run_native_transcription_impl`, while the streaming path consumes
        // the exact `VerifiedPack` attached to `NativeRuntimeModelAdapter`.
        if !model_pack.root.exists() {
            return NativeAsrRuntimeReadiness::MissingLocalModelAsset {
                path: model_pack.root.clone(),
            };
        }
        match native_path::validate_local_native_runtime_source(&model_pack.root) {
            Ok(_) => NativeAsrRuntimeReadiness::Ready,
            Err(error) => NativeAsrRuntimeReadiness::UnsupportedModelPack {
                reason: error.to_string(),
            },
        }
    }

    fn transcribe(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        request: NativeAsrOfflineRequest,
    ) -> Result<Transcription, NativeAsrError> {
        match native_runtime_policy_readiness(adapter, model_pack, target) {
            NativeAsrRuntimeReadiness::Ready => {}
            other => {
                return Err(NativeAsrError::try_from(other)
                    .expect("non-ready runtime readiness converts to NativeAsrError"));
            }
        }
        let execution_target = native_execution_target_from_hardware_target(target)
            .ok_or(NativeAsrError::UnsupportedHardwareTarget { target })?;
        let execution_intent = execution_intent_from_hardware_target(target)?;
        let adapter_capabilities = adapter.capabilities();
        reject_unsupported_native_phrase_bias(
            adapter.adapter_id(),
            adapter.model_family(),
            adapter_capabilities.supports_phrase_bias,
            request.options.phrase_bias.as_ref(),
        )?;
        let request =
            native_offline_request_to_transcription_request(model_pack, execution_target, request);
        let verified_pack = Arc::clone(model_pack.verified_pack()?);
        native_transcribe::run_native_transcription_with_verified_pack(
            request,
            Arc::clone(&self.execution_services),
            Some(execution_intent),
            verified_pack,
        )
        .map_err(native_backend_error_to_asr)
    }

    fn start_session(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        let _ = (adapter, model_pack, target, context, options);
        Err(NativeAsrError::BackendDoesNotSupportTrueStreaming {
            backend: self.executor_id().to_string(),
        })
    }

    fn start_streaming_session(
        &self,
        adapter: &dyn NativeAsrModelAdapter,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        let mut session_config = session_config;
        session_config.validate()?;
        match native_runtime_policy_readiness(adapter, model_pack, target) {
            NativeAsrRuntimeReadiness::Ready => {}
            other => {
                return Err(NativeAsrError::try_from(other)
                    .expect("non-ready runtime readiness converts to NativeAsrError"));
            }
        }
        let adapter_capabilities = adapter.capabilities();
        if !adapter_capabilities.supports_true_streaming {
            return Err(NativeAsrError::BackendDoesNotSupportTrueStreaming {
                backend: adapter.adapter_id().to_string(),
            });
        }
        reject_unsupported_native_phrase_bias(
            adapter.adapter_id(),
            adapter.model_family(),
            adapter_capabilities.supports_phrase_bias,
            options.phrase_bias.as_ref(),
        )?;
        session_config.partial_results = session_config.partial_results
            && options.partial_results
            && adapter_capabilities.supports_partials;
        session_config.word_timestamps = session_config.word_timestamps
            && options.word_timestamps
            && adapter_capabilities.supports_timestamps;
        adapter.start_streaming_session(
            Arc::clone(&self.execution_services),
            model_pack,
            target,
            context,
            options,
            session_config,
        )
    }
}

/// Family/target readiness that does no filesystem I/O. Execution paths call
/// this and then cross exactly one package-verification seam; the public
/// `runtime_readiness` diagnostic adds a path probe only when explicitly
/// requested by a caller.
fn native_runtime_policy_readiness(
    adapter: &dyn NativeAsrModelAdapter,
    model_pack: &NativeAsrModelPackRef,
    target: NativeAsrHardwareTarget,
) -> NativeAsrRuntimeReadiness {
    if !adapter.supports_model_pack(model_pack) {
        return NativeAsrRuntimeReadiness::UnsupportedModelPack {
            reason: format!(
                "adapter '{}' for family '{}' does not support model pack '{}' ({})",
                adapter.adapter_id(),
                adapter.model_family(),
                model_pack.id,
                model_pack.family
            ),
        };
    }
    let intent = match execution_intent_from_hardware_target(target) {
        Ok(intent) => intent,
        Err(_) => return NativeAsrRuntimeReadiness::UnsupportedHardwareTarget { target },
    };
    if let ExecutionIntent::ConstrainedAcceleratedOnly(constraint) = intent {
        let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
        if !inventory
            .iter()
            .any(|device| accelerated_constraint_matches(constraint, device))
        {
            return NativeAsrRuntimeReadiness::ProviderUnavailable {
                provider: hardware_target_provider_label(target).to_string(),
            };
        }
    }
    NativeAsrRuntimeReadiness::Ready
}

fn native_execution_target_from_hardware_target(
    target: NativeAsrHardwareTarget,
) -> Option<ExecutionTarget> {
    match target {
        NativeAsrHardwareTarget::Auto => Some(ExecutionTarget::Auto),
        NativeAsrHardwareTarget::Cpu | NativeAsrHardwareTarget::IntelCpu => {
            Some(ExecutionTarget::Cpu)
        }
        NativeAsrHardwareTarget::Accelerated
        | NativeAsrHardwareTarget::AppleSilicon
        | NativeAsrHardwareTarget::NvidiaCuda
        | NativeAsrHardwareTarget::AmdGpu
        | NativeAsrHardwareTarget::IntelGpu => Some(ExecutionTarget::Accelerated),
        NativeAsrHardwareTarget::IntelNpu => None,
    }
}

fn execution_intent_from_hardware_target(
    target: NativeAsrHardwareTarget,
) -> Result<ExecutionIntent, NativeAsrError> {
    match target {
        NativeAsrHardwareTarget::Auto => Ok(ExecutionIntent::Auto),
        NativeAsrHardwareTarget::Cpu | NativeAsrHardwareTarget::IntelCpu => {
            Ok(ExecutionIntent::CpuOnly)
        }
        NativeAsrHardwareTarget::Accelerated => Ok(ExecutionIntent::AcceleratedOnly),
        NativeAsrHardwareTarget::AppleSilicon => {
            if cfg!(all(target_vendor = "apple", target_arch = "aarch64")) {
                Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
                    AcceleratedDeviceConstraint::Provider(ExecutionProvider::Metal),
                ))
            } else {
                Err(NativeAsrError::UnsupportedHardwareTarget { target })
            }
        }
        NativeAsrHardwareTarget::NvidiaCuda => Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::Provider(ExecutionProvider::Cuda),
        )),
        NativeAsrHardwareTarget::AmdGpu => Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::HardwareVendor(ExecutionHardwareVendor::Amd),
        )),
        NativeAsrHardwareTarget::IntelGpu => Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::HardwareVendor(ExecutionHardwareVendor::Intel),
        )),
        NativeAsrHardwareTarget::IntelNpu => {
            Err(NativeAsrError::UnsupportedHardwareTarget { target })
        }
    }
}

fn resolve_native_execution_plan_for_hardware_target(
    execution_services: &NativeExecutionServices,
    descriptor: &GgmlFamilyAdapterDescriptor,
    target: NativeAsrHardwareTarget,
) -> Result<ExecutionPlan, NativeAsrError> {
    let intent = execution_intent_from_hardware_target(target)?;
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(
            intent,
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                descriptor.model_architecture,
            ),
            descriptor.execution_capabilities,
            &inventory,
        )
        .map_err(|error| execution_policy_error_to_native(error, target))
}

fn execution_policy_error_to_native(
    error: ExecutionPolicyError,
    target: NativeAsrHardwareTarget,
) -> NativeAsrError {
    match error {
        ExecutionPolicyError::Route(error) => NativeAsrError::from_execution_route_error(error),
        ExecutionPolicyError::ConstrainedAcceleratedUnavailable { .. } => {
            NativeAsrError::ProviderUnavailable {
                provider: hardware_target_provider_label(target).to_string(),
            }
        }
        ExecutionPolicyError::UnsupportedPlacement { .. }
        | ExecutionPolicyError::NoAcceleratedPlacement
        | ExecutionPolicyError::NoAcceleratedPlacementForProvider { .. }
        | ExecutionPolicyError::NoAcceleratedPlacementForConstraint { .. }
        | ExecutionPolicyError::NoSupportedCandidate { .. } => {
            NativeAsrError::UnsupportedHardwareTarget { target }
        }
        other => NativeAsrError::SessionFailed {
            message: format!("could not resolve native execution policy: {other}"),
        },
    }
}

fn hardware_target_provider_label(target: NativeAsrHardwareTarget) -> &'static str {
    match target {
        NativeAsrHardwareTarget::AppleSilicon => "Metal on Apple silicon",
        NativeAsrHardwareTarget::NvidiaCuda => "NVIDIA CUDA",
        NativeAsrHardwareTarget::AmdGpu => "AMD GPU (HIP or proven AMD Vulkan)",
        NativeAsrHardwareTarget::IntelGpu => "Intel GPU (proven Intel Vulkan)",
        NativeAsrHardwareTarget::Accelerated => "accelerated",
        NativeAsrHardwareTarget::Cpu | NativeAsrHardwareTarget::IntelCpu => "CPU",
        NativeAsrHardwareTarget::IntelNpu => "Intel NPU",
        NativeAsrHardwareTarget::Auto => "auto",
    }
}

fn accelerated_constraint_matches(
    constraint: AcceleratedDeviceConstraint,
    device: &crate::device::execution_route::EnumeratedComputeDevice,
) -> bool {
    match constraint {
        AcceleratedDeviceConstraint::Provider(provider) => device.provider == provider,
        AcceleratedDeviceConstraint::HardwareVendor(vendor) => {
            device.hardware_vendor == Some(vendor)
        }
    }
}

fn coarse_backend_preference_for_candidate(
    candidate: &ExecutionCandidate,
) -> GgmlAsrBackendPreference {
    match candidate.placement {
        ExecutionPlacement::CpuOnly => GgmlAsrBackendPreference::CpuOnly,
        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => {
            GgmlAsrBackendPreference::Accelerated
        }
    }
}

fn resolved_runtime_for_candidate(
    candidate: &ExecutionCandidate,
    auto_gpu_policy: crate::ggml_runtime::AutoGpuPolicy,
) -> crate::ggml_runtime::ResolvedFamilyRuntimeInput {
    let preference = match candidate.placement {
        ExecutionPlacement::CpuOnly => Some(RequestBackendPreference::CpuOnly),
        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => Some(
            RequestBackendPreference::Exact(candidate.device.route.clone()),
        ),
    };
    crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(preference, auto_gpu_policy)
}

fn native_ggml_streaming_error_to_asr(
    adapter_id: &'static str,
    error: GgmlAsrExecutionError,
) -> NativeAsrError {
    match error {
        GgmlAsrExecutionError::ExecutorUnavailable { .. } => {
            NativeAsrError::BackendDoesNotSupportTrueStreaming {
                backend: adapter_id.to_string(),
            }
        }
        GgmlAsrExecutionError::ExecutionRoute(error) => {
            NativeAsrError::from_execution_route_error(error)
        }
        other => {
            if let Some(route_error) =
                crate::device::execution_route::ExecutionRouteError::from_embedded_message(
                    &other.to_string(),
                )
            {
                return NativeAsrError::from_execution_route_error(route_error);
            }
            NativeAsrError::SessionFailed {
                message: format!("native ggml streaming session failed: {other}"),
            }
        }
    }
}

fn reject_unsupported_native_phrase_bias(
    adapter: &'static str,
    model_family: &'static str,
    supported: bool,
    phrase_bias: Option<&crate::PhraseBiasConfig>,
) -> Result<(), NativeAsrError> {
    if supported || phrase_bias.is_none_or(crate::PhraseBiasConfig::is_empty) {
        return Ok(());
    }

    Err(NativeAsrError::PhraseBiasUnsupportedByModel {
        adapter: adapter.to_string(),
        model_family: model_family.to_string(),
    })
}

fn native_offline_request_to_transcription_request(
    model_pack: &NativeAsrModelPackRef,
    execution_target: ExecutionTarget,
    request: NativeAsrOfflineRequest,
) -> TranscriptionRequest {
    let segmenter = request.voice_id_segmenter;
    let mut converted = TranscriptionRequest::new(request.input_path, model_pack.id.clone())
        .with_model_pack_path(Some(model_pack.root.clone()))
        .with_language(request.options.language)
        .with_task(request.options.task)
        .with_prompt(request.options.prompt)
        .with_phrase_bias(request.options.phrase_bias)
        .with_inference_threads(request.options.inference_threads)
        .with_execution_target(Some(execution_target))
        .with_word_timestamps(request.options.word_timestamps)
        .with_word_timestamps_refine(request.options.word_timestamps_refine)
        .with_voice_id(request.options.voice_id)
        .with_longform(request.longform)
        .with_display_file_name(request.display_file_name)
        .with_source(request.source)
        .with_source_audio_format(request.source_sample_rate_hz, request.source_channels)
        .with_source_container(request.source_container)
        .with_prepared_samples(request.prepared_samples)
        .with_execution_context(request.execution_context)
        .with_serve_batch_max_native_sessions(request.serve_batch_max_native_sessions);
    converted.voice_id_segmenter = segmenter;
    converted
}

fn native_backend_error_to_asr(error: BackendError) -> NativeAsrError {
    let message = match error {
        BackendError::NativeFailClosed { reason } => reason,
        error => error.to_string(),
    };
    NativeAsrError::SessionFailed { message }
}

pub fn validate_local_native_model_pack_path(
    path: &Path,
) -> Result<std::path::PathBuf, BackendError> {
    native_path::validate_local_native_model_pack_path(path)
}

const NATIVE_RUNTIME_LOOKUP_CACHE_CAPACITY: usize = 64;

type NativeRuntimeIdentityCacheKey = (String, PathBuf, Option<String>);
type NativeRuntimeIdentityCacheValue =
    Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError>;

static NATIVE_RUNTIME_MODEL_IDENTITY_CACHE: OnceLock<
    Mutex<
        HashMap<
            NativeRuntimeIdentityCacheKey,
            Arc<OnceLock<Option<NativeRuntimeIdentityCacheValue>>>,
        >,
    >,
> = OnceLock::new();

/// Content-keyed, bounded single-flight cache.
///
/// A failed build is shared only by callers already waiting on the same
/// in-flight cell, then removed immediately. This prevents permanent negative
/// caching while ensuring concurrent cold misses parse one content generation
/// once. Eviction never removes an in-flight entry; a burst may temporarily
/// exceed `capacity`, then converges as completed entries become evictable.
fn native_content_cache_get_or_build<K, V>(
    cache: &Mutex<HashMap<K, Arc<OnceLock<Option<V>>>>>,
    capacity: usize,
    key: K,
    build: impl FnOnce() -> Option<V>,
) -> Option<V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    debug_assert!(capacity > 0);
    let cell = match cache.lock() {
        Ok(mut cache) => {
            if let Some(cell) = cache.get(&key) {
                Arc::clone(cell)
            } else {
                if cache.len() >= capacity
                    && let Some(evicted_key) = cache
                        .iter()
                        .find_map(|(key, cell)| cell.get().is_some().then(|| key.clone()))
                {
                    cache.remove(&evicted_key);
                }
                let cell = Arc::new(OnceLock::new());
                cache.insert(key.clone(), Arc::clone(&cell));
                cell
            }
        }
        Err(_) => return build(),
    };

    let value = cell.get_or_init(build).clone();
    if let Ok(mut cache) = cache.lock() {
        if value.is_none() {
            if cache
                .get(&key)
                .is_some_and(|cached| Arc::ptr_eq(cached, &cell))
            {
                cache.remove(&key);
            }
        } else {
            while cache.len() > capacity {
                let Some(evicted_key) = cache.iter().find_map(|(candidate, cached)| {
                    (candidate != &key && cached.get().is_some()).then(|| candidate.clone())
                }) else {
                    break;
                };
                cache.remove(&evicted_key);
            }
        }
    }
    value
}

/// Cached path ingress for
/// [`native_model_id::resolve_native_runtime_model_identity_from_source`].
///
/// The content id prevents same-path replacement from inheriting stale
/// metadata. The logical path remains part of the key because its file stem is
/// the final identity fallback; the explicit fallback is request data and must
/// also participate. Invalid paths are not cached. Resolution errors for a
/// valid immutable generation are deterministic and may be cached until that
/// bounded content entry is evicted.
pub fn resolve_local_native_runtime_model_identity(
    runtime_path: &Path,
    explicit_model_id_fallback: Option<&str>,
) -> Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError> {
    let runtime_source =
        crate::validate_ggml_runtime_source_path(runtime_path).map_err(|error| {
            NativeRuntimeModelIdentityError::RuntimeSourceValidation {
                path: runtime_path.display().to_string(),
                reason: error.to_string(),
            }
        })?;
    let cache = NATIVE_RUNTIME_MODEL_IDENTITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = (
        runtime_source.content_id().to_string(),
        runtime_source.path().to_path_buf(),
        explicit_model_id_fallback.map(str::to_string),
    );
    native_content_cache_get_or_build(
        cache,
        NATIVE_RUNTIME_LOOKUP_CACHE_CAPACITY,
        cache_key,
        || {
            Some(
                native_model_id::resolve_native_runtime_model_identity_from_source(
                    &runtime_source,
                    explicit_model_id_fallback,
                ),
            )
        },
    )
    .expect("identity cache builder always returns one deterministic result")
}

pub fn native_runtime_transcription_capabilities_for_path(
    path: &Path,
) -> TranscriptionBackendCapabilities {
    // Build the adapter once and derive every facet (phrase_bias,
    // diarization, language) from it, instead of re-reading and re-parsing
    // the pack's GGUF metadata/tensor index once per facet.
    let adapter = native_runtime_model_adapter_for_path(path);
    let mut capabilities = TranscriptionBackendCapabilities::for_backend_kind(BackendKind::Native);
    capabilities.phrase_bias = native_phrase_bias_capability_for_adapter(adapter.as_ref());
    capabilities.diarization = native_diarization_capability_for_adapter(adapter.as_ref());
    if let Some(adapter) = adapter.as_ref() {
        capabilities.language = super::LanguageCapability::from(adapter.language_mode());
    }
    capabilities
}

pub(crate) const NATIVE_PHRASE_BIAS_UNAVAILABLE_REASON: &str = "Phrase bias / hotword boosting is not implemented for this native model; requests with phrase_bias or hotword fields are rejected.";

fn native_phrase_bias_capability_for_adapter(
    adapter: Option<&NativeRuntimeModelAdapter>,
) -> BackendFeatureCapability {
    if adapter.is_some_and(|adapter| adapter.capabilities().supports_phrase_bias) {
        BackendFeatureCapability::supported()
    } else {
        BackendFeatureCapability::reject_request(NATIVE_PHRASE_BIAS_UNAVAILABLE_REASON)
    }
}

/// Reason reported when the acoustic identity stack required by Voice ID is
/// incomplete. Every model needs ReDimNet2-B6; models without native speaker
/// tracks additionally need the active recording-level segmenter.
pub(crate) const NATIVE_DIARIZATION_UNAVAILABLE_REASON: &str = "Voice ID needs the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn) for acoustic identity. Models that do not separate speakers themselves also need an active local speaker segmenter pack (pyannote-segmentation-3.0, or an installed optional provider selected by the global policy); install the required Voice ID packs or omit diarize=true.";

/// Voice ID capability for a runtime pack: ReDim must be installed for every
/// source, and a family without in-decoder speaker tracks also needs the full
/// external segment/embed/cluster pipeline.
fn native_diarization_capability_for_adapter(
    adapter: Option<&NativeRuntimeModelAdapter>,
) -> BackendFeatureCapability {
    native_diarization_capability(
        native_runtime_adapter_segments_speakers_in_decoder(adapter),
        crate::diarize::embed::embedder_pack_installed(),
        crate::diarize::external_diarization_available(),
    )
}

/// The rule itself, separated from the two live lookups it consults: one
/// segmentation source plus the shared acoustic identity model is enough.
/// Kept as a pure function so every source/dependency row is assertable.
fn native_diarization_capability(
    segments_speakers_in_decoder: bool,
    embedder_available: bool,
    external_pipeline_available: bool,
) -> BackendFeatureCapability {
    if embedder_available && (segments_speakers_in_decoder || external_pipeline_available) {
        BackendFeatureCapability::supported()
    } else {
        BackendFeatureCapability::reject_request(NATIVE_DIARIZATION_UNAVAILABLE_REASON)
    }
}

pub fn native_runtime_realtime_capabilities_for_path(path: &Path) -> RealtimeBackendCapabilities {
    // Same single-build discipline as the transcription facets above: one
    // adapter build serves the one realtime facet derived from it today, and
    // keeps this entry point structurally consistent if more facets are
    // added later.
    let adapter = native_runtime_model_adapter_for_path(path);
    RealtimeBackendCapabilities::from_native_capabilities(&native_runtime_capabilities_for_adapter(
        adapter.as_ref(),
    ))
}

fn native_runtime_capabilities_for_adapter(
    adapter: Option<&NativeRuntimeModelAdapter>,
) -> NativeAsrCapabilities {
    adapter
        .map(|adapter| adapter.capabilities())
        .unwrap_or_else(NativeAsrCapabilities::unsupported)
}

static NATIVE_RUNTIME_MODEL_ADAPTER_CACHE: OnceLock<
    Mutex<HashMap<String, Arc<OnceLock<Option<NativeRuntimeModelProjection>>>>>,
> = OnceLock::new();

/// Resolves and caches the runtime adapter for one immutable pack content.
///
/// Full package/runtime verification and content-id derivation happen before
/// the cache lookup so a same-path replacement gets a new key. The capability
/// projection borrows that exact proof; invalid packs are never cached as
/// negative entries, so a later valid replacement can resolve normally.
pub fn native_runtime_model_adapter_for_path(path: &Path) -> Option<NativeRuntimeModelAdapter> {
    if !native_path::has_supported_native_runtime_pack_path_shape(path) {
        return None;
    }
    let verified_pack = crate::models::pack_verifier::PackVerifier
        .verify_candidate(crate::models::pack_verifier::PackCandidate::new(path))
        .ok()?;
    if !matches!(
        verified_pack.route(),
        crate::models::pack_verifier::PackRoute::Asr { .. }
    ) {
        return None;
    }
    let cache_key = verified_pack.content_id().to_string();
    let cache = NATIVE_RUNTIME_MODEL_ADAPTER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let projection = native_content_cache_get_or_build(
        cache,
        NATIVE_RUNTIME_LOOKUP_CACHE_CAPACITY,
        cache_key,
        || build_native_runtime_model_adapter_from_preflight(verified_pack.preflight()),
    )?;
    // The global cache owns only the small, content-keyed family projection.
    // The caller owns the exact open/mapped proof so a bounded cache cannot
    // pin dozens of model files and mappings after their sessions are gone.
    Some(NativeRuntimeModelAdapter::from_verified(
        projection,
        verified_pack,
    ))
}

fn build_native_runtime_model_adapter_from_preflight(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Option<NativeRuntimeModelProjection> {
    let selection_metadata = selection_metadata_from_gguf(&preflight.metadata);
    let selected = OpenAsrArchitectureRegistry::with_builtins()
        .select_ggml_adapter_from_gguf_metadata_v1(&selection_metadata)
        .ok()?;
    let descriptor = crate::arch::builtin_adapter_descriptor(selected.identity.model_architecture);
    Some(NativeRuntimeModelProjection::new(
        descriptor,
        &preflight.metadata,
        Some(preflight.tensor_index.as_ref()),
    ))
}

pub fn verify_native_runtime_model_pack_path(path: &Path) -> Result<(), String> {
    crate::models::pack_verifier::PackVerifier
        .verify_candidate(crate::models::pack_verifier::PackCandidate::new(path))
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn native_runtime_adapter_segments_speakers_in_decoder(
    adapter: Option<&NativeRuntimeModelAdapter>,
) -> bool {
    adapter.is_some_and(|adapter| adapter.capabilities().supports_in_decoder_speakers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::native::NativeAsrStreamingSessionConfig;
    use crate::testing::{
        TinyGgufFixtureSpec, WhisperExecutionFailureStage,
        classify_whisper_execution_failure_stage, with_forced_cpu_backend_for_test,
        write_tiny_gguf_runtime_source,
    };
    use std::{
        env, fs,
        path::{Path, PathBuf},
        sync::Barrier,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn native_backend_error_mapping_does_not_nest_the_fail_closed_envelope() {
        let error = native_backend_error_to_asr(BackendError::NativeFailClosed {
            reason: "first causal failure".to_string(),
        });

        assert_eq!(
            error,
            NativeAsrError::SessionFailed {
                message: "first causal failure".to_string(),
            }
        );
    }

    #[test]
    fn published_granite_identity_compatibility_is_content_bound() {
        for (requested, content_id) in [
            (
                "granite-speech-4.1-2b:fp16",
                GRANITE_SPEECH_PUBLISHED_FP16_CONTENT_ID,
            ),
            (
                "granite-speech-4.1-2b:q8",
                GRANITE_SPEECH_PUBLISHED_Q8_CONTENT_ID,
            ),
            (
                "granite-speech-4.1-2b:q8_0",
                GRANITE_SPEECH_PUBLISHED_Q8_CONTENT_ID,
            ),
            (
                "granite-speech-4.1-2b:q4",
                GRANITE_SPEECH_PUBLISHED_Q4_CONTENT_ID,
            ),
            (
                "granite-speech-4.1-2b:q4_k_m",
                GRANITE_SPEECH_PUBLISHED_Q4_CONTENT_ID,
            ),
        ] {
            assert!(!native_transcribe::native_runtime_model_refs_match(
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
            ));
            assert!(granite_speech_published_identity_compatibility_matches(
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                content_id,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ));
        }

        let requested = "granite-speech-4.1-2b:q8";

        for (
            label,
            requested,
            runtime_id,
            content_id,
            route,
            adapter_id,
            model_family,
            model_architecture,
        ) in [
            (
                "wrong SHA",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                "not-the-published-content",
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "bare SHA must not match the content id",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                "7368242e65f8f907bae8002a609f966a25fcb32af6575b1baae6057c48e6566c",
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "wrong runtime id",
                requested,
                "ibm-granite/other",
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "wrong route",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                false,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "wrong adapter",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                "other-adapter",
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "wrong adapter family",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                "other-family",
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "wrong adapter architecture",
                requested,
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                "other-architecture",
            ),
            (
                "wrong catalog family",
                "other-granite:q8",
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "q4 must not inherit q8 compatibility",
                "granite-speech-4.1-2b:q4_k_m",
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            (
                "fp16 must not inherit q8 compatibility",
                "granite-speech-4.1-2b:f16",
                GRANITE_SPEECH_PUBLISHED_RUNTIME_MODEL_ID,
                GRANITE_SPEECH_PUBLISHED_CONTENT_ID,
                true,
                GRANITE_SPEECH_GGML_ADAPTER_ID,
                GRANITE_SPEECH_MODEL_FAMILY,
                GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
        ] {
            assert!(
                !granite_speech_published_identity_compatibility_matches(
                    requested,
                    runtime_id,
                    content_id,
                    route,
                    adapter_id,
                    model_family,
                    model_architecture,
                ),
                "{label} must reject"
            );
        }
    }

    #[test]
    fn native_content_cache_singleflights_concurrent_cold_misses() {
        let cache: Arc<Mutex<HashMap<String, Arc<OnceLock<Option<usize>>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let barrier = Arc::new(Barrier::new(8));
        let builds = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                let builds = Arc::clone(&builds);
                std::thread::spawn(move || {
                    barrier.wait();
                    native_content_cache_get_or_build(&cache, 4, "same-content".to_string(), || {
                        builds.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Some(42)
                    })
                })
            })
            .collect();

        for worker in workers {
            assert_eq!(worker.join().expect("cache worker"), Some(42));
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn native_content_cache_does_not_retain_failed_builds() {
        let cache = Mutex::new(HashMap::new());
        let builds = AtomicUsize::new(0);
        assert_eq!(
            native_content_cache_get_or_build(&cache, 4, "replaceable", || {
                builds.fetch_add(1, Ordering::SeqCst);
                None::<usize>
            }),
            None
        );
        assert_eq!(
            native_content_cache_get_or_build(&cache, 4, "replaceable", || {
                builds.fetch_add(1, Ordering::SeqCst);
                Some(7)
            }),
            Some(7)
        );
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    fn native_execution_services_for_test() -> Arc<NativeExecutionServices> {
        crate::models::native_execution_services::test_native_execution_services()
    }

    fn native_backend_for_test() -> NativeBackend {
        NativeBackend::new(native_execution_services_for_test())
    }

    fn native_executor_for_test() -> NativeBackendExecutor {
        NativeBackendExecutor::new(native_execution_services_for_test())
    }

    fn sample_wav_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist")
    }

    #[test]
    fn offline_round_trip_preserves_serve_batch_max_native_sessions() {
        // The server sets `serve_batch_max_native_sessions` from its per-model
        // admission width, then hands the request to `NativeBackendExecutor`,
        // which rebuilds it through
        // `native_offline_request_to_transcription_request`. If that rebuild
        // drops the field, `native_transcribe` reads a serial width of 1 and
        // serve-batch never engages on the server transcription path -- a gap
        // the real-pack parity lane misses because it drives the batch kernel
        // directly via a test hook, bypassing this offline round-trip.
        let pack =
            NativeAsrModelPackRef::new("moonshine-tiny", "moonshine", PathBuf::from("/tmp/pack"));
        let rebuilt = native_offline_request_to_transcription_request(
            &pack,
            ExecutionTarget::Auto,
            NativeAsrOfflineRequest::new(PathBuf::from("/tmp/audio.wav"))
                .with_serve_batch_max_native_sessions(Some(4)),
        );
        assert_eq!(rebuilt.serve_batch_max_native_sessions, Some(4));

        // No admission width still round-trips to `None`, leaving the
        // consumer's serial fallback (width 1) unchanged.
        let rebuilt_default = native_offline_request_to_transcription_request(
            &pack,
            ExecutionTarget::Auto,
            NativeAsrOfflineRequest::new(PathBuf::from("/tmp/audio.wav")),
        );
        assert_eq!(rebuilt_default.serve_batch_max_native_sessions, None);
    }

    fn write_mono_pcm16_wav(path: &Path, sample_rate_hz: u32, frames: u32) {
        let channels = 1_u16;
        let bits_per_sample = 16_u16;
        let bytes_per_sample = (bits_per_sample / 8) as u32;
        let data_size = frames * channels as u32 * bytes_per_sample;
        let mut bytes = Vec::with_capacity((44 + data_size) as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
        let byte_rate = sample_rate_hz * channels as u32 * bytes_per_sample;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = channels * (bits_per_sample / 8);
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for _ in 0..frames {
            bytes.extend_from_slice(&0_i16.to_le_bytes());
        }
        fs::write(path, bytes).expect("write short wav fixture");
    }

    fn whisper_tiny_context_wav_fixture(temp: &tempfile::TempDir) -> PathBuf {
        let path = temp.path().join("whisper-tiny-context.wav");
        // Tiny Whisper metadata fixtures advertise only 128 encoder positions
        // (2.56 s at the real frontend geometry). Keep these boundary tests
        // inside that truthful contract so they reach the tensor/tokenizer
        // condition each test is intended to assert.
        write_mono_pcm16_wav(&path, 16_000, 3_200);
        path
    }

    fn read_wav_mono_16k_pcm16(path: &Path) -> Result<Vec<i16>, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("could not read '{}': {error}", path.display()))?;
        if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(format!("'{}' is not a RIFF/WAVE file", path.display()));
        }

        let mut channels = None;
        let mut sample_rate = None;
        let mut bits_per_sample = None;
        let mut data = None;
        let mut i = 12;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            let start = i + 8;
            let end = start.saturating_add(size).min(bytes.len());
            if id == b"fmt " && size >= 16 && end <= bytes.len() {
                channels = Some(u16::from_le_bytes([bytes[start + 2], bytes[start + 3]]));
                sample_rate = Some(u32::from_le_bytes([
                    bytes[start + 4],
                    bytes[start + 5],
                    bytes[start + 6],
                    bytes[start + 7],
                ]));
                bits_per_sample = Some(u16::from_le_bytes([bytes[start + 14], bytes[start + 15]]));
            } else if id == b"data" && end <= bytes.len() {
                data = Some(&bytes[start..end]);
            }
            i += 8 + size + (size & 1);
        }

        if channels != Some(1) || sample_rate != Some(16_000) || bits_per_sample != Some(16) {
            return Err(format!(
                "'{}' must be 16 kHz mono PCM16 WAV (got channels={channels:?}, sample_rate={sample_rate:?}, bits={bits_per_sample:?})",
                path.display()
            ));
        }
        let data = data.ok_or_else(|| format!("'{}' has no data chunk", path.display()))?;
        Ok(data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn required_env_path(name: &str) -> PathBuf {
        let value = env::var(name).unwrap_or_else(|_| {
            panic!("{name} must point to a local file for this ignored smoke test")
        });
        let path = PathBuf::from(value);
        assert!(
            path.exists(),
            "{name} path does not exist: {}",
            path.display()
        );
        path
    }

    fn env_u64(name: &str, default: u64) -> u64 {
        env::var(name)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default)
    }

    fn env_f64(name: &str, default: f64) -> f64 {
        env::var(name)
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(default)
    }

    fn add_qwen_audio_layer_shapes(
        spec: TinyGgufFixtureSpec,
        layer_idx: usize,
    ) -> TinyGgufFixtureSpec {
        let prefix = format!("audio.blk.{layer_idx}.");
        spec.with_tensor_shape(format!("{prefix}attn_norm.weight"), [16_u64])
            .with_tensor_shape(format!("{prefix}attn_norm.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}attn_q.weight"), [16_u64, 16_u64])
            .with_tensor_shape(format!("{prefix}attn_q.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}attn_k.weight"), [16_u64, 16_u64])
            .with_tensor_shape(format!("{prefix}attn_k.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}attn_v.weight"), [16_u64, 16_u64])
            .with_tensor_shape(format!("{prefix}attn_v.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}attn_out.weight"), [16_u64, 16_u64])
            .with_tensor_shape(format!("{prefix}attn_out.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}ffn_norm.weight"), [16_u64])
            .with_tensor_shape(format!("{prefix}ffn_norm.bias"), [16_u64])
            .with_tensor_shape(format!("{prefix}ffn_up.weight"), [32_u64, 16_u64])
            .with_tensor_shape(format!("{prefix}ffn_up.bias"), [32_u64])
            .with_tensor_shape(format!("{prefix}ffn_down.weight"), [16_u64, 32_u64])
            .with_tensor_shape(format!("{prefix}ffn_down.bias"), [16_u64])
    }

    fn streaming_runtime_fixture_spec(
        family: &str,
        architecture: &str,
        frontend: &str,
        decode_policy: &str,
        tokenizer: &str,
    ) -> TinyGgufFixtureSpec {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            crate::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            crate::models::oasr_metadata::OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            family.to_string(),
        );
        metadata.insert(
            crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            architecture.to_string(),
        );
        // Decoder-state topology reads the model's native GGUF architecture,
        // just as the production pack contracts do. Keep the generic fixture
        // valid at both the OpenASR routing layer and the model-semantic layer.
        metadata.insert("general.architecture".to_string(), architecture.to_string());
        metadata.insert(
            crate::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            frontend.to_string(),
        );
        metadata.insert(
            crate::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            decode_policy.to_string(),
        );
        metadata.insert("openasr.tokenizer.id".to_string(), tokenizer.to_string());
        TinyGgufFixtureSpec::new(metadata)
    }

    fn whisper_streaming_runtime_fixture_spec(model_id: &str) -> TinyGgufFixtureSpec {
        TinyGgufFixtureSpec::whisper_oasr_v1_metadata_ready_for_streaming_fail_closed(model_id)
    }

    fn cohere_streaming_runtime_fixture_spec(model_id: &str) -> TinyGgufFixtureSpec {
        TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(model_id)
    }

    fn xasr_zipformer_streaming_runtime_fixture_spec(_model_id: &str) -> TinyGgufFixtureSpec {
        streaming_runtime_fixture_spec(
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            crate::XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
            crate::XASR_ZIPFORMER_DECODE_POLICY_ID,
            crate::XASR_ZIPFORMER_TOKENIZER_ID,
        )
    }

    #[derive(Clone, Copy)]
    struct StreamingRuntimeFixtureCase {
        slug: &'static str,
        adapter_id: &'static str,
    }

    fn streaming_runtime_fixture_cases() -> [StreamingRuntimeFixtureCase; 6] {
        [
            StreamingRuntimeFixtureCase {
                slug: "cohere",
                adapter_id: crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
            },
            StreamingRuntimeFixtureCase {
                slug: "moonshine",
                adapter_id: crate::MOONSHINE_GGML_ADAPTER_ID,
            },
            StreamingRuntimeFixtureCase {
                slug: "parakeet-ctc",
                adapter_id: crate::PARAKEET_CTC_GGML_ADAPTER_ID,
            },
            StreamingRuntimeFixtureCase {
                slug: "wav2vec2-ctc",
                adapter_id: crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
            },
            StreamingRuntimeFixtureCase {
                slug: "qwen",
                adapter_id: crate::QWEN3_ASR_GGML_ADAPTER_ID,
            },
            StreamingRuntimeFixtureCase {
                slug: "whisper",
                adapter_id: crate::WHISPER_GGML_ADAPTER_ID,
            },
        ]
    }

    struct TestNativeRuntimeAdapter {
        family: &'static str,
    }

    struct TestStreamingRuntimeAdapter {
        family: &'static str,
        supports_partials: bool,
        supports_timestamps: bool,
        expected_partial_results: bool,
        expected_word_timestamps: bool,
    }

    struct TestDelegatedStreamingSession {
        session_id: String,
        next_seq: u64,
    }

    impl NativeAsrModelAdapter for TestNativeRuntimeAdapter {
        fn adapter_id(&self) -> &'static str {
            "test-native-runtime-adapter"
        }

        fn model_family(&self) -> &'static str {
            self.family
        }

        fn capabilities(&self) -> NativeAsrCapabilities {
            NativeAsrCapabilities::native_offline()
        }

        fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
            model_pack.family == self.family
        }
    }

    impl NativeAsrModelAdapter for TestStreamingRuntimeAdapter {
        fn adapter_id(&self) -> &'static str {
            "test-streaming-runtime-adapter"
        }

        fn model_family(&self) -> &'static str {
            self.family
        }

        fn capabilities(&self) -> NativeAsrCapabilities {
            NativeAsrCapabilities::native_true_streaming()
                .with_partial_results(self.supports_partials)
                .with_timestamps(self.supports_timestamps)
        }

        fn supports_model_pack(&self, model_pack: &NativeAsrModelPackRef) -> bool {
            model_pack.family == self.family
        }

        fn start_streaming_session(
            &self,
            _execution_services: Arc<NativeExecutionServices>,
            _model_pack: &NativeAsrModelPackRef,
            _target: NativeAsrHardwareTarget,
            context: NativeAsrSessionContext,
            _options: NativeAsrRequestOptions,
            session_config: NativeAsrStreamingSessionConfig,
        ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
            assert_eq!(
                session_config.partial_results, self.expected_partial_results,
                "NativeBackendExecutor must gate requested partials before adapter dispatch"
            );
            assert_eq!(
                session_config.word_timestamps, self.expected_word_timestamps,
                "NativeBackendExecutor must gate requested word timestamps before adapter dispatch"
            );
            Ok(Box::new(TestDelegatedStreamingSession {
                session_id: context.session_id.0,
                next_seq: 1,
            }))
        }
    }

    impl NativeAsrSession for TestDelegatedStreamingSession {
        fn session_id(&self) -> &str {
            &self.session_id
        }

        fn push_audio(
            &mut self,
            frame: crate::realtime::RealtimeAudioFrame,
        ) -> Result<Vec<crate::realtime::RealtimeEventEnvelope>, NativeAsrError> {
            let event = crate::realtime::RealtimeEvent::Transcript(
                crate::realtime::RealtimeTranscriptEvent::Partial(
                    crate::realtime::RealtimeTranscriptPartial {
                        utterance_id: crate::realtime::TranscriptUtteranceId(
                            "utt_delegate_000001".to_string(),
                        ),
                        segment_id: crate::realtime::TranscriptSegmentId(
                            "seg_delegate_000001".to_string(),
                        ),
                        revision: frame.seq,
                        text: "adapter partial".to_string(),
                        start_ms: frame.start_ms,
                        end_ms: frame.end_ms(),
                        is_final: false,
                        words: Vec::new(),
                        language: None,
                        speaker: None,
                        speaker_label: None,
                        speaker_person_id: None,
                        speaker_snapshot_label: None,
                    },
                ),
            );
            let envelope = crate::realtime::RealtimeEventEnvelope {
                event_type: event.event_type(),
                session_id: crate::realtime::RealtimeSessionId(self.session_id.clone()),
                event_id: crate::realtime::RealtimeEventId(format!("evt_{:06}", self.next_seq)),
                seq: self.next_seq,
                created_at: "2026-06-04T00:00:00Z".to_string(),
                trace_id: None,
                request_id: None,
                event,
            };
            self.next_seq += 1;
            Ok(vec![envelope])
        }

        fn poll_events(
            &mut self,
        ) -> Result<Vec<crate::realtime::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }

        fn finish(
            &mut self,
        ) -> Result<Vec<crate::realtime::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }

        fn cancel(
            &mut self,
        ) -> Result<Vec<crate::realtime::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn service_scoped_unload_is_safe_before_any_dispatch_use() {
        // Constructing a service builds registry tables but not model weights;
        // unloading before the first request is therefore a safe no-op.
        native_execution_services_for_test().unload_idle_native_model_runtime_caches();
    }

    #[test]
    fn native_runtime_model_adapter_selects_descriptor_and_capabilities_from_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();

        assert_eq!(
            adapter.adapter_id(),
            crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        );
        assert_eq!(adapter.model_family(), "cohere-transcribe");
        let capabilities = adapter.capabilities();
        assert!(capabilities.is_native_adapter());
        assert!(capabilities.supports_phrase_bias);
        assert!(capabilities.supports_timestamps);
        // Family-level fact, read from the arch descriptor. Deliberately NOT
        // re-derived from per-pack metadata -- what a family's decode can do is
        // a property of the architecture, not of a pack's declarations.
        assert!(!capabilities.supports_in_decoder_speakers);
        assert!(capabilities.supports_quantized_models);
        assert!(capabilities.supports_hardware_acceleration);
        // Realtime cadence is registry-driven: cohere-transcribe registers a
        // streaming executor, so any of its packs advertises true streaming
        // regardless of pack metadata.
        assert!(capabilities.supports_true_streaming);
        assert_eq!(
            adapter.tensor_layout().unwrap().name,
            crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID
        );
    }

    /// The per-family segmentation source is what the runtime reads, and it
    /// comes from the arch descriptor rather than from anything a pack
    /// declares -- pinned here at both ends: the registry fact for every
    /// builtin family, and the capability a resolved pack reports for the ones
    /// a tiny fixture can build.
    #[test]
    fn in_decoder_speaker_capability_is_family_level_not_pack_declared() {
        // moss-transcribe-diarize is the only builtin family that carries
        // speaker structure in its own decode today; every other family takes
        // it from an external source (cohere's decoder has the mode but no
        // publishable pack -- see its arch descriptor).
        let registry = crate::arch::OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in registry.descriptors() {
            let expected =
                descriptor.identity.model_architecture == crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID;
            assert_eq!(
                descriptor
                    .execution_contract
                    .speaker_segmentation
                    .is_in_decoder(),
                expected,
                "'{}' speaker segmentation source mismatch",
                descriptor.identity.model_architecture
            );
        }

        let cases = [
            ("cohere-transcribe", "cohere"),
            ("whisper", "whisper"),
            ("qwen3-asr", "qwen"),
        ];
        let architecture_registry = OpenAsrArchitectureRegistry::with_builtins();
        for (family, name) in cases {
            let descriptor = architecture_registry
                .descriptors()
                .iter()
                .find(|descriptor| descriptor.identity.model_family == family)
                .unwrap_or_else(|| panic!("{name} family descriptor must be registered"))
                .ggml_family_adapter_descriptor();
            let adapter =
                NativeRuntimeModelAdapter::new(descriptor, &crate::GgufMetadata::default(), None);
            assert!(
                !adapter.segments_speakers_in_decoder(),
                "'{name}' takes its speaker structure from an external source"
            );
            assert_eq!(
                adapter.capabilities().supports_in_decoder_speakers,
                adapter.segments_speakers_in_decoder(),
                "'{name}' capability must mirror the descriptor"
            );
        }
    }

    /// moss-transcribe-diarize is the builtin family that carries speaker
    /// structure in its own decode. A fully verified pack -- crossing the same
    /// `PackVerifier` -> adapter-selection seam as an installed pack, and
    /// therefore the family's metadata+tensor runtime contract -- must report
    /// that capability from the descriptor, and nothing about the pack can
    /// flip it.
    #[test]
    fn moss_adapter_reports_in_decoder_speakers_from_a_verified_pack() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("moss-runtime.oasr");
        let spec = TinyGgufFixtureSpec::moss_td_oasr_v1_runtime_ready("moss-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("verified moss pack must resolve its native adapter");

        assert_eq!(adapter.adapter_id(), crate::arch::MOSS_TD_GGML_ADAPTER_ID);
        assert_eq!(adapter.model_family(), crate::arch::MOSS_TD_MODEL_FAMILY);
        let capabilities = adapter.capabilities();
        assert!(
            capabilities.supports_in_decoder_speakers,
            "moss capability must mirror its descriptor's InDecoder speaker source"
        );
        assert!(adapter.segments_speakers_in_decoder());
        assert!(
            !capabilities.supports_phrase_bias,
            "moss declares phrase bias Unsupported on its execution facet"
        );
        assert_eq!(
            adapter.tensor_layout().unwrap().name,
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID
        );
    }

    /// Voice ID always needs acoustic identity. Native speaker tracks remove
    /// only the external segmenter requirement; they never remove ReDim.
    #[test]
    fn voice_id_requires_redim_for_both_speaker_sources() {
        for (segments_in_decoder, embedder_available, external_pipeline_available, expected) in [
            (true, true, false, true),
            (true, false, false, false),
            (false, true, true, true),
            (false, false, true, false),
            (false, true, false, false),
        ] {
            let capability = native_diarization_capability(
                segments_in_decoder,
                embedder_available,
                external_pipeline_available,
            );
            assert_eq!(
                capability.supported, expected,
                "in_decoder={segments_in_decoder} embedder={embedder_available} external={external_pipeline_available}"
            );
            if !expected {
                assert!(
                    capability
                        .reason
                        .is_some_and(|reason| reason.contains("redimnet2-b6-cn"))
                );
            }
        }
    }

    #[test]
    fn native_streaming_rejects_voice_id_and_keeps_speakers_out_of_decode_options() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-redimnet-only-streaming.gguf");
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(
            "whisper-redimnet-only-streaming",
        )
        .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let adapter = NativeRuntimeModelAdapter::new(
            crate::arch::builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
            &crate::GgufMetadata::default(),
            None,
        );
        let realtime_capabilities =
            RealtimeBackendCapabilities::from_native_capabilities(&adapter.capabilities());
        assert!(adapter.capabilities().supports_true_streaming);
        assert!(!realtime_capabilities.diarization.supported);
        assert!(
            !adapter.segments_speakers_in_decoder(),
            "whisper takes its speaker structure from the external source"
        );

        let session_options = NativeAsrRequestOptions::new()
            .with_voice_id(true)
            .with_partial_results(true)
            .with_word_timestamps(true);
        let request_options =
            native_streaming_request_options_from_session_options(&session_options);

        assert!(
            !request_options.in_decoder_speakers,
            "the external speaker source must not also switch the decoder into in-decoder mode"
        );
        assert!(request_options.word_timestamps);

        let pack = NativeAsrModelPackRef::new(
            "whisper-redimnet-only-streaming",
            adapter.model_family(),
            runtime_path,
        );
        let error = match adapter.start_streaming_session(
            native_execution_services_for_test(),
            &pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_voice_id_rejected"),
            session_options,
            NativeAsrStreamingSessionConfig::default(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("realtime Voice ID must fail at the native adapter boundary"),
        };
        assert_eq!(error, NativeAsrError::VoiceIdUnsupportedForRealtime);
    }

    #[test]
    fn native_runtime_phrase_bias_capability_matrix_is_per_family() {
        let cases: [(&str, &str, bool); 7] = [
            ("whisper", crate::WHISPER_GGML_ARCHITECTURE_ID, true),
            (
                "cohere",
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                true,
            ),
            ("qwen", crate::QWEN3_ASR_GGML_ARCHITECTURE_ID, true),
            ("moonshine", crate::MOONSHINE_GGML_ARCHITECTURE_ID, true),
            (
                "parakeet-ctc",
                crate::PARAKEET_CTC_GGML_ARCHITECTURE_ID,
                true,
            ),
            (
                "wav2vec2-ctc",
                crate::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                true,
            ),
            (
                "xasr-zipformer",
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                false,
            ),
        ];
        let architecture_registry = OpenAsrArchitectureRegistry::with_builtins();

        for (slug, architecture, expected_phrase_bias) in cases {
            let descriptor = architecture_registry
                .descriptors()
                .iter()
                .find(|descriptor| descriptor.identity.model_architecture == architecture)
                .unwrap_or_else(|| panic!("{slug} descriptor must be registered"))
                .ggml_family_adapter_descriptor();
            let adapter =
                NativeRuntimeModelAdapter::new(descriptor, &crate::GgufMetadata::default(), None);
            let transcription = native_phrase_bias_capability_for_adapter(Some(&adapter));
            let realtime =
                RealtimeBackendCapabilities::from_native_capabilities(&adapter.capabilities());

            assert_eq!(
                adapter.capabilities().supports_phrase_bias,
                expected_phrase_bias,
                "{slug} adapter capability"
            );
            assert_eq!(
                transcription.supported, expected_phrase_bias,
                "{slug} transcription capability"
            );
            assert_eq!(
                realtime.phrase_bias.supported, expected_phrase_bias,
                "{slug} realtime capability"
            );
            assert_eq!(
                realtime.mode,
                crate::realtime::RealtimeBackendMode::TrueStreaming,
                "{slug} realtime mode"
            );
        }
    }

    #[test]
    fn native_executor_rejects_xasr_phrase_bias_before_offline_dispatch() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("xasr-zipformer.gguf");
        let spec = xasr_zipformer_streaming_runtime_fixture_spec("xasr-zipformer");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let adapter = NativeRuntimeModelAdapter::new(
            crate::arch::builtin_adapter_descriptor(
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            &crate::GgufMetadata::default(),
            None,
        );
        let model_pack = NativeAsrModelPackRef::new(
            "xasr-zipformer",
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            &runtime_path,
        );
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();

        let error = NativeAsrExecutor::transcribe(
            &native_executor_for_test(),
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrOfflineRequest::new(temp.path().join("missing-input.wav"))
                .with_options(NativeAsrRequestOptions::new().with_phrase_bias(Some(phrase_bias))),
        )
        .expect_err("xasr phrase bias must fail before offline dispatch");

        assert!(matches!(
            error,
            NativeAsrError::PhraseBiasUnsupportedByModel { .. }
        ));
        assert!(error.to_string().contains("xasr-zipformer"));
    }

    #[test]
    fn native_executor_rejects_xasr_phrase_bias_before_streaming_runtime_checkout() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("xasr-zipformer.gguf");
        let spec = xasr_zipformer_streaming_runtime_fixture_spec("xasr-zipformer");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let adapter = NativeRuntimeModelAdapter::new(
            crate::arch::builtin_adapter_descriptor(
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            &crate::GgufMetadata::default(),
            None,
        );
        let model_pack = NativeAsrModelPackRef::new(
            "xasr-zipformer",
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            &runtime_path,
        );
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();

        let error = match NativeAsrExecutor::start_streaming_session(
            &native_executor_for_test(),
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_xasr_hotword_reject"),
            NativeAsrRequestOptions::new().with_phrase_bias(Some(phrase_bias)),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        ) {
            Ok(_) => panic!("xasr phrase bias must fail before streaming runtime checkout"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            NativeAsrError::PhraseBiasUnsupportedByModel { .. }
        ));
        assert!(error.to_string().contains("xasr-zipformer"));
    }

    #[test]
    fn native_runtime_model_adapters_advertise_streaming_when_executor_is_registered() {
        let architecture_registry = OpenAsrArchitectureRegistry::with_builtins();
        for case in streaming_runtime_fixture_cases() {
            let descriptor = architecture_registry
                .find_by_adapter_id(case.adapter_id)
                .unwrap_or_else(|| panic!("{} adapter must be registered", case.slug))
                .ggml_family_adapter_descriptor();
            let adapter =
                NativeRuntimeModelAdapter::new(descriptor, &crate::GgufMetadata::default(), None);
            let capabilities = adapter.capabilities();
            assert_eq!(adapter.adapter_id(), case.adapter_id, "{}", case.slug);
            assert!(capabilities.supports_true_streaming, "{}", case.slug);
            assert!(capabilities.supports_partials, "{}", case.slug);

            let realtime = RealtimeBackendCapabilities::from_native_capabilities(&capabilities);
            assert_eq!(
                realtime.mode,
                crate::realtime::RealtimeBackendMode::TrueStreaming,
                "{}",
                case.slug
            );
            assert!(realtime.is_true_streaming, "{}", case.slug);
            assert!(realtime.supports_partial_results, "{}", case.slug);
        }
    }

    /// Public product-route smoke for one verifier-admitted Cohere pack:
    /// admission -> adapter selection -> native executor -> streaming
    /// dispatch. Other families' registry-derived streaming capabilities are
    /// covered above; their metadata-only placeholders are intentionally not
    /// treated as executable packs under fail-closed admission.
    #[test]
    fn native_backend_admits_and_dispatches_cohere_streaming_session() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-streaming-runtime.oasr");
        let spec = cohere_streaming_runtime_fixture_spec("cohere-streaming-runtime");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("complete Cohere fixture must pass package admission");
        let model_pack = NativeAsrModelPackRef::new(
            "cohere-streaming-runtime",
            "cohere-transcribe",
            &runtime_path,
        );
        let backend = native_executor_for_test();
        let session_id = "rt_cohere_backend_streaming";
        let mut session = NativeAsrExecutor::start_streaming_session(
            &backend,
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new(session_id),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        )
        .unwrap();

        assert_eq!(session.session_id(), session_id);
        let _ = session.poll_events().unwrap();

        let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(20).unwrap();
        // push_audio only buffers; the decode runs in poll_events once enough
        // audio passes the first-decode floor. Feed ~1.2s, then poll.
        for seq in 1..=60u64 {
            session
                .push_audio(
                    crate::realtime::RealtimeAudioFrame::new(
                        seq,
                        (seq - 1) * 20,
                        format,
                        vec![0; sample_count],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        // The tiny Cohere fixture is runnable enough to yield a partial; an
        // executor-level error is also acceptable as long as it came through
        // the declared streaming dispatch components.
        match session.poll_events() {
            Ok(events) => assert!(
                events
                    .iter()
                    .any(|event| event.event_type == "transcript.partial"),
                "expected a Cohere streaming partial"
            ),
            Err(error) => {
                let error = error.to_string();
                assert!(
                    error.contains("cohere-transcribe-ggml-snapshot-streaming-executor-v1"),
                    "{error}"
                );
                assert!(
                    error.contains("cohere-transcribe-ggml-executor-v1"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires OPENASR_NATIVE_STREAMING_SMOKE_PACK and OPENASR_NATIVE_STREAMING_SMOKE_WAV"]
    fn native_streaming_real_runtime_smoke_from_env() {
        let runtime_path = required_env_path("OPENASR_NATIVE_STREAMING_SMOKE_PACK");
        let wav_path = required_env_path("OPENASR_NATIVE_STREAMING_SMOKE_WAV");
        let max_ms = env::var("OPENASR_NATIVE_STREAMING_SMOKE_MAX_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5_000);
        let request_partials = env::var("OPENASR_NATIVE_STREAMING_SMOKE_PARTIALS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let poll_ms = env::var("OPENASR_NATIVE_STREAMING_SMOKE_POLL_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(200);
        let max_first_partial_end_ms = env_u64(
            "OPENASR_NATIVE_STREAMING_SMOKE_MAX_FIRST_PARTIAL_END_MS",
            1_200,
        );
        let max_first_partial_prefix_wer = env_f64(
            "OPENASR_NATIVE_STREAMING_SMOKE_MAX_FIRST_PARTIAL_PREFIX_WER",
            0.0,
        );
        let expected_final_text = env::var("OPENASR_NATIVE_STREAMING_SMOKE_EXPECTED_FINAL")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let samples = read_wav_mono_16k_pcm16(&wav_path).unwrap();
        assert!(!samples.is_empty(), "smoke WAV must contain audio samples");

        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("smoke runtime must be a valid native runtime pack");
        let capabilities = adapter.capabilities();
        assert!(
            capabilities.supports_true_streaming,
            "smoke runtime must declare true streaming and have a registered executor"
        );

        let model_id = runtime_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("native-streaming-smoke-runtime");
        let model_pack =
            NativeAsrModelPackRef::new(model_id, adapter.model_family(), &runtime_path);
        let backend = native_executor_for_test();
        let mut session = NativeAsrExecutor::start_streaming_session(
            &backend,
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Auto,
            NativeAsrSessionContext::new("rt_real_runtime_smoke"),
            NativeAsrRequestOptions::new().with_partial_results(request_partials),
            NativeAsrStreamingSessionConfig::new().with_partial_results(request_partials),
        )
        .expect("real runtime streaming session should start");

        let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
        let frame_duration_ms = 20_u64;
        let frame_sample_count = format.sample_count_for_duration_ms(20).unwrap();
        let partial_poll_every_frames = poll_ms.div_ceil(frame_duration_ms).max(1) as usize;
        let requested_samples = (max_ms as usize).saturating_mul(16).max(frame_sample_count);
        let max_samples = samples.len().min(requested_samples);
        let smoke_samples = &samples[..max_samples];
        let mut events = session.poll_events().unwrap();
        for (index, chunk) in smoke_samples.chunks(frame_sample_count).enumerate() {
            let mut frame_samples = chunk.to_vec();
            if frame_samples.len() < frame_sample_count {
                frame_samples.resize(frame_sample_count, 0);
            }
            let frame = crate::realtime::RealtimeAudioFrame::new(
                index as u64 + 1,
                index as u64 * 20,
                format,
                frame_samples,
            )
            .unwrap();
            events.extend(session.push_audio(frame).unwrap());
            if request_partials && (index + 1) % partial_poll_every_frames == 0 {
                events.extend(session.poll_events().unwrap());
            }
        }
        events.extend(session.finish().unwrap());

        let event_types = events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert!(event_types.contains(&"session.created"), "{event_types:?}");
        assert!(
            event_types.contains(&"session.configured"),
            "{event_types:?}"
        );
        assert!(
            event_types.contains(&"audio.input.started"),
            "{event_types:?}"
        );
        assert!(event_types.contains(&"transcript.final"), "{event_types:?}");
        if request_partials {
            assert!(
                event_types.contains(&"transcript.partial"),
                "{event_types:?}"
            );
        }
        assert!(
            event_types.contains(&"audio.input.stopped"),
            "{event_types:?}"
        );

        let final_event = events
            .iter()
            .find_map(|event| match &event.event {
                crate::RealtimeEvent::Transcript(crate::RealtimeTranscriptEvent::Final(final_)) => {
                    Some(final_)
                }
                _ => None,
            })
            .expect("real runtime smoke must emit a final transcript");
        assert!(final_event.is_final);
        assert!(final_event.revision >= 1);
        assert!(
            !final_event.text.trim().is_empty(),
            "real runtime smoke must emit non-empty text"
        );
        if let Some(expected) = expected_final_text.as_deref() {
            assert_eq!(
                crate::normalize_text(&final_event.text),
                crate::normalize_text(expected),
                "native streaming smoke final drifted"
            );
        }
        eprintln!(
            "native streaming smoke final text ({} ms): {}",
            max_ms,
            final_event.text.trim()
        );
        if request_partials {
            let partials = events
                .iter()
                .filter_map(|event| match &event.event {
                    crate::RealtimeEvent::Transcript(crate::RealtimeTranscriptEvent::Partial(
                        partial,
                    )) => Some(partial),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let first_partial = partials
                .first()
                .expect("partial-enabled real runtime smoke must emit a partial transcript");
            assert!(!first_partial.text.trim().is_empty());
            assert!(
                first_partial.end_ms <= max_first_partial_end_ms,
                "first partial ended at {}ms, above {}ms; text={:?}",
                first_partial.end_ms,
                max_first_partial_end_ms,
                first_partial.text
            );
            let prefix_reference = expected_final_text.as_deref().unwrap_or(&final_event.text);
            let first_partial_prefix_wer =
                crate::word_prefix_error_rate(&first_partial.text, prefix_reference)
                    .expect("first partial and final prefix must be non-empty");
            assert!(
                first_partial_prefix_wer <= max_first_partial_prefix_wer,
                "first partial prefix WER {first_partial_prefix_wer:.3} exceeded {max_first_partial_prefix_wer:.3}; first_partial={:?}; reference={:?}",
                first_partial.text,
                prefix_reference
            );
            eprintln!(
                "native streaming smoke partials: count={}, poll_ms={}, first_end_ms={}, first_prefix_wer={:.3}, first_text={}",
                partials.len(),
                poll_ms,
                first_partial.end_ms,
                first_partial_prefix_wer,
                first_partial.text.trim()
            );
        }
    }

    #[test]
    fn native_runtime_model_adapter_enforces_product_pack_shapes() {
        let temp = tempfile::tempdir().unwrap();
        let invalid_path = temp.path().join("not-a-runtime.oasr");
        fs::write(&invalid_path, b"not gguf").unwrap();

        assert!(native_runtime_model_adapter_for_path(&invalid_path).is_none());

        let realtime = native_runtime_realtime_capabilities_for_path(&invalid_path);
        assert_eq!(
            realtime.mode,
            crate::realtime::RealtimeBackendMode::Unsupported
        );
        assert!(!realtime.supports_realtime_sessions);

        // A structurally valid GGUF payload is still not a product runtime
        // package when it is exposed under the raw `.gguf` extension. The
        // product adapter seam accepts `.oasr` or the installed object shape.
        let valid_gguf_path = temp.path().join("valid-runtime.gguf");
        let valid_oasr_path = temp.path().join("valid-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("valid-runtime");
        write_tiny_gguf_runtime_source(&valid_gguf_path, &spec).unwrap();
        assert!(
            native_runtime_model_adapter_for_path(&valid_gguf_path).is_none(),
            "a valid raw GGUF must not cross the product native adapter ingress"
        );

        write_tiny_gguf_runtime_source(&valid_oasr_path, &spec).unwrap();
        assert!(
            native_runtime_model_adapter_for_path(&valid_oasr_path).is_some(),
            "the same valid payload is accepted once wrapped at the .oasr ingress"
        );

        let installed_object_path = temp
            .path()
            .join("objects/sha256")
            .join("0".repeat(64))
            .join("content");
        fs::create_dir_all(installed_object_path.parent().unwrap()).unwrap();
        write_tiny_gguf_runtime_source(&installed_object_path, &spec).unwrap();
        assert!(
            native_runtime_model_adapter_for_path(&installed_object_path).is_some(),
            "an installed content-addressed pack must remain executable without a .oasr suffix"
        );
    }

    #[test]
    fn native_runtime_model_adapter_does_not_cache_invalid_path_before_valid_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("replacement.oasr");
        fs::write(&runtime_path, b"not gguf").unwrap();

        assert!(native_runtime_model_adapter_for_path(&runtime_path).is_none());

        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("replacement-valid");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("a valid replacement must not inherit an invalid negative cache entry");
        assert_eq!(
            adapter.adapter_id(),
            crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        );
    }

    #[test]
    fn native_runtime_identity_cache_rekeys_same_path_content_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("identity-replacement.gguf");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("identity-before"),
        )
        .unwrap();
        let before = resolve_local_native_runtime_model_identity(&runtime_path, None).unwrap();
        assert_eq!(before.model_id, "identity-before");

        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("identity-after"),
        )
        .unwrap();
        let after = resolve_local_native_runtime_model_identity(&runtime_path, None).unwrap();
        assert_eq!(after.model_id, "identity-after");
    }

    #[test]
    fn native_runtime_model_adapter_cache_rekeys_same_path_content_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("dolphin-replacement.oasr");

        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::dolphin_oasr_v1_runtime_ready("dolphin-replacement-base"),
        )
        .unwrap();
        let base_adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("base Dolphin fixture must resolve");
        assert!(!base_adapter.capabilities().supports_phrase_bias);

        let hotword_spec =
            TinyGgufFixtureSpec::dolphin_oasr_v1_runtime_ready("dolphin-replacement-hotword")
                .with_added_tensor(
                crate::models::dolphin::hotword_context::CONTEXT_MODULE_WORD_EMBEDDING_TENSOR_NAME,
            );
        write_tiny_gguf_runtime_source(&runtime_path, &hotword_spec).unwrap();
        let hotword_adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("same-path Dolphin replacement must resolve");
        assert!(
            hotword_adapter.capabilities().supports_phrase_bias,
            "content-keyed cache must not reuse the base pack's negative phrase-bias result"
        );
    }

    #[test]
    fn native_backend_product_executor_reports_runtime_readiness() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = native_executor_for_test();
        let adapter = TestNativeRuntimeAdapter { family: "cohere" };
        let model_pack =
            NativeAsrModelPackRef::new("cohere-runtime-fixture", "cohere", runtime_path.clone());

        assert_eq!(
            NativeAsrExecutor::runtime_readiness(
                &backend,
                &adapter,
                &model_pack,
                NativeAsrHardwareTarget::Cpu
            ),
            NativeAsrRuntimeReadiness::Ready
        );

        let missing_pack = NativeAsrModelPackRef::new(
            "cohere-runtime-fixture",
            "cohere",
            temp.path().join("missing.oasr"),
        );
        assert!(matches!(
            NativeAsrExecutor::runtime_readiness(
                &backend,
                &adapter,
                &missing_pack,
                NativeAsrHardwareTarget::Cpu
            ),
            NativeAsrRuntimeReadiness::MissingLocalModelAsset { .. }
        ));

        assert!(matches!(
            NativeAsrExecutor::runtime_readiness(
                &backend,
                &adapter,
                &model_pack,
                NativeAsrHardwareTarget::IntelNpu
            ),
            NativeAsrRuntimeReadiness::UnsupportedHardwareTarget {
                target: NativeAsrHardwareTarget::IntelNpu
            }
        ));
    }

    #[test]
    fn native_hardware_target_mapping_preserves_generic_execution_targets() {
        assert_eq!(
            native_execution_target_from_hardware_target(NativeAsrHardwareTarget::Auto),
            Some(ExecutionTarget::Auto)
        );
        assert_eq!(
            native_execution_target_from_hardware_target(NativeAsrHardwareTarget::Accelerated),
            Some(ExecutionTarget::Accelerated)
        );
        assert_eq!(
            native_execution_target_from_hardware_target(NativeAsrHardwareTarget::Cpu),
            Some(ExecutionTarget::Cpu)
        );
        assert_eq!(
            native_execution_target_from_hardware_target(NativeAsrHardwareTarget::IntelNpu),
            None
        );
    }

    #[test]
    fn native_hardware_target_mapping_preserves_policy_constraints() {
        assert_eq!(
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::Auto).unwrap(),
            ExecutionIntent::Auto
        );
        assert_eq!(
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::Cpu).unwrap(),
            ExecutionIntent::CpuOnly
        );
        assert_eq!(
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::NvidiaCuda).unwrap(),
            ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                ExecutionProvider::Cuda
            ))
        );
        assert_eq!(
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::AmdGpu).unwrap(),
            ExecutionIntent::ConstrainedAcceleratedOnly(
                AcceleratedDeviceConstraint::HardwareVendor(ExecutionHardwareVendor::Amd)
            )
        );
        let apple_silicon =
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::AppleSilicon);
        if cfg!(all(target_vendor = "apple", target_arch = "aarch64")) {
            assert_eq!(
                apple_silicon.unwrap(),
                ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                    ExecutionProvider::Metal
                ))
            );
        } else {
            assert!(matches!(
                apple_silicon,
                Err(NativeAsrError::UnsupportedHardwareTarget {
                    target: NativeAsrHardwareTarget::AppleSilicon
                })
            ));
        }
        assert!(matches!(
            execution_intent_from_hardware_target(NativeAsrHardwareTarget::IntelNpu),
            Err(NativeAsrError::UnsupportedHardwareTarget {
                target: NativeAsrHardwareTarget::IntelNpu
            })
        ));
    }

    struct TestStreamingCandidateBuilder {
        services: Arc<NativeExecutionServices>,
        builds: Arc<Mutex<Vec<ExecutionProvider>>>,
        fail_build_on_accelerated: bool,
        fail_build_on_cpu_untyped: bool,
        fail_warmup_on_accelerated: bool,
        fail_push_on_accelerated: bool,
        auxiliary_initializations: Arc<AtomicUsize>,
    }

    impl NativeStreamingSessionCandidateBuilder for TestStreamingCandidateBuilder {
        fn execution_services(&self) -> Arc<NativeExecutionServices> {
            Arc::clone(&self.services)
        }

        fn initialize_auxiliary_runtimes(&self) -> Result<(), NativeAsrError> {
            self.auxiliary_initializations
                .fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn build(
            &self,
            candidate: &ExecutionCandidate,
        ) -> crate::models::native_execution_services::ExecutionCandidateAttemptOutcome<
            Box<dyn NativeAsrSession>,
            NativeAsrError,
        > {
            crate::models::native_execution_services::run_execution_candidate_attempt(
                self.services.as_ref(),
                candidate,
                || {
                    let provider = candidate.device.route.provider;
                    self.builds.lock().unwrap().push(provider);
                    let accelerated = provider != ExecutionProvider::Cpu;
                    if accelerated && self.fail_build_on_accelerated {
                        crate::models::native_execution_services::record_current_execution_candidate_failure(
                            crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                                "test-session-build",
                                "typed build failure",
                            ),
                        );
                        return Err(NativeAsrError::SessionFailed {
                            message: "opaque build error".to_string(),
                        });
                    }
                    if !accelerated && self.fail_build_on_cpu_untyped {
                        return Err(NativeAsrError::SessionFailed {
                            message: "opaque CPU build error".to_string(),
                        });
                    }
                    Ok(Box::new(TestPolicyNativeSession {
                        provider,
                        fail_warmup_on_accelerated: self.fail_warmup_on_accelerated,
                        fail_push_on_accelerated: self.fail_push_on_accelerated,
                    }) as Box<dyn NativeAsrSession>)
                },
            )
        }
    }

    struct TestPolicyNativeSession {
        provider: ExecutionProvider,
        fail_warmup_on_accelerated: bool,
        fail_push_on_accelerated: bool,
    }

    impl NativeAsrSession for TestPolicyNativeSession {
        fn session_id(&self) -> &str {
            "policy-streaming-test"
        }

        fn push_audio(
            &mut self,
            _frame: crate::RealtimeAudioFrame,
        ) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
            if self.provider != ExecutionProvider::Cpu && self.fail_push_on_accelerated {
                crate::models::native_execution_services::record_current_execution_candidate_failure(
                    crate::device::execution_policy::ExecutionCandidateFailure::device_lost(
                        "test-push",
                        "typed post-audio failure",
                    ),
                );
                return Err(NativeAsrError::SessionFailed {
                    message: "opaque push error".to_string(),
                });
            }
            Ok(Vec::new())
        }

        fn poll_events(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }

        fn warm_up(&mut self) -> Result<(), NativeAsrError> {
            if self.provider != ExecutionProvider::Cpu && self.fail_warmup_on_accelerated {
                crate::models::native_execution_services::record_current_execution_candidate_failure(
                    crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                        "test-warm-up",
                        "typed warm-up failure",
                    ),
                );
                return Err(NativeAsrError::SessionFailed {
                    message: "opaque warm-up error".to_string(),
                });
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }

        fn cancel(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, NativeAsrError> {
            Ok(Vec::new())
        }
    }

    fn streaming_policy_candidate(
        provider: ExecutionProvider,
        placement: ExecutionPlacement,
    ) -> ExecutionCandidate {
        let kind = if provider == ExecutionProvider::Cpu {
            crate::RouteDeviceKind::Cpu
        } else {
            crate::RouteDeviceKind::Accelerated
        };
        ExecutionCandidate {
            device: crate::device::execution_policy::ExecutionDeviceSnapshot {
                route: crate::ResolvedExecutionRoute {
                    provider,
                    stable_id: format!("{provider}-test"),
                    registry_ordinal: 0,
                    kind,
                    addressability: crate::DeviceAddressability::NotExactlyAddressable {
                        reason: "test candidate",
                    },
                },
                ggml_kind: if provider == ExecutionProvider::Cpu {
                    crate::GgmlBackendKind::Cpu
                } else {
                    crate::GgmlBackendKind::Gpu
                },
                memory: None,
                buffer_alignment: None,
            },
            placement,
        }
    }

    fn streaming_auto_test_plan() -> ExecutionPlan {
        ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                streaming_policy_candidate(ExecutionProvider::Vulkan, ExecutionPlacement::Hybrid),
                streaming_policy_candidate(ExecutionProvider::Cpu, ExecutionPlacement::CpuOnly),
            ],
        )
    }

    fn streaming_test_builder(
        fail_build: bool,
        fail_warmup: bool,
        fail_push: bool,
    ) -> (
        Arc<dyn NativeStreamingSessionCandidateBuilder>,
        Arc<Mutex<Vec<ExecutionProvider>>>,
        Arc<AtomicUsize>,
    ) {
        let builds = Arc::new(Mutex::new(Vec::new()));
        let auxiliary_initializations = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(TestStreamingCandidateBuilder {
                services: native_execution_services_for_test(),
                builds: Arc::clone(&builds),
                fail_build_on_accelerated: fail_build,
                fail_build_on_cpu_untyped: false,
                fail_warmup_on_accelerated: fail_warmup,
                fail_push_on_accelerated: fail_push,
                auxiliary_initializations: Arc::clone(&auxiliary_initializations),
            }),
            builds,
            auxiliary_initializations,
        )
    }

    #[test]
    fn streaming_session_acquisition_advances_only_on_typed_candidate_failure() {
        let (builder, builds, _) = streaming_test_builder(true, false, false);
        let session =
            PolicyResolvedNativeStreamingSession::start(builder, streaming_auto_test_plan())
                .expect("typed accelerated build failure should construct CPU candidate");
        assert_eq!(session.session_id(), "policy-streaming-test");
        assert_eq!(
            *builds.lock().unwrap(),
            vec![ExecutionProvider::Vulkan, ExecutionProvider::Cpu]
        );
    }

    #[test]
    fn streaming_warmup_can_replace_candidate_before_audio() {
        let (builder, builds, auxiliary_initializations) =
            streaming_test_builder(false, true, false);
        let mut session =
            PolicyResolvedNativeStreamingSession::start(builder, streaming_auto_test_plan())
                .unwrap();
        session
            .warm_up()
            .expect("typed warm-up failure should rebuild and warm CPU candidate");
        assert_eq!(
            *builds.lock().unwrap(),
            vec![ExecutionProvider::Vulkan, ExecutionProvider::Cpu]
        );
        assert_eq!(auxiliary_initializations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn streaming_never_retries_after_first_audio_enters_session() {
        let (builder, builds, _) = streaming_test_builder(false, false, true);
        let mut session =
            PolicyResolvedNativeStreamingSession::start(builder, streaming_auto_test_plan())
                .unwrap();
        let format = crate::RealtimeAudioFormat::pcm16_mono_16khz();
        let frame = crate::RealtimeAudioFrame::new(0, 0, format, vec![0; 320]).unwrap();
        let error = session
            .push_audio(frame)
            .expect_err("post-audio typed device failure must fail without replay");
        assert!(error.to_string().contains("opaque push error"));
        assert_eq!(*builds.lock().unwrap(), vec![ExecutionProvider::Vulkan]);
    }

    #[test]
    fn streaming_cancel_before_audio_never_initializes_auxiliary_models() {
        let (builder, _, auxiliary_initializations) = streaming_test_builder(false, false, false);
        let mut session =
            PolicyResolvedNativeStreamingSession::start(builder, streaming_auto_test_plan())
                .unwrap();

        session.cancel().unwrap();

        assert_eq!(auxiliary_initializations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn streaming_failed_replacement_becomes_terminal_without_panicking() {
        let builds = Arc::new(Mutex::new(Vec::new()));
        let builder: Arc<dyn NativeStreamingSessionCandidateBuilder> =
            Arc::new(TestStreamingCandidateBuilder {
                services: native_execution_services_for_test(),
                builds: Arc::clone(&builds),
                fail_build_on_accelerated: false,
                fail_build_on_cpu_untyped: true,
                fail_warmup_on_accelerated: true,
                fail_push_on_accelerated: false,
                auxiliary_initializations: Arc::new(AtomicUsize::new(0)),
            });
        let mut session =
            PolicyResolvedNativeStreamingSession::start(builder, streaming_auto_test_plan())
                .unwrap();
        let warmup_error = session
            .warm_up()
            .expect_err("CPU replacement construction is deliberately untyped and terminal");
        assert!(warmup_error.to_string().contains("opaque CPU build error"));
        assert_eq!(session.session_id(), "policy-streaming-test");
        let later_error = session
            .poll_events()
            .expect_err("a terminal wrapper must fail closed instead of panicking");
        assert_eq!(later_error, warmup_error);
        assert_eq!(
            *builds.lock().unwrap(),
            vec![ExecutionProvider::Vulkan, ExecutionProvider::Cpu]
        );
    }

    #[test]
    fn streaming_terminal_candidate_warmup_failure_stays_fail_closed() {
        let (builder, _, _) = streaming_test_builder(false, true, false);
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::AcceleratedOnly,
            vec![streaming_policy_candidate(
                ExecutionProvider::Vulkan,
                ExecutionPlacement::Hybrid,
            )],
        );
        let mut session = PolicyResolvedNativeStreamingSession::start(builder, plan).unwrap();
        let warmup_error = session
            .warm_up()
            .expect_err("the only candidate deliberately fails warm-up");
        let later_error = session
            .poll_events()
            .expect_err("a terminal wrapper must fail closed instead of panicking");
        assert_eq!(later_error, warmup_error);
    }

    #[test]
    fn native_offline_request_conversion_preserves_server_request_fields() {
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();
        let longform = crate::LongFormOptions {
            mode: crate::LongFormMode::Energy,
            ..crate::LongFormOptions::default()
        };
        let model_pack = NativeAsrModelPackRef::new(
            "qwen3-asr-0.6b:q8_0",
            crate::QWEN3_ASR_MODEL_FAMILY,
            "/tmp/openasr/qwen3-asr-0.6b-q8_0.gguf",
        );
        let request = NativeAsrOfflineRequest::new("/tmp/openasr/input.wav")
            .with_options(
                NativeAsrRequestOptions::new()
                    .with_language(Some("zh".to_string()))
                    .with_prompt(Some("domain prompt".to_string()))
                    .with_phrase_bias(Some(phrase_bias.clone()))
                    .with_inference_threads(Some(6))
                    .with_voice_id(true)
                    .with_word_timestamps(true),
            )
            .with_voice_id_segmenter(crate::config::VoiceIdSegmenterPreference::Segmentation3_0)
            .with_longform(Some(longform.clone()))
            .with_display_file_name(Some("meeting.wav".to_string()));

        let converted = native_offline_request_to_transcription_request(
            &model_pack,
            ExecutionTarget::Accelerated,
            request,
        );

        assert!(converted.input_path.ends_with("input.wav"));
        assert_eq!(
            converted.voice_id_segmenter,
            crate::config::VoiceIdSegmenterPreference::Segmentation3_0
        );
        assert_eq!(converted.model_id, "qwen3-asr-0.6b:q8_0");
        assert_eq!(
            converted.model_pack_path.as_deref(),
            Some(model_pack.root.as_path())
        );
        assert_eq!(converted.language.as_deref(), Some("zh"));
        assert_eq!(converted.prompt.as_deref(), Some("domain prompt"));
        assert_eq!(converted.phrase_bias, Some(phrase_bias));
        assert_eq!(converted.inference_threads, Some(6));
        assert_eq!(
            converted.execution_target,
            Some(ExecutionTarget::Accelerated)
        );
        assert!(converted.word_timestamps);
        assert!(converted.voice_id);
        assert_eq!(converted.longform, Some(longform));
        assert_eq!(converted.display_file_name.as_deref(), Some("meeting.wav"));
    }

    #[test]
    fn native_backend_product_executor_dispatches_offline_transcription() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-runtime.oasr");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
            let backend = native_executor_for_test();
            let adapter = native_runtime_model_adapter_for_path(&runtime_path)
                .expect("fixture must resolve through the production verifier");
            let model_pack = adapter
                .model_pack_ref("cohere-runtime-fixture")
                .expect("verified adapter must carry its exact pack proof");
            let request = NativeAsrOfflineRequest::new(sample_wav_fixture_path())
                .with_options(NativeAsrRequestOptions::new().with_word_timestamps(true));

            let transcription = NativeAsrExecutor::transcribe(
                &backend,
                &adapter,
                &model_pack,
                NativeAsrHardwareTarget::Cpu,
                request,
            )
            .unwrap();

            assert!(transcription.text.is_ascii() || !transcription.text.is_empty());
            assert!(!transcription.segments.is_empty());
        });
    }

    #[test]
    fn native_backend_product_executor_delegates_true_streaming_to_adapter() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = native_executor_for_test();
        let adapter = TestStreamingRuntimeAdapter {
            family: "cohere",
            supports_partials: true,
            supports_timestamps: true,
            expected_partial_results: true,
            expected_word_timestamps: true,
        };
        let model_pack =
            NativeAsrModelPackRef::new("cohere-runtime-fixture", "cohere", runtime_path);

        let mut session = NativeAsrExecutor::start_streaming_session(
            &backend,
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_product_streaming"),
            NativeAsrRequestOptions::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
            NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
        )
        .unwrap();

        assert_eq!(session.session_id(), "rt_native_product_streaming");
        let events = session
            .push_audio(
                crate::realtime::RealtimeAudioFrame::new(
                    1,
                    0,
                    crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz(),
                    vec![0; 320],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(events[0].event_type, "transcript.partial");
    }

    #[test]
    fn native_backend_product_executor_gates_adapter_streaming_options() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = native_executor_for_test();
        let adapter = TestStreamingRuntimeAdapter {
            family: "cohere",
            supports_partials: false,
            supports_timestamps: false,
            expected_partial_results: false,
            expected_word_timestamps: false,
        };
        let model_pack =
            NativeAsrModelPackRef::new("cohere-runtime-fixture", "cohere", runtime_path);

        let session = NativeAsrExecutor::start_streaming_session(
            &backend,
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_product_streaming_gated"),
            NativeAsrRequestOptions::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
            NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
        )
        .unwrap();

        assert_eq!(session.session_id(), "rt_native_product_streaming_gated");
    }

    #[test]
    fn native_backend_product_executor_keeps_streaming_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = native_executor_for_test();
        let adapter = TestNativeRuntimeAdapter { family: "cohere" };
        let model_pack =
            NativeAsrModelPackRef::new("cohere-runtime-fixture", "cohere", runtime_path);

        let error = match NativeAsrExecutor::start_streaming_session(
            &backend,
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_product"),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        ) {
            Ok(_) => panic!("native product executor must not pretend true streaming is available"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            NativeAsrError::BackendDoesNotSupportTrueStreaming { backend }
                if backend == "test-native-runtime-adapter"
        ));
    }

    #[test]
    fn native_runtime_model_adapter_routes_declared_true_streaming_to_ggml_dispatch() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture"),
        )
        .unwrap();
        let descriptor = crate::arch::builtin_adapter_descriptor(
            crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
        );
        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("valid Cohere fixture resolves a verified runtime adapter");
        let model_pack = NativeAsrModelPackRef::new(
            "cohere-runtime-fixture",
            descriptor.model_family,
            runtime_path,
        );

        let mut session = adapter
            .start_streaming_session(
                native_execution_services_for_test(),
                &model_pack,
                NativeAsrHardwareTarget::Cpu,
                NativeAsrSessionContext::new("rt_native_adapter_ggml_streaming"),
                NativeAsrRequestOptions::new()
                    .with_partial_results(true)
                    .with_word_timestamps(true),
                NativeAsrStreamingSessionConfig::new()
                    .with_partial_results(true)
                    .with_word_timestamps(true),
            )
            .expect("registered cohere streaming executor should create a session");
        let _ = session.poll_events().unwrap();
        let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(20).unwrap();
        // push_audio only buffers; the decode (which loads the fixture runtime and
        // fails) runs in poll_events once enough audio passes the first-decode
        // floor. Feed ~1.2s, then poll to surface the error.
        for seq in 1..=60u64 {
            session
                .push_audio(
                    crate::realtime::RealtimeAudioFrame::new(
                        seq,
                        (seq - 1) * 20,
                        format,
                        vec![0; sample_count],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let events = session.poll_events().unwrap();
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "transcript.partial"),
            "the runtime-ready Cohere fixture must decode through the registered streaming executor"
        );
    }

    #[test]
    fn native_streaming_start_consumes_the_adapter_preflight_proof_without_reparse() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-single-preflight.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-single-preflight"),
        )
        .unwrap();

        let before = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();
        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("valid Cohere fixture resolves a verified runtime adapter");
        let model_pack = NativeAsrModelPackRef::new(
            "cohere-single-preflight",
            adapter.model_family(),
            runtime_path,
        );
        let session = NativeAsrExecutor::start_streaming_session(
            &native_executor_for_test(),
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_single_preflight"),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        )
        .expect("streaming session construction consumes the attached proof");
        let after = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();
        drop(session);

        assert_eq!(
            after - before,
            1,
            "adapter resolution and streaming start must share one bounded GGUF parse"
        );
    }

    #[test]
    fn verified_identity_and_model_pack_ref_project_without_reparse() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-identity-single-preflight.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-identity-single-preflight"),
        )
        .unwrap();

        let before = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();
        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("valid Cohere fixture resolves a verified runtime adapter");
        let identity = adapter
            .verified_runtime_model_identity(None)
            .expect("identity projects from verifier metadata");
        let model_pack = adapter
            .model_pack_ref(identity.model_id)
            .expect("pack ref projects from the same proof");
        let after = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();

        assert_eq!(model_pack.root, runtime_path);
        assert_eq!(
            after - before,
            1,
            "adapter selection, admission identity, and execution pack ref must share one bounded GGUF parse"
        );
    }

    #[test]
    fn native_offline_start_consumes_the_model_pack_proof_without_reparse() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-offline-single-preflight.oasr");
        write_tiny_gguf_runtime_source(
            &runtime_path,
            &TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-offline-single-preflight"),
        )
        .unwrap();

        let before = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();
        let adapter = native_runtime_model_adapter_for_path(&runtime_path)
            .expect("valid Cohere fixture resolves a verified runtime adapter");
        let model_pack = adapter
            .model_pack_ref("cohere-offline-single-preflight")
            .expect("product adapter carries a verified model-pack proof");
        let _error = NativeAsrExecutor::transcribe(
            &native_executor_for_test(),
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrOfflineRequest::new(temp.path().join("missing-audio.wav")),
        )
        .expect_err("fixture has no audio input");
        let after = crate::ggml_runtime::bounded_parse_call_count_for_current_thread();

        assert_eq!(
            after - before,
            1,
            "adapter resolution and offline start must share one bounded GGUF parse"
        );
    }

    #[test]
    fn native_backend_requires_model_pack_path() {
        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-small");

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("requires an explicit local runtime pack path"));
        assert!(error.contains("fail-closed"));
    }

    #[test]
    fn native_backend_rejects_voice_id_when_shared_embedder_is_missing() {
        // Flattened into one multi-key override instead of nesting
        // `with_forced_cpu_backend_for_test` inside a second env guard: the
        // process env lock is not reentrant, so two nested guards on the same
        // thread would self-deadlock on the second `lock()` call.
        let temp = tempfile::tempdir().unwrap();
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_GGML_BACKEND", Some("cpu".into())),
                ("OPENASR_REDIMNET_PACK", None),
                ("OPENASR_PYANNOTE_PACK", None),
                ("OPENASR_HOME", Some(temp.path().as_os_str().to_os_string())),
            ],
            || {
                // Every Voice ID route needs the shared acoustic identity
                // space, so the runtime must stop before loading either the ASR
                // model or the external segmenter when ReDim is absent.
                let runtime_path = temp.path().join("whisper-runtime.oasr");
                let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(
                    "whisper-runtime-fixture",
                )
                .with_whisper_minimal_tokenizer();
                write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

                let backend = native_backend_for_test();
                let request =
                    TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                        .with_model_pack_path(Some(runtime_path))
                        .with_voice_id(true);

                let error = backend.transcribe(request).unwrap_err().to_string();

                assert!(error.contains("ReDimNet2-B6"), "{error}");
                assert!(error.contains("redimnet2-b6-cn"), "{error}");
            },
        );
    }

    #[test]
    fn pull_contract_validation_routes_diarize_packs_to_their_loader() {
        let temp = tempfile::tempdir().unwrap();
        // Diarization aux packs (ReDimNet2-B6 / pyannote) load a generic weight
        // bag at pull-time and only fail closed on missing tensors at forward.
        // What this gate must still prove: a pack declaring a diarization
        // architecture is routed through the aux table and is NEVER rejected by
        // ASR runtime adapter selection.
        let pack_path = temp.path().join("redimnet-stub.oasr");
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("redimnet2".to_string()),
        );
        metadata.insert(
            "openasr.package.version".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("1".to_string()),
        );
        let tensors = [crate::ggml_runtime::GgufWriteTensor {
            name: "stub.weight".to_string(),
            dims: vec![1],
            tensor_type: crate::ggml_runtime::GgufWriteTensorType::F32,
            data: vec![0u8; 4],
        }];
        crate::ggml_runtime::write_gguf_file_v0(&pack_path, &metadata, &tensors).unwrap();

        match verify_native_runtime_model_pack_path(&pack_path) {
            Ok(()) => {
                // Aux loader accepted the stub weight bag -- routing still succeeded.
            }
            Err(error) => {
                assert!(
                    error.contains("diarization pack validation failed"),
                    "got: {error}"
                );
                assert!(
                    !error.contains("runtime adapter selection failed"),
                    "got: {error}"
                );
            }
        }

        // Negative control: an unknown architecture still hits ASR selection.
        let unknown_path = temp.path().join("unknown-stub.oasr");
        let mut unknown_metadata = std::collections::BTreeMap::new();
        unknown_metadata.insert(
            "general.architecture".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("not-a-real-family".to_string()),
        );
        unknown_metadata.insert(
            "openasr.package.version".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("1".to_string()),
        );
        crate::ggml_runtime::write_gguf_file_v0(&unknown_path, &unknown_metadata, &tensors)
            .unwrap();
        let unknown_error = verify_native_runtime_model_pack_path(&unknown_path)
            .expect_err("unknown architecture must fail ASR adapter selection");
        assert!(
            unknown_error.contains("runtime adapter selection failed"),
            "got: {unknown_error}"
        );
    }

    /// Regression test for the install-time defect this fix closes: a real
    /// Qwen3-ForcedAligner-0.6B pack carries no `openasr.audio.frontend` /
    /// `openasr.decode.policy` (verified against the published pack's GGUF
    /// header), so before `qwen3-forced-aligner` was registered in
    /// `aux_pack_registry`, EVERY quant of this capability pack fell through
    /// to ASR family-adapter selection and failed `openasr pull` /
    /// `--word-timestamps=aligned` auto-install with
    /// `InvalidMetadata(MissingKey("openasr.audio.frontend"))` -- a defect
    /// present since the family shipped, unrelated to any quantization
    /// incident. Twin of
    /// `pull_contract_validation_routes_diarize_packs_to_their_loader` above,
    /// but through the full public entry point (a temporary GGUF writer -> a
    /// real `.oasr` file on disk -> `verify_native_runtime_model_pack_path`)
    /// rather than the internal aux-table function directly, and asserting
    /// both directions: complete metadata is accepted, and a bare-minimum
    /// pack is claimed by the aux table (never ASR selection) yet still
    /// rejected.
    #[test]
    fn pull_contract_validation_enforces_forced_aligner_mixed_floor_and_metadata() {
        let temp = tempfile::tempdir().unwrap();

        let mut complete_metadata = std::collections::BTreeMap::new();
        complete_metadata.insert(
            "general.architecture".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("qwen3-forced-aligner".to_string()),
        );
        complete_metadata.insert(
            "openasr.package.version".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("1".to_string()),
        );
        complete_metadata.insert(
            "openasr.model.id".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("qwen3-forced-aligner-0.6b".to_string()),
        );
        for key in [
            "qwen3_forced_aligner.audio.sample_rate_hz",
            "qwen3_forced_aligner.audio.n_mels",
            "qwen3_forced_aligner.audio.n_fft",
            "qwen3_forced_aligner.audio.win_length",
            "qwen3_forced_aligner.audio.hop_length",
            "qwen3_forced_aligner.audio.n_layers",
            "qwen3_forced_aligner.audio.d_model",
            "qwen3_forced_aligner.audio.n_heads",
            "qwen3_forced_aligner.llm.n_layers",
            "qwen3_forced_aligner.llm.d_model",
            "qwen3_forced_aligner.llm.n_heads",
            "qwen3_forced_aligner.llm.n_kv_heads",
            "qwen3_forced_aligner.llm.head_dim",
            "qwen3_forced_aligner.llm.embed_vocab_size",
            "qwen3_forced_aligner.llm.classify_num",
            "qwen3_forced_aligner.llm.max_positions",
            "qwen3_forced_aligner.audio_start_token_id",
            "qwen3_forced_aligner.audio_end_token_id",
            "qwen3_forced_aligner.audio_pad_token_id",
            "qwen3_forced_aligner.timestamp_token_id",
            "qwen3_forced_aligner.timestamp_segment_time_ms",
        ] {
            complete_metadata.insert(key.to_string(), crate::ggml_runtime::GgufWriteValue::U32(1));
        }
        complete_metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            crate::ggml_runtime::GgufWriteValue::StringArray(vec!["<pad>".to_string()]),
        );
        complete_metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            crate::ggml_runtime::GgufWriteValue::StringArray(Vec::new()),
        );
        let tensors = [crate::ggml_runtime::GgufWriteTensor {
            name: "blk.0.attn_q.weight".to_string(),
            dims: vec![32, 32],
            tensor_type: crate::ggml_runtime::GgufWriteTensorType::F16,
            data: vec![0u8; 32 * 32 * 2],
        }];
        let complete_path = temp.path().join("forced-aligner-complete.oasr");
        crate::ggml_runtime::write_gguf_file_v0(&complete_path, &complete_metadata, &tensors)
            .unwrap();
        assert!(
            verify_native_runtime_model_pack_path(&complete_path).is_ok(),
            "a pack carrying every qwen3_forced_aligner.* key plus the BPE tokenizer arrays \
             must pass install-time validation"
        );

        let q4_values = vec![0.0f32; 256 * 256];
        let q4_tensors = [crate::ggml_runtime::GgufWriteTensor {
            name: "audio.blk.0.attn_q.weight".to_string(),
            dims: vec![256, 256],
            tensor_type: crate::ggml_runtime::GgufWriteTensorType::Q4_K,
            data: crate::ggml_runtime::quantize_f32_to_ggml_tensor_data(
                crate::ggml_runtime::GgufWriteTensorType::Q4_K,
                &[256, 256],
                &q4_values,
            )
            .expect("quantize forced-aligner Q4 fixture"),
        }];
        let q4_path = temp.path().join("forced-aligner-q4.oasr");
        crate::ggml_runtime::write_gguf_file_v0(&q4_path, &complete_metadata, &q4_tensors).unwrap();
        let q4_error = verify_native_runtime_model_pack_path(&q4_path)
            .expect_err("a Q4 forced-aligner audio matrix must fail the public pack verifier");
        assert!(q4_error.contains("Q8_0"), "got: {q4_error}");

        crate::test_process_env::with_test_process_env(
            [(
                "OPENASR_FORCED_ALIGNER_PACK",
                Some(q4_path.into_os_string()),
            )],
            || {
                assert!(
                    !crate::word_timestamp_forced_aligner_available(),
                    "an installed legacy Q4 pack must not suppress Q8_0 replacement"
                );
            },
        );
        crate::test_process_env::with_test_process_env(
            [(
                "OPENASR_FORCED_ALIGNER_PACK",
                Some(complete_path.clone().into_os_string()),
            )],
            || {
                assert!(
                    crate::word_timestamp_forced_aligner_available(),
                    "an FP16 matrix satisfies the Q8 floor"
                );
            },
        );

        let mut bare_metadata = std::collections::BTreeMap::new();
        bare_metadata.insert(
            "general.architecture".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("qwen3-forced-aligner".to_string()),
        );
        bare_metadata.insert(
            "openasr.package.version".to_string(),
            crate::ggml_runtime::GgufWriteValue::String("1".to_string()),
        );
        let bare_path = temp.path().join("forced-aligner-bare.oasr");
        crate::ggml_runtime::write_gguf_file_v0(&bare_path, &bare_metadata, &tensors).unwrap();
        let bare_error = verify_native_runtime_model_pack_path(&bare_path)
            .expect_err("a pack missing every qwen3_forced_aligner.* key must fail closed");
        assert!(
            bare_error.contains("forced-alignment pack validation failed"),
            "got: {bare_error}"
        );
        assert!(
            !bare_error.contains("runtime adapter selection failed"),
            "a forced-aligner pack must never be diverted into ASR adapter selection: got: {bare_error}"
        );
    }

    /// `verify_native_runtime_model_pack_path` routes through the shared
    /// PackVerifier and registry-owned family contracts. This coverage test
    /// must never silently miss an architecture that *should* have fail-closed
    /// validation wired up.
    ///
    /// Mirrors the sibling completeness tests in `builtin_execution_dispatch`
    /// (`builtins_cover_all_dedicated_runtime_architectures` /
    /// `..._native_graph_lowering_architectures`), but black-box: rather than
    /// checking a lookup table contains a key, it feeds every builtin
    /// architecture a pack that carries only the bare adapter-selection
    /// metadata (family/architecture/audio-frontend/decode-policy) and
    /// nothing else, and asserts install-time validation rejects it. Every
    /// architecture wired up today (via a dedicated `runtime_contract`
    /// parser, or via the qwen3/cohere tensor-contract check, or via the
    /// aux-pack table) fails closed on such a bare-bones pack, so if a future
    /// architecture is missing from the registry-owned contract projection,
    /// it would fall through to `Ok(())` here and this test would catch it.
    #[test]
    fn install_time_family_metadata_validation_covers_every_builtin_architecture() {
        use crate::arch::OpenAsrArchitectureRegistry;
        use crate::models::oasr_metadata::{
            OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
            OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
            OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
        };
        use std::collections::BTreeMap;

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let mut metadata = BTreeMap::new();
            metadata.insert(
                OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
                OASR_PACKAGE_VERSION_V1.to_string(),
            );
            metadata.insert(
                OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
                descriptor.identity.model_family.to_string(),
            );
            metadata.insert(
                OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
                descriptor.identity.model_architecture.to_string(),
            );
            metadata.insert(
                OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
                descriptor.pack_contract.audio_frontend_id.to_string(),
            );
            metadata.insert(
                OASR_METADATA_KEY_DECODE_POLICY.to_string(),
                descriptor
                    .topology_contract
                    .decode_driver
                    .decode_policy_id()
                    .to_string(),
            );
            let spec = TinyGgufFixtureSpec::new(metadata);
            let temp = tempfile::tempdir().unwrap();
            let pack_path = temp.path().join("fixture.oasr");
            write_tiny_gguf_runtime_source(&pack_path, &spec).unwrap();

            let result = verify_native_runtime_model_pack_path(&pack_path);
            assert!(
                result.is_err(),
                "{} accepted an install-time pack that carries only bare \
                 adapter-selection metadata; every builtin architecture must fail \
                 closed on missing family-specific runtime metadata (a silent \
                 `_ => Ok(())` dispatch arm would let this through)",
                descriptor.identity.model_architecture,
            );
        }
    }

    #[test]
    fn native_backend_rejects_speakers_hint_without_diarize() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-runtime.oasr");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let backend = native_backend_for_test();
            let request =
                TranscriptionRequest::new(sample_wav_fixture_path(), "cohere-runtime-fixture")
                    .with_model_pack_path(Some(runtime_path))
                    .with_diarize_speakers(Some(2));

            let error = backend.transcribe(request).unwrap_err().to_string();

            assert!(error.contains("speakers hint requires diarize=true"));
        });
    }

    #[test]
    fn native_runtime_capabilities_reject_voice_id_for_an_external_family_with_no_embedder() {
        let temp = tempfile::tempdir().unwrap();
        // Hermetic: the capability probe also consults the host's installed
        // ReDimNet2-B6 pack, so pin the lookup to an empty home.
        let _env = crate::test_process_env::TestProcessEnvGuard::new([
            ("OPENASR_REDIMNET_PACK", None),
            ("OPENASR_HOME", Some(temp.path().as_os_str().to_os_string())),
        ]);
        let runtime_path = temp.path().join("whisper-runtime.oasr");
        let spec = whisper_streaming_runtime_fixture_spec("whisper-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let capabilities = native_runtime_transcription_capabilities_for_path(&runtime_path);

        assert!(!capabilities.diarization.supported);
        assert!(
            capabilities
                .diarization
                .reason
                .is_some_and(|reason| reason.contains("speaker-embedder pack")
                    && reason.contains("redimnet2-b6-cn"))
        );
    }

    #[test]
    fn native_runtime_capabilities_require_embedder_and_segmenter_for_external_voice_id() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-runtime.oasr");
        let spec = whisper_streaming_runtime_fixture_spec("whisper-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let redimnet_pack = temp.path().join("redimnet.oasr");
        let segmenter_pack = temp.path().join("segmenter.oasr");
        std::fs::write(&redimnet_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        std::fs::write(&segmenter_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        let capabilities = crate::test_process_env::with_test_process_env(
            [
                (
                    "OPENASR_REDIMNET_PACK",
                    Some(redimnet_pack.into_os_string()),
                ),
                (
                    "OPENASR_PYANNOTE_PACK",
                    Some(segmenter_pack.into_os_string()),
                ),
            ],
            || native_runtime_transcription_capabilities_for_path(&runtime_path),
        );

        // The VAD + ReDimNet2-B6 path is model-agnostic: a family that takes
        // its speaker structure from an external source reports Voice ID
        // supported only once both external pipeline packs are installed.
        assert!(capabilities.diarization.supported);
    }

    #[test]
    fn native_runtime_realtime_capabilities_are_runtime_owned_and_conservative() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let capabilities = native_runtime_realtime_capabilities_for_path(&runtime_path);

        // Realtime capability is owned by the streaming-executor registry, not the
        // pack: cohere-transcribe registers a (buffered) streaming executor, so any
        // of its packs advertises true streaming with partials and no VAD-boundary
        // requirement -- regardless of pack metadata.
        assert_eq!(
            capabilities.mode,
            crate::realtime::RealtimeBackendMode::TrueStreaming
        );
        assert!(capabilities.supports_realtime_sessions);
        assert!(capabilities.phrase_bias.supported);
        assert!(capabilities.supports_partial_results);
        assert!(!capabilities.requires_vad_utterance_boundaries);
        // Buffered granularity (re-decode), not frame-sync.
        assert!(!capabilities.frame_sync_partials);
    }

    #[test]
    fn native_backend_does_not_reject_phrase_bias_before_runtime_dispatch() {
        let backend = native_backend_for_test();
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 3.0)]).unwrap();
        let request = TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-small")
            .with_phrase_bias(Some(phrase_bias));

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("requires an explicit local runtime pack path"));
        assert!(!error.contains("silently ignoring phrase_bias"));
    }

    #[test]
    fn dolphin_phrase_bias_probe_reports_true_only_when_context_module_tensor_is_baked() {
        let dolphin_descriptor =
            crate::arch::builtin_adapter_descriptor(crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID);
        let temp = tempfile::tempdir().unwrap();

        // Base-tier pack: no `context_module.*` weights baked -- must not
        // report phrase-bias support (this used to be a family-wide `true`
        // that let requests reach `hotword_context.rs` and hard-fail there).
        let base_path = temp.path().join("dolphin-base.gguf");
        write_tiny_gguf_runtime_source(&base_path, &TinyGgufFixtureSpec::new(Default::default()))
            .unwrap();
        let base_tensor_index = crate::read_gguf_tensor_index(&base_path).unwrap();
        assert!(
            !native_runtime_descriptor_supports_phrase_bias(
                &dolphin_descriptor,
                Some(&base_tensor_index),
            ),
            "a pack without the context-module tensor must not advertise phrase bias"
        );
        // No tensor index at all (best-effort read failure) must also fail closed.
        assert!(!native_runtime_descriptor_supports_phrase_bias(
            &dolphin_descriptor,
            None,
        ));

        // Hotword-tier pack: the deep-biasing context module tensor is baked.
        let hotword_path = temp.path().join("dolphin-cn-dialect-small.gguf");
        let hotword_spec = TinyGgufFixtureSpec::new(Default::default()).with_added_tensor(
            crate::models::dolphin::hotword_context::CONTEXT_MODULE_WORD_EMBEDDING_TENSOR_NAME,
        );
        write_tiny_gguf_runtime_source(&hotword_path, &hotword_spec).unwrap();
        let hotword_tensor_index = crate::read_gguf_tensor_index(&hotword_path).unwrap();
        assert!(
            native_runtime_descriptor_supports_phrase_bias(
                &dolphin_descriptor,
                Some(&hotword_tensor_index),
            ),
            "a pack with the baked context-module tensor must advertise phrase bias"
        );

        // Every non-Dolphin architecture keeps the prior architecture-level
        // answer regardless of the (irrelevant) tensor index passed in.
        let whisper_descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .find(|descriptor| {
                descriptor.identity.model_architecture == crate::WHISPER_GGML_ARCHITECTURE_ID
            })
            .expect("whisper descriptor registered")
            .ggml_family_adapter_descriptor();
        assert!(native_runtime_descriptor_supports_phrase_bias(
            &whisper_descriptor,
            Some(&base_tensor_index),
        ));
    }

    #[test]
    fn dolphin_adapter_builder_still_probes_tensor_index_end_to_end() {
        // Regression test for the tensor-index read gating in
        // The shared preflight carries both metadata and tensor index, so this
        // exercises the full `native_runtime_model_adapter_for_path` ->
        // architecture-registry selection -> Dolphin tensor-index probe path end to
        // end (not just the already-covered
        // `native_runtime_descriptor_supports_phrase_bias` unit above).
        let temp = tempfile::tempdir().unwrap();

        let base_path = temp.path().join("dolphin-base-e2e.oasr");
        write_tiny_gguf_runtime_source(
            &base_path,
            &TinyGgufFixtureSpec::dolphin_oasr_v1_runtime_ready("dolphin-base-e2e"),
        )
        .unwrap();
        let base_adapter = native_runtime_model_adapter_for_path(&base_path).unwrap();
        assert_eq!(
            base_adapter.descriptor.model_architecture,
            crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID
        );
        assert!(
            !base_adapter.capabilities().supports_phrase_bias,
            "a Dolphin pack without the context-module tensor must not advertise phrase bias"
        );

        let hotword_path = temp.path().join("dolphin-hotword-e2e.oasr");
        let hotword_spec =
            TinyGgufFixtureSpec::dolphin_oasr_v1_runtime_ready("dolphin-hotword-e2e")
                .with_added_tensor(
                crate::models::dolphin::hotword_context::CONTEXT_MODULE_WORD_EMBEDDING_TENSOR_NAME,
            );
        write_tiny_gguf_runtime_source(&hotword_path, &hotword_spec).unwrap();
        let hotword_adapter = native_runtime_model_adapter_for_path(&hotword_path).unwrap();
        assert!(
            hotword_adapter.capabilities().supports_phrase_bias,
            "a Dolphin pack with the baked context-module tensor must advertise phrase bias"
        );

        // Non-Dolphin architectures never consult the tensor index; the
        // Cohere/whisper full-path tests elsewhere in this module already
        // cover their `supports_phrase_bias` derivation running through this
        // same builder with the read now skipped.
    }

    #[test]
    fn native_model_pack_path_rejects_remote_url() {
        let error = validate_local_native_model_pack_path(Path::new(
            "https://example.invalid/whisper-small.oasr",
        ))
        .unwrap_err()
        .to_string();

        assert!(error.contains("remote URL is not supported"));
    }

    #[test]
    fn native_model_pack_path_rejects_missing_path() {
        let error =
            validate_local_native_model_pack_path(Path::new("this-pack-should-not-exist.oasr"))
                .unwrap_err()
                .to_string();

        assert!(error.contains("path does not exist"));
    }

    #[test]
    fn native_model_pack_path_rejects_local_file_path() {
        let temp = tempfile::tempdir().unwrap();
        let file_path = temp.path().join("not-a-pack.oasr");
        std::fs::write(&file_path, b"not a directory").unwrap();

        let error = validate_local_native_model_pack_path(&file_path)
            .unwrap_err()
            .to_string();

        assert!(error.contains("Expected a local GGUF-backed OpenASR runtime package"));
    }

    #[test]
    fn native_model_pack_path_accepts_oasr_single_file_path() {
        let temp = tempfile::tempdir().unwrap();
        let pack_file = temp.path().join("fixture-pack.oasr");
        std::fs::write(&pack_file, b"GGUFpayload").unwrap();

        let validated = validate_local_native_model_pack_path(&pack_file).unwrap();
        assert_eq!(validated, pack_file);
    }

    #[test]
    fn native_model_pack_path_rejects_directory_even_with_openasr_suffix() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("not-a-pack.oasr");
        std::fs::create_dir_all(&directory).unwrap();

        let error = validate_local_native_model_pack_path(&directory)
            .unwrap_err()
            .to_string();

        assert!(error.contains("must be a regular file"));
    }

    #[test]
    fn native_model_pack_path_rejects_raw_gguf_file_path() {
        let temp = tempfile::tempdir().unwrap();
        let pack_file = temp.path().join("valid-pack.gguf");
        std::fs::write(&pack_file, b"GGUFpayload").unwrap();

        let error = validate_local_native_model_pack_path(&pack_file)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("expected a .oasr file or an installed content-addressed pack object"),
            "{error}"
        );
    }

    #[test]
    fn native_model_pack_path_accepts_installed_content_addressed_object() {
        let temp = tempfile::tempdir().unwrap();
        let object = temp
            .path()
            .join("objects/sha256")
            .join("f".repeat(64))
            .join("content");
        std::fs::create_dir_all(object.parent().unwrap()).unwrap();
        std::fs::write(&object, b"GGUFpayload").unwrap();

        let validated = validate_local_native_model_pack_path(&object).unwrap();
        assert_eq!(validated, object);
    }

    #[test]
    fn native_backend_fails_closed_when_gguf_oasr_metadata_is_incomplete() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-incomplete.oasr");
        let fixture_spec = TinyGgufFixtureSpec::new(
            [
                ("openasr.model.id", "whisper-runtime-fixture"),
                ("openasr.package.version", "1"),
                ("openasr.model.family", "whisper"),
                ("openasr.model.architecture", "whisper-encoder-decoder"),
                ("openasr.audio.frontend", "whisper.logmel.16khz.mono.v0"),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("runtime adapter selection failed"),
            "{error}"
        );
        assert!(error.contains("openasr.decode.policy"), "{error}");
    }

    #[test]
    fn native_backend_fails_closed_when_gguf_metadata_has_no_registered_family_adapter() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("unknown-family.oasr");
        let fixture_spec = TinyGgufFixtureSpec::new(
            [
                ("openasr.model.id", "unknown-family-fixture"),
                ("openasr.package.version", "1"),
                ("openasr.model.family", "unknown-family"),
                ("openasr.model.architecture", "unknown-arch"),
                ("openasr.audio.frontend", "unknown.frontend.v0"),
                ("openasr.decode.policy", "unknown.decode.v0"),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "unknown-family-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("runtime adapter selection failed"),
            "{error}"
        );
        assert!(error.contains("UnknownFamily"), "{error}");
    }

    #[test]
    fn native_backend_rejects_qwen_pack_missing_audio_stem_tensor_before_execution() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.oasr");
        let fixture_spec =
            TinyGgufFixtureSpec::qwen3_asr_oasr_v1_metadata_ready_for_runtime_fail_closed(
                "qwen3-asr-0.6b-q4_k",
            )
            .with_metadata("qwen3-asr.n_mels", "80")
            .with_metadata("qwen3-asr.llm.max_pos", "256")
            .with_tensor_shape("audio.mel_filters", [80_u64, 201_u64])
            .with_tensor_shape("audio.mel_window", [400_u64])
            .with_tensor_shape("audio.conv_out.weight", [3_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_norm.weight", [16_u64])
            .with_tensor_shape("blk.0.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_output.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.0.attn_q_norm.weight", [8_u64])
            .with_tensor_shape("blk.0.attn_k_norm.weight", [8_u64])
            .with_tensor_shape("blk.0.ffn_norm.weight", [16_u64])
            .with_tensor_shape("blk.0.ffn_gate.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.0.ffn_up.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.0.ffn_down.weight", [16_u64, 32_u64])
            .with_tensor_shape("blk.1.attn_norm.weight", [16_u64])
            .with_tensor_shape("blk.1.attn_q.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_k.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_v.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_output.weight", [16_u64, 16_u64])
            .with_tensor_shape("blk.1.attn_q_norm.weight", [8_u64])
            .with_tensor_shape("blk.1.attn_k_norm.weight", [8_u64])
            .with_tensor_shape("blk.1.ffn_norm.weight", [16_u64])
            .with_tensor_shape("blk.1.ffn_gate.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.1.ffn_up.weight", [32_u64, 16_u64])
            .with_tensor_shape("blk.1.ffn_down.weight", [16_u64, 32_u64])
            .with_tensor_shape("token_embd.weight", [16_u64, 32_u64])
            .with_tensor_shape("output.weight", [16_u64, 32_u64])
            .with_tensor_shape("output_norm.weight", [16_u64]);
        let fixture_spec =
            add_qwen_audio_layer_shapes(add_qwen_audio_layer_shapes(fixture_spec, 0), 1);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(sample_wav_fixture_path(), "qwen3-asr-0.6b-q4_k")
            .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();
        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("qwen3-asr missing required GGUF tensor"),
            "{error}"
        );
        assert!(error.contains("audio.conv.1.weight"), "{error}");
        assert!(!error.contains("qwen3-asr.ggml-executor.v1"), "{error}");
    }

    #[test]
    fn native_backend_routes_a_complete_cohere_pack_to_its_executor() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-transcribe-q4_k.oasr");
            let fixture_spec =
                TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

            let backend = native_backend_for_test();
            let request =
                TranscriptionRequest::new(sample_wav_fixture_path(), "cohere-runtime-fixture")
                    .with_model_pack_path(Some(runtime_path));
            let transcription = backend.transcribe(request).unwrap();
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

    #[test]
    fn native_backend_rejects_whisper_pack_missing_tensor_anchor_before_execution() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-runtime.oasr");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_missing_tensor(
            "whisper-runtime-fixture",
            "model.encoder.conv1.weight",
        )
        .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("missing required whisper gguf tensor slot"),
            "{error}"
        );
        assert!(error.contains("model.encoder.conv1.weight"), "{error}");
        assert!(!error.contains("whisper-ggml-executor-v1"), "{error}");
    }

    #[test]
    fn native_backend_rejects_whisper_pack_missing_tokenizer_before_execution() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-encoder-graph.oasr");
        let wav_path = temp.path().join("whisper-short.wav");
        write_mono_pcm16_wav(&wav_path, 16_000, 3_200);
        let fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(wav_path, "whisper-runtime-fixture")
            .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("Whisper GGUF tokenizer is missing required key 'tokenizer.ggml.model'"),
            "{error}"
        );
        assert!(!error.contains("whisper-ggml-executor-v1"), "{error}");
        let stage = classify_whisper_execution_failure_stage(&error);
        assert!(
            matches!(stage, WhisperExecutionFailureStage::MetadataPreflight),
            "unexpected whisper fail-closed stage {stage:?}: {error}"
        );
    }

    #[test]
    fn native_backend_rejects_whisper_pack_with_incomplete_runtime_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-metadata-incomplete.oasr");
        let mut fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        fixture_spec.metadata.remove("n_audio_layer");
        fixture_spec.metadata.remove("whisper.encoder.block_count");
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("whisper runtime metadata contract validation failed"),
            "{error}"
        );
        assert!(error.contains("whisper.encoder.block_count"), "{error}");
    }

    #[test]
    fn native_backend_whisper_executor_accepts_decoder_tensor_alias_and_executes() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-alias.oasr");
        let fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture")
                .with_whisper_required_tensor_alias(
                    "model.decoder.embed_tokens.weight",
                    "model.decoder.token_embedding.weight",
                )
                .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let transcription = backend
            .transcribe(request)
            .expect("the aliased decoder embedding must bind and execute");
        assert!(!transcription.text.is_empty());
        assert!(!transcription.segments.is_empty());
    }

    #[test]
    fn native_backend_rejects_whisper_pack_with_layer_tensor_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-layer-mismatch.oasr");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_layer_count_mismatch(
            "whisper-runtime-fixture",
            2,
            2,
        )
        .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("missing required whisper gguf tensor slot"),
            "{error}"
        );
        assert!(error.contains("encoder.layers.1."), "{error}");
    }

    #[test]
    fn native_backend_rejects_whisper_pack_with_required_tensor_shape_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-shape-mismatch.oasr");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_shape_mismatch(
            "whisper-runtime-fixture",
            "model.encoder.conv2.bias",
            [2_u64],
        )
        .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = native_backend_for_test();
        let request = TranscriptionRequest::new(
            whisper_tiny_context_wav_fixture(&temp),
            "whisper-runtime-fixture",
        )
        .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("runtime pack verification failed"),
            "{error}"
        );
        assert!(
            error.contains("whisper gguf tensor slot 'encoder.conv2.bias'"),
            "{error}"
        );
        assert!(error.contains("invalid shape [2]"), "{error}");
    }

    #[test]
    fn native_model_pack_path_rejects_reserved_non_gguf_oasr_container_magic() {
        let temp = tempfile::tempdir().unwrap();
        let pack_file = temp.path().join("reserved-non-gguf-pack.oasr");
        std::fs::write(&pack_file, b"OASRPKG\0legacy").unwrap();

        let error = validate_local_native_model_pack_path(&pack_file)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reserved OASR container magic"));
    }
}
