use std::{path::Path, sync::OnceLock};

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
use crate::ggml_runtime::{
    read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
};
use crate::models::builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch;
use crate::models::executor_component_registry::builtin_executor_supports_phrase_bias_for_model_architecture;
use crate::models::ggml_family_adapter::GgmlFamilyAdapterDescriptor;
use crate::models::ggml_family_registry::{
    COHERE_TRANSCRIBE_GGML_ADAPTER_ID, COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
    DOLPHIN_GGML_ARCHITECTURE_ID, GgmlFamilyRegistry, MOONSHINE_GGML_ARCHITECTURE_ID,
    PARAKEET_CTC_GGML_ARCHITECTURE_ID, QWEN3_ASR_GGML_ARCHITECTURE_ID,
    WAV2VEC2_CTC_GGML_ARCHITECTURE_ID, WHISPER_GGML_ARCHITECTURE_ID,
    XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
};
use crate::models::oasr_metadata::{
    OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1, OASR_METADATA_KEY_FEATURE_DIARIZATION,
};
use crate::models::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::models::runtime_tensor_contract_registry::validate_builtin_runtime_tensor_contract_for_architecture;
use crate::realtime::RealtimeBackendCapabilities;
use crate::{
    ExecutionTarget, GgmlAsrBackendPreference, GgmlAsrExecutionDispatch, GgmlAsrExecutionError,
    GgmlAsrExecutionOptions, GgmlAsrStreamingSessionRequest,
};

#[path = "cue_segmentation.rs"]
mod cue_segmentation;
#[path = "native_model_id.rs"]
mod native_model_id;
#[path = "native_path.rs"]
mod native_path;
#[path = "native_transcribe.rs"]
mod native_transcribe;
#[path = "transcription_control.rs"]
mod transcription_control;
pub use native_model_id::{
    NativeRuntimeModelIdSource, NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError,
};
pub use native_transcribe::{
    NativeTranscriptionPhase, NativeTranscriptionProgress, native_transcription_progress,
};
pub use transcription_control::{
    ActiveTranscriptionControlGuard, SliceBoundaryControl, TranscriptionControl,
    install_active_transcription_control,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeBackend;

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeBackendExecutor;

static NATIVE_GGML_STREAMING_EXECUTION_DISPATCH: OnceLock<
    Result<GgmlAsrExecutionDispatch, String>,
> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRuntimeModelAdapter {
    descriptor: GgmlFamilyAdapterDescriptor,
    capabilities: NativeAsrCapabilities,
    language_mode: crate::models::language::LanguageMode,
}

impl NativeRuntimeModelAdapter {
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
            .with_diarization(native_runtime_metadata_supports_diarization(
                metadata,
                descriptor.adapter_id,
            ))
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

    fn model_self_diarizes(&self) -> bool {
        self.capabilities.supports_diarization
    }

    pub(crate) fn language_mode(&self) -> crate::models::language::LanguageMode {
        self.language_mode
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
    let Ok(dispatch) = shared_native_ggml_streaming_execution_dispatch() else {
        return NativeAsrCapabilities::native_offline();
    };
    if !dispatch.has_streaming_executor_for(descriptor) {
        return NativeAsrCapabilities::native_offline();
    }
    // Partial granularity is a property of the registered streaming executor:
    // frame-sync (append-only, never revises) vs buffered (re-decodes a growing
    // window). Only xasr-zipformer is frame-sync today.
    NativeAsrCapabilities::native_true_streaming()
        .with_partial_results(true)
        .with_frame_sync_partials(dispatch.is_frame_sync_for(descriptor))
}

/// Phrase-bias capability for one runtime pack.
///
/// This is family/architecture-level (`builtin_executor_supports_phrase_bias_for_model_architecture`)
/// for every architecture except Dolphin, where the deep-biasing `context_module.*`
/// weights are only present on some packs within the family (the multi-lingual
/// `small`/`base` catalog tiers never trained them) -- reporting the family-wide
/// `true` there let requests reach `hotword_context.rs`, which then hard-fails
/// with a `MissingWeight` error instead of a clean, pre-decode capability
/// rejection. Dolphin therefore probes the pack's own GGUF tensor index for the
/// context-module tensor rather than trusting the architecture constant; every
/// other family keeps the prior architecture-level answer since their executors
/// require the family's tensors unconditionally.
fn native_runtime_descriptor_supports_phrase_bias(
    descriptor: &GgmlFamilyAdapterDescriptor,
    tensor_index: Option<&crate::GgufTensorIndex>,
) -> bool {
    if descriptor.model_architecture == DOLPHIN_GGML_ARCHITECTURE_ID {
        return tensor_index.is_some_and(|tensor_index| {
            tensor_index
                .get(crate::models::dolphin::hotword_context::CONTEXT_MODULE_WORD_EMBEDDING_TENSOR_NAME)
                .is_some()
        });
    }
    builtin_executor_supports_phrase_bias_for_model_architecture(descriptor.model_architecture)
        .unwrap_or(false)
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
        model_pack.family == self.descriptor.model_family
    }

    fn start_streaming_session(
        &self,
        model_pack: &NativeAsrModelPackRef,
        target: NativeAsrHardwareTarget,
        context: NativeAsrSessionContext,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Box<dyn NativeAsrSession>, NativeAsrError> {
        session_config.validate()?;
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
        let backend_preference = native_ggml_backend_preference_from_hardware_target(target)?;
        let request_options = native_streaming_request_options_from_session_options(
            &options,
            self.model_self_diarizes(),
        );
        let request = GgmlAsrStreamingSessionRequest {
            runtime_source_path: model_pack.root.clone(),
            runtime_source_preflight: None,
            selected_family: self.descriptor.clone(),
            request_options,
            configured_diarize: options.diarize,
            backend_preference,
            session_context: context,
            session_config: session_config.into(),
        };
        shared_native_ggml_streaming_execution_dispatch()?
            .start_streaming_session(&request)
            .map_err(|error| native_ggml_streaming_error_to_asr(self.adapter_id(), error))
    }
}

fn native_streaming_request_options_from_session_options(
    options: &NativeAsrRequestOptions,
    model_self_diarizes: bool,
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
    // `NativeAsrRequestOptions::diarize` is the accepted session-level request:
    // realtime uses it to emit `session.configured` and run the external
    // VAD + speaker-embedder diarizer. GGML consumes this flag only for
    // model-native self-diarization, so do not forward post-hoc diarization into decoder
    // prompt construction.
    request_options.diarize = options.diarize && model_self_diarizes;
    request_options
}

impl TranscriptionBackend for NativeBackend {
    fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcription, BackendError> {
        native_transcribe::run_native_transcription(request)
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
        if native_execution_target_from_hardware_target(target).is_none() {
            return NativeAsrRuntimeReadiness::UnsupportedHardwareTarget { target };
        }
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
        match self.runtime_readiness(adapter, model_pack, target) {
            NativeAsrRuntimeReadiness::Ready => {}
            other => {
                return Err(NativeAsrError::try_from(other)
                    .expect("non-ready runtime readiness converts to NativeAsrError"));
            }
        }
        let execution_target = native_execution_target_from_hardware_target(target)
            .ok_or(NativeAsrError::UnsupportedHardwareTarget { target })?;
        let adapter_capabilities = adapter.capabilities();
        reject_unsupported_native_phrase_bias(
            adapter.adapter_id(),
            adapter.model_family(),
            adapter_capabilities.supports_phrase_bias,
            request.options.phrase_bias.as_ref(),
        )?;
        let request =
            native_offline_request_to_transcription_request(model_pack, execution_target, request);
        native_transcribe::run_native_transcription(request).map_err(native_backend_error_to_asr)
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
        match self.runtime_readiness(adapter, model_pack, target) {
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
        adapter.start_streaming_session(model_pack, target, context, options, session_config)
    }
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

fn native_ggml_backend_preference_from_hardware_target(
    target: NativeAsrHardwareTarget,
) -> Result<GgmlAsrBackendPreference, NativeAsrError> {
    match target {
        NativeAsrHardwareTarget::Auto => Ok(GgmlAsrBackendPreference::Auto),
        NativeAsrHardwareTarget::Cpu | NativeAsrHardwareTarget::IntelCpu => {
            Ok(GgmlAsrBackendPreference::CpuOnly)
        }
        NativeAsrHardwareTarget::Accelerated
        | NativeAsrHardwareTarget::AppleSilicon
        | NativeAsrHardwareTarget::NvidiaCuda
        | NativeAsrHardwareTarget::AmdGpu
        | NativeAsrHardwareTarget::IntelGpu => {
            let has_accelerated_device = crate::ggml_available_devices()
                .iter()
                .any(|device| device.kind.is_gpu());
            if has_accelerated_device {
                Ok(GgmlAsrBackendPreference::Accelerated)
            } else {
                Err(NativeAsrError::SessionFailed {
                    message: format!(
                        "hardware target '{target}' was requested, but no ggml GPU device is available"
                    ),
                })
            }
        }
        NativeAsrHardwareTarget::IntelNpu => {
            Err(NativeAsrError::UnsupportedHardwareTarget { target })
        }
    }
}

fn shared_native_ggml_streaming_execution_dispatch()
-> Result<&'static GgmlAsrExecutionDispatch, NativeAsrError> {
    match NATIVE_GGML_STREAMING_EXECUTION_DISPATCH.get_or_init(|| {
        build_builtin_ggml_streaming_execution_dispatch().map_err(|error| error.to_string())
    }) {
        Ok(dispatch) => Ok(dispatch),
        Err(message) => Err(NativeAsrError::SessionFailed {
            message: format!("could not build builtin ggml streaming dispatch: {message}"),
        }),
    }
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
        other => NativeAsrError::SessionFailed {
            message: format!("native ggml streaming session failed: {other}"),
        },
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
    TranscriptionRequest::new(request.input_path, model_pack.id.clone())
        .with_model_pack_path(Some(model_pack.root.clone()))
        .with_language(request.options.language)
        .with_task(request.options.task)
        .with_prompt(request.options.prompt)
        .with_phrase_bias(request.options.phrase_bias)
        .with_inference_threads(request.options.inference_threads)
        .with_execution_target(Some(execution_target))
        .with_word_timestamps(request.options.word_timestamps)
        .with_word_timestamps_refine(request.options.word_timestamps_refine)
        .with_diarization(request.options.diarize)
        .with_longform(request.longform)
        .with_display_file_name(request.display_file_name)
}

fn native_backend_error_to_asr(error: BackendError) -> NativeAsrError {
    NativeAsrError::SessionFailed {
        message: error.to_string(),
    }
}

pub fn validate_local_native_model_pack_path(
    path: &Path,
) -> Result<std::path::PathBuf, BackendError> {
    native_path::validate_local_native_model_pack_path(path)
}

pub fn resolve_local_native_runtime_model_identity(
    runtime_path: &Path,
    explicit_model_id_fallback: Option<&str>,
) -> Result<NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError> {
    native_model_id::resolve_local_native_runtime_model_identity(
        runtime_path,
        explicit_model_id_fallback,
    )
}

pub fn native_runtime_transcription_capabilities_for_path(
    path: &Path,
) -> TranscriptionBackendCapabilities {
    let mut capabilities = TranscriptionBackendCapabilities::for_backend_kind(BackendKind::Native);
    capabilities.phrase_bias = native_phrase_bias_capability_for_path(path);
    capabilities.diarization = native_diarization_capability_for_path(path);
    if let Some(adapter) = native_runtime_model_adapter_for_path(path) {
        capabilities.language = super::LanguageCapability::from(adapter.language_mode());
    }
    capabilities
}

pub(crate) const NATIVE_PHRASE_BIAS_UNAVAILABLE_REASON: &str = "Phrase bias / hotword boosting is not implemented for this native model; requests with phrase_bias or hotword fields are rejected.";

fn native_phrase_bias_capability_for_path(path: &Path) -> BackendFeatureCapability {
    if native_runtime_model_adapter_for_path(path)
        .is_some_and(|adapter| adapter.capabilities().supports_phrase_bias)
    {
        BackendFeatureCapability::supported()
    } else {
        BackendFeatureCapability::reject_request(NATIVE_PHRASE_BIAS_UNAVAILABLE_REASON)
    }
}

/// Reason reported when neither a self-diarizing pack nor the model-agnostic
/// VAD + active speaker-embedder path is installed.
pub(crate) const NATIVE_DIARIZATION_UNAVAILABLE_REASON: &str = "Diarization needs the WeSpeaker speaker-embedder pack (wespeaker-voxceleb-resnet34-lm) or a self-diarizing model pack; install one or omit diarize=true.";

/// Diarization capability for a runtime pack: supported when the model
/// self-diarizes (e.g. the cohere token-stream) or the model-agnostic
/// VAD + active speaker-embedder path is installed for this process.
fn native_diarization_capability_for_path(path: &Path) -> BackendFeatureCapability {
    if native_runtime_path_supports_diarization(path) || crate::diarize::vad_diarization_available()
    {
        BackendFeatureCapability::supported()
    } else {
        BackendFeatureCapability::reject_request(NATIVE_DIARIZATION_UNAVAILABLE_REASON)
    }
}

pub fn native_runtime_realtime_capabilities_for_path(path: &Path) -> RealtimeBackendCapabilities {
    RealtimeBackendCapabilities::from_native_capabilities(
        &native_runtime_asr_capabilities_for_path(path),
    )
}

fn native_runtime_asr_capabilities_for_path(path: &Path) -> NativeAsrCapabilities {
    native_runtime_model_adapter_for_path(path)
        .map(|adapter| adapter.capabilities())
        .unwrap_or_else(NativeAsrCapabilities::unsupported)
}

pub fn native_runtime_model_adapter_for_path(path: &Path) -> Option<NativeRuntimeModelAdapter> {
    let runtime_source = native_path::validate_local_native_runtime_source(path).ok()?;
    let metadata = read_gguf_metadata_from_runtime_source(&runtime_source).ok()?;
    let selection_metadata = selection_metadata_from_gguf(&metadata);
    let registry = GgmlFamilyRegistry::with_builtin_adapters();
    let descriptor = registry
        .select_from_gguf_metadata_v1(&selection_metadata)
        .ok()?
        .clone();
    // Best-effort: a tensor index read failure here should not fail the whole
    // capability lookup (the adapter still resolves from metadata alone); it
    // only narrows the Dolphin per-pack phrase-bias probe to "unsupported".
    let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source).ok();
    Some(NativeRuntimeModelAdapter::new(
        descriptor,
        &metadata,
        tensor_index.as_ref(),
    ))
}

pub fn validate_native_runtime_model_pack_contract(path: &Path) -> Result<(), String> {
    let runtime_source = native_path::validate_local_native_runtime_source(path)
        .map_err(|error| error.to_string())?;
    let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
        .map_err(|error| format!("metadata read failed: {error}"))?;
    // Diarization support packs (speaker embedder, pyannote segmenter) never
    // select an ASR runtime adapter; their contract is "the diarize loader can
    // construct the model from this pack" — checked here so `openasr pull`
    // stays fail-closed for them too.
    if let Some(result) = crate::diarize::validate_diarize_runtime_pack_contract(path, &metadata) {
        return result.map_err(|error| format!("diarization pack validation failed: {error}"));
    }
    // Translation packs (Hy-MT2) never select an ASR runtime adapter either;
    // their contract is "the translation runtime probe accepts this pack" —
    // checked here so `openasr pull` stays fail-closed for them too.
    if let Some(result) =
        crate::models::hymt2::validate_translation_runtime_pack_contract(path, &metadata)
    {
        return result.map_err(|error| format!("translation pack validation failed: {error}"));
    }
    // Punctuation packs (FireRedPunc) are a text-in/labels-out post-processor,
    // not an ASR runtime adapter; their contract is "the punctuation metadata
    // geometry validates" -- checked here so `openasr pull` stays fail-closed
    // for them too.
    if let Some(result) =
        crate::models::firered_punc::validate_punctuation_runtime_pack_contract(path, &metadata)
    {
        return result.map_err(|error| format!("punctuation pack validation failed: {error}"));
    }
    let selection_metadata = selection_metadata_from_gguf(&metadata);
    let registry = GgmlFamilyRegistry::with_builtin_adapters();
    let descriptor = registry
        .select_from_gguf_metadata_v1(&selection_metadata)
        .map_err(|error| format!("runtime adapter selection failed: {error:?}"))?;
    if matches!(
        descriptor.model_architecture,
        QWEN3_ASR_GGML_ARCHITECTURE_ID | COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID
    ) {
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .map_err(|error| format!("tensor index read failed: {error}"))?;
        return validate_builtin_runtime_tensor_contract_for_architecture(
            descriptor.model_architecture,
            &metadata,
            &tensor_index,
        )
        .map(|_| ())
        .map_err(|error| format!("runtime tensor contract validation failed: {error}"));
    }
    // Adapter selection above only checks the `openasr.*` family/architecture
    // routing keys; it does not require the family's *runtime* scalar keys
    // (e.g. whisper's decoder head_count). Without the check below, a pack
    // missing those keys "installs successfully" and only fails closed the
    // first time it is actually loaded for inference (see the turbo pack
    // that shipped without `whisper.decoder.attention.head_count`). Dispatch
    // to each family's existing required-metadata parser so install stays
    // fail-closed at the same gate `openasr pull` already uses for
    // qwen3/cohere above. Only families with such a parser are covered;
    // families without one keep prior (adapter-selection-only) behavior and
    // still fail closed later, at first load, via their executor.
    match descriptor.model_architecture {
        WHISPER_GGML_ARCHITECTURE_ID => {
            crate::models::whisper::runtime_contract::validate_whisper_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "whisper runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        MOONSHINE_GGML_ARCHITECTURE_ID => {
            crate::models::moonshine::runtime_contract::parse_moonshine_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "moonshine runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        PARAKEET_CTC_GGML_ARCHITECTURE_ID => {
            crate::models::parakeet_ctc::runtime_contract::parse_parakeet_ctc_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "parakeet-ctc runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID => {
            crate::models::parakeet_tdt::runtime_contract::parse_parakeet_tdt_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "parakeet-tdt runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        WAV2VEC2_CTC_GGML_ARCHITECTURE_ID => {
            crate::models::wav2vec2_ctc::runtime_contract::parse_wav2vec2_ctc_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "wav2vec2-ctc runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        XASR_ZIPFORMER_GGML_ARCHITECTURE_ID => {
            crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "xasr-zipformer runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        DOLPHIN_GGML_ARCHITECTURE_ID => {
            // `max_ctx` resolution needs the tensor index (the baked
            // position-table tensor's own shape is authoritative over the
            // metadata scalar when present); see
            // `runtime_contract::resolve_position_table_max_ctx`.
            let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
                .map_err(|error| format!("tensor index read failed: {error}"))?;
            crate::models::dolphin::runtime_contract::parse_dolphin_execution_metadata(
                &metadata,
                &tensor_index,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "dolphin runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID => {
            crate::models::sensevoice::runtime_contract::parse_sensevoice_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "sensevoice runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID => {
            crate::models::firered_aed::runtime_contract::parse_firered_aed_execution_metadata(
                &metadata,
            )
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "firered-aed runtime metadata contract validation failed: {error} ({RUNTIME_CONTRACT_OUTDATED_PACK_HINT})"
                )
            })
        }
        // No dedicated required-metadata parser for this architecture (yet):
        // stay Ok() here, same as before this check existed. The executor
        // still fails closed at first load if the pack is incomplete.
        _ => Ok(()),
    }
}

/// Shared hint appended to install-time runtime-contract failures: these
/// always mean the pack is missing keys a current conversion pipeline would
/// have written, not that the file is corrupt.
const RUNTIME_CONTRACT_OUTDATED_PACK_HINT: &str = "this pack was likely produced by an outdated or incompatible conversion pipeline; re-convert or re-pull the model pack";

pub(crate) fn native_runtime_path_supports_diarization(path: &Path) -> bool {
    native_runtime_model_adapter_for_path(path)
        .is_some_and(|adapter| adapter.capabilities().supports_diarization)
}

pub(crate) fn native_runtime_metadata_supports_diarization(
    metadata: &crate::GgufMetadata,
    adapter_id: &str,
) -> bool {
    adapter_id == COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        && metadata
            .get_string(OASR_METADATA_KEY_FEATURE_DIARIZATION)
            .is_some_and(|value| value.trim() == OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1)
        && metadata
            .get_string_array("tokenizer.ggml.tokens")
            .is_some_and(|tokens| {
                tokens.iter().any(|token| token == "<|diarize|>")
                    && tokens.iter().any(|token| token == "<|timestamp|>")
                    && tokens.iter().any(|token| token == "<|spltoken0|>")
            })
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
    };

    fn sample_wav_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist")
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

    fn qwen_streaming_runtime_fixture_spec(_model_id: &str) -> TinyGgufFixtureSpec {
        streaming_runtime_fixture_spec(
            crate::models::qwen::QWEN3_ASR_MODEL_FAMILY,
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            crate::QWEN3_ASR_AUDIO_FRONTEND_ID,
            crate::QWEN3_ASR_DECODE_POLICY_ID,
            crate::QWEN3_ASR_TOKENIZER_ID,
        )
    }

    fn whisper_streaming_runtime_fixture_spec(model_id: &str) -> TinyGgufFixtureSpec {
        TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu(model_id)
    }

    fn cohere_streaming_runtime_fixture_spec(model_id: &str) -> TinyGgufFixtureSpec {
        TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(model_id)
    }

    fn moonshine_streaming_runtime_fixture_spec(_model_id: &str) -> TinyGgufFixtureSpec {
        streaming_runtime_fixture_spec(
            crate::MOONSHINE_MODEL_FAMILY,
            crate::MOONSHINE_GGML_ARCHITECTURE_ID,
            crate::MOONSHINE_AUDIO_FRONTEND_ID,
            crate::MOONSHINE_DECODE_POLICY_ID,
            crate::MOONSHINE_TOKENIZER_ID,
        )
    }

    fn parakeet_ctc_streaming_runtime_fixture_spec(_model_id: &str) -> TinyGgufFixtureSpec {
        streaming_runtime_fixture_spec(
            "parakeet-ctc",
            crate::PARAKEET_CTC_GGML_ARCHITECTURE_ID,
            crate::PARAKEET_CTC_AUDIO_FRONTEND_ID,
            crate::PARAKEET_CTC_DECODE_POLICY_ID,
            crate::PARAKEET_CTC_TOKENIZER_ID,
        )
    }

    fn wav2vec2_ctc_streaming_runtime_fixture_spec(_model_id: &str) -> TinyGgufFixtureSpec {
        streaming_runtime_fixture_spec(
            "wav2vec2-ctc",
            crate::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
            crate::WAV2VEC2_CTC_AUDIO_FRONTEND_ID,
            crate::WAV2VEC2_CTC_DECODE_POLICY_ID,
            crate::WAV2VEC2_CTC_TOKENIZER_ID,
        )
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
        model_id: &'static str,
        family: &'static str,
        adapter_id: &'static str,
        expected_executor_id: &'static str,
        expected_secondary_executor_id: Option<&'static str>,
        fixture_spec: fn(&str) -> TinyGgufFixtureSpec,
    }

    fn streaming_runtime_fixture_cases() -> [StreamingRuntimeFixtureCase; 6] {
        [
            StreamingRuntimeFixtureCase {
                slug: "cohere",
                model_id: "cohere-streaming-runtime",
                family: "cohere-transcribe",
                adapter_id: crate::COHERE_TRANSCRIBE_GGML_ADAPTER_ID,
                expected_executor_id: "cohere-transcribe-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: Some("cohere-transcribe-ggml-executor-v1"),
                fixture_spec: cohere_streaming_runtime_fixture_spec,
            },
            StreamingRuntimeFixtureCase {
                slug: "moonshine",
                model_id: "moonshine-streaming-runtime",
                family: crate::MOONSHINE_MODEL_FAMILY,
                adapter_id: crate::MOONSHINE_GGML_ADAPTER_ID,
                expected_executor_id: "moonshine-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: None,
                fixture_spec: moonshine_streaming_runtime_fixture_spec,
            },
            StreamingRuntimeFixtureCase {
                slug: "parakeet-ctc",
                model_id: "parakeet-ctc-streaming-runtime",
                family: "parakeet-ctc",
                adapter_id: crate::PARAKEET_CTC_GGML_ADAPTER_ID,
                expected_executor_id: "parakeet-ctc-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: None,
                fixture_spec: parakeet_ctc_streaming_runtime_fixture_spec,
            },
            StreamingRuntimeFixtureCase {
                slug: "wav2vec2-ctc",
                model_id: "wav2vec2-ctc-streaming-runtime",
                family: "wav2vec2-ctc",
                adapter_id: crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
                expected_executor_id: "wav2vec2-ctc-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: None,
                fixture_spec: wav2vec2_ctc_streaming_runtime_fixture_spec,
            },
            StreamingRuntimeFixtureCase {
                slug: "qwen",
                model_id: "qwen-streaming-runtime",
                family: crate::models::qwen::QWEN3_ASR_MODEL_FAMILY,
                adapter_id: crate::QWEN3_ASR_GGML_ADAPTER_ID,
                expected_executor_id: "qwen3-asr-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: None,
                fixture_spec: qwen_streaming_runtime_fixture_spec,
            },
            StreamingRuntimeFixtureCase {
                slug: "whisper",
                model_id: "whisper-streaming-runtime",
                family: crate::WHISPER_MODEL_FAMILY,
                adapter_id: crate::WHISPER_GGML_ADAPTER_ID,
                expected_executor_id: "whisper-ggml-snapshot-streaming-executor-v1",
                expected_secondary_executor_id: None,
                fixture_spec: whisper_streaming_runtime_fixture_spec,
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
                        speaker_profile_id: None,
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
    fn native_runtime_model_adapter_selects_descriptor_and_capabilities_from_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_metadata(
                crate::models::oasr_metadata::OASR_METADATA_KEY_FEATURE_DIARIZATION,
                crate::models::oasr_metadata::OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1,
            )
            .with_string_array_metadata(
                "tokenizer.ggml.tokens",
                [
                    "<|startofcontext|>",
                    "<|startoftranscript|>",
                    "<|emo:undefined|>",
                    "<|en|>",
                    "<|pnc|>",
                    "<|noitn|>",
                    "<|notimestamp|>",
                    "<|timestamp|>",
                    "<|nodiarize|>",
                    "<|diarize|>",
                    "<|endoftext|>",
                    "<|spltoken0|>",
                ],
            );
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();

        assert_eq!(adapter.adapter_id(), COHERE_TRANSCRIBE_GGML_ADAPTER_ID);
        assert_eq!(adapter.model_family(), "cohere-transcribe");
        let capabilities = adapter.capabilities();
        assert!(capabilities.is_native_adapter());
        assert!(capabilities.supports_phrase_bias);
        assert!(capabilities.supports_timestamps);
        assert!(capabilities.supports_diarization);
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

    #[test]
    fn native_streaming_request_keeps_embedder_diarization_out_of_ggml_decode_options() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-wespeaker-only-streaming.gguf");
        let spec = cohere_streaming_runtime_fixture_spec("cohere-wespeaker-only-streaming");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let wespeaker_pack = temp.path().join("wespeaker.oasr");
        std::fs::write(&wespeaker_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        unsafe { std::env::set_var("OPENASR_WESPEAKER_PACK", &wespeaker_pack) };

        let realtime_capabilities = native_runtime_realtime_capabilities_for_path(&runtime_path);
        unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
        let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
        let adapter_capabilities = adapter.capabilities();
        assert!(adapter_capabilities.supports_true_streaming);
        assert!(
            realtime_capabilities.diarization.supported,
            "active embedder pack presence should accept the session-level diarize request"
        );
        assert!(
            !adapter.model_self_diarizes(),
            "fixture has no self-diarize metadata/tokens"
        );

        let session_options = NativeAsrRequestOptions::new()
            .with_diarization(true)
            .with_partial_results(true)
            .with_word_timestamps(true);
        let request_options = native_streaming_request_options_from_session_options(
            &session_options,
            adapter.model_self_diarizes(),
        );

        assert!(
            !request_options.diarize,
            "embedder realtime diarization must not switch GGML into self-diarize decode mode"
        );
        assert!(request_options.word_timestamps);
        assert!(
            native_streaming_request_options_from_session_options(&session_options, true).diarize
        );
    }

    #[test]
    fn native_runtime_phrase_bias_capability_matrix_is_per_family() {
        let cases: [(&str, fn(&str) -> TinyGgufFixtureSpec, bool); 7] = [
            ("whisper", whisper_streaming_runtime_fixture_spec, true),
            ("cohere", cohere_streaming_runtime_fixture_spec, true),
            ("qwen", qwen_streaming_runtime_fixture_spec, true),
            ("moonshine", moonshine_streaming_runtime_fixture_spec, true),
            (
                "parakeet-ctc",
                parakeet_ctc_streaming_runtime_fixture_spec,
                true,
            ),
            (
                "wav2vec2-ctc",
                wav2vec2_ctc_streaming_runtime_fixture_spec,
                true,
            ),
            (
                "xasr-zipformer",
                xasr_zipformer_streaming_runtime_fixture_spec,
                false,
            ),
        ];

        for (slug, fixture_spec, expected_phrase_bias) in cases {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join(format!("{slug}.gguf"));
            let spec = fixture_spec(slug);
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
            let transcription = native_runtime_transcription_capabilities_for_path(&runtime_path);
            let realtime = native_runtime_realtime_capabilities_for_path(&runtime_path);

            assert_eq!(
                adapter.capabilities().supports_phrase_bias,
                expected_phrase_bias,
                "{slug} adapter capability"
            );
            assert_eq!(
                transcription.phrase_bias.supported, expected_phrase_bias,
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
        let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
        let model_pack = NativeAsrModelPackRef::new(
            "xasr-zipformer",
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            &runtime_path,
        );
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();

        let error = NativeAsrExecutor::transcribe(
            &NativeBackendExecutor,
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
        let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
        let model_pack = NativeAsrModelPackRef::new(
            "xasr-zipformer",
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            &runtime_path,
        );
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();

        let error = match NativeAsrExecutor::start_streaming_session(
            &NativeBackendExecutor,
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
        for case in streaming_runtime_fixture_cases() {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join(format!("{}.gguf", case.model_id));
            let spec = (case.fixture_spec)(case.model_id);
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
            let capabilities = adapter.capabilities();
            assert_eq!(adapter.adapter_id(), case.adapter_id, "{}", case.slug);
            assert!(capabilities.supports_true_streaming, "{}", case.slug);
            assert!(capabilities.supports_partials, "{}", case.slug);

            let realtime = native_runtime_realtime_capabilities_for_path(&runtime_path);
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

    #[test]
    fn native_backend_starts_declared_streaming_sessions_through_product_route() {
        for case in streaming_runtime_fixture_cases() {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join(format!("{}.gguf", case.model_id));
            let spec = (case.fixture_spec)(case.model_id);
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let adapter = native_runtime_model_adapter_for_path(&runtime_path).unwrap();
            let model_pack = NativeAsrModelPackRef::new(case.model_id, case.family, &runtime_path);
            let backend = NativeBackendExecutor;
            let session_id = format!("rt_{}_backend_streaming", case.slug.replace('-', "_"));
            let mut session = NativeAsrExecutor::start_streaming_session(
                &backend,
                &adapter,
                &model_pack,
                NativeAsrHardwareTarget::Cpu,
                NativeAsrSessionContext::new(&session_id),
                NativeAsrRequestOptions::new().with_partial_results(true),
                NativeAsrStreamingSessionConfig::new().with_partial_results(true),
            )
            .unwrap();

            assert_eq!(session.session_id(), session_id, "{}", case.slug);
            let _ = session.poll_events().unwrap();

            let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
            let sample_count = format.sample_count_for_duration_ms(20).unwrap();
            // push_audio only buffers; the decode (which loads the fixture runtime
            // and fails) runs in poll_events once enough audio passes the
            // first-decode floor. Feed ~1.2s, then poll to surface the error.
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
            // A working fixture runtime yields a partial; a non-loadable one errors
            // via its declared executor. Either proves the session routed correctly.
            match session.poll_events() {
                Ok(events) => assert!(
                    events
                        .iter()
                        .any(|event| event.event_type == "transcript.partial"),
                    "{}: expected a streaming partial via {}",
                    case.slug,
                    case.expected_executor_id
                ),
                Err(error) => {
                    let error = error.to_string();
                    assert!(
                        error.contains(case.expected_executor_id),
                        "{}: {error}",
                        case.slug
                    );
                    if let Some(expected) = case.expected_secondary_executor_id {
                        assert!(error.contains(expected), "{}: {error}", case.slug);
                    }
                }
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
        let backend = NativeBackendExecutor;
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
    fn native_runtime_model_adapter_rejects_invalid_runtime_source() {
        let temp = tempfile::tempdir().unwrap();
        let invalid_path = temp.path().join("not-a-runtime.gguf");
        fs::write(&invalid_path, b"not gguf").unwrap();

        assert!(native_runtime_model_adapter_for_path(&invalid_path).is_none());

        let realtime = native_runtime_realtime_capabilities_for_path(&invalid_path);
        assert_eq!(
            realtime.mode,
            crate::realtime::RealtimeBackendMode::Unsupported
        );
        assert!(!realtime.supports_realtime_sessions);
    }

    #[test]
    fn native_backend_product_executor_reports_runtime_readiness() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = NativeBackendExecutor;
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
            temp.path().join("missing.gguf"),
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
    fn native_hardware_target_mapping_preserves_ggml_streaming_backend_preferences() {
        assert_eq!(
            native_ggml_backend_preference_from_hardware_target(NativeAsrHardwareTarget::Auto)
                .unwrap(),
            GgmlAsrBackendPreference::Auto
        );
        assert_eq!(
            native_ggml_backend_preference_from_hardware_target(NativeAsrHardwareTarget::Cpu)
                .unwrap(),
            GgmlAsrBackendPreference::CpuOnly
        );
        assert!(matches!(
            native_ggml_backend_preference_from_hardware_target(NativeAsrHardwareTarget::IntelNpu),
            Err(NativeAsrError::UnsupportedHardwareTarget {
                target: NativeAsrHardwareTarget::IntelNpu
            })
        ));
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
                    .with_diarization(true)
                    .with_word_timestamps(true),
            )
            .with_longform(Some(longform.clone()))
            .with_display_file_name(Some("meeting.wav".to_string()));

        let converted = native_offline_request_to_transcription_request(
            &model_pack,
            ExecutionTarget::Accelerated,
            request,
        );

        assert!(converted.input_path.ends_with("input.wav"));
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
        assert!(converted.diarize);
        assert_eq!(converted.longform, Some(longform));
        assert_eq!(converted.display_file_name.as_deref(), Some("meeting.wav"));
    }

    #[test]
    fn native_backend_product_executor_dispatches_offline_transcription() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
            let backend = NativeBackendExecutor;
            let adapter = TestNativeRuntimeAdapter { family: "cohere" };
            let model_pack =
                NativeAsrModelPackRef::new("cohere-runtime-fixture", "cohere", runtime_path);
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
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = NativeBackendExecutor;
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
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = NativeBackendExecutor;
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
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let backend = NativeBackendExecutor;
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
        let descriptor = crate::cohere_transcribe_runtime_descriptor_v1();
        let adapter = NativeRuntimeModelAdapter {
            descriptor: descriptor.clone(),
            capabilities: NativeAsrCapabilities::native_true_streaming()
                .with_partial_results(true)
                .with_timestamps(true),
            language_mode: crate::models::language::LanguageMode::SpecifyOnly {
                default_language: "en",
            },
        };
        let model_pack = NativeAsrModelPackRef::new(
            "cohere-runtime-fixture",
            descriptor.model_family,
            "/tmp/openasr/cohere-runtime.gguf",
        );

        let mut session = adapter
            .start_streaming_session(
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
        let error = session.poll_events().unwrap_err().to_string();

        assert!(
            error.contains("cohere-transcribe-ggml-snapshot-streaming-executor-v1"),
            "{error}"
        );
        assert!(
            error.contains("could not load runtime preflight"),
            "{error}"
        );
    }

    #[test]
    fn native_backend_requires_model_pack_path() {
        let backend = NativeBackend;
        let request = TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-small");

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("requires an explicit local runtime pack path"));
        assert!(error.contains("fail-closed"));
    }

    #[test]
    fn native_backend_rejects_diarization_requests() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            // Hermetic: the run-time gate probes the host's installed
            // WeSpeaker pack, so pin the lookup to an empty home.
            unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
            unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let backend = NativeBackend;
            let request =
                TranscriptionRequest::new(sample_wav_fixture_path(), "cohere-runtime-fixture")
                    .with_model_pack_path(Some(runtime_path))
                    .with_diarization(true);

            let error = backend.transcribe(request).unwrap_err().to_string();

            assert!(error.contains("speaker-embedder pack"));
            assert!(error.contains("native backend"));
        });
    }

    #[test]
    fn pull_contract_validation_routes_diarize_packs_to_their_loader() {
        let temp = tempfile::tempdir().unwrap();
        let pack_path = temp.path().join("wespeaker-stub.oasr");
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            crate::ggml_runtime::GgufWriteValue::String(
                crate::models::wespeaker::WESPEAKER_GGML_ARCHITECTURE_ID.to_string(),
            ),
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

        // A diarization-architecture pack must be validated by its own loader
        // (which rejects this stub for missing tensors), not by ASR runtime
        // adapter selection (which would reject ALL diarize packs).
        let error = validate_native_runtime_model_pack_contract(&pack_path)
            .expect_err("stub wespeaker pack must fail its loader contract");
        assert!(
            error.contains("diarization pack validation failed"),
            "got: {error}"
        );
        assert!(
            !error.contains("runtime adapter selection failed"),
            "got: {error}"
        );
    }

    #[test]
    fn native_backend_rejects_speakers_hint_without_diarize() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-runtime.gguf");
            let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

            let backend = NativeBackend;
            let request =
                TranscriptionRequest::new(sample_wav_fixture_path(), "cohere-runtime-fixture")
                    .with_model_pack_path(Some(runtime_path))
                    .with_diarize_speakers(Some(2));

            let error = backend.transcribe(request).unwrap_err().to_string();

            assert!(error.contains("speakers hint requires diarize=true"));
        });
    }

    #[test]
    fn native_runtime_capabilities_enable_declared_cohere_diarization_pack() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-diarize-runtime.gguf");
        let spec =
            TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-diarize-runtime-fixture")
                .with_metadata(
                    crate::models::oasr_metadata::OASR_METADATA_KEY_FEATURE_DIARIZATION,
                    crate::models::oasr_metadata::OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1,
                )
                .with_string_array_metadata(
                    "tokenizer.ggml.tokens",
                    [
                        "<|startofcontext|>",
                        "<|startoftranscript|>",
                        "<|emo:undefined|>",
                        "<|en|>",
                        "<|pnc|>",
                        "<|noitn|>",
                        "<|notimestamp|>",
                        "<|timestamp|>",
                        "<|nodiarize|>",
                        "<|diarize|>",
                        "<|endoftext|>",
                        "<|spltoken0|>",
                        "▁fixture11",
                        "▁fixture12",
                        "▁fixture13",
                        "▁fixture14",
                        "▁fixture15",
                        "▁fixture16",
                        "▁fixture17",
                        "▁fixture18",
                        "▁fixture19",
                        "▁fixture20",
                        "▁fixture21",
                        "▁fixture22",
                        "▁fixture23",
                        "▁fixture24",
                        "▁fixture25",
                        "▁fixture26",
                        "▁fixture27",
                        "▁fixture28",
                        "▁fixture29",
                        "▁fixture30",
                        "▁fixture31",
                    ],
                );
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let capabilities = native_runtime_transcription_capabilities_for_path(&runtime_path);

        assert!(capabilities.diarization.supported);
    }

    #[test]
    fn native_runtime_capabilities_keep_base_cohere_diarization_unsupported() {
        let temp = tempfile::tempdir().unwrap();
        // Hermetic: the capability probe also consults the host's installed
        // WeSpeaker pack, so pin the lookup to an empty home.
        unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
        unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let capabilities = native_runtime_transcription_capabilities_for_path(&runtime_path);

        assert!(!capabilities.diarization.supported);
        assert!(
            capabilities
                .diarization
                .reason
                .is_some_and(|reason| reason.contains("speaker-embedder pack"))
        );
    }

    #[test]
    fn native_runtime_capabilities_enable_vad_diarization_when_wespeaker_pack_installed() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.gguf");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();
        let wespeaker_pack = temp.path().join("wespeaker.oasr");
        std::fs::write(&wespeaker_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        unsafe { std::env::set_var("OPENASR_WESPEAKER_PACK", &wespeaker_pack) };

        let capabilities = native_runtime_transcription_capabilities_for_path(&runtime_path);
        unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };

        // The VAD + WeSpeaker path is model-agnostic: a pack with no
        // self-diarize metadata reports diarization supported once the embedder
        // pack is installed.
        assert!(capabilities.diarization.supported);
    }

    #[test]
    fn native_runtime_realtime_capabilities_are_runtime_owned_and_conservative() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-runtime.gguf");
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
        let backend = NativeBackend;
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
            crate::models::ggml_family_registry::dolphin_runtime_descriptor_v1();
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
        let whisper_descriptor =
            crate::models::ggml_family_registry::GgmlFamilyRegistry::with_builtin_adapters()
                .descriptors()
                .iter()
                .find(|descriptor| descriptor.model_architecture == WHISPER_GGML_ARCHITECTURE_ID)
                .expect("whisper descriptor registered")
                .clone();
        assert!(native_runtime_descriptor_supports_phrase_bias(
            &whisper_descriptor,
            Some(&base_tensor_index),
        ));
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

        assert!(error.contains("Expected a local GGUF-backed runtime file"));
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
    fn native_model_pack_path_rejects_directory_without_openasr_suffix() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("not-a-pack");
        std::fs::create_dir_all(&directory).unwrap();

        let error = validate_local_native_model_pack_path(&directory)
            .unwrap_err()
            .to_string();

        assert!(error.contains("must be a regular file"));
    }

    #[test]
    fn native_model_pack_path_preserves_input_file_path() {
        let temp = tempfile::tempdir().unwrap();
        let pack_file = temp.path().join("valid-pack.gguf");
        std::fs::write(&pack_file, b"GGUFpayload").unwrap();

        let validated = validate_local_native_model_pack_path(&pack_file).unwrap();
        assert_eq!(validated, pack_file);
    }

    #[test]
    fn native_backend_fails_closed_when_gguf_oasr_metadata_is_incomplete() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-incomplete.gguf");
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

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("gguf metadata is missing required OASR v1 key 'openasr.decode.policy'"),
            "{error}"
        );
    }

    #[test]
    fn native_backend_fails_closed_when_gguf_metadata_has_no_registered_family_adapter() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("unknown-family.gguf");
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

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "unknown-family-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("gguf metadata does not match any registered family adapter"),
            "{error}"
        );
    }

    #[test]
    fn native_backend_synthesizes_oasr_selection_keys_from_qwen_general_architecture() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = TinyGgufFixtureSpec::new(
            [
                ("general.architecture", "qwen3-asr"),
                ("qwen3-asr.sample_rate", "16000"),
                ("qwen3-asr.n_mels", "80"),
                ("qwen3-asr.n_fft", "400"),
                ("qwen3-asr.win_length", "400"),
                ("qwen3-asr.hop_length", "160"),
                ("qwen3-asr.audio.n_layers", "2"),
                ("qwen3-asr.audio.d_model", "16"),
                ("qwen3-asr.audio.n_heads", "2"),
                ("qwen3-asr.llm.n_layers", "2"),
                ("qwen3-asr.llm.d_model", "16"),
                ("qwen3-asr.llm.n_heads", "2"),
                ("qwen3-asr.llm.n_kv_heads", "2"),
                ("qwen3-asr.llm.head_dim", "8"),
                ("qwen3-asr.llm.vocab_size", "32"),
                ("qwen3-asr.llm.max_pos", "256"),
                ("qwen3-asr.audio_start_token_id", "2"),
                ("qwen3-asr.audio_end_token_id", "3"),
                ("qwen3-asr.audio_pad_token_id", "4"),
                ("qwen3-asr.eos_token_id", "5"),
                ("qwen3-asr.pad_token_id", "6"),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        )
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

        let backend = NativeBackend;
        let request = TranscriptionRequest::new(sample_wav_fixture_path(), "qwen3-asr-0.6b-q4_k")
            .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();
        assert!(error.contains("qwen3-asr-ggml-executor-v1"), "{error}");
        assert!(error.contains("qwen3-asr audio encoder"), "{error}");
        assert!(error.contains("audio.conv.1.weight"), "{error}");
    }

    #[test]
    fn native_backend_synthesizes_oasr_selection_keys_from_cohere_general_architecture() {
        with_forced_cpu_backend_for_test(|| {
            let temp = tempfile::tempdir().unwrap();
            let runtime_path = temp.path().join("cohere-transcribe-q4_k.gguf");
            let fixture_spec =
                TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
            write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

            let backend = NativeBackend;
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
    fn native_backend_selects_whisper_executor_and_fails_on_missing_tensor_anchor() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-runtime.gguf");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_missing_tensor(
            "whisper-runtime-fixture",
            "model.encoder.conv1.weight",
        );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("whisper-ggml-executor-v1"), "{error}");
        assert!(
            error.contains("whisper ggml executor missing required GGUF tensor"),
            "{error}"
        );
        assert!(error.contains("model.encoder.conv1.weight"), "{error}");
    }

    #[test]
    fn native_backend_whisper_encoder_graph_fixture_fails_closed_at_tokenizer_preflight() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-encoder-graph.gguf");
        let wav_path = temp.path().join("whisper-short.wav");
        write_mono_pcm16_wav(&wav_path, 16_000, 3_200);
        let fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request = TranscriptionRequest::new(wav_path, "whisper-runtime-fixture")
            .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("whisper-ggml-executor-v1"), "{error}");
        assert!(
            error.contains("whisper ggml executor tokenizer is missing"),
            "{error}"
        );
        assert!(error.contains("tokenizer.ggml.model"), "{error}");
        let stage = classify_whisper_execution_failure_stage(&error);
        assert!(
            matches!(stage, WhisperExecutionFailureStage::MetadataPreflight),
            "unexpected whisper fail-closed stage {stage:?}: {error}"
        );
    }

    #[test]
    fn native_backend_whisper_executor_fails_closed_when_whisper_gguf_metadata_is_incomplete() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-metadata-incomplete.gguf");
        let mut fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        fixture_spec.metadata.remove("n_audio_layer");
        fixture_spec.metadata.remove("whisper.encoder.block_count");
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("whisper-ggml-executor-v1"), "{error}");
        assert!(
            error.contains("whisper ggml executor missing required GGUF metadata key"),
            "{error}"
        );
        assert!(error.contains("whisper.encoder.block_count"), "{error}");
    }

    #[test]
    fn native_backend_whisper_executor_accepts_decoder_tensor_alias_and_reaches_tokenizer_boundary()
    {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-alias.gguf");
        let fixture_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture")
                .with_whisper_required_tensor_alias(
                    "model.decoder.embed_tokens.weight",
                    "model.decoder.token_embedding.weight",
                );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("whisper ggml executor tokenizer is missing"),
            "{error}"
        );
        assert!(error.contains("tokenizer.ggml.model"), "{error}");
    }

    #[test]
    fn native_backend_whisper_executor_detects_layer_tensor_mismatch_from_fixture_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-layer-mismatch.gguf");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_layer_count_mismatch(
            "whisper-runtime-fixture",
            2,
            2,
        );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains("whisper ggml executor missing required GGUF tensor"),
            "{error}"
        );
        assert!(error.contains("model.encoder.layers.1."), "{error}");
    }

    #[test]
    fn native_backend_whisper_executor_detects_required_tensor_shape_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("whisper-shape-mismatch.gguf");
        let fixture_spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_shape_mismatch(
            "whisper-runtime-fixture",
            "model.encoder.conv2.bias",
            [2_u64],
        );
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).unwrap();

        let backend = NativeBackend;
        let request =
            TranscriptionRequest::new(sample_wav_fixture_path(), "whisper-runtime-fixture")
                .with_model_pack_path(Some(runtime_path));
        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(
            error.contains(
                "whisper ggml executor tensor 'model.encoder.conv2.bias' failed binding validation"
            ),
            "{error}"
        );
        assert!(error.contains("shape=[2]"), "{error}");
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
