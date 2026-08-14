mod atomic_file;
mod backends_manifest_security;
mod catalog_security;
mod catalog_series;
// This module exists in the compiled library only to host a regression test:
// the actual CUDA-arch-target logic runs at build-script time, not at
// runtime, and `build.rs` gets its own copy via `include!` (see
// `cuda_targets.rs`'s doc comment) since a build script cannot depend on the
// crate it configures. Gating the whole module on `cfg(test)` keeps a plain
// (non-test) build free of otherwise-dead-code warnings for functions this
// crate's runtime never calls.
#[cfg(test)]
mod cuda_targets;
mod http;
#[cfg(test)]
mod windows_cmake_cache;

// Module visibility is scoped to the actual external API surface. Modules that
// external crates (openasr-cli, openasr-server, desktop src-tauri) reach into by
// path stay `pub`; most others are `pub(crate)` and expose only the individual
// items re-exported below. A few modules (adapter_pack, device, ggml_runtime)
// still expose additional items only through their module path and remain
// `pub` rather than dropping that reachable-but-currently-unexercised API.
// `testing` is gated so its fixtures do not ship in the default public surface
// (workspace consumers enable the `testing` feature).
pub mod adapter_pack;
pub mod api;
pub mod apikeys;
mod arch;
pub use arch::{
    COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID, COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    COHERE_TRANSCRIBE_GGML_ADAPTER_ID, COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
    COHERE_TRANSCRIBE_TOKENIZER_ID, DOLPHIN_AUDIO_FRONTEND_ID, DOLPHIN_DECODE_POLICY_ID,
    DOLPHIN_GGML_ADAPTER_ID, DOLPHIN_GGML_ARCHITECTURE_ID, DOLPHIN_TOKENIZER_ID,
    MOONSHINE_AUDIO_FRONTEND_ID, MOONSHINE_DECODE_POLICY_ID, MOONSHINE_GGML_ADAPTER_ID,
    MOONSHINE_GGML_ARCHITECTURE_ID, MOONSHINE_TOKENIZER_ID, PARAKEET_CTC_AUDIO_FRONTEND_ID,
    PARAKEET_CTC_DECODE_POLICY_ID, PARAKEET_CTC_GGML_ADAPTER_ID, PARAKEET_CTC_GGML_ARCHITECTURE_ID,
    PARAKEET_CTC_TOKENIZER_ID, PARAKEET_TDT_AUDIO_FRONTEND_ID, PARAKEET_TDT_DECODE_POLICY_ID,
    PARAKEET_TDT_GGML_ADAPTER_ID, PARAKEET_TDT_GGML_ARCHITECTURE_ID, PARAKEET_TDT_TOKENIZER_ID,
    QWEN3_ASR_AUDIO_FRONTEND_ID, QWEN3_ASR_DECODE_POLICY_ID, QWEN3_ASR_GGML_ADAPTER_ID,
    QWEN3_ASR_GGML_ARCHITECTURE_ID, QWEN3_ASR_TOKENIZER_ID, SENSEVOICE_AUDIO_FRONTEND_ID,
    SENSEVOICE_DECODE_POLICY_ID, SENSEVOICE_GGML_ADAPTER_ID, SENSEVOICE_GGML_ARCHITECTURE_ID,
    SENSEVOICE_TOKENIZER_ID, WAV2VEC2_CTC_AUDIO_FRONTEND_ID, WAV2VEC2_CTC_DECODE_POLICY_ID,
    WAV2VEC2_CTC_GGML_ADAPTER_ID, WAV2VEC2_CTC_GGML_ARCHITECTURE_ID, WAV2VEC2_CTC_TOKENIZER_ID,
    WHISPER_AUDIO_FRONTEND_ID, WHISPER_DECODE_POLICY_ID, WHISPER_GGML_ADAPTER_ID,
    WHISPER_GGML_ARCHITECTURE_ID, WHISPER_TOKENIZER_ID, XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
    XASR_ZIPFORMER_DECODE_POLICY_ID, XASR_ZIPFORMER_GGML_ADAPTER_ID,
    XASR_ZIPFORMER_GGML_ARCHITECTURE_ID, XASR_ZIPFORMER_TOKENIZER_ID,
};
pub(crate) mod audio;
pub mod family_inventory;
pub use family_inventory::{
    ExecutionCapabilitiesInventoryV1, ExecutionProviderInventoryV1, ModelFamilyInventoryEntryV1,
    ModelFamilyInventoryV1, builtin_model_family_inventory,
};
// `pub` (not `pub(crate)`): the desktop app reaches this by path
// (`openasr_core::backend_manifest::verify_and_parse`) to verify the
// downloaded inference-kernel manifest -- see the module doc comment.
pub mod backend_manifest;
pub(crate) mod batch;
pub(crate) mod benchmark;
pub(crate) mod capability_pack;
pub(crate) mod capacity;
pub mod config;
pub(crate) mod content_store;
pub mod default_selection;
pub mod device;
pub mod diarize;
pub(crate) mod download_source;
pub(crate) mod format;
pub mod ggml_runtime;
pub(crate) mod home;
pub(crate) mod host;
pub(crate) mod hotword;
pub mod installed_model_store;
pub(crate) mod launch_pack;
pub(crate) mod longform;
pub(crate) mod metrics;
pub mod model_store_gc;
pub mod models;
mod nn;
pub(crate) mod output;
pub(crate) mod pull;
pub(crate) mod punctuation;
pub mod realtime;
pub(crate) mod registry;
pub(crate) mod remote_compute;
pub(crate) mod safety;
pub(crate) mod short_audio_receipt;
pub mod stage_timing;
pub mod subtitle;
mod tensor;
#[cfg(test)]
mod test_process_env;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub(crate) mod translation;

/// Whether the optional forced-aligner pack resolves from the explicit
/// environment override or installed content-addressed model store and
/// satisfies the current runtime contract.
///
/// This intentionally verifies the resolved pack instead of checking only
/// that a file exists. Older Q3/Q4 installs must be treated as unavailable so
/// callers can replace them with the supported Q8_0 tier before inference.
pub fn word_timestamp_forced_aligner_available() -> bool {
    models::qwen::forced_aligner_pack::resolve_forced_aligner_pack_path()
        .is_some_and(|path| models::qwen::verify_forced_aligner_pack(&path).is_ok())
}

pub use api::backend::{
    BackendError, BackendKind, DecodeTruncation, DecodeTruncationReason, ExecutionTarget,
    FailureCategory, GgmlAbortCallbackGuard, NATIVE_RUNTIME_MODEL_ID_AUTO, NativeBackend,
    NativeBackendExecutor, NativeRuntimeModelAdapter, NativeRuntimeModelIdSource,
    NativeRuntimeModelIdentity, NativeRuntimeModelIdentityError, RequestExecutionContext,
    RequestSource, Segment, SliceBoundaryControl, Transcription, TranscriptionBackend,
    TranscriptionControl, TranscriptionRequest, TranscriptionTask, TruncatedDecode, WordTimestamp,
    add_segment_word_timestamps, describe_native_runtime_model_mismatch,
    format_failure_context_line, format_request_context_line,
    native_adapter_supports_source_language_hint, native_runtime_model_adapter_for_path,
    native_runtime_model_refs_match, native_runtime_realtime_capabilities_for_path,
    native_runtime_transcription_capabilities_for_path, refine_existing_transcription_timeline,
    resolve_local_native_runtime_model_identity, validate_local_native_model_pack_path,
    verify_native_runtime_model_pack_path,
};
pub use api::native::{
    NativeAsrBackpressurePolicy, NativeAsrBenchmarkStatus, NativeAsrCapabilities,
    NativeAsrCapabilityClass, NativeAsrError, NativeAsrExecutor, NativeAsrHardwareTarget,
    NativeAsrModelAdapter, NativeAsrModelPackRef, NativeAsrOfflineRequest, NativeAsrRequestOptions,
    NativeAsrRuntimeReadiness, NativeAsrSession, NativeAsrSessionContext,
    NativeAsrStreamingSessionConfig, NativeAsrTensorLayoutRef, load_native_wav_16khz_mono_f32_v0,
};
pub use api::streaming::{StreamingConfig, StreamingEvent, StreamingEventKind, StreamingSession};
pub use atomic_file::write_owner_only_file_atomically;
pub use audio::{
    AudioInputError, AudioInputInfo, AudioInputIssue, AudioPreparationError,
    AudioPreparationOptions, PreparedAudioInput, prepare_audio_input, probe_audio_input,
    probe_wav_duration, recognized_audio_extensions, validate_audio_input,
};
pub(crate) use audio::{PcmBuffer, PcmSlice};
pub use backends_manifest_security::{
    BACKENDS_MANIFEST_PRODUCTION_KEY_ID, BACKENDS_MANIFEST_SIGNATURE_ALGORITHM,
    BACKENDS_MANIFEST_SIGNATURE_FILE_NAME, BACKENDS_MANIFEST_SIGNATURE_SCHEMA_VERSION,
    BackendsManifestSecurityError, BackendsManifestSignature, BackendsManifestSignatureValue,
    VerifiedBackendsManifestSignature, render_backends_manifest_signature,
    verify_backends_manifest_signature,
};
pub use batch::{
    BatchError, BatchFailure, BatchInput, BatchItem, BatchOutput, BatchSummary, batch_output_path,
    discover_batch_inputs, render_batch_summary, response_format_extension,
};
pub use benchmark::{
    BenchmarkFormat, BenchmarkResult, RegressionFinding, RegressionKind, SuiteBaseline,
    SuiteConfig, SuiteEntry, SuiteEntryMetrics, Tolerances, check_quant_ordering, check_vs_cpp,
    compare_to_baseline, probe_audio_duration_seconds, quant_rank, render_benchmark,
    render_suite_json, render_suite_markdown,
};
pub use catalog_security::{
    CATALOG_DEGRADED_MARKER_FILE_NAME, CATALOG_EPOCH_FILE_NAME, CATALOG_SIGNATURE_ALGORITHM,
    CATALOG_SIGNATURE_FILE_NAME, CATALOG_SIGNATURE_KEY_ID, CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID,
    CATALOG_SIGNATURE_SCHEMA_VERSION, CatalogDegradedStatus, CatalogSecurityError,
    CatalogSignature, CatalogSignatureManifest, LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX,
    VerifiedCatalogSignature, catalog_signature_source, clear_catalog_degraded,
    default_catalog_degraded_marker_path, default_catalog_epoch_path,
    default_catalog_signature_cache_path, derive_catalog_public_key_hex,
    read_catalog_degraded_status, record_catalog_degraded, render_catalog_signature_manifest,
    verify_catalog_signature_manifest, verify_local_catalog_signature_manifest,
};
pub use ggml_runtime::{
    GgmlExecutionNodeSample, GgmlExecutionPlacementSummary, GgmlExecutionTelemetryCollector,
    GgmlExecutionTelemetryGuard,
};
pub use metrics::{
    ProcessMemorySnapshot, WerCounts, cer_counts, current_rss_bytes, normalize_text,
    peak_rss_bytes, process_memory_snapshot, wer, wer_counts, word_prefix_error_rate,
};
pub use models::pack_verifier::{PackCandidate, PackVerificationError, PackVerifier, VerifiedPack};
pub use short_audio_receipt::{
    SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK,
    SHORT_AUDIO_RECEIPT_SCHEMA, ShortAudioReceipt, ShortAudioReceiptAudio, ShortAudioReceiptError,
    ShortAudioReceiptLoadError, ShortAudioReceiptMetrics, ShortAudioReceiptPack,
    ShortAudioReceiptRun, ShortAudioReceiptTranscript, median_f64, receipt_os_id,
    resolve_core_commit, sha256_file, sha256_hex_bytes, validate_core_commit,
};
pub use subtitle::{
    TimelinePrecisionPolicy, TimelineQuality, WordAnchorQuality, WordAnchorValidation,
    decide_forced_alignment, project_transcription, validate_word_anchors,
};

pub use config::{
    ConfigError, ConfigKey, DEFAULT_BACKEND_ID, DEFAULT_MODEL_BOOTSTRAP_QUANT, DEFAULT_MODEL_ID,
    MAX_INFERENCE_THREADS, OPENASR_MODELS_DIR_ENV, OpenAsrConfig, OpenAsrConfigDocument,
    config_path, load_config, load_config_document, models_dir, resolve_models_dir, save_config,
    save_config_document, save_default_model_selection,
};
pub use content_store::{ContentLease, ContentStoreError, is_content_addressed_object_path};
pub use device::capabilities::{
    ApplePlatformHints, CpuArchitectureFamily, CpuCapabilities, HardwareCapabilities,
    HardwareFallbackPolicy, HardwareProvider, ProviderAvailability, ProviderAvailabilityState,
    detect_hardware_capabilities,
};
pub use device::compute_devices::{
    ComputeDevice, compute_devices_from_runtime, default_execution_target,
};
pub use device::execution_route::{
    DeviceAddressability, EnumeratedComputeDevice, ExactDeviceSelector, ExecutionProvider,
    ExecutionRouteCacheKey, ExecutionRouteError, ExecutionRouteRequest, PhysicalResourceKey,
    ResolvedExecutionRoute, RouteDeviceKind, admission_identity_for_route,
    enumerate_compute_devices_from_ggml, resolve_execution_route, worker_route_isolation_key,
};
pub use device::types::{CapabilityClass, DeviceCapabilities};
pub use download_source::{DownloadSource, DownloadSourcePref, resolve_chain};
pub use format::{ResponseFormat, render_transcription};
pub use ggml_runtime::{
    GGUF_C_PARSER_SANDBOX_HELPER_ARG, GgmlBackend, GgmlBackendDevice, GgmlBackendKind,
    GgmlCpuBinaryOp, GgmlCpuFeatures, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
    GgmlPackageExtensionHint, GgmlPackageFormat, GgmlPackageModelIdentityProbe, GgmlPackageProbe,
    GgmlPackageProbeError, GgmlRuntimeError, GgmlRuntimeInfo, GgmlRuntimeSource,
    GgmlRuntimeSourcePathError, GgufCParserSandboxError, GgufHostTensorPayload, GgufMetadata,
    GgufMetadataReadError, GgufMetadataValue, GgufRuntimeSourcePreflight, GgufTensorDataReadError,
    GgufTensorDataReader, GgufTensorIndex, GgufTensorIndexReadError, GgufTensorMetadata,
    OPENASR_RUNTIME_PACK_EXTENSION, RuntimeSourceMetadataAndTensorIndexPreflightError,
    ggml_available_devices, ggml_hip_tuning_summary, ggml_native_build_enabled,
    ggml_runtime_boot_summary, ggml_runtime_info, has_openasr_runtime_pack_extension,
    probe_ggml_package_model_identity, probe_ggml_package_path, read_gguf_metadata,
    read_gguf_metadata_from_runtime_source, read_gguf_tensor_index,
    read_gguf_tensor_index_from_runtime_source, render_gguf_c_parser_sandbox_child_output,
    resolve_request_execution_route, validate_ggml_runtime_source_path,
};
pub use home::{OpenAsrHomeError, openasr_home, resolve_openasr_home};
pub use host::{
    host_available_memory_bytes, host_cpu_model, host_os_name_and_version,
    host_quant_recommendation_profile, host_system_boot_summary, host_total_memory_bytes,
};
pub use hotword::{
    DEFAULT_PHRASE_BIAS_BOOST, MAX_PHRASE_BIAS_BOOST, MAX_PHRASE_BIAS_ENTRIES,
    MAX_PHRASE_BIAS_PHRASE_CHARS, MAX_PHRASE_BIAS_TOTAL_CHARS, PhraseBiasConfig, PhraseBiasEntry,
    PhraseBiasError,
};
pub use installed_model_store::{InstalledModelDiagnostic, InstalledModelStore};
pub use launch_pack::{
    LaunchPackError, LaunchPackNotice, LaunchPackRequest, LaunchPackSelection,
    LaunchSelectionReason, QuantPreference, installed_packs_for_model, resolve_launch_pack,
};
pub use longform::{
    AudioSlice, AudioSliceKind, LongFormAssembleStats, LongFormBenchmarkMetadata, LongFormMode,
    LongFormOptions, LongFormOptionsError, LongFormSlicePlan, LongFormSliceStats,
    LongFormVadOptions, LongFormVadProvider, LongFormVadSlice, SegmentMergePolicy,
    SegmentTimeDomain, SliceTranscript, TimelineAnchor, TimelineMap, TranscriptAssembler,
    plan_longform_slices,
};
pub use model_store_gc::{
    ModelStoreEntry, ModelStoreGcReport, ModelStoreRefVerification, ModelStoreUsage,
    ModelStoreVerification, collect_model_store_garbage, model_store_usage, verify_model_store,
};
pub(crate) use models::ggml_asr_executor::{
    GgmlAsrExecutionViewRequest, GgmlAsrPreparedAudioView, GgmlAsrViewExecutor,
};
pub use models::native_execution_services::{
    NativeExecutionScopeId, NativeExecutionServices, NativeExecutionServicesError,
};
pub use models::{
    cohere::COHERE_TRANSCRIBE_MODEL_FAMILY,
    cohere::{
        CohereLocalSourceError, CohereLocalSourceImportRequest,
        CohereLocalSourceImportRuntimeResult, CohereRuntimeQuantizationMode,
        convert_local_cohere_source_to_runtime_pack,
    },
    dolphin::{
        DolphinImportRequest, DolphinImportResult, DolphinLanguageScheme, DolphinQuantizationMode,
        convert_local_dolphin_wenet_source_to_runtime_pack,
    },
    firered_aed::{
        FireRedAedImportRequest, FireRedAedImportResult, FireRedAedQuantizationMode,
        convert_local_firered_aed_source_to_runtime_pack,
    },
    firered_llm::{
        FireRedLlmImportRequest, FireRedLlmImportResult, FireRedLlmQuantizationMode,
        convert_local_firered_llm_source_to_runtime_pack,
    },
    firered_punc::package_import::{
        FireRedPuncImportRequest, FireRedPuncImportResult, FireRedPuncQuantizationMode,
        convert_local_firered_punc_source_to_runtime_pack,
    },
    funasr_nano::{
        FunasrNanoImportRequest, FunasrNanoImportResult, FunasrNanoQuantizationMode,
        convert_local_funasr_nano_source_to_runtime_pack,
    },
    ggml_asr_executor::{
        GgmlAsrBackendPreference, GgmlAsrExecutionDispatch, GgmlAsrExecutionError,
        GgmlAsrExecutionOptions, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
        GgmlAsrPreparedAudio, GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionConfig,
        GgmlAsrStreamingSessionRequest, RuntimeBuildIdentity, StreamingPartialGranularity,
    },
    ggml_family_adapter::{
        GGML_TOKENIZER_ID_KEY, GgmlAdapterMetadataSource, GgmlExecutionCapability,
        GgmlFamilyAdapterDescriptor, GgmlFamilyAdapterSelectionFields,
        GgmlFamilyAdapterSelectionSpec, OasrV1AdapterSelectionMetadata, OasrV1MetadataError,
    },
    hymt2::{
        HYMT2_PINNED_SOURCE_GGUF_SHA256, Hymt2ConfigError, Hymt2DecodeResult, Hymt2DecodeTimings,
        Hymt2ExecutionMetadata, Hymt2ImportError, Hymt2ImportRequest, Hymt2ImportResult,
        Hymt2PrefixCacheConfig, Hymt2PrefixReuseReport, Hymt2Runtime, Hymt2RuntimeError,
        Hymt2TranslationSessionCache, PolicyResolvedHymt2Error,
        PolicyResolvedHymt2TranslationRuntime, import_hymt2_gguf_to_runtime_pack,
    },
    moonshine::{
        MOONSHINE_MODEL_FAMILY, MoonshineLocalSourceError, MoonshineLocalSourceImportRequest,
        MoonshineLocalSourceImportRuntimeResult, MoonshineRuntimeQuantizationMode,
        convert_local_moonshine_source_to_runtime_pack,
    },
    moss_transcribe_diarize::package_import::{
        MossTdImportRequest, MossTdImportResult,
        convert_local_moss_transcribe_diarize_source_to_runtime_pack,
    },
    parakeet_ctc::{
        ParakeetCtcImportRequest, ParakeetCtcImportResult, ParakeetCtcQuantizationMode,
        convert_local_parakeet_ctc_source_to_runtime_pack,
    },
    parakeet_tdt::{
        ParakeetTdtImportRequest, ParakeetTdtImportResult, ParakeetTdtQuantizationMode,
        convert_local_parakeet_tdt_source_to_runtime_pack,
    },
    pyannote::package_import::{
        PyannoteImportRequest, PyannoteImportResult, convert_local_pyannote_source_to_runtime_pack,
    },
    qwen::{
        QWEN3_ASR_MODEL_FAMILY, QWEN3_FORCED_ALIGNER_MODEL_FAMILY, Qwen3AsrLocalSourceError,
        Qwen3AsrLocalSourceImportRequest, Qwen3AsrLocalSourceImportRuntimeResult,
        Qwen3AsrRuntimeQuantizationMode, Qwen3ForcedAlignerLocalSourceError,
        Qwen3ForcedAlignerLocalSourceImportRequest,
        Qwen3ForcedAlignerLocalSourceImportRuntimeResult,
        convert_local_qwen_forced_aligner_source_to_runtime_pack,
        convert_local_qwen_source_to_runtime_pack,
    },
    sensevoice::{
        SenseVoiceImportRequest, SenseVoiceImportResult, SenseVoiceQuantizationMode,
        convert_local_sensevoice_source_to_runtime_pack,
    },
    wav2vec2_ctc::{
        Wav2Vec2CtcImportRequest, Wav2Vec2CtcImportResult, Wav2Vec2CtcQuantizationMode,
        convert_local_wav2vec2_ctc_source_to_runtime_pack,
    },
    whisper::{
        WHISPER_MODEL_FAMILY, WhisperLocalSourceError, WhisperLocalSourceImportRequest,
        WhisperLocalSourceImportRuntimeResult, WhisperRuntimeQuantizationMode, WhisperTokenizer,
        convert_local_whisper_hf_source_to_runtime_pack, whisper_log_mel_spectrogram_16khz_mono_v0,
    },
    xasr_zipformer::{
        XasrZipformerImportRequest, XasrZipformerImportResult, XasrZipformerQuantizationMode,
        convert_local_xasr_zipformer_source_to_runtime_pack,
    },
};
pub use output::{OutputWriteError, atomic_write_text};
pub use pull::{
    BackendFileFormat, DefaultPackPointer, InstalledBackend, InstalledPack, LegacyMigrationFailure,
    LegacyMigrationReport, ModelPackPreflightReceipt, PullError, PullModelPackRequest,
    PullProgress, available_disk_space_bytes, default_pack_pointer_path, install_backend_pack,
    install_catalog_model_pack_from_path,
    install_catalog_model_pack_from_path_with_execution_services, install_model_pack_from_path,
    install_model_pack_from_path_with_execution_services, list_installed_packs,
    migrate_legacy_model_store, migrate_model_store_at_startup, open_installed_content_lease,
    persist_default_pack_pointer, preflight_model_pack_for_install,
    preflight_model_pack_with_receipt, pull_model_pack, read_default_pack_pointer,
    remove_model_pack, remove_model_pack_with_execution_services,
    resolve_catalog_model_pack_from_path, resolve_installed_pack_path,
    resolve_installed_pack_reference, resolve_installed_pack_reference_with_catalog,
};
pub use realtime::{
    BufferedUtterance, CaptureBackpressureQueue, CaptureEngine, CaptureEngineError,
    CaptureInputFormat, CapturePushOutcome, CaptureSample, DEFAULT_REALTIME_CHANNELS,
    DEFAULT_REALTIME_SAMPLE_RATE_HZ, RealtimeAudioEncoding, RealtimeAudioFormat,
    RealtimeAudioFrame, RealtimeAudioInputEvent, RealtimeBackendCapabilities, RealtimeBackendMode,
    RealtimeBuffer, RealtimeBufferConfig, RealtimeBufferError, RealtimeErrorCode,
    RealtimeErrorEvent, RealtimeEvent, RealtimeEventEnvelope, RealtimeEventId, RealtimeEventSeq,
    RealtimeEventSequencer, RealtimeExportFormat, RealtimeFrameError, RealtimeHistoryApplyResult,
    RealtimeHistoryEntry, RealtimeHistoryExportError, RealtimeHistoryRevision,
    RealtimeLifecycleAction, RealtimeLifecycleEvent, RealtimePostProcessOutput,
    RealtimePostProcessor, RealtimeSessionConfig, RealtimeSessionController, RealtimeSessionError,
    RealtimeSessionId, RealtimeSessionState, RealtimeTranscriptEvent, RealtimeTranscriptFinal,
    RealtimeTranscriptHistory, RealtimeTranscriptPartial, RealtimeTranscriptRevision,
    RealtimeTranscriptWord, RealtimeTranslationCapability, RealtimeTranslationEvent,
    RealtimeTranslationFinal, RealtimeTranslationPartial, RealtimeTranslationStatus,
    RealtimeTranslationTombstone, RealtimeUtteranceEndReason, RealtimeVadEvent,
    SessionCapabilitiesEvent, SessionTranslationSummary, SpeechBoundaryEvent,
    TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION, TRANSCRIPT_REVISION_REASONS,
    TranscriptLifecycle, TranscriptLifecycleResult, TranscriptRevisionPolicy, TranscriptSegmentId,
    TranscriptUpdate, TranscriptUtteranceId, VadConfig, VadConfigError, VadDecision,
    VadFrameDecision, VadMode, VadSpeechStartedEvent, VadSpeechStoppedEvent, VadState,
    VadStateMachine,
};
pub use registry::{
    BackendResolutionError, CATALOG_FEATURE_SPEAKER_DIARIZATION, CATALOG_FEATURE_WORD_TIMESTAMPS,
    CatalogBackend, CatalogBackendFile, CatalogBackendFileRole, CatalogBackendVendor,
    CatalogCapability, CatalogCapabilityRole, CatalogError, CatalogLanguageMode, CatalogMirror,
    CatalogModel, CatalogModelKind, CatalogProse, CatalogPullRequest, CatalogQuant,
    CatalogQuantPerf, CatalogQuantRecommendationProfile, CatalogSpeakerSource,
    CatalogWordTimestampSource, LicenseClass, LocalCatalogEnvOverride, ModelAvailability,
    ModelCard, ModelCatalog, ModelInstallLicenseDecision, ModelRef, ModelResolutionError,
    ModelVariantMetadata, OPENASR_CATALOG_FILE_ENV_VAR, OPENASR_CATALOG_IDENTITY_ENV_VAR,
    RegistryError, ResolvedCatalogBackendPull, ResolvedCatalogPull, ResolvedModel,
    ResolvedRuntimeModelRef, RuntimeModelRefSource, RuntimeModelResolutionError,
    RuntimeRegistryError, canonical_quant_tag, current_cli_version, default_catalog_cache_path,
    default_catalog_url, default_registry_dir, embedded_catalog_fingerprint,
    load_embedded_signed_catalog, load_local_catalog_file_with_identity, load_model_catalog,
    load_registry, model_cards_from_catalog, model_install_license_decision,
    model_reference_matches_resolved_source, model_refs_match_with_optional_tag_alias,
    parse_model_catalog, parse_model_ref, preview_local_catalog_file_with_identity,
    recommend_catalog_quant, resolve_catalog_backend_pull, resolve_catalog_pull,
    resolve_catalog_pull_with_profile, resolve_local_catalog_env_override,
    resolve_registry_model_ref, resolve_runtime_catalog, resolve_runtime_model_ref,
    runtime_registry,
};
pub use remote_compute::{
    certificate_fingerprint_sha256, pairing_safety_code_for_certificate_fingerprint,
};
pub use safety::{
    current_platform_key, validate_platform_key, validate_platform_key_field,
    validate_safe_relative_path, validate_sha256,
};
pub use translation::{
    ClauseBoundaryReason, ClauseId, ClauseSegment, ClauseSegmentationConfig,
    ClauseSegmentationUpdate, ClauseSegmenter, ClauseStatus, FinalizedTranslationContext,
    LatestOnlyTranslationQueue, StabilityGate, StabilityGateConfig, StabilityGateDecision,
    StabilityGateInput, StabilityGateReason, TargetLang, TranslationOutput, TranslationQueueError,
    TranslationQueueSubmit, TranslationRequest, TranslationSession, TranslationSessionError,
    TranslationTimings, TranslationWorkerOutput, TranslationWorkerReadiness,
};
