use super::*;
use crate::{BackendKind, RealtimeAudioFormat, RealtimeBackendCapabilities, RealtimeBackendMode};
use std::path::PathBuf;

use super::support::assert_session_failed_contains;

#[test]
fn capability_metadata_represents_native_fallback_and_unsupported() {
    let native = NativeAsrCapabilities::native_true_streaming()
        .with_partial_results(true)
        .with_quantized_models(true)
        .with_hardware_acceleration(true);
    assert_eq!(native.class, NativeAsrCapabilityClass::NativeModelAdapter);
    assert!(native.is_native_adapter());
    assert!(native.supports_true_streaming);
    assert!(native.supports_partials);
    assert!(!native.supports_phrase_bias);
    assert!(native.supports_quantized_models);
    assert!(native.supports_hardware_acceleration);

    let fallback = NativeAsrCapabilities::file_per_utterance_fallback();
    assert_eq!(
        fallback.class,
        NativeAsrCapabilityClass::FilePerUtteranceFallback
    );
    assert!(!fallback.is_native_adapter());
    assert!(!fallback.supports_true_streaming);
    assert!(!fallback.supports_partials);
    assert!(!fallback.supports_phrase_bias);

    let unsupported = NativeAsrCapabilities::unsupported();
    assert_eq!(unsupported.class, NativeAsrCapabilityClass::Unsupported);
    assert!(!unsupported.supports_true_streaming);
    assert!(!unsupported.supports_phrase_bias);
}

#[test]
fn file_per_utterance_fallback_is_separate_from_native_true_streaming() {
    let fallback = NativeAsrCapabilities::file_per_utterance_fallback();
    let streaming = NativeAsrCapabilities::native_true_streaming().with_partial_results(true);

    assert_eq!(
        fallback.class,
        NativeAsrCapabilityClass::FilePerUtteranceFallback
    );
    assert_eq!(
        streaming.class,
        NativeAsrCapabilityClass::NativeModelAdapter
    );
    assert!(!fallback.supports_true_streaming);
    assert!(streaming.supports_true_streaming);

    let realtime_fallback = RealtimeBackendCapabilities::from_native_capabilities(&fallback);
    assert_eq!(
        realtime_fallback.mode,
        RealtimeBackendMode::FilePerUtteranceFallback
    );
    assert!(realtime_fallback.requires_vad_utterance_boundaries);
    assert!(!realtime_fallback.supports_partial_results);
    assert!(realtime_fallback.word_timestamps.supported);

    let realtime_streaming = RealtimeBackendCapabilities::from_native_capabilities(&streaming);
    assert_eq!(realtime_streaming.mode, RealtimeBackendMode::TrueStreaming);
    assert!(realtime_streaming.supports_partial_results);
    assert!(!realtime_streaming.requires_vad_utterance_boundaries);
    assert!(!realtime_streaming.word_timestamps.supported);
}

#[test]
fn realtime_mapping_covers_native_offline_and_unsupported() {
    let native_offline = NativeAsrCapabilities::native_offline();
    let realtime_offline = RealtimeBackendCapabilities::from_native_capabilities(&native_offline);
    assert_eq!(
        realtime_offline.mode,
        RealtimeBackendMode::FilePerUtteranceFallback
    );
    assert!(realtime_offline.supports_realtime_sessions);
    assert!(!realtime_offline.supports_partial_results);
    assert!(!realtime_offline.word_timestamps.supported);
    assert!(!realtime_offline.is_true_streaming);
    assert!(realtime_offline.requires_vad_utterance_boundaries);
    assert!(realtime_offline.is_file_per_utterance_fallback);
    let unsupported = NativeAsrCapabilities::unsupported();
    let realtime_unsupported = RealtimeBackendCapabilities::from_native_capabilities(&unsupported);
    assert_eq!(realtime_unsupported.mode, RealtimeBackendMode::Unsupported);
}

#[test]
fn true_streaming_and_partial_support_are_independent_capability_bits() {
    let streaming_without_partials = NativeAsrCapabilities::native_true_streaming()
        .with_benchmark_status(NativeAsrBenchmarkStatus::NotBenchmarked);
    assert!(!streaming_without_partials.supports_partials);

    let realtime =
        RealtimeBackendCapabilities::from_native_capabilities(&streaming_without_partials);

    assert_eq!(realtime.mode, RealtimeBackendMode::TrueStreaming);
    assert!(realtime.supports_realtime_sessions);
    assert!(realtime.is_true_streaming);
    assert!(!realtime.supports_partial_results);
    assert!(!realtime.word_timestamps.supported);
    assert!(!realtime.effective_partial_results(true));

    let streaming_with_partials = NativeAsrCapabilities::native_true_streaming()
        .with_partial_results(true)
        .with_timestamps(true);
    let realtime_streaming =
        RealtimeBackendCapabilities::from_native_capabilities(&streaming_with_partials);
    assert_eq!(realtime_streaming.mode, RealtimeBackendMode::TrueStreaming);
    assert!(realtime_streaming.supports_partial_results);
    assert!(realtime_streaming.word_timestamps.supported);
    assert!(realtime_streaming.is_true_streaming);
    assert!(!realtime_streaming.is_file_per_utterance_fallback);
}

#[test]
fn existing_backends_do_not_report_true_streaming_without_native_capability() {
    for backend in [BackendKind::Mock, BackendKind::Native, BackendKind::Native] {
        let capabilities = RealtimeBackendCapabilities::for_backend_kind(backend);
        assert_ne!(capabilities.mode, RealtimeBackendMode::TrueStreaming);
        assert!(!capabilities.is_true_streaming);
        assert!(!capabilities.supports_partial_results);
    }
}

#[test]
fn runtime_readiness_errors_are_explicit() {
    assert!(NativeAsrRuntimeReadiness::Ready.is_ready());
    assert_eq!(
        NativeAsrError::try_from(NativeAsrRuntimeReadiness::Ready),
        Err(NativeAsrRuntimeReadiness::Ready)
    );

    let cases = [
        NativeAsrRuntimeReadiness::UnsupportedModelPack {
            reason: "missing native manifest".to_string(),
        },
        NativeAsrRuntimeReadiness::MissingLocalModelAsset {
            path: PathBuf::from("/tmp/openasr/model.oasr"),
        },
        NativeAsrRuntimeReadiness::UnsupportedHardwareTarget {
            target: NativeAsrHardwareTarget::NvidiaCuda,
        },
        NativeAsrRuntimeReadiness::ProviderUnavailable {
            provider: "cuda".to_string(),
        },
        NativeAsrRuntimeReadiness::BackendDoesNotSupportTrueStreaming {
            backend: "native".to_string(),
        },
    ];

    for readiness in cases {
        assert!(!readiness.is_ready());
        let error = NativeAsrError::try_from(readiness).expect("error readiness converts");
        assert!(!error.to_string().trim().is_empty());
    }
}

#[test]
fn model_pack_ref_is_a_local_boundary_without_source_url() {
    let model_pack = NativeAsrModelPackRef::new(
        "native-placeholder",
        "native-placeholder",
        "/tmp/openasr/models/native-placeholder",
    )
    .with_variant("q8_0")
    .with_manifest_path("/tmp/openasr/models/native-placeholder/model.oasr");

    assert_eq!(model_pack.id, "native-placeholder");
    assert_eq!(model_pack.family, "native-placeholder");
    assert_eq!(model_pack.variant.as_deref(), Some("q8_0"));
    assert!(model_pack.root.ends_with("native-placeholder"));
    assert!(model_pack.manifest_path.is_some());
}

#[test]
fn request_options_cover_current_offline_and_realtime_preferences() {
    let options = NativeAsrRequestOptions::new()
        .with_language(Some("en".to_string()))
        .with_prompt(Some("domain terms".to_string()))
        .with_phrase_bias(Some(
            crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap(),
        ))
        .with_diarization(true)
        .with_partial_results(true);
    let request =
        NativeAsrOfflineRequest::new("/tmp/openasr/input.wav").with_options(options.clone());

    assert_eq!(options.language.as_deref(), Some("en"));
    assert_eq!(options.prompt.as_deref(), Some("domain terms"));
    assert_eq!(
        options.phrase_bias.as_ref().unwrap().entries()[0].phrase(),
        "OpenASR"
    );
    assert!(options.diarize);
    assert!(options.partial_results);
    assert!(request.input_path.ends_with("input.wav"));
    assert_eq!(request.options, options);

    let context = NativeAsrSessionContext::new("rt_m55")
        .with_trace_id(Some("trace_m55".to_string()))
        .with_request_id(Some("req_m55".to_string()));
    assert_eq!(context.session_id.0, "rt_m55");
    assert_eq!(context.trace_id.as_deref(), Some("trace_m55"));
    assert_eq!(context.request_id.as_deref(), Some("req_m55"));
}

#[test]
fn streaming_session_config_validates_audio_and_backpressure() {
    assert!(
        NativeAsrStreamingSessionConfig::new()
            .with_partial_results(true)
            .validate()
            .is_ok()
    );

    let invalid_audio =
        NativeAsrStreamingSessionConfig::new().with_audio_format(RealtimeAudioFormat {
            sample_rate_hz: 48_000,
            ..RealtimeAudioFormat::pcm16_mono_16khz()
        });
    assert_session_failed_contains(
        invalid_audio.validate(),
        "invalid Native ASR streaming session config",
    );

    let invalid_backpressure =
        NativeAsrStreamingSessionConfig::new().with_backpressure(NativeAsrBackpressurePolicy {
            max_queued_audio_frames: 0,
            max_queued_events: 64,
        });
    assert_session_failed_contains(
        invalid_backpressure.validate(),
        "invalid Native ASR streaming session config",
    );

    let invalid_event_backpressure =
        NativeAsrStreamingSessionConfig::new().with_backpressure(NativeAsrBackpressurePolicy {
            max_queued_audio_frames: 64,
            max_queued_events: 3,
        });
    assert_session_failed_contains(
        invalid_event_backpressure.validate(),
        "invalid Native ASR streaming session config",
    );
}
