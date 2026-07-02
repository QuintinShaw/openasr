use crate::{
    BackendKind,
    api::{
        backend::BackendFeatureCapability,
        native::{NativeAsrCapabilities, NativeAsrCapabilityClass},
    },
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeBackendMode {
    Unsupported,
    FilePerUtteranceFallback,
    TrueStreaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RealtimeBackendCapabilities {
    pub mode: RealtimeBackendMode,
    pub supports_realtime_sessions: bool,
    pub supports_partial_results: bool,
    pub phrase_bias: BackendFeatureCapability,
    pub word_timestamps: BackendFeatureCapability,
    pub diarization: BackendFeatureCapability,
    pub translation: RealtimeTranslationCapability,
    pub requires_vad_utterance_boundaries: bool,
    pub is_file_per_utterance_fallback: bool,
    pub is_true_streaming: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RealtimeTranslationCapability {
    pub supported: bool,
    pub installed: bool,
    pub experimental: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<&'static str>,
    pub source_langs: &'static [&'static str],
    pub target_langs: &'static [&'static str],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_pack: Option<&'static str>,
    pub reason: Option<&'static str>,
}

impl RealtimeTranslationCapability {
    pub const MODE_CLAUSE_RETRANSLATION: &'static str = "clause_retranslation";
    pub const MODEL_ID_HYMT2_1_8B_Q4_K_M: &'static str = "hymt2-1.8b-q4_k_m";
    pub const REASON_PACK_MISSING: &'static str = "translation_pack_missing";
    pub const REASON_MODEL_UNSUPPORTED: &'static str = "translation_model_unsupported";

    pub const fn unavailable(reason: &'static str) -> Self {
        Self {
            supported: false,
            installed: false,
            experimental: true,
            mode: Some(Self::MODE_CLAUSE_RETRANSLATION),
            source_langs: &["zh"],
            target_langs: &["en"],
            model_id: Some(Self::MODEL_ID_HYMT2_1_8B_Q4_K_M),
            requires_pack: Some(Self::MODEL_ID_HYMT2_1_8B_Q4_K_M),
            reason: Some(reason),
        }
    }

    pub const fn installed_hymt2() -> Self {
        Self {
            supported: true,
            installed: true,
            experimental: true,
            mode: Some(Self::MODE_CLAUSE_RETRANSLATION),
            source_langs: &["zh"],
            target_langs: &["en"],
            model_id: Some(Self::MODEL_ID_HYMT2_1_8B_Q4_K_M),
            requires_pack: Some(Self::MODEL_ID_HYMT2_1_8B_Q4_K_M),
            reason: None,
        }
    }
}

impl RealtimeBackendCapabilities {
    const fn build(
        mode: RealtimeBackendMode,
        supports_realtime_sessions: bool,
        supports_partial_results: bool,
        requires_vad_utterance_boundaries: bool,
        phrase_bias: BackendFeatureCapability,
        word_timestamps: BackendFeatureCapability,
    ) -> Self {
        Self {
            mode,
            supports_realtime_sessions,
            supports_partial_results,
            phrase_bias,
            word_timestamps,
            // The diarization capability is mode- and install-dependent, so the
            // const builders stay conservative; the non-const constructors
            // overwrite it via [`realtime_diarization_capability`].
            diarization: realtime_diarization_unprobed(),
            translation: RealtimeTranslationCapability::unavailable(
                RealtimeTranslationCapability::REASON_PACK_MISSING,
            ),
            requires_vad_utterance_boundaries,
            is_file_per_utterance_fallback: matches!(
                mode,
                RealtimeBackendMode::FilePerUtteranceFallback
            ),
            is_true_streaming: matches!(mode, RealtimeBackendMode::TrueStreaming),
        }
    }

    pub const fn unsupported() -> Self {
        Self::build(
            RealtimeBackendMode::Unsupported,
            false,
            false,
            false,
            realtime_phrase_bias_unsupported(),
            realtime_word_timestamps_unsupported(),
        )
    }

    pub const fn file_per_utterance_fallback() -> Self {
        Self::build(
            RealtimeBackendMode::FilePerUtteranceFallback,
            true,
            false,
            true,
            realtime_phrase_bias_unsupported(),
            BackendFeatureCapability::supported(),
        )
    }

    pub const fn file_per_utterance_fallback_with_phrase_bias() -> Self {
        Self::build(
            RealtimeBackendMode::FilePerUtteranceFallback,
            true,
            false,
            true,
            BackendFeatureCapability::supported(),
            BackendFeatureCapability::supported(),
        )
    }

    pub const fn true_streaming_local() -> Self {
        Self::build(
            RealtimeBackendMode::TrueStreaming,
            true,
            true,
            false,
            realtime_phrase_bias_unsupported(),
            BackendFeatureCapability::supported(),
        )
    }

    pub fn for_backend_kind(backend: BackendKind) -> Self {
        let mut capabilities = match backend {
            BackendKind::Mock => Self::file_per_utterance_fallback(),
            BackendKind::Native => Self::file_per_utterance_fallback_with_phrase_bias(),
        };
        capabilities.diarization = realtime_diarization_capability(capabilities.mode);
        capabilities
    }

    pub fn from_native_capabilities(capabilities: &NativeAsrCapabilities) -> Self {
        let mut realtime = Self::from_native_capabilities_without_diarization(capabilities);
        realtime.diarization = realtime_diarization_capability(realtime.mode);
        realtime
    }

    fn from_native_capabilities_without_diarization(capabilities: &NativeAsrCapabilities) -> Self {
        match capabilities.class {
            NativeAsrCapabilityClass::Unsupported => Self::unsupported(),
            NativeAsrCapabilityClass::NativeModelAdapter
                if capabilities.supports_true_streaming =>
            {
                Self::build(
                    RealtimeBackendMode::TrueStreaming,
                    true,
                    capabilities.supports_partials,
                    false,
                    if capabilities.supports_phrase_bias {
                        BackendFeatureCapability::supported()
                    } else {
                        realtime_phrase_bias_unsupported()
                    },
                    if capabilities.supports_timestamps {
                        BackendFeatureCapability::supported()
                    } else {
                        realtime_word_timestamps_unsupported()
                    },
                )
            }
            NativeAsrCapabilityClass::NativeModelAdapter => Self::build(
                RealtimeBackendMode::FilePerUtteranceFallback,
                true,
                false,
                true,
                if capabilities.supports_phrase_bias {
                    BackendFeatureCapability::supported()
                } else {
                    realtime_phrase_bias_unsupported()
                },
                if capabilities.supports_timestamps {
                    BackendFeatureCapability::supported()
                } else {
                    realtime_word_timestamps_unsupported()
                },
            ),
            NativeAsrCapabilityClass::FilePerUtteranceFallback => {
                Self::file_per_utterance_fallback()
            }
        }
    }

    pub fn effective_partial_results(self, requested: bool) -> bool {
        requested && self.supports_partial_results
    }
}

/// Realtime diarization capability for a session mode, derived fresh on every
/// call: the streaming diarizer labels utterances inside the session loop, so
/// it is model-agnostic and needs only the installed active speaker-embedder
/// pack. The file-per-utterance path diarizes the buffered utterance audio;
/// true-streaming sessions retain a bounded speech-gated copy of each
/// utterance for the same purpose. Presence-only probe; pack load failures
/// still fail closed at session configure time.
pub fn realtime_diarization_capability(mode: RealtimeBackendMode) -> BackendFeatureCapability {
    match mode {
        RealtimeBackendMode::Unsupported => realtime_diarization_unprobed(),
        RealtimeBackendMode::FilePerUtteranceFallback | RealtimeBackendMode::TrueStreaming => {
            if crate::diarize::vad_diarization_available() {
                BackendFeatureCapability::supported()
            } else {
                BackendFeatureCapability::reject_request(
                    "Realtime diarization needs the WeSpeaker speaker-embedder pack (wespeaker-voxceleb-resnet34-lm); install it or omit diarize=true.",
                )
            }
        }
    }
}

/// Conservative const default for the const capability builders; the non-const
/// constructors overwrite it via [`realtime_diarization_capability`].
const fn realtime_diarization_unprobed() -> BackendFeatureCapability {
    BackendFeatureCapability::reject_request(
        "Realtime diarization needs the WeSpeaker speaker-embedder pack (wespeaker-voxceleb-resnet34-lm); install it or omit diarize=true.",
    )
}

const fn realtime_phrase_bias_unsupported() -> BackendFeatureCapability {
    BackendFeatureCapability::reject_request(
        "Realtime phrase bias / hotword boosting is not implemented for this active backend/model; session.start requests with phrase_bias or hotwords are rejected.",
    )
}

const fn realtime_word_timestamps_unsupported() -> BackendFeatureCapability {
    BackendFeatureCapability::reject_request(
        "Realtime word timestamps are not implemented for this backend; session.start requests with word_timestamps=true are rejected.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RealtimeEventId, RealtimeTranscriptEvent, TranscriptLifecycle, TranscriptLifecycleResult,
        TranscriptUpdate,
    };

    #[test]
    fn current_backends_are_file_per_utterance_without_partials() {
        for backend in [BackendKind::Mock, BackendKind::Native] {
            let capabilities = RealtimeBackendCapabilities::for_backend_kind(backend);
            assert_eq!(
                capabilities.mode,
                RealtimeBackendMode::FilePerUtteranceFallback
            );
            assert!(capabilities.supports_realtime_sessions);
            assert!(!capabilities.supports_partial_results);
            assert_eq!(
                capabilities.phrase_bias.supported,
                backend == BackendKind::Native
            );
            assert!(capabilities.word_timestamps.supported);
            assert!(capabilities.requires_vad_utterance_boundaries);
            assert!(capabilities.is_file_per_utterance_fallback);
            assert!(!capabilities.is_true_streaming);
            assert!(!capabilities.effective_partial_results(true));
        }
    }

    #[test]
    fn realtime_diarization_capability_depends_on_mode_and_pack() {
        let temp = tempfile::tempdir().unwrap();
        // Hermetic: the probe consults the installed WeSpeaker embedder pack.
        unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
        unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };

        let fallback =
            realtime_diarization_capability(RealtimeBackendMode::FilePerUtteranceFallback);
        assert!(!fallback.supported);
        assert!(
            fallback
                .reason
                .is_some_and(|reason| reason.contains("speaker-embedder pack"))
        );
        let streaming = realtime_diarization_capability(RealtimeBackendMode::TrueStreaming);
        assert!(!streaming.supported);
        assert!(
            streaming
                .reason
                .is_some_and(|reason| reason.contains("speaker-embedder pack"))
        );

        let install_dir = temp
            .path()
            .join("models/wespeaker-voxceleb-resnet34-lm/f32");
        std::fs::create_dir_all(&install_dir).unwrap();
        let installed_pack = install_dir.join("wespeaker-voxceleb-resnet34-lm-f32.oasr");
        std::fs::write(&installed_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        let installed_meta = serde_json::json!({
            "model_id": "wespeaker-voxceleb-resnet34-lm",
            "display_name": "WeSpeaker ResNet34 Speaker Embedder (VoxCeleb)",
            "quant": "f32",
            "suffix": "f32",
            "pull": "wespeaker-voxceleb-resnet34-lm:f32",
            "filename": "wespeaker-voxceleb-resnet34-lm-f32.oasr",
            "path": installed_pack,
            "url": "https://example.invalid/OpenASR/wespeaker-voxceleb-resnet34-lm/wespeaker-voxceleb-resnet34-lm-f32.oasr",
            "hf_revision": "0123456789abcdef0123456789abcdef01234567",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "size_bytes": 8,
            "installed_at_unix_seconds": 1
        });
        std::fs::write(
            install_dir.join("installed.json"),
            format!("{installed_meta}\n"),
        )
        .unwrap();
        assert!(
            realtime_diarization_capability(RealtimeBackendMode::FilePerUtteranceFallback)
                .supported
        );
        assert!(realtime_diarization_capability(RealtimeBackendMode::TrueStreaming).supported);

        let wespeaker = temp.path().join("wespeaker.oasr");
        std::fs::write(&wespeaker, b"GGUF\x00\x00\x00\x00").unwrap();
        unsafe { std::env::set_var("OPENASR_WESPEAKER_PACK", &wespeaker) };
        assert!(
            realtime_diarization_capability(RealtimeBackendMode::FilePerUtteranceFallback)
                .supported
        );
        // True-streaming sessions retain a bounded copy of each utterance's
        // speech, so the pack is the only gate there as well.
        assert!(realtime_diarization_capability(RealtimeBackendMode::TrueStreaming).supported);
        unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
    }

    #[test]
    fn true_streaming_capability_can_enable_requested_partials_without_downloads() {
        let capabilities = RealtimeBackendCapabilities::true_streaming_local();
        assert_eq!(capabilities.mode, RealtimeBackendMode::TrueStreaming);
        assert!(capabilities.supports_realtime_sessions);
        assert!(capabilities.supports_partial_results);
        assert!(capabilities.word_timestamps.supported);
        assert!(!capabilities.phrase_bias.supported);
        assert!(!capabilities.requires_vad_utterance_boundaries);
        assert!(!capabilities.is_file_per_utterance_fallback);
        assert!(capabilities.is_true_streaming);
        assert!(capabilities.effective_partial_results(true));
        assert!(!capabilities.effective_partial_results(false));
    }

    #[test]
    fn test_only_streaming_lifecycle_exercises_partial_final_revision_events() {
        let mut lifecycle = TranscriptLifecycle::default();
        let partial = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_test", "seg_test", 1, "hel", 0, 120,
        ));
        assert!(matches!(
            partial,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(_))
        ));

        let final_event = lifecycle.apply_final(
            TranscriptUpdate::new("utt_test", "seg_test", 2, "hello", 0, 240),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert!(matches!(
            final_event,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(_))
        ));

        let revision = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_test",
            "seg_test",
            3,
            "hello world",
            0,
            360,
        ));
        assert!(matches!(
            revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(_))
        ));
    }

    #[test]
    fn test_only_streaming_lifecycle_rejects_stale_final() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_test", "seg_test", 2, "hello", 0, 240,
        ));

        let stale_final = lifecycle.apply_final(
            TranscriptUpdate::new("utt_test", "seg_test", 1, "hel", 0, 120),
            Some(RealtimeEventId("evt_stale_final".to_string())),
        );

        assert_eq!(
            stale_final,
            TranscriptLifecycleResult::IgnoredOutOfOrder {
                current_revision: 2,
                incoming_revision: 1,
            }
        );
    }
}
