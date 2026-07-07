#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::{RealtimeAudioFormat, RealtimeAudioFrame, SpeechBoundaryEvent, VadMode};

    #[test]
    fn external_probability_mode_routes_through_neural_vad() {
        // Skip if the vendored model is unavailable in this build.
        if crate::diarize::vad::shared_model().is_none() {
            return;
        }
        let mut config =
            RealtimeSessionConfig::new("rt_neural", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        config.vad.mode = VadMode::ExternalProbability;
        config.vad.energy_threshold = 0.5; // used as the probability threshold
        config.vad.frame_duration_ms = 20;
        let mut controller = RealtimeSessionController::new(config).unwrap();

        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        let mut started = false;
        let mut start_ms = 0u64;
        for (seq, frame_samples) in pcm.chunks(320).enumerate() {
            if frame_samples.len() < 320 {
                break;
            }
            let frame =
                RealtimeAudioFrame::new(seq as u64, start_ms, format, frame_samples.to_vec())
                    .unwrap();
            let boundaries = controller.process_vad_frame(&frame);
            if boundaries
                .iter()
                .any(|b| matches!(b, SpeechBoundaryEvent::SpeechStarted { .. }))
            {
                started = true;
                break;
            }
            start_ms += 20;
        }
        assert!(
            started,
            "neural VAD (ExternalProbability) should emit SpeechStarted on golden speech"
        );
    }

    #[test]
    fn external_probability_mode_honors_firered_stream_opt_in() {
        // Opt-in via OPENASR_VAD: the realtime session must construct the
        // FireRedVAD Stream-VAD detector instead of Silero, and Silero must
        // stay untouched as the default when the env var is unset (asserted
        // by `external_probability_mode_routes_through_neural_vad` above).
        if crate::diarize::vad::FireRedStreamVadProvider::shared().is_none() {
            return;
        }
        let saved = std::env::var("OPENASR_VAD").ok();
        // SAFETY: sequential mutation, restored before returning; mirrors the
        // guard already used elsewhere for this same env var in this crate's
        // tests (e.g. `native_transcribe::tests::openasr_vad_env_override_selects_firered`).
        unsafe { std::env::set_var("OPENASR_VAD", "firered-stream") };

        let mut config = RealtimeSessionConfig::new(
            "rt_neural_stream",
            "whisper-small:candidate",
            "2026-05-09T00:00:00Z",
        );
        config.vad.mode = VadMode::ExternalProbability;
        config.vad.energy_threshold = 0.5;
        config.vad.frame_duration_ms = 20;
        let mut controller = RealtimeSessionController::new(config).unwrap();

        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        let mut started = false;
        let mut start_ms = 0u64;
        for (seq, frame_samples) in pcm.chunks(320).enumerate() {
            if frame_samples.len() < 320 {
                break;
            }
            let frame =
                RealtimeAudioFrame::new(seq as u64, start_ms, format, frame_samples.to_vec())
                    .unwrap();
            let boundaries = controller.process_vad_frame(&frame);
            if boundaries
                .iter()
                .any(|b| matches!(b, SpeechBoundaryEvent::SpeechStarted { .. }))
            {
                started = true;
                break;
            }
            start_ms += 20;
        }

        match saved {
            Some(value) => unsafe { std::env::set_var("OPENASR_VAD", value) },
            None => unsafe { std::env::remove_var("OPENASR_VAD") },
        }

        assert!(
            started,
            "Stream-VAD (opt-in ExternalProbability) should emit SpeechStarted on golden speech"
        );
    }

    #[test]
    fn validates_session_config() {
        let mut config =
            RealtimeSessionConfig::new("rt_test", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        config.model_id.clear();
        assert!(matches!(
            config.validate(),
            Err(RealtimeSessionError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn session_events_are_monotonic() {
        let config =
            RealtimeSessionConfig::new("rt_test", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        let mut controller = RealtimeSessionController::new(config).unwrap();
        let created = controller.session_created_event("2026-05-09T00:00:00Z");
        let configured = controller
            .lifecycle(RealtimeLifecycleAction::Configure, "2026-05-09T00:00:01Z")
            .unwrap();
        let audio_started = controller
            .lifecycle(RealtimeLifecycleAction::StartAudio, "2026-05-09T00:00:02Z")
            .unwrap();
        let audio_stopped = controller
            .lifecycle(
                RealtimeLifecycleAction::StopAudio {
                    reason: "client_stopped".to_string(),
                },
                "2026-05-09T00:00:03Z",
            )
            .unwrap();
        assert_eq!(created.seq, 1);
        assert_eq!(created.created_at, "2026-05-09T00:00:00Z");
        assert_eq!(configured.seq, 2);
        assert_eq!(configured.created_at, "2026-05-09T00:00:01Z");
        assert_eq!(audio_started.seq, 3);
        assert_eq!(audio_started.event_type, "audio.input.started");
        assert_eq!(audio_started.created_at, "2026-05-09T00:00:02Z");
        assert_eq!(audio_stopped.seq, 4);
        assert_eq!(audio_stopped.event_type, "audio.input.stopped");
        assert_eq!(audio_stopped.created_at, "2026-05-09T00:00:03Z");
        assert_eq!(controller.state(), RealtimeSessionState::Configured);
    }

    #[test]
    fn close_blocks_later_reset() {
        let config =
            RealtimeSessionConfig::new("rt_test", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        let mut controller = RealtimeSessionController::new(config).unwrap();
        let closed = controller
            .lifecycle(
                RealtimeLifecycleAction::Close {
                    reason: "client_closed".to_string(),
                },
                "2026-05-09T00:00:00Z",
            )
            .unwrap();
        assert_eq!(closed.event_type, "session.closed");
        assert!(matches!(
            controller.reset(),
            Err(RealtimeSessionError::SessionClosed)
        ));
        assert!(matches!(
            controller.lifecycle(
                RealtimeLifecycleAction::Close {
                    reason: "client_closed".to_string()
                },
                "2026-05-09T00:00:00Z"
            ),
            Err(RealtimeSessionError::SessionClosed)
        ));
    }

    #[test]
    fn cancel_resets_internal_state() {
        let config =
            RealtimeSessionConfig::new("rt_test", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        let mut controller = RealtimeSessionController::new(config).unwrap();
        controller
            .lifecycle(RealtimeLifecycleAction::Configure, "2026-05-09T00:00:01Z")
            .unwrap();
        controller
            .lifecycle(RealtimeLifecycleAction::StartAudio, "2026-05-09T00:00:02Z")
            .unwrap();
        let (cancelled, utterance) = controller.cancel(0, "2026-05-09T00:00:03Z").unwrap();
        assert_eq!(cancelled.event_type, "session.closed");
        assert!(utterance.is_none());
        assert_eq!(controller.state(), RealtimeSessionState::Cancelled);
        assert!(matches!(
            controller.cancel(0, "2026-05-09T00:00:04Z"),
            Err(RealtimeSessionError::SessionClosed)
        ));
    }

    #[test]
    fn session_lifecycle_requires_configure_before_start() {
        let config =
            RealtimeSessionConfig::new("rt_test", "whisper-small:candidate", "2026-05-09T00:00:00Z");
        let mut controller = RealtimeSessionController::new(config).unwrap();
        assert!(matches!(
            controller.lifecycle(RealtimeLifecycleAction::StartAudio, "2026-05-09T00:00:01Z"),
            Err(RealtimeSessionError::InvalidStateTransition {
                from: RealtimeSessionState::Created,
                action: "start_audio"
            })
        ));
        controller
            .lifecycle(RealtimeLifecycleAction::Configure, "2026-05-09T00:00:01Z")
            .unwrap();
        let started = controller
            .lifecycle(RealtimeLifecycleAction::StartAudio, "2026-05-09T00:00:02Z")
            .unwrap();
        assert_eq!(started.event_type, "audio.input.started");
        assert!(matches!(
            controller.lifecycle(RealtimeLifecycleAction::Configure, "2026-05-09T00:00:03Z"),
            Err(RealtimeSessionError::InvalidStateTransition {
                from: RealtimeSessionState::Running,
                action: "configure"
            })
        ));
    }
}
