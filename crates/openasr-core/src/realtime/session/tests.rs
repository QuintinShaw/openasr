#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::realtime::{RealtimeAudioFormat, RealtimeAudioFrame, SpeechBoundaryEvent, VadMode};

    fn neural_vad_config_with_frame_ms(
        session_id: &str,
        frame_duration_ms: u32,
    ) -> RealtimeSessionConfig {
        let mut config = RealtimeSessionConfig::new(
            session_id,
            "whisper-small:candidate",
            "2026-05-09T00:00:00Z",
        );
        config.vad.mode = VadMode::ExternalProbability;
        config.vad.energy_threshold = 0.5;
        config.vad.frame_duration_ms = frame_duration_ms;
        config
    }

    fn neural_vad_config(session_id: &str) -> RealtimeSessionConfig {
        neural_vad_config_with_frame_ms(session_id, 20)
    }

    fn collect_golden_boundaries(
        controller: &mut RealtimeSessionController,
    ) -> Vec<SpeechBoundaryEvent> {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let frame_duration_ms = controller.config().vad.frame_duration_ms;
        let frame_samples = format
            .sample_count_for_duration_ms(frame_duration_ms)
            .unwrap();
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        let mut boundaries = Vec::new();
        for (seq, samples) in pcm.chunks(frame_samples).enumerate() {
            if samples.len() < frame_samples {
                break;
            }
            let frame = RealtimeAudioFrame::new(
                seq as u64,
                seq as u64 * u64::from(frame_duration_ms),
                format,
                samples.to_vec(),
            )
            .unwrap();
            boundaries.extend(controller.process_vad_frame(&frame).unwrap());
        }
        boundaries
    }

    fn boundary_geometry(boundary: &SpeechBoundaryEvent) -> (&'static str, u64, u64) {
        match boundary {
            SpeechBoundaryEvent::SpeechStarted { start_ms, .. } => ("start", *start_ms, *start_ms),
            SpeechBoundaryEvent::SpeechStopped {
                start_ms, end_ms, ..
            } => ("stop", *start_ms, *end_ms),
            SpeechBoundaryEvent::MaxUtterance {
                start_ms, end_ms, ..
            } => ("max", *start_ms, *end_ms),
            SpeechBoundaryEvent::NoSpeechTimeout { timeout_ms, at_ms } => {
                ("timeout", u64::from(*timeout_ms), *at_ms)
            }
        }
    }

    #[test]
    fn external_probability_mode_routes_through_stream_vad() {
        // Skip if the vendored model is unavailable in this build.
        if crate::diarize::vad::shared_model().is_none() {
            return;
        }
        let config = neural_vad_config("rt_neural");
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
            let boundaries = controller.process_vad_frame(&frame).unwrap();
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
            "Stream-VAD (ExternalProbability) should emit SpeechStarted on golden speech"
        );
    }

    #[test]
    fn reset_clears_neural_vad_frontend_and_dfsmn_state() {
        let mut controller =
            RealtimeSessionController::new(neural_vad_config("rt_neural_reset")).unwrap();
        controller
            .lifecycle(
                RealtimeLifecycleAction::Configure,
                "2026-05-09T00:00:01Z",
            )
            .unwrap();
        let first = collect_golden_boundaries(&mut controller);
        controller.reset().unwrap();
        let second = collect_golden_boundaries(&mut controller);
        assert_eq!(
            first.iter().map(boundary_geometry).collect::<Vec<_>>(),
            second.iter().map(boundary_geometry).collect::<Vec<_>>()
        );
        assert!(first.iter().any(|boundary| matches!(
            boundary,
            SpeechBoundaryEvent::SpeechStarted { .. }
        )));
    }

    #[test]
    fn explicit_metal_realtime_vad_matches_host_and_reports_only_metal_compute() {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return;
        }
        let services = Arc::new(
            crate::NativeExecutionServices::for_local_process()
                .expect("test execution services must construct"),
        );
        let telemetry = crate::GgmlExecutionTelemetryCollector::new();
        let _telemetry = telemetry.install();

        for frame_ms in [10_u32, 20, 30] {
            let mut host = RealtimeSessionController::new(neural_vad_config_with_frame_ms(
                &format!("rt_neural_host_{frame_ms}"),
                frame_ms,
            ))
            .unwrap();
            let expected = collect_golden_boundaries(&mut host);
            let mut metal = RealtimeSessionController::new_with_execution(
                neural_vad_config_with_frame_ms(
                    &format!("rt_neural_metal_{frame_ms}"),
                    frame_ms,
                ),
                Arc::clone(&services),
                crate::ExecutionTarget::Accelerated,
            )
            .unwrap();
            let first_actor = metal
                .neural_vad_actor_identity_for_test()
                .expect("explicit Metal owns an actor");
            assert_eq!(collect_golden_boundaries(&mut metal), expected);
            drop(metal);

            // Re-checkout must reuse the same idle actor for this cadence and
            // reset both frontend/cache state while replaying placement truth.
            let mut reused = RealtimeSessionController::new_with_execution(
                neural_vad_config_with_frame_ms(
                    &format!("rt_neural_metal_reused_{frame_ms}"),
                    frame_ms,
                ),
                Arc::clone(&services),
                crate::ExecutionTarget::Accelerated,
            )
            .unwrap();
            assert_eq!(
                reused.neural_vad_actor_identity_for_test(),
                Some(first_actor)
            );
            assert_eq!(collect_golden_boundaries(&mut reused), expected);
        }

        let observed = telemetry.snapshot();
        assert!(
            !observed.observed_compute_nodes_by_backend.is_empty(),
            "explicit Metal realtime VAD must execute a device graph"
        );
        assert!(
            observed
                .observed_compute_nodes_by_backend
                .keys()
                .all(|backend| backend.to_ascii_lowercase().contains("mtl")
                    || backend.to_ascii_lowercase().contains("metal")),
            "explicit Metal realtime VAD observed non-Metal compute: {:?}",
            observed.observed_compute_nodes_by_backend
        );
    }

    #[test]
    fn neural_vad_rejects_a_frame_that_disagrees_with_session_cadence() {
        let mut controller = RealtimeSessionController::new(neural_vad_config_with_frame_ms(
            "rt_neural_frame_mismatch",
            10,
        ))
        .unwrap();
        let frame = RealtimeAudioFrame::new(
            0,
            0,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            vec![0_i16; 480],
        )
        .unwrap();
        let error = controller.process_vad_frame(&frame).unwrap_err();
        assert!(matches!(error, RealtimeSessionError::StreamVadExecution(_)));
        assert!(error.to_string().contains("expected 160"));
    }

    #[test]
    fn terminal_controller_releases_accelerated_vad_checkout() {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return;
        }
        let services = Arc::new(
            crate::NativeExecutionServices::for_local_process()
                .expect("test execution services must construct"),
        );
        let mut first = RealtimeSessionController::new_with_execution(
            neural_vad_config("rt_neural_terminal_first"),
            Arc::clone(&services),
            crate::ExecutionTarget::Accelerated,
        )
        .unwrap();
        let actor = first
            .neural_vad_actor_identity_for_test()
            .expect("explicit Metal owns an actor");
        first
            .lifecycle(
                RealtimeLifecycleAction::Close {
                    reason: "test-complete".to_string(),
                },
                "2026-05-09T00:00:02Z",
            )
            .unwrap();
        assert_eq!(first.neural_vad_actor_identity_for_test(), None);

        let second = RealtimeSessionController::new_with_execution(
            neural_vad_config("rt_neural_terminal_second"),
            services,
            crate::ExecutionTarget::Accelerated,
        )
        .unwrap();
        assert_eq!(second.neural_vad_actor_identity_for_test(), Some(actor));
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
