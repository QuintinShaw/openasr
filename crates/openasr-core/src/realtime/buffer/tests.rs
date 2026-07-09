#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::audio::{RealtimeAudioFormat, RealtimeAudioFrame};

    fn frame(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
        frame_with_duration(seq, start_ms, 20)
    }

    fn frame_with_duration(seq: u64, start_ms: u64, duration_ms: u32) -> RealtimeAudioFrame {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(duration_ms).unwrap();
        RealtimeAudioFrame::new(seq, start_ms, format, vec![0; sample_count]).unwrap()
    }

    fn config() -> RealtimeBufferConfig {
        RealtimeBufferConfig {
            frame_duration_ms: 20,
            pre_roll_ms: 40,
            max_buffered_frames: 10,
            max_buffered_samples: 10_000,
        }
    }

    #[test]
    fn includes_configured_pre_roll() {
        let mut buffer = RealtimeBuffer::new(config()).unwrap();
        assert!(buffer.push_frame(frame(1, 0), &[]).unwrap().is_empty());
        assert!(buffer.push_frame(frame(2, 20), &[]).unwrap().is_empty());
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(3, 40),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: utterance_id.clone(),
                    start_ms: 40,
                }],
            )
            .unwrap();
        let utterances = buffer
            .push_frame(
                frame(4, 60),
                &[SpeechBoundaryEvent::SpeechStopped {
                    utterance_id,
                    start_ms: 40,
                    end_ms: 80,
                }],
            )
            .unwrap();
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].frames.len(), 4);
        assert_eq!(utterances[0].start_ms, 0);
        assert_eq!(utterances[0].end_ms, 80);
        assert_eq!(utterances[0].reason, RealtimeUtteranceEndReason::VadStop);
    }

    #[test]
    fn trims_pre_roll_by_actual_frame_duration() {
        let mut buffer = RealtimeBuffer::new(config()).unwrap();
        for index in 0..5 {
            buffer
                .push_frame(frame_with_duration(index + 1, index * 10, 10), &[])
                .unwrap();
        }

        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame_with_duration(6, 50, 10),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: utterance_id.clone(),
                    start_ms: 50,
                }],
            )
            .unwrap();
        let utterances = buffer
            .push_frame(
                frame_with_duration(7, 60, 10),
                &[SpeechBoundaryEvent::SpeechStopped {
                    utterance_id,
                    start_ms: 50,
                    end_ms: 70,
                }],
            )
            .unwrap();

        assert_eq!(
            utterances[0]
                .frames
                .iter()
                .map(|frame| frame.start_ms)
                .collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50, 60]
        );
    }

    #[test]
    fn pre_roll_overflow_returns_typed_error_before_speech() {
        let mut buffer = RealtimeBuffer::new(RealtimeBufferConfig {
            pre_roll_ms: 100,
            max_buffered_frames: 2,
            ..config()
        })
        .unwrap();
        buffer.push_frame(frame(1, 0), &[]).unwrap();
        buffer.push_frame(frame(2, 20), &[]).unwrap();
        let error = buffer.push_frame(frame(3, 40), &[]).unwrap_err();

        assert!(matches!(
            error,
            RealtimeBufferError::AudioBufferOverflow {
                buffered_frames: 3,
                max_buffered_frames: 2,
                ..
            }
        ));
        assert!(buffer.last_overflow().is_some());
        assert_eq!(buffer.pre_roll.len(), 2);
    }

    #[test]
    fn consumes_pre_roll_when_utterance_starts() {
        let mut buffer = RealtimeBuffer::new(config()).unwrap();
        buffer.push_frame(frame(1, 0), &[]).unwrap();
        buffer.push_frame(frame(2, 20), &[]).unwrap();

        let first_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(3, 40),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: first_id.clone(),
                    start_ms: 40,
                }],
            )
            .unwrap();
        buffer
            .push_frame(
                frame(4, 60),
                &[SpeechBoundaryEvent::SpeechStopped {
                    utterance_id: first_id,
                    start_ms: 40,
                    end_ms: 80,
                }],
            )
            .unwrap();

        buffer.push_frame(frame(5, 80), &[]).unwrap();
        let second_id = TranscriptUtteranceId("utt_2".to_string());
        buffer
            .push_frame(
                frame(6, 100),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: second_id.clone(),
                    start_ms: 100,
                }],
            )
            .unwrap();
        let utterances = buffer
            .push_frame(
                frame(7, 120),
                &[SpeechBoundaryEvent::SpeechStopped {
                    utterance_id: second_id,
                    start_ms: 100,
                    end_ms: 140,
                }],
            )
            .unwrap();

        assert_eq!(utterances[0].start_ms, 80);
        assert_eq!(
            utterances[0]
                .frames
                .iter()
                .map(|frame| frame.start_ms)
                .collect::<Vec<_>>(),
            vec![80, 100, 120]
        );
    }

    #[test]
    fn returns_typed_overflow_error() {
        let mut buffer = RealtimeBuffer::new(RealtimeBufferConfig {
            max_buffered_frames: 2,
            ..config()
        })
        .unwrap();
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(1, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id,
                    start_ms: 0,
                }],
            )
            .unwrap();
        buffer.push_frame(frame(2, 20), &[]).unwrap();
        let error = buffer.push_frame(frame(3, 40), &[]).unwrap_err();
        assert!(matches!(
            error,
            RealtimeBufferError::AudioBufferOverflow {
                buffered_frames: 3,
                max_buffered_frames: 2,
                ..
            }
        ));
        assert!(buffer.last_overflow().is_some());
    }

    #[test]
    fn max_utterance_boundary_finishes_with_reason() {
        let mut buffer = RealtimeBuffer::new(config()).unwrap();
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(1, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: utterance_id.clone(),
                    start_ms: 0,
                }],
            )
            .unwrap();
        let utterances = buffer
            .push_frame(
                frame(2, 20),
                &[SpeechBoundaryEvent::MaxUtterance {
                    utterance_id,
                    start_ms: 0,
                    end_ms: 40,
                }],
            )
            .unwrap();
        assert_eq!(
            utterances[0].reason,
            RealtimeUtteranceEndReason::MaxUtterance
        );
    }

    #[test]
    fn reset_and_cancel_clear_active_buffer() {
        let mut buffer = RealtimeBuffer::new(config()).unwrap();
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(1, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id,
                    start_ms: 0,
                }],
            )
            .unwrap();
        let cancelled = buffer.cancel(20).unwrap();
        assert_eq!(cancelled.reason, RealtimeUtteranceEndReason::Cancel);
        assert!(buffer.flush(20).is_none());
        buffer.reset();
        assert!(buffer.last_overflow().is_none());
    }

    /// Regression test for the active-frame ingest capacity check: previously
    /// `push_active` cloned the whole accumulated frame `Vec` on every single
    /// frame push just to size-check it (O(n) clone per push -> O(n^2)
    /// memcpy over a long utterance). It now projects the post-push
    /// frame/sample counts from a running counter instead. This asserts the
    /// frame-count-based overflow decision is unchanged.
    #[test]
    fn active_frame_overflow_matches_pre_fix_boundary() {
        let mut buffer = RealtimeBuffer::new(RealtimeBufferConfig {
            max_buffered_frames: 2,
            ..config()
        })
        .unwrap();
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(1, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id,
                    start_ms: 0,
                }],
            )
            .unwrap();
        // Second frame is exactly at capacity: must still succeed.
        buffer.push_frame(frame(2, 20), &[]).unwrap();
        // Third frame pushes frame count past the limit: must fail, with the
        // same reported (buffered_frames, buffered_samples) as before.
        let error = buffer.push_frame(frame(3, 40), &[]).unwrap_err();
        assert!(matches!(
            error,
            RealtimeBufferError::AudioBufferOverflow {
                buffered_frames: 3,
                max_buffered_frames: 2,
                buffered_samples: 960,
                max_buffered_samples: 10_000,
            }
        ));
        assert!(buffer.last_overflow().is_some());
    }

    /// Companion regression test exercising the *sample*-count branch of the
    /// same capacity check (frame count stays well under its limit), so the
    /// running-sample-counter projection introduced by the fix is verified
    /// independently of the frame-count projection.
    #[test]
    fn active_sample_overflow_matches_pre_fix_boundary() {
        let mut buffer = RealtimeBuffer::new(RealtimeBufferConfig {
            max_buffered_frames: 100,
            // Two 20ms@16kHz frames = 640 samples fits; a third (960) does not.
            max_buffered_samples: 700,
            ..config()
        })
        .unwrap();
        let utterance_id = TranscriptUtteranceId("utt_1".to_string());
        buffer
            .push_frame(
                frame(1, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id,
                    start_ms: 0,
                }],
            )
            .unwrap();
        buffer.push_frame(frame(2, 20), &[]).unwrap();
        let error = buffer.push_frame(frame(3, 40), &[]).unwrap_err();
        assert!(matches!(
            error,
            RealtimeBufferError::AudioBufferOverflow {
                buffered_frames: 3,
                max_buffered_frames: 100,
                buffered_samples: 960,
                max_buffered_samples: 700,
            }
        ));
        assert!(buffer.last_overflow().is_some());
    }

    /// Perf regression guard: active-utterance ingest must stay linear in
    /// the number of frames pushed. Before the fix, `push_active` cloned the
    /// entire accumulated frame buffer (including each frame's owned sample
    /// Vec) on every push, so a long utterance triggered O(n^2) memcpy (a 30s
    /// utterance was ~730MB of copying). This pushes several thousand frames
    /// and bounds the wall-clock time generously above what the O(n) fixed
    /// path takes, so a reintroduced quadratic clone would fail this test
    /// long before it gets anywhere near production-scale utterances.
    #[test]
    fn active_ingest_stays_linear_for_long_utterances() {
        use std::time::Instant;

        let long_config = RealtimeBufferConfig {
            frame_duration_ms: 20,
            pre_roll_ms: 0,
            max_buffered_frames: 5_000,
            max_buffered_samples: 5_000 * 320,
        };
        let mut buffer = RealtimeBuffer::new(long_config).unwrap();
        let utterance_id = TranscriptUtteranceId("utt_long".to_string());
        buffer
            .push_frame(
                frame(0, 0),
                &[SpeechBoundaryEvent::SpeechStarted {
                    utterance_id: utterance_id.clone(),
                    start_ms: 0,
                }],
            )
            .unwrap();

        const EXTRA_FRAMES: u64 = 3_000;
        let started = Instant::now();
        for seq in 1..=EXTRA_FRAMES {
            buffer.push_frame(frame(seq, seq * 20), &[]).unwrap();
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "active-frame ingest took {elapsed:?} for {EXTRA_FRAMES} frames; \
             this looks like a reintroduced O(n^2) clone in push_active"
        );

        let utterances = buffer
            .push_frame(
                frame(EXTRA_FRAMES + 1, (EXTRA_FRAMES + 1) * 20),
                &[SpeechBoundaryEvent::SpeechStopped {
                    utterance_id,
                    start_ms: 0,
                    end_ms: (EXTRA_FRAMES + 2) * 20,
                }],
            )
            .unwrap();
        assert_eq!(utterances[0].frames.len(), (EXTRA_FRAMES + 2) as usize);
    }
}
