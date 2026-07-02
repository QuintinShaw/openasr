#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::audio::{RealtimeAudioFormat, RealtimeAudioFrame};

    fn frame(seq: u64, start_ms: u64, sample: i16) -> RealtimeAudioFrame {
        frame_with_duration(seq, start_ms, 20, sample)
    }

    fn frame_with_duration(
        seq: u64,
        start_ms: u64,
        duration_ms: u32,
        sample: i16,
    ) -> RealtimeAudioFrame {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(duration_ms).unwrap();
        RealtimeAudioFrame::new(seq, start_ms, format, vec![sample; sample_count]).unwrap()
    }

    fn config() -> VadConfig {
        VadConfig {
            frame_duration_ms: 20,
            speech_start_ms: 40,
            speech_stop_ms: 60,
            pre_roll_ms: 40,
            max_utterance_ms: Some(100),
            no_speech_timeout_ms: Some(80),
            mode: VadMode::Energy,
            energy_threshold: 0.02,
        }
    }

    #[test]
    fn speech_start_requires_debounce() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        assert!(
            vad.process_decision(&frame(1, 0, 1000), VadFrameDecision::speech())
                .is_empty()
        );
        let events = vad.process_decision(&frame(2, 20, 1000), VadFrameDecision::speech());
        assert!(matches!(
            events.as_slice(),
            [SpeechBoundaryEvent::SpeechStarted { start_ms: 0, .. }]
        ));
    }

    #[test]
    fn speech_start_uses_actual_frame_duration() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        for index in 0..3 {
            assert!(
                vad.process_decision(
                    &frame_with_duration(index + 1, index * 10, 10, 1000),
                    VadFrameDecision::speech()
                )
                .is_empty()
            );
        }
        let events = vad.process_decision(
            &frame_with_duration(4, 30, 10, 1000),
            VadFrameDecision::speech(),
        );
        assert!(matches!(
            events.as_slice(),
            [SpeechBoundaryEvent::SpeechStarted { start_ms: 0, .. }]
        ));
    }

    #[test]
    fn speech_stop_requires_debounce() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        vad.process_decision(&frame(1, 0, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(2, 20, 1000), VadFrameDecision::speech());
        assert!(
            vad.process_decision(&frame(3, 40, 0), VadFrameDecision::silence())
                .is_empty()
        );
        assert!(
            vad.process_decision(&frame(4, 60, 0), VadFrameDecision::silence())
                .is_empty()
        );
        let events = vad.process_decision(&frame(5, 80, 0), VadFrameDecision::silence());
        assert!(matches!(
            events.as_slice(),
            [SpeechBoundaryEvent::SpeechStopped { end_ms: 100, .. }]
        ));
    }

    #[test]
    fn speech_stop_uses_actual_frame_duration() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        vad.process_decision(&frame(1, 0, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(2, 20, 1000), VadFrameDecision::speech());
        assert!(
            vad.process_decision(
                &frame_with_duration(3, 40, 30, 0),
                VadFrameDecision::silence()
            )
            .is_empty()
        );
        let events = vad.process_decision(
            &frame_with_duration(4, 70, 30, 0),
            VadFrameDecision::silence(),
        );
        assert!(matches!(
            events.as_slice(),
            [SpeechBoundaryEvent::SpeechStopped { end_ms: 100, .. }]
        ));
    }

    #[test]
    fn max_utterance_flushes_active_speech() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        vad.process_decision(&frame(1, 0, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(2, 20, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(3, 40, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(4, 60, 1000), VadFrameDecision::speech());
        let events = vad.process_decision(&frame(5, 80, 1000), VadFrameDecision::speech());
        assert!(matches!(
            events.as_slice(),
            [SpeechBoundaryEvent::MaxUtterance { end_ms: 100, .. }]
        ));
    }

    #[test]
    fn emits_no_speech_timeout_once() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        for index in 0..3 {
            assert!(
                vad.process_decision(
                    &frame(index + 1, index * 20, 0),
                    VadFrameDecision::silence()
                )
                .is_empty()
            );
        }
        let events = vad.process_decision(&frame(4, 60, 0), VadFrameDecision::silence());
        assert_eq!(
            events,
            vec![SpeechBoundaryEvent::NoSpeechTimeout {
                timeout_ms: 80,
                at_ms: 80
            }]
        );
        assert!(
            vad.process_decision(&frame(5, 80, 0), VadFrameDecision::silence())
                .is_empty()
        );
    }

    #[test]
    fn no_speech_timeout_uses_waiting_window_after_reset() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        vad.reset();
        for index in 0..3 {
            assert!(
                vad.process_decision(
                    &frame(index + 1, 1_000 + index * 20, 0),
                    VadFrameDecision::silence()
                )
                .is_empty()
            );
        }
        let events = vad.process_decision(&frame(4, 1_060, 0), VadFrameDecision::silence());
        assert_eq!(
            events,
            vec![SpeechBoundaryEvent::NoSpeechTimeout {
                timeout_ms: 80,
                at_ms: 1_080
            }]
        );
    }

    #[test]
    fn no_speech_timeout_does_not_fire_after_speech_started() {
        let mut vad = VadStateMachine::new(config()).unwrap();
        vad.process_decision(&frame(1, 0, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(2, 20, 1000), VadFrameDecision::speech());
        vad.process_decision(&frame(3, 40, 0), VadFrameDecision::silence());
        vad.process_decision(&frame(4, 60, 0), VadFrameDecision::silence());
        vad.process_decision(&frame(5, 80, 0), VadFrameDecision::silence());

        assert!(
            vad.process_decision(&frame(6, 100, 0), VadFrameDecision::silence())
                .is_empty()
        );
    }

    #[test]
    fn energy_decision_is_deterministic() {
        let quiet = frame(1, 0, 10);
        let loud = frame(2, 20, 2000);
        assert_eq!(
            VadFrameDecision::from_energy(&quiet, 0.02).decision,
            VadDecision::Silence
        );
        assert_eq!(
            VadFrameDecision::from_energy(&loud, 0.02).decision,
            VadDecision::Speech
        );
    }
}
