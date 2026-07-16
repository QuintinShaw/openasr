#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::buffer::{RealtimeBuffer, RealtimeBufferConfig};
    use crate::realtime::vad::{VadConfig, VadStateMachine};

    fn engine(sample_rate_hz: u32, channels: u16, frame_duration_ms: u32) -> CaptureEngine {
        let input = CaptureInputFormat::new(sample_rate_hz, channels).unwrap();
        CaptureEngine::new(input, frame_duration_ms).unwrap()
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let error = CaptureInputFormat::new(0, 1).unwrap_err();
        assert_eq!(error, CaptureEngineError::ZeroSampleRate);
        assert!(error.to_string().contains("sample rate"));
    }

    #[test]
    fn rejects_zero_channels() {
        let error = CaptureInputFormat::new(16_000, 0).unwrap_err();
        assert_eq!(error, CaptureEngineError::ZeroChannels);
        assert!(error.to_string().contains("channel"));
    }

    #[test]
    fn engine_new_rejects_zero_fields_even_via_struct_literal() {
        // Fields are public for platform glue that builds this from a
        // device-reported config struct directly, bypassing `new()`.
        let bypassed = CaptureInputFormat {
            sample_rate_hz: 0,
            channels: 1,
        };
        assert_eq!(
            CaptureEngine::new(bypassed, 20).unwrap_err(),
            CaptureEngineError::ZeroSampleRate
        );
    }

    #[test]
    fn rejects_unsupported_frame_duration() {
        let input = CaptureInputFormat::new(16_000, 1).unwrap();
        let error = CaptureEngine::new(input, 25).unwrap_err();
        assert!(error.to_string().contains("frame duration"));
    }

    #[test]
    fn rejects_chunk_not_divisible_by_channel_count() {
        let mut capture = engine(16_000, 2, 20);
        let error = capture.push_f32_interleaved(&[0.0, 0.1, 0.2]).unwrap_err();
        assert!(error.to_string().contains("not divisible"));
        assert_eq!(
            error,
            CaptureEngineError::NonMultipleOfChannels {
                sample_count: 3,
                channels: 2
            }
        );
    }

    #[test]
    fn passes_through_16khz_mono_and_splits_frames() {
        let mut capture = engine(16_000, 1, 20);
        let samples = vec![0.5_f32; 640];
        let frames = capture.push_f32_interleaved(&samples).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].seq, 1);
        assert_eq!(frames[0].start_ms, 0);
        assert_eq!(frames[0].samples()[0], 16384);
        assert_eq!(frames[1].seq, 2);
        assert_eq!(frames[1].start_ms, 20);
        assert_eq!(capture.next_frame_start_ms(), 40);
    }

    #[test]
    fn resamples_multiple_input_rates_to_16khz_within_tolerance() {
        // Each candidate input rate a mobile/desktop mic is realistically
        // negotiated at: 8 kHz (telephony-grade), 16 kHz (pass-through),
        // 22.05 kHz, 44.1 kHz, and 48 kHz (the common device default).
        for &input_rate in &[8_000_u32, 16_000, 22_050, 44_100, 48_000] {
            let mut capture = engine(input_rate, 1, 20);
            let one_second: Vec<f32> = (0..input_rate)
                .map(|index| (index as f32 / input_rate as f32 * std::f32::consts::TAU * 440.0).sin() * 0.5)
                .collect();
            let frames = capture.push_f32_interleaved(&one_second).unwrap();
            let produced_samples: usize = frames.iter().map(|frame| frame.sample_count()).sum();
            // 1 second of input should resample to ~16_000 output samples
            // (minus whatever remains buffered in a partial frame / the
            // resampler's fractional tail).
            let tolerance = 16_000 / 20; // one frame's worth of slack
            assert!(
                produced_samples.abs_diff(16_000) <= tolerance,
                "input_rate={input_rate}: expected ~16000 output samples, got {produced_samples}"
            );
            for frame in &frames {
                assert_eq!(frame.format, RealtimeAudioFormat::pcm16_mono_16khz());
            }
        }
    }

    #[test]
    fn resamples_48khz_mono_to_16khz_frames_exactly() {
        let mut capture = engine(48_000, 1, 20);
        let samples = vec![0.25_f32; 960];
        let frames = capture.push_f32_interleaved(&samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_count(), 320);
        assert!(frames[0].samples().iter().all(|sample| *sample == 8192));
    }

    #[test]
    fn downmixes_stereo_and_converts_i16_u16() {
        let mut i16_capture = engine(16_000, 2, 20);
        let i16_samples = (0..320)
            .flat_map(|_| [32767_i16, -32768_i16])
            .collect::<Vec<_>>();
        let frames = i16_capture.push_i16_interleaved(&i16_samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].samples()[0].abs() <= 1);

        let mut u16_capture = engine(16_000, 1, 20);
        let u16_samples = vec![u16::MAX; 320];
        let frames = u16_capture.push_u16_interleaved(&u16_samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples()[0], 32766);
    }

    #[test]
    fn carries_partial_samples_across_chunks() {
        let mut capture = engine(16_000, 1, 20);
        assert!(
            capture
                .push_f32_interleaved(&vec![0.1_f32; 200])
                .unwrap()
                .is_empty()
        );
        let frames = capture.push_f32_interleaved(&vec![0.1_f32; 120]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_count(), 320);
    }

    #[test]
    fn push_dispatches_on_capture_sample_variant() {
        let mut capture = engine(16_000, 1, 20);
        let frames = capture.push(CaptureSample::F32(vec![0.5_f32; 320])).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples()[0], 16384);
    }

    // -- CaptureBackpressureQueue -------------------------------------------------

    #[test]
    fn queue_rejects_zero_capacity() {
        let error = CaptureBackpressureQueue::<i32>::new(0).unwrap_err();
        assert_eq!(error, CaptureEngineError::ZeroQueueCapacity);
    }

    #[test]
    fn queue_accepts_up_to_capacity_then_overflows_without_evicting() {
        let queue = CaptureBackpressureQueue::new(2).unwrap();
        assert_eq!(queue.try_push(1), CapturePushOutcome::Accepted);
        assert_eq!(queue.try_push(2), CapturePushOutcome::Accepted);
        assert_eq!(queue.try_push(3), CapturePushOutcome::Overflowed);
        // The third item was dropped, not swapped in for the oldest.
        assert_eq!(queue.len(), 2);
        assert!(queue.take_overflowed());
        // The overflow flag is sticky until read, then clears.
        assert!(!queue.take_overflowed());
    }

    #[test]
    fn queue_drains_in_fifo_order_and_empties() {
        let queue = CaptureBackpressureQueue::new(4).unwrap();
        queue.try_push("a");
        queue.try_push("b");
        queue.try_push("c");
        assert_eq!(queue.drain(), vec!["a", "b", "c"]);
        assert!(queue.is_empty());
        assert_eq!(queue.drain(), Vec::<&str>::new());
    }

    #[test]
    fn queue_is_send_and_sync_for_cross_thread_audio_callback_handoff() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CaptureBackpressureQueue<CaptureSample>>();
    }

    // -- VAD / RealtimeBuffer integration point ------------------------------------

    /// Proves the engine's normalized output frames need zero adaptation to
    /// drive the existing `VadStateMachine` + `RealtimeBuffer` pair that
    /// desktop live mode and the server realtime session already use --
    /// this *is* the "capture engine talks to VadStateMachine/RealtimeBuffer"
    /// contract the mobile capture engine plan calls for.
    #[test]
    fn engine_output_frames_drive_vad_and_buffer_to_a_completed_utterance() {
        let mut capture = engine(48_000, 2, 20); // stereo 48 kHz, like a typical device mic.
        let mut vad = VadStateMachine::new(VadConfig {
            frame_duration_ms: 20,
            speech_start_ms: 40,
            speech_stop_ms: 40,
            pre_roll_ms: 20,
            max_utterance_ms: Some(5_000),
            no_speech_timeout_ms: None,
            mode: crate::realtime::VadMode::Energy,
            energy_threshold: 0.02,
        })
        .unwrap();
        let mut buffer = RealtimeBuffer::new(RealtimeBufferConfig {
            frame_duration_ms: 20,
            pre_roll_ms: 20,
            max_buffered_frames: 1_000,
            max_buffered_samples: 320_000,
        })
        .unwrap();

        let silence_chunk = vec![0.0_f32; 20 * 48_000 / 1_000 * 2]; // 20ms stereo
        let loud_chunk: Vec<f32> = (0..20 * 48_000 / 1_000 * 2)
            .map(|index| {
                (index as f32 / 48_000.0 * std::f32::consts::TAU * 440.0).sin() * 0.8
            })
            .collect();

        let mut completed = Vec::new();
        let mut started = false;
        let mut push_and_drive = |capture: &mut CaptureEngine, chunk: &[f32]| {
            for frame in capture.push_f32_interleaved(chunk).unwrap() {
                let boundaries = vad.process_energy_frame(&frame);
                if boundaries
                    .iter()
                    .any(|event| matches!(event, crate::realtime::SpeechBoundaryEvent::SpeechStarted { .. }))
                {
                    started = true;
                }
                completed.extend(buffer.push_frame(frame, &boundaries).unwrap());
            }
        };

        // A few silent frames (pre-roll), then enough loud frames to cross
        // speech_start_ms, then silence again to cross speech_stop_ms.
        for _ in 0..3 {
            push_and_drive(&mut capture, &silence_chunk);
        }
        for _ in 0..5 {
            push_and_drive(&mut capture, &loud_chunk);
        }
        for _ in 0..5 {
            push_and_drive(&mut capture, &silence_chunk);
        }

        assert!(started, "expected the VAD to detect speech start");
        assert!(
            !completed.is_empty(),
            "expected at least one completed utterance out of RealtimeBuffer"
        );
        assert!(completed[0].sample_count() > 0);
    }
}
