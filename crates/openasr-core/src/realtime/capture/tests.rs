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
    fn resample_consumed_len_never_exceeds_held_samples() {
        assert_eq!(resample_consumed_len(513.3, 512), 512);
        assert_eq!(resample_consumed_len(512.0, 512), 512);
        assert_eq!(resample_consumed_len(100.2, 512), 100);
        assert_eq!(resample_consumed_len(0.9, 512), 0);
    }

    fn collect_pcm(frames: &[RealtimeAudioFrame]) -> Vec<i16> {
        frames
            .iter()
            .flat_map(|frame| frame.samples().iter().copied())
            .collect()
    }

    fn tone_interleaved(sample_rate_hz: u32, channels: u16, frames: usize) -> Vec<f32> {
        let channels = channels as usize;
        (0..frames * channels)
            .map(|index| {
                let frame = index / channels;
                (frame as f32 / sample_rate_hz as f32 * std::f32::consts::TAU * 440.0).sin() * 0.5
            })
            .collect()
    }

    fn chunked_resample_matches_oneshot(
        sample_rate_hz: u32,
        channels: u16,
        frames_per_chunk: usize,
        chunks: usize,
    ) {
        let tone = tone_interleaved(sample_rate_hz, channels, frames_per_chunk * chunks);
        let mut oneshot = engine(sample_rate_hz, channels, 20);
        let oneshot_pcm = collect_pcm(&oneshot.push_f32_interleaved(&tone).unwrap());

        let mut chunked = engine(sample_rate_hz, channels, 20);
        let mut chunked_pcm = Vec::new();
        let stride = frames_per_chunk * channels as usize;
        for chunk in tone.chunks(stride) {
            chunked_pcm.extend(collect_pcm(&chunked.push_f32_interleaved(chunk).unwrap()));
        }

        assert_eq!(
            chunked_pcm, oneshot_pcm,
            "chunked resample diverged from oneshot at {sample_rate_hz} Hz / {channels} ch / {frames_per_chunk}-frame callbacks"
        );
        assert!(
            !chunked_pcm.is_empty(),
            "expected at least one 16 kHz frame from {chunks} chunks"
        );
    }

    #[test]
    fn downsampling_read_head_may_overshoot_a_512_sample_chunk() {
        // Live capture commonly delivers 512 frames. After the first 44.1 kHz
        // chunk the leftover pos plus step>1 puts floor(pos) past the buffer
        // (513 vs 512). 48 kHz / 512 is the same class (step = 3).
        chunked_resample_matches_oneshot(44_100, 2, 512, 8);
        chunked_resample_matches_oneshot(44_100, 1, 512, 8);
        chunked_resample_matches_oneshot(48_000, 1, 512, 8);
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

    // -- Property tests (#352-class: live resample/downmix/framing) ---------------

    const PROP_RATES_HZ: [u32; 6] = [8_000, 16_000, 22_050, 44_100, 48_000, 96_000];
    const PROP_CHANNELS: [u16; 3] = [1, 2, 6];
    const OUTPUT_FRAME_SAMPLES: usize = 320; // 20 ms at 16 kHz

    #[derive(Clone, Copy, Debug)]
    enum LiveFormat {
        I16,
        F32,
        U16,
    }

    #[derive(Clone, Copy, Debug)]
    enum LiveDurationKind {
        Empty,
        OneInputFrame,
        PartialOutputFrame,
        HardwareCallback512,
        QuarterSecond,
    }

    const LIVE_DURATION_KINDS: [LiveDurationKind; 5] = [
        LiveDurationKind::Empty,
        LiveDurationKind::OneInputFrame,
        LiveDurationKind::PartialOutputFrame,
        LiveDurationKind::HardwareCallback512,
        LiveDurationKind::QuarterSecond,
    ];

    fn input_frames_for(kind: LiveDurationKind, sample_rate_hz: u32) -> usize {
        match kind {
            LiveDurationKind::Empty => 0,
            LiveDurationKind::OneInputFrame => 1,
            LiveDurationKind::PartialOutputFrame => {
                let full_output_frame =
                    (OUTPUT_FRAME_SAMPLES * sample_rate_hz as usize) / 16_000;
                full_output_frame.saturating_sub(1).clamp(1, 512)
            }
            LiveDurationKind::HardwareCallback512 => 512,
            LiveDurationKind::QuarterSecond => (sample_rate_hz as usize / 4).clamp(1, 48_000),
        }
    }

    fn expected_pcm16_len(input_frames: usize, input_rate_hz: u32) -> usize {
        if input_rate_hz == 16_000 {
            input_frames
        } else {
            input_frames.saturating_mul(16_000) / input_rate_hz as usize
        }
    }

    fn resample_len_slack(input_rate_hz: u32) -> usize {
        // Linear interpolation uses a 2-sample window and a fractional read
        // head; bound the leftover by one input-rate step in 16 kHz units.
        (input_rate_hz as usize / 16_000).max(1) + 2
    }

    fn interleaved_tone(sample_rate_hz: u32, channels: u16, frames: usize) -> Vec<f32> {
        let channels = channels as usize;
        (0..frames.saturating_mul(channels))
            .map(|index| {
                let frame = index / channels;
                (frame as f32 / sample_rate_hz.max(1) as f32 * std::f32::consts::TAU * 440.0).sin()
                    * 0.5
            })
            .collect()
    }

    fn push_live(
        engine: &mut CaptureEngine,
        format: LiveFormat,
        samples_f32: &[f32],
    ) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        match format {
            LiveFormat::F32 => engine.push_f32_interleaved(samples_f32),
            LiveFormat::I16 => {
                let samples: Vec<i16> = samples_f32
                    .iter()
                    .map(|sample| {
                        let clamped = sample.clamp(-1.0, 1.0);
                        (clamped * i16::MAX as f32).round() as i16
                    })
                    .collect();
                engine.push_i16_interleaved(&samples)
            }
            LiveFormat::U16 => {
                let samples: Vec<u16> = samples_f32
                    .iter()
                    .map(|sample| {
                        let clamped = sample.clamp(-1.0, 1.0);
                        (clamped.mul_add(32767.0, 32768.0)).round() as u16
                    })
                    .collect();
                engine.push_u16_interleaved(&samples)
            }
        }
    }

    fn assert_live_frames(
        frames: &[RealtimeAudioFrame],
        input_frames: usize,
        input_rate_hz: u32,
        context: &str,
    ) {
        let emitted: usize = frames.iter().map(RealtimeAudioFrame::sample_count).sum();
        for frame in frames {
            assert_eq!(
                frame.format,
                RealtimeAudioFormat::pcm16_mono_16khz(),
                "{context}: emitted frame is not 16 kHz mono pcm16"
            );
            assert_eq!(
                frame.sample_count(),
                OUTPUT_FRAME_SAMPLES,
                "{context}: 20 ms frame must contain {OUTPUT_FRAME_SAMPLES} samples"
            );
        }

        let expected = expected_pcm16_len(input_frames, input_rate_hz);
        let slack = resample_len_slack(input_rate_hz);
        let min_produced = expected.saturating_sub(slack);
        let max_produced = expected.saturating_add(slack);
        let min_emitted = (min_produced / OUTPUT_FRAME_SAMPLES) * OUTPUT_FRAME_SAMPLES;
        let max_emitted = (max_produced / OUTPUT_FRAME_SAMPLES) * OUTPUT_FRAME_SAMPLES;
        assert!(
            emitted >= min_emitted && emitted <= max_emitted,
            "{context}: emitted {emitted} pcm16 samples, expected frames in \
             [{min_emitted}, {max_emitted}] (produced ~{expected} +/- {slack}, \
             remainder stays in the engine)"
        );
        assert_eq!(
            emitted % OUTPUT_FRAME_SAMPLES,
            0,
            "{context}: partial frames must remain buffered, not emitted"
        );
    }

    fn run_live_case(
        sample_rate_hz: u32,
        channels: u16,
        format: LiveFormat,
        input_frames: usize,
        samples_f32: &[f32],
    ) -> Result<Vec<RealtimeAudioFrame>, String> {
        let context = format!(
            "rate={sample_rate_hz} ch={channels} format={format:?} frames={input_frames} len={}",
            samples_f32.len()
        );
        let mut capture = engine(sample_rate_hz, channels, 20);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            push_live(&mut capture, format, samples_f32)
        }));
        match caught {
            Ok(Ok(frames)) => {
                assert_live_frames(&frames, input_frames, sample_rate_hz, &context);
                Ok(frames)
            }
            Ok(Err(error)) => Err(format!("{context}: unexpected error {error}")),
            Err(_) => Err(format!(
                "{context} head={:?}: panicked",
                &samples_f32[..samples_f32.len().min(8)]
            )),
        }
    }

    #[test]
    fn live_capture_discrete_grid_does_not_panic() {
        for &sample_rate_hz in &PROP_RATES_HZ {
            for &channels in &PROP_CHANNELS {
                for format in [LiveFormat::I16, LiveFormat::F32, LiveFormat::U16] {
                    for kind in LIVE_DURATION_KINDS {
                        let input_frames = input_frames_for(kind, sample_rate_hz);
                        let samples =
                            interleaved_tone(sample_rate_hz, channels, input_frames);
                        run_live_case(
                            sample_rate_hz,
                            channels,
                            format,
                            input_frames,
                            &samples,
                        )
                        .unwrap_or_else(|error| panic!("{error}"));
                    }
                }
            }
        }
    }

    #[test]
    fn live_capture_non_multiple_of_channels_is_error_not_panic() {
        for &channels in &[2_u16, 6] {
            for extra in 1..channels {
                let samples = vec![0.1_f32; channels as usize * 8 + extra as usize];
                let mut capture = engine(16_000, channels, 20);
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    capture.push_f32_interleaved(&samples)
                }));
                let result = caught.unwrap_or_else(|_| {
                    panic!(
                        "rate=16000 ch={channels} leftover={extra} len={}: panicked",
                        samples.len()
                    )
                });
                assert_eq!(
                    result,
                    Err(CaptureEngineError::NonMultipleOfChannels {
                        sample_count: samples.len(),
                        channels,
                    })
                );
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 48,
            ..proptest::test_runner::Config::default()
        })]

        #[test]
        fn live_capture_random_f32_payload_does_not_panic(
            rate_idx in 0..PROP_RATES_HZ.len(),
            ch_idx in 0..PROP_CHANNELS.len(),
            kind_idx in 0..LIVE_DURATION_KINDS.len(),
            payload in proptest::collection::vec(proptest::num::f32::ANY, 0..=48_000),
        ) {
            let sample_rate_hz = PROP_RATES_HZ[rate_idx];
            let channels = PROP_CHANNELS[ch_idx];
            let input_frames = input_frames_for(LIVE_DURATION_KINDS[kind_idx], sample_rate_hz);
            let len = input_frames.saturating_mul(channels as usize);
            let mut samples = payload;
            samples.truncate(len);
            if samples.len() < len {
                samples.resize(len, 0.0);
            }
            run_live_case(
                sample_rate_hz,
                channels,
                LiveFormat::F32,
                input_frames,
                &samples,
            )
            .unwrap_or_else(|error| panic!("{error}"));
        }

        #[test]
        fn live_capture_random_i16_payload_does_not_panic(
            rate_idx in 0..PROP_RATES_HZ.len(),
            ch_idx in 0..PROP_CHANNELS.len(),
            kind_idx in 0..LIVE_DURATION_KINDS.len(),
            payload in proptest::collection::vec(proptest::num::i16::ANY, 0..=48_000),
        ) {
            let sample_rate_hz = PROP_RATES_HZ[rate_idx];
            let channels = PROP_CHANNELS[ch_idx];
            let input_frames = input_frames_for(LIVE_DURATION_KINDS[kind_idx], sample_rate_hz);
            let len = input_frames.saturating_mul(channels as usize);
            let mut samples = payload;
            samples.truncate(len);
            if samples.len() < len {
                samples.resize(len, 0);
            }
            let mut capture = engine(sample_rate_hz, channels, 20);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture.push_i16_interleaved(&samples)
            }));
            let frames = match caught {
                Ok(Ok(frames)) => frames,
                Ok(Err(error)) => panic!(
                    "rate={sample_rate_hz} ch={channels} i16 frames={input_frames}: {error}"
                ),
                Err(_) => panic!(
                    "rate={sample_rate_hz} ch={channels} i16 frames={input_frames} len={} head={:?}: panicked",
                    samples.len(),
                    &samples[..samples.len().min(8)]
                ),
            };
            assert_live_frames(
                &frames,
                input_frames,
                sample_rate_hz,
                &format!("rate={sample_rate_hz} ch={channels} i16 frames={input_frames}"),
            );
        }

        #[test]
        fn live_capture_random_u16_payload_does_not_panic(
            rate_idx in 0..PROP_RATES_HZ.len(),
            ch_idx in 0..PROP_CHANNELS.len(),
            payload in proptest::collection::vec(proptest::num::u16::ANY, 0..=4_000),
        ) {
            let sample_rate_hz = PROP_RATES_HZ[rate_idx];
            let channels = PROP_CHANNELS[ch_idx];
            let len = (payload.len() / channels as usize) * channels as usize;
            let samples = &payload[..len];
            let input_frames = samples.len() / channels as usize;
            let mut capture = engine(sample_rate_hz, channels, 20);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture.push_u16_interleaved(samples)
            }));
            let frames = match caught {
                Ok(Ok(frames)) => frames,
                Ok(Err(error)) => panic!(
                    "rate={sample_rate_hz} ch={channels} u16 frames={input_frames}: {error}"
                ),
                Err(_) => panic!(
                    "rate={sample_rate_hz} ch={channels} u16 frames={input_frames} len={} head={:?}: panicked",
                    samples.len(),
                    &samples[..samples.len().min(8)]
                ),
            };
            assert_live_frames(
                &frames,
                input_frames,
                sample_rate_hz,
                &format!("rate={sample_rate_hz} ch={channels} u16 frames={input_frames}"),
            );
        }

        #[test]
        fn live_capture_non_multiple_of_channels_proptest(
            ch_idx in 0..2usize,
            frames in 0usize..=1_024,
            extra in 1u16..=5,
            payload in proptest::collection::vec(proptest::num::f32::ANY, 0..=8_192),
        ) {
            let channels = [2_u16, 6][ch_idx];
            let leftover = (extra % channels).max(1);
            let len = frames.saturating_mul(channels as usize) + leftover as usize;
            let mut samples = payload;
            samples.truncate(len);
            if samples.len() < len {
                samples.resize(len, 0.0);
            }
            let mut capture = engine(44_100, channels, 20);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture.push_f32_interleaved(&samples)
            }));
            let result = match caught {
                Ok(result) => result,
                Err(_) => panic!(
                    "rate=44100 ch={channels} leftover={leftover} len={}: panicked",
                    samples.len()
                ),
            };
            proptest::prop_assert_eq!(
                result,
                Err(CaptureEngineError::NonMultipleOfChannels {
                    sample_count: samples.len(),
                    channels,
                })
            );
        }
    }
}
