use super::{LongFormOptions, LongFormVadProvider, LongFormVadProviderKind, LongFormVadSlice};

const DEFAULT_FRAME_MS: usize = 20;

#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyLongFormVadProvider;

impl LongFormVadProvider for EnergyLongFormVadProvider {
    fn provider_kind(&self) -> LongFormVadProviderKind {
        LongFormVadProviderKind::EnergyLike
    }

    fn compute_speech_slices(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
    ) -> Result<Vec<LongFormVadSlice>, String> {
        if sample_rate_hz == 0 {
            return Err("sample_rate_hz must be greater than zero".to_string());
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let frame_samples = ((sample_rate_hz as usize) * DEFAULT_FRAME_MS / 1000).max(1);
        let frame_rms = compute_frame_rms(samples, frame_samples);
        if frame_rms.is_empty() {
            return Ok(Vec::new());
        }
        let max_rms = frame_rms.iter().copied().fold(0.0_f32, f32::max);
        if max_rms <= f32::EPSILON {
            return Ok(Vec::new());
        }

        let threshold = options.vad.threshold.clamp(0.0, 1.0);
        let noise_floor = percentile(&frame_rms, 0.20);
        let speech_peak = percentile(&frame_rms, 0.95).max(noise_floor);
        let relative_gate = noise_floor + (speech_peak - noise_floor) * threshold;
        // The relative gate is anchored to *this clip's own* 95th-percentile peak,
        // so a loud opening section inflates `speech_peak` and can drag the gate
        // above quieter speech later in the same recording -- observed in
        // production to silently drop an entire trailing utterance: a loud
        // Chinese opener (~-25 dBFS) pushed the gate to -34.2 dBFS, just above a
        // quieter English tail peaking at -34.6 dBFS, so the tail was never
        // detected as speech and never reached the decoder (see the long-form
        // code-switch investigation). Cap the gate at the module's existing
        // "definitely not silence" floor -- `energy_silence_threshold_db`
        // (-38 dBFS default), the same line `estimate_elision_penalty`,
        // `estimate_gap_edge_penalty`, `choose_forced_cut`, and
        // `subdivide_processed_spans_silence_aware` in `slicing.rs` already use to
        // decide a span is non-silent -- rather than a second, independently
        // tuned constant. This keeps one shared definition of "silence" across
        // the whole longform pipeline and, on the repro numbers above, is low
        // enough (-38 < -34.6) that the dropped English tail passes the gate.
        // VAD/energy slicing is a performance optimization, not a content
        // filter: a frame louder than the shared silence floor must never be
        // gated out purely because something else in the clip was louder still.
        let absolute_floor = 10.0_f32.powf(options.energy_silence_threshold_db / 20.0);
        let gate = relative_gate.min(absolute_floor);
        let min_speech_frames = duration_ms_to_frames(
            options.vad.min_speech_duration_ms as usize,
            DEFAULT_FRAME_MS,
        );
        let min_silence_frames = duration_ms_to_frames(
            options.vad.min_silence_duration_ms as usize,
            DEFAULT_FRAME_MS,
        );
        let mask: Vec<bool> = frame_rms.iter().map(|value| *value >= gate).collect();

        let mut speech_ranges = Vec::new();
        let mut in_speech = false;
        let mut speech_start = 0usize;
        let mut trailing_silence = 0usize;
        for (frame_idx, active) in mask.iter().copied().enumerate() {
            if active {
                if !in_speech {
                    in_speech = true;
                    speech_start = frame_idx;
                }
                trailing_silence = 0;
                continue;
            }
            if !in_speech {
                continue;
            }
            trailing_silence = trailing_silence.saturating_add(1);
            if trailing_silence < min_silence_frames {
                continue;
            }
            let speech_end = frame_idx.saturating_add(1).saturating_sub(trailing_silence);
            push_if_long_enough(
                &mut speech_ranges,
                speech_start,
                speech_end,
                min_speech_frames,
            );
            in_speech = false;
            trailing_silence = 0;
        }
        if in_speech {
            let speech_end = mask.len().saturating_sub(trailing_silence);
            push_if_long_enough(
                &mut speech_ranges,
                speech_start,
                speech_end,
                min_speech_frames,
            );
        }

        let mut slices = Vec::with_capacity(speech_ranges.len());
        for (start_frame, end_frame) in speech_ranges {
            let start_sample = (start_frame * frame_samples).min(samples.len());
            let end_sample = (end_frame * frame_samples).min(samples.len());
            if end_sample <= start_sample {
                continue;
            }
            slices.push(LongFormVadSlice {
                start_sample,
                end_sample,
            });
        }
        Ok(slices)
    }
}

fn duration_ms_to_frames(duration_ms: usize, frame_ms: usize) -> usize {
    if duration_ms == 0 {
        return 1;
    }
    duration_ms.saturating_add(frame_ms - 1) / frame_ms
}

fn push_if_long_enough(
    target: &mut Vec<(usize, usize)>,
    start_frame: usize,
    end_frame: usize,
    min_frames: usize,
) {
    if end_frame <= start_frame {
        return;
    }
    if end_frame.saturating_sub(start_frame) < min_frames {
        return;
    }
    target.push((start_frame, end_frame));
}

fn compute_frame_rms(samples: &[f32], frame_samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len() / frame_samples + 1);
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_samples).min(samples.len());
        if end <= start {
            break;
        }
        let mut sum = 0.0_f64;
        for sample in &samples[start..end] {
            let value = *sample as f64;
            sum += value * value;
        }
        out.push((sum / (end - start) as f64).sqrt() as f32);
        start = end;
    }
    out
}

fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clamped = fraction.clamp(0.0, 1.0);
    let index = ((sorted.len() - 1) as f32 * clamped).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longform::LongFormOptions;

    fn samples_with_two_speech_regions() -> Vec<f32> {
        let sr = 16_000usize;
        let mut samples = vec![0.0; sr];
        for sample in &mut samples[sr / 2..sr] {
            *sample = 0.2;
        }
        samples.extend(vec![0.0; sr]);
        let mut tail = vec![0.0; sr];
        for sample in &mut tail[..sr / 2] {
            *sample = 0.22;
        }
        samples.extend(tail);
        samples
    }

    #[test]
    fn detects_multiple_speech_slices() {
        let provider = EnergyLongFormVadProvider;
        let options = LongFormOptions::default();
        let slices = provider
            .compute_speech_slices(&samples_with_two_speech_regions(), 16_000, &options)
            .expect("vad");
        assert_eq!(slices.len(), 2);
        assert!(slices[0].start_sample < slices[0].end_sample);
        assert!(slices[1].start_sample < slices[1].end_sample);
        assert!(slices[1].start_sample > slices[0].end_sample);
    }

    #[test]
    fn silence_returns_empty_slices() {
        let provider = EnergyLongFormVadProvider;
        let options = LongFormOptions::default();
        let slices = provider
            .compute_speech_slices(&vec![0.0; 16_000], 16_000, &options)
            .expect("vad");
        assert!(slices.is_empty());
    }

    /// Reproduces the code-switch long-form bug at unit scale: a loud opener
    /// followed by true silence followed by a much quieter (but still clearly
    /// audible, above the -38 dBFS absolute floor) tail. Without the absolute
    /// floor, `speech_peak` is pinned to the loud opener, the relative gate
    /// sits at half that peak (0.25 amplitude), and the 0.02-amplitude tail
    /// (-33.9 dBFS) never crosses it -- exactly how the English tail in the
    /// original recording (peaking at -34.6 dBFS against a gate of -34.2 dBFS)
    /// was silently dropped. With the floor capping the gate at
    /// `energy_silence_threshold_db` (-38 dBFS, amplitude ~0.0126), the tail
    /// clears the gate and is detected as its own speech region.
    #[test]
    fn absolute_floor_prevents_loud_opening_from_masking_quiet_tail() {
        let provider = EnergyLongFormVadProvider;
        let options = LongFormOptions::default();
        let mut samples = vec![0.5_f32; 16_000]; // 1s loud opener
        samples.extend(vec![0.0_f32; 16_000 * 3]); // 3s true silence
        samples.extend(vec![0.02_f32; 16_000]); // 1s quiet-but-audible tail
        let slices = provider
            .compute_speech_slices(&samples, 16_000, &options)
            .expect("vad");
        assert_eq!(
            slices.len(),
            2,
            "expected the quiet tail to be detected as its own speech region: {slices:?}"
        );
        let tail = &slices[1];
        assert!(
            tail.start_sample >= 16_000 * 4,
            "tail slice should start at or after the silence gap: {tail:?}"
        );
        assert!(
            tail.end_sample <= samples.len(),
            "tail slice should not run past the audio: {tail:?}"
        );
    }
}
