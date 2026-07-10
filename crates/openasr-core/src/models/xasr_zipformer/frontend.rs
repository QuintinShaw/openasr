//! Kaldi-style 80-bin fbank frontend for X-ASR.
//!
//! This is the production-side Rust frontend scaffold. The exact constants are
//! intentionally local to X-ASR because sherpa/icefall fbank parity is part of
//! this family contract, not a generic OpenASR mel frontend.
//!
//! Deliberately **not** built on the shared batch engine in
//! [`crate::models::kaldi_fbank`] (which firered_aed/dolphin/sensevoice do
//! share): this frontend runs `snip_edges=false` framing with incremental,
//! cacheable frame-range computation (`features_for_frame_range_from`,
//! `earliest_sample_needed_for_frame`) for O(1)-memory streaming, skips
//! pre-emphasis/DC-removal/int16 rescale entirely, and stores mel filters
//! densely with precomputed per-mel bin ranges for that range slicing --
//! folding it into the shared engine would mean threading streaming-only
//! control flow through a batch-oriented API, i.e. a shared-layer special
//! case rather than a config difference.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub(crate) const XASR_SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const XASR_CHANNELS: u16 = 1;
/// Trailing silence appended exactly once at final flush (0.8 s, the tail
/// padding sherpa-onnx zipformer recipes feed before `InputFinished`). The
/// streaming zipformer needs right acoustic context to emit trailing tokens --
/// most visibly the terminal punctuation of the last sentence, which the model
/// only produces after it has seen the following silence. Audio that ends
/// abruptly at speech offset would otherwise lose that punctuation.
pub(crate) const XASR_FINAL_FLUSH_TAIL_PAD_SAMPLES: usize = XASR_SAMPLE_RATE_HZ as usize * 8 / 10;
const FRAME_LENGTH: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const N_FFT: usize = 512;
const N_MELS: usize = 80;
pub(crate) const XASR_N_MELS: usize = N_MELS;
const MEL_FMIN: f32 = 20.0;
const MEL_FMAX: f32 = 7_600.0;
const LOG_EPSILON: f32 = 1.192_092_9e-7;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrFbankFeatures {
    /// Row-major `[frame][mel]`; matches ONNX `x: [N, T, 80]`.
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum XasrFrontendError {
    #[error("xasr frontend requires finite 16 kHz mono f32 audio")]
    UnsupportedAudio,
    #[error("xasr frontend produced no fbank frames from {samples} samples")]
    NoFrames { samples: usize },
    #[error("xasr frontend was asked for frame {frame} whose samples were already drained")]
    MissingPrefix { frame: usize },
}

#[derive(Clone)]
pub(crate) struct XasrFbankFrontend {
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    /// Per-mel half-open `[start, end)` range of nonzero filter bins; the
    /// triangular filters cover only a narrow band, so iterating the full
    /// 257-bin row wastes most of the multiply-adds.
    mel_bin_ranges: Vec<(usize, usize)>,
    fft_bins: usize,
    /// Planned once: building the planner + twiddle tables per call dominated
    /// the per-push cost of incremental streaming fbank.
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for XasrFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XasrFbankFrontend")
            .field("fft_bins", &self.fft_bins)
            .finish_non_exhaustive()
    }
}

impl Default for XasrFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl XasrFbankFrontend {
    pub(crate) fn new() -> Self {
        let fft_bins = N_FFT / 2 + 1;
        let mel_filters = htk_mel_filterbank(N_MELS, N_FFT, fft_bins);
        let mel_bin_ranges = (0..N_MELS)
            .map(|mel| {
                let row = &mel_filters[mel * fft_bins..(mel + 1) * fft_bins];
                let start = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let end = row.iter().rposition(|&v| v > 0.0).map_or(0, |i| i + 1);
                (start, end.max(start))
            })
            .collect();
        Self {
            window: povey_window(FRAME_LENGTH),
            mel_filters,
            mel_bin_ranges,
            fft_bins,
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT),
        }
    }

    pub(crate) fn features_from_samples(
        &self,
        samples: &[f32],
    ) -> Result<XasrFbankFeatures, XasrFrontendError> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err(XasrFrontendError::UnsupportedAudio);
        }
        let n_frames = snip_edges_false_frame_count(samples.len());
        if n_frames == 0 {
            return Err(XasrFrontendError::NoFrames {
                samples: samples.len(),
            });
        }
        let mut output = vec![0.0_f32; n_frames * N_MELS];
        self.compute_frames_into(samples, 0, 0, n_frames, &mut output)?;
        Ok(XasrFbankFeatures {
            data: output,
            n_frames,
            n_mels: N_MELS,
        })
    }

    /// Computes fbank rows for `start_frame..end_frame` only.
    ///
    /// Frame `i` reads samples `[i*160-120, i*160+280)`; once those samples are
    /// all real (no right-edge reflection), its value is independent of any
    /// audio appended later, so streaming callers can cache rows and only pay
    /// for newly clean frames per push. Callers validate sample finiteness.
    pub(crate) fn features_for_frame_range(
        &self,
        samples: &[f32],
        start_frame: usize,
        end_frame: usize,
    ) -> Result<Vec<f32>, XasrFrontendError> {
        self.features_for_frame_range_from(samples, 0, start_frame, end_frame)
    }

    /// Like [`Self::features_for_frame_range`], but `samples` is the tail of a
    /// longer stream whose first `sample_offset` samples were already drained.
    /// Frame indices stay absolute (against the full stream), which lets
    /// streaming callers drop consumed audio while keeping O(1) memory.
    pub(crate) fn features_for_frame_range_from(
        &self,
        samples: &[f32],
        sample_offset: usize,
        start_frame: usize,
        end_frame: usize,
    ) -> Result<Vec<f32>, XasrFrontendError> {
        if samples.is_empty() {
            return Err(XasrFrontendError::UnsupportedAudio);
        }
        let total_len = sample_offset
            .checked_add(samples.len())
            .ok_or(XasrFrontendError::UnsupportedAudio)?;
        let total_frames = snip_edges_false_frame_count(total_len);
        if end_frame > total_frames || start_frame > end_frame {
            return Err(XasrFrontendError::NoFrames { samples: total_len });
        }
        if earliest_sample_needed_for_frame(start_frame) < sample_offset {
            return Err(XasrFrontendError::MissingPrefix { frame: start_frame });
        }
        let mut output = vec![0.0_f32; (end_frame - start_frame) * N_MELS];
        self.compute_frames_into(samples, sample_offset, start_frame, end_frame, &mut output)?;
        Ok(output)
    }

    fn compute_frames_into(
        &self,
        samples: &[f32],
        sample_offset: usize,
        start_frame: usize,
        end_frame: usize,
        output: &mut [f32],
    ) -> Result<(), XasrFrontendError> {
        let total_len = sample_offset + samples.len();
        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        let mut power = vec![0.0_f32; self.fft_bins];

        for frame in start_frame..end_frame {
            fft_in.fill(0.0);
            let start = frame as isize * FRAME_SHIFT as isize - (FRAME_LENGTH as isize / 2)
                + (FRAME_SHIFT as isize / 2);
            for (i, sample) in fft_in.iter_mut().enumerate().take(FRAME_LENGTH) {
                let absolute_idx = reflect_sample_index(start + i as isize, total_len);
                let sample_idx = absolute_idx
                    .checked_sub(sample_offset)
                    .ok_or(XasrFrontendError::MissingPrefix { frame })?;
                *sample = samples[sample_idx] * self.window[i];
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("rfft");
            for (bin, value) in fft_out.iter().enumerate() {
                power[bin] = value.norm_sqr();
            }
            let row_offset = (frame - start_frame) * N_MELS;
            for mel in 0..N_MELS {
                let row = &self.mel_filters[mel * self.fft_bins..(mel + 1) * self.fft_bins];
                let (bin_start, bin_end) = self.mel_bin_ranges[mel];
                let mut energy = 0.0_f32;
                for bin in bin_start..bin_end {
                    energy += row[bin] * power[bin];
                }
                output[row_offset + mel] = energy.max(LOG_EPSILON).ln();
            }
        }
        Ok(())
    }
}

pub(crate) fn total_frame_count_for_samples(num_samples: usize) -> usize {
    snip_edges_false_frame_count(num_samples)
}

/// First sample index frame `frame` reads (its window is
/// `[frame*SHIFT - (LEN/2 - SHIFT/2), frame*SHIFT + ...)`), clamped to 0 at the
/// stream start where the left edge reflects.
pub(crate) fn earliest_sample_needed_for_frame(frame: usize) -> usize {
    (frame * FRAME_SHIFT).saturating_sub(FRAME_LENGTH / 2 - FRAME_SHIFT / 2)
}

pub(crate) fn clean_frame_count_for_samples(num_samples: usize) -> usize {
    let right_edge_samples = FRAME_LENGTH - (FRAME_LENGTH / 2 - FRAME_SHIFT / 2);
    if num_samples < right_edge_samples {
        0
    } else {
        (num_samples - right_edge_samples) / FRAME_SHIFT + 1
    }
}

fn snip_edges_false_frame_count(num_samples: usize) -> usize {
    if num_samples == 0 {
        return 0;
    }
    (num_samples + FRAME_SHIFT / 2) / FRAME_SHIFT
}

fn reflect_sample_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let last = len as isize - 1;
    let period = 2 * last;
    let mut folded = index % period;
    if folded < 0 {
        folded += period;
    }
    if folded > last {
        (period - folded) as usize
    } else {
        folded as usize
    }
}

fn povey_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos();
            hann.powf(0.85)
        })
        .collect()
}

fn hz_to_mel_htk(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

fn htk_mel_filterbank(n_mels: usize, n_fft: usize, fft_bins: usize) -> Vec<f32> {
    let mel_min = hz_to_mel_htk(MEL_FMIN);
    let mel_max = hz_to_mel_htk(MEL_FMAX);
    let mel_points = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz_htk)
        .collect::<Vec<_>>();
    let fft_freqs = (0..fft_bins)
        .map(|i| i as f32 * XASR_SAMPLE_RATE_HZ as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0_f32; n_mels * fft_bins];
    for mel in 0..n_mels {
        let left = mel_points[mel];
        let center = mel_points[mel + 1];
        let right = mel_points[mel + 2];
        for (bin, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / (center - left);
            let upper = (right - freq) / (right - center);
            filters[mel * fft_bins + bin] = lower.min(upper).max(0.0);
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_finite_80_bin_fbank() {
        let samples = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.1)
            .collect::<Vec<_>>();
        let features = XasrFbankFrontend::new()
            .features_from_samples(&samples)
            .expect("features");
        assert_eq!(features.n_mels, 80);
        assert!(features.n_frames >= 99 && features.n_frames <= 101);
        assert_eq!(features.data.len(), features.n_frames * features.n_mels);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn frame_range_matches_full_computation() {
        let samples = (0..8_000)
            .map(|i| (2.0 * std::f32::consts::PI * 313.0 * i as f32 / 16_000.0).sin() * 0.1)
            .collect::<Vec<_>>();
        let frontend = XasrFbankFrontend::new();
        let full = frontend.features_from_samples(&samples).expect("full");
        let head = frontend
            .features_for_frame_range(&samples, 0, 7)
            .expect("head");
        let tail = frontend
            .features_for_frame_range(&samples, 7, full.n_frames)
            .expect("tail");
        assert_eq!(head.len(), 7 * full.n_mels);
        assert_eq!(head, full.data[..7 * full.n_mels]);
        assert_eq!(tail, full.data[7 * full.n_mels..]);
    }

    #[test]
    fn clean_frames_wait_until_right_edge_is_real_audio() {
        assert_eq!(clean_frame_count_for_samples(279), 0);
        assert_eq!(clean_frame_count_for_samples(280), 1);
        assert_eq!(clean_frame_count_for_samples(440), 2);
    }

    #[test]
    fn clean_frames_are_stable_as_audio_grows() {
        let long = (0..8_000)
            .map(|i| (2.0 * std::f32::consts::PI * 313.0 * i as f32 / 16_000.0).sin() * 0.1)
            .collect::<Vec<_>>();
        // Frame i reads samples [i*160-120, i*160+280); with 4000 samples the
        // first (4000-280)/160+1 = 24 frames are free of right-edge reflection.
        let short = &long[..4_000];
        let frontend = XasrFbankFrontend::new();
        let from_short = frontend
            .features_for_frame_range(short, 0, 24)
            .expect("short clean frames");
        let from_long = frontend
            .features_for_frame_range(&long, 0, 24)
            .expect("long clean frames");
        assert_eq!(from_short, from_long);
    }

    #[test]
    fn reflection_stays_in_bounds() {
        let values = (-20..20)
            .map(|i| reflect_sample_index(i, 5))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|&i| i < 5));
        assert_eq!(reflect_sample_index(-1, 5), 1);
        assert_eq!(reflect_sample_index(5, 5), 3);
    }
}
