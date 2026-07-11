//! Kaldi-style 80-bin fbank frontend for X-ASR.
//!
//! This is the production-side Rust frontend scaffold. The exact constants are
//! intentionally local to X-ASR because sherpa/icefall fbank parity is part of
//! this family contract, not a generic OpenASR mel frontend.
//!
//! The window/FFT/power-spectrum step now goes through
//! [`crate::models::audio_frontend`]'s `StftFramer` via
//! `PadMode::KaldiSnipEdgesFalse` (kaldi's asymmetric `snip_edges=false`
//! offset, reflected against the *total* stream length rather than a
//! once-materialized padded buffer), which preserves this frontend's O(1)
//! streaming property: `power_spectrogram_kaldi_snip_edges_false_range`
//! takes an explicit frame range plus a `sample_offset` for a tail-only
//! buffer, so callers can drop consumed audio exactly as before
//! (`features_for_frame_range_from`, `earliest_sample_needed_for_frame`).
//! This is deliberately **not** the shared batch engine in
//! [`crate::models::kaldi_fbank`] (which firered_aed/dolphin/sensevoice
//! share): that engine computes a whole buffer in one call, and skips
//! pre-emphasis/DC-removal/int16 rescale entirely here, plus stores mel
//! filters densely with precomputed per-mel bin ranges for that range
//! slicing -- folding *that* into the batch engine would still be a
//! shared-layer special case rather than a config difference, so the
//! frame-range caching and sparse-bin-range projection loop stay local.

use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale};
use crate::models::audio_frontend::{
    KaldiFrameRangeError, PadMode, StftFramer, povey_window_left_aligned,
};

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
    /// Owns the analysis window and the (planned-once) FFT; planning the
    /// twiddle tables per call dominated the per-push cost of incremental
    /// streaming fbank, which is why `StftFramer` builds its plan once at
    /// construction rather than per `compute_frames_into` call.
    framer: StftFramer,
    mel_filters: Vec<f32>,
    /// Per-mel half-open `[start, end)` range of nonzero filter bins; the
    /// triangular filters cover only a narrow band, so iterating the full
    /// 257-bin row wastes most of the multiply-adds.
    mel_bin_ranges: Vec<(usize, usize)>,
    fft_bins: usize,
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
        let mel_filters = crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::Htk,
            sample_rate_hz: XASR_SAMPLE_RATE_HZ as f32,
            n_fft: N_FFT,
            n_mels: N_MELS,
            fmin: MEL_FMIN,
            fmax: MEL_FMAX,
            mel_point_order: MelPointOrder::SpanTimesIndexFirst,
        });
        let mel_bin_ranges = (0..N_MELS)
            .map(|mel| {
                let row = &mel_filters[mel * fft_bins..(mel + 1) * fft_bins];
                let start = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let end = row.iter().rposition(|&v| v > 0.0).map_or(0, |i| i + 1);
                (start, end.max(start))
            })
            .collect();
        let framer = StftFramer::new(
            N_FFT,
            FRAME_LENGTH,
            FRAME_SHIFT,
            PadMode::KaldiSnipEdgesFalse,
            povey_window_left_aligned(FRAME_LENGTH, N_FFT),
        );
        Self {
            framer,
            mel_filters,
            mel_bin_ranges,
            fft_bins,
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
        let spectrogram = self
            .framer
            .power_spectrogram_kaldi_snip_edges_false_range(
                samples,
                sample_offset,
                total_len,
                start_frame,
                end_frame,
            )
            .map_err(|error| match error {
                KaldiFrameRangeError::MissingPrefix { frame } => {
                    XasrFrontendError::MissingPrefix { frame }
                }
            })?;

        for frame in start_frame..end_frame {
            let power = &spectrogram.data
                [(frame - start_frame) * self.fft_bins..(frame - start_frame + 1) * self.fft_bins];
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
        // Same reflect-index math xasr shipped before this module existed,
        // now `crate::models::audio_frontend::reflect_index_no_repeat`
        // (pinned bit-exact there); re-checked here against xasr's own
        // fixed points since this frontend depends on it directly.
        use crate::models::audio_frontend::reflect_index_no_repeat;
        let values = (-20i64..20)
            .map(|i| reflect_index_no_repeat(5, i))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|&i| i < 5));
        assert_eq!(reflect_index_no_repeat(5, -1), 1);
        assert_eq!(reflect_index_no_repeat(5, 5), 3);
    }
}
