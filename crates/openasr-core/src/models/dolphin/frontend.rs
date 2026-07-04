//! Dolphin `small.cn` kaldi-fbank frontend + global CMVN (WeNet format).
//!
//! Reproduces the reference feature pipeline used to fit `small.cn`:
//! `torchaudio.compliance.kaldi.fbank` with the `train.yaml` `fbank_conf`
//! (25 ms window, 10 ms shift, 80 mel bins, `dither=0`) over the waveform scaled
//! by `1 << 15` (float `[-1, 1]` back to the int16 magnitude kaldi expects),
//! followed by the checkpoint's `global_cmvn` `(x - mean) * istd`.
//!
//! Fixed to the kaldi defaults the reference uses: Povey window, `remove_dc_offset`,
//! pre-emphasis 0.97, HTK mel scale over 20-8000 Hz, `snip_edges=true`, and the
//! `log(max(energy, eps))` floor. The frontend is family-local (like the X-ASR and
//! whisper frontends) so nothing generic grows a Dolphin special case.
//!
//! Parity: the pre-CMVN log-mel reproduces the golden `logmel_feats` fixture to a
//! max abs diff ~1e-4 (f32 FFT/mel noise), and `(x - mean) * istd` reproduces
//! `logmel_feats_cmvn` (the exact tensor fed to the encoder). Both are exercised
//! by the executor's end-to-end harness against the committed golden.

#![allow(dead_code)]

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// 16 kHz, 25 ms window / 10 ms shift, 80 mel bins (train.yaml `fbank_conf`).
pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_LENGTH: usize = 400; // 25 ms @ 16 kHz
const FRAME_SHIFT: usize = 160; // 10 ms @ 16 kHz
const FFT_SIZE: usize = 512; // next pow2 >= 400 (kaldi rounds the window up)
pub(crate) const NUM_MEL_BINS: usize = 80;
const MEL_LOW_HZ: f32 = 20.0;
const MEL_HIGH_HZ: f32 = 8_000.0; // high_freq <= 0 in kaldi => Nyquist (8 kHz)
const PREEMPH_COEFF: f32 = 0.97;
/// float `[-1, 1]` -> int16 magnitude, the domain kaldi fbank operates in.
const INPUT_SCALE: f32 = 32_768.0;
/// kaldi/torchaudio mel-energy floor before the log (`torch.finfo(f32).eps`).
const LOG_ENERGY_FLOOR: f32 = 1.192_092_9e-7;

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinFrontendError {
    #[error("dolphin frontend requires finite 16 kHz mono f32 audio")]
    UnsupportedAudio,
    #[error("dolphin frontend produced no fbank frames from {samples} samples")]
    NoFrames { samples: usize },
    #[error(
        "dolphin global_cmvn vector '{name}' has {actual} values, expected {expected} (feature dim)"
    )]
    CmvnLen {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin fbank produced {frames}x{stride} features, expected feature dim {expected}")]
    FeatureShape {
        frames: usize,
        stride: usize,
        expected: usize,
    },
}

/// Pre-CMVN log-mel features, row-major `[frame][mel]` (mel innermost), matching
/// the golden `logmel_feats` layout and the tensor the encoder graph consumes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DolphinFbankFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// One triangular mel filter as a contiguous weight run over FFT power bins.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

pub(crate) struct DolphinFbankFrontend {
    window: [f32; FRAME_LENGTH],
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for DolphinFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DolphinFbankFrontend")
            .field("n_mels", &self.filters.len())
            .finish_non_exhaustive()
    }
}

impl Default for DolphinFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl DolphinFbankFrontend {
    pub(crate) fn new() -> Self {
        let mut window = [0.0f32; FRAME_LENGTH];
        for (n, w) in window.iter_mut().enumerate() {
            // Povey window: a Hann raised to 0.85 (kaldi/torchaudio default).
            let hann = 0.5
                - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (FRAME_LENGTH as f32 - 1.0)).cos();
            *w = hann.powf(0.85);
        }
        Self {
            window,
            filters: build_mel_filters(),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE),
        }
    }

    /// Compute pre-CMVN kaldi log-mel features for 16 kHz mono `samples`
    /// (float in `[-1, 1]`). `snip_edges=true`: frame count is
    /// `1 + (len - FRAME_LENGTH) / FRAME_SHIFT`, frame `i` starts at
    /// `i * FRAME_SHIFT` with no edge reflection.
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<DolphinFbankFeatures, DolphinFrontendError> {
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(DolphinFrontendError::UnsupportedAudio);
        }
        if samples.len() < FRAME_LENGTH {
            return Err(DolphinFrontendError::NoFrames {
                samples: samples.len(),
            });
        }
        let n_frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        let mut power = vec![0.0f32; FFT_SIZE / 2 + 1];

        let mut feats = vec![0.0f32; n_frames * NUM_MEL_BINS];
        let mut frame = [0.0f32; FRAME_LENGTH];
        for fr in 0..n_frames {
            let start = fr * FRAME_SHIFT;
            for (dst, src) in frame.iter_mut().zip(&samples[start..start + FRAME_LENGTH]) {
                *dst = *src * INPUT_SCALE;
            }
            // remove_dc_offset: subtract the frame mean.
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for v in &mut frame {
                *v -= mean;
            }
            // pre-emphasis with replicate padding: x[i] -= 0.97 x[i-1] (i>=1),
            // x[0] -= 0.97 x[0].
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPH_COEFF * frame[i - 1];
            }
            frame[0] -= PREEMPH_COEFF * frame[0];
            // Povey window into the zero-padded FFT input.
            fft_in.fill(0.0);
            for (slot, (sample, w)) in fft_in.iter_mut().zip(frame.iter().zip(self.window.iter())) {
                *slot = *sample * *w;
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("dolphin fbank rfft");
            for (bin, value) in fft_out.iter().enumerate() {
                power[bin] = value.re * value.re + value.im * value.im;
            }
            for (bin, filter) in self.filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (j, weight) in filter.weights.iter().enumerate() {
                    energy += weight * power[filter.first_bin + j];
                }
                feats[fr * NUM_MEL_BINS + bin] = energy.max(LOG_ENERGY_FLOOR).ln();
            }
        }
        Ok(DolphinFbankFeatures {
            data: feats,
            n_frames,
            n_mels: NUM_MEL_BINS,
        })
    }
}

/// Apply the checkpoint's `global_cmvn` `(x - mean) * istd` in place, per mel bin.
/// `features` is row-major `[frames, feature_dim]`; `mean`/`istd` are the
/// `encoder.global_cmvn.*` vectors baked in the pack (both length `feature_dim`).
pub(crate) fn apply_global_cmvn(
    features: &mut [f32],
    feature_dim: usize,
    mean: &[f32],
    istd: &[f32],
) -> Result<(), DolphinFrontendError> {
    if mean.len() != feature_dim {
        return Err(DolphinFrontendError::CmvnLen {
            name: "encoder.global_cmvn.mean",
            expected: feature_dim,
            actual: mean.len(),
        });
    }
    if istd.len() != feature_dim {
        return Err(DolphinFrontendError::CmvnLen {
            name: "encoder.global_cmvn.istd",
            expected: feature_dim,
            actual: istd.len(),
        });
    }
    if !features.len().is_multiple_of(feature_dim) {
        return Err(DolphinFrontendError::FeatureShape {
            frames: features.len() / feature_dim.max(1),
            stride: feature_dim,
            expected: feature_dim,
        });
    }
    for frame in features.chunks_exact_mut(feature_dim) {
        for ((value, m), s) in frame.iter_mut().zip(mean).zip(istd) {
            *value = (*value - *m) * *s;
        }
    }
    Ok(())
}

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi triangular mel filterbank over `FFT_SIZE/2 + 1` power bins: `NUM_MEL_BINS`
/// filters spanning `[MEL_LOW_HZ, MEL_HIGH_HZ]` on the HTK mel scale, each a peak-
/// normalized triangle (no Slaney area norm), gated to bins strictly inside the
/// filter band.
fn build_mel_filters() -> Vec<MelFilter> {
    let n_fft_bins = FFT_SIZE / 2 + 1;
    let fft_bin_width = SAMPLE_RATE_HZ as f32 / FFT_SIZE as f32;
    let mel_low = mel_scale(MEL_LOW_HZ);
    let mel_high = mel_scale(MEL_HIGH_HZ);
    let mel_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f32 + 1.0);

    let mut filters = Vec::with_capacity(NUM_MEL_BINS);
    for bin in 0..NUM_MEL_BINS {
        let left = mel_low + bin as f32 * mel_delta;
        let center = mel_low + (bin as f32 + 1.0) * mel_delta;
        let right = mel_low + (bin as f32 + 2.0) * mel_delta;
        let mut first_bin = None;
        let mut weights = Vec::new();
        for k in 0..n_fft_bins {
            let mel = mel_scale(fft_bin_width * k as f32);
            if mel > left && mel < right {
                let weight = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
                if first_bin.is_none() {
                    first_bin = Some(k);
                }
                weights.push(weight);
            } else if first_bin.is_some() {
                break;
            }
        }
        filters.push(MelFilter {
            first_bin: first_bin.unwrap_or(0),
            weights,
        });
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_80_bin_snip_edges_fbank() {
        // 2380 ms of audio -> 236 frames (1 + (38080 - 400) / 160), matching the
        // golden clip's frame count.
        let samples = vec![0.01f32; 38_080];
        let features = DolphinFbankFrontend::new()
            .compute(&samples)
            .expect("fbank");
        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 236);
        assert_eq!(features.data.len(), 236 * 80);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_too_short_audio() {
        let samples = vec![0.0f32; 200];
        assert!(matches!(
            DolphinFbankFrontend::new().compute(&samples),
            Err(DolphinFrontendError::NoFrames { samples: 200 })
        ));
    }

    #[test]
    fn global_cmvn_applies_per_bin_affine() {
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        apply_global_cmvn(&mut feats, 2, &[0.5, 1.0], &[2.0, 0.5]).expect("cmvn");
        // row0: ((1-0.5)*2, (2-1)*0.5) = (1.0, 0.5); row1: ((3-0.5)*2, (4-1)*0.5).
        assert_eq!(feats, vec![1.0, 0.5, 5.0, 1.5]);
    }

    #[test]
    fn global_cmvn_rejects_mismatched_vector() {
        let mut feats = vec![0.0; 4];
        assert!(apply_global_cmvn(&mut feats, 2, &[0.0], &[1.0, 1.0]).is_err());
    }
}
