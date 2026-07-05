//! SenseVoiceSmall (`FunAudioLLM/SenseVoiceSmall`) audio frontend.
//!
//! Reproduces FunASR's `WavFrontend` feature pipeline for the SenseVoice
//! SAN-M/CTC model:
//!
//! ```text
//!   16 kHz mono f32
//!     -> kaldi 80-mel fbank (25 ms window, 10 ms shift, Povey window,
//!        pre-emphasis 0.97, remove_dc_offset, HTK mel 20..8000 Hz, snip_edges)
//!     -> LFR stacking (m = 7 frames stacked, n = 6 stride)   [80 -> 560 dim]
//!     -> CMVN normalization  (x + neg_mean) * inv_stddev     [560-dim am.mvn]
//! ```
//!
//! The fbank stage is byte-for-byte the same kaldi computation the Dolphin
//! frontend uses (both derive from `torchaudio.compliance.kaldi.fbank` at the
//! WeNet/FunASR defaults), so its numeric behavior is exercised by the same
//! discipline. What is *SenseVoice-specific* and owned here is the **LFR
//! stacking** and the fact that CMVN is applied to the LFR-stacked 560-dim
//! feature rather than the raw 80-dim mel. Both are unit-tested here with
//! hand-computed expectations so the stacking/padding math is pinned
//! independent of any downloaded model.
//!
//! Numeric parity of the *fbank window/mel* stage against reference `funasr`
//! Python output is deferred until a converted `.oasr` pack + reference features
//! are available (it requires the model + reference download); the structural
//! stages (LFR geometry, CMVN affine) are verified here without weights.

#![allow(dead_code)]

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// 16 kHz, 25 ms window / 10 ms shift, 80 mel bins (FunASR `WavFrontend` fbank).
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

/// LFR (low frame rate) stacking parameters for SenseVoiceSmall: stack `m = 7`
/// consecutive fbank frames per output frame, advancing by `n = 6` frames.
pub(crate) const LFR_M: usize = 7;
pub(crate) const LFR_N: usize = 6;
/// SenseVoice CMVN operates on the LFR-stacked feature: `80 * 7 = 560` dims.
pub(crate) const LFR_FEATURE_DIM: usize = NUM_MEL_BINS * LFR_M;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SenseVoiceFrontendError {
    #[error("sensevoice frontend requires finite 16 kHz mono f32 audio")]
    UnsupportedAudio,
    #[error("sensevoice frontend produced no fbank frames from {samples} samples")]
    NoFrames { samples: usize },
    #[error(
        "sensevoice cmvn vector '{name}' has {actual} values, expected {expected} (LFR feature dim)"
    )]
    CmvnLen {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("sensevoice LFR features {frames}x{stride} do not match feature dim {expected}")]
    FeatureShape {
        frames: usize,
        stride: usize,
        expected: usize,
    },
}

/// Pre-CMVN 80-mel fbank features, row-major `[frame][mel]` (mel innermost).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SenseVoiceFbankFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// LFR-stacked features, row-major `[lfr_frame][560]` (stacked mel innermost).
/// This is the layout the SenseVoice encoder input projection consumes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SenseVoiceLfrFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub feature_dim: usize,
}

/// One triangular mel filter as a contiguous weight run over FFT power bins.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

pub(crate) struct SenseVoiceFbankFrontend {
    window: [f32; FRAME_LENGTH],
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for SenseVoiceFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SenseVoiceFbankFrontend")
            .field("n_mels", &self.filters.len())
            .finish_non_exhaustive()
    }
}

impl Default for SenseVoiceFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl SenseVoiceFbankFrontend {
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
    /// `1 + (len - FRAME_LENGTH) / FRAME_SHIFT`.
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<SenseVoiceFbankFeatures, SenseVoiceFrontendError> {
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(SenseVoiceFrontendError::UnsupportedAudio);
        }
        if samples.len() < FRAME_LENGTH {
            return Err(SenseVoiceFrontendError::NoFrames {
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
                .expect("sensevoice fbank rfft");
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
        Ok(SenseVoiceFbankFeatures {
            data: feats,
            n_frames,
            n_mels: NUM_MEL_BINS,
        })
    }
}

/// Apply FunASR LFR (low frame rate) stacking to `[n_frames, feature_dim]`
/// row-major features, producing `[ceil(n_frames / LFR_N), feature_dim * LFR_M]`.
///
/// Matches FunASR's `apply_lfr(inputs, lfr_m, lfr_n)`:
/// - left-pad `(LFR_M - 1) / 2` copies of the *first* frame,
/// - emit `ceil(n_frames / LFR_N)` output frames; output `i` concatenates frames
///   `[i*LFR_N, i*LFR_N + LFR_M)` of the padded input,
/// - the final output frame, if it runs past the end, is padded by repeating the
///   *last* input frame until it holds `LFR_M` frames.
pub(crate) fn apply_lfr(
    features: &[f32],
    feature_dim: usize,
) -> Result<SenseVoiceLfrFeatures, SenseVoiceFrontendError> {
    if feature_dim == 0 || !features.len().is_multiple_of(feature_dim) {
        return Err(SenseVoiceFrontendError::FeatureShape {
            frames: features.len() / feature_dim.max(1),
            stride: feature_dim,
            expected: feature_dim,
        });
    }
    let n_frames = features.len() / feature_dim;
    if n_frames == 0 {
        return Err(SenseVoiceFrontendError::NoFrames { samples: 0 });
    }
    let frame = |idx: usize| -> &[f32] {
        let clamped = idx.min(n_frames - 1);
        &features[clamped * feature_dim..(clamped + 1) * feature_dim]
    };
    // Padded input = left_pad copies of frame 0, then the frames themselves.
    let left_pad = (LFR_M - 1) / 2;
    let padded_len = n_frames + left_pad;
    // padded index -> original frame index: [0..left_pad) -> frame 0.
    let padded_frame = |p_idx: usize| -> &[f32] {
        if p_idx < left_pad {
            frame(0)
        } else {
            frame(p_idx - left_pad)
        }
    };

    let out_frames = padded_len.div_ceil(LFR_N);
    let out_dim = feature_dim * LFR_M;
    let mut out = vec![0.0f32; out_frames * out_dim];
    for i in 0..out_frames {
        let base = i * LFR_N;
        for m in 0..LFR_M {
            // frames past the padded end repeat the last real frame.
            let src_padded = base + m;
            let src = if src_padded < padded_len {
                padded_frame(src_padded)
            } else {
                frame(n_frames - 1)
            };
            let dst = &mut out[i * out_dim + m * feature_dim..i * out_dim + (m + 1) * feature_dim];
            dst.copy_from_slice(src);
        }
    }
    Ok(SenseVoiceLfrFeatures {
        data: out,
        n_frames: out_frames,
        feature_dim: out_dim,
    })
}

/// Apply the pack's `am.mvn` CMVN `(x + neg_mean) * inv_stddev` in place, per LFR
/// feature dim. `features` is row-major `[frames, feature_dim]`; `neg_mean` and
/// `inv_stddev` are the baked CMVN vectors (both length `feature_dim`, i.e. 560).
pub(crate) fn apply_cmvn(
    features: &mut [f32],
    feature_dim: usize,
    neg_mean: &[f32],
    inv_stddev: &[f32],
) -> Result<(), SenseVoiceFrontendError> {
    if neg_mean.len() != feature_dim {
        return Err(SenseVoiceFrontendError::CmvnLen {
            name: "am.mvn.neg_mean",
            expected: feature_dim,
            actual: neg_mean.len(),
        });
    }
    if inv_stddev.len() != feature_dim {
        return Err(SenseVoiceFrontendError::CmvnLen {
            name: "am.mvn.inv_stddev",
            expected: feature_dim,
            actual: inv_stddev.len(),
        });
    }
    if feature_dim == 0 || !features.len().is_multiple_of(feature_dim) {
        return Err(SenseVoiceFrontendError::FeatureShape {
            frames: features.len() / feature_dim.max(1),
            stride: feature_dim,
            expected: feature_dim,
        });
    }
    for frame in features.chunks_exact_mut(feature_dim) {
        for ((value, add), scale) in frame.iter_mut().zip(neg_mean).zip(inv_stddev) {
            *value = (*value + *add) * *scale;
        }
    }
    Ok(())
}

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi triangular mel filterbank over `FFT_SIZE/2 + 1` power bins: `NUM_MEL_BINS`
/// filters spanning `[MEL_LOW_HZ, MEL_HIGH_HZ]` on the HTK mel scale, each a peak-
/// normalized triangle gated to bins strictly inside the filter band.
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
        // 2380 ms -> 236 frames (1 + (38080 - 400) / 160).
        let samples = vec![0.01f32; 38_080];
        let features = SenseVoiceFbankFrontend::new()
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
            SenseVoiceFbankFrontend::new().compute(&samples),
            Err(SenseVoiceFrontendError::NoFrames { samples: 200 })
        ));
    }

    /// Hand-computed LFR check with a tiny feature_dim (2) so the stacking and
    /// padding geometry is pinned exactly. FunASR `apply_lfr` with m=7, n=6:
    /// left_pad = (7-1)/2 = 3 copies of frame 0; padded_len = n + 3.
    #[test]
    fn lfr_stacks_seven_frames_stride_six_with_edge_padding() {
        // 5 frames, feature_dim 2: frame k = [k, k+100].
        let feature_dim = 2usize;
        let n_frames = 5usize;
        let mut feats = Vec::new();
        for k in 0..n_frames {
            feats.push(k as f32);
            feats.push(k as f32 + 100.0);
        }
        let lfr = apply_lfr(&feats, feature_dim).expect("lfr");
        assert_eq!(lfr.feature_dim, feature_dim * LFR_M); // 14
        // padded_len = 5 + 3 = 8; out_frames = ceil(8 / 6) = 2.
        assert_eq!(lfr.n_frames, 2);

        // Padded frame sequence (by original index, clamped):
        //   [0, 0, 0, 0, 1, 2, 3, 4]  (3 left-pad copies of frame 0)
        // Output 0: padded[0..7] = frames [0,0,0,0,1,2,3].
        let out0 = &lfr.data[0..lfr.feature_dim];
        let expect0: Vec<f32> = [0, 0, 0, 0, 1, 2, 3]
            .iter()
            .flat_map(|&k| [k as f32, k as f32 + 100.0])
            .collect();
        assert_eq!(out0, expect0.as_slice());

        // Output 1: base = 6, padded[6..13]; padded has 8 entries [0..8), so
        //   padded[6]=frame3, padded[7]=frame4, then indices 8..12 run past the
        //   padded end and repeat the LAST real frame (frame 4).
        let out1 = &lfr.data[lfr.feature_dim..2 * lfr.feature_dim];
        let expect1: Vec<f32> = [3, 4, 4, 4, 4, 4, 4]
            .iter()
            .flat_map(|&k| [k as f32, k as f32 + 100.0])
            .collect();
        assert_eq!(out1, expect1.as_slice());
    }

    #[test]
    fn lfr_output_dim_is_560_for_80_mel() {
        // 60 fbank frames of 80-mel -> stacked 560-dim; out frames = ceil(63/6).
        let feats = vec![0.5f32; 60 * NUM_MEL_BINS];
        let lfr = apply_lfr(&feats, NUM_MEL_BINS).expect("lfr");
        assert_eq!(lfr.feature_dim, LFR_FEATURE_DIM);
        assert_eq!(lfr.feature_dim, 560);
        assert_eq!(lfr.n_frames, (60 + 3usize).div_ceil(LFR_N));
        assert_eq!(lfr.data.len(), lfr.n_frames * 560);
    }

    #[test]
    fn cmvn_applies_per_dim_affine() {
        // (x + neg_mean) * inv_stddev, per dim.
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        apply_cmvn(&mut feats, 2, &[-0.5, -1.0], &[2.0, 0.5]).expect("cmvn");
        // row0: ((1-0.5)*2, (2-1)*0.5) = (1.0, 0.5); row1: ((3-0.5)*2, (4-1)*0.5).
        assert_eq!(feats, vec![1.0, 0.5, 5.0, 1.5]);
    }

    #[test]
    fn cmvn_rejects_mismatched_vector() {
        let mut feats = vec![0.0; 4];
        assert!(apply_cmvn(&mut feats, 2, &[0.0], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn full_pipeline_frontend_to_lfr_to_cmvn_is_finite() {
        let samples = vec![0.02f32; 32_000]; // 2 s
        let fbank = SenseVoiceFbankFrontend::new()
            .compute(&samples)
            .expect("fbank");
        let mut lfr = apply_lfr(&fbank.data, fbank.n_mels).expect("lfr");
        assert_eq!(lfr.feature_dim, LFR_FEATURE_DIM);
        let neg_mean = vec![0.0f32; LFR_FEATURE_DIM];
        let inv_stddev = vec![1.0f32; LFR_FEATURE_DIM];
        apply_cmvn(&mut lfr.data, lfr.feature_dim, &neg_mean, &inv_stddev).expect("cmvn");
        assert!(lfr.data.iter().all(|v| v.is_finite()));
    }
}
