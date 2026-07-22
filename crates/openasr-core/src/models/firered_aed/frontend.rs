//! firered-aed kaldi-fbank frontend + global CMVN.
//!
//! Reproduces the upstream feature pipeline (`fireredasr/data/asr_feat.py`):
//! `kaldi_native_fbank` with `num_mel_bins=80, frame_length=25, frame_shift=10,
//! dither=0, snip_edges=true` over int16-magnitude samples, followed by the
//! checkpoint CMVN `(x - mean) * istd` -- baked in the pack as
//! `frontend.cmvn.neg_mean` / `frontend.cmvn.inv_stddev`, applied as
//! `(x + neg_mean) * inv_stddev`.
//!
//! The fbank math itself is the shared [`crate::models::kaldi_fbank`] engine
//! (Povey window, remove_dc_offset, pre-emphasis 0.97, HTK mel 20 Hz..Nyquist,
//! `log(max(energy, eps))`), which the dolphin and sensevoice frontends also
//! use with the same 80/25ms/10ms/16kHz/dither-0 config -- see that module's
//! doc for why the engine is shared but the CMVN/error types stay family-local.

#![allow(dead_code)]

use crate::models::kaldi_fbank::{
    KaldiFbankConfig, KaldiFbankError, KaldiFbankFrontend, KaldiWindowKind,
};

pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const NUM_MEL_BINS: usize = 80;
/// 25 ms @ 16 kHz. Exposed (not just baked into [`FRONTEND_CONFIG`]) so
/// [`super::decoder_graph`]'s cross-KV capacity sizing can predict the mel
/// frame count for a given audio duration without duplicating this constant.
pub(crate) const FRAME_LENGTH_SAMPLES: usize = 400;
/// 10 ms @ 16 kHz (`snip_edges=true`: frame count is `1 + (len - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES`).
pub(crate) const FRAME_SHIFT_SAMPLES: usize = 160;

const FRONTEND_CONFIG: KaldiFbankConfig = KaldiFbankConfig {
    sample_rate_hz: SAMPLE_RATE_HZ,
    frame_length: FRAME_LENGTH_SAMPLES,
    frame_shift: FRAME_SHIFT_SAMPLES,
    fft_size: 512, // next pow2 >= 400 (kaldi rounds the window up)
    num_mel_bins: NUM_MEL_BINS,
    mel_low_hz: 20.0,
    mel_high_hz: 8_000.0, // high_freq <= 0 in kaldi => Nyquist
    preemph_coeff: 0.97,
    input_scale: 32_768.0, // float [-1, 1] -> int16 magnitude
    log_energy_floor: 1.192_092_9e-7,
    window: KaldiWindowKind::Povey,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedFrontendError {
    #[error("firered frontend requires finite 16 kHz mono f32 audio")]
    UnsupportedAudio,
    #[error("firered frontend produced no fbank frames from {samples} samples")]
    NoFrames { samples: usize },
    #[error("firered CMVN vector '{name}' has {actual} values, expected {expected} (feature dim)")]
    CmvnLen {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("firered fbank produced {frames}x{stride} features, expected feature dim {expected}")]
    FeatureShape {
        frames: usize,
        stride: usize,
        expected: usize,
    },
}

impl From<KaldiFbankError> for FireRedFrontendError {
    fn from(error: KaldiFbankError) -> Self {
        match error {
            KaldiFbankError::UnsupportedAudio => Self::UnsupportedAudio,
            KaldiFbankError::NoFrames { samples } => Self::NoFrames { samples },
        }
    }
}

/// Pre-CMVN log-mel features, row-major `[frame][mel]` (mel innermost).
pub(crate) type FireRedFbankFeatures = crate::models::kaldi_fbank::KaldiFbankFeatures;

pub(crate) struct FireRedFbankFrontend {
    inner: KaldiFbankFrontend,
}

impl std::fmt::Debug for FireRedFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FireRedFbankFrontend")
            .finish_non_exhaustive()
    }
}

impl Default for FireRedFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl FireRedFbankFrontend {
    pub(crate) fn new() -> Self {
        Self {
            inner: KaldiFbankFrontend::new(FRONTEND_CONFIG),
        }
    }

    /// Compute pre-CMVN kaldi log-mel features for 16 kHz mono `samples`
    /// (float in `[-1, 1]`). `snip_edges=true`: frame count is
    /// `1 + (len - FRAME_LENGTH) / FRAME_SHIFT`, frame `i` starts at
    /// `i * FRAME_SHIFT` with no edge reflection.
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<FireRedFbankFeatures, FireRedFrontendError> {
        Ok(self.inner.compute(samples)?)
    }
}

/// Apply the pack CMVN `(x + neg_mean) * inv_stddev` in place, per mel bin.
/// `features` is row-major `[frames, feature_dim]`; the vectors are the
/// `frontend.cmvn.*` tensors baked in the pack (both length `feature_dim`).
pub(crate) fn apply_cmvn(
    features: &mut [f32],
    feature_dim: usize,
    neg_mean: &[f32],
    inv_stddev: &[f32],
) -> Result<(), FireRedFrontendError> {
    if neg_mean.len() != feature_dim {
        return Err(FireRedFrontendError::CmvnLen {
            name: "frontend.cmvn.neg_mean",
            expected: feature_dim,
            actual: neg_mean.len(),
        });
    }
    if inv_stddev.len() != feature_dim {
        return Err(FireRedFrontendError::CmvnLen {
            name: "frontend.cmvn.inv_stddev",
            expected: feature_dim,
            actual: inv_stddev.len(),
        });
    }
    if !features.len().is_multiple_of(feature_dim) {
        return Err(FireRedFrontendError::FeatureShape {
            frames: features.len() / feature_dim.max(1),
            stride: feature_dim,
            expected: feature_dim,
        });
    }
    for frame in features.chunks_exact_mut(feature_dim) {
        for ((value, m), s) in frame.iter_mut().zip(neg_mean).zip(inv_stddev) {
            *value = (*value + *m) * *s;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_80_bin_snip_edges_fbank() {
        let samples = vec![0.01f32; 38_080];
        let features = FireRedFbankFrontend::new()
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
            FireRedFbankFrontend::new().compute(&samples),
            Err(FireRedFrontendError::NoFrames { samples: 200 })
        ));
    }

    #[test]
    fn cmvn_applies_neg_mean_then_scale() {
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        // (x + neg_mean) * inv_stddev with neg_mean = [-0.5, -1.0].
        apply_cmvn(&mut feats, 2, &[-0.5, -1.0], &[2.0, 0.5]).expect("cmvn");
        assert_eq!(feats, vec![1.0, 0.5, 5.0, 1.5]);
    }

    #[test]
    fn cmvn_rejects_mismatched_vector() {
        let mut feats = vec![0.0; 4];
        assert!(apply_cmvn(&mut feats, 2, &[0.0], &[1.0, 1.0]).is_err());
    }

    // Weight-free, committable regression net for the fbank math itself
    // (windowing / FFT / mel filterbank / log-energy): the two tests above
    // only feed a constant-DC signal, which is degenerate -- pre-emphasis and
    // the FFT collapse it to (near-)zero magnitude in every bin except the
    // log-energy floor, so a bug in the mel filterbank weights or the window
    // function would not move the output at all. A synthetic 440 Hz sine
    // (generated in-code, no fixture file, no model weights) actually
    // exercises the frequency response.
    //
    // This does NOT pin a bit-exact hash of the full output: `rustfft` (the
    // shared `kaldi_fbank` engine's FFT) dispatches to different SIMD kernels
    // per target architecture, and a cross-arch check found up to 0.017 max
    // absolute difference per log-mel value between this repo's aarch64 dev
    // host and CI's `ubuntu-latest` (x86_64) runner for this exact synthetic
    // input -- a real, expected floating-point-reduction-order difference,
    // not a bug. So every numeric assertion below is a `< TOLERANCE`
    // (2e-2, comfortably above that measured 0.017 spread) comparison against
    // reference values captured on this repo's aarch64 dev host, plus a small
    // number of topology-level exact assertions (frame/bin counts, the peak
    // mel bin) that a bit-exact FFT reduction order cannot move.
    const CROSS_ARCH_TOLERANCE: f32 = 2e-2;

    fn synthetic_sine_wave_samples(duration_seconds: f32, frequency_hz: f32) -> Vec<f32> {
        let n = (SAMPLE_RATE_HZ as f32 * duration_seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ as f32;
                0.5 * (2.0 * std::f32::consts::PI * frequency_hz * t).sin()
            })
            .collect()
    }

    // Reference-platform (macOS aarch64) pre-CMVN log-mel row for frame 49
    // (the temporal midpoint of the 98-frame output) of a 1s/440Hz synthetic
    // sine, captured via `FireRedFbankFrontend::compute`.
    const REFERENCE_MID_FRAME_INDEX: usize = 49;
    const REFERENCE_MID_FRAME_MEL: [f32; 80] = [
        9.15403, 9.707858, 9.0794, 8.10379, 10.367656, 11.684205, 12.609817, 13.0649605,
        12.0948515, 13.158659, 16.220291, 18.155163, 21.218346, 24.451693, 25.201832, 24.18605,
        20.647932, 17.800747, 14.544067, 13.942957, 13.465343, 11.708978, 10.721327, 11.125276,
        9.636799, 9.035014, 9.323213, 7.888769, 8.032621, 7.8527403, 6.703881, 7.238613, 5.9116964,
        6.362684, 5.814136, 5.619769, 5.7077184, 5.0359716, 4.839378, 4.5809956, 4.500667,
        5.9902954, 6.2484126, 5.969748, 4.3605266, 3.8930109, 4.786309, 4.589755, 3.3749213,
        3.2095275, 5.80585, 6.5169654, 5.7522435, 5.657481, 5.732948, 4.62743, 4.8528843,
        5.7505183, 5.0832577, 6.365632, 6.578811, 5.2467036, 5.3258343, 6.3744135, 6.318425,
        7.2973347, 5.370094, 5.7292542, 6.311184, 7.035475, 7.1788545, 6.216628, 6.36065, 8.939394,
        7.266076, 5.47633, 6.591452, 9.103914, 7.3653455, 7.268999,
    ];
    // The 440 Hz fundamental's log-mel peak, at reference-platform mel bin
    // 14 -- topology, not float magnitude, so it must hold exactly across
    // architectures regardless of FFT reduction order.
    const REFERENCE_PEAK_MEL_BIN: usize = 14;
    // Reference-platform total log-mel energy (sum over all 98*80 values);
    // a wide-but-bounded absolute tolerance (well under 0.1% relative) so a
    // real regression (wrong filterbank weights, dropped frames, mis-scaled
    // window) still trips this even though per-value cross-arch FFT noise
    // does not.
    const REFERENCE_TOTAL_ENERGY: f64 = 65021.198688641656;
    const TOTAL_ENERGY_TOLERANCE: f64 = 50.0;

    #[test]
    fn golden_diff_fbank_matches_pinned_reference_for_synthetic_440hz_sine() {
        let samples = synthetic_sine_wave_samples(1.0, 440.0);
        let features = FireRedFbankFrontend::new()
            .compute(&samples)
            .expect("fbank");

        // Exact, architecture-independent topology.
        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 98);
        assert!(features.data.iter().all(|v| v.is_finite()));

        let mid = features.n_frames / 2;
        assert_eq!(mid, REFERENCE_MID_FRAME_INDEX);
        let row = &features.data[mid * features.n_mels..(mid + 1) * features.n_mels];

        let peak_bin = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .expect("mid frame is non-empty");
        assert_eq!(
            peak_bin, REFERENCE_PEAK_MEL_BIN,
            "440 Hz fundamental's peak mel bin moved"
        );

        for (bin, (actual, expected)) in row.iter().zip(REFERENCE_MID_FRAME_MEL).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < CROSS_ARCH_TOLERANCE,
                "mid-frame mel bin {bin} drifted past cross-arch tolerance: \
                 actual={actual} expected={expected} diff={diff} tolerance={CROSS_ARCH_TOLERANCE}"
            );
        }

        let total_energy: f64 = features.data.iter().map(|v| *v as f64).sum();
        let energy_diff = (total_energy - REFERENCE_TOTAL_ENERGY).abs();
        assert!(
            energy_diff < TOTAL_ENERGY_TOLERANCE,
            "total fbank log-mel energy drifted: actual={total_energy} \
             expected={REFERENCE_TOTAL_ENERGY} diff={energy_diff} \
             tolerance={TOTAL_ENERGY_TOLERANCE}"
        );
    }
}
