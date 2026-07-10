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

const FRONTEND_CONFIG: KaldiFbankConfig = KaldiFbankConfig {
    sample_rate_hz: SAMPLE_RATE_HZ,
    frame_length: 400, // 25 ms @ 16 kHz
    frame_shift: 160,  // 10 ms @ 16 kHz
    fft_size: 512,     // next pow2 >= 400 (kaldi rounds the window up)
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
}
