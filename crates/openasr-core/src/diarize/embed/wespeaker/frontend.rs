//! Kaldi Hamming fbank + utterance CMN for WeSpeaker ResNet.
//!
//! Reuses [`crate::models::kaldi_fbank`] with Hamming window, 80 mel, 25/10 ms,
//! 16 kHz, preemph 0.97, `input_scale=32768`, `log_energy_floor=f32::EPSILON`,
//! fmin 20 / fmax 8000, snip_edges. Output is per-utterance mean-only CMN.

use crate::models::kaldi_fbank::{KaldiFbankConfig, KaldiFbankFrontend, KaldiWindowKind};

use super::super::weights::{WeightsError, allocation_commitment};
use super::config::N_MELS;

const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 8_000.0;
const PREEMPH: f32 = 0.97;
const INPUT_SCALE: f32 = 32_768.0;

pub(crate) struct WeSpeakerFrontend {
    inner: KaldiFbankFrontend,
}

impl WeSpeakerFrontend {
    pub(crate) fn new() -> Self {
        Self {
            inner: KaldiFbankFrontend::new(KaldiFbankConfig {
                sample_rate_hz: 16_000,
                frame_length: FRAME_LENGTH,
                frame_shift: FRAME_SHIFT,
                fft_size: FFT_SIZE,
                num_mel_bins: N_MELS,
                mel_low_hz: LOW_FREQ,
                mel_high_hz: HIGH_FREQ,
                preemph_coeff: PREEMPH,
                input_scale: INPUT_SCALE,
                log_energy_floor: f32::EPSILON,
                window: KaldiWindowKind::Hamming,
            }),
        }
    }

    /// CMN-normalized log-mel features as `[T, 80]` row-major (`feats[t * 80 + f]`).
    pub(crate) fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let Ok(features) = self.inner.compute(samples) else {
            return (Vec::new(), 0);
        };
        let mut data = features.data;
        cepstral_mean_normalize(&mut data, features.n_frames, features.n_mels);
        (data, features.n_frames)
    }

    /// Transpose `[T, F]` row-major into ggml `ne=[T, F]` (T innermost).
    pub(crate) fn features_to_ggml_layout(features: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * N_MELS];
        for t in 0..frames {
            for f in 0..N_MELS {
                out[t + f * frames] = features[t * N_MELS + f];
            }
        }
        out
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, WeightsError> {
        quoted_frontend_commitment()
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes() -> Result<u64, WeightsError> {
        quoted_frontend_commitment()
    }
}

fn quoted_frontend_commitment() -> Result<u64, WeightsError> {
    let mut bytes = allocation_commitment(std::mem::size_of::<WeSpeakerFrontend>())?;
    // Hamming window + ~80 sparse mel filters (each a short weight run) +
    // planner metadata. Page-rounded so a larger observed capacity still
    // reconciles with admission.
    let window_bytes = FRAME_LENGTH
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| WeightsError::InvalidInput("wespeaker frontend window overflow".into()))?;
    let filter_bytes = N_MELS
        .checked_mul(64 * std::mem::size_of::<f32>())
        .ok_or_else(|| WeightsError::InvalidInput("wespeaker frontend filter overflow".into()))?;
    for payload in [window_bytes, filter_bytes, 4096usize] {
        bytes = bytes
            .checked_add(allocation_commitment(payload)?)
            .ok_or_else(|| {
                WeightsError::InvalidInput("wespeaker frontend retained byte sum overflow".into())
            })?;
    }
    Ok(bytes)
}

fn cepstral_mean_normalize(feats: &mut [f32], n_frames: usize, n_mels: usize) {
    if n_frames == 0 {
        return;
    }
    for bin in 0..n_mels {
        let mut mean = 0.0f32;
        for fr in 0..n_frames {
            mean += feats[fr * n_mels + bin];
        }
        mean /= n_frames as f32;
        for fr in 0..n_frames {
            feats[fr * n_mels + bin] -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_samples(n: usize, freq_hz: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / 16_000.0).sin() * 0.3)
            .collect()
    }

    #[test]
    fn too_short_audio_yields_no_frames() {
        let frontend = WeSpeakerFrontend::new();
        let (feats, frames) = frontend.compute(&[0.0f32; 200]);
        assert_eq!(frames, 0);
        assert!(feats.is_empty());
    }

    #[test]
    fn cmn_is_mean_only_per_bin() {
        let frontend = WeSpeakerFrontend::new();
        let samples = sine_samples(16_000, 220.0);
        let (feats, frames) = frontend.compute(&samples);
        assert!(frames > 1);
        for bin in 0..N_MELS {
            let mean: f32 =
                (0..frames).map(|fr| feats[fr * N_MELS + bin]).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-4, "bin {bin} mean after CMN is {mean}");
        }
    }

    #[test]
    fn ggml_layout_is_t_innermost() {
        let frames = 3;
        let mut feats = vec![0.0f32; frames * N_MELS];
        for t in 0..frames {
            for f in 0..N_MELS {
                feats[t * N_MELS + f] = (t * 100 + f) as f32;
            }
        }
        let ggml = WeSpeakerFrontend::features_to_ggml_layout(&feats, frames);
        for t in 0..frames {
            for f in 0..N_MELS {
                assert_eq!(ggml[t + f * frames], (t * 100 + f) as f32);
            }
        }
    }
}
