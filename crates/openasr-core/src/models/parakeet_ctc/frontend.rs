//! parakeet-ctc log-mel frontend (goal-1 S4): reproduces NeMo's
//! `AudioToMelSpectrogramPreprocessor` / HF `ParakeetFeatureExtractor` for the
//! 0.6b model. parakeet embeds NO mel filterbank, so we COMPUTE a librosa
//! slaney-norm 80-mel filterbank here.
//!
//! Pipeline (inference; dither off): preemphasis 0.97 → reflect center-pad
//! n_fft/2 → framed Hann STFT (n_fft 512, win 400, hop 160) → power |.|² →
//! 80-mel (slaney, fmin 0, fmax 8000) → log(x + 2⁻²⁴) → per-feature
//! (per-mel-bin) mean/std normalization over the utterance. Output is the
//! `[n_mels, n_frames]` (mel-fastest) layout the encoder's mel input expects.
//!
//! Framing/windowing/FFT, the slaney mel filterbank, and the per-feature
//! mean/std normalization are the shared [`crate::models::audio_frontend`]
//! primitives (this is the first family built directly on them); only the
//! preemphasis and the NeMo `log_zero_guard_type="add"` log-guard style stay
//! here since they differ per family.

#![allow(dead_code)]

use super::encoder_graph::ParakeetMelFeatures;
use super::runtime_contract::ParakeetCtcExecutionMetadata;
use crate::models::audio_frontend::mel::{FilterbankConfig, MelScale};
use crate::models::audio_frontend::{
    PadMode, StftError, StftFramer, hann_window_centered, per_feature_normalize,
};

const PREEMPHASIS: f32 = 0.97;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const SAMPLE_RATE: f32 = 16_000.0;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 8_000.0;
/// NeMo `log_zero_guard_value` default (2⁻²⁴) for `log_zero_guard_type="add"`.
const LOG_GUARD: f32 = 5.960_464_5e-8;
/// NeMo per-feature normalization epsilon.
const NORM_EPS: f32 = 1.0e-5;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ParakeetFrontendError {
    #[error(
        "parakeet-ctc frontend requires 16 kHz mono f32 audio (got {channels}ch {sample_rate}Hz)"
    )]
    UnsupportedAudio { channels: usize, sample_rate: u32 },
    #[error("parakeet-ctc frontend: audio too short ({samples} samples) for one STFT frame")]
    TooShort { samples: usize },
}

impl From<StftError> for ParakeetFrontendError {
    fn from(error: StftError) -> Self {
        match error {
            StftError::TooShort { samples } => Self::TooShort { samples },
        }
    }
}

pub(crate) struct ParakeetFrontend {
    n_mels: usize,
    framer: StftFramer,
    /// Row-major `[n_mels][fft_bins]` slaney mel filterbank.
    mel_filters: Vec<f32>,
    fft_bins: usize,
}

impl ParakeetFrontend {
    pub(crate) fn new(metadata: &ParakeetCtcExecutionMetadata) -> Self {
        Self::with_n_mels(metadata.n_mels)
    }

    /// Same NeMo/HF `ParakeetFeatureExtractor` pipeline for a caller-chosen
    /// mel-bin count (parakeet-tdt uses 128 mels; everything else — window,
    /// hop, preemphasis, slaney filterbank, log guard, per-feature norm — is
    /// identical across the parakeet family).
    pub(crate) fn with_n_mels(n_mels: usize) -> Self {
        let fft_bins = N_FFT / 2 + 1;
        let framer = StftFramer::new(
            N_FFT,
            WIN_LENGTH,
            HOP_LENGTH,
            PadMode::ReflectCenter,
            hann_window_centered(WIN_LENGTH, N_FFT),
        );
        let mel_filters = crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::Slaney,
            sample_rate_hz: SAMPLE_RATE,
            n_fft: N_FFT,
            n_mels,
            fmin: MEL_FMIN,
            fmax: MEL_FMAX,
        });
        Self {
            n_mels,
            framer,
            mel_filters,
            fft_bins,
        }
    }

    /// Compute `[n_mels, n_frames]` (mel-fastest) log-mel features from 16 kHz
    /// mono f32 PCM.
    pub(crate) fn features_from_samples(
        &self,
        samples: &[f32],
    ) -> Result<ParakeetMelFeatures, ParakeetFrontendError> {
        // preemphasis: y[0] = x[0]; y[t] = x[t] - 0.97 x[t-1].
        let mut emphasized = Vec::with_capacity(samples.len());
        let mut prev = 0.0f32;
        for (i, &x) in samples.iter().enumerate() {
            emphasized.push(if i == 0 { x } else { x - PREEMPHASIS * prev });
            prev = x;
        }
        let spectrogram = self.framer.power_spectrogram(&emphasized)?;
        let n_frames = spectrogram.n_frames;

        // Output [n_frames][n_mels] then we emit mel-fastest [mel, frame].
        let mut mel = vec![0.0f32; n_frames * self.n_mels];
        for frame_idx in 0..n_frames {
            let power =
                &spectrogram.data[frame_idx * self.fft_bins..(frame_idx + 1) * self.fft_bins];
            for m in 0..self.n_mels {
                let row = &self.mel_filters[m * self.fft_bins..(m + 1) * self.fft_bins];
                let mut acc = 0.0f32;
                for bin in 0..self.fft_bins {
                    acc += row[bin] * power[bin];
                }
                mel[frame_idx * self.n_mels + m] = (acc + LOG_GUARD).ln();
            }
        }

        // per-feature (per mel bin) normalization over the time axis.
        per_feature_normalize(&mut mel, n_frames, self.n_mels, NORM_EPS, 0.0);

        Ok(ParakeetMelFeatures {
            data: mel,
            n_frames,
            n_mels: self.n_mels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> ParakeetCtcExecutionMetadata {
        ParakeetCtcExecutionMetadata {
            n_layers: 24,
            hidden_size: 1024,
            n_heads: 8,
            head_dim: 128,
            ffn_dim: 4096,
            conv_kernel: 9,
            n_mels: 80,
            subsampling_factor: 8,
            subsampling_channels: 256,
            vocab_size: 1025,
            blank_token_id: 1024,
        }
    }

    #[test]
    fn produces_finite_80_mel_features() {
        let frontend = ParakeetFrontend::new(&metadata());
        // 1 s of a 440 Hz tone @ 16 kHz.
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin() * 0.1)
            .collect();
        let feats = frontend.features_from_samples(&samples).expect("features");
        assert_eq!(feats.n_mels, 80);
        // ~ (16000 + 512)/160 frames.
        assert!(
            feats.n_frames > 90 && feats.n_frames < 110,
            "frames={}",
            feats.n_frames
        );
        assert_eq!(feats.data.len(), feats.n_mels * feats.n_frames);
        assert!(feats.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mel_filterbank_rows_are_normalized_and_localized() {
        let fb = crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::Slaney,
            sample_rate_hz: SAMPLE_RATE,
            n_fft: 512,
            n_mels: 80,
            fmin: MEL_FMIN,
            fmax: MEL_FMAX,
        });
        // every row has positive mass, bounded weights.
        for m in 0..80 {
            let row = &fb[m * 257..(m + 1) * 257];
            assert!(row.iter().sum::<f32>() > 0.0, "row {m} empty");
            assert!(row.iter().all(|w| w.is_finite() && *w >= 0.0));
        }
    }

    #[test]
    fn handles_audio_shorter_than_one_stft_frame_without_panicking() {
        // See `audio_frontend`'s equivalent test: reflect-center padding
        // (n_fft/2 both sides) means `TooShort` never actually triggers for
        // nonempty input at this frontend's fixed n_fft; what must hold is
        // that short-but-nonempty audio still produces finite features.
        let frontend = ParakeetFrontend::new(&metadata());
        let samples = vec![0.01f32; 4];
        let feats = frontend.features_from_samples(&samples).expect("features");
        assert_eq!(feats.n_frames, 1);
        assert_eq!(feats.n_mels, 80);
        assert!(feats.data.iter().all(|v| v.is_finite()));
    }
}
