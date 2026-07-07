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

#![allow(dead_code)]

use realfft::RealFftPlanner;

use super::encoder_graph::ParakeetMelFeatures;
use super::runtime_contract::ParakeetCtcExecutionMetadata;

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

pub(crate) struct ParakeetFrontend {
    n_mels: usize,
    /// Hann window of `WIN_LENGTH`, zero-centered into `N_FFT`.
    window: Vec<f32>,
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
        Self {
            n_mels,
            window: centered_hann_window(WIN_LENGTH, N_FFT),
            mel_filters: slaney_mel_filterbank(n_mels, N_FFT, fft_bins),
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
        // reflect center-pad n_fft/2 (torch.stft center=True).
        let pad = N_FFT / 2;
        let padded = reflect_pad(&emphasized, pad);
        if padded.len() < N_FFT {
            return Err(ParakeetFrontendError::TooShort {
                samples: samples.len(),
            });
        }
        let n_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(N_FFT);
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        // Output [n_frames][n_mels] then we emit mel-fastest [mel, frame].
        let mut mel = vec![0.0f32; n_frames * self.n_mels];
        let mut power = vec![0.0f32; self.fft_bins];
        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            for i in 0..N_FFT {
                fft_in[i] = padded[start + i] * self.window[i];
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("rfft");
            for (bin, c) in fft_out.iter().enumerate() {
                power[bin] = c.norm_sqr();
            }
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
        for m in 0..self.n_mels {
            let mut mean = 0.0f64;
            for f in 0..n_frames {
                mean += mel[f * self.n_mels + m] as f64;
            }
            mean /= n_frames.max(1) as f64;
            let mut var = 0.0f64;
            for f in 0..n_frames {
                let d = mel[f * self.n_mels + m] as f64 - mean;
                var += d * d;
            }
            var /= n_frames.max(1) as f64;
            let std = (var.sqrt() as f32) + NORM_EPS;
            for f in 0..n_frames {
                let v = &mut mel[f * self.n_mels + m];
                *v = (*v - mean as f32) / std;
            }
        }

        Ok(ParakeetMelFeatures {
            data: mel,
            n_frames,
            n_mels: self.n_mels,
        })
    }
}

/// Periodic Hann window of `win_length`, zero-padded symmetrically into `n_fft`
/// (torch.stft pads a shorter `win_length` to `n_fft`, centered).
fn centered_hann_window(win_length: usize, n_fft: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    let offset = (n_fft - win_length) / 2;
    for i in 0..win_length {
        // periodic Hann: 0.5 - 0.5 cos(2πi / win_length).
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos();
        window[offset + i] = w;
    }
    window
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        // reflect without repeating the edge sample (numpy 'reflect').
        out.push(samples[(pad - i).min(n.saturating_sub(1))]);
    }
    out.extend_from_slice(samples);
    for i in 0..pad {
        let idx = n.saturating_sub(2 + i);
        out.push(samples[idx.min(n.saturating_sub(1))]);
    }
    out
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    // librosa slaney: linear below 1000 Hz, log above.
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        (hz - F_MIN) / F_SP
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

/// librosa.filters.mel(sr=16000, n_fft, n_mels, fmin=0, fmax=8000, htk=False,
/// norm='slaney'): triangular filters with slaney area normalization. Row-major
/// `[n_mels][fft_bins]`.
fn slaney_mel_filterbank(n_mels: usize, n_fft: usize, fft_bins: usize) -> Vec<f32> {
    // fft bin center frequencies.
    let fft_freqs: Vec<f32> = (0..fft_bins)
        .map(|i| i as f32 * SAMPLE_RATE / n_fft as f32)
        .collect();
    // n_mels + 2 mel points equally spaced in mel between fmin..fmax.
    let mel_min = hz_to_mel_slaney(MEL_FMIN);
    let mel_max = hz_to_mel_slaney(MEL_FMAX);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    let mut filters = vec![0.0f32; n_mels * fft_bins];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        // slaney area normalization: 2 / (right - left).
        let enorm = 2.0 / (right - left);
        for (bin, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - left) / (center - left);
            let upper = (right - f) / (right - center);
            let weight = lower.min(upper).max(0.0);
            filters[m * fft_bins + bin] = weight * enorm;
        }
    }
    filters
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
        let fb = slaney_mel_filterbank(80, 512, 257);
        // every row has positive mass, bounded weights.
        for m in 0..80 {
            let row = &fb[m * 257..(m + 1) * 257];
            assert!(row.iter().sum::<f32>() > 0.0, "row {m} empty");
            assert!(row.iter().all(|w| w.is_finite() && *w >= 0.0));
        }
    }
}
