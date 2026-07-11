//! Kaldi-compatible 80-bin log-mel filterbank front-end for speaker embedders.
//!
//! Matches the WeSpeaker/pyannote Kaldi fbank frontend: 25/10 ms, Hamming
//! window, pre-emphasis 0.97, power spectrum, log, mel range 20-8000 Hz,
//! `dither=0`, `snip_edges=true`, and pyannote's `waveform * 32768` scaling
//! before feature extraction. The output is per-utterance cepstral mean
//! normalized (subtract each bin's time-mean). Inputs are `f32` in `[-1, 1]`.
//! Validated against torchaudio (see `tests`).
//!
//! The fbank math (framing, Hamming window, DC removal, pre-emphasis, FFT,
//! HTK mel filterbank, log floor) is the same computation
//! `firered_aed`/`dolphin`/`sensevoice` share via
//! [`crate::models::kaldi_fbank`] (Povey-window config there; this frontend
//! selects [`crate::models::kaldi_fbank::KaldiWindowKind::Hamming`]). Only
//! the per-utterance CMN (mean-only, no variance scaling -- distinct from
//! that shared module's CMVN affine, which is a per-checkpoint tensor
//! applied by the family frontend, not computed here) stays local.

use crate::models::kaldi_fbank::{KaldiFbankConfig, KaldiFbankFrontend, KaldiWindowKind};

const FRAME_LENGTH: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512; // next pow2 >= 400
const NUM_BINS: usize = 80;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 8_000.0;
const PREEMPH: f32 = 0.97;

#[derive(Debug, Clone, Copy)]
pub(crate) struct FbankConfig {
    input_scale: f32,
}

impl FbankConfig {
    pub(crate) fn wespeaker() -> Self {
        Self {
            input_scale: 32768.0,
        }
    }
}

pub(crate) struct Fbank {
    inner: KaldiFbankFrontend,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::wespeaker()
    }
}

impl Fbank {
    pub(crate) fn wespeaker() -> Self {
        Self::with_config(FbankConfig::wespeaker())
    }

    pub(crate) fn with_config(config: FbankConfig) -> Self {
        Self {
            inner: KaldiFbankFrontend::new(KaldiFbankConfig {
                sample_rate_hz: 16_000,
                frame_length: FRAME_LENGTH,
                frame_shift: FRAME_SHIFT,
                fft_size: FFT_SIZE,
                num_mel_bins: NUM_BINS,
                mel_low_hz: LOW_FREQ,
                mel_high_hz: HIGH_FREQ,
                preemph_coeff: PREEMPH,
                input_scale: config.input_scale,
                log_energy_floor: f32::EPSILON,
                window: KaldiWindowKind::Hamming,
            }),
        }
    }

    /// Compute the CMN-normalized log-mel features for `samples`, returning
    /// `(features [t*80] row-major, t)`. Returns `t == 0` for too-short input
    /// (or, since the shared engine also fail-closes on non-finite input
    /// where this frontend previously had no such check, for non-finite
    /// samples).
    pub(crate) fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let Ok(features) = self.inner.compute(samples) else {
            return (Vec::new(), 0);
        };
        let mut data = features.data;
        cepstral_mean_normalize(&mut data, features.n_frames, features.n_mels);
        (data, features.n_frames)
    }
}

/// CMN: subtract each mel bin's mean over time (no variance scaling --
/// distinct from [`crate::models::audio_frontend::per_feature_normalize`],
/// which also divides by std).
fn cepstral_mean_normalize(feats: &mut [f32], n_frames: usize, n_mels: usize) {
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

    /// Frozen copy of the pre-refactor `Fbank` (own FFT/mel-filter/window
    /// construction, before it delegated to `crate::models::kaldi_fbank`),
    /// kept only to pin the migrated implementation to byte-identical
    /// output.
    mod reference {
        use realfft::{RealFftPlanner, RealToComplex};
        use std::sync::Arc;

        const FRAME_LENGTH: usize = 400;
        const FRAME_SHIFT: usize = 160;
        const FFT_SIZE: usize = 512;
        const NUM_BINS: usize = 80;
        const LOW_FREQ: f32 = 20.0;
        const HIGH_FREQ: f32 = 8_000.0;
        const PREEMPH: f32 = 0.97;
        const SAMPLE_RATE: f32 = 16_000.0;

        fn mel_scale(freq: f32) -> f32 {
            1127.0 * (1.0 + freq / 700.0).ln()
        }

        struct MelFilter {
            first_bin: usize,
            weights: Vec<f32>,
        }

        pub(super) struct ReferenceFbank {
            window: [f32; FRAME_LENGTH],
            filters: Vec<MelFilter>,
            fft: Arc<dyn RealToComplex<f32>>,
            input_scale: f32,
        }

        impl ReferenceFbank {
            pub(super) fn wespeaker() -> Self {
                let mut window = [0.0f32; FRAME_LENGTH];
                for (n, w) in window.iter_mut().enumerate() {
                    let phase = 2.0 * std::f32::consts::PI * n as f32 / (FRAME_LENGTH as f32 - 1.0);
                    *w = 0.54 - 0.46 * phase.cos();
                }
                Self {
                    window,
                    filters: build_mel_filters(),
                    fft: RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE),
                    input_scale: 32768.0,
                }
            }

            pub(super) fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
                if samples.len() < FRAME_LENGTH {
                    return (Vec::new(), 0);
                }
                let n_frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
                let r2c = &self.fft;
                let mut fft_in = r2c.make_input_vec();
                let mut fft_out = r2c.make_output_vec();

                let mut feats = vec![0.0f32; n_frames * NUM_BINS];
                let mut frame = [0.0f32; FRAME_LENGTH];
                for fr in 0..n_frames {
                    let start = fr * FRAME_SHIFT;
                    for (dst, src) in frame.iter_mut().zip(&samples[start..start + FRAME_LENGTH]) {
                        *dst = *src * self.input_scale;
                    }
                    let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
                    for v in &mut frame {
                        *v -= mean;
                    }
                    for i in (1..FRAME_LENGTH).rev() {
                        frame[i] -= PREEMPH * frame[i - 1];
                    }
                    frame[0] -= PREEMPH * frame[0];
                    for slot in fft_in.iter_mut() {
                        *slot = 0.0;
                    }
                    for i in 0..FRAME_LENGTH {
                        fft_in[i] = frame[i] * self.window[i];
                    }
                    r2c.process(&mut fft_in, &mut fft_out).expect("fft");
                    let mut power = [0.0f32; FFT_SIZE / 2 + 1];
                    for (k, c) in fft_out.iter().enumerate() {
                        power[k] = c.re * c.re + c.im * c.im;
                    }
                    for (bin, filter) in self.filters.iter().enumerate() {
                        let mut energy = 0.0f32;
                        for (j, weight) in filter.weights.iter().enumerate() {
                            energy += weight * power[filter.first_bin + j];
                        }
                        feats[fr * NUM_BINS + bin] = energy.max(f32::EPSILON).ln();
                    }
                }
                for bin in 0..NUM_BINS {
                    let mut mean = 0.0f32;
                    for fr in 0..n_frames {
                        mean += feats[fr * NUM_BINS + bin];
                    }
                    mean /= n_frames as f32;
                    for fr in 0..n_frames {
                        feats[fr * NUM_BINS + bin] -= mean;
                    }
                }
                (feats, n_frames)
            }
        }

        fn build_mel_filters() -> Vec<MelFilter> {
            let n_fft_bins = FFT_SIZE / 2 + 1;
            let fft_bin_width = SAMPLE_RATE / FFT_SIZE as f32;
            let mel_low = mel_scale(LOW_FREQ);
            let mel_high = mel_scale(HIGH_FREQ);
            let mel_delta = (mel_high - mel_low) / (NUM_BINS as f32 + 1.0);

            let mut filters = Vec::with_capacity(NUM_BINS);
            for bin in 0..NUM_BINS {
                let left = mel_low + bin as f32 * mel_delta;
                let center = mel_low + (bin as f32 + 1.0) * mel_delta;
                let right = mel_low + (bin as f32 + 2.0) * mel_delta;
                let mut first_bin = None;
                let mut weights = Vec::new();
                for k in 0..n_fft_bins {
                    let freq = fft_bin_width * k as f32;
                    let mel = mel_scale(freq);
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
    }

    fn sine_samples(n: usize, freq_hz: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / 16_000.0).sin() * 0.3)
            .collect()
    }

    #[test]
    fn migrated_fbank_is_byte_identical_to_pre_refactor_reference() {
        let migrated = Fbank::wespeaker();
        let reference = reference::ReferenceFbank::wespeaker();
        for samples in [
            vec![0.01f32; 38_080],
            sine_samples(48_000, 220.0),
            sine_samples(16_500, 880.0),
        ] {
            let (migrated_feats, migrated_frames) = migrated.compute(&samples);
            let (reference_feats, reference_frames) = reference.compute(&samples);
            assert_eq!(migrated_frames, reference_frames);
            assert_eq!(migrated_feats, reference_feats);
        }
    }

    #[test]
    fn migrated_fbank_matches_reference_on_too_short_audio() {
        let migrated = Fbank::wespeaker();
        let reference = reference::ReferenceFbank::wespeaker();
        let samples = vec![0.0f32; 200];
        assert_eq!(migrated.compute(&samples), reference.compute(&samples));
    }
}
