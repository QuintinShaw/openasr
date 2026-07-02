//! Kaldi-compatible 80-bin log-mel filterbank front-end for speaker embedders.
//!
//! Matches the WeSpeaker/pyannote Kaldi fbank frontend: 25/10 ms, Hamming
//! window, pre-emphasis 0.97, power spectrum, log, mel range 20-8000 Hz,
//! `dither=0`, `snip_edges=true`, and pyannote's `waveform * 32768` scaling
//! before feature extraction. The output is per-utterance cepstral mean
//! normalized (subtract each bin's time-mean). Inputs are `f32` in `[-1, 1]`.
//! Validated against torchaudio (see `tests`).

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

const SAMPLE_RATE: f32 = 16_000.0;
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

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// One triangular mel filter as a contiguous weight run over FFT power bins.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

pub(crate) struct Fbank {
    window: [f32; FRAME_LENGTH],
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
    input_scale: f32,
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
        let mut window = [0.0f32; FRAME_LENGTH];
        for (n, w) in window.iter_mut().enumerate() {
            let phase = 2.0 * std::f32::consts::PI * n as f32 / (FRAME_LENGTH as f32 - 1.0);
            *w = 0.54 - 0.46 * phase.cos();
        }
        Self {
            window,
            filters: build_mel_filters(),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE),
            input_scale: config.input_scale,
        }
    }

    /// Compute the CMN-normalized log-mel features for `samples`, returning
    /// `(features [t*80] row-major, t)`. Returns `t == 0` for too-short input.
    pub(crate) fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
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
            // remove DC offset (subtract frame mean).
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for v in &mut frame {
                *v -= mean;
            }
            // pre-emphasis: x[i] -= 0.97 x[i-1] (back-to-front), x[0] -= 0.97 x[0].
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            // Apply the window into the zero-padded FFT input.
            for slot in fft_in.iter_mut() {
                *slot = 0.0;
            }
            for i in 0..FRAME_LENGTH {
                fft_in[i] = frame[i] * self.window[i];
            }
            r2c.process(&mut fft_in, &mut fft_out).expect("fft");
            // power spectrum.
            let mut power = [0.0f32; FFT_SIZE / 2 + 1];
            for (k, c) in fft_out.iter().enumerate() {
                power[k] = c.re * c.re + c.im * c.im;
            }
            // mel + log.
            for (bin, filter) in self.filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (j, weight) in filter.weights.iter().enumerate() {
                    energy += weight * power[filter.first_bin + j];
                }
                feats[fr * NUM_BINS + bin] = energy.max(f32::EPSILON).ln();
            }
        }
        // CMN: subtract each bin's mean over time.
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
