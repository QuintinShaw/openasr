//! FireRedVAD kaldi-fbank frontend + global CMVN.
//!
//! FireRedVAD's own feature extractor (`fireredvad/core/audio_feat.py`) runs
//! `kaldi_native_fbank` with `num_mel_bins=80, frame_length=25, frame_shift=10,
//! dither=0, snip_edges=true` over int16-magnitude samples, then applies a
//! global CMVN `(x - mean) * inv_stddev` fit on the training corpus (baked
//! into the vendored checkpoint as `frontend.cmvn.mean` /
//! `frontend.cmvn.inv_stddev`).
//!
//! The fbank math is byte-identical in structure to
//! `crate::models::firered_aed::frontend` (both upstreams -- same team --
//! run kaldi fbank with the same 80/25ms/10ms/16kHz/dither-0 config and
//! defaults: Povey window, remove_dc_offset, pre-emphasis 0.97, HTK mel
//! 20 Hz..Nyquist, `log(max(energy, eps))`). Kept family-local like every
//! other frontend so nothing generic grows a FireRed special case.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;
// `pub(crate)`: the Stream-VAD streaming detector needs these to know how
// many raw samples must accumulate before the next fbank frame is ready.
pub(crate) const FRAME_LENGTH: usize = 400; // 25 ms @ 16 kHz
pub(crate) const FRAME_SHIFT: usize = 160; // 10 ms @ 16 kHz
const FFT_SIZE: usize = 512; // next pow2 >= 400 (kaldi rounds the window up)
pub(crate) const NUM_MEL_BINS: usize = 80;
const MEL_LOW_HZ: f32 = 20.0;
const MEL_HIGH_HZ: f32 = 8_000.0; // high_freq <= 0 in kaldi => Nyquist
const PREEMPH_COEFF: f32 = 0.97;
/// float `[-1, 1]` -> int16 magnitude, the domain kaldi fbank operates in
/// (upstream feeds raw int16-valued waveforms via `soundfile.read(..,
/// dtype="int16")`).
const INPUT_SCALE: f32 = 32_768.0;
/// kaldi mel-energy floor before the log.
const LOG_ENERGY_FLOOR: f32 = 1.192_092_9e-7;

/// Pre-CMVN log-mel features, row-major `[frame][mel]` (mel innermost).
pub(crate) struct FbankFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
}

/// One triangular mel filter as a contiguous weight run over FFT power bins.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

pub(crate) struct FireRedVadFbankFrontend {
    window: [f32; FRAME_LENGTH],
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl FireRedVadFbankFrontend {
    pub(crate) fn new() -> Self {
        let mut window = [0.0f32; FRAME_LENGTH];
        for (n, w) in window.iter_mut().enumerate() {
            // Povey window: a Hann raised to 0.85 (kaldi default).
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
    /// `i * FRAME_SHIFT` with no edge reflection. Returns `n_frames == 0`
    /// (empty data) for audio shorter than one frame.
    pub(crate) fn compute(&self, samples: &[f32]) -> FbankFeatures {
        if samples.len() < FRAME_LENGTH {
            return FbankFeatures {
                data: Vec::new(),
                n_frames: 0,
            };
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
                .expect("firered VAD fbank rfft");
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
        FbankFeatures {
            data: feats,
            n_frames,
        }
    }
}

/// Apply the vendored global CMVN `(x - mean) * inv_stddev` in place, per mel
/// bin. `features` is row-major `[frames, NUM_MEL_BINS]`.
pub(crate) fn apply_cmvn(
    features: &mut [f32],
    mean: &[f32; NUM_MEL_BINS],
    inv_stddev: &[f32; NUM_MEL_BINS],
) {
    for frame in features.chunks_exact_mut(NUM_MEL_BINS) {
        for ((value, m), s) in frame.iter_mut().zip(mean).zip(inv_stddev) {
            *value = (*value - *m) * *s;
        }
    }
}

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi triangular mel filterbank over `FFT_SIZE/2 + 1` power bins:
/// `NUM_MEL_BINS` peak-normalized triangles spanning `[MEL_LOW_HZ, MEL_HIGH_HZ]`
/// on the HTK mel scale, gated to bins strictly inside the filter band.
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
        let samples = vec![0.01f32; 48_000];
        let features = FireRedVadFbankFrontend::new().compute(&samples);
        assert_eq!(features.n_frames, 298);
        assert_eq!(features.data.len(), 298 * NUM_MEL_BINS);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_too_short_audio_by_returning_zero_frames() {
        let samples = vec![0.0f32; 200];
        let features = FireRedVadFbankFrontend::new().compute(&samples);
        assert_eq!(features.n_frames, 0);
        assert!(features.data.is_empty());
    }
}
