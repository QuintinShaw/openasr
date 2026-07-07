//! Dolphin frontends: kaldi-fbank (cn-dialect, WeNet format) and ESPnet
//! `DefaultFrontend` (multilingual) + global CMVN.
//!
//! [`DolphinFbankFrontend`] reproduces the reference feature pipeline used to fit
//! `small.cn`/`cn-dialect-base`: `torchaudio.compliance.kaldi.fbank` with the
//! `train.yaml` `fbank_conf` (25 ms window, 10 ms shift, 80 mel bins, `dither=0`)
//! over the waveform scaled by `1 << 15` (float `[-1, 1]` back to the int16
//! magnitude kaldi expects), followed by the checkpoint's `global_cmvn`
//! `(x - mean) * istd`.
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
//!
//! [`DolphinEspnetFrontend`] is the multilingual (dolphin-small/dolphin-base)
//! counterpart: those checkpoints' `train.yaml` carries a `dataset_conf`
//! with *both* a `fbank_conf` and a `frontend_conf` key, and DataoceanAI's
//! own `dolphin/processor.py::extract_feats` picks the ESPnet `DefaultFrontend`
//! (`frontend_conf`) whenever it is present, ignoring `fbank_conf` entirely --
//! confirmed by reading `DataoceanAI/Dolphin`'s inference source directly
//! (`dolphin/model.py::Stft`/`LogMel`/`DefaultFrontend`), not assumed. That
//! frontend is a materially different pipeline from the kaldi one above: a
//! periodic Hann window (not Povey), `torch.stft`-style reflect-centered
//! framing (not kaldi `snip_edges`), no pre-emphasis/DC-removal/int16 scaling,
//! and a **Slaney**-normalized librosa mel scale/filterbank (not HTK
//! peak-normalized) -- see `build_slaney_mel_filters`.

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

// --- ESPnet DefaultFrontend (multilingual dolphin-small/dolphin-base) -------

/// `frontend_conf` shared by dolphin-small and dolphin-base `train.yaml`:
/// `n_fft: 512, win_length: 400, hop_length: 160, fs: 16000`. `LogMel`'s own
/// defaults fill in the rest (unset in `frontend_conf`): `n_mels=80`,
/// `fmin=0`, `fmax=fs/2`, `htk=False` (librosa Slaney mel).
const ESPNET_N_FFT: usize = 512;
const ESPNET_WIN_LENGTH: usize = 400;
const ESPNET_HOP_LENGTH: usize = 160;
pub(crate) const ESPNET_NUM_MEL_BINS: usize = 80;
const ESPNET_FMIN_HZ: f64 = 0.0;
/// `LogMel.__init__`'s `fmax = fs / 2` default (16 kHz Nyquist).
const ESPNET_FMAX_HZ: f64 = 8_000.0;
/// `torch.clamp(mel_feat, min=1e-10)` before `.log()` in `LogMel.forward`.
const ESPNET_LOG_ENERGY_FLOOR: f32 = 1.0e-10;

/// One triangular Slaney mel filter over the full `0..=n_fft/2` power-bin
/// range (unlike [`MelFilter`], not gated to a contiguous nonzero run: the
/// area-normalized Slaney weights are small enough near the band edges that
/// hand-rolling a first/last-bin search buys little, and dense storage keeps
/// the librosa reference formula ported verbatim below).
struct EspnetMelFilter {
    weights: Vec<f32>,
}

pub(crate) struct DolphinEspnetFrontend {
    /// Periodic Hann window (`torch.hann_window(win_length)` default,
    /// `periodic=True`), zero-padded/centered into the `n_fft`-length FFT
    /// input buffer (`left = (n_fft - win_length) / 2`), mirroring
    /// `torch.stft`'s own window centering when `win_length < n_fft`.
    windowed_zero_pad: [f32; ESPNET_N_FFT],
    filters: Vec<EspnetMelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for DolphinEspnetFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DolphinEspnetFrontend")
            .field("n_mels", &self.filters.len())
            .finish_non_exhaustive()
    }
}

impl Default for DolphinEspnetFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl DolphinEspnetFrontend {
    pub(crate) fn new() -> Self {
        let mut windowed_zero_pad = [0.0f32; ESPNET_N_FFT];
        let left = (ESPNET_N_FFT - ESPNET_WIN_LENGTH) / 2;
        for (n, slot) in windowed_zero_pad[left..left + ESPNET_WIN_LENGTH]
            .iter_mut()
            .enumerate()
        {
            // Periodic Hann: `0.5 - 0.5*cos(2*pi*n/N)` (N = win_length in the
            // denominator, not N-1 -- `torch.hann_window`'s default).
            *slot = 0.5
                - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / ESPNET_WIN_LENGTH as f32).cos();
        }
        Self {
            windowed_zero_pad,
            filters: build_slaney_mel_filters(
                SAMPLE_RATE_HZ,
                ESPNET_N_FFT,
                ESPNET_NUM_MEL_BINS,
                ESPNET_FMIN_HZ,
                ESPNET_FMAX_HZ,
            ),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(ESPNET_N_FFT),
        }
    }

    /// Compute pre-CMVN ESPnet log-mel features for 16 kHz mono `samples`
    /// (float in `[-1, 1]`). `torch.stft(..., center=True, pad_mode="reflect")`
    /// framing: the signal is conceptually reflect-padded by `n_fft/2` on each
    /// side (exactly `win_length`'s slack either side of `hop_length`), giving
    /// `n_frames = 1 + samples.len() / hop_length` (the padding cancels the
    /// `n_fft` term because `2 * (n_fft/2) == n_fft`). No pre-emphasis, DC
    /// removal, or dither: this frontend runs directly on the float PCM the
    /// caller already decoded (no int16 rescale, unlike the kaldi frontend).
    pub(crate) fn compute(
        &self,
        samples: &[f32],
    ) -> Result<DolphinFbankFeatures, DolphinFrontendError> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err(DolphinFrontendError::UnsupportedAudio);
        }
        let pad = ESPNET_N_FFT / 2;
        let n_frames = 1 + samples.len() / ESPNET_HOP_LENGTH;
        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        let mut power = vec![0.0f32; ESPNET_N_FFT / 2 + 1];

        let n_mels = self.filters.len();
        let mut feats = vec![0.0f32; n_frames * n_mels];
        for fr in 0..n_frames {
            let frame_start = (fr * ESPNET_HOP_LENGTH) as i64 - pad as i64;
            for (j, slot) in fft_in.iter_mut().enumerate() {
                let sample = reflect_sample(samples, frame_start + j as i64);
                *slot = sample * self.windowed_zero_pad[j];
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("dolphin espnet fbank rfft");
            for (bin, value) in fft_out.iter().enumerate() {
                power[bin] = value.re * value.re + value.im * value.im;
            }
            for (bin, filter) in self.filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (weight, p) in filter.weights.iter().zip(power.iter()) {
                    energy += weight * p;
                }
                feats[fr * n_mels + bin] = energy.max(ESPNET_LOG_ENERGY_FLOOR).ln();
            }
        }
        Ok(DolphinFbankFeatures {
            data: feats,
            n_frames,
            n_mels,
        })
    }
}

/// `numpy`/`torch` `mode="reflect"` boundary sample at signal index `k`
/// (may be negative or `>= samples.len()`): mirrors the signal about each
/// edge without repeating the edge sample itself, matching `torch.stft`'s
/// `pad_mode="reflect"` centering. Single-sample signals fold to that sample.
fn reflect_sample(samples: &[f32], k: i64) -> f32 {
    let len = samples.len();
    if len <= 1 {
        return samples.first().copied().unwrap_or(0.0);
    }
    let period = 2 * (len as i64 - 1);
    let mut m = k % period;
    if m < 0 {
        m += period;
    }
    let m = if m < len as i64 { m } else { period - m };
    samples[m as usize]
}

/// librosa's Slaney (non-HTK) mel scale: linear below 1 kHz, logarithmic
/// above. `htk=False` is `librosa.filters.mel`'s default and what
/// `DefaultFrontend`'s `LogMel` uses when `htk` is left unset (as
/// dolphin-small/dolphin-base's `frontend_conf` does).
fn hz_to_mel_slaney(freq: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
    let logstep = 6.4f64.ln() / 27.0;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep
    } else {
        freq / F_SP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False)` (default
/// `norm="slaney"`): `n_mels` overlapping triangles over `n_fft/2+1` linear FFT
/// bins, with edges chosen evenly spaced in Slaney-mel space and each row
/// scaled by `2 / (mel_edge[i+2] - mel_edge[i])` (area, not peak, normalized --
/// the opposite convention from the kaldi/HTK filters above).
fn build_slaney_mel_filters(
    sample_rate_hz: u32,
    n_fft: usize,
    n_mels: usize,
    fmin_hz: f64,
    fmax_hz: f64,
) -> Vec<EspnetMelFilter> {
    let n_fft_bins = n_fft / 2 + 1;
    let sr = f64::from(sample_rate_hz);

    // `fft_frequencies`: linear bins `k * sr / n_fft`.
    let fft_freqs: Vec<f64> = (0..n_fft_bins)
        .map(|k| k as f64 * sr / n_fft as f64)
        .collect();

    // `mel_frequencies(n_mels + 2, fmin, fmax)`: `n_mels+2` Hz edges, evenly
    // spaced in mel space then mapped back to Hz.
    let min_mel = hz_to_mel_slaney(fmin_hz);
    let max_mel = hz_to_mel_slaney(fmax_hz);
    let n_edges = n_mels + 2;
    // `mel_frequencies`: Hz values of `n_edges` mel-evenly-spaced points.
    // NOTE: librosa's `mel()` keeps `mel_f`/`fdiff`/`ramps` in **Hz** space
    // from here on (only the edge placement itself uses the mel scale) --
    // it never round-trips back through `hz_to_mel`.
    let mel_edges_hz: Vec<f64> = (0..n_edges)
        .map(|i| {
            let mel = if n_edges == 1 {
                min_mel
            } else {
                min_mel + (max_mel - min_mel) * i as f64 / (n_edges - 1) as f64
            };
            mel_to_hz_slaney(mel)
        })
        .collect();
    let fdiff: Vec<f64> = mel_edges_hz.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_edges_hz[m + 2] - mel_edges_hz[m]);
        let mut weights = vec![0.0f32; n_fft_bins];
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = -(mel_edges_hz[m] - freq) / fdiff[m];
            let upper = (mel_edges_hz[m + 2] - freq) / fdiff[m + 1];
            let weight = lower.min(upper).max(0.0) * enorm;
            weights[k] = weight as f32;
        }
        filters.push(EspnetMelFilter { weights });
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

    // --- ESPnet DefaultFrontend parity (dolphin-small/dolphin-base) --------
    //
    // Golden values below are from a direct port of
    // `dolphin/model.py::Stft`+`LogMel` (the exact classes DataoceanAI's own
    // inference code runs for these checkpoints), computed with
    // `torch.stft(n_fft=512, win_length=400, hop_length=160, window=hann_window(400),
    // center=True, pad_mode="reflect")` -> power -> `librosa.filters.mel(sr=16000,
    // n_fft=512, n_mels=80, fmin=0, fmax=8000, htk=False) @ power` -> `clamp(1e-10).log()`,
    // over `fixtures/jfk.wav` (16 kHz mono, 176000 samples).

    fn jfk_samples() -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
        crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &path,
            "dolphin_espnet_frontend_test",
            "jfk.wav",
        )
        .expect("load jfk.wav")
    }

    #[test]
    fn slaney_mel_filterbank_matches_librosa_reference() {
        let filters = build_slaney_mel_filters(
            SAMPLE_RATE_HZ,
            ESPNET_N_FFT,
            ESPNET_NUM_MEL_BINS,
            0.0,
            8_000.0,
        );
        assert_eq!(filters.len(), 80);

        // librosa.filters.mel(...)[0][:10]
        let expected_row0: [f32; 10] = [
            0.0,
            2.253_456e-2,
            8.637_710_5e-3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (bin, &expected) in expected_row0.iter().enumerate() {
            let actual = filters[0].weights[bin];
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "mel row0 bin{bin}: actual {actual} expected {expected}"
            );
        }

        // librosa.filters.mel(...)[79][240:257]
        let expected_row79_tail: [f32; 17] = [
            1.066_234_2e-3,
            1.430_553_3e-3,
            1.794_872_1e-3,
            2.159_191_3e-3,
            2.523_510_3e-3,
            2.887_829_2e-3,
            3.252_148_4e-3,
            3.155_337e-3,
            2.804_744e-3,
            2.454_151e-3,
            2.103_558e-3,
            1.752_965e-3,
            1.402_372e-3,
            1.051_779e-3,
            7.011_86e-4,
            3.505_93e-4,
            0.0,
        ];
        for (offset, &expected) in expected_row79_tail.iter().enumerate() {
            let actual = filters[79].weights[240 + offset];
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "mel row79 bin{}: actual {actual} expected {expected}",
                240 + offset
            );
        }
    }

    #[test]
    fn espnet_frontend_matches_reference_logmel_frame_count() {
        let samples = jfk_samples();
        assert_eq!(samples.len(), 176_000);
        let features = DolphinEspnetFrontend::new()
            .compute(&samples)
            .expect("espnet fbank");
        // 1 + 176000 / 160 = 1101 (torch.stft center=True reflect framing).
        assert_eq!(features.n_frames, 1_101);
        assert_eq!(features.n_mels, 80);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn espnet_frontend_matches_reference_logmel_values() {
        let samples = jfk_samples();
        let features = DolphinEspnetFrontend::new()
            .compute(&samples)
            .expect("espnet fbank");

        let frame_at = |frame: usize| &features.data[frame * 80..frame * 80 + 8];

        // frame 0 is fully inside the reflect-padded silence lead-in -> the
        // 1e-10 clamp floor, ln(1e-10) = -23.02585.
        for &value in frame_at(0) {
            assert!(
                (value - (-23.025_85)).abs() < 1.0e-3,
                "frame0 expected clamp floor, got {value}"
            );
        }

        // frame 500 first 8 mel bins (torch/librosa reference).
        let expected_frame500: [f32; 8] = [
            -6.403_16,
            -5.750_341_4,
            -5.296_384_3,
            -4.686_787,
            -4.055_321,
            -4.389_283,
            -3.961_102_7,
            -3.866_433_4,
        ];
        for (bin, &expected) in expected_frame500.iter().enumerate() {
            let actual = frame_at(500)[bin];
            assert!(
                (actual - expected).abs() < 1.0e-2,
                "frame500 bin{bin}: actual {actual} expected {expected}"
            );
        }

        // last frame (1100) first 8 mel bins.
        let expected_frame_last: [f32; 8] = [
            -9.255_437,
            -8.575_424,
            -7.676_475,
            -6.291_418,
            -3.404_137_6,
            -3.624_057_8,
            -3.642_896_4,
            -2.373_729_7,
        ];
        for (bin, &expected) in expected_frame_last.iter().enumerate() {
            let actual = frame_at(1_100)[bin];
            assert!(
                (actual - expected).abs() < 1.0e-2,
                "frame1100 bin{bin}: actual {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn espnet_frontend_rejects_empty_audio() {
        assert!(matches!(
            DolphinEspnetFrontend::new().compute(&[]),
            Err(DolphinFrontendError::UnsupportedAudio)
        ));
    }
}
