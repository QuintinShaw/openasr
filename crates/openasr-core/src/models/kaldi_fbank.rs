//! Shared kaldi-style fbank frontend engine (batch, `snip_edges=true`).
//!
//! `firered_aed`, dolphin's kaldi frontend (`DolphinFbankFrontend`, the
//! `small.cn`/`cn-dialect-base` pipeline), and sensevoice all run the exact
//! same `torchaudio.compliance.kaldi.fbank`-equivalent computation: 25 ms
//! window / 10 ms shift, N mel bins, `dither=0`, `remove_dc_offset`,
//! pre-emphasis, HTK mel scale over `[low_hz, high_hz]`, `snip_edges=true`
//! framing (no edge reflection), and `log(max(energy, floor))`. The three
//! families differed only in their analysis window (Povey vs Hamming) and a
//! couple of numeric constants -- the FFT/DC-removal/pre-emphasis/mel-filter
//! math was byte-for-byte duplicated three times. This module is that shared
//! engine, parameterized by [`KaldiFbankConfig`]; the family modules keep
//! their own error types, CMVN affine (the tensor names and sign convention
//! differ per checkpoint), and any family-specific post-processing (e.g.
//! sensevoice's LFR stacking).
//!
//! `xasr_zipformer`'s HTK-variant frontend is deliberately **not** folded in
//! here: it runs `snip_edges=false` streaming framing (reflect-padded,
//! incremental frame-range computation with O(1)-memory caching), skips
//! pre-emphasis/DC-removal/int16 rescale entirely, and stores mel filters
//! densely (not the sparse first-bin+run representation below) to support
//! O(1) range slicing. Making this engine cover that shape would mean
//! threading streaming-specific control flow through a batch-oriented API --
//! a shared-layer special case, not a config difference. See
//! `models/xasr_zipformer/frontend.rs`'s module doc for its own rationale.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

/// Analysis window applied to each frame before the FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KaldiWindowKind {
    /// Hann raised to the kaldi/torchaudio Povey exponent (0.85).
    Povey,
    /// `0.54 - 0.46 * cos(2*pi*n / (N-1))`.
    Hamming,
}

/// Frontend geometry + kaldi-fbank knobs. All fields are the same ones each
/// family's frontend module documents on its own local `const`s; the values
/// firered_aed/dolphin/sensevoice each currently pass are byte-identical
/// except for `window`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KaldiFbankConfig {
    pub sample_rate_hz: u32,
    pub frame_length: usize,
    pub frame_shift: usize,
    pub fft_size: usize,
    pub num_mel_bins: usize,
    pub mel_low_hz: f32,
    pub mel_high_hz: f32,
    pub preemph_coeff: f32,
    /// float `[-1, 1]` -> int16 magnitude, the domain kaldi fbank operates in.
    pub input_scale: f32,
    /// mel-energy floor before the log.
    pub log_energy_floor: f32,
    pub window: KaldiWindowKind,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum KaldiFbankError {
    #[error("kaldi fbank frontend requires finite audio")]
    UnsupportedAudio,
    #[error("kaldi fbank frontend produced no frames from {samples} samples")]
    NoFrames { samples: usize },
}

/// Pre-CMVN log-mel features, row-major `[frame][mel]` (mel innermost).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KaldiFbankFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

/// One triangular mel filter as a contiguous weight run over FFT power bins.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

pub(crate) struct KaldiFbankFrontend {
    config: KaldiFbankConfig,
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for KaldiFbankFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KaldiFbankFrontend")
            .field("n_mels", &self.filters.len())
            .finish_non_exhaustive()
    }
}

impl KaldiFbankFrontend {
    pub(crate) fn new(config: KaldiFbankConfig) -> Self {
        let window = build_window(config.window, config.frame_length);
        let filters = build_mel_filters(&config);
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(config.fft_size);
        Self {
            config,
            window,
            filters,
            fft,
        }
    }

    /// Compute pre-CMVN kaldi log-mel features for mono `samples` (float in
    /// `[-1, 1]`). `snip_edges=true`: frame count is
    /// `1 + (len - frame_length) / frame_shift`, frame `i` starts at
    /// `i * frame_shift` with no edge reflection.
    pub(crate) fn compute(&self, samples: &[f32]) -> Result<KaldiFbankFeatures, KaldiFbankError> {
        let cfg = &self.config;
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(KaldiFbankError::UnsupportedAudio);
        }
        if samples.len() < cfg.frame_length {
            return Err(KaldiFbankError::NoFrames {
                samples: samples.len(),
            });
        }
        let n_frames = 1 + (samples.len() - cfg.frame_length) / cfg.frame_shift;
        let r2c = &self.fft;
        let mut fft_in = r2c.make_input_vec();
        let mut fft_out = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        let mut power = vec![0.0f32; cfg.fft_size / 2 + 1];

        let mut feats = vec![0.0f32; n_frames * cfg.num_mel_bins];
        let mut frame = vec![0.0f32; cfg.frame_length];
        for fr in 0..n_frames {
            let start = fr * cfg.frame_shift;
            for (dst, src) in frame
                .iter_mut()
                .zip(&samples[start..start + cfg.frame_length])
            {
                *dst = *src * cfg.input_scale;
            }
            // remove_dc_offset: subtract the frame mean.
            let mean = frame.iter().sum::<f32>() / cfg.frame_length as f32;
            for v in &mut frame {
                *v -= mean;
            }
            // pre-emphasis with replicate padding: x[i] -= coeff * x[i-1] (i>=1),
            // x[0] -= coeff * x[0].
            for i in (1..cfg.frame_length).rev() {
                frame[i] -= cfg.preemph_coeff * frame[i - 1];
            }
            frame[0] -= cfg.preemph_coeff * frame[0];
            // Window into the zero-padded FFT input.
            fft_in.fill(0.0);
            for (slot, (sample, w)) in fft_in.iter_mut().zip(frame.iter().zip(self.window.iter())) {
                *slot = *sample * *w;
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .expect("kaldi fbank rfft");
            for (bin, value) in fft_out.iter().enumerate() {
                power[bin] = value.re * value.re + value.im * value.im;
            }
            for (bin, filter) in self.filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (j, weight) in filter.weights.iter().enumerate() {
                    energy += weight * power[filter.first_bin + j];
                }
                feats[fr * cfg.num_mel_bins + bin] = energy.max(cfg.log_energy_floor).ln();
            }
        }
        Ok(KaldiFbankFeatures {
            data: feats,
            n_frames,
            n_mels: cfg.num_mel_bins,
        })
    }
}

fn build_window(kind: KaldiWindowKind, frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|n| match kind {
            KaldiWindowKind::Povey => {
                // Povey window: a Hann raised to 0.85 (kaldi/torchaudio default).
                let hann = 0.5
                    - 0.5
                        * (2.0 * std::f32::consts::PI * n as f32 / (frame_length as f32 - 1.0))
                            .cos();
                hann.powf(0.85)
            }
            KaldiWindowKind::Hamming => {
                0.54 - 0.46
                    * (2.0 * std::f32::consts::PI * n as f32 / (frame_length as f32 - 1.0)).cos()
            }
        })
        .collect()
}

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi triangular mel filterbank over `fft_size/2 + 1` power bins:
/// `num_mel_bins` peak-normalized triangles spanning `[mel_low_hz, mel_high_hz]`
/// on the HTK mel scale, gated to bins strictly inside the filter band.
fn build_mel_filters(cfg: &KaldiFbankConfig) -> Vec<MelFilter> {
    let n_fft_bins = cfg.fft_size / 2 + 1;
    let fft_bin_width = cfg.sample_rate_hz as f32 / cfg.fft_size as f32;
    let mel_low = mel_scale(cfg.mel_low_hz);
    let mel_high = mel_scale(cfg.mel_high_hz);
    let mel_delta = (mel_high - mel_low) / (cfg.num_mel_bins as f32 + 1.0);

    let mut filters = Vec::with_capacity(cfg.num_mel_bins);
    for bin in 0..cfg.num_mel_bins {
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

    fn firered_style_config(window: KaldiWindowKind) -> KaldiFbankConfig {
        KaldiFbankConfig {
            sample_rate_hz: 16_000,
            frame_length: 400,
            frame_shift: 160,
            fft_size: 512,
            num_mel_bins: 80,
            mel_low_hz: 20.0,
            mel_high_hz: 8_000.0,
            preemph_coeff: 0.97,
            input_scale: 32_768.0,
            log_energy_floor: 1.192_092_9e-7,
            window,
        }
    }

    #[test]
    fn produces_80_bin_snip_edges_fbank_povey() {
        let samples = vec![0.01f32; 38_080];
        let features = KaldiFbankFrontend::new(firered_style_config(KaldiWindowKind::Povey))
            .compute(&samples)
            .expect("fbank");
        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 236);
        assert_eq!(features.data.len(), 236 * 80);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn produces_80_bin_snip_edges_fbank_hamming() {
        let samples = vec![0.01f32; 38_080];
        let features = KaldiFbankFrontend::new(firered_style_config(KaldiWindowKind::Hamming))
            .compute(&samples)
            .expect("fbank");
        assert_eq!(features.n_mels, 80);
        assert_eq!(features.n_frames, 236);
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_too_short_audio() {
        let samples = vec![0.0f32; 200];
        assert!(matches!(
            KaldiFbankFrontend::new(firered_style_config(KaldiWindowKind::Povey)).compute(&samples),
            Err(KaldiFbankError::NoFrames { samples: 200 })
        ));
    }

    #[test]
    fn rejects_non_finite_audio() {
        let samples = vec![f32::NAN; 500];
        assert!(matches!(
            KaldiFbankFrontend::new(firered_style_config(KaldiWindowKind::Povey)).compute(&samples),
            Err(KaldiFbankError::UnsupportedAudio)
        ));
    }
}
