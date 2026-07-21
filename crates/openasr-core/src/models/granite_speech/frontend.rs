//! Granite Speech mel front-end: `GraniteSpeechFeatureExtractor`
//! (`transformers.models.granite_speech.feature_extraction_granite_speech`),
//! a `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512,
//! win_length=400, hop_length=160, n_mels=80)` (torchaudio defaults:
//! `mel_scale="htk"`, `norm=None` i.e. NOT area-normalized, `power=2.0`,
//! `center=True`, `pad_mode="reflect"`, periodic Hann window) followed by
//! `log10 -> per-utterance floor-clip -> /4+1 -> drop-odd-frame -> 2x
//! frame-stack`.
//!
//! Built entirely on the shared model-agnostic DSP primitives in
//! `models::audio_frontend` (`StftFramer`, `PadMode::ReflectCenter`,
//! `hann_window_centered`, `mel::{MelScale::Htk, filterbank}`) -- per
//! `AGENTS.md`'s "keep infrastructure model-agnostic" rule and the 0.1.13
//! mel/fbank single-source consolidation, this module parameterizes that
//! shared engine rather than hand-rolling a second FFT/filterbank. Whisper's
//! frontend (`n_fft=400`, Slaney scale) is the nearest sibling but genuinely
//! different on both axes Granite differs from it (`n_fft=512` != `win=400`
//! here, so the window needs centering slack whisper's `win==n_fft` case
//! does not; `Htk`/no-norm scale, not `Slaney`) -- both are config
//! differences on the same shared primitives, not a second DSP stack.
//!
//! Numeric parity against the HF `torchaudio` reference (`input_features`
//! golden dumped from the real feature extractor) lives in `parity`'s
//! `granite_speech_frontend_parity`.

#![allow(dead_code)]

use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale, filterbank};
use crate::models::audio_frontend::{PadMode, StftFramer, hann_window_centered};

pub(crate) const SAMPLE_RATE_HZ: f32 = 16000.0;
pub(crate) const N_FFT: usize = 512;
pub(crate) const WIN_LENGTH: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;
pub(crate) const N_MELS: usize = 80;

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechFrontendError {
    #[error("granite-speech frontend stft failed: {0}")]
    Stft(#[from] crate::models::audio_frontend::StftError),
}

pub(crate) struct GraniteSpeechMelFrontend {
    framer: StftFramer,
    filters: Vec<f32>,
}

impl GraniteSpeechMelFrontend {
    pub(crate) fn new() -> Self {
        let window = hann_window_centered(WIN_LENGTH, N_FFT);
        let framer = StftFramer::new(
            N_FFT,
            WIN_LENGTH,
            HOP_LENGTH,
            PadMode::ReflectCenter,
            window,
        );
        let filters = filterbank(FilterbankConfig {
            scale: MelScale::Htk,
            sample_rate_hz: SAMPLE_RATE_HZ,
            n_fft: N_FFT,
            n_mels: N_MELS,
            fmin: 0.0,
            fmax: SAMPLE_RATE_HZ / 2.0,
            mel_point_order: MelPointOrder::SpanTimesIndexFirst,
        });
        Self { framer, filters }
    }

    /// `samples` is mono f32 PCM in `[-1, 1]` at 16 kHz. Returns row-major
    /// `[frames, 160]` (80 log-mel bins x2 frame-stacking), matching
    /// `GraniteSpeechFeatureExtractor.__call__`'s `input_features` exactly
    /// (see the module doc for the op sequence).
    pub(crate) fn extract(
        &self,
        samples: &[f32],
    ) -> Result<(Vec<f32>, usize), GraniteSpeechFrontendError> {
        let spectrogram = self.framer.power_spectrogram(samples)?;
        let n_frames = spectrogram.n_frames;
        let fft_bins = spectrogram.n_fft_bins;

        // mel[frame, m] = sum_bin power[frame, bin] * filters[m, bin]
        let mut logmel = vec![0.0f32; n_frames * N_MELS];
        for frame in 0..n_frames {
            let power_row = &spectrogram.data[frame * fft_bins..(frame + 1) * fft_bins];
            for m in 0..N_MELS {
                let filter_row = &self.filters[m * fft_bins..(m + 1) * fft_bins];
                let mut energy = 0.0f32;
                for (p, f) in power_row.iter().zip(filter_row.iter()) {
                    energy += p * f;
                }
                logmel[frame * N_MELS + m] = energy.max(1.0e-10).log10();
            }
        }

        // Per-utterance floor clip at (max - 8), then /4 + 1 (matches
        // `torch.maximum(logmel, mx - 8.0).div_(4).add_(1)`).
        let max_val = logmel.iter().copied().fold(f32::MIN, f32::max);
        let floor = max_val - 8.0;
        for value in logmel.iter_mut() {
            *value = (value.max(floor)) / 4.0 + 1.0;
        }

        // Drop the last frame if the frame count is odd, then stack pairs of
        // consecutive 80-dim frames into 160-dim (concatenation, not
        // averaging), halving the frame rate.
        let usable_frames = n_frames - (n_frames % 2);
        let stacked_frames = usable_frames / 2;
        let mut stacked = vec![0.0f32; stacked_frames * 2 * N_MELS];
        stacked[..stacked_frames * 2 * N_MELS]
            .copy_from_slice(&logmel[..stacked_frames * 2 * N_MELS]);

        Ok((stacked, stacked_frames))
    }
}

impl Default for GraniteSpeechMelFrontend {
    fn default() -> Self {
        Self::new()
    }
}
