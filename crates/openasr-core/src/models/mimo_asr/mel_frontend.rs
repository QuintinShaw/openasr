//! MiMo-Audio-Tokenizer mel front-end: `torchaudio.transforms.MelSpectrogram`
//! shape (24kHz / n_fft=960 / hop=240 / win=960 / n_mels=128, htk scale,
//! `norm=None`, `power=1` magnitude spectrogram, natural-log with a
//! `1e-7` clip floor, `center=True` reflect padding) -- see
//! `GGUF_MANIFEST.md`'s `mimo.mel.*` keys and
//! `tooling/mimo-asr/convert_mimo_asr.py::mel_filters_and_window`, which bakes
//! the exact filterbank + Hann window this module reads back rather than
//! recomputing (so filter construction can never drift between the converter
//! and the runtime).

use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::audio_frontend::{PadMode, StftFramer};

use super::runtime_contract::MimoMelMetadata;
use super::tensor_names::{AUDIOTOK_MEL_FILTERS, AUDIOTOK_MEL_WINDOW};

#[derive(Debug, Error)]
pub(crate) enum MimoMelFrontendError {
    #[error("mimo-asr mel frontend could not read GGUF tensor '{tensor_name}': {source}")]
    TensorRead {
        tensor_name: &'static str,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("mimo-asr mel frontend requires non-empty finite mono samples")]
    InvalidAudioSamples,
    #[error("mimo-asr mel frontend produced zero frames")]
    NoFrames,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MimoMelFrontendPlan {
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate_hz: usize,
    pub n_mels: usize,
    pub log_clip: f32,
    pub window: Vec<f32>,
    /// Freq-major `[n_fft/2+1][n_mels]` contiguous (GGUF ne=[n_mels, n_freqs]
    /// -> mels vary fastest), matching `qwen::frontend`'s baked-filter layout.
    pub mel_filters: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MimoMelFeatures {
    pub n_mels: usize,
    pub n_frames: usize,
    /// Mel-major (`[mel][frame]`) contiguous f32 -- ggml `conv_1d` input
    /// convention (time innermost per channel row).
    pub data: Vec<f32>,
}

pub(crate) fn load_mimo_mel_frontend_plan_from_reader(
    reader: &GgufTensorDataReader,
    metadata: &MimoMelMetadata,
) -> Result<MimoMelFrontendPlan, MimoMelFrontendError> {
    let fft_bins = metadata.n_fft / 2 + 1;
    let mel_filters = reader
        .host_tensor_f32_copy_by_name(
            AUDIOTOK_MEL_FILTERS,
            &[metadata.n_mels as u64, fft_bins as u64],
        )
        .map_err(|source| MimoMelFrontendError::TensorRead {
            tensor_name: AUDIOTOK_MEL_FILTERS,
            source,
        })?;
    let window = reader
        .host_tensor_f32_copy_by_name(AUDIOTOK_MEL_WINDOW, &[metadata.win_length as u64])
        .map_err(|source| MimoMelFrontendError::TensorRead {
            tensor_name: AUDIOTOK_MEL_WINDOW,
            source,
        })?;
    Ok(MimoMelFrontendPlan {
        n_fft: metadata.n_fft,
        hop_length: metadata.hop_length,
        sample_rate_hz: metadata.sample_rate_hz,
        n_mels: metadata.n_mels,
        log_clip: metadata.log_clip,
        window,
        mel_filters,
    })
}

pub(crate) fn mimo_mel_features_from_samples(
    samples: &[f32],
    plan: &MimoMelFrontendPlan,
) -> Result<MimoMelFeatures, MimoMelFrontendError> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(MimoMelFrontendError::InvalidAudioSamples);
    }
    let framer = StftFramer::new(
        plan.n_fft,
        plan.n_fft,
        plan.hop_length,
        PadMode::ReflectCenter,
        plan.window.clone(),
    );
    // `StftFramer::power_spectrogram` returns |X|^2 (see its doc); MiMo's
    // `torchaudio.MelSpectrogram(power=1.0)` wants the magnitude spectrogram,
    // so take the elementwise sqrt before projecting through the mel filters.
    let spectrogram = framer
        .power_spectrogram(samples)
        .map_err(|_| MimoMelFrontendError::NoFrames)?;
    let n_frames = spectrogram.n_frames;
    if n_frames == 0 {
        return Err(MimoMelFrontendError::NoFrames);
    }
    let fft_bins = framer.n_fft_bins();
    let magnitude: Vec<f32> = spectrogram
        .data
        .iter()
        .map(|power| power.max(0.0).sqrt())
        .collect();

    let n_mels = plan.n_mels;
    let mut mel_values = vec![0.0_f32; n_mels * n_frames];
    for frame_idx in 0..n_frames {
        let row = &magnitude[frame_idx * fft_bins..(frame_idx + 1) * fft_bins];
        for mel_idx in 0..n_mels {
            let mut sum = 0.0_f64;
            for (freq_idx, mag) in row.iter().enumerate() {
                // GGUF ne=[n_mels, n_freqs]: contiguous memory is freq-major
                // with mels innermost (same convention as qwen::frontend).
                let filter_idx = freq_idx * n_mels + mel_idx;
                sum += f64::from(*mag) * f64::from(plan.mel_filters[filter_idx]);
            }
            let clipped = (sum as f32).max(plan.log_clip);
            mel_values[mel_idx * n_frames + frame_idx] = clipped.ln();
        }
    }
    Ok(MimoMelFeatures {
        n_mels,
        n_frames,
        data: mel_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_features_emit_expected_frame_count_and_ln_clip_floor() {
        let n_fft = 8;
        let hop = 2;
        let n_mels = 2;
        let fft_bins = n_fft / 2 + 1;
        let plan = MimoMelFrontendPlan {
            n_fft,
            hop_length: hop,
            sample_rate_hz: 24_000,
            n_mels,
            log_clip: 1e-7,
            window: vec![1.0; n_fft],
            mel_filters: vec![0.0; n_mels * fft_bins],
        };
        // All-silence input with zero filters -> every mel value clips to
        // log_clip and ln(1e-7) is the floor everywhere.
        let samples = vec![0.0_f32; 32];
        let features = mimo_mel_features_from_samples(&samples, &plan).expect("mel features");
        assert_eq!(features.n_mels, 2);
        assert!(features.n_frames > 0);
        let expected_floor = plan.log_clip.ln();
        assert!(
            features
                .data
                .iter()
                .all(|value| (*value - expected_floor).abs() < 1e-6)
        );
    }

    #[test]
    fn rejects_empty_samples() {
        let plan = MimoMelFrontendPlan {
            n_fft: 8,
            hop_length: 2,
            sample_rate_hz: 24_000,
            n_mels: 2,
            log_clip: 1e-7,
            window: vec![1.0; 8],
            mel_filters: vec![0.0; 2 * 5],
        };
        let error = mimo_mel_features_from_samples(&[], &plan).expect_err("must fail");
        assert!(matches!(error, MimoMelFrontendError::InvalidAudioSamples));
    }
}
