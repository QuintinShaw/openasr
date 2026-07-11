use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::audio_frontend::{PadMode, StftFramer, center_embed_window};
use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::{AUDIO_MEL_FILTERS, AUDIO_MEL_WINDOW};

const LOG_SPEC_FLOOR: f32 = 1e-10;
const LOG_SPEC_DYNAMIC_RANGE: f32 = 8.0;
const LOG_SPEC_SHIFT: f32 = 4.0;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrMelFrontendPlan {
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub window: Vec<f32>,
    pub mel_filters: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrMelFeatures {
    pub n_mels: usize,
    pub n_frames: usize,
    // Layout: mel-major ([mel][frame]) contiguous f32.
    pub data: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrMelFrontendError {
    #[error("qwen3-asr mel frontend metadata is unsupported: {reason}")]
    UnsupportedMetadata { reason: String },
    #[error("qwen3-asr mel frontend could not read GGUF tensor '{tensor_name}': {source}")]
    TensorRead {
        tensor_name: &'static str,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("qwen3-asr mel frontend requires non-empty finite mono samples")]
    InvalidAudioSamples,
    #[error("qwen3-asr mel frontend frame calculation overflowed")]
    FrameCountOverflow,
    #[error("qwen3-asr mel frontend mel projection failed: {reason}")]
    MelProjectionFailed { reason: String },
}

pub(crate) fn load_qwen3_mel_frontend_plan_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Qwen3AsrMelFrontendPlan, Qwen3AsrMelFrontendError> {
    if metadata.win_length != metadata.n_fft {
        return Err(Qwen3AsrMelFrontendError::UnsupportedMetadata {
            reason: format!(
                "win_length={} must equal n_fft={} for current frontend",
                metadata.win_length, metadata.n_fft
            ),
        });
    }
    if metadata.hop_length > metadata.n_fft {
        return Err(Qwen3AsrMelFrontendError::UnsupportedMetadata {
            reason: format!(
                "hop_length={} must be <= n_fft={}",
                metadata.hop_length, metadata.n_fft
            ),
        });
    }
    let expected_fft_bins = metadata
        .n_fft
        .checked_div(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?;
    let mel_filters = reader
        .host_tensor_f32_copy_by_name(
            AUDIO_MEL_FILTERS,
            &[metadata.n_mels as u64, expected_fft_bins as u64],
        )
        .map_err(|source| Qwen3AsrMelFrontendError::TensorRead {
            tensor_name: AUDIO_MEL_FILTERS,
            source,
        })?;
    let window = reader
        .host_tensor_f32_copy_by_name(AUDIO_MEL_WINDOW, &[metadata.win_length as u64])
        .map_err(|source| Qwen3AsrMelFrontendError::TensorRead {
            tensor_name: AUDIO_MEL_WINDOW,
            source,
        })?;

    Ok(Qwen3AsrMelFrontendPlan {
        n_fft: metadata.n_fft,
        hop_length: metadata.hop_length,
        sample_rate_hz: metadata.sample_rate_hz,
        n_mels: metadata.n_mels,
        window,
        mel_filters,
    })
}

pub(crate) fn qwen3_mel_features_from_prepared_audio(
    prepared_audio: &GgmlAsrPreparedAudio,
    plan: &Qwen3AsrMelFrontendPlan,
) -> Result<Qwen3AsrMelFeatures, Qwen3AsrMelFrontendError> {
    if prepared_audio.sample_rate_hz != plan.sample_rate_hz
        || prepared_audio.channels != 1
        || prepared_audio.samples_f32.is_empty()
        || prepared_audio
            .samples_f32
            .iter()
            .any(|sample| !sample.is_finite())
    {
        return Err(Qwen3AsrMelFrontendError::InvalidAudioSamples);
    }
    qwen3_mel_features_from_samples(&prepared_audio.samples_f32, plan)
}

fn qwen3_mel_features_from_samples(
    samples: &[f32],
    plan: &Qwen3AsrMelFrontendPlan,
) -> Result<Qwen3AsrMelFeatures, Qwen3AsrMelFrontendError> {
    if samples.is_empty() {
        return Err(Qwen3AsrMelFrontendError::InvalidAudioSamples);
    }

    // `win_length == n_fft` is enforced by `load_qwen3_mel_frontend_plan_from_reader`,
    // so `center_embed_window` is a no-op copy here (offset 0); kept for
    // parity with `cohere`, whose `win_length` can be shorter than `n_fft`.
    let framer = StftFramer::new(
        plan.n_fft,
        plan.n_fft,
        plan.hop_length,
        PadMode::ZeroCenter,
        center_embed_window(&plan.window, plan.n_fft),
    );
    let spectrogram = framer
        .power_spectrogram(samples)
        .map_err(|_| Qwen3AsrMelFrontendError::FrameCountOverflow)?;
    // `torch.stft`-shaped framing over a signal zero-padded to `n_fft/2` on
    // each side yields one more frame than the audio's "real" frame count;
    // the pre-refactor frontend always dropped that trailing frame.
    let frame_count = spectrogram
        .n_frames
        .checked_sub(1)
        .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?;
    if frame_count == 0 {
        return Err(Qwen3AsrMelFrontendError::InvalidAudioSamples);
    }
    let fft_bins = framer.n_fft_bins();
    let power_spectrogram = &spectrogram.data[..frame_count * fft_bins];

    let mut mel_values = project_power_spectrogram_to_mels_time(
        power_spectrogram,
        frame_count,
        fft_bins,
        &plan.mel_filters,
        plan.n_mels,
    )?;
    normalize_log_mel_in_place(&mut mel_values);
    Ok(Qwen3AsrMelFeatures {
        n_mels: plan.n_mels,
        n_frames: frame_count,
        data: mel_values,
    })
}

fn project_power_spectrogram_to_mels_time(
    power_spectrogram: &[f32],
    frame_count: usize,
    fft_bins: usize,
    mel_filters: &[f32],
    n_mels: usize,
) -> Result<Vec<f32>, Qwen3AsrMelFrontendError> {
    let expected_power_len = frame_count
        .checked_mul(fft_bins)
        .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?;
    let expected_filter_len = fft_bins
        .checked_mul(n_mels)
        .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?;
    if power_spectrogram.len() != expected_power_len || mel_filters.len() != expected_filter_len {
        return Err(Qwen3AsrMelFrontendError::MelProjectionFailed {
            reason: format!(
                "power len {} expected {expected_power_len}, filter len {} expected {expected_filter_len}",
                power_spectrogram.len(),
                mel_filters.len()
            ),
        });
    }

    let mut mel_values = vec![
        0.0_f32;
        n_mels
            .checked_mul(frame_count)
            .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?
    ];
    for frame_idx in 0..frame_count {
        let power_row = &power_spectrogram[frame_idx * fft_bins..(frame_idx + 1) * fft_bins];
        for mel_idx in 0..n_mels {
            let mut sum = 0.0_f64;
            for (freq_idx, power) in power_row.iter().enumerate() {
                // GGUF stores this tensor as ggml ne=[n_mels, n_freqs], so
                // contiguous memory is freq-major with mels as the inner axis.
                let filter_idx = freq_idx
                    .checked_mul(n_mels)
                    .and_then(|value| value.checked_add(mel_idx))
                    .ok_or(Qwen3AsrMelFrontendError::FrameCountOverflow)?;
                sum += f64::from(*power) * f64::from(mel_filters[filter_idx]);
            }
            mel_values[mel_idx * frame_count + frame_idx] = sum as f32;
        }
    }
    Ok(mel_values)
}

fn normalize_log_mel_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = value.max(LOG_SPEC_FLOOR).log10();
    }
    let peak = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(f32::NEG_INFINITY);
    let floor = peak - LOG_SPEC_DYNAMIC_RANGE;
    for value in values.iter_mut() {
        *value = value.max(floor);
        *value = (*value + LOG_SPEC_SHIFT) / LOG_SPEC_SHIFT;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;

    #[test]
    fn mel_frontend_emits_expected_frame_count_for_center_padded_stft() {
        let n_fft = 400;
        let hop_length = 160;
        let n_mels = 2;
        let fft_bins = n_fft / 2 + 1;
        let plan = Qwen3AsrMelFrontendPlan {
            n_fft,
            hop_length,
            sample_rate_hz: 16_000,
            n_mels,
            window: vec![1.0; n_fft],
            mel_filters: vec![0.0; n_mels * fft_bins],
        };
        let samples = vec![0.0_f32; 3_200];
        let prepared = GgmlAsrPreparedAudio::mono_16khz(samples);
        let features =
            qwen3_mel_features_from_prepared_audio(&prepared, &plan).expect("mel features");
        assert_eq!(features.n_frames, 20);
        assert_eq!(features.data.len(), n_mels * 20);
    }

    #[test]
    fn mel_projection_interprets_ggml_filterbank_as_freqs_mels() {
        let power = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ];
        let filterbank = vec![
            10.0_f32, 100.0, //
            20.0, 200.0, //
            30.0, 300.0,
        ];

        let mel = project_power_spectrogram_to_mels_time(&power, 2, 3, &filterbank, 2)
            .expect("mel projection");

        assert_eq!(mel, vec![140.0, 320.0, 1400.0, 3200.0]);
    }
}
