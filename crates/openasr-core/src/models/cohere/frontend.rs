use realfft::RealFftPlanner;
use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;

use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tensor_names::{
    FE_MEL_FB as COHERE_FILTER_TENSOR_NAME, FE_WINDOW as COHERE_WINDOW_TENSOR_NAME,
};

const COHERE_PREEMPHASIS: f32 = 0.97;
const COHERE_LOG_EPSILON: f32 = 1.0 / (1_u32 << 24) as f32;
const COHERE_NORM_EPSILON: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereTranscribeFrontendPlan {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub window: Vec<f32>,
    pub mel_filters: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CohereTranscribeMelFeatures {
    pub n_frames: usize,
    pub n_mels: usize,
    // Layout: time-major ([frame][mel]) contiguous f32.
    pub data: Vec<f32>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereTranscribeFrontendError {
    #[error("cohere-transcribe frontend metadata is unsupported: {reason}")]
    UnsupportedMetadata { reason: String },
    #[error("cohere-transcribe frontend could not read GGUF tensor '{tensor_name}': {source}")]
    TensorRead {
        tensor_name: &'static str,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error("cohere-transcribe frontend requires non-empty finite mono 16 kHz audio")]
    InvalidAudioSamples,
    #[error("cohere-transcribe frontend frame calculation overflowed")]
    FrameCountOverflow,
    #[error("cohere-transcribe frontend FFT failed: {reason}")]
    FftFailed { reason: String },
}

pub(crate) fn load_cohere_transcribe_frontend_plan_from_reader(
    reader: &GgufTensorDataReader,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<CohereTranscribeFrontendPlan, CohereTranscribeFrontendError> {
    if metadata.hop_length > metadata.win_length || metadata.win_length > metadata.n_fft {
        return Err(CohereTranscribeFrontendError::UnsupportedMetadata {
            reason: format!(
                "expected hop_length <= win_length <= n_fft, got {} <= {} <= {}",
                metadata.hop_length, metadata.win_length, metadata.n_fft
            ),
        });
    }
    let expected_fft_bins = metadata
        .n_fft
        .checked_div(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(CohereTranscribeFrontendError::FrameCountOverflow)?;
    let mel_filters = reader
        .host_tensor_f32_copy_dequantized_by_name(
            COHERE_FILTER_TENSOR_NAME,
            &[expected_fft_bins as u64, metadata.n_mels as u64],
        )
        .map_err(|source| CohereTranscribeFrontendError::TensorRead {
            tensor_name: COHERE_FILTER_TENSOR_NAME,
            source,
        })?;
    let window = reader
        .host_tensor_f32_copy_dequantized_by_name(
            COHERE_WINDOW_TENSOR_NAME,
            &[metadata.win_length as u64],
        )
        .map_err(|source| CohereTranscribeFrontendError::TensorRead {
            tensor_name: COHERE_WINDOW_TENSOR_NAME,
            source,
        })?;

    Ok(CohereTranscribeFrontendPlan {
        n_fft: metadata.n_fft,
        hop_length: metadata.hop_length,
        win_length: metadata.win_length,
        sample_rate_hz: metadata.sample_rate_hz,
        n_mels: metadata.n_mels,
        window,
        mel_filters,
    })
}

pub(crate) fn cohere_transcribe_features_from_prepared_audio(
    prepared_audio: &GgmlAsrPreparedAudio,
    plan: &CohereTranscribeFrontendPlan,
) -> Result<CohereTranscribeMelFeatures, CohereTranscribeFrontendError> {
    if prepared_audio.sample_rate_hz != plan.sample_rate_hz
        || prepared_audio.channels != 1
        || prepared_audio.samples_f32.is_empty()
        || prepared_audio
            .samples_f32
            .iter()
            .any(|sample| !sample.is_finite())
    {
        return Err(CohereTranscribeFrontendError::InvalidAudioSamples);
    }
    cohere_transcribe_features_from_samples(&prepared_audio.samples_f32, plan)
}

fn cohere_transcribe_features_from_samples(
    samples: &[f32],
    plan: &CohereTranscribeFrontendPlan,
) -> Result<CohereTranscribeMelFeatures, CohereTranscribeFrontendError> {
    if samples.is_empty() {
        return Err(CohereTranscribeFrontendError::InvalidAudioSamples);
    }

    let emphasized = apply_preemphasis(samples);
    let padded = center_pad_samples(&emphasized, plan.n_fft / 2)?;
    let all_frames = padded
        .len()
        .checked_sub(plan.n_fft)
        .ok_or(CohereTranscribeFrontendError::FrameCountOverflow)?
        .checked_div(plan.hop_length)
        .and_then(|value| value.checked_add(1))
        .ok_or(CohereTranscribeFrontendError::FrameCountOverflow)?;
    let frame_count = all_frames
        .checked_sub(1)
        .ok_or(CohereTranscribeFrontendError::FrameCountOverflow)?;
    if frame_count == 0 {
        return Err(CohereTranscribeFrontendError::InvalidAudioSamples);
    }

    let fft_bins = plan.n_fft / 2 + 1;
    let padded_window = zero_pad_window(plan);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(plan.n_fft);
    let mut fft_input = r2c.make_input_vec();
    let mut fft_output = r2c.make_output_vec();
    let mut fft_scratch = r2c.make_scratch_vec();
    let mut power_spectrogram = vec![0.0_f32; frame_count * fft_bins];

    for frame_idx in 0..frame_count {
        let start = frame_idx * plan.hop_length;
        let frame = &padded[start..start + plan.n_fft];
        for (index, value) in fft_input.iter_mut().enumerate() {
            *value = frame[index] * padded_window[index];
        }
        r2c.process_with_scratch(&mut fft_input, &mut fft_output, &mut fft_scratch)
            .map_err(|error| CohereTranscribeFrontendError::FftFailed {
                reason: error.to_string(),
            })?;
        for (bin, complex) in fft_output.iter().enumerate() {
            power_spectrogram[frame_idx * fft_bins + bin] = complex.norm_sqr();
        }
    }

    let mut mel_values = project_power_spectrogram_to_time_major_mels(
        &power_spectrogram,
        frame_count,
        fft_bins,
        plan,
    );
    log_and_normalize_in_place(&mut mel_values, frame_count, plan.n_mels);
    Ok(CohereTranscribeMelFeatures {
        n_frames: frame_count,
        n_mels: plan.n_mels,
        data: mel_values,
    })
}

fn apply_preemphasis(samples: &[f32]) -> Vec<f32> {
    let mut emphasized = Vec::with_capacity(samples.len());
    if let Some(first) = samples.first().copied() {
        emphasized.push(first);
    }
    for pair in samples.windows(2) {
        emphasized.push(pair[1] - COHERE_PREEMPHASIS * pair[0]);
    }
    emphasized
}

fn center_pad_samples(
    samples: &[f32],
    pad_each_side: usize,
) -> Result<Vec<f32>, CohereTranscribeFrontendError> {
    let total_len = samples
        .len()
        .checked_add(pad_each_side)
        .and_then(|value| value.checked_add(pad_each_side))
        .ok_or(CohereTranscribeFrontendError::FrameCountOverflow)?;
    let mut padded = vec![0.0_f32; total_len];
    let start = pad_each_side;
    let end = start + samples.len();
    padded[start..end].copy_from_slice(samples);
    Ok(padded)
}

fn zero_pad_window(plan: &CohereTranscribeFrontendPlan) -> Vec<f32> {
    let mut padded = vec![0.0_f32; plan.n_fft];
    let left_pad = (plan.n_fft - plan.win_length) / 2;
    padded[left_pad..left_pad + plan.win_length].copy_from_slice(&plan.window);
    padded
}

fn project_power_spectrogram_to_time_major_mels(
    power_spectrogram: &[f32],
    frame_count: usize,
    fft_bins: usize,
    plan: &CohereTranscribeFrontendPlan,
) -> Vec<f32> {
    let mut mel_values = vec![0.0_f32; frame_count * plan.n_mels];
    for frame_idx in 0..frame_count {
        let power_row = &power_spectrogram[frame_idx * fft_bins..(frame_idx + 1) * fft_bins];
        for mel_idx in 0..plan.n_mels {
            let mut sum = 0.0f32;
            for (freq_idx, power) in power_row.iter().enumerate() {
                // GGUF stores the raw tensor dims as [n_freqs, n_mels], but the
                // flattened payload is laid out as mel-major rows. Match the
                // reference implementation's fb[m * n_freqs + k] indexing.
                let filter_idx = mel_idx * fft_bins + freq_idx;
                sum += *power * plan.mel_filters[filter_idx];
            }
            mel_values[frame_idx * plan.n_mels + mel_idx] = sum;
        }
    }
    mel_values
}

fn log_and_normalize_in_place(values: &mut [f32], frame_count: usize, n_mels: usize) {
    for value in values.iter_mut() {
        *value = (*value + COHERE_LOG_EPSILON).ln();
    }
    for mel_idx in 0..n_mels {
        let mut sum = 0.0f64;
        for frame_idx in 0..frame_count {
            sum += f64::from(values[frame_idx * n_mels + mel_idx]);
        }
        let mean = sum / frame_count as f64;
        let mut var_sum = 0.0f64;
        for frame_idx in 0..frame_count {
            let diff = f64::from(values[frame_idx * n_mels + mel_idx]) - mean;
            var_sum += diff * diff;
        }
        let variance = if frame_count > 1 {
            var_sum / (frame_count - 1) as f64
        } else {
            var_sum
        };
        let mut scale = variance.sqrt();
        if scale.is_nan() {
            scale = 0.0;
        }
        scale += f64::from(COHERE_NORM_EPSILON);
        for frame_idx in 0..frame_count {
            let index = frame_idx * n_mels + mel_idx;
            values[index] = ((f64::from(values[index]) - mean) / scale) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_plan() -> CohereTranscribeFrontendPlan {
        CohereTranscribeFrontendPlan {
            n_fft: 8,
            hop_length: 4,
            win_length: 4,
            sample_rate_hz: 16_000,
            n_mels: 2,
            window: vec![1.0, 1.0, 1.0, 1.0],
            mel_filters: vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.5, 0.5, //
                0.0, 0.0, //
                0.0, 0.0, //
            ],
        }
    }

    #[test]
    fn frontend_rejects_invalid_audio() {
        let plan = test_plan();
        let error = cohere_transcribe_features_from_prepared_audio(
            &GgmlAsrPreparedAudio {
                sample_rate_hz: 8_000,
                channels: 1,
                samples_f32: vec![0.0, 1.0],
            },
            &plan,
        )
        .expect_err("invalid audio must fail");
        assert!(matches!(
            error,
            CohereTranscribeFrontendError::InvalidAudioSamples
        ));
    }

    #[test]
    fn frontend_emits_finite_time_major_features() {
        let plan = test_plan();
        let samples = vec![
            0.0, 0.2, 0.6, -0.3, -0.2, 0.4, 0.8, 0.1, -0.5, -0.1, 0.3, 0.7, 0.2, -0.4, -0.2, 0.5,
        ];
        let features =
            cohere_transcribe_features_from_samples(&samples, &plan).expect("features must build");
        assert_eq!(features.n_mels, 2);
        assert!(features.n_frames >= 2);
        assert_eq!(features.data.len(), features.n_frames * features.n_mels);
        assert!(features.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn frontend_applies_per_feature_zero_mean_normalization() {
        let plan = test_plan();
        let samples = vec![
            0.0, 0.3, 0.7, -0.2, -0.1, 0.5, 0.9, 0.2, -0.4, -0.2, 0.2, 0.6, 0.1, -0.3, -0.1, 0.4,
        ];
        let features =
            cohere_transcribe_features_from_samples(&samples, &plan).expect("features must build");
        for mel_idx in 0..features.n_mels {
            let mean = (0..features.n_frames)
                .map(|frame_idx| features.data[frame_idx * features.n_mels + mel_idx])
                .sum::<f32>()
                / features.n_frames as f32;
            assert!(mean.abs() < 1e-4, "mel {mel_idx} mean={mean}");
        }
    }

    #[test]
    fn frontend_projects_mel_filters_using_mel_major_payload_layout() {
        let plan = test_plan();
        let power = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mel_values = project_power_spectrogram_to_time_major_mels(&power, 1, 5, &plan);
        assert_eq!(mel_values, vec![7.5, 0.5]);
    }
}
