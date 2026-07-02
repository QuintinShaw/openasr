use crate::NativeAsrError;
use crate::tensor::TensorOwnedF32;

use super::mel::{
    WhisperMelFeatures, WhisperMelFrontendPlan, build_whisper_mel_frontend_plan_v0,
    whisper_mel_features_from_samples_with_plan_v0,
};

pub type WhisperFrontendPlan = WhisperMelFrontendPlan;

pub fn whisper_log_mel_spectrogram_16khz_mono_v0(
    samples: &[f32],
    n_mels: usize,
    target_frames: usize,
) -> Result<TensorOwnedF32, NativeAsrError> {
    let plan = build_whisper_frontend_plan_v0(n_mels, target_frames)?;
    whisper_log_mel_spectrogram_with_plan_v0(samples, &plan)
}

pub fn build_whisper_frontend_plan_v0(
    n_mels: usize,
    target_frames: usize,
) -> Result<WhisperFrontendPlan, NativeAsrError> {
    build_whisper_mel_frontend_plan_v0(n_mels, target_frames)
}

pub fn whisper_log_mel_spectrogram_with_plan_v0(
    samples: &[f32],
    plan: &WhisperFrontendPlan,
) -> Result<TensorOwnedF32, NativeAsrError> {
    let mel = whisper_mel_features_from_samples_with_plan_v0(samples, plan)?;
    whisper_mel_features_to_tensor_v0(&mel)
}

pub fn whisper_mel_features_to_tensor_v0(
    mel: &WhisperMelFeatures,
) -> Result<TensorOwnedF32, NativeAsrError> {
    mel.to_tensor_owned_v0()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_rejects_empty_input() {
        let error = whisper_log_mel_spectrogram_16khz_mono_v0(&[], 80, 3000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one audio sample"), "{error}");
    }

    #[test]
    fn frontend_rejects_non_finite_input() {
        let error = whisper_log_mel_spectrogram_16khz_mono_v0(&[0.0, f32::NAN], 80, 3000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-finite"), "{error}");
    }

    #[test]
    fn frontend_produces_expected_shape_for_one_second_waveform() {
        let samples = (0..16_000)
            .map(|index| {
                let angle = 2.0 * std::f32::consts::PI * 440.0 * index as f32 / 16_000.0;
                angle.sin() * 0.25
            })
            .collect::<Vec<_>>();
        let mel = whisper_log_mel_spectrogram_16khz_mono_v0(&samples, 80, 3000).unwrap();
        assert_eq!(mel.layout().shape(), &[1, 80, 3000]);
        assert!(mel.data().iter().all(|value| value.is_finite()));
        assert!(mel.data().iter().any(|value| *value > 0.0));
    }
}
