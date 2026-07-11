use crate::NativeAsrError;
use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale};
use crate::models::audio_frontend::{PadMode, StftFramer, hann_window_periodic_scale_first};
use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;
use crate::tensor::{TensorOwnedF32, TensorViewF32, linear_f32};

pub const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
pub const WHISPER_CHANNELS: u16 = 1;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
const WHISPER_MEL_FMIN: f32 = 0.0;
const WHISPER_MEL_FMAX: f32 = 8_000.0;
const WHISPER_LOG_SPEC_FLOOR: f32 = 1e-10;
const WHISPER_LOG_SPEC_DYNAMIC_RANGE: f32 = 8.0;
const WHISPER_LOG_SPEC_SHIFT: f32 = 4.0;

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperMelFrontendPlan {
    pub n_mels: usize,
    pub target_frames: usize,
    /// Periodic Hann window, `n_fft` (== `WHISPER_N_FFT`) long -- whisper
    /// never needs the `win_length < n_fft` centering slack
    /// [`crate::models::audio_frontend::hann_window_centered`] handles for
    /// other families.
    pub window: Vec<f32>,
    pub mel_filters: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperMelFeatures {
    pub n_mels: usize,
    pub n_frames: usize,
    // Layout: mel-major ([mel][frame]) contiguous f32.
    pub data: Vec<f32>,
}

impl WhisperMelFeatures {
    pub fn to_tensor_owned_v0(&self) -> Result<TensorOwnedF32, NativeAsrError> {
        let expected = self.n_mels.checked_mul(self.n_frames).ok_or_else(|| {
            NativeAsrError::SessionFailed {
                message: "Whisper mel feature shape overflows value count".to_string(),
            }
        })?;
        if self.data.len() != expected {
            return Err(NativeAsrError::SessionFailed {
                message: format!(
                    "Whisper mel feature value count mismatch: got {}, expected {}",
                    self.data.len(),
                    expected
                ),
            });
        }
        TensorOwnedF32::contiguous(self.data.clone(), &[1, self.n_mels, self.n_frames]).map_err(
            |error| NativeAsrError::SessionFailed {
                message: format!("Whisper frontend could not materialize log-mel tensor: {error}"),
            },
        )
    }
}

pub fn build_whisper_mel_frontend_plan_v0(
    n_mels: usize,
    target_frames: usize,
) -> Result<WhisperMelFrontendPlan, NativeAsrError> {
    if n_mels == 0 {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper frontend requires num_mel_bins > 0".to_string(),
        });
    }
    if target_frames == 0 {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper frontend requires target_frames > 0".to_string(),
        });
    }
    Ok(WhisperMelFrontendPlan {
        n_mels,
        target_frames,
        window: hann_window_periodic_scale_first(WHISPER_N_FFT),
        mel_filters: crate::models::audio_frontend::mel::filterbank(FilterbankConfig {
            scale: MelScale::Slaney,
            sample_rate_hz: WHISPER_SAMPLE_RATE_HZ as f32,
            n_fft: WHISPER_N_FFT,
            n_mels,
            fmin: WHISPER_MEL_FMIN,
            fmax: WHISPER_MEL_FMAX,
            // whisper's own pre-refactor filterbank builder computed the
            // mel-domain edge ratio before scaling by the mel span; see
            // `MelPointOrder::RatioFirst`'s doc for why this must stay
            // pinned rather than reuse `parakeet_ctc`'s edge-construction
            // order.
            mel_point_order: MelPointOrder::RatioFirst,
        }),
    })
}

pub fn whisper_mel_features_from_prepared_audio_v0(
    prepared_audio: &GgmlAsrPreparedAudio,
    n_mels: usize,
    target_frames: usize,
) -> Result<WhisperMelFeatures, NativeAsrError> {
    if prepared_audio.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ {
        return Err(NativeAsrError::SessionFailed {
            message: format!(
                "Whisper frontend requires sample_rate_hz={}, got {}",
                WHISPER_SAMPLE_RATE_HZ, prepared_audio.sample_rate_hz
            ),
        });
    }
    if prepared_audio.channels != WHISPER_CHANNELS {
        return Err(NativeAsrError::SessionFailed {
            message: format!(
                "Whisper frontend requires channels={}, got {}",
                WHISPER_CHANNELS, prepared_audio.channels
            ),
        });
    }

    let plan = build_whisper_mel_frontend_plan_v0(n_mels, target_frames)?;
    whisper_mel_features_from_samples_with_plan_v0(&prepared_audio.samples_f32, &plan)
}

pub fn whisper_mel_features_from_samples_with_plan_v0(
    samples: &[f32],
    plan: &WhisperMelFrontendPlan,
) -> Result<WhisperMelFeatures, NativeAsrError> {
    if samples.is_empty() {
        return Err(NativeAsrError::SessionFailed {
            message: "Whisper frontend requires at least one audio sample".to_string(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(NativeAsrError::SessionFailed {
            message: "Whisper frontend input contains non-finite samples".to_string(),
        });
    }
    if plan.n_mels == 0 {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper frontend requires num_mel_bins > 0".to_string(),
        });
    }
    if plan.target_frames == 0 {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: "Whisper frontend requires target_frames > 0".to_string(),
        });
    }

    // OpenAI whisper's `log_mel_spectrogram` pads/trims to a fixed
    // `target_frames * hop` sample window before framing (untrusted-pack
    // sizes stay checked-arithmetic since `target_frames` comes from parsed
    // `.oasr` metadata).
    let target_audio_samples = plan
        .target_frames
        .checked_mul(WHISPER_HOP_LENGTH)
        .ok_or_else(|| NativeAsrError::SessionFailed {
            message: "Whisper frontend target frame count overflows audio length".to_string(),
        })?;
    let audio = pad_or_trim_audio(samples, target_audio_samples);

    let framer = StftFramer::new(
        WHISPER_N_FFT,
        WHISPER_N_FFT,
        WHISPER_HOP_LENGTH,
        PadMode::ReflectCenter,
        plan.window.clone(),
    );
    let spectrogram =
        framer
            .power_spectrogram(&audio)
            .map_err(|error| NativeAsrError::SessionFailed {
                message: format!("Whisper frontend STFT failed: {error}"),
            })?;

    // `torch.stft(..., center=True, pad_mode="reflect")` on a signal padded
    // to exactly `target_frames * hop` samples yields `target_frames + 1`
    // frames; OpenAI whisper's `log_mel_spectrogram` always drops the
    // trailing frame (`log_spec[..., :-1]`) to land on exactly
    // `target_frames`.
    let fft_bins = framer.n_fft_bins();
    if spectrogram.n_frames <= plan.target_frames {
        return Err(NativeAsrError::SessionFailed {
            message: format!(
                "Whisper frontend produced {} spectrogram frames, expected more than {}",
                spectrogram.n_frames, plan.target_frames
            ),
        });
    }
    let frame_count = plan.target_frames;
    let power_spectrogram = &spectrogram.data[..frame_count * fft_bins];

    let power_view = TensorViewF32::contiguous(power_spectrogram, &[frame_count, fft_bins])
        .map_err(|error| NativeAsrError::SessionFailed {
            message: format!("Whisper frontend power spectrogram view failed: {error}"),
        })?;
    let mel_filter_view = TensorViewF32::contiguous(&plan.mel_filters, &[plan.n_mels, fft_bins])
        .map_err(|error| NativeAsrError::SessionFailed {
            message: format!("Whisper frontend mel filter view failed: {error}"),
        })?;
    let mel_rows = linear_f32(&power_view, &mel_filter_view, None).map_err(|error| {
        NativeAsrError::SessionFailed {
            message: format!("Whisper frontend mel projection failed: {error}"),
        }
    })?;
    let mut mel_values = vec![0.0_f32; plan.n_mels * frame_count];
    for frame_idx in 0..frame_count {
        let src = &mel_rows[frame_idx * plan.n_mels..(frame_idx + 1) * plan.n_mels];
        for (mel_idx, value) in src.iter().enumerate() {
            mel_values[mel_idx * frame_count + frame_idx] = *value;
        }
    }
    normalize_log_mel_in_place(&mut mel_values);
    emit_mel_probe_trace(&mel_values, plan.n_mels, frame_count);
    Ok(WhisperMelFeatures {
        n_mels: plan.n_mels,
        n_frames: frame_count,
        data: mel_values,
    })
}

fn emit_mel_probe_trace(values: &[f32], n_mels: usize, n_frames: usize) {
    if std::env::var_os("OPENASR_WHISPER_MEL_TRACE").is_none() {
        return;
    }
    let items = values
        .iter()
        .take(12)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let (min, max, sum_abs) = values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0_f32),
        |(min, max, sum_abs), value| (min.min(value), max.max(value), sum_abs + value.abs()),
    );
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        sum_abs / values.len() as f32
    };
    eprintln!(
        "openasr_whisper_mel_trace n_mels={n_mels} n_frames={n_frames} first_mel_major={items} min={min:.6} max={max:.6} mean_abs={mean_abs:.6}"
    );
}

fn pad_or_trim_audio(samples: &[f32], target_audio_samples: usize) -> Vec<f32> {
    let trimmed = if samples.len() > target_audio_samples {
        &samples[..target_audio_samples]
    } else {
        samples
    };
    let mut output = vec![0.0_f32; target_audio_samples];
    output[..trimmed.len()].copy_from_slice(trimmed);
    output
}

fn normalize_log_mel_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = value.max(WHISPER_LOG_SPEC_FLOOR).log10();
    }
    let max_value = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |left, right| left.max(right));
    let floor = max_value - WHISPER_LOG_SPEC_DYNAMIC_RANGE;
    for value in values.iter_mut() {
        *value = ((*value).max(floor) + WHISPER_LOG_SPEC_SHIFT) / WHISPER_LOG_SPEC_SHIFT;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_features_are_finite_with_expected_shape() {
        let audio = GgmlAsrPreparedAudio::mono_16khz(vec![0.0_f32; 16_000]);
        let mel = whisper_mel_features_from_prepared_audio_v0(&audio, 80, 32).unwrap();
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.n_frames, 32);
        assert_eq!(mel.data.len(), 80 * 32);
        assert!(mel.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn impulse_features_are_finite_with_expected_shape() {
        let mut samples = vec![0.0_f32; 16_000];
        samples[0] = 1.0;
        let audio = GgmlAsrPreparedAudio::mono_16khz(samples);
        let mel = whisper_mel_features_from_prepared_audio_v0(&audio, 80, 64).unwrap();
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.n_frames, 64);
        assert_eq!(mel.data.len(), 80 * 64);
        assert!(mel.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn mel_filterbank_matches_openai_whisper_slaney_reference_points() {
        let plan = build_whisper_mel_frontend_plan_v0(80, 3000).unwrap();
        assert!((plan.mel_filters[1] - 0.02486259).abs() < 1.0e-7);
        let fft_bins = WHISPER_N_FFT / 2 + 1;
        let row1_sum = plan.mel_filters[fft_bins..2 * fft_bins]
            .iter()
            .copied()
            .sum::<f32>();
        assert!((row1_sum - 0.02486259).abs() < 1.0e-7);
    }

    #[test]
    fn reject_invalid_sample_rate_fail_closed() {
        let audio = GgmlAsrPreparedAudio {
            sample_rate_hz: 8_000,
            channels: 1,
            samples_f32: vec![0.0, 0.1, 0.2],
        };
        let error = whisper_mel_features_from_prepared_audio_v0(&audio, 80, 16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sample_rate_hz=16000"), "{error}");
    }

    #[test]
    fn reject_non_finite_input_fail_closed() {
        let audio = GgmlAsrPreparedAudio::mono_16khz(vec![0.0, f32::NAN, 0.2]);
        let error = whisper_mel_features_from_prepared_audio_v0(&audio, 80, 16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-finite"), "{error}");
    }
}
