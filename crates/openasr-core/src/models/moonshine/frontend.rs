use std::borrow::Cow;

use thiserror::Error;

use crate::models::ggml_asr_executor::GgmlAsrPreparedAudioView;

/// Raw 16 kHz mono PCM samples fed directly to the conv stem.
///
/// Moonshine sets `do_normalize=false`, so unlike wav2vec2-960h there is NO zero-mean /
/// unit-variance normalization: the raw f32 waveform is the feature input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MoonshineWaveformFeatures<'a> {
    pub samples: Cow<'a, [f32]>,
}

#[derive(Debug, Error)]
pub(crate) enum MoonshineFrontendError {
    #[error("moonshine frontend received unexpected sample rate {got} (expected {expected})")]
    UnexpectedSampleRate { got: u32, expected: u32 },
    #[error("moonshine frontend received empty audio")]
    EmptyAudio,
    #[error("moonshine frontend received non-finite samples")]
    NonFiniteSamples,
}

pub(crate) fn moonshine_waveform_from_prepared_audio<'a>(
    audio: &'a GgmlAsrPreparedAudioView<'_>,
    expected_sample_rate_hz: u32,
) -> Result<MoonshineWaveformFeatures<'a>, MoonshineFrontendError> {
    if audio.sample_rate_hz != expected_sample_rate_hz {
        return Err(MoonshineFrontendError::UnexpectedSampleRate {
            got: audio.sample_rate_hz,
            expected: expected_sample_rate_hz,
        });
    }
    let samples = downmix_to_mono(&audio.samples_f32, audio.channels);
    if samples.is_empty() {
        return Err(MoonshineFrontendError::EmptyAudio);
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(MoonshineFrontendError::NonFiniteSamples);
    }
    Ok(MoonshineWaveformFeatures { samples })
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Cow<'_, [f32]> {
    if channels <= 1 {
        return Cow::Borrowed(samples);
    }
    let channels = channels as usize;
    Cow::Owned(
        samples
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_raw_mono_samples_without_normalization() {
        let audio = GgmlAsrPreparedAudioView::mono_16khz(vec![0.5, -0.25, 1.0]);
        let features = moonshine_waveform_from_prepared_audio(&audio, 16_000).expect("features");
        assert!(matches!(&features.samples, Cow::Borrowed(_)));
        assert_eq!(features.samples.as_ref(), &[0.5, -0.25, 1.0]);
        assert_eq!(features.samples.as_ptr(), audio.samples_f32.as_ptr());
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut audio = GgmlAsrPreparedAudioView::mono_16khz(vec![0.1, 0.2]);
        audio.sample_rate_hz = 8_000;
        let error = moonshine_waveform_from_prepared_audio(&audio, 16_000).expect_err("must fail");
        assert!(matches!(
            error,
            MoonshineFrontendError::UnexpectedSampleRate { got: 8_000, .. }
        ));
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        let audio = GgmlAsrPreparedAudioView {
            sample_rate_hz: 16_000,
            channels: 2,
            samples_f32: vec![1.0, 3.0, 0.0, 4.0].into(),
        };
        let features = moonshine_waveform_from_prepared_audio(&audio, 16_000).expect("features");
        assert!(matches!(&features.samples, Cow::Owned(_)));
        assert_eq!(features.samples.as_ref(), &[2.0, 2.0]);
    }
}
