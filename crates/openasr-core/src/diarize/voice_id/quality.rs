//! Enrollment audio quality gate.
//!
//! Raw audio is never retained. Only the quality summary and the embedding
//! produced from accepted speech leave this module.

use thiserror::Error;

use super::domain::SampleQuality;

/// Minimum accepted speech for a single enrollment sample.
pub const MIN_SAMPLE_SPEECH_SECONDS: f32 = 5.0;
/// Soft target used when scoring quality weight (does not reject).
pub const TARGET_SAMPLE_SPEECH_SECONDS: f32 = 12.0;
const SAMPLE_RATE_HZ: usize = 16_000;
const FRAME_SAMPLES: usize = SAMPLE_RATE_HZ / 50; // 20 ms
const RMS_SPEECH_FLOOR: f32 = 0.01;
const CLIP_ABS: f32 = 0.98;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum QualityError {
    #[error("enrollment audio is silent: no speech was detected")]
    NoSpeech,
    #[error(
        "enrollment audio is too short: need at least {required:.1} seconds of speech, got {actual:.2}"
    )]
    TooShortSpeech { required: f32, actual: f32 },
    #[error("enrollment audio is heavily clipped ({ratio:.1}% of frames)")]
    TooMuchClipping { ratio: f32 },
    #[error("enrollment audio has too little speech coverage ({coverage:.1}%)")]
    LowVadCoverage { coverage: f32 },
}

/// Assess enrollment PCM and return a quality summary, or reject the sample.
pub fn assess_enrollment_quality(samples: &[f32]) -> Result<SampleQuality, QualityError> {
    if samples.is_empty() {
        return Err(QualityError::NoSpeech);
    }

    let mut speech_frames = 0usize;
    let mut total_frames = 0usize;
    let mut clipped_frames = 0usize;
    let mut speech_energy = 0.0f32;
    let mut noise_energy = 0.0f32;
    let mut noise_frames = 0usize;
    let mut speech_seconds = 0.0f32;

    for chunk in samples.chunks(FRAME_SAMPLES) {
        if chunk.is_empty() {
            continue;
        }
        total_frames += 1;
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        let clipped = chunk.iter().any(|s| s.abs() >= CLIP_ABS);
        if clipped {
            clipped_frames += 1;
        }
        if rms >= RMS_SPEECH_FLOOR {
            speech_frames += 1;
            speech_seconds += chunk.len() as f32 / SAMPLE_RATE_HZ as f32;
            speech_energy += rms * rms;
        } else {
            noise_frames += 1;
            noise_energy += rms * rms;
        }
    }

    if speech_frames == 0 || speech_seconds <= f32::EPSILON {
        return Err(QualityError::NoSpeech);
    }
    if speech_seconds < MIN_SAMPLE_SPEECH_SECONDS {
        return Err(QualityError::TooShortSpeech {
            required: MIN_SAMPLE_SPEECH_SECONDS,
            actual: speech_seconds,
        });
    }

    let clipping_ratio = if total_frames == 0 {
        0.0
    } else {
        clipped_frames as f32 / total_frames as f32
    };
    if clipping_ratio > 0.08 {
        return Err(QualityError::TooMuchClipping {
            ratio: clipping_ratio * 100.0,
        });
    }

    let vad_coverage = if total_frames == 0 {
        0.0
    } else {
        speech_frames as f32 / total_frames as f32
    };
    if vad_coverage < 0.15 {
        return Err(QualityError::LowVadCoverage {
            coverage: vad_coverage * 100.0,
        });
    }

    let mean_speech = if speech_frames == 0 {
        0.0
    } else {
        (speech_energy / speech_frames as f32).max(1e-9)
    };
    let mean_noise = if noise_frames == 0 {
        1e-6
    } else {
        (noise_energy / noise_frames as f32).max(1e-9)
    };
    let snr_estimate = 10.0 * (mean_speech / mean_noise).log10();

    Ok(SampleQuality {
        speech_seconds,
        snr_estimate,
        clipping_ratio,
        vad_coverage,
        accepted_reason: "quality_gate_v1".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, amp: f32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE_HZ as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * amp
            })
            .collect()
    }

    #[test]
    fn accepts_clean_speech_like_audio() {
        let q = assess_enrollment_quality(&tone(8.0, 0.2)).unwrap();
        assert!(q.speech_seconds >= MIN_SAMPLE_SPEECH_SECONDS);
        assert!(q.weight() > 0.2);
    }

    #[test]
    fn rejects_silent_and_short_audio() {
        assert!(matches!(
            assess_enrollment_quality(&vec![0.0; SAMPLE_RATE_HZ]),
            Err(QualityError::NoSpeech)
        ));
        assert!(matches!(
            assess_enrollment_quality(&tone(2.0, 0.2)),
            Err(QualityError::TooShortSpeech { .. })
        ));
    }

    #[test]
    fn rejects_heavy_clipping() {
        let mut samples = tone(8.0, 1.0);
        for s in &mut samples {
            *s = s.signum();
        }
        assert!(matches!(
            assess_enrollment_quality(&samples),
            Err(QualityError::TooMuchClipping { .. })
        ));
    }
}
