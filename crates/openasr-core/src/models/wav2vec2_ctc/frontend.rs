//! wav2vec2-ctc RAW-waveform frontend.
//!
//! wav2vec2 takes raw 16 kHz mono PCM directly — NO STFT/mel. The HF
//! `Wav2Vec2FeatureExtractor` with `do_normalize=True` zero-mean/unit-var
//! normalizes the whole utterance, then the raw f32 samples feed the in-graph
//! conv feature extractor (see `encoder_graph`). This file produces the
//! normalized sample buffer; the conv stack runs in the encoder graph.

#![allow(dead_code)]

const NORM_EPS: f64 = 1.0e-7;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Wav2Vec2FrontendError {
    #[error(
        "wav2vec2-ctc frontend: audio too short ({samples} samples) for the conv feature extractor"
    )]
    TooShort { samples: usize },
}

/// Zero-mean / unit-variance normalized raw 16 kHz PCM for one utterance.
#[derive(Debug, Clone)]
pub(crate) struct Wav2Vec2RawAudio {
    pub samples: Vec<f32>,
    pub n_samples: usize,
}

pub(crate) struct Wav2Vec2Frontend;

impl Wav2Vec2Frontend {
    pub(crate) fn new() -> Self {
        Self
    }

    /// Normalize 16 kHz mono f32 PCM to zero-mean/unit-var over the utterance
    /// (`(x - mean) / sqrt(var + 1e-7)`, population variance — matches HF
    /// `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`).
    pub(crate) fn features_from_samples(
        &self,
        samples: &[f32],
    ) -> Result<Wav2Vec2RawAudio, Wav2Vec2FrontendError> {
        // The conv stack needs at least one full receptive field; the first conv
        // kernel is 10 with stride 5, so a handful of hundred samples minimum.
        if samples.len() < 400 {
            return Err(Wav2Vec2FrontendError::TooShort {
                samples: samples.len(),
            });
        }
        let n = samples.len();
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var = samples
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let inv_std = 1.0 / (var + NORM_EPS).sqrt();
        let normalized: Vec<f32> = samples
            .iter()
            .map(|&x| ((x as f64 - mean) * inv_std) as f32)
            .collect();
        Ok(Wav2Vec2RawAudio {
            samples: normalized,
            n_samples: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_zero_mean_unit_var() {
        let frontend = Wav2Vec2Frontend::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.3 + 0.1)
            .collect();
        let out = frontend.features_from_samples(&samples).expect("features");
        assert_eq!(out.n_samples, 16_000);
        let n = out.samples.len() as f64;
        let mean = out.samples.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = out
            .samples
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 1.0e-4, "mean {mean}");
        assert!((var - 1.0).abs() < 1.0e-3, "var {var}");
    }

    #[test]
    fn rejects_too_short_audio() {
        let frontend = Wav2Vec2Frontend::new();
        assert!(frontend.features_from_samples(&[0.0; 100]).is_err());
    }
}
