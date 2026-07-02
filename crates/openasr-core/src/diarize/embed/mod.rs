//! Speaker embedding.
//!
//! Pure-Rust WeSpeaker ResNet34 embeddings for clustering and enrollment.
//! Weights are loaded from pulled/local `.oasr` packs and are not vendored.

mod fbank;
pub(crate) mod ops;
mod pack;
pub(crate) mod weights;
mod wespeaker;

#[cfg(test)]
mod tests;

pub use pack::{
    SpeakerEmbedderIdentity, embedder_pack_installed, shared_embedder, shared_embedder_identity,
};

use thiserror::Error;

use super::calibration::{SpeakerCalibrationProfile, WESPEAKER_CALIBRATION};
use super::contract::SpeakerEmbedding;
use fbank::Fbank;
use wespeaker::WeSpeakerResNet34Model;

/// Sample rate the embedder requires.
const SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("speaker-embedding model is unavailable: {0}")]
    Unavailable(String),
    #[error("audio is too short to embed (need at least one frame)")]
    TooShort,
    #[error(
        "audio is too short for WeSpeaker stats pooling after ResNet stride (input frames {frames}, post-stride time length {post_stride_frames}, need at least 2)"
    )]
    WeSpeakerPostStrideTooShort {
        frames: usize,
        post_stride_frames: usize,
    },
    #[error("speaker embedder requires 16 kHz mono audio, got {0} Hz")]
    UnsupportedSampleRate(u32),
}

/// Turns a speech segment (16 kHz mono `f32`) into a speaker embedding.
pub trait SpeakerEmbedder: Send + Sync {
    /// Embed `samples`; the result is L2-normalized.
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError>;

    /// Embedding dimensionality (WeSpeaker ResNet34 = 256).
    fn embedding_dim(&self) -> usize;

    /// Calibration profile for clustering and streaming gates in this embedder's
    /// cosine space.
    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        WESPEAKER_CALIBRATION
    }
}

/// pyannote/WeSpeaker ResNet34 embedder: hamming Kaldi-fbank + pure-Rust network.
pub struct WeSpeakerEmbedder {
    model: WeSpeakerResNet34Model,
    fbank: Fbank,
}

impl WeSpeakerEmbedder {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, EmbedError> {
        let model = WeSpeakerResNet34Model::from_safetensors(bytes)
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;
        Ok(Self {
            model,
            fbank: Fbank::wespeaker(),
        })
    }

    pub fn from_oasr(path: &std::path::Path) -> Result<Self, EmbedError> {
        let model = WeSpeakerResNet34Model::from_oasr(path)
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;
        Ok(Self {
            model,
            fbank: Fbank::wespeaker(),
        })
    }
}

impl SpeakerEmbedder for WeSpeakerEmbedder {
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        let (features, frames) = self.fbank.compute(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
        }
        let post_stride_frames = WeSpeakerResNet34Model::post_stride_time_len(frames);
        if post_stride_frames < 2 {
            return Err(EmbedError::WeSpeakerPostStrideTooShort {
                frames,
                post_stride_frames,
            });
        }
        let raw = self
            .model
            .forward(&features, frames)
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;
        Ok(SpeakerEmbedding::l2_normalized(raw))
    }

    fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }

    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        WESPEAKER_CALIBRATION
    }
}
