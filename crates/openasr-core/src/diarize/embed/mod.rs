//! Speaker embedding.
//!
//! The only supported speaker embedder is ReDimNet2-B6 (192-d, ggml graph,
//! Chinese-enhanced). Weights load from a pulled/local `.oasr` pack and are not
//! vendored. When the pack is missing, diarization and Voice ID fail closed.

mod pack;
// ReDimNet2-B6 embedder (192-d, ggml graph).
pub(crate) mod ops;
mod redimnet;
pub(crate) mod weights;

#[cfg(test)]
mod tests;

pub use pack::{
    SpeakerEmbedderIdentity, embedder_pack_installed, shared_embedder, shared_embedder_identity,
};

use thiserror::Error;

use super::calibration::{REDIMNET_CALIBRATION, SpeakerCalibrationProfile};
use super::contract::SpeakerEmbedding;
use redimnet::backbone::RedimNet2Model;
use redimnet::frontend::RedimNetFrontend;

/// Sample rate the embedder requires.
const SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("speaker-embedding model is unavailable: {0}")]
    Unavailable(String),
    #[error("audio is too short to embed (need at least one frame)")]
    TooShort,
    #[error("speaker embedder requires 16 kHz mono audio, got {0} Hz")]
    UnsupportedSampleRate(u32),
}

/// Turns a speech segment (16 kHz mono `f32`) into a speaker embedding.
pub trait SpeakerEmbedder: Send + Sync {
    /// Embed `samples`; the result is L2-normalized.
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError>;

    /// Embedding dimensionality (ReDimNet2-B6 = 192).
    fn embedding_dim(&self) -> usize;

    /// Calibration profile for clustering and streaming gates in this embedder's
    /// cosine space. Defaults to the ReDimNet2-B6 profile.
    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        REDIMNET_CALIBRATION
    }
}

/// ReDimNet2-B6 embedder: `TFMelBanks` front end + ggml-graph backbone,
/// Chinese-enhanced (vb2+vox2+cnc2) checkpoint. `embedding_dim() == 192`.
/// Compatibility across packs is gated by `SpeakerProfile::is_compatible_with`
/// (keyed on `embedding_dim` + `pack_fingerprint`).
pub struct RedimNet2Embedder {
    model: RedimNet2Model,
    frontend: RedimNetFrontend,
}

impl RedimNet2Embedder {
    pub fn from_oasr(path: &std::path::Path) -> Result<Self, EmbedError> {
        let model =
            RedimNet2Model::from_oasr(path).map_err(|e| EmbedError::Unavailable(e.to_string()))?;
        Ok(Self {
            model,
            frontend: RedimNetFrontend::new(),
        })
    }

    /// Human-readable identifier for this embedder's embedding space; see
    /// `pack::REDIMNET_EMBEDDING_SPACE_VERSION` for what changes it (and, more
    /// importantly, what does not -- the actual compatibility gate is the pack
    /// content fingerprint, not this label).
    pub fn embedding_space_version(&self) -> &'static str {
        pack::REDIMNET_EMBEDDING_SPACE_VERSION
    }
}

impl SpeakerEmbedder for RedimNet2Embedder {
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        let (features, frames) = self.frontend.forward(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
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
        REDIMNET_CALIBRATION
    }
}
