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

pub(crate) use pack::speaker_attribution_admission_bytes;
pub use pack::{
    DIARIZATION_EMBEDDER_LOAD_FAILED_REASON, REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON,
    SPEAKER_EMBEDDER_PACK_ID, SPEAKER_EMBEDDER_PACK_LABEL, SpeakerEmbedderIdentity,
    VOICE_ID_EMBEDDER_PACK_MISSING_REASON, VOICE_ID_NAMING_EMBEDDER_MISSING_REASON,
    VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON, embedder_pack_installed, shared_embedder,
    shared_embedder_identity,
};

use rayon::prelude::*;
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
    #[error("speaker embedding was canceled")]
    Canceled,
}

/// Turns a speech segment (16 kHz mono `f32`) into a speaker embedding.
pub trait SpeakerEmbedder: Send + Sync {
    /// Embed `samples`; the result is L2-normalized.
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError>;

    /// Embed independent clips in input order. The default stays object-safe
    /// and preserves compatibility for simple/test embedders; runtimes with a
    /// safe session pool can override it for parallel execution.
    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        let cancel = crate::ggml_runtime::thread_job_cancel_flag();
        clips
            .iter()
            .map(|samples| {
                if cancel.as_ref().is_some_and(cancel_requested) {
                    Err(EmbedError::Canceled)
                } else {
                    self.embed(samples, sample_rate_hz)
                }
            })
            .collect()
    }

    /// Embedding dimensionality (ReDimNet2-B6 = 192).
    fn embedding_dim(&self) -> usize;

    /// Calibration profile for clustering and streaming gates in this embedder's
    /// cosine space. Defaults to the ReDimNet2-B6 profile.
    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        REDIMNET_CALIBRATION
    }

    /// Content identity of this exact embedding space, when the embedder is
    /// backed by a model pack. Returning an owned value keeps the trait
    /// object-safe and prevents a path replacement from invalidating a borrow.
    fn identity(&self) -> Option<SpeakerEmbedderIdentity> {
        None
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

    pub(crate) fn from_runtime_source(
        source: &crate::GgmlRuntimeSource,
    ) -> Result<Self, EmbedError> {
        let model = RedimNet2Model::from_runtime_source(source)
            .map_err(|e| EmbedError::Unavailable(e.to_string()))?;
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

    fn embed_with_threads(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        n_threads: Option<usize>,
    ) -> Result<SpeakerEmbedding, EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        if crate::ggml_runtime::thread_job_cancel_flag()
            .as_ref()
            .is_some_and(cancel_requested)
        {
            return Err(EmbedError::Canceled);
        }
        #[cfg(test)]
        let _active_probe = RedimActiveProbe::enter();
        let (features, frames) = self.frontend.forward(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
        }
        let raw = self
            .model
            .forward_with_threads(&features, frames, n_threads)
            .map_err(|error| {
                if error.is_canceled() {
                    EmbedError::Canceled
                } else {
                    EmbedError::Unavailable(error.to_string())
                }
            })?;
        Ok(SpeakerEmbedding::l2_normalized(raw))
    }

    #[cfg(test)]
    fn embed_uncached_for_bench(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<SpeakerEmbedding, EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        let (features, frames) = self.frontend.forward(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
        }
        let raw = self
            .model
            .forward_uncached_for_bench(&features, frames)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        Ok(SpeakerEmbedding::l2_normalized(raw))
    }
}

impl SpeakerEmbedder for RedimNet2Embedder {
    fn embed(&self, samples: &[f32], sample_rate_hz: u32) -> Result<SpeakerEmbedding, EmbedError> {
        self.embed_with_threads(samples, sample_rate_hz, None)
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        if clips.len() <= 1 {
            return clips
                .iter()
                .map(|samples| self.embed(samples, sample_rate_hz))
                .collect();
        }
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let workers = clips.len().min(available).max(1);
        let threads_per_runner = (available / workers).max(1);
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        clips
            .par_iter()
            .map(|samples| {
                let _cancel_guard = inherited_cancel.as_ref().map(InheritedCancelGuard::arm);
                if inherited_cancel.as_ref().is_some_and(cancel_requested) {
                    Err(EmbedError::Canceled)
                } else {
                    self.embed_with_threads(samples, sample_rate_hz, Some(threads_per_runner))
                }
            })
            .collect()
    }

    fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }

    fn calibration_profile(&self) -> SpeakerCalibrationProfile {
        REDIMNET_CALIBRATION
    }

    fn identity(&self) -> Option<SpeakerEmbedderIdentity> {
        Some(SpeakerEmbedderIdentity {
            embedding_dim: self.embedding_dim(),
            pack_fingerprint: self.model.pack_content_id().to_string(),
        })
    }
}

fn cancel_requested(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> bool {
    flag.load(std::sync::atomic::Ordering::SeqCst)
}

struct InheritedCancelGuard {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    previous: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl InheritedCancelGuard {
    fn arm(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        let flag = std::sync::Arc::clone(flag);
        let previous =
            crate::ggml_runtime::arm_thread_job_cancel_flag(Some(std::sync::Arc::clone(&flag)));
        Self { flag, previous }
    }
}

impl Drop for InheritedCancelGuard {
    fn drop(&mut self) {
        let _ = crate::ggml_runtime::disarm_thread_job_cancel_flag_if_current(
            &self.flag,
            self.previous.take(),
        );
    }
}

#[cfg(test)]
static REDIM_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static REDIM_MAX_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
struct RedimActiveProbe;

#[cfg(test)]
impl RedimActiveProbe {
    fn enter() -> Self {
        use std::sync::atomic::Ordering;
        let active = REDIM_ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
        REDIM_MAX_ACTIVE.fetch_max(active, Ordering::SeqCst);
        Self
    }
}

#[cfg(test)]
impl Drop for RedimActiveProbe {
    fn drop(&mut self) {
        REDIM_ACTIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
fn reset_redim_max_active() {
    REDIM_MAX_ACTIVE.store(0, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
fn redim_max_active() -> usize {
    REDIM_MAX_ACTIVE.load(std::sync::atomic::Ordering::SeqCst)
}
