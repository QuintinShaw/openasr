//! Speaker embedding.
//!
//! The only supported speaker embedder is ReDimNet2-B6 (192-d, ggml graph,
//! Chinese-enhanced). Weights load from a pulled/local `.oasr` pack and are not
//! vendored. When the pack is missing, diarization and Voice ID fail closed.

mod pack;
mod policy_runtime;
// ReDimNet2-B6 embedder (192-d, ggml graph).
pub(crate) mod ops;
mod redimnet;
pub(crate) mod weights;

#[cfg(test)]
mod tests;

pub use pack::{
    DIARIZATION_EMBEDDER_LOAD_FAILED_REASON, REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON,
    SPEAKER_EMBEDDER_PACK_ID, SPEAKER_EMBEDDER_PACK_LABEL, SpeakerEmbedderIdentity,
    VOICE_ID_EMBEDDER_PACK_MISSING_REASON, VOICE_ID_NAMING_EMBEDDER_MISSING_REASON,
    VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON, embedder_pack_installed,
};
pub use policy_runtime::PolicyResolvedSpeakerRuntime;
pub(crate) use redimnet::backbone::RedimNetResidentRuntime;

#[cfg(test)]
use std::sync::OnceLock;
use thiserror::Error;

use super::calibration::{REDIMNET_CALIBRATION, SpeakerCalibrationProfile};
use super::contract::SpeakerEmbedding;
use redimnet::backbone::RedimNet2Model;
use redimnet::frontend::RedimNetFrontend;

/// Sample rate the embedder requires.
const SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const REDIMNET_MAX_BATCH_WORKERS: usize = 4;

pub(crate) const fn redimnet_frontend_payload_quote(samples: usize) -> (u64, u64) {
    RedimNetFrontend::quoted_forward_payload_bytes(samples)
}

#[cfg(test)]
const REDIMNET_BENCH_WORKERS_ENV: &str = "OPENASR_REDIMNET_BENCH_WORKERS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpeakerEmbeddingExecutionPlan {
    pub(crate) workers: usize,
    pub(crate) threads_per_runner: usize,
}

impl SpeakerEmbeddingExecutionPlan {
    pub(crate) fn for_clips(clips: usize, available: usize, pool_threads: usize) -> Self {
        let workers = clips.max(1).min(pool_threads.max(1));
        Self {
            workers,
            threads_per_runner: (available.max(1) / workers).max(1),
        }
    }

    pub(crate) fn worker_range(self, worker: usize, clips: usize) -> std::ops::Range<usize> {
        worker * clips / self.workers..(worker + 1) * clips / self.workers
    }
}

pub(crate) fn redimnet_batch_worker_limit(pool_threads: usize) -> usize {
    #[cfg(test)]
    if let Some(limit) = std::env::var(REDIMNET_BENCH_WORKERS_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|limit| (1..=REDIMNET_MAX_BATCH_WORKERS).contains(limit))
    {
        return pool_threads.max(1).min(limit);
    }
    pool_threads.clamp(1, REDIMNET_MAX_BATCH_WORKERS)
}

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("speaker-embedding model is unavailable: {0}")]
    Unavailable(String),
    #[error("speaker-embedding backend failed terminally: {0}")]
    TerminalBackend(String),
    #[error("speaker-embedding batch aborted after a terminal backend failure: {0}")]
    BatchAbortedAfterTerminalBackend(String),
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

    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, EmbedError> {
        let model = RedimNet2Model::from_preflight(preflight)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        Ok(Self {
            model,
            frontend: RedimNetFrontend::new(),
        })
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, EmbedError> {
        let model_bytes = self
            .model
            .persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        let frontend_bytes = self
            .frontend
            .persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        model_bytes.checked_add(frontend_bytes).ok_or_else(|| {
            EmbedError::Unavailable("redimnet persistent host byte sum overflow".to_string())
        })
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, EmbedError> {
        let model_bytes = RedimNet2Model::quoted_persistent_host_commitment_bytes(tensor_index)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        let frontend_bytes = RedimNetFrontend::quoted_persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        model_bytes.checked_add(frontend_bytes).ok_or_else(|| {
            EmbedError::Unavailable("redimnet quoted host byte sum overflow".to_string())
        })
    }

    /// Human-readable identifier for this embedder's embedding space; see
    /// `pack::REDIMNET_EMBEDDING_SPACE_VERSION` for what changes it (and, more
    /// importantly, what does not -- the actual compatibility gate is the pack
    /// content fingerprint, not this label).
    pub fn embedding_space_version(&self) -> &'static str {
        pack::REDIMNET_EMBEDDING_SPACE_VERSION
    }

    pub(crate) fn prepare_embedding_input(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(Vec<f32>, usize), EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        if crate::ggml_runtime::thread_job_cancel_flag()
            .as_ref()
            .is_some_and(cancel_requested)
        {
            return Err(EmbedError::Canceled);
        }
        let (features, frames) = self.frontend.forward(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
        }
        Ok((features, frames))
    }

    pub(crate) fn shared_weights(&self) -> std::sync::Arc<weights::Weights> {
        self.model.shared_weights()
    }
}

pub(crate) fn cancel_requested(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> bool {
    flag.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(test)]
fn embed_batch_worker_range(
    clips: &[&[f32]],
    inherited_cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
    terminal_failure: &OnceLock<String>,
    mut embed: impl FnMut(&[f32]) -> Result<SpeakerEmbedding, EmbedError>,
) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
    let mut results = Vec::with_capacity(clips.len());
    for samples in clips {
        let _cancel_guard = inherited_cancel.map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
        let result = if inherited_cancel.is_some_and(cancel_requested) {
            Err(EmbedError::Canceled)
        } else if let Some(reason) = terminal_failure.get() {
            Err(EmbedError::BatchAbortedAfterTerminalBackend(reason.clone()))
        } else {
            embed(samples)
        };
        if let Err(EmbedError::TerminalBackend(reason)) = &result {
            let _ = terminal_failure.set(reason.clone());
        }
        results.push(result);
    }
    results
}

fn abort_successful_results_after_terminal_failure(
    results: &mut [Result<SpeakerEmbedding, EmbedError>],
    reason: &str,
) {
    for result in results {
        if result.is_ok() {
            *result = Err(EmbedError::BatchAbortedAfterTerminalBackend(
                reason.to_string(),
            ));
        }
    }
}
