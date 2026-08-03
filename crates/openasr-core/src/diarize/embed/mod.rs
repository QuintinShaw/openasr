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
    DIARIZATION_EMBEDDER_LOAD_FAILED_REASON, REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON,
    SPEAKER_EMBEDDER_PACK_ID, SPEAKER_EMBEDDER_PACK_LABEL, SpeakerEmbedderIdentity,
    VOICE_ID_EMBEDDER_PACK_MISSING_REASON, VOICE_ID_NAMING_EMBEDDER_MISSING_REASON,
    VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON, embedder_pack_installed, shared_embedder,
    shared_embedder_identity,
};
pub(crate) use pack::{prepare_shared_embedder_snapshot, unload_idle_embedder_cache};

use rayon::prelude::*;
use std::sync::OnceLock;
use thiserror::Error;

use super::calibration::{REDIMNET_CALIBRATION, SpeakerCalibrationProfile};
use super::contract::SpeakerEmbedding;
use crate::models::thread_local_runtime_cache::{
    DedicatedWorkerRuntimeOwnerLease, DedicatedWorkerRuntimeOwnerTracker,
};
use redimnet::backbone::RedimNet2Model;
use redimnet::frontend::RedimNetFrontend;

/// Sample rate the embedder requires.
const SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const REDIMNET_MAX_BATCH_WORKERS: usize = 4;

#[cfg(test)]
const REDIMNET_BENCH_WORKERS_ENV: &str = "OPENASR_REDIMNET_BENCH_WORKERS";

static REDIMNET_BATCH_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
static REDIMNET_RUNTIME_OWNERS: DedicatedWorkerRuntimeOwnerTracker =
    DedicatedWorkerRuntimeOwnerTracker::new(unload_idle_redimnet_worker_runtimes);

fn redimnet_batch_pool() -> &'static Result<rayon::ThreadPool, String> {
    REDIMNET_BATCH_POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(REDIMNET_MAX_BATCH_WORKERS);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("openasr-redimnet-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
}

/// Eager idle-unload for the dedicated, persistent ReDim worker pool. A mere
/// generation bump is insufficient here: unlike tokio blocking workers these
/// threads intentionally remain alive, so without a broadcast their uploaded
/// arenas would stay resident forever when no later Voice ID request arrives.
pub(crate) fn unload_idle_redimnet_worker_runtimes() {
    if let Some(Ok(pool)) = REDIMNET_BATCH_POOL.get() {
        pool.broadcast(|_| redimnet::backbone::clear_current_thread_runtime_cache());
    }
}

#[cfg(test)]
fn redimnet_worker_runtime_entry_count() -> usize {
    let Some(Ok(pool)) = REDIMNET_BATCH_POOL.get() else {
        return 0;
    };
    pool.broadcast(|_| redimnet::backbone::current_thread_runtime_cache_len())
        .into_iter()
        .sum()
}

#[cfg(test)]
fn install_worker_graph_compute_device_lost() {
    let pool = redimnet_batch_pool()
        .as_ref()
        .expect("ReDimNet test worker pool");
    pool.broadcast(|_| crate::ggml_runtime::install_test_graph_compute_device_lost());
}

#[cfg(test)]
fn clear_worker_graph_compute_status_override() {
    let pool = redimnet_batch_pool()
        .as_ref()
        .expect("ReDimNet test worker pool");
    pool.broadcast(|_| crate::ggml_runtime::clear_test_graph_compute_status_override());
}

#[cfg(test)]
fn redimnet_runtime_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpeakerEmbeddingExecutionPlan {
    workers: usize,
    threads_per_runner: usize,
}

impl SpeakerEmbeddingExecutionPlan {
    fn for_clips(clips: usize, available: usize, pool_threads: usize) -> Self {
        let workers = clips.max(1).min(pool_threads.max(1));
        Self {
            workers,
            threads_per_runner: (available.max(1) / workers).max(1),
        }
    }

    fn worker_range(self, worker: usize, clips: usize) -> std::ops::Range<usize> {
        worker * clips / self.workers..(worker + 1) * clips / self.workers
    }
}

fn redimnet_batch_worker_limit(pool_threads: usize) -> usize {
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
    // Drop first so worker-owned device runtimes are released before the
    // parsed model they were built from. The shared product cache retains the
    // lease across requests; direct low-level users release on final drop.
    _worker_runtime_owner: DedicatedWorkerRuntimeOwnerLease,
    model: RedimNet2Model,
    frontend: RedimNetFrontend,
}

impl RedimNet2Embedder {
    pub fn from_oasr(path: &std::path::Path) -> Result<Self, EmbedError> {
        let model =
            RedimNet2Model::from_oasr(path).map_err(|e| EmbedError::Unavailable(e.to_string()))?;
        Ok(Self {
            _worker_runtime_owner: REDIMNET_RUNTIME_OWNERS.acquire(),
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
            _worker_runtime_owner: REDIMNET_RUNTIME_OWNERS.acquire(),
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

    pub(crate) fn logical_f32_weight_bytes(&self) -> u64 {
        self.model.logical_f32_weight_bytes()
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
                } else if error.is_terminal_backend_failure() {
                    EmbedError::TerminalBackend(error.to_string())
                } else {
                    EmbedError::Unavailable(error.to_string())
                }
            })?;
        Ok(SpeakerEmbedding::l2_normalized(raw))
    }

    fn embed_on_bounded_pool(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        if clips.is_empty() {
            return Vec::new();
        }
        let pool = match redimnet_batch_pool().as_ref() {
            Ok(pool) => pool,
            Err(error) => {
                return clips
                    .iter()
                    .map(|_| {
                        Err(EmbedError::Unavailable(format!(
                            "could not create bounded ReDimNet worker pool: {error}"
                        )))
                    })
                    .collect();
            }
        };
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let plan = SpeakerEmbeddingExecutionPlan::for_clips(
            clips.len(),
            available,
            redimnet_batch_worker_limit(pool.current_num_threads()),
        );
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        let terminal_failure = OnceLock::new();
        let mut results: Vec<Result<SpeakerEmbedding, EmbedError>> = pool.install(|| {
            (0..plan.workers)
                .into_par_iter()
                .map(|worker| {
                    embed_batch_worker_range(
                        &clips[plan.worker_range(worker, clips.len())],
                        inherited_cancel.as_ref(),
                        &terminal_failure,
                        |samples| {
                            self.embed_with_threads(
                                samples,
                                sample_rate_hz,
                                Some(plan.threads_per_runner),
                            )
                        },
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });
        if let Some(reason) = terminal_failure.get() {
            // A terminal device/backend failure invalidates every resident
            // handle in this request's worker pool. Evict only after the batch
            // has stopped; the failed batch is never retried, and the next
            // request is the first place a runtime may be rebuilt.
            pool.broadcast(|_| redimnet::backbone::clear_current_thread_runtime_cache());
            abort_successful_results_after_terminal_failure(&mut results, reason);
        }
        results
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
        self.embed_on_bounded_pool(&[samples], sample_rate_hz)
            .into_iter()
            .next()
            .expect("one ReDimNet input produces one result")
    }

    fn embed_batch(
        &self,
        clips: &[&[f32]],
        sample_rate_hz: u32,
    ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
        self.embed_on_bounded_pool(clips, sample_rate_hz)
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
