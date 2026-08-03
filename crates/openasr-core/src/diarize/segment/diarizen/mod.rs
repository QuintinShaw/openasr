//! Native DiariZen Large-s80-v2 overlap-segmentation runtime.
//!
//! The public primitive deliberately accepts one exact 16-second, 16 kHz
//! window.  Sliding-window aggregation belongs to the diarization pipeline;
//! keeping it out of this low-level runtime makes the model contract explicit
//! and prevents accidental geometry drift.

mod config;
mod pack;
mod runtime;
mod weights;

use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;

use thiserror::Error;

use crate::diarize::contract::{SpeakerId, SpeakerTurn, TimeRange};
use crate::ggml_runtime::{
    GgmlCpuGraphError, GgufTensorDataReader, read_gguf_metadata_from_runtime_source,
    validate_ggml_runtime_source_path,
};
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DedicatedWorkerRuntimeOwnerLease, DedicatedWorkerRuntimeOwnerTracker,
    PackContentKey, with_thread_local_cached_mut_by_key,
};

pub use config::ARCHITECTURE_ID as DIARIZEN_GGML_ARCHITECTURE_ID;
pub const DIARIZEN_MODEL_ID: &str = config::MODEL_ID;
pub(crate) use pack::{
    PreparedDiariZenSegmenter, prepare_diarizen_segmenter_snapshot, unload_idle_diarizen_cache,
};
pub use pack::{diarizen_pack_installed, load_diarizen_segmenter, shared_diarizen_segmenter};

pub(crate) const DIARIZEN_SAMPLE_RATE_HZ: u32 = config::SAMPLE_RATE_HZ;
pub(crate) const DIARIZEN_WINDOW_SAMPLES: usize = config::WINDOW_SAMPLES;
pub(crate) const DIARIZEN_WINDOW_STEP_SAMPLES: usize = config::WINDOW_STEP_SAMPLES;
pub(crate) const DIARIZEN_FRAME_DURATION_SAMPLES: u32 = 400;
pub(crate) const DIARIZEN_FRAME_STEP_SAMPLES: u32 = config::FRAME_STRIDE_SAMPLES as u32;
pub(crate) const DIARIZEN_LOCAL_SPEAKERS: usize = config::LOCAL_SPEAKERS;

use config::{
    FRAME_STRIDE_SAMPLES, LOCAL_SPEAKERS, POWERSET_CLASSES, SAMPLE_RATE_HZ, WINDOW_SAMPLES,
};
use runtime::DiariZenRuntime;
use weights::validate_tensor_contract;

const DIARIZEN_RUNTIME_CACHE_CAPACITY: usize = 1;
static DIARIZEN_WORKER_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
static DIARIZEN_RUNTIME_OWNERS: DedicatedWorkerRuntimeOwnerTracker =
    DedicatedWorkerRuntimeOwnerTracker::new(unload_idle_worker_runtimes);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiariZenWorkerRuntimeKey {
    pack: PackContentKey,
    execution: super::SegmenterExecutionKey,
}

thread_local! {
    static DIARIZEN_RUNTIME_BY_KEY: RefCell<
        BoundedRuntimeCache<DiariZenWorkerRuntimeKey, DiariZenRuntime>
    > = RefCell::new(BoundedRuntimeCache::new());
}

fn diarizen_worker_pool() -> &'static Result<rayon::ThreadPool, String> {
    DIARIZEN_WORKER_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "openasr-diarizen-0".to_string())
            .build()
            .map_err(|error| error.to_string())
    })
}

fn clear_current_thread_runtime_cache() {
    DIARIZEN_RUNTIME_BY_KEY.with(|cache| cache.borrow_mut().clear_for_idle_unload());
}

pub(crate) fn unload_idle_worker_runtimes() {
    if let Some(Ok(pool)) = DIARIZEN_WORKER_POOL.get() {
        pool.broadcast(|_| clear_current_thread_runtime_cache());
    }
}

#[cfg(test)]
fn diarizen_worker_runtime_entry_count() -> usize {
    let Some(Ok(pool)) = DIARIZEN_WORKER_POOL.get() else {
        return 0;
    };
    pool.broadcast(|_| DIARIZEN_RUNTIME_BY_KEY.with(|cache| cache.borrow().len()))
        .into_iter()
        .sum()
}

#[cfg(test)]
fn install_worker_graph_compute_abort() {
    let pool = diarizen_worker_pool()
        .as_ref()
        .expect("DiariZen test worker pool");
    pool.broadcast(|_| crate::ggml_runtime::install_test_graph_compute_abort());
}

#[cfg(test)]
fn install_worker_graph_compute_device_lost() {
    let pool = diarizen_worker_pool()
        .as_ref()
        .expect("DiariZen test worker pool");
    pool.broadcast(|_| crate::ggml_runtime::install_test_graph_compute_device_lost());
}

#[cfg(test)]
fn diarizen_runtime_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug, Error)]
pub enum DiariZenSegmenterError {
    #[error("could not open DiariZen pack: {0}")]
    PackSource(String),
    #[error("could not read DiariZen pack: {0}")]
    PackRead(String),
    #[error("DiariZen pack is missing metadata key '{key}'")]
    MissingMetadata { key: &'static str },
    #[error("DiariZen metadata '{key}' mismatch: expected {expected}, got {actual}")]
    MetadataMismatch {
        key: &'static str,
        expected: String,
        actual: String,
    },
    #[error("DiariZen metadata '{key}' contains invalid JSON: {source}")]
    InvalidMetadataJson {
        key: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("DiariZen pack is missing tensor '{0}'")]
    MissingTensor(String),
    #[error(
        "DiariZen tensor set mismatch: expected {expected}, got {actual}; missing=[{missing}]; unexpected=[{unexpected}]"
    )]
    TensorSetMismatch {
        expected: usize,
        actual: usize,
        missing: String,
        unexpected: String,
    },
    #[error("DiariZen tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    TensorShapeMismatch {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("DiariZen tensor '{name}' has unsupported type '{tensor_type}'")]
    UnsupportedTensorType { name: String, tensor_type: String },
    #[error("DiariZen requires 16000 Hz audio, got {actual} Hz")]
    UnsupportedSampleRate { actual: u32 },
    #[error("DiariZen requires exactly {expected} samples per window, got {actual}")]
    WindowSize { expected: usize, actual: usize },
    #[error("DiariZen worker pool could not be created: {0}")]
    WorkerPool(String),
    #[error("DiariZen execution route could not be resolved: {0}")]
    ExecutionRoute(#[from] crate::device::execution_route::ExecutionRouteError),
    #[error("DiariZen graph step '{step}' failed: {source}")]
    Graph {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
}

impl DiariZenSegmenterError {
    fn graph(step: &'static str, source: GgmlCpuGraphError) -> Self {
        Self::Graph { step, source }
    }

    pub fn is_canceled(&self) -> bool {
        matches!(
            self,
            Self::Graph {
                source: GgmlCpuGraphError::Aborted | GgmlCpuGraphError::Canceled,
                ..
            }
        )
    }

    fn is_terminal_backend_failure(&self) -> bool {
        matches!(
            self,
            Self::Graph {
                source: GgmlCpuGraphError::DeviceLost | GgmlCpuGraphError::BackendPoisoned,
                ..
            }
        )
    }

    fn is_route_initialization_failure(&self) -> bool {
        matches!(
            self,
            Self::Graph {
                source: GgmlCpuGraphError::ExecutionRoute(_),
                ..
            }
        )
    }
}

/// One model-window result. Logits are frame-major (`[frame, class]`) and
/// activity is frame-major (`[frame, local_speaker]`) after powerset argmax
/// and the checkpoint's pinned 11-frame median filter.
#[derive(Debug, Clone, PartialEq)]
pub struct DiariZenWindowOutput {
    pub frame_count: usize,
    pub logits: Vec<f32>,
    pub powerset_class: Vec<u8>,
    pub activity: Vec<u8>,
}

/// Native DiariZen Large-s80-v2 adapter. Only the immutable mapped source and a
/// request-resolved execution plan cross threads. Native runners remain in the
/// dedicated worker's TLS, alongside the backend cache that owns their raw
/// handles; no `unsafe Send` bridge is needed.
pub struct DiariZenSegmenter {
    // Drop first so worker-owned device runtimes are gone before their mapped
    // source and execution metadata. The active product cache keeps this lease
    // alive across requests; standalone users release on their final drop.
    _worker_runtime_owner: DedicatedWorkerRuntimeOwnerLease,
    source: crate::ggml_runtime::GgmlRuntimeSource,
    pack_key: PackContentKey,
    runtime_input: super::SegmenterRuntimeInput,
}

impl DiariZenSegmenter {
    pub fn from_oasr(path: &Path) -> Result<Self, DiariZenSegmenterError> {
        let source = validate_ggml_runtime_source_path(path)
            .map_err(|error| DiariZenSegmenterError::PackSource(error.to_string()))?;
        let runtime_input =
            super::SegmenterRuntimeInput::resolve(crate::ggml_runtime::request_backend_override())?;
        Self::from_runtime_source(&source, runtime_input)
    }

    pub(super) fn from_runtime_source(
        source: &crate::ggml_runtime::GgmlRuntimeSource,
        runtime_input: super::SegmenterRuntimeInput,
    ) -> Result<Self, DiariZenSegmenterError> {
        Self::probe_runtime_source(source)?;
        Ok(Self {
            _worker_runtime_owner: DIARIZEN_RUNTIME_OWNERS.acquire(),
            source: source.clone(),
            pack_key: PackContentKey::for_runtime_source(source),
            runtime_input,
        })
    }

    /// Cheap install-time contract probe. This parses metadata and the tensor
    /// index only; it does not materialize weights or construct a compute graph.
    pub fn probe_oasr(path: &Path) -> Result<(), DiariZenSegmenterError> {
        let source = validate_ggml_runtime_source_path(path)
            .map_err(|error| DiariZenSegmenterError::PackSource(error.to_string()))?;
        Self::probe_runtime_source(&source)
    }

    pub(super) fn probe_runtime_source(
        source: &crate::ggml_runtime::GgmlRuntimeSource,
    ) -> Result<(), DiariZenSegmenterError> {
        let metadata = read_gguf_metadata_from_runtime_source(source)
            .map_err(|error| DiariZenSegmenterError::PackRead(error.to_string()))?;
        config::validate_metadata(&metadata)?;
        let reader = GgufTensorDataReader::from_runtime_source(source)
            .map_err(|error| DiariZenSegmenterError::PackRead(error.to_string()))?;
        validate_tensor_contract(reader.tensor_index())
    }

    pub fn infer_window(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<DiariZenWindowOutput, DiariZenSegmenterError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(DiariZenSegmenterError::UnsupportedSampleRate {
                actual: sample_rate_hz,
            });
        }
        if samples.len() != WINDOW_SAMPLES {
            return Err(DiariZenSegmenterError::WindowSize {
                expected: WINDOW_SAMPLES,
                actual: samples.len(),
            });
        }
        let pool = diarizen_worker_pool()
            .as_ref()
            .map_err(|error| DiariZenSegmenterError::WorkerPool(error.clone()))?;
        let inherited_cancel = crate::ggml_runtime::thread_job_cancel_flag();
        pool.install(|| {
            let _cancel_guard = inherited_cancel
                .as_ref()
                .map(crate::ggml_runtime::InheritedJobCancelGuard::arm);
            self.infer_window_on_current_worker(samples)
        })
    }

    fn infer_window_on_current_worker(
        &self,
        samples: &[f32],
    ) -> Result<DiariZenWindowOutput, DiariZenSegmenterError> {
        let mut last_route_error = None;
        for candidate in self.runtime_input.candidates() {
            let _backend_guard = crate::ggml_runtime::install_request_backend_override(
                candidate.backend_preference.clone(),
            );
            let key = DiariZenWorkerRuntimeKey {
                pack: self.pack_key.clone(),
                execution: candidate.key.clone(),
            };
            let result = with_thread_local_cached_mut_by_key(
                &DIARIZEN_RUNTIME_BY_KEY,
                key,
                DIARIZEN_RUNTIME_CACHE_CAPACITY,
                || {
                    DiariZenRuntime::from_runtime_source(
                        &self.source,
                        WINDOW_SAMPLES,
                        false,
                        Some(self.runtime_input.backend()),
                    )
                },
                |runtime| runtime.infer(samples),
            );
            match result {
                Err(error) if error.is_terminal_backend_failure() => {
                    clear_current_thread_runtime_cache();
                    return Err(error);
                }
                Err(error) if error.is_route_initialization_failure() => {
                    last_route_error = Some(error);
                }
                other => return other,
            }
        }
        Err(last_route_error.unwrap_or({
            DiariZenSegmenterError::ExecutionRoute(
                crate::device::execution_route::ExecutionRouteError::AcceleratedUnavailable,
            )
        }))
    }

    /// Decode the exact-window output into window-local overlap-aware turns.
    pub fn segment(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<Vec<SpeakerTurn>, DiariZenSegmenterError> {
        let output = self.infer_window(samples, sample_rate_hz)?;
        Ok(decode_segments(&output.activity, output.frame_count))
    }
}

fn decode_segments(activity: &[u8], frames: usize) -> Vec<SpeakerTurn> {
    debug_assert_eq!(activity.len(), frames * LOCAL_SPEAKERS);
    let time = |frame: usize| frame as f64 * FRAME_STRIDE_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let mut turns = Vec::new();
    for speaker in 0..LOCAL_SPEAKERS {
        let mut start = None;
        let mut overlap = false;
        for frame in 0..frames {
            let row = &activity[frame * LOCAL_SPEAKERS..(frame + 1) * LOCAL_SPEAKERS];
            if row[speaker] != 0 {
                start.get_or_insert(frame);
                overlap |= row.iter().filter(|&&value| value != 0).count() > 1;
            } else if let Some(begin) = start.take() {
                turns.push(SpeakerTurn {
                    range: TimeRange::new(time(begin), time(frame)),
                    speaker: SpeakerId(speaker as u32),
                    overlap,
                });
                overlap = false;
            }
        }
        if let Some(begin) = start {
            turns.push(SpeakerTurn {
                range: TimeRange::new(time(begin), time(frames)),
                speaker: SpeakerId(speaker as u32),
                overlap,
            });
        }
    }
    turns
}

const POWERSET: [[u8; LOCAL_SPEAKERS]; POWERSET_CLASSES] = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
];

fn postprocess_logits(logits: &[f32], frames: usize) -> (Vec<u8>, Vec<u8>) {
    let classes = logits
        .chunks_exact(POWERSET_CLASSES)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index as u8)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    debug_assert_eq!(classes.len(), frames);
    let mut raw = vec![0_u8; frames * LOCAL_SPEAKERS];
    for (frame, &class) in classes.iter().enumerate() {
        raw[frame * LOCAL_SPEAKERS..(frame + 1) * LOCAL_SPEAKERS]
            .copy_from_slice(&POWERSET[class as usize]);
    }
    (classes, median_filter_activity(&raw, frames))
}

fn median_filter_activity(input: &[u8], frames: usize) -> Vec<u8> {
    let radius = config::MEDIAN_FILTER_FRAMES / 2;
    let reflect = |index: isize| -> usize {
        let frames = frames as isize;
        let mut index = index;
        while index < 0 || index >= frames {
            index = if index < 0 {
                -index - 1
            } else {
                2 * frames - index - 1
            };
        }
        index as usize
    };
    let mut output = vec![0_u8; input.len()];
    for frame in 0..frames {
        for speaker in 0..LOCAL_SPEAKERS {
            let active = (-(radius as isize)..=radius as isize)
                .filter(|offset| {
                    input[reflect(frame as isize + offset) * LOCAL_SPEAKERS + speaker] != 0
                })
                .count();
            output[frame * LOCAL_SPEAKERS + speaker] =
                u8::from(active > config::MEDIAN_FILTER_FRAMES / 2);
        }
    }
    output
}

#[cfg(test)]
mod tests;
