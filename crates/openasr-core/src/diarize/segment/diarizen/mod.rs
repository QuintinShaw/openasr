//! Native DiariZen Base-s80 overlap-segmentation runtime.
//!
//! The public primitive deliberately accepts one exact 16-second, 16 kHz
//! window.  Sliding-window aggregation belongs to the diarization pipeline;
//! keeping it out of this low-level runtime makes the model contract explicit
//! and prevents accidental geometry drift.

mod config;
mod pack;
mod runtime;
mod weights;

use std::path::Path;
use std::sync::Mutex;

use thiserror::Error;

use crate::diarize::contract::{SpeakerId, SpeakerTurn, TimeRange};
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphError, GgufTensorDataReader,
    read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path,
};

pub use config::ARCHITECTURE_ID as DIARIZEN_GGML_ARCHITECTURE_ID;
pub use pack::{diarizen_pack_installed, load_diarizen_segmenter, shared_diarizen_segmenter};

use config::{
    FRAME_STRIDE_SAMPLES, LOCAL_SPEAKERS, POWERSET_CLASSES, SAMPLE_RATE_HZ, WINDOW_SAMPLES,
};
use runtime::DiariZenRuntime;
use weights::validate_tensor_contract;

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
    #[error("DiariZen runtime mutex is poisoned")]
    RuntimePoisoned,
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

/// Native, resident DiariZen Base-s80 segmenter.
pub struct DiariZenSegmenter {
    runtime: Mutex<SendableRuntime>,
}

struct SendableRuntime(DiariZenRuntime);

// GGML runtime handles are not thread-affine, but contain Rc-backed owners and
// raw backend pointers. Moving them between threads is safe only while access
// remains exclusive; the enclosing Mutex provides exactly that ownership
// boundary and no handle escapes a locked call.
unsafe impl Send for SendableRuntime {}

impl DiariZenSegmenter {
    pub fn from_oasr(path: &Path) -> Result<Self, DiariZenSegmenterError> {
        Self::from_oasr_with_backend(path, None)
    }

    fn from_runtime_source(
        source: &crate::ggml_runtime::GgmlRuntimeSource,
    ) -> Result<Self, DiariZenSegmenterError> {
        let runtime = DiariZenRuntime::from_runtime_source(source, WINDOW_SAMPLES, false, None)?;
        Ok(Self {
            runtime: Mutex::new(SendableRuntime(runtime)),
        })
    }

    fn from_oasr_with_backend(
        path: &Path,
        backend: Option<GgmlCpuGraphBackend>,
    ) -> Result<Self, DiariZenSegmenterError> {
        let runtime = DiariZenRuntime::new(path, WINDOW_SAMPLES, false, backend)?;
        Ok(Self {
            runtime: Mutex::new(SendableRuntime(runtime)),
        })
    }

    /// Cheap install-time contract probe. This parses metadata and the tensor
    /// index only; it does not materialize weights or construct a compute graph.
    pub fn probe_oasr(path: &Path) -> Result<(), DiariZenSegmenterError> {
        let source = validate_ggml_runtime_source_path(path)
            .map_err(|error| DiariZenSegmenterError::PackSource(error.to_string()))?;
        let metadata = read_gguf_metadata_from_runtime_source(&source)
            .map_err(|error| DiariZenSegmenterError::PackRead(error.to_string()))?;
        config::validate_metadata(&metadata)?;
        let reader = GgufTensorDataReader::from_runtime_source(&source)
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
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| DiariZenSegmenterError::RuntimePoisoned)?;
        runtime.0.infer(samples)
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

    #[cfg(test)]
    fn from_test_geometry(path: &Path, samples: usize) -> Result<Self, DiariZenSegmenterError> {
        let runtime = DiariZenRuntime::new(path, samples, true, Some(GgmlCpuGraphBackend::Cpu))?;
        Ok(Self {
            runtime: Mutex::new(SendableRuntime(runtime)),
        })
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
