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

use std::path::Path;

use thiserror::Error;

#[cfg(test)]
use crate::diarize::contract::{SpeakerId, SpeakerTurn, TimeRange};
use crate::ggml_runtime::{GgmlCpuGraphError, GgufMetadata, GgufTensorIndex};
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_verifier::{PackCandidate, PackRoute, PackVerificationError, PackVerifier, VerifiedPack},
};

pub use config::ARCHITECTURE_ID as DIARIZEN_GGML_ARCHITECTURE_ID;
pub const DIARIZEN_MODEL_ID: &str = config::MODEL_ID;
#[cfg(test)]
pub(crate) use pack::DIARIZEN_PACK_PREFERENCE;
pub use pack::diarizen_pack_installed;
pub(crate) use pack::diarizen_pack_path;

pub(crate) const DIARIZEN_SAMPLE_RATE_HZ: u32 = config::SAMPLE_RATE_HZ;
pub(crate) const DIARIZEN_WINDOW_SAMPLES: usize = config::WINDOW_SAMPLES;
pub(crate) const DIARIZEN_WINDOW_STEP_SAMPLES: usize = config::WINDOW_STEP_SAMPLES;
pub(crate) const DIARIZEN_FRAME_DURATION_SAMPLES: u32 = 400;
pub(crate) const DIARIZEN_FRAME_STEP_SAMPLES: u32 = config::FRAME_STRIDE_SAMPLES as u32;
pub(crate) const DIARIZEN_LOCAL_SPEAKERS: usize = config::LOCAL_SPEAKERS;
pub(crate) const DIARIZEN_POWERSET_CLASSES: usize = config::POWERSET_CLASSES;

#[cfg(test)]
use config::{FRAME_STRIDE_SAMPLES, SAMPLE_RATE_HZ};
use config::{LOCAL_SPEAKERS, POWERSET_CLASSES};
pub(crate) use runtime::DiariZenRuntime;
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
    #[error("DiariZen system-memory capacity failed: {0}")]
    Capacity(String),
    #[error("DiariZen owner-thread runtime failed: {0}")]
    Actor(String),
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

/// Stateless pack-contract facade. Execution runtimes are deliberately not
/// constructible here: production materialization must pass through the
/// injected policy-owned actor in `segment::policy_runtime`.
pub struct DiariZenSegmenter;

impl DiariZenSegmenter {
    /// Cheap install-time contract probe. This parses metadata and the tensor
    /// index only; it does not materialize weights or construct a compute graph.
    pub fn probe_oasr(path: &Path) -> Result<(), DiariZenSegmenterError> {
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(path))
            .map_err(map_pack_verification_error)?;
        ensure_diarization_pack_route(&verified_pack)
    }

    pub(crate) fn probe_preflight_parts(
        metadata: &GgufMetadata,
        tensor_index: &GgufTensorIndex,
    ) -> Result<(), DiariZenSegmenterError> {
        config::validate_metadata(metadata)?;
        validate_tensor_contract(tensor_index)
    }
}

fn map_pack_verification_error(error: PackVerificationError) -> DiariZenSegmenterError {
    match error {
        PackVerificationError::RuntimeSource { source, .. } => {
            DiariZenSegmenterError::PackSource(source.to_string())
        }
        other => DiariZenSegmenterError::PackRead(other.to_string()),
    }
}

fn ensure_diarization_pack_route(
    verified_pack: &VerifiedPack,
) -> Result<(), DiariZenSegmenterError> {
    if matches!(
        verified_pack.route(),
        PackRoute::Aux {
            kind: AuxPackKind::Diarization,
            ..
        }
    ) {
        return Ok(());
    }
    Err(DiariZenSegmenterError::PackRead(format!(
        "pack route is not auxiliary diarization: {:?}",
        verified_pack.route()
    )))
}

#[cfg(test)]
fn decode_segments(activity: &[u8], frames: usize) -> Vec<SpeakerTurn> {
    debug_assert_eq!(activity.len(), frames * LOCAL_SPEAKERS);
    let time = |frame: usize| frame as f64 * FRAME_STRIDE_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;
    let mut turns = Vec::new();
    for speaker in 0..LOCAL_SPEAKERS {
        let mut active_run: Option<(usize, bool)> = None;
        for frame in 0..frames {
            let row = &activity[frame * LOCAL_SPEAKERS..(frame + 1) * LOCAL_SPEAKERS];
            let speaker_active = row[speaker] != 0;
            let overlap = speaker_active && row.iter().filter(|&&value| value != 0).count() > 1;
            match active_run {
                None if speaker_active => active_run = Some((frame, overlap)),
                Some((begin, run_overlap)) if !speaker_active || overlap != run_overlap => {
                    turns.push(SpeakerTurn {
                        range: TimeRange::new(time(begin), time(frame)),
                        speaker: SpeakerId(speaker as u32),
                        overlap: run_overlap,
                    });
                    active_run = speaker_active.then_some((frame, overlap));
                }
                _ => {}
            }
        }
        if let Some((begin, overlap)) = active_run {
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
