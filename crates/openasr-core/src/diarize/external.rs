//! Deep recording-level external diarization module.
//!
//! Callers provide 16 kHz recording audio and receive normalized
//! recording-local turns plus centroids. Model selection, sliding activity,
//! VAD union, embedding windows, automatic clustering, and overlap
//! reconstruction stay local to this implementation.

use std::collections::BTreeMap;
use std::sync::Arc;

use thiserror::Error;

use super::clustering::AutomaticClusterer;
#[cfg(test)]
use super::clustering::{AutomaticClusteringDiagnostics, AutomaticClusteringStrategy};
use super::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, SpeakerTurn, TimeRange};
use super::embed::{EmbedError, SpeakerEmbedder};
use super::pipeline::Diarization;
use super::segment::{
    ActivityFrameClock, LocalActivity, PreparedSelectedSegmenter, SegmentError, SegmenterProvider,
    SegmenterRuntimeInput, SelectedSegmenter,
};
use crate::config::VoiceIdSegmenterPreference;
use crate::ggml_runtime::{GgmlCpuGraphBackend, RequestBackendPreference};

const SAMPLE_RATE_HZ: u32 = 16_000;
const EMBEDDING_WINDOW_S: f64 = 1.5;
const EMBEDDING_STEP_S: f64 = 0.75;

#[derive(Debug, Error)]
pub enum ExternalDiarizationError {
    #[error(transparent)]
    Segmenter(#[from] SegmentError),
    #[error("external Voice ID could not load the vendored FireRed Stream-VAD")]
    VadUnavailable,
    #[error("external Voice ID FireRed VAD failed: {0}")]
    Vad(String),
    #[error("external Voice ID ReDim embedding failed: {0}")]
    Embedding(String),
    #[error("external Voice ID requires 16 kHz mono audio, got {0} Hz")]
    UnsupportedSampleRate(u32),
    #[error("external Voice ID was canceled")]
    Canceled,
    #[error("external Voice ID execution route could not be resolved: {0}")]
    ExecutionRoute(#[from] crate::device::execution_route::ExecutionRouteError),
}

/// Lightweight request plan. It pins the selected provider, pack mappings,
/// actual execution-route candidates, and admission estimate, but owns no
/// parsed/dequantized weights, runner, VAD, or graph.
pub(crate) struct PreparedExternalDiarizer {
    segmenter: PreparedSelectedSegmenter,
}

impl PreparedExternalDiarizer {
    pub(crate) fn prepare(
        preference: VoiceIdSegmenterPreference,
        backend_preference: Option<RequestBackendPreference>,
    ) -> Result<Self, ExternalDiarizationError> {
        let runtime_input = SegmenterRuntimeInput::resolve(backend_preference)?;
        let segmenter = super::segment::prepare_segmenter(preference, runtime_input)?;
        Ok(Self { segmenter })
    }

    pub(crate) fn segmenter_admission_bytes(&self) -> u64 {
        self.segmenter.admission_bytes()
    }

    pub(crate) fn segmenter_admission_backend(&self) -> GgmlCpuGraphBackend {
        self.segmenter.admission_backend()
    }

    pub(crate) fn segmenter_discrete_vram_budget_bytes(&self) -> Option<u64> {
        self.segmenter.discrete_vram_budget_bytes()
    }

    #[cfg(test)]
    pub(crate) fn segmenter_content_id(&self) -> &str {
        self.segmenter.content_id()
    }

    pub(crate) fn materialize(
        self,
        embedder: Arc<dyn SpeakerEmbedder>,
    ) -> Result<ExternalDiarizer, ExternalDiarizationError> {
        let segmenter = self.segmenter.materialize()?;
        let vad = super::vad::FireRedStreamVadProvider::shared()
            .ok_or(ExternalDiarizationError::VadUnavailable)?;
        Ok(ExternalDiarizer {
            segmenter,
            embedder,
            vad,
            clusterer: AutomaticClusterer,
        })
    }
}

/// One preflighted recording-level pipeline. The chosen segmenter adapter is
/// retained for the full request, preventing load/inference fallback after
/// selection.
pub(crate) struct ExternalDiarizer {
    segmenter: SelectedSegmenter,
    embedder: Arc<dyn SpeakerEmbedder>,
    vad: super::vad::FireRedStreamVadProvider,
    clusterer: AutomaticClusterer,
}

struct PreparedExternalRecording {
    activity: LocalActivity,
    embedded_chunks: Vec<TimeRange>,
    embeddings: Vec<SpeakerEmbedding>,
    audio_duration_s: f64,
}

#[cfg(test)]
#[derive(Debug, serde::Serialize)]
struct NativeDiarizationDiagnostics {
    schema: &'static str,
    chunks: Vec<NativeDiarizationDiagnosticChunk>,
    embeddings: Vec<Vec<f32>>,
    clustering: NativeClusteringDiagnostics,
}

#[cfg(test)]
#[derive(Debug, serde::Serialize)]
struct NativeDiarizationDiagnosticChunk {
    start_s: f64,
    end_s: f64,
}

#[cfg(test)]
#[derive(Debug, serde::Serialize)]
struct NativeClusteringDiagnostics {
    strategy: &'static str,
    spectral_eigenvalues: Vec<f64>,
    eigengap_speakers: Option<usize>,
    selected_speakers: usize,
    raw_labels: Vec<u32>,
    minor_filtered_labels: Vec<u32>,
    final_labels: Vec<u32>,
}

#[cfg(test)]
impl NativeDiarizationDiagnostics {
    fn from_pipeline(
        chunks: &[TimeRange],
        embeddings: &[SpeakerEmbedding],
        clustering: AutomaticClusteringDiagnostics,
    ) -> Self {
        assert_eq!(
            chunks.len(),
            embeddings.len(),
            "native diagnostics require one embedding per successful chunk"
        );
        assert_eq!(
            chunks.len(),
            clustering.raw_labels.len(),
            "native diagnostics require one raw label per embedding"
        );
        assert_eq!(
            chunks.len(),
            clustering.minor_filtered_labels.len(),
            "native diagnostics require one filtered label per embedding"
        );
        assert_eq!(
            chunks.len(),
            clustering.final_labels.len(),
            "native diagnostics require one final label per embedding"
        );
        let strategy = match clustering.strategy {
            AutomaticClusteringStrategy::Ahc => "ahc",
            AutomaticClusteringStrategy::Spectral => "spectral",
        };
        Self {
            schema: "openasr.native-diarization-diagnostics.v1",
            chunks: chunks
                .iter()
                .map(|range| NativeDiarizationDiagnosticChunk {
                    start_s: range.start_s,
                    end_s: range.end_s,
                })
                .collect(),
            embeddings: embeddings
                .iter()
                .map(|embedding| embedding.0.clone())
                .collect(),
            clustering: NativeClusteringDiagnostics {
                strategy,
                spectral_eigenvalues: clustering.spectral_eigenvalues,
                eigengap_speakers: clustering.eigengap_speakers,
                selected_speakers: clustering.selected_speakers,
                raw_labels: speaker_label_values(clustering.raw_labels),
                minor_filtered_labels: speaker_label_values(clustering.minor_filtered_labels),
                final_labels: speaker_label_values(clustering.final_labels),
            },
        }
    }
}

#[cfg(test)]
fn speaker_label_values(labels: Vec<SpeakerId>) -> Vec<u32> {
    labels.into_iter().map(|speaker| speaker.0).collect()
}

#[cfg(test)]
fn native_diagnostics_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

impl ExternalDiarizer {
    pub(crate) fn selected_segmenter(&self) -> SegmenterProvider {
        self.segmenter.provider
    }

    pub(crate) fn diarize(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Diarization, ExternalDiarizationError> {
        self.diarize_with_clustering(
            samples,
            sample_rate_hz,
            hint,
            canceled,
            |clusterer, _chunks, embeddings, hint| (clusterer.cluster(embeddings, hint), ()),
        )
        .map(|(diarization, ())| diarization)
    }

    #[cfg(test)]
    fn diarize_with_diagnostics(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
    ) -> Result<(Diarization, NativeDiarizationDiagnostics), ExternalDiarizationError> {
        self.diarize_with_clustering(
            samples,
            sample_rate_hz,
            hint,
            canceled,
            |clusterer, chunks, embeddings, hint| {
                let clustering = clusterer.diagnostics(embeddings, hint);
                let labels = clustering.final_labels.clone();
                (
                    labels,
                    NativeDiarizationDiagnostics::from_pipeline(chunks, embeddings, clustering),
                )
            },
        )
    }

    fn diarize_with_clustering<T>(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
        cluster: impl FnOnce(
            &AutomaticClusterer,
            &[TimeRange],
            &[SpeakerEmbedding],
            DiarizeHint,
        ) -> (Vec<SpeakerId>, T),
    ) -> Result<(Diarization, T), ExternalDiarizationError> {
        let prepared = self.prepare_recording(samples, sample_rate_hz, canceled)?;
        if !prepared.embeddings.is_empty() {
            cancel_checkpoint(canceled)?;
        }
        let (labels, output) = cluster(
            &self.clusterer,
            &prepared.embedded_chunks,
            &prepared.embeddings,
            hint,
        );
        debug_assert_eq!(labels.len(), prepared.embeddings.len());
        Ok((assemble_recording(&prepared, &labels), output))
    }

    fn prepare_recording(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
    ) -> Result<PreparedExternalRecording, ExternalDiarizationError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(ExternalDiarizationError::UnsupportedSampleRate(
                sample_rate_hz,
            ));
        }
        cancel_checkpoint(canceled)?;
        let activity =
            self.segmenter
                .adapter
                .segment_local_activity(samples, sample_rate_hz, canceled)?;
        cancel_checkpoint(canceled)?;
        let vad_regions = self.vad_regions(samples, sample_rate_hz, canceled)?;
        let activity_regions = activity.valid_regions(samples.len() as f64 / sample_rate_hz as f64);
        let speech = union_regions(vad_regions.into_iter().chain(activity_regions));
        let chunks = embedding_chunks(&speech);
        let (embedded_chunks, embeddings) = embed_chunks(
            self.embedder.as_ref(),
            samples,
            sample_rate_hz,
            &chunks,
            canceled,
        )?;
        Ok(PreparedExternalRecording {
            activity,
            embedded_chunks,
            embeddings,
            audio_duration_s: samples.len() as f64 / sample_rate_hz as f64,
        })
    }

    fn vad_regions(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<TimeRange>, ExternalDiarizationError> {
        self.vad
            .compute_speech_slices_cancellable(
                samples,
                sample_rate_hz,
                &crate::LongFormOptions::default(),
                canceled,
            )
            .map(|slices| {
                slices
                    .into_iter()
                    .map(|slice| {
                        TimeRange::new(
                            slice.start_sample as f64 / sample_rate_hz as f64,
                            slice.end_sample as f64 / sample_rate_hz as f64,
                        )
                    })
                    .collect()
            })
            .map_err(|error| match error {
                super::vad::FireRedStreamVadError::Canceled => ExternalDiarizationError::Canceled,
                other => ExternalDiarizationError::Vad(other.to_string()),
            })
    }
}

fn assemble_recording(prepared: &PreparedExternalRecording, labels: &[SpeakerId]) -> Diarization {
    let cluster_segments = compress_cluster_segments(&prepared.embedded_chunks, labels);
    let speaker_count = labels
        .iter()
        .map(|speaker| speaker.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let turns = reconstruct_global_turns(
        &prepared.activity,
        &cluster_segments,
        speaker_count,
        prepared.audio_duration_s,
    );
    let centroids = speaker_centroids(labels, &prepared.embeddings);
    Diarization { turns, centroids }
}

fn cancel_checkpoint(canceled: &dyn Fn() -> bool) -> Result<(), ExternalDiarizationError> {
    if canceled() {
        Err(ExternalDiarizationError::Canceled)
    } else {
        Ok(())
    }
}

fn union_regions(regions: impl IntoIterator<Item = TimeRange>) -> Vec<TimeRange> {
    let mut regions: Vec<_> = regions
        .into_iter()
        .filter(|region| region.duration_s() > 0.0)
        .collect();
    regions.sort_by(|left, right| {
        left.start_s
            .total_cmp(&right.start_s)
            .then_with(|| left.end_s.total_cmp(&right.end_s))
    });
    let mut merged: Vec<TimeRange> = Vec::new();
    for region in regions {
        if let Some(last) = merged.last_mut()
            && region.start_s <= last.end_s
        {
            last.end_s = last.end_s.max(region.end_s);
        } else {
            merged.push(region);
        }
    }
    merged
}

fn embedding_chunks(speech: &[TimeRange]) -> Vec<TimeRange> {
    let mut chunks = Vec::new();
    for region in speech {
        let mut start_s = region.start_s;
        while start_s + EMBEDDING_WINDOW_S < region.end_s + EMBEDDING_STEP_S {
            chunks.push(TimeRange::new(
                start_s,
                (start_s + EMBEDDING_WINDOW_S).min(region.end_s),
            ));
            start_s += EMBEDDING_STEP_S;
        }
    }
    chunks
}

fn embed_chunks(
    embedder: &dyn SpeakerEmbedder,
    samples: &[f32],
    sample_rate_hz: u32,
    chunks: &[TimeRange],
    canceled: &dyn Fn() -> bool,
) -> Result<(Vec<TimeRange>, Vec<SpeakerEmbedding>), ExternalDiarizationError> {
    cancel_checkpoint(canceled)?;
    let raw: Vec<Vec<f32>> = chunks
        .iter()
        .map(|range| {
            let start = (range.start_s * sample_rate_hz as f64).max(0.0) as usize;
            let end = ((range.end_s * sample_rate_hz as f64) as usize).min(samples.len());
            samples[start.min(end)..end].to_vec()
        })
        .collect();
    let target_len = (EMBEDDING_WINDOW_S * sample_rate_hz as f64).round() as usize;
    let padded: Vec<Vec<f32>> = raw
        .iter()
        .map(|chunk| circle_pad(chunk, target_len))
        .collect();
    let borrowed: Vec<&[f32]> = padded.iter().map(Vec::as_slice).collect();
    let results = embedder.embed_batch(&borrowed, sample_rate_hz);
    if results.len() != chunks.len() {
        return Err(ExternalDiarizationError::Embedding(format!(
            "embedder returned {} results for {} diarization windows",
            results.len(),
            chunks.len()
        )));
    }
    cancel_checkpoint(canceled)?;
    let mut successful_chunks = Vec::new();
    let mut embeddings = Vec::new();
    for (range, result) in chunks.iter().copied().zip(results) {
        match result {
            Ok(embedding) => {
                successful_chunks.push(range);
                embeddings.push(embedding);
            }
            Err(EmbedError::TooShort) => {}
            Err(EmbedError::Canceled) => return Err(ExternalDiarizationError::Canceled),
            Err(error) => {
                return Err(ExternalDiarizationError::Embedding(error.to_string()));
            }
        }
    }
    Ok((successful_chunks, embeddings))
}

fn circle_pad(samples: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    (0..target_len)
        .map(|index| samples[index % samples.len()])
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ClusterSegment {
    range: TimeRange,
    speaker: SpeakerId,
}

fn compress_cluster_segments(ranges: &[TimeRange], labels: &[SpeakerId]) -> Vec<ClusterSegment> {
    let mut compressed: Vec<ClusterSegment> = Vec::new();
    for (&range, &speaker) in ranges.iter().zip(labels) {
        if let Some(last) = compressed.last_mut() {
            if speaker == last.speaker {
                if range.start_s <= last.range.end_s {
                    last.range.end_s = range.end_s.max(last.range.end_s);
                    continue;
                }
            } else if range.start_s < last.range.end_s {
                let midpoint = (last.range.end_s + range.start_s) * 0.5;
                last.range.end_s = midpoint;
                compressed.push(ClusterSegment {
                    range: TimeRange::new(midpoint, range.end_s),
                    speaker,
                });
                continue;
            }
        }
        compressed.push(ClusterSegment { range, speaker });
    }
    compressed
}

fn reconstruct_global_turns(
    activity: &LocalActivity,
    clusters: &[ClusterSegment],
    speaker_count: usize,
    audio_duration_s: f64,
) -> Vec<SpeakerTurn> {
    if speaker_count == 0 || activity.speaker_count.is_empty() {
        return Vec::new();
    }
    let frames = activity.speaker_count.len();
    let mut cluster_frames = vec![0u8; frames * speaker_count];
    for cluster in clusters {
        let start = activity
            .frame_clock
            .closest_frame(cluster.range.start_s + activity.frame_clock.duration_s() * 0.5)
            .min(frames);
        let end = activity
            .frame_clock
            .closest_frame(cluster.range.end_s + activity.frame_clock.duration_s() * 0.5)
            .min(frames);
        for frame in start..end {
            cluster_frames[frame * speaker_count + cluster.speaker.0 as usize] = 1;
        }
    }

    let mut activations = vec![0u16; frames * speaker_count];
    for window in &activity.windows {
        let start = activity
            .frame_clock
            .closest_frame_for_window_start(window.start_sample);
        if start >= frames {
            continue;
        }
        let usable = window.frame_activity.len().min(frames - start);
        debug_assert!(activity.local_speaker_slots <= u8::BITS as u8);
        let local_slots = activity.local_speaker_slots.min(u8::BITS as u8) as usize;
        let mut overlap = vec![vec![-1i64; speaker_count]; local_slots];
        for (local, local_scores) in overlap.iter_mut().enumerate() {
            let bit = 1u8 << local;
            let active = window.frame_activity[..usable]
                .iter()
                .any(|mask| mask & bit != 0);
            if !active {
                continue;
            }
            for (speaker, score) in local_scores.iter_mut().enumerate() {
                *score = (0..usable)
                    .filter(|&offset| {
                        window.frame_activity[offset] & bit != 0
                            && cluster_frames[(start + offset) * speaker_count + speaker] != 0
                    })
                    .count() as i64;
            }
        }
        for (local, speaker) in hungarian_maximize(&overlap) {
            if overlap[local][speaker] <= 0 {
                continue;
            }
            let bit = 1u8 << local;
            for (offset, &mask) in window.frame_activity[..usable].iter().enumerate() {
                if mask & bit != 0 {
                    activations[(start + offset) * speaker_count + speaker] =
                        activations[(start + offset) * speaker_count + speaker].saturating_add(1);
                }
            }
        }
    }

    let mut binary = vec![false; frames * speaker_count];
    for (frame, &count) in activity.speaker_count.iter().enumerate() {
        let mut speakers: Vec<usize> = (0..speaker_count).collect();
        speakers.sort_by(|&left, &right| {
            activations[frame * speaker_count + right]
                .cmp(&activations[frame * speaker_count + left])
                .then_with(|| left.cmp(&right))
        });
        for &speaker in speakers.iter().take((count as usize).min(speaker_count)) {
            if activations[frame * speaker_count + speaker] > 0 {
                binary[frame * speaker_count + speaker] = true;
            }
        }
        let selected = (0..speaker_count).any(|speaker| binary[frame * speaker_count + speaker]);
        if !selected {
            for speaker in 0..speaker_count {
                binary[frame * speaker_count + speaker] =
                    cluster_frames[frame * speaker_count + speaker] != 0;
            }
        }
    }
    binary_to_turns(
        &binary,
        speaker_count,
        activity.frame_clock,
        audio_duration_s,
    )
}

fn binary_to_turns(
    binary: &[bool],
    speaker_count: usize,
    clock: ActivityFrameClock,
    audio_duration_s: f64,
) -> Vec<SpeakerTurn> {
    let frames = binary.len() / speaker_count;
    let mut turns = Vec::new();
    for speaker in 0..speaker_count {
        let mut start = None;
        for frame in 0..frames {
            let active = binary[frame * speaker_count + speaker];
            if active && start.is_none() {
                start = Some(frame);
            }
            if !active && let Some(begin) = start.take() {
                turns.push(SpeakerTurn {
                    range: TimeRange::new(
                        clock.midpoint_s(begin),
                        clock.midpoint_s(frame).min(audio_duration_s),
                    ),
                    speaker: SpeakerId(speaker as u32),
                    overlap: false,
                });
            }
        }
        if let Some(begin) = start {
            turns.push(SpeakerTurn {
                range: TimeRange::new(
                    clock.midpoint_s(begin),
                    clock.midpoint_s(frames).min(audio_duration_s),
                ),
                speaker: SpeakerId(speaker as u32),
                overlap: false,
            });
        }
    }
    turns.sort_by(|left, right| {
        left.range
            .start_s
            .total_cmp(&right.range.start_s)
            .then_with(|| left.speaker.cmp(&right.speaker))
    });
    for index in 0..turns.len() {
        turns[index].overlap = turns.iter().enumerate().any(|(other_index, other)| {
            index != other_index
                && turns[index].speaker != other.speaker
                && turns[index].range.overlaps(&other.range)
        });
    }
    turns
}

/// Rectangular Hungarian assignment, maximizing integer overlap counts.
fn hungarian_maximize(scores: &[Vec<i64>]) -> Vec<(usize, usize)> {
    let rows = scores.len();
    let columns = scores.first().map_or(0, Vec::len);
    if rows == 0 || columns == 0 {
        return Vec::new();
    }
    if rows > columns {
        let transposed: Vec<Vec<i64>> = (0..columns)
            .map(|column| (0..rows).map(|row| scores[row][column]).collect())
            .collect();
        return hungarian_maximize(&transposed)
            .into_iter()
            .map(|(column, row)| (row, column))
            .collect();
    }

    let mut u = vec![0i64; rows + 1];
    let mut v = vec![0i64; columns + 1];
    let mut matched_row = vec![0usize; columns + 1];
    let mut way = vec![0usize; columns + 1];
    for row in 1..=rows {
        matched_row[0] = row;
        let mut column0 = 0usize;
        let mut minimum = vec![i64::MAX; columns + 1];
        let mut used = vec![false; columns + 1];
        loop {
            used[column0] = true;
            let row0 = matched_row[column0];
            let mut delta = i64::MAX;
            let mut column1 = 0usize;
            for column in 1..=columns {
                if used[column] {
                    continue;
                }
                let current = -scores[row0 - 1][column - 1] - u[row0] - v[column];
                if current < minimum[column] {
                    minimum[column] = current;
                    way[column] = column0;
                }
                if minimum[column] < delta || (minimum[column] == delta && column < column1) {
                    delta = minimum[column];
                    column1 = column;
                }
            }
            for column in 0..=columns {
                if used[column] {
                    u[matched_row[column]] += delta;
                    v[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            column0 = column1;
            if matched_row[column0] == 0 {
                break;
            }
        }
        loop {
            let column1 = way[column0];
            matched_row[column0] = matched_row[column1];
            column0 = column1;
            if column0 == 0 {
                break;
            }
        }
    }
    let mut assignment: Vec<_> = (1..=columns)
        .filter(|&column| matched_row[column] != 0)
        .map(|column| (matched_row[column] - 1, column - 1))
        .collect();
    assignment.sort_unstable();
    assignment
}

fn speaker_centroids(
    labels: &[SpeakerId],
    embeddings: &[SpeakerEmbedding],
) -> Vec<(SpeakerId, SpeakerEmbedding)> {
    let dimensions = embeddings.first().map_or(0, SpeakerEmbedding::dim);
    let mut sums: BTreeMap<SpeakerId, Vec<f32>> = BTreeMap::new();
    for (&speaker, embedding) in labels.iter().zip(embeddings) {
        let sum = sums.entry(speaker).or_insert_with(|| vec![0.0; dimensions]);
        for (accumulator, &value) in sum.iter_mut().zip(&embedding.0) {
            *accumulator += value;
        }
    }
    sums.into_iter()
        .map(|(speaker, sum)| (speaker, SpeakerEmbedding::l2_normalized(sum)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::segment::LocalActivityWindow;
    use sha2::Digest;

    #[derive(serde::Deserialize)]
    struct NativeDiarizationFixture {
        id: String,
        wav: std::path::PathBuf,
    }

    struct CanceledEmbedder;

    impl SpeakerEmbedder for CanceledEmbedder {
        fn embed(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, EmbedError> {
            Err(EmbedError::Canceled)
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    struct ShortBatchEmbedder;

    impl SpeakerEmbedder for ShortBatchEmbedder {
        fn embed(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, EmbedError> {
            unreachable!("the batch seam is overridden")
        }

        fn embed_batch(
            &self,
            _clips: &[&[f32]],
            _sample_rate_hz: u32,
        ) -> Vec<Result<SpeakerEmbedding, EmbedError>> {
            Vec::new()
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    fn clock() -> ActivityFrameClock {
        ActivityFrameClock::new(0, 2, 1, 10)
    }

    #[test]
    fn firered_and_segmenter_valid_regions_are_unioned() {
        let merged = union_regions([
            TimeRange::new(0.0, 1.0),
            TimeRange::new(0.8, 1.4),
            TimeRange::new(2.0, 3.0),
        ]);
        assert_eq!(
            merged,
            vec![TimeRange::new(0.0, 1.4), TimeRange::new(2.0, 3.0)]
        );
    }

    #[test]
    fn embedding_protocol_is_one_point_five_by_zero_point_seven_five() {
        assert_eq!(
            embedding_chunks(&[TimeRange::new(0.0, 3.0)]),
            vec![
                TimeRange::new(0.0, 1.5),
                TimeRange::new(0.75, 2.25),
                TimeRange::new(1.5, 3.0),
            ]
        );
        assert_eq!(circle_pad(&[1.0, 2.0], 5), vec![1.0, 2.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn cancellation_checkpoint_is_typed() {
        assert!(matches!(
            cancel_checkpoint(&|| true),
            Err(ExternalDiarizationError::Canceled)
        ));
    }

    #[test]
    fn redim_batch_cancel_is_not_stringified() {
        let error = embed_chunks(
            &CanceledEmbedder,
            &vec![0.0; 24_000],
            16_000,
            &[TimeRange::new(0.0, 1.5)],
            &|| false,
        )
        .expect_err("embedding cancellation must stop external diarization");
        assert!(matches!(error, ExternalDiarizationError::Canceled));
    }

    #[test]
    fn malformed_embedder_batch_length_fails_closed() {
        let error = embed_chunks(
            &ShortBatchEmbedder,
            &vec![0.0; 24_000],
            16_000,
            &[TimeRange::new(0.0, 1.5)],
            &|| false,
        )
        .expect_err("a short batch result must not silently drop a window");
        assert!(matches!(
            error,
            ExternalDiarizationError::Embedding(reason)
                if reason.contains("0 results for 1 diarization windows")
        ));
    }

    #[test]
    fn hungarian_alignment_is_maximum_and_deterministic() {
        let scores = vec![vec![8, 1, 0], vec![1, 7, 2], vec![0, 2, 6]];
        assert_eq!(hungarian_maximize(&scores), vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(hungarian_maximize(&scores), hungarian_maximize(&scores));
    }

    #[test]
    fn count_reconstruction_preserves_overlap() {
        let activity = LocalActivity {
            frame_clock: clock(),
            windows: vec![LocalActivityWindow {
                start_sample: 0,
                frame_activity: vec![0b01, 0b11, 0b10, 0],
            }],
            local_speaker_slots: 3,
            speaker_count: vec![1, 2, 1, 0],
        };
        let clusters = vec![
            ClusterSegment {
                range: TimeRange::new(0.0, 0.2),
                speaker: SpeakerId(0),
            },
            ClusterSegment {
                range: TimeRange::new(0.2, 0.4),
                speaker: SpeakerId(1),
            },
        ];
        let turns = reconstruct_global_turns(&activity, &clusters, 2, 0.4);
        assert!(
            turns
                .iter()
                .any(|turn| turn.speaker == SpeakerId(0) && turn.overlap)
        );
        assert!(
            turns
                .iter()
                .any(|turn| turn.speaker == SpeakerId(1) && turn.overlap)
        );
    }

    #[test]
    fn reconstruction_keeps_a_fourth_local_speaker_slot() {
        let activity = LocalActivity {
            frame_clock: clock(),
            windows: vec![LocalActivityWindow {
                start_sample: 0,
                frame_activity: vec![0b0001, 0b0010, 0b0100, 0b1000],
            }],
            local_speaker_slots: 4,
            speaker_count: vec![1, 1, 1, 1],
        };
        let clusters = (0..4)
            .map(|speaker| ClusterSegment {
                range: TimeRange::new(speaker as f64 * 0.1, (speaker + 1) as f64 * 0.1),
                speaker: SpeakerId(speaker),
            })
            .collect::<Vec<_>>();

        let turns = reconstruct_global_turns(&activity, &clusters, 4, 0.4);

        assert!(
            turns.iter().any(|turn| turn.speaker == SpeakerId(3)),
            "the fourth DiariZen-local slot must survive Hungarian alignment"
        );
    }

    #[test]
    fn native_diagnostics_require_explicit_one() {
        assert!(!native_diagnostics_enabled(None));
        assert!(!native_diagnostics_enabled(Some("")));
        assert!(!native_diagnostics_enabled(Some("0")));
        assert!(!native_diagnostics_enabled(Some("true")));
        assert!(native_diagnostics_enabled(Some("1")));
    }

    #[test]
    fn native_diagnostics_serialize_exact_pipeline_artifacts() {
        let chunks = vec![TimeRange::new(0.0, 1.5), TimeRange::new(0.75, 2.25)];
        let embeddings = vec![
            SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            SpeakerEmbedding::l2_normalized(vec![0.0, 1.0]),
        ];
        let clustering = AutomaticClusterer.diagnostics(&embeddings, DiarizeHint::NumSpeakers(2));
        let expected_raw: Vec<_> = clustering
            .raw_labels
            .iter()
            .map(|speaker| speaker.0)
            .collect();
        let expected_minor: Vec<_> = clustering
            .minor_filtered_labels
            .iter()
            .map(|speaker| speaker.0)
            .collect();
        let expected_final: Vec<_> = clustering
            .final_labels
            .iter()
            .map(|speaker| speaker.0)
            .collect();

        let value = serde_json::to_value(NativeDiarizationDiagnostics::from_pipeline(
            &chunks,
            &embeddings,
            clustering,
        ))
        .expect("serialize native diagnostics fixture");

        assert_eq!(value["schema"], "openasr.native-diarization-diagnostics.v1");
        assert_eq!(
            value["chunks"],
            serde_json::json!([
                {"start_s": 0.0, "end_s": 1.5},
                {"start_s": 0.75, "end_s": 2.25}
            ])
        );
        assert_eq!(
            value["embeddings"],
            serde_json::json!([[1.0, 0.0], [0.0, 1.0]])
        );
        assert_eq!(value["clustering"]["strategy"], "spectral");
        assert_eq!(
            value["clustering"]["spectral_eigenvalues"]
                .as_array()
                .map(Vec::len),
            Some(2)
        );
        assert!(value["clustering"]["eigengap_speakers"].is_null());
        assert_eq!(value["clustering"]["selected_speakers"], 2);
        assert_eq!(
            value["clustering"]["raw_labels"],
            serde_json::to_value(expected_raw).expect("serialize expected raw labels")
        );
        assert_eq!(
            value["clustering"]["minor_filtered_labels"],
            serde_json::to_value(expected_minor).expect("serialize expected filtered labels")
        );
        assert_eq!(
            value["clustering"]["final_labels"],
            serde_json::to_value(expected_final).expect("serialize expected final labels")
        );
    }

    /// Exact, ASR-independent hypothesis emitter for the locked diarization
    /// corpus. Unlike `OPENASR_DIARIZE_DEBUG`, this runs the same production
    /// recording-level module directly and writes full-precision raw turns.
    /// Corpus scoring and the DER threshold are a separate gate: naming this
    /// an emitter prevents a successful inference run from being mistaken for
    /// quality acceptance. The caller owns fixture/output paths so every large
    /// or private asset can stay under one disposable research root.
    #[test]
    #[ignore = "requires OPENASR_NATIVE_DIARIZATION_FIXTURES/OUTPUT plus local model packs"]
    fn native_locked_fixture_manifest_emits_exact_rttm() {
        use std::fmt::Write as _;

        let manifest = std::env::var_os("OPENASR_NATIVE_DIARIZATION_FIXTURES")
            .map(std::path::PathBuf::from)
            .expect("OPENASR_NATIVE_DIARIZATION_FIXTURES must point to fixtures.json");
        let output = std::env::var_os("OPENASR_NATIVE_DIARIZATION_OUTPUT")
            .map(std::path::PathBuf::from)
            .expect("OPENASR_NATIVE_DIARIZATION_OUTPUT must name a disposable run directory");
        let provider = std::env::var("OPENASR_NATIVE_DIARIZATION_PROVIDER")
            .expect("OPENASR_NATIVE_DIARIZATION_PROVIDER must be segmentation_3_0 or diarizen");
        let core_revision = std::env::var("OPENASR_NATIVE_DIARIZATION_CORE_REV")
            .expect("OPENASR_NATIVE_DIARIZATION_CORE_REV must pin the tested commit");
        let segmenter_quant = std::env::var("OPENASR_NATIVE_DIARIZATION_SEGMENTER_QUANT")
            .expect("OPENASR_NATIVE_DIARIZATION_SEGMENTER_QUANT must state the tested pack tier");
        let embedder_quant = std::env::var("OPENASR_NATIVE_DIARIZATION_EMBEDDER_QUANT")
            .expect("OPENASR_NATIVE_DIARIZATION_EMBEDDER_QUANT must state the tested pack tier");
        let diagnostics_env = std::env::var("OPENASR_NATIVE_DIARIZATION_DIAGNOSTICS").ok();
        let emit_diagnostics = native_diagnostics_enabled(diagnostics_env.as_deref());
        let (preference, expected_provider) = match provider.as_str() {
            "segmentation_3_0" => (
                VoiceIdSegmenterPreference::Segmentation3_0,
                SegmenterProvider::Segmentation3_0,
            ),
            "diarizen" => (
                VoiceIdSegmenterPreference::Auto,
                SegmenterProvider::DiariZen,
            ),
            other => panic!("unsupported native diarization provider '{other}'"),
        };
        let backend = std::env::var("OPENASR_NATIVE_DIARIZATION_BACKEND")
            .unwrap_or_else(|_| "cpu".to_string());
        let backend_preference = match backend.as_str() {
            "cpu" => Some(RequestBackendPreference::CpuOnly),
            "accelerated" => Some(RequestBackendPreference::Accelerated),
            other => panic!("unsupported native diarization backend '{other}'"),
        };
        let expected_backend = SegmenterRuntimeInput::resolve(backend_preference.clone())
            .expect("resolve exact acceptance backend")
            .backend();
        let manifest_root = manifest
            .parent()
            .and_then(std::path::Path::parent)
            .expect("fixtures.json must live under <research-root>/scripts");
        let fixtures: Vec<NativeDiarizationFixture> = serde_json::from_slice(
            &std::fs::read(&manifest).expect("read native diarization fixture manifest"),
        )
        .expect("parse native diarization fixture manifest");

        let runtime_owner = crate::NativeRuntimeShutdownGuard::new();
        let _backend_guard =
            crate::ggml_runtime::install_request_backend_override(backend_preference.clone());
        let embedder_plan = crate::diarize::embed::prepare_shared_embedder_snapshot()
            .expect("OPENASR_REDIMNET_PACK must resolve to a valid ReDimNet2-B6 pack");
        let diarizer_plan = PreparedExternalDiarizer::prepare(preference, backend_preference)
            .expect("prepare native external diarizer");
        let embedder_content_id = embedder_plan.content_id().to_string();
        let segmenter_content_id = diarizer_plan.segmenter_content_id().to_string();
        assert_eq!(
            diarizer_plan.segmenter_admission_backend(),
            expected_backend,
            "the exact acceptance harness must use the requested backend"
        );
        let embedder_bytes = embedder_plan.admission_bytes();
        let segmenter_bytes = diarizer_plan.segmenter_admission_bytes();
        if let Some(total_memory) = crate::host::host_total_memory_bytes() {
            crate::capacity::evaluate_static_host_memory_admission(
                0,
                embedder_bytes.saturating_add(if expected_backend != GgmlCpuGraphBackend::Gpu {
                    segmenter_bytes
                } else {
                    0
                }),
                total_memory,
                crate::capacity::MemoryAdmissionDomain::UnifiedMemory {
                    swap_bytes: crate::host::host_total_swap_bytes().unwrap_or(0),
                },
            )
            .expect("native diarization fixture run must pass production-shaped admission");
        }
        if expected_backend == GgmlCpuGraphBackend::Gpu {
            let budget = diarizer_plan
                .segmenter_discrete_vram_budget_bytes()
                .filter(|budget| *budget > 0)
                .expect("the exact GPU route must report a VRAM admission budget");
            crate::capacity::evaluate_static_host_memory_admission(
                0,
                segmenter_bytes,
                0,
                crate::capacity::MemoryAdmissionDomain::DiscreteVram {
                    budget_bytes: budget,
                },
            )
            .expect("native diarization GPU fixture run must pass VRAM admission");
        }
        let embedder = embedder_plan
            .materialize()
            .expect("materialize exact ReDimNet snapshot after admission");
        let diarizer = diarizer_plan
            .materialize(embedder)
            .expect("materialize exact segmentation snapshot after admission");
        assert_eq!(diarizer.selected_segmenter(), expected_provider);

        std::fs::create_dir_all(&output).expect("create native diarization output directory");
        let manifest_sha256 = format!(
            "{:x}",
            sha2::Sha256::digest(
                std::fs::read(&manifest).expect("read fixture manifest for provenance")
            )
        );
        let provenance = serde_json::json!({
            "schema": "openasr.native-diarization-emitter.v1",
            "core_revision": core_revision,
            "fixture_manifest_sha256": manifest_sha256,
            "provider": provider,
            "segmenter_content_id": segmenter_content_id,
            "segmenter_quant": segmenter_quant,
            "embedder": "redimnet2-b6-cn",
            "embedder_content_id": embedder_content_id,
            "embedder_quant": embedder_quant,
            "requested_backend": backend,
            "resolved_backend": format!("{expected_backend:?}").to_ascii_lowercase(),
            "overlap_output": "raw-turns-preserved",
        });
        std::fs::write(
            output.join("provenance.json"),
            serde_json::to_vec_pretty(&provenance).expect("serialize native run provenance"),
        )
        .expect("write native run provenance");
        for fixture in fixtures {
            eprintln!(
                "NATIVE_DIARIZATION_FIXTURE provider={provider} backend={backend} id={} stage=start",
                fixture.id
            );
            let wav = manifest_root.join(&fixture.wav);
            let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                &wav,
                "native diarization acceptance",
                "native diarization acceptance",
            )
            .unwrap_or_else(|error| panic!("load fixture '{}': {error}", wav.display()));
            let (diarization, diagnostics) = if emit_diagnostics {
                let (diarization, diagnostics) = diarizer
                    .diarize_with_diagnostics(&samples, SAMPLE_RATE_HZ, DiarizeHint::Auto, &|| {
                        false
                    })
                    .unwrap_or_else(|error| panic!("diarize fixture '{}': {error}", fixture.id));
                (diarization, Some(diagnostics))
            } else {
                let diarization = diarizer
                    .diarize(&samples, SAMPLE_RATE_HZ, DiarizeHint::Auto, &|| false)
                    .unwrap_or_else(|error| panic!("diarize fixture '{}': {error}", fixture.id));
                (diarization, None)
            };
            assert!(
                !diarization.turns.is_empty(),
                "native diarization emitter produced no turns for '{}'",
                fixture.id
            );
            if let Some(diagnostics) = diagnostics {
                std::fs::write(
                    output.join(format!("{}.diagnostics.json", fixture.id)),
                    serde_json::to_vec_pretty(&diagnostics)
                        .expect("serialize native diarization diagnostics"),
                )
                .expect("write native diarization diagnostics");
            }
            let mut rttm = String::new();
            for turn in diarization.turns {
                let duration = turn.range.duration_s();
                if duration <= 0.0 {
                    continue;
                }
                writeln!(
                    rttm,
                    "SPEAKER {} 1 {:.9} {:.9} <NA> <NA> {} <NA> <NA>",
                    fixture.id,
                    turn.range.start_s,
                    duration,
                    turn.speaker.label()
                )
                .expect("write RTTM line");
            }
            std::fs::write(output.join(format!("{}.rttm", fixture.id)), rttm)
                .expect("write native diarization RTTM");
            eprintln!(
                "NATIVE_DIARIZATION_FIXTURE provider={provider} backend={backend} id={} stage=done",
                fixture.id
            );
        }
        drop(diarizer);
        drop(runtime_owner);
        drop(_backend_guard);
    }
}
