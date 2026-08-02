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
use super::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, SpeakerTurn, TimeRange};
use super::embed::{EmbedError, SpeakerEmbedder};
use super::pipeline::Diarization;
use super::segment::{ActivityFrameClock, LocalActivity, SegmentError, SelectedSegmenter};
use crate::config::VoiceIdSegmenterPreference;
use crate::longform::LongFormVadProvider;

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

impl ExternalDiarizer {
    pub(crate) fn preflight(
        preference: VoiceIdSegmenterPreference,
        embedder: Arc<dyn SpeakerEmbedder>,
    ) -> Result<Self, ExternalDiarizationError> {
        let segmenter = super::segment::resolve_segmenter(preference)?;
        let vad = super::vad::FireRedStreamVadProvider::shared()
            .ok_or(ExternalDiarizationError::VadUnavailable)?;
        Ok(Self {
            segmenter,
            embedder,
            vad,
            clusterer: AutomaticClusterer,
        })
    }

    pub(crate) fn selected_segmenter(&self) -> VoiceIdSegmenterPreference {
        self.segmenter.preference
    }

    pub(crate) fn diarize(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Diarization, ExternalDiarizationError> {
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
        let vad_regions = self.vad_regions(samples, sample_rate_hz)?;
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
        if embeddings.is_empty() {
            return Ok(Diarization {
                turns: Vec::new(),
                centroids: Vec::new(),
            });
        }
        cancel_checkpoint(canceled)?;
        let labels = self.clusterer.cluster(&embeddings, hint);
        let cluster_segments = compress_cluster_segments(&embedded_chunks, &labels);
        let speaker_count = labels
            .iter()
            .map(|speaker| speaker.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let turns = reconstruct_global_turns(
            &activity,
            &cluster_segments,
            speaker_count,
            samples.len() as f64 / sample_rate_hz as f64,
        );
        let centroids = speaker_centroids(&labels, &embeddings);
        Ok(Diarization { turns, centroids })
    }

    fn vad_regions(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<Vec<TimeRange>, ExternalDiarizationError> {
        self.vad
            .compute_speech_slices(samples, sample_rate_hz, &crate::LongFormOptions::default())
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
            .map_err(ExternalDiarizationError::Vad)
    }
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
            .closest_frame(cluster.range.start_s + activity.frame_clock.duration_s * 0.5)
            .min(frames);
        let end = activity
            .frame_clock
            .closest_frame(cluster.range.end_s + activity.frame_clock.duration_s * 0.5)
            .min(frames);
        for frame in start..end {
            cluster_frames[frame * speaker_count + cluster.speaker.0 as usize] = 1;
        }
    }

    let mut activations = vec![0u16; frames * speaker_count];
    for window in &activity.windows {
        let start = activity
            .frame_clock
            .closest_frame(window.start_s + activity.frame_clock.duration_s * 0.5);
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

    fn clock() -> ActivityFrameClock {
        ActivityFrameClock {
            start_s: 0.0,
            duration_s: 0.2,
            step_s: 0.1,
        }
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
                start_s: 0.0,
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
                start_s: 0.0,
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
}
