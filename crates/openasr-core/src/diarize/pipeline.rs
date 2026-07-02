//! Batch diarization pipeline: speech segments → speaker embeddings →
//! clustering → speaker turns.
//!
//! Speech regions arrive pre-computed (one assumed speaker per region): the
//! caller (`native_transcribe::resolve_speech_segments`) prefers pyannote
//! segmentation regions (speaker-change + overlap aware, P3-full) when that
//! pack is installed and falls back to the neural VAD's slices (P3-lite),
//! which works for turn-taking conversations with pauses.

use std::collections::BTreeMap;

use super::clustering::{ClusterContext, SpeakerClusterer};
use super::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, SpeakerTurn, TimeRange};
use super::embed::SpeakerEmbedder;

/// Resolve the speech regions to embed: pyannote segmentation when its pack is
/// installed (finer, speaker-change + overlap aware), else the neural VAD
/// slices. Both batch attribution and enrollment go through this single
/// resolver so their embeddings live in the same space.
pub fn resolve_speech_regions(samples: &[f32]) -> Option<Vec<TimeRange>> {
    Some(
        resolve_diarization_regions(samples)?
            .iter()
            .map(|region| region.range)
            .collect(),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct DiarizationRegion {
    pub range: TimeRange,
    pub local_speaker: Option<SpeakerId>,
    pub overlap: bool,
}

/// Resolve speech regions plus optional pyannote local-speaker metadata for
/// context-aware clustering.
pub fn resolve_diarization_regions(samples: &[f32]) -> Option<Vec<DiarizationRegion>> {
    if let Some(segmenter) = super::segment::shared_segmenter() {
        let turns = segmenter.segment(samples, 16_000).unwrap_or_default();
        if !turns.is_empty() {
            return Some(
                turns
                    .iter()
                    .map(|turn| DiarizationRegion {
                        range: turn.range,
                        local_speaker: Some(turn.speaker),
                        overlap: turn.overlap,
                    })
                    .collect(),
            );
        }
    }

    use crate::longform::LongFormVadProvider;
    let vad = super::vad::SileroVadProvider::shared()?;
    let options = crate::LongFormOptions::default();
    let slices = vad.compute_speech_slices(samples, 16_000, &options).ok()?;
    Some(
        slices
            .iter()
            .map(|s| {
                let range = TimeRange::new(
                    s.start_sample as f64 / 16_000.0,
                    s.end_sample as f64 / 16_000.0,
                );
                DiarizationRegion {
                    range,
                    local_speaker: None,
                    overlap: false,
                }
            })
            .collect(),
    )
}

/// Shortest speech segment (seconds) worth embedding; shorter regions give
/// unreliable speaker embeddings and are skipped.
const MIN_SEGMENT_S: f64 = 0.5;
const MAX_EMBED_CHUNK_S: f64 = 5.0;

/// The result of batch diarization: speaker turns plus each speaker's mean
/// (L2-normalized) embedding centroid, used for optional enrollment matching.
pub struct Diarization {
    pub turns: Vec<SpeakerTurn>,
    pub centroids: Vec<(SpeakerId, SpeakerEmbedding)>,
}

struct EmbeddedRegion {
    source_index: usize,
    range: TimeRange,
    context: ClusterContext,
    embedding: SpeakerEmbedding,
}

#[derive(Clone)]
struct LabeledRegion {
    range: TimeRange,
    context: ClusterContext,
    speaker: SpeakerId,
}

/// Composes an embedder and a clusterer into batch diarization.
pub struct BatchDiarizer<'a> {
    embedder: &'a dyn SpeakerEmbedder,
    clusterer: &'a dyn SpeakerClusterer,
    min_segment_s: f64,
}

impl<'a> BatchDiarizer<'a> {
    pub fn new(embedder: &'a dyn SpeakerEmbedder, clusterer: &'a dyn SpeakerClusterer) -> Self {
        Self {
            embedder,
            clusterer,
            min_segment_s: MIN_SEGMENT_S,
        }
    }

    /// Diarize `samples` given pre-computed `speech` regions (from the VAD),
    /// returning one speaker turn per embeddable region. Regions shorter than
    /// `min_segment_s`, or that fail to embed, are dropped.
    pub fn diarize(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        speech: &[TimeRange],
        hint: DiarizeHint,
    ) -> Diarization {
        let regions: Vec<DiarizationRegion> = speech
            .iter()
            .map(|range| DiarizationRegion {
                range: *range,
                local_speaker: None,
                overlap: false,
            })
            .collect();
        self.diarize_regions(samples, sample_rate_hz, &regions, hint)
    }

    pub fn diarize_regions(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        speech: &[DiarizationRegion],
        hint: DiarizeHint,
    ) -> Diarization {
        let mut embedded_regions: Vec<EmbeddedRegion> = Vec::new();
        for (source_index, region) in speech.iter().enumerate() {
            for range in embedding_ranges(region.range, self.min_segment_s) {
                let start = (range.start_s * sample_rate_hz as f64).max(0.0) as usize;
                let end = ((range.end_s * sample_rate_hz as f64) as usize).min(samples.len());
                if end <= start {
                    continue;
                }
                if let Ok(embedding) = self.embedder.embed(&samples[start..end], sample_rate_hz) {
                    let context = ClusterContext {
                        range,
                        local_speaker: region.local_speaker,
                        overlap: region.overlap,
                    };
                    embedded_regions.push(EmbeddedRegion {
                        source_index,
                        range,
                        context,
                        embedding,
                    });
                }
            }
        }
        let embeddings: Vec<SpeakerEmbedding> = embedded_regions
            .iter()
            .map(|region| region.embedding.clone())
            .collect();
        if embeddings.is_empty() {
            return Diarization {
                turns: Vec::new(),
                centroids: Vec::new(),
            };
        }
        let context: Vec<ClusterContext> = embedded_regions
            .iter()
            .map(|region| region.context)
            .collect();
        let mut labels = self
            .clusterer
            .cluster_with_context(&embeddings, &context, hint);
        if let Some(refined) = super::vbx::refine_labels(&embeddings, &context, &labels) {
            labels = refined;
        }
        if let Some(refined) = super::vbx::refine_dense_labels(
            samples,
            sample_rate_hz,
            self.embedder,
            &context,
            &labels,
        ) {
            labels = refined;
        }
        let centroids = speaker_centroids(&labels, &embeddings);
        let turn_regions = labeled_turn_regions(speech, &embedded_regions, &labels);
        let turns = build_speaker_turns(&turn_regions, centroids.len());
        Diarization { turns, centroids }
    }
}

fn labeled_turn_regions(
    speech: &[DiarizationRegion],
    embedded_regions: &[EmbeddedRegion],
    labels: &[SpeakerId],
) -> Vec<LabeledRegion> {
    let mut embedded_by_source: BTreeMap<usize, Vec<LabeledRegion>> = BTreeMap::new();
    for (region, &speaker) in embedded_regions.iter().zip(labels) {
        embedded_by_source
            .entry(region.source_index)
            .or_default()
            .push(LabeledRegion {
                range: region.range,
                context: region.context,
                speaker,
            });
    }
    let local_labels = local_speaker_label_map(embedded_regions, labels);
    let mut regions = Vec::new();
    for (source_index, region) in speech.iter().enumerate() {
        if let Some(chunks) = embedded_by_source.get(&source_index) {
            regions.extend(chunks.iter().cloned());
        } else {
            let Some(speaker) = nearest_local_speaker_label(region, embedded_regions, labels)
                .or_else(|| {
                    region
                        .local_speaker
                        .and_then(|local| local_labels.get(&local).copied())
                })
            else {
                continue;
            };
            regions.push(LabeledRegion {
                range: region.range,
                context: ClusterContext {
                    range: region.range,
                    local_speaker: region.local_speaker,
                    overlap: region.overlap,
                },
                speaker,
            });
        }
    }
    regions
}

fn embedding_ranges(range: TimeRange, min_segment_s: f64) -> Vec<TimeRange> {
    if range.duration_s() < min_segment_s {
        return Vec::new();
    }
    if range.duration_s() <= MAX_EMBED_CHUNK_S {
        return vec![range];
    }
    let mut ranges = Vec::new();
    let mut start_s = range.start_s;
    while start_s < range.end_s {
        let end_s = (start_s + MAX_EMBED_CHUNK_S).min(range.end_s);
        if end_s - start_s >= min_segment_s {
            ranges.push(TimeRange::new(start_s, end_s));
        }
        start_s = end_s;
    }
    ranges
}

fn nearest_local_speaker_label(
    region: &DiarizationRegion,
    embedded_regions: &[EmbeddedRegion],
    labels: &[SpeakerId],
) -> Option<SpeakerId> {
    let local = region.local_speaker?;
    embedded_regions
        .iter()
        .zip(labels)
        .filter(|(candidate, _)| candidate.context.local_speaker == Some(local))
        .max_by(|(left, left_label), (right, right_label)| {
            let left_overlap = region.range.intersection_s(&left.range);
            let right_overlap = region.range.intersection_s(&right.range);
            left_overlap
                .total_cmp(&right_overlap)
                .then_with(|| {
                    local_region_gap_s(region.range, right.range)
                        .total_cmp(&local_region_gap_s(region.range, left.range))
                })
                .then_with(|| left.range.duration_s().total_cmp(&right.range.duration_s()))
                .then_with(|| right_label.cmp(left_label))
        })
        .map(|(_, &label)| label)
}

fn local_region_gap_s(left: TimeRange, right: TimeRange) -> f64 {
    if left.overlaps(&right) {
        0.0
    } else if left.end_s <= right.start_s {
        right.start_s - left.end_s
    } else {
        left.start_s - right.end_s
    }
}

fn local_speaker_label_map(
    embedded_regions: &[EmbeddedRegion],
    labels: &[SpeakerId],
) -> BTreeMap<SpeakerId, SpeakerId> {
    let mut scores: BTreeMap<SpeakerId, BTreeMap<SpeakerId, f64>> = BTreeMap::new();
    for (region, &label) in embedded_regions.iter().zip(labels) {
        let Some(local) = region.context.local_speaker else {
            continue;
        };
        *scores.entry(local).or_default().entry(label).or_default() += region.range.duration_s();
    }
    scores
        .into_iter()
        .filter_map(|(local, labels)| {
            let speaker = labels
                .into_iter()
                .max_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| right.0.cmp(&left.0))
                })?
                .0;
            Some((local, speaker))
        })
        .collect()
}

fn build_speaker_turns(regions: &[LabeledRegion], speaker_count: usize) -> Vec<SpeakerTurn> {
    let turns = if regions_have_overlap_context(regions) && speaker_count > 1 {
        overlap_aware_turns(regions)
    } else {
        regions
            .iter()
            .map(|region| SpeakerTurn {
                range: region.range,
                speaker: region.speaker,
                overlap: region.context.overlap,
            })
            .collect()
    };
    collapse_adjacent_turns(turns)
}

fn regions_have_overlap_context(regions: &[LabeledRegion]) -> bool {
    for (left_index, left) in regions.iter().enumerate() {
        for right in regions.iter().skip(left_index + 1) {
            if distinct_local_speakers_overlap(left, right) {
                return true;
            }
        }
    }
    false
}

fn overlap_aware_turns(regions: &[LabeledRegion]) -> Vec<SpeakerTurn> {
    let mut boundaries: Vec<f64> = regions
        .iter()
        .flat_map(|region| [region.range.start_s, region.range.end_s])
        .collect();
    boundaries.sort_by(|left, right| left.total_cmp(right));
    boundaries.dedup_by(|left, right| (*left - *right).abs() < 1e-9);

    let mut turns = Vec::new();
    for pair in boundaries.windows(2) {
        let range = TimeRange::new(pair[0], pair[1]);
        if range.duration_s() <= 0.0 {
            continue;
        }
        let midpoint = (range.start_s + range.end_s) * 0.5;
        let active: Vec<usize> = regions
            .iter()
            .enumerate()
            .filter_map(|(index, region)| {
                (region.range.start_s <= midpoint && midpoint < region.range.end_s).then_some(index)
            })
            .collect();
        if active.is_empty() {
            continue;
        }

        let overlap = active_regions_overlap(regions, &active);
        let speakers = if overlap {
            active_local_speaker_labels(regions, &active, 2)
        } else {
            vec![dominant_active_speaker(regions, &active)]
        };
        for speaker in speakers {
            push_or_extend_turn(
                &mut turns,
                SpeakerTurn {
                    range,
                    speaker,
                    overlap,
                },
            );
        }
    }
    turns
}

fn active_regions_overlap(regions: &[LabeledRegion], active: &[usize]) -> bool {
    for (left_pos, &left_index) in active.iter().enumerate() {
        for &right_index in active.iter().skip(left_pos + 1) {
            let left = &regions[left_index];
            let right = &regions[right_index];
            if distinct_local_speakers_overlap(left, right) {
                return true;
            }
        }
    }
    false
}

fn dominant_active_speaker(regions: &[LabeledRegion], active: &[usize]) -> SpeakerId {
    active
        .iter()
        .copied()
        .max_by(|&left, &right| {
            regions[left]
                .range
                .duration_s()
                .total_cmp(&regions[right].range.duration_s())
                .then_with(|| regions[right].speaker.cmp(&regions[left].speaker))
        })
        .map(|index| regions[index].speaker)
        .unwrap_or(SpeakerId(0))
}

fn distinct_local_speakers_overlap(left: &LabeledRegion, right: &LabeledRegion) -> bool {
    matches!(
        (left.context.local_speaker, right.context.local_speaker),
        (Some(left_speaker), Some(right_speaker))
            if left_speaker != right_speaker && left.range.overlaps(&right.range)
    )
}

fn active_local_speaker_labels(
    regions: &[LabeledRegion],
    active: &[usize],
    limit: usize,
) -> Vec<SpeakerId> {
    let mut by_local: BTreeMap<SpeakerId, usize> = BTreeMap::new();
    for &index in active {
        let Some(local_speaker) = regions[index].context.local_speaker else {
            continue;
        };
        by_local
            .entry(local_speaker)
            .and_modify(|existing| {
                if regions[index].range.duration_s() > regions[*existing].range.duration_s() {
                    *existing = index;
                }
            })
            .or_insert(index);
    }

    let mut speakers = Vec::new();
    for (_, index) in by_local.into_iter().take(limit) {
        let speaker = regions[index].speaker;
        if !speakers.contains(&speaker) {
            speakers.push(speaker);
        }
    }
    speakers
}

fn push_or_extend_turn(turns: &mut Vec<SpeakerTurn>, turn: SpeakerTurn) {
    if let Some(last) = turns.last_mut()
        && last.speaker == turn.speaker
        && last.overlap == turn.overlap
        && (last.range.end_s - turn.range.start_s).abs() < 1e-9
    {
        last.range.end_s = turn.range.end_s;
        return;
    }
    turns.push(turn);
}

fn collapse_adjacent_turns(turns: Vec<SpeakerTurn>) -> Vec<SpeakerTurn> {
    let mut collapsed = Vec::with_capacity(turns.len());
    for turn in turns {
        push_or_extend_turn(&mut collapsed, turn);
    }
    collapsed
}

/// Mean L2-normalized embedding per speaker id.
fn speaker_centroids(
    labels: &[SpeakerId],
    embeddings: &[SpeakerEmbedding],
) -> Vec<(SpeakerId, SpeakerEmbedding)> {
    let dim = embeddings.first().map(|e| e.dim()).unwrap_or(0);
    let mut sums: BTreeMap<SpeakerId, (Vec<f32>, usize)> = BTreeMap::new();
    for (label, embedding) in labels.iter().zip(embeddings) {
        let entry = sums.entry(*label).or_insert_with(|| (vec![0.0; dim], 0));
        for (acc, v) in entry.0.iter_mut().zip(&embedding.0) {
            *acc += v;
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(id, (sum, _count))| (id, SpeakerEmbedding::l2_normalized(sum)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::clustering::AgglomerativeClusterer;
    use crate::diarize::embed::EmbedError;

    /// Mock embedder: returns a fixed embedding per "speaker", chosen by the
    /// segment's mean amplitude sign, so two speakers are separable.
    struct MockEmbedder;
    impl SpeakerEmbedder for MockEmbedder {
        fn embed(&self, samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let v = if mean >= 0.0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
            Ok(SpeakerEmbedding::l2_normalized(v))
        }
        fn embedding_dim(&self) -> usize {
            2
        }
    }

    struct FixedClusterer {
        labels: Vec<SpeakerId>,
    }

    impl SpeakerClusterer for FixedClusterer {
        fn cluster(&self, _embeddings: &[SpeakerEmbedding], _hint: DiarizeHint) -> Vec<SpeakerId> {
            self.labels.clone()
        }

        fn cluster_with_context(
            &self,
            _embeddings: &[SpeakerEmbedding],
            _context: &[ClusterContext],
            _hint: DiarizeHint,
        ) -> Vec<SpeakerId> {
            self.labels.clone()
        }
    }

    #[test]
    fn clusters_two_speakers_from_segments() {
        let sr = 16_000u32;
        // 0-1s positive (spkA), 1-2s negative (spkB), 2-3s positive (spkA).
        let mut samples = vec![0.0f32; sr as usize * 3];
        for s in &mut samples[0..sr as usize] {
            *s = 0.5;
        }
        for s in &mut samples[sr as usize..2 * sr as usize] {
            *s = -0.5;
        }
        for s in &mut samples[2 * sr as usize..] {
            *s = 0.5;
        }
        let speech = vec![
            TimeRange::new(0.0, 1.0),
            TimeRange::new(1.0, 2.0),
            TimeRange::new(2.0, 3.0),
        ];
        let clusterer = AgglomerativeClusterer::default();
        let diarizer = BatchDiarizer::new(&MockEmbedder, &clusterer);
        let result = diarizer.diarize(&samples, sr, &speech, DiarizeHint::Auto);
        let turns = result.turns;
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].speaker, turns[2].speaker, "same speaker A");
        assert_ne!(turns[0].speaker, turns[1].speaker, "B differs");
        // Two speakers -> two centroids.
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn drops_too_short_segments() {
        let diarizer_clusterer = AgglomerativeClusterer::default();
        let diarizer = BatchDiarizer::new(&MockEmbedder, &diarizer_clusterer);
        let samples = vec![0.5f32; 16_000];
        let speech = vec![TimeRange::new(0.0, 0.1)]; // 100 ms < 0.5 s
        assert!(
            diarizer
                .diarize(&samples, 16_000, &speech, DiarizeHint::Auto)
                .turns
                .is_empty()
        );
    }

    #[test]
    fn long_single_speaker_region_does_not_emit_embedding_chunk_boundaries() {
        let sr = 16_000u32;
        let samples = vec![0.5f32; sr as usize * 12];
        let speech = vec![TimeRange::new(0.0, 12.0)];
        let clusterer = FixedClusterer {
            labels: vec![SpeakerId(0), SpeakerId(0), SpeakerId(0)],
        };
        let diarizer = BatchDiarizer::new(&MockEmbedder, &clusterer);

        let turns = diarizer
            .diarize(&samples, sr, &speech, DiarizeHint::Auto)
            .turns;

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].range, TimeRange::new(0.0, 12.0));
        assert_eq!(turns[0].speaker, SpeakerId(0));
        assert!(!turns[0].overlap);
    }

    #[test]
    fn overlap_region_gets_active_local_speakers_not_global_top_two() {
        let sr = 16_000u32;
        let mut samples = vec![0.0f32; sr as usize * 3];
        for s in &mut samples[0..sr as usize] {
            *s = 0.5;
        }
        for s in &mut samples[sr as usize..2 * sr as usize] {
            *s = -0.5;
        }
        for s in &mut samples[2 * sr as usize..] {
            *s = 0.5;
        }
        let speech = vec![
            DiarizationRegion {
                range: TimeRange::new(0.0, 2.0),
                local_speaker: Some(SpeakerId(0)),
                overlap: true,
            },
            DiarizationRegion {
                range: TimeRange::new(1.0, 2.0),
                local_speaker: Some(SpeakerId(1)),
                overlap: true,
            },
            DiarizationRegion {
                range: TimeRange::new(2.0, 3.0),
                local_speaker: Some(SpeakerId(2)),
                overlap: false,
            },
        ];
        let clusterer = FixedClusterer {
            labels: vec![SpeakerId(0), SpeakerId(2), SpeakerId(1)],
        };
        let diarizer = BatchDiarizer::new(&MockEmbedder, &clusterer);

        let turns = diarizer
            .diarize_regions(&samples, sr, &speech, DiarizeHint::Auto)
            .turns;

        assert!(
            turns.iter().any(|turn| {
                turn.range == TimeRange::new(1.0, 2.0)
                    && turn.speaker == SpeakerId(0)
                    && turn.overlap
            }),
            "primary speaker remains active in overlap"
        );
        assert!(
            turns.iter().any(|turn| {
                turn.range == TimeRange::new(1.0, 2.0)
                    && turn.speaker == SpeakerId(2)
                    && turn.overlap
            }),
            "overlap span receives the second active local speaker"
        );
        assert!(
            turns.iter().all(|turn| {
                turn.range != TimeRange::new(1.0, 2.0)
                    || !turn.overlap
                    || turn.speaker != SpeakerId(1)
            }),
            "inactive global speaker must not be injected into overlap"
        );
        assert_eq!(
            turns
                .iter()
                .filter(|turn| turn.range == TimeRange::new(1.0, 2.0))
                .count(),
            2
        );
    }
}
