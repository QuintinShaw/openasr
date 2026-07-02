use crate::{Segment, Transcription, WordTimestamp};

use super::slicing::{AudioSlice, LongFormBenchmarkMetadata};
use super::timeline::TimelineMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentTimeDomain {
    RelativeToSliceContent,
    AbsoluteOriginal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceTranscript {
    pub slice: AudioSlice,
    pub text: String,
    pub segments: Vec<Segment>,
    pub time_domain: SegmentTimeDomain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMergePolicy {
    pub max_gap_seconds: f32,
    pub redundant_overlap_ratio: f32,
    pub redundant_min_words: usize,
}

impl Default for SegmentMergePolicy {
    fn default() -> Self {
        Self {
            max_gap_seconds: 1.2,
            redundant_overlap_ratio: 0.8,
            redundant_min_words: 4,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongFormAssembleStats {
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
}

#[derive(Debug)]
pub struct TranscriptAssembler {
    timeline: TimelineMap,
    merge_policy: SegmentMergePolicy,
    segments: Vec<Segment>,
    stats: LongFormAssembleStats,
}

impl TranscriptAssembler {
    pub fn new(timeline: TimelineMap, merge_policy: SegmentMergePolicy) -> Self {
        Self {
            timeline,
            merge_policy,
            segments: Vec::new(),
            stats: LongFormAssembleStats::default(),
        }
    }

    pub fn push_slice_result(&mut self, mut transcript: SliceTranscript) {
        transcript.text = transcript.text.trim().to_string();
        if transcript.text.is_empty()
            && transcript
                .segments
                .iter()
                .all(|segment| segment.text.trim().is_empty())
        {
            self.stats.skipped_silent_chunks += 1;
            return;
        }
        let synthesized_slice_fallback = transcript.segments.is_empty();
        if transcript.segments.is_empty() {
            let sample_rate = 16_000.0_f32;
            transcript.segments.push(Segment {
                start: 0.0,
                end: transcript.slice.content_duration_samples() as f32 / sample_rate,
                text: transcript.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            });
            transcript.time_domain = SegmentTimeDomain::RelativeToSliceContent;
        }
        let time_domain = transcript.time_domain;
        let slice = transcript.slice.clone();
        for mut segment in transcript.segments {
            segment.text = segment.text.trim().to_string();
            if segment.text.is_empty() {
                continue;
            }
            let mapped = self.map_segment_time(&segment, &slice, time_domain);
            if self.try_drop_redundant_segment(&mapped) {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            if !synthesized_slice_fallback && self.try_merge_adjacent_segment(mapped.clone()) {
                continue;
            }
            self.segments.push(mapped);
        }
    }

    pub fn into_transcription(self) -> Transcription {
        self.into_parts().0
    }

    pub fn into_parts(self) -> (Transcription, LongFormAssembleStats) {
        let text = self
            .segments
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        (
            Transcription {
                text,
                segments: self.segments,
                longform: None,
                language: None,
            },
            self.stats,
        )
    }

    pub fn benchmark_metadata(&self) -> LongFormBenchmarkMetadata {
        LongFormBenchmarkMetadata {
            chunk_count: self.segments.len(),
            skipped_silent_chunks: self.stats.skipped_silent_chunks,
            duplicate_merge_count: self.stats.duplicate_merge_count,
            provenance: vec!["core.longform.assembler".to_string()],
        }
    }

    fn map_segment_time(
        &self,
        segment: &Segment,
        slice: &AudioSlice,
        time_domain: SegmentTimeDomain,
    ) -> Segment {
        let sample_rate = 16_000.0_f32;
        let mut start = segment.start.max(0.0);
        let mut end = segment.end.max(start);
        let mut content_offset = 0.0_f32;
        if time_domain == SegmentTimeDomain::RelativeToSliceContent {
            content_offset = slice.content_start_sample as f32 / sample_rate;
            start += content_offset;
            end += content_offset;
        }
        let original_start = self.timeline.map_processed_to_original_seconds(start);
        let original_end = self
            .timeline
            .map_processed_to_original_seconds(end)
            .max(original_start);
        let words = segment
            .words
            .iter()
            .filter_map(|word| {
                map_word_time_to_original(
                    word,
                    content_offset,
                    time_domain,
                    &self.timeline,
                    original_start,
                    original_end,
                )
            })
            .collect();
        Segment {
            start: original_start,
            end: original_end,
            text: segment.text.clone(),
            speaker: segment.speaker.clone(),
            speaker_label: segment.speaker_label.clone(),
            speaker_profile_id: segment.speaker_profile_id.clone(),
            words,
        }
    }

    fn try_drop_redundant_segment(&self, current: &Segment) -> bool {
        let Some(previous) = self.segments.last() else {
            return false;
        };
        if current.start < previous.end {
            return false;
        }
        let gap_seconds = current.start - previous.end;
        if gap_seconds > self.merge_policy.max_gap_seconds {
            return false;
        }
        let previous_words = normalize_words(&previous.text);
        let current_words = normalize_words(&current.text);
        if previous_words.is_empty() || current_words.is_empty() {
            return false;
        }
        if previous_words == current_words {
            return true;
        }
        let overlap = longest_common_window_len(&previous_words, &current_words);
        let min_len = previous_words.len().min(current_words.len());
        min_len >= self.merge_policy.redundant_min_words
            && overlap as f32 / min_len as f32 >= self.merge_policy.redundant_overlap_ratio
    }

    fn try_merge_adjacent_segment(&mut self, current: Segment) -> bool {
        let Some(previous) = self.segments.last_mut() else {
            return false;
        };
        if current.start < previous.end {
            return false;
        }
        let gap = current.start - previous.end;
        if gap > self.merge_policy.max_gap_seconds {
            return false;
        }
        // Never merge across a speaker change — the merged segment would
        // misattribute one speaker's words to the other.
        if previous.speaker != current.speaker {
            return false;
        }
        if !previous.text.ends_with(' ') {
            previous.text.push(' ');
        }
        previous.text.push_str(current.text.trim());
        previous.end = previous.end.max(current.end);
        // The merged segment's words already carry original-timeline timestamps
        // (mapped by map_segment_time before merge); append them so word-level
        // timing spans the whole merged segment instead of only its first half.
        previous.words.extend(current.words);
        true
    }
}

fn map_word_time_to_original(
    word: &WordTimestamp,
    content_offset: f32,
    time_domain: SegmentTimeDomain,
    timeline: &TimelineMap,
    segment_start: f32,
    segment_end: f32,
) -> Option<WordTimestamp> {
    let text = word.word.trim();
    if text.is_empty() || !word.start.is_finite() || !word.end.is_finite() {
        return None;
    }
    let mut start = word.start.max(0.0);
    let mut end = word.end.max(start);
    if time_domain == SegmentTimeDomain::RelativeToSliceContent {
        start += content_offset;
        end += content_offset;
    }
    let original_start = timeline
        .map_processed_to_original_seconds(start)
        .clamp(segment_start, segment_end);
    let original_end = timeline
        .map_processed_to_original_seconds(end)
        .max(original_start)
        .clamp(original_start, segment_end);
    Some(WordTimestamp {
        word: text.to_string(),
        start: original_start,
        end: original_end,
        confidence: word.confidence,
    })
}

fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn contains_window(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn longest_common_window_len(left: &[String], right: &[String]) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let mut longest = 0usize;
    for start in 0..left.len() {
        for end in (start + 1)..=left.len() {
            let candidate = &left[start..end];
            if candidate.len() <= longest {
                continue;
            }
            if contains_window(right, candidate) {
                longest = candidate.len();
            }
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longform::{AudioSlice, AudioSliceKind, TimelineAnchor};

    fn slice(start: usize, end: usize) -> AudioSlice {
        AudioSlice {
            index: 0,
            kind: AudioSliceKind::Fixed,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        }
    }

    #[test]
    fn assembler_maps_relative_segment_times() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 10.0,
                original_seconds: 10.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "hello".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 0.5,
                text: "hello".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: vec![WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.1,
                    end: 0.4,
                    confidence: None,
                }],
            }],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
        assert!(transcription.segments[0].start >= 1.0);
        assert_eq!(transcription.segments[0].words.len(), 1);
        assert_eq!(transcription.segments[0].words[0].word, "hello");
        assert!(transcription.segments[0].words[0].start >= 1.1);
        assert!(transcription.segments[0].words[0].end <= 1.4);
    }

    #[test]
    fn assembler_drops_redundant_overlap() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(15_000, 31_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 1.05,
                end: 2.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
    }

    #[test]
    fn assembler_merge_preserves_words_from_both_segments() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 32_000),
            text: "hello world".to_string(),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello world".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: vec![
                        WordTimestamp {
                            word: "hello".to_string(),
                            start: 0.1,
                            end: 0.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "world".to_string(),
                            start: 0.5,
                            end: 0.9,
                            confidence: None,
                        },
                    ],
                },
                Segment {
                    start: 1.0,
                    end: 2.0,
                    text: "from openasr".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: vec![
                        WordTimestamp {
                            word: "from".to_string(),
                            start: 1.1,
                            end: 1.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "openasr".to_string(),
                            start: 1.5,
                            end: 1.9,
                            confidence: None,
                        },
                    ],
                },
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription.segments.len(),
            1,
            "adjacent non-overlapping segments should merge into one"
        );
        let merged = &transcription.segments[0];
        assert_eq!(merged.text, "hello world from openasr");
        let merged_words: Vec<&str> = merged.words.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(merged_words, ["hello", "world", "from", "openasr"]);
    }

    #[test]
    fn assembler_preserves_slice_boundaries_for_synthesized_slice_segments() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "first chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "second chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].start, 0.0);
        assert_eq!(transcription.segments[0].end, 1.0);
        assert_eq!(transcription.segments[1].start, 1.0);
        assert_eq!(transcription.segments[1].end, 2.0);
    }

    #[test]
    fn assembler_maps_packed_slice_segments_back_to_original_timeline() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 1.0,
                original_seconds: 1.0,
            },
            TimelineAnchor {
                processed_seconds: 1.2,
                original_seconds: 12.0,
            },
            TimelineAnchor {
                processed_seconds: 2.2,
                original_seconds: 13.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 35_200,
                content_start_sample: 0,
                content_end_sample: 35_200,
            },
            text: "first second".to_string(),
            segments: vec![
                Segment {
                    start: 0.1,
                    end: 0.9,
                    text: "first".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                },
                Segment {
                    start: 1.3,
                    end: 2.0,
                    text: "second".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                },
            ],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert!(transcription.segments[0].end <= 1.0);
        assert!(transcription.segments[1].start >= 12.0);
        assert!(transcription.segments[1].end <= 13.0);
    }
}
