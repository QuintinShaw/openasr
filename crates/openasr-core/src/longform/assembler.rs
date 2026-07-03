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
    /// End of the audio region (original-timeline seconds) already committed by
    /// prior slices. Consecutive slices overlap at forced/energy cuts, so the
    /// next slice re-decodes that region; anything it emits before this point is
    /// a redundant re-read (or a weak-model hallucination of partial audio) and
    /// is trimmed by time before it can survive into the transcript.
    committed_end_original: Option<f32>,
}

impl TranscriptAssembler {
    pub fn new(timeline: TimelineMap, merge_policy: SegmentMergePolicy) -> Self {
        Self {
            timeline,
            merge_policy,
            segments: Vec::new(),
            stats: LongFormAssembleStats::default(),
            committed_end_original: None,
        }
    }

    pub fn push_slice_result(&mut self, mut transcript: SliceTranscript) {
        // The trim boundary is the region committed by *prior* slices; this
        // slice's own span is folded into the boundary afterwards so the next
        // slice trims against it (even if this slice is silent / emits nothing).
        let trim_boundary = self.committed_end_original;
        let slice_committed_end = self.slice_content_end_original(&transcript.slice);
        self.committed_end_original = Some(match self.committed_end_original {
            Some(previous) => previous.max(slice_committed_end),
            None => slice_committed_end,
        });
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
            let mut mapped = self.map_segment_time(&segment, &slice, time_domain);
            // Time-domain overlap trim: drop any leading words / whole segments
            // whose audio lies in the region a prior slice already committed.
            // This is text-agnostic, so it catches weak-model hallucinations of
            // the re-read overlap (e.g. "belief" mis-decoded as "If,") that the
            // text-equality dedup below cannot see.
            if let Some(boundary) = trim_boundary
                && trim_committed_overlap(&mut mapped, boundary)
            {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            if self.try_drop_redundant_segment(&mapped) {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            // Distinct segments are kept distinct: the post-ASR cue
            // re-segmentation pass owns subtitle granularity, so the assembler
            // no longer coalesces adjacent same-speaker segments into paragraph
            // blobs. Only exact / high-overlap duplicates from slice overlap are
            // dropped above.
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

    /// End of this slice's content span in original-timeline seconds. The
    /// `content_end_sample` indexes the same processed/plan audio domain that
    /// [`Self::map_segment_time`] maps from, so mapping it through the timeline
    /// yields the original-time cut point this slice commits up to.
    fn slice_content_end_original(&self, slice: &AudioSlice) -> f32 {
        let sample_rate = 16_000.0_f32;
        let processed_end = slice.content_end_sample as f32 / sample_rate;
        self.timeline
            .map_processed_to_original_seconds(processed_end)
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

/// Trim the part of a mapped segment that lies in the audio region a prior slice
/// already committed (`[.., boundary)` in original-timeline seconds). Returns
/// `true` when the whole segment falls inside that region and should be dropped.
///
/// A word is assigned to whichever side of `boundary` holds the majority of it
/// (midpoint rule), so a word straddling the cut is kept in exactly one slice.
/// Leading committed words are dropped and the segment text is reconstructed
/// from the surviving word span (exact substring of the original text, so CJK
/// and glued punctuation stay intact); a segment left empty is dropped.
fn trim_committed_overlap(segment: &mut Segment, boundary: f32) -> bool {
    // Whole segment already behind the committed frontier: drop it outright.
    // (This is the standalone-orphan shape, e.g. a hallucinated 1-word cue.)
    if segment.end <= boundary {
        return true;
    }
    if segment.words.is_empty() {
        // No word anchors: fall back to the segment midpoint.
        let midpoint = 0.5 * (segment.start + segment.end);
        return midpoint < boundary;
    }
    let first_keep = segment
        .words
        .iter()
        .position(|word| 0.5 * (word.start + word.end) >= boundary);
    let Some(first_keep) = first_keep else {
        // Every word's majority sits in the committed region.
        return true;
    };
    if first_keep == 0 {
        return false;
    }
    let chars: Vec<char> = segment.text.chars().collect();
    let new_text = match leading_word_char_offset(&chars, &segment.words, first_keep) {
        Some(offset) => chars[offset..]
            .iter()
            .collect::<String>()
            .trim()
            .to_string(),
        // Words did not align to the text (unexpected): rebuild from the kept
        // word tokens rather than mis-slice the string.
        None => segment.words[first_keep..]
            .iter()
            .map(|word| word.word.trim())
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
    };
    segment.words.drain(0..first_keep);
    if let Some(first) = segment.words.first() {
        segment.start = first.start;
    }
    segment.text = new_text;
    segment.text.trim().is_empty()
}

/// Char offset at which `words[first_keep]` begins within `chars`, found by the
/// same greedy whitespace-delimited match the cue splitter uses. Returns `None`
/// if a leading word does not align to the text.
fn leading_word_char_offset(
    chars: &[char],
    words: &[WordTimestamp],
    first_keep: usize,
) -> Option<usize> {
    let mut idx = 0usize;
    for word in &words[..first_keep] {
        while idx < chars.len() && chars[idx].is_whitespace() {
            idx += 1;
        }
        let token: Vec<char> = word.word.trim().chars().collect();
        if token.is_empty() {
            continue;
        }
        if idx + token.len() > chars.len() || chars[idx..idx + token.len()] != token[..] {
            return None;
        }
        idx += token.len();
    }
    while idx < chars.len() && chars[idx].is_whitespace() {
        idx += 1;
    }
    Some(idx)
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
    fn assembler_keeps_adjacent_segments_distinct() {
        // Adjacent, non-overlapping segments are no longer coalesced into a
        // paragraph blob: the post-ASR cue re-segmentation pass owns subtitle
        // granularity. Each segment survives with its own words and timing, and
        // the joined transcript text still reads as one paragraph.
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
            2,
            "adjacent segments must stay distinct"
        );
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "from openasr");
        assert_eq!(transcription.text, "hello world from openasr");
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

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn absolute_segment(text: &str, start: f32, end: f32, words: Vec<WordTimestamp>) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words,
        }
    }

    #[test]
    fn assembler_time_trims_hallucinated_leading_overlap_word() {
        // Field defect shape: a forced cut at 1.0s widens the overlap so the next
        // slice re-reads the straddling audio. A weak model hallucinates the head
        // of its monolithic segment ("If,") from that already-committed region;
        // its text does not match anything in slice 1, so text-equality dedup is
        // blind to it. The time trim drops it by timestamp and keeps the rest.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If, mad indeed would".to_string(),
            segments: vec![absolute_segment(
                "If, mad indeed would",
                0.80,
                1.90,
                vec![
                    word("If,", 0.80, 0.95),
                    word("mad", 1.10, 1.30),
                    word("indeed", 1.35, 1.60),
                    word("would", 1.65, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "mad indeed would");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(transcription.segments[1].words[0].word, "mad");
        assert!((transcription.segments[1].start - 1.10).abs() < 1e-4);
        assert_eq!(transcription.text, "hello world mad indeed would");
        // The trimmed word is not a dropped segment, so no whole-segment drop.
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_drops_standalone_orphan_inside_committed_span() {
        // The "If," rendered as its own leading cue: the whole segment sits behind
        // the committed frontier and is dropped outright.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If,".to_string(),
            segments: vec![
                absolute_segment("If,", 0.80, 0.95, vec![word("If,", 0.80, 0.95)]),
                absolute_segment(
                    "mad indeed",
                    1.10,
                    1.60,
                    vec![word("mad", 1.10, 1.30), word("indeed", 1.35, 1.60)],
                ),
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "mad indeed");
        assert_eq!(transcription.text, "hello world mad indeed");
        assert_eq!(stats.duplicate_merge_count, 1);
    }

    #[test]
    fn assembler_keeps_straddling_word_with_majority_after_boundary() {
        // A word straddling the 1.0s cut whose midpoint (1.025s) is past the
        // boundary belongs to the new slice and is kept whole.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.85,
                1.60,
                vec![word("straddle", 0.85, 1.20), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "straddle tail");
        assert_eq!(transcription.segments[1].words.len(), 2);
    }

    #[test]
    fn assembler_trims_straddling_word_with_majority_before_boundary() {
        // Same cut, but the leading word's midpoint (0.925s) is before the
        // boundary, so the word belongs to the prior slice and is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.75,
                1.60,
                vec![word("straddle", 0.75, 1.10), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "tail");
        assert_eq!(transcription.segments[1].words.len(), 1);
        assert_eq!(transcription.segments[1].words[0].word, "tail");
    }

    #[test]
    fn assembler_does_not_trim_without_overlap() {
        // Abutting, non-overlapping slices: the second slice's words all sit past
        // the committed frontier, so nothing is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "next words here".to_string(),
            segments: vec![absolute_segment(
                "next words here",
                1.20,
                1.90,
                vec![
                    word("next", 1.20, 1.40),
                    word("words", 1.50, 1.70),
                    word("here", 1.75, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "next words here");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(stats.duplicate_merge_count, 0);
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
