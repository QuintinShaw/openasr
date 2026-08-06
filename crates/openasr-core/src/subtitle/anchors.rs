//! Runtime validation of native (or aligned) word timestamps.
//!
//! Catalog `word_timestamp_source = native` is a capability declaration only.
//! Before a precise timeline is treated as trustworthy, the finished transcript
//! must pass these checks. Thresholds live here so the policy is unit-testable
//! without spinning up an aligner.

use crate::api::backend::{Segment, Transcription};

/// How far a word end may exceed the audio duration before it is treated as
/// out-of-bounds. Native decoders sometimes stamp the final word a few tens of
/// milliseconds past the clip end.
pub const AUDIO_DURATION_TOLERANCE_S: f32 = 0.35;

/// Minimum ratio of content characters in speech segments that must be covered
/// by non-empty word tokens. Below this the word list is treated as a sparse
/// sample rather than a reliable anchor stream.
pub const MIN_TEXT_COVERAGE_RATIO: f32 = 0.55;

/// A speech segment longer than this (seconds) with no word timestamps at all
/// is a systematic gap, not an empty interjection.
pub const MIN_SPEECH_SEGMENT_SECONDS_REQUIRING_WORDS: f32 = 0.40;

/// Largest contiguous gap (seconds) allowed between consecutive word ends/starts
/// inside a single speech segment before the timeline is considered hollow.
pub const MAX_WORDLESS_GAP_S: f32 = 4.0;

/// Maximum fraction of speech segments that may lack words before validation
/// fails (systemic missing words, not a single empty interjection).
pub const MAX_SPEECH_SEGMENTS_WITHOUT_WORDS_RATIO: f32 = 0.15;

/// Result of validating word anchors against a finished transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct WordAnchorValidation {
    pub quality: WordAnchorQuality,
    pub issues: Vec<WordAnchorIssue>,
}

impl WordAnchorValidation {
    pub fn is_reliable(&self) -> bool {
        matches!(self.quality, WordAnchorQuality::Reliable)
    }
}

/// Whether the transcript's word timestamps are trustworthy as a precise
/// timeline source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordAnchorQuality {
    Reliable,
    Unreliable,
}

/// One concrete reason a word-anchor stream failed validation.
#[derive(Debug, Clone, PartialEq)]
pub enum WordAnchorIssue {
    /// A timed speech segment carried no `words[]`.
    MissingWordsOnSpeech {
        segment_index: usize,
        start: f32,
        end: f32,
    },
    /// Too many speech segments lacked words.
    SystemicMissingWords {
        speech_segments: usize,
        missing: usize,
    },
    /// A word timestamp was NaN/Inf.
    NonFiniteTime {
        segment_index: usize,
        word_index: usize,
    },
    /// A word started or ended below zero (beyond a tiny float epsilon).
    NegativeTime {
        segment_index: usize,
        word_index: usize,
    },
    /// Word starts or ends were not non-decreasing inside a segment.
    NonMonotonic {
        segment_index: usize,
        word_index: usize,
    },
    /// A word's end was before its start (beyond float epsilon).
    InvertedInterval {
        segment_index: usize,
        word_index: usize,
    },
    /// A word end lay outside the audio duration (plus tolerance).
    OutsideAudioDuration {
        segment_index: usize,
        word_index: usize,
        end: f32,
        audio_duration_s: f32,
    },
    /// Word text covered too little of the segment manuscript text.
    InsufficientTextCoverage { segment_index: usize, coverage: f32 },
    /// A large span of speech had no word anchors between consecutive words.
    LargeWordlessGap { segment_index: usize, gap_s: f32 },
}

/// Validate word anchors on a finished transcription against the audio length.
///
/// Empty transcripts (no speech segments) are treated as reliable: there is
/// nothing to align and nothing to project.
pub fn validate_word_anchors(
    transcription: &Transcription,
    audio_duration_s: f32,
) -> WordAnchorValidation {
    let mut issues = Vec::new();
    let speech_segments: Vec<(usize, &Segment)> = transcription
        .segments
        .iter()
        .enumerate()
        .filter(|(_, segment)| is_speech_segment(segment))
        .collect();

    if speech_segments.is_empty() {
        return WordAnchorValidation {
            quality: WordAnchorQuality::Reliable,
            issues,
        };
    }

    let mut missing_words = 0usize;
    for &(index, segment) in &speech_segments {
        if segment.words.is_empty() {
            missing_words += 1;
            issues.push(WordAnchorIssue::MissingWordsOnSpeech {
                segment_index: index,
                start: segment.start,
                end: segment.end,
            });
            continue;
        }
        validate_segment_words(index, segment, audio_duration_s, &mut issues);
    }

    let speech_count = speech_segments.len();
    if speech_count > 0 {
        let missing_ratio = missing_words as f32 / speech_count as f32;
        if missing_ratio > MAX_SPEECH_SEGMENTS_WITHOUT_WORDS_RATIO
            || (missing_words > 0 && speech_count <= 2 && missing_words == speech_count)
        {
            issues.push(WordAnchorIssue::SystemicMissingWords {
                speech_segments: speech_count,
                missing: missing_words,
            });
        }
    }

    let quality = if issues.is_empty() {
        WordAnchorQuality::Reliable
    } else {
        WordAnchorQuality::Unreliable
    };
    WordAnchorValidation { quality, issues }
}

fn is_speech_segment(segment: &Segment) -> bool {
    if segment.text.chars().any(char_has_content) {
        return true;
    }
    let duration = (segment.end - segment.start).max(0.0);
    duration >= MIN_SPEECH_SEGMENT_SECONDS_REQUIRING_WORDS
}

fn validate_segment_words(
    segment_index: usize,
    segment: &Segment,
    audio_duration_s: f32,
    issues: &mut Vec<WordAnchorIssue>,
) {
    let mut prev_start = f32::NEG_INFINITY;
    let mut prev_end = f32::NEG_INFINITY;
    for (word_index, word) in segment.words.iter().enumerate() {
        if !word.start.is_finite() || !word.end.is_finite() {
            issues.push(WordAnchorIssue::NonFiniteTime {
                segment_index,
                word_index,
            });
            continue;
        }
        if word.start < -1e-3 || word.end < -1e-3 {
            issues.push(WordAnchorIssue::NegativeTime {
                segment_index,
                word_index,
            });
        }
        // end must be >= start within float noise (zero-duration is allowed).
        if word.end + 1e-3 < word.start {
            issues.push(WordAnchorIssue::InvertedInterval {
                segment_index,
                word_index,
            });
        }
        if word.start + 1e-3 < prev_start || word.end + 1e-3 < prev_end {
            issues.push(WordAnchorIssue::NonMonotonic {
                segment_index,
                word_index,
            });
        }
        prev_start = word.start;
        prev_end = word.end;
        if audio_duration_s.is_finite()
            && audio_duration_s > 0.0
            && word.end > audio_duration_s + AUDIO_DURATION_TOLERANCE_S
        {
            issues.push(WordAnchorIssue::OutsideAudioDuration {
                segment_index,
                word_index,
                end: word.end,
                audio_duration_s,
            });
        }
    }

    for window in segment.words.windows(2) {
        let gap = window[1].start - window[0].end;
        if gap > MAX_WORDLESS_GAP_S {
            issues.push(WordAnchorIssue::LargeWordlessGap {
                segment_index,
                gap_s: gap,
            });
        }
    }

    let coverage = text_coverage_ratio(segment);
    if coverage < MIN_TEXT_COVERAGE_RATIO {
        issues.push(WordAnchorIssue::InsufficientTextCoverage {
            segment_index,
            coverage,
        });
    }
}

fn text_coverage_ratio(segment: &Segment) -> f32 {
    let segment_chars = content_char_count(&segment.text);
    if segment_chars == 0 {
        return 1.0;
    }
    let word_chars: usize = segment
        .words
        .iter()
        .map(|word| content_char_count(&word.word))
        .sum();
    (word_chars as f32 / segment_chars as f32).min(1.0)
}

fn content_char_count(text: &str) -> usize {
    text.chars().filter(|c| char_has_content(*c)).count()
}

fn char_has_content(c: char) -> bool {
    !c.is_whitespace()
        && !matches!(
            c,
            '.' | '!'
                | '?'
                | ','
                | ';'
                | ':'
                | '"'
                | '\''
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '\u{3002}'
                | '\u{ff01}'
                | '\u{ff1f}'
                | '\u{ff0c}'
                | '\u{3001}'
                | '\u{ff1b}'
                | '\u{ff1a}'
                | '\u{2026}'
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::backend::{Segment, Transcription, WordTimestamp};

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn segment(text: &str, words: Vec<WordTimestamp>) -> Segment {
        let start = words.first().map_or(0.0, |w| w.start);
        let end = words.last().map_or(1.0, |w| w.end);
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }
    }

    fn transcription(segments: Vec<Segment>) -> Transcription {
        Transcription {
            text: segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            segments,
            ..Default::default()
        }
    }

    #[test]
    fn good_word_anchors_are_reliable() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.0, 0.4), word("world", 0.5, 1.0)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(result.is_reliable(), "issues: {:?}", result.issues);
    }

    #[test]
    fn missing_words_on_speech_is_unreliable() {
        let t = transcription(vec![Segment {
            start: 0.0,
            end: 2.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }]);
        let result = validate_word_anchors(&t, 2.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::MissingWordsOnSpeech { .. }))
        );
    }

    #[test]
    fn non_monotonic_words_are_unreliable() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.5, 0.8), word("world", 0.2, 0.4)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::NonMonotonic { .. }))
        );
    }

    #[test]
    fn outside_audio_duration_is_unreliable() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.0, 0.4), word("world", 0.5, 5.0)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::OutsideAudioDuration { .. }))
        );
    }

    #[test]
    fn large_wordless_gap_is_unreliable() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.0, 0.3), word("world", 6.0, 6.4)],
        )]);
        let result = validate_word_anchors(&t, 7.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::LargeWordlessGap { .. }))
        );
    }

    #[test]
    fn sparse_word_text_coverage_is_unreliable() {
        let t = transcription(vec![segment(
            "the quick brown fox jumps over the lazy dog",
            vec![word("the", 0.0, 0.2), word("dog", 2.0, 2.3)],
        )]);
        let result = validate_word_anchors(&t, 3.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::InsufficientTextCoverage { .. }))
        );
    }

    #[test]
    fn empty_transcript_is_reliable() {
        let t = transcription(Vec::new());
        assert!(validate_word_anchors(&t, 0.0).is_reliable());
    }

    #[test]
    fn inverted_word_interval_is_unreliable() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.5, 0.2), word("world", 0.6, 1.0)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::InvertedInterval { .. }))
        );
    }

    #[test]
    fn non_monotonic_word_ends_are_unreliable() {
        // Starts are non-decreasing, but ends go backwards (e.g. mis-aligned
        // sparse anchors that would poison multi-speaker split midpoints).
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.0, 0.9), word("world", 0.2, 0.4)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(!result.is_reliable());
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, WordAnchorIssue::NonMonotonic { .. }))
        );
    }

    #[test]
    fn zero_duration_word_is_allowed() {
        let t = transcription(vec![segment(
            "hello world",
            vec![word("hello", 0.0, 0.4), word("world", 0.5, 0.5)],
        )]);
        let result = validate_word_anchors(&t, 1.0);
        assert!(
            result.is_reliable(),
            "zero-duration trailing word is valid: {:?}",
            result.issues
        );
    }
}
