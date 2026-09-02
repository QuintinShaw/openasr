//! Fail-closed checks for a forced-alignment of an external transcript.
//!
//! Forced alignment always maps the given words onto the audio; it does not
//! score semantic agreement the way ASR WER would. These checks reject
//! degenerate outputs (empty word lists, collapsed timestamp bins, unreliable
//! anchors) so a mismatched script cannot be silently exported as a timeline.

use super::anchors::{WordAnchorIssue, validate_word_anchors};
use crate::api::backend::{Transcription, WordTimestamp};

/// Forced-aligner classify head uses 80 ms bins. Unique-start collapse is
/// measured in these bins so the threshold is independent of floating point.
pub const TIMESTAMP_BIN_S: f32 = 0.080;

/// Word lists shorter than this are too small for a collapse ratio to be
/// meaningful (a handful of stacked bins can be a real short utterance).
pub const MIN_WORDS_FOR_COLLAPSE_CHECK: usize = 8;

/// Minimum unique start-bin count / word count before a timeline is treated
/// as collapsed (most words piled onto the same few bins).
pub const MIN_UNIQUE_START_BIN_RATIO: f32 = 0.25;

/// Maximum fraction of words that may have zero duration (`start == end`).
pub const MAX_ZERO_DURATION_WORD_RATIO: f32 = 0.50;

/// Why a forced-alignment of an external transcript was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum ForcedAlignmentMismatch {
    EmptyWordList,
    UnreliableAnchors {
        issues: Vec<WordAnchorIssue>,
    },
    CollapsedTimeline {
        unique_starts: usize,
        word_count: usize,
    },
    TooManyZeroDurationWords {
        zero_duration: usize,
        word_count: usize,
    },
}

impl std::fmt::Display for ForcedAlignmentMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyWordList => {
                write!(
                    f,
                    "transcript produced no alignable words after normalization (letters, numbers, and apostrophes are kept; other punctuation is stripped)"
                )
            }
            Self::UnreliableAnchors { issues } => {
                write!(
                    f,
                    "aligned word timestamps failed validation ({} issue{}); the transcript and audio are treated as a mismatch rather than exporting a fabricated timeline",
                    issues.len(),
                    if issues.len() == 1 { "" } else { "s" }
                )
            }
            Self::CollapsedTimeline {
                unique_starts,
                word_count,
            } => write!(
                f,
                "aligned timeline collapsed: {unique_starts} unique start bins for {word_count} words (severe transcript/audio mismatch)"
            ),
            Self::TooManyZeroDurationWords {
                zero_duration,
                word_count,
            } => write!(
                f,
                "aligned timeline is degenerate: {zero_duration}/{word_count} words have zero duration (severe transcript/audio mismatch)"
            ),
        }
    }
}

/// Reject a forced-alignment result that cannot be trusted as a timeline.
pub fn reject_degenerate_forced_alignment(
    transcription: &Transcription,
    audio_duration_s: f32,
) -> Result<(), ForcedAlignmentMismatch> {
    let words = alignment_words(transcription);
    if words.is_empty() {
        return Err(ForcedAlignmentMismatch::EmptyWordList);
    }

    let validation = validate_word_anchors(transcription, audio_duration_s);
    if !validation.is_reliable() {
        return Err(ForcedAlignmentMismatch::UnreliableAnchors {
            issues: validation.issues,
        });
    }

    let word_count = words.len();
    let zero_duration = words
        .iter()
        .filter(|word| (word.end - word.start).abs() <= f32::EPSILON)
        .count();
    if word_count > 0 {
        let zero_ratio = zero_duration as f32 / word_count as f32;
        if zero_ratio > MAX_ZERO_DURATION_WORD_RATIO {
            return Err(ForcedAlignmentMismatch::TooManyZeroDurationWords {
                zero_duration,
                word_count,
            });
        }
    }

    if word_count >= MIN_WORDS_FOR_COLLAPSE_CHECK {
        let unique_starts = unique_start_bins(&words);
        let ratio = unique_starts as f32 / word_count as f32;
        if ratio < MIN_UNIQUE_START_BIN_RATIO {
            return Err(ForcedAlignmentMismatch::CollapsedTimeline {
                unique_starts,
                word_count,
            });
        }
    }

    Ok(())
}

fn alignment_words(transcription: &Transcription) -> Vec<&WordTimestamp> {
    transcription
        .segments
        .iter()
        .flat_map(|segment| segment.words.iter())
        .collect()
}

fn unique_start_bins(words: &[&WordTimestamp]) -> usize {
    let mut bins: Vec<i64> = words
        .iter()
        .map(|word| (word.start / TIMESTAMP_BIN_S).round() as i64)
        .collect();
    bins.sort_unstable();
    bins.dedup();
    bins.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::backend::{Segment, Transcription};

    fn transcription_with_words(words: Vec<WordTimestamp>) -> Transcription {
        let text = words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let end = words.last().map(|word| word.end).unwrap_or(0.0);
        Transcription {
            text: text.clone(),
            language: Some("en".into()),
            segments: vec![Segment {
                start: 0.0,
                end,
                text,
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words,
            }],
            ..Default::default()
        }
    }

    fn spread_words(count: usize, duration_s: f32) -> Vec<WordTimestamp> {
        let step = duration_s / count as f32;
        (0..count)
            .map(|index| {
                let start = index as f32 * step;
                WordTimestamp {
                    word: format!("w{index}"),
                    start,
                    end: start + step * 0.8,
                    confidence: None,
                }
            })
            .collect()
    }

    #[test]
    fn empty_word_list_is_rejected() {
        let transcription = Transcription {
            text: "hello".into(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello".into(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            ..Default::default()
        };
        let error = reject_degenerate_forced_alignment(&transcription, 1.0)
            .expect_err("empty words must fail");
        assert!(matches!(error, ForcedAlignmentMismatch::EmptyWordList));
    }

    #[test]
    fn collapsed_bins_are_rejected() {
        let words = (0..12)
            .map(|index| WordTimestamp {
                word: format!("w{index}"),
                start: 0.0,
                end: 0.08,
                confidence: None,
            })
            .collect();
        let transcription = transcription_with_words(words);
        let error = reject_degenerate_forced_alignment(&transcription, 11.0)
            .expect_err("collapsed bins must fail");
        assert!(
            matches!(error, ForcedAlignmentMismatch::CollapsedTimeline { .. }),
            "got {error}"
        );
    }

    #[test]
    fn zero_duration_majority_is_rejected() {
        let words = (0..10)
            .map(|index| {
                let start = index as f32 * 0.2;
                WordTimestamp {
                    word: format!("w{index}"),
                    start,
                    end: start,
                    confidence: None,
                }
            })
            .collect();
        let transcription = transcription_with_words(words);
        let error = reject_degenerate_forced_alignment(&transcription, 2.0)
            .expect_err("zero-duration majority must fail");
        assert!(matches!(
            error,
            ForcedAlignmentMismatch::TooManyZeroDurationWords { .. }
        ));
    }

    #[test]
    fn spread_jfk_like_timeline_is_accepted() {
        let transcription = transcription_with_words(spread_words(21, 11.0));
        reject_degenerate_forced_alignment(&transcription, 11.0)
            .expect("a spread timeline must pass");
    }
}
