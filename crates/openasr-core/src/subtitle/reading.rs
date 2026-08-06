//! Speaker-aware reading paragraphs for the manuscript view.
//!
//! Consecutive segments that share the same speaker identity are merged into
//! longer paragraphs so the reading pane is not fragmented into subtitle-sized
//! lines. Word timestamps are concatenated in order; text is joined with a
//! single ASCII space between non-empty pieces (CJK segments that already
//! abut without a space keep their original text, and the join only inserts a
//! space when both sides carry content).
//!
//! ## V1 merge caps (tunable)
//!
//! Same-speaker merges stop when any of these soft limits would be exceeded,
//! so a long monologue still breaks into readable manuscript paragraphs:
//! - [`MAX_PARAGRAPH_SECONDS`]: wall-clock span of the merged paragraph
//! - [`MAX_PARAGRAPH_CHARS`]: content characters in the merged text
//! - [`MAX_PARAGRAPH_PIECES`]: number of original segments fused into one
//!   paragraph (a rough sentence/clause count when the upstream ASR already
//!   segmented at natural boundaries)

use crate::api::backend::{Segment, WordTimestamp};

/// Soft ceiling on merged paragraph duration (seconds). Tunable V1 constant.
pub const MAX_PARAGRAPH_SECONDS: f32 = 45.0;
/// Soft ceiling on merged paragraph content characters. Tunable V1 constant.
pub const MAX_PARAGRAPH_CHARS: usize = 400;
/// Soft ceiling on how many source segments fuse into one paragraph.
pub const MAX_PARAGRAPH_PIECES: usize = 8;

/// Merge consecutive same-speaker attributed segments into reading paragraphs.
///
/// Speaker identity is compared on the resolved display label (`speaker`),
/// falling back to `speaker_label` when `speaker` is absent, so anonymous and
/// named turns do not accidentally fuse. Segments with no speaker identity
/// only merge with adjacent segments that are also unattributed. Merges also
/// stop when a V1 paragraph cap would be exceeded (see module docs).
pub fn merge_reading_segments(segments: Vec<Segment>) -> Vec<Segment> {
    let mut paragraphs: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut piece_counts: Vec<usize> = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.text.trim().is_empty() && segment.words.is_empty() {
            continue;
        }
        if let Some(last) = paragraphs.last_mut() {
            let pieces = *piece_counts.last().unwrap_or(&1);
            if same_speaker(last, &segment) && can_merge(last, &segment, pieces) {
                merge_into(last, segment);
                *piece_counts.last_mut().unwrap() = pieces + 1;
                continue;
            }
        }
        paragraphs.push(segment);
        piece_counts.push(1);
    }
    paragraphs
}

/// Whether fusing `next` into `target` would stay within the V1 paragraph caps.
fn can_merge(target: &Segment, next: &Segment, pieces_in_target: usize) -> bool {
    if pieces_in_target >= MAX_PARAGRAPH_PIECES {
        return false;
    }
    let start = target.start.min(next.start);
    let end = target.end.max(next.end);
    if end - start > MAX_PARAGRAPH_SECONDS {
        return false;
    }
    let merged_chars = estimate_merged_content_chars(&target.text, &next.text);
    merged_chars <= MAX_PARAGRAPH_CHARS
}

fn estimate_merged_content_chars(left: &str, right: &str) -> usize {
    let left = left.trim_end();
    let right = right.trim_start();
    if left.is_empty() {
        return content_char_count(right);
    }
    if right.is_empty() {
        return content_char_count(left);
    }
    // Join space is whitespace and does not count as content.
    content_char_count(left) + content_char_count(right)
}

fn content_char_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

fn same_speaker(left: &Segment, right: &Segment) -> bool {
    speaker_key(left) == speaker_key(right)
}

fn speaker_key(segment: &Segment) -> (Option<&str>, Option<&str>) {
    (segment.speaker.as_deref(), segment.speaker_label.as_deref())
}

fn merge_into(target: &mut Segment, next: Segment) {
    let left = target.text.trim_end();
    let right = next.text.trim_start();
    if left.is_empty() {
        target.text = right.to_string();
    } else if right.is_empty() {
        target.text = left.to_string();
    } else if needs_ascii_space_join(left, right) {
        target.text = format!("{left} {right}");
    } else {
        target.text = format!("{left}{right}");
    }
    target.end = target.end.max(next.end);
    if next.start < target.start {
        target.start = next.start;
    }
    // Prefer identity fields already on the target; fill gaps from `next`.
    if target.speaker.is_none() {
        target.speaker = next.speaker;
    }
    if target.speaker_label.is_none() {
        target.speaker_label = next.speaker_label;
    }
    if target.speaker_person_id.is_none() {
        target.speaker_person_id = next.speaker_person_id;
    }
    if target.speaker_snapshot_label.is_none() {
        target.speaker_snapshot_label = next.speaker_snapshot_label;
    }
    append_words(&mut target.words, next.words);
}

fn needs_ascii_space_join(left: &str, right: &str) -> bool {
    let Some(prev) = left.chars().next_back() else {
        return false;
    };
    let Some(next) = right.chars().next() else {
        return false;
    };
    if prev.is_whitespace() || next.is_whitespace() {
        return false;
    }
    // CJK / fullwidth runs concatenate without a Latin word space.
    if is_cjk_or_fullwidth(prev) || is_cjk_or_fullwidth(next) {
        return false;
    }
    true
}

fn is_cjk_or_fullwidth(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x1100..=0x115F
            | 0x2E80..=0x2EFF
            | 0x3000..=0x303F
            | 0x3040..=0x30FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xAC00..=0xD7A3
            | 0xF900..=0xFAFF
            | 0xFF00..=0xFF60
            | 0x20000..=0x3134F
    )
}

fn append_words(target: &mut Vec<WordTimestamp>, next: Vec<WordTimestamp>) {
    if next.is_empty() {
        return;
    }
    if target.is_empty() {
        *target = next;
        return;
    }
    target.extend(next);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::backend::WordTimestamp;

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn seg(text: &str, start: f32, end: f32, speaker: Option<&str>) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: speaker.map(str::to_string),
            speaker_label: speaker.map(str::to_string),
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: text
                .split_whitespace()
                .enumerate()
                .map(|(i, w)| {
                    let s = start + i as f32 * 0.3;
                    word(w, s, s + 0.25)
                })
                .collect(),
        }
    }

    #[test]
    fn merges_consecutive_same_speaker() {
        let paragraphs = merge_reading_segments(vec![
            seg("hello world", 0.0, 1.0, Some("SPEAKER_00")),
            seg("next sentence", 1.0, 2.0, Some("SPEAKER_00")),
            seg("other voice", 2.0, 3.0, Some("SPEAKER_01")),
        ]);
        assert_eq!(paragraphs.len(), 2);
        assert_eq!(paragraphs[0].text, "hello world next sentence");
        assert_eq!(paragraphs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(paragraphs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(paragraphs[0].words.len(), 4);
    }

    #[test]
    fn does_not_merge_across_speakers() {
        let paragraphs = merge_reading_segments(vec![
            seg("alpha", 0.0, 0.5, Some("A")),
            seg("bravo", 0.5, 1.0, Some("B")),
        ]);
        assert_eq!(paragraphs.len(), 2);
    }

    #[test]
    fn joins_cjk_without_ascii_space() {
        let paragraphs = merge_reading_segments(vec![
            seg("你好", 0.0, 0.5, Some("SPEAKER_00")),
            seg("世界", 0.5, 1.0, Some("SPEAKER_00")),
        ]);
        assert_eq!(paragraphs.len(), 1);
        assert_eq!(paragraphs[0].text, "你好世界");
    }

    #[test]
    fn stops_merging_when_piece_cap_is_reached() {
        let mut segments = Vec::new();
        for i in 0..(MAX_PARAGRAPH_PIECES + 2) {
            let start = i as f32 * 0.5;
            segments.push(seg(&format!("piece{i}"), start, start + 0.4, Some("A")));
        }
        let paragraphs = merge_reading_segments(segments);
        assert!(
            paragraphs.len() >= 2,
            "piece cap must open a new paragraph: {paragraphs:?}"
        );
        // First paragraph absorbs exactly MAX_PARAGRAPH_PIECES source segments.
        let first_words = paragraphs[0].words.len();
        assert_eq!(first_words, MAX_PARAGRAPH_PIECES);
    }

    #[test]
    fn stops_merging_when_duration_cap_is_reached() {
        let paragraphs = merge_reading_segments(vec![
            seg("start", 0.0, 1.0, Some("A")),
            seg("middle", 20.0, 21.0, Some("A")),
            // Spanning from 0 to > MAX_PARAGRAPH_SECONDS must open a new paragraph.
            seg(
                "late",
                MAX_PARAGRAPH_SECONDS + 1.0,
                MAX_PARAGRAPH_SECONDS + 2.0,
                Some("A"),
            ),
        ]);
        assert!(
            paragraphs.len() >= 2,
            "duration cap must open a new paragraph: {paragraphs:?}"
        );
        assert!(paragraphs.last().unwrap().text.contains("late"));
    }
}
