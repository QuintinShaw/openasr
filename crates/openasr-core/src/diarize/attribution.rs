//! "Who said what" attribution — a pure interval function with zero model
//! coupling (plan §3.3).
//!
//! Each transcript segment is assigned the speaker whose turns overlap it the
//! most (WhisperX `assign_word_speakers` semantics). When a segment overlaps
//! turns from MULTIPLE distinct speakers and carries word timestamps, it is
//! split at speaker-turn boundaries snapped to word boundaries: each word goes
//! to the turn covering its midpoint, runs of same-speaker words become
//! separate segments, and the segment text is carved at the corresponding word
//! positions (no text is invented or lost). Without word anchors the text
//! cannot be split faithfully, so the segment keeps the dominant-overlap
//! speaker unchanged — the batch path force-enables word timestamps when
//! diarization is active precisely so this fallback stays rare.

use std::collections::BTreeMap;

use super::contract::{SpeakerId, SpeakerTurn, TimeRange};
use super::debug::diarize_debug_enabled;
use super::enrollment::SpeakerDisplayAssignment;
use crate::Segment;
use crate::api::backend::WordTimestamp;

/// Assign `Segment.speaker` from speaker turns, splitting multi-speaker
/// segments at word-snapped turn boundaries. Returns the (possibly longer)
/// segment list; segment order and the concatenated text are preserved.
///
/// A segment with no overlapping turn is left unassigned (`speaker` unchanged).
/// Ties break toward the lower `SpeakerId` (deterministic). `identities`
/// optionally relabel compatible voice-match profiles with display names while
/// preserving the stable anonymous session label in `speaker_label`.
pub fn assign_speakers(
    turns: &[SpeakerTurn],
    segments: Vec<Segment>,
    identities: &BTreeMap<SpeakerId, SpeakerDisplayAssignment>,
) -> Vec<Segment> {
    let mut output = Vec::with_capacity(segments.len());
    for segment in segments {
        attribute_segment(segment, turns, identities, &mut output);
    }
    output
}

fn attribute_segment(
    mut segment: Segment,
    turns: &[SpeakerTurn],
    identities: &BTreeMap<SpeakerId, SpeakerDisplayAssignment>,
    output: &mut Vec<Segment>,
) {
    let overlap = overlap_by_speaker(&segment, turns);
    let Some(dominant) = dominant_speaker(&overlap) else {
        log_attribution(&segment, &overlap, "unattributed", 1);
        output.push(segment);
        return;
    };
    if overlap.len() == 1 {
        log_attribution(&segment, &overlap, "single-speaker", 1);
        apply_speaker(&mut segment, dominant, identities);
        output.push(segment);
        return;
    }
    if segment.words.is_empty() {
        // No word anchors: the text cannot be split faithfully, keep the
        // dominant-overlap assignment (legacy behavior).
        log_attribution(&segment, &overlap, "multi-speaker-no-words", 1);
        apply_speaker(&mut segment, dominant, identities);
        output.push(segment);
        return;
    }
    match split_segment_at_turn_boundaries(&segment, turns) {
        Some(pieces) => {
            log_attribution(&segment, &overlap, "split", pieces.len());
            for piece in pieces {
                let mut piece_segment = piece.segment;
                apply_speaker(&mut piece_segment, piece.speaker, identities);
                if diarize_debug_enabled() {
                    eprintln!(
                        "openasr_diarize_debug stage=attribution piece start={:.2} end={:.2} speaker={} words={} text={:?}",
                        piece_segment.start,
                        piece_segment.end,
                        piece.speaker.label(),
                        piece_segment.words.len(),
                        piece_segment.text
                    );
                }
                output.push(piece_segment);
            }
        }
        None => {
            // Word/text mismatch — fall back to the unsplit dominant speaker
            // rather than emit text that no longer matches the transcript.
            log_attribution(&segment, &overlap, "split-fallback-dominant", 1);
            apply_speaker(&mut segment, dominant, identities);
            output.push(segment);
        }
    }
}

/// Per-speaker overlap duration between `segment` and `turns`.
fn overlap_by_speaker(segment: &Segment, turns: &[SpeakerTurn]) -> BTreeMap<SpeakerId, f64> {
    let range = TimeRange::new(segment.start as f64, segment.end as f64);
    let mut overlap: BTreeMap<SpeakerId, f64> = BTreeMap::new();
    for turn in turns {
        let shared = range.intersection_s(&turn.range);
        if shared > 0.0 {
            *overlap.entry(turn.speaker).or_insert(0.0) += shared;
        }
    }
    overlap
}

/// Max overlap, ties -> lower `SpeakerId`. `max_by` returns the LAST of equal
/// maxima, so iterate descending (BTreeMap keys reversed) to land on the
/// lowest key on ties.
fn dominant_speaker(overlap: &BTreeMap<SpeakerId, f64>) -> Option<SpeakerId> {
    overlap
        .iter()
        .rev()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(speaker, _)| *speaker)
}

fn apply_speaker(
    segment: &mut Segment,
    speaker: SpeakerId,
    identities: &BTreeMap<SpeakerId, SpeakerDisplayAssignment>,
) {
    if let Some(assignment) = identities.get(&speaker) {
        segment.speaker = Some(assignment.speaker.clone());
        segment.speaker_label = Some(assignment.speaker_label.clone());
        segment.speaker_profile_id = assignment.speaker_profile_id.clone();
    } else {
        segment.speaker = Some(speaker.label());
        segment.speaker_label = None;
        segment.speaker_profile_id = None;
    }
}

struct SplitPiece {
    speaker: SpeakerId,
    segment: Segment,
}

/// Split a multi-speaker segment into per-speaker pieces at word boundaries.
///
/// Each word is assigned the speaker of the turn covering its midpoint (or the
/// nearest turn when the midpoint falls in an inter-turn gap); runs of
/// consecutive same-speaker words become pieces. The original text is carved
/// at the byte position where each run's first word starts, so punctuation and
/// whitespace between runs stay attached to the preceding piece and the
/// concatenation of all piece texts reproduces the original text exactly
/// (modulo the trimmed inter-piece whitespace).
///
/// Returns `None` when a word cannot be located in the segment text in order
/// (the caller falls back to unsplit dominant-speaker attribution).
fn split_segment_at_turn_boundaries(
    segment: &Segment,
    turns: &[SpeakerTurn],
) -> Option<Vec<SplitPiece>> {
    let word_starts = locate_words_in_text(&segment.text, &segment.words)?;
    let mut speakers: Vec<SpeakerId> = segment
        .words
        .iter()
        .map(|word| word_speaker(word, turns))
        .collect::<Option<Vec<_>>>()?;
    // Punctuation-only words carry no voice of their own and CTC-style
    // decoders emit them late (often into the inter-turn silence), so their
    // own timing must not vote: glue them to the preceding word's speaker
    // (or the following word's for segment-leading punctuation).
    let is_punctuation: Vec<bool> = segment
        .words
        .iter()
        .map(|word| word.word.chars().all(|ch| !ch.is_alphanumeric()))
        .collect();
    for index in 0..speakers.len() {
        if !is_punctuation[index] {
            continue;
        }
        let neighbor = (0..index)
            .rev()
            .find(|&j| !is_punctuation[j])
            .or_else(|| (index + 1..speakers.len()).find(|&j| !is_punctuation[j]));
        if let Some(neighbor) = neighbor {
            speakers[index] = speakers[neighbor];
        }
    }

    // Group consecutive words by speaker: (speaker, first_word, last_word).
    let mut runs: Vec<(SpeakerId, usize, usize)> = Vec::new();
    for (index, speaker) in speakers.iter().copied().enumerate() {
        match runs.last_mut() {
            Some((last_speaker, _, last_index)) if *last_speaker == speaker => {
                *last_index = index;
            }
            _ => runs.push((speaker, index, index)),
        }
    }
    if runs.len() == 1 {
        // All word midpoints agree on one speaker: no split, just assign it.
        return Some(vec![SplitPiece {
            speaker: runs[0].0,
            segment: segment.clone(),
        }]);
    }

    let mut pieces = Vec::with_capacity(runs.len());
    for (run_index, (speaker, first_word, last_word)) in runs.iter().copied().enumerate() {
        let is_first = run_index == 0;
        let is_last = run_index + 1 == runs.len();
        let text_start = if is_first { 0 } else { word_starts[first_word] };
        let text_end = if is_last {
            segment.text.len()
        } else {
            word_starts[runs[run_index + 1].1]
        };
        let words = segment.words[first_word..=last_word].to_vec();
        let start = if is_first {
            segment.start
        } else {
            words
                .first()
                .map(|word| word.start)
                .unwrap_or(segment.start)
        };
        let end = if is_last {
            segment.end
        } else {
            words.last().map(|word| word.end).unwrap_or(segment.end)
        };
        pieces.push(SplitPiece {
            speaker,
            segment: Segment {
                start,
                end: end.max(start),
                text: segment.text[text_start..text_end].trim().to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words,
            },
        });
    }
    // A glued punctuation word emitted late into the inter-turn gap can drag a
    // piece's end past the next piece's start. Clamp so pieces never overlap
    // (and never invert: end stays >= start).
    for index in 0..pieces.len().saturating_sub(1) {
        let next_start = pieces[index + 1].segment.start;
        let piece = &mut pieces[index].segment;
        if piece.end > next_start {
            piece.end = next_start.max(piece.start);
        }
    }
    Some(pieces)
}

/// Byte offset in `text` where each word starts, located sequentially.
/// `None` when any word cannot be found in order (defensive: words are derived
/// from the text by the decoders, so this only trips on a contract drift).
fn locate_words_in_text(text: &str, words: &[WordTimestamp]) -> Option<Vec<usize>> {
    locate_word_starts(text, words.iter().map(|word| word.word.as_str()))
}

fn locate_word_starts<'a>(text: &str, words: impl Iterator<Item = &'a str>) -> Option<Vec<usize>> {
    let mut starts = Vec::new();
    let mut cursor = 0usize;
    for word in words {
        let needle = word.trim();
        if needle.is_empty() {
            starts.push(cursor);
            continue;
        }
        let found = text.get(cursor..)?.find(needle)?;
        starts.push(cursor + found);
        cursor = cursor + found + needle.len();
    }
    Some(starts)
}

/// Word-snapped tail split of a finalized realtime transcript at an acoustic
/// speaker-change estimate, for retroactive speaker reattribution.
///
/// The realtime change-point detector lags the acoustic change by up to its
/// analysis window, so text decoded during the lag is finalized into the OLD
/// speaker's segment. Given the estimated change time, this carves the
/// transcript at the first word whose midpoint falls after the change point
/// (punctuation-only words glue to the preceding word, mirroring the batch
/// attribution split). `max_tail_ms` bounds how much trailing speech may be
/// reattributed (≈ the detection lag), so a bad change estimate cannot move
/// more than the lag window.
///
/// Returns `None` when the split is not faithful or not useful: no word
/// timestamps, the change point precedes all words / trails all words, or a
/// word cannot be located in the text in order.
pub fn split_transcript_tail_at_change(
    text: &str,
    words: &[crate::realtime::RealtimeTranscriptWord],
    change_ms: u64,
    max_tail_ms: u64,
) -> Option<TranscriptTailSplit> {
    if text.trim().is_empty() || words.is_empty() {
        return None;
    }
    let end_ms = words.last()?.end_ms;
    let change_ms = change_ms.max(end_ms.saturating_sub(max_tail_ms));
    // Assign each word a side by midpoint; punctuation-only words carry no
    // voice of their own and often land late, so they follow their
    // preceding word.
    let mut prev_moved = false;
    let mut last_kept: Option<usize> = None;
    let mut any_moved = false;
    for (index, word) in words.iter().enumerate() {
        let punctuation_only = word.word.chars().all(|ch| !ch.is_alphanumeric());
        let moved = if punctuation_only {
            prev_moved
        } else {
            word.start_ms.midpoint(word.end_ms) > change_ms
        };
        prev_moved = moved;
        if moved {
            any_moved = true;
        } else {
            last_kept = Some(index);
        }
    }
    if !any_moved {
        return None;
    }
    // Only a contiguous TRAILING run can be reattributed; words after the
    // last kept word form that run.
    let first_moved = match last_kept {
        // Everything would move: that is not a tail split (the whole segment
        // already gets the new speaker through the normal label path).
        None => return None,
        Some(last_kept) if last_kept + 1 >= words.len() => return None,
        Some(last_kept) => last_kept + 1,
    };
    let starts = locate_word_starts(text, words.iter().map(|word| word.word.as_str()))?;
    let cut = starts[first_moved];
    let kept_text = text[..cut].trim();
    let moved_text = text[cut..].trim();
    if kept_text.is_empty() || moved_text.is_empty() {
        return None;
    }
    Some(TranscriptTailSplit {
        kept_text: kept_text.to_string(),
        kept_end_ms: words[first_moved - 1].end_ms,
        moved_text: moved_text.to_string(),
        moved_from_word: first_moved,
        moved_start_ms: words[first_moved].start_ms,
    })
}

/// Result of [`split_transcript_tail_at_change`]: the original text carved at
/// a word boundary; nothing is invented or lost.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptTailSplit {
    /// Text that stays with the OLD speaker's segment.
    pub kept_text: String,
    /// End of the last kept word — the revised end of the old segment.
    pub kept_end_ms: u64,
    /// Text reattributed to the NEW speaker.
    pub moved_text: String,
    /// Index of the first moved word in the input `words`.
    pub moved_from_word: usize,
    /// Start of the first moved word — the start of the reattributed piece.
    pub moved_start_ms: u64,
}

/// The speaker of the turn covering the word's midpoint, falling back to the
/// turn whose range is nearest to the midpoint (inter-turn gaps and turn-edge
/// drift). `None` only when `turns` is empty.
fn word_speaker(word: &WordTimestamp, turns: &[SpeakerTurn]) -> Option<SpeakerId> {
    let midpoint = f64::from(word.start + word.end) / 2.0;
    let mut nearest: Option<(f64, SpeakerId)> = None;
    for turn in turns {
        if midpoint >= turn.range.start_s && midpoint < turn.range.end_s {
            return Some(turn.speaker);
        }
        let distance = if midpoint < turn.range.start_s {
            turn.range.start_s - midpoint
        } else {
            midpoint - turn.range.end_s
        };
        let closer = match nearest {
            Some((best, _)) => distance < best,
            None => true,
        };
        if closer {
            nearest = Some((distance, turn.speaker));
        }
    }
    nearest.map(|(_, speaker)| speaker)
}

fn log_attribution(
    segment: &Segment,
    overlap: &BTreeMap<SpeakerId, f64>,
    decision: &str,
    pieces: usize,
) {
    if !diarize_debug_enabled() {
        return;
    }
    let overlaps = overlap
        .iter()
        .map(|(speaker, seconds)| format!("{}:{seconds:.2}s", speaker.label()))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "openasr_diarize_debug stage=attribution segment start={:.2} end={:.2} words={} overlaps=[{}] decision={} pieces={}",
        segment.start,
        segment.end,
        segment.words.len(),
        overlaps,
        decision,
        pieces
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            range: TimeRange::new(start, end),
            speaker: SpeakerId(spk),
            overlap: false,
        }
    }

    fn seg(start: f32, end: f32) -> Segment {
        Segment {
            start,
            end,
            text: "x".into(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        }
    }

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn worded_seg(start: f32, end: f32, text: &str, words: Vec<WordTimestamp>) -> Segment {
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
    fn assigns_dominant_overlap_speaker() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 5.0, 1)];
        let segs = assign_speakers(
            &turns,
            vec![seg(0.0, 1.5), seg(2.5, 4.0), seg(1.8, 2.4)],
            &BTreeMap::new(),
        );
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].speaker_label, None);
        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].speaker_label, None);
        // straddling segment without words: 0.2s in spk0, 0.4s in spk1 ->
        // spk1 dominates, no split possible.
        assert_eq!(segs[2].speaker.as_deref(), Some("SPEAKER_01"));
    }

    #[test]
    fn exact_overlap_tie_breaks_to_lower_speaker() {
        // segment 1.0-3.0 overlaps spk0 (1.0-2.0) and spk1 (2.0-3.0) by 1.0s each.
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 3.0, 1)];
        let segs = assign_speakers(&turns, vec![seg(1.0, 3.0)], &BTreeMap::new());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
    }

    #[test]
    fn voice_match_relabels_speaker_and_preserves_session_label() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 5.0, 1)];
        let mut identities = BTreeMap::new();
        identities.insert(
            SpeakerId(1),
            SpeakerDisplayAssignment {
                speaker_id: SpeakerId(1),
                speaker: "Alice".to_string(),
                speaker_label: "SPEAKER_01".to_string(),
                speaker_profile_id: Some("vp_aaaaaaaaaaaaaaaa".to_string()),
            },
        );
        let segs = assign_speakers(&turns, vec![seg(0.0, 1.5), seg(2.5, 4.0)], &identities);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].speaker_label, None);
        assert_eq!(segs[1].speaker.as_deref(), Some("Alice"));
        assert_eq!(segs[1].speaker_label.as_deref(), Some("SPEAKER_01"));
        assert_eq!(
            segs[1].speaker_profile_id.as_deref(),
            Some("vp_aaaaaaaaaaaaaaaa")
        );
    }

    #[test]
    fn no_overlap_leaves_speaker_unassigned() {
        let turns = vec![turn(0.0, 1.0, 0)];
        let segs = assign_speakers(&turns, vec![seg(2.0, 3.0)], &BTreeMap::new());
        assert_eq!(segs[0].speaker, None);
    }

    #[test]
    fn empty_turns_is_noop() {
        let segs = assign_speakers(&[], vec![seg(0.0, 1.0)], &BTreeMap::new());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, None);
    }

    #[test]
    fn multi_speaker_segment_splits_at_word_snapped_turn_boundaries() {
        // One monolithic segment covering an A / B / A conversation (the X-ASR
        // batch shape): the split must produce three segments with the right
        // speakers, word-boundary-snapped times, and exactly carved text.
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1), turn(4.0, 6.0, 0)];
        let segment = worded_seg(
            0.0,
            6.0,
            "hello there, general kenobi! you are bold",
            vec![
                word("hello", 0.2, 0.8),
                word("there,", 0.9, 1.6),
                word("general", 2.2, 2.9),
                word("kenobi!", 3.0, 3.7),
                word("you", 4.2, 4.5),
                word("are", 4.6, 4.9),
                word("bold", 5.0, 5.6),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 3, "A/B/A turns must yield three segments");

        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, "hello there,");
        // First piece keeps the segment start; interior boundary snaps to the
        // last word of the run.
        assert_eq!(segs[0].start, 0.0);
        assert_eq!(segs[0].end, 1.6);
        assert_eq!(segs[0].words.len(), 2);

        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].text, "general kenobi!");
        // Interior piece is bounded by its own words on both sides.
        assert_eq!(segs[1].start, 2.2);
        assert_eq!(segs[1].end, 3.7);
        assert_eq!(segs[1].words.len(), 2);

        assert_eq!(segs[2].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[2].text, "you are bold");
        // Last piece keeps the segment end.
        assert_eq!(segs[2].start, 4.2);
        assert_eq!(segs[2].end, 6.0);
        assert_eq!(segs[2].words.len(), 3);

        // No text invented or lost across the split.
        let rejoined = segs
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(rejoined, "hello there, general kenobi! you are bold");
    }

    #[test]
    fn cjk_text_without_spaces_is_carved_at_word_starts() {
        // CJK words carry no whitespace; carving at word byte starts must keep
        // the inter-run punctuation with the left piece and lose nothing.
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let segment = worded_seg(
            0.0,
            4.0,
            "你好，今天",
            vec![
                word("你", 0.4, 0.8),
                word("好", 0.9, 1.3),
                word("今", 2.3, 2.7),
                word("天", 2.8, 3.2),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, "你好，");
        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].text, "今天");
    }

    #[test]
    fn word_in_inter_turn_gap_goes_to_nearest_turn() {
        // "uh" sits in the 2.0-3.0 gap, 0.2s from spk0's turn end and 0.6s from
        // spk1's start: it must ride with spk0.
        let turns = vec![turn(0.0, 2.0, 0), turn(3.0, 5.0, 1)];
        let segment = worded_seg(
            0.0,
            5.0,
            "yes uh okay",
            vec![
                word("yes", 1.0, 1.8),
                word("uh", 2.1, 2.3),
                word("okay", 3.4, 4.0),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, "yes uh");
        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].text, "okay");
    }

    #[test]
    fn late_emitted_punctuation_sticks_with_the_preceding_speaker() {
        // CTC decoders emit terminal punctuation late, often into the
        // inter-turn gap nearer the NEXT speaker's turn ("？" at 5.9s between a
        // user turn ending 3.5s and a video turn starting 5.8s). It must stay
        // with the utterance that produced it — and because the glued "？"
        // ends (6.15s) past the next run's first word start (6.0s), the split
        // must clamp the piece boundary so pieces never overlap.
        let turns = vec![turn(1.4, 3.5, 0), turn(5.8, 13.9, 1)];
        let segment = worded_seg(
            0.0,
            14.0,
            "可以听见吗？一旦",
            vec![
                word("可", 3.0, 3.1),
                word("以", 3.1, 3.2),
                word("听", 3.3, 3.4),
                word("见", 3.6, 3.7),
                word("吗", 3.9, 4.0),
                word("？", 5.9, 6.15),
                word("一", 6.0, 6.1),
                word("旦", 6.1, 6.2),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, "可以听见吗？");
        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].text, "一旦");
        // The late "？" must not drag piece 0 past piece 1's start: its end is
        // clamped to the next piece's start.
        assert_eq!(segs[0].end, 6.0);
        assert_eq!(segs[1].start, 6.0);
        // Strict non-overlap and no inverted spans across all pieces.
        for piece in &segs {
            assert!(
                piece.start <= piece.end,
                "piece start must not exceed end: {piece:?}"
            );
        }
        for pair in segs.windows(2) {
            assert!(
                pair[0].end <= pair[1].start,
                "split pieces must not overlap: {pair:?}"
            );
        }
    }

    #[test]
    fn single_speaker_segment_with_words_is_untouched() {
        let turns = vec![turn(0.0, 3.0, 0)];
        let original = worded_seg(
            0.0,
            3.0,
            "all one speaker",
            vec![
                word("all", 0.2, 0.7),
                word("one", 0.9, 1.4),
                word("speaker", 1.6, 2.4),
            ],
        );
        let segs = assign_speakers(&turns, vec![original.clone()], &BTreeMap::new());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, original.text);
        assert_eq!(segs[0].start, original.start);
        assert_eq!(segs[0].end, original.end);
        assert_eq!(segs[0].words, original.words);
    }

    #[test]
    fn words_agreeing_on_one_speaker_do_not_split_despite_multi_turn_overlap() {
        // The segment time-range grazes spk1's turn, but every word midpoint is
        // inside spk0's turns: assign spk0, one segment.
        let turns = vec![turn(0.0, 3.0, 0), turn(3.0, 3.4, 1)];
        let segment = worded_seg(
            0.0,
            3.3,
            "only speaker zero",
            vec![
                word("only", 0.2, 0.7),
                word("speaker", 1.0, 1.7),
                word("zero", 2.0, 2.6),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segs[0].text, "only speaker zero");
    }

    #[test]
    fn word_text_mismatch_falls_back_to_dominant_without_split() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let segment = worded_seg(
            0.0,
            4.0,
            "actual transcript text",
            vec![word("unrelated", 0.5, 1.0), word("words", 2.5, 3.0)],
        );
        let segs = assign_speakers(&turns, vec![segment], &BTreeMap::new());
        assert_eq!(segs.len(), 1, "mismatched words must not split the text");
        assert_eq!(segs[0].text, "actual transcript text");
        assert!(segs[0].speaker.is_some());
    }

    #[test]
    fn split_pieces_carry_voice_match_identities() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let mut identities = BTreeMap::new();
        identities.insert(
            SpeakerId(0),
            SpeakerDisplayAssignment {
                speaker_id: SpeakerId(0),
                speaker: "Alice".to_string(),
                speaker_label: "SPEAKER_00".to_string(),
                speaker_profile_id: Some("vp_bbbbbbbbbbbbbbbb".to_string()),
            },
        );
        let segment = worded_seg(
            0.0,
            4.0,
            "hi there video audio",
            vec![
                word("hi", 0.3, 0.6),
                word("there", 0.8, 1.4),
                word("video", 2.2, 2.8),
                word("audio", 3.0, 3.6),
            ],
        );
        let segs = assign_speakers(&turns, vec![segment], &identities);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker.as_deref(), Some("Alice"));
        assert_eq!(segs[0].speaker_label.as_deref(), Some("SPEAKER_00"));
        assert_eq!(
            segs[0].speaker_profile_id.as_deref(),
            Some("vp_bbbbbbbbbbbbbbbb")
        );
        assert_eq!(segs[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segs[1].speaker_label, None);
        assert_eq!(segs[1].speaker_profile_id, None);
    }

    fn rt_word(word: &str, start_ms: u64, end_ms: u64) -> crate::realtime::RealtimeTranscriptWord {
        crate::realtime::RealtimeTranscriptWord {
            word: word.to_string(),
            start_ms,
            end_ms,
            confidence: None,
        }
    }

    #[test]
    fn tail_split_carves_cjk_text_at_the_change_point_word() {
        // "那现在又回到了我。" decoded into the old speaker's segment during
        // the detection lag: a change at 25.5 s moves the trailing words.
        let words = vec![
            rt_word("还特意", 22_000, 23_000),
            rt_word("给出了", 23_000, 24_000),
            rt_word("具体的过程", 24_000, 25_400),
            rt_word("那现在", 25_700, 26_500),
            rt_word("又回到了我", 26_500, 27_600),
        ];
        let split = split_transcript_tail_at_change(
            "还特意给出了具体的过程。那现在又回到了我。",
            &words,
            25_500,
            3_000,
        )
        .expect("trailing words after the change point must split");
        assert_eq!(split.kept_text, "还特意给出了具体的过程。");
        assert_eq!(split.kept_end_ms, 25_400);
        assert_eq!(split.moved_text, "那现在又回到了我。");
        assert_eq!(split.moved_from_word, 3);
        assert_eq!(split.moved_start_ms, 25_700);
    }

    #[test]
    fn tail_split_is_bounded_by_the_max_tail_window() {
        // A change estimate far before the end may only move ~max_tail_ms of
        // trailing speech, so a bad estimate cannot rewrite the segment.
        let words: Vec<_> = (0..10)
            .map(|i| rt_word("w", i * 1_000, i * 1_000 + 900))
            .collect();
        let text = "w w w w w w w w w w";
        let split = split_transcript_tail_at_change(text, &words, 0, 3_000)
            .expect("bounded split still moves the trailing window");
        // end=9 900 ms, bound clamps change to 6 900: words 7..10 move.
        assert_eq!(split.moved_from_word, 7);
    }

    #[test]
    fn tail_split_rejects_unhelpful_or_unfaithful_cuts() {
        let words = vec![rt_word("hello", 0, 500), rt_word("there", 600, 1_200)];
        // Change after all words: nothing to move.
        assert_eq!(
            split_transcript_tail_at_change("hello there", &words, 5_000, 3_000),
            None
        );
        // Change before all words with a window covering everything: not a
        // tail split (the whole segment is the new speaker's).
        assert_eq!(
            split_transcript_tail_at_change("hello there", &words, 0, 60_000),
            None
        );
        // No word anchors: cannot carve faithfully.
        assert_eq!(
            split_transcript_tail_at_change("hello there", &[], 500, 3_000),
            None
        );
        // Word missing from the text (contract drift): refuse to carve.
        assert_eq!(
            split_transcript_tail_at_change("hello world", &words, 500, 3_000),
            None
        );
    }

    #[test]
    fn tail_split_glues_trailing_punctuation_to_its_word() {
        let words = vec![
            rt_word("具体的过程", 24_000, 25_400),
            rt_word("那现在", 25_700, 26_500),
            rt_word("。", 27_900, 27_950),
        ];
        // The lone punctuation word lands late but follows its preceding
        // word's side, so a change at 26.6 s (after 那现在's midpoint) moves
        // nothing -- punctuation alone is never reattributed.
        assert_eq!(
            split_transcript_tail_at_change("具体的过程。那现在。", &words, 26_600, 3_000),
            None
        );
    }
}
