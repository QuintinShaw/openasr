//! Parses moss-transcribe-diarize's inline `[start][end][SNN]text` speaker /
//! time-anchor markup -- ordinary BPE tokens the Qwen3 decoder emits as
//! literal transcript *characters* (see the module doc) -- into the shared
//! [`Segment`] speaker-turn shape (`speaker`/`start`/`end`/`text`) the rest of
//! the engine already understands.
//!
//! This mirrors the one existing precedent for turning a family's own inline
//! diarization markup into `Segment`s:
//! `cohere::decoder_graph::cohere_diarized_segments_from_generated_tokens`.
//! The shapes differ because the underlying signal differs -- cohere's
//! `<|spltoken0|>` / `<|t:2.4|>` are dedicated vocabulary entries the tokenizer
//! can recognize by token id before any text decode, so a malformed stream is
//! not really reachable; moss-td's tags are ordinary characters the model
//! free-generates as part of its text, so a malformed tag stream is a real,
//! reachable failure mode this parser must handle without guessing. Both
//! parsers make the same "never invent a speaker" call (see
//! [`parse_moss_td_speaker_segments`]'s fail-closed policy below) and both
//! produce the same `Segment` shape, which is what will let a future
//! `DiarizerBackend` trait extraction treat "VAD+embedder turns" and
//! "in-decoder self-diarization tags" as two producers of one interface
//! without reshaping either family's output again.
//!
//! That future two-producer interface will, however, want fields neither
//! source populates today: a per-turn confidence (cf.
//! [`crate::api::backend::WordTimestamp::confidence`], already an `Option`) and
//! an `overlap` flag (cf. [`crate::diarize::contract::SpeakerTurn::overlap`],
//! which the VAD path sets but this in-decoder path has no signal for).
//! [`Segment`] carries neither, and moss-td asserts neither, so nothing is lost
//! now -- but a `DiarizerBackend` extraction that wants to keep the VAD path's
//! overlap/confidence must grow [`Segment`] additively (a new
//! `Option`/`#[serde(default)]` field) rather than reshape it. Flagged here so
//! that growth stays a conscious additive step, not a breaking change.
//!
//! # Tags are ordinary characters: an inherent ambiguity
//!
//! Because moss-td's `[t]`/`[Sxx]` markers are ordinary transcript characters
//! rather than reserved control tokens, this parser cannot tell a structural
//! tag apart from transcript content that merely *looks* like one. If the
//! decoded text itself contains a bracketed number (say the model wrote
//! `meeting at [3.30] pm`) that span is consumed as a time anchor and the
//! segment splits there; a bracketed `[Sxx]` sitting inside content is likewise
//! read as a speaker change and absorbed. This is unavoidable given the format
//! and is deliberately accepted: the worst case is a mis-split or an absorbed
//! bracket, never a panic and never a dropped transcript -- and if such a stray
//! bracket makes time run backwards or strands text before an anchor, the
//! fail-closed policy below degrades the whole decode back to the untouched raw
//! text. The reference decode does not emit bracketed numerics as free text, so
//! this stays a theoretical edge, but callers must treat the segment overlay as
//! best-effort structure over a plain-text signal, not a guaranteed lossless
//! parse of arbitrary transcript content.
//!
//! # Grammar
//!
//! Observed from the reference HF decode (`docs/model-audits/
//! moss-transcribe-diarize.md`, pinned in `executor.rs`'s golden fixtures): a
//! segment opens with a numeric time anchor `[t]`, a speaker tag `[Sxx]`, then
//! free text. The anchor that closes one segment doubles as the opener of the
//! next, so two anchors appear back to back between segments, e.g.
//! `...for you,[7.71][8.12][S01] ask what...`. A final trailing anchor closes
//! the last segment.
//!
//! # Fail-closed policy
//!
//! Any deviation from that grammar -- an unterminated `[`, a tag that is
//! neither a finite non-negative float nor `Sxx`, a time anchor that goes
//! backwards, or text/a speaker change emitted before the first anchor or
//! speaker tag has ever appeared -- returns a typed
//! [`MossTdSpeakerSegmentParseError`] instead of guessing at a boundary or
//! silently dropping the offending span. The caller (`executor.rs`) treats
//! any such error, and the "well-formed but zero speaker tags found" case, the
//! same way: this decode's tag structure is not trustworthy, so it falls back
//! to the pre-existing single speaker-less segment carrying the untouched raw
//! text. The transcript text itself is never dropped or rewritten -- only the
//! speaker/segment overlay is withheld -- which mirrors this crate's existing
//! diarization degrade path (`SpeakerAttribution` with empty turns is a
//! silent no-op, never an error surfaced to the caller).
//!
//! A speaker-number *gap* (e.g. `S01` then `S05` with no `S02`-`S04` in
//! between) is deliberately NOT an error: the model's own numbering is passed
//! through verbatim, on the same "never invent speakers" principle as
//! `cohere_diarized_segments_from_generated_tokens`'s
//! `does_not_invent_speakers` test -- renumbering to close the gap would
//! fabricate an ordering/count the model never asserted.

use crate::api::backend::Segment;

/// Why [`parse_moss_td_speaker_segments`] gave up rather than guess. See the
/// module doc's "Fail-closed policy" for how each variant is triggered and
/// what the caller does with it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MossTdSpeakerSegmentParseError {
    /// A `[` was never followed by a matching `]`.
    UnclosedTag,
    /// A bracketed tag's content was neither a finite, non-negative time
    /// value nor an `Sxx` speaker marker.
    UnknownTag { raw: String },
    /// A later time anchor is smaller than an earlier one, e.g.
    /// `[2.0]...[1.0]`.
    TimeWentBackwards { previous: f32, next: f32 },
    /// Text (or a speaker tag) appeared before the stream ever produced an
    /// opening time anchor, so no `start` value exists to attribute it to.
    TextBeforeTimestamp,
    /// Text appeared before any `[Sxx]` speaker tag was seen, so there is no
    /// speaker to attribute it to.
    TextBeforeSpeaker,
}

impl std::fmt::Display for MossTdSpeakerSegmentParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnclosedTag => write!(f, "unterminated '[' in moss-td tag stream"),
            Self::UnknownTag { raw } => write!(f, "unrecognized moss-td tag content '{raw}'"),
            Self::TimeWentBackwards { previous, next } => write!(
                f,
                "moss-td time anchor went backwards: {previous} -> {next}"
            ),
            Self::TextBeforeTimestamp => {
                write!(f, "moss-td text appeared before any time anchor")
            }
            Self::TextBeforeSpeaker => {
                write!(f, "moss-td text appeared before any speaker tag")
            }
        }
    }
}

impl std::error::Error for MossTdSpeakerSegmentParseError {}

enum MossTdTag {
    Anchor(f32),
    Speaker(String),
}

/// Parses one bracketed tag's inner content (without the `[`/`]`). Speaker
/// tags are tried first since `"S01"` would otherwise also fail the float
/// parse and fall through anyway; trying it first just avoids the wasted
/// `parse::<f32>()` call on the common case.
fn parse_tag_content(raw: &str) -> Result<MossTdTag, MossTdSpeakerSegmentParseError> {
    if let Some(digits) = raw.strip_prefix('S')
        && !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        // Digits are ASCII-checked above, so this only fails on overflow of a
        // number no real pack would ever emit; treat that the same as any
        // other unrecognized tag rather than panicking.
        return digits
            .parse::<u32>()
            .map(|number| MossTdTag::Speaker(format!("SPEAKER_{number:02}")))
            .map_err(|_| MossTdSpeakerSegmentParseError::UnknownTag {
                raw: raw.to_string(),
            });
    }
    if let Ok(value) = raw.trim().parse::<f32>()
        && value.is_finite()
        && value >= 0.0
    {
        return Ok(MossTdTag::Anchor(value));
    }
    Err(MossTdSpeakerSegmentParseError::UnknownTag {
        raw: raw.to_string(),
    })
}

fn plain_segment(speaker: String, start: f32, end: f32, text: String) -> Segment {
    Segment {
        start,
        end: end.max(start),
        text,
        speaker: Some(speaker),
        speaker_label: None,
        speaker_profile_id: None,
        words: Vec::new(),
    }
}

/// Parses a moss-transcribe-diarize decoded transcript's inline
/// `[start][end][SNN]text` markup into ordered, non-overlapping [`Segment`]s.
/// `audio_duration_seconds` closes a final segment that never received a
/// trailing anchor (premature EOS), the same permissive end-of-stream
/// handling as the cohere parser this mirrors.
///
/// Returns `Ok(vec![])` (never an error) when the stream is empty or well
/// formed but carries no speaker tags/text at all -- e.g. a bare anchor/tag
/// skeleton with no free text -- since there is nothing to invent a segment
/// from. See the module doc for the fail-closed policy on genuinely malformed
/// input.
pub(crate) fn parse_moss_td_speaker_segments(
    text: &str,
    audio_duration_seconds: f32,
) -> Result<Vec<Segment>, MossTdSpeakerSegmentParseError> {
    let mut segments = Vec::new();
    let mut pending_start: Option<f32> = None;
    let mut last_anchor: Option<f32> = None;
    let mut current_speaker: Option<String> = None;
    let mut buffer = String::new();
    let mut rest = text;

    while let Some(open_rel) = rest.find('[') {
        buffer.push_str(&rest[..open_rel]);
        let after_open = &rest[open_rel + 1..];
        let Some(close_rel) = after_open.find(']') else {
            return Err(MossTdSpeakerSegmentParseError::UnclosedTag);
        };
        let raw_tag = &after_open[..close_rel];
        rest = &after_open[close_rel + 1..];

        match parse_tag_content(raw_tag)? {
            MossTdTag::Anchor(timestamp) => {
                if let Some(previous) = last_anchor
                    && timestamp < previous
                {
                    // MOSS occasionally emits a corrected turn-start anchor
                    // immediately after an initial anchor and before any text
                    // (for example `[125.31][124.34][S01]`). It denotes the
                    // same pending start, not a temporal reversal. Preserve
                    // strict monotonicity once text has been attached.
                    if buffer.trim().is_empty() && pending_start == Some(previous) {
                        pending_start = Some(timestamp);
                        last_anchor = Some(timestamp);
                        continue;
                    }
                    return Err(MossTdSpeakerSegmentParseError::TimeWentBackwards {
                        previous,
                        next: timestamp,
                    });
                }
                last_anchor = Some(timestamp);
                let trimmed = buffer.trim();
                if !trimmed.is_empty() {
                    let speaker = current_speaker
                        .clone()
                        .ok_or(MossTdSpeakerSegmentParseError::TextBeforeSpeaker)?;
                    let start =
                        pending_start.ok_or(MossTdSpeakerSegmentParseError::TextBeforeTimestamp)?;
                    segments.push(plain_segment(
                        speaker,
                        start,
                        timestamp,
                        trimmed.to_string(),
                    ));
                }
                buffer.clear();
                pending_start = Some(timestamp);
            }
            MossTdTag::Speaker(label) => {
                current_speaker = Some(label);
            }
        }
    }
    buffer.push_str(rest);
    let trimmed = buffer.trim();
    if !trimmed.is_empty() {
        let speaker = current_speaker.ok_or(MossTdSpeakerSegmentParseError::TextBeforeSpeaker)?;
        let start = pending_start.ok_or(MossTdSpeakerSegmentParseError::TextBeforeTimestamp)?;
        segments.push(plain_segment(
            speaker,
            start,
            audio_duration_seconds.max(start),
            trimmed.to_string(),
        ));
    }
    Ok(segments)
}

/// The executor's segment-overlay decision, centralized next to the parser it
/// guards instead of inlined in `executor.rs`. Returns the parsed per-speaker
/// segments when the decode's tag stream is well formed AND carried at least
/// one attributable turn; otherwise -- a typed parse error, or a well-formed
/// stream with no speaker tags/text at all -- returns the single, speaker-less
/// segment carrying the untouched raw `text` (tags included, verbatim), i.e.
/// the exact shape that existed before inline-tag structuring. Structure is
/// never fabricated for a decode that did not assert it.
pub(crate) fn moss_td_segments_or_degrade(text: &str, audio_duration_seconds: f32) -> Vec<Segment> {
    match parse_moss_td_speaker_segments(text, audio_duration_seconds) {
        Ok(segments) if !segments.is_empty() => segments,
        _ => vec![Segment {
            start: 0.0,
            end: audio_duration_seconds.max(0.0),
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_an_adjacent_pending_start_correction() {
        let segments = parse_moss_td_speaker_segments("[1.0][0.9][S01]hello[2.0]", 2.0)
            .expect("adjacent corrected start should parse");
        assert_eq!(segments[0].start, 0.9);
        assert_eq!(segments[0].end, 2.0);
    }

    #[test]
    fn rejects_a_backwards_anchor_after_text() {
        let error = parse_moss_td_speaker_segments("[1.0][S01]hello[0.9]", 2.0)
            .expect_err("text-attached backwards anchor must fail closed");
        assert!(matches!(
            error,
            MossTdSpeakerSegmentParseError::TimeWentBackwards { .. }
        ));
    }

    #[test]
    fn empty_stream_yields_no_segments() {
        assert_eq!(parse_moss_td_speaker_segments("", 5.0), Ok(Vec::new()));
    }

    #[test]
    fn tags_only_with_no_text_yields_no_segments() {
        let segments = parse_moss_td_speaker_segments("[0.0][S01][1.0][S02][2.0]", 5.0)
            .expect("well-formed tag-only stream parses");
        assert!(segments.is_empty());
    }

    #[test]
    fn parses_the_jfk_golden_shape() {
        let text = concat!(
            "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S01] ask not what your ",
            "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
        );
        let segments = parse_moss_td_speaker_segments(text, 10.59).expect("jfk golden parses");
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[0].start, 0.28);
        assert_eq!(segments[0].end, 2.32);
        assert_eq!(segments[0].text, "And so, my fellow Americans,");
        assert_eq!(segments[1].start, 3.22);
        assert_eq!(segments[1].end, 7.71);
        assert_eq!(segments[2].start, 8.12);
        assert_eq!(segments[2].end, 10.59);
        assert_eq!(segments[2].text, "ask what you can do for your country.");
    }

    #[test]
    fn parses_a_speaker_change() {
        let text = "[0.0][S01]hello[1.0][2.0][S02]world[3.0]";
        let segments =
            parse_moss_td_speaker_segments(text, 3.0).expect("two-speaker stream parses");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_02"));
        assert_eq!(segments[1].text, "world");
    }

    #[test]
    fn speaker_number_gap_is_accepted_verbatim() {
        let text = "[0.0][S01]hello[1.0][2.0][S05]world[3.0]";
        let segments =
            parse_moss_td_speaker_segments(text, 3.0).expect("a numbering gap is not malformed");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_05"));
    }

    #[test]
    fn trailing_text_without_a_closing_anchor_uses_audio_duration() {
        let segments = parse_moss_td_speaker_segments("[0.0][S01]hello", 4.5)
            .expect("premature EOS still parses");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end, 4.5);
    }

    #[test]
    fn unclosed_tag_is_rejected() {
        let error = parse_moss_td_speaker_segments("[0.0][S01]hello[1.0", 5.0)
            .expect_err("unterminated '[' must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::UnclosedTag);
    }

    #[test]
    fn unknown_tag_content_is_rejected() {
        let error = parse_moss_td_speaker_segments("[0.0][S01]hello[oops]", 5.0)
            .expect_err("a tag that is neither a timestamp nor Sxx must fail closed");
        assert_eq!(
            error,
            MossTdSpeakerSegmentParseError::UnknownTag {
                raw: "oops".to_string()
            }
        );
    }

    #[test]
    fn time_reversal_is_rejected() {
        let error = parse_moss_td_speaker_segments("[2.0][S01]hi[1.0]", 5.0)
            .expect_err("a time anchor going backwards must fail closed");
        assert_eq!(
            error,
            MossTdSpeakerSegmentParseError::TimeWentBackwards {
                previous: 2.0,
                next: 1.0
            }
        );
    }

    #[test]
    fn text_before_any_timestamp_is_rejected() {
        let error = parse_moss_td_speaker_segments("[S01]hello", 5.0)
            .expect_err("text before the first anchor must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::TextBeforeTimestamp);
    }

    #[test]
    fn text_before_any_speaker_tag_is_rejected() {
        let error = parse_moss_td_speaker_segments("[0.0]hello[1.0]", 5.0)
            .expect_err("text before the first speaker tag must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::TextBeforeSpeaker);
    }

    /// The degrade shape the executor keeps for a malformed decode: exactly one
    /// speaker-less segment spanning the whole clip, carrying the raw text
    /// verbatim with its tags still in it -- never empty, never rewritten. This
    /// is the verbose_json/SRT/VTT overlay-withheld case (a single unattributed
    /// cue), asserted here so it cannot silently regress into an empty segment
    /// list or a stripped transcript.
    #[test]
    fn malformed_decode_degrades_to_one_raw_speaker_less_segment() {
        // Time runs backwards -> a typed parse error -> degrade.
        let raw = "[2.0][S01]hi[1.0][S01]bye";
        let segments = moss_td_segments_or_degrade(raw, 5.0);
        assert_eq!(
            segments,
            vec![Segment {
                start: 0.0,
                end: 5.0,
                text: raw.to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }]
        );
    }

    /// A well-formed decode that simply carried no speaker tags/text degrades
    /// the same way (single speaker-less segment), not to an empty list.
    #[test]
    fn tag_skeleton_with_no_text_degrades_to_one_speaker_less_segment() {
        let raw = "[0.0][1.0][2.0]";
        let segments = moss_td_segments_or_degrade(raw, 4.0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, None);
        assert_eq!(segments[0].text, raw);
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 4.0);
    }

    /// A well-formed decode keeps its structured per-speaker turns (the happy
    /// path the degrade helper must NOT swallow).
    #[test]
    fn well_formed_decode_keeps_structured_segments() {
        let segments = moss_td_segments_or_degrade("[0.0][S01]hello[1.0][2.0][S02]world[3.0]", 3.0);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_02"));
    }

    /// Documented inherent ambiguity (see the module doc's "Tags are ordinary
    /// characters" section): a bracketed numeric that is really transcript
    /// content is indistinguishable from a time anchor and splits the segment.
    /// Pinned so the behavior is a conscious, reviewed contract rather than a
    /// surprise -- the fail-closed worst case is a mis-split, never a panic.
    #[test]
    fn bracketed_numeric_content_is_consumed_as_an_anchor_by_design() {
        let segments = parse_moss_td_speaker_segments("[0.0][S01]meeting at [3.30] pm[5.0]", 6.0)
            .expect("well-formed once the stray bracket is read as an anchor");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "meeting at");
        assert_eq!(segments[0].end, 3.30);
        assert_eq!(segments[1].text, "pm");
    }
}
