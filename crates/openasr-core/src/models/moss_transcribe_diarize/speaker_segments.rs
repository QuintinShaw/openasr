//! Normalizes moss-transcribe-diarize's inline `[start][end][SNN]` speaker /
//! time-anchor markup -- ordinary BPE tokens the Qwen3 decoder emits as
//! literal transcript *characters* (see the module doc) -- into the engine's
//! shared representation: [`Segment`]s carrying clean text plus, when the
//! request asked for speakers, the recording-local `SPEAKER_NN` labels the
//! model asserted.
//!
//! # Normalization is unconditional; keeping the labels is not
//!
//! The decode prompt is a fixed instruction the checkpoint was fine-tuned
//! against (see `decode_prompt`), so the model writes its markers whether or
//! not the user asked for speakers -- there is no "plain transcript" decode
//! mode to switch to. That makes stripping the markup this layer's job, not the
//! caller's: the markers are an internal transport for structure, never
//! transcript content, so they are removed from the text on every path. With
//! Voice ID off the speaker labels are dropped as well, and what a caller gets
//! back is byte-for-byte what a model that cannot separate speakers would have
//! produced. Leaving the stripping to a renderer would leak the markers into
//! every copy/export path and make this family behave differently from every
//! other one under the same switch.
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
//! write the same shared field, [`Segment::speaker_label`] -- the one
//! recording-local speaker representation the engine carries, which an
//! external source (`crate::diarize::pipeline::Diarization` -> attribution)
//! also lands on. That is what lets in-decoder and external sources stay
//! interchangeable downstream, including for the identity stage
//! (`crate::diarize::voice_id`).
//!
//! An external source additionally carries a per-turn `overlap` flag this
//! in-decoder path has no signal for, plus a per-turn confidence neither
//! source populates today (cf.
//! [`crate::api::backend::WordTimestamp::confidence`]). [`Segment`] carries
//! neither, and moss-td asserts neither, so nothing is lost now -- but a
//! future consumer that wants the VAD path's overlap/confidence must grow
//! [`Segment`] additively (a new `Option`/`#[serde(default)]` field) rather
//! than reshape it. Flagged here so that growth stays a conscious additive
//! step, not a breaking change.
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
//! fail-closed policy below degrades the whole decode back to a single
//! unstructured segment. The reference decode does not emit bracketed numerics
//! as free text, so this stays a theoretical edge, but callers must treat the
//! segment overlay as best-effort structure over a plain-text signal, not a
//! guaranteed lossless parse of arbitrary transcript content.
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
//! silently dropping the offending span. The caller treats any such error, and
//! the "well-formed but zero speaker tags found" case, the same way: this
//! decode's tag structure is not trustworthy, so it degrades to a single
//! speaker-less segment spanning the clip. The transcript *words* are never
//! dropped or rewritten -- only the structure overlay is withheld, and the
//! markup characters themselves are removed by the same rule the parser uses to
//! recognize them (see [`strip_moss_td_markup`]) so a degraded decode cannot
//! leak markers a successful one would have consumed. That mirrors this
//! crate's existing diarization degrade path (an empty turn list is a silent
//! no-op, never an error surfaced to the caller).
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
        speaker: Some(speaker.clone()),
        speaker_label: Some(speaker),
        speaker_person_id: None,
        speaker_snapshot_label: None,
        words: Vec::new(),
    }
}

/// How a moss-td decode ended, which is what the parser needs to close a final
/// segment that never received its trailing anchor.
///
/// The distinction is load-bearing, not bookkeeping. Both cases look identical
/// in the decoded text -- trailing words with no closing anchor -- but they mean
/// opposite things:
/// - a decode that stopped on its own token really did reach the end of the
///   audio, so closing the last segment at the clip's end is right;
/// - a decode the driver cut short (degenerate-repeat guard, exhausted budget)
///   stopped somewhere in the middle, and everything after that point is audio
///   this decode never looked at. Closing the last segment at the clip's end
///   there paints one speaker's label across every real turn that followed --
///   the transcript reads complete and confidently attributes minutes of other
///   people's speech to whoever happened to be talking when the decode died.
///
/// So a truncated decode closes its final segment at the last anchor it
/// actually emitted and says nothing past it. That direction is deliberate:
/// under-covering the audio costs recall, over-covering it invents speaker
/// attribution, and this family's product contract puts a wrong answer well
/// below a missing one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MossTdDecodeExtent {
    /// Duration of the audio THIS decode was given. Under longform slicing that
    /// is one slice, not the recording, so a truncated slice can never blanket
    /// past its own end.
    pub audio_duration_seconds: f32,
    /// Whether the decode stopped short of that audio.
    pub truncated: bool,
}

impl MossTdDecodeExtent {
    /// A decode that ran to the end of the audio it was given.
    #[cfg(test)]
    pub fn complete(audio_duration_seconds: f32) -> Self {
        Self {
            audio_duration_seconds,
            truncated: false,
        }
    }

    /// A decode the driver stopped before the end of the audio it was given.
    #[cfg(test)]
    pub fn truncated(audio_duration_seconds: f32) -> Self {
        Self {
            audio_duration_seconds,
            truncated: true,
        }
    }

    /// Where an unanchored final segment ends: the clip's end for a complete
    /// decode, the last anchor (its own start) for a truncated one.
    fn close_final_segment_at(&self, last_anchor_seconds: f32) -> f32 {
        if self.truncated {
            last_anchor_seconds
        } else {
            self.audio_duration_seconds.max(last_anchor_seconds)
        }
    }
}

/// Parses a moss-transcribe-diarize decoded transcript's inline
/// `[start][end][SNN]text` markup into ordered, non-overlapping [`Segment`]s.
/// [`MossTdDecodeExtent`] closes a final segment that never received a
/// trailing anchor -- at the clip's end when the model stopped on its own
/// (the same permissive end-of-stream handling as the cohere parser this
/// mirrors), at the last anchor when the decode was cut short.
///
/// Returns `Ok(vec![])` (never an error) when the stream is empty or well
/// formed but carries no speaker tags/text at all -- e.g. a bare anchor/tag
/// skeleton with no free text -- since there is nothing to invent a segment
/// from. See the module doc for the fail-closed policy on genuinely malformed
/// input.
pub(crate) fn parse_moss_td_speaker_segments(
    text: &str,
    extent: MossTdDecodeExtent,
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
            extent.close_final_segment_at(start),
            trimmed.to_string(),
        ));
    }
    Ok(segments)
}

/// Remove moss-td's structural markup from a decoded string without parsing it
/// into segments: drops exactly the bracketed spans [`parse_tag_content`]
/// recognizes as a tag (an `Sxx` speaker marker or a finite non-negative time
/// anchor) and leaves every other `[...]` span, and all other characters,
/// untouched. Collapses the ASCII space runs a removal leaves behind so the
/// result reads like ordinary prose.
///
/// Used on the degrade path, where the tag stream is not trustworthy enough to
/// carve segments from but the markers must still not reach the caller.
pub(crate) fn strip_moss_td_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open_rel) = rest.find('[') {
        let after_open = &rest[open_rel + 1..];
        let Some(close_rel) = after_open.find(']') else {
            break;
        };
        out.push_str(&rest[..open_rel]);
        if parse_tag_content(&after_open[..close_rel]).is_err() {
            // Not a marker: content that merely looks bracketed stays verbatim.
            out.push('[');
            out.push_str(&after_open[..close_rel]);
            out.push(']');
        }
        rest = &after_open[close_rel + 1..];
    }
    out.push_str(rest);
    collapse_ascii_space_runs(out.trim())
}

fn collapse_ascii_space_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_was_space = false;
    for character in text.chars() {
        let is_space = character == ' ';
        if !(is_space && previous_was_space) {
            out.push(character);
        }
        previous_was_space = is_space;
    }
    out
}

/// One decode normalized into the engine's shared representation.
pub(crate) struct MossTdNormalizedDecode {
    /// Ordered, non-overlapping segments with markup-free text. Speaker labels
    /// are present only when the request asked for them.
    pub segments: Vec<Segment>,
    /// The flat transcript, markup-free, consistent with `segments`.
    pub text: String,
    /// Set when the decode stopped short of the audio it was given: the point
    /// (in this decode's own seconds) past which the transcript says nothing.
    ///
    /// Without this a truncated decode is indistinguishable from a complete one
    /// at the API boundary -- same shape, same success status -- and a caller
    /// has no way to know that the last stretch of audio is simply absent. A
    /// caller that slices can use it to retry or flag the affected slice
    /// instead of shipping a silently short transcript.
    pub truncated_at_seconds: Option<f32>,
}

/// Normalize one moss-td decode. `keep_speaker_labels` is the request's Voice
/// ID switch as it reaches this family: the markup is stripped either way, and
/// only the recording-local `SPEAKER_NN` labels depend on it.
///
/// Returns the parsed per-speaker segments when the decode's tag stream is well
/// formed AND carried at least one attributable turn; otherwise -- a typed
/// parse error, or a well-formed stream with no speaker tags/text at all -- a
/// single speaker-less segment spanning the clip, carrying the same words with
/// the markers stripped. Structure is never fabricated for a decode that did
/// not assert it.
pub(crate) fn normalize_moss_td_decode(
    text: &str,
    extent: MossTdDecodeExtent,
    keep_speaker_labels: bool,
) -> MossTdNormalizedDecode {
    let mut segments = match parse_moss_td_speaker_segments(text, extent) {
        Ok(segments) if !segments.is_empty() => segments,
        // No trustworthy tag structure at all. The span stays the whole clip
        // even for a truncated decode: there is no anchor to tighten to, and
        // with no speaker label on it this segment cannot mis-attribute anyone
        // -- the blanket risk this parser guards against is a *labeled* segment
        // covering other people's turns.
        _ => vec![Segment {
            start: 0.0,
            end: extent.audio_duration_seconds.max(0.0),
            text: strip_moss_td_markup(text),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }],
    };
    let truncated_at_seconds = extent.truncated.then(|| {
        segments
            .last()
            .map(|segment| segment.end.max(segment.start))
            .unwrap_or(0.0)
    });
    if !keep_speaker_labels {
        for segment in &mut segments {
            segment.speaker = None;
            segment.speaker_label = None;
        }
    }
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|segment_text| !segment_text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    MossTdNormalizedDecode {
        segments,
        text,
        truncated_at_seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_an_adjacent_pending_start_correction() {
        let segments = parse_moss_td_speaker_segments(
            "[1.0][0.9][S01]hello[2.0]",
            MossTdDecodeExtent::complete(2.0),
        )
        .expect("adjacent corrected start should parse");
        assert_eq!(segments[0].start, 0.9);
        assert_eq!(segments[0].end, 2.0);
    }

    #[test]
    fn rejects_a_backwards_anchor_after_text() {
        let error = parse_moss_td_speaker_segments(
            "[1.0][S01]hello[0.9]",
            MossTdDecodeExtent::complete(2.0),
        )
        .expect_err("text-attached backwards anchor must fail closed");
        assert!(matches!(
            error,
            MossTdSpeakerSegmentParseError::TimeWentBackwards { .. }
        ));
    }

    #[test]
    fn empty_stream_yields_no_segments() {
        assert_eq!(
            parse_moss_td_speaker_segments("", MossTdDecodeExtent::complete(5.0)),
            Ok(Vec::new())
        );
    }

    #[test]
    fn tags_only_with_no_text_yields_no_segments() {
        let segments = parse_moss_td_speaker_segments(
            "[0.0][S01][1.0][S02][2.0]",
            MossTdDecodeExtent::complete(5.0),
        )
        .expect("well-formed tag-only stream parses");
        assert!(segments.is_empty());
    }

    #[test]
    fn parses_the_jfk_golden_shape() {
        let text = concat!(
            "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S01] ask not what your ",
            "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
        );
        let segments = parse_moss_td_speaker_segments(text, MossTdDecodeExtent::complete(10.59))
            .expect("jfk golden parses");
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
        let segments = parse_moss_td_speaker_segments(text, MossTdDecodeExtent::complete(3.0))
            .expect("two-speaker stream parses");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_02"));
        assert_eq!(segments[1].text, "world");
    }

    #[test]
    fn speaker_number_gap_is_accepted_verbatim() {
        let text = "[0.0][S01]hello[1.0][2.0][S05]world[3.0]";
        let segments = parse_moss_td_speaker_segments(text, MossTdDecodeExtent::complete(3.0))
            .expect("a numbering gap is not malformed");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_05"));
    }

    #[test]
    fn trailing_text_without_a_closing_anchor_uses_audio_duration() {
        let segments =
            parse_moss_td_speaker_segments("[0.0][S01]hello", MossTdDecodeExtent::complete(4.5))
                .expect("premature EOS still parses");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end, 4.5);
    }

    #[test]
    fn unclosed_tag_is_rejected() {
        let error = parse_moss_td_speaker_segments(
            "[0.0][S01]hello[1.0",
            MossTdDecodeExtent::complete(5.0),
        )
        .expect_err("unterminated '[' must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::UnclosedTag);
    }

    #[test]
    fn unknown_tag_content_is_rejected() {
        let error = parse_moss_td_speaker_segments(
            "[0.0][S01]hello[oops]",
            MossTdDecodeExtent::complete(5.0),
        )
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
        let error =
            parse_moss_td_speaker_segments("[2.0][S01]hi[1.0]", MossTdDecodeExtent::complete(5.0))
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
        let error = parse_moss_td_speaker_segments("[S01]hello", MossTdDecodeExtent::complete(5.0))
            .expect_err("text before the first anchor must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::TextBeforeTimestamp);
    }

    #[test]
    fn text_before_any_speaker_tag_is_rejected() {
        let error =
            parse_moss_td_speaker_segments("[0.0]hello[1.0]", MossTdDecodeExtent::complete(5.0))
                .expect_err("text before the first speaker tag must fail closed");
        assert_eq!(error, MossTdSpeakerSegmentParseError::TextBeforeSpeaker);
    }

    /// The blanket-segment regression, in its exact field shape: a long
    /// recording where the decode was cut short a couple of minutes in. The
    /// last speaker's final segment must stop at the last anchor the model
    /// emitted -- not run to the end of the clip, which would attribute every
    /// remaining minute of the meeting to whoever was talking when the decode
    /// died, and would do it while looking like an ordinary complete result.
    #[test]
    fn a_truncated_decode_does_not_blanket_the_rest_of_the_clip() {
        let raw = "[0.0][S01]first[120.0][121.5][S02]cut off here";
        let normalized = normalize_moss_td_decode(raw, MossTdDecodeExtent::truncated(600.0), true);
        assert_eq!(normalized.segments.len(), 2);
        let last = normalized.segments.last().expect("last segment");
        assert_eq!(last.start, 121.5);
        assert_eq!(
            last.end, 121.5,
            "truncated tail must not stretch to the clip end"
        );
        // The same decode reported as complete legitimately closes at the clip
        // end -- the two differ only in what the driver said about the stop.
        let complete = normalize_moss_td_decode(raw, MossTdDecodeExtent::complete(600.0), true);
        assert_eq!(complete.segments.last().expect("last segment").end, 600.0);
        assert_eq!(complete.truncated_at_seconds, None);
    }

    /// The truncation point is reported, not just acted on: a caller (and the
    /// longform slicer, which may retry or flag the slice) has to be able to
    /// see that the audio past this second was never transcribed.
    #[test]
    fn a_truncated_decode_reports_where_it_stopped() {
        let normalized = normalize_moss_td_decode(
            "[0.0][S01]first[120.0][121.5][S02]cut off here",
            MossTdDecodeExtent::truncated(600.0),
            true,
        );
        assert_eq!(normalized.truncated_at_seconds, Some(121.5));
    }

    /// A truncated decode that ended cleanly on an anchor has no unanchored
    /// tail to tighten, so its segments are untouched and the reported
    /// truncation point is that last anchor.
    #[test]
    fn a_truncated_decode_ending_on_an_anchor_keeps_its_segments() {
        let normalized = normalize_moss_td_decode(
            "[0.0][S01]first[12.5]",
            MossTdDecodeExtent::truncated(600.0),
            true,
        );
        assert_eq!(normalized.segments.len(), 1);
        assert_eq!(normalized.segments[0].end, 12.5);
        assert_eq!(normalized.truncated_at_seconds, Some(12.5));
    }

    /// The degrade shape for a malformed decode: exactly one speaker-less
    /// segment spanning the whole clip, carrying the same words with the
    /// markers stripped -- never empty, never missing words. This is the
    /// verbose_json/SRT/VTT overlay-withheld case (a single unattributed cue),
    /// asserted here so it cannot silently regress into an empty segment list,
    /// a dropped transcript, or a transcript that leaks markup.
    #[test]
    fn malformed_decode_degrades_to_one_markup_free_speaker_less_segment() {
        // Time runs backwards -> a typed parse error -> degrade.
        let raw = "[2.0][S01]hi[1.0][S01]bye";
        let normalized = normalize_moss_td_decode(raw, MossTdDecodeExtent::complete(5.0), true);
        assert_eq!(
            normalized.segments,
            vec![Segment {
                start: 0.0,
                end: 5.0,
                text: "hibye".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }]
        );
        assert_eq!(normalized.text, "hibye");
    }

    /// A well-formed decode that simply carried no speaker tags/text degrades
    /// the same way (single speaker-less segment), not to an empty list.
    #[test]
    fn tag_skeleton_with_no_text_degrades_to_one_speaker_less_segment() {
        let normalized =
            normalize_moss_td_decode("[0.0][1.0][2.0]", MossTdDecodeExtent::complete(4.0), true);
        assert_eq!(normalized.segments.len(), 1);
        assert_eq!(normalized.segments[0].speaker, None);
        assert_eq!(normalized.segments[0].text, "");
        assert_eq!(normalized.segments[0].start, 0.0);
        assert_eq!(normalized.segments[0].end, 4.0);
    }

    /// A well-formed decode keeps its structured per-speaker turns, and the
    /// flat transcript it reports never carries the markup the model wrote.
    #[test]
    fn well_formed_decode_keeps_structured_segments_and_clean_text() {
        let normalized = normalize_moss_td_decode(
            "[0.0][S01]hello[1.0][2.0][S02]world[3.0]",
            MossTdDecodeExtent::complete(3.0),
            true,
        );
        assert_eq!(normalized.segments.len(), 2);
        assert_eq!(
            normalized.segments[0].speaker.as_deref(),
            Some("SPEAKER_01")
        );
        assert_eq!(
            normalized.segments[1].speaker.as_deref(),
            Some("SPEAKER_02")
        );
        assert_eq!(normalized.text, "hello world");

        let labels: Vec<_> = normalized
            .segments
            .iter()
            .filter_map(|segment| segment.speaker_label.as_deref())
            .collect();
        assert_eq!(labels, vec!["SPEAKER_01", "SPEAKER_02"]);
    }

    /// Voice ID off: same words, same timings, no speaker structure anywhere --
    /// the transcript is what a model that cannot separate speakers at all
    /// would have produced.
    #[test]
    fn voice_id_off_drops_every_trace_of_the_speaker_markup() {
        let raw = "[0.0][S01]hello[1.0][2.0][S02]world[3.0]";
        let on = normalize_moss_td_decode(raw, MossTdDecodeExtent::complete(3.0), true);
        let off = normalize_moss_td_decode(raw, MossTdDecodeExtent::complete(3.0), false);

        assert_eq!(off.text, on.text);
        assert!(!off.text.contains('['));
        assert_eq!(off.segments.len(), on.segments.len());
        for (off_segment, on_segment) in off.segments.iter().zip(&on.segments) {
            assert_eq!(off_segment.text, on_segment.text);
            assert_eq!(off_segment.start, on_segment.start);
            assert_eq!(off_segment.end, on_segment.end);
            assert!(!off_segment.text.contains("[S"));
            assert!(off_segment.speaker.is_none());
            assert!(off_segment.speaker_label.is_none());
        }
    }

    /// The degrade path strips markers by exactly the rule the parser uses to
    /// recognize them, and leaves bracketed spans that are not markers alone.
    #[test]
    fn markup_stripping_removes_only_recognized_tags() {
        assert_eq!(strip_moss_td_markup("[0.28][S01] And so,[2.32]"), "And so,");
        assert_eq!(strip_moss_td_markup("see [note] here"), "see [note] here");
        assert_eq!(strip_moss_td_markup("a [1.5] b"), "a b");
        assert_eq!(
            strip_moss_td_markup("unterminated [1.5"),
            "unterminated [1.5"
        );
        assert_eq!(strip_moss_td_markup("plain text"), "plain text");
    }

    /// Documented inherent ambiguity (see the module doc's "Tags are ordinary
    /// characters" section): a bracketed numeric that is really transcript
    /// content is indistinguishable from a time anchor and splits the segment.
    /// Pinned so the behavior is a conscious, reviewed contract rather than a
    /// surprise -- the fail-closed worst case is a mis-split, never a panic.
    #[test]
    fn bracketed_numeric_content_is_consumed_as_an_anchor_by_design() {
        let segments = parse_moss_td_speaker_segments(
            "[0.0][S01]meeting at [3.30] pm[5.0]",
            MossTdDecodeExtent::complete(6.0),
        )
        .expect("well-formed once the stray bracket is read as an anchor");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "meeting at");
        assert_eq!(segments[0].end, 3.30);
        assert_eq!(segments[1].text, "pm");
    }
}
