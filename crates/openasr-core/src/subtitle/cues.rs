//! Family-agnostic re-segmentation of attributed segments into subtitle-grade
//! cues.
//!
//! Model families differ wildly in how coarsely they segment: whisper emits
//! sentence-ish segments, but X-ASR / qwen / cohere / moonshine each emit one
//! monolithic segment per decode (per long-form slice), which renders as a
//! single 30-60s subtitle cue. This pass runs after speaker attribution and
//! rebalances every segment into short cues, splitting at sentence-final
//! punctuation first, then clause punctuation, then word gaps, while honouring
//! duration and line-length caps.
//!
//! Invariants:
//! - It never reorders or rewrites words, so the joined transcript text is
//!   unchanged (streaming==batch parity and `transcription.text` stay intact).
//! - It splits *within* the segments it is given and never merges across them.
//!   A speaker change is therefore a hard cue boundary: a cue never spans two
//!   speakers.
//! - Word timestamps drive the boundaries when present; otherwise a segment's
//!   words are synthesised proportionally from its character span so the same
//!   packer applies.

use crate::api::backend::{Segment, Transcription, WordTimestamp};

/// Preferred cue duration. Cues are grown up to this bound before a cut is
/// forced, so most cues land at or under it.
const TARGET_CUE_SECONDS: f32 = 6.0;
/// Hard ceiling used only when merging a dangling orphan tail back into its
/// neighbour; a normal cue is already bounded by [`TARGET_CUE_SECONDS`].
const MAX_CUE_SECONDS: f32 = 8.0;
/// ~42 characters x 2 lines for Latin-script cues.
const LATIN_MAX_CHARS: usize = 84;
/// ~18 fullwidth characters x 2 lines for CJK-script cues.
const CJK_MAX_CHARS: usize = 36;
/// Upper reading-speed bound (content characters per second) for Latin cues.
/// Calibrated to common subtitle guidelines (~17-21 CPS); V1 uses the top of
/// that range so well-paced speech is not over-split.
const LATIN_MAX_CPS: f32 = 21.0;
/// Upper reading-speed bound for CJK cues (fullwidth characters per second).
/// Common Chinese subtitle guidance is roughly 4-9 CPS; V1 uses 9.
const CJK_MAX_CPS: f32 = 9.0;
/// Inter-word gap treated as a deliberate pause when choosing a forced cut.
const MIN_PAUSE_GAP_S: f32 = 0.35;
/// A trailing piece of this many words or fewer is treated as an orphan and
/// merged back into the previous cue when it fits within the hard caps.
const ORPHAN_MAX_WORDS: usize = 2;

/// Re-segment every segment of `transcription` into subtitle-grade cues,
/// overwriting `transcription.segments`. Prefer
/// [`resegment_segments_into_cues`] + reading projection for the dual-view
/// pipeline; this helper remains for callers that only need cue segments.
///
/// Without a known audio duration the final cue is not CPS-stretched (no
/// unbounded display end); pass a duration via
/// [`resegment_segments_into_cues`] when available.
pub fn resegment_transcription_cues(mut transcription: Transcription) -> Transcription {
    if transcription.segments.is_empty() {
        return transcription;
    }
    transcription.segments =
        resegment_segments_into_cues(std::mem::take(&mut transcription.segments), None);
    transcription
}

/// Split attributed segments into short subtitle cues, then layout display ends
/// with CPS stretch clamped to the next cue start or `audio_duration_s`.
///
/// Speaker identity on each input segment is copied onto every child cue;
/// segments are never merged, so a speaker change is always a hard boundary.
///
/// Display-time priority (never rewrites text):
/// 1. do not cross the next cue start or audio end
/// 2. do not overlap neighbouring cues
/// 3. stretch toward `content / max_cps` when that target fits inside the hard end
/// 4. otherwise keep the acoustic end (clamped to the hard end) -- never
///    fabricate an out-of-bounds display time just to meet CPS
pub fn resegment_segments_into_cues(
    segments: Vec<Segment>,
    audio_duration_s: Option<f32>,
) -> Vec<Segment> {
    let mut cues = Vec::with_capacity(segments.len());
    for segment in segments {
        cues.extend(segment_into_cues(segment));
    }
    layout_cue_display_ends(cues, audio_duration_s)
}

/// A word-sized unit the packer reasons over: its character span within the
/// parent segment text plus its time span. Real word timestamps are used when
/// they align to the segment text; otherwise units are synthesised from
/// whitespace tokens with times interpolated proportionally.
struct CueToken {
    char_start: usize,
    char_end: usize,
    start: f32,
    end: f32,
}

/// Split one attributed segment into zero or more short subtitle cues.
///
/// Emit acoustic start/end only. Cross-cue CPS display stretch is applied
/// later by [`layout_cue_display_ends`] so a dense cue cannot invent an end
/// that overlaps the next speaker or runs past the audio.
pub fn segment_into_cues(segment: Segment) -> Vec<Segment> {
    let chars: Vec<char> = segment.text.chars().collect();
    let (tokens, real_words) = build_tokens(&segment, &chars);
    let limits = cue_limits(&chars);
    if tokens.len() < 2 {
        return vec![segment];
    }
    let ranges = pack_tokens(&chars, &tokens, limits);
    if ranges.len() <= 1 {
        return vec![segment];
    }
    let mut cues = Vec::with_capacity(ranges.len());
    for (first, last) in ranges {
        let text: String = chars[tokens[first].char_start..tokens[last].char_end]
            .iter()
            .collect();
        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let start = tokens[first].start.max(segment.start);
        // Acoustic end only; CPS stretch is a later layout step that sees the
        // next cue / audio hard end and refuses to fabricate out-of-bounds times.
        let end = tokens[last].end.max(start).min(segment.end.max(start));
        let words: Vec<WordTimestamp> = if real_words {
            segment.words[first..=last].to_vec()
        } else {
            Vec::new()
        };
        cues.push(Segment {
            start,
            end,
            text,
            speaker: segment.speaker.clone(),
            speaker_label: segment.speaker_label.clone(),
            speaker_person_id: segment.speaker_person_id.clone(),
            speaker_snapshot_label: segment.speaker_snapshot_label.clone(),
            words,
        });
    }
    if cues.len() <= 1 {
        return vec![segment];
    }
    cues
}

/// Apply bounded CPS display stretch across packed cues.
///
/// `hard_end` for cue `i` is the next cue's start, or `audio_duration_s` for
/// the last cue. Stretch only when the CPS target fits inside that bound;
/// otherwise keep the acoustic end (clamped). Text is never rewritten.
fn layout_cue_display_ends(mut cues: Vec<Segment>, audio_duration_s: Option<f32>) -> Vec<Segment> {
    let n = cues.len();
    if n == 0 {
        return cues;
    }
    let audio_hard_end = audio_duration_s.filter(|d| d.is_finite() && *d >= 0.0);
    for i in 0..n {
        let hard_end = if i + 1 < n {
            Some(cues[i + 1].start)
        } else {
            audio_hard_end
        };
        let start = cues[i].start;
        let acoustic_end = cues[i].end.max(start);
        let Some(hard_end) = hard_end else {
            // No next cue and no audio bound: refuse unbounded stretch.
            cues[i].end = acoustic_end;
            continue;
        };
        let chars: Vec<char> = cues[i].text.chars().collect();
        let limits = cue_limits(&chars);
        let content = chars
            .iter()
            .copied()
            .filter(|c| char_has_content(*c))
            .count();
        cues[i].end = display_end_for_cps(start, acoustic_end, hard_end, content, limits.max_cps);
    }
    cues
}

/// Resolve a cue's display end under a hard bound.
///
/// When `start + content/max_cps` fits inside `hard_end`, stretch (or keep)
/// to that target clamped by `hard_end`. When it does not, keep the acoustic
/// end clamped to `hard_end` -- never invent a time past the hard bound.
fn display_end_for_cps(
    start: f32,
    acoustic_end: f32,
    hard_end: f32,
    content_chars: usize,
    max_cps: f32,
) -> f32 {
    let acoustic_end = acoustic_end.max(start);
    if content_chars == 0 || max_cps <= 0.0 {
        return acoustic_end.min(hard_end).max(start);
    }
    let target = start + content_chars as f32 / max_cps;
    if target <= hard_end {
        hard_end.min(acoustic_end.max(target)).max(start)
    } else {
        acoustic_end.min(hard_end).max(start)
    }
}

/// Build the token stream for a segment. Returns `(tokens, real_words)` where
/// `real_words` is true when the tokens map 1:1 onto `segment.words` (so the
/// caller can slice the original word timestamps into each cue).
fn build_tokens(segment: &Segment, chars: &[char]) -> (Vec<CueToken>, bool) {
    if segment.words.len() >= 2
        && let Some(spans) = word_char_spans(chars, &segment.words)
    {
        let tokens = segment
            .words
            .iter()
            .zip(spans)
            .map(|(word, (char_start, char_end))| CueToken {
                char_start,
                char_end,
                start: word.start,
                end: word.end.max(word.start),
            })
            .collect();
        return (tokens, true);
    }
    (synthesize_tokens(segment, chars), false)
}

/// Synthesise word-sized tokens when real `words[]` are missing or cannot be
/// aligned to the segment text.
///
/// - Latin / space-delimited runs keep whitespace tokenisation (one token per
///   orthographic word, punctuation glued as emitted).
/// - CJK / wide-script continuous text has no spaces: split at character
///   boundaries with sentence/clause punctuation as its own token so the
///   packer can still cut long unpunctuated runs into short cues under the
///   char-budget and CPS caps.
///
/// Times are interpolated proportionally across
/// `[segment.start, segment.end]` by character position.
fn synthesize_tokens(segment: &Segment, chars: &[char]) -> Vec<CueToken> {
    let total = chars.len();
    if total == 0 {
        return Vec::new();
    }
    let span_start = segment.start;
    let span = (segment.end - segment.start).max(0.0);
    let at = |char_index: usize| span_start + span * (char_index as f32 / total as f32);
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < total {
        while index < total && chars[index].is_whitespace() {
            index += 1;
        }
        if index >= total {
            break;
        }
        let char_start = index;
        if is_wide_script(chars[index]) || is_cjk_or_fullwidth_punct(chars[index]) {
            // One wide-script character (or a standalone CJK punctuation mark)
            // per token. The packer groups them under char budget / CPS / pauses.
            index += 1;
        } else {
            // Latin (or other non-wide) orthographic word until whitespace or
            // a wide-script / CJK-punct boundary.
            index += 1;
            while index < total
                && !chars[index].is_whitespace()
                && !is_wide_script(chars[index])
                && !is_cjk_or_fullwidth_punct(chars[index])
            {
                index += 1;
            }
        }
        tokens.push(CueToken {
            char_start,
            char_end: index,
            start: at(char_start),
            end: at(index),
        });
    }
    tokens
}

fn is_cjk_or_fullwidth_punct(ch: char) -> bool {
    matches!(
        ch,
        '\u{3002}'
            | '\u{ff01}'
            | '\u{ff1f}'
            | '\u{ff0c}'
            | '\u{3001}'
            | '\u{ff1b}'
            | '\u{ff1a}'
            | '\u{2026}'
            | '\u{300c}'
            | '\u{300d}'
            | '\u{300e}'
            | '\u{300f}'
            | '\u{3010}'
            | '\u{3011}'
            | '\u{ff08}'
            | '\u{ff09}'
    )
}

/// Script-aware presentation caps applied by the packer (two-line char budget
/// + max reading speed + hard duration).
#[derive(Debug, Clone, Copy)]
struct CueLimits {
    char_budget: usize,
    max_cps: f32,
}

/// Greedily pack tokens into cue ranges (inclusive `(first, last)` token
/// indices). Cues grow up to the target caps and break at the first sentence
/// boundary or a deliberate inter-word pause, preferring deliberate pauses,
/// then clause punctuation, then the widest word gap when a long sentence must
/// be split. Every emitted range is re-checked against the target caps so a
/// forced natural cut cannot silently leave a multi-token CPS/duration breach.
fn pack_tokens(chars: &[char], tokens: &[CueToken], limits: CueLimits) -> Vec<(usize, usize)> {
    let n = tokens.len();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n {
        let mut end = start;
        // Grow the cue until it hits a content-bearing sentence boundary, a
        // deliberate pause before the next token, runs out of tokens, or the
        // next token would overflow the target caps.
        while !(ends_sentence(chars, &tokens[end]) && range_has_content(chars, tokens, start, end))
            && end + 1 < n
            && fits(chars, tokens, start, end + 1, limits, TARGET_CUE_SECONDS)
        {
            let gap = tokens[end + 1].start - tokens[end].end;
            // Active pause cut: end the cue at a real breath even when the
            // running total is still under char/duration/CPS caps. The
            // MIN_PAUSE_GAP_S floor (0.35s) avoids chopping after the first
            // word on ordinary short inter-word gaps.
            if gap >= MIN_PAUSE_GAP_S && range_has_content(chars, tokens, start, end) {
                break;
            }
            end += 1;
        }
        let cut = if (ends_sentence(chars, &tokens[end])
            && range_has_content(chars, tokens, start, end))
            || end == n - 1
        {
            end
        } else if end + 1 < n
            && (tokens[end + 1].start - tokens[end].end) >= MIN_PAUSE_GAP_S
            && range_has_content(chars, tokens, start, end)
        {
            // Grow stopped on a deliberate pause; keep the cut there rather
            // than re-running choose_cut (which would pick the same pause).
            end
        } else {
            choose_cut(chars, tokens, start, end)
        };
        let cut = enforce_range_fits(chars, tokens, start, cut, limits);
        ranges.push((start, cut));
        start = cut + 1;
    }
    merge_orphan_tails(chars, tokens, ranges, limits)
}

/// Final CPS / duration / char-budget gate before a range is committed.
///
/// The grow loop already refuses to *add* a token that would overflow, but a
/// subsequent natural cut (pause / clause / widest gap) can shrink duration
/// faster than content and raise CPS above the cap. Shrink until the range
/// fits. A single token that still fails is retained for emission; later
/// layout may stretch its display end only when a hard bound (next cue /
/// audio end) has room -- never by inventing an unbounded end.
fn enforce_range_fits(
    chars: &[char],
    tokens: &[CueToken],
    start: usize,
    mut end: usize,
    limits: CueLimits,
) -> usize {
    if fits(chars, tokens, start, end, limits, TARGET_CUE_SECONDS) {
        return end;
    }
    if start >= end {
        // Physically unsplittable single token; keep it (bounded layout later).
        return end;
    }
    // Prefer a natural cut inside the failing window when that sub-range fits.
    let preferred = choose_cut(chars, tokens, start, end);
    if preferred < end && fits(chars, tokens, start, preferred, limits, TARGET_CUE_SECONDS) {
        return preferred;
    }
    // Walk end backward until the prefix fits, keeping at least one token
    // (and preferring to leave content when the tail is pure punctuation).
    while end > start {
        let candidate = end - 1;
        if fits(chars, tokens, start, candidate, limits, TARGET_CUE_SECONDS)
            && range_has_content(chars, tokens, start, candidate)
        {
            return candidate;
        }
        // Still failing, or candidate is punctuation-only: keep shrinking as
        // long as a content-bearing shorter prefix remains.
        if candidate > start && range_has_content(chars, tokens, start, candidate) {
            end = candidate;
            continue;
        }
        break;
    }
    // Last resort: single leading token; layout may stretch only inside bounds.
    start
}

/// Pick the split point within `[start, end]` for a sentence that is too long
/// to keep whole: latest deliberate pause, else latest clause boundary, else
/// the token before the widest inter-word gap, else pack to `end`.
fn choose_cut(chars: &[char], tokens: &[CueToken], start: usize, end: usize) -> usize {
    // Prefer a real pause so cues breathe with speech rhythm.
    for k in (start..end).rev() {
        let gap = tokens[k + 1].start - tokens[k].end;
        if gap >= MIN_PAUSE_GAP_S && range_has_content(chars, tokens, start, k) {
            return k;
        }
    }
    for k in (start..=end).rev() {
        if ends_clause(chars, &tokens[k]) && range_has_content(chars, tokens, start, k) {
            return k;
        }
    }
    let mut best_k = end;
    let mut best_gap = 0.0f32;
    for k in start..end {
        let gap = tokens[k + 1].start - tokens[k].end;
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k
}

/// Merge a trailing 1-2 word cue back into its predecessor when they belong to
/// the same sentence (the predecessor did not end one) and the union still fits
/// the hard caps -- avoids leaving a dangling orphan word on its own line.
///
/// The forced aligner's reference timestamp repair can legitimately collapse a
/// trailing anomaly to the preceding timestamp. Such a zero-duration range is
/// never useful as a standalone subtitle cue, so merge it even when the prior
/// token ended a sentence. The original word timestamp remains untouched.
fn merge_orphan_tails(
    chars: &[char],
    tokens: &[CueToken],
    ranges: Vec<(usize, usize)>,
    limits: CueLimits,
) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (first, last) in ranges {
        if let Some(&(prev_first, prev_last)) = merged.last() {
            let word_count = last - first + 1;
            let prev_ends_sentence = ends_sentence(chars, &tokens[prev_last]);
            let zero_duration = tokens[last].end <= tokens[first].start;
            // Do not re-glue cues the packer split on a deliberate pause:
            // that cut is intentional speech rhythm, not an accidental orphan.
            // Zero-duration forced-aligner repairs still merge (gap <= 0).
            let pause_between =
                !zero_duration && (tokens[first].start - tokens[prev_last].end) >= MIN_PAUSE_GAP_S;
            if word_count <= ORPHAN_MAX_WORDS
                && (zero_duration || !prev_ends_sentence)
                && !pause_between
                && fits(chars, tokens, prev_first, last, limits, MAX_CUE_SECONDS)
            {
                *merged.last_mut().unwrap() = (prev_first, last);
                continue;
            }
        }
        merged.push((first, last));
    }
    merged
}

/// Whether `tokens[start..=end]` fits the two-line char budget, max reading
/// speed (CPS), and `max_seconds` hard duration.
fn fits(
    chars: &[char],
    tokens: &[CueToken],
    start: usize,
    end: usize,
    limits: CueLimits,
    max_seconds: f32,
) -> bool {
    let span_chars = tokens[end]
        .char_end
        .saturating_sub(tokens[start].char_start);
    if span_chars > limits.char_budget {
        return false;
    }
    let duration = tokens[end].end - tokens[start].start;
    if duration > max_seconds {
        return false;
    }
    // Reading-speed gate: dense text on a short cue is unreadable. Tiny /
    // zero-duration ranges skip the CPS check so zero-duration FA repairs can
    // still merge as orphans.
    if duration > 1e-3 {
        let content = content_char_count_in_range(chars, tokens, start, end);
        if content as f32 / duration > limits.max_cps {
            return false;
        }
    }
    true
}

fn content_char_count_in_range(
    chars: &[char],
    tokens: &[CueToken],
    start: usize,
    end: usize,
) -> usize {
    tokens[start..=end]
        .iter()
        .flat_map(|token| chars[token.char_start..token.char_end].iter().copied())
        .filter(|c| char_has_content(*c))
        .count()
}

/// Presentation caps for the segment's dominant script: CJK cues carry far
/// fewer (wider) characters per line and a lower CPS ceiling than Latin cues.
fn cue_limits(chars: &[char]) -> CueLimits {
    let mut wide = 0usize;
    let mut total = 0usize;
    for &ch in chars {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if is_wide_script(ch) {
            wide += 1;
        }
    }
    if total > 0 && wide * 2 >= total {
        CueLimits {
            char_budget: CJK_MAX_CHARS,
            max_cps: CJK_MAX_CPS,
        }
    } else {
        CueLimits {
            char_budget: LATIN_MAX_CHARS,
            max_cps: LATIN_MAX_CPS,
        }
    }
}

fn is_wide_script(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x1100..=0x115F      // Hangul Jamo
        | 0x2E80..=0x2EFF    // CJK radicals
        | 0x3000..=0x303F    // CJK symbols and punctuation
        | 0x3040..=0x30FF    // Hiragana + Katakana
        | 0x3400..=0x4DBF    // CJK Ext A
        | 0x4E00..=0x9FFF    // CJK Unified
        | 0xAC00..=0xD7A3    // Hangul syllables
        | 0xF900..=0xFAFF    // CJK compatibility ideographs
        | 0xFF00..=0xFF60    // Fullwidth forms
        | 0x20000..=0x3134F  // CJK Ext B..H
    )
}

/// Whether the token ends a sentence: its last non-closing character is
/// sentence-final punctuation. The mark may be its own token (`" . "`) or glued
/// to the last word (`"country."`).
fn ends_sentence(chars: &[char], token: &CueToken) -> bool {
    last_significant_char(chars, token).is_some_and(is_sentence_terminal_char)
}

/// Whether the token ends a clause: its last non-closing character is clause
/// punctuation (comma / semicolon / colon, ASCII or fullwidth).
fn ends_clause(chars: &[char], token: &CueToken) -> bool {
    last_significant_char(chars, token).is_some_and(is_clause_punct)
}

/// The token's last character, skipping trailing closing punctuation and
/// whitespace.
fn last_significant_char(chars: &[char], token: &CueToken) -> Option<char> {
    chars[token.char_start..token.char_end]
        .iter()
        .copied()
        .rev()
        .find(|c| !is_segment_closing_punct(*c) && !c.is_whitespace())
}

/// Whether `tokens[start..=end]` carries any non-punctuation content, so a cue
/// never consists solely of a stray punctuation token.
fn range_has_content(chars: &[char], tokens: &[CueToken], start: usize, end: usize) -> bool {
    tokens[start..=end]
        .iter()
        .flat_map(|token| chars[token.char_start..token.char_end].iter().copied())
        .any(char_has_content)
}

fn is_sentence_terminal_char(c: char) -> bool {
    matches!(
        c,
        '.' | '!' | '?' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}' | '\u{2026}'
    )
}

fn is_clause_punct(c: char) -> bool {
    matches!(
        c,
        ',' | ';' | ':' | '\u{ff0c}' | '\u{3001}' | '\u{ff1b}' | '\u{ff1a}'
    )
}

fn is_segment_closing_punct(c: char) -> bool {
    matches!(
        c,
        '"' | '\''
            | ')'
            | ']'
            | '}'
            | '\u{201d}'
            | '\u{2019}'
            | '\u{ff09}'
            | '\u{3011}'
            | '\u{300d}'
            | '\u{300f}'
    )
}

fn char_has_content(c: char) -> bool {
    !is_sentence_terminal_char(c)
        && !is_clause_punct(c)
        && !is_segment_closing_punct(c)
        && !c.is_whitespace()
}

/// Map each word token to its `[start, end)` char span in the segment `chars`
/// by greedy forward matching. Native word timestamps may retain punctuation,
/// while forced aligners commonly strip it (`hello, world` -> `hello`,
/// `world`; `你好，今天` -> one timestamp per ideograph). Try the exact form
/// first, then allow only punctuation/whitespace to be skipped while matching
/// an alphanumeric/apostrophe-only token. Separator text is attached to the
/// preceding span so cue slicing preserves the original transcript verbatim.
/// Returns `None` if content characters disagree, so the caller falls back to
/// synthesised tokens rather than mis-slicing text.
fn word_char_spans(chars: &[char], words: &[WordTimestamp]) -> Option<Vec<(usize, usize)>> {
    let mut spans = Vec::with_capacity(words.len());
    let mut idx = 0usize;
    for word in words {
        while idx < chars.len() && chars[idx].is_whitespace() {
            idx += 1;
        }
        let token: Vec<char> = word.word.trim().chars().collect();
        if token.is_empty() {
            spans.push((idx, idx));
            continue;
        }
        let (start, end) = match_word_span(chars, idx, &token)?;
        spans.push((start, end));
        idx = end;
    }
    if let Some(first) = spans.first_mut() {
        first.0 = 0;
    }
    for index in 0..spans.len().saturating_sub(1) {
        spans[index].1 = spans[index + 1].0;
    }
    if let Some(last) = spans.last_mut() {
        last.1 = chars.len();
    }
    Some(spans)
}

fn match_word_span(chars: &[char], start: usize, token: &[char]) -> Option<(usize, usize)> {
    if start + token.len() <= chars.len() && chars[start..start + token.len()] == token[..] {
        return Some((start, start + token.len()));
    }
    if !token.iter().copied().all(is_forced_alignment_char) {
        return None;
    }

    let mut cursor = start;
    let mut first = None;
    for &expected in token {
        while cursor < chars.len() && !is_forced_alignment_char(chars[cursor]) {
            cursor += 1;
        }
        if chars.get(cursor).copied() != Some(expected) {
            return None;
        }
        first.get_or_insert(cursor);
        cursor += 1;
    }
    Some((first?, cursor))
}

fn is_forced_alignment_char(ch: char) -> bool {
    ch == '\'' || ch.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let end = words.last().map_or(0.0, |w| w.end);
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
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" "),
            segments,
            ..Default::default()
        }
    }

    #[test]
    fn splits_latin_monolithic_segment_at_sentence_punctuation() {
        // Real X-ASR jfk output (detok already glued `.` to the prior word).
        let text = "And so my fellow americans ask not what your country can do for you. Ask what you can do for your country";
        let words = vec![
            word("And", 0.96, 1.00),
            word("so", 1.43, 1.47),
            word("my", 1.55, 1.59),
            word("fellow", 1.71, 1.91),
            word("americans", 2.19, 3.19),
            word("ask", 4.11, 4.14),
            word("not", 4.90, 4.94),
            word("what", 5.74, 5.78),
            word("your", 6.22, 6.26),
            word("country", 6.50, 6.54),
            word("can", 6.86, 6.89),
            word("do", 7.13, 7.17),
            word("for", 7.49, 7.53),
            word("you.", 7.93, 7.97),
            word("Ask", 8.77, 9.01),
            word("what", 9.21, 9.25),
            word("you", 9.41, 9.45),
            word("can", 9.61, 9.64),
            word("do", 9.84, 9.88),
            word("for", 10.08, 10.12),
            word("your", 10.28, 10.32),
            word("country", 10.80, 10.84),
        ];
        let cues = segment_into_cues(segment(text, words));
        // First sentence is >6s, so it splits at a clause/gap boundary too; the
        // whole thing must be at least the two sentences, none over the caps.
        assert!(cues.len() >= 2, "cues: {cues:?}");
        // Every cue is <= the hard duration cap.
        for cue in &cues {
            assert!(
                cue.end - cue.start <= MAX_CUE_SECONDS + 1e-3,
                "cue too long: {cue:?}"
            );
            assert!(cue.text.chars().count() <= LATIN_MAX_CHARS);
        }
        // Words are preserved in order across all cues.
        let joined: Vec<&str> = cues
            .iter()
            .flat_map(|c| c.words.iter().map(|w| w.word.as_str()))
            .collect();
        assert_eq!(joined.len(), 22);
        assert_eq!(joined[0], "And");
        assert_eq!(joined[21], "country");
        // A cue boundary lands on the sentence end.
        assert!(cues.iter().any(|c| c.text.ends_with("you.")));
    }

    #[test]
    fn keeps_short_single_sentence_whole() {
        let text = "hello world this is short";
        let words = vec![
            word("hello", 0.0, 0.3),
            word("world", 0.4, 0.7),
            word("this", 0.8, 1.0),
            word("is", 1.1, 1.2),
            word("short", 1.3, 1.6),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hello world this is short");
    }

    #[test]
    fn splits_cjk_segment_at_ideographic_period() {
        // Unspaced CJK with a fullwidth period; each ideograph is its own word.
        let text = "\u{4f60}\u{597d}\u{4e16}\u{754c}\u{3002}\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}";
        let words = vec![
            word("\u{4f60}", 0.0, 0.3),
            word("\u{597d}", 0.3, 0.6),
            word("\u{4e16}", 0.6, 0.9),
            word("\u{754c}\u{3002}", 0.9, 1.2),
            word("\u{4eca}", 1.3, 1.6),
            word("\u{5929}", 1.6, 1.9),
            word("\u{5929}", 1.9, 2.2),
            word("\u{6c14}", 2.2, 2.5),
            word("\u{5f88}", 2.5, 2.8),
            word("\u{597d}", 2.8, 3.1),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(cues.len(), 2, "cues: {cues:?}");
        assert_eq!(cues[0].text, "\u{4f60}\u{597d}\u{4e16}\u{754c}\u{3002}");
        assert_eq!(
            cues[1].text,
            "\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}"
        );
    }

    #[test]
    fn preserves_forced_aligner_words_across_latin_punctuation() {
        let text = "hello, world; this is a deliberately long sentence. another clause follows";
        let tokens = [
            "hello",
            "world",
            "this",
            "is",
            "a",
            "deliberately",
            "long",
            "sentence",
            "another",
            "clause",
            "follows",
        ];
        let words = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| word(token, index as f32, index as f32 + 0.5))
            .collect::<Vec<_>>();

        let cues = segment_into_cues(segment(text, words));

        assert!(
            cues.len() >= 2,
            "sentence punctuation should remain visible: {cues:?}"
        );
        let preserved = cues
            .iter()
            .flat_map(|cue| cue.words.iter().map(|word| word.word.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(preserved, tokens);
        assert!(cues.iter().any(|cue| cue.text.ends_with("sentence.")));
    }

    #[test]
    fn preserves_forced_aligner_words_across_cjk_punctuation() {
        let text = "你好世界。今天天气很好";
        let tokens = ["你", "好", "世", "界", "今", "天", "天", "气", "很", "好"];
        let words = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| word(token, index as f32 * 0.3, index as f32 * 0.3 + 0.2))
            .collect::<Vec<_>>();

        let cues = segment_into_cues(segment(text, words));

        assert_eq!(
            cues.len(),
            2,
            "ideographic period should remain visible: {cues:?}"
        );
        let preserved = cues
            .iter()
            .flat_map(|cue| cue.words.iter().map(|word| word.word.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(preserved, tokens);
        assert_eq!(cues[0].text, "你好世界。");
        assert_eq!(cues[1].text, "今天天气很好");
    }

    #[test]
    fn merges_zero_duration_forced_aligner_tail_across_sentence_boundary() {
        let text = "嗯。嗯。";
        let words = vec![word("嗯", 55.60, 55.92), word("嗯", 55.92, 55.92)];

        let cues = segment_into_cues(segment(text, words));

        assert_eq!(cues.len(), 1, "zero-duration tail must not stand alone");
        assert_eq!(cues[0].text, text);
        assert_eq!(cues[0].words.len(), 2);
        assert_eq!(cues[0].start, 55.60);
        assert_eq!(cues[0].end, 55.92);
    }

    #[test]
    fn splits_long_unpunctuated_segment_by_duration() {
        // Raw X-ASR zh-en without punctuation: a >6s run must still break by
        // duration / word gap rather than render one long cue.
        let words: Vec<WordTimestamp> = (0..10)
            .map(|i| {
                let start = i as f32 * 1.0;
                word("word", start, start + 0.5)
            })
            .collect();
        let text = "word word word word word word word word word word";
        let cues = segment_into_cues(segment(text, words));
        assert!(cues.len() >= 2, "a 9.5s run must split: {cues:?}");
        for cue in &cues {
            assert!(cue.end - cue.start <= MAX_CUE_SECONDS + 1e-3);
        }
    }

    #[test]
    fn never_crosses_speaker_turns() {
        // Two segments, distinct speakers: re-segmentation stays within each and
        // never merges across the turn boundary. Display ends also must not
        // overlap the next speaker's start.
        let mut a = segment(
            "alpha bravo charlie. delta echo foxtrot.",
            vec![
                word("alpha", 0.0, 0.3),
                word("bravo", 0.4, 0.7),
                word("charlie.", 0.8, 1.1),
                word("delta", 1.2, 1.5),
                word("echo", 1.6, 1.9),
                word("foxtrot.", 2.0, 2.3),
            ],
        );
        a.speaker = Some("SPEAKER_00".to_string());
        let mut b = segment(
            "golf hotel.",
            vec![word("golf", 2.5, 2.8), word("hotel.", 2.9, 3.2)],
        );
        b.speaker = Some("SPEAKER_01".to_string());
        let out = resegment_segments_into_cues(vec![a, b], Some(3.2));
        // Each cue carries exactly one speaker; the SPEAKER_01 content is never
        // fused with SPEAKER_00 content.
        for cue in &out {
            let speaker = cue.speaker.as_deref().unwrap();
            if cue.text.contains("golf") || cue.text.contains("hotel") {
                assert_eq!(speaker, "SPEAKER_01");
            } else {
                assert_eq!(speaker, "SPEAKER_00");
            }
        }
        assert!(
            out.iter()
                .any(|c| c.speaker.as_deref() == Some("SPEAKER_01"))
        );
        // Time assertions: no fabricated overlap across the speaker turn.
        for window in out.windows(2) {
            assert!(
                window[0].end <= window[1].start + 1e-4,
                "cues must not overlap: {:?} then {:?}",
                window[0],
                window[1]
            );
        }
        let speaker_00_end = out
            .iter()
            .filter(|c| c.speaker.as_deref() == Some("SPEAKER_00"))
            .map(|c| c.end)
            .fold(0.0f32, f32::max);
        let speaker_01_start = out
            .iter()
            .filter(|c| c.speaker.as_deref() == Some("SPEAKER_01"))
            .map(|c| c.start)
            .fold(f32::INFINITY, f32::min);
        assert!(
            speaker_00_end <= speaker_01_start + 1e-4,
            "SPEAKER_00 display end {speaker_00_end} must not cross SPEAKER_01 start {speaker_01_start}"
        );
    }

    #[test]
    fn cps_stretch_clamps_to_next_cue_start() {
        // Dense speaker-A cue abutting speaker B: CPS would want ~1.43s of
        // display for 30 Latin chars, but must not cross B at 0.5s.
        let text_a = "abcdefghijabcdefghijabcdefghij"; // 30 content chars
        let mut a = segment(text_a, vec![word(text_a, 0.0, 0.5)]);
        a.speaker = Some("SPEAKER_00".to_string());
        let mut b = segment("ok", vec![word("ok", 0.5, 0.8)]);
        b.speaker = Some("SPEAKER_01".to_string());
        let cues = resegment_segments_into_cues(vec![a, b], Some(1.0));
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, text_a);
        assert!(
            cues[0].end <= 0.5 + 1e-4,
            "A.end must not cross B.start: got {}",
            cues[0].end
        );
        assert!(
            cues[0].end <= cues[1].start + 1e-4,
            "cues must not overlap: A.end={} B.start={}",
            cues[0].end,
            cues[1].start
        );
        // Text unchanged; time is acoustic (could not meet CPS inside hard end).
        assert!((cues[0].end - 0.5).abs() < 1e-3 || cues[0].end <= 0.5);
    }

    #[test]
    fn cps_stretch_clamps_to_audio_duration() {
        // Lone dense cue: stretch toward CPS when room exists inside audio,
        // but never past the recording end.
        let text = "abcdefghijabcdefghijabcdefghij"; // 30 chars; target ~1.429s
        let seg = segment(text, vec![word(text, 0.0, 0.5)]);
        let cues = resegment_segments_into_cues(vec![seg], Some(1.0));
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, text);
        assert!(
            cues[0].end <= 1.0 + 1e-4,
            "display end must clamp to audio duration: got {}",
            cues[0].end
        );
        // 30/21 ≈ 1.429 > 1.0 hard end: refuse to fabricate past audio; keep
        // acoustic end clamped to audio.
        assert!(
            (cues[0].end - 0.5).abs() < 1e-3,
            "when CPS target exceeds audio end, keep acoustic: got {}",
            cues[0].end
        );

        // With enough audio room the same dense cue does stretch to the CPS target.
        let seg = segment(text, vec![word(text, 0.0, 0.5)]);
        let cues = resegment_segments_into_cues(vec![seg], Some(3.0));
        let content = text.chars().filter(|c| char_has_content(*c)).count();
        let target = cues[0].start + content as f32 / LATIN_MAX_CPS;
        assert!(
            (cues[0].end - target).abs() < 1e-3,
            "with room under audio end, stretch to CPS target {target}, got {}",
            cues[0].end
        );
    }

    #[test]
    fn merges_trailing_orphan_into_previous_cue() {
        // Duration forces a cut that leaves a dangling two-word tail of the
        // same sentence. Inter-word gaps stay below MIN_PAUSE_GAP_S so the
        // orphan merge is allowed (pause-split cues must not be re-glued).
        let words = vec![
            word("alpha", 0.0, 0.5),
            word("bravo", 0.6, 1.1),
            word("charlie", 1.2, 1.7),
            word("delta", 1.8, 2.3),
            word("echo", 2.4, 2.9),
            word("foxtrot", 3.0, 3.5),
            word("golf", 3.6, 4.1),
            word("hotel", 4.2, 4.7),
            word("india", 4.8, 5.3),
            word("juliet", 5.4, 5.9),
            // 0.30s gap (< MIN_PAUSE_GAP_S) is the widest, so choose_cut lands
            // here; the remaining two words are an orphan tail.
            word("kilo", 6.2, 6.5),
            word("lima", 6.55, 6.8),
        ];
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
        let cues = segment_into_cues(segment(text, words));
        assert!(
            cues.last().map(|c| c.words.len()).unwrap_or(0) > ORPHAN_MAX_WORDS || cues.len() == 1,
            "orphan tail was not merged: {cues:?}"
        );
    }

    #[test]
    fn splits_on_mid_segment_pause_even_when_under_caps() {
        // Deliberate 0.5s+ pause in the middle of a short, under-budget run:
        // the packer must cut there even though total duration < TARGET and
        // the char budget is nowhere near full.
        let text = "hello there friend today";
        let words = vec![
            word("hello", 0.0, 0.3),
            word("there", 0.4, 0.7),
            // 0.55s pause (>= MIN_PAUSE_GAP_S) between "there" and "friend".
            word("friend", 1.25, 1.55),
            word("today", 1.65, 1.95),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert!(
            cues.len() >= 2,
            "mid-segment pause must produce >=2 cues: {cues:?}"
        );
        assert!(
            cues[0].text.contains("there"),
            "first cue should end at or before the pause: {cues:?}"
        );
        assert!(
            cues.iter().any(|c| c.text.contains("friend")),
            "post-pause words must remain: {cues:?}"
        );
        // The cut lands on the pause: first cue ends at "there", second starts
        // at "friend" (no re-merge across the deliberate gap).
        assert!(
            cues[0].text.ends_with("there")
                || cues[0].words.last().map(|w| w.word.as_str()) == Some("there"),
            "pause cut should end the first cue at 'there': {cues:?}"
        );
    }

    #[test]
    fn splits_high_cps_multi_token_range() {
        // Dense multi-token burst: many content chars packed into a short
        // window so the whole run exceeds LATIN_MAX_CPS. The packer must
        // split; layout may only stretch inside the next-cue / audio hard end
        // and must not invent unbounded display times.
        let tokens = [
            "abcdefghij", // 10 chars
            "klmnopqrst", // 10
            "uvwxyzabcd", // 10
            "efghijklmn", // 10
            "opqrstuvwx", // 10
        ];
        // Five 10-char words over 1.0s total -> 50 chars / 1.0s = 50 CPS >> 21.
        let words = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let start = i as f32 * 0.2;
                word(t, start, start + 0.15)
            })
            .collect::<Vec<_>>();
        let text = tokens.join(" ");
        let acoustic = segment_into_cues(segment(&text, words.clone()));
        assert!(
            acoustic.len() >= 2,
            "high-CPS multi-token run must split: {acoustic:?}"
        );
        // Multi-token acoustic ranges must not silently exceed the CPS cap;
        // single-token leftovers may still be dense until layout stretches
        // (and only when a hard end has room).
        for cue in &acoustic {
            if cue.words.len() > 1 {
                let duration = (cue.end - cue.start).max(1e-6);
                let content = cue.text.chars().filter(|c| char_has_content(*c)).count();
                let cps = content as f32 / duration;
                assert!(
                    cps <= LATIN_MAX_CPS + 1e-3,
                    "multi-token cue CPS {cps} exceeds cap: {cue:?}"
                );
            }
        }
        let laid_out = resegment_segments_into_cues(vec![segment(&text, words)], Some(2.0));
        for window in laid_out.windows(2) {
            assert!(
                window[0].end <= window[1].start + 1e-4,
                "layout must not overlap cues: {:?} then {:?}",
                window[0],
                window[1]
            );
        }
        assert!(
            laid_out.last().unwrap().end <= 2.0 + 1e-4,
            "layout must not pass audio end"
        );
    }

    #[test]
    fn segment_into_cues_keeps_acoustic_end_without_unbounded_stretch() {
        // Packer emit path must not invent display time past the acoustic end;
        // stretch is a later layout step that sees hard bounds.
        let text = "abcdefghijabcdefghijabcdefghij"; // 30 content chars
        let words = vec![word(text, 0.0, 0.5)]; // 30/0.5 = 60 CPS >> 21
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, text);
        assert!(
            (cues[0].end - 0.5).abs() < 1e-4,
            "segment_into_cues must keep acoustic end, got {}",
            cues[0].end
        );
    }

    #[test]
    fn does_not_merge_orphan_across_deliberate_pause() {
        // Pause-split cues of orphan length must stay split; merge_orphan_tails
        // must not glue them back across gap >= MIN_PAUSE_GAP_S.
        let text = "hello world";
        let words = vec![
            word("hello", 0.0, 0.3),
            // 0.5s pause.
            word("world", 0.8, 1.1),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(
            cues.len(),
            2,
            "pause between two words must keep two cues: {cues:?}"
        );
        assert_eq!(cues[0].text, "hello");
        assert_eq!(cues[1].text, "world");
    }

    #[test]
    fn preserves_transcription_text_verbatim() {
        let text = "one two three. four five six. seven eight nine.";
        let words = vec![
            word("one", 0.0, 0.3),
            word("two", 0.4, 0.7),
            word("three.", 0.8, 1.1),
            word("four", 1.2, 1.5),
            word("five", 1.6, 1.9),
            word("six.", 2.0, 2.3),
            word("seven", 2.4, 2.7),
            word("eight", 2.8, 3.1),
            word("nine.", 3.2, 3.5),
        ];
        let original = transcription(vec![segment(text, words)]);
        let text_before = original.text.clone();
        let out = resegment_transcription_cues(original);
        assert_eq!(out.text, text_before, "joined text must be untouched");
        assert!(out.segments.len() >= 3);
    }

    #[test]
    fn synthesizes_cjk_char_tokens_without_whitespace() {
        // Unspaced CJK with no real word anchors: must still pack into short
        // cues under the CJK two-line budget (not one monolithic cue).
        let text = "你好世界今天天气很好我们一起去公园散步看花然后回家吃饭";
        let segment = Segment {
            start: 0.0,
            end: 12.0,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        };
        let cues = segment_into_cues(segment);
        assert!(
            cues.len() >= 2,
            "unspaced CJK without words must split: {cues:?}"
        );
        for cue in &cues {
            assert!(
                cue.text.chars().count() <= CJK_MAX_CHARS,
                "cue exceeds CJK two-line budget: {cue:?}"
            );
            assert!(
                cue.end - cue.start <= MAX_CUE_SECONDS + 1e-3,
                "cue exceeds hard duration: {cue:?}"
            );
        }
        let joined: String = cues.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(joined, text, "synthesised path must not rewrite text");
    }

    #[test]
    fn synthesizes_cjk_splits_at_ideographic_period() {
        let text = "你好世界。今天天气很好";
        let segment = Segment {
            start: 0.0,
            end: 4.0,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        };
        let cues = segment_into_cues(segment);
        assert_eq!(cues.len(), 2, "cues: {cues:?}");
        assert_eq!(cues[0].text, "你好世界。");
        assert_eq!(cues[1].text, "今天天气很好");
    }
}
