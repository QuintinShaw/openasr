//! Family-agnostic re-segmentation of an assembled transcription into
//! subtitle-grade cues.
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
//! - It splits *within* the segments it is given and never merges across them,
//!   so speaker turns are preserved (a cue never spans two speakers).
//! - Word timestamps drive the boundaries when present; otherwise a segment's
//!   words are synthesised proportionally from its character span so the same
//!   packer applies.

use crate::{Segment, Transcription, WordTimestamp};

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
/// A trailing piece of this many words or fewer is treated as an orphan and
/// merged back into the previous cue when it fits within the hard caps.
const ORPHAN_MAX_WORDS: usize = 2;

/// Re-segment every segment of `transcription` into subtitle-grade cues. The
/// segment order, speaker attribution, and word sequence are preserved.
pub(crate) fn resegment_transcription_cues(mut transcription: Transcription) -> Transcription {
    if transcription.segments.is_empty() {
        return transcription;
    }
    let mut cues = Vec::with_capacity(transcription.segments.len());
    for segment in std::mem::take(&mut transcription.segments) {
        cues.extend(segment_into_cues(segment));
    }
    transcription.segments = cues;
    transcription
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

fn segment_into_cues(segment: Segment) -> Vec<Segment> {
    let chars: Vec<char> = segment.text.chars().collect();
    let (tokens, real_words) = build_tokens(&segment, &chars);
    if tokens.len() < 2 {
        return vec![segment];
    }
    let budget = char_budget(&chars);
    let ranges = pack_tokens(&chars, &tokens, budget);
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
            speaker_profile_id: segment.speaker_profile_id.clone(),
            words,
        });
    }
    if cues.len() <= 1 {
        return vec![segment];
    }
    cues
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

/// Synthesise word-sized tokens from whitespace runs, interpolating times
/// proportionally across `[segment.start, segment.end]` by character position.
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
        while index < total && !chars[index].is_whitespace() {
            index += 1;
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

/// Greedily pack tokens into cue ranges (inclusive `(first, last)` token
/// indices). Cues grow up to the target caps and break at the first sentence
/// boundary, preferring clause punctuation and then the widest word gap when a
/// long sentence must be split.
fn pack_tokens(chars: &[char], tokens: &[CueToken], budget: usize) -> Vec<(usize, usize)> {
    let n = tokens.len();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n {
        let mut end = start;
        // Grow the cue until it hits a content-bearing sentence boundary, runs
        // out of tokens, or the next token would overflow the target caps.
        while !(ends_sentence(chars, &tokens[end]) && range_has_content(chars, tokens, start, end))
            && end + 1 < n
            && fits(tokens, start, end + 1, budget, TARGET_CUE_SECONDS)
        {
            end += 1;
        }
        let cut = if (ends_sentence(chars, &tokens[end])
            && range_has_content(chars, tokens, start, end))
            || end == n - 1
        {
            end
        } else {
            choose_cut(chars, tokens, start, end)
        };
        ranges.push((start, cut));
        start = cut + 1;
    }
    merge_orphan_tails(chars, tokens, ranges, budget)
}

/// Pick the split point within `[start, end]` for a sentence that is too long
/// to keep whole: the latest clause boundary if any, else the token before the
/// widest inter-word gap, else pack to `end`.
fn choose_cut(chars: &[char], tokens: &[CueToken], start: usize, end: usize) -> usize {
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
fn merge_orphan_tails(
    chars: &[char],
    tokens: &[CueToken],
    ranges: Vec<(usize, usize)>,
    budget: usize,
) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (first, last) in ranges {
        if let Some(&(prev_first, prev_last)) = merged.last() {
            let word_count = last - first + 1;
            let prev_ends_sentence = ends_sentence(chars, &tokens[prev_last]);
            if word_count <= ORPHAN_MAX_WORDS
                && !prev_ends_sentence
                && fits(tokens, prev_first, last, budget, MAX_CUE_SECONDS)
            {
                *merged.last_mut().unwrap() = (prev_first, last);
                continue;
            }
        }
        merged.push((first, last));
    }
    merged
}

/// Whether `tokens[start..=end]` fits both the character budget and `max_seconds`.
fn fits(tokens: &[CueToken], start: usize, end: usize, budget: usize, max_seconds: f32) -> bool {
    let chars = tokens[end]
        .char_end
        .saturating_sub(tokens[start].char_start);
    if chars > budget {
        return false;
    }
    let duration = tokens[end].end - tokens[start].start;
    duration <= max_seconds
}

/// Character budget for the segment's dominant script: CJK cues carry far fewer
/// (wider) characters per line than Latin cues.
fn char_budget(chars: &[char]) -> usize {
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
        CJK_MAX_CHARS
    } else {
        LATIN_MAX_CHARS
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
/// by greedy forward matching (words are whitespace-separated tokens of the
/// text). Returns `None` if a token does not align, so the caller falls back to
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
        if idx + token.len() > chars.len() {
            return None;
        }
        if chars[idx..idx + token.len()] != token[..] {
            return None;
        }
        spans.push((idx, idx + token.len()));
        idx += token.len();
    }
    Some(spans)
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
            speaker_profile_id: None,
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
            longform: None,
            language: None,
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
        // never merges across the turn boundary.
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
        let out = resegment_transcription_cues(transcription(vec![a, b]));
        // Each cue carries exactly one speaker; the SPEAKER_01 content is never
        // fused with SPEAKER_00 content.
        for cue in &out.segments {
            let speaker = cue.speaker.as_deref().unwrap();
            if cue.text.contains("golf") || cue.text.contains("hotel") {
                assert_eq!(speaker, "SPEAKER_01");
            } else {
                assert_eq!(speaker, "SPEAKER_00");
            }
        }
        assert!(
            out.segments
                .iter()
                .any(|c| c.speaker.as_deref() == Some("SPEAKER_01"))
        );
    }

    #[test]
    fn merges_trailing_orphan_into_previous_cue() {
        // A long clause followed by a dangling two-word tail of the same
        // sentence: the tail must not become its own cue.
        let words = vec![
            word("the", 0.0, 0.3),
            word("quick", 0.5, 0.9),
            word("brown", 1.2, 1.6),
            word("fox", 2.0, 2.4),
            word("jumps,", 3.0, 3.4),
            word("over", 5.6, 5.9),
            word("it", 6.0, 6.2),
        ];
        let text = "the quick brown fox jumps, over it";
        let cues = segment_into_cues(segment(text, words));
        // "over it" (2 words) would orphan after the clause cut; it is merged
        // back so no cue is a lone 1-2 word tail.
        assert!(
            cues.last().map(|c| c.words.len()).unwrap_or(0) > ORPHAN_MAX_WORDS || cues.len() == 1,
            "orphan tail was not merged: {cues:?}"
        );
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
}
