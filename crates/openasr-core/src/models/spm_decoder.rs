//! Shared SentencePiece-style detokenizer.
//!
//! Every SPM/metaspace family (cohere, moonshine, parakeet-ctc, parakeet-tdt,
//! sensevoice, firered-aed, dolphin) hand-rolled its own version of the same
//! four mechanics: turning the `▁` (U+2581) word-start marker into whitespace,
//! fusing `<0xXX>` byte-fallback pieces into UTF-8, dropping bracketed
//! `<...>` special/structural tokens, and trimming the joined output. The
//! mechanics are identical across families; only *which* of them apply (and
//! how aggressively) differs, so this module expresses each family's
//! detokenizer as a [`SpmDecoderConfig`] value instead of a fresh
//! hand-written loop.
//!
//! Deliberately NOT centralized here: id -> token-string resolution. That
//! part genuinely differs per family (typed `NativeAsrError` vs `String`
//! errors, an owned `Vec<String>` vs an externally-supplied token table,
//! fail-closed-on-unknown-id vs silently-skip-on-unknown-id), so each
//! tokenizer keeps resolving its own ids and hands the resolved `&str`
//! pieces to [`decode_spm_pieces`].

/// SentencePiece word-start marker, U+2581 (`▁`). Kept in sync with the
/// separate copy in [`crate::models::sentencepiece_word_timestamps`], which
/// folds *timed* pieces into word spans -- a distinct concern (alignment)
/// from the plain text-joining done here.
pub(crate) const WORD_START_MARKER: char = '\u{2581}';

/// How a `▁` marker inside a piece becomes whitespace in the joined output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WordMarkerJoin {
    /// Replace every `▁` in a piece with a literal space, wherever it occurs
    /// in the piece (the marker never appears anywhere else in these
    /// families' vocabs). Used by cohere, moonshine, parakeet-ctc,
    /// parakeet-tdt, sensevoice, firered-aed.
    ReplaceEverywhere,
    /// Only a LEADING `▁` opens a new word: it is stripped and replaced by a
    /// separating space (suppressed for the very first word, so no leading
    /// space ever appears). Pieces without the marker concatenate directly
    /// with no separator, so a run of bare CJK-character pieces needs none.
    /// Used by dolphin's mixed unigram vocab, where the marker is
    /// (by construction of the upstream SentencePiece model) never anything
    /// but a piece prefix.
    LeadingMarkerOnly,
}

/// Whether bracketed `<...>` pieces (special/structural tokens) are dropped
/// from the joined output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecialTokenPolicy {
    /// Drop any piece that is fully bracketed, e.g. `<sos>`, `<eos>`,
    /// `<sil>`, `<|endoftext|>`. Used by cohere, moonshine, firered-aed,
    /// dolphin.
    DropBracketed,
    /// Keep every piece verbatim. Used by parakeet-ctc/parakeet-tdt (their
    /// emitted ids never contain bracketed pieces to begin with) and
    /// sensevoice (whose leading `<|lang|><|emotion|><|event|><|itn|>` tag
    /// pieces the executor deliberately parses back out of the raw text).
    KeepAll,
}

/// Whether `<0xXX>` byte-fallback pieces are fused into UTF-8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteFallbackPolicy {
    /// Consecutive `<0xXX>` pieces accumulate as raw bytes and are flushed as
    /// UTF-8 text as soon as a non-byte-fallback piece is seen (or at the end
    /// of the sequence); a run of bytes that is not valid UTF-8 is dropped
    /// rather than emitted lossy or garbled. Used by cohere, moonshine.
    Enabled,
    /// No byte-fallback pieces are ever produced by this family's vocab; skip
    /// the check entirely. Used by parakeet-ctc/tdt, sensevoice, firered-aed,
    /// dolphin.
    Disabled,
}

/// How the fully-joined output is trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrimPolicy {
    /// Strip a single leading space only. [`WordMarkerJoin::ReplaceEverywhere`]
    /// always turns the very first word-start marker into a leading space
    /// (there is nothing before it to separate); this removes exactly that,
    /// and nothing else. Used by cohere, moonshine.
    LeadingSpaceOnly,
    /// `str::trim()` both ends. Used by parakeet-ctc/tdt, sensevoice,
    /// firered-aed.
    Full,
    /// No trimming: the join algorithm never produces edge whitespace to
    /// begin with. Used by dolphin.
    None,
}

/// One family's full detokenize configuration. Construct via the named
/// per-family presets below rather than field literals, so a config change
/// reads as "this family now behaves like X" instead of an opaque struct
/// literal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpmDecoderConfig {
    pub word_marker_join: WordMarkerJoin,
    pub special_token_policy: SpecialTokenPolicy,
    pub byte_fallback: ByteFallbackPolicy,
    pub trim: TrimPolicy,
}

impl SpmDecoderConfig {
    /// cohere, moonshine: llama-style SPM-BPE with `<0xXX>` byte-fallback.
    /// Mirrors HF `Sequence[Replace(▁→space), ByteFallback, Fuse,
    /// Strip(start=1)]`.
    pub(crate) const BYTE_FALLBACK_BPE: Self = Self {
        word_marker_join: WordMarkerJoin::ReplaceEverywhere,
        special_token_policy: SpecialTokenPolicy::DropBracketed,
        byte_fallback: ByteFallbackPolicy::Enabled,
        trim: TrimPolicy::LeadingSpaceOnly,
    };

    /// parakeet-ctc, parakeet-tdt, sensevoice: plain unigram SentencePiece,
    /// no byte-fallback.
    pub(crate) const PLAIN_UNIGRAM: Self = Self {
        word_marker_join: WordMarkerJoin::ReplaceEverywhere,
        special_token_policy: SpecialTokenPolicy::KeepAll,
        byte_fallback: ByteFallbackPolicy::Disabled,
        trim: TrimPolicy::Full,
    };

    /// firered-aed: char + SentencePiece hybrid vocab, structural tokens
    /// dropped.
    pub(crate) const CHAR_SPM_HYBRID: Self = Self {
        word_marker_join: WordMarkerJoin::ReplaceEverywhere,
        special_token_policy: SpecialTokenPolicy::DropBracketed,
        byte_fallback: ByteFallbackPolicy::Disabled,
        trim: TrimPolicy::Full,
    };

    /// dolphin: mixed unigram vocab (bare CJK chars + marker-prefixed Latin
    /// word pieces).
    pub(crate) const MIXED_UNIGRAM_LEADING_MARKER: Self = Self {
        word_marker_join: WordMarkerJoin::LeadingMarkerOnly,
        special_token_policy: SpecialTokenPolicy::DropBracketed,
        byte_fallback: ByteFallbackPolicy::Disabled,
        trim: TrimPolicy::None,
    };
}

/// Join already-resolved SentencePiece token strings into text per `config`.
/// Callers resolve token ids to `&str` pieces themselves (see module docs for
/// why) and skip/error on unknown ids however their family requires;
/// everything downstream of "I have the piece strings in order" lives here.
pub(crate) fn decode_spm_pieces<'a>(
    pieces: impl IntoIterator<Item = &'a str>,
    config: SpmDecoderConfig,
) -> String {
    let joined = match config.word_marker_join {
        WordMarkerJoin::ReplaceEverywhere => {
            join_replace_everywhere(pieces, config.byte_fallback, config.special_token_policy)
        }
        WordMarkerJoin::LeadingMarkerOnly => {
            join_leading_marker_only(pieces, config.special_token_policy)
        }
    };
    apply_trim(joined, config.trim)
}

fn join_replace_everywhere<'a>(
    pieces: impl IntoIterator<Item = &'a str>,
    byte_fallback: ByteFallbackPolicy,
    special_token_policy: SpecialTokenPolicy,
) -> String {
    let mut output = String::new();
    let mut pending_bytes: Vec<u8> = Vec::new();

    for piece in pieces {
        if byte_fallback == ByteFallbackPolicy::Enabled
            && let Some(byte) = parse_byte_fallback_piece(piece)
        {
            pending_bytes.push(byte);
            continue;
        }
        flush_pending_bytes(&mut output, &mut pending_bytes);
        if special_token_policy == SpecialTokenPolicy::DropBracketed && is_bracketed(piece) {
            continue;
        }
        output.push_str(&piece.replace(WORD_START_MARKER, " "));
    }
    flush_pending_bytes(&mut output, &mut pending_bytes);
    output
}

fn join_leading_marker_only<'a>(
    pieces: impl IntoIterator<Item = &'a str>,
    special_token_policy: SpecialTokenPolicy,
) -> String {
    let mut output = String::new();
    for piece in pieces {
        if special_token_policy == SpecialTokenPolicy::DropBracketed && is_bracketed(piece) {
            continue;
        }
        match piece.strip_prefix(WORD_START_MARKER) {
            Some(rest) => {
                if !output.is_empty() {
                    output.push(' ');
                }
                output.push_str(rest);
            }
            None => output.push_str(piece),
        }
    }
    output
}

fn flush_pending_bytes(output: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }
    if let Ok(text) = std::str::from_utf8(pending_bytes.as_slice()) {
        output.push_str(text);
    }
    pending_bytes.clear();
}

fn apply_trim(text: String, trim: TrimPolicy) -> String {
    match trim {
        TrimPolicy::LeadingSpaceOnly => text.strip_prefix(' ').unwrap_or(&text).to_string(),
        TrimPolicy::Full => text.trim().to_string(),
        TrimPolicy::None => text,
    }
}

fn is_bracketed(piece: &str) -> bool {
    piece.len() >= 2 && piece.starts_with('<') && piece.ends_with('>')
}

/// Parse a llama-style SentencePiece byte-fallback token `<0xXX>` (exactly
/// six ASCII bytes: `<`, `0`, `x`, two hex digits, `>`) into its byte value.
pub(crate) fn parse_byte_fallback_piece(piece: &str) -> Option<u8> {
    if !piece.starts_with("<0x") || !piece.ends_with('>') || piece.len() != 6 {
        return None;
    }
    u8::from_str_radix(&piece[3..5], 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(pieces: &[&str], config: SpmDecoderConfig) -> String {
        decode_spm_pieces(pieces.iter().copied(), config)
    }

    // -- metaspace --------------------------------------------------------

    #[test]
    fn replace_everywhere_turns_every_marker_into_a_space_and_strips_one_leading_space() {
        let text = decode(
            &["\u{2581}hello", "\u{2581}world"],
            SpmDecoderConfig::PLAIN_UNIGRAM,
        );
        assert_eq!(text, "hello world");
    }

    #[test]
    fn replace_everywhere_full_trim_handles_marker_only_first_piece() {
        // First piece is the bare marker: replace-everywhere makes it a
        // leading space, full trim removes any amount of edge whitespace.
        let text = decode(&["\u{2581}", "hi"], SpmDecoderConfig::PLAIN_UNIGRAM);
        assert_eq!(text, "hi");
    }

    #[test]
    fn leading_marker_only_concatenates_bare_pieces_with_no_separator() {
        // Bare CJK-style pieces with no marker: direct concatenation.
        let text = decode(
            &["\u{4f60}", "\u{597d}"],
            SpmDecoderConfig::MIXED_UNIGRAM_LEADING_MARKER,
        );
        assert_eq!(text, "\u{4f60}\u{597d}"); // "你好"
    }

    #[test]
    fn leading_marker_only_opens_a_new_word_without_a_leading_space() {
        let text = decode(
            &["\u{2581}hello", "\u{2581}world"],
            SpmDecoderConfig::MIXED_UNIGRAM_LEADING_MARKER,
        );
        assert_eq!(text, "hello world");
    }

    #[test]
    fn leading_marker_only_mixes_cjk_and_latin_with_a_single_boundary_space() {
        // 你 ▁HELLO 好 -> "你 HELLO好": the marker is the only whitespace
        // source, so there is no space between HELLO and 好.
        let text = decode(
            &["\u{4f60}", "\u{2581}HELLO", "\u{597d}"],
            SpmDecoderConfig::MIXED_UNIGRAM_LEADING_MARKER,
        );
        assert_eq!(text, "\u{4f60} HELLO\u{597d}");
    }

    // -- byte fallback ------------------------------------------------------

    #[test]
    fn byte_fallback_fuses_multi_byte_utf8_sequences() {
        // <0xE4><0xBD><0xA0> is the UTF-8 encoding of '你'.
        let text = decode(
            &[
                "\u{2581}hello",
                "<0xE4>",
                "<0xBD>",
                "<0xA0>",
                "\u{2581}world",
            ],
            SpmDecoderConfig::BYTE_FALLBACK_BPE,
        );
        assert_eq!(text, "hello\u{4f60} world");
    }

    #[test]
    fn byte_fallback_drops_invalid_utf8_byte_runs() {
        // 0xFF is never a valid UTF-8 lead byte; the run is dropped, not
        // emitted lossily or left as raw bytes.
        let text = decode(
            &["\u{2581}a", "<0xFF>", "\u{2581}b"],
            SpmDecoderConfig::BYTE_FALLBACK_BPE,
        );
        assert_eq!(text, "a b");
    }

    #[test]
    fn non_byte_fallback_bracketed_token_is_not_mistaken_for_one() {
        assert_eq!(parse_byte_fallback_piece("<0xZZ>"), None);
        assert_eq!(parse_byte_fallback_piece("<eos>"), None);
        assert_eq!(parse_byte_fallback_piece("<0x1>"), None); // wrong length
        assert_eq!(parse_byte_fallback_piece("<0xAB>"), Some(0xAB));
    }

    // -- special token boundary ---------------------------------------------

    #[test]
    fn drop_bracketed_removes_special_and_structural_tokens() {
        let text = decode(
            &["<sos>", "\u{2581}hi", "<eos>"],
            SpmDecoderConfig::CHAR_SPM_HYBRID,
        );
        assert_eq!(text, "hi");
    }

    #[test]
    fn keep_all_leaves_tag_pieces_in_the_output() {
        let text = decode(
            &["<|zh|>", "\u{5f00}", "\u{9970}"],
            SpmDecoderConfig::PLAIN_UNIGRAM,
        );
        assert_eq!(text, "<|zh|>\u{5f00}\u{9970}");
    }

    #[test]
    fn drop_bracketed_does_not_eat_short_or_unterminated_angle_brackets() {
        // A single '<' or '>' is not itself a special token.
        let text = decode(&["<", "hi", ">"], SpmDecoderConfig::CHAR_SPM_HYBRID);
        assert_eq!(text, "<hi>");
    }

    // -- byte-fallback config end-to-end (moonshine/cohere golden shape) ----

    #[test]
    fn byte_fallback_bpe_matches_moonshine_cohere_golden_shape() {
        let text = decode(
            &[
                "\u{2581}hello",
                "<0xE4>",
                "<0xBD>",
                "<0xA0>",
                "\u{2581}world",
                "<|endoftext|>",
            ],
            SpmDecoderConfig::BYTE_FALLBACK_BPE,
        );
        assert_eq!(text, "hello\u{4f60} world");
    }
}
