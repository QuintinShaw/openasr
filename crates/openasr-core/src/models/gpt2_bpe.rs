use std::collections::BTreeMap;

use crate::NativeAsrError;

pub(crate) fn build_merge_rank(merges: &[String]) -> BTreeMap<String, usize> {
    merges
        .iter()
        .enumerate()
        .map(|(index, merge)| (merge.clone(), index))
        .collect()
}

pub(crate) fn build_token_to_id(
    tokens: &[String],
    tokenizer_name: &str,
) -> Result<BTreeMap<String, u32>, NativeAsrError> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let token_id =
                u32::try_from(index).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "{tokenizer_name} tokenizer token index {index} exceeds u32 range"
                    ),
                })?;
            Ok((token.clone(), token_id))
        })
        .collect()
}

pub(crate) fn token_to_bytes(token: &str) -> Vec<u8> {
    token
        .chars()
        .map(|ch| byte_from_unicode_char(ch).unwrap_or(b'?'))
        .collect()
}

pub(crate) fn bytes_to_unicode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| unicode_char_from_byte(*byte))
        .collect()
}

pub(crate) fn encode_prompt_text(
    text: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let mut token_ids = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        if let Some((token_id, next_cursor)) = try_match_special_token(text, cursor, token_to_id) {
            token_ids.push(token_id);
            cursor = next_cursor;
            continue;
        }

        let next_special = find_next_special_token_boundary(text, cursor, token_to_id);
        let chunk = &text[cursor..next_special];
        token_ids.extend(encode_plain_text_chunk(
            chunk,
            token_to_id,
            merge_rank,
            tokenizer_name,
        )?);
        cursor = next_special;
    }
    Ok(token_ids)
}

fn try_match_special_token(
    text: &str,
    cursor: usize,
    token_to_id: &BTreeMap<String, u32>,
) -> Option<(u32, usize)> {
    let rest = text.get(cursor..)?;
    if !rest.starts_with("<|") {
        return None;
    }
    let end_offset = rest.find("|>")?;
    let token = &rest[..end_offset + 2];
    let token_id = *token_to_id.get(token)?;
    Some((token_id, cursor + end_offset + 2))
}

fn find_next_special_token_boundary(
    text: &str,
    start: usize,
    token_to_id: &BTreeMap<String, u32>,
) -> usize {
    let mut cursor = start;
    while cursor < text.len() {
        let Some(relative) = text[cursor..].find("<|") else {
            return text.len();
        };
        let candidate = cursor + relative;
        if try_match_special_token(text, candidate, token_to_id).is_some() {
            return candidate;
        }
        cursor = candidate.saturating_add(1);
    }
    text.len()
}

fn encode_plain_text_chunk(
    chunk: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let mut token_ids = Vec::new();
    for piece in gpt2_style_pretokenize(chunk) {
        if piece.is_empty() {
            continue;
        }
        token_ids.extend(encode_byte_level_piece(
            piece,
            token_to_id,
            merge_rank,
            tokenizer_name,
        )?);
    }
    Ok(token_ids)
}

/// Splits `text` into pretoken pieces per the standard "Qwen tokenizer"
/// byte-level-BPE pretokenizer regex -- the exact pattern baked into every
/// real checkpoint's `tokenizer.json` this crate has converted so far
/// (`moss-transcribe-diarize`, the official Qwen2 tokenizer firered-llm/
/// firered2-llm pack, and MiMo-V2.5-ASR all declare the byte-identical
/// `Sequence[Split(<this regex>), ByteLevel]` pretokenizer), and the same
/// pattern documented as `tiktoken`'s `cl100k_base` pretokenizer except this
/// variant matches exactly one digit per number token (`\p{N}`) instead of
/// `cl100k_base`'s up-to-three (`\p{N}{1,3}`):
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// This is hard-coded rather than read per-pack: every real pack observed
/// declares the identical regex (there is no per-family variance to honor
/// yet), `.oasr`'s `tokenizer.ggml.*` metadata contract has no slot for an
/// arbitrary pretokenizer regex today, and running an attacker-supplied
/// regex from a downloaded pack would need its own ReDoS/engine-parity
/// review before it could join the trust boundary. If a future family's
/// `tokenizer.json` ever declares a genuinely different pretokenizer, that
/// is the point to add a per-family selector here (still not raw
/// pack-supplied regex) -- see this module's doc for the exact evidence.
///
/// Implemented as a manual scanner (not a regex engine) to avoid a new
/// dependency for one fixed pattern (`AGENTS.md`'s "reuse over invent").
/// Uses `char::is_alphabetic`/`is_numeric`/`is_whitespace` as this crate's
/// existing approximation of `\p{L}`/`\p{N}`/`\s` (matches `Regex`'s classes
/// for every character this crate's real transcripts and ChatML templates
/// use: ASCII, CJK, and common punctuation; combining marks and Nl-only
/// letter-likes such as Roman numerals are out of scope).
fn gpt2_style_pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_end = |index: usize| -> usize {
        if index < n {
            chars[index].0
        } else {
            text.len()
        }
    };
    let is_punct_class =
        |ch: char| -> bool { !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric() };

    let mut pieces = Vec::new();
    let mut i = 0usize;
    while i < n {
        let start_byte = chars[i].0;
        let ch = chars[i].1;

        // (?i:'s|'t|'re|'ve|'m|'ll|'d): contraction suffixes.
        if ch == '\''
            && let Some(end) = match_contraction(&chars, i)
        {
            pieces.push(&text[start_byte..byte_end(end)]);
            i = end;
            continue;
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+: an optional single non-letter/number/CRLF
        // char immediately followed by one-or-more letters.
        if ch.is_alphabetic() {
            let mut j = i + 1;
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            pieces.push(&text[start_byte..byte_end(j)]);
            i = j;
            continue;
        }
        if ch != '\r'
            && ch != '\n'
            && !ch.is_numeric()
            && i + 1 < n
            && chars[i + 1].1.is_alphabetic()
        {
            let mut j = i + 2;
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            pieces.push(&text[start_byte..byte_end(j)]);
            i = j;
            continue;
        }

        // \p{N}: exactly one digit (this variant never groups a number run).
        if ch.is_numeric() {
            pieces.push(&text[start_byte..byte_end(i + 1)]);
            i += 1;
            continue;
        }

        // ' '?[^\s\p{L}\p{N}]+[\r\n]*: an optional single leading space, then
        // a greedy punctuation/symbol run, then any trailing CR/LF.
        let punct_run_start = if ch == ' ' && i + 1 < n && is_punct_class(chars[i + 1].1) {
            Some(i + 1)
        } else if ch != ' ' && is_punct_class(ch) {
            Some(i)
        } else {
            None
        };
        if let Some(run_start) = punct_run_start {
            let mut j = run_start;
            while j < n && is_punct_class(chars[j].1) {
                j += 1;
            }
            while j < n && (chars[j].1 == '\r' || chars[j].1 == '\n') {
                j += 1;
            }
            pieces.push(&text[start_byte..byte_end(j)]);
            i = j;
            continue;
        }

        // \s*[\r\n]+ | \s+(?!\S) | \s+: whitespace-run handling. By this
        // point every non-whitespace case above has already failed to
        // match, so `ch` is guaranteed whitespace.
        debug_assert!(
            ch.is_whitespace(),
            "unclassified non-whitespace char '{ch}'"
        );
        let mut run_end = i;
        while run_end < n && chars[run_end].1.is_whitespace() {
            run_end += 1;
        }
        let last_newline = (i..run_end)
            .rev()
            .find(|&k| matches!(chars[k].1, '\r' | '\n'));
        let consumed_end = if let Some(k) = last_newline {
            // \s*[\r\n]+: consume through the last CR/LF in this run (any
            // interior plain whitespace before it rides along with `\s*`).
            k + 1
        } else if run_end == n {
            // End of chunk: `\s+(?!\S)`'s lookahead is trivially satisfied,
            // equivalent to plain `\s+` here -- consume the whole run.
            run_end
        } else {
            // `\s+(?!\S)` prefers to leave exactly one trailing whitespace
            // char unconsumed (so it can prefix the following word via the
            // letter-run branch above), unless the run is only one char
            // long (nothing left to leave behind).
            (run_end - 1).max(i + 1)
        };
        pieces.push(&text[start_byte..byte_end(consumed_end)]);
        i = consumed_end;
    }
    pieces
}

/// Matches `(?i:'s|'t|'re|'ve|'m|'ll|'d)` starting at `chars[i]` (which must
/// be `'`). Returns the end index (exclusive, in `chars`) on success.
fn match_contraction(chars: &[(usize, char)], i: usize) -> Option<usize> {
    const SUFFIXES: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    'outer: for suffix in SUFFIXES {
        let mut j = i + 1;
        for expected in suffix.chars() {
            let Some(&(_, actual)) = chars.get(j) else {
                continue 'outer;
            };
            if !actual.eq_ignore_ascii_case(&expected) {
                continue 'outer;
            }
            j += 1;
        }
        return Some(j);
    }
    None
}

pub(crate) fn encode_byte_level_piece(
    piece: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let encoded = bytes_to_unicode(piece.as_bytes());
    bpe_encode_piece(&encoded, token_to_id, merge_rank, tokenizer_name)
}

fn bpe_encode_piece(
    encoded_piece: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    if encoded_piece.is_empty() {
        return Ok(Vec::new());
    }
    let mut symbols = encoded_piece
        .chars()
        .map(|ch| ch.to_string())
        .collect::<Vec<_>>();
    while symbols.len() > 1 {
        let mut best_pair: Option<(usize, usize)> = None;
        for pair_index in 0..symbols.len() - 1 {
            let key = format!("{} {}", symbols[pair_index], symbols[pair_index + 1]);
            let Some(rank) = merge_rank.get(&key).copied() else {
                continue;
            };
            match best_pair {
                Some((_, best_rank)) if rank >= best_rank => {}
                _ => best_pair = Some((pair_index, rank)),
            }
        }
        let Some((pair_index, _)) = best_pair else {
            break;
        };
        let merged = format!("{}{}", symbols[pair_index], symbols[pair_index + 1]);
        symbols.splice(pair_index..=pair_index + 1, [merged]);
    }

    symbols
        .into_iter()
        .map(|symbol| {
            token_to_id.get(&symbol).copied().ok_or_else(|| {
                NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "{tokenizer_name} tokenizer cannot encode BPE piece '{symbol}' from prompt text"
                    ),
                }
            })
        })
        .collect()
}

fn byte_from_unicode_char(ch: char) -> Option<u8> {
    let code = u32::from(ch);
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return u8::try_from(code).ok();
    }
    if code < 256 {
        return None;
    }

    let mut extra_code = 256u32;
    for byte in 0u32..=255 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
        {
            continue;
        }
        if extra_code == code {
            return u8::try_from(byte).ok();
        }
        extra_code += 1;
    }
    None
}

fn unicode_char_from_byte(byte: u8) -> char {
    let code = u32::from(byte);
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return char::from(byte);
    }

    let mut extra_code = 256u32;
    for candidate in 0u32..=255 {
        if (33..=126).contains(&candidate)
            || (161..=172).contains(&candidate)
            || (174..=255).contains(&candidate)
        {
            continue;
        }
        if candidate == code {
            return char::from_u32(extra_code).unwrap_or('?');
        }
        extra_code += 1;
    }
    '?'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_splits_letter_and_punctuation_runs_at_the_category_boundary() {
        // "assistant." must NOT stay one pretoken piece: a letter run and a
        // trailing punctuation char are different categories in the real
        // regex, even with no whitespace between them.
        assert_eq!(gpt2_style_pretokenize("assistant."), vec!["assistant", "."]);
    }

    #[test]
    fn pretokenize_gives_each_digit_its_own_piece_not_a_run() {
        // `\p{N}` (no `+`/`{1,3}`): this is the Qwen-specific single-digit
        // variant, unlike GPT-2/cl100k's up-to-3-digit grouping.
        assert_eq!(gpt2_style_pretokenize("101"), vec!["1", "0", "1"]);
    }

    #[test]
    fn pretokenize_matches_the_real_moss_golden_bracket_sequence() {
        // The exact fragment that exposed the original bug: real golden
        // `prompt_input_ids` decode to "[" "S" "0" "1" "]" "、" "[" ... as
        // separate tokens, never a merged "[S" -- letters/digits/punctuation
        // are three different regex categories and a bracket immediately
        // followed by a letter can never join it in one pretoken piece.
        assert_eq!(
            gpt2_style_pretokenize("（[S01]、[S02]、[S03]…）开头"),
            vec![
                "（[", "S", "0", "1", "]、[", "S", "0", "2", "]、[", "S", "0", "3", "]…）", "开头"
            ]
        );
    }

    #[test]
    fn pretokenize_attaches_a_single_leading_space_to_the_following_word() {
        assert_eq!(gpt2_style_pretokenize("You are"), vec!["You", " are"]);
    }

    #[test]
    fn pretokenize_collapses_a_multi_space_run_leaving_one_space_for_the_next_word() {
        // Three spaces before "b": `\s+(?!\S)` backtracks to leave exactly
        // one trailing space, which then prefixes "b" as its own piece.
        assert_eq!(gpt2_style_pretokenize("a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn pretokenize_treats_a_newline_run_as_one_piece_separate_from_following_whitespace() {
        assert_eq!(gpt2_style_pretokenize("a\n\nb"), vec!["a", "\n\n", "b"]);
        // A newline followed by plain spaces before a word: the newline run
        // stops at the last CR/LF; trailing spaces are handled separately.
        assert_eq!(gpt2_style_pretokenize("a\n  b"), vec!["a", "\n", " ", " b"]);
    }

    #[test]
    fn pretokenize_splits_contractions_from_the_preceding_word() {
        assert_eq!(gpt2_style_pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(
            gpt2_style_pretokenize("we'REGOING"),
            vec!["we", "'RE", "GOING"]
        );
    }

    #[test]
    fn pretokenize_glues_a_single_leading_cjk_punctuation_char_to_a_following_cjk_word() {
        // Verified directly against the real checkpoint's
        // `tokenizers::pre_tokenizer.pre_tokenize_str` output: a single
        // punctuation char immediately followed by a letter is the SAME
        // "optional leading char + word" shape as an ASCII space glued to a
        // following English word (alt2 in this module's doc comment does
        // not special-case CJK) -- "，" is not itself excluded, so it glues
        // to "我打算" exactly like a leading space glues to " are".
        assert_eq!(
            gpt2_style_pretokenize("今天天气非常好，我打算"),
            vec!["今天天气非常好", "，我打算"]
        );
    }

    #[test]
    fn pretokenize_whitespace_at_end_of_text_consumes_the_whole_run() {
        assert_eq!(gpt2_style_pretokenize("hi  "), vec!["hi", "  "]);
    }

    #[test]
    fn pretokenize_glues_a_lone_leading_bracket_to_a_following_letter() {
        // Also verified against the real tokenizer: "[S01]" in ISOLATION
        // (nothing punctuation-class immediately before the "[") DOES glue
        // "[" to "S" into one pretoken piece -- alt2's "optional leading
        // char" rule has no bracket/letter exception. This is the same rule
        // that, in the real MOSS instruction text, does NOT fire for its
        // "[S" occurrences, because there each "[" is immediately preceded
        // by another punctuation-class char ("（" or "、") that greedily
        // claims it into a punctuation run first (see the bracket-sequence
        // test below) -- the bug this module fixes was letting that
        // boundary get crossed even when nothing preceded the bracket.
        assert_eq!(gpt2_style_pretokenize("[S01]"), vec!["[S", "0", "1", "]"]);
    }

    fn tiny_vocab_with_bracket_letter_merge() -> (BTreeMap<String, u32>, BTreeMap<String, usize>) {
        // A synthetic vocab reproducing the real bug's shape: every raw
        // byte-level char used below gets its own base token (all plain
        // printable ASCII, so `bytes_to_unicode` maps each one to itself and
        // the vocab entries can be written as literal chars), PLUS an
        // anomalous "[S" entry (mirroring the real vocab's actual token
        // 42474) reachable by one merge rule bridging punctuation and a
        // letter.
        let base_chars = ["(", "[", "S", "0", "1", "]"];
        let mut tokens: Vec<String> = base_chars.iter().map(|s| s.to_string()).collect();
        tokens.push("[S".to_string());
        let token_to_id = build_token_to_id(&tokens, "test").expect("token_to_id");
        let merge_rank = build_merge_rank(&["[ S".to_string()]);
        (token_to_id, merge_rank)
    }

    #[test]
    fn encode_prompt_text_never_lets_a_punctuation_letter_merge_cross_the_pretoken_boundary() {
        // Unlike the isolated "[S01]" case above, a "(" immediately before
        // "[" claims it into a punctuation-run pretoken first (the same
        // shape as the real MOSS golden fixture's "（[" context), so "[" and
        // "S" must land in different pretoken pieces and the "[S" merge
        // rule must never get a chance to fire here even though the vocab
        // has it.
        let (token_to_id, merge_rank) = tiny_vocab_with_bracket_letter_merge();
        let ids = encode_prompt_text("([S01]", &token_to_id, &merge_rank, "test").expect("encode");
        let expected: Vec<u32> = ["(", "[", "S", "0", "1", "]"]
            .iter()
            .map(|token| *token_to_id.get(*token).unwrap())
            .collect();
        assert_eq!(
            ids, expected,
            "must never resolve to the anomalous merged '[S' token when a punctuation run claims '[' first"
        );
    }
}
