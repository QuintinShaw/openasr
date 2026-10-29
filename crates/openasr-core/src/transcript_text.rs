//! Shared text projection for already segmented transcripts.

/// Characters retained in forced-alignment word units. Keep the tokenizer and
/// manuscript matching on the same policy without changing the original text.
pub(crate) fn is_alignment_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '\''
}

/// Locate the earliest native or cleaned alignment word, returning byte offsets
/// into the original UTF-8 text. Only cleaned needles may omit punctuation;
/// whitespace and retained characters must always match, including case.
pub(crate) fn find_transcript_word(text: &str, word: &str) -> Option<(usize, usize)> {
    let first = word.chars().next()?;
    let cleaned = word.chars().all(is_alignment_word_char);
    for (start, ch) in text.char_indices() {
        if ch != first {
            continue;
        }
        let candidate = &text[start..];
        if candidate.starts_with(word) {
            return Some((start, start + word.len()));
        }
        if !cleaned {
            continue;
        }
        let mut expected = word.chars().peekable();
        for (offset, ch) in candidate.char_indices() {
            if expected.peek() == Some(&ch) {
                expected.next();
                if expected.peek().is_none() {
                    return Some((start, start + offset + ch.len_utf8()));
                }
            } else if is_alignment_word_char(ch) || ch.is_whitespace() {
                break;
            }
        }
    }
    None
}

/// Trim segment boundaries and join non-empty pieces without introducing
/// spaces into CJK/fullwidth runs. Interior manuscript text is untouched.
pub(crate) fn join_segment_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for text in texts {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if needs_ascii_space_join(&out, text) {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

pub(crate) fn needs_ascii_space_join(left: &str, right: &str) -> bool {
    let (Some(prev), Some(next)) = (left.chars().next_back(), right.chars().next()) else {
        return false;
    };
    !prev.is_whitespace()
        && !next.is_whitespace()
        && !is_cjk_or_fullwidth(prev)
        && !is_cjk_or_fullwidth(next)
}

fn is_cjk_or_fullwidth(ch: char) -> bool {
    matches!(
        u32::from(ch),
        0x2E80..=0x2EFF
            | 0x3000..=0x303F
            | 0x3040..=0x30FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0xFF00..=0xFFEF
            | 0x20000..=0x3134F
    )
}

#[cfg(test)]
mod tests {
    use super::{find_transcript_word, join_segment_texts};

    #[test]
    fn alignment_matching_preserves_native_tokens_and_maps_cleaned_ranges() {
        for (text, word, original) in [
            ("中文 peut-être", "peutêtre", "peut-être"),
            ("aujourd\u{2019}hui", "aujourdhui", "aujourd\u{2019}hui"),
            ("U.S. today", "US", "U.S"),
            ("c'est-à-dire", "c'estàdire", "c'est-à-dire"),
            ("e\u{301}te\u{301}", "ete", "e\u{301}te"),
            ("hello-world", "hello-world", "hello-world"),
            ("。", "。", "。"),
        ] {
            let (start, end) = find_transcript_word(text, word).expect("matching word");
            assert_eq!(&text[start..end], original);
        }
    }

    #[test]
    fn joins_segment_boundaries_without_rewriting_interior_text() {
        for (pieces, expected) in [
            (
                vec![" 我 ", "通常会", "读书。", "", "然后散步。"],
                "我通常会读书。然后散步。",
            ),
            (
                vec![" hello ", "", " world. ", "Next sentence."],
                "hello world. Next sentence.",
            ),
            (vec!["今日は", "晴れです。"], "今日は晴れです。"),
            (vec!["안녕하세요", "세계"], "안녕하세요 세계"),
            (vec!["𠀀", "世界"], "𠀀世界"),
            (vec!["使用 OpenASR", "软件"], "使用 OpenASR软件"),
            (vec!["inside  spaces", "stay"], "inside  spaces stay"),
        ] {
            assert_eq!(join_segment_texts(pieces), expected);
        }
    }
}
