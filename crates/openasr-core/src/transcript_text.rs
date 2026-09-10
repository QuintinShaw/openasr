//! Shared text projection for already segmented transcripts.

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
    use super::join_segment_texts;

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
