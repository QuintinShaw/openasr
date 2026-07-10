use std::collections::{BTreeMap, BTreeSet};

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_byte_level_piece, token_to_bytes,
};
use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    required_metadata_string, required_metadata_string_array,
};
use unicode_general_category::{GeneralCategory, get_general_category};

const HYMT2_TOKENIZER_FAMILY: &str = "Hy-MT2";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_PRE_KEY: &str = "tokenizer.ggml.pre";
const TOKENIZER_GGML_PRE_VALUE_HUNYUAN_DENSE: &str = "hunyuan-dense";

pub(crate) const HYMT2_BOS_TOKEN_ID: u32 = 120_000;
pub(crate) const HYMT2_PAD_TOKEN_ID: u32 = 120_002;
pub(crate) const HYMT2_USER_TOKEN_ID: u32 = 120_006;
pub(crate) const HYMT2_ASSISTANT_TOKEN_ID: u32 = 120_007;
pub(crate) const HYMT2_EOT_TOKEN_ID: u32 = 120_008;
pub(crate) const HYMT2_EOS_TOKEN_ID: u32 = 120_020;

pub(crate) const HYMT2_BOS_TOKEN: &str = "<｜hy_begin▁of▁sentence｜>";
pub(crate) const HYMT2_PAD_TOKEN: &str = "<｜hy_▁pad▁｜>";
pub(crate) const HYMT2_USER_TOKEN: &str = "<｜hy_User｜>";
pub(crate) const HYMT2_ASSISTANT_TOKEN: &str = "<｜hy_Assistant｜>";
pub(crate) const HYMT2_EOT_TOKEN: &str = "<｜hy_EOT｜>";
pub(crate) const HYMT2_EOS_TOKEN: &str = "<｜hy_place▁holder▁no▁2｜>";

const ASCII_PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;

#[derive(Debug, Clone)]
pub struct Hymt2Tokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    special_token_ids: BTreeSet<u32>,
}

impl Hymt2Tokenizer {
    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model =
            required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY, HYMT2_TOKENIZER_FAMILY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Hy-MT2 GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }
        let tokenizer_pre =
            required_metadata_string(metadata, TOKENIZER_GGML_PRE_KEY, HYMT2_TOKENIZER_FAMILY)?;
        if !tokenizer_pre.eq_ignore_ascii_case(TOKENIZER_GGML_PRE_VALUE_HUNYUAN_DENSE) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Hy-MT2 GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_PRE_KEY, TOKENIZER_GGML_PRE_VALUE_HUNYUAN_DENSE, tokenizer_pre
                ),
            });
        }

        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            HYMT2_TOKENIZER_FAMILY,
        )?;
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            HYMT2_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() || merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: "Hy-MT2 GGUF tokenizer tokens and merges must be non-empty".to_string(),
            });
        }

        let mut id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let mut token_to_id = build_token_to_id(tokens, "Hy-MT2")?;
        patch_known_special_tokens(&mut id_to_token, &mut token_to_id);

        let merge_rank = build_merge_rank(merges);
        let special_token_ids = [
            HYMT2_BOS_TOKEN_ID,
            HYMT2_PAD_TOKEN_ID,
            HYMT2_USER_TOKEN_ID,
            HYMT2_ASSISTANT_TOKEN_ID,
            HYMT2_EOT_TOKEN_ID,
            HYMT2_EOS_TOKEN_ID,
        ]
        .into_iter()
        .map(|token_id| {
            validate_token_id_in_range(&id_to_token, token_id)?;
            Ok(token_id)
        })
        .collect::<Result<BTreeSet<_>, NativeAsrError>>()?;

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special_token_ids,
        })
    }

    pub fn encode_user_chat_prompt(&self, content: &str) -> Result<Vec<u32>, NativeAsrError> {
        let mut token_ids = Vec::with_capacity(content.len().saturating_add(3));
        token_ids.push(HYMT2_BOS_TOKEN_ID);
        token_ids.push(HYMT2_USER_TOKEN_ID);
        token_ids.extend(self.encode_content_text(content)?);
        token_ids.push(HYMT2_ASSISTANT_TOKEN_ID);
        Ok(token_ids)
    }

    pub(crate) fn encode_content_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        self.encode_plain_text_chunk(text)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        let mut token_ids = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            if let Some((token_id, next_cursor)) = self.try_match_special_token(text, cursor) {
                token_ids.push(token_id);
                cursor = next_cursor;
                continue;
            }

            let next_special = self.find_next_special_token_boundary(text, cursor);
            let chunk = &text[cursor..next_special];
            token_ids.extend(self.encode_plain_text_chunk(chunk)?);
            cursor = next_special;
        }
        Ok(token_ids)
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if self.special_token_ids.contains(token_id) {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("Hy-MT2 tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Hy-MT2 tokenizer id {token_id} is not in vocab"),
                });
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn stop_token_ids(&self) -> BTreeSet<u32> {
        [
            HYMT2_EOS_TOKEN_ID,
            HYMT2_EOT_TOKEN_ID,
            HYMT2_BOS_TOKEN_ID,
            HYMT2_USER_TOKEN_ID,
            HYMT2_ASSISTANT_TOKEN_ID,
        ]
        .into_iter()
        .collect()
    }

    fn try_match_special_token(&self, text: &str, cursor: usize) -> Option<(u32, usize)> {
        let rest = text.get(cursor..)?;
        if !rest.starts_with("<｜hy_") {
            return None;
        }
        [
            (HYMT2_BOS_TOKEN, HYMT2_BOS_TOKEN_ID),
            (HYMT2_ASSISTANT_TOKEN, HYMT2_ASSISTANT_TOKEN_ID),
            (HYMT2_USER_TOKEN, HYMT2_USER_TOKEN_ID),
            (HYMT2_EOS_TOKEN, HYMT2_EOS_TOKEN_ID),
            (HYMT2_EOT_TOKEN, HYMT2_EOT_TOKEN_ID),
            (HYMT2_PAD_TOKEN, HYMT2_PAD_TOKEN_ID),
        ]
        .into_iter()
        .filter_map(|(token, token_id)| {
            rest.starts_with(token)
                .then_some((token_id, cursor + token.len()))
        })
        .max_by_key(|(_, next_cursor)| *next_cursor)
    }

    fn find_next_special_token_boundary(&self, text: &str, start: usize) -> usize {
        let mut cursor = start;
        while cursor < text.len() {
            let Some(relative) = text[cursor..].find("<｜hy_") else {
                return text.len();
            };
            let candidate = cursor + relative;
            if self.try_match_special_token(text, candidate).is_some() {
                return candidate;
            }
            cursor = candidate.saturating_add(1);
        }
        text.len()
    }

    fn encode_plain_text_chunk(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        let mut token_ids = Vec::new();
        for piece in hunyuan_dense_pretokenize(text) {
            token_ids.extend(encode_byte_level_piece(
                piece,
                &self.token_to_id,
                &self.merge_rank,
                "Hy-MT2",
            )?);
        }
        Ok(token_ids)
    }
}

fn hunyuan_dense_pretokenize(text: &str) -> Vec<&str> {
    let numeric = split_numeric_1_to_3(text);
    let cjk = split_cjk_kana_runs(&numeric);
    cjk.into_iter()
        .flat_map(|piece| split_hunyuan_regex(piece).into_iter())
        .filter(|piece| !piece.is_empty())
        .collect()
}

fn split_numeric_1_to_3(text: &str) -> Vec<&str> {
    split_by_run(text, |ch| ch.is_numeric(), Some(3))
}

fn split_cjk_kana_runs<'a>(pieces: &[&'a str]) -> Vec<&'a str> {
    pieces
        .iter()
        .flat_map(|piece| split_by_run(piece, is_hunyuan_cjk_kana, None))
        .collect()
}

fn split_by_run(
    text: &str,
    predicate: impl Fn(char) -> bool + Copy,
    max_match_chars: Option<usize>,
) -> Vec<&str> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some(ch) = text[cursor..].chars().next() else {
            break;
        };
        if predicate(ch) {
            let start = cursor;
            let mut count = 0usize;
            while cursor < text.len() {
                let Some(next) = text[cursor..].chars().next() else {
                    break;
                };
                if !predicate(next) || max_match_chars.is_some_and(|max| count >= max) {
                    break;
                }
                cursor += next.len_utf8();
                count += 1;
            }
            out.push(&text[start..cursor]);
        } else {
            let start = cursor;
            cursor += ch.len_utf8();
            while cursor < text.len() {
                let Some(next) = text[cursor..].chars().next() else {
                    break;
                };
                if predicate(next) {
                    break;
                }
                cursor += next.len_utf8();
            }
            out.push(&text[start..cursor]);
        }
    }
    out
}

fn split_hunyuan_regex(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        if let Some(next) = match_ascii_punct_ascii_letters(text, cursor)
            .or_else(|| match_optional_prefix_letters(text, cursor))
            .or_else(|| match_optional_space_punct_symbols(text, cursor))
            .or_else(|| match_whitespace(text, cursor))
        {
            out.push(&text[cursor..next]);
            cursor = next;
            continue;
        }
        let unmatched_start = cursor;
        while cursor < text.len() {
            if match_ascii_punct_ascii_letters(text, cursor)
                .or_else(|| match_optional_prefix_letters(text, cursor))
                .or_else(|| match_optional_space_punct_symbols(text, cursor))
                .or_else(|| match_whitespace(text, cursor))
                .is_some()
            {
                break;
            }
            let ch = text[cursor..]
                .chars()
                .next()
                .expect("cursor is inside text");
            cursor += ch.len_utf8();
        }
        out.push(&text[unmatched_start..cursor]);
    }
    out
}

fn match_ascii_punct_ascii_letters(text: &str, cursor: usize) -> Option<usize> {
    let ch = text[cursor..].chars().next()?;
    if !is_ascii_punctuation(ch) {
        return None;
    }
    let mut next_cursor = cursor + ch.len_utf8();
    let first = text[next_cursor..].chars().next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    next_cursor += first.len_utf8();
    while next_cursor < text.len() {
        let next = text[next_cursor..].chars().next()?;
        if !next.is_ascii_alphabetic() {
            break;
        }
        next_cursor += next.len_utf8();
    }
    Some(next_cursor)
}

fn match_optional_prefix_letters(text: &str, cursor: usize) -> Option<usize> {
    let ch = text[cursor..].chars().next()?;
    let mut next_cursor = cursor;
    let mut letter = ch;
    if !is_cr_lf(ch) && !is_letter_or_mark(ch) && !is_punctuation_or_symbol(ch) {
        next_cursor += ch.len_utf8();
        letter = text[next_cursor..].chars().next()?;
    }
    if !is_letter_or_mark(letter) {
        return None;
    }
    next_cursor += letter.len_utf8();
    while next_cursor < text.len() {
        let next = text[next_cursor..].chars().next()?;
        if !is_letter_or_mark(next) {
            break;
        }
        next_cursor += next.len_utf8();
    }
    Some(next_cursor)
}

fn match_optional_space_punct_symbols(text: &str, cursor: usize) -> Option<usize> {
    let ch = text[cursor..].chars().next()?;
    let mut next_cursor = cursor;
    let mut symbol = ch;
    if ch == ' ' {
        next_cursor += ch.len_utf8();
        symbol = text[next_cursor..].chars().next()?;
    }
    if !is_punctuation_or_symbol(symbol) {
        return None;
    }
    next_cursor += symbol.len_utf8();
    while next_cursor < text.len() {
        let next = text[next_cursor..].chars().next()?;
        if !is_punctuation_or_symbol(next) {
            break;
        }
        next_cursor += next.len_utf8();
    }
    while next_cursor < text.len() {
        let next = text[next_cursor..].chars().next()?;
        if !is_cr_lf(next) {
            break;
        }
        next_cursor += next.len_utf8();
    }
    Some(next_cursor)
}

fn match_whitespace(text: &str, cursor: usize) -> Option<usize> {
    let ch = text[cursor..].chars().next()?;
    if !ch.is_whitespace() {
        return None;
    }
    let mut next_cursor = cursor + ch.len_utf8();
    while next_cursor < text.len() {
        let next = text[next_cursor..].chars().next()?;
        if !next.is_whitespace() {
            break;
        }
        next_cursor += next.len_utf8();
    }
    Some(next_cursor)
}

fn is_hunyuan_cjk_kana(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff
    )
}

fn is_ascii_punctuation(ch: char) -> bool {
    ch.is_ascii() && ASCII_PUNCTUATION.contains(ch)
}

fn is_letter_or_mark(ch: char) -> bool {
    ch.is_alphabetic() || is_unicode_mark(ch)
}

fn is_unicode_mark(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036f
            | 0x1ab0..=0x1aff
            | 0x1dc0..=0x1dff
            | 0x20d0..=0x20ff
            | 0xfe20..=0xfe2f
    )
}

fn is_punctuation_or_symbol(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::ClosePunctuation
            | GeneralCategory::ConnectorPunctuation
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::DashPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::MathSymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::OtherSymbol
    )
}

fn is_cr_lf(ch: char) -> bool {
    matches!(ch, '\r' | '\n')
}

const KNOWN_SPECIAL_TOKEN_PATCHES: &[(u32, &str)] = &[
    (HYMT2_BOS_TOKEN_ID, HYMT2_BOS_TOKEN),
    (120_001, "<｜hy_end▁of▁sentence｜>"),
    (HYMT2_PAD_TOKEN_ID, HYMT2_PAD_TOKEN),
    (120_003, "<｜hy_fim▁hole｜>"),
    (120_004, "<｜hy_fim▁begin｜>"),
    (120_005, "<｜hy_fim▁end｜>"),
    (HYMT2_USER_TOKEN_ID, HYMT2_USER_TOKEN),
    (HYMT2_ASSISTANT_TOKEN_ID, HYMT2_ASSISTANT_TOKEN),
    (HYMT2_EOT_TOKEN_ID, HYMT2_EOT_TOKEN),
    (120_009, "<｜hy_place▁holder▁no▁11｜>"),
    (120_010, "<｜hy_place▁holder▁no▁12｜>"),
    (120_011, "<｜hy_place▁holder▁no▁13｜>"),
    (120_012, "<｜hy_place▁holder▁no▁14｜>"),
    (120_013, "<｜hy_place▁holder▁no▁15｜>"),
    (120_014, "<｜hy_place▁holder▁no▁16｜>"),
    (120_015, "<｜hy_place▁holder▁no▁17｜>"),
    (120_016, "<｜hy_place▁holder▁no▁18｜>"),
    (120_017, "<｜hy_place▁holder▁no▁19｜>"),
    (120_018, "<｜hy_place▁holder▁no▁0｜>"),
    (120_019, "<｜hy_place▁holder▁no▁1｜>"),
    (HYMT2_EOS_TOKEN_ID, HYMT2_EOS_TOKEN),
    (120_021, "<｜hy_place▁holder▁no▁3｜>"),
    (120_022, "<｜hy_place▁holder▁no▁4｜>"),
    (120_023, "<｜hy_place▁holder▁no▁5｜>"),
    (120_024, "<｜hy_place▁holder▁no▁6｜>"),
    (120_025, "<｜hy_place▁holder▁no▁7｜>"),
    (120_026, "<｜hy_place▁holder▁no▁8｜>"),
    (120_027, "<｜hy_place▁holder▁no▁9｜>"),
    (120_028, "<｜hy_place▁holder▁no▁10｜>"),
];

fn patch_known_special_tokens(
    id_to_token: &mut [Option<String>],
    token_to_id: &mut BTreeMap<String, u32>,
) {
    for &(token_id, token_text) in KNOWN_SPECIAL_TOKEN_PATCHES {
        let Ok(index) = usize::try_from(token_id) else {
            continue;
        };
        let Some(slot) = id_to_token.get_mut(index) else {
            continue;
        };
        if let Some(previous) = slot.replace(token_text.to_string()) {
            token_to_id.remove(&previous);
        }
        token_to_id.insert(token_text.to_string(), token_id);
    }
}

fn validate_token_id_in_range(
    id_to_token: &[Option<String>],
    token_id: u32,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!("Hy-MT2 tokenizer token id {token_id} does not fit into usize"),
    })?;
    if index < id_to_token.len() {
        return Ok(());
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "Hy-MT2 tokenizer token id {token_id} is out of range for vocab size {}",
            id_to_token.len()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER_GOLDEN_JSON: &str = include_str!("fixtures/tokenizer_golden.json");

    #[test]
    fn hunyuan_pretokenizer_splits_digits_cjk_kana_and_punct_words() {
        let pieces = hunyuan_dense_pretokenize(" abc12345中文テスト!hello 🙂\n");
        assert_eq!(
            pieces,
            vec![" abc", "123", "45", "中文テスト", "!hello", " 🙂\n"]
        );
    }

    #[test]
    fn punctuation_symbol_uses_unicode_general_category_not_format_chars() {
        assert!(is_punctuation_or_symbol('。'));
        assert!(is_punctuation_or_symbol('🙂'));
        assert!(is_punctuation_or_symbol('<'));
        assert!(!is_punctuation_or_symbol('\u{200d}'));
    }

    #[test]
    fn stop_tokens_include_eos_and_role_markers() {
        let stop = [
            HYMT2_EOS_TOKEN_ID,
            HYMT2_EOT_TOKEN_ID,
            HYMT2_BOS_TOKEN_ID,
            HYMT2_USER_TOKEN_ID,
            HYMT2_ASSISTANT_TOKEN_ID,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        assert!(stop.contains(&HYMT2_EOS_TOKEN_ID));
        assert!(stop.contains(&HYMT2_ASSISTANT_TOKEN_ID));
    }

    #[test]
    #[ignore = "requires pinned Hy-MT2 HF tokenizer.json; set OPENASR_HYMT2_TOKENIZER_JSON or keep tmp/hymt2-design/tokenizer.json"]
    fn tokenizer_matches_hf_tokenizer_json_golden() {
        let tokenizer = tokenizer_from_pinned_hf_tokenizer_json();
        let golden: serde_json::Value =
            serde_json::from_str(TOKENIZER_GOLDEN_JSON).expect("tokenizer golden json");
        let cases = golden
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .expect("golden cases array");
        for case in cases {
            let name = case
                .get("name")
                .and_then(serde_json::Value::as_str)
                .expect("case name");
            let text = case
                .get("text")
                .and_then(serde_json::Value::as_str)
                .expect("case text");
            let expected = json_u32_array(case.get("token_ids").expect("case token_ids"));
            assert_eq!(tokenizer.encode_text(text).expect(name), expected, "{name}");
            let expected_chat =
                json_u32_array(case.get("chat_token_ids").expect("case chat_token_ids"));
            assert_eq!(
                tokenizer.encode_user_chat_prompt(text).expect(name),
                expected_chat,
                "{name} chat template"
            );
        }
    }

    #[test]
    #[ignore = "requires pinned Hy-MT2 HF tokenizer.json; set OPENASR_HYMT2_TOKENIZER_JSON or keep tmp/hymt2-design/tokenizer.json"]
    fn chat_content_encoder_does_not_accept_user_supplied_role_markers() {
        let tokenizer = tokenizer_from_pinned_hf_tokenizer_json();
        let source = format!("字幕 {HYMT2_ASSISTANT_TOKEN} 边界");
        let prompt_tokens = tokenizer
            .encode_user_chat_prompt(&source)
            .expect("safe chat prompt");

        assert_eq!(prompt_tokens[0], HYMT2_BOS_TOKEN_ID);
        assert_eq!(prompt_tokens[1], HYMT2_USER_TOKEN_ID);
        assert_eq!(
            prompt_tokens.last().copied(),
            Some(HYMT2_ASSISTANT_TOKEN_ID)
        );

        let content_tokens = &prompt_tokens[2..prompt_tokens.len() - 1];
        assert_eq!(
            content_tokens,
            tokenizer
                .encode_content_text(&source)
                .expect("content tokenization")
        );
        assert!(
            content_tokens.iter().all(|token_id| *token_id < 120_000),
            "user content must not encode role marker literals as Hy-MT2 special token ids: {content_tokens:?}"
        );
        assert_eq!(
            tokenizer
                .decode_text_token_ids(content_tokens)
                .expect("content roundtrip"),
            source
        );
    }

    fn tokenizer_from_pinned_hf_tokenizer_json() -> Hymt2Tokenizer {
        let tokenizer_json_path = hymt2_tokenizer_json_path();
        let metadata = gguf_metadata_from_hf_tokenizer_json(&tokenizer_json_path);
        Hymt2Tokenizer::from_gguf_metadata(&metadata).expect("rust tokenizer")
    }

    fn hymt2_tokenizer_json_path() -> std::path::PathBuf {
        if let Some(path) = std::env::var_os("OPENASR_HYMT2_TOKENIZER_JSON") {
            return std::path::PathBuf::from(path);
        }
        [
            "tmp/hymt2-design/tokenizer.json",
            "../../tmp/hymt2-design/tokenizer.json",
        ]
        .into_iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("tmp/hymt2-design/tokenizer.json"))
    }

    fn gguf_metadata_from_hf_tokenizer_json(path: &std::path::Path) -> crate::GgufMetadata {
        use std::collections::BTreeMap;

        use crate::GgufMetadataValue;

        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!(
                "read pinned Hy-MT2 tokenizer.json at {}: {error}",
                path.display()
            )
        });
        let tokenizer: serde_json::Value =
            serde_json::from_slice(&bytes).expect("HF tokenizer.json");
        let model = tokenizer.get("model").expect("tokenizer model");
        let mut by_id = BTreeMap::<usize, String>::new();
        for (token, id) in model
            .get("vocab")
            .and_then(serde_json::Value::as_object)
            .expect("model vocab")
        {
            let id = id.as_u64().expect("vocab id") as usize;
            by_id.insert(id, token.clone());
        }
        for token in tokenizer
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
            .expect("added_tokens")
        {
            let id = token
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .expect("added token id") as usize;
            let content = token
                .get("content")
                .and_then(serde_json::Value::as_str)
                .expect("added token content");
            by_id.insert(id, content.to_string());
        }
        let max_id = *by_id.keys().max().expect("non-empty vocab");
        let mut tokens = Vec::with_capacity(max_id + 1);
        for token_id in 0..=max_id {
            tokens.push(
                by_id
                    .remove(&token_id)
                    .unwrap_or_else(|| panic!("tokenizer id {token_id} is missing")),
            );
        }

        let merges = model
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .expect("model merges")
            .iter()
            .map(|merge| {
                if let Some(parts) = merge.as_array() {
                    let left = parts
                        .first()
                        .and_then(serde_json::Value::as_str)
                        .expect("merge left");
                    let right = parts
                        .get(1)
                        .and_then(serde_json::Value::as_str)
                        .expect("merge right");
                    format!("{left} {right}")
                } else {
                    merge.as_str().expect("merge string").to_string()
                }
            })
            .collect::<Vec<_>>();

        let mut values = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_PRE_KEY.to_string(),
            GgufMetadataValue::String(TOKENIZER_GGML_PRE_VALUE_HUNYUAN_DENSE.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(merges),
        );
        crate::GgufMetadata::from_values_for_test(values)
    }

    fn json_u32_array(value: &serde_json::Value) -> Vec<u32> {
        value
            .as_array()
            .expect("u32 array")
            .iter()
            .map(|value| {
                let token_id = value.as_u64().expect("token id");
                u32::try_from(token_id).expect("token id fits u32")
            })
            .collect()
    }
}
