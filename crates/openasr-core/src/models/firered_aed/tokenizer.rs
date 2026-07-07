//! firered-aed detokenizer over the pack-baked `tokenizer.ggml.tokens` vocab.
//!
//! The upstream tokenizer (`ChineseCharEnglishSpmTokenizer`) is a char + SPM
//! hybrid: one Chinese char per token, English split into SentencePiece pieces
//! carrying the `\u{2581}` word-boundary marker. Inference only needs the
//! id -> text direction (`detokenize`): join token strings, replace the SPM
//! space marker with a space, trim. The SentencePiece *encoder* model is a
//! training-side asset and is deliberately not shipped in the pack.

#![allow(dead_code)]

const SPM_SPACE: char = '\u{2581}';

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedTokenizerError {
    #[error("firered token id {id} out of range for vocab of {vocab_size}")]
    TokenIdOutOfRange { id: u32, vocab_size: usize },
}

#[derive(Debug, Clone)]
pub(crate) struct FireRedTokenizer {
    tokens: Vec<String>,
}

impl FireRedTokenizer {
    pub(crate) fn new(tokens: Vec<String>) -> Self {
        Self { tokens }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn token_content(&self, id: u32) -> Result<&str, FireRedTokenizerError> {
        self.tokens.get(id as usize).map(String::as_str).ok_or(
            FireRedTokenizerError::TokenIdOutOfRange {
                id,
                vocab_size: self.tokens.len(),
            },
        )
    }

    /// Decode a token-id sequence to text: concatenate token contents, map the
    /// SPM `\u{2581}` word-boundary marker to a space, trim the edges (the
    /// upstream `detokenize` with `join_symbol=""`, `replace_spm_space=True`).
    pub(crate) fn decode(&self, token_ids: &[u32]) -> Result<String, FireRedTokenizerError> {
        let mut joined = String::new();
        for &id in token_ids {
            joined.push_str(self.token_content(id)?);
        }
        let replaced: String = joined
            .chars()
            .map(|c| if c == SPM_SPACE { ' ' } else { c })
            .collect();
        Ok(replaced.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> FireRedTokenizer {
        FireRedTokenizer::new(
            [
                "<blank>",
                "<unk>",
                "<pad>",
                "<sos>",
                "<eos>",
                "你",
                "好",
                "\u{2581}HELLO",
                "\u{2581}WORL",
                "D",
                "\u{2581}",
            ]
            .iter()
            .map(|token| token.to_string())
            .collect(),
        )
    }

    #[test]
    fn decodes_chinese_chars_without_spaces() {
        assert_eq!(tokenizer().decode(&[5, 6]).unwrap(), "你好");
    }

    #[test]
    fn decodes_spm_pieces_with_word_boundaries() {
        // ▁HELLO ▁WORL D -> "HELLO WORLD"
        assert_eq!(tokenizer().decode(&[7, 8, 9]).unwrap(), "HELLO WORLD");
    }

    #[test]
    fn mixed_zh_en_keeps_spm_boundary_only() {
        // 你 ▁HELLO 好 -> "你 HELLO好" (upstream semantics: the marker is the
        // only whitespace source).
        assert_eq!(tokenizer().decode(&[5, 7, 6]).unwrap(), "你 HELLO好");
    }

    #[test]
    fn trims_leading_boundary_marker() {
        assert_eq!(tokenizer().decode(&[10, 5]).unwrap(), "你");
    }

    #[test]
    fn rejects_out_of_range_id() {
        assert!(matches!(
            tokenizer().decode(&[99]),
            Err(FireRedTokenizerError::TokenIdOutOfRange { id: 99, .. })
        ));
    }
}
