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

    /// Decode a token-id sequence to text: drop non-lexical structural
    /// tokens (`<sos>`, `<eos>`, `<blank>`, `<unk>`, `<pad>`, `<sil>`, ...),
    /// concatenate the remaining token contents, map the SPM `\u{2581}`
    /// word-boundary marker to a space, trim the edges (the upstream
    /// `detokenize` with `join_symbol=""`, `replace_spm_space=True`).
    ///
    /// The caller already excludes the leading `<sos>` prompt token and
    /// stops generation at `<eos>` (see
    /// `run_firered_aed_decoder_greedy_with_runtime`), but the model is free
    /// to emit *other* structural tokens mid-sequence -- most notably
    /// `<sil>` for trailing/leading silence -- and those must not leak into
    /// user-visible text. Rather than hardcoding a guessed token name list,
    /// treat any `dict.txt` entry that is fully bracketed (`<...>`) as
    /// structural: FireRed's char+SPM vocab never produces bracketed
    /// entries for real Chinese/English content, so this cannot misfire on
    /// normal text.
    pub(crate) fn decode(&self, token_ids: &[u32]) -> Result<String, FireRedTokenizerError> {
        let mut joined = String::new();
        for &id in token_ids {
            let content = self.token_content(id)?;
            if is_structural_token(content) {
                continue;
            }
            joined.push_str(content);
        }
        let replaced: String = joined
            .chars()
            .map(|c| if c == SPM_SPACE { ' ' } else { c })
            .collect();
        Ok(replaced.trim().to_string())
    }
}

/// A `dict.txt` entry is structural (not real transcript text) iff it is
/// wrapped in angle brackets, e.g. `<sos>`, `<eos>`, `<blank>`, `<unk>`,
/// `<pad>`, `<sil>`. The literal `<space>` entry is remapped to a plain `" "`
/// by `read_dict_txt` before it ever reaches the tokenizer, so it never hits
/// this check.
fn is_structural_token(content: &str) -> bool {
    content.len() >= 2 && content.starts_with('<') && content.ends_with('>')
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
                "<sil>",
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

    #[test]
    fn strips_trailing_sil_token_from_chinese_text() {
        // 你好 <sil> -> "你好", not "你好<sil>" (trailing-silence leak, the
        // reported bug: holding to talk and waiting past the end of speech
        // before releasing produced a visible `<sil>` in the transcript).
        assert_eq!(tokenizer().decode(&[5, 6, 11]).unwrap(), "你好");
    }

    #[test]
    fn strips_trailing_sil_token_from_english_text() {
        // ▁HELLO ▁WORL D <sil> -> "HELLO WORLD"
        assert_eq!(tokenizer().decode(&[7, 8, 9, 11]).unwrap(), "HELLO WORLD");
    }

    #[test]
    fn strips_leading_and_interior_sil_tokens() {
        // <sil> 你 <sil> 好 -> "你好" (mid-utterance silence markers must not
        // surface either).
        assert_eq!(tokenizer().decode(&[11, 5, 11, 6]).unwrap(), "你好");
    }

    #[test]
    fn strips_all_structural_tokens_leaving_only_text() {
        // <blank> <unk> <pad> <sos> <eos> 你 好 <sil> -> "你好": every
        // bracketed dict.txt entry is dropped, not just sos/eos.
        assert_eq!(
            tokenizer().decode(&[0, 1, 2, 3, 4, 5, 6, 11]).unwrap(),
            "你好"
        );
    }
}
