//! sensevoice SentencePiece-BPE detokenizer: map collapsed CTC token ids back
//! to text. The vocab (25055 pieces, `▁` word-start marker for Latin scripts,
//! bare CJK pieces, and the `<|...|>` tag pieces) is embedded in the pack as
//! `tokenizer.ggml.tokens`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::ggml_runtime::GgufMetadata;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};
use crate::models::sentencepiece_word_timestamps::WORD_START_MARKER;

pub(crate) struct SenseVoiceTokenizer {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl SenseVoiceTokenizer {
    pub(crate) fn from_metadata(metadata: &GgufMetadata) -> Result<Self, String> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "sensevoice pack missing tokenizer.ggml.tokens".to_string())?
            .to_vec();
        if tokens.is_empty() {
            return Err("sensevoice tokenizer vocab is empty".to_string());
        }
        let token_to_id = build_token_to_id(&tokens)?;
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    /// SentencePiece detokenize: concatenate token pieces, turn the `▁` word
    /// marker into a space, and trim. The raw output keeps SenseVoice's leading
    /// `<|lang|><|emotion|><|event|><|itn|>` tag pieces; the executor strips
    /// them into structured fields afterwards.
    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut out = String::new();
        for &id in ids {
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                format!("token id {id} out of range (vocab {})", self.tokens.len())
            })?;
            out.push_str(token);
        }
        Ok(out.replace(WORD_START_MARKER, " ").trim().to_string())
    }
}

impl PhraseBiasTokenEncoder for SenseVoiceTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_sentencepiece_phrase_bias_tokens(phrase, &self.token_to_id, "sensevoice")
    }

    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        let Some(primary) = self.encode_phrase_bias_tokens(phrase)? else {
            return Ok(None);
        };
        let mut variants = vec![primary.clone()];
        // SentencePiece prepends the U+2581 word-start marker to each phrase, and
        // for CJK the marker greedily encodes as its OWN standalone piece before
        // the first character. The CTC decode of CJK text emits bare character
        // pieces with no marker, so that phantom leading marker token would never
        // match the decoded stream and the hotword would never bias. Add a bare
        // variant with the standalone leading marker stripped so a CJK hotword
        // aligns with the per-character emission (the marker-prefixed form is kept
        // for phrases whose first piece legitimately carries the marker, e.g.
        // Latin words at an utterance boundary).
        if let Some((&first, rest)) = primary.split_first()
            && !rest.is_empty()
            && self
                .tokens
                .get(first as usize)
                .is_some_and(|piece| piece.chars().eq(std::iter::once(WORD_START_MARKER)))
            && !variants.contains(&rest.to_vec())
        {
            variants.push(rest.to_vec());
        }
        Ok(Some(variants))
    }
}

fn build_token_to_id(tokens: &[String]) -> Result<BTreeMap<String, u32>, String> {
    let mut token_to_id = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let token_id =
            u32::try_from(index).map_err(|_| "sensevoice token index exceeds u32".to_string())?;
        token_to_id.entry(token.clone()).or_insert(token_id);
    }
    Ok(token_to_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_mixed_cjk_and_latin_pieces() {
        let tokens: Vec<String> = ["<unk>", "▁the", "▁tribal", "开", "饭", "<|zh|>"]
            .iter()
            .map(|t| t.to_string())
            .collect();
        let tokenizer = SenseVoiceTokenizer {
            token_to_id: build_token_to_id(&tokens).unwrap(),
            tokens,
        };
        assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), "the tribal");
        assert_eq!(tokenizer.decode(&[3, 4]).unwrap(), "开饭");
        assert_eq!(tokenizer.decode(&[5, 3]).unwrap(), "<|zh|>开");
        assert!(tokenizer.decode(&[99]).is_err());
    }

    #[test]
    fn cjk_phrase_bias_adds_a_bare_variant_without_the_phantom_word_marker() {
        // A CJK phrase greedily encodes as [marker, char, char, ...] because the
        // U+2581 marker has no glued CJK piece; the model emits bare characters, so
        // the encoder must ALSO offer the marker-stripped form or the hotword never
        // aligns. tokens: 124-style marker at id 0, bare CJK chars after.
        let tokens: Vec<String> = ["\u{2581}", "\u{5201}", "\u{5929}", "\u{5bb8}"]
            .iter()
            .map(|t| t.to_string())
            .collect();
        let tokenizer = SenseVoiceTokenizer {
            token_to_id: build_token_to_id(&tokens).unwrap(),
            tokens,
        };
        let primary = tokenizer
            .encode_phrase_bias_tokens("\u{5201}\u{5929}\u{5bb8}")
            .unwrap()
            .unwrap();
        assert_eq!(primary, vec![0, 1, 2, 3]); // [marker, 刁, 天, 宸]
        let variants = tokenizer
            .encode_phrase_bias_variants("\u{5201}\u{5929}\u{5bb8}")
            .unwrap()
            .unwrap();
        // Both the marker-prefixed and the bare per-character forms are offered.
        assert!(variants.contains(&vec![0, 1, 2, 3]));
        assert!(variants.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn latin_phrase_bias_keeps_only_the_marker_glued_form() {
        // A Latin word glues the marker into its first piece (no standalone marker
        // token), so there is no phantom to strip and no spurious bare variant.
        let tokens: Vec<String> = ["\u{2581}open", "asr"]
            .iter()
            .map(|t| t.to_string())
            .collect();
        let tokenizer = SenseVoiceTokenizer {
            token_to_id: build_token_to_id(&tokens).unwrap(),
            tokens,
        };
        let variants = tokenizer
            .encode_phrase_bias_variants("openasr")
            .unwrap()
            .unwrap();
        assert_eq!(variants, vec![vec![0, 1]]);
    }
}
