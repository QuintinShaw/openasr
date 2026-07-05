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
}
