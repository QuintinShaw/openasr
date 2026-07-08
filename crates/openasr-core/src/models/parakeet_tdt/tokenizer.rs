//! parakeet-tdt SentencePiece-BPE detokenizer: map emitted token ids back to
//! text, and assemble NATIVE word timestamps from the TDT duration head's
//! frame spans (the model's own alignment output — not a uniform
//! approximation).

use std::collections::BTreeMap;

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::GgufMetadata;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};
use crate::models::sentencepiece_word_timestamps::{
    TimedSentencePieceToken, assemble_sentencepiece_word_timestamps, frame_to_seconds,
};
use crate::models::spm_decoder::{SpmDecoderConfig, decode_spm_pieces};

use super::greedy::ParakeetTdtEmittedToken;

pub(crate) struct ParakeetTdtTokenizer {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl ParakeetTdtTokenizer {
    pub(crate) fn from_metadata(metadata: &GgufMetadata) -> Result<Self, String> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "parakeet-tdt pack missing tokenizer.ggml.tokens".to_string())?
            .to_vec();
        if tokens.is_empty() {
            return Err("parakeet-tdt tokenizer vocab is empty".to_string());
        }
        let token_to_id = build_token_to_id(&tokens)?;
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    /// SentencePiece detokenize: concatenate pieces, turn the `▁` word marker
    /// into a space, trim. (TDT emissions are already blank-free.)
    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut pieces = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                format!("token id {id} out of range (vocab {})", self.tokens.len())
            })?;
            pieces.push(token.as_str());
        }
        Ok(decode_spm_pieces(pieces, SpmDecoderConfig::PLAIN_UNIGRAM))
    }

    /// Assemble word timestamps from the emitted tokens' TDT frame spans.
    /// `frame_count` is the encoder frame count the spans index into.
    pub(crate) fn word_timestamps_from_emitted(
        &self,
        emitted: &[ParakeetTdtEmittedToken],
        duration_seconds: f32,
        frame_count: usize,
    ) -> Result<Vec<WordTimestamp>, String> {
        if emitted.is_empty() || frame_count == 0 || !duration_seconds.is_finite() {
            return Ok(Vec::new());
        }
        let mut timed = Vec::with_capacity(emitted.len());
        for token in emitted {
            let piece = self.tokens.get(token.token_id as usize).ok_or_else(|| {
                format!(
                    "token id {} out of range (vocab {})",
                    token.token_id,
                    self.tokens.len()
                )
            })?;
            let start_seconds = frame_to_seconds(token.start_frame, frame_count, duration_seconds);
            let end_seconds =
                frame_to_seconds(token.end_frame, frame_count, duration_seconds).max(start_seconds);
            timed.push(TimedSentencePieceToken {
                token: piece,
                start_seconds,
                end_seconds,
                probability: Some(token.probability),
            });
        }
        Ok(assemble_sentencepiece_word_timestamps(timed))
    }
}

impl PhraseBiasTokenEncoder for ParakeetTdtTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_sentencepiece_phrase_bias_tokens(phrase, &self.token_to_id, "parakeet-tdt")
    }
}

fn build_token_to_id(tokens: &[String]) -> Result<BTreeMap<String, u32>, String> {
    let mut token_to_id = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let token_id =
            u32::try_from(index).map_err(|_| "parakeet-tdt token index exceeds u32".to_string())?;
        token_to_id.entry(token.clone()).or_insert(token_id);
    }
    Ok(token_to_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> ParakeetTdtTokenizer {
        let tokens = ["▁hello", "▁wor", "ld", "<blank>"]
            .iter()
            .map(|token| token.to_string())
            .collect::<Vec<_>>();
        ParakeetTdtTokenizer {
            token_to_id: build_token_to_id(&tokens).unwrap(),
            tokens,
        }
    }

    #[test]
    fn decodes_sentencepiece_pieces() {
        assert_eq!(tokenizer().decode(&[0, 1, 2]).unwrap(), "hello world");
    }

    #[test]
    fn builds_native_word_timestamps_from_tdt_spans() {
        let emitted = vec![
            ParakeetTdtEmittedToken {
                token_id: 0,
                start_frame: 0,
                end_frame: 2,
                probability: 0.9,
            },
            ParakeetTdtEmittedToken {
                token_id: 1,
                start_frame: 2,
                end_frame: 3,
                probability: 0.8,
            },
            ParakeetTdtEmittedToken {
                token_id: 2,
                start_frame: 3,
                end_frame: 3,
                probability: 0.4,
            },
        ];
        let words = tokenizer()
            .word_timestamps_from_emitted(&emitted, 3.0, 6)
            .expect("word timestamps");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 1.0);
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].start, 1.0);
        // "world" = "▁wor" (0.8) + "ld" (0.4) -> mean 0.6.
        assert!((words[1].confidence.unwrap() - 0.6).abs() < 1e-6);
    }
}
