//! parakeet-ctc SentencePiece-BPE detokenizer (goal-1 S4): map collapsed CTC
//! token ids back to text. The vocab (with the `▁` U+2581 word-start marker) is
//! embedded in the pack as `tokenizer.ggml.tokens`.

use std::collections::BTreeMap;

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::GgufMetadata;
use crate::models::ctc_greedy_decode::CtcTokenFrameSpan;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};
use crate::models::sentencepiece_word_timestamps::{
    TimedSentencePieceToken, assemble_sentencepiece_word_timestamps, frame_to_seconds,
};
use crate::models::spm_decoder::{SpmDecoderConfig, decode_spm_pieces};

pub(crate) struct ParakeetTokenizer {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl ParakeetTokenizer {
    pub(crate) fn from_metadata(metadata: &GgufMetadata) -> Result<Self, String> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "parakeet-ctc pack missing tokenizer.ggml.tokens".to_string())?
            .to_vec();
        if tokens.is_empty() {
            return Err("parakeet-ctc tokenizer vocab is empty".to_string());
        }
        let token_to_id = build_token_to_id(&tokens)?;
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    /// SentencePiece detokenize: concatenate token pieces, turn the `▁` word
    /// marker into a space, and trim. (The CTC ids are already blank-collapsed.)
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

    pub(crate) fn word_timestamps_from_token_spans(
        &self,
        token_spans: &[CtcTokenFrameSpan],
        duration_seconds: f32,
        frame_count: usize,
    ) -> Result<Vec<WordTimestamp>, String> {
        if token_spans.is_empty() || frame_count == 0 || !duration_seconds.is_finite() {
            return Ok(Vec::new());
        }

        let mut timed = Vec::with_capacity(token_spans.len());
        for span in token_spans {
            let token = self.tokens.get(span.token_id as usize).ok_or_else(|| {
                format!(
                    "token id {} out of range (vocab {})",
                    span.token_id,
                    self.tokens.len()
                )
            })?;
            let start_seconds = frame_to_seconds(span.start_frame, frame_count, duration_seconds);
            let end_seconds =
                frame_to_seconds(span.end_frame, frame_count, duration_seconds).max(start_seconds);
            timed.push(TimedSentencePieceToken {
                token,
                start_seconds,
                end_seconds,
                probability: Some(span.probability),
            });
        }
        Ok(assemble_sentencepiece_word_timestamps(timed))
    }
}

impl PhraseBiasTokenEncoder for ParakeetTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_sentencepiece_phrase_bias_tokens(phrase, &self.token_to_id, "parakeet-ctc")
    }
}

fn build_token_to_id(tokens: &[String]) -> Result<BTreeMap<String, u32>, String> {
    let mut token_to_id = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let token_id =
            u32::try_from(index).map_err(|_| "parakeet-ctc token index exceeds u32".to_string())?;
        token_to_id.entry(token.clone()).or_insert(token_id);
    }
    Ok(token_to_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> ParakeetTokenizer {
        let tokens = ["▁hello", "▁wor", "ld", "▁again"]
            .iter()
            .map(|token| token.to_string())
            .collect::<Vec<_>>();
        ParakeetTokenizer {
            token_to_id: build_token_to_id(&tokens).unwrap(),
            tokens,
        }
    }

    #[test]
    fn builds_word_timestamps_from_sentencepiece_frame_spans() {
        let words = tokenizer()
            .word_timestamps_from_token_spans(
                &[
                    CtcTokenFrameSpan {
                        token_id: 0,
                        start_frame: 0,
                        end_frame: 2,
                        probability: 0.9,
                    },
                    CtcTokenFrameSpan {
                        token_id: 1,
                        start_frame: 2,
                        end_frame: 3,
                        probability: 0.8,
                    },
                    CtcTokenFrameSpan {
                        token_id: 2,
                        start_frame: 3,
                        end_frame: 4,
                        probability: 0.4,
                    },
                    CtcTokenFrameSpan {
                        token_id: 3,
                        start_frame: 4,
                        end_frame: 6,
                        probability: 1.0,
                    },
                ],
                3.0,
                6,
            )
            .expect("word timestamps");

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 1.0);
        assert_eq!(words[0].confidence, Some(0.9));
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].start, 1.0);
        assert_eq!(words[1].end, 2.0);
        // "world" = pieces "▁wor" (0.8) + "ld" (0.4) -> mean 0.6.
        assert!((words[1].confidence.unwrap() - 0.6).abs() < 1e-6);
        assert_eq!(words[2].word, "again");
        assert_eq!(words[2].start, 2.0);
        assert_eq!(words[2].end, 3.0);
        assert_eq!(words[2].confidence, Some(1.0));
    }
}
