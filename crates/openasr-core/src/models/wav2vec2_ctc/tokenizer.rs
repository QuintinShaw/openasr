//! wav2vec2-ctc character-CTC detokenizer.
//!
//! wav2vec2's CTC vocab is a character vocab (e.g. `E`, `T`, `'`) plus a `|`
//! word delimiter and the special tokens `<pad>` (= the CTC blank), `<s>`,
//! `</s>`, `<unk>`. Detokenize = concatenate the (already blank-collapsed) ids'
//! characters, map `|` → space, drop the special tokens, collapse repeated
//! spaces, and trim. The vocab is embedded in the pack as `tokenizer.ggml.tokens`.

use std::collections::BTreeMap;

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::GgufMetadata;
use crate::models::ctc_greedy_decode::CtcTokenFrameSpan;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_character_ctc_phrase_bias_tokens,
};

const WORD_DELIMITER: &str = "|";
const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<s>", "</s>", "<unk>"];

pub(crate) struct Wav2Vec2Tokenizer {
    tokens: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl Wav2Vec2Tokenizer {
    pub(crate) fn from_metadata(metadata: &GgufMetadata) -> Result<Self, String> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "wav2vec2-ctc pack missing tokenizer.ggml.tokens".to_string())?
            .to_vec();
        if tokens.is_empty() {
            return Err("wav2vec2-ctc tokenizer vocab is empty".to_string());
        }
        let token_to_id = build_token_to_id(&tokens)?;
        Ok(Self {
            tokens,
            token_to_id,
        })
    }

    /// Character-CTC detokenize: ids are already blank-collapsed. Map each id to
    /// its character, turn `|` into a space, skip special tokens, then collapse
    /// repeated spaces + trim.
    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut raw = String::new();
        for &id in ids {
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                format!("token id {id} out of range (vocab {})", self.tokens.len())
            })?;
            if SPECIAL_TOKENS.contains(&token.as_str()) {
                continue;
            }
            if token == WORD_DELIMITER {
                raw.push(' ');
            } else {
                raw.push_str(token);
            }
        }
        // Collapse repeated whitespace and trim (a `|`-terminated word followed
        // by another `|`, or trailing delimiters, would otherwise leave doubled
        // spaces).
        let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        Ok(collapsed)
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

        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut current_start = 0.0_f32;
        let mut current_end = 0.0_f32;
        let mut current_probability_sum = 0.0_f32;
        let mut current_probability_count = 0_usize;

        for span in token_spans {
            let token = self.tokens.get(span.token_id as usize).ok_or_else(|| {
                format!(
                    "token id {} out of range (vocab {})",
                    span.token_id,
                    self.tokens.len()
                )
            })?;
            if SPECIAL_TOKENS.contains(&token.as_str()) {
                continue;
            }
            if token == WORD_DELIMITER {
                push_word_timestamp(
                    &mut words,
                    &mut current_word,
                    current_start,
                    current_end,
                    &mut current_probability_sum,
                    &mut current_probability_count,
                );
                continue;
            }
            let start = frame_to_seconds(span.start_frame, frame_count, duration_seconds);
            let end = frame_to_seconds(span.end_frame, frame_count, duration_seconds).max(start);
            if current_word.is_empty() {
                current_start = start;
            }
            current_word.push_str(token);
            current_end = end;
            current_probability_sum += span.probability;
            current_probability_count += 1;
        }
        push_word_timestamp(
            &mut words,
            &mut current_word,
            current_start,
            current_end,
            &mut current_probability_sum,
            &mut current_probability_count,
        );
        Ok(words)
    }
}

impl PhraseBiasTokenEncoder for Wav2Vec2Tokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_character_ctc_phrase_bias_tokens(
            phrase,
            &self.token_to_id,
            WORD_DELIMITER,
            true,
            "wav2vec2-ctc",
        )
    }
}

fn build_token_to_id(tokens: &[String]) -> Result<BTreeMap<String, u32>, String> {
    let mut token_to_id = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let token_id =
            u32::try_from(index).map_err(|_| "wav2vec2-ctc token index exceeds u32".to_string())?;
        token_to_id.entry(token.clone()).or_insert(token_id);
    }
    Ok(token_to_id)
}

fn push_word_timestamp(
    words: &mut Vec<WordTimestamp>,
    current_word: &mut String,
    start: f32,
    end: f32,
    probability_sum: &mut f32,
    probability_count: &mut usize,
) {
    if current_word.trim().is_empty() {
        current_word.clear();
        return;
    }
    let confidence = (*probability_count > 0)
        .then(|| (*probability_sum / *probability_count as f32).clamp(0.0, 1.0));
    words.push(WordTimestamp {
        word: current_word.trim().to_string(),
        start,
        end: end.max(start),
        confidence,
    });
    current_word.clear();
    *probability_sum = 0.0;
    *probability_count = 0;
}

fn frame_to_seconds(frame: usize, frame_count: usize, duration_seconds: f32) -> f32 {
    duration_seconds.max(0.0) * frame.min(frame_count) as f32 / frame_count as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Wav2Vec2Tokenizer {
        // The wav2vec2-base-960h vocab order (ids 0..=31).
        let chars = [
            "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D",
            "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z",
        ];
        Wav2Vec2Tokenizer {
            tokens: chars.iter().map(|s| s.to_string()).collect(),
            token_to_id: build_token_to_id(
                &chars.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            )
            .unwrap(),
        }
    }

    #[test]
    fn decodes_chars_word_delimiter_to_space() {
        let tk = tokenizer();
        // "THE|CAT" -> ids: T=6,H=11,E=5,|=4,C=19,A=7,T=6
        let out = tk.decode(&[6, 11, 5, 4, 19, 7, 6]).expect("decode");
        assert_eq!(out, "THE CAT");
    }

    #[test]
    fn drops_special_tokens_and_collapses_spaces() {
        let tk = tokenizer();
        // <s> A | | T </s>  ->  "A T"
        let out = tk.decode(&[1, 7, 4, 4, 6, 2]).expect("decode");
        assert_eq!(out, "A T");
    }

    #[test]
    fn rejects_out_of_range_id() {
        let tk = tokenizer();
        assert!(tk.decode(&[99]).is_err());
    }

    #[test]
    fn builds_word_timestamps_from_ctc_frame_spans() {
        let tk = tokenizer();
        let words = tk
            .word_timestamps_from_token_spans(
                &[
                    CtcTokenFrameSpan {
                        token_id: 6,
                        start_frame: 0,
                        end_frame: 1,

                        probability: 0.9,
                    },
                    CtcTokenFrameSpan {
                        token_id: 11,
                        start_frame: 1,
                        end_frame: 2,

                        probability: 0.7,
                    },
                    CtcTokenFrameSpan {
                        token_id: 5,
                        start_frame: 2,
                        end_frame: 3,

                        probability: 0.8,
                    },
                    CtcTokenFrameSpan {
                        token_id: 4,
                        start_frame: 3,
                        end_frame: 4,

                        probability: 0.5,
                    },
                    CtcTokenFrameSpan {
                        token_id: 19,
                        start_frame: 4,
                        end_frame: 5,

                        probability: 0.6,
                    },
                    CtcTokenFrameSpan {
                        token_id: 7,
                        start_frame: 5,
                        end_frame: 6,

                        probability: 1.0,
                    },
                    CtcTokenFrameSpan {
                        token_id: 6,
                        start_frame: 6,
                        end_frame: 8,

                        probability: 0.4,
                    },
                ],
                4.0,
                8,
            )
            .expect("word timestamps");

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "THE");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 1.5);
        // T(0.9) H(0.7) E(0.8) -> mean 0.8; the "|" delimiter contributes none.
        assert!((words[0].confidence.unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(words[1].word, "CAT");
        assert_eq!(words[1].start, 2.0);
        assert_eq!(words[1].end, 4.0);
        // C(0.6) A(1.0) T(0.4) -> mean 2/3.
        assert!((words[1].confidence.unwrap() - 2.0 / 3.0).abs() < 1e-6);
    }
}
