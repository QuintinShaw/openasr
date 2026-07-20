use thiserror::Error;

use crate::PhraseBiasConfig;
use crate::models::decode_policy_component_registry::{
    BuiltinSeq2SeqDecodePolicyConfigInput, BuiltinSeq2SeqDecodePolicyTokenSource,
    run_builtin_seq2seq_decode_policy,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperGreedyDecodeResult {
    pub generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    pub generated_probabilities: Vec<f32>,
    pub text: String,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum WhisperGreedyDecodeError {
    #[error("whisper greedy decode requires at least one initial prompt token")]
    EmptyInitialPrompt,
    #[error("whisper greedy decode requires vocab_size > 0")]
    EmptyVocab,
    #[error("whisper greedy decode requires max_generated_tokens > 0")]
    EmptyMaxGeneratedTokens,
    #[error("whisper greedy decode step {step_index} produced no logits")]
    EmptyStepLogits { step_index: usize },
    #[error(
        "whisper greedy decode step {step_index} logits width mismatch: got {got}, expected vocab_size={expected}"
    )]
    StepLogitsVocabMismatch {
        step_index: usize,
        got: usize,
        expected: usize,
    },
    #[error("whisper greedy decode step {step_index} logits contain non-finite values")]
    NonFiniteStepLogits { step_index: usize },
    #[error(
        "whisper greedy decode step {step_index} selected token id {token_id} not in vocab_size={vocab_size}"
    )]
    SelectedTokenOutOfVocab {
        step_index: usize,
        token_id: u32,
        vocab_size: usize,
    },
    #[error("whisper greedy decode reached max_generated_tokens={max_generated_tokens} before EOT")]
    EotNotReachedBeforeMaxTokens {
        max_generated_tokens: usize,
        /// Parallel to `generated_probabilities`. Preserved (not dropped)
        /// through the shared-error round trip so a caller can degrade to
        /// this partial prefix instead of failing the whole decode -- the
        /// same "no-EOT partial" pattern cohere/moonshine/qwen already use
        /// (see `ggml_executor::whisper_greedy_decode_output_or_partial`).
        generated_tokens: Vec<u32>,
        generated_probabilities: Vec<f32>,
    },
    #[error("whisper greedy decode decoder step failed: {reason}")]
    DecoderStepFailed { reason: String },
    #[error("whisper greedy decode tokenizer decode failed: {reason}")]
    TokenizerDecodeFailed { reason: String },
}

pub(crate) fn run_whisper_greedy_decode_loop(
    config: &BuiltinSeq2SeqDecodePolicyConfigInput,
    token_source: &dyn BuiltinSeq2SeqDecodePolicyTokenSource,
    phrase_bias: Option<&PhraseBiasConfig>,
    step_executor: &mut dyn Seq2SeqGreedyDecodeStepExecutor,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, WhisperGreedyDecodeError>,
) -> Result<WhisperGreedyDecodeResult, WhisperGreedyDecodeError> {
    let shared = run_builtin_seq2seq_decode_policy(
        crate::WHISPER_DECODE_POLICY_ID,
        config,
        token_source,
        phrase_bias,
        step_executor,
        decode_text_token_ids,
        map_whisper_error_to_shared,
        map_shared_error,
        map_registry_error,
    )?;
    Ok(WhisperGreedyDecodeResult {
        generated_tokens: shared.generated_tokens,
        generated_probabilities: shared.generated_probabilities,
        text: shared.text,
    })
}

fn map_whisper_error_to_shared(error: WhisperGreedyDecodeError) -> Seq2SeqGreedyDecodeError {
    match error {
        WhisperGreedyDecodeError::EmptyInitialPrompt => {
            Seq2SeqGreedyDecodeError::EmptyInitialPrompt
        }
        WhisperGreedyDecodeError::EmptyVocab => Seq2SeqGreedyDecodeError::EmptyVocab,
        WhisperGreedyDecodeError::EmptyMaxGeneratedTokens => {
            Seq2SeqGreedyDecodeError::EmptyMaxGeneratedTokens
        }
        WhisperGreedyDecodeError::EmptyStepLogits { step_index } => {
            Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index }
        }
        WhisperGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        } => Seq2SeqGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        },
        WhisperGreedyDecodeError::NonFiniteStepLogits { step_index } => {
            Seq2SeqGreedyDecodeError::NonFiniteStepLogits { step_index }
        }
        WhisperGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        } => Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        },
        WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        } => Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        },
        WhisperGreedyDecodeError::DecoderStepFailed { reason } => {
            Seq2SeqGreedyDecodeError::DecoderStepFailed { reason }
        }
        WhisperGreedyDecodeError::TokenizerDecodeFailed { reason } => {
            Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason }
        }
    }
}

fn map_shared_error(error: Seq2SeqGreedyDecodeError) -> WhisperGreedyDecodeError {
    match error {
        Seq2SeqGreedyDecodeError::EmptyInitialPrompt => {
            WhisperGreedyDecodeError::EmptyInitialPrompt
        }
        Seq2SeqGreedyDecodeError::EmptyVocab => WhisperGreedyDecodeError::EmptyVocab,
        Seq2SeqGreedyDecodeError::EmptyMaxGeneratedTokens => {
            WhisperGreedyDecodeError::EmptyMaxGeneratedTokens
        }
        Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index } => {
            WhisperGreedyDecodeError::EmptyStepLogits { step_index }
        }
        Seq2SeqGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        } => WhisperGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        },
        Seq2SeqGreedyDecodeError::NonFiniteStepLogits { step_index } => {
            WhisperGreedyDecodeError::NonFiniteStepLogits { step_index }
        }
        Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        } => WhisperGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        },
        Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        } => WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        },
        Seq2SeqGreedyDecodeError::DecoderStepFailed { reason } => {
            WhisperGreedyDecodeError::DecoderStepFailed { reason }
        }
        Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason } => {
            WhisperGreedyDecodeError::TokenizerDecodeFailed { reason }
        }
    }
}

fn map_registry_error(
    error: crate::models::decode_policy_component_registry::BuiltinDecodePolicyComponentRegistryError,
) -> WhisperGreedyDecodeError {
    WhisperGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::models::seq2seq_greedy_decode::{
        Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    };

    struct SyntheticStepExecutor {
        vocab_size: usize,
        sequence: Vec<u32>,
        logits_calls: usize,
    }

    impl Seq2SeqGreedyDecodeStepExecutor for SyntheticStepExecutor {
        fn decode_step_logits(
            &mut self,
            input: Seq2SeqGreedyDecodeStepInput<'_>,
        ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
            self.logits_calls = self.logits_calls.saturating_add(1);
            let token_id = self
                .sequence
                .get(input.step_index)
                .copied()
                .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: format!("missing synthetic token for step {}", input.step_index),
                })?;
            let token_idx = usize::try_from(token_id).map_err(|_| {
                Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: format!("synthetic token {token_id} cannot fit usize"),
                }
            })?;
            if token_idx >= self.vocab_size {
                return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: format!("synthetic token {token_id} out of vocab"),
                });
            }
            let mut logits = vec![-1000.0_f32; self.vocab_size];
            logits[token_idx] = 1000.0;
            Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            })
        }
    }

    struct SyntheticTokenSource {
        token_ids_by_content: BTreeMap<&'static str, u32>,
    }

    impl BuiltinSeq2SeqDecodePolicyTokenSource for SyntheticTokenSource {
        fn start_of_transcript_token_id(&self) -> Option<u32> {
            Some(10)
        }

        fn transcribe_token_id(&self) -> Option<u32> {
            Some(11)
        }

        fn no_timestamps_token_id(&self) -> Option<u32> {
            Some(12)
        }

        fn token_id_by_content(&self, content: &str) -> Option<u32> {
            self.token_ids_by_content.get(content).copied()
        }
    }

    impl crate::models::phrase_bias_decode::PhraseBiasTokenEncoder for SyntheticTokenSource {
        fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Ok(None)
        }
    }

    #[test]
    fn greedy_decode_turns_token_sequence_into_text() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 7],
            logits_calls: 0,
        };
        let token_table = BTreeMap::from([(1, "he"), (2, "llo")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = token_table.get(token_id) else {
                    return Err(WhisperGreedyDecodeError::TokenizerDecodeFailed {
                        reason: format!("token {token_id} missing from synthetic decoder table"),
                    });
                };
                out.push_str(piece);
            }
            Ok(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![42, 43],
            eot_token_id: 7,
            vocab_size: 16,
            max_generated_tokens: 8,
        };

        let output = run_whisper_greedy_decode_loop(
            &config,
            &SyntheticTokenSource {
                token_ids_by_content: BTreeMap::new(),
            },
            None,
            &mut step_executor,
            &decode_text_token_ids,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hello");
        assert_eq!(step_executor.logits_calls, 3);
    }

    #[test]
    fn greedy_decode_fails_closed_when_eot_is_missing() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 8,
            sequence: vec![1, 2, 3],
            logits_calls: 0,
        };
        let token_table = BTreeMap::from([(1, "a"), (2, "b"), (3, "c")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = token_table.get(token_id) else {
                    return Err(WhisperGreedyDecodeError::TokenizerDecodeFailed {
                        reason: format!("token {token_id} missing from synthetic decoder table"),
                    });
                };
                out.push_str(piece);
            }
            Ok(out)
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: vec![99],
            eot_token_id: 7,
            vocab_size: 8,
            max_generated_tokens: 3,
        };

        let error = run_whisper_greedy_decode_loop(
            &config,
            &SyntheticTokenSource {
                token_ids_by_content: BTreeMap::new(),
            },
            None,
            &mut step_executor,
            &decode_text_token_ids,
        )
        .expect_err("no EOT should fail closed");

        // Regression: this error must carry the actually-generated partial
        // prefix, not silently drop it. Before this fix,
        // `map_shared_error` discarded the shared driver's
        // `generated_tokens`/`generated_probabilities` with a `..` pattern
        // (the field didn't even exist on `WhisperGreedyDecodeError` yet),
        // so a caller had no partial output to degrade to and had to
        // hard-fail the whole transcription on any max-tokens-cap runaway
        // (e.g. an OOD decode with no `--language` hint). The executor layer
        // (`ggml_executor::run_whisper_decode_loop`) now degrades this exact
        // case to the generated prefix instead of erroring the call, mirroring
        // cohere/moonshine/qwen -- this test pins that the data survives the
        // round trip so that degrade path has something to degrade to.
        match error {
            WhisperGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                max_generated_tokens,
                generated_tokens,
                generated_probabilities,
            } => {
                assert_eq!(max_generated_tokens, 3);
                assert_eq!(generated_tokens, vec![1, 2, 3]);
                assert_eq!(generated_probabilities.len(), generated_tokens.len());
            }
            other => panic!("expected EotNotReachedBeforeMaxTokens, got {other:?}"),
        }
    }
}
