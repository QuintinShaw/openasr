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
pub(crate) struct Qwen3AsrGreedyDecodeResult {
    pub generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    pub generated_probabilities: Vec<f32>,
    pub text: String,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum Qwen3AsrGreedyDecodeError {
    #[error("qwen3-asr greedy decode requires at least one initial prompt token")]
    EmptyInitialPrompt,
    #[error("qwen3-asr greedy decode requires vocab_size > 0")]
    EmptyVocab,
    #[error("qwen3-asr greedy decode requires max_generated_tokens > 0")]
    EmptyMaxGeneratedTokens,
    #[error("qwen3-asr greedy decode step {step_index} produced no logits")]
    EmptyStepLogits { step_index: usize },
    #[error(
        "qwen3-asr greedy decode step {step_index} logits width mismatch: got {got}, expected vocab_size={expected}"
    )]
    StepLogitsVocabMismatch {
        step_index: usize,
        got: usize,
        expected: usize,
    },
    #[error("qwen3-asr greedy decode step {step_index} logits contain non-finite values")]
    NonFiniteStepLogits { step_index: usize },
    #[error(
        "qwen3-asr greedy decode step {step_index} selected token id {token_id} not in vocab_size={vocab_size}"
    )]
    SelectedTokenOutOfVocab {
        step_index: usize,
        token_id: u32,
        vocab_size: usize,
    },
    #[error(
        "qwen3-asr greedy decode reached max_generated_tokens={max_generated_tokens} before EOT; generated_tokens={generated_tokens:?}"
    )]
    EotNotReachedBeforeMaxTokens {
        max_generated_tokens: usize,
        generated_tokens: Vec<u32>,
        /// Parallel to `generated_tokens` (see the shared variant).
        generated_probabilities: Vec<f32>,
    },
    #[error("qwen3-asr greedy decode decoder step failed: {reason}")]
    DecoderStepFailed { reason: String },
    #[error("qwen3-asr greedy decode tokenizer decode failed: {reason}")]
    TokenizerDecodeFailed { reason: String },
}

pub(crate) fn run_qwen3_greedy_decode_loop(
    config: &BuiltinSeq2SeqDecodePolicyConfigInput,
    token_source: &dyn BuiltinSeq2SeqDecodePolicyTokenSource,
    phrase_bias: Option<&PhraseBiasConfig>,
    step_executor: &mut dyn Seq2SeqGreedyDecodeStepExecutor,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, Qwen3AsrGreedyDecodeError>,
) -> Result<Qwen3AsrGreedyDecodeResult, Qwen3AsrGreedyDecodeError> {
    let output = run_builtin_seq2seq_decode_policy(
        crate::QWEN3_ASR_DECODE_POLICY_ID,
        config,
        token_source,
        phrase_bias,
        step_executor,
        decode_text_token_ids,
        map_qwen_error_to_shared,
        map_shared_error,
        map_registry_error,
    )?;
    Ok(Qwen3AsrGreedyDecodeResult {
        generated_tokens: output.generated_tokens,
        generated_probabilities: output.generated_probabilities,
        text: output.text,
    })
}

fn map_qwen_error_to_shared(error: Qwen3AsrGreedyDecodeError) -> Seq2SeqGreedyDecodeError {
    match error {
        Qwen3AsrGreedyDecodeError::EmptyInitialPrompt => {
            Seq2SeqGreedyDecodeError::EmptyInitialPrompt
        }
        Qwen3AsrGreedyDecodeError::EmptyVocab => Seq2SeqGreedyDecodeError::EmptyVocab,
        Qwen3AsrGreedyDecodeError::EmptyMaxGeneratedTokens => {
            Seq2SeqGreedyDecodeError::EmptyMaxGeneratedTokens
        }
        Qwen3AsrGreedyDecodeError::EmptyStepLogits { step_index } => {
            Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index }
        }
        Qwen3AsrGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        } => Seq2SeqGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        },
        Qwen3AsrGreedyDecodeError::NonFiniteStepLogits { step_index } => {
            Seq2SeqGreedyDecodeError::NonFiniteStepLogits { step_index }
        }
        Qwen3AsrGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        } => Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        },
        Qwen3AsrGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        } => Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        },
        Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason } => {
            Seq2SeqGreedyDecodeError::DecoderStepFailed { reason }
        }
        Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed { reason } => {
            Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason }
        }
    }
}

fn map_shared_error(error: Seq2SeqGreedyDecodeError) -> Qwen3AsrGreedyDecodeError {
    match error {
        Seq2SeqGreedyDecodeError::EmptyInitialPrompt => {
            Qwen3AsrGreedyDecodeError::EmptyInitialPrompt
        }
        Seq2SeqGreedyDecodeError::EmptyVocab => Qwen3AsrGreedyDecodeError::EmptyVocab,
        Seq2SeqGreedyDecodeError::EmptyMaxGeneratedTokens => {
            Qwen3AsrGreedyDecodeError::EmptyMaxGeneratedTokens
        }
        Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index } => {
            Qwen3AsrGreedyDecodeError::EmptyStepLogits { step_index }
        }
        Seq2SeqGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        } => Qwen3AsrGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got,
            expected,
        },
        Seq2SeqGreedyDecodeError::NonFiniteStepLogits { step_index } => {
            Qwen3AsrGreedyDecodeError::NonFiniteStepLogits { step_index }
        }
        Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        } => Qwen3AsrGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        },
        Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        } => Qwen3AsrGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens,
            generated_tokens,
            generated_probabilities,
        },
        Seq2SeqGreedyDecodeError::DecoderStepFailed { reason } => {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason }
        }
        Seq2SeqGreedyDecodeError::TokenizerDecodeFailed { reason } => {
            Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed { reason }
        }
    }
}

fn map_registry_error(
    error: crate::models::decode_policy_component_registry::BuiltinDecodePolicyComponentRegistryError,
) -> Qwen3AsrGreedyDecodeError {
    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
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
    }

    impl Seq2SeqGreedyDecodeStepExecutor for SyntheticStepExecutor {
        fn decode_step_logits(
            &mut self,
            input: Seq2SeqGreedyDecodeStepInput<'_>,
        ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
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

    struct SyntheticTokenSource;

    impl BuiltinSeq2SeqDecodePolicyTokenSource for SyntheticTokenSource {
        fn audio_end_token_id(&self) -> Option<u32> {
            Some(9)
        }

        fn audio_pad_token_id(&self) -> Option<u32> {
            Some(8)
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
        };
        let token_table = BTreeMap::from([(1, "he"), (2, "llo")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = token_table.get(token_id) else {
                    return Err(Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
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

        let output = run_qwen3_greedy_decode_loop(
            &config,
            &SyntheticTokenSource,
            None,
            &mut step_executor,
            &decode_text_token_ids,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hello");
    }

    #[test]
    fn greedy_decode_strips_qwen_asr_control_prefix() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 3, 7],
        };
        let token_table = BTreeMap::from([
            (1, "language English"),
            (2, "<asr_text>"),
            (3, " transcript "),
        ]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = token_table.get(token_id) else {
                    return Err(Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
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

        let output = run_qwen3_greedy_decode_loop(
            &config,
            &SyntheticTokenSource,
            None,
            &mut step_executor,
            &decode_text_token_ids,
        )
        .unwrap();

        assert_eq!(output.text, "transcript");
    }

    #[test]
    fn greedy_decode_fails_closed_when_eot_is_missing() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 8,
            sequence: vec![1, 2, 3],
        };
        let token_table = BTreeMap::from([(1, "a"), (2, "b"), (3, "c")]);
        let decode_text_token_ids = |token_ids: &[u32]| {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = token_table.get(token_id) else {
                    return Err(Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
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

        let error = run_qwen3_greedy_decode_loop(
            &config,
            &SyntheticTokenSource,
            None,
            &mut step_executor,
            &decode_text_token_ids,
        )
        .expect_err("no EOT should fail closed");
        assert!(matches!(
            error,
            Qwen3AsrGreedyDecodeError::EotNotReachedBeforeMaxTokens { .. }
        ));
    }
}
