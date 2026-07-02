use thiserror::Error;

use crate::models::phrase_bias_decode::{TokenPhraseBias, apply_phrase_bias_to_logits};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Seq2SeqGreedyDecodeConfig {
    pub initial_prompt_tokens: Vec<u32>,
    pub eot_token_id: u32,
    pub stop_token_ids: Vec<u32>,
    pub vocab_size: usize,
    pub max_generated_tokens: usize,
    pub suppress_first_step_token_ids: Vec<u32>,
    pub suppress_token_ids: Vec<u32>,
    pub phrase_biases: Vec<TokenPhraseBias>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Seq2SeqGreedyDecodeStepInput<'a> {
    pub initial_prompt_tokens: &'a [u32],
    pub generated_tokens: &'a [u32],
    pub step_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Seq2SeqGreedyDecodeStepLogitsOutput {
    pub logits: Vec<f32>,
    pub greedy_token_hint: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Seq2SeqGreedyStepSelection {
    pub token_id: u32,
    pub reached_eot: bool,
    /// Softmax probability of the selected token over this step's logit row
    /// (the suppressed/biased row on the host-argmax path; the raw row on the
    /// device-hint path — suppress lists are a handful of special tokens, so
    /// the denominators differ negligibly).
    pub probability: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Seq2SeqGreedyDecodeResult {
    pub generated_tokens: Vec<u32>,
    /// Per-token softmax probability, parallel to `generated_tokens`.
    pub generated_probabilities: Vec<f32>,
    pub text: String,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum Seq2SeqGreedyDecodeError {
    #[error("seq2seq greedy decode requires at least one initial prompt token")]
    EmptyInitialPrompt,
    #[error("seq2seq greedy decode requires vocab_size > 0")]
    EmptyVocab,
    #[error("seq2seq greedy decode requires max_generated_tokens > 0")]
    EmptyMaxGeneratedTokens,
    #[error("seq2seq greedy decode step {step_index} produced no logits")]
    EmptyStepLogits { step_index: usize },
    #[error(
        "seq2seq greedy decode step {step_index} logits width mismatch: got {got}, expected vocab_size={expected}"
    )]
    StepLogitsVocabMismatch {
        step_index: usize,
        got: usize,
        expected: usize,
    },
    #[error("seq2seq greedy decode step {step_index} logits contain non-finite values")]
    NonFiniteStepLogits { step_index: usize },
    #[error(
        "seq2seq greedy decode step {step_index} selected token id {token_id} not in vocab_size={vocab_size}"
    )]
    SelectedTokenOutOfVocab {
        step_index: usize,
        token_id: u32,
        vocab_size: usize,
    },
    #[error("seq2seq greedy decode reached max_generated_tokens={max_generated_tokens} before EOT")]
    EotNotReachedBeforeMaxTokens {
        max_generated_tokens: usize,
        generated_tokens: Vec<u32>,
        /// Parallel to `generated_tokens`: callers that degrade to the partial
        /// prefix keep its word confidence instead of discarding real scores.
        generated_probabilities: Vec<f32>,
    },
    #[error("seq2seq greedy decode decoder step failed: {reason}")]
    DecoderStepFailed { reason: String },
    #[error("seq2seq greedy decode tokenizer decode failed: {reason}")]
    TokenizerDecodeFailed { reason: String },
}

pub(crate) trait Seq2SeqGreedyDecodeStepExecutor {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError>;
}

pub(crate) trait Seq2SeqGreedyTokenDecoder {
    fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, Seq2SeqGreedyDecodeError>;
}

pub(crate) fn run_seq2seq_greedy_decode_loop_with_adapter_v0<E>(
    config: &Seq2SeqGreedyDecodeConfig,
    step_executor: &mut dyn Seq2SeqGreedyDecodeStepExecutor,
    decode_text_token_ids: &dyn Fn(&[u32]) -> Result<String, E>,
    map_token_decoder_error_to_shared: fn(E) -> Seq2SeqGreedyDecodeError,
    map_shared_error_to_family: fn(Seq2SeqGreedyDecodeError) -> E,
    normalize_text: &dyn Fn(String) -> String,
    trace_token: &mut dyn FnMut(usize, u32, bool),
    on_topk: &mut dyn FnMut(usize, &[f32]),
) -> Result<Seq2SeqGreedyDecodeResult, E> {
    struct ClosureTokenDecoder<'a, E> {
        decode_text_token_ids: &'a dyn Fn(&[u32]) -> Result<String, E>,
        map_family_error_to_shared: fn(E) -> Seq2SeqGreedyDecodeError,
    }

    impl<E> Seq2SeqGreedyTokenDecoder for ClosureTokenDecoder<'_, E> {
        fn decode_text_token_ids(
            &self,
            token_ids: &[u32],
        ) -> Result<String, Seq2SeqGreedyDecodeError> {
            (self.decode_text_token_ids)(token_ids).map_err(self.map_family_error_to_shared)
        }
    }

    let token_decoder = ClosureTokenDecoder {
        decode_text_token_ids,
        map_family_error_to_shared: map_token_decoder_error_to_shared,
    };
    let output = run_seq2seq_greedy_decode_loop_v0(
        config,
        step_executor,
        &token_decoder,
        trace_token,
        on_topk,
    )
    .map_err(map_shared_error_to_family)?;
    Ok(Seq2SeqGreedyDecodeResult {
        generated_tokens: output.generated_tokens,
        generated_probabilities: output.generated_probabilities,
        text: normalize_text(output.text),
    })
}

pub(crate) fn run_seq2seq_greedy_decode_loop_v0(
    config: &Seq2SeqGreedyDecodeConfig,
    step_executor: &mut dyn Seq2SeqGreedyDecodeStepExecutor,
    token_decoder: &dyn Seq2SeqGreedyTokenDecoder,
    trace_token: &mut dyn FnMut(usize, u32, bool),
    on_topk: &mut dyn FnMut(usize, &[f32]),
) -> Result<Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeError> {
    if config.initial_prompt_tokens.is_empty() {
        return Err(Seq2SeqGreedyDecodeError::EmptyInitialPrompt);
    }
    if config.vocab_size == 0 {
        return Err(Seq2SeqGreedyDecodeError::EmptyVocab);
    }
    if config.max_generated_tokens == 0 {
        return Err(Seq2SeqGreedyDecodeError::EmptyMaxGeneratedTokens);
    }

    let stop_token_ids = build_seq2seq_greedy_stop_token_ids(config);
    let mut generated = Vec::new();
    let mut generated_probabilities = Vec::new();
    let mut reached_eot = false;

    for step_index in 0..config.max_generated_tokens {
        let step_input = Seq2SeqGreedyDecodeStepInput {
            initial_prompt_tokens: &config.initial_prompt_tokens,
            generated_tokens: &generated,
            step_index,
        };
        let step_logits = step_executor.decode_step_logits(step_input)?;
        let selection = select_seq2seq_greedy_step_token(
            config,
            &generated,
            step_index,
            step_logits,
            stop_token_ids.as_slice(),
            on_topk,
        )?;
        trace_token(step_index, selection.token_id, selection.reached_eot);
        if selection.reached_eot {
            reached_eot = true;
            break;
        }
        generated.push(selection.token_id);
        generated_probabilities.push(selection.probability);
    }

    if !reached_eot {
        return Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            max_generated_tokens: config.max_generated_tokens,
            generated_tokens: generated,
            generated_probabilities,
        });
    }

    let text = token_decoder.decode_text_token_ids(&generated)?;
    Ok(Seq2SeqGreedyDecodeResult {
        generated_tokens: generated,
        generated_probabilities,
        text,
    })
}

pub(crate) fn select_seq2seq_greedy_step_token(
    config: &Seq2SeqGreedyDecodeConfig,
    generated_tokens: &[u32],
    step_index: usize,
    step_logits: Seq2SeqGreedyDecodeStepLogitsOutput,
    stop_token_ids: &[u32],
    on_topk: &mut dyn FnMut(usize, &[f32]),
) -> Result<Seq2SeqGreedyStepSelection, Seq2SeqGreedyDecodeError> {
    if step_logits.logits.is_empty() {
        return Err(Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index });
    }
    if step_logits.logits.len() != config.vocab_size {
        return Err(Seq2SeqGreedyDecodeError::StepLogitsVocabMismatch {
            step_index,
            got: step_logits.logits.len(),
            expected: config.vocab_size,
        });
    }
    if config.phrase_biases.is_empty()
        && let Some(next_token) = step_logits.greedy_token_hint
    {
        validate_selected_token(step_index, next_token, config.vocab_size)?;
        let is_suppressed = config.suppress_token_ids.contains(&next_token)
            || (step_index == 0 && config.suppress_first_step_token_ids.contains(&next_token));
        if !is_suppressed {
            return Ok(Seq2SeqGreedyStepSelection {
                token_id: next_token,
                reached_eot: is_stop_token(next_token, stop_token_ids),
                probability: token_softmax_probability(&step_logits.logits, next_token as usize),
            });
        }
    }

    let mut logits = step_logits.logits;
    suppress_logits(&mut logits, &config.suppress_token_ids);
    if step_index == 0 {
        suppress_logits(&mut logits, &config.suppress_first_step_token_ids);
    }
    apply_phrase_bias_to_logits(&mut logits, generated_tokens, &config.phrase_biases);
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(Seq2SeqGreedyDecodeError::NonFiniteStepLogits { step_index });
    }
    on_topk(step_index, &logits);
    let next_token_idx =
        argmax_index(&logits).ok_or(Seq2SeqGreedyDecodeError::EmptyStepLogits { step_index })?;
    let next_token = u32::try_from(next_token_idx).map_err(|_| {
        Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id: u32::MAX,
            vocab_size: config.vocab_size,
        }
    })?;
    validate_selected_token(step_index, next_token, config.vocab_size)?;
    Ok(Seq2SeqGreedyStepSelection {
        token_id: next_token,
        reached_eot: is_stop_token(next_token, stop_token_ids),
        probability: token_softmax_probability(&logits, next_token_idx),
    })
}

/// Softmax probability of `token` over a host logit row (one max + one
/// sum-exp pass — negligible next to the matmul that produced the row).
/// Suppressed entries are `-inf`, so they contribute zero mass. Shared by the
/// seq2seq selection above and the transducer greedy loop (xasr).
pub(crate) fn token_softmax_probability(logits: &[f32], token: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let denominator: f32 = logits.iter().map(|value| (value - max).exp()).sum();
    if denominator <= 0.0 || !denominator.is_finite() {
        return 0.0;
    }
    ((logits[token] - max).exp() / denominator).clamp(0.0, 1.0)
}

fn validate_selected_token(
    step_index: usize,
    token_id: u32,
    vocab_size: usize,
) -> Result<(), Seq2SeqGreedyDecodeError> {
    if usize::try_from(token_id)
        .ok()
        .is_none_or(|token| token >= vocab_size)
    {
        return Err(Seq2SeqGreedyDecodeError::SelectedTokenOutOfVocab {
            step_index,
            token_id,
            vocab_size,
        });
    }
    Ok(())
}

fn suppress_logits(logits: &mut [f32], token_ids: &[u32]) {
    const SUPPRESSED_LOGIT: f32 = -1.0e30;
    for token_id in token_ids {
        let Some(index) = usize::try_from(*token_id).ok() else {
            continue;
        };
        if let Some(logit) = logits.get_mut(index) {
            *logit = SUPPRESSED_LOGIT;
        }
    }
}

fn argmax_index(values: &[f32]) -> Option<usize> {
    let mut best_index = None::<usize>;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = Some(idx);
        }
    }
    best_index
}

pub(crate) fn build_seq2seq_greedy_stop_token_ids(config: &Seq2SeqGreedyDecodeConfig) -> Vec<u32> {
    let mut stop = Vec::with_capacity(config.stop_token_ids.len().saturating_add(1));
    stop.push(config.eot_token_id);
    for token_id in &config.stop_token_ids {
        if *token_id != config.eot_token_id && !stop.contains(token_id) {
            stop.push(*token_id);
        }
    }
    stop
}

fn is_stop_token(token_id: u32, stop_token_ids: &[u32]) -> bool {
    stop_token_ids.contains(&token_id)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

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

    struct SyntheticTokenDecoder {
        table: BTreeMap<u32, &'static str>,
    }

    impl Seq2SeqGreedyTokenDecoder for SyntheticTokenDecoder {
        fn decode_text_token_ids(
            &self,
            token_ids: &[u32],
        ) -> Result<String, Seq2SeqGreedyDecodeError> {
            let mut out = String::new();
            for token_id in token_ids {
                let Some(piece) = self.table.get(token_id) else {
                    return Err(Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                        reason: format!("token {token_id} missing from synthetic decoder table"),
                    });
                };
                out.push_str(piece);
            }
            Ok(out)
        }
    }

    #[test]
    fn seq2seq_greedy_decode_turns_token_sequence_into_text() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2, 7],
            logits_calls: 0,
        };
        let token_decoder = SyntheticTokenDecoder {
            table: BTreeMap::from([(1, "he"), (2, "llo")]),
        };
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42, 43],
            eot_token_id: 7,
            stop_token_ids: Vec::new(),
            vocab_size: 16,
            max_generated_tokens: 8,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        let mut no_token_trace = |_: usize, _: u32, _: bool| {};
        let mut no_topk_trace = |_: usize, _: &[f32]| {};

        let output = run_seq2seq_greedy_decode_loop_v0(
            &config,
            &mut step_executor,
            &token_decoder,
            &mut no_token_trace,
            &mut no_topk_trace,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hello");
        assert_eq!(step_executor.logits_calls, 3);
    }

    #[test]
    fn seq2seq_step_selection_uses_unsuppressed_hint_without_topk() {
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            stop_token_ids: Vec::new(),
            vocab_size: 16,
            max_generated_tokens: 8,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        let stop = build_seq2seq_greedy_stop_token_ids(&config);
        let mut topk_calls = 0usize;
        let mut on_topk = |_: usize, _: &[f32]| {
            topk_calls += 1;
        };

        let selection = select_seq2seq_greedy_step_token(
            &config,
            &[],
            0,
            Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: vec![0.0; 16],
                greedy_token_hint: Some(3),
            },
            &stop,
            &mut on_topk,
        )
        .expect("hint should select");

        assert_eq!(
            selection,
            Seq2SeqGreedyStepSelection {
                token_id: 3,
                reached_eot: false,
                // Uniform logits over 16 tokens -> exactly 1/16.
                probability: 1.0 / 16.0,
            }
        );
        assert_eq!(topk_calls, 0);
    }

    #[test]
    fn seq2seq_step_selection_falls_back_when_hint_is_suppressed() {
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            stop_token_ids: Vec::new(),
            vocab_size: 16,
            max_generated_tokens: 8,
            suppress_first_step_token_ids: vec![3],
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        let stop = build_seq2seq_greedy_stop_token_ids(&config);
        let mut topk_calls = 0usize;
        let mut on_topk = |_: usize, logits: &[f32]| {
            topk_calls += 1;
            assert_eq!(logits[3], -1.0e30);
        };
        let mut logits = vec![-1000.0_f32; 16];
        logits[3] = 1000.0;
        logits[4] = 900.0;

        let selection = select_seq2seq_greedy_step_token(
            &config,
            &[],
            0,
            Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: Some(3),
            },
            &stop,
            &mut on_topk,
        )
        .expect("suppressed hint should fall back to logits");

        assert_eq!(
            selection,
            Seq2SeqGreedyStepSelection {
                token_id: 4,
                reached_eot: false,
                // The runner-up dominates after the hint is suppressed: every
                // other exp() term underflows to zero in f32.
                probability: 1.0,
            }
        );
        assert_eq!(topk_calls, 1);
    }

    #[test]
    fn seq2seq_truncation_error_keeps_probabilities_parallel_to_tokens() {
        // Callers degrade a no-EOT decode to the generated prefix; the error
        // must carry the per-token scores so that prefix keeps its confidence.
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 2],
            logits_calls: 0,
        };
        let token_decoder = SyntheticTokenDecoder {
            table: BTreeMap::from([(1, "he"), (2, "llo")]),
        };
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            stop_token_ids: Vec::new(),
            vocab_size: 16,
            max_generated_tokens: 2,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        let mut no_token_trace = |_: usize, _: u32, _: bool| {};
        let mut no_topk_trace = |_: usize, _: &[f32]| {};

        let error = run_seq2seq_greedy_decode_loop_v0(
            &config,
            &mut step_executor,
            &token_decoder,
            &mut no_token_trace,
            &mut no_topk_trace,
        )
        .unwrap_err();

        let Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
            generated_tokens,
            generated_probabilities,
            ..
        } = error
        else {
            panic!("expected truncation error, got {error:?}");
        };
        assert_eq!(generated_tokens, vec![1, 2]);
        assert_eq!(generated_probabilities.len(), generated_tokens.len());
        // One-hot synthetic logits: the winner's softmax saturates to 1.
        assert!(generated_probabilities.iter().all(|p| *p > 0.99));
    }

    #[test]
    fn seq2seq_stop_tokens_include_eot_once() {
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 7,
            stop_token_ids: vec![9, 7, 9],
            vocab_size: 16,
            max_generated_tokens: 8,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };

        assert_eq!(build_seq2seq_greedy_stop_token_ids(&config), vec![7, 9]);
    }

    #[test]
    fn seq2seq_greedy_decode_stops_on_additional_stop_token() {
        let mut step_executor = SyntheticStepExecutor {
            vocab_size: 16,
            sequence: vec![1, 9, 7],
            logits_calls: 0,
        };
        let token_decoder = SyntheticTokenDecoder {
            table: BTreeMap::from([(1, "he"), (2, "llo")]),
        };
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42, 43],
            eot_token_id: 7,
            stop_token_ids: vec![9],
            vocab_size: 16,
            max_generated_tokens: 8,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: Vec::new(),
        };
        let mut no_token_trace = |_: usize, _: u32, _: bool| {};
        let mut no_topk_trace = |_: usize, _: &[f32]| {};

        let output = run_seq2seq_greedy_decode_loop_v0(
            &config,
            &mut step_executor,
            &token_decoder,
            &mut no_token_trace,
            &mut no_topk_trace,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1]);
        assert_eq!(output.text, "he");
        assert_eq!(step_executor.logits_calls, 2);
    }

    #[test]
    fn seq2seq_phrase_bias_can_change_first_and_continuation_argmax() {
        struct FixedLogitsExecutor {
            rows: Vec<Vec<f32>>,
        }

        impl Seq2SeqGreedyDecodeStepExecutor for FixedLogitsExecutor {
            fn decode_step_logits(
                &mut self,
                input: Seq2SeqGreedyDecodeStepInput<'_>,
            ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
                Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits: self.rows[input.step_index].clone(),
                    greedy_token_hint: None,
                })
            }
        }

        let mut step_executor = FixedLogitsExecutor {
            rows: vec![
                vec![0.0, 0.9, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.9, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        };
        let token_decoder = SyntheticTokenDecoder {
            table: BTreeMap::from([(1, "hot"), (2, "word")]),
        };
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 4,
            stop_token_ids: Vec::new(),
            vocab_size: 5,
            max_generated_tokens: 4,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: vec![TokenPhraseBias::new(vec![vec![1, 2]], 0.2).unwrap()],
        };
        let mut no_token_trace = |_: usize, _: u32, _: bool| {};
        let mut no_topk_trace = |_: usize, _: &[f32]| {};

        let output = run_seq2seq_greedy_decode_loop_v0(
            &config,
            &mut step_executor,
            &token_decoder,
            &mut no_token_trace,
            &mut no_topk_trace,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1, 2]);
        assert_eq!(output.text, "hotword");
    }

    #[test]
    fn seq2seq_phrase_bias_uses_logits_instead_of_greedy_hint() {
        struct HintingExecutor;

        impl Seq2SeqGreedyDecodeStepExecutor for HintingExecutor {
            fn decode_step_logits(
                &mut self,
                input: Seq2SeqGreedyDecodeStepInput<'_>,
            ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
                let mut logits = vec![0.0, 0.9, 1.0, 0.0];
                if input.step_index == 1 {
                    logits = vec![0.0, 0.0, 0.0, 1.0];
                }
                Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits,
                    greedy_token_hint: Some(2),
                })
            }
        }

        let token_decoder = SyntheticTokenDecoder {
            table: BTreeMap::from([(1, "hot")]),
        };
        let config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: vec![42],
            eot_token_id: 3,
            stop_token_ids: Vec::new(),
            vocab_size: 4,
            max_generated_tokens: 3,
            suppress_first_step_token_ids: Vec::new(),
            suppress_token_ids: Vec::new(),
            phrase_biases: vec![TokenPhraseBias::new(vec![vec![1]], 0.2).unwrap()],
        };
        let mut no_token_trace = |_: usize, _: u32, _: bool| {};
        let mut no_topk_trace = |_: usize, _: &[f32]| {};
        let mut step_executor = HintingExecutor;

        let output = run_seq2seq_greedy_decode_loop_v0(
            &config,
            &mut step_executor,
            &token_decoder,
            &mut no_token_trace,
            &mut no_topk_trace,
        )
        .unwrap();

        assert_eq!(output.generated_tokens, vec![1]);
    }
}
