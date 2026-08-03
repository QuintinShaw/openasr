//! Cohere Transcribe's family-owned integer decode budget.
//!
//! The audio-derived runaway guard is part of decode semantics, not a memory
//! heuristic.  Both the capacity topology and `Seq2SeqGreedyDecodeConfig`
//! consume this oracle so resident self-KV and the loop's legal token count
//! cannot drift.

use thiserror::Error;

pub(crate) const COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV: &str =
    "OPENASR_COHERE_MAX_GENERATED_TOKENS_OVERRIDE";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum CohereDecodeBudgetError {
    #[error("cohere decode prompt must contain at least one token")]
    EmptyPrompt,
    #[error(
        "cohere decode prompt length {prompt_positions} exhausts decoder context {decoder_position_cap}"
    )]
    PromptExhaustsContext {
        prompt_positions: usize,
        decoder_position_cap: usize,
    },
    #[error("{COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV} must be a positive integer, got '{value}'")]
    InvalidOverride { value: String },
    #[error("cohere causal decode schedule is invalid: {reason}")]
    InvalidCausalSchedule { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CohereDecodeBudget {
    pub prompt_positions: usize,
    pub max_generated_tokens: usize,
    pub self_kv_positions: usize,
}

pub(crate) fn cohere_decode_budget(
    prompt_positions: usize,
    encoder_frame_count: usize,
    decoder_position_cap: usize,
) -> Result<CohereDecodeBudget, CohereDecodeBudgetError> {
    if prompt_positions == 0 {
        return Err(CohereDecodeBudgetError::EmptyPrompt);
    }
    let context_budget = decoder_position_cap
        .checked_sub(prompt_positions)
        .filter(|&positions| positions > 0)
        .ok_or(CohereDecodeBudgetError::PromptExhaustsContext {
            prompt_positions,
            decoder_position_cap,
        })?;
    // Existing family semantics: allow four output tokens per encoder frame,
    // with a short-audio floor and a runaway ceiling.
    let audio_budget = encoder_frame_count.saturating_mul(4).clamp(64, 512);
    let mut max_generated_tokens = context_budget.min(audio_budget);
    if let Some(raw) = std::env::var_os(COHERE_MAX_GENERATED_TOKENS_OVERRIDE_ENV) {
        let value = raw.to_string_lossy().trim().to_string();
        if !value.is_empty() {
            let override_value = value
                .parse::<usize>()
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| CohereDecodeBudgetError::InvalidOverride {
                    value: value.clone(),
                })?;
            max_generated_tokens = max_generated_tokens.min(override_value);
        }
    }
    let self_kv_positions = crate::capacity::topology::causal_prefix_positions_with_context_cap(
        "cohere.decoder.self_kv",
        prompt_positions,
        max_generated_tokens,
        decoder_position_cap,
    )
    .map_err(|error| CohereDecodeBudgetError::InvalidCausalSchedule {
        reason: error.to_string(),
    })?;
    Ok(CohereDecodeBudget {
        prompt_positions,
        max_generated_tokens,
        self_kv_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_budget_is_distinct_from_context_ceiling() {
        let budget = cohere_decode_budget(9, 100, 1_024).unwrap();
        assert_eq!(budget.max_generated_tokens, 400);
        assert_eq!(budget.self_kv_positions, 408);
    }

    #[test]
    fn family_floor_and_ceiling_are_integer_exact() {
        assert_eq!(
            cohere_decode_budget(9, 1, 1_024)
                .unwrap()
                .max_generated_tokens,
            64
        );
        assert_eq!(
            cohere_decode_budget(9, 10_000, 1_024)
                .unwrap()
                .max_generated_tokens,
            512
        );
    }
}
