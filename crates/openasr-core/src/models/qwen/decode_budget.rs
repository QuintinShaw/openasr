//! Family-owned Qwen3-ASR generation-budget oracle.
//!
//! Both decoder-state planning and the real `DecodeConfig` call this module;
//! neither side carries a seconds-based approximation of the other.

use thiserror::Error;

use crate::models::decode_token_history::context_window_budget;

pub(crate) const QWEN3_DECODE_MIN_GENERATED_TOKENS: usize = 128;
pub(crate) const QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND: usize = 12;
pub(crate) const QWEN3_DECODE_TOKEN_BUDGET_MARGIN: usize = 32;

pub(crate) fn qwen3_desired_generated_tokens(
    sample_count: usize,
    sample_rate_hz: usize,
) -> Result<usize, Qwen3DecodeBudgetError> {
    if sample_rate_hz == 0 {
        return Err(Qwen3DecodeBudgetError::ZeroSampleRate);
    }
    let audio_rate_budget = sample_count
        .checked_mul(QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND)
        .and_then(|value| value.checked_add(sample_rate_hz - 1))
        .and_then(|value| value.checked_div(sample_rate_hz))
        .ok_or(Qwen3DecodeBudgetError::ArithmeticOverflow)?;
    audio_rate_budget
        .checked_add(QWEN3_DECODE_TOKEN_BUDGET_MARGIN)
        .map(|tokens| tokens.max(QWEN3_DECODE_MIN_GENERATED_TOKENS))
        .ok_or(Qwen3DecodeBudgetError::ArithmeticOverflow)
}

pub(crate) fn qwen3_generated_token_budget(
    sample_count: usize,
    sample_rate_hz: usize,
    prompt_tokens: usize,
    max_positions: usize,
) -> Result<usize, Qwen3DecodeBudgetError> {
    let context_remaining = context_window_budget(max_positions, prompt_tokens).ok_or(
        Qwen3DecodeBudgetError::ContextExhausted {
            prompt_tokens,
            max_positions,
        },
    )?;
    Ok(qwen3_desired_generated_tokens(sample_count, sample_rate_hz)?.min(context_remaining))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum Qwen3DecodeBudgetError {
    #[error("sample_rate_hz must be greater than zero")]
    ZeroSampleRate,
    #[error("audio duration token budget overflowed")]
    ArithmeticOverflow,
    #[error("prompt_tokens={prompt_tokens} exhausts llm_max_positions={max_positions}")]
    ContextExhausted {
        prompt_tokens: usize,
        max_positions: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_budget_scales_and_clamps_to_context() {
        assert_eq!(qwen3_desired_generated_tokens(16_000, 16_000), Ok(128));
        assert_eq!(qwen3_desired_generated_tokens(240_000, 16_000), Ok(212));
        assert_eq!(
            qwen3_generated_token_budget(240_000, 16_000, 240, 256),
            Ok(16)
        );
    }

    #[test]
    fn invalid_shape_fails_closed() {
        assert_eq!(
            qwen3_generated_token_budget(16_000, 16_000, 256, 256),
            Err(Qwen3DecodeBudgetError::ContextExhausted {
                prompt_tokens: 256,
                max_positions: 256,
            })
        );
        assert_eq!(
            qwen3_desired_generated_tokens(16_000, 0),
            Err(Qwen3DecodeBudgetError::ZeroSampleRate)
        );
    }
}
