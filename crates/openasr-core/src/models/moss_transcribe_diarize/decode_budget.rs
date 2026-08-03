//! Family-owned MOSS-TD generation-budget oracle shared by planning and
//! execution.

use thiserror::Error;

/// Absolute fail-closed backstop for a non-terminating decode.
pub(crate) const MOSS_TD_MAX_GENERATED_TOKENS: usize = 4_096;
/// Conservative output allowance for dense timestamped/overlapping speech.
/// 23 tokens/s is the rate at which a 180-second invocation reaches the
/// checkpoint's 4096-token runaway backstop. Shorter inputs remain
/// proportional, so they do not reserve that entire ceiling.
pub(crate) const MOSS_TD_GENERATED_TOKENS_PER_AUDIO_SECOND: usize = 23;
pub(crate) const MOSS_TD_MIN_GENERATED_TOKENS: usize = 128;
pub(crate) const MOSS_TD_GENERATED_TOKEN_BUDGET_MARGIN: usize = 128;

pub(crate) fn moss_td_desired_generated_tokens(
    sample_count: usize,
    sample_rate_hz: usize,
) -> Result<usize, MossTdDecodeBudgetError> {
    if sample_rate_hz == 0 {
        return Err(MossTdDecodeBudgetError::ZeroSampleRate);
    }
    let audio_tokens = sample_count
        .checked_mul(MOSS_TD_GENERATED_TOKENS_PER_AUDIO_SECOND)
        .and_then(|value| value.checked_add(sample_rate_hz - 1))
        .and_then(|value| value.checked_div(sample_rate_hz))
        .ok_or(MossTdDecodeBudgetError::ArithmeticOverflow)?;
    audio_tokens
        .checked_add(MOSS_TD_GENERATED_TOKEN_BUDGET_MARGIN)
        .map(|value| value.clamp(MOSS_TD_MIN_GENERATED_TOKENS, MOSS_TD_MAX_GENERATED_TOKENS))
        .ok_or(MossTdDecodeBudgetError::ArithmeticOverflow)
}

pub(crate) fn moss_td_generated_token_budget(
    sample_count: usize,
    sample_rate_hz: usize,
    prompt_tokens: usize,
    kv_capacity: usize,
) -> Result<usize, MossTdDecodeBudgetError> {
    let remaining_context = kv_capacity.checked_sub(prompt_tokens).ok_or(
        MossTdDecodeBudgetError::PromptExhaustsContext {
            prompt_tokens,
            kv_capacity,
        },
    )?;
    if remaining_context < MOSS_TD_MIN_GENERATED_TOKENS {
        return Err(MossTdDecodeBudgetError::MinimumGenerationDoesNotFit {
            remaining_context,
            minimum_generated_tokens: MOSS_TD_MIN_GENERATED_TOKENS,
        });
    }
    Ok(moss_td_desired_generated_tokens(sample_count, sample_rate_hz)?.min(remaining_context))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum MossTdDecodeBudgetError {
    #[error("sample_rate_hz must be greater than zero")]
    ZeroSampleRate,
    #[error("audio-duration token budget overflowed")]
    ArithmeticOverflow,
    #[error("prompt_tokens={prompt_tokens} exhausts KV capacity {kv_capacity}")]
    PromptExhaustsContext {
        prompt_tokens: usize,
        kv_capacity: usize,
    },
    #[error(
        "remaining KV context {remaining_context} cannot fit the minimum generation budget {minimum_generated_tokens}"
    )]
    MinimumGenerationDoesNotFit {
        remaining_context: usize,
        minimum_generated_tokens: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_windows_have_exact_integer_budgets() {
        assert_eq!(
            moss_td_desired_generated_tokens(30 * 16_000, 16_000),
            Ok(818)
        );
        assert_eq!(
            moss_td_desired_generated_tokens(60 * 16_000, 16_000),
            Ok(1_508)
        );
        assert_eq!(
            moss_td_desired_generated_tokens(180 * 16_000, 16_000),
            Ok(4_096)
        );
    }

    #[test]
    fn insufficient_context_fails_before_decode() {
        assert!(matches!(
            moss_td_generated_token_budget(16_000, 16_000, 900, 1_000),
            Err(MossTdDecodeBudgetError::MinimumGenerationDoesNotFit { .. })
        ));
    }
}
