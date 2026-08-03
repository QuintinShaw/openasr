//! FireRed-AED's decoder-length contract.
//!
//! Upstream's default `decode_max_len=0` resolves to the encoder output length
//! (`Ti`).  The position table remains a hard mathematical ceiling; it is not
//! the ordinary decode budget.  Keeping this integer oracle shared by the
//! topology and runtime makes the resident self-KV span exactly follow the
//! largest legal encoder invocation without duplicating the rule.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FireRedAedDecodeBudgetError {
    #[error("firered-aed encoder output must contain at least one frame")]
    EmptyEncoderOutput,
    #[error("firered-aed decoder position cap must leave room for the SOS prompt")]
    DecoderPositionCapExhausted,
    #[error("firered-aed causal decode schedule is invalid: {reason}")]
    InvalidCausalSchedule { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FireRedAedDecodeBudget {
    pub prompt_positions: usize,
    pub max_generated_tokens: usize,
    pub self_kv_positions: usize,
}

pub(crate) fn firered_aed_decode_budget(
    encoder_frame_count: usize,
    decoder_position_cap: usize,
) -> Result<FireRedAedDecodeBudget, FireRedAedDecodeBudgetError> {
    if encoder_frame_count == 0 {
        return Err(FireRedAedDecodeBudgetError::EmptyEncoderOutput);
    }
    const PROMPT_POSITIONS: usize = 1;
    let context_budget = decoder_position_cap
        .checked_sub(PROMPT_POSITIONS)
        .filter(|&positions| positions > 0)
        .ok_or(FireRedAedDecodeBudgetError::DecoderPositionCapExhausted)?;
    let max_generated_tokens = encoder_frame_count.min(context_budget);
    let self_kv_positions = crate::capacity::topology::causal_prefix_positions_with_context_cap(
        "firered-aed.decoder.self_kv",
        PROMPT_POSITIONS,
        max_generated_tokens,
        decoder_position_cap,
    )
    .map_err(|error| FireRedAedDecodeBudgetError::InvalidCausalSchedule {
        reason: error.to_string(),
    })?;
    Ok(FireRedAedDecodeBudget {
        prompt_positions: PROMPT_POSITIONS,
        max_generated_tokens,
        self_kv_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_default_tracks_encoder_length_not_position_table_length() {
        let budget = firered_aed_decode_budget(750, 5_000).unwrap();
        assert_eq!(budget.prompt_positions, 1);
        assert_eq!(budget.max_generated_tokens, 750);
        assert_eq!(budget.self_kv_positions, 750);
    }

    #[test]
    fn position_table_remains_a_strict_hard_cap() {
        let budget = firered_aed_decode_budget(9_000, 5_000).unwrap();
        assert_eq!(budget.max_generated_tokens, 4_999);
        assert_eq!(budget.self_kv_positions, 4_999);
    }
}
