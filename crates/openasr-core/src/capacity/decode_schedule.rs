//! Decoder-step schedule arithmetic shared by topology and execution.
//!
//! This module counts cache writes, not output tokens. Keeping that distinction
//! explicit prevents every family from independently reserving one dead KV row
//! for the final sampled token, which is returned but never fed back.

use thiserror::Error;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeScheduleError {
    #[error("greedy decode generation budget must contain at least one step")]
    EmptyGenerationBudget,
    #[error("decoder cache position count overflowed")]
    PositionOverflow,
}

/// Incremental KV writes after a prompt prefill for the shared greedy driver.
pub(crate) fn greedy_incremental_cache_writes(
    max_generated_steps: usize,
) -> Result<usize, DecodeScheduleError> {
    max_generated_steps
        .checked_sub(1)
        .ok_or(DecodeScheduleError::EmptyGenerationBudget)
}

/// Minimum addressable self-KV rows for one legal greedy invocation.
pub(crate) fn greedy_self_kv_positions(
    prompt_positions: usize,
    max_generated_steps: usize,
) -> Result<usize, DecodeScheduleError> {
    prompt_positions
        .checked_add(greedy_incremental_cache_writes(max_generated_steps)?)
        .ok_or(DecodeScheduleError::PositionOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_sampled_token_does_not_consume_a_cache_row() {
        assert_eq!(greedy_incremental_cache_writes(818), Ok(817));
        assert_eq!(greedy_self_kv_positions(472, 818), Ok(1_289));
        assert_eq!(greedy_self_kv_positions(1, 1), Ok(1));
        assert_eq!(
            greedy_self_kv_positions(472, 0),
            Err(DecodeScheduleError::EmptyGenerationBudget)
        );
        assert_eq!(
            greedy_self_kv_positions(usize::MAX, 2),
            Err(DecodeScheduleError::PositionOverflow)
        );
    }

    #[test]
    fn computed_span_is_both_sufficient_and_strictly_minimal() {
        // Exhaust the small Cartesian product and compare the formula with the
        // actual write schedule. Prefill writes P rows; only the first G-1
        // sampled tokens are fed back and write another row. Therefore the
        // final written address is K-1, proving K sufficient and K-1 unsafe.
        for prompt_positions in 1..=32 {
            for generated_steps in 1..=32 {
                let planned = greedy_self_kv_positions(prompt_positions, generated_steps).unwrap();
                let written_positions = (0..prompt_positions).chain(
                    (0..greedy_incremental_cache_writes(generated_steps).unwrap())
                        .map(|offset| prompt_positions + offset),
                );
                let last_written_position = written_positions.max().unwrap();
                let required = last_written_position + 1;
                assert_eq!(planned, required);
                assert_eq!(last_written_position, planned - 1);
            }
        }
    }
}
