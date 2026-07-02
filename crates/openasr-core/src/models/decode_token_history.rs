/// Returns `max_positions - prompt_len`, or `None` if the prompt already fills
/// or exceeds the context window. Shared by all model families to compute the
/// remaining generation budget before capping by any audio-based heuristic.
pub(crate) fn context_window_budget(max_positions: usize, prompt_len: usize) -> Option<usize> {
    max_positions.checked_sub(prompt_len).filter(|v| *v > 0)
}

pub(crate) fn trim_prompt_token_tail(
    mut prompt_token_ids: Vec<u32>,
    max_prompt_tokens: usize,
    longform_enabled: bool,
    longform_prompt_token_tail_limit: usize,
) -> Vec<u32> {
    let prompt_tail_budget = if longform_enabled {
        max_prompt_tokens.min(longform_prompt_token_tail_limit)
    } else {
        max_prompt_tokens
    };
    if prompt_token_ids.len() > prompt_tail_budget {
        let split_at = prompt_token_ids.len().saturating_sub(prompt_tail_budget);
        prompt_token_ids = prompt_token_ids.split_off(split_at);
    }
    prompt_token_ids
}

pub(crate) fn build_longform_token_history_carry(
    longform_enabled: bool,
    mut prompt_token_ids: Vec<u32>,
    generated_tokens: &[u32],
    longform_prompt_token_tail_limit: usize,
) -> Option<Vec<u32>> {
    if !longform_enabled {
        return None;
    }
    prompt_token_ids.extend_from_slice(generated_tokens);
    if prompt_token_ids.is_empty() {
        return None;
    }
    if prompt_token_ids.len() > longform_prompt_token_tail_limit {
        let split_at = prompt_token_ids
            .len()
            .saturating_sub(longform_prompt_token_tail_limit);
        prompt_token_ids = prompt_token_ids.split_off(split_at);
    }
    Some(prompt_token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_window_budget_returns_remaining_space() {
        assert_eq!(context_window_budget(100, 40), Some(60));
    }

    #[test]
    fn context_window_budget_returns_none_when_prompt_fills_window() {
        assert_eq!(context_window_budget(100, 100), None);
        assert_eq!(context_window_budget(100, 120), None);
    }

    #[test]
    fn trim_prompt_token_tail_keeps_full_prompt_when_within_budget() {
        let tokens = trim_prompt_token_tail(vec![1, 2, 3], 4, false, 2);

        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn trim_prompt_token_tail_trims_to_longform_tail_limit() {
        let tokens = trim_prompt_token_tail(vec![1, 2, 3, 4, 5], 8, true, 3);

        assert_eq!(tokens, vec![3, 4, 5]);
    }

    #[test]
    fn build_longform_token_history_carry_returns_none_when_not_longform() {
        let carry = build_longform_token_history_carry(false, vec![1, 2], &[3, 4], 4);

        assert!(carry.is_none());
    }

    #[test]
    fn build_longform_token_history_carry_trims_to_tail_limit() {
        let carry =
            build_longform_token_history_carry(true, vec![1, 2, 3], &[4, 5, 6], 4).expect("carry");

        assert_eq!(carry, vec![3, 4, 5, 6]);
    }
}
