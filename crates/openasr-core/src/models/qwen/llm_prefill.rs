use thiserror::Error;

use super::prompt_embedding::Qwen3AsrPromptEmbeddings;

const MASK_ALLOW: f32 = 0.0;
const MASK_BLOCK: f32 = -1.0e30;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrLlmPrefillInput {
    pub token_count: usize,
    pub hidden_size: usize,
    // Layout: token-major row-contiguous ([token][hidden]) f32.
    pub token_major_embeddings: Vec<f32>,
    // Layout: token-major ([token]) i32.
    pub position_ids: Vec<i32>,
    // Layout: row-major ([query_token][key_token]) f32.
    pub causal_mask: Vec<f32>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum Qwen3AsrLlmPrefillInputError {
    #[error(
        "qwen3-asr llm prefill embeddings shape is invalid: token_count={token_count} hidden_size={hidden_size} values_len={values_len}"
    )]
    InvalidEmbeddingShape {
        token_count: usize,
        hidden_size: usize,
        values_len: usize,
    },
    #[error("qwen3-asr llm prefill token_count={token_count} does not fit i32 position ids")]
    TokenCountOutOfI32Range { token_count: usize },
    #[error("qwen3-asr llm prefill token_count={token_count} overflows causal mask allocation")]
    CausalMaskAllocationOverflow { token_count: usize },
    #[error("qwen3-asr llm prefill embeddings contain non-finite values")]
    NonFiniteEmbeddings,
}

pub(crate) fn build_qwen3_llm_prefill_input(
    prompt_embeddings: &Qwen3AsrPromptEmbeddings,
) -> Result<Qwen3AsrLlmPrefillInput, Qwen3AsrLlmPrefillInputError> {
    let token_count = prompt_embeddings.token_count;
    let hidden_size = prompt_embeddings.hidden_size;
    let expected_embeddings = token_count.checked_mul(hidden_size).ok_or(
        Qwen3AsrLlmPrefillInputError::InvalidEmbeddingShape {
            token_count,
            hidden_size,
            values_len: prompt_embeddings.token_major_values.len(),
        },
    )?;
    if prompt_embeddings.token_major_values.len() != expected_embeddings {
        return Err(Qwen3AsrLlmPrefillInputError::InvalidEmbeddingShape {
            token_count,
            hidden_size,
            values_len: prompt_embeddings.token_major_values.len(),
        });
    }
    if prompt_embeddings
        .token_major_values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(Qwen3AsrLlmPrefillInputError::NonFiniteEmbeddings);
    }

    let mut position_ids = Vec::with_capacity(token_count);
    for idx in 0..token_count {
        let id = i32::try_from(idx)
            .map_err(|_| Qwen3AsrLlmPrefillInputError::TokenCountOutOfI32Range { token_count })?;
        position_ids.push(id);
    }

    let mask_len = token_count
        .checked_mul(token_count)
        .ok_or(Qwen3AsrLlmPrefillInputError::CausalMaskAllocationOverflow { token_count })?;
    let mut causal_mask = vec![MASK_BLOCK; mask_len];
    for query_idx in 0..token_count {
        for key_idx in 0..=query_idx {
            causal_mask[query_idx * token_count + key_idx] = MASK_ALLOW;
        }
    }

    Ok(Qwen3AsrLlmPrefillInput {
        token_count,
        hidden_size,
        token_major_embeddings: prompt_embeddings.token_major_values.clone(),
        position_ids,
        causal_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_prefill_builds_monotonic_positions_and_causal_mask() {
        let prompt = Qwen3AsrPromptEmbeddings {
            hidden_size: 2,
            token_count: 3,
            token_major_values: vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
            ],
        };
        let input = build_qwen3_llm_prefill_input(&prompt).expect("prefill input");
        assert_eq!(input.position_ids, vec![0, 1, 2]);
        assert_eq!(
            input.causal_mask,
            vec![
                0.0, -1.0e30, -1.0e30, //
                0.0, 0.0, -1.0e30, //
                0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn llm_prefill_rejects_non_finite_embeddings() {
        let prompt = Qwen3AsrPromptEmbeddings {
            hidden_size: 1,
            token_count: 1,
            token_major_values: vec![f32::NAN],
        };
        let error = build_qwen3_llm_prefill_input(&prompt).expect_err("must fail");
        assert!(matches!(
            error,
            Qwen3AsrLlmPrefillInputError::NonFiniteEmbeddings
        ));
    }
}
