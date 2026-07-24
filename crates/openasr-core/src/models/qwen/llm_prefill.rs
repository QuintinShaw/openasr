use thiserror::Error;

use super::prompt_embedding::Qwen3AsrPromptEmbeddings;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrLlmPrefillInput {
    pub token_count: usize,
    pub hidden_size: usize,
    // Layout: token-major row-contiguous ([token][hidden]) f32.
    pub token_major_embeddings: Vec<f32>,
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

    Ok(Qwen3AsrLlmPrefillInput {
        token_count,
        hidden_size,
        token_major_embeddings: prompt_embeddings.token_major_values.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_prefill_passes_through_token_major_embeddings() {
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
        assert_eq!(input.token_count, 3);
        assert_eq!(input.hidden_size, 2);
        assert_eq!(
            input.token_major_embeddings,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
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
