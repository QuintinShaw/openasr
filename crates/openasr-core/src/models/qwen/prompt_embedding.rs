use thiserror::Error;

use super::decode_prompt::Qwen3AsrDecodePrompt;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrPromptEmbeddings {
    pub hidden_size: usize,
    pub token_count: usize,
    // Layout: token-major row-contiguous ([token][hidden]) f32.
    pub token_major_values: Vec<f32>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum Qwen3AsrPromptEmbeddingError {
    #[error(
        "qwen3-asr prompt embedding token rows shape is invalid: token_count={token_count} hidden_size={hidden_size} values_len={values_len}"
    )]
    InvalidTokenRowsShape {
        token_count: usize,
        hidden_size: usize,
        values_len: usize,
    },
    #[error(
        "qwen3-asr prompt embedding audio rows shape is invalid: audio_frame_count={audio_frame_count} hidden_size={hidden_size} values_len={values_len}"
    )]
    InvalidAudioRowsShape {
        audio_frame_count: usize,
        hidden_size: usize,
        values_len: usize,
    },
    #[error(
        "qwen3-asr prompt embedding audio splice span is invalid: pad_start={audio_pad_start_index} pad_count={audio_pad_count} token_count={token_count}"
    )]
    InvalidAudioPadSpan {
        audio_pad_start_index: usize,
        audio_pad_count: usize,
        token_count: usize,
    },
    #[error("qwen3-asr prompt embedding values contain non-finite elements")]
    NonFiniteValues,
}

pub(crate) fn build_qwen3_prompt_embeddings_with_audio_splice(
    decode_prompt: &Qwen3AsrDecodePrompt,
    hidden_size: usize,
    token_rows: &[f32],
    audio_rows: &[f32],
) -> Result<Qwen3AsrPromptEmbeddings, Qwen3AsrPromptEmbeddingError> {
    if hidden_size == 0 {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
            token_count: decode_prompt.token_ids.len(),
            hidden_size,
            values_len: token_rows.len(),
        });
    }

    let token_count = decode_prompt.token_ids.len();
    let expected_token_values = token_count.checked_mul(hidden_size).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
            token_count,
            hidden_size,
            values_len: token_rows.len(),
        },
    )?;
    if token_rows.len() != expected_token_values {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
            token_count,
            hidden_size,
            values_len: token_rows.len(),
        });
    }
    if token_rows.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrPromptEmbeddingError::NonFiniteValues);
    }

    let audio_frame_count = decode_prompt.audio_pad_count;
    let expected_audio_values = audio_frame_count.checked_mul(hidden_size).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape {
            audio_frame_count,
            hidden_size,
            values_len: audio_rows.len(),
        },
    )?;
    if audio_rows.len() != expected_audio_values {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape {
            audio_frame_count,
            hidden_size,
            values_len: audio_rows.len(),
        });
    }
    if audio_rows.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrPromptEmbeddingError::NonFiniteValues);
    }

    let pad_start = decode_prompt.audio_pad_start_index;
    let pad_end = pad_start.checked_add(audio_frame_count).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidAudioPadSpan {
            audio_pad_start_index: pad_start,
            audio_pad_count: audio_frame_count,
            token_count,
        },
    )?;
    if pad_end > token_count {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidAudioPadSpan {
            audio_pad_start_index: pad_start,
            audio_pad_count: audio_frame_count,
            token_count,
        });
    }

    let mut combined = token_rows.to_vec();
    for frame_idx in 0..audio_frame_count {
        let src_start = frame_idx.checked_mul(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape {
                audio_frame_count,
                hidden_size,
                values_len: audio_rows.len(),
            },
        )?;
        let src_end = src_start.checked_add(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape {
                audio_frame_count,
                hidden_size,
                values_len: audio_rows.len(),
            },
        )?;
        let dst_token_idx = pad_start + frame_idx;
        let dst_start = dst_token_idx.checked_mul(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
                token_count,
                hidden_size,
                values_len: combined.len(),
            },
        )?;
        let dst_end = dst_start.checked_add(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
                token_count,
                hidden_size,
                values_len: combined.len(),
            },
        )?;
        combined[dst_start..dst_end].copy_from_slice(&audio_rows[src_start..src_end]);
    }

    Ok(Qwen3AsrPromptEmbeddings {
        hidden_size,
        token_count,
        token_major_values: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen::decode_prompt::Qwen3AsrDecodePrompt;

    #[test]
    fn prompt_embedding_splice_replaces_audio_pad_rows_only() {
        let prompt = Qwen3AsrDecodePrompt {
            token_ids: vec![1, 2, 2, 3],
            audio_pad_start_index: 1,
            audio_pad_count: 2,
        };
        let token_rows = vec![
            10.0, 11.0, //
            20.0, 21.0, //
            30.0, 31.0, //
            40.0, 41.0,
        ];
        let audio_rows = vec![
            100.0, 101.0, //
            200.0, 201.0,
        ];
        let spliced =
            build_qwen3_prompt_embeddings_with_audio_splice(&prompt, 2, &token_rows, &audio_rows)
                .expect("splice");
        assert_eq!(
            spliced.token_major_values,
            vec![
                10.0, 11.0, //
                100.0, 101.0, //
                200.0, 201.0, //
                40.0, 41.0
            ]
        );
    }

    #[test]
    fn prompt_embedding_splice_rejects_audio_shape_mismatch() {
        let prompt = Qwen3AsrDecodePrompt {
            token_ids: vec![1, 2, 2, 3],
            audio_pad_start_index: 1,
            audio_pad_count: 2,
        };
        let error =
            build_qwen3_prompt_embeddings_with_audio_splice(&prompt, 2, &[0.0; 8], &[0.0; 3])
                .expect_err("audio shape mismatch must fail");
        assert!(matches!(
            error,
            Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape { .. }
        ));
    }
}
