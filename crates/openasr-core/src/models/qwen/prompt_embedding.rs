use thiserror::Error;

use super::decode_prompt::Qwen3AsrDecodePrompt;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrPromptEmbeddings {
    pub hidden_size: usize,
    pub token_count: usize,
    // Layout: token-major row-contiguous ([token][hidden]) f32.
    pub token_major_values: Vec<f32>,
}

/// Canonical prompt source retained until the first decoder step decides
/// whether token lookup can stay on the selected device. Audio rows are
/// encoder-produced activations; `audio_positions` identifies the token rows
/// they replace and supports both contiguous Qwen-style spans and MOSS's
/// sparse marker layout.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrPromptTokenInput {
    pub token_ids: Vec<u32>,
    pub audio_rows: Vec<f32>,
    pub audio_positions: Vec<usize>,
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
    #[error(
        "qwen3-asr prompt embedding audio positions are invalid: positions_len={positions_len} audio_row_count={audio_row_count} token_count={token_count}"
    )]
    InvalidAudioPositions {
        positions_len: usize,
        audio_row_count: usize,
        token_count: usize,
    },
    #[error("qwen3-asr prompt embedding values contain non-finite elements")]
    NonFiniteValues,
}

/// Sparse-position counterpart to
/// [`build_qwen3_prompt_embeddings_with_audio_splice`]. This is the shared
/// CPU fallback for device prompt prefill: GPU-class direct graphs consume the
/// same [`Qwen3AsrPromptTokenInput`] through `get_rows` + `set_rows`, while a
/// CPU/scheduler path gathers the canonical token table once and calls here.
pub(crate) fn build_qwen3_prompt_embeddings_with_audio_positions(
    token_count: usize,
    audio_positions: &[usize],
    hidden_size: usize,
    mut token_rows: Vec<f32>,
    audio_rows: &[f32],
) -> Result<Qwen3AsrPromptEmbeddings, Qwen3AsrPromptEmbeddingError> {
    let expected_token_values = token_count.checked_mul(hidden_size).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
            token_count,
            hidden_size,
            values_len: token_rows.len(),
        },
    )?;
    if hidden_size == 0 || token_rows.len() != expected_token_values {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidTokenRowsShape {
            token_count,
            hidden_size,
            values_len: token_rows.len(),
        });
    }
    if token_rows.iter().any(|value| !value.is_finite())
        || audio_rows.iter().any(|value| !value.is_finite())
    {
        return Err(Qwen3AsrPromptEmbeddingError::NonFiniteValues);
    }
    if !audio_rows.len().is_multiple_of(hidden_size) {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape {
            audio_frame_count: audio_positions.len(),
            hidden_size,
            values_len: audio_rows.len(),
        });
    }
    let audio_row_count = audio_rows.len() / hidden_size;
    let positions_valid = audio_positions.len() == audio_row_count
        && audio_positions
            .iter()
            .copied()
            .enumerate()
            .all(|(index, position)| {
                position < token_count
                    && index
                        .checked_sub(1)
                        .is_none_or(|previous| audio_positions[previous] < position)
            });
    if !positions_valid {
        return Err(Qwen3AsrPromptEmbeddingError::InvalidAudioPositions {
            positions_len: audio_positions.len(),
            audio_row_count,
            token_count,
        });
    }
    for (audio_index, &token_position) in audio_positions.iter().enumerate() {
        let source_start = audio_index.checked_mul(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidAudioPositions {
                positions_len: audio_positions.len(),
                audio_row_count,
                token_count,
            },
        )?;
        let target_start = token_position.checked_mul(hidden_size).ok_or(
            Qwen3AsrPromptEmbeddingError::InvalidAudioPositions {
                positions_len: audio_positions.len(),
                audio_row_count,
                token_count,
            },
        )?;
        token_rows[target_start..target_start + hidden_size]
            .copy_from_slice(&audio_rows[source_start..source_start + hidden_size]);
    }
    Ok(Qwen3AsrPromptEmbeddings {
        hidden_size,
        token_count,
        token_major_values: token_rows,
    })
}

/// Splice the audio-encoder rows into the token-embedding buffer at the decode
/// prompt's audio pad span, producing the combined prompt embeddings.
///
/// `token_rows` is consumed by value and spliced IN PLACE: every family
/// executor (qwen / firered-llm / mimo / forced-aligner) gathers an owned
/// token-row buffer it never touches again, so the historical `to_vec()`
/// copy was a pure per-utterance `token_count x hidden` clone on the
/// encoder-to-prefill critical path (and the result was cloned once more by
/// `build_qwen3_llm_prefill_input`). Both buffers are row-major
/// `[token|frame][hidden]`, so the audio span is contiguous on each side and
/// the splice is a single span copy. All fail-closed validation (shape,
/// checked span arithmetic, non-finite elements) runs unchanged before the
/// copy.
pub(crate) fn build_qwen3_prompt_embeddings_with_audio_splice(
    decode_prompt: &Qwen3AsrDecodePrompt,
    hidden_size: usize,
    mut token_rows: Vec<f32>,
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

    // The audio span is contiguous on both sides (row-major [token][hidden]),
    // and the checks above pin it inside both buffers:
    // `expected_audio_values == audio_rows.len()` and `pad_end <= token_count`
    // (so the dst span ends within `token_rows`). The checked offsets keep the
    // overflow guard the per-frame loop used to carry.
    let dst_start = pad_start.checked_mul(hidden_size).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidAudioPadSpan {
            audio_pad_start_index: pad_start,
            audio_pad_count: audio_frame_count,
            token_count,
        },
    )?;
    let dst_end = dst_start.checked_add(expected_audio_values).ok_or(
        Qwen3AsrPromptEmbeddingError::InvalidAudioPadSpan {
            audio_pad_start_index: pad_start,
            audio_pad_count: audio_frame_count,
            token_count,
        },
    )?;
    token_rows[dst_start..dst_end].copy_from_slice(&audio_rows[..expected_audio_values]);

    Ok(Qwen3AsrPromptEmbeddings {
        hidden_size,
        token_count,
        token_major_values: token_rows,
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
            build_qwen3_prompt_embeddings_with_audio_splice(&prompt, 2, token_rows, &audio_rows)
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
            build_qwen3_prompt_embeddings_with_audio_splice(&prompt, 2, vec![0.0; 8], &[0.0; 3])
                .expect_err("audio shape mismatch must fail");
        assert!(matches!(
            error,
            Qwen3AsrPromptEmbeddingError::InvalidAudioRowsShape { .. }
        ));
    }

    #[test]
    fn sparse_prompt_embedding_splice_replaces_only_declared_rows() {
        let result = build_qwen3_prompt_embeddings_with_audio_positions(
            5,
            &[1, 4],
            2,
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            &[100.0, 101.0, 400.0, 401.0],
        )
        .expect("sparse splice");
        assert_eq!(
            result.token_major_values,
            vec![0.0, 1.0, 100.0, 101.0, 20.0, 21.0, 30.0, 31.0, 400.0, 401.0]
        );
    }

    #[test]
    fn sparse_prompt_embedding_splice_rejects_duplicate_positions() {
        let error = build_qwen3_prompt_embeddings_with_audio_positions(
            3,
            &[1, 1],
            2,
            vec![0.0; 6],
            &[1.0; 4],
        )
        .expect_err("duplicate positions must fail");
        assert!(matches!(
            error,
            Qwen3AsrPromptEmbeddingError::InvalidAudioPositions { .. }
        ));
    }
}
