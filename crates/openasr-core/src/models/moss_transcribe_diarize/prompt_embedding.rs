//! Splice adaptor output rows into the token embedding sequence at the exact
//! (possibly non-contiguous) `<|audio_pad|>` positions
//! `decode_prompt::build_moss_td_decode_prompt` records. Cannot reuse
//! `qwen::build_qwen3_prompt_embeddings_with_audio_splice` (that function
//! assumes one contiguous pad run; MOSS's audio span is interrupted by
//! digit-marker tokens -- see `decode_prompt`'s module doc), so this is the
//! sparse-position generalization of the same "replace embedding rows at
//! placeholder positions" idea upstream implements via `masked_scatter`
//! (`modeling_moss_transcribe_diarize.py`'s `inject_audio_features`).

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MossTdPromptEmbeddings {
    pub hidden_size: usize,
    pub token_count: usize,
    /// Token-major, row-contiguous ([token][hidden]) f32.
    pub token_major_values: Vec<f32>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum MossTdPromptEmbeddingError {
    #[error(
        "moss-transcribe-diarize prompt embedding token rows shape is invalid: token_count={token_count} hidden_size={hidden_size} values_len={values_len}"
    )]
    InvalidTokenRowsShape {
        token_count: usize,
        hidden_size: usize,
        values_len: usize,
    },
    #[error(
        "moss-transcribe-diarize prompt embedding audio rows shape is invalid: audio_row_count={audio_row_count} hidden_size={hidden_size} values_len={values_len}"
    )]
    InvalidAudioRowsShape {
        audio_row_count: usize,
        hidden_size: usize,
        values_len: usize,
    },
    #[error(
        "moss-transcribe-diarize prompt embedding audio_pad_positions length {positions_len} != audio row count {audio_row_count}"
    )]
    AudioRowCountMismatch {
        positions_len: usize,
        audio_row_count: usize,
    },
    #[error(
        "moss-transcribe-diarize prompt embedding audio_pad position {position} is out of range for token_count {token_count}"
    )]
    PositionOutOfRange { position: usize, token_count: usize },
    #[error("moss-transcribe-diarize prompt embedding values contain non-finite elements")]
    NonFiniteValues,
}

pub(crate) fn build_moss_td_prompt_embeddings_with_audio_splice(
    token_ids_len: usize,
    audio_pad_positions: &[usize],
    hidden_size: usize,
    token_rows: &[f32],
    audio_rows: &[f32],
) -> Result<MossTdPromptEmbeddings, MossTdPromptEmbeddingError> {
    if hidden_size == 0 {
        return Err(MossTdPromptEmbeddingError::InvalidTokenRowsShape {
            token_count: token_ids_len,
            hidden_size,
            values_len: token_rows.len(),
        });
    }
    let expected_token_values = token_ids_len.checked_mul(hidden_size).ok_or(
        MossTdPromptEmbeddingError::InvalidTokenRowsShape {
            token_count: token_ids_len,
            hidden_size,
            values_len: token_rows.len(),
        },
    )?;
    if token_rows.len() != expected_token_values {
        return Err(MossTdPromptEmbeddingError::InvalidTokenRowsShape {
            token_count: token_ids_len,
            hidden_size,
            values_len: token_rows.len(),
        });
    }
    if token_rows.iter().any(|value| !value.is_finite()) {
        return Err(MossTdPromptEmbeddingError::NonFiniteValues);
    }

    let audio_row_count = audio_pad_positions.len();
    let expected_audio_values = audio_row_count.checked_mul(hidden_size).ok_or(
        MossTdPromptEmbeddingError::InvalidAudioRowsShape {
            audio_row_count,
            hidden_size,
            values_len: audio_rows.len(),
        },
    )?;
    if audio_rows.len() != expected_audio_values {
        return Err(MossTdPromptEmbeddingError::AudioRowCountMismatch {
            positions_len: audio_row_count,
            audio_row_count: audio_rows.len() / hidden_size.max(1),
        });
    }
    if audio_rows.iter().any(|value| !value.is_finite()) {
        return Err(MossTdPromptEmbeddingError::NonFiniteValues);
    }

    let mut combined = token_rows.to_vec();
    for (row_idx, &position) in audio_pad_positions.iter().enumerate() {
        if position >= token_ids_len {
            return Err(MossTdPromptEmbeddingError::PositionOutOfRange {
                position,
                token_count: token_ids_len,
            });
        }
        let src_start = row_idx * hidden_size;
        let dst_start = position * hidden_size;
        combined[dst_start..dst_start + hidden_size]
            .copy_from_slice(&audio_rows[src_start..src_start + hidden_size]);
    }

    Ok(MossTdPromptEmbeddings {
        hidden_size,
        token_count: token_ids_len,
        token_major_values: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splice_replaces_only_the_listed_positions() {
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
        // Positions 1 and 3 (non-contiguous, position 2 stays a real token).
        let spliced = build_moss_td_prompt_embeddings_with_audio_splice(
            4,
            &[1, 3],
            2,
            &token_rows,
            &audio_rows,
        )
        .expect("splice");
        assert_eq!(
            spliced.token_major_values,
            vec![10.0, 11.0, 100.0, 101.0, 30.0, 31.0, 200.0, 201.0]
        );
    }

    #[test]
    fn splice_rejects_position_out_of_range() {
        let error =
            build_moss_td_prompt_embeddings_with_audio_splice(2, &[5], 2, &[0.0; 4], &[0.0; 2])
                .expect_err("must fail");
        assert!(matches!(
            error,
            MossTdPromptEmbeddingError::PositionOutOfRange { .. }
        ));
    }
}
