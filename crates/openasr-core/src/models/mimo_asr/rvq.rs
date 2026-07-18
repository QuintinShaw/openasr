//! RVQ (residual vector quantization) encode over the first 8 packed
//! codebooks -- a pure host-side computation (mirrors `firered_llm::adapter_graph`'s
//! precedent: small, runs once per utterance, not per decode step, so a ggml
//! graph would be plumbing for no benefit).
//!
//! Reference (`quantization.py::EuclideanCodebook.quantize` /
//! `ResidualVectorQuantization.encode`, P2.0 findings SS2): for each of the 8
//! RVQ levels in turn, pick the nearest codebook row to the current residual
//! (`argmax(2*x.C^T - ||C||^2)`, the constant `-||x||^2` term dropped since it
//! doesn't affect the argmax), subtract that row from the residual, and feed
//! the new residual into the next level. All distance math runs in f32 (the
//! upstream `self.quantizer.float()` cast, not an extra conservatism here).

use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::runtime_contract::MimoAudiotokMetadata;
use super::tensor_names::audiotok_codebook_name;

#[derive(Debug, Error)]
pub(crate) enum MimoRvqError {
    #[error("mimo-asr RVQ codebook '{name}' could not be read: {source}")]
    TensorRead {
        name: String,
        #[source]
        source: GgufTensorDataReadError,
    },
    #[error(
        "mimo-asr RVQ encoder hidden rows shape is invalid: frame_count={frame_count} d_model={d_model} values_len={values_len}"
    )]
    InvalidHiddenRowsShape {
        frame_count: usize,
        d_model: usize,
        values_len: usize,
    },
}

pub(crate) struct MimoRvqCodebooks {
    d_model: usize,
    /// One `[vocab_size][d_model]` row-major table per packed level.
    levels: Vec<Vec<f32>>,
    vocab_sizes: Vec<usize>,
}

pub(crate) fn load_mimo_rvq_codebooks_from_reader(
    reader: &GgufTensorDataReader,
    metadata: &MimoAudiotokMetadata,
) -> Result<MimoRvqCodebooks, MimoRvqError> {
    let mut levels = Vec::with_capacity(metadata.rvq_packed);
    let mut vocab_sizes = Vec::with_capacity(metadata.rvq_packed);
    for (level, &vocab_size) in metadata.codebook_sizes.iter().enumerate() {
        let vocab_size = vocab_size as usize;
        let name = audiotok_codebook_name(level);
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(
                &name,
                &[metadata.d_model as u64, vocab_size as u64],
            )
            .map_err(|source| MimoRvqError::TensorRead { name, source })?;
        levels.push(values);
        vocab_sizes.push(vocab_size);
    }
    Ok(MimoRvqCodebooks {
        d_model: metadata.d_model,
        levels,
        vocab_sizes,
    })
}

/// Residual-quantize `hidden_rows` (`[frame_count][d_model]` row-major) into
/// `[frame_count][rvq_packed]` codebook indices, one nearest-code lookup per
/// level per frame, feeding each level's residual into the next.
pub(crate) fn encode_rvq_codes(
    codebooks: &MimoRvqCodebooks,
    hidden_rows: &[f32],
    frame_count: usize,
) -> Result<Vec<Vec<u32>>, MimoRvqError> {
    let d_model = codebooks.d_model;
    let expected_len = frame_count.saturating_mul(d_model);
    if hidden_rows.len() != expected_len {
        return Err(MimoRvqError::InvalidHiddenRowsShape {
            frame_count,
            d_model,
            values_len: hidden_rows.len(),
        });
    }
    let rvq_packed = codebooks.levels.len();
    let mut codes = vec![vec![0u32; rvq_packed]; frame_count];
    let mut residual = vec![0.0_f32; d_model];
    for frame_idx in 0..frame_count {
        residual.copy_from_slice(&hidden_rows[frame_idx * d_model..(frame_idx + 1) * d_model]);
        for (level, code_slot) in codes[frame_idx].iter_mut().enumerate() {
            let table = &codebooks.levels[level];
            let vocab_size = codebooks.vocab_sizes[level];
            let (best_idx, best_row) = nearest_code(&residual, table, vocab_size, d_model);
            *code_slot = best_idx as u32;
            for (r, c) in residual.iter_mut().zip(best_row.iter()) {
                *r -= *c;
            }
        }
    }
    Ok(codes)
}

/// `argmax_v(2 * x.dot(C[v]) - ||C[v]||^2)` -- mathematically equivalent to
/// minimizing `||x - C[v]||^2` (the constant `-||x||^2` term is dropped since
/// it does not depend on `v`; see `quantization.py`'s own derivation, P2.0
/// findings SS2 step 9). Returns `(index, row)`.
fn nearest_code<'a>(
    x: &[f32],
    table: &'a [f32],
    vocab_size: usize,
    d_model: usize,
) -> (usize, &'a [f32]) {
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for v in 0..vocab_size {
        let row = &table[v * d_model..(v + 1) * d_model];
        let mut dot = 0.0_f32;
        let mut norm_sq = 0.0_f32;
        for (xi, ci) in x.iter().zip(row.iter()) {
            dot += xi * ci;
            norm_sq += ci * ci;
        }
        let score = 2.0 * dot - norm_sq;
        if score > best_score {
            best_score = score;
            best_idx = v;
        }
    }
    (
        best_idx,
        &table[best_idx * d_model..(best_idx + 1) * d_model],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_codebooks() -> MimoRvqCodebooks {
        // d_model=2, 2 packed levels, vocab 2 each.
        MimoRvqCodebooks {
            d_model: 2,
            levels: vec![
                vec![1.0, 0.0, 0.0, 1.0], // level 0: code0=(1,0) code1=(0,1)
                vec![0.5, 0.0, 0.0, 0.5], // level 1 (residual-scale codes)
            ],
            vocab_sizes: vec![2, 2],
        }
    }

    #[test]
    fn nearest_code_picks_closest_row() {
        let table = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0];
        let (idx, row) = nearest_code(&[0.9, 0.1], &table, 3, 2);
        assert_eq!(idx, 0);
        assert_eq!(row, &[1.0, 0.0]);
    }

    #[test]
    fn encode_rvq_codes_is_residual_and_sequential() {
        let codebooks = toy_codebooks();
        // x = (1.4, 0.1): level0 picks code0=(1,0) [closer], residual=(0.4,0.1);
        // level1 picks code0=(0.5,0) [closer to (0.4,0.1) than (0,0.5)].
        let hidden = vec![1.4_f32, 0.1];
        let codes = encode_rvq_codes(&codebooks, &hidden, 1).expect("encode");
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0], vec![0, 0]);
    }

    #[test]
    fn encode_rvq_codes_rejects_shape_mismatch() {
        let codebooks = toy_codebooks();
        let error = encode_rvq_codes(&codebooks, &[1.0, 2.0, 3.0], 2).expect_err("must fail");
        assert!(matches!(error, MimoRvqError::InvalidHiddenRowsShape { .. }));
    }
}
