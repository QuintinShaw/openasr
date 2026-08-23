//! RVQ (residual vector quantization) encode over the packed codebooks. The
//! selected backend produces the encoder hidden rows; host code reads complete
//! score vectors and applies the strict-first nearest-code oracle. No device
//! argmax is used because there is no independent RVQ compact capability.
//!
//! Reference (`quantization.py::EuclideanCodebook.quantize` /
//! `ResidualVectorQuantization.encode`, P2.0 findings SS2): for each of the 8
//! RVQ levels in turn, pick the nearest codebook row to the current residual
//! (`argmax(2*x.C^T - ||C||^2)`, the constant `-||x||^2` term dropped since it
//! doesn't affect the argmax), subtract that row from the residual, and feed
//! the new residual into the next level. All distance math runs in f32 (the
//! upstream `self.quantizer.float()` cast, not an extra conservatism here).

use thiserror::Error;

use crate::ggml_runtime::{
    GGML_TYPE_F16, GGML_TYPE_F32, GgufTensorDataReadError, GgufTensorDataReader, ggml_is_quantized,
};

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
    #[cfg(test)]
    #[error(
        "mimo-asr RVQ codebook shape is invalid: vocab_size={vocab_size} d_model={d_model} values_len={values_len}"
    )]
    InvalidCodebookShape {
        vocab_size: usize,
        d_model: usize,
        values_len: usize,
    },
    #[error(
        "mimo-asr RVQ code tensor layout is invalid: frame_count={frame_count} channels={channels} values_len={values_len}"
    )]
    InvalidCodeLayout {
        frame_count: usize,
        channels: usize,
        values_len: usize,
    },
}

/// Compact channel-major `[channel][frame]` RVQ ids. The encoder hidden rows
/// and complete score vectors are read back only at the host-oracle boundary;
/// this compact payload is then uploaded to the input-local graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MimoRvqCodes {
    frame_count: usize,
    channels: usize,
    values: Vec<i32>,
}

impl MimoRvqCodes {
    pub(crate) fn from_channel_major(
        frame_count: usize,
        channels: usize,
        values: Vec<i32>,
    ) -> Result<Self, MimoRvqError> {
        if values.len() != frame_count.saturating_mul(channels)
            || values.iter().any(|&value| value < 0)
        {
            return Err(MimoRvqError::InvalidCodeLayout {
                frame_count,
                channels,
                values_len: values.len(),
            });
        }
        Ok(Self {
            frame_count,
            channels,
            values,
        })
    }

    pub(crate) const fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub(crate) const fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn values(&self) -> &[i32] {
        &self.values
    }

    pub(crate) fn code(&self, frame: usize, channel: usize) -> Option<u32> {
        if frame >= self.frame_count || channel >= self.channels {
            return None;
        }
        u32::try_from(self.values[channel * self.frame_count + frame]).ok()
    }

    pub(crate) fn truncate_frames(&mut self, frame_count: usize) -> Result<(), MimoRvqError> {
        if frame_count > self.frame_count {
            return Err(MimoRvqError::InvalidCodeLayout {
                frame_count,
                channels: self.channels,
                values_len: self.values.len(),
            });
        }
        if frame_count == self.frame_count {
            return Ok(());
        }
        let mut truncated = Vec::with_capacity(frame_count.saturating_mul(self.channels));
        for channel in 0..self.channels {
            let start = channel * self.frame_count;
            truncated.extend_from_slice(&self.values[start..start + frame_count]);
        }
        self.frame_count = frame_count;
        self.values = truncated;
        Ok(())
    }
}

pub(crate) struct MimoRvqCodebooks {
    d_model: usize,
    /// One `[vocab_size][d_model]` row-major table per packed level.
    levels: Vec<Vec<f32>>,
    vocab_sizes: Vec<usize>,
}

impl MimoRvqCodebooks {
    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: &MimoAudiotokMetadata,
    ) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_usize(
            metadata
                .rvq_packed
                .checked_mul(std::mem::size_of::<Vec<f32>>())
                .ok_or_else(|| "mimo-asr RVQ table descriptors quote overflowed".to_string())?,
            "mimo-asr RVQ table descriptors quote",
        )?;
        let value_count = metadata
            .codebook_sizes
            .iter()
            .try_fold(0usize, |total, &vocab_size| {
                let level = (vocab_size as usize).checked_mul(metadata.d_model)?;
                total.checked_add(level)
            })
            .ok_or_else(|| "mimo-asr RVQ values quote overflowed".to_string())?;
        bytes.add_usize(
            value_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "mimo-asr RVQ value bytes quote overflowed".to_string())?,
            "mimo-asr RVQ values quote",
        )?;
        bytes.add_usize(
            metadata
                .rvq_packed
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| "mimo-asr RVQ vocabulary quote overflowed".to_string())?,
            "mimo-asr RVQ vocabulary quote",
        )?;
        Ok(bytes.finish())
    }

    pub(crate) fn quoted_construction_system_memory_bytes(
        reader: &GgufTensorDataReader,
        metadata: &MimoAudiotokMetadata,
    ) -> Result<(u64, u64), String> {
        let retained = Self::quoted_retained_system_memory_bytes(metadata)?;
        let mut largest_extra = 0_u64;
        for (level, &vocab_size) in metadata.codebook_sizes.iter().enumerate() {
            let name = audiotok_codebook_name(level);
            let tensor = reader
                .tensor_index()
                .get(&name)
                .ok_or_else(|| format!("mimo-asr RVQ codebook '{name}' is missing"))?;
            let elements = tensor.num_elements().ok_or_else(|| {
                format!("mimo-asr RVQ codebook '{name}' element count overflowed")
            })?;
            let expected_elements = u64::from(vocab_size)
                .checked_mul(metadata.d_model as u64)
                .ok_or_else(|| format!("mimo-asr RVQ codebook '{name}' shape overflowed"))?;
            if elements != expected_elements {
                return Err(format!(
                    "mimo-asr RVQ codebook '{name}' element count {elements} does not match expected {expected_elements}"
                ));
            }
            largest_extra =
                largest_extra.max(materialization_extra_bytes(tensor.ggml_type, elements)?);
        }
        let peak = retained
            .checked_add(largest_extra)
            .ok_or_else(|| "mimo-asr RVQ construction peak quote overflowed".to_string())?;
        Ok((peak, retained))
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.levels, "mimo-asr RVQ level tables")?;
        for level in &self.levels {
            bytes.add_vec(level, "mimo-asr RVQ level table values")?;
        }
        bytes.add_vec(&self.vocab_sizes, "mimo-asr RVQ vocabulary sizes")?;
        Ok(bytes.finish())
    }
}

fn materialization_extra_bytes(ggml_type: i32, elements: u64) -> Result<u64, String> {
    let elements = usize::try_from(elements)
        .map_err(|_| "mimo-asr RVQ materialization element count exceeds usize".to_string())?;
    match ggml_type {
        GGML_TYPE_F32 => Ok(0),
        GGML_TYPE_F16 => u64::try_from(
            elements
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| "mimo-asr RVQ F16 transient quote overflowed".to_string())?,
        )
        .map_err(|_| "mimo-asr RVQ F16 transient quote exceeds u64".to_string()),
        other if unsafe { ggml_is_quantized(other) } => Ok(0),
        other => Err(format!(
            "mimo-asr RVQ codebook ggml type {other} is unsupported for host materialization"
        )),
    }
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
/// `[frame_count][rvq_packed]` codebook indices, one complete score-vector
/// readback per level and host nearest-code lookup per frame. The residual is
/// updated only from that host oracle, so no backend argmax can affect codes.
pub(crate) fn encode_rvq_codes(
    codebooks: &MimoRvqCodebooks,
    hidden_rows: &[f32],
    frame_count: usize,
) -> Result<MimoRvqCodes, MimoRvqError> {
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
    let mut codes = vec![0_i32; frame_count.saturating_mul(rvq_packed)];
    let mut residual = vec![0.0_f32; d_model];
    let largest_vocab = codebooks.vocab_sizes.iter().copied().max().unwrap_or(0);
    let mut scores = Vec::with_capacity(largest_vocab);
    for frame_idx in 0..frame_count {
        residual.copy_from_slice(&hidden_rows[frame_idx * d_model..(frame_idx + 1) * d_model]);
        for level in 0..rvq_packed {
            let table = &codebooks.levels[level];
            let vocab_size = codebooks.vocab_sizes[level];
            complete_scores_for_residual(&residual, table, vocab_size, d_model, &mut scores);
            let best_idx = nearest_code_from_complete_scores(&scores, vocab_size);
            codes[level * frame_count + frame_idx] = best_idx as i32;
            let best_row = &table[best_idx * d_model..(best_idx + 1) * d_model];
            for (r, c) in residual.iter_mut().zip(best_row.iter()) {
                *r -= *c;
            }
        }
    }
    MimoRvqCodes::from_channel_major(frame_count, rvq_packed, codes)
}

/// Per-codebook squared row norms retained for quantized-shape validation and
/// diagnostics. RVQ selection itself never uses a device-side argmax.
#[cfg(test)]
pub(crate) fn codebook_row_norm_sq(
    table: &[f32],
    vocab_size: usize,
    d_model: usize,
) -> Result<Vec<f32>, MimoRvqError> {
    if table.len() != vocab_size.saturating_mul(d_model) {
        return Err(MimoRvqError::InvalidCodebookShape {
            vocab_size,
            d_model,
            values_len: table.len(),
        });
    }
    Ok(table
        .chunks_exact(d_model)
        .map(|row| row.iter().map(|value| value * value).sum())
        .collect())
}

/// Fill one complete RVQ score vector using f32 arithmetic. Keeping this as a
/// separate host operation makes the result contract explicit: callers must
/// read all scores and then apply [`nearest_code_from_complete_scores`].
fn complete_scores_for_residual(
    x: &[f32],
    table: &[f32],
    vocab_size: usize,
    d_model: usize,
    scores: &mut Vec<f32>,
) {
    scores.clear();
    scores.extend((0..vocab_size).map(|v| {
        let row = &table[v * d_model..(v + 1) * d_model];
        let (mut dot, mut norm_sq) = (0.0_f32, 0.0_f32);
        for (xi, ci) in x.iter().zip(row.iter()) {
            dot += xi * ci;
            norm_sq += ci * ci;
        }
        2.0 * dot - norm_sq
    }));
}

/// Select the first maximal score. The strict `>` update is the RVQ oracle's
/// tie contract and must not be replaced with a provider-specific argmax.
fn nearest_code_from_complete_scores(scores: &[f32], vocab_size: usize) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, &score) in scores.iter().take(vocab_size).enumerate() {
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

/// `argmax_v(2 * x.dot(C[v]) - ||C[v]||^2)` -- mathematically equivalent to
/// minimizing `||x - C[v]||^2` (the constant `-||x||^2` term is dropped since it
/// does not depend on `v`). Returns `(index, row)` using strict first-max ties.
#[cfg(test)]
fn nearest_code<'a>(
    x: &[f32],
    table: &'a [f32],
    vocab_size: usize,
    d_model: usize,
) -> (usize, &'a [f32]) {
    let mut scores = Vec::with_capacity(vocab_size);
    complete_scores_for_residual(x, table, vocab_size, d_model, &mut scores);
    let best_idx = nearest_code_from_complete_scores(&scores, vocab_size);
    (
        best_idx,
        &table[best_idx * d_model..(best_idx + 1) * d_model],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GGML_TYPE_Q4_K;

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
    fn nearest_code_uses_strict_first_max_on_exact_tie() {
        let scores = [1.0_f32, 1.0, 0.5];
        assert_eq!(nearest_code_from_complete_scores(&scores, 3), 0);
        assert_eq!(
            nearest_code_from_complete_scores(&[2.0, 1.0, 5.0, 5.0], 4),
            2,
            "MiMo RVQ host first-max must keep the first equal code"
        );
    }

    #[test]
    fn complete_scores_have_the_requested_quantized_shape() {
        let table = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut scores = Vec::new();
        complete_scores_for_residual(&[0.5, 0.5], &table, 2, 2, &mut scores);
        assert_eq!(scores.len(), 2);
        assert_eq!(scores, vec![0.0, 0.0]);
    }

    #[test]
    fn encode_rvq_codes_is_residual_and_sequential() {
        let codebooks = toy_codebooks();
        // x = (1.4, 0.1): level0 picks code0=(1,0) [closer], residual=(0.4,0.1);
        // level1 picks code0=(0.5,0) [closer to (0.4,0.1) than (0,0.5)].
        let hidden = vec![1.4_f32, 0.1];
        let codes = encode_rvq_codes(&codebooks, &hidden, 1).expect("encode");
        assert_eq!(codes.frame_count(), 1);
        assert_eq!(codes.channels(), 2);
        assert_eq!(codes.code(0, 0), Some(0));
        assert_eq!(codes.code(0, 1), Some(0));
    }

    #[test]
    fn encode_rvq_codes_rejects_shape_mismatch() {
        let codebooks = toy_codebooks();
        let error = encode_rvq_codes(&codebooks, &[1.0, 2.0, 3.0], 2).expect_err("must fail");
        assert!(matches!(error, MimoRvqError::InvalidHiddenRowsShape { .. }));
    }

    #[test]
    fn materialization_peak_quotes_dtype_specific_transient() {
        assert_eq!(materialization_extra_bytes(GGML_TYPE_F32, 10), Ok(0));
        assert_eq!(materialization_extra_bytes(GGML_TYPE_F16, 10), Ok(20));
        assert_eq!(materialization_extra_bytes(GGML_TYPE_Q4_K, 256), Ok(0));
    }

    #[test]
    fn materialization_peak_rejects_unsupported_type_and_overflow() {
        assert!(materialization_extra_bytes(999, 10).is_err());
        let too_many = (usize::MAX as u64 / 2).saturating_add(1);
        assert!(materialization_extra_bytes(GGML_TYPE_F16, too_many).is_err());
    }

    #[test]
    fn codebook_row_norms_reject_quantized_shape_mismatch() {
        let error = codebook_row_norm_sq(&[1.0, 2.0, 3.0], 2, 2).expect_err("must fail");
        assert!(matches!(error, MimoRvqError::InvalidCodebookShape { .. }));
    }
}
