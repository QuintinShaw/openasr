//! RVQ (residual vector quantization) encode over the first 8 packed
//! codebooks. CPU keeps the scalar f32 implementation as the exact numerical
//! oracle. Accelerated execution builds the same sequential residual loop in
//! the audio-tokenizer ggml graph, so the large hidden rows and codebooks never
//! round-trip through host memory.
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
    GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor, GgufTensorDataReadError,
    GgufTensorDataReader,
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
    #[error("mimo-asr RVQ graph construction failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
}

fn graph_err(step: &'static str, source: GgmlCpuGraphError) -> MimoRvqError {
    MimoRvqError::GraphBuildFailed { step, source }
}

/// Compact channel-major `[channel][frame]` RVQ ids. This is the only payload
/// copied from the audio-tokenizer device graph back to the host before the
/// input-local graph uploads it again; the f32 hidden rows and both large table
/// families remain device-resident.
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

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.levels, "mimo-asr RVQ level tables")?;
        for level in &self.levels {
            bytes.add_vec(level, "mimo-asr RVQ level table values")?;
        }
        bytes.add_vec(&self.vocab_sizes, "mimo-asr RVQ vocabulary sizes")?;
        Ok(bytes.finish())
    }

    /// Peak temporary host allocation used by the accelerated constructor to
    /// derive one norm vector at a time. The full codebooks themselves stay in
    /// their native GGUF storage and are not retained as host f32 tables.
    pub(crate) fn quoted_device_construction_peak_system_memory_bytes(
        metadata: &MimoAudiotokMetadata,
    ) -> Result<u64, String> {
        let largest_vocab = metadata.codebook_sizes.iter().copied().max().unwrap_or(0) as usize;
        let values = largest_vocab
            .checked_mul(metadata.d_model)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "mimo-asr device RVQ construction quote overflowed".to_string())?;
        let norms = largest_vocab
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "mimo-asr device RVQ norm quote overflowed".to_string())?;
        let peak = values
            .checked_add(norms)
            .ok_or_else(|| "mimo-asr device RVQ construction quote overflowed".to_string())?;
        u64::try_from(peak)
            .map_err(|_| "mimo-asr device RVQ construction quote exceeds u64".to_string())
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
/// `[frame_count][rvq_packed]` codebook indices, one nearest-code lookup per
/// level per frame, feeding each level's residual into the next.
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
    for frame_idx in 0..frame_count {
        residual.copy_from_slice(&hidden_rows[frame_idx * d_model..(frame_idx + 1) * d_model]);
        for level in 0..rvq_packed {
            let table = &codebooks.levels[level];
            let vocab_size = codebooks.vocab_sizes[level];
            let (best_idx, best_row) = nearest_code(&residual, table, vocab_size, d_model);
            codes[level * frame_count + frame_idx] = best_idx as i32;
            for (r, c) in residual.iter_mut().zip(best_row.iter()) {
                *r -= *c;
            }
        }
    }
    MimoRvqCodes::from_channel_major(frame_count, rvq_packed, codes)
}

/// Per-codebook squared row norms used by both the scalar oracle's distance
/// identity and the accelerated graph's broadcast subtraction.
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

/// Build the sequential device RVQ loop. `hidden` is `[d_model, frames]`;
/// each level is `(codebook[d_model,vocab], row_norm_sq[vocab])`. The output
/// is one contiguous i32 tensor `[frames, levels]`, whose physical order is
/// channel-major and can be uploaded directly to the device-side speech-table
/// lookup graph.
pub(crate) fn build_rvq_codes_graph<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    hidden: GgmlCpuTensor<'a>,
    levels: &[(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>)],
    frame_count: usize,
) -> Result<GgmlCpuTensor<'a>, MimoRvqError> {
    if levels.is_empty() || frame_count == 0 {
        return Err(MimoRvqError::InvalidCodeLayout {
            frame_count,
            channels: levels.len(),
            values_len: 0,
        });
    }
    let mut residual = hidden;
    let mut packed_codes = None;
    for &(codebook, norm_sq) in levels {
        let dot = graph
            .mul_mat(codebook, residual)
            .map_err(|source| graph_err("rvq_dot", source))?;
        let doubled = graph
            .scale(dot, 2.0)
            .map_err(|source| graph_err("rvq_double_dot", source))?;
        let scores = graph
            .sub(doubled, norm_sq)
            .map_err(|source| graph_err("rvq_distance_scores", source))?;
        // ggml backends may choose different indices for an exact score tie.
        // Real codebooks do not contain duplicate rows; CPU remains the exact
        // first-max oracle and the real-pack CPU/Metal bridge gate catches any
        // numerically meaningful decision drift.
        let indices = graph
            .top1_argmax(scores)
            .map_err(|source| graph_err("rvq_argmax", source))?;
        let selected = graph
            .get_rows(codebook, indices)
            .map_err(|source| graph_err("rvq_select", source))?;
        residual = graph
            .sub(residual, selected)
            .map_err(|source| graph_err("rvq_residual", source))?;
        let indices = graph
            .reshape_2d(indices, frame_count, 1)
            .map_err(|source| graph_err("rvq_codes_row", source))?;
        packed_codes = Some(match packed_codes {
            Some(previous) => graph
                .concat(previous, indices, 1)
                .map_err(|source| graph_err("rvq_codes_concat", source))?,
            None => indices,
        });
    }
    packed_codes.ok_or(MimoRvqError::InvalidCodeLayout {
        frame_count,
        channels: 0,
        values_len: 0,
    })
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
    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphRunner};

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
    fn device_rvq_graph_matches_scalar_sequential_oracle() {
        let codebooks = toy_codebooks();
        let hidden = vec![1.4_f32, 0.1, 0.1, 1.4];
        let expected = encode_rvq_codes(&codebooks, &hidden, 2).expect("scalar RVQ");

        let mut runner =
            GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("CPU graph runner");
        let mut graph = runner.start_graph();
        let hidden_tensor = graph
            .new_tensor_2d_f32(2, 2, "rvq_hidden")
            .expect("hidden tensor");
        graph.set_input(hidden_tensor).expect("hidden input");

        let mut level_tensors = Vec::with_capacity(codebooks.levels.len());
        let mut uploads = Vec::with_capacity(codebooks.levels.len());
        for (level, table) in codebooks.levels.iter().enumerate() {
            let vocab_size = codebooks.vocab_sizes[level];
            let codebook = graph
                .new_tensor_2d_f32(2, vocab_size, "rvq_codebook")
                .expect("codebook tensor");
            let norms = graph
                .new_tensor_1d_f32(vocab_size, "rvq_norms")
                .expect("norm tensor");
            graph.set_input(codebook).expect("codebook input");
            graph.set_input(norms).expect("norm input");
            level_tensors.push((codebook, norms));
            uploads.push((
                codebook,
                table,
                norms,
                codebook_row_norm_sq(table, vocab_size, 2).expect("row norms"),
            ));
        }

        let codes =
            build_rvq_codes_graph(&graph, hidden_tensor, &level_tensors, 2).expect("RVQ graph");
        graph.set_output(codes).expect("RVQ graph output");
        graph
            .set_f32_slice(hidden_tensor, &hidden, "rvq_hidden")
            .expect("hidden upload");
        for (codebook, table, norms, norm_values) in uploads {
            graph
                .set_f32_slice(codebook, table, "rvq_codebook")
                .expect("codebook upload");
            graph
                .set_f32_slice(norms, &norm_values, "rvq_norms")
                .expect("norm upload");
        }
        let actual = graph
            .compute_output_i32(codes, expected.values().len())
            .expect("RVQ compute");
        assert_eq!(actual, expected.values());
    }
}
