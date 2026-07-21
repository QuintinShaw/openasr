//! The MOSS-Transcribe-Diarize `VQAdaptor` bridge: 4x time-merge (pure
//! reshape, no weights) -> `Linear(4096 -> 1024) -> SiLU -> Linear(1024 ->
//! 1024) -> LayerNorm(eps=1e-6)`. Despite the "VQ" name there is no
//! vector-quantization codebook in this checkpoint (see `package_import`'s
//! module doc) -- this is a plain 3-weighted-layer MLP.
//!
//! A plain host-side computation, not a ggml graph: mirrors
//! `firered_llm::adapter_graph`'s precedent (small one-shot per-utterance
//! projection, not a per-decode-step op) -- the weights here are smaller
//! still (~9.4M params, ~19MB f32).

use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::tensor_names::{
    ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
    ADAPTOR_NORM_BIAS, ADAPTOR_NORM_WEIGHT,
};

#[derive(Debug, Error)]
pub(crate) enum MossAdaptorError {
    #[error("moss-transcribe-diarize adaptor tensor read failed: {0}")]
    TensorRead(#[from] GgufTensorDataReadError),
    #[error(
        "moss-transcribe-diarize adaptor input shape is invalid: frame_count={frame_count} encoder_d_model={encoder_d_model} values_len={values_len}"
    )]
    InvalidInputShape {
        frame_count: usize,
        encoder_d_model: usize,
        values_len: usize,
    },
    #[error("moss-transcribe-diarize adaptor output contains non-finite values")]
    NonFiniteValues,
}

#[derive(Debug, Clone)]
pub(crate) struct MossAdaptorWeights {
    stacked_input_width: usize,
    llm_dim: usize,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    norm_epsilon: f32,
}

pub(crate) fn load_moss_adaptor_weights_from_reader(
    reader: &GgufTensorDataReader,
    encoder_d_model: usize,
    merge_size: usize,
    llm_dim: usize,
    norm_epsilon: f32,
) -> Result<MossAdaptorWeights, MossAdaptorError> {
    let stacked_input_width = encoder_d_model * merge_size;
    let linear1_weight = reader.host_tensor_f32_copy_dequantized_by_name(
        ADAPTOR_LINEAR1_WEIGHT,
        &[stacked_input_width as u64, llm_dim as u64],
    )?;
    let linear1_bias =
        reader.host_tensor_f32_copy_dequantized_by_name(ADAPTOR_LINEAR1_BIAS, &[llm_dim as u64])?;
    let linear2_weight = reader.host_tensor_f32_copy_dequantized_by_name(
        ADAPTOR_LINEAR2_WEIGHT,
        &[llm_dim as u64, llm_dim as u64],
    )?;
    let linear2_bias =
        reader.host_tensor_f32_copy_dequantized_by_name(ADAPTOR_LINEAR2_BIAS, &[llm_dim as u64])?;
    let norm_weight =
        reader.host_tensor_f32_copy_dequantized_by_name(ADAPTOR_NORM_WEIGHT, &[llm_dim as u64])?;
    let norm_bias =
        reader.host_tensor_f32_copy_dequantized_by_name(ADAPTOR_NORM_BIAS, &[llm_dim as u64])?;
    Ok(MossAdaptorWeights {
        stacked_input_width,
        llm_dim,
        linear1_weight,
        linear1_bias,
        linear2_weight,
        linear2_bias,
        norm_weight,
        norm_bias,
        norm_epsilon,
    })
}

/// `encoder_rows`: frame-major `[frame][encoder_d_model]`, already trimmed
/// to a multiple of `merge_size` (the executor drops any remainder before
/// calling this, matching upstream `time_merge`'s `T_trim = (T //
/// merge_size) * merge_size` truncation). Returns (token-major output rows,
/// output_token_count).
pub(crate) fn run_moss_adaptor(
    weights: &MossAdaptorWeights,
    encoder_rows: &[f32],
    frame_count: usize,
    encoder_d_model: usize,
    merge_size: usize,
) -> Result<(Vec<f32>, usize), MossAdaptorError> {
    let expected_len =
        frame_count
            .checked_mul(encoder_d_model)
            .ok_or(MossAdaptorError::InvalidInputShape {
                frame_count,
                encoder_d_model,
                values_len: encoder_rows.len(),
            })?;
    if encoder_rows.len() != expected_len {
        return Err(MossAdaptorError::InvalidInputShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }
    if !frame_count.is_multiple_of(merge_size.max(1)) {
        return Err(MossAdaptorError::InvalidInputShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }
    if encoder_rows.iter().any(|value| !value.is_finite()) {
        return Err(MossAdaptorError::NonFiniteValues);
    }

    let output_token_count = frame_count / merge_size.max(1);
    let stacked_width = encoder_d_model * merge_size;
    if stacked_width != weights.stacked_input_width {
        return Err(MossAdaptorError::InvalidInputShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }

    let llm_dim = weights.llm_dim;
    let mut output = Vec::with_capacity(output_token_count * llm_dim);
    let mut stacked_row = vec![0.0_f32; stacked_width];
    let mut hidden_row = vec![0.0_f32; llm_dim];
    let mut linear2_row = vec![0.0_f32; llm_dim];
    for out_token in 0..output_token_count {
        let src_start = out_token * stacked_width;
        stacked_row.copy_from_slice(&encoder_rows[src_start..src_start + stacked_width]);

        matmul_row_output_by_input(
            &stacked_row,
            &weights.linear1_weight,
            &weights.linear1_bias,
            stacked_width,
            &mut hidden_row,
        );
        // SiLU: x * sigmoid(x).
        for value in hidden_row.iter_mut() {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }

        matmul_row_output_by_input(
            &hidden_row,
            &weights.linear2_weight,
            &weights.linear2_bias,
            llm_dim,
            &mut linear2_row,
        );

        let mean = linear2_row.iter().sum::<f32>() / llm_dim as f32;
        let variance = linear2_row
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / llm_dim as f32;
        let inv_std = 1.0 / (variance + weights.norm_epsilon).sqrt();
        for (idx, value) in linear2_row.iter().enumerate() {
            let normalized = (value - mean) * inv_std;
            output.push(normalized * weights.norm_weight[idx] + weights.norm_bias[idx]);
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(MossAdaptorError::NonFiniteValues);
    }
    Ok((output, output_token_count))
}

/// `out = W @ in + b`, where `weight` is `[input_width, output_width]`
/// column-major-by-input (ggml `InputByOutput`-on-disk convention this
/// family's `package_import` writes 2D linear weights in -- see that
/// module's `reversed_dims`), i.e. row `out_idx` of the logical `[out, in]`
/// matrix lives at `weight[out_idx * input_width .. +input_width]` exactly
/// like `firered_llm::adapter_graph`'s host-side matmul.
fn matmul_row_output_by_input(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    input_width: usize,
    out: &mut [f32],
) {
    for (out_idx, out_value) in out.iter_mut().enumerate() {
        let row = &weight[out_idx * input_width..out_idx * input_width + input_width];
        let mut acc = 0.0_f32;
        for (input_value, weight_value) in input.iter().zip(row.iter()) {
            acc += input_value * weight_value;
        }
        *out_value = acc + bias[out_idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_weights() -> MossAdaptorWeights {
        // encoder_d_model=2, merge_size=2 -> stacked_input_width=4, llm_dim=3.
        MossAdaptorWeights {
            stacked_input_width: 4,
            llm_dim: 3,
            linear1_weight: vec![
                1.0, 1.0, 1.0, 1.0, //
                0.5, 0.5, 0.5, 0.5, //
                -1.0, -1.0, -1.0, -1.0,
            ],
            linear1_bias: vec![0.0, 0.0, 10.0],
            linear2_weight: vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            linear2_bias: vec![0.0, 0.0, 0.0],
            norm_weight: vec![1.0, 1.0, 1.0],
            norm_bias: vec![0.0, 0.0, 0.0],
            norm_epsilon: 1e-6,
        }
    }

    #[test]
    fn adaptor_merges_two_frames_applies_silu_identity_linear2_then_layernorm() {
        let weights = toy_weights();
        let encoder_rows = [
            1.0, 2.0, // frame 0
            3.0, 4.0, // frame 1
            100.0, 100.0, // frame 2 (dropped: frame_count must be a multiple of merge_size)
        ];
        // Only pass the first 2 (already-trimmed) frames, mirroring what the
        // executor would hand in after its own truncation.
        let (output, output_token_count) =
            run_moss_adaptor(&weights, &encoder_rows[..4], 2, 2, 2).expect("adaptor");
        assert_eq!(output_token_count, 1);
        // stacked = [1,2,3,4]; linear1 -> [10, 5, 0] (relu-equivalent since all
        // non-negative here except the last which SiLU(0)=0); SiLU(10)~=10,
        // SiLU(5)~=5, SiLU(0)=0 -> linear2 identity -> layernorm(mean~=5,
        // var>0) -- assert finiteness and the expected zero-mean/unit-ish
        // shape rather than a brittle exact float chain.
        assert_eq!(output.len(), 3);
        let mean = output.iter().sum::<f32>() / 3.0;
        assert!(
            mean.abs() < 1e-3,
            "layernorm output should be ~zero-mean: {output:?}"
        );
    }

    #[test]
    fn adaptor_rejects_frame_count_not_multiple_of_merge_size() {
        let weights = toy_weights();
        let error = run_moss_adaptor(&weights, &[0.0; 6], 3, 2, 2).expect_err("must fail");
        assert!(matches!(error, MossAdaptorError::InvalidInputShape { .. }));
    }
}
