//! The FireRedASR2-LLM Adapter: 2x frame-stacking + `Linear -> ReLU -> Linear`
//! (upstream `fireredasr2/models/module/adapter.py`, 32 lines, fully
//! transcribed here). Deliberately a plain host-side computation, not a ggml
//! graph: the two weight matrices are small (~22M params total, ~88MB f32,
//! see `scratchpad/fr2/T1-findings.md` S5) and this runs exactly once per
//! utterance (not per decode step), so a hand-rolled `mul_mat` graph would add
//! ggml-runtime plumbing for no measurable benefit -- this mirrors the
//! project's existing "small one-shot projection -> plain Rust" precedent
//! (e.g. `qwen::prompt_embedding`'s splice, also host-side).

use thiserror::Error;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};

use super::tensor_names::{
    ADAPTER_LINEAR1_BIAS, ADAPTER_LINEAR1_WEIGHT, ADAPTER_LINEAR2_BIAS, ADAPTER_LINEAR2_WEIGHT,
};

#[derive(Debug, Error)]
pub(crate) enum FireRedLlmAdapterError {
    #[error("firered-llm adapter tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("firered-llm adapter tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: &'static str,
        shape: String,
        reason: String,
    },
    #[error(
        "firered-llm adapter encoder rows shape is invalid: frame_count={frame_count} encoder_d_model={encoder_d_model} values_len={values_len}"
    )]
    InvalidEncoderRowsShape {
        frame_count: usize,
        encoder_d_model: usize,
        values_len: usize,
    },
    #[error("firered-llm adapter output contains non-finite values")]
    NonFiniteValues,
}

#[derive(Debug, Clone)]
pub(crate) struct FireRedLlmAdapterWeights {
    stacked_input_width: usize,
    llm_dim: usize,
    // [out, in] row-major (ggml OutputByInput convention -- see this module's
    // doc comment for the on-disk layout derivation).
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

pub(crate) fn load_firered_llm_adapter_weights_from_reader(
    reader: &GgufTensorDataReader,
    encoder_d_model: usize,
    downsample_rate: usize,
    llm_dim: usize,
) -> Result<FireRedLlmAdapterWeights, FireRedLlmAdapterError> {
    let stacked_input_width = encoder_d_model.checked_mul(downsample_rate).ok_or(
        FireRedLlmAdapterError::InvalidTensorShape {
            tensor_name: ADAPTER_LINEAR1_WEIGHT,
            shape: "[]".to_string(),
            reason: "encoder_d_model * downsample_rate overflowed".to_string(),
        },
    )?;
    let linear1_weight = load_matrix(reader, ADAPTER_LINEAR1_WEIGHT, llm_dim, stacked_input_width)?;
    let linear1_bias = load_vector(reader, ADAPTER_LINEAR1_BIAS, llm_dim)?;
    let linear2_weight = load_matrix(reader, ADAPTER_LINEAR2_WEIGHT, llm_dim, llm_dim)?;
    let linear2_bias = load_vector(reader, ADAPTER_LINEAR2_BIAS, llm_dim)?;
    Ok(FireRedLlmAdapterWeights {
        stacked_input_width,
        llm_dim,
        linear1_weight,
        linear1_bias,
        linear2_weight,
        linear2_bias,
    })
}

fn load_matrix(
    reader: &GgufTensorDataReader,
    tensor_name: &'static str,
    output_width: usize,
    input_width: usize,
) -> Result<Vec<f32>, FireRedLlmAdapterError> {
    let dims = [input_width as u64, output_width as u64];
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
        .map_err(map_tensor_read_error)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FireRedLlmAdapterError::NonFiniteValues);
    }
    Ok(values)
}

fn load_vector(
    reader: &GgufTensorDataReader,
    tensor_name: &'static str,
    width: usize,
) -> Result<Vec<f32>, FireRedLlmAdapterError> {
    let dims = [width as u64];
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
        .map_err(map_tensor_read_error)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FireRedLlmAdapterError::NonFiniteValues);
    }
    Ok(values)
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> FireRedLlmAdapterError {
    FireRedLlmAdapterError::TensorReadFailed {
        reason: error.to_string(),
    }
}

/// Run the Adapter over a full utterance's encoder output. Upstream
/// (`adapter.py::forward`): drop the trailing `seq_len % downsample_rate`
/// frames, reshape adjacent `downsample_rate` frames into one wider row,
/// `linear1 -> relu -> linear2`. Returns (token-major output rows,
/// output_frame_count); `output_frame_count = frame_count / downsample_rate`
/// (integer division, matching upstream's truncation, not rounding).
pub(crate) fn run_firered_llm_adapter(
    weights: &FireRedLlmAdapterWeights,
    encoder_rows: &[f32],
    frame_count: usize,
    encoder_d_model: usize,
    downsample_rate: usize,
) -> Result<(Vec<f32>, usize), FireRedLlmAdapterError> {
    let expected_len = frame_count.checked_mul(encoder_d_model).ok_or(
        FireRedLlmAdapterError::InvalidEncoderRowsShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        },
    )?;
    if encoder_rows.len() != expected_len {
        return Err(FireRedLlmAdapterError::InvalidEncoderRowsShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }
    if encoder_rows.iter().any(|value| !value.is_finite()) {
        return Err(FireRedLlmAdapterError::NonFiniteValues);
    }

    let output_frame_count = frame_count / downsample_rate.max(1);
    let stacked_width = encoder_d_model * downsample_rate;
    if stacked_width != weights.stacked_input_width {
        return Err(FireRedLlmAdapterError::InvalidEncoderRowsShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }

    let llm_dim = weights.llm_dim;
    let mut output = Vec::with_capacity(output_frame_count * llm_dim);
    let mut stacked_row = vec![0.0_f32; stacked_width];
    let mut hidden_row = vec![0.0_f32; llm_dim];
    for out_frame in 0..output_frame_count {
        // Concatenate `downsample_rate` adjacent encoder frames into one row.
        let src_start = out_frame * downsample_rate * encoder_d_model;
        stacked_row.copy_from_slice(&encoder_rows[src_start..src_start + stacked_width]);

        // linear1 + ReLU.
        matmul_row_output_by_input(
            &stacked_row,
            &weights.linear1_weight,
            &weights.linear1_bias,
            stacked_width,
            &mut hidden_row,
        );
        for value in hidden_row.iter_mut() {
            *value = value.max(0.0);
        }

        // linear2.
        let mut out_row = vec![0.0_f32; llm_dim];
        matmul_row_output_by_input(
            &hidden_row,
            &weights.linear2_weight,
            &weights.linear2_bias,
            llm_dim,
            &mut out_row,
        );
        if out_row.iter().any(|value| !value.is_finite()) {
            return Err(FireRedLlmAdapterError::NonFiniteValues);
        }
        output.extend_from_slice(&out_row);
    }
    Ok((output, output_frame_count))
}

/// `out = W @ in + b`, where `weight` is `[output_width, input_width]`
/// row-major (ggml `OutputByInput` on-disk convention -- see this module's
/// doc comment).
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

    fn toy_weights() -> FireRedLlmAdapterWeights {
        // encoder_d_model=2, downsample_rate=2 -> stacked_input_width=4, llm_dim=3.
        FireRedLlmAdapterWeights {
            stacked_input_width: 4,
            llm_dim: 3,
            // Identity-ish: sum the 4 stacked inputs into each of 3 outputs,
            // scaled distinctly so per-output values are all different.
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
        }
    }

    #[test]
    fn adapter_stacks_two_frames_and_applies_relu_then_identity_linear2() {
        let weights = toy_weights();
        // 3 encoder frames of width 2 -> downsample_rate=2 keeps floor(3/2)=1
        // output frame from the first 2 input frames; the trailing frame is
        // dropped (matches upstream's truncation).
        let encoder_rows = vec![
            1.0, 2.0, // frame 0
            3.0, 4.0, // frame 1
            100.0, 100.0, // frame 2 (dropped)
        ];
        let (output, output_frame_count) =
            run_firered_llm_adapter(&weights, &encoder_rows, 3, 2, 2).expect("adapter");
        assert_eq!(output_frame_count, 1);
        // stacked = [1,2,3,4]; linear1 -> [10, 5, -10+10=0] -> relu -> [10,5,0]
        // linear2 identity -> [10,5,0].
        assert_eq!(output, vec![10.0, 5.0, 0.0]);
    }

    #[test]
    fn adapter_rejects_shape_mismatch() {
        let weights = toy_weights();
        let error = run_firered_llm_adapter(&weights, &[1.0, 2.0, 3.0], 3, 1, 2)
            .expect_err("width mismatch must fail");
        assert!(matches!(
            error,
            FireRedLlmAdapterError::InvalidEncoderRowsShape { .. }
        ));
    }
}
