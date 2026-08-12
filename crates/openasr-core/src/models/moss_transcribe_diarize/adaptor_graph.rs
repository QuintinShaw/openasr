//! The MOSS-Transcribe-Diarize `VQAdaptor` bridge: 4x time merge (a view/
//! reshape, no weights) -> `Linear(4096 -> 1024) -> SiLU -> Linear(1024 ->
//! 1024) -> LayerNorm(eps=1e-6)`. Despite the "VQ" name there is no vector-
//! quantization codebook in this checkpoint.
//!
//! On accelerated lanes the adaptor is appended directly to the resident
//! Whisper encoder graph. Its two large linear weights stay in their native
//! GGUF storage and bind through the encoder's already-open
//! [`GgmlLoadedWeightContext`]. This keeps the neural bridge on the selected
//! backend, avoids a scalar host matmul and a second runner/weight context, and
//! avoids materializing a ~19 MiB f32 copy. CPU deliberately retains the
//! original scalar implementation as the official numerical oracle.

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgufTensorDataReadError, GgufTensorDataReader,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::tensor_names::{
    ADAPTOR_LINEAR1_BIAS, ADAPTOR_LINEAR1_WEIGHT, ADAPTOR_LINEAR2_BIAS, ADAPTOR_LINEAR2_WEIGHT,
    ADAPTOR_NORM_BIAS, ADAPTOR_NORM_WEIGHT,
};

#[derive(Debug, Error)]
pub(crate) enum MossAdaptorError {
    #[error("moss-transcribe-diarize adaptor tensor read failed: {0}")]
    TensorRead(#[from] GgufTensorDataReadError),
    #[error("moss-transcribe-diarize adaptor is missing tensor '{name}'")]
    MissingTensor { name: &'static str },
    #[error(
        "moss-transcribe-diarize host adaptor input shape is invalid: frame_count={frame_count} encoder_d_model={encoder_d_model} values_len={values_len}"
    )]
    InvalidHostInputShape {
        frame_count: usize,
        encoder_d_model: usize,
        values_len: usize,
    },
    #[error(
        "moss-transcribe-diarize graph adaptor input shape is invalid: frame_count={frame_count} encoder_d_model={encoder_d_model} merge_size={merge_size} input_dim={input_dim} llm_dim={llm_dim}"
    )]
    InvalidGraphInputShape {
        frame_count: usize,
        encoder_d_model: usize,
        merge_size: usize,
        input_dim: usize,
        llm_dim: usize,
    },
    #[error("moss-transcribe-diarize adaptor shape overflowed")]
    ShapeOverflow,
    #[error("moss-transcribe-diarize adaptor graph failed at '{step}': {source}")]
    GraphBuild {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("moss-transcribe-diarize adaptor output contains non-finite values")]
    NonFiniteValues,
}

/// The original scalar implementation is the CPU numerical oracle. It stays
/// CPU-only because the equivalent ggml reduction order can move a delicate
/// decoder boundary even when the graph itself also runs on CPU. Accelerated
/// routes use [`MossAdaptorGraphWeights`] instead and therefore never retain
/// these f32 copies.
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

impl MossAdaptorWeights {
    pub(crate) fn system_memory_quote(
        tensor_index: &crate::GgufTensorIndex,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        let mut quote = crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder::new::<
            Self,
        >(pack_content_id);
        for name in [
            ADAPTOR_LINEAR1_WEIGHT,
            ADAPTOR_LINEAR1_BIAS,
            ADAPTOR_LINEAR2_WEIGHT,
            ADAPTOR_LINEAR2_BIAS,
            ADAPTOR_NORM_WEIGHT,
            ADAPTOR_NORM_BIAS,
        ] {
            quote.add_tensor_f32(tensor_index, name)?;
        }
        let mut quote = quote.finish()?;
        quote.resource_id = format!("moss-td-cpu-adaptor:{pack_content_id}");
        Ok(quote)
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        for (values, label) in [
            (&self.linear1_weight, "moss adaptor linear1 weight"),
            (&self.linear1_bias, "moss adaptor linear1 bias"),
            (&self.linear2_weight, "moss adaptor linear2 weight"),
            (&self.linear2_bias, "moss adaptor linear2 bias"),
            (&self.norm_weight, "moss adaptor norm weight"),
            (&self.norm_bias, "moss adaptor norm bias"),
        ] {
            bytes.add_vec(values, label)?;
        }
        Ok(bytes.finish())
    }
}

pub(crate) fn load_moss_adaptor_weights_from_reader(
    reader: &GgufTensorDataReader,
    encoder_d_model: usize,
    merge_size: usize,
    llm_dim: usize,
    norm_epsilon: f32,
) -> Result<MossAdaptorWeights, MossAdaptorError> {
    let stacked_input_width = encoder_d_model
        .checked_mul(merge_size)
        .ok_or(MossAdaptorError::ShapeOverflow)?;
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

/// CPU numerical oracle. `encoder_rows` is frame-major
/// `[frame][encoder_d_model]` and already merge-size aligned.
pub(crate) fn run_moss_adaptor(
    weights: &MossAdaptorWeights,
    encoder_rows: &[f32],
    frame_count: usize,
    encoder_d_model: usize,
    merge_size: usize,
) -> Result<(Vec<f32>, usize), MossAdaptorError> {
    let expected_len = frame_count.checked_mul(encoder_d_model).ok_or(
        MossAdaptorError::InvalidHostInputShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        },
    )?;
    if encoder_rows.len() != expected_len
        || merge_size == 0
        || !frame_count.is_multiple_of(merge_size)
        || encoder_d_model.checked_mul(merge_size) != Some(weights.stacked_input_width)
    {
        return Err(MossAdaptorError::InvalidHostInputShape {
            frame_count,
            encoder_d_model,
            values_len: encoder_rows.len(),
        });
    }
    if encoder_rows.iter().any(|value| !value.is_finite()) {
        return Err(MossAdaptorError::NonFiniteValues);
    }

    let output_token_count = frame_count / merge_size;
    let stacked_width = weights.stacked_input_width;
    let llm_dim = weights.llm_dim;
    let output_capacity = output_token_count
        .checked_mul(llm_dim)
        .ok_or(MossAdaptorError::ShapeOverflow)?;
    let mut output = Vec::with_capacity(output_capacity);
    let mut hidden_row = vec![0.0_f32; llm_dim];
    let mut linear2_row = vec![0.0_f32; llm_dim];
    for stacked_row in encoder_rows.chunks_exact(stacked_width) {
        matmul_row_output_by_input(
            stacked_row,
            &weights.linear1_weight,
            &weights.linear1_bias,
            stacked_width,
            &mut hidden_row,
        );
        for value in &mut hidden_row {
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
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / llm_dim as f32;
        let inv_std = 1.0 / (variance + weights.norm_epsilon).sqrt();
        for (index, value) in linear2_row.iter().enumerate() {
            output.push(
                (value - mean) * inv_std * weights.norm_weight[index] + weights.norm_bias[index],
            );
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(MossAdaptorError::NonFiniteValues);
    }
    Ok((output, output_token_count))
}

fn matmul_row_output_by_input(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    input_width: usize,
    out: &mut [f32],
) {
    for (out_index, out_value) in out.iter_mut().enumerate() {
        let row = &weight[out_index * input_width..out_index * input_width + input_width];
        let mut accumulator = 0.0_f32;
        for (input_value, weight_value) in input.iter().zip(row) {
            accumulator += input_value * weight_value;
        }
        *out_value = accumulator + bias[out_index];
    }
}

fn map_graph_error(step: &'static str, source: GgmlCpuGraphError) -> MossAdaptorError {
    MossAdaptorError::GraphBuild { step, source }
}

fn tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &'static str,
) -> Result<GgmlLoadedTensor, MossAdaptorError> {
    loaded
        .tensor(name)
        .ok_or(MossAdaptorError::MissingTensor { name })
}

/// Loaded handles for the six adaptor tensors. The owning encoder runtime keeps
/// the corresponding [`GgmlLoadedWeightContext`] alive for the handles' entire
/// lifetime.
pub(crate) struct MossAdaptorGraphWeights {
    linear1_weight: GgmlLoadedTensor,
    linear1_bias: GgmlLoadedTensor,
    linear2_weight: GgmlLoadedTensor,
    linear2_bias: GgmlLoadedTensor,
    norm_weight: GgmlLoadedTensor,
    norm_bias: GgmlLoadedTensor,
}

impl MossAdaptorGraphWeights {
    pub(crate) fn from_loaded(loaded: &GgmlLoadedWeightContext) -> Result<Self, MossAdaptorError> {
        Ok(Self {
            linear1_weight: tensor(loaded, ADAPTOR_LINEAR1_WEIGHT)?,
            linear1_bias: tensor(loaded, ADAPTOR_LINEAR1_BIAS)?,
            linear2_weight: tensor(loaded, ADAPTOR_LINEAR2_WEIGHT)?,
            linear2_bias: tensor(loaded, ADAPTOR_LINEAR2_BIAS)?,
            norm_weight: tensor(loaded, ADAPTOR_NORM_WEIGHT)?,
            norm_bias: tensor(loaded, ADAPTOR_NORM_BIAS)?,
        })
    }

    fn graph_tensors<'a>(&self) -> MossAdaptorGraphTensors<'a> {
        MossAdaptorGraphTensors {
            linear1_weight: self.linear1_weight.as_graph_tensor(),
            linear1_bias: self.linear1_bias.as_graph_tensor(),
            linear2_weight: self.linear2_weight.as_graph_tensor(),
            linear2_bias: self.linear2_bias.as_graph_tensor(),
            norm_weight: self.norm_weight.as_graph_tensor(),
            norm_bias: self.norm_bias.as_graph_tensor(),
        }
    }

    /// Appends the adaptor to an encoder state shaped
    /// `[encoder_d_model, encoder_output_frames]`. Only the valid, merge-size-
    /// aligned prefix is viewed, so padded frames never enter the adaptor.
    pub(crate) fn apply<'a>(
        &self,
        graph: &GgmlCpuGraphBuilder<'a>,
        encoder_state: GgmlCpuTensor<'a>,
        frame_count: usize,
        encoder_d_model: usize,
        merge_size: usize,
        input_dim: usize,
        llm_dim: usize,
        norm_epsilon: f32,
    ) -> Result<GgmlCpuTensor<'a>, MossAdaptorError> {
        apply_moss_adaptor_with_tensors(
            graph,
            encoder_state,
            self.graph_tensors(),
            frame_count,
            encoder_d_model,
            merge_size,
            input_dim,
            llm_dim,
            norm_epsilon,
        )
    }
}

#[derive(Clone, Copy)]
struct MossAdaptorGraphTensors<'a> {
    linear1_weight: GgmlCpuTensor<'a>,
    linear1_bias: GgmlCpuTensor<'a>,
    linear2_weight: GgmlCpuTensor<'a>,
    linear2_bias: GgmlCpuTensor<'a>,
    norm_weight: GgmlCpuTensor<'a>,
    norm_bias: GgmlCpuTensor<'a>,
}

#[allow(clippy::too_many_arguments)]
fn apply_moss_adaptor_with_tensors<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    encoder_state: GgmlCpuTensor<'a>,
    weights: MossAdaptorGraphTensors<'a>,
    frame_count: usize,
    encoder_d_model: usize,
    merge_size: usize,
    input_dim: usize,
    llm_dim: usize,
    norm_epsilon: f32,
) -> Result<GgmlCpuTensor<'a>, MossAdaptorError> {
    let expected_input_dim = encoder_d_model
        .checked_mul(merge_size)
        .ok_or(MossAdaptorError::ShapeOverflow)?;
    if frame_count == 0
        || merge_size == 0
        || !frame_count.is_multiple_of(merge_size)
        || input_dim != expected_input_dim
        || llm_dim == 0
    {
        return Err(MossAdaptorError::InvalidGraphInputShape {
            frame_count,
            encoder_d_model,
            merge_size,
            input_dim,
            llm_dim,
        });
    }

    let row_stride = encoder_d_model
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(MossAdaptorError::ShapeOverflow)?;
    let valid_state = graph
        .view_2d(encoder_state, encoder_d_model, frame_count, row_stride, 0)
        .map_err(|source| map_graph_error("ggml_view_2d(adaptor_valid_frames)", source))?;
    let token_count = frame_count / merge_size;
    let stacked = graph
        .reshape_2d(valid_state, input_dim, token_count)
        .map_err(|source| map_graph_error("ggml_reshape_2d(adaptor_time_merge)", source))?;

    let mut hidden = graph
        .mul_mat(weights.linear1_weight, stacked)
        .map_err(|source| map_graph_error("ggml_mul_mat(adaptor_linear1)", source))?;
    hidden = graph
        .add(hidden, weights.linear1_bias)
        .map_err(|source| map_graph_error("ggml_add(adaptor_linear1_bias)", source))?;
    hidden = graph
        .silu(hidden)
        .map_err(|source| map_graph_error("ggml_silu(adaptor)", source))?;

    let mut output = graph
        .mul_mat(weights.linear2_weight, hidden)
        .map_err(|source| map_graph_error("ggml_mul_mat(adaptor_linear2)", source))?;
    output = graph
        .add(output, weights.linear2_bias)
        .map_err(|source| map_graph_error("ggml_add(adaptor_linear2_bias)", source))?;
    apply_affine_layer_norm(
        graph,
        output,
        norm_epsilon,
        weights.norm_weight,
        weights.norm_bias,
        AffineLayerNormSteps {
            norm: "ggml_norm(adaptor)",
            scale: "ggml_mul(adaptor_norm_weight)",
            bias: "ggml_add(adaptor_norm_bias)",
        },
        map_graph_error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    fn toy_weights() -> MossAdaptorWeights {
        // encoder_d_model=2, merge_size=2 -> input_dim=4, llm_dim=3.
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
            norm_epsilon: 1.0e-6,
        }
    }

    #[test]
    fn adaptor_graph_matches_scalar_family_math() {
        let weights = toy_weights();
        let encoder_rows = [1.0_f32, 2.0, 3.0, 4.0];
        let (expected, expected_tokens) =
            run_moss_adaptor(&weights, &encoder_rows, 2, 2, 2).expect("host adaptor");
        assert_eq!(expected_tokens, 1);

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("runner");
        let mut graph = runner.start_graph();
        let encoder = graph
            .new_tensor_2d_f32(2, 2, "moss_adaptor_test_encoder")
            .expect("encoder tensor");
        let linear1_weight = graph
            .new_tensor_2d_f32(4, 3, "moss_adaptor_test_linear1_weight")
            .expect("linear1 weight");
        let linear1_bias = graph
            .new_tensor_1d_f32(3, "moss_adaptor_test_linear1_bias")
            .expect("linear1 bias");
        let linear2_weight = graph
            .new_tensor_2d_f32(3, 3, "moss_adaptor_test_linear2_weight")
            .expect("linear2 weight");
        let linear2_bias = graph
            .new_tensor_1d_f32(3, "moss_adaptor_test_linear2_bias")
            .expect("linear2 bias");
        let norm_weight = graph
            .new_tensor_1d_f32(3, "moss_adaptor_test_norm_weight")
            .expect("norm weight");
        let norm_bias = graph
            .new_tensor_1d_f32(3, "moss_adaptor_test_norm_bias")
            .expect("norm bias");
        for input in [
            encoder,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            norm_weight,
            norm_bias,
        ] {
            graph.set_input(input).expect("set input");
        }
        let output = apply_moss_adaptor_with_tensors(
            &graph,
            encoder,
            MossAdaptorGraphTensors {
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
                norm_weight,
                norm_bias,
            },
            2,
            2,
            2,
            4,
            3,
            1.0e-6,
        )
        .expect("adaptor graph");
        graph.set_output(output).expect("set output");
        graph
            .prepare_outputs_for_upload(&[output])
            .expect("prepare");
        for (tensor, values, name) in [
            (encoder, encoder_rows.as_slice(), "encoder"),
            (
                linear1_weight,
                weights.linear1_weight.as_slice(),
                "linear1_weight",
            ),
            (
                linear1_bias,
                weights.linear1_bias.as_slice(),
                "linear1_bias",
            ),
            (
                linear2_weight,
                weights.linear2_weight.as_slice(),
                "linear2_weight",
            ),
            (
                linear2_bias,
                weights.linear2_bias.as_slice(),
                "linear2_bias",
            ),
            (norm_weight, weights.norm_weight.as_slice(), "norm_weight"),
            (norm_bias, weights.norm_bias.as_slice(), "norm_bias"),
        ] {
            graph.set_f32_slice(tensor, values, name).expect("upload");
        }
        let actual = graph.compute_output_f32(output, 3).expect("compute");
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "adaptor output[{idx}] {actual} != {expected}"
            );
        }
    }

    #[test]
    fn adaptor_rejects_frame_count_not_multiple_of_merge_size() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("runner");
        let graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 3, "moss_adaptor_invalid_input")
            .expect("input");
        let weights = MossAdaptorGraphTensors {
            linear1_weight: input,
            linear1_bias: input,
            linear2_weight: input,
            linear2_bias: input,
            norm_weight: input,
            norm_bias: input,
        };
        let error = apply_moss_adaptor_with_tensors(&graph, input, weights, 3, 2, 2, 4, 3, 1.0e-6)
            .expect_err("must fail");
        assert!(matches!(
            error,
            MossAdaptorError::InvalidGraphInputShape { .. }
        ));
    }

    #[test]
    fn host_adaptor_rejects_frame_count_not_multiple_of_merge_size() {
        let error = run_moss_adaptor(&toy_weights(), &[0.0; 6], 3, 2, 2).expect_err("must fail");
        assert!(matches!(
            error,
            MossAdaptorError::InvalidHostInputShape { .. }
        ));
    }
}
