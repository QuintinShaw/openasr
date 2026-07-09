use crate::ggml_runtime::GGML_TYPE_F32;
use crate::ggml_runtime::GgufTensorDataReader;

use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tensor_names::{
    ENC_PRE_OUT_BIAS, ENC_PRE_OUT_WEIGHT, ENC_PROJ_BIAS, ENC_PROJ_WEIGHT, enc_pre_conv_bias,
    enc_pre_conv_weight, encoder_layer_tensor_names,
};
use super::weights::{
    CohereMatrixWeight, CohereTensorWeight, CohereVectorWeight, CohereWeightLoadError,
    load_matrix_weight, load_matrix_weight_for_runtime,
    load_tensor_weight_with_rank_for_runtime_expected_type,
    load_tensor_weight_with_required_dims_and_ranks,
    load_tensor_weight_with_required_dims_and_ranks_for_runtime,
    load_tensor_weight_with_required_dims_and_ranks_for_runtime_expected_type, load_vector_weight,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereEncoderLayerWeights {
    pub ff1_norm_weight: CohereVectorWeight,
    pub ff1_norm_bias: CohereVectorWeight,
    pub ff1_up_weight: CohereMatrixWeight,
    pub ff1_up_bias: CohereVectorWeight,
    pub ff1_down_weight: CohereMatrixWeight,
    pub ff1_down_bias: CohereVectorWeight,
    pub attn_norm_weight: CohereVectorWeight,
    pub attn_norm_bias: CohereVectorWeight,
    pub attn_q_weight: CohereMatrixWeight,
    pub attn_q_bias: CohereVectorWeight,
    pub attn_k_weight: CohereMatrixWeight,
    pub attn_k_bias: CohereVectorWeight,
    pub attn_v_weight: CohereMatrixWeight,
    pub attn_v_bias: CohereVectorWeight,
    pub attn_out_weight: CohereMatrixWeight,
    pub attn_out_bias: CohereVectorWeight,
    pub attn_pos_weight: CohereMatrixWeight,
    pub attn_pos_bias_u: CohereMatrixWeight,
    pub attn_pos_bias_v: CohereMatrixWeight,
    pub conv_norm_weight: CohereVectorWeight,
    pub conv_norm_bias: CohereVectorWeight,
    pub conv_pw1_weight: CohereTensorWeight,
    pub conv_pw1_bias: CohereVectorWeight,
    pub conv_dw_weight: CohereTensorWeight,
    pub conv_dw_bias: CohereVectorWeight,
    pub conv_bn_weight: CohereVectorWeight,
    pub conv_bn_bias: CohereVectorWeight,
    pub conv_bn_mean: CohereVectorWeight,
    pub conv_bn_var: CohereVectorWeight,
    pub conv_pw2_weight: CohereTensorWeight,
    pub conv_pw2_bias: CohereVectorWeight,
    pub ff2_norm_weight: CohereVectorWeight,
    pub ff2_norm_bias: CohereVectorWeight,
    pub ff2_up_weight: CohereMatrixWeight,
    pub ff2_up_bias: CohereVectorWeight,
    pub ff2_down_weight: CohereMatrixWeight,
    pub ff2_down_bias: CohereVectorWeight,
    pub out_norm_weight: CohereVectorWeight,
    pub out_norm_bias: CohereVectorWeight,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereTranscribeEncoderWeights {
    pub pre_conv0_weight: CohereTensorWeight,
    pub pre_conv0_bias: CohereVectorWeight,
    pub pre_conv2_weight: CohereTensorWeight,
    pub pre_conv2_bias: CohereVectorWeight,
    pub pre_conv3_weight: CohereTensorWeight,
    pub pre_conv3_bias: CohereVectorWeight,
    pub pre_conv5_weight: CohereTensorWeight,
    pub pre_conv5_bias: CohereVectorWeight,
    pub pre_conv6_weight: CohereTensorWeight,
    pub pre_conv6_bias: CohereVectorWeight,
    pub pre_out_weight: CohereMatrixWeight,
    pub pre_out_bias: CohereVectorWeight,
    pub encoder_projection_weight: CohereMatrixWeight,
    pub encoder_projection_bias: CohereVectorWeight,
    pub layers: Vec<CohereEncoderLayerWeights>,
}

pub(crate) type CohereEncoderWeightsError = CohereWeightLoadError;
const GGML_TYPE_F16: i32 = 1;
const COHERE_ENCODER_CONV_BN_EPSILON: f32 = 1.0e-5;

pub(crate) fn load_cohere_transcribe_encoder_weights_from_reader(
    reader: &GgufTensorDataReader,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<CohereTranscribeEncoderWeights, CohereEncoderWeightsError> {
    let enc_d_model = metadata.encoder_d_model;
    let dec_d_model = metadata.decoder_d_model;
    let enc_heads = metadata.encoder_heads;
    let enc_head_dim = metadata.encoder_head_dim;
    let enc_ffn_dim = metadata.encoder_ffn_dim;
    let conv_kernel = metadata.encoder_conv_kernel;

    let pre_conv0_weight_name = enc_pre_conv_weight(0);
    let pre_conv0_bias_name = enc_pre_conv_bias(0);
    let pre_conv2_weight_name = enc_pre_conv_weight(2);
    let pre_conv2_bias_name = enc_pre_conv_bias(2);
    let pre_conv3_weight_name = enc_pre_conv_weight(3);
    let pre_conv3_bias_name = enc_pre_conv_bias(3);
    let pre_conv5_weight_name = enc_pre_conv_weight(5);
    let pre_conv5_bias_name = enc_pre_conv_bias(5);
    let pre_conv6_weight_name = enc_pre_conv_weight(6);
    let pre_conv6_bias_name = enc_pre_conv_bias(6);

    let pre_conv0_weight = load_tensor_weight_with_rank_for_runtime_expected_type(
        reader,
        &pre_conv0_weight_name,
        4,
        GGML_TYPE_F32,
    )?;
    let pre_conv0_bias = load_vector_weight(
        reader,
        &pre_conv0_bias_name,
        infer_conv_bias_len(&pre_conv0_weight)?,
    )?;
    let pre_conv2_weight = load_tensor_weight_with_rank_for_runtime_expected_type(
        reader,
        &pre_conv2_weight_name,
        4,
        GGML_TYPE_F16,
    )?;
    let pre_conv2_bias = load_vector_weight(
        reader,
        &pre_conv2_bias_name,
        infer_conv_bias_len(&pre_conv2_weight)?,
    )?;
    let pre_conv3_weight = normalize_pointwise_preconv_weight(
        load_tensor_weight_with_required_dims_and_ranks_for_runtime_expected_type(
            reader,
            &pre_conv3_weight_name,
            &[2, 4],
            &[256],
            GGML_TYPE_F32,
        )?,
    );
    let pre_conv3_bias = load_vector_weight(
        reader,
        &pre_conv3_bias_name,
        infer_conv_bias_len(&pre_conv3_weight)?,
    )?;
    let pre_conv5_weight = load_tensor_weight_with_rank_for_runtime_expected_type(
        reader,
        &pre_conv5_weight_name,
        4,
        GGML_TYPE_F16,
    )?;
    let pre_conv5_bias = load_vector_weight(
        reader,
        &pre_conv5_bias_name,
        infer_conv_bias_len(&pre_conv5_weight)?,
    )?;
    let pre_conv6_weight = normalize_pointwise_preconv_weight(
        load_tensor_weight_with_required_dims_and_ranks_for_runtime_expected_type(
            reader,
            &pre_conv6_weight_name,
            &[2, 4],
            &[256],
            GGML_TYPE_F32,
        )?,
    );
    let pre_conv6_bias = load_vector_weight(
        reader,
        &pre_conv6_bias_name,
        infer_conv_bias_len(&pre_conv6_weight)?,
    )?;

    let pre_out_weight =
        load_matrix_weight_with_required_dim(reader, ENC_PRE_OUT_WEIGHT, enc_d_model)?;
    let pre_out_bias = load_vector_weight(reader, ENC_PRE_OUT_BIAS, enc_d_model)?;
    let encoder_projection_weight =
        load_matrix_weight_for_runtime(reader, ENC_PROJ_WEIGHT, dec_d_model, enc_d_model)?;
    let encoder_projection_bias = load_vector_weight(reader, ENC_PROJ_BIAS, dec_d_model)?;

    let mut layers = Vec::with_capacity(metadata.encoder_layers);
    for layer_idx in 0..metadata.encoder_layers {
        let names = encoder_layer_tensor_names(layer_idx);
        let mut layer = CohereEncoderLayerWeights {
            ff1_norm_weight: load_vector_weight(reader, &names.ff1_norm_weight, enc_d_model)?,
            ff1_norm_bias: load_vector_weight(reader, &names.ff1_norm_bias, enc_d_model)?,
            ff1_up_weight: load_matrix_weight_for_runtime(
                reader,
                &names.ff1_up_weight,
                enc_ffn_dim,
                enc_d_model,
            )?,
            ff1_up_bias: load_vector_weight(reader, &names.ff1_up_bias, enc_ffn_dim)?,
            ff1_down_weight: load_matrix_weight_for_runtime(
                reader,
                &names.ff1_down_weight,
                enc_d_model,
                enc_ffn_dim,
            )?,
            ff1_down_bias: load_vector_weight(reader, &names.ff1_down_bias, enc_d_model)?,
            attn_norm_weight: load_vector_weight(reader, &names.attn_norm_weight, enc_d_model)?,
            attn_norm_bias: load_vector_weight(reader, &names.attn_norm_bias, enc_d_model)?,
            attn_q_weight: load_matrix_weight_for_runtime(
                reader,
                &names.attn_q_weight,
                enc_d_model,
                enc_d_model,
            )?,
            attn_q_bias: load_vector_weight(reader, &names.attn_q_bias, enc_d_model)?,
            attn_k_weight: load_matrix_weight_for_runtime(
                reader,
                &names.attn_k_weight,
                enc_d_model,
                enc_d_model,
            )?,
            attn_k_bias: load_vector_weight(reader, &names.attn_k_bias, enc_d_model)?,
            attn_v_weight: load_matrix_weight_for_runtime(
                reader,
                &names.attn_v_weight,
                enc_d_model,
                enc_d_model,
            )?,
            attn_v_bias: load_vector_weight(reader, &names.attn_v_bias, enc_d_model)?,
            attn_out_weight: load_matrix_weight_for_runtime(
                reader,
                &names.attn_out_weight,
                enc_d_model,
                enc_d_model,
            )?,
            attn_out_bias: load_vector_weight(reader, &names.attn_out_bias, enc_d_model)?,
            attn_pos_weight: load_matrix_weight_for_runtime(
                reader,
                &names.attn_pos_weight,
                enc_d_model,
                enc_d_model,
            )?,
            attn_pos_bias_u: load_matrix_weight(
                reader,
                &names.attn_pos_bias_u,
                enc_heads,
                enc_head_dim,
            )?,
            attn_pos_bias_v: load_matrix_weight(
                reader,
                &names.attn_pos_bias_v,
                enc_heads,
                enc_head_dim,
            )?,
            conv_norm_weight: load_vector_weight(reader, &names.conv_norm_weight, enc_d_model)?,
            conv_norm_bias: load_vector_weight(reader, &names.conv_norm_bias, enc_d_model)?,
            conv_pw1_weight: load_tensor_weight_with_required_dims_and_ranks_for_runtime(
                reader,
                &names.conv_pw1_weight,
                &[2, 3],
                &[enc_d_model * 2, enc_d_model],
            )?,
            conv_pw1_bias: load_vector_weight(reader, &names.conv_pw1_bias, enc_d_model * 2)?,
            conv_dw_weight: load_tensor_weight_with_required_dims_and_ranks(
                reader,
                &names.conv_dw_weight,
                &[2, 3],
                &[enc_d_model, conv_kernel],
            )?,
            conv_dw_bias: load_vector_weight(reader, &names.conv_dw_bias, enc_d_model)?,
            conv_bn_weight: load_vector_weight(reader, &names.conv_bn_weight, enc_d_model)?,
            conv_bn_bias: load_vector_weight(reader, &names.conv_bn_bias, enc_d_model)?,
            conv_bn_mean: load_vector_weight(reader, &names.conv_bn_mean, enc_d_model)?,
            conv_bn_var: load_vector_weight(reader, &names.conv_bn_var, enc_d_model)?,
            conv_pw2_weight: load_tensor_weight_with_required_dims_and_ranks_for_runtime(
                reader,
                &names.conv_pw2_weight,
                &[2, 3],
                &[enc_d_model, enc_d_model],
            )?,
            conv_pw2_bias: load_vector_weight(reader, &names.conv_pw2_bias, enc_d_model)?,
            ff2_norm_weight: load_vector_weight(reader, &names.ff2_norm_weight, enc_d_model)?,
            ff2_norm_bias: load_vector_weight(reader, &names.ff2_norm_bias, enc_d_model)?,
            ff2_up_weight: load_matrix_weight_for_runtime(
                reader,
                &names.ff2_up_weight,
                enc_ffn_dim,
                enc_d_model,
            )?,
            ff2_up_bias: load_vector_weight(reader, &names.ff2_up_bias, enc_ffn_dim)?,
            ff2_down_weight: load_matrix_weight_for_runtime(
                reader,
                &names.ff2_down_weight,
                enc_d_model,
                enc_ffn_dim,
            )?,
            ff2_down_bias: load_vector_weight(reader, &names.ff2_down_bias, enc_d_model)?,
            out_norm_weight: load_vector_weight(reader, &names.out_norm_weight, enc_d_model)?,
            out_norm_bias: load_vector_weight(reader, &names.out_norm_bias, enc_d_model)?,
        };
        if std::env::var_os("OPENASR_COHERE_DISABLE_BN_FOLD").is_none() {
            fold_conformer_conv_batchnorm_into_depthwise(&mut layer, enc_d_model)?;
        }
        if layer_idx == 0 && std::env::var_os("OPENASR_COHERE_DEBUG_CONV_DIMS").is_some() {
            eprintln!(
                "openasr cohere conv dims: pw1={:?} dw={:?} pw2={:?}",
                layer.conv_pw1_weight.dims, layer.conv_dw_weight.dims, layer.conv_pw2_weight.dims
            );
        }
        layers.push(layer);
    }

    Ok(CohereTranscribeEncoderWeights {
        pre_conv0_weight,
        pre_conv0_bias,
        pre_conv2_weight,
        pre_conv2_bias,
        pre_conv3_weight,
        pre_conv3_bias,
        pre_conv5_weight,
        pre_conv5_bias,
        pre_conv6_weight,
        pre_conv6_bias,
        pre_out_weight,
        pre_out_bias,
        encoder_projection_weight,
        encoder_projection_bias,
        layers,
    })
}

fn fold_conformer_conv_batchnorm_into_depthwise(
    layer: &mut CohereEncoderLayerWeights,
    channel_count: usize,
) -> Result<(), CohereEncoderWeightsError> {
    if layer.conv_dw_bias.values.len() != channel_count
        || layer.conv_bn_weight.values.len() != channel_count
        || layer.conv_bn_bias.values.len() != channel_count
        || layer.conv_bn_mean.values.len() != channel_count
        || layer.conv_bn_var.values.len() != channel_count
    {
        return Err(CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: layer.conv_dw_weight.name.clone(),
            shape: format!(
                "conv_dw_bias={} conv_bn_weight={} conv_bn_bias={} conv_bn_mean={} conv_bn_var={} expected={channel_count}",
                layer.conv_dw_bias.values.len(),
                layer.conv_bn_weight.values.len(),
                layer.conv_bn_bias.values.len(),
                layer.conv_bn_mean.values.len(),
                layer.conv_bn_var.values.len()
            ),
            reason:
                "conformer conv batchnorm fold requires per-channel tensors with matching width"
                    .to_string(),
        });
    }
    let channel_axis = find_channel_axis(&layer.conv_dw_weight.dims, channel_count).ok_or_else(|| {
        CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: layer.conv_dw_weight.name.clone(),
            shape: format!("{:?}", layer.conv_dw_weight.dims),
            reason: format!(
                "conformer conv batchnorm fold expected exactly one axis with channel width {channel_count}"
            ),
        }
    })?;
    let per_channel_scale = (0..channel_count)
        .map(|channel| {
            let var = layer.conv_bn_var.values[channel];
            let gamma = layer.conv_bn_weight.values[channel];
            gamma / (var + COHERE_ENCODER_CONV_BN_EPSILON).sqrt()
        })
        .collect::<Vec<_>>();
    if std::env::var_os("OPENASR_COHERE_DEBUG_BN_FOLD").is_some() {
        let mut scale_min = f32::INFINITY;
        let mut scale_max = f32::NEG_INFINITY;
        for value in &per_channel_scale {
            scale_min = scale_min.min(*value);
            scale_max = scale_max.max(*value);
        }
        eprintln!(
            "openasr cohere bn-fold: tensor='{}' dims={:?} channel_axis={} scale_min={:.6} scale_max={:.6}",
            layer.conv_dw_weight.name,
            layer.conv_dw_weight.dims,
            channel_axis,
            scale_min,
            scale_max
        );
    }
    for (index, value) in layer.conv_dw_weight.values.iter_mut().enumerate() {
        let channel = index_to_axis_coordinate(index, &layer.conv_dw_weight.dims, channel_axis)?;
        *value *= per_channel_scale[channel];
    }
    // The folded values no longer match the original GGUF payload bytes.
    // Force downstream uploads to use the updated f32 values instead of
    // reusing the stale raw tensor payload.
    layer.conv_dw_weight.raw_ggml = None;
    for (channel, bias_value) in layer
        .conv_dw_bias
        .values
        .iter_mut()
        .enumerate()
        .take(channel_count)
    {
        let bias = *bias_value;
        let mean = layer.conv_bn_mean.values[channel];
        let beta = layer.conv_bn_bias.values[channel];
        let scale = per_channel_scale[channel];
        *bias_value = (bias - mean) * scale + beta;
    }
    Ok(())
}

fn find_channel_axis(dims: &[usize], channel_count: usize) -> Option<usize> {
    let mut axis = None;
    for (idx, dim) in dims.iter().copied().enumerate() {
        if dim == channel_count {
            if axis.is_some() {
                return None;
            }
            axis = Some(idx);
        }
    }
    axis
}

fn index_to_axis_coordinate(
    flat_index: usize,
    dims: &[usize],
    axis: usize,
) -> Result<usize, CohereEncoderWeightsError> {
    if axis >= dims.len() {
        return Err(CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: "index_to_axis_coordinate".to_string(),
            shape: format!("{dims:?}"),
            reason: format!("axis {axis} out of bounds"),
        });
    }
    let mut index = flat_index;
    for (current_axis, dim) in dims.iter().copied().enumerate() {
        if dim == 0 {
            return Err(CohereEncoderWeightsError::InvalidTensorShape {
                tensor_name: "index_to_axis_coordinate".to_string(),
                shape: format!("{dims:?}"),
                reason: "zero-sized dimensions are unsupported".to_string(),
            });
        }
        let coordinate = index % dim;
        if current_axis == axis {
            return Ok(coordinate);
        }
        index /= dim;
    }
    Err(CohereEncoderWeightsError::InvalidTensorShape {
        tensor_name: "index_to_axis_coordinate".to_string(),
        shape: format!("{dims:?}"),
        reason: format!("flat index {flat_index} overflows shape"),
    })
}

fn infer_conv_bias_len(weight: &CohereTensorWeight) -> Result<usize, CohereEncoderWeightsError> {
    weight
        .dims
        .iter()
        .copied()
        .max()
        .ok_or_else(|| CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: weight.name.clone(),
            shape: "[]".to_string(),
            reason: "rank-4 conv tensor cannot be empty".to_string(),
        })
}

fn normalize_pointwise_preconv_weight(mut weight: CohereTensorWeight) -> CohereTensorWeight {
    if weight.dims.len() == 2 {
        weight.dims.extend([1, 1]);
        if let Some(raw) = &mut weight.raw_ggml
            && raw.dims.len() == 2
        {
            raw.dims.extend([1, 1]);
        }
    }
    weight
}

fn load_matrix_weight_with_required_dim(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    required_dim: usize,
) -> Result<CohereMatrixWeight, CohereEncoderWeightsError> {
    let tensor = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    if tensor.dims.len() != 2 {
        return Err(CohereEncoderWeightsError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: super::weights::render_shape_u64(&tensor.dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let dim0 = tensor.dims[0] as usize;
    let dim1 = tensor.dims[1] as usize;
    if dim0 == required_dim {
        return load_matrix_weight_for_runtime(reader, tensor_name, dim0, dim1);
    }
    if dim1 == required_dim {
        return load_matrix_weight_for_runtime(reader, tensor_name, dim1, dim0);
    }
    Err(CohereEncoderWeightsError::InvalidTensorShape {
        tensor_name: tensor_name.to_string(),
        shape: super::weights::render_shape_u64(&tensor.dims),
        reason: format!("expected one dimension to equal {required_dim}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cohere::tensor_names::{
        ENC_PROJ_BIAS, ENC_PROJ_WEIGHT, encoder_layer_tensor_names,
    };
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::validate_ggml_runtime_source_path;
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source,
    };
    use std::sync::Arc;
    use tempfile::{NamedTempFile, TempPath};

    fn write_runtime_ready_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn loads_encoder_weights_from_runtime_ready_fixture() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let weights =
            load_cohere_transcribe_encoder_weights_from_reader(&reader, metadata).expect("weights");
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.pre_conv0_bias.values.len(), 4);
        assert_eq!(weights.pre_out_bias.values.len(), 16);
        assert_eq!(weights.encoder_projection_bias.values.len(), 16);
        assert_eq!(weights.layers[0].conv_pw1_bias.values.len(), 32);
        assert_eq!(weights.layers[1].attn_pos_bias_u.rows, 2);
        assert!(weights.layers[1].conv_dw_weight.raw_ggml.is_none());
    }

    #[test]
    fn accepts_rank2_encoder_conv_pointwise_weight() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let layer0 = encoder_layer_tensor_names(0);
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_tensor_shape(layer0.conv_pw1_weight, [32_u64, 16_u64]);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let weights =
            load_cohere_transcribe_encoder_weights_from_reader(&reader, metadata).expect("weights");
        assert_eq!(weights.layers[0].conv_pw1_weight.dims, vec![32, 16]);
    }

    #[test]
    fn loads_non_square_encoder_projection_with_decoder_output_rows() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_metadata("cohere_transcribe.decoder.d_model", "24")
            .with_metadata("cohere_transcribe.decoder.head_dim", "12")
            .with_tensor_shape(ENC_PROJ_WEIGHT, [24_u64, 16_u64])
            .with_tensor_shape(ENC_PROJ_BIAS, [24_u64]);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let weights =
            load_cohere_transcribe_encoder_weights_from_reader(&reader, metadata).expect("weights");
        assert_eq!(weights.encoder_projection_weight.rows, 24);
        assert_eq!(weights.encoder_projection_weight.cols, 16);
        assert_eq!(weights.encoder_projection_bias.values.len(), 24);
    }

    #[test]
    fn rejects_missing_encoder_tensor() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let layer1 = encoder_layer_tensor_names(1);
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .without_tensor(&layer1.attn_out_bias);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let result = load_cohere_transcribe_encoder_weights_from_reader(&reader, metadata);
        assert!(result.is_err(), "missing tensor must fail closed");
    }
}
