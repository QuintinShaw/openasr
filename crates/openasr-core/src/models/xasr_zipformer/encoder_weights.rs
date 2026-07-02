//! Structured Zipformer2 encoder weight loading for X-ASR.
//!
//! This module intentionally stops at the pack contract: it resolves the
//! semantic icefall names through the shared compaction layer and validates
//! shapes from GGUF metadata. Graph execution lives separately so name/shape
//! drift cannot hide inside operator code.

use crate::ggml_runtime::GgufTensorDataReader;

use super::runtime_contract::XasrZipformerExecutionMetadata;
use super::weights::{
    NamedTensor, StoredLinear, XasrWeightsError, load_linear, load_named, load_vector,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderWeights {
    pub embed: XasrEncoderEmbedWeights,
    pub stacks: Vec<XasrEncoderStackWeights>,
    pub downsample_output_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderEmbedWeights {
    pub conv0: XasrConv2dWeights,
    pub conv4: XasrConv2dWeights,
    pub conv7: XasrConv2dWeights,
    pub convnext_depthwise: XasrConv2dWeights,
    pub convnext_pointwise1: XasrConv2dWeights,
    pub convnext_pointwise2: XasrConv2dWeights,
    pub out: XasrLinearWithBias,
    pub out_norm_bias: Vec<f32>,
    pub out_norm_log_scale: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderStackWeights {
    pub stack: usize,
    pub dim: usize,
    pub downsampling_factor: usize,
    pub layers: Vec<XasrEncoderLayerWeights>,
    pub downsample_bias: Option<Vec<f32>>,
    pub out_combiner_bypass_scale: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrEncoderLayerWeights {
    pub feed_forward1: XasrLinearPairWeights,
    pub feed_forward2: XasrLinearPairWeights,
    pub feed_forward3: XasrLinearPairWeights,
    pub self_attn_weights: XasrSelfAttentionWeightsWeights,
    pub self_attn1: XasrLinearPairWeights,
    pub self_attn2: XasrLinearPairWeights,
    pub nonlin_attention: XasrNonlinAttentionWeights,
    pub conv_module1: XasrConvolutionModuleWeights,
    pub conv_module2: XasrConvolutionModuleWeights,
    pub norm_bias: Vec<f32>,
    pub norm_log_scale: Vec<f32>,
    pub bypass_scale: Vec<f32>,
    pub bypass_mid_scale: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrLinearWithBias {
    pub weight: StoredLinear,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrLinearPairWeights {
    pub in_proj: XasrLinearWithBias,
    pub out_proj: XasrLinearWithBias,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrSelfAttentionWeightsWeights {
    pub in_proj: XasrLinearWithBias,
    pub linear_pos: StoredLinear,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrNonlinAttentionWeights {
    pub in_proj: XasrLinearWithBias,
    pub out_proj: XasrLinearWithBias,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrConvolutionModuleWeights {
    pub in_proj: XasrLinearWithBias,
    pub depthwise_causal_conv: XasrConv1dWeights,
    pub depthwise_chunkwise_conv: XasrConv1dWeights,
    pub chunkwise_conv_scale: NamedTensor,
    pub out_proj: XasrLinearWithBias,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrConv1dWeights {
    pub weight: NamedTensor,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrConv2dWeights {
    pub weight: NamedTensor,
    pub bias: Vec<f32>,
}

pub(crate) fn load_xasr_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrEncoderWeights, XasrWeightsError> {
    let embed = load_embed_weights(reader, metadata)?;
    let mut stacks = Vec::with_capacity(metadata.num_stacks);
    for stack in 0..metadata.num_stacks {
        stacks.push(load_stack_weights(reader, metadata, stack)?);
    }
    let output_downsampling_factor = metadata.downsampling_factors.last().copied().unwrap_or(2);
    let downsample_output_bias = load_vector(
        reader,
        "encoder.downsample_output.bias",
        output_downsampling_factor,
    )?;
    Ok(XasrEncoderWeights {
        embed,
        stacks,
        downsample_output_bias,
    })
}

fn load_embed_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrEncoderEmbedWeights, XasrWeightsError> {
    let first_dim = metadata.encoder_dims[0];
    Ok(XasrEncoderEmbedWeights {
        conv0: load_conv2d(reader, "encoder_embed.conv.0", &[3, 3, 1, 8])?,
        conv4: load_conv2d(reader, "encoder_embed.conv.4", &[3, 3, 8, 32])?,
        conv7: load_conv2d(reader, "encoder_embed.conv.7", &[3, 3, 32, 128])?,
        convnext_depthwise: load_conv2d(
            reader,
            "encoder_embed.convnext.depthwise_conv",
            &[7, 7, 1, 128],
        )?,
        convnext_pointwise1: load_conv2d(
            reader,
            "encoder_embed.convnext.pointwise_conv1",
            &[1, 1, 128, 384],
        )?,
        convnext_pointwise2: load_conv2d(
            reader,
            "encoder_embed.convnext.pointwise_conv2",
            &[1, 1, 384, 128],
        )?,
        out: load_linear_with_bias(reader, "encoder_embed.out", 2432, first_dim)?,
        out_norm_bias: load_vector(reader, "encoder_embed.out_norm.bias", first_dim)?,
        out_norm_log_scale: load_vector(reader, "encoder_embed.out_norm.log_scale", 1)?,
    })
}

fn load_stack_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
) -> Result<XasrEncoderStackWeights, XasrWeightsError> {
    let dim = metadata.encoder_dims[stack];
    let mut layers = Vec::with_capacity(metadata.num_encoder_layers[stack]);
    for layer in 0..metadata.num_encoder_layers[stack] {
        layers.push(load_layer_weights(reader, metadata, stack, layer)?);
    }
    let (downsample_bias, out_combiner_bypass_scale) = if stack == 0 {
        (None, None)
    } else {
        (
            Some(load_vector(
                reader,
                &format!("encoder.encoders.{stack}.downsample.bias"),
                metadata.downsampling_factors[stack],
            )?),
            Some(load_vector(
                reader,
                &format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
                dim,
            )?),
        )
    };
    Ok(XasrEncoderStackWeights {
        stack,
        dim,
        downsampling_factor: metadata.downsampling_factors[stack],
        layers,
        downsample_bias,
        out_combiner_bypass_scale,
    })
}

fn load_layer_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
    layer: usize,
) -> Result<XasrEncoderLayerWeights, XasrWeightsError> {
    let dim = metadata.encoder_dims[stack];
    let prefix = layer_prefix(stack, layer);
    Ok(XasrEncoderLayerWeights {
        feed_forward1: load_feed_forward(reader, &prefix, "feed_forward1", dim)?,
        feed_forward2: load_feed_forward(reader, &prefix, "feed_forward2", dim)?,
        feed_forward3: load_feed_forward(reader, &prefix, "feed_forward3", dim)?,
        self_attn_weights: load_self_attention_weights(reader, metadata, &prefix, stack, dim)?,
        self_attn1: load_attention_value_projection(reader, &prefix, "self_attn1", dim)?,
        self_attn2: load_attention_value_projection(reader, &prefix, "self_attn2", dim)?,
        nonlin_attention: load_nonlin_attention(reader, &prefix, dim)?,
        conv_module1: load_convolution_module(
            reader,
            metadata,
            &prefix,
            stack,
            "conv_module1",
            dim,
        )?,
        conv_module2: load_convolution_module(
            reader,
            metadata,
            &prefix,
            stack,
            "conv_module2",
            dim,
        )?,
        norm_bias: load_vector(reader, &format!("{prefix}.norm.bias"), dim)?,
        norm_log_scale: load_vector(reader, &format!("{prefix}.norm.log_scale"), 1)?,
        bypass_scale: load_vector(reader, &format!("{prefix}.bypass.bypass_scale"), dim)?,
        bypass_mid_scale: load_vector(reader, &format!("{prefix}.bypass_mid.bypass_scale"), dim)?,
    })
}

fn load_feed_forward(
    reader: &GgufTensorDataReader,
    prefix: &str,
    name: &str,
    dim: usize,
) -> Result<XasrLinearPairWeights, XasrWeightsError> {
    let in_proj = load_dynamic_linear_with_bias(reader, &format!("{prefix}.{name}.in_proj"), dim)?;
    let hidden_dim = in_proj.weight.output_dim;
    let out_proj = load_linear_with_bias(
        reader,
        &format!("{prefix}.{name}.out_proj"),
        hidden_dim,
        dim,
    )?;
    Ok(XasrLinearPairWeights { in_proj, out_proj })
}

fn load_attention_value_projection(
    reader: &GgufTensorDataReader,
    prefix: &str,
    name: &str,
    dim: usize,
) -> Result<XasrLinearPairWeights, XasrWeightsError> {
    let in_proj = load_dynamic_linear_with_bias(reader, &format!("{prefix}.{name}.in_proj"), dim)?;
    let value_dim = in_proj.weight.output_dim;
    let out_proj =
        load_linear_with_bias(reader, &format!("{prefix}.{name}.out_proj"), value_dim, dim)?;
    Ok(XasrLinearPairWeights { in_proj, out_proj })
}

fn load_self_attention_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
    prefix: &str,
    stack: usize,
    dim: usize,
) -> Result<XasrSelfAttentionWeightsWeights, XasrWeightsError> {
    let linear_pos = load_rank2_linear_by_actual_dims(
        reader,
        &format!("{prefix}.self_attn_weights.linear_pos.weight"),
    )?;
    let query_dim = metadata.num_heads[stack] * metadata.query_head_dims[stack];
    let expected_output = 2 * query_dim + linear_pos.output_dim;
    let in_proj = load_linear_with_bias(
        reader,
        &format!("{prefix}.self_attn_weights.in_proj"),
        dim,
        expected_output,
    )?;
    Ok(XasrSelfAttentionWeightsWeights {
        in_proj,
        linear_pos,
    })
}

fn load_nonlin_attention(
    reader: &GgufTensorDataReader,
    prefix: &str,
    dim: usize,
) -> Result<XasrNonlinAttentionWeights, XasrWeightsError> {
    let in_proj =
        load_dynamic_linear_with_bias(reader, &format!("{prefix}.nonlin_attention.in_proj"), dim)?;
    let out_input_dim = in_proj.weight.output_dim / 3;
    let out_proj = load_linear_with_bias(
        reader,
        &format!("{prefix}.nonlin_attention.out_proj"),
        out_input_dim,
        dim,
    )?;
    Ok(XasrNonlinAttentionWeights { in_proj, out_proj })
}

fn load_convolution_module(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
    prefix: &str,
    stack: usize,
    name: &str,
    dim: usize,
) -> Result<XasrConvolutionModuleWeights, XasrWeightsError> {
    let kernel = metadata.cnn_module_kernels[stack];
    let causal_kernel = kernel.div_ceil(2);
    Ok(XasrConvolutionModuleWeights {
        in_proj: load_linear_with_bias(reader, &format!("{prefix}.{name}.in_proj"), dim, 2 * dim)?,
        depthwise_causal_conv: load_conv1d(
            reader,
            &format!("{prefix}.{name}.depthwise_conv.causal_conv"),
            &[causal_kernel, 1, dim],
        )?,
        depthwise_chunkwise_conv: load_conv1d(
            reader,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv"),
            &[kernel, 1, dim],
        )?,
        chunkwise_conv_scale: load_named_with_dims(
            reader,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
            &[2, dim, kernel],
        )?,
        out_proj: load_linear_with_bias(reader, &format!("{prefix}.{name}.out_proj"), dim, dim)?,
    })
}

fn layer_prefix(stack: usize, layer: usize) -> String {
    if stack == 0 {
        format!("encoder.encoders.{stack}.layers.{layer}")
    } else {
        format!("encoder.encoders.{stack}.encoder.layers.{layer}")
    }
}

fn load_linear_with_bias(
    reader: &GgufTensorDataReader,
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<XasrLinearWithBias, XasrWeightsError> {
    Ok(XasrLinearWithBias {
        weight: load_linear(reader, &format!("{prefix}.weight"), input_dim, output_dim)?,
        bias: load_vector(reader, &format!("{prefix}.bias"), output_dim)?,
    })
}

fn load_dynamic_linear_with_bias(
    reader: &GgufTensorDataReader,
    prefix: &str,
    input_dim: usize,
) -> Result<XasrLinearWithBias, XasrWeightsError> {
    let bias = load_named(reader, &format!("{prefix}.bias"))?;
    ensure_dims(&bias, &[bias.values.len()])?;
    let output_dim = bias.values.len();
    Ok(XasrLinearWithBias {
        weight: load_linear(reader, &format!("{prefix}.weight"), input_dim, output_dim)?,
        bias: bias.values,
    })
}

fn load_rank2_linear_by_actual_dims(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
) -> Result<StoredLinear, XasrWeightsError> {
    let tensor = load_named(reader, upstream_name)?;
    if tensor.dims.len() != 2 {
        return Err(XasrWeightsError::Rank {
            name: tensor.name,
            rank: tensor.dims.len(),
            expected_rank: 2,
        });
    }
    Ok(StoredLinear {
        name: tensor.name,
        input_dim: tensor.dims[0],
        output_dim: tensor.dims[1],
        values: tensor.values,
    })
}

fn load_conv1d(
    reader: &GgufTensorDataReader,
    prefix: &str,
    expected_dims: &[usize],
) -> Result<XasrConv1dWeights, XasrWeightsError> {
    Ok(XasrConv1dWeights {
        weight: load_named_with_dims(reader, &format!("{prefix}.weight"), expected_dims)?,
        bias: load_vector(reader, &format!("{prefix}.bias"), expected_dims[2])?,
    })
}

fn load_conv2d(
    reader: &GgufTensorDataReader,
    prefix: &str,
    expected_dims: &[usize],
) -> Result<XasrConv2dWeights, XasrWeightsError> {
    Ok(XasrConv2dWeights {
        weight: load_named_with_dims(reader, &format!("{prefix}.weight"), expected_dims)?,
        bias: load_vector(reader, &format!("{prefix}.bias"), expected_dims[3])?,
    })
}

fn load_named_with_dims(
    reader: &GgufTensorDataReader,
    upstream_name: &str,
    expected_dims: &[usize],
) -> Result<NamedTensor, XasrWeightsError> {
    let tensor = load_named(reader, upstream_name)?;
    ensure_dims(&tensor, expected_dims)?;
    Ok(tensor)
}

fn ensure_dims(tensor: &NamedTensor, expected_dims: &[usize]) -> Result<(), XasrWeightsError> {
    if tensor.dims == expected_dims {
        return Ok(());
    }
    Err(XasrWeightsError::Dims {
        name: tensor.name.clone(),
        dims: tensor.dims.clone(),
        expected: expected_dims.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
    use crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata;
    use std::path::Path;

    #[test]
    #[ignore = "host-local: loads the full ONNX-derived X-ASR pack"]
    fn loads_xasr_encoder_weights_when_onnx_pack_present() {
        let pack = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-fp16.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr ONNX fp16 pack absent at {}", pack.display());
            return;
        }
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("xasr metadata");
        let weights = load_xasr_encoder_weights(&reader, &metadata).expect("encoder weights");

        assert_eq!(weights.stacks.len(), 6);
        assert_eq!(
            weights.stacks.iter().map(|s| s.layers.len()).sum::<usize>(),
            19
        );
        assert_eq!(weights.embed.conv0.weight.dims, vec![3, 3, 1, 8]);
        assert_eq!(weights.embed.out.weight.input_dim, 2432);
        assert_eq!(weights.embed.out.weight.output_dim, 192);
        assert_eq!(weights.downsample_output_bias.len(), 2);

        let stack0_layer0 = &weights.stacks[0].layers[0];
        assert_eq!(weights.stacks[0].downsample_bias, None);
        assert_eq!(weights.stacks[0].out_combiner_bypass_scale, None);
        assert_eq!(stack0_layer0.feed_forward1.in_proj.weight.output_dim, 384);
        assert_eq!(stack0_layer0.feed_forward2.in_proj.weight.output_dim, 512);
        assert_eq!(stack0_layer0.feed_forward3.in_proj.weight.output_dim, 640);
        assert_eq!(
            stack0_layer0
                .conv_module1
                .depthwise_chunkwise_conv
                .weight
                .dims,
            vec![31, 1, 192]
        );
        assert_eq!(
            stack0_layer0.conv_module1.chunkwise_conv_scale.dims,
            vec![2, 192, 31]
        );
        assert_eq!(stack0_layer0.self_attn_weights.linear_pos.input_dim, 48);
        assert_eq!(stack0_layer0.self_attn_weights.linear_pos.output_dim, 16);
        assert_eq!(
            stack0_layer0.self_attn_weights.in_proj.weight.output_dim,
            272
        );

        let stack3_layer0 = &weights.stacks[3].layers[0];
        assert_eq!(weights.stacks[3].dim, 768);
        assert_eq!(weights.stacks[3].downsample_bias.as_ref().unwrap().len(), 8);
        assert!(weights.stacks[3].out_combiner_bypass_scale.is_some());
        assert_eq!(stack3_layer0.self_attn1.in_proj.weight.output_dim, 96);
        assert_eq!(stack3_layer0.self_attn_weights.linear_pos.output_dim, 32);
        assert_eq!(
            stack3_layer0.self_attn_weights.in_proj.weight.output_dim,
            544
        );
        assert_eq!(
            stack3_layer0.nonlin_attention.out_proj.weight.input_dim,
            576
        );
        assert_eq!(
            stack3_layer0.conv_module2.chunkwise_conv_scale.dims,
            vec![2, 768, 15]
        );
    }
}
