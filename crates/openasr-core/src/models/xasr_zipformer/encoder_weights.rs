//! Structured Zipformer2 encoder weight loading for X-ASR.
//!
//! This module intentionally stops at the pack contract: it resolves the
//! semantic icefall names through the shared compaction layer and validates
//! shapes from GGUF metadata. Graph execution lives separately so name/shape
//! drift cannot hide inside operator code.

use crate::ggml_runtime::GgufTensorDataReader;

use super::runtime_contract::{XasrRuntimeTensorContract, XasrZipformerExecutionMetadata};
use super::weights::{
    NamedTensor, StoredLinear, XasrWeightsError, load_named, load_native_linear,
    load_native_linear_by_actual_dims, load_native_linear_from_contract, load_vector,
    load_vector_from_contract,
};

#[derive(Debug, Clone)]
pub(crate) struct XasrEncoderWeights {
    pub embed: XasrEncoderEmbedWeights,
    pub stacks: Vec<XasrEncoderStackWeights>,
    pub downsample_output_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub(crate) struct XasrEncoderStackWeights {
    pub stack: usize,
    pub dim: usize,
    pub downsampling_factor: usize,
    pub layers: Vec<XasrEncoderLayerWeights>,
    pub downsample_bias: Option<Vec<f32>>,
    pub out_combiner_bypass_scale: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub(crate) struct XasrLinearWithBias {
    pub weight: StoredLinear,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct XasrLinearPairWeights {
    pub in_proj: XasrLinearWithBias,
    pub out_proj: XasrLinearWithBias,
}

#[derive(Debug, Clone)]
pub(crate) struct XasrSelfAttentionWeightsWeights {
    pub in_proj: XasrLinearWithBias,
    pub linear_pos: StoredLinear,
}

#[derive(Debug, Clone)]
pub(crate) struct XasrNonlinAttentionWeights {
    pub in_proj: XasrLinearWithBias,
    pub out_proj: XasrLinearWithBias,
}

#[derive(Debug, Clone)]
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

impl XasrEncoderWeights {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        add_embed_bytes(&self.embed, &mut bytes)?;
        bytes.add_vec(&self.stacks, "xasr encoder stack descriptors")?;
        for stack in &self.stacks {
            bytes.add_vec(&stack.layers, "xasr encoder layer descriptors")?;
            if let Some(values) = &stack.downsample_bias {
                bytes.add_vec(values, "xasr encoder stack downsample bias")?;
            }
            if let Some(values) = &stack.out_combiner_bypass_scale {
                bytes.add_vec(values, "xasr encoder stack combiner scale")?;
            }
            for layer in &stack.layers {
                add_layer_bytes(layer, &mut bytes)?;
            }
        }
        bytes.add_vec(
            &self.downsample_output_bias,
            "xasr encoder output downsample bias",
        )?;
        Ok(bytes.finish())
    }
}

fn add_embed_bytes(
    embed: &XasrEncoderEmbedWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
) -> Result<(), String> {
    for (label, conv) in [
        ("xasr encoder embed conv0", &embed.conv0),
        ("xasr encoder embed conv4", &embed.conv4),
        ("xasr encoder embed conv7", &embed.conv7),
        (
            "xasr encoder embed convnext depthwise",
            &embed.convnext_depthwise,
        ),
        (
            "xasr encoder embed convnext pointwise1",
            &embed.convnext_pointwise1,
        ),
        (
            "xasr encoder embed convnext pointwise2",
            &embed.convnext_pointwise2,
        ),
    ] {
        add_conv2d_bytes(conv, bytes, label)?;
    }
    add_linear_with_bias_bytes(&embed.out, bytes, "xasr encoder embed output")?;
    bytes.add_vec(&embed.out_norm_bias, "xasr encoder embed norm bias")?;
    bytes.add_vec(
        &embed.out_norm_log_scale,
        "xasr encoder embed norm log scale",
    )
}

fn add_layer_bytes(
    layer: &XasrEncoderLayerWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
) -> Result<(), String> {
    for (label, pair) in [
        ("xasr encoder feed forward1", &layer.feed_forward1),
        ("xasr encoder feed forward2", &layer.feed_forward2),
        ("xasr encoder feed forward3", &layer.feed_forward3),
        ("xasr encoder self attention1", &layer.self_attn1),
        ("xasr encoder self attention2", &layer.self_attn2),
    ] {
        add_linear_pair_bytes(pair, bytes, label)?;
    }
    add_linear_with_bias_bytes(
        &layer.self_attn_weights.in_proj,
        bytes,
        "xasr encoder attention qkv projection",
    )?;
    layer
        .self_attn_weights
        .linear_pos
        .add_retained_system_memory_bytes(bytes, "xasr encoder attention position projection")?;
    add_linear_with_bias_bytes(
        &layer.nonlin_attention.in_proj,
        bytes,
        "xasr encoder nonlinear attention input",
    )?;
    add_linear_with_bias_bytes(
        &layer.nonlin_attention.out_proj,
        bytes,
        "xasr encoder nonlinear attention output",
    )?;
    add_convolution_module_bytes(&layer.conv_module1, bytes, "xasr encoder conv module1")?;
    add_convolution_module_bytes(&layer.conv_module2, bytes, "xasr encoder conv module2")?;
    for (label, values) in [
        ("xasr encoder norm bias", &layer.norm_bias),
        ("xasr encoder norm log scale", &layer.norm_log_scale),
        ("xasr encoder bypass scale", &layer.bypass_scale),
        ("xasr encoder mid bypass scale", &layer.bypass_mid_scale),
    ] {
        bytes.add_vec(values, label)?;
    }
    Ok(())
}

fn add_linear_pair_bytes(
    pair: &XasrLinearPairWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    add_linear_with_bias_bytes(&pair.in_proj, bytes, &format!("{label} input"))?;
    add_linear_with_bias_bytes(&pair.out_proj, bytes, &format!("{label} output"))
}

fn add_linear_with_bias_bytes(
    linear: &XasrLinearWithBias,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    linear
        .weight
        .add_retained_system_memory_bytes(bytes, &format!("{label} weight"))?;
    bytes.add_vec(&linear.bias, &format!("{label} bias"))
}

fn add_convolution_module_bytes(
    module: &XasrConvolutionModuleWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    add_linear_with_bias_bytes(&module.in_proj, bytes, &format!("{label} input"))?;
    add_conv1d_bytes(
        &module.depthwise_causal_conv,
        bytes,
        &format!("{label} causal"),
    )?;
    add_conv1d_bytes(
        &module.depthwise_chunkwise_conv,
        bytes,
        &format!("{label} chunkwise"),
    )?;
    module
        .chunkwise_conv_scale
        .add_retained_system_memory_bytes(bytes, &format!("{label} chunkwise scale"))?;
    add_linear_with_bias_bytes(&module.out_proj, bytes, &format!("{label} output"))
}

fn add_conv1d_bytes(
    conv: &XasrConv1dWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    conv.weight
        .add_retained_system_memory_bytes(bytes, &format!("{label} weight"))?;
    bytes.add_vec(&conv.bias, &format!("{label} bias"))
}

fn add_conv2d_bytes(
    conv: &XasrConv2dWeights,
    bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    conv.weight
        .add_retained_system_memory_bytes(bytes, &format!("{label} weight"))?;
    bytes.add_vec(&conv.bias, &format!("{label} bias"))
}

pub(crate) fn load_xasr_encoder_weights(
    reader: &GgufTensorDataReader,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrEncoderWeights, XasrWeightsError> {
    let contract = XasrRuntimeTensorContract::for_metadata(metadata);
    load_xasr_encoder_weights_with_contract(reader, &contract, metadata)
}

fn load_xasr_encoder_weights_with_contract(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<XasrEncoderWeights, XasrWeightsError> {
    let embed = load_embed_weights(reader, contract)?;
    let mut stacks = Vec::with_capacity(metadata.num_stacks);
    for stack in 0..metadata.num_stacks {
        stacks.push(load_stack_weights(reader, contract, metadata, stack)?);
    }
    let downsample_output_bias =
        load_vector_from_contract(reader, contract, "encoder.downsample_output.bias")?;
    Ok(XasrEncoderWeights {
        embed,
        stacks,
        downsample_output_bias,
    })
}

fn load_embed_weights(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
) -> Result<XasrEncoderEmbedWeights, XasrWeightsError> {
    // The conv stem's kernel shapes and the `out` projection's input width
    // are architecture constants the contract pins exactly; the loader sources
    // them from the contract instead of duplicating the literals.
    Ok(XasrEncoderEmbedWeights {
        conv0: load_conv2d_from_contract(reader, contract, "encoder_embed.conv.0")?,
        conv4: load_conv2d_from_contract(reader, contract, "encoder_embed.conv.4")?,
        conv7: load_conv2d_from_contract(reader, contract, "encoder_embed.conv.7")?,
        convnext_depthwise: load_conv2d_from_contract(
            reader,
            contract,
            "encoder_embed.convnext.depthwise_conv",
        )?,
        convnext_pointwise1: load_conv2d_from_contract(
            reader,
            contract,
            "encoder_embed.convnext.pointwise_conv1",
        )?,
        convnext_pointwise2: load_conv2d_from_contract(
            reader,
            contract,
            "encoder_embed.convnext.pointwise_conv2",
        )?,
        out: load_linear_with_bias_from_contract(reader, contract, "encoder_embed.out")?,
        out_norm_bias: load_vector_from_contract(reader, contract, "encoder_embed.out_norm.bias")?,
        out_norm_log_scale: load_vector_from_contract(
            reader,
            contract,
            "encoder_embed.out_norm.log_scale",
        )?,
    })
}

fn load_stack_weights(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
) -> Result<XasrEncoderStackWeights, XasrWeightsError> {
    let dim = metadata.encoder_dims[stack];
    let mut layers = Vec::with_capacity(metadata.num_encoder_layers[stack]);
    for layer in 0..metadata.num_encoder_layers[stack] {
        layers.push(load_layer_weights(
            reader, contract, metadata, stack, layer,
        )?);
    }
    let (downsample_bias, out_combiner_bypass_scale) = if stack == 0 {
        (None, None)
    } else {
        (
            Some(load_vector_from_contract(
                reader,
                contract,
                &format!("encoder.encoders.{stack}.downsample.bias"),
            )?),
            Some(load_vector_from_contract(
                reader,
                contract,
                &format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
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
    contract: &XasrRuntimeTensorContract,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
    layer: usize,
) -> Result<XasrEncoderLayerWeights, XasrWeightsError> {
    let dim = metadata.encoder_dims[stack];
    let prefix = layer_prefix(stack, layer);
    Ok(XasrEncoderLayerWeights {
        feed_forward1: load_feed_forward(reader, contract, &prefix, "feed_forward1", dim)?,
        feed_forward2: load_feed_forward(reader, contract, &prefix, "feed_forward2", dim)?,
        feed_forward3: load_feed_forward(reader, contract, &prefix, "feed_forward3", dim)?,
        self_attn_weights: load_self_attention_weights(
            reader, contract, metadata, &prefix, stack, dim,
        )?,
        self_attn1: load_attention_value_projection(reader, contract, &prefix, "self_attn1", dim)?,
        self_attn2: load_attention_value_projection(reader, contract, &prefix, "self_attn2", dim)?,
        nonlin_attention: load_nonlin_attention(reader, contract, &prefix, dim)?,
        conv_module1: load_convolution_module(reader, contract, &prefix, "conv_module1")?,
        conv_module2: load_convolution_module(reader, contract, &prefix, "conv_module2")?,
        norm_bias: load_vector_from_contract(reader, contract, &format!("{prefix}.norm.bias"))?,
        norm_log_scale: load_vector_from_contract(
            reader,
            contract,
            &format!("{prefix}.norm.log_scale"),
        )?,
        bypass_scale: load_vector_from_contract(
            reader,
            contract,
            &format!("{prefix}.bypass.bypass_scale"),
        )?,
        bypass_mid_scale: load_vector_from_contract(
            reader,
            contract,
            &format!("{prefix}.bypass_mid.bypass_scale"),
        )?,
    })
}

fn load_feed_forward(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    name: &str,
    dim: usize,
) -> Result<XasrLinearPairWeights, XasrWeightsError> {
    let in_proj =
        load_dynamic_linear_with_bias(reader, contract, &format!("{prefix}.{name}.in_proj"), dim)?;
    let hidden_dim = in_proj.weight.output_dim;
    let out_proj = load_linear_with_bias(
        reader,
        contract,
        &format!("{prefix}.{name}.out_proj"),
        hidden_dim,
        dim,
    )?;
    Ok(XasrLinearPairWeights { in_proj, out_proj })
}

fn load_attention_value_projection(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    name: &str,
    dim: usize,
) -> Result<XasrLinearPairWeights, XasrWeightsError> {
    let in_proj =
        load_dynamic_linear_with_bias(reader, contract, &format!("{prefix}.{name}.in_proj"), dim)?;
    let value_dim = in_proj.weight.output_dim;
    let out_proj = load_linear_with_bias(
        reader,
        contract,
        &format!("{prefix}.{name}.out_proj"),
        value_dim,
        dim,
    )?;
    Ok(XasrLinearPairWeights { in_proj, out_proj })
}

fn load_self_attention_weights(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    metadata: &XasrZipformerExecutionMetadata,
    prefix: &str,
    stack: usize,
    dim: usize,
) -> Result<XasrSelfAttentionWeightsWeights, XasrWeightsError> {
    let linear_pos = load_native_linear_by_actual_dims(
        reader,
        contract,
        &format!("{prefix}.self_attn_weights.linear_pos.weight"),
    )?;
    // Checked arithmetic: `linear_pos.output_dim` is pack-derived, so the
    // expectation must fail closed instead of wrapping into an admitting
    // comparison (parse-time caps already bound the metadata factors).
    let expected_output = metadata.num_heads[stack]
        .checked_mul(metadata.query_head_dims[stack])
        .and_then(|query_dim| query_dim.checked_mul(2))
        .and_then(|value| value.checked_add(linear_pos.output_dim))
        .ok_or_else(|| XasrWeightsError::ExpectationOverflow {
            reason: format!("{prefix}.self_attn_weights.in_proj expected output width overflows"),
        })?;
    let in_proj = load_linear_with_bias(
        reader,
        contract,
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
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    dim: usize,
) -> Result<XasrNonlinAttentionWeights, XasrWeightsError> {
    let in_proj = load_dynamic_linear_with_bias(
        reader,
        contract,
        &format!("{prefix}.nonlin_attention.in_proj"),
        dim,
    )?;
    let out_input_dim = in_proj.weight.output_dim / 3;
    let out_proj = load_linear_with_bias(
        reader,
        contract,
        &format!("{prefix}.nonlin_attention.out_proj"),
        out_input_dim,
        dim,
    )?;
    Ok(XasrNonlinAttentionWeights { in_proj, out_proj })
}

/// Every conv-module tensor's shape is fully pinned by the metadata-derived
/// contract (kernel sizes, gate widths, scales), so the loader sources all of
/// them from the contract instead of recomputing the literals.
fn load_convolution_module(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    name: &str,
) -> Result<XasrConvolutionModuleWeights, XasrWeightsError> {
    Ok(XasrConvolutionModuleWeights {
        in_proj: load_linear_with_bias_from_contract(
            reader,
            contract,
            &format!("{prefix}.{name}.in_proj"),
        )?,
        depthwise_causal_conv: load_conv1d_from_contract(
            reader,
            contract,
            &format!("{prefix}.{name}.depthwise_conv.causal_conv"),
        )?,
        depthwise_chunkwise_conv: load_conv1d_from_contract(
            reader,
            contract,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv"),
        )?,
        chunkwise_conv_scale: load_named(
            reader,
            contract,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
        )?,
        out_proj: load_linear_with_bias_from_contract(
            reader,
            contract,
            &format!("{prefix}.{name}.out_proj"),
        )?,
    })
}

/// Per-(stack, layer) tensor-name prefix. Shared by the runtime weight loader
/// and the admission-time runtime tensor-contract validator so both resolve the
/// same pack names through one contract.
pub(super) fn layer_prefix(stack: usize, layer: usize) -> String {
    if stack == 0 {
        format!("encoder.encoders.{stack}.layers.{layer}")
    } else {
        format!("encoder.encoders.{stack}.encoder.layers.{layer}")
    }
}

fn load_linear_with_bias(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
) -> Result<XasrLinearWithBias, XasrWeightsError> {
    Ok(XasrLinearWithBias {
        weight: load_native_linear(
            reader,
            contract,
            &format!("{prefix}.weight"),
            input_dim,
            output_dim,
        )?,
        bias: load_vector(reader, contract, &format!("{prefix}.bias"), output_dim)?,
    })
}

/// Load a rank-2 projection plus its bias with both weight extents sourced
/// from the contract's `Exact` pin.
fn load_linear_with_bias_from_contract(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
) -> Result<XasrLinearWithBias, XasrWeightsError> {
    Ok(XasrLinearWithBias {
        weight: load_native_linear_from_contract(reader, contract, &format!("{prefix}.weight"))?,
        bias: load_vector_from_contract(reader, contract, &format!("{prefix}.bias"))?,
    })
}

fn load_dynamic_linear_with_bias(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
    input_dim: usize,
) -> Result<XasrLinearWithBias, XasrWeightsError> {
    let bias = load_named(reader, contract, &format!("{prefix}.bias"))?;
    ensure_dims(&bias, &[bias.values.len()])?;
    let output_dim = bias.values.len();
    Ok(XasrLinearWithBias {
        weight: load_native_linear(
            reader,
            contract,
            &format!("{prefix}.weight"),
            input_dim,
            output_dim,
        )?,
        bias: bias.values,
    })
}

/// Load a rank-1 depthwise conv kernel plus its bias; both shapes come from
/// the contract's exact pins.
fn load_conv1d_from_contract(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
) -> Result<XasrConv1dWeights, XasrWeightsError> {
    let weight = load_named(reader, contract, &format!("{prefix}.weight"))?;
    let bias = load_vector_from_contract(reader, contract, &format!("{prefix}.bias"))?;
    Ok(XasrConv1dWeights { weight, bias })
}

/// Load a conv-stem conv2d kernel plus its bias; both shapes come from the
/// contract's exact pins.
fn load_conv2d_from_contract(
    reader: &GgufTensorDataReader,
    contract: &XasrRuntimeTensorContract,
    prefix: &str,
) -> Result<XasrConv2dWeights, XasrWeightsError> {
    let weight = load_named(reader, contract, &format!("{prefix}.weight"))?;
    let bias = load_vector_from_contract(reader, contract, &format!("{prefix}.bias"))?;
    Ok(XasrConv2dWeights { weight, bias })
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
