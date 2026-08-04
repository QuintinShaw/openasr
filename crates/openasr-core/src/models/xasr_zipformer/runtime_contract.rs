//! X-ASR Zipformer2 execution metadata parsed from the `.oasr` GGUF header,
//! plus the admission-time runtime tensor contract that proves the pack carries
//! every tensor the runtime will load (metadata-derived shapes checked against
//! the tensor index) before the pack is admitted.

use thiserror::Error;

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::{GgufTensorIndex, GgufTensorMetadata};

use super::encoder_weights::layer_prefix;
use super::package_import::compact_xasr_name;

pub(crate) const XASR_NUM_STACKS_KEY: &str = "xasr.num_stacks";
pub(crate) const XASR_NUM_ENCODER_LAYERS_KEY: &str = "xasr.num_encoder_layers";
pub(crate) const XASR_ENCODER_DIMS_KEY: &str = "xasr.encoder_dims";
pub(crate) const XASR_QUERY_HEAD_DIMS_KEY: &str = "xasr.query_head_dims";
pub(crate) const XASR_VALUE_HEAD_DIMS_KEY: &str = "xasr.value_head_dims";
pub(crate) const XASR_NUM_HEADS_KEY: &str = "xasr.num_heads";
pub(crate) const XASR_CNN_MODULE_KERNELS_KEY: &str = "xasr.cnn_module_kernels";
pub(crate) const XASR_LEFT_CONTEXT_LEN_KEY: &str = "xasr.left_context_len";
pub(crate) const XASR_DOWNSAMPLING_FACTORS_KEY: &str = "xasr.downsampling_factors";
pub(crate) const XASR_FEATURE_DIM_KEY: &str = "xasr.feature_dim";
pub(crate) const XASR_DECODE_CHUNK_LEN_KEY: &str = "xasr.decode_chunk_len";
pub(crate) const XASR_JOINER_DIM_KEY: &str = "xasr.joiner_dim";
pub(crate) const XASR_DECODER_CONTEXT_SIZE_KEY: &str = "xasr.decoder_context_size";
pub(crate) const XASR_VOCAB_SIZE_KEY: &str = "xasr.vocab_size";
pub(crate) const XASR_BLANK_ID_KEY: &str = "xasr.blank_id";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XasrZipformerExecutionMetadata {
    pub num_stacks: usize,
    pub num_encoder_layers: Vec<usize>,
    pub encoder_dims: Vec<usize>,
    pub query_head_dims: Vec<usize>,
    pub value_head_dims: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub cnn_module_kernels: Vec<usize>,
    pub left_context_len: Vec<usize>,
    pub downsampling_factors: Vec<usize>,
    pub feature_dim: usize,
    pub decode_chunk_len: usize,
    pub joiner_dim: usize,
    pub decoder_context_size: usize,
    pub vocab_size: usize,
    pub blank_id: u32,
}

impl XasrZipformerExecutionMetadata {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        for (label, values) in [
            (
                "xasr metadata encoder layer counts",
                &self.num_encoder_layers,
            ),
            ("xasr metadata encoder dims", &self.encoder_dims),
            ("xasr metadata query head dims", &self.query_head_dims),
            ("xasr metadata value head dims", &self.value_head_dims),
            ("xasr metadata head counts", &self.num_heads),
            (
                "xasr metadata convolution kernels",
                &self.cnn_module_kernels,
            ),
            ("xasr metadata left context", &self.left_context_len),
            (
                "xasr metadata downsampling factors",
                &self.downsampling_factors,
            ),
        ] {
            bytes.add_vec(values, label)?;
        }
        Ok(bytes.finish())
    }

    pub(crate) fn total_encoder_layers(&self) -> usize {
        self.num_encoder_layers.iter().sum()
    }

    pub(crate) fn decoder_dim(&self) -> usize {
        self.joiner_dim
    }

    pub(crate) fn encoder_output_dim(&self) -> usize {
        self.encoder_dims
            .iter()
            .copied()
            .max()
            .unwrap_or(self.joiner_dim)
    }
}

pub(crate) fn parse_xasr_zipformer_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<XasrZipformerExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };

    let num_stacks = usize_key(XASR_NUM_STACKS_KEY)?;
    validate_positive_usize(num_stacks, XASR_NUM_STACKS_KEY)?;

    let num_encoder_layers =
        required_usize_list(metadata, XASR_NUM_ENCODER_LAYERS_KEY, num_stacks)?;
    let encoder_dims = required_usize_list(metadata, XASR_ENCODER_DIMS_KEY, num_stacks)?;
    let query_head_dims = required_usize_list(metadata, XASR_QUERY_HEAD_DIMS_KEY, num_stacks)?;
    let value_head_dims = required_usize_list(metadata, XASR_VALUE_HEAD_DIMS_KEY, num_stacks)?;
    let num_heads = required_usize_list(metadata, XASR_NUM_HEADS_KEY, num_stacks)?;
    let cnn_module_kernels =
        required_usize_list(metadata, XASR_CNN_MODULE_KERNELS_KEY, num_stacks)?;
    let left_context_len = required_usize_list(metadata, XASR_LEFT_CONTEXT_LEN_KEY, num_stacks)?;
    let downsampling_factors =
        required_usize_list(metadata, XASR_DOWNSAMPLING_FACTORS_KEY, num_stacks)?;

    let feature_dim = usize_key(XASR_FEATURE_DIM_KEY)?;
    let decode_chunk_len = usize_key(XASR_DECODE_CHUNK_LEN_KEY)?;
    let joiner_dim = usize_key(XASR_JOINER_DIM_KEY)?;
    let decoder_context_size = usize_key(XASR_DECODER_CONTEXT_SIZE_KEY)?;
    let vocab_size = usize_key(XASR_VOCAB_SIZE_KEY)?;
    let blank_id = u32_key(XASR_BLANK_ID_KEY)?;

    for (key, value) in [
        (XASR_FEATURE_DIM_KEY, feature_dim),
        (XASR_DECODE_CHUNK_LEN_KEY, decode_chunk_len),
        (XASR_JOINER_DIM_KEY, joiner_dim),
        (XASR_DECODER_CONTEXT_SIZE_KEY, decoder_context_size),
        (XASR_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    if blank_id as usize >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: XASR_BLANK_ID_KEY,
            reason: format!("blank_id {blank_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if decoder_context_size != 2 {
        return Err(MetadataContractError::InvalidValue {
            key: XASR_DECODER_CONTEXT_SIZE_KEY,
            reason: format!(
                "stateless X-ASR predictor expects context_size=2, got {decoder_context_size}"
            ),
        });
    }
    for (stack, ((heads, q_dim), v_dim)) in num_heads
        .iter()
        .zip(query_head_dims.iter())
        .zip(value_head_dims.iter())
        .enumerate()
    {
        validate_positive_usize(*heads, XASR_NUM_HEADS_KEY)?;
        validate_positive_usize(*q_dim, XASR_QUERY_HEAD_DIMS_KEY)?;
        validate_positive_usize(*v_dim, XASR_VALUE_HEAD_DIMS_KEY)?;
        let attn_dim =
            heads
                .checked_mul(*q_dim)
                .ok_or_else(|| MetadataContractError::InvalidValue {
                    key: XASR_QUERY_HEAD_DIMS_KEY,
                    reason: format!("stack {stack} attention dim overflows"),
                })?;
        if attn_dim == 0 {
            return Err(MetadataContractError::InvalidValue {
                key: XASR_QUERY_HEAD_DIMS_KEY,
                reason: format!("stack {stack} attention dim must be > 0"),
            });
        }
    }

    Ok(XasrZipformerExecutionMetadata {
        num_stacks,
        num_encoder_layers,
        encoder_dims,
        query_head_dims,
        value_head_dims,
        num_heads,
        cnn_module_kernels,
        left_context_len,
        downsampling_factors,
        feature_dim,
        decode_chunk_len,
        joiner_dim,
        decoder_context_size,
        vocab_size,
        blank_id,
    })
}

fn required_usize_list<M: ScalarMetadataView>(
    metadata: &M,
    key: &'static str,
    expected_len: usize,
) -> Result<Vec<usize>, MetadataContractError> {
    let raw = required_string_scalar(metadata, key)?;
    let values = raw
        .split(',')
        .map(str::trim)
        .enumerate()
        .map(|(index, item)| {
            if item.is_empty() {
                return Err(MetadataContractError::InvalidValue {
                    key,
                    reason: format!("entry {index} is empty"),
                });
            }
            let value =
                item.parse::<usize>()
                    .map_err(|source| MetadataContractError::InvalidValue {
                        key,
                        reason: format!("entry {index} '{item}' is not usize: {source}"),
                    })?;
            validate_positive_usize(value, key)?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected_len {
        return Err(MetadataContractError::InvalidValue {
            key,
            reason: format!("expected {expected_len} entries, got {}", values.len()),
        });
    }
    Ok(values)
}

/// Admission-time runtime tensor contract for xasr-zipformer. The runtime
/// resolves every weight through `compact_xasr_name(upstream_name)` and
/// validates exact dims while loading; this gate re-checks the same tensor set
/// against the lightweight tensor index so a pack that is missing a required
/// tensor (or carries a structurally wrong shape) fails closed at admission
/// instead of mid-execution. Shapes fully determined by the parsed metadata are
/// checked exactly; architecture-constant and data-derived dims fall back to a
/// rank check (the loader still enforces the exact value at runtime).
pub(crate) fn validate_xasr_zipformer_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<(), XasrTensorContractError> {
    validate_embed_tensors(index, metadata)?;
    for stack in 0..metadata.num_stacks {
        validate_stack_tensors(index, metadata, stack)?;
    }
    let output_downsampling_factor = metadata.downsampling_factors.last().copied().unwrap_or(2);
    require_vector(
        index,
        "encoder.downsample_output.bias",
        output_downsampling_factor,
    )?;
    validate_decoder_tensors(index, metadata)?;
    validate_joiner_tensors(index, metadata)?;
    Ok(())
}

fn validate_embed_tensors(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<(), XasrTensorContractError> {
    let first_dim = metadata.encoder_dims[0];
    // Conv stem kernels are architecture constants; rank-check the weights and
    // exact-check the biases (their length equals the constant out-channel count
    // the loader pins). The loader enforces the exact kernel shapes at runtime.
    for (weight, bias_len) in [
        ("encoder_embed.conv.0", 8usize),
        ("encoder_embed.conv.4", 32),
        ("encoder_embed.conv.7", 128),
        ("encoder_embed.convnext.depthwise_conv", 128),
        ("encoder_embed.convnext.pointwise_conv1", 384),
        ("encoder_embed.convnext.pointwise_conv2", 128),
    ] {
        require_rank(index, weight, 4)?;
        require_vector(index, &format!("{weight}.bias"), bias_len)?;
    }
    // `encoder_embed.out` output width is `first_dim`; its input width is an
    // architecture constant the loader pins (rank-check only here).
    require_rank2_out(index, "encoder_embed.out.weight", first_dim)?;
    require_vector(index, "encoder_embed.out.bias", first_dim)?;
    require_vector(index, "encoder_embed.out_norm.bias", first_dim)?;
    require_vector(index, "encoder_embed.out_norm.log_scale", 1)?;
    Ok(())
}

fn validate_stack_tensors(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
) -> Result<(), XasrTensorContractError> {
    let dim = metadata.encoder_dims[stack];
    if stack > 0 {
        require_vector(
            index,
            &format!("encoder.encoders.{stack}.downsample.bias"),
            metadata.downsampling_factors[stack],
        )?;
        require_vector(
            index,
            &format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
            dim,
        )?;
    }
    for layer in 0..metadata.num_encoder_layers[stack] {
        validate_layer_tensors(index, metadata, stack, layer)?;
    }
    Ok(())
}

fn validate_layer_tensors(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
    layer: usize,
) -> Result<(), XasrTensorContractError> {
    let dim = metadata.encoder_dims[stack];
    let kernel = metadata.cnn_module_kernels[stack];
    let causal_kernel = kernel.div_ceil(2);
    let prefix = layer_prefix(stack, layer);

    for name in ["feed_forward1", "feed_forward2", "feed_forward3"] {
        // Hidden width is data-derived; the loader pins it from the bias at
        // runtime. Here the input/output contract is the checkable surface.
        require_rank2_in(index, &format!("{prefix}.{name}.in_proj.weight"), dim)?;
        require_rank(index, &format!("{prefix}.{name}.in_proj.bias"), 1)?;
        require_rank2_out(index, &format!("{prefix}.{name}.out_proj.weight"), dim)?;
        require_vector(index, &format!("{prefix}.{name}.out_proj.bias"), dim)?;
    }

    // Self-attention qkv projection output width depends on the data-derived
    // `linear_pos` width; rank-check it and exact-check its input.
    require_rank2_in(
        index,
        &format!("{prefix}.self_attn_weights.in_proj.weight"),
        dim,
    )?;
    require_rank(
        index,
        &format!("{prefix}.self_attn_weights.in_proj.bias"),
        1,
    )?;
    require_rank(
        index,
        &format!("{prefix}.self_attn_weights.linear_pos.weight"),
        2,
    )?;

    for name in ["self_attn1", "self_attn2"] {
        require_rank2_in(index, &format!("{prefix}.{name}.in_proj.weight"), dim)?;
        require_rank(index, &format!("{prefix}.{name}.in_proj.bias"), 1)?;
        require_rank2_out(index, &format!("{prefix}.{name}.out_proj.weight"), dim)?;
        require_vector(index, &format!("{prefix}.{name}.out_proj.bias"), dim)?;
    }

    require_rank2_in(
        index,
        &format!("{prefix}.nonlin_attention.in_proj.weight"),
        dim,
    )?;
    require_rank(index, &format!("{prefix}.nonlin_attention.in_proj.bias"), 1)?;
    require_rank2_out(
        index,
        &format!("{prefix}.nonlin_attention.out_proj.weight"),
        dim,
    )?;
    require_vector(
        index,
        &format!("{prefix}.nonlin_attention.out_proj.bias"),
        dim,
    )?;

    for name in ["conv_module1", "conv_module2"] {
        require_exact(
            index,
            &format!("{prefix}.{name}.in_proj.weight"),
            &[dim as u64, (2 * dim) as u64],
        )?;
        require_vector(index, &format!("{prefix}.{name}.in_proj.bias"), 2 * dim)?;
        require_exact(
            index,
            &format!("{prefix}.{name}.depthwise_conv.causal_conv.weight"),
            &[causal_kernel as u64, 1, dim as u64],
        )?;
        require_vector(
            index,
            &format!("{prefix}.{name}.depthwise_conv.causal_conv.bias"),
            dim,
        )?;
        require_exact(
            index,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.weight"),
            &[kernel as u64, 1, dim as u64],
        )?;
        require_vector(
            index,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.bias"),
            dim,
        )?;
        require_exact(
            index,
            &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
            &[2, dim as u64, kernel as u64],
        )?;
        require_exact(
            index,
            &format!("{prefix}.{name}.out_proj.weight"),
            &[dim as u64, dim as u64],
        )?;
        require_vector(index, &format!("{prefix}.{name}.out_proj.bias"), dim)?;
    }

    require_vector(index, &format!("{prefix}.norm.bias"), dim)?;
    require_vector(index, &format!("{prefix}.norm.log_scale"), 1)?;
    require_vector(index, &format!("{prefix}.bypass.bypass_scale"), dim)?;
    require_vector(index, &format!("{prefix}.bypass_mid.bypass_scale"), dim)?;
    Ok(())
}

fn validate_decoder_tensors(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<(), XasrTensorContractError> {
    let decoder_dim = metadata.decoder_dim();
    require_exact(
        index,
        "decoder.embedding.weight",
        &[decoder_dim as u64, metadata.vocab_size as u64],
    )?;
    // The conv kernel's middle dim is `decoder_dim / 128` (grouped conv); the
    // loader enforces that exact shape at runtime. Rank-check here keeps tiny
    // admission fixtures (whose small joiner_dim would make the group count 0)
    // representable while still proving the tensor exists with the right rank.
    require_rank(index, "decoder.conv.weight", 3)?;
    Ok(())
}

fn validate_joiner_tensors(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<(), XasrTensorContractError> {
    let encoder_output_dim = metadata.encoder_output_dim();
    let joiner_dim = metadata.joiner_dim;
    let vocab_size = metadata.vocab_size;
    require_exact(
        index,
        "joiner.encoder_proj.weight",
        &[encoder_output_dim as u64, joiner_dim as u64],
    )?;
    require_vector(index, "joiner.encoder_proj.bias", joiner_dim)?;
    require_exact(
        index,
        "joiner.decoder_proj.weight",
        &[joiner_dim as u64, joiner_dim as u64],
    )?;
    require_vector(index, "joiner.decoder_proj.bias", joiner_dim)?;
    require_exact(
        index,
        "joiner.output_linear.weight",
        &[joiner_dim as u64, vocab_size as u64],
    )?;
    require_vector(index, "joiner.output_linear.bias", vocab_size)?;
    Ok(())
}

/// Generates the minimal runtime tensor set (compacted pack names plus valid
/// dims) that satisfies the xasr-zipformer runtime tensor contract for
/// `metadata`. Data-derived dims that the loader only learns by reading bytes
/// are filled with a small valid placeholder. The runtime-ready test fixture
/// stamps exactly this set, so the fixture and the admission validator agree on
/// the required tensors through one enumeration.
pub(crate) fn xasr_zipformer_minimal_runtime_tensors(
    metadata: &XasrZipformerExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    fn push(out: &mut Vec<(String, Vec<u64>)>, name: &str, dims: Vec<u64>) {
        out.push((compact_xasr_name(name), dims));
    }

    let first_dim = metadata.encoder_dims[0] as u64;
    for (weight, bias_len) in [
        ("encoder_embed.conv.0", 8u64),
        ("encoder_embed.conv.4", 32),
        ("encoder_embed.conv.7", 128),
        ("encoder_embed.convnext.depthwise_conv", 128),
        ("encoder_embed.convnext.pointwise_conv1", 384),
        ("encoder_embed.convnext.pointwise_conv2", 128),
    ] {
        push(&mut out, weight, vec![3, 3, 1, bias_len]);
        push(&mut out, &format!("{weight}.bias"), vec![bias_len]);
    }
    push(&mut out, "encoder_embed.out.weight", vec![32, first_dim]);
    push(&mut out, "encoder_embed.out.bias", vec![first_dim]);
    push(&mut out, "encoder_embed.out_norm.bias", vec![first_dim]);
    push(&mut out, "encoder_embed.out_norm.log_scale", vec![1]);

    for stack in 0..metadata.num_stacks {
        let dim = metadata.encoder_dims[stack] as u64;
        if stack > 0 {
            push(
                &mut out,
                &format!("encoder.encoders.{stack}.downsample.bias"),
                vec![metadata.downsampling_factors[stack] as u64],
            );
            push(
                &mut out,
                &format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
                vec![dim],
            );
        }
        let kernel = metadata.cnn_module_kernels[stack] as u64;
        let causal_kernel = (kernel as usize).div_ceil(2) as u64;
        for layer in 0..metadata.num_encoder_layers[stack] {
            let prefix = layer_prefix(stack, layer);
            for name in ["feed_forward1", "feed_forward2", "feed_forward3"] {
                push(
                    &mut out,
                    &format!("{prefix}.{name}.in_proj.weight"),
                    vec![dim, 2 * dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.in_proj.bias"),
                    vec![2 * dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.weight"),
                    vec![2 * dim, dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.bias"),
                    vec![dim],
                );
            }
            push(
                &mut out,
                &format!("{prefix}.self_attn_weights.in_proj.weight"),
                vec![dim, 20],
            );
            push(
                &mut out,
                &format!("{prefix}.self_attn_weights.in_proj.bias"),
                vec![20],
            );
            push(
                &mut out,
                &format!("{prefix}.self_attn_weights.linear_pos.weight"),
                vec![4, 4],
            );
            for name in ["self_attn1", "self_attn2"] {
                push(
                    &mut out,
                    &format!("{prefix}.{name}.in_proj.weight"),
                    vec![dim, 4],
                );
                push(&mut out, &format!("{prefix}.{name}.in_proj.bias"), vec![4]);
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.weight"),
                    vec![4, dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.bias"),
                    vec![dim],
                );
            }
            push(
                &mut out,
                &format!("{prefix}.nonlin_attention.in_proj.weight"),
                vec![dim, 6],
            );
            push(
                &mut out,
                &format!("{prefix}.nonlin_attention.in_proj.bias"),
                vec![6],
            );
            push(
                &mut out,
                &format!("{prefix}.nonlin_attention.out_proj.weight"),
                vec![2, dim],
            );
            push(
                &mut out,
                &format!("{prefix}.nonlin_attention.out_proj.bias"),
                vec![dim],
            );
            for name in ["conv_module1", "conv_module2"] {
                push(
                    &mut out,
                    &format!("{prefix}.{name}.in_proj.weight"),
                    vec![dim, 2 * dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.in_proj.bias"),
                    vec![2 * dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.depthwise_conv.causal_conv.weight"),
                    vec![causal_kernel, 1, dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.depthwise_conv.causal_conv.bias"),
                    vec![dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.weight"),
                    vec![kernel, 1, dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.bias"),
                    vec![dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
                    vec![2, dim, kernel],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.weight"),
                    vec![dim, dim],
                );
                push(
                    &mut out,
                    &format!("{prefix}.{name}.out_proj.bias"),
                    vec![dim],
                );
            }
            push(&mut out, &format!("{prefix}.norm.bias"), vec![dim]);
            push(&mut out, &format!("{prefix}.norm.log_scale"), vec![1]);
            push(
                &mut out,
                &format!("{prefix}.bypass.bypass_scale"),
                vec![dim],
            );
            push(
                &mut out,
                &format!("{prefix}.bypass_mid.bypass_scale"),
                vec![dim],
            );
        }
    }
    push(
        &mut out,
        "encoder.downsample_output.bias",
        vec![*metadata.downsampling_factors.last().unwrap_or(&2) as u64],
    );

    let decoder_dim = metadata.decoder_dim() as u64;
    let vocab = metadata.vocab_size as u64;
    push(
        &mut out,
        "decoder.embedding.weight",
        vec![decoder_dim, vocab],
    );
    push(
        &mut out,
        "decoder.conv.weight",
        vec![metadata.decoder_context_size as u64, 1, decoder_dim],
    );

    let joiner = metadata.joiner_dim as u64;
    let enc_out = metadata.encoder_output_dim() as u64;
    push(
        &mut out,
        "joiner.encoder_proj.weight",
        vec![enc_out, joiner],
    );
    push(&mut out, "joiner.encoder_proj.bias", vec![joiner]);
    push(&mut out, "joiner.decoder_proj.weight", vec![joiner, joiner]);
    push(&mut out, "joiner.decoder_proj.bias", vec![joiner]);
    push(&mut out, "joiner.output_linear.weight", vec![joiner, vocab]);
    push(&mut out, "joiner.output_linear.bias", vec![vocab]);
    out
}

/// Typed tensor-contract failure for a single xasr-zipformer pack.
#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum XasrTensorContractError {
    #[error("xasr-zipformer missing required runtime tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("xasr-zipformer runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

fn require_tensor<'a>(
    index: &'a GgufTensorIndex,
    upstream_name: &str,
) -> Result<&'a GgufTensorMetadata, XasrTensorContractError> {
    let name = compact_xasr_name(upstream_name);
    index
        .get(&name)
        .ok_or(XasrTensorContractError::MissingRequiredTensor { name })
}

fn require_vector(
    index: &GgufTensorIndex,
    upstream_name: &str,
    len: usize,
) -> Result<(), XasrTensorContractError> {
    let name = compact_xasr_name(upstream_name);
    let tensor = index
        .get(&name)
        .ok_or(XasrTensorContractError::MissingRequiredTensor { name })?;
    if tensor.dims.len() == 1 && tensor.dims[0] == len as u64 {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        format!("expected a rank-1 vector of length {len}"),
    ))
}

fn require_exact(
    index: &GgufTensorIndex,
    upstream_name: &str,
    expected: &[u64],
) -> Result<(), XasrTensorContractError> {
    let name = compact_xasr_name(upstream_name);
    let tensor = index
        .get(&name)
        .ok_or(XasrTensorContractError::MissingRequiredTensor { name })?;
    if tensor.dims.as_slice() == expected {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        format!("expected shape {expected:?}"),
    ))
}

fn require_rank(
    index: &GgufTensorIndex,
    upstream_name: &str,
    rank: usize,
) -> Result<(), XasrTensorContractError> {
    let tensor = require_tensor(index, upstream_name)?;
    if tensor.dims.len() == rank {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        format!("expected rank {rank}"),
    ))
}

fn require_rank2_in(
    index: &GgufTensorIndex,
    upstream_name: &str,
    input_dim: usize,
) -> Result<(), XasrTensorContractError> {
    let tensor = require_tensor(index, upstream_name)?;
    if tensor.dims.len() == 2 && tensor.dims[0] == input_dim as u64 {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        format!("expected a rank-2 matrix with input dim {input_dim}"),
    ))
}

fn require_rank2_out(
    index: &GgufTensorIndex,
    upstream_name: &str,
    output_dim: usize,
) -> Result<(), XasrTensorContractError> {
    let tensor = require_tensor(index, upstream_name)?;
    if tensor.dims.len() == 2 && tensor.dims[1] == output_dim as u64 {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        format!("expected a rank-2 matrix with output dim {output_dim}"),
    ))
}

fn invalid_shape(name: &str, shape: &[u64], reason: String) -> XasrTensorContractError {
    let rendered = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    XasrTensorContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: format!("[{rendered}]"),
        reason,
    }
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata =
        parse_xasr_zipformer_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("xasr-zipformer", error)
        })?;
    validate_xasr_zipformer_runtime_tensors_with_index(preflight.tensor_index(), &metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn metadata() -> BTreeMap<String, String> {
        [
            (XASR_NUM_STACKS_KEY, "6"),
            (XASR_NUM_ENCODER_LAYERS_KEY, "2,2,4,5,4,2"),
            (XASR_ENCODER_DIMS_KEY, "192,256,512,768,512,256"),
            (XASR_QUERY_HEAD_DIMS_KEY, "32,32,32,32,32,32"),
            (XASR_VALUE_HEAD_DIMS_KEY, "12,12,12,12,12,12"),
            (XASR_NUM_HEADS_KEY, "4,4,4,8,4,4"),
            (XASR_CNN_MODULE_KERNELS_KEY, "31,31,15,15,15,31"),
            (XASR_LEFT_CONTEXT_LEN_KEY, "256,128,64,32,64,128"),
            (XASR_DOWNSAMPLING_FACTORS_KEY, "1,2,4,8,4,2"),
            (XASR_FEATURE_DIM_KEY, "80"),
            (XASR_DECODE_CHUNK_LEN_KEY, "48"),
            (XASR_JOINER_DIM_KEY, "512"),
            (XASR_DECODER_CONTEXT_SIZE_KEY, "2"),
            (XASR_VOCAB_SIZE_KEY, "5000"),
            (XASR_BLANK_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_xasr_metadata() {
        let parsed = parse_xasr_zipformer_execution_metadata(&metadata()).expect("parse");
        assert_eq!(parsed.num_stacks, 6);
        assert_eq!(parsed.total_encoder_layers(), 19);
        assert_eq!(parsed.encoder_output_dim(), 768);
        assert_eq!(parsed.decoder_dim(), 512);
        assert_eq!(parsed.left_context_len, vec![256, 128, 64, 32, 64, 128]);
        assert_eq!(parsed.decode_chunk_len, 48);
        assert_eq!(parsed.blank_id, 0);
    }

    #[test]
    fn rejects_list_length_drift() {
        let mut metadata = metadata();
        metadata.insert(XASR_ENCODER_DIMS_KEY.to_string(), "192,256".to_string());
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_unsupported_context_size() {
        let mut metadata = metadata();
        metadata.insert(XASR_DECODER_CONTEXT_SIZE_KEY.to_string(), "3".to_string());
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
    }

    // --- Runtime tensor contract ---

    fn tensor_contract_metadata() -> XasrZipformerExecutionMetadata {
        XasrZipformerExecutionMetadata {
            num_stacks: 1,
            num_encoder_layers: vec![1],
            encoder_dims: vec![16],
            query_head_dims: vec![4],
            value_head_dims: vec![4],
            num_heads: vec![2],
            cnn_module_kernels: vec![3],
            left_context_len: vec![4],
            downsampling_factors: vec![1],
            feature_dim: 80,
            decode_chunk_len: 4,
            joiner_dim: 128,
            decoder_context_size: 2,
            vocab_size: 5,
            blank_id: 0,
        }
    }

    fn tensor_entry(name: &str, dims: Vec<u64>) -> GgufTensorMetadata {
        GgufTensorMetadata {
            name: compact_xasr_name(name),
            dims,
            ggml_type: 0,
            type_name: "f32".to_string(),
            size_bytes: 0,
            offset_bytes: 0,
        }
    }

    /// Builds the full runtime tensor set for `metadata`, mirroring the names
    /// the runtime loader resolves (through `compact_xasr_name` / `layer_prefix`).
    fn required_tensors(metadata: &XasrZipformerExecutionMetadata) -> Vec<GgufTensorMetadata> {
        let mut tensors = Vec::new();
        let first_dim = metadata.encoder_dims[0] as u64;
        for (weight, bias_len) in [
            ("encoder_embed.conv.0", 8u64),
            ("encoder_embed.conv.4", 32),
            ("encoder_embed.conv.7", 128),
            ("encoder_embed.convnext.depthwise_conv", 128),
            ("encoder_embed.convnext.pointwise_conv1", 384),
            ("encoder_embed.convnext.pointwise_conv2", 128),
        ] {
            tensors.push(tensor_entry(weight, vec![3, 3, 1, bias_len]));
            tensors.push(tensor_entry(&format!("{weight}.bias"), vec![bias_len]));
        }
        tensors.push(tensor_entry(
            "encoder_embed.out.weight",
            vec![32, first_dim],
        ));
        tensors.push(tensor_entry("encoder_embed.out.bias", vec![first_dim]));
        tensors.push(tensor_entry("encoder_embed.out_norm.bias", vec![first_dim]));
        tensors.push(tensor_entry("encoder_embed.out_norm.log_scale", vec![1]));

        for stack in 0..metadata.num_stacks {
            let dim = metadata.encoder_dims[stack] as u64;
            if stack > 0 {
                tensors.push(tensor_entry(
                    &format!("encoder.encoders.{stack}.downsample.bias"),
                    vec![metadata.downsampling_factors[stack] as u64],
                ));
                tensors.push(tensor_entry(
                    &format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
                    vec![dim],
                ));
            }
            let kernel = metadata.cnn_module_kernels[stack] as u64;
            let causal_kernel = (kernel as usize).div_ceil(2) as u64;
            for layer in 0..metadata.num_encoder_layers[stack] {
                let prefix = layer_prefix(stack, layer);
                for name in ["feed_forward1", "feed_forward2", "feed_forward3"] {
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.weight"),
                        vec![dim, 2 * dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.bias"),
                        vec![2 * dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.weight"),
                        vec![2 * dim, dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.bias"),
                        vec![dim],
                    ));
                }
                tensors.push(tensor_entry(
                    &format!("{prefix}.self_attn_weights.in_proj.weight"),
                    vec![dim, 20],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.self_attn_weights.in_proj.bias"),
                    vec![20],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.self_attn_weights.linear_pos.weight"),
                    vec![4, 4],
                ));
                for name in ["self_attn1", "self_attn2"] {
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.weight"),
                        vec![dim, 4],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.bias"),
                        vec![4],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.weight"),
                        vec![4, dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.bias"),
                        vec![dim],
                    ));
                }
                tensors.push(tensor_entry(
                    &format!("{prefix}.nonlin_attention.in_proj.weight"),
                    vec![dim, 6],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.nonlin_attention.in_proj.bias"),
                    vec![6],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.nonlin_attention.out_proj.weight"),
                    vec![2, dim],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.nonlin_attention.out_proj.bias"),
                    vec![dim],
                ));
                for name in ["conv_module1", "conv_module2"] {
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.weight"),
                        vec![dim, 2 * dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.in_proj.bias"),
                        vec![2 * dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.depthwise_conv.causal_conv.weight"),
                        vec![causal_kernel, 1, dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.depthwise_conv.causal_conv.bias"),
                        vec![dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.weight"),
                        vec![kernel, 1, dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.bias"),
                        vec![dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
                        vec![2, dim, kernel],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.weight"),
                        vec![dim, dim],
                    ));
                    tensors.push(tensor_entry(
                        &format!("{prefix}.{name}.out_proj.bias"),
                        vec![dim],
                    ));
                }
                tensors.push(tensor_entry(&format!("{prefix}.norm.bias"), vec![dim]));
                tensors.push(tensor_entry(&format!("{prefix}.norm.log_scale"), vec![1]));
                tensors.push(tensor_entry(
                    &format!("{prefix}.bypass.bypass_scale"),
                    vec![dim],
                ));
                tensors.push(tensor_entry(
                    &format!("{prefix}.bypass_mid.bypass_scale"),
                    vec![dim],
                ));
            }
        }
        tensors.push(tensor_entry(
            "encoder.downsample_output.bias",
            vec![*metadata.downsampling_factors.last().unwrap() as u64],
        ));

        let decoder_dim = metadata.decoder_dim() as u64;
        let vocab = metadata.vocab_size as u64;
        tensors.push(tensor_entry(
            "decoder.embedding.weight",
            vec![decoder_dim, vocab],
        ));
        tensors.push(tensor_entry(
            "decoder.conv.weight",
            vec![
                metadata.decoder_context_size as u64,
                decoder_dim / 128,
                decoder_dim,
            ],
        ));

        let joiner = metadata.joiner_dim as u64;
        let enc_out = metadata.encoder_output_dim() as u64;
        tensors.push(tensor_entry(
            "joiner.encoder_proj.weight",
            vec![enc_out, joiner],
        ));
        tensors.push(tensor_entry("joiner.encoder_proj.bias", vec![joiner]));
        tensors.push(tensor_entry(
            "joiner.decoder_proj.weight",
            vec![joiner, joiner],
        ));
        tensors.push(tensor_entry("joiner.decoder_proj.bias", vec![joiner]));
        tensors.push(tensor_entry(
            "joiner.output_linear.weight",
            vec![joiner, vocab],
        ));
        tensors.push(tensor_entry("joiner.output_linear.bias", vec![vocab]));
        tensors
    }

    fn tensor_index_from(tensors: Vec<GgufTensorMetadata>) -> GgufTensorIndex {
        GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("xasr-tensor-contract.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("tensor names must be unique")
    }

    #[test]
    fn tensor_contract_accepts_a_complete_runtime_tensor_set() {
        let metadata = tensor_contract_metadata();
        let index = tensor_index_from(required_tensors(&metadata));
        validate_xasr_zipformer_runtime_tensors_with_index(&index, &metadata)
            .expect("complete tensor set must pass");
    }

    #[test]
    fn tensor_contract_rejects_a_missing_tensor() {
        let metadata = tensor_contract_metadata();
        let missing = compact_xasr_name("joiner.output_linear.bias");
        let tensors = required_tensors(&metadata)
            .into_iter()
            .filter(|tensor| tensor.name != missing)
            .collect();
        let index = tensor_index_from(tensors);
        let error = validate_xasr_zipformer_runtime_tensors_with_index(&index, &metadata)
            .expect_err("a missing required tensor must fail closed");
        assert!(matches!(
            error,
            XasrTensorContractError::MissingRequiredTensor { .. }
        ));
    }

    #[test]
    fn tensor_contract_rejects_a_wrong_shape() {
        let metadata = tensor_contract_metadata();
        // `decoder.embedding.weight` is exact-checked ([decoder_dim, vocab]);
        // corrupting its shape must fail closed.
        let target = compact_xasr_name("decoder.embedding.weight");
        let tensors = required_tensors(&metadata)
            .into_iter()
            .map(|mut tensor| {
                if tensor.name == target {
                    tensor.dims = vec![1, 1];
                }
                tensor
            })
            .collect();
        let index = tensor_index_from(tensors);
        let error = validate_xasr_zipformer_runtime_tensors_with_index(&index, &metadata)
            .expect_err("a wrong tensor shape must fail closed");
        assert!(matches!(
            error,
            XasrTensorContractError::InvalidTensorShape { .. }
        ));
    }
}
