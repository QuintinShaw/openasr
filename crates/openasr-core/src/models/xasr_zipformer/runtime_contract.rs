//! X-ASR Zipformer2 execution metadata parsed from the `.oasr` GGUF header,
//! plus the admission-time runtime tensor contract that proves the pack carries
//! every tensor the runtime will load (metadata-derived shapes checked against
//! the tensor index) before the pack is admitted.

use thiserror::Error;

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::GgufTensorIndex;

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

/// The shape contract a single required xasr-zipformer tensor must satisfy at
/// admission. Shapes fully determined by the parsed metadata (or by an
/// architecture constant the loader pins) are checked exactly or by a fixed
/// input/output dim; data-derived dims the loader only learns by reading bytes
/// fall back to a rank check (the loader still enforces the exact value at
/// runtime).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XasrTensorShape {
    /// Dims must equal `expected` exactly.
    Exact(Vec<usize>),
    /// Rank-1 vector of exactly `len` elements.
    Vector(usize),
    /// Only the rank is pinned (data-derived / loader-enforced dims).
    Rank(usize),
    /// Rank-2 matrix whose input dim (`dims[0]`) is `input_dim`.
    Rank2In { input_dim: usize },
    /// Rank-2 matrix whose output dim (`dims[1]`) is `output_dim`.
    Rank2Out { output_dim: usize },
}

/// One tensor the xasr-zipformer runtime loads, named by its upstream icefall
/// name (resolved to the pack name through `compact_xasr_name`) plus the shape
/// contract the admission validator enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XasrTensorRequirement {
    pub upstream_name: String,
    pub shape: XasrTensorShape,
}

fn requirement(upstream_name: String, shape: XasrTensorShape) -> XasrTensorRequirement {
    XasrTensorRequirement {
        upstream_name,
        shape,
    }
}

/// The single source of truth for the xasr-zipformer runtime tensor set: every
/// tensor the weight loader reads (`encoder_weights` / `weights`), named exactly
/// as the loader resolves it and shaped with the strictness the loader applies.
/// The admission validator checks precisely this set against the tensor index,
/// and the runtime-ready fixture projects it, so validator / loader / fixture
/// cannot drift: a pack missing any entry (or carrying a structurally wrong
/// shape) fails closed at admission instead of mid-execution.
pub(crate) fn xasr_zipformer_runtime_tensor_requirements(
    metadata: &XasrZipformerExecutionMetadata,
) -> Vec<XasrTensorRequirement> {
    let mut out = Vec::new();
    embed_requirements(&mut out, metadata);
    for stack in 0..metadata.num_stacks {
        stack_requirements(&mut out, metadata, stack);
    }
    let output_downsampling_factor = metadata.downsampling_factors.last().copied().unwrap_or(2);
    out.push(requirement(
        "encoder.downsample_output.bias".to_string(),
        XasrTensorShape::Vector(output_downsampling_factor),
    ));
    decoder_requirements(&mut out, metadata);
    joiner_requirements(&mut out, metadata);
    out
}

/// Admission-time runtime tensor contract for xasr-zipformer. Validates the
/// pack tensor index against the single required-tensor enumeration
/// ([`xasr_zipformer_runtime_tensor_requirements`]); a missing tensor or a shape
/// the declared geometry cannot construct fails closed with the offending pack
/// tensor named.
pub(crate) fn validate_xasr_zipformer_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &XasrZipformerExecutionMetadata,
) -> Result<(), XasrTensorContractError> {
    for requirement in xasr_zipformer_runtime_tensor_requirements(metadata) {
        check_requirement(index, &requirement)?;
    }
    Ok(())
}

fn check_requirement(
    index: &GgufTensorIndex,
    requirement: &XasrTensorRequirement,
) -> Result<(), XasrTensorContractError> {
    let name = compact_xasr_name(&requirement.upstream_name);
    let tensor = index
        .get(&name)
        .ok_or(XasrTensorContractError::MissingRequiredTensor { name })?;
    let valid = match &requirement.shape {
        XasrTensorShape::Exact(expected) => {
            tensor.dims.len() == expected.len()
                && tensor
                    .dims
                    .iter()
                    .zip(expected.iter())
                    .all(|(actual, expected)| *actual == *expected as u64)
        }
        XasrTensorShape::Vector(len) => tensor.dims.len() == 1 && tensor.dims[0] == *len as u64,
        XasrTensorShape::Rank(rank) => tensor.dims.len() == *rank,
        XasrTensorShape::Rank2In { input_dim } => {
            tensor.dims.len() == 2 && tensor.dims[0] == *input_dim as u64
        }
        XasrTensorShape::Rank2Out { output_dim } => {
            tensor.dims.len() == 2 && tensor.dims[1] == *output_dim as u64
        }
    };
    if valid {
        return Ok(());
    }
    let reason = match &requirement.shape {
        XasrTensorShape::Exact(expected) => format!("expected shape {expected:?}"),
        XasrTensorShape::Vector(len) => {
            format!("expected a rank-1 vector of length {len}")
        }
        XasrTensorShape::Rank(rank) => format!("expected rank {rank}"),
        XasrTensorShape::Rank2In { input_dim } => {
            format!("expected a rank-2 matrix with input dim {input_dim}")
        }
        XasrTensorShape::Rank2Out { output_dim } => {
            format!("expected a rank-2 matrix with output dim {output_dim}")
        }
    };
    Err(invalid_shape(&tensor.name, &tensor.dims, reason))
}

fn embed_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
) {
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
        out.push(requirement(
            format!("{weight}.weight"),
            XasrTensorShape::Rank(4),
        ));
        out.push(requirement(
            format!("{weight}.bias"),
            XasrTensorShape::Vector(bias_len),
        ));
    }
    // `encoder_embed.out` output width is `first_dim`; its input width is an
    // architecture constant the loader pins (rank-check only here).
    out.push(requirement(
        "encoder_embed.out.weight".to_string(),
        XasrTensorShape::Rank2Out {
            output_dim: first_dim,
        },
    ));
    out.push(requirement(
        "encoder_embed.out.bias".to_string(),
        XasrTensorShape::Vector(first_dim),
    ));
    out.push(requirement(
        "encoder_embed.out_norm.bias".to_string(),
        XasrTensorShape::Vector(first_dim),
    ));
    out.push(requirement(
        "encoder_embed.out_norm.log_scale".to_string(),
        XasrTensorShape::Vector(1),
    ));
}

fn stack_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
) {
    let dim = metadata.encoder_dims[stack];
    if stack > 0 {
        out.push(requirement(
            format!("encoder.encoders.{stack}.downsample.bias"),
            XasrTensorShape::Vector(metadata.downsampling_factors[stack]),
        ));
        out.push(requirement(
            format!("encoder.encoders.{stack}.out_combiner.bypass_scale"),
            XasrTensorShape::Vector(dim),
        ));
    }
    for layer in 0..metadata.num_encoder_layers[stack] {
        layer_requirements(out, metadata, stack, layer);
    }
}

fn layer_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
    stack: usize,
    layer: usize,
) {
    let dim = metadata.encoder_dims[stack];
    let kernel = metadata.cnn_module_kernels[stack];
    let causal_kernel = kernel.div_ceil(2);
    let prefix = layer_prefix(stack, layer);

    for name in ["feed_forward1", "feed_forward2", "feed_forward3"] {
        // Hidden width is data-derived; the loader pins it from the bias at
        // runtime. Here the input/output contract is the checkable surface.
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.weight"),
            XasrTensorShape::Rank2In { input_dim: dim },
        ));
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.bias"),
            XasrTensorShape::Rank(1),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.weight"),
            XasrTensorShape::Rank2Out { output_dim: dim },
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.bias"),
            XasrTensorShape::Vector(dim),
        ));
    }

    // Self-attention qkv projection output width depends on the data-derived
    // `linear_pos` width; rank-check it and exact-check its input.
    out.push(requirement(
        format!("{prefix}.self_attn_weights.in_proj.weight"),
        XasrTensorShape::Rank2In { input_dim: dim },
    ));
    out.push(requirement(
        format!("{prefix}.self_attn_weights.in_proj.bias"),
        XasrTensorShape::Rank(1),
    ));
    out.push(requirement(
        format!("{prefix}.self_attn_weights.linear_pos.weight"),
        XasrTensorShape::Rank(2),
    ));

    for name in ["self_attn1", "self_attn2"] {
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.weight"),
            XasrTensorShape::Rank2In { input_dim: dim },
        ));
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.bias"),
            XasrTensorShape::Rank(1),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.weight"),
            XasrTensorShape::Rank2Out { output_dim: dim },
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.bias"),
            XasrTensorShape::Vector(dim),
        ));
    }

    out.push(requirement(
        format!("{prefix}.nonlin_attention.in_proj.weight"),
        XasrTensorShape::Rank2In { input_dim: dim },
    ));
    out.push(requirement(
        format!("{prefix}.nonlin_attention.in_proj.bias"),
        XasrTensorShape::Rank(1),
    ));
    out.push(requirement(
        format!("{prefix}.nonlin_attention.out_proj.weight"),
        XasrTensorShape::Rank2Out { output_dim: dim },
    ));
    out.push(requirement(
        format!("{prefix}.nonlin_attention.out_proj.bias"),
        XasrTensorShape::Vector(dim),
    ));

    for name in ["conv_module1", "conv_module2"] {
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.weight"),
            XasrTensorShape::Exact(vec![dim, 2 * dim]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.bias"),
            XasrTensorShape::Vector(2 * dim),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.depthwise_conv.causal_conv.weight"),
            XasrTensorShape::Exact(vec![causal_kernel, 1, dim]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.depthwise_conv.causal_conv.bias"),
            XasrTensorShape::Vector(dim),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.weight"),
            XasrTensorShape::Exact(vec![kernel, 1, dim]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.depthwise_conv.chunkwise_conv.bias"),
            XasrTensorShape::Vector(dim),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.depthwise_conv.chunkwise_conv_scale"),
            XasrTensorShape::Exact(vec![2, dim, kernel]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.weight"),
            XasrTensorShape::Exact(vec![dim, dim]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.out_proj.bias"),
            XasrTensorShape::Vector(dim),
        ));
    }

    out.push(requirement(
        format!("{prefix}.norm.bias"),
        XasrTensorShape::Vector(dim),
    ));
    out.push(requirement(
        format!("{prefix}.norm.log_scale"),
        XasrTensorShape::Vector(1),
    ));
    out.push(requirement(
        format!("{prefix}.bypass.bypass_scale"),
        XasrTensorShape::Vector(dim),
    ));
    out.push(requirement(
        format!("{prefix}.bypass_mid.bypass_scale"),
        XasrTensorShape::Vector(dim),
    ));
}

fn decoder_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
) {
    let decoder_dim = metadata.decoder_dim();
    out.push(requirement(
        "decoder.embedding.weight".to_string(),
        XasrTensorShape::Exact(vec![decoder_dim, metadata.vocab_size]),
    ));
    // The conv kernel's middle dim is `decoder_dim / 128` (grouped conv); the
    // loader enforces that exact shape at runtime. Rank-check here keeps tiny
    // admission fixtures (whose small joiner_dim would make the group count 0)
    // representable while still proving the tensor exists with the right rank.
    out.push(requirement(
        "decoder.conv.weight".to_string(),
        XasrTensorShape::Rank(3),
    ));
}

fn joiner_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
) {
    let encoder_output_dim = metadata.encoder_output_dim();
    let joiner_dim = metadata.joiner_dim;
    let vocab_size = metadata.vocab_size;
    out.push(requirement(
        "joiner.encoder_proj.weight".to_string(),
        XasrTensorShape::Exact(vec![encoder_output_dim, joiner_dim]),
    ));
    out.push(requirement(
        "joiner.encoder_proj.bias".to_string(),
        XasrTensorShape::Vector(joiner_dim),
    ));
    out.push(requirement(
        "joiner.decoder_proj.weight".to_string(),
        XasrTensorShape::Exact(vec![joiner_dim, joiner_dim]),
    ));
    out.push(requirement(
        "joiner.decoder_proj.bias".to_string(),
        XasrTensorShape::Vector(joiner_dim),
    ));
    out.push(requirement(
        "joiner.output_linear.weight".to_string(),
        XasrTensorShape::Exact(vec![joiner_dim, vocab_size]),
    ));
    out.push(requirement(
        "joiner.output_linear.bias".to_string(),
        XasrTensorShape::Vector(vocab_size),
    ));
}

/// Projects the single runtime tensor contract
/// ([`xasr_zipformer_runtime_tensor_requirements`]) into a minimal runtime-ready
/// tensor set: compacted pack names plus one valid dims choice per shape
/// contract. Data-derived dims the loader only learns by reading bytes get a
/// small valid placeholder. The runtime-ready test fixture stamps exactly this
/// set, so fixture and admission validator agree on the required tensors
/// through one enumeration.
pub(crate) fn xasr_zipformer_minimal_runtime_tensors(
    metadata: &XasrZipformerExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    xasr_zipformer_runtime_tensor_requirements(metadata)
        .into_iter()
        .map(|requirement| {
            // Project one valid dims choice per shape contract. GGUF reads back
            // a canonical rank trimmed of trailing ones, so every projection
            // keeps its last dim > 1 (rank-1 `Vector(1)` is the lone `[1]`
            // exception that survives trimming).
            let dims = match &requirement.shape {
                XasrTensorShape::Exact(expected) => {
                    expected.iter().map(|dim| *dim as u64).collect()
                }
                XasrTensorShape::Vector(len) => vec![*len as u64],
                XasrTensorShape::Rank(1) => vec![2],
                XasrTensorShape::Rank(rank) => {
                    let mut dims = vec![1_u64; *rank];
                    *dims.last_mut().expect("rank > 0") = 2;
                    dims
                }
                XasrTensorShape::Rank2In { input_dim } => vec![*input_dim as u64, 2],
                XasrTensorShape::Rank2Out { output_dim } => vec![2, *output_dim as u64],
            };
            (compact_xasr_name(&requirement.upstream_name), dims)
        })
        .collect()
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
    use crate::GgufTensorMetadata;
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

    /// Builds the full runtime tensor index set for `metadata` by projecting
    /// the single runtime tensor contract (`xasr_zipformer_minimal_runtime_tensors`),
    /// so the fixture index and the admission validator can never drift: both
    /// are projections of one enumeration.
    fn required_tensors(metadata: &XasrZipformerExecutionMetadata) -> Vec<GgufTensorMetadata> {
        xasr_zipformer_minimal_runtime_tensors(metadata)
            .into_iter()
            .map(|(name, dims)| GgufTensorMetadata {
                name,
                dims,
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: 0,
            })
            .collect()
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

    /// The requirement enumeration IS the loader's read set: pin it on the full
    /// production geometry (the published X-ASR checkpoint's architecture). Any
    /// drift between what the validator requires and what the loader resolves
    /// shows up here as a name/count change. Regression anchor for the incident
    /// where the admission validator demanded compacted names without the
    /// `.weight` suffix (`EE.conv.0`) while the loader reads `EE.conv.0.weight`,
    /// rejecting every published pack at preflight.
    #[test]
    fn requirement_set_matches_the_loader_read_set_on_production_geometry() {
        let parsed = parse_xasr_zipformer_execution_metadata(&metadata()).expect("parse");
        let requirements = xasr_zipformer_runtime_tensor_requirements(&parsed);
        // The published X-ASR pack ships exactly 966 tensors and the loader
        // reads every one of them (see the ONNX round-trip import test).
        assert_eq!(
            requirements.len(),
            966,
            "the requirement set must stay exactly the loader read set"
        );
        let names: std::collections::BTreeSet<String> = requirements
            .iter()
            .map(|requirement| compact_xasr_name(&requirement.upstream_name))
            .collect();
        assert_eq!(
            names.len(),
            requirements.len(),
            "compacted requirement names must be unique"
        );
        // Embed conv weights carry the `.weight` suffix the loader reads.
        for name in [
            "EE.conv.0.weight",
            "EE.conv.4.weight",
            "EE.conv.7.weight",
            "EE.CX.DW.weight",
            "EE.CX.PW1.weight",
            "EE.CX.PW2.weight",
            "EE.out.weight",
            "E0.L0.FF1.IP.weight",
            "E5.L1.CM2.OP.bias",
            "E3.DS.bias",
            "E3.out_combiner.BY_scale",
            "encoder.DS_output.bias",
            "decoder.EMB.weight",
            "decoder.conv.weight",
            "joiner.encoder_proj.weight",
            "joiner.OL.weight",
            "joiner.OL.bias",
        ] {
            assert!(names.contains(name), "requirement set must contain {name}");
        }
    }

    /// Ground-truth set equality against a host-local imported pack (skipped
    /// when absent so weight-free CI stays green): the required set must equal
    /// the stored tensor set of the real ONNX-derived X-ASR pack, which is
    /// exactly the set the runtime loader reads.
    #[test]
    #[ignore = "host-local: validates against the imported ONNX X-ASR pack"]
    fn requirement_set_equals_the_real_pack_tensor_set() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-fp16.oasr");
        if !path.exists() {
            eprintln!("skipping: xasr ONNX fp16 pack absent at {}", path.display());
            return;
        }
        let index = crate::read_gguf_tensor_index(&path).expect("read pack tensor index");
        let pack_metadata = crate::ggml_runtime::read_gguf_metadata(&path).expect("metadata");
        let parsed = parse_xasr_zipformer_execution_metadata(&pack_metadata).expect("parse");
        let required: std::collections::BTreeSet<String> =
            xasr_zipformer_runtime_tensor_requirements(&parsed)
                .iter()
                .map(|requirement| compact_xasr_name(&requirement.upstream_name))
                .collect();
        let stored: std::collections::BTreeSet<String> = index
            .tensors()
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        let missing: Vec<&String> = stored.difference(&required).collect();
        let extra: Vec<&String> = required.difference(&stored).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "requirement set and pack tensor set must be equal; \
             required-but-absent={missing:?} present-but-not-required={extra:?}"
        );
    }
}
