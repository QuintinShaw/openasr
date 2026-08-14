//! X-ASR Zipformer2 execution metadata parsed from the `.oasr` GGUF header,
//! plus the admission-time runtime tensor contract that proves the pack carries
//! every tensor the runtime will load (metadata-derived shapes checked against
//! the tensor index) before the pack is admitted.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_string_scalar, required_u64_scalar,
    u64_to_u32, u64_to_usize, validate_bounded_usize, validate_positive_usize,
};

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

/// The decoder's causal conv runs as a grouped convolution with exactly 128
/// groups: the loader and the graph hardcode the group count, so the stored
/// kernel's middle extent is `decoder_dim / 128` and the runtime tensor
/// contract pins the full `[context_size, decoder_dim / 128, decoder_dim]`
/// shape. Parsing rejects a decoder dim that cannot form those 128 groups.
pub(crate) const XASR_DECODER_CONV_GROUPS: usize = 128;

/// Architecture-constant input width of the `encoder_embed.out` projection
/// (the flattened conv-stem output the loader pins), carried by the runtime
/// tensor contract instead of duplicated loader literals.
pub(crate) const XASR_ENCODER_EMBED_INPUT_DIM: usize = 2432;

/// The exported X-ASR Zipformer encoder emits one transducer frame for every
/// four feature-hop frames. This is an architecture contract (the final
/// `downsample_output` width is four), not a tuning knob.
pub(crate) const XASR_OUTPUT_DOWNSAMPLING_FACTOR: usize = 4;

/// Architecture ceilings for pack-supplied geometry, with generous headroom
/// over the published checkpoint (6 stacks, 19 encoder layers, dims up to
/// 768, joiner 512, vocab 5000). They bound every contract-derived
/// arithmetic expression and the requirement count a malicious metadata set
/// can construct, so contract building stays allocation-bounded and
/// overflow-free on untrusted input; parse fails closed above them.
pub(crate) const XASR_MAX_NUM_STACKS: usize = 64;
pub(crate) const XASR_MAX_TOTAL_ENCODER_LAYERS: usize = 4096;
pub(crate) const XASR_MAX_ENCODER_DIM: usize = 65_536;
pub(crate) const XASR_MAX_HEAD_COUNT: usize = 1_024;
pub(crate) const XASR_MAX_HEAD_DIM: usize = 65_536;
pub(crate) const XASR_MAX_CNN_KERNEL: usize = 4_096;
pub(crate) const XASR_MAX_LEFT_CONTEXT: usize = 65_536;
pub(crate) const XASR_MAX_DOWNSAMPLING_FACTOR: usize = 1_024;
pub(crate) const XASR_MAX_FEATURE_DIM: usize = 4_096;
pub(crate) const XASR_MAX_DECODE_CHUNK_LEN: usize = 4_096;
pub(crate) const XASR_MAX_JOINER_DIM: usize = 65_536;
pub(crate) const XASR_MAX_VOCAB_SIZE: usize = 1_000_000;

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
    // Architecture ceilings: keep contract construction bounded and
    // overflow-free on untrusted metadata (fail closed above them).
    validate_bounded_usize(num_stacks, XASR_NUM_STACKS_KEY, XASR_MAX_NUM_STACKS)?;
    for (key, values, max) in [
        (
            XASR_NUM_ENCODER_LAYERS_KEY,
            &num_encoder_layers,
            XASR_MAX_TOTAL_ENCODER_LAYERS,
        ),
        (XASR_ENCODER_DIMS_KEY, &encoder_dims, XASR_MAX_ENCODER_DIM),
        (
            XASR_QUERY_HEAD_DIMS_KEY,
            &query_head_dims,
            XASR_MAX_HEAD_DIM,
        ),
        (
            XASR_VALUE_HEAD_DIMS_KEY,
            &value_head_dims,
            XASR_MAX_HEAD_DIM,
        ),
        (XASR_NUM_HEADS_KEY, &num_heads, XASR_MAX_HEAD_COUNT),
        (
            XASR_CNN_MODULE_KERNELS_KEY,
            &cnn_module_kernels,
            XASR_MAX_CNN_KERNEL,
        ),
        (
            XASR_LEFT_CONTEXT_LEN_KEY,
            &left_context_len,
            XASR_MAX_LEFT_CONTEXT,
        ),
        (
            XASR_DOWNSAMPLING_FACTORS_KEY,
            &downsampling_factors,
            XASR_MAX_DOWNSAMPLING_FACTOR,
        ),
    ] {
        for value in values {
            validate_bounded_usize(*value, key, max)?;
        }
    }
    // Bound the total encoder layer count (and therefore the requirement
    // enumeration size) with checked accumulation, fail-closed.
    let total_layers = num_encoder_layers
        .iter()
        .try_fold(0usize, |total, layers| total.checked_add(*layers))
        .ok_or_else(|| MetadataContractError::InvalidValue {
            key: XASR_NUM_ENCODER_LAYERS_KEY,
            reason: "total encoder layer count overflows".to_string(),
        })?;
    validate_bounded_usize(
        total_layers,
        XASR_NUM_ENCODER_LAYERS_KEY,
        XASR_MAX_TOTAL_ENCODER_LAYERS,
    )?;
    for (key, value, max) in [
        (XASR_FEATURE_DIM_KEY, feature_dim, XASR_MAX_FEATURE_DIM),
        (
            XASR_DECODE_CHUNK_LEN_KEY,
            decode_chunk_len,
            XASR_MAX_DECODE_CHUNK_LEN,
        ),
        (XASR_JOINER_DIM_KEY, joiner_dim, XASR_MAX_JOINER_DIM),
        (XASR_VOCAB_SIZE_KEY, vocab_size, XASR_MAX_VOCAB_SIZE),
    ] {
        validate_bounded_usize(value, key, max)?;
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
    // The decoder causal conv is a grouped convolution with a hardcoded group
    // count; a decoder dim that cannot form those groups has no loadable
    // kernel shape, so fail closed at parse time instead of mid-load.
    if joiner_dim % XASR_DECODER_CONV_GROUPS != 0 {
        return Err(MetadataContractError::InvalidValue {
            key: XASR_JOINER_DIM_KEY,
            reason: format!(
                "joiner_dim {joiner_dim} must be a multiple of the decoder conv group count \
                 {XASR_DECODER_CONV_GROUPS}"
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
/// admission AND at load time: shapes fully determined by the parsed metadata
/// (or by an architecture constant the loader pins) are carried as exact dims;
/// data-derived dims the loader only learns by reading bytes keep the partial
/// pin the loader applies (the loader still enforces its data-derived
/// cross-checks at runtime).
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

impl XasrTensorShape {
    /// Check stored dims against this shape at the precision it declares.
    pub(crate) fn matches(&self, dims: &[usize]) -> bool {
        match self {
            XasrTensorShape::Exact(expected) => dims == expected.as_slice(),
            XasrTensorShape::Vector(len) => dims.len() == 1 && dims[0] == *len,
            XasrTensorShape::Rank(rank) => dims.len() == *rank,
            XasrTensorShape::Rank2In { input_dim } => dims.len() == 2 && dims[0] == *input_dim,
            XasrTensorShape::Rank2Out { output_dim } => dims.len() == 2 && dims[1] == *output_dim,
        }
    }

    /// Human-readable rendering of the expectation, for fail-closed errors.
    pub(crate) fn describe(&self) -> String {
        match self {
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
        }
    }

    /// The exact dims an `Exact` shape pins, when this shape is one. The
    /// weight loader uses this to source its pinned-tensor expectations from
    /// the contract instead of duplicating them.
    pub(crate) fn exact_dims(&self) -> Option<&[usize]> {
        match self {
            XasrTensorShape::Exact(expected) => Some(expected),
            _ => None,
        }
    }
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
    let dims: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();
    if requirement.shape.matches(&dims) {
        return Ok(());
    }
    Err(invalid_shape(
        &tensor.name,
        &tensor.dims,
        requirement.shape.describe(),
    ))
}

/// The resolved per-pack-name view of the single runtime tensor contract
/// ([`xasr_zipformer_runtime_tensor_requirements`]). The weight loaders
/// consume this table: every tensor they read must be present in it and must
/// satisfy the shape it pins, so the requirement enumeration is the loader's
/// authoritative read list instead of a parallel set of loader literals. A
/// read the contract does not cover fails closed.
#[derive(Debug, Clone)]
pub(crate) struct XasrRuntimeTensorContract {
    by_pack_name: std::collections::BTreeMap<String, XasrTensorShape>,
}

impl XasrRuntimeTensorContract {
    pub(crate) fn for_metadata(metadata: &XasrZipformerExecutionMetadata) -> Self {
        let mut by_pack_name = std::collections::BTreeMap::new();
        for requirement in xasr_zipformer_runtime_tensor_requirements(metadata) {
            let name = compact_xasr_name(&requirement.upstream_name);
            if by_pack_name
                .insert(name.clone(), requirement.shape)
                .is_some()
            {
                unreachable!("requirement names are unique per the contract enumeration");
            }
        }
        Self { by_pack_name }
    }

    /// The shape contract one pack tensor must satisfy, by its compacted pack
    /// name. `None` means the tensor is not part of the runtime contract and
    /// a loader reading it must fail closed.
    pub(crate) fn shape(&self, pack_name: &str) -> Option<&XasrTensorShape> {
        self.by_pack_name.get(pack_name)
    }

    /// The pinned exact dims for one tensor; loader expectations for
    /// exact-pinned tensors are sourced here. Fails closed when the tensor is
    /// absent from the contract or its shape is not `Exact`.
    pub(crate) fn exact_dims(&self, pack_name: &str) -> Result<Vec<usize>, String> {
        let shape = self.shape(pack_name).ok_or_else(|| {
            format!("tensor '{pack_name}' is not part of the xasr-zipformer runtime contract")
        })?;
        shape.exact_dims().map(|dims| dims.to_vec()).ok_or_else(|| {
            format!(
                "tensor '{pack_name}' contract shape is not exact: {}",
                shape.describe()
            )
        })
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn names(&self) -> impl Iterator<Item = &String> {
        self.by_pack_name.keys()
    }
}

fn embed_requirements(
    out: &mut Vec<XasrTensorRequirement>,
    metadata: &XasrZipformerExecutionMetadata,
) {
    let first_dim = metadata.encoder_dims[0];
    // Conv stem kernels are architecture constants; the contract carries the
    // loader's exact pinned shapes so admission and the loader enforce one
    // geometry from one enumeration.
    for (weight, dims, bias_len) in [
        ("encoder_embed.conv.0", [3usize, 3, 1, 8], 8usize),
        ("encoder_embed.conv.4", [3, 3, 8, 32], 32),
        ("encoder_embed.conv.7", [3, 3, 32, 128], 128),
        ("encoder_embed.convnext.depthwise_conv", [7, 7, 1, 128], 128),
        (
            "encoder_embed.convnext.pointwise_conv1",
            [1, 1, 128, 384],
            384,
        ),
        (
            "encoder_embed.convnext.pointwise_conv2",
            [1, 1, 384, 128],
            128,
        ),
    ] {
        out.push(requirement(
            format!("{weight}.weight"),
            XasrTensorShape::Exact(dims.to_vec()),
        ));
        out.push(requirement(
            format!("{weight}.bias"),
            XasrTensorShape::Vector(bias_len),
        ));
    }
    // `encoder_embed.out` maps the flattened conv-stem output (an
    // architecture constant) into `first_dim`; both extents are pinned.
    out.push(requirement(
        "encoder_embed.out.weight".to_string(),
        XasrTensorShape::Exact(vec![XASR_ENCODER_EMBED_INPUT_DIM, first_dim]),
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

    // GLU gate width; parsing caps `dim` well below overflow, so the
    // saturating product is defense in depth that stays fail-closed at
    // validation (no pack tensor can match a saturated extent).
    let glu_dim = dim.saturating_mul(2);
    for name in ["conv_module1", "conv_module2"] {
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.weight"),
            XasrTensorShape::Exact(vec![dim, glu_dim]),
        ));
        out.push(requirement(
            format!("{prefix}.{name}.in_proj.bias"),
            XasrTensorShape::Vector(glu_dim),
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
    // The conv kernel's middle dim is `decoder_dim / 128`: the grouped
    // convolution's group count is an architecture constant the loader and
    // the graph hardcode. Parsing already rejects a decoder dim that cannot
    // form the groups, so the contract pins the exact loader-enforced shape
    // (admission and the loader admit a pack on one geometry).
    out.push(requirement(
        "decoder.conv.weight".to_string(),
        XasrTensorShape::Exact(vec![
            metadata.decoder_context_size,
            decoder_dim / XASR_DECODER_CONV_GROUPS,
            decoder_dim,
        ]),
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

/// Projects one loadable dims choice per requirement for a pack the runtime
/// weight loaders can read end to end: exact-pinned tensors take their pinned
/// dims, and data-derived tensors the loader learns from the pack get one
/// self-consistent assignment per role (e.g. a feed-forward `in_proj.bias`
/// length and the matching `out_proj.weight` input extent agree). The
/// full-load access-trace equivalence test stamps a synthetic pack with
/// exactly this set, runs the real loaders, and proves their read set equals
/// the requirement set name for name and shape for shape.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn xasr_zipformer_loader_ready_runtime_tensors(
    metadata: &XasrZipformerExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    xasr_zipformer_runtime_tensor_requirements(metadata)
        .iter()
        .map(|requirement| {
            let dims = loader_ready_dims(metadata, requirement);
            (compact_xasr_name(&requirement.upstream_name), dims)
        })
        .collect()
}

#[cfg(any(test, feature = "testing"))]
fn loader_ready_dims(
    metadata: &XasrZipformerExecutionMetadata,
    requirement: &XasrTensorRequirement,
) -> Vec<u64> {
    let upstream = requirement.upstream_name.as_str();
    // Data-derived widths, one consistent assignment per tensor role. The
    // stack dim / query width come from the requirement's own stack scope.
    let stack_scope = upstream
        .strip_prefix("encoder.encoders.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|stack| stack.parse::<usize>().ok())
        .filter(|stack| *stack < metadata.num_stacks);
    if let Some(stack) = stack_scope {
        let dim = metadata.encoder_dims[stack] as u64;
        let query_dim = (metadata.num_heads[stack] * metadata.query_head_dims[stack]) as u64;
        // Feed-forward: the bias length IS the hidden width the loader
        // derives; keep it 2 for every stack and match the projections.
        if upstream.ends_with(".in_proj.bias") && upstream.contains(".feed_forward") {
            return vec![2];
        }
        if upstream.contains(".feed_forward") {
            if upstream.ends_with(".in_proj.weight") {
                return vec![dim, 2];
            }
            if upstream.ends_with(".out_proj.weight") {
                return vec![2, dim];
            }
        }
        // Relative-position projection: the loader derives its output width
        // from the pack; the qkv in_proj must then span 2*query_dim + that
        // width, so the two roles share one choice (output width 2).
        if upstream.ends_with(".self_attn_weights.linear_pos.weight") {
            return vec![2, 2];
        }
        if upstream.ends_with(".self_attn_weights.in_proj.bias") {
            return vec![2 * query_dim + 2];
        }
        if upstream.ends_with(".self_attn_weights.in_proj.weight") {
            return vec![dim, 2 * query_dim + 2];
        }
        // Value projections: bias length = value width = 2 everywhere.
        if upstream.contains(".self_attn") {
            if upstream.ends_with(".in_proj.bias") {
                return vec![2];
            }
            if upstream.ends_with(".in_proj.weight") {
                return vec![dim, 2];
            }
            if upstream.ends_with(".out_proj.weight") {
                return vec![2, dim];
            }
        }
        // Nonlinear attention: in_proj output must be divisible by 3; the
        // out_proj input is that quotient.
        if upstream.contains(".nonlin_attention.") {
            if upstream.ends_with(".in_proj.bias") {
                return vec![3];
            }
            if upstream.ends_with(".in_proj.weight") {
                return vec![dim, 3];
            }
            if upstream.ends_with(".out_proj.weight") {
                return vec![1, dim];
            }
        }
    }
    // Everything else is pinned by its requirement shape.
    match &requirement.shape {
        XasrTensorShape::Exact(expected) => expected.iter().map(|dim| *dim as u64).collect(),
        XasrTensorShape::Vector(len) => vec![*len as u64],
        XasrTensorShape::Rank(1) => vec![2],
        XasrTensorShape::Rank(rank) => {
            let mut dims = vec![1_u64; *rank];
            *dims.last_mut().expect("rank > 0") = 2;
            dims
        }
        XasrTensorShape::Rank2In { input_dim } => vec![*input_dim as u64, 2],
        XasrTensorShape::Rank2Out { output_dim } => vec![2, *output_dim as u64],
    }
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

    /// Shape pins the incident review found loose: on the full production
    /// geometry the contract must carry the LOADER's exact shapes for the
    /// tensors a rank check used to admit structurally wrong packs for --
    /// the grouped decoder conv kernel (including its `decoder_dim / 128`
    /// middle group extent), every conv-stem kernel, and the embed output
    /// projection. The full-load access-trace test below proves the whole
    /// enumeration equals the loader read set; this test holds the exact
    /// production values that enumeration produces.
    #[test]
    fn production_geometry_contract_pins_the_loader_enforced_shapes() {
        let parsed = parse_xasr_zipformer_execution_metadata(&metadata()).expect("parse");
        let contract = XasrRuntimeTensorContract::for_metadata(&parsed);
        // The published X-ASR pack ships exactly 966 tensors; the contract
        // enumeration covers every one of them (see the ONNX round-trip
        // import test and the ignored real-pack equality test).
        assert_eq!(
            contract.names().count(),
            966,
            "the contract must stay exactly the loader read set on production geometry"
        );
        let exact = |name: &str| -> Vec<usize> {
            contract
                .exact_dims(name)
                .unwrap_or_else(|reason| panic!("{name} must be exact-pinned: {reason}"))
        };
        assert_eq!(exact("decoder.conv.weight"), vec![2, 512 / 128, 512]);
        assert_eq!(exact("decoder.EMB.weight"), vec![512, 5000]);
        assert_eq!(exact("EE.conv.0.weight"), vec![3, 3, 1, 8]);
        assert_eq!(exact("EE.conv.4.weight"), vec![3, 3, 8, 32]);
        assert_eq!(exact("EE.conv.7.weight"), vec![3, 3, 32, 128]);
        assert_eq!(exact("EE.CX.DW.weight"), vec![7, 7, 1, 128]);
        assert_eq!(exact("EE.CX.PW1.weight"), vec![1, 1, 128, 384]);
        assert_eq!(exact("EE.CX.PW2.weight"), vec![1, 1, 384, 128]);
        assert_eq!(exact("EE.out.weight"), vec![2432, 192]);
        assert_eq!(exact("joiner.encoder_proj.weight"), vec![768, 512]);
        assert_eq!(exact("joiner.OL.weight"), vec![512, 5000]);
    }

    /// A decoder dim that cannot form the conv's 128 groups has no loadable
    /// kernel shape; parsing fails closed instead of loading a group count 0.
    #[test]
    fn rejects_decoder_dim_not_divisible_by_the_conv_group_count() {
        let mut metadata = metadata();
        metadata.insert(XASR_JOINER_DIM_KEY.to_string(), "100".to_string());
        let error = parse_xasr_zipformer_execution_metadata(&metadata)
            .expect_err("joiner_dim 100 cannot form 128 conv groups");
        assert!(matches!(
            error,
            MetadataContractError::InvalidValue {
                key: XASR_JOINER_DIM_KEY,
                ..
            }
        ));
    }

    /// Architecture ceilings fail closed on untrusted metadata, keeping
    /// contract construction allocation-bounded and overflow-free.
    #[test]
    fn rejects_geometry_above_architecture_ceilings() {
        let base = metadata();
        for (key, value) in [
            (
                XASR_FEATURE_DIM_KEY,
                (XASR_MAX_FEATURE_DIM as u64 + 1).to_string(),
            ),
            (
                XASR_DECODE_CHUNK_LEN_KEY,
                (XASR_MAX_DECODE_CHUNK_LEN as u64 + 1).to_string(),
            ),
            (
                XASR_JOINER_DIM_KEY,
                (XASR_MAX_JOINER_DIM as u64 + 128).to_string(),
            ),
            (
                XASR_VOCAB_SIZE_KEY,
                (XASR_MAX_VOCAB_SIZE as u64 + 1).to_string(),
            ),
        ] {
            let mut metadata = base.clone();
            metadata.insert(key.to_string(), value);
            assert!(
                parse_xasr_zipformer_execution_metadata(&metadata).is_err(),
                "must reject {key} above its ceiling"
            );
        }
        // List-valued ceilings.
        let mut metadata = base.clone();
        metadata.insert(
            XASR_ENCODER_DIMS_KEY.to_string(),
            "192,256,512,768,512,65537".to_string(),
        );
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
        // Total encoder layer count ceiling.
        let mut metadata = base.clone();
        metadata.insert(
            XASR_NUM_ENCODER_LAYERS_KEY.to_string(),
            "2048,2047,1,1,1,1".to_string(),
        );
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_err());
    }

    /// Boundary: geometry exactly at the ceilings stays admissible (the
    /// ceilings bound, they do not shrink the production envelope).
    #[test]
    fn accepts_geometry_at_the_architecture_ceilings() {
        let mut metadata = metadata();
        metadata.insert(
            XASR_FEATURE_DIM_KEY.to_string(),
            XASR_MAX_FEATURE_DIM.to_string(),
        );
        metadata.insert(
            XASR_DECODE_CHUNK_LEN_KEY.to_string(),
            XASR_MAX_DECODE_CHUNK_LEN.to_string(),
        );
        metadata.insert(
            XASR_VOCAB_SIZE_KEY.to_string(),
            XASR_MAX_VOCAB_SIZE.to_string(),
        );
        assert!(parse_xasr_zipformer_execution_metadata(&metadata).is_ok());
    }

    /// The equivalence evidence the count-plus-sampling pin used to fake: run
    /// the REAL weight loaders (encoder + decoder + joiner) over a synthetic
    /// pack whose tensor set is projected from the requirement enumeration,
    /// with the tensor index's access trace enabled, and assert the recorded
    /// read set equals the requirement set name for name and shape for shape.
    /// Any drift -- a loader reading a tensor the contract does not list, a
    /// contract entry no loader reads, or a read whose shape violates the
    /// contract's precision -- fails here.
    #[test]
    fn full_loader_read_trace_equals_the_requirement_set() {
        use crate::ggml_runtime::GgufTensorDataReader;
        use crate::models::xasr_zipformer::encoder_weights::load_xasr_encoder_weights;
        use crate::models::xasr_zipformer::weights::{
            load_xasr_decoder_weights, load_xasr_joiner_weights,
        };
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

        let parsed =
            parse_xasr_zipformer_execution_metadata(&trace_geometry_metadata()).expect("parse");
        let tensors = xasr_zipformer_loader_ready_runtime_tensors(&parsed);
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("xasr-loader-trace.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in tensors {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write trace pack");

        let reader = GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();
        load_xasr_encoder_weights(&reader, &parsed).expect("full encoder load");
        load_xasr_decoder_weights(&reader, &parsed).expect("full decoder load");
        load_xasr_joiner_weights(&reader, &parsed).expect("full joiner load");

        assert_trace_equals_the_requirement_set(&reader.tensor_index().access_trace(), &parsed);
    }

    /// Two stacks (covering the stack-0 vs stack-N layer-prefix split and the
    /// per-stack downsample/combiner tensors), two layers in stack 0, and
    /// distinct per-stack dims/kernels so every name template and every
    /// metadata-derived shape branch runs in the traced full load.
    fn trace_geometry_metadata() -> BTreeMap<String, String> {
        [
            (XASR_NUM_STACKS_KEY, "2"),
            (XASR_NUM_ENCODER_LAYERS_KEY, "2,1"),
            (XASR_ENCODER_DIMS_KEY, "16,24"),
            (XASR_QUERY_HEAD_DIMS_KEY, "4,4"),
            (XASR_VALUE_HEAD_DIMS_KEY, "4,4"),
            (XASR_NUM_HEADS_KEY, "2,2"),
            (XASR_CNN_MODULE_KERNELS_KEY, "3,5"),
            (XASR_LEFT_CONTEXT_LEN_KEY, "4,4"),
            (XASR_DOWNSAMPLING_FACTORS_KEY, "1,2"),
            (XASR_FEATURE_DIM_KEY, "80"),
            (XASR_DECODE_CHUNK_LEN_KEY, "4"),
            (XASR_JOINER_DIM_KEY, "128"),
            (XASR_DECODER_CONTEXT_SIZE_KEY, "2"),
            (XASR_VOCAB_SIZE_KEY, "7"),
            (XASR_BLANK_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Compare a traced full load against the requirement enumeration:
    /// exact name-set equality both directions, then each traced read's dims
    /// must satisfy its requirement's shape at the precision it declares.
    pub(crate) fn assert_trace_equals_the_requirement_set(
        trace: &[crate::ggml_runtime::GgufTensorAccessRecord],
        metadata: &XasrZipformerExecutionMetadata,
    ) {
        let requirements = xasr_zipformer_runtime_tensor_requirements(metadata);
        let mut required: std::collections::BTreeMap<String, &XasrTensorShape> =
            std::collections::BTreeMap::new();
        for requirement in &requirements {
            let name = compact_xasr_name(&requirement.upstream_name);
            if required.insert(name.clone(), &requirement.shape).is_some() {
                panic!("requirement names must be unique: {name}");
            }
        }
        let mut traced: std::collections::BTreeMap<String, Vec<u64>> =
            std::collections::BTreeMap::new();
        for record in trace {
            if let Some(previous) = traced.get(&record.name) {
                assert_eq!(
                    previous, &record.dims,
                    "traced dims for '{}' must be stable across reads",
                    record.name
                );
            } else {
                traced.insert(record.name.clone(), record.dims.clone());
            }
        }
        let missing: Vec<&String> = required
            .keys()
            .filter(|name| !traced.contains_key(name.as_str()))
            .collect();
        let extra: Vec<&String> = traced
            .keys()
            .filter(|name| !required.contains_key(name.as_str()))
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "loader read set and requirement set must be equal; \
             required-but-never-read={missing:?} read-but-not-required={extra:?}"
        );
        for (name, shape) in &required {
            let dims = &traced[name];
            let dims_usize: Vec<usize> = dims.iter().map(|&dim| dim as usize).collect();
            assert!(
                shape.matches(&dims_usize),
                "loader read '{name}' with dims {dims:?}, but the contract says {}",
                shape.describe()
            );
        }
    }

    /// Ground-truth set equality against a host-local imported pack (skipped
    /// when absent so weight-free CI stays green): the required set must equal
    /// the stored tensor set of a real X-ASR pack name for name, and every
    /// stored tensor must satisfy its requirement's shape at the precision the
    /// requirement declares -- the runtime loader reads exactly this set.
    #[test]
    #[ignore = "host-local: validates against an imported X-ASR pack"]
    fn requirement_set_equals_the_real_pack_tensor_set() {
        let candidates = [
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tmp/family-migration-batch1/pack-regress/xasr-zh-en-q8.oasr"),
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-fp16.oasr"),
        ];
        let Some(path) = candidates.iter().find(|path| path.exists()) else {
            eprintln!("skipping: no host-local xasr pack present at {candidates:?}");
            return;
        };
        let index = crate::read_gguf_tensor_index(path).expect("read pack tensor index");
        let pack_metadata = crate::ggml_runtime::read_gguf_metadata(path).expect("metadata");
        let parsed = parse_xasr_zipformer_execution_metadata(&pack_metadata).expect("parse");
        let requirements = xasr_zipformer_runtime_tensor_requirements(&parsed);
        let required: std::collections::BTreeSet<String> = requirements
            .iter()
            .map(|requirement| compact_xasr_name(&requirement.upstream_name))
            .collect();
        let stored: std::collections::BTreeSet<String> = index
            .tensors()
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        let missing: Vec<&String> = required.difference(&stored).collect();
        let extra: Vec<&String> = stored.difference(&required).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "requirement set and pack tensor set must be equal; \
             required-but-absent={missing:?} present-but-not-required={extra:?}"
        );
        for requirement in &requirements {
            let name = compact_xasr_name(&requirement.upstream_name);
            let tensor = index.get(&name).expect("required tensor present");
            let dims: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();
            assert!(
                requirement.shape.matches(&dims),
                "pack tensor '{name}' has dims {:?}, but the contract says {}",
                tensor.dims,
                requirement.shape.describe()
            );
        }
    }
}
