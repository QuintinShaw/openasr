//! sensevoice execution metadata + runtime tensor contract parsed from the
//! `.oasr` GGUF header. The validator is depth-complete: a pack must satisfy
//! the metadata contract AND the full runtime tensor binding (every tensor the
//! executor loads, with the shapes the SAN-M graph consumes) before it can be
//! admitted, so a malformed pack fails closed at verification instead of deep
//! inside the weight loader.

use crate::GgufTensorIndex;
use crate::ggml_runtime::GgufMetadata;
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

pub(crate) const SENSEVOICE_N_LAYERS_KEY: &str = "sensevoice.n_layers";
pub(crate) const SENSEVOICE_TP_LAYERS_KEY: &str = "sensevoice.tp_layers";
pub(crate) const SENSEVOICE_D_MODEL_KEY: &str = "sensevoice.d_model";
pub(crate) const SENSEVOICE_N_HEADS_KEY: &str = "sensevoice.n_heads";
pub(crate) const SENSEVOICE_FFN_DIM_KEY: &str = "sensevoice.ffn_dim";
pub(crate) const SENSEVOICE_FSMN_KERNEL_KEY: &str = "sensevoice.fsmn_kernel";
pub(crate) const SENSEVOICE_FEATURE_DIM_KEY: &str = "sensevoice.feature_dim";
pub(crate) const SENSEVOICE_VOCAB_SIZE_KEY: &str = "sensevoice.vocab_size";
pub(crate) const SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";
const SENSEVOICE_TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// Rows the prompt-embedding table must have at minimum: the prompt builder
/// addresses indices 0..=15 (`auto`, event/emotion queries, the six language
/// ids, and both textnorm slots; see `language.rs`).
pub(crate) const SENSEVOICE_PROMPT_EMBED_MIN_ROWS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SenseVoiceExecutionMetadata {
    /// SAN-M encoder blocks: `enc.blk.0` (the 560-dim input layer) .. `enc.blk.{n-1}`.
    pub n_layers: usize,
    /// `tp.blk.*` blocks after `enc.after_norm`.
    pub tp_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub fsmn_kernel: usize,
    /// LFR-stacked input feature dim (80 * 7 = 560), also the prompt-embed dim.
    pub feature_dim: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
}

/// Fail-closed tensor-contract errors, surfaced by the pack verifier before a
/// sensevoice pack can be admitted.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum SenseVoiceRuntimeContractError {
    #[error("sensevoice runtime tensor contract is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("sensevoice runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

fn missing_required_tensor(name: &str) -> SenseVoiceRuntimeContractError {
    SenseVoiceRuntimeContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> SenseVoiceRuntimeContractError {
    SenseVoiceRuntimeContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

pub(crate) fn parse_sensevoice_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<SenseVoiceExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(SENSEVOICE_N_LAYERS_KEY)?;
    let tp_layers = usize_key(SENSEVOICE_TP_LAYERS_KEY)?;
    let d_model = usize_key(SENSEVOICE_D_MODEL_KEY)?;
    let n_heads = usize_key(SENSEVOICE_N_HEADS_KEY)?;
    let ffn_dim = usize_key(SENSEVOICE_FFN_DIM_KEY)?;
    let fsmn_kernel = usize_key(SENSEVOICE_FSMN_KERNEL_KEY)?;
    let feature_dim = usize_key(SENSEVOICE_FEATURE_DIM_KEY)?;
    let vocab_size = usize_key(SENSEVOICE_VOCAB_SIZE_KEY)?;
    let blank_token_id = u64_to_u32(
        required_u64_scalar(metadata, SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY)?,
        SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY,
    )?;

    for (key, value) in [
        (SENSEVOICE_N_LAYERS_KEY, n_layers),
        (SENSEVOICE_TP_LAYERS_KEY, tp_layers),
        (SENSEVOICE_D_MODEL_KEY, d_model),
        (SENSEVOICE_N_HEADS_KEY, n_heads),
        (SENSEVOICE_FFN_DIM_KEY, ffn_dim),
        (SENSEVOICE_FSMN_KERNEL_KEY, fsmn_kernel),
        (SENSEVOICE_FEATURE_DIM_KEY, feature_dim),
        (SENSEVOICE_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    if (blank_token_id as usize) >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY,
            reason: format!("blank {blank_token_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if !d_model.is_multiple_of(n_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: SENSEVOICE_N_HEADS_KEY,
            reason: format!("n_heads {n_heads} does not divide d_model {d_model}"),
        });
    }
    if fsmn_kernel.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: SENSEVOICE_FSMN_KERNEL_KEY,
            reason: format!("fsmn kernel {fsmn_kernel} must be odd (symmetric sanm_shift 0 pad)"),
        });
    }

    Ok(SenseVoiceExecutionMetadata {
        n_layers,
        tp_layers,
        d_model,
        n_heads,
        head_dim: d_model / n_heads,
        ffn_dim,
        fsmn_kernel,
        feature_dim,
        vocab_size,
        blank_token_id,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockScope {
    Encoder,
    TwoPass,
}

impl BlockScope {
    fn tensor_name_scope(self) -> &'static str {
        match self {
            BlockScope::Encoder => "enc.blk",
            BlockScope::TwoPass => "tp.blk",
        }
    }
}

/// The input width of one SAN-M block: `enc.blk.0` consumes the LFR+prompt
/// feature width, every other block (including all `tp.blk.*`) consumes
/// `d_model`.
fn block_input_dim(
    metadata: &SenseVoiceExecutionMetadata,
    scope: BlockScope,
    layer: usize,
) -> usize {
    match scope {
        BlockScope::Encoder if layer == 0 => metadata.feature_dim,
        BlockScope::Encoder | BlockScope::TwoPass => metadata.d_model,
    }
}

fn descriptor(
    tensor_name: String,
    requirement: TensorBindingDescriptorRequirement,
    reason: &str,
) -> TensorBindingDescriptor {
    TensorBindingDescriptor {
        tensor_name,
        requirement,
        reason: reason.to_string(),
    }
}

/// One SAN-M block's runtime tensor bindings (the 13 tensors
/// `encoder_weights::load_layer` reads and `nn::encoder::sanm_fsmn_encoder_layer`
/// consumes), shaped for the block's `input_dim`.
fn block_tensor_descriptors(
    metadata: &SenseVoiceExecutionMetadata,
    scope: BlockScope,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    let input_dim = block_input_dim(metadata, scope, layer);
    let d_model = metadata.d_model;
    let qkv_dim = 3 * d_model;
    let name = |suffix: &str| format!("{}.{}.{suffix}", scope.tensor_name_scope(), layer);
    vec![
        descriptor(
            name("attn.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(input_dim),
            "pre-attention LayerNorm gamma must span the block input width",
        ),
        descriptor(
            name("attn.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(input_dim),
            "pre-attention LayerNorm beta must span the block input width",
        ),
        descriptor(
            name("attn.qkv.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(input_dim, qkv_dim),
            "fused QKV projection must map the block input width to 3*d_model",
        ),
        descriptor(
            name("attn.qkv.bias"),
            TensorBindingDescriptorRequirement::VectorLen(qkv_dim),
            "fused QKV bias must span 3*d_model",
        ),
        descriptor(
            name("attn.out.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, d_model),
            "attention output projection must be d_model x d_model",
        ),
        descriptor(
            name("attn.out.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "attention output bias must span d_model",
        ),
        descriptor(
            name("attn.fsmn.weight"),
            TensorBindingDescriptorRequirement::ExactDims(vec![metadata.fsmn_kernel, 1, d_model]),
            "FSMN depthwise kernel must be [fsmn_kernel, 1, d_model] for the im2col conv path",
        ),
        descriptor(
            name("ffn.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm gamma must span d_model",
        ),
        descriptor(
            name("ffn.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "pre-FFN LayerNorm beta must span d_model",
        ),
        descriptor(
            name("ffn.up.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, metadata.ffn_dim),
            "FFN up projection must map d_model to ffn_dim",
        ),
        descriptor(
            name("ffn.up.bias"),
            TensorBindingDescriptorRequirement::VectorLen(metadata.ffn_dim),
            "FFN up bias must span ffn_dim",
        ),
        descriptor(
            name("ffn.down.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(metadata.ffn_dim, d_model),
            "FFN down projection must map ffn_dim to d_model",
        ),
        descriptor(
            name("ffn.down.bias"),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "FFN down bias must span d_model",
        ),
    ]
}

/// The runtime tensor contract for one sensevoice pack: every tensor the
/// executor materializes (`encoder_weights` plus the frontend CMVN/prompt
/// tables), with the exact shapes the SAN-M/FSMN graph and the CTC head
/// consume. Derived from the parsed metadata, so a checkpoint with different
/// layer counts validates its own geometry.
pub(crate) fn sensevoice_runtime_tensor_binding_descriptors(
    metadata: SenseVoiceExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors = Vec::new();
    for layer in 0..metadata.n_layers {
        descriptors.extend(block_tensor_descriptors(
            &metadata,
            BlockScope::Encoder,
            layer,
        ));
    }
    for layer in 0..metadata.tp_layers {
        descriptors.extend(block_tensor_descriptors(
            &metadata,
            BlockScope::TwoPass,
            layer,
        ));
    }
    let d_model = metadata.d_model;
    descriptors.extend([
        descriptor(
            "enc.after_norm.weight".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "encoder tail LayerNorm gamma must span d_model",
        ),
        descriptor(
            "enc.after_norm.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "encoder tail LayerNorm beta must span d_model",
        ),
        descriptor(
            "tp.norm.weight".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "two-pass tail LayerNorm gamma must span d_model",
        ),
        descriptor(
            "tp.norm.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "two-pass tail LayerNorm beta must span d_model",
        ),
        descriptor(
            "ctc.head.weight".to_string(),
            TensorBindingDescriptorRequirement::Rank2EitherDims(d_model, metadata.vocab_size),
            "CTC head must project d_model to the vocab",
        ),
        descriptor(
            "ctc.head.bias".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(metadata.vocab_size),
            "CTC head bias must span the vocab",
        ),
        descriptor(
            "embed.prompt.weight".to_string(),
            TensorBindingDescriptorRequirement::Rank2WithDim(metadata.feature_dim),
            "prompt embedding rows must be feature_dim wide",
        ),
        descriptor(
            "frontend.cmvn.neg_mean".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(metadata.feature_dim),
            "CMVN neg-mean must span the LFR feature dim",
        ),
        descriptor(
            "frontend.cmvn.inv_stddev".to_string(),
            TensorBindingDescriptorRequirement::VectorLen(metadata.feature_dim),
            "CMVN inverse-stddev must span the LFR feature dim",
        ),
    ]);
    descriptors
}

/// Validate the full runtime tensor set against the pack's tensor index.
pub(crate) fn validate_sensevoice_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: SenseVoiceExecutionMetadata,
) -> Result<(), SenseVoiceRuntimeContractError> {
    let descriptors = sensevoice_runtime_tensor_binding_descriptors(metadata);
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )?;
    // The prompt table must hold every index the prompt builder can address
    // (the binding batch above only pins the row width).
    let embed = index.get("embed.prompt.weight").ok_or_else(|| {
        SenseVoiceRuntimeContractError::MissingRequiredTensor {
            name: "embed.prompt.weight".to_string(),
        }
    })?;
    let rows_ok = embed.dims.len() == 2 && {
        let lhs = embed.dims[0] as usize;
        let rhs = embed.dims[1] as usize;
        let has_feature_dim = lhs == metadata.feature_dim || rhs == metadata.feature_dim;
        // Rows are the dim that is not the row width; a square table keeps
        // either dim.
        let rows = if lhs == metadata.feature_dim && rhs != metadata.feature_dim {
            rhs
        } else if rhs == metadata.feature_dim && lhs != metadata.feature_dim {
            lhs
        } else {
            lhs.max(rhs)
        };
        has_feature_dim && rows >= SENSEVOICE_PROMPT_EMBED_MIN_ROWS
    };
    if !rows_ok {
        return Err(invalid_tensor_shape(
            "embed.prompt.weight",
            &embed.dims,
            format!(
                "prompt embedding table must have at least {SENSEVOICE_PROMPT_EMBED_MIN_ROWS} rows of feature_dim {}",
                metadata.feature_dim
            ),
        ));
    }
    Ok(())
}

/// The pack's embedded SentencePiece vocab must match the CTC vocab the
/// metadata declares (the importer pins them equal; a truncated or foreign
/// pack fails closed here instead of decoding out-of-range ids).
pub(crate) fn validate_sensevoice_tokenizer_contract(
    metadata: &GgufMetadata,
    vocab_size: usize,
) -> Result<(), MetadataContractError> {
    let tokens = metadata
        .get_string_array(SENSEVOICE_TOKENIZER_TOKENS_KEY)
        .ok_or(MetadataContractError::MissingRequiredKey {
            key: SENSEVOICE_TOKENIZER_TOKENS_KEY,
        })?;
    if tokens.is_empty() {
        return Err(MetadataContractError::InvalidValue {
            key: SENSEVOICE_TOKENIZER_TOKENS_KEY,
            reason: "tokenizer vocab is empty".to_string(),
        });
    }
    if tokens.len() != vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: SENSEVOICE_TOKENIZER_TOKENS_KEY,
            reason: format!(
                "tokenizer vocab has {} pieces but {SENSEVOICE_VOCAB_SIZE_KEY} declares vocab_size {vocab_size}",
                tokens.len()
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata = parse_sensevoice_execution_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("sensevoice", error)
    })?;
    validate_sensevoice_runtime_tensors_with_index(preflight.tensor_index(), metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)?;
    validate_sensevoice_tokenizer_contract(preflight.metadata(), metadata.vocab_size).map_err(
        |error| {
            crate::models::runtime_pack_contract::metadata_validation_error(
                "sensevoice tokenizer",
                error,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};
    use crate::testing::TinyGgufFixtureSpec;
    use std::collections::BTreeMap;

    /// Rebuild the fixture's GGUF metadata view (scalars as strings + the
    /// string-array keys) so the tokenizer contract can be checked against it.
    fn gguf_metadata_from_spec(spec: &TinyGgufFixtureSpec) -> GgufMetadata {
        let mut values: BTreeMap<String, GgufMetadataValue> = spec
            .metadata
            .iter()
            .map(|(key, value)| (key.clone(), GgufMetadataValue::String(value.clone())))
            .collect();
        for (key, entries) in &spec.metadata_string_arrays {
            values.insert(key.clone(), GgufMetadataValue::StringArray(entries.clone()));
        }
        GgufMetadata::from_values_for_test(values)
    }

    fn sensevoice_metadata() -> BTreeMap<String, String> {
        [
            (SENSEVOICE_N_LAYERS_KEY, "50"),
            (SENSEVOICE_TP_LAYERS_KEY, "20"),
            (SENSEVOICE_D_MODEL_KEY, "512"),
            (SENSEVOICE_N_HEADS_KEY, "4"),
            (SENSEVOICE_FFN_DIM_KEY, "2048"),
            (SENSEVOICE_FSMN_KERNEL_KEY, "11"),
            (SENSEVOICE_FEATURE_DIM_KEY, "560"),
            (SENSEVOICE_VOCAB_SIZE_KEY, "25055"),
            (SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY, "0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn tiny_execution_metadata() -> SenseVoiceExecutionMetadata {
        SenseVoiceExecutionMetadata {
            n_layers: 2,
            tp_layers: 1,
            d_model: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            fsmn_kernel: 5,
            feature_dim: 28,
            vocab_size: 12,
            blank_token_id: 0,
        }
    }

    #[test]
    fn parses_sensevoice_small_metadata() {
        let parsed = parse_sensevoice_execution_metadata(&sensevoice_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 50);
        assert_eq!(parsed.tp_layers, 20);
        assert_eq!(parsed.d_model, 512);
        assert_eq!(parsed.head_dim, 128);
        assert_eq!(parsed.vocab_size, 25055);
        assert_eq!(parsed.blank_token_id, 0);
    }

    #[test]
    fn rejects_blank_out_of_vocab() {
        let mut metadata = sensevoice_metadata();
        metadata.insert(
            SENSEVOICE_CTC_BLANK_TOKEN_ID_KEY.to_string(),
            "30000".to_string(),
        );
        assert!(parse_sensevoice_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_even_fsmn_kernel() {
        let mut metadata = sensevoice_metadata();
        metadata.insert(SENSEVOICE_FSMN_KERNEL_KEY.to_string(), "10".to_string());
        assert!(parse_sensevoice_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_heads_not_dividing_d_model() {
        let mut metadata = sensevoice_metadata();
        metadata.insert(SENSEVOICE_N_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_sensevoice_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_missing_key() {
        let mut metadata = sensevoice_metadata();
        metadata.remove(SENSEVOICE_N_LAYERS_KEY);
        assert!(parse_sensevoice_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn binding_descriptors_cover_every_runtime_tensor_exactly_once() {
        let descriptors = sensevoice_runtime_tensor_binding_descriptors(tiny_execution_metadata());
        // 13 tensors per SAN-M block (2 enc + 1 tp) + 9 tail tensors.
        assert_eq!(descriptors.len(), 13 * 3 + 9);
        let mut names = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            descriptors.len(),
            "every runtime tensor must be bound exactly once"
        );
        for required in [
            "enc.blk.0.attn.qkv.weight",
            "enc.blk.1.attn.fsmn.weight",
            "tp.blk.0.ffn.down.bias",
            "enc.after_norm.weight",
            "tp.norm.bias",
            "ctc.head.weight",
            "embed.prompt.weight",
            "frontend.cmvn.neg_mean",
            "frontend.cmvn.inv_stddev",
        ] {
            assert!(
                names.contains(&required),
                "binding list must cover {required}"
            );
        }
    }

    #[test]
    fn validates_runtime_ready_fixture_tensors() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec =
            TinyGgufFixtureSpec::sensevoice_oasr_v1_runtime_ready("sensevoice-runtime-fixture");
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_sensevoice_execution_metadata(&spec.metadata).expect("metadata must parse");
        validate_sensevoice_runtime_tensors_with_index(&index, metadata)
            .expect("runtime-ready tensor fixture must validate");
        validate_sensevoice_tokenizer_contract(
            &gguf_metadata_from_spec(&spec),
            metadata.vocab_size,
        )
        .expect("tokenizer vocab must match the declared vocab size");
    }

    #[test]
    fn rejects_fixture_missing_required_tensor() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = crate::testing::TinyGgufFixtureSpec::sensevoice_oasr_v1_runtime_ready(
            "sensevoice-runtime-fixture",
        )
        .without_tensor("tp.blk.0.attn.out.bias");
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_sensevoice_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_sensevoice_runtime_tensors_with_index(&index, metadata)
            .expect_err("missing tensor must fail closed");
        assert!(
            matches!(error, SenseVoiceRuntimeContractError::MissingRequiredTensor { ref name } if name == "tp.blk.0.attn.out.bias"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_fixture_shape_mismatch() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = crate::testing::TinyGgufFixtureSpec::sensevoice_oasr_v1_runtime_ready(
            "sensevoice-runtime-fixture",
        )
        .with_tensor_shape("enc.blk.0.attn.qkv.weight", [28_u64, 47_u64]);
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_sensevoice_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_sensevoice_runtime_tensors_with_index(&index, metadata)
            .expect_err("shape mismatch must fail closed");
        assert!(
            matches!(error, SenseVoiceRuntimeContractError::InvalidTensorShape { ref name, .. } if name == "enc.blk.0.attn.qkv.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_prompt_embed_with_too_few_rows() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let spec = crate::testing::TinyGgufFixtureSpec::sensevoice_oasr_v1_runtime_ready(
            "sensevoice-runtime-fixture",
        )
        .with_tensor_shape("embed.prompt.weight", [28_u64, 8_u64]);
        crate::testing::write_tiny_gguf_runtime_source(file.path(), &spec).expect("write");

        let index = crate::read_gguf_tensor_index(file.path()).expect("read tensor index");
        let metadata =
            parse_sensevoice_execution_metadata(&spec.metadata).expect("metadata must parse");
        let error = validate_sensevoice_runtime_tensors_with_index(&index, metadata)
            .expect_err("short prompt table must fail closed");
        assert!(
            matches!(error, SenseVoiceRuntimeContractError::InvalidTensorShape { ref name, .. } if name == "embed.prompt.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_tokenizer_vocab_size_drift() {
        // No tokenizer.ggml.tokens key at all: fail closed.
        let values = sensevoice_metadata()
            .into_iter()
            .map(|(key, value)| (key, GgufMetadataValue::String(value)))
            .collect();
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = validate_sensevoice_tokenizer_contract(&metadata, 25055)
            .expect_err("missing vocab must fail closed");
        assert!(matches!(
            error,
            MetadataContractError::MissingRequiredKey { .. }
        ));

        // Vocab count drifting from the declared vocab_size: fail closed.
        let mut values: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec!["a".to_string(), "b".to_string()]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = validate_sensevoice_tokenizer_contract(&metadata, 3)
            .expect_err("vocab drift must fail closed");
        assert!(matches!(error, MetadataContractError::InvalidValue { .. }));
    }
}
