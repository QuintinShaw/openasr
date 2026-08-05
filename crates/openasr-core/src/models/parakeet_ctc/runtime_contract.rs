//! parakeet-ctc execution metadata + runtime tensor contract parsed from the
//! `.oasr` GGUF header. The validator is depth-complete: a pack must satisfy
//! the metadata contract AND the full runtime tensor binding (the shared
//! FastConformer encoder plus the CTC head) before it can be admitted, so a
//! malformed pack fails closed at verification instead of deep inside the
//! weight loader.

use crate::GgufTensorIndex;
use crate::models::fastconformer::{
    FastConformerContractGeometry, fastconformer_encoder_tensor_descriptors,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

pub(crate) const PARAKEET_N_LAYERS_KEY: &str = "parakeet.n_layers";
pub(crate) const PARAKEET_HIDDEN_SIZE_KEY: &str = "parakeet.hidden_size";
pub(crate) const PARAKEET_N_HEADS_KEY: &str = "parakeet.n_heads";
pub(crate) const PARAKEET_HEAD_DIM_KEY: &str = "parakeet.head_dim";
pub(crate) const PARAKEET_FFN_DIM_KEY: &str = "parakeet.ffn_dim";
pub(crate) const PARAKEET_CONV_KERNEL_KEY: &str = "parakeet.conv_kernel";
pub(crate) const PARAKEET_N_MELS_KEY: &str = "parakeet.n_mels";
pub(crate) const PARAKEET_SUBSAMPLING_FACTOR_KEY: &str = "parakeet.subsampling_factor";
pub(crate) const PARAKEET_SUBSAMPLING_CHANNELS_KEY: &str = "parakeet.subsampling_channels";
pub(crate) const PARAKEET_VOCAB_SIZE_KEY: &str = "parakeet.vocab_size";
pub(crate) const PARAKEET_CTC_BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParakeetCtcExecutionMetadata {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub conv_kernel: usize,
    pub n_mels: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
}

pub(crate) fn parse_parakeet_ctc_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<ParakeetCtcExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(PARAKEET_N_LAYERS_KEY)?;
    let hidden_size = usize_key(PARAKEET_HIDDEN_SIZE_KEY)?;
    let n_heads = usize_key(PARAKEET_N_HEADS_KEY)?;
    let head_dim = usize_key(PARAKEET_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(PARAKEET_FFN_DIM_KEY)?;
    let conv_kernel = usize_key(PARAKEET_CONV_KERNEL_KEY)?;
    let n_mels = usize_key(PARAKEET_N_MELS_KEY)?;
    let subsampling_factor = usize_key(PARAKEET_SUBSAMPLING_FACTOR_KEY)?;
    let subsampling_channels = usize_key(PARAKEET_SUBSAMPLING_CHANNELS_KEY)?;
    let vocab_size = usize_key(PARAKEET_VOCAB_SIZE_KEY)?;
    let blank_token_id = u64_to_u32(
        required_u64_scalar(metadata, PARAKEET_CTC_BLANK_TOKEN_ID_KEY)?,
        PARAKEET_CTC_BLANK_TOKEN_ID_KEY,
    )?;

    for (key, value) in [
        (PARAKEET_N_LAYERS_KEY, n_layers),
        (PARAKEET_HIDDEN_SIZE_KEY, hidden_size),
        (PARAKEET_N_HEADS_KEY, n_heads),
        (PARAKEET_HEAD_DIM_KEY, head_dim),
        (PARAKEET_FFN_DIM_KEY, ffn_dim),
        (PARAKEET_CONV_KERNEL_KEY, conv_kernel),
        (PARAKEET_N_MELS_KEY, n_mels),
        (PARAKEET_SUBSAMPLING_FACTOR_KEY, subsampling_factor),
        (PARAKEET_SUBSAMPLING_CHANNELS_KEY, subsampling_channels),
        (PARAKEET_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    // The blank id must be the last vocab slot (vocab includes the blank).
    if (blank_token_id as usize) >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_CTC_BLANK_TOKEN_ID_KEY,
            reason: format!("blank {blank_token_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if head_dim * n_heads != hidden_size {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_HEAD_DIM_KEY,
            reason: format!("head_dim {head_dim} * n_heads {n_heads} != hidden_size {hidden_size}"),
        });
    }

    Ok(ParakeetCtcExecutionMetadata {
        n_layers,
        hidden_size,
        n_heads,
        head_dim,
        ffn_dim,
        conv_kernel,
        n_mels,
        subsampling_factor,
        subsampling_channels,
        vocab_size,
        blank_token_id,
    })
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata =
        parse_parakeet_ctc_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("parakeet-ctc", error)
        })?;
    validate_parakeet_ctc_runtime_tensors_with_index(preflight.tensor_index(), &metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

/// Fail-closed tensor-contract errors, surfaced by the pack verifier before a
/// parakeet-ctc pack can be admitted.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum ParakeetCtcTensorContractError {
    #[error("parakeet-ctc runtime tensor contract is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("parakeet-ctc runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

fn missing_required_tensor(name: &str) -> ParakeetCtcTensorContractError {
    ParakeetCtcTensorContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> ParakeetCtcTensorContractError {
    ParakeetCtcTensorContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn parakeet_ctc_contract_geometry(
    metadata: &ParakeetCtcExecutionMetadata,
) -> FastConformerContractGeometry {
    FastConformerContractGeometry {
        n_layers: metadata.n_layers,
        hidden_size: metadata.hidden_size,
        ffn_dim: metadata.ffn_dim,
        conv_kernel: metadata.conv_kernel,
        n_heads: metadata.n_heads,
        head_dim: metadata.head_dim,
        subsampling_channels: metadata.subsampling_channels,
        // The parakeet-ctc checkpoint ships every conformer bias tensor.
        bias_present: true,
    }
}

/// The runtime tensor contract for one parakeet-ctc pack: the shared
/// FastConformer encoder (subsampling prelude + conformer stack, bias-complete)
/// plus the CTC head tail the family executor materializes.
pub(crate) fn parakeet_ctc_runtime_tensor_binding_descriptors(
    metadata: &ParakeetCtcExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors =
        fastconformer_encoder_tensor_descriptors(&parakeet_ctc_contract_geometry(metadata));
    descriptors.extend([
        TensorBindingDescriptor {
            tensor_name: "ctc.head.weight".to_string(),
            requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(
                metadata.hidden_size,
                metadata.vocab_size,
            ),
            reason: "CTC head must project hidden_size to the vocab".to_string(),
        },
        TensorBindingDescriptor {
            tensor_name: "ctc.head.bias".to_string(),
            requirement: TensorBindingDescriptorRequirement::VectorLen(metadata.vocab_size),
            reason: "CTC head bias must span the vocab".to_string(),
        },
    ]);
    descriptors
}

/// Validate the full runtime tensor set against the pack's tensor index.
pub(crate) fn validate_parakeet_ctc_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &ParakeetCtcExecutionMetadata,
) -> Result<(), ParakeetCtcTensorContractError> {
    let descriptors = parakeet_ctc_runtime_tensor_binding_descriptors(metadata);
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )
}

/// Projects the single tensor contract into a runtime-ready fixture tensor set
/// (pack names plus valid dims); the runtime-ready test fixture
/// stamps exactly this set, so fixture and validator agree through one
/// enumeration.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn parakeet_ctc_runtime_tensors(
    metadata: &ParakeetCtcExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    crate::models::tensor_binding::project_fixture_tensors(
        &parakeet_ctc_runtime_tensor_binding_descriptors(metadata),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn parakeet_metadata() -> BTreeMap<String, String> {
        [
            (PARAKEET_N_LAYERS_KEY, "24"),
            (PARAKEET_HIDDEN_SIZE_KEY, "1024"),
            (PARAKEET_N_HEADS_KEY, "8"),
            (PARAKEET_HEAD_DIM_KEY, "128"),
            (PARAKEET_FFN_DIM_KEY, "4096"),
            (PARAKEET_CONV_KERNEL_KEY, "9"),
            (PARAKEET_N_MELS_KEY, "80"),
            (PARAKEET_SUBSAMPLING_FACTOR_KEY, "8"),
            (PARAKEET_SUBSAMPLING_CHANNELS_KEY, "256"),
            (PARAKEET_VOCAB_SIZE_KEY, "1025"),
            (PARAKEET_CTC_BLANK_TOKEN_ID_KEY, "1024"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_parakeet_ctc_06b_metadata() {
        let parsed = parse_parakeet_ctc_execution_metadata(&parakeet_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 24);
        assert_eq!(parsed.hidden_size, 1024);
        assert_eq!(parsed.head_dim, 128);
        assert_eq!(parsed.vocab_size, 1025);
        assert_eq!(parsed.blank_token_id, 1024);
    }

    #[test]
    fn rejects_blank_out_of_vocab() {
        let mut metadata = parakeet_metadata();
        metadata.insert(
            PARAKEET_CTC_BLANK_TOKEN_ID_KEY.to_string(),
            "2000".to_string(),
        );
        assert!(parse_parakeet_ctc_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_inconsistent_head_dim() {
        let mut metadata = parakeet_metadata();
        metadata.insert(PARAKEET_HEAD_DIM_KEY.to_string(), "100".to_string());
        assert!(parse_parakeet_ctc_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_missing_key() {
        let mut metadata = parakeet_metadata();
        metadata.remove(PARAKEET_N_LAYERS_KEY);
        assert!(parse_parakeet_ctc_execution_metadata(&metadata).is_err());
    }

    // --- Runtime tensor contract ---

    fn tiny_execution_metadata() -> ParakeetCtcExecutionMetadata {
        ParakeetCtcExecutionMetadata {
            n_layers: 1,
            hidden_size: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            conv_kernel: 9,
            n_mels: 80,
            subsampling_factor: 8,
            subsampling_channels: 24,
            vocab_size: 12,
            blank_token_id: 11,
        }
    }

    fn tensor_index_from_shapes(shapes: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let tensors = shapes
            .iter()
            .enumerate()
            .map(|(index, (name, dims))| crate::GgufTensorMetadata {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("parakeet-ctc-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    /// The requirement enumeration IS the loader read set: pin it on the full
    /// production geometry (parakeet-ctc-0.6b: 24 bias-complete layers). The
    /// shared encoder contributes 12 subsampling + 39 per-layer tensors
    /// (24 always-loaded + 11 biases + 4 BatchNorm-fold statistics), plus the
    /// 2-tensor CTC head tail.
    #[test]
    fn descriptor_set_matches_the_loader_read_set_on_production_geometry() {
        let parsed = parse_parakeet_ctc_execution_metadata(&parakeet_metadata()).expect("parse");
        let descriptors = parakeet_ctc_runtime_tensor_binding_descriptors(&parsed);
        assert_eq!(descriptors.len(), 12 + 24 * 39 + 2);
        let names: std::collections::BTreeSet<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        assert_eq!(names.len(), descriptors.len(), "names must be unique");
        for required in [
            "enc.sub.layers.0.weight",
            "enc.sub.layers.6.bias",
            "enc.sub.linear.weight",
            "enc.blk.0.ff1.up.bias",
            "enc.blk.0.attn.pos_bias_u",
            "enc.blk.0.conv.bn.var",
            "enc.blk.23.out.norm.bias",
            "ctc.head.weight",
            "ctc.head.bias",
        ] {
            assert!(names.contains(required), "contract must cover {required}");
        }
    }

    #[test]
    fn validates_the_projected_tiny_tensor_set() {
        let metadata = tiny_execution_metadata();
        let shapes = parakeet_ctc_runtime_tensors(&metadata);
        let index = tensor_index_from_shapes(&shapes);
        validate_parakeet_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect("projected tensor set must satisfy the contract");
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_ctc_runtime_tensors(&metadata);
        shapes.retain(|(name, _)| name != "ctc.head.weight");
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("missing CTC head must fail closed");
        assert!(
            matches!(error, ParakeetCtcTensorContractError::MissingRequiredTensor { ref name } if name == "ctc.head.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_a_wrong_shape() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_ctc_runtime_tensors(&metadata);
        for (name, dims) in shapes.iter_mut() {
            if name == "enc.blk.0.conv.dw.weight" {
                *dims = vec![1, 1];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("corrupted depthwise kernel must fail closed");
        assert!(
            matches!(error, ParakeetCtcTensorContractError::InvalidTensorShape { ref name, .. } if name == "enc.blk.0.conv.dw.weight"),
            "unexpected error: {error}"
        );
    }
}
