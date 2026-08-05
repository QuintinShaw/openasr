//! parakeet-ctc execution metadata + runtime tensor contract parsed from the
//! `.oasr` GGUF header. The validator is depth-complete: a pack must satisfy
//! the metadata contract AND the full runtime tensor binding (the shared
//! FastConformer encoder plus the CTC head) before it can be admitted, so a
//! malformed pack fails closed at verification instead of deep inside the
//! weight loader.

use crate::GgufTensorIndex;
use crate::models::fastconformer::{
    FastConformerContractGeometry, fastconformer_encoder_descriptor_count,
    fastconformer_encoder_tensor_descriptors,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_bounded_usize, validate_positive_usize,
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

/// The dw-striding subsampling prelude + the frontend/frame bookkeeping the
/// shared graph hardcodes amount to exactly three stride-2 stages; the
/// contract admits no other factor (parakeet-tdt carries the same pin).
pub(crate) const PARAKEET_CTC_SUBSAMPLING_FACTOR: usize = 8;

/// Architecture ceilings for pack-supplied geometry, with generous headroom
/// over the production checkpoint (24 layers, hidden 1024, ffn 4096, vocab
/// 1025). They bound every contract-derived arithmetic expression and the
/// tensor-obligation count a malicious metadata set can construct, so
/// contract building stays allocation-bounded and overflow-free on untrusted
/// input; parse fails closed above them.
pub(crate) const PARAKEET_CTC_MAX_N_LAYERS: usize = 512;
pub(crate) const PARAKEET_CTC_MAX_HIDDEN_SIZE: usize = 65_536;
pub(crate) const PARAKEET_CTC_MAX_FFN_DIM: usize = 262_144;
pub(crate) const PARAKEET_CTC_MAX_N_HEADS: usize = 1_024;
pub(crate) const PARAKEET_CTC_MAX_HEAD_DIM: usize = 65_536;
pub(crate) const PARAKEET_CTC_MAX_CONV_KERNEL: usize = 4_096;
pub(crate) const PARAKEET_CTC_MAX_N_MELS: usize = 4_096;
pub(crate) const PARAKEET_CTC_MAX_SUBSAMPLING_CHANNELS: usize = 65_536;
pub(crate) const PARAKEET_CTC_MAX_VOCAB_SIZE: usize = 1_000_000;
/// Global ceiling on the tensor obligations one pack's contract may
/// construct; far above the production 950, far below anything that could
/// exhaust the verifier.
pub(crate) const PARAKEET_CTC_MAX_TENSOR_OBLIGATIONS: usize = 1_000_000;
/// The CTC head tensors appended after the shared FastConformer encoder.
const PARAKEET_CTC_TAIL_DESCRIPTOR_COUNT: usize = 2;

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
    // Architecture ceilings: keep contract construction bounded and
    // overflow-free on untrusted metadata (fail closed above them).
    for (key, value, max) in [
        (PARAKEET_N_LAYERS_KEY, n_layers, PARAKEET_CTC_MAX_N_LAYERS),
        (
            PARAKEET_HIDDEN_SIZE_KEY,
            hidden_size,
            PARAKEET_CTC_MAX_HIDDEN_SIZE,
        ),
        (PARAKEET_N_HEADS_KEY, n_heads, PARAKEET_CTC_MAX_N_HEADS),
        (PARAKEET_HEAD_DIM_KEY, head_dim, PARAKEET_CTC_MAX_HEAD_DIM),
        (PARAKEET_FFN_DIM_KEY, ffn_dim, PARAKEET_CTC_MAX_FFN_DIM),
        (
            PARAKEET_CONV_KERNEL_KEY,
            conv_kernel,
            PARAKEET_CTC_MAX_CONV_KERNEL,
        ),
        (PARAKEET_N_MELS_KEY, n_mels, PARAKEET_CTC_MAX_N_MELS),
        (
            PARAKEET_SUBSAMPLING_CHANNELS_KEY,
            subsampling_channels,
            PARAKEET_CTC_MAX_SUBSAMPLING_CHANNELS,
        ),
        (
            PARAKEET_VOCAB_SIZE_KEY,
            vocab_size,
            PARAKEET_CTC_MAX_VOCAB_SIZE,
        ),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    // The shared subsampling prelude and the frontend frame bookkeeping
    // hardcode three stride-2 stages; admit no other factor.
    if subsampling_factor != PARAKEET_CTC_SUBSAMPLING_FACTOR {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_SUBSAMPLING_FACTOR_KEY,
            reason: format!(
                "the shared graph fixes three stride-2 subsampling stages (factor \
                 {PARAKEET_CTC_SUBSAMPLING_FACTOR}), got {subsampling_factor}"
            ),
        });
    }
    // The blank id must be the last vocab slot (vocab includes the blank).
    if (blank_token_id as usize) >= vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_CTC_BLANK_TOKEN_ID_KEY,
            reason: format!("blank {blank_token_id} out of range for vocab_size {vocab_size}"),
        });
    }
    if head_dim.checked_mul(n_heads) != Some(hidden_size) {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_HEAD_DIM_KEY,
            reason: format!("head_dim {head_dim} * n_heads {n_heads} != hidden_size {hidden_size}"),
        });
    }
    // Bound the tensor obligations this geometry's contract will construct,
    // fail-closed, before any descriptor allocation happens. The count is
    // derived from the same builders the contract uses, so it cannot drift.
    let total_obligations =
        fastconformer_encoder_descriptor_count(&FastConformerContractGeometry {
            n_layers,
            hidden_size,
            ffn_dim,
            conv_kernel,
            n_heads,
            head_dim,
            n_mels,
            subsampling_channels,
            bias_present: true,
        })
        .saturating_add(PARAKEET_CTC_TAIL_DESCRIPTOR_COUNT);
    if total_obligations > PARAKEET_CTC_MAX_TENSOR_OBLIGATIONS {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_N_LAYERS_KEY,
            reason: format!(
                "geometry constructs {total_obligations} tensor obligations, exceeding the \
                 ceiling {PARAKEET_CTC_MAX_TENSOR_OBLIGATIONS}"
            ),
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
        n_mels: metadata.n_mels,
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

    /// Structural pins on the production geometry (parakeet-ctc-0.6b: 24
    /// bias-complete layers). The loader-equivalence evidence lives in
    /// `full_loader_read_trace_equals_the_descriptor_set` (encoder_weights);
    /// this test holds the enumeration's production shape stable: the shared
    /// encoder contributes 12 subsampling + 39 per-layer tensors (24
    /// always-loaded + 11 biases + 4 BatchNorm-fold statistics), plus the
    /// 2-tensor CTC head tail.
    #[test]
    fn descriptor_set_stays_pinned_on_production_geometry() {
        let parsed = parse_parakeet_ctc_execution_metadata(&parakeet_metadata()).expect("parse");
        assert_eq!(parsed.subsampling_factor, PARAKEET_CTC_SUBSAMPLING_FACTOR);
        let descriptors = parakeet_ctc_runtime_tensor_binding_descriptors(&parsed);
        assert_eq!(descriptors.len(), 12 + 24 * 39 + 2);
        let names: std::collections::BTreeSet<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        assert_eq!(names.len(), descriptors.len(), "names must be unique");
    }

    #[test]
    fn rejects_a_subsampling_factor_the_shared_graph_cannot_express() {
        for factor in [1u64, 2, 4, 16, 32] {
            let mut metadata = parakeet_metadata();
            metadata.insert(
                PARAKEET_SUBSAMPLING_FACTOR_KEY.to_string(),
                factor.to_string(),
            );
            let error = parse_parakeet_ctc_execution_metadata(&metadata).expect_err(
                "the shared graph fixes three stride-2 stages; only factor 8 is admissible",
            );
            assert!(matches!(
                error,
                MetadataContractError::InvalidValue {
                    key: PARAKEET_SUBSAMPLING_FACTOR_KEY,
                    ..
                }
            ));
        }
    }

    /// Architecture ceilings fail closed on untrusted metadata, keeping
    /// contract construction allocation-bounded and overflow-free.
    #[test]
    fn rejects_geometry_above_architecture_ceilings() {
        for (key, value) in [
            (PARAKEET_N_LAYERS_KEY, PARAKEET_CTC_MAX_N_LAYERS as u64 + 1),
            (PARAKEET_FFN_DIM_KEY, PARAKEET_CTC_MAX_FFN_DIM as u64 + 1),
            (
                PARAKEET_CONV_KERNEL_KEY,
                PARAKEET_CTC_MAX_CONV_KERNEL as u64 + 1,
            ),
            (PARAKEET_N_MELS_KEY, PARAKEET_CTC_MAX_N_MELS as u64 + 1),
            (
                PARAKEET_SUBSAMPLING_CHANNELS_KEY,
                PARAKEET_CTC_MAX_SUBSAMPLING_CHANNELS as u64 + 1,
            ),
            (
                PARAKEET_VOCAB_SIZE_KEY,
                PARAKEET_CTC_MAX_VOCAB_SIZE as u64 + 1,
            ),
        ] {
            let mut metadata = parakeet_metadata();
            metadata.insert(key.to_string(), value.to_string());
            assert!(
                parse_parakeet_ctc_execution_metadata(&metadata).is_err(),
                "must reject {key} = {value} above its ceiling"
            );
        }
    }

    /// Boundary: geometry exactly at the ceilings stays admissible.
    #[test]
    fn accepts_geometry_at_the_architecture_ceilings() {
        let mut metadata = parakeet_metadata();
        metadata.insert(
            PARAKEET_N_LAYERS_KEY.to_string(),
            PARAKEET_CTC_MAX_N_LAYERS.to_string(),
        );
        metadata.insert(
            PARAKEET_FFN_DIM_KEY.to_string(),
            PARAKEET_CTC_MAX_FFN_DIM.to_string(),
        );
        assert!(parse_parakeet_ctc_execution_metadata(&metadata).is_ok());
    }

    /// Overflowing head geometry must fail closed through checked arithmetic
    /// instead of wrapping into an accidentally satisfying product.
    #[test]
    fn rejects_overflowing_head_geometry_without_wrapping() {
        let mut metadata = parakeet_metadata();
        metadata.insert(PARAKEET_HEAD_DIM_KEY.to_string(), u64::MAX.to_string());
        metadata.insert(PARAKEET_N_HEADS_KEY.to_string(), u64::MAX.to_string());
        // The head_dim ceiling fires first; either way parse fails closed
        // without panicking or wrapping the product.
        assert!(parse_parakeet_ctc_execution_metadata(&metadata).is_err());
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

    /// Transposed rectangular weights used to pass under Rank2EitherDims; the
    /// ordered ExactDims contract must reject them at the production validator
    /// entry so a pack cannot ship the HF [out, in] orientation the graph
    /// cannot consume.
    #[test]
    fn rejects_transposed_subsampling_linear_weight() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_ctc_runtime_tensors(&metadata);
        let linear = shapes
            .iter_mut()
            .find(|(name, _)| name == "enc.sub.linear.weight")
            .expect("subsampling linear");
        // Correct is ggml [flatten, hidden]; swap to [hidden, flatten].
        linear.1 = vec![metadata.hidden_size as u64, linear.1[0]];
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("transposed enc.sub.linear.weight must fail closed");
        assert!(
            matches!(
                error,
                ParakeetCtcTensorContractError::InvalidTensorShape { ref name, .. }
                    if name == "enc.sub.linear.weight"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_transposed_ffn_up_weight() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_ctc_runtime_tensors(&metadata);
        let ff1_up = shapes
            .iter_mut()
            .find(|(name, _)| name == "enc.blk.0.ff1.up.weight")
            .expect("ff1.up");
        // Correct is ggml [hidden, ffn]; swap to [ffn, hidden].
        ff1_up.1 = vec![metadata.ffn_dim as u64, metadata.hidden_size as u64];
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_ctc_runtime_tensors_with_index(&index, &metadata)
            .expect_err("transposed ff1.up.weight must fail closed");
        assert!(
            matches!(
                error,
                ParakeetCtcTensorContractError::InvalidTensorShape { ref name, .. }
                    if name == "enc.blk.0.ff1.up.weight"
            ),
            "unexpected error: {error}"
        );
    }
}
