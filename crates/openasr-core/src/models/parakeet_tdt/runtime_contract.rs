//! parakeet-tdt execution metadata + runtime tensor contract parsed from the
//! `.oasr` GGUF header. The validator is depth-complete: a pack must satisfy
//! the metadata contract AND the full runtime tensor binding (the shared
//! bias-free FastConformer encoder plus the joint encoder projection, LSTM
//! predictor, and fused joint head) before it can be admitted, so a malformed
//! pack fails closed at verification instead of deep inside the weight loader.

use crate::GgufTensorIndex;
use crate::ggml_runtime::GgufMetadata;
use crate::models::fastconformer::{
    FastConformerContractGeometry, fastconformer_encoder_tensor_descriptors,
};
use crate::models::runtime_contract::{
    MetadataContractError, required_u64_scalar, u64_to_u32, u64_to_usize, validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

pub(crate) const PARAKEET_TDT_N_LAYERS_KEY: &str = "parakeet-tdt.n_layers";
pub(crate) const PARAKEET_TDT_HIDDEN_SIZE_KEY: &str = "parakeet-tdt.hidden_size";
pub(crate) const PARAKEET_TDT_N_HEADS_KEY: &str = "parakeet-tdt.n_heads";
pub(crate) const PARAKEET_TDT_HEAD_DIM_KEY: &str = "parakeet-tdt.head_dim";
pub(crate) const PARAKEET_TDT_FFN_DIM_KEY: &str = "parakeet-tdt.ffn_dim";
pub(crate) const PARAKEET_TDT_CONV_KERNEL_KEY: &str = "parakeet-tdt.conv_kernel";
pub(crate) const PARAKEET_TDT_N_MELS_KEY: &str = "parakeet-tdt.n_mels";
pub(crate) const PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY: &str = "parakeet-tdt.subsampling_factor";
pub(crate) const PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY: &str = "parakeet-tdt.subsampling_channels";
pub(crate) const PARAKEET_TDT_SCALE_INPUT_KEY: &str = "parakeet-tdt.scale_input";
pub(crate) const PARAKEET_TDT_VOCAB_SIZE_KEY: &str = "parakeet-tdt.vocab_size";
pub(crate) const PARAKEET_TDT_BLANK_TOKEN_ID_KEY: &str = "parakeet-tdt.blank_token_id";
pub(crate) const PARAKEET_TDT_PRED_HIDDEN_KEY: &str = "parakeet-tdt.pred_hidden";
pub(crate) const PARAKEET_TDT_PRED_LAYERS_KEY: &str = "parakeet-tdt.pred_layers";
pub(crate) const PARAKEET_TDT_JOINT_HIDDEN_KEY: &str = "parakeet-tdt.joint_hidden";
pub(crate) const PARAKEET_TDT_N_DURATIONS_KEY: &str = "parakeet-tdt.n_durations";
pub(crate) const PARAKEET_TDT_DURATIONS_KEY: &str = "parakeet-tdt.durations";
pub(crate) const PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY: &str = "parakeet-tdt.max_symbols_per_step";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParakeetTdtExecutionMetadata {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub conv_kernel: usize,
    pub n_mels: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    /// NeMo/HF `scale_input`: multiply the subsampled input by sqrt(d_model)
    /// before the conformer stack. FALSE for parakeet-tdt-0.6b-v3 (the HF
    /// conversion this pack imports from does not scale); stored per pack so
    /// a future checkpoint that scales stays honest.
    pub scale_input: bool,
    /// Token vocab INCLUDING the blank (8193 for v3; blank = 8192 = last id).
    pub vocab_size: usize,
    pub blank_token_id: u32,
    pub pred_hidden: usize,
    pub pred_layers: usize,
    pub joint_hidden: usize,
    /// Number of TDT duration bins. The duration values are the CONTIGUOUS
    /// range `0..n_durations` (validated at import and again here), so the
    /// decode loop can use the argmax duration index as the frame skip.
    pub n_durations: usize,
    pub max_symbols_per_step: usize,
}

pub(crate) fn parse_parakeet_tdt_execution_metadata(
    metadata: &GgufMetadata,
) -> Result<ParakeetTdtExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(PARAKEET_TDT_N_LAYERS_KEY)?;
    let hidden_size = usize_key(PARAKEET_TDT_HIDDEN_SIZE_KEY)?;
    let n_heads = usize_key(PARAKEET_TDT_N_HEADS_KEY)?;
    let head_dim = usize_key(PARAKEET_TDT_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(PARAKEET_TDT_FFN_DIM_KEY)?;
    let conv_kernel = usize_key(PARAKEET_TDT_CONV_KERNEL_KEY)?;
    let n_mels = usize_key(PARAKEET_TDT_N_MELS_KEY)?;
    let subsampling_factor = usize_key(PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY)?;
    let subsampling_channels = usize_key(PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY)?;
    let scale_input = required_u64_scalar(metadata, PARAKEET_TDT_SCALE_INPUT_KEY)? != 0;
    let vocab_size = usize_key(PARAKEET_TDT_VOCAB_SIZE_KEY)?;
    let blank_token_id = u64_to_u32(
        required_u64_scalar(metadata, PARAKEET_TDT_BLANK_TOKEN_ID_KEY)?,
        PARAKEET_TDT_BLANK_TOKEN_ID_KEY,
    )?;
    let pred_hidden = usize_key(PARAKEET_TDT_PRED_HIDDEN_KEY)?;
    let pred_layers = usize_key(PARAKEET_TDT_PRED_LAYERS_KEY)?;
    let joint_hidden = usize_key(PARAKEET_TDT_JOINT_HIDDEN_KEY)?;
    let n_durations = usize_key(PARAKEET_TDT_N_DURATIONS_KEY)?;
    let max_symbols_per_step = usize_key(PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY)?;

    for (key, value) in [
        (PARAKEET_TDT_N_LAYERS_KEY, n_layers),
        (PARAKEET_TDT_HIDDEN_SIZE_KEY, hidden_size),
        (PARAKEET_TDT_N_HEADS_KEY, n_heads),
        (PARAKEET_TDT_HEAD_DIM_KEY, head_dim),
        (PARAKEET_TDT_FFN_DIM_KEY, ffn_dim),
        (PARAKEET_TDT_CONV_KERNEL_KEY, conv_kernel),
        (PARAKEET_TDT_N_MELS_KEY, n_mels),
        (PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY, subsampling_factor),
        (PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY, subsampling_channels),
        (PARAKEET_TDT_VOCAB_SIZE_KEY, vocab_size),
        (PARAKEET_TDT_PRED_HIDDEN_KEY, pred_hidden),
        (PARAKEET_TDT_PRED_LAYERS_KEY, pred_layers),
        (PARAKEET_TDT_JOINT_HIDDEN_KEY, joint_hidden),
        (PARAKEET_TDT_N_DURATIONS_KEY, n_durations),
        (PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY, max_symbols_per_step),
    ] {
        validate_positive_usize(value, key)?;
    }
    // The blank must be the last vocab slot (NeMo RNNT/TDT convention; the
    // vocab_size here already includes it).
    if (blank_token_id as usize) + 1 != vocab_size {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_TDT_BLANK_TOKEN_ID_KEY,
            reason: format!(
                "blank {blank_token_id} must be the last vocab slot (vocab_size {vocab_size})"
            ),
        });
    }
    if head_dim * n_heads != hidden_size {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_TDT_HEAD_DIM_KEY,
            reason: format!("head_dim {head_dim} * n_heads {n_heads} != hidden_size {hidden_size}"),
        });
    }
    // TDT decode requires a 2-layer LSTM predictor (v3's shape); fail closed on
    // anything else rather than run a structurally different prediction net.
    if pred_layers != 2 {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_TDT_PRED_LAYERS_KEY,
            reason: format!("parakeet-tdt runtime supports pred_layers 2 only, got {pred_layers}"),
        });
    }
    // The decode loop uses the duration argmax INDEX as the frame skip, which
    // is only sound when the trained duration bins are exactly 0..n. Enforce
    // the stored `durations` array agrees (import wrote it from config.json).
    let durations = metadata
        .get_u32_array(PARAKEET_TDT_DURATIONS_KEY)
        .ok_or_else(|| MetadataContractError::InvalidValue {
            key: PARAKEET_TDT_DURATIONS_KEY,
            reason: "missing durations array".to_string(),
        })?;
    let contiguous = durations.len() == n_durations
        && durations
            .iter()
            .enumerate()
            .all(|(index, &value)| value as usize == index);
    if !contiguous {
        return Err(MetadataContractError::InvalidValue {
            key: PARAKEET_TDT_DURATIONS_KEY,
            reason: format!(
                "durations {durations:?} must be the contiguous range 0..{n_durations}"
            ),
        });
    }

    Ok(ParakeetTdtExecutionMetadata {
        n_layers,
        hidden_size,
        n_heads,
        head_dim,
        ffn_dim,
        conv_kernel,
        n_mels,
        subsampling_factor,
        subsampling_channels,
        scale_input,
        vocab_size,
        blank_token_id,
        pred_hidden,
        pred_layers,
        joint_hidden,
        n_durations,
        max_symbols_per_step,
    })
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let metadata =
        parse_parakeet_tdt_execution_metadata(preflight.metadata()).map_err(|error| {
            crate::models::runtime_pack_contract::metadata_validation_error("parakeet-tdt", error)
        })?;
    validate_parakeet_tdt_runtime_tensors_with_index(preflight.tensor_index(), &metadata)
        .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
}

/// Fail-closed tensor-contract errors, surfaced by the pack verifier before a
/// parakeet-tdt pack can be admitted.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum ParakeetTdtTensorContractError {
    #[error("parakeet-tdt runtime tensor contract is missing required tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("parakeet-tdt runtime tensor '{name}' has shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
}

fn missing_required_tensor(name: &str) -> ParakeetTdtTensorContractError {
    ParakeetTdtTensorContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> ParakeetTdtTensorContractError {
    ParakeetTdtTensorContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

fn parakeet_tdt_contract_geometry(
    metadata: &ParakeetTdtExecutionMetadata,
) -> FastConformerContractGeometry {
    FastConformerContractGeometry {
        n_layers: metadata.n_layers,
        hidden_size: metadata.hidden_size,
        ffn_dim: metadata.ffn_dim,
        conv_kernel: metadata.conv_kernel,
        n_heads: metadata.n_heads,
        head_dim: metadata.head_dim,
        subsampling_channels: metadata.subsampling_channels,
        // v3 ships no attn/conv/FFN bias tensors; the loader synthesizes zeros.
        bias_present: false,
    }
}

/// The runtime tensor contract for one parakeet-tdt pack: the shared
/// FastConformer encoder (subsampling prelude + bias-free conformer stack),
/// the joint encoder projection, the LSTM prediction network, and the fused
/// joint head the family executor materializes.
pub(crate) fn parakeet_tdt_runtime_tensor_binding_descriptors(
    metadata: &ParakeetTdtExecutionMetadata,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors =
        fastconformer_encoder_tensor_descriptors(&parakeet_tdt_contract_geometry(metadata));
    let hidden = metadata.hidden_size;
    let joint = metadata.joint_hidden;
    let pred = metadata.pred_hidden;
    let out_rows = metadata.vocab_size + metadata.n_durations;
    let mut push =
        |tensor_name: &str, requirement: TensorBindingDescriptorRequirement, reason: &str| {
            descriptors.push(TensorBindingDescriptor {
                tensor_name: tensor_name.to_string(),
                requirement,
                reason: reason.to_string(),
            });
        };
    push(
        "enc.proj.weight",
        TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, joint),
        "joint encoder projection must map hidden_size to joint_hidden",
    );
    push(
        "enc.proj.bias",
        TensorBindingDescriptorRequirement::VectorLen(joint),
        "joint encoder projection bias must span joint_hidden",
    );
    push(
        "dec.embed.weight",
        TensorBindingDescriptorRequirement::Rank2EitherDims(metadata.vocab_size, pred),
        "predictor embedding must be vocab_size x pred_hidden",
    );
    for layer in 0..metadata.pred_layers {
        for suffix in ["w_ih", "w_hh"] {
            push(
                &format!("dec.lstm.{layer}.{suffix}"),
                TensorBindingDescriptorRequirement::Rank2EitherDims(4 * pred, pred),
                "LSTM gate weight must be 4*pred_hidden x pred_hidden",
            );
        }
        for suffix in ["b_ih", "b_hh"] {
            push(
                &format!("dec.lstm.{layer}.{suffix}"),
                TensorBindingDescriptorRequirement::VectorLen(4 * pred),
                "LSTM gate bias must span 4*pred_hidden",
            );
        }
    }
    push(
        "joint.pred.weight",
        TensorBindingDescriptorRequirement::Rank2EitherDims(pred, joint),
        "predictor projection must map pred_hidden to joint_hidden",
    );
    push(
        "joint.pred.bias",
        TensorBindingDescriptorRequirement::VectorLen(joint),
        "predictor projection bias must span joint_hidden",
    );
    push(
        "joint.out.weight",
        TensorBindingDescriptorRequirement::Rank2EitherDims(joint, out_rows),
        "fused joint head must map joint_hidden to vocab + durations",
    );
    push(
        "joint.out.bias",
        TensorBindingDescriptorRequirement::VectorLen(out_rows),
        "fused joint head bias must span vocab + durations",
    );
    descriptors
}

/// Validate the full runtime tensor set against the pack's tensor index.
pub(crate) fn validate_parakeet_tdt_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    metadata: &ParakeetTdtExecutionMetadata,
) -> Result<(), ParakeetTdtTensorContractError> {
    let descriptors = parakeet_tdt_runtime_tensor_binding_descriptors(metadata);
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )
}

/// Projects the single tensor contract into a runtime-ready fixture tensor set
/// (pack names plus valid dims); the runtime-ready test fixture stamps exactly
/// this set, so fixture and validator agree through one enumeration.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn parakeet_tdt_runtime_tensors(
    metadata: &ParakeetTdtExecutionMetadata,
) -> Vec<(String, Vec<u64>)> {
    crate::models::tensor_binding::project_fixture_tensors(
        &parakeet_tdt_runtime_tensor_binding_descriptors(metadata),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};
    use std::collections::BTreeMap;

    fn tdt_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        let mut put = |key: &str, value: u64| {
            values.insert(key.to_string(), GgufMetadataValue::U64(value));
        };
        put(PARAKEET_TDT_N_LAYERS_KEY, 24);
        put(PARAKEET_TDT_HIDDEN_SIZE_KEY, 1024);
        put(PARAKEET_TDT_N_HEADS_KEY, 8);
        put(PARAKEET_TDT_HEAD_DIM_KEY, 128);
        put(PARAKEET_TDT_FFN_DIM_KEY, 4096);
        put(PARAKEET_TDT_CONV_KERNEL_KEY, 9);
        put(PARAKEET_TDT_N_MELS_KEY, 128);
        put(PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY, 8);
        put(PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY, 256);
        put(PARAKEET_TDT_SCALE_INPUT_KEY, 0);
        put(PARAKEET_TDT_VOCAB_SIZE_KEY, 8193);
        put(PARAKEET_TDT_BLANK_TOKEN_ID_KEY, 8192);
        put(PARAKEET_TDT_PRED_HIDDEN_KEY, 640);
        put(PARAKEET_TDT_PRED_LAYERS_KEY, 2);
        put(PARAKEET_TDT_JOINT_HIDDEN_KEY, 640);
        put(PARAKEET_TDT_N_DURATIONS_KEY, 5);
        put(PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY, 10);
        values.insert(
            PARAKEET_TDT_DURATIONS_KEY.to_string(),
            GgufMetadataValue::U32Array(vec![0, 1, 2, 3, 4]),
        );
        GgufMetadata::from_values_for_test(values)
    }

    fn with_u64(metadata: GgufMetadata, key: &str, value: u64) -> GgufMetadata {
        let mut values = metadata.values().clone();
        values.insert(key.to_string(), GgufMetadataValue::U64(value));
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn parses_parakeet_tdt_06b_v3_metadata() {
        let parsed = parse_parakeet_tdt_execution_metadata(&tdt_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 24);
        assert_eq!(parsed.n_mels, 128);
        assert!(!parsed.scale_input);
        assert_eq!(parsed.vocab_size, 8193);
        assert_eq!(parsed.blank_token_id, 8192);
        assert_eq!(parsed.pred_hidden, 640);
        assert_eq!(parsed.n_durations, 5);
        assert_eq!(parsed.max_symbols_per_step, 10);
    }

    #[test]
    fn rejects_blank_not_last_vocab_slot() {
        let metadata = with_u64(tdt_metadata(), PARAKEET_TDT_BLANK_TOKEN_ID_KEY, 100);
        assert!(parse_parakeet_tdt_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_non_contiguous_durations() {
        let mut values = tdt_metadata().values().clone();
        values.insert(
            PARAKEET_TDT_DURATIONS_KEY.to_string(),
            GgufMetadataValue::U32Array(vec![0, 2, 3, 4, 8]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(parse_parakeet_tdt_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_unsupported_pred_layers() {
        let metadata = with_u64(tdt_metadata(), PARAKEET_TDT_PRED_LAYERS_KEY, 1);
        assert!(parse_parakeet_tdt_execution_metadata(&metadata).is_err());
    }

    // --- Runtime tensor contract ---

    fn tiny_execution_metadata() -> ParakeetTdtExecutionMetadata {
        ParakeetTdtExecutionMetadata {
            n_layers: 1,
            hidden_size: 16,
            n_heads: 2,
            head_dim: 8,
            ffn_dim: 32,
            conv_kernel: 9,
            n_mels: 128,
            subsampling_factor: 8,
            subsampling_channels: 24,
            scale_input: false,
            vocab_size: 12,
            blank_token_id: 11,
            pred_hidden: 20,
            pred_layers: 2,
            joint_hidden: 24,
            n_durations: 5,
            max_symbols_per_step: 10,
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
            path: std::path::PathBuf::from("parakeet-tdt-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    /// The requirement enumeration IS the loader read set: pin it on the full
    /// production geometry (parakeet-tdt-0.6b-v3: 24 bias-free layers). The
    /// bias-free encoder contributes 12 subsampling + 28 per-layer tensors
    /// (24 always-loaded + 4 BatchNorm-fold statistics, no projection biases),
    /// plus the 15-tensor TDT tail (2 enc.proj, 1 embed, 8 LSTM, 4 joint).
    #[test]
    fn descriptor_set_matches_the_loader_read_set_on_production_geometry() {
        let parsed = parse_parakeet_tdt_execution_metadata(&tdt_metadata()).expect("parse");
        let descriptors = parakeet_tdt_runtime_tensor_binding_descriptors(&parsed);
        assert_eq!(descriptors.len(), 12 + 24 * 28 + 15);
        let names: std::collections::BTreeSet<&str> = descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.as_str())
            .collect();
        assert_eq!(names.len(), descriptors.len(), "names must be unique");
        for required in [
            "enc.sub.layers.0.weight",
            "enc.sub.linear.bias",
            "enc.blk.0.ff1.up.weight",
            "enc.blk.0.conv.bn.var",
            "enc.blk.23.out.norm.bias",
            "enc.proj.weight",
            "dec.embed.weight",
            "dec.lstm.1.b_hh",
            "joint.pred.weight",
            "joint.out.bias",
        ] {
            assert!(names.contains(required), "contract must cover {required}");
        }
        // The bias-free checkpoint must NOT be required to carry projection
        // biases: the loader synthesizes them as zeros.
        for forbidden in ["enc.blk.0.ff1.up.bias", "enc.blk.0.attn.q.bias"] {
            assert!(
                !names.contains(forbidden),
                "bias-free contract must not require {forbidden}"
            );
        }
    }

    #[test]
    fn validates_the_projected_tiny_tensor_set() {
        let metadata = tiny_execution_metadata();
        let shapes = parakeet_tdt_runtime_tensors(&metadata);
        let index = tensor_index_from_shapes(&shapes);
        validate_parakeet_tdt_runtime_tensors_with_index(&index, &metadata)
            .expect("projected tensor set must satisfy the contract");
    }

    #[test]
    fn rejects_a_missing_required_tensor() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_tdt_runtime_tensors(&metadata);
        shapes.retain(|(name, _)| name != "joint.out.weight");
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_tdt_runtime_tensors_with_index(&index, &metadata)
            .expect_err("missing joint head must fail closed");
        assert!(
            matches!(error, ParakeetTdtTensorContractError::MissingRequiredTensor { ref name } if name == "joint.out.weight"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_a_wrong_shape() {
        let metadata = tiny_execution_metadata();
        let mut shapes = parakeet_tdt_runtime_tensors(&metadata);
        for (name, dims) in shapes.iter_mut() {
            if name == "dec.lstm.0.w_ih" {
                *dims = vec![3];
            }
        }
        let index = tensor_index_from_shapes(&shapes);
        let error = validate_parakeet_tdt_runtime_tensors_with_index(&index, &metadata)
            .expect_err("corrupted LSTM gate weight must fail closed");
        assert!(
            matches!(error, ParakeetTdtTensorContractError::InvalidTensorShape { ref name, .. } if name == "dec.lstm.0.w_ih"),
            "unexpected error: {error}"
        );
    }
}
