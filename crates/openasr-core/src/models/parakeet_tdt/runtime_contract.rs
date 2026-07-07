//! parakeet-tdt execution metadata parsed from the `.oasr` GGUF header.

// Consumed by the encoder/executor wired in the follow-up stages; tested
// standalone meanwhile (the parakeet-ctc staging precedent).
#![allow(dead_code)]

use crate::ggml_runtime::GgufMetadata;
use crate::models::runtime_contract::{
    MetadataContractError, required_u64_scalar, u64_to_u32, u64_to_usize, validate_positive_usize,
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
}
