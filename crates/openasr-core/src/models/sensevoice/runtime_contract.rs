//! sensevoice execution metadata parsed from the `.oasr` GGUF header.

#![allow(dead_code)]

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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
}
