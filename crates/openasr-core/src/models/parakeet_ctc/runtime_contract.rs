//! parakeet-ctc execution metadata parsed from the `.oasr` GGUF header.

// Consumed by the encoder/executor wired in S3/S4; tested standalone meanwhile.
#![allow(dead_code)]

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
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
}
