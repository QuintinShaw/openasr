//! firered-aed execution metadata parsed from the `.oasr` GGUF header.

#![allow(dead_code)]

use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_u32, u64_to_usize,
    validate_positive_usize,
};

pub(crate) const FIRERED_ENCODER_N_LAYERS_KEY: &str = "firered.encoder.n_layers";
pub(crate) const FIRERED_ENCODER_D_MODEL_KEY: &str = "firered.encoder.d_model";
pub(crate) const FIRERED_ENCODER_N_HEADS_KEY: &str = "firered.encoder.n_heads";
pub(crate) const FIRERED_ENCODER_HEAD_DIM_KEY: &str = "firered.encoder.head_dim";
pub(crate) const FIRERED_ENCODER_FFN_DIM_KEY: &str = "firered.encoder.ffn_dim";
pub(crate) const FIRERED_ENCODER_CONV_KERNEL_KEY: &str = "firered.encoder.conv_kernel";
pub(crate) const FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY: &str =
    "firered.encoder.subsample_channels";
pub(crate) const FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY: &str = "firered.encoder.subsample_out_dim";
pub(crate) const FIRERED_ENCODER_FEATURE_DIM_KEY: &str = "firered.encoder.feature_dim";
pub(crate) const FIRERED_ENCODER_PE_LEN_KEY: &str = "firered.encoder.pe_len";
pub(crate) const FIRERED_DECODER_N_LAYERS_KEY: &str = "firered.decoder.n_layers";
pub(crate) const FIRERED_DECODER_FFN_DIM_KEY: &str = "firered.decoder.ffn_dim";
pub(crate) const FIRERED_DECODER_PE_LEN_KEY: &str = "firered.decoder.pe_len";
pub(crate) const FIRERED_VOCAB_SIZE_KEY: &str = "firered.vocab_size";
pub(crate) const FIRERED_SOS_TOKEN_ID_KEY: &str = "firered.sos_token_id";
pub(crate) const FIRERED_EOS_TOKEN_ID_KEY: &str = "firered.eos_token_id";
pub(crate) const FIRERED_PAD_TOKEN_ID_KEY: &str = "firered.pad_token_id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FireRedAedExecutionMetadata {
    pub encoder_n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub encoder_ffn_dim: usize,
    pub conv_kernel: usize,
    pub subsample_channels: usize,
    pub subsample_out_dim: usize,
    pub feature_dim: usize,
    /// Relative-position table rows (`2 * max_frames - 1`, odd).
    pub encoder_pe_len: usize,
    pub decoder_n_layers: usize,
    pub decoder_ffn_dim: usize,
    /// Absolute sinusoidal position rows == decoder max context.
    pub decoder_pe_len: usize,
    pub vocab_size: usize,
    pub sos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl FireRedAedExecutionMetadata {
    /// Maximum encoder frame count the baked rel-pos table supports.
    pub(crate) fn encoder_max_frames(&self) -> usize {
        (self.encoder_pe_len + 1) / 2
    }
}

pub(crate) fn parse_firered_aed_execution_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedAedExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let encoder_n_layers = usize_key(FIRERED_ENCODER_N_LAYERS_KEY)?;
    let d_model = usize_key(FIRERED_ENCODER_D_MODEL_KEY)?;
    let n_heads = usize_key(FIRERED_ENCODER_N_HEADS_KEY)?;
    let head_dim = usize_key(FIRERED_ENCODER_HEAD_DIM_KEY)?;
    let encoder_ffn_dim = usize_key(FIRERED_ENCODER_FFN_DIM_KEY)?;
    let conv_kernel = usize_key(FIRERED_ENCODER_CONV_KERNEL_KEY)?;
    let subsample_channels = usize_key(FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY)?;
    let subsample_out_dim = usize_key(FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY)?;
    let feature_dim = usize_key(FIRERED_ENCODER_FEATURE_DIM_KEY)?;
    let encoder_pe_len = usize_key(FIRERED_ENCODER_PE_LEN_KEY)?;
    let decoder_n_layers = usize_key(FIRERED_DECODER_N_LAYERS_KEY)?;
    let decoder_ffn_dim = usize_key(FIRERED_DECODER_FFN_DIM_KEY)?;
    let decoder_pe_len = usize_key(FIRERED_DECODER_PE_LEN_KEY)?;
    let vocab_size = usize_key(FIRERED_VOCAB_SIZE_KEY)?;
    let sos_token_id = u32_key(FIRERED_SOS_TOKEN_ID_KEY)?;
    let eos_token_id = u32_key(FIRERED_EOS_TOKEN_ID_KEY)?;
    let pad_token_id = u32_key(FIRERED_PAD_TOKEN_ID_KEY)?;

    for (key, value) in [
        (FIRERED_ENCODER_N_LAYERS_KEY, encoder_n_layers),
        (FIRERED_ENCODER_D_MODEL_KEY, d_model),
        (FIRERED_ENCODER_N_HEADS_KEY, n_heads),
        (FIRERED_ENCODER_HEAD_DIM_KEY, head_dim),
        (FIRERED_ENCODER_FFN_DIM_KEY, encoder_ffn_dim),
        (FIRERED_ENCODER_CONV_KERNEL_KEY, conv_kernel),
        (FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, subsample_channels),
        (FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY, subsample_out_dim),
        (FIRERED_ENCODER_FEATURE_DIM_KEY, feature_dim),
        (FIRERED_ENCODER_PE_LEN_KEY, encoder_pe_len),
        (FIRERED_DECODER_N_LAYERS_KEY, decoder_n_layers),
        (FIRERED_DECODER_FFN_DIM_KEY, decoder_ffn_dim),
        (FIRERED_DECODER_PE_LEN_KEY, decoder_pe_len),
        (FIRERED_VOCAB_SIZE_KEY, vocab_size),
    ] {
        validate_positive_usize(value, key)?;
    }
    if n_heads * head_dim != d_model {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_HEAD_DIM_KEY,
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    if conv_kernel.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_CONV_KERNEL_KEY,
            reason: format!("conv kernel {conv_kernel} must be odd (symmetric padding)"),
        });
    }
    if encoder_pe_len.is_multiple_of(2) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_PE_LEN_KEY,
            reason: format!("rel-pos table length {encoder_pe_len} must be odd (2*max-1)"),
        });
    }
    let expected_subsample = subsample_channels * (((feature_dim - 1) / 2 - 1) / 2);
    if subsample_out_dim != expected_subsample {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
            reason: format!(
                "subsample_out_dim {subsample_out_dim} != channels {subsample_channels} x \
                 subsampled {feature_dim}-mel width ({expected_subsample})"
            ),
        });
    }
    for (key, id) in [
        (FIRERED_SOS_TOKEN_ID_KEY, sos_token_id),
        (FIRERED_EOS_TOKEN_ID_KEY, eos_token_id),
        (FIRERED_PAD_TOKEN_ID_KEY, pad_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(FireRedAedExecutionMetadata {
        encoder_n_layers,
        d_model,
        n_heads,
        head_dim,
        encoder_ffn_dim,
        conv_kernel,
        subsample_channels,
        subsample_out_dim,
        feature_dim,
        encoder_pe_len,
        decoder_n_layers,
        decoder_ffn_dim,
        decoder_pe_len,
        vocab_size,
        sos_token_id,
        eos_token_id,
        pad_token_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn aed_l_metadata() -> BTreeMap<String, String> {
        [
            (FIRERED_ENCODER_N_LAYERS_KEY, "16"),
            (FIRERED_ENCODER_D_MODEL_KEY, "1280"),
            (FIRERED_ENCODER_N_HEADS_KEY, "20"),
            (FIRERED_ENCODER_HEAD_DIM_KEY, "64"),
            (FIRERED_ENCODER_FFN_DIM_KEY, "5120"),
            (FIRERED_ENCODER_CONV_KERNEL_KEY, "33"),
            (FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, "32"),
            (FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY, "608"),
            (FIRERED_ENCODER_FEATURE_DIM_KEY, "80"),
            (FIRERED_ENCODER_PE_LEN_KEY, "9999"),
            (FIRERED_DECODER_N_LAYERS_KEY, "16"),
            (FIRERED_DECODER_FFN_DIM_KEY, "5120"),
            (FIRERED_DECODER_PE_LEN_KEY, "5000"),
            (FIRERED_VOCAB_SIZE_KEY, "7832"),
            (FIRERED_SOS_TOKEN_ID_KEY, "3"),
            (FIRERED_EOS_TOKEN_ID_KEY, "4"),
            (FIRERED_PAD_TOKEN_ID_KEY, "2"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_aed_l_metadata() {
        let parsed = parse_firered_aed_execution_metadata(&aed_l_metadata()).expect("parse");
        assert_eq!(parsed.encoder_n_layers, 16);
        assert_eq!(parsed.d_model, 1280);
        assert_eq!(parsed.head_dim, 64);
        assert_eq!(parsed.encoder_max_frames(), 5000);
        assert_eq!(parsed.sos_token_id, 3);
        assert_eq!(parsed.eos_token_id, 4);
    }

    #[test]
    fn rejects_head_geometry_mismatch() {
        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_ENCODER_HEAD_DIM_KEY.to_string(), "60".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_even_conv_kernel_and_pe_len() {
        let mut metadata = aed_l_metadata();
        metadata.insert(
            FIRERED_ENCODER_CONV_KERNEL_KEY.to_string(),
            "32".to_string(),
        );
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());

        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_ENCODER_PE_LEN_KEY.to_string(), "10000".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_subsample_out_dim_mismatch() {
        let mut metadata = aed_l_metadata();
        metadata.insert(
            FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY.to_string(),
            "600".to_string(),
        );
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_special_token_out_of_vocab() {
        let mut metadata = aed_l_metadata();
        metadata.insert(FIRERED_EOS_TOKEN_ID_KEY.to_string(), "9000".to_string());
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_missing_key() {
        let mut metadata = aed_l_metadata();
        metadata.remove(FIRERED_VOCAB_SIZE_KEY);
        assert!(parse_firered_aed_execution_metadata(&metadata).is_err());
    }
}
