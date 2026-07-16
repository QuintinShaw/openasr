//! firered-llm execution metadata parsed from the `.oasr` GGUF header.
//!
//! The encoder branch reuses `firered_aed`'s own key namespace
//! (`firered.encoder.*` / `firered.audio.*` -- see `package_import`'s module
//! doc), but `firered_aed::runtime_contract::parse_firered_aed_execution_metadata`
//! cannot be reused as-is: it also requires `firered.decoder.*` /
//! `firered.vocab_size` / `firered.{sos,eos,pad}_token_id` keys that only
//! exist for the AED decoder branch, which this family has none of (LLM
//! decode is Qwen2, not the AED Transformer decoder). This module parses just
//! the encoder-relevant subset directly with the SAME key constants
//! (`firered_aed::runtime_contract`'s `pub(crate)` `FIRERED_ENCODER_*_KEY`
//! constants), then bridges into `firered_aed::encoder_graph`'s
//! `FireRedAedExecutionMetadata` shape with inert placeholder values for the
//! fields `FireRedEncoderGraphRuntime` never reads for an encoder-only run
//! (`decoder_n_layers`/`decoder_ffn_dim`/`decoder_pe_len`/`vocab_size`/
//! `sos_token_id`/`eos_token_id`/`pad_token_id` -- verified by reading
//! `encoder_graph.rs`, which only ever touches `metadata.encoder_*` /
//! `d_model` / `n_heads` / `head_dim` / `feature_dim` / `subsample_*`).

use crate::models::firered_aed::runtime_contract::{
    FIRERED_ENCODER_CONV_KERNEL_KEY, FIRERED_ENCODER_D_MODEL_KEY, FIRERED_ENCODER_FEATURE_DIM_KEY,
    FIRERED_ENCODER_FFN_DIM_KEY, FIRERED_ENCODER_HEAD_DIM_KEY, FIRERED_ENCODER_N_HEADS_KEY,
    FIRERED_ENCODER_N_LAYERS_KEY, FIRERED_ENCODER_PE_LEN_KEY,
    FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
    FireRedAedExecutionMetadata,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_usize,
    validate_positive_usize,
};

pub(crate) const FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY: &str =
    "firered_llm.adapter.downsample_rate";
pub(crate) const FIRERED_LLM_ADAPTER_LLM_DIM_KEY: &str = "firered_llm.adapter.llm_dim";
pub(crate) const FIRERED_LLM_LLM_N_LAYERS_KEY: &str = "firered_llm.llm.n_layers";
pub(crate) const FIRERED_LLM_LLM_D_MODEL_KEY: &str = "firered_llm.llm.d_model";
pub(crate) const FIRERED_LLM_LLM_N_HEADS_KEY: &str = "firered_llm.llm.n_heads";
pub(crate) const FIRERED_LLM_LLM_N_KV_HEADS_KEY: &str = "firered_llm.llm.n_kv_heads";
pub(crate) const FIRERED_LLM_LLM_HEAD_DIM_KEY: &str = "firered_llm.llm.head_dim";
pub(crate) const FIRERED_LLM_LLM_FFN_DIM_KEY: &str = "firered_llm.llm.ffn_dim";
pub(crate) const FIRERED_LLM_LLM_VOCAB_SIZE_KEY: &str = "firered_llm.llm.vocab_size";
pub(crate) const FIRERED_LLM_LLM_MAX_POSITIONS_KEY: &str = "firered_llm.llm.max_positions";
pub(crate) const FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY: &str =
    "firered_llm.llm.chatml_im_start_token_id";
pub(crate) const FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY: &str =
    "firered_llm.llm.chatml_im_end_token_id";
pub(crate) const FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY: &str = "firered_llm.llm.endoftext_token_id";
pub(crate) const FIRERED_LLM_SPEECH_TOKEN_ID_KEY: &str = "firered_llm.llm.speech_token_id";

/// rope_theta and the RMSNorm epsilon are fixed properties of the official
/// Qwen2-7B-Instruct architecture (`config.json`'s `rope_theta` /
/// `rms_norm_eps`, verified in `scratchpad/fr2/T1-findings.md`), not derived
/// from the checkpoint -- the same "family constant, not a metadata key"
/// convention `qwen::llm_transformer`'s `DEFAULT_RMS_NORM_EPSILON` /
/// `rope_theta: 1_000_000.0` already use for qwen3-asr.
pub(crate) const FIRERED_LLM_ROPE_THETA: f32 = 1_000_000.0;
pub(crate) const FIRERED_LLM_RMS_NORM_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FireRedLlmAdapterMetadata {
    pub downsample_rate: usize,
    pub llm_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FireRedLlmDecoderMetadata {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub chatml_im_start_token_id: u32,
    pub chatml_im_end_token_id: u32,
    pub endoftext_token_id: u32,
    pub speech_token_id: u32,
}

/// Parse the `firered.encoder.*` / `firered.audio.*` subset into the exact
/// shape `firered_aed::encoder_graph::FireRedEncoderGraphRuntime::new` wants,
/// so the encoder graph/weights code (architecturally identical, see this
/// module's doc comment) can be reused byte-for-byte against a firered-llm
/// pack's OWN `enc.*` tensors (never the published `firered-aed-l-v2` pack --
/// the two families' encoder weights are independently trained, see
/// `package_import`'s module doc).
pub(crate) fn parse_firered_llm_encoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedAedExecutionMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
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
        // Inert placeholders: this family has no AED Transformer decoder, and
        // `FireRedEncoderGraphRuntime` never reads these fields for an
        // encoder-only run (verified against `encoder_graph.rs`). Kept
        // internally consistent (positive, in-range) rather than zeroed, so a
        // future accidental read fails on a wrong-looking value instead of a
        // suspicious-looking zero.
        decoder_n_layers: 1,
        decoder_ffn_dim: 1,
        decoder_pe_len: 1,
        vocab_size: 1,
        sos_token_id: 0,
        eos_token_id: 0,
        pad_token_id: 0,
    })
}

pub(crate) fn parse_firered_llm_adapter_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedLlmAdapterMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let downsample_rate = usize_key(FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY)?;
    let llm_dim = usize_key(FIRERED_LLM_ADAPTER_LLM_DIM_KEY)?;
    validate_positive_usize(downsample_rate, FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY)?;
    validate_positive_usize(llm_dim, FIRERED_LLM_ADAPTER_LLM_DIM_KEY)?;
    Ok(FireRedLlmAdapterMetadata {
        downsample_rate,
        llm_dim,
    })
}

pub(crate) fn parse_firered_llm_decoder_metadata<M: ScalarMetadataView>(
    metadata: &M,
) -> Result<FireRedLlmDecoderMetadata, MetadataContractError> {
    let usize_key = |key: &'static str| -> Result<usize, MetadataContractError> {
        u64_to_usize(required_u64_scalar(metadata, key)?, key)
    };
    let u32_key = |key: &'static str| -> Result<u32, MetadataContractError> {
        crate::models::runtime_contract::u64_to_u32(required_u64_scalar(metadata, key)?, key)
    };
    let n_layers = usize_key(FIRERED_LLM_LLM_N_LAYERS_KEY)?;
    let d_model = usize_key(FIRERED_LLM_LLM_D_MODEL_KEY)?;
    let n_heads = usize_key(FIRERED_LLM_LLM_N_HEADS_KEY)?;
    let n_kv_heads = usize_key(FIRERED_LLM_LLM_N_KV_HEADS_KEY)?;
    let head_dim = usize_key(FIRERED_LLM_LLM_HEAD_DIM_KEY)?;
    let ffn_dim = usize_key(FIRERED_LLM_LLM_FFN_DIM_KEY)?;
    let vocab_size = usize_key(FIRERED_LLM_LLM_VOCAB_SIZE_KEY)?;
    let max_positions = usize_key(FIRERED_LLM_LLM_MAX_POSITIONS_KEY)?;
    let chatml_im_start_token_id = u32_key(FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY)?;
    let chatml_im_end_token_id = u32_key(FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY)?;
    let endoftext_token_id = u32_key(FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY)?;
    let speech_token_id = u32_key(FIRERED_LLM_SPEECH_TOKEN_ID_KEY)?;

    for (key, value) in [
        (FIRERED_LLM_LLM_N_LAYERS_KEY, n_layers),
        (FIRERED_LLM_LLM_D_MODEL_KEY, d_model),
        (FIRERED_LLM_LLM_N_HEADS_KEY, n_heads),
        (FIRERED_LLM_LLM_N_KV_HEADS_KEY, n_kv_heads),
        (FIRERED_LLM_LLM_HEAD_DIM_KEY, head_dim),
        (FIRERED_LLM_LLM_FFN_DIM_KEY, ffn_dim),
        (FIRERED_LLM_LLM_VOCAB_SIZE_KEY, vocab_size),
        (FIRERED_LLM_LLM_MAX_POSITIONS_KEY, max_positions),
    ] {
        validate_positive_usize(value, key)?;
    }
    if n_heads * head_dim != d_model {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_LLM_LLM_HEAD_DIM_KEY,
            reason: format!("n_heads {n_heads} * head_dim {head_dim} != d_model {d_model}"),
        });
    }
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_LLM_LLM_N_KV_HEADS_KEY,
            reason: format!("n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"),
        });
    }
    for (key, id) in [
        (
            FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY,
            chatml_im_start_token_id,
        ),
        (
            FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY,
            chatml_im_end_token_id,
        ),
        (FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY, endoftext_token_id),
        (FIRERED_LLM_SPEECH_TOKEN_ID_KEY, speech_token_id),
    ] {
        if (id as usize) >= vocab_size {
            return Err(MetadataContractError::InvalidValue {
                key,
                reason: format!("token id {id} out of range for vocab_size {vocab_size}"),
            });
        }
    }

    Ok(FireRedLlmDecoderMetadata {
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_dim,
        vocab_size,
        max_positions,
        chatml_im_start_token_id,
        chatml_im_end_token_id,
        endoftext_token_id,
        speech_token_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn full_metadata() -> BTreeMap<String, String> {
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
            (FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY, "2"),
            (FIRERED_LLM_ADAPTER_LLM_DIM_KEY, "3584"),
            (FIRERED_LLM_LLM_N_LAYERS_KEY, "28"),
            (FIRERED_LLM_LLM_D_MODEL_KEY, "3584"),
            (FIRERED_LLM_LLM_N_HEADS_KEY, "28"),
            (FIRERED_LLM_LLM_N_KV_HEADS_KEY, "4"),
            (FIRERED_LLM_LLM_HEAD_DIM_KEY, "128"),
            (FIRERED_LLM_LLM_FFN_DIM_KEY, "18944"),
            (FIRERED_LLM_LLM_VOCAB_SIZE_KEY, "152064"),
            (FIRERED_LLM_LLM_MAX_POSITIONS_KEY, "32768"),
            (FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY, "151644"),
            (FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY, "151645"),
            (FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY, "151643"),
            (FIRERED_LLM_SPEECH_TOKEN_ID_KEY, "151646"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    #[test]
    fn parses_encoder_metadata_matching_t2_dump() {
        let parsed = parse_firered_llm_encoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.encoder_n_layers, 16);
        assert_eq!(parsed.d_model, 1280);
        assert_eq!(parsed.head_dim, 64);
    }

    #[test]
    fn parses_adapter_metadata_matching_t2_dump() {
        let parsed = parse_firered_llm_adapter_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.downsample_rate, 2);
        assert_eq!(parsed.llm_dim, 3584);
    }

    #[test]
    fn parses_decoder_metadata_matching_t2_dump() {
        let parsed = parse_firered_llm_decoder_metadata(&full_metadata()).expect("parse");
        assert_eq!(parsed.n_layers, 28);
        assert_eq!(parsed.n_kv_heads, 4);
        assert_eq!(parsed.speech_token_id, 151_646);
        assert_eq!(parsed.chatml_im_end_token_id, 151_645);
    }

    #[test]
    fn rejects_kv_heads_not_dividing_heads() {
        let mut metadata = full_metadata();
        metadata.insert(FIRERED_LLM_LLM_N_KV_HEADS_KEY.to_string(), "3".to_string());
        assert!(parse_firered_llm_decoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_token_id_out_of_vocab() {
        let mut metadata = full_metadata();
        metadata.insert(
            FIRERED_LLM_SPEECH_TOKEN_ID_KEY.to_string(),
            "999999".to_string(),
        );
        assert!(parse_firered_llm_decoder_metadata(&metadata).is_err());
    }
}
