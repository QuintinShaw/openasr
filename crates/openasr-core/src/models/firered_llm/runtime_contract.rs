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
//!
//! Admission depth: `validate_runtime_pack_contract` checks the three metadata
//! segments AND the complete runtime tensor set against the parsed geometry
//! (moonshine/whisper admission-gate depth), so a pack with a missing or
//! mis-shaped encoder/adapter/LLM tensor fails closed at `PackVerifier` time
//! instead of at first execution. The two tensors the importer carries for
//! provenance only (`enc.pos_enc.pe`, `firered.mel_filters`) are deliberately
//! not part of this contract: the runtime rebuilds the rel-pos table and the
//! fbank filterbank host-side and never binds them.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::models::firered_aed::runtime_contract::{
    FIRERED_ENCODER_CONV_KERNEL_KEY, FIRERED_ENCODER_D_MODEL_KEY, FIRERED_ENCODER_FEATURE_DIM_KEY,
    FIRERED_ENCODER_FFN_DIM_KEY, FIRERED_ENCODER_HEAD_DIM_KEY, FIRERED_ENCODER_N_HEADS_KEY,
    FIRERED_ENCODER_N_LAYERS_KEY, FIRERED_ENCODER_PE_LEN_KEY,
    FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
    FireRedAedExecutionMetadata,
};
use crate::models::runtime_contract::{
    MetadataContractError, ScalarMetadataView, required_u64_scalar, u64_to_usize,
    validate_bounded_usize, validate_positive_usize,
};
use crate::models::tensor_binding::{
    TensorBindingDescriptor, TensorBindingDescriptorRequirement, render_shape,
    validate_tensor_binding_descriptors,
};

use super::tensor_names::{
    ADAPTER_LINEAR1_BIAS, ADAPTER_LINEAR1_WEIGHT, ADAPTER_LINEAR2_BIAS, ADAPTER_LINEAR2_WEIGHT,
    LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, qwen2_llm_layer_tensor_names,
};

pub(crate) const FIRERED_LLM_CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
pub(crate) const FIRERED_LLM_CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";

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
/// Local ceiling for RoPE position tables; generous over production 32768.
pub(crate) const FIRERED_LLM_MAX_POSITIONS: usize = 1_048_576;

/// Architecture ceilings for the Conformer encoder / adapter admitted from
/// untrusted pack metadata. Production FireRed2-LLM encoder is 16L / d1280 /
/// 20 heads / ffn 5120 / feature_dim 80; ceilings match FunASR/Qwen headroom.
pub(crate) const FIRERED_LLM_MAX_ENCODER_LAYERS: usize = 512;
pub(crate) const FIRERED_LLM_MAX_D_MODEL: usize = 65_536;
pub(crate) const FIRERED_LLM_MAX_N_HEADS: usize = 1_024;
pub(crate) const FIRERED_LLM_MAX_HEAD_DIM: usize = 1_024;
pub(crate) const FIRERED_LLM_MAX_FFN_DIM: usize = 262_144;
pub(crate) const FIRERED_LLM_MAX_CONV_KERNEL: usize = 4_096;
pub(crate) const FIRERED_LLM_MAX_SUBSAMPLE_CHANNELS: usize = 4_096;
pub(crate) const FIRERED_LLM_MAX_FEATURE_DIM: usize = 4_096;
pub(crate) const FIRERED_LLM_MAX_PE_LEN: usize = 1_048_576;
pub(crate) const FIRERED_LLM_MAX_ADAPTER_DOWNSAMPLE: usize = 64;
/// Two stride-2 convs need feature_dim large enough that
/// `(((feature_dim - 1) / 2) - 1) / 2` stays non-negative; the minimum odd width
/// that survives both halvings with a positive residual is 5.
const FIRERED_LLM_MIN_FEATURE_DIM_FOR_SUBSAMPLE: usize = 5;
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
    for (key, value, max) in [
        (
            FIRERED_ENCODER_N_LAYERS_KEY,
            encoder_n_layers,
            FIRERED_LLM_MAX_ENCODER_LAYERS,
        ),
        (
            FIRERED_ENCODER_D_MODEL_KEY,
            d_model,
            FIRERED_LLM_MAX_D_MODEL,
        ),
        (
            FIRERED_ENCODER_N_HEADS_KEY,
            n_heads,
            FIRERED_LLM_MAX_N_HEADS,
        ),
        (
            FIRERED_ENCODER_HEAD_DIM_KEY,
            head_dim,
            FIRERED_LLM_MAX_HEAD_DIM,
        ),
        (
            FIRERED_ENCODER_FFN_DIM_KEY,
            encoder_ffn_dim,
            FIRERED_LLM_MAX_FFN_DIM,
        ),
        (
            FIRERED_ENCODER_CONV_KERNEL_KEY,
            conv_kernel,
            FIRERED_LLM_MAX_CONV_KERNEL,
        ),
        (
            FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY,
            subsample_channels,
            FIRERED_LLM_MAX_SUBSAMPLE_CHANNELS,
        ),
        (
            FIRERED_ENCODER_FEATURE_DIM_KEY,
            feature_dim,
            FIRERED_LLM_MAX_FEATURE_DIM,
        ),
        (
            FIRERED_ENCODER_PE_LEN_KEY,
            encoder_pe_len,
            FIRERED_LLM_MAX_PE_LEN,
        ),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    // Bound subsample_out_dim by the product of the two source ceilings.
    validate_bounded_usize(
        subsample_out_dim,
        FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY,
        FIRERED_LLM_MAX_SUBSAMPLE_CHANNELS.saturating_mul(FIRERED_LLM_MAX_FEATURE_DIM),
    )?;
    if n_heads.checked_mul(head_dim) != Some(d_model) {
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
    // Two successive stride-2 convs compute
    // `channels * (((feature_dim - 1) / 2 - 1) / 2)`. feature_dim must be large
    // enough that neither half underflows into a wrapping usize, and the
    // channel * width product must not overflow.
    if feature_dim < FIRERED_LLM_MIN_FEATURE_DIM_FOR_SUBSAMPLE {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_FEATURE_DIM_KEY,
            reason: format!(
                "feature_dim {feature_dim} is too small for two stride-2 subsampling stages (need >= {FIRERED_LLM_MIN_FEATURE_DIM_FOR_SUBSAMPLE})"
            ),
        });
    }
    let after_first = (feature_dim - 1) / 2;
    if after_first < 1 {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_FEATURE_DIM_KEY,
            reason: format!(
                "feature_dim {feature_dim} underflows the first stride-2 subsample stage"
            ),
        });
    }
    let subsampled_width = (after_first - 1) / 2;
    if subsampled_width == 0 {
        return Err(MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_FEATURE_DIM_KEY,
            reason: format!(
                "feature_dim {feature_dim} underflows the second stride-2 subsample stage"
            ),
        });
    }
    let expected_subsample = subsample_channels
        .checked_mul(subsampled_width)
        .ok_or_else(|| MetadataContractError::InvalidValue {
            key: FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY,
            reason: format!(
                "subsample_channels {subsample_channels} * subsampled_width {subsampled_width} overflows"
            ),
        })?;
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
    validate_bounded_usize(
        downsample_rate,
        FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY,
        FIRERED_LLM_MAX_ADAPTER_DOWNSAMPLE,
    )?;
    validate_bounded_usize(
        llm_dim,
        FIRERED_LLM_ADAPTER_LLM_DIM_KEY,
        FIRERED_LLM_MAX_D_MODEL,
    )?;
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
    use crate::models::qwen::{
        QWEN_DECODER_MAX_D_MODEL, QWEN_DECODER_MAX_FFN_DIM, QWEN_DECODER_MAX_HEAD_DIM,
        QWEN_DECODER_MAX_LAYERS, QWEN_DECODER_MAX_N_HEADS, QWEN_DECODER_MAX_VOCAB_SIZE,
    };
    for (key, value, max) in [
        (
            FIRERED_LLM_LLM_N_LAYERS_KEY,
            n_layers,
            QWEN_DECODER_MAX_LAYERS,
        ),
        (
            FIRERED_LLM_LLM_D_MODEL_KEY,
            d_model,
            QWEN_DECODER_MAX_D_MODEL,
        ),
        (
            FIRERED_LLM_LLM_N_HEADS_KEY,
            n_heads,
            QWEN_DECODER_MAX_N_HEADS,
        ),
        (
            FIRERED_LLM_LLM_N_KV_HEADS_KEY,
            n_kv_heads,
            QWEN_DECODER_MAX_N_HEADS,
        ),
        (
            FIRERED_LLM_LLM_HEAD_DIM_KEY,
            head_dim,
            QWEN_DECODER_MAX_HEAD_DIM,
        ),
        (
            FIRERED_LLM_LLM_FFN_DIM_KEY,
            ffn_dim,
            QWEN_DECODER_MAX_FFN_DIM,
        ),
        (
            FIRERED_LLM_LLM_VOCAB_SIZE_KEY,
            vocab_size,
            QWEN_DECODER_MAX_VOCAB_SIZE,
        ),
        (
            FIRERED_LLM_LLM_MAX_POSITIONS_KEY,
            max_positions,
            FIRERED_LLM_MAX_POSITIONS,
        ),
    ] {
        validate_bounded_usize(value, key, max)?;
    }
    if n_heads.checked_mul(head_dim) != Some(d_model) {
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

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FireRedLlmRuntimeTensorContractError {
    #[error("firered-llm missing required GGUF tensor '{name}'")]
    MissingRequiredTensor { name: String },
    #[error("firered-llm GGUF tensor '{name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        name: String,
        shape: String,
        reason: String,
    },
    #[error("firered-llm runtime tensor geometry overflowed: {reason}")]
    GeometryOverflow { reason: String },
}

/// Map firered-llm decoder metadata onto the shared Qwen-shaped geometry.
pub(crate) fn firered_llm_qwen_decoder_geometry(
    decoder: &FireRedLlmDecoderMetadata,
) -> crate::models::qwen::QwenDecoderContractGeometry {
    crate::models::qwen::QwenDecoderContractGeometry {
        n_layers: decoder.n_layers,
        d_model: decoder.d_model,
        n_heads: decoder.n_heads,
        n_kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
        ffn_dim: decoder.ffn_dim,
        vocab_size: decoder.vocab_size,
    }
}

/// Layer name provider for the Qwen2 decoder (`llm.blk.{i}.*`).
///
/// FireRed spells the attention output projection `attn_out.weight`; the
/// shared contract field is `attn_output_name`.
pub(crate) fn firered_llm_qwen_family_layer_names(
    layer: usize,
) -> crate::models::qwen::QwenFamilyLlmLayerTensorNames {
    let names = qwen2_llm_layer_tensor_names(layer);
    crate::models::qwen::QwenFamilyLlmLayerTensorNames {
        attn_norm_name: names.attn_norm_weight,
        attn_q_name: names.attn_q_weight,
        attn_k_name: names.attn_k_weight,
        attn_v_name: names.attn_v_weight,
        attn_output_name: names.attn_out_weight,
        q_norm_name: None,
        k_norm_name: None,
        q_bias_name: Some(names.attn_q_bias),
        k_bias_name: Some(names.attn_k_bias),
        v_bias_name: Some(names.attn_v_bias),
        ffn_norm_name: names.ffn_norm_weight,
        ffn_gate_name: names.ffn_gate_weight,
        ffn_up_name: names.ffn_up_weight,
        ffn_down_name: names.ffn_down_weight,
    }
}

/// Adapter-local Qwen2 profile for FireRedASR2-LLM: closed variant, layer
/// names, and tail. It is immediately geometry-bound into the contract
/// consumed by admission, planning, tail load, host quote, and compilation.
pub(crate) fn firered_llm_qwen_decoder_profile() -> crate::models::qwen::QwenFamilyDecoderProfile {
    crate::models::qwen::QwenFamilyDecoderProfile::new(
        crate::models::qwen::QwenDecoderVariant::Qwen2,
        firered_llm_qwen_family_layer_names,
        firered_llm_qwen_decoder_tail_names(),
    )
}

/// The Qwen2 decoder half: every `llm.blk.*` layer plus token embd / logits /
/// final norm. Expanded from the shared Qwen decoder contract Module
/// (ordered `ExactDims`) so the per-layer tensor set (base 9 + Qwen2 qkv-bias
/// 3 = 12) cannot drift from FunASR-Nano / MOSS / MiMo.
pub(crate) fn firered_llm_qwen_decoder_contract(
    decoder: &FireRedLlmDecoderMetadata,
) -> Result<crate::models::qwen::QwenDecoderContract, FireRedLlmRuntimeTensorContractError> {
    crate::models::qwen::QwenDecoderContract::bind(
        firered_llm_qwen_decoder_geometry(decoder),
        firered_llm_qwen_decoder_profile(),
    )
    .map_err(|reason| FireRedLlmRuntimeTensorContractError::GeometryOverflow { reason })
}

pub(crate) fn firered_llm_decoder_tensor_descriptors(
    decoder: &FireRedLlmDecoderMetadata,
) -> Result<Vec<TensorBindingDescriptor>, FireRedLlmRuntimeTensorContractError> {
    firered_llm_qwen_decoder_contract(decoder)?
        .runtime_tensor_descriptors()
        .map_err(|reason| FireRedLlmRuntimeTensorContractError::GeometryOverflow { reason })
}

/// Static tail tensor names shared by admission descriptors and the contract-
/// projected tail loader. Keep this the single spelling source for FireRed2-LLM.
pub(crate) fn firered_llm_qwen_decoder_tail_names()
-> crate::models::qwen::QwenDecoderTailTensorNames<'static> {
    crate::models::qwen::QwenDecoderTailTensorNames {
        output_norm: LLM_OUTPUT_NORM_WEIGHT,
        output_weight: Some(LLM_OUTPUT_WEIGHT),
        token_embd: LLM_TOKEN_EMBD_WEIGHT,
    }
}

/// The complete runtime-bound tensor set for one firered-llm pack, expressed
/// against the three parsed metadata segments: fbank/CMVN frontend vectors,
/// the `enc.*` Conformer branch (reused from `firered_aed`'s encoder graph),
/// the `adapter.*` frame-stacking projector, and the `llm.*` Qwen2 decoder.
/// Every tensor the executor/encoder/adapter/decoder graphs bind appears
/// exactly once; the importer is the only writer, so this descriptor list and
/// `package_import`'s tensor map are the two fail-closed ends of the same
/// contract.
pub(crate) fn firered_llm_runtime_tensor_binding_descriptors(
    encoder: &FireRedAedExecutionMetadata,
    adapter: &FireRedLlmAdapterMetadata,
    decoder: &FireRedLlmDecoderMetadata,
) -> Result<Vec<TensorBindingDescriptor>, FireRedLlmRuntimeTensorContractError> {
    let d_model = encoder.d_model;
    let ffn_dim = encoder.encoder_ffn_dim;
    let conv_kernel = encoder.conv_kernel;
    let subsample_channels = encoder.subsample_channels;
    let subsample_out_dim = encoder.subsample_out_dim;
    // FireRed AED/LLM encoder conv-module GLU: pw1 expands d_model -> 4*d_model,
    // depthwise/ln stay on 2*d_model after the GLU split (same geometry as
    // `firered_aed::runtime_contract` / `encoder_graph`).
    let double_d_model = d_model
        .checked_mul(2)
        .ok_or_else(|| geometry_overflow("2 x encoder d_model"))?;
    let quad_d_model = d_model
        .checked_mul(4)
        .ok_or_else(|| geometry_overflow("4 x encoder d_model"))?;

    let vector = |name: String, len: usize, what: &str| TensorBindingDescriptor {
        tensor_name: name,
        requirement: TensorBindingDescriptorRequirement::VectorLen(len),
        reason: format!("expected {what} vector"),
    };
    let matrix = |name: String, lhs: usize, rhs: usize, what: &str| TensorBindingDescriptor {
        tensor_name: name,
        // Ordered ggml [in, out] — same rule as firered_aed encoder admission.
        requirement: TensorBindingDescriptorRequirement::ExactDims(vec![lhs, rhs]),
        reason: format!("expected {what} matrix"),
    };

    let mut descriptors = vec![
        vector(
            FIRERED_LLM_CMVN_NEG_MEAN_TENSOR.to_string(),
            encoder.feature_dim,
            "feature_dim-sized CMVN neg_mean",
        ),
        vector(
            FIRERED_LLM_CMVN_INV_STDDEV_TENSOR.to_string(),
            encoder.feature_dim,
            "feature_dim-sized CMVN inv_stddev",
        ),
        TensorBindingDescriptor {
            tensor_name: "enc.subsample.conv1.weight".to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                3,
                3,
                1,
                subsample_channels,
            ]),
            reason: "expected [3, 3, 1, subsample_channels] subsampling conv1 kernel".to_string(),
        },
        vector(
            "enc.subsample.conv1.bias".to_string(),
            subsample_channels,
            "subsample_channels-sized conv1 bias",
        ),
        TensorBindingDescriptor {
            tensor_name: "enc.subsample.conv2.weight".to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                3,
                3,
                subsample_channels,
                subsample_channels,
            ]),
            reason:
                "expected [3, 3, subsample_channels, subsample_channels] subsampling conv2 kernel"
                    .to_string(),
        },
        vector(
            "enc.subsample.conv2.bias".to_string(),
            subsample_channels,
            "subsample_channels-sized conv2 bias",
        ),
        matrix(
            "enc.subsample.out.weight".to_string(),
            subsample_out_dim,
            d_model,
            "subsample projection",
        ),
        vector(
            "enc.subsample.out.bias".to_string(),
            d_model,
            "d_model-sized subsample bias",
        ),
    ];

    for layer_idx in 0..encoder.encoder_n_layers {
        let prefix = format!("enc.blk.{layer_idx}.");
        for ffn in ["ffn1", "ffn2"] {
            descriptors.extend([
                vector(
                    format!("{prefix}{ffn}.norm.weight"),
                    d_model,
                    "d_model-sized norm",
                ),
                vector(
                    format!("{prefix}{ffn}.norm.bias"),
                    d_model,
                    "d_model-sized norm",
                ),
                matrix(
                    format!("{prefix}{ffn}.up.weight"),
                    d_model,
                    ffn_dim,
                    "FFN up",
                ),
                vector(
                    format!("{prefix}{ffn}.up.bias"),
                    ffn_dim,
                    "ffn_dim-sized FFN up bias",
                ),
                matrix(
                    format!("{prefix}{ffn}.down.weight"),
                    ffn_dim,
                    d_model,
                    "FFN down",
                ),
                vector(
                    format!("{prefix}{ffn}.down.bias"),
                    d_model,
                    "d_model-sized FFN down bias",
                ),
            ]);
        }
        for norm in ["norm_q", "norm_k", "norm_v"] {
            descriptors.extend([
                vector(
                    format!("{prefix}attn.{norm}.weight"),
                    d_model,
                    "d_model-sized attention norm",
                ),
                vector(
                    format!("{prefix}attn.{norm}.bias"),
                    d_model,
                    "d_model-sized attention norm",
                ),
            ]);
        }
        for projection in ["q", "k", "v", "out", "pos"] {
            descriptors.push(matrix(
                format!("{prefix}attn.{projection}.weight"),
                d_model,
                d_model,
                "attention projection",
            ));
        }
        descriptors.extend([
            // Flattened `n_heads x head_dim`; metadata validation already
            // proves `n_heads * head_dim == d_model`.
            vector(
                format!("{prefix}attn.pos_bias_u"),
                d_model,
                "flattened rel-pos bias",
            ),
            vector(
                format!("{prefix}attn.pos_bias_v"),
                d_model,
                "flattened rel-pos bias",
            ),
            vector(
                format!("{prefix}conv.norm.weight"),
                d_model,
                "d_model-sized conv norm",
            ),
            vector(
                format!("{prefix}conv.norm.bias"),
                d_model,
                "d_model-sized conv norm",
            ),
            matrix(
                format!("{prefix}conv.pw1.weight"),
                d_model,
                quad_d_model,
                "GLU pointwise conv1 (kernel-1 squeezed) d_model x 4*d_model",
            ),
            TensorBindingDescriptor {
                tensor_name: format!("{prefix}conv.dw.weight"),
                requirement: TensorBindingDescriptorRequirement::ExactDims(vec![
                    conv_kernel,
                    1,
                    double_d_model,
                ]),
                reason: "expected [conv_kernel, 1, 2 x d_model] depthwise conv kernel".to_string(),
            },
            vector(
                format!("{prefix}conv.ln.weight"),
                double_d_model,
                "2*d_model-sized conv mid-block layer norm",
            ),
            vector(
                format!("{prefix}conv.ln.bias"),
                double_d_model,
                "2*d_model-sized conv mid-block layer norm",
            ),
            matrix(
                format!("{prefix}conv.pw2.weight"),
                double_d_model,
                d_model,
                "pointwise conv2 restore 2*d_model x d_model",
            ),
            vector(
                format!("{prefix}out_norm.weight"),
                d_model,
                "d_model-sized block out norm",
            ),
            vector(
                format!("{prefix}out_norm.bias"),
                d_model,
                "d_model-sized block out norm",
            ),
        ]);
    }

    let stacked_adapter_input = d_model
        .checked_mul(adapter.downsample_rate)
        .ok_or_else(|| geometry_overflow("encoder d_model x adapter downsample_rate"))?;
    descriptors.extend([
        matrix(
            ADAPTER_LINEAR1_WEIGHT.to_string(),
            stacked_adapter_input,
            adapter.llm_dim,
            "adapter linear1 (stacked encoder frames -> llm_dim)",
        ),
        vector(
            ADAPTER_LINEAR1_BIAS.to_string(),
            adapter.llm_dim,
            "llm_dim-sized adapter linear1 bias",
        ),
        matrix(
            ADAPTER_LINEAR2_WEIGHT.to_string(),
            adapter.llm_dim,
            adapter.llm_dim,
            "adapter linear2",
        ),
        vector(
            ADAPTER_LINEAR2_BIAS.to_string(),
            adapter.llm_dim,
            "llm_dim-sized adapter linear2 bias",
        ),
    ]);

    // Qwen2 decoder half via the shared contract (ExactDims, not local EitherDims matrix).
    descriptors.extend(firered_llm_decoder_tensor_descriptors(decoder)?);
    Ok(descriptors)
}

pub(crate) fn validate_firered_llm_runtime_tensors_with_index(
    index: &GgufTensorIndex,
    encoder: &FireRedAedExecutionMetadata,
    adapter: &FireRedLlmAdapterMetadata,
    decoder: &FireRedLlmDecoderMetadata,
) -> Result<(), FireRedLlmRuntimeTensorContractError> {
    let descriptors = firered_llm_runtime_tensor_binding_descriptors(encoder, adapter, decoder)?;
    validate_tensor_binding_descriptors(
        index,
        &descriptors,
        missing_required_tensor,
        invalid_tensor_shape,
    )
}

fn geometry_overflow(what: &str) -> FireRedLlmRuntimeTensorContractError {
    FireRedLlmRuntimeTensorContractError::GeometryOverflow {
        reason: format!("{what} overflowed usize"),
    }
}

fn missing_required_tensor(name: &str) -> FireRedLlmRuntimeTensorContractError {
    FireRedLlmRuntimeTensorContractError::MissingRequiredTensor {
        name: name.to_string(),
    }
}

fn invalid_tensor_shape(
    name: &str,
    shape: &[u64],
    reason: String,
) -> FireRedLlmRuntimeTensorContractError {
    FireRedLlmRuntimeTensorContractError::InvalidTensorShape {
        name: name.to_string(),
        shape: render_shape(shape),
        reason,
    }
}

pub(crate) fn validate_runtime_pack_contract(
    preflight: &crate::GgufRuntimeSourcePreflight,
) -> Result<(), String> {
    let encoder = parse_firered_llm_encoder_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("firered-llm", error)
    })?;
    let adapter = parse_firered_llm_adapter_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("firered-llm", error)
    })?;
    let decoder = parse_firered_llm_decoder_metadata(preflight.metadata()).map_err(|error| {
        crate::models::runtime_pack_contract::metadata_validation_error("firered-llm", error)
    })?;
    // Cross-segment contract: the adapter splices its rows into the LLM
    // decoder's embedding stream, so its output width must equal the LLM
    // hidden size. `package_import` proves this at conversion time; the
    // admission gate re-checks it so a hand-edited or corrupted header
    // cannot smuggle an inconsistent geometry past the executor.
    if adapter.llm_dim != decoder.d_model {
        return Err(
            crate::models::runtime_pack_contract::metadata_validation_error(
                "firered-llm",
                MetadataContractError::InvalidValue {
                    key: FIRERED_LLM_ADAPTER_LLM_DIM_KEY,
                    reason: format!(
                        "adapter llm_dim {} != firered_llm.llm.d_model {}",
                        adapter.llm_dim, decoder.d_model
                    ),
                },
            ),
        );
    }
    validate_firered_llm_runtime_tensors_with_index(
        preflight.tensor_index(),
        &encoder,
        &adapter,
        &decoder,
    )
    .map_err(crate::models::runtime_pack_contract::tensor_validation_error)
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

    /// Capacity regression anchor: the shared KV byte derivation on this
    /// family's real-checkpoint Qwen2 decoder geometry (28 layers, 4 KV
    /// heads, head_dim 128 -- the fixture values above), split by storage
    /// copy. Runs the derivation golden for every `Derived` family, not just
    /// the one that consumes an integral window today.
    #[test]
    fn kv_bytes_per_position_matches_the_reference_decoder_geometry() {
        use crate::capacity::{KvGeometry, kv_bytes_per_position};
        use crate::nn::decoder::LlmKvCacheSpec;

        let geometry = KvGeometry {
            n_layers: 28,
            kv_heads: 4,
            head_dim: 128,
        };
        // 28 layers * 2 (K+V) * 4 kv-heads = 224 rows per position.
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 224 * 512); // f32 rows
        assert_eq!(default.resident, 224 * 256); // f16 rows
        let q8_0 = kv_bytes_per_position(&geometry, LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 224 * 136); // 128 / 32 * 34 B q8_0 rows
        assert_eq!(q8_0.resident, 224 * 136);
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
    fn rejects_encoder_n_layers_above_architecture_ceiling() {
        let mut metadata = full_metadata();
        metadata.insert(
            FIRERED_ENCODER_N_LAYERS_KEY.to_string(),
            (FIRERED_LLM_MAX_ENCODER_LAYERS as u64 + 1).to_string(),
        );
        assert!(parse_firered_llm_encoder_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_feature_dim_too_small_for_subsample() {
        let mut metadata = full_metadata();
        metadata.insert(FIRERED_ENCODER_FEATURE_DIM_KEY.to_string(), "1".to_string());
        // Keep subsample_out_dim consistent with the broken formula so the
        // failure is the feature_dim underflow gate, not the out_dim mismatch.
        metadata.insert(
            FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY.to_string(),
            "0".to_string(),
        );
        assert!(parse_firered_llm_encoder_metadata(&metadata).is_err());
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

    // --- Admission-depth (metadata + tensor) fixtures ---------------------
    //
    // Small synthetic geometry (1 encoder layer, 1 decoder layer) so the
    // complete runtime-bound tensor set fits in a tiny GGUF fixture. Kept
    // deliberately distinct from the T2-dump values above so the shape math
    // (2 x d_model conv widths, stacked adapter input, GQA kv widths) is
    // exercised on non-square numbers.

    fn tensor_fixture_metadata() -> BTreeMap<String, String> {
        [
            (FIRERED_ENCODER_N_LAYERS_KEY, "1"),
            (FIRERED_ENCODER_D_MODEL_KEY, "8"),
            (FIRERED_ENCODER_N_HEADS_KEY, "2"),
            (FIRERED_ENCODER_HEAD_DIM_KEY, "4"),
            (FIRERED_ENCODER_FFN_DIM_KEY, "16"),
            (FIRERED_ENCODER_CONV_KERNEL_KEY, "3"),
            (FIRERED_ENCODER_SUBSAMPLE_CHANNELS_KEY, "4"),
            // 4 channels x (((8 - 1) / 2 - 1) / 2) = 4 subsampled mel rows.
            (FIRERED_ENCODER_SUBSAMPLE_OUT_DIM_KEY, "4"),
            (FIRERED_ENCODER_FEATURE_DIM_KEY, "8"),
            (FIRERED_ENCODER_PE_LEN_KEY, "5"),
            (FIRERED_LLM_ADAPTER_DOWNSAMPLE_RATE_KEY, "2"),
            (FIRERED_LLM_ADAPTER_LLM_DIM_KEY, "16"),
            (FIRERED_LLM_LLM_N_LAYERS_KEY, "1"),
            (FIRERED_LLM_LLM_D_MODEL_KEY, "16"),
            (FIRERED_LLM_LLM_N_HEADS_KEY, "4"),
            (FIRERED_LLM_LLM_N_KV_HEADS_KEY, "2"),
            (FIRERED_LLM_LLM_HEAD_DIM_KEY, "4"),
            (FIRERED_LLM_LLM_FFN_DIM_KEY, "32"),
            (FIRERED_LLM_LLM_VOCAB_SIZE_KEY, "64"),
            (FIRERED_LLM_LLM_MAX_POSITIONS_KEY, "128"),
            (FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY, "1"),
            (FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY, "2"),
            (FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY, "0"),
            (FIRERED_LLM_SPEECH_TOKEN_ID_KEY, "3"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Every runtime-bound tensor of the fixture pack. Encoder + adapter halves
    /// stay hand-written; the decoder half is projected from the shared Qwen
    /// descriptor set so positive fixtures cannot drift from admission.
    fn tensor_fixture_shapes() -> Vec<(String, Vec<u64>)> {
        let mut shapes: Vec<(String, Vec<u64>)> = vec![
            (FIRERED_LLM_CMVN_NEG_MEAN_TENSOR.to_string(), vec![8]),
            (FIRERED_LLM_CMVN_INV_STDDEV_TENSOR.to_string(), vec![8]),
            ("enc.subsample.conv1.weight".to_string(), vec![3, 3, 1, 4]),
            ("enc.subsample.conv1.bias".to_string(), vec![4]),
            ("enc.subsample.conv2.weight".to_string(), vec![3, 3, 4, 4]),
            ("enc.subsample.conv2.bias".to_string(), vec![4]),
            ("enc.subsample.out.weight".to_string(), vec![4, 8]),
            ("enc.subsample.out.bias".to_string(), vec![8]),
        ];
        let p = "enc.blk.0.";
        for (name, dims) in [
            ("ffn1.norm.weight", vec![8]),
            ("ffn1.norm.bias", vec![8]),
            ("ffn1.up.weight", vec![8, 16]),
            ("ffn1.up.bias", vec![16]),
            ("ffn1.down.weight", vec![16, 8]),
            ("ffn1.down.bias", vec![8]),
            ("attn.norm_q.weight", vec![8]),
            ("attn.norm_q.bias", vec![8]),
            ("attn.norm_k.weight", vec![8]),
            ("attn.norm_k.bias", vec![8]),
            ("attn.norm_v.weight", vec![8]),
            ("attn.norm_v.bias", vec![8]),
            ("attn.q.weight", vec![8, 8]),
            ("attn.k.weight", vec![8, 8]),
            ("attn.v.weight", vec![8, 8]),
            ("attn.out.weight", vec![8, 8]),
            ("attn.pos.weight", vec![8, 8]),
            ("attn.pos_bias_u", vec![8]),
            ("attn.pos_bias_v", vec![8]),
            ("conv.norm.weight", vec![8]),
            ("conv.norm.bias", vec![8]),
            // GLU: pw1 is d x 4d; dw/ln stay on 2d after the split (matches firered_aed).
            ("conv.pw1.weight", vec![8, 32]),
            ("conv.dw.weight", vec![3, 1, 16]),
            ("conv.ln.weight", vec![16]),
            ("conv.ln.bias", vec![16]),
            ("conv.pw2.weight", vec![16, 8]),
            ("ffn2.norm.weight", vec![8]),
            ("ffn2.norm.bias", vec![8]),
            ("ffn2.up.weight", vec![8, 16]),
            ("ffn2.up.bias", vec![16]),
            ("ffn2.down.weight", vec![16, 8]),
            ("ffn2.down.bias", vec![8]),
            ("out_norm.weight", vec![8]),
            ("out_norm.bias", vec![8]),
        ] {
            shapes.push((format!("{p}{name}"), dims));
        }
        shapes.extend([
            (ADAPTER_LINEAR1_WEIGHT.to_string(), vec![16, 16]),
            (ADAPTER_LINEAR1_BIAS.to_string(), vec![16]),
            (ADAPTER_LINEAR2_WEIGHT.to_string(), vec![16, 16]),
            (ADAPTER_LINEAR2_BIAS.to_string(), vec![16]),
        ]);
        let decoder =
            parse_firered_llm_decoder_metadata(&tensor_fixture_metadata()).expect("tiny decoder");
        shapes.extend(crate::models::tensor_binding::project_fixture_tensors(
            &firered_llm_decoder_tensor_descriptors(&decoder).expect("decoder descriptors"),
        ));
        shapes
    }

    fn write_tensor_fixture(
        shapes: &[(String, Vec<u64>)],
        metadata: BTreeMap<String, String>,
    ) -> tempfile::NamedTempFile {
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

        let mut spec = TinyGgufFixtureSpec::new(metadata);
        for (name, dims) in shapes {
            spec = spec.with_tensor_shape(name.clone(), dims.clone());
        }
        let file = tempfile::NamedTempFile::new().expect("temp file");
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write gguf fixture");
        file
    }

    fn run_admission_validator(file: &tempfile::NamedTempFile) -> Result<(), String> {
        let runtime_source =
            crate::validate_ggml_runtime_source_path(file.path()).expect("runtime source");
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
                &runtime_source,
            )
            .expect("runtime preflight");
        validate_runtime_pack_contract(&preflight)
    }

    #[test]
    fn admission_validator_accepts_a_complete_pack() {
        let shapes = tensor_fixture_shapes();
        let file = write_tensor_fixture(&shapes, tensor_fixture_metadata());
        run_admission_validator(&file).expect("complete pack must pass admission");
    }

    #[test]
    fn admission_validator_rejects_a_missing_llm_layer_tensor() {
        let missing = qwen2_llm_layer_tensor_names(0).ffn_gate_weight.to_string();
        let shapes: Vec<(String, Vec<u64>)> = tensor_fixture_shapes()
            .into_iter()
            .filter(|(name, _)| *name != missing)
            .collect();
        let file = write_tensor_fixture(&shapes, tensor_fixture_metadata());
        let error = run_admission_validator(&file).expect_err("missing tensor must fail closed");
        assert!(
            error.contains("llm.blk.0.ffn_gate.weight"),
            "error must name the missing tensor: {error}"
        );
    }

    #[test]
    fn admission_validator_rejects_a_misshapen_adapter_weight() {
        let shapes: Vec<(String, Vec<u64>)> = tensor_fixture_shapes()
            .into_iter()
            .map(|(name, dims)| {
                if name == ADAPTER_LINEAR1_WEIGHT {
                    // stacked input (8 x 2 = 16) x llm_dim (16) is the only
                    // admitted geometry; 15 rows cannot bind.
                    (name, vec![15, 16])
                } else {
                    (name, dims)
                }
            })
            .collect();
        let file = write_tensor_fixture(&shapes, tensor_fixture_metadata());
        let error = run_admission_validator(&file).expect_err("misshapen tensor must fail closed");
        assert!(
            error.contains(ADAPTER_LINEAR1_WEIGHT),
            "error must name the misshapen tensor: {error}"
        );
    }

    #[test]
    fn admission_validator_rejects_adapter_llm_dim_decoder_mismatch() {
        let mut metadata = tensor_fixture_metadata();
        metadata.insert(
            FIRERED_LLM_ADAPTER_LLM_DIM_KEY.to_string(),
            "24".to_string(),
        );
        let file = write_tensor_fixture(&tensor_fixture_shapes(), metadata);
        let error = run_admission_validator(&file)
            .expect_err("adapter/decoder width mismatch must fail closed");
        assert!(
            error.contains(FIRERED_LLM_ADAPTER_LLM_DIM_KEY),
            "error must name the inconsistent key: {error}"
        );
    }

    #[test]
    fn tensor_descriptors_cover_the_whole_runtime_bound_set() {
        let metadata = tensor_fixture_metadata();
        let encoder = parse_firered_llm_encoder_metadata(&metadata).expect("encoder");
        let adapter = parse_firered_llm_adapter_metadata(&metadata).expect("adapter");
        let decoder = parse_firered_llm_decoder_metadata(&metadata).expect("decoder");
        let descriptors =
            firered_llm_runtime_tensor_binding_descriptors(&encoder, &adapter, &decoder)
                .expect("descriptors");
        let descriptor_names: std::collections::BTreeSet<&str> =
            descriptors.iter().map(|d| d.tensor_name.as_str()).collect();
        let fixture_shapes = tensor_fixture_shapes();
        let fixture_names: std::collections::BTreeSet<&str> = fixture_shapes
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(
            descriptor_names, fixture_names,
            "descriptor list and the runtime-bound fixture set must match exactly"
        );
        assert_eq!(descriptors.len(), 61);
    }

    #[test]
    fn rejects_transposed_encoder_and_adapter_projections() {
        // Ordered ExactDims must reject HF [out, in] that Rank2EitherDims admitted.
        for (tensor_name, transposed) in [
            ("enc.blk.0.ffn1.up.weight", vec![16_u64, 8]),
            ("enc.blk.0.conv.pw1.weight", vec![32_u64, 8]),
            ("enc.subsample.out.weight", vec![8_u64, 4]),
            // Tiny adapter linear1 is square [16,16]; pin a wrong ordered pair.
            (ADAPTER_LINEAR1_WEIGHT, vec![8_u64, 16]),
        ] {
            let shapes: Vec<(String, Vec<u64>)> = tensor_fixture_shapes()
                .into_iter()
                .map(|(name, dims)| {
                    if name == tensor_name {
                        (name, transposed.clone())
                    } else {
                        (name, dims)
                    }
                })
                .collect();
            let file = write_tensor_fixture(&shapes, tensor_fixture_metadata());
            let error =
                run_admission_validator(&file).expect_err("transposed weight must fail closed");
            assert!(
                error.contains(tensor_name),
                "error must name {tensor_name}: {error}"
            );
        }
    }
}
