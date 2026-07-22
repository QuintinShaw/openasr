//! GGUF metadata contract for granite-speech: parses the
//! `granite_speech.{encoder,projector,decoder}.*` keys `package_import.rs`
//! writes back into the same config structs
//! `encoder_graph`/`qformer`/`decoder_graph` already accept, so an install-
//! time pack check (`api::backend::native::validate_native_runtime_model_pack_contract`)
//! and the executor read the exact same parsed values -- no second copy of
//! the hparam list to drift. Mirrors `arch::hparams::GRANITE_SPEECH_HPARAM_SCHEMA`'s
//! key list exactly.

#![allow(dead_code)]

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::oasr_metadata::{required_metadata_string, required_metadata_u32};

use super::decoder_graph::GraniteSpeechDecoderConfig;
use super::encoder_graph::GraniteSpeechEncoderConfig;
use super::qformer::GraniteSpeechProjectorConfig;

const GRANITE_SPEECH_CONTRACT_FAMILY: &str = "granite-speech";

fn required_metadata_f32(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<f32, NativeAsrError> {
    let raw = required_metadata_string(metadata, key, GRANITE_SPEECH_CONTRACT_FAMILY)?;
    raw.parse::<f32>()
        .map_err(|error| NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "granite-speech GGUF metadata key '{key}' is not a valid f32 ('{raw}'): {error}"
            ),
        })
}

fn u32_to_usize(key: &'static str, value: u32) -> Result<usize, NativeAsrError> {
    usize::try_from(value).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!("granite-speech GGUF metadata key '{key}' does not fit usize"),
    })
}

pub(crate) fn parse_encoder_metadata(
    metadata: &GgufMetadata,
) -> Result<GraniteSpeechEncoderConfig, NativeAsrError> {
    let u = |key: &'static str| -> Result<usize, NativeAsrError> {
        u32_to_usize(
            key,
            required_metadata_u32(metadata, key, GRANITE_SPEECH_CONTRACT_FAMILY)?,
        )
    };
    Ok(GraniteSpeechEncoderConfig {
        input_dim: u("granite_speech.encoder.input_dim")?,
        hidden_dim: u("granite_speech.encoder.hidden_dim")?,
        num_layers: u("granite_speech.encoder.num_layers")?,
        num_heads: u("granite_speech.encoder.num_heads")?,
        dim_head: u("granite_speech.encoder.dim_head")?,
        feedforward_mult: u("granite_speech.encoder.feedforward_mult")?,
        conv_kernel_size: u("granite_speech.encoder.conv_kernel_size")?,
        conv_expansion_factor: u("granite_speech.encoder.conv_expansion_factor")?,
        context_size: u("granite_speech.encoder.context_size")?,
        max_pos_emb: u("granite_speech.encoder.max_pos_emb")?,
        output_dim: u("granite_speech.encoder.output_dim")?,
        // Not stored as pack metadata (fixed architectural constants, never a
        // per-checkpoint variable for this architecture): mirrors the shipped
        // 4.1-2b checkpoint's values, same convention as e.g. whisper's fixed
        // layer-norm epsilon.
        layer_norm_eps: 1.0e-5,
        batch_norm_eps: 1.0e-5,
    })
}

pub(crate) fn parse_projector_metadata(
    metadata: &GgufMetadata,
) -> Result<GraniteSpeechProjectorConfig, NativeAsrError> {
    let u = |key: &'static str| -> Result<usize, NativeAsrError> {
        u32_to_usize(
            key,
            required_metadata_u32(metadata, key, GRANITE_SPEECH_CONTRACT_FAMILY)?,
        )
    };
    Ok(GraniteSpeechProjectorConfig {
        encoder_hidden_size: u("granite_speech.projector.encoder_hidden_size")?,
        llm_hidden_size: u("granite_speech.decoder.hidden_size")?,
        window_size: u32_to_usize(
            "granite_speech.window_size",
            required_metadata_u32(
                metadata,
                "granite_speech.window_size",
                GRANITE_SPEECH_CONTRACT_FAMILY,
            )?,
        )?,
        downsample_rate: u32_to_usize(
            "granite_speech.downsample_rate",
            required_metadata_u32(
                metadata,
                "granite_speech.downsample_rate",
                GRANITE_SPEECH_CONTRACT_FAMILY,
            )?,
        )?,
        num_hidden_layers: u("granite_speech.projector.num_hidden_layers")?,
        num_attention_heads: u("granite_speech.projector.num_attention_heads")?,
        intermediate_size: u("granite_speech.projector.intermediate_size")?,
        // Fixed architectural constant (BLIP-2 Q-Former), not a per-checkpoint
        // pack value.
        layer_norm_eps: 1.0e-12,
    })
}

pub(crate) fn parse_decoder_metadata(
    metadata: &GgufMetadata,
) -> Result<GraniteSpeechDecoderConfig, NativeAsrError> {
    let u = |key: &'static str| -> Result<usize, NativeAsrError> {
        u32_to_usize(
            key,
            required_metadata_u32(metadata, key, GRANITE_SPEECH_CONTRACT_FAMILY)?,
        )
    };
    let hidden_size = u("granite_speech.decoder.hidden_size")?;
    let num_heads = u("granite_speech.decoder.num_attention_heads")?;
    Ok(GraniteSpeechDecoderConfig {
        hidden_size,
        num_layers: u("granite_speech.decoder.num_hidden_layers")?,
        num_heads,
        num_kv_heads: u("granite_speech.decoder.num_key_value_heads")?,
        head_dim: u("granite_speech.decoder.head_dim")?,
        intermediate_size: u("granite_speech.decoder.intermediate_size")?,
        vocab_size: u("granite_speech.decoder.vocab_size")?,
        rms_norm_eps: required_metadata_f32(metadata, "granite_speech.decoder.rms_norm_eps")?,
        rope_theta: required_metadata_f32(metadata, "granite_speech.decoder.rope_theta")?,
        attention_multiplier: required_metadata_f32(
            metadata,
            "granite_speech.decoder.attention_multiplier",
        )?,
        embedding_multiplier: required_metadata_f32(
            metadata,
            "granite_speech.decoder.embedding_multiplier",
        )?,
        residual_multiplier: required_metadata_f32(
            metadata,
            "granite_speech.decoder.residual_multiplier",
        )?,
        logits_scaling: required_metadata_f32(metadata, "granite_speech.decoder.logits_scaling")?,
    })
}
