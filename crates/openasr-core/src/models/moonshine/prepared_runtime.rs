use thiserror::Error;

use super::runtime_contract::{
    MoonshineExecutionMetadata, parse_moonshine_execution_metadata,
    validate_moonshine_runtime_tensors_with_index,
};
use super::tokenizer::MoonshineTokenizer;
use super::weights::{
    MoonshineDecoderWeights, MoonshineEncoderWeights, load_moonshine_decoder_weights,
    load_moonshine_encoder_weights,
};
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;

#[derive(Debug, Clone)]
pub(crate) struct MoonshinePreparedRuntime {
    pub metadata: MoonshineExecutionMetadata,
    pub tokenizer: MoonshineTokenizer,
    pub encoder_weights: MoonshineEncoderWeights,
    pub decoder_weights: MoonshineDecoderWeights,
}

impl MoonshinePreparedRuntime {
    pub(crate) fn system_memory_quote(
        metadata: &crate::GgufMetadata,
        tensor_index: &crate::GgufTensorIndex,
        pack_content_id: &str,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        crate::models::system_memory_owner::SystemMemoryOwnerError,
    > {
        use crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder;

        let execution = parse_moonshine_execution_metadata(metadata).map_err(|error| {
            crate::models::system_memory_owner::SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                error.to_string(),
            )
        })?;
        let mut quote = PreparedRuntimeQuoteBuilder::new::<Self>(pack_content_id);
        quote.add_tokenizer_metadata(metadata, false)?;
        for name in [
            "enc.conv1.weight",
            "enc.conv2.weight",
            "enc.conv2.bias",
            "enc.conv3.weight",
            "enc.conv3.bias",
            "enc.groupnorm.weight",
            "enc.groupnorm.bias",
            "enc.out_norm.weight",
            "dec.emb.weight",
            "dec.out_norm.weight",
        ] {
            quote.add_tensor_f32(tensor_index, name)?;
        }
        quote.add_structural_bytes(
            checked_layer_descriptor_bytes::<super::weights::MoonshineEncoderLayerWeights>(
                execution.encoder_layers,
                "moonshine encoder layer descriptors",
            )?,
            "moonshine encoder layer descriptors",
        )?;
        quote.add_structural_bytes(
            checked_layer_descriptor_bytes::<super::weights::MoonshineDecoderLayerWeights>(
                execution.decoder_layers,
                "moonshine decoder layer descriptors",
            )?,
            "moonshine decoder layer descriptors",
        )?;
        for layer_idx in 0..execution.encoder_layers {
            let prefix = format!("enc.blk.{layer_idx}.");
            for suffix in [
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_up.bias",
                "ffn_down.bias",
            ] {
                quote.add_tensor_f32(tensor_index, &format!("{prefix}{suffix}"))?;
            }
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_o.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                quote.add_tensor_metadata(tensor_index, &format!("{prefix}{suffix}"))?;
            }
        }
        for layer_idx in 0..execution.decoder_layers {
            let prefix = format!("dec.blk.{layer_idx}.");
            for suffix in [
                "attn_norm.weight",
                "cross_norm.weight",
                "ffn_norm.weight",
                "ffn_up.bias",
                "ffn_down.bias",
            ] {
                quote.add_tensor_f32(tensor_index, &format!("{prefix}{suffix}"))?;
            }
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_o.weight",
                "cross_q.weight",
                "cross_k.weight",
                "cross_v.weight",
                "cross_o.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                quote.add_tensor_metadata(tensor_index, &format!("{prefix}{suffix}"))?;
            }
        }
        quote.finish()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "moonshine prepared tokenizer",
        )?;
        bytes.add(
            self.encoder_weights.retained_system_memory_bytes()?,
            "moonshine prepared encoder weights",
        )?;
        bytes.add(
            self.decoder_weights.retained_system_memory_bytes()?,
            "moonshine prepared decoder weights",
        )?;
        Ok(bytes.finish())
    }
}

fn checked_layer_descriptor_bytes<T>(
    count: usize,
    label: &str,
) -> Result<u64, crate::models::system_memory_owner::SystemMemoryOwnerError> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        crate::models::system_memory_owner::SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            format!("{label} byte count overflowed"),
        )
    })?;
    u64::try_from(bytes).map_err(|_| {
        crate::models::system_memory_owner::SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            format!("{label} byte count does not fit u64"),
        )
    })
}

#[derive(Debug, Error)]
pub(crate) enum MoonshinePreparedRuntimeError {
    #[error("moonshine runtime contract check failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("moonshine runtime tensor reader build failed: {reason}")]
    TensorReaderBuildFailed { reason: String },
    #[error("moonshine tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("moonshine encoder weight build failed: {reason}")]
    EncoderWeightsBuildFailed { reason: String },
    #[error("moonshine decoder weight build failed: {reason}")]
    DecoderWeightsBuildFailed { reason: String },
}

pub(crate) fn build_moonshine_prepared_runtime(
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<MoonshinePreparedRuntime, MoonshinePreparedRuntimeError> {
    let metadata = parse_moonshine_execution_metadata(&preflight.metadata).map_err(|error| {
        MoonshinePreparedRuntimeError::RuntimeContractViolation {
            reason: error.to_string(),
        }
    })?;
    validate_moonshine_runtime_tensors_with_index(&preflight.tensor_index, metadata).map_err(
        |error| MoonshinePreparedRuntimeError::RuntimeContractViolation {
            reason: error.to_string(),
        },
    )?;
    let tensor_reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        MoonshinePreparedRuntimeError::TensorReaderBuildFailed {
            reason: error.to_string(),
        }
    })?;
    let tokenizer =
        MoonshineTokenizer::from_gguf_metadata(&preflight.metadata).map_err(|error| {
            MoonshinePreparedRuntimeError::TokenizerBuildFailed {
                reason: error.to_string(),
            }
        })?;
    let encoder_weights =
        load_moonshine_encoder_weights(&tensor_reader, metadata).map_err(|error| {
            MoonshinePreparedRuntimeError::EncoderWeightsBuildFailed {
                reason: error.to_string(),
            }
        })?;
    let decoder_weights =
        load_moonshine_decoder_weights(&tensor_reader, metadata).map_err(|error| {
            MoonshinePreparedRuntimeError::DecoderWeightsBuildFailed {
                reason: error.to_string(),
            }
        })?;
    Ok(MoonshinePreparedRuntime {
        metadata,
        tokenizer,
        encoder_weights,
        decoder_weights,
    })
}
