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
