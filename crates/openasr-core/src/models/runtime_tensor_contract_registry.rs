//! Runtime tensor-contract validation + descriptor expansion for the
//! **data-driven composer** families only (Cohere Transcribe + Qwen3-ASR) — the
//! two whose runtime tensors are validated/expanded centrally before graph
//! assembly. Reached only from those families' import + prepared-runtime paths.
//!
//! Dedicated-executor families validate their own tensor sets in their family
//! modules (e.g. `validate_stage_against_descriptor`) and never route a
//! contract id through here. Known dedicated-executor contract ids are matched
//! explicitly so generic tooling sees a first-class fail-closed marker instead
//! of a generic unknown-contract error.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::arch::{
    COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID, MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID,
    OpenAsrArchitectureRegistry, PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID,
    QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID, WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID,
    XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID,
};
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;
use crate::models::runtime_contract::ScalarMetadataView;

use super::cohere::runtime_contract::{
    CohereTranscribeExecutionMetadata, cohere_transcribe_runtime_tensor_descriptors,
    parse_cohere_transcribe_execution_metadata,
    validate_cohere_transcribe_runtime_tensors_with_index,
};
use super::qwen::runtime_contract::{
    Qwen3AsrExecutionMetadata, parse_qwen3_execution_metadata, qwen3_runtime_tensor_descriptors,
    validate_qwen3_runtime_tensors_with_index,
};
use super::tensor_binding::TensorBindingDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeTensorContractMetadata {
    CohereTranscribe(CohereTranscribeExecutionMetadata),
    Qwen3Asr(Qwen3AsrExecutionMetadata),
}

impl RuntimeTensorContractMetadata {
    fn kind_label(self) -> &'static str {
        match self {
            Self::CohereTranscribe(_) => "cohere-transcribe",
            Self::Qwen3Asr(_) => QWEN3_ASR_MODEL_FAMILY,
        }
    }

    pub(crate) fn into_cohere_transcribe(self) -> Option<CohereTranscribeExecutionMetadata> {
        match self {
            Self::CohereTranscribe(metadata) => Some(metadata),
            Self::Qwen3Asr(_) => None,
        }
    }

    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrExecutionMetadata> {
        match self {
            Self::Qwen3Asr(metadata) => Some(metadata),
            Self::CohereTranscribe(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DedicatedRuntimeTensorContractFamily {
    ParakeetCtc,
    ParakeetTdt,
    Wav2Vec2Ctc,
    XasrZipformer,
    Moonshine,
    SenseVoice,
    FireRedAed,
    FireRedLlm,
}

impl DedicatedRuntimeTensorContractFamily {
    fn label(self) -> &'static str {
        match self {
            Self::ParakeetCtc => "parakeet-ctc",
            Self::ParakeetTdt => "parakeet-tdt",
            Self::Wav2Vec2Ctc => "wav2vec2-ctc",
            Self::XasrZipformer => "xasr-zipformer",
            Self::Moonshine => "moonshine",
            Self::SenseVoice => "sensevoice",
            Self::FireRedAed => "firered-aed",
            Self::FireRedLlm => "firered-llm",
        }
    }
}

impl std::fmt::Display for DedicatedRuntimeTensorContractFamily {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeTensorContractRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown runtime tensor contract '{contract_id}'")]
    UnknownContract { contract_id: String },
    #[error(
        "runtime tensor contract '{contract_id}' belongs to dedicated executor family '{family}' and is not materialized by the composer registry"
    )]
    DedicatedExecutorContract {
        contract_id: String,
        family: DedicatedRuntimeTensorContractFamily,
    },
    #[error(
        "runtime tensor contract '{contract_id}' expected metadata for '{expected_kind}', got '{found_kind}'"
    )]
    MetadataKindMismatch {
        contract_id: String,
        expected_kind: &'static str,
        found_kind: &'static str,
    },
    #[error("runtime tensor contract '{contract_id}' metadata parse failed: {reason}")]
    MetadataParseFailed { contract_id: String, reason: String },
    #[error("runtime tensor contract '{contract_id}' validation failed: {reason}")]
    ValidationFailed { contract_id: String, reason: String },
}

pub(crate) fn resolve_builtin_runtime_tensor_contract_descriptors(
    contract_id: &str,
    metadata: RuntimeTensorContractMetadata,
) -> Result<Vec<TensorBindingDescriptor>, RuntimeTensorContractRegistryError> {
    match (contract_id, metadata) {
        (
            QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::Qwen3Asr(metadata),
        ) => Ok(qwen3_runtime_tensor_descriptors(metadata)),
        (
            COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::CohereTranscribe(metadata),
        ) => Ok(cohere_transcribe_runtime_tensor_descriptors(metadata)),
        (QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID, metadata) => {
            Err(RuntimeTensorContractRegistryError::MetadataKindMismatch {
                contract_id: contract_id.to_string(),
                expected_kind: QWEN3_ASR_MODEL_FAMILY,
                found_kind: metadata.kind_label(),
            })
        }
        (COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID, metadata) => {
            Err(RuntimeTensorContractRegistryError::MetadataKindMismatch {
                contract_id: contract_id.to_string(),
                expected_kind: "cohere-transcribe",
                found_kind: metadata.kind_label(),
            })
        }
        (contract_id, _) => dedicated_executor_contract_error(contract_id).map_or_else(
            || {
                Err(RuntimeTensorContractRegistryError::UnknownContract {
                    contract_id: contract_id.to_string(),
                })
            },
            Err,
        ),
    }
}

pub(crate) fn validate_builtin_runtime_tensor_contract_for_architecture<M: ScalarMetadataView>(
    model_architecture: &str,
    metadata: &M,
    tensor_index: &GgufTensorIndex,
) -> Result<RuntimeTensorContractMetadata, RuntimeTensorContractRegistryError> {
    let descriptor = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .ok_or_else(|| RuntimeTensorContractRegistryError::UnknownArchitecture {
            model_architecture: model_architecture.to_string(),
        })?;
    match descriptor.runtime_tensor_contract_id {
        QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID => {
            let metadata = parse_qwen3_execution_metadata(metadata).map_err(|error| {
                RuntimeTensorContractRegistryError::MetadataParseFailed {
                    contract_id: descriptor.runtime_tensor_contract_id.to_string(),
                    reason: error.to_string(),
                }
            })?;
            validate_qwen3_runtime_tensors_with_index(tensor_index, metadata).map_err(|error| {
                RuntimeTensorContractRegistryError::ValidationFailed {
                    contract_id: descriptor.runtime_tensor_contract_id.to_string(),
                    reason: error.to_string(),
                }
            })?;
            Ok(RuntimeTensorContractMetadata::Qwen3Asr(metadata))
        }
        COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID => {
            let metadata =
                parse_cohere_transcribe_execution_metadata(metadata).map_err(|error| {
                    RuntimeTensorContractRegistryError::MetadataParseFailed {
                        contract_id: descriptor.runtime_tensor_contract_id.to_string(),
                        reason: error.to_string(),
                    }
                })?;
            validate_cohere_transcribe_runtime_tensors_with_index(tensor_index, metadata).map_err(
                |error| RuntimeTensorContractRegistryError::ValidationFailed {
                    contract_id: descriptor.runtime_tensor_contract_id.to_string(),
                    reason: error.to_string(),
                },
            )?;
            Ok(RuntimeTensorContractMetadata::CohereTranscribe(metadata))
        }
        contract_id => dedicated_executor_contract_error(contract_id).map_or_else(
            || {
                Err(RuntimeTensorContractRegistryError::UnknownContract {
                    contract_id: contract_id.to_string(),
                })
            },
            Err,
        ),
    }
}

pub(crate) fn validate_builtin_runtime_tensor_contract_preflight(
    model_architecture: &str,
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<RuntimeTensorContractMetadata, RuntimeTensorContractRegistryError> {
    validate_builtin_runtime_tensor_contract_for_architecture(
        model_architecture,
        &preflight.metadata,
        &preflight.tensor_index,
    )
}

fn dedicated_runtime_tensor_contract_family(
    contract_id: &str,
) -> Option<DedicatedRuntimeTensorContractFamily> {
    match contract_id {
        PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::ParakeetCtc)
        }
        crate::arch::PARAKEET_TDT_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::ParakeetTdt)
        }
        WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::Wav2Vec2Ctc)
        }
        XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::XasrZipformer)
        }
        MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::Moonshine)
        }
        crate::arch::SENSEVOICE_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::SenseVoice)
        }
        crate::arch::FIRERED_AED_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::FireRedAed)
        }
        crate::arch::FIRERED_LLM_RUNTIME_TENSOR_CONTRACT_ID => {
            Some(DedicatedRuntimeTensorContractFamily::FireRedLlm)
        }
        _ => None,
    }
}

fn dedicated_executor_contract_error(
    contract_id: &str,
) -> Option<RuntimeTensorContractRegistryError> {
    dedicated_runtime_tensor_contract_family(contract_id).map(|family| {
        RuntimeTensorContractRegistryError::DedicatedExecutorContract {
            contract_id: contract_id.to_string(),
            family,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 80,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 2,
            audio_d_model: 16,
            audio_heads: 2,
            llm_layers: 2,
            llm_d_model: 16,
            llm_heads: 2,
            llm_kv_heads: 2,
            llm_head_dim: 8,
            vocab_size: 32,
            llm_max_positions: 256,
            audio_start_token_id: 2,
            audio_end_token_id: 3,
            audio_pad_token_id: 4,
            eos_token_id: 5,
            pad_token_id: 6,
        }
    }

    fn cohere_metadata() -> CohereTranscribeExecutionMetadata {
        CohereTranscribeExecutionMetadata {
            vocab_size: 16_384,
            encoder_layers: 2,
            encoder_d_model: 1_280,
            encoder_heads: 8,
            encoder_head_dim: 160,
            encoder_ffn_dim: 5_120,
            encoder_conv_kernel: 9,
            decoder_layers: 2,
            decoder_d_model: 1_024,
            decoder_heads: 8,
            decoder_head_dim: 128,
            decoder_ffn_dim: 4_096,
            decoder_max_context: 1_024,
            decoder_start_token_id: 13_764,
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
        }
    }

    #[test]
    fn resolves_qwen_builtin_contract() {
        let descriptors = resolve_builtin_runtime_tensor_contract_descriptors(
            QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::Qwen3Asr(qwen_metadata()),
        )
        .expect("qwen descriptors");

        assert!(
            descriptors
                .iter()
                .any(|descriptor| descriptor.tensor_name == "audio.blk.0.attn_norm.weight")
        );
    }

    #[test]
    fn resolves_cohere_builtin_contract() {
        let descriptors = resolve_builtin_runtime_tensor_contract_descriptors(
            COHERE_TRANSCRIBE_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::CohereTranscribe(cohere_metadata()),
        )
        .expect("cohere descriptors");

        assert!(
            descriptors
                .iter()
                .any(|descriptor| descriptor.tensor_name == "enc.blk.1.conv.pw2.weight")
        );
    }

    #[test]
    fn rejects_mismatched_metadata_kind() {
        let error = resolve_builtin_runtime_tensor_contract_descriptors(
            QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::CohereTranscribe(cohere_metadata()),
        )
        .expect_err("mismatched metadata kind must fail");

        assert!(matches!(
            error,
            RuntimeTensorContractRegistryError::MetadataKindMismatch { .. }
        ));
    }

    #[test]
    fn identifies_dedicated_executor_contract_markers() {
        assert_eq!(
            dedicated_runtime_tensor_contract_family(PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID),
            Some(DedicatedRuntimeTensorContractFamily::ParakeetCtc)
        );
        assert_eq!(
            dedicated_runtime_tensor_contract_family(WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID),
            Some(DedicatedRuntimeTensorContractFamily::Wav2Vec2Ctc)
        );
        assert_eq!(
            dedicated_runtime_tensor_contract_family(MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID),
            Some(DedicatedRuntimeTensorContractFamily::Moonshine)
        );
        assert_eq!(
            dedicated_runtime_tensor_contract_family(XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID),
            Some(DedicatedRuntimeTensorContractFamily::XasrZipformer)
        );
    }

    #[test]
    fn dedicated_executor_contracts_fail_closed_without_unknown_contract() {
        for (contract_id, family) in [
            (
                PARAKEET_CTC_RUNTIME_TENSOR_CONTRACT_ID,
                DedicatedRuntimeTensorContractFamily::ParakeetCtc,
            ),
            (
                WAV2VEC2_CTC_RUNTIME_TENSOR_CONTRACT_ID,
                DedicatedRuntimeTensorContractFamily::Wav2Vec2Ctc,
            ),
            (
                MOONSHINE_RUNTIME_TENSOR_CONTRACT_ID,
                DedicatedRuntimeTensorContractFamily::Moonshine,
            ),
            (
                XASR_ZIPFORMER_RUNTIME_TENSOR_CONTRACT_ID,
                DedicatedRuntimeTensorContractFamily::XasrZipformer,
            ),
        ] {
            let error = resolve_builtin_runtime_tensor_contract_descriptors(
                contract_id,
                RuntimeTensorContractMetadata::Qwen3Asr(qwen_metadata()),
            )
            .expect_err("dedicated contract should not materialize composer descriptors");

            assert_eq!(
                error,
                RuntimeTensorContractRegistryError::DedicatedExecutorContract {
                    contract_id: contract_id.to_string(),
                    family,
                }
            );
        }
    }
}
