//! Runtime tensor-contract validation + descriptor expansion for the
//! **data-driven composer** family only (Qwen3-ASR), whose runtime tensors are
//! validated/expanded centrally before graph assembly.
//!
//! Families that select `FamilyOwned` prepared runtimes validate their tensor
//! sets in their own modules (e.g. `validate_stage_against_descriptor`) and
//! never route a contract id through here. The canonical architecture
//! inventory identifies those contracts so generic tooling sees a first-class
//! fail-closed marker instead of a generic unknown-contract error. Dedicated
//! executor families validate their own tensor contracts.

use thiserror::Error;

use crate::GgufTensorIndex;
use crate::arch::{
    OpenAsrArchitectureRegistry, OpenAsrPreparedRuntimeStrategy,
    QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
};
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::models::qwen::QwenDecoderContract;
use crate::models::runtime_contract::ScalarMetadataView;

#[cfg(test)]
use super::qwen::runtime_contract::qwen3_runtime_tensor_descriptors;
use super::qwen::runtime_contract::{
    Qwen3AsrExecutionMetadata, parse_qwen3_execution_metadata,
    validate_qwen3_runtime_tensors_with_index,
};
#[cfg(test)]
use super::tensor_binding::TensorBindingDescriptor;

#[derive(Debug, Clone, Copy)]
pub(crate) enum RuntimeTensorContractMetadata {
    Qwen3Asr {
        metadata: Qwen3AsrExecutionMetadata,
        decoder_contract: QwenDecoderContract,
    },
}

impl RuntimeTensorContractMetadata {
    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrExecutionMetadata> {
        match self {
            Self::Qwen3Asr { metadata, .. } => Some(metadata),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeTensorContractRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown runtime tensor contract '{contract_id}'")]
    UnknownContract { contract_id: String },
    #[error(
        "runtime tensor contract '{contract_id}' belongs to family-owned runtime '{family}' and is not materialized by the shared composer registry"
    )]
    FamilyOwnedRuntimeContract {
        contract_id: String,
        family: &'static str,
    },
    #[error("runtime tensor contract '{contract_id}' metadata parse failed: {reason}")]
    MetadataParseFailed { contract_id: String, reason: String },
    #[error("runtime tensor contract '{contract_id}' validation failed: {reason}")]
    ValidationFailed { contract_id: String, reason: String },
}

#[cfg(test)]
pub(crate) fn resolve_builtin_runtime_tensor_contract_descriptors(
    contract_id: &str,
    metadata: RuntimeTensorContractMetadata,
) -> Result<Vec<TensorBindingDescriptor>, RuntimeTensorContractRegistryError> {
    match (contract_id, metadata) {
        (
            QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            RuntimeTensorContractMetadata::Qwen3Asr {
                metadata,
                decoder_contract,
            },
        ) => qwen3_runtime_tensor_descriptors(metadata, &decoder_contract).map_err(|error| {
            RuntimeTensorContractRegistryError::ValidationFailed {
                contract_id: contract_id.to_string(),
                reason: error.to_string(),
            }
        }),
        (contract_id, _) => family_owned_runtime_contract_error(contract_id).map_or_else(
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
    match descriptor.pack_contract.runtime_tensor_contract_id {
        QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID => {
            let metadata = parse_qwen3_execution_metadata(metadata).map_err(|error| {
                RuntimeTensorContractRegistryError::MetadataParseFailed {
                    contract_id: descriptor
                        .pack_contract
                        .runtime_tensor_contract_id
                        .to_string(),
                    reason: error.to_string(),
                }
            })?;
            let decoder_contract =
                validate_qwen3_runtime_tensors_with_index(tensor_index, metadata).map_err(
                    |error| RuntimeTensorContractRegistryError::ValidationFailed {
                        contract_id: descriptor
                            .pack_contract
                            .runtime_tensor_contract_id
                            .to_string(),
                        reason: error.to_string(),
                    },
                )?;
            Ok(RuntimeTensorContractMetadata::Qwen3Asr {
                metadata,
                decoder_contract,
            })
        }
        contract_id => family_owned_runtime_contract_error(contract_id).map_or_else(
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
    preflight: &GgufRuntimeSourcePreflight,
) -> Result<RuntimeTensorContractMetadata, RuntimeTensorContractRegistryError> {
    validate_builtin_runtime_tensor_contract_for_architecture(
        model_architecture,
        &preflight.metadata,
        &preflight.tensor_index,
    )
}

fn family_owned_runtime_tensor_contract_family(contract_id: &str) -> Option<&'static str> {
    OpenAsrArchitectureRegistry::with_builtins()
        .descriptors()
        .iter()
        .find(|descriptor| {
            descriptor.pack_contract.runtime_tensor_contract_id == contract_id
                && descriptor.execution_contract.prepared_runtime
                    == OpenAsrPreparedRuntimeStrategy::FamilyOwned
        })
        .map(|descriptor| descriptor.identity.model_family)
}

fn family_owned_runtime_contract_error(
    contract_id: &str,
) -> Option<RuntimeTensorContractRegistryError> {
    family_owned_runtime_tensor_contract_family(contract_id).map(|family| {
        RuntimeTensorContractRegistryError::FamilyOwnedRuntimeContract {
            contract_id: contract_id.to_string(),
            family,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen::QwenDecoderContractGeometry;
    use crate::models::qwen::runtime_contract::qwen3_asr_decoder_profile;

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

    fn qwen_contract() -> QwenDecoderContract {
        let metadata = qwen_metadata();
        QwenDecoderContract::bind(
            QwenDecoderContractGeometry {
                n_layers: metadata.llm_layers,
                d_model: metadata.llm_d_model,
                n_heads: metadata.llm_heads,
                n_kv_heads: metadata.llm_kv_heads,
                head_dim: metadata.llm_head_dim,
                ffn_dim: 64,
                vocab_size: metadata.vocab_size,
            },
            qwen3_asr_decoder_profile(),
        )
        .expect("qwen contract")
    }

    fn qwen_contract_metadata() -> RuntimeTensorContractMetadata {
        RuntimeTensorContractMetadata::Qwen3Asr {
            metadata: qwen_metadata(),
            decoder_contract: qwen_contract(),
        }
    }
    #[test]
    fn resolves_qwen_builtin_contract() {
        let descriptors = resolve_builtin_runtime_tensor_contract_descriptors(
            QWEN3_ASR_RUNTIME_TENSOR_CONTRACT_ID,
            qwen_contract_metadata(),
        )
        .expect("qwen descriptors");

        assert!(
            descriptors
                .iter()
                .any(|descriptor| descriptor.tensor_name == "audio.blk.0.attn_norm.weight")
        );
    }

    #[test]
    fn identifies_every_family_owned_runtime_contract_from_architecture_inventory() {
        let family_owned_descriptors = OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .filter(|descriptor| {
                descriptor.execution_contract.prepared_runtime
                    == OpenAsrPreparedRuntimeStrategy::FamilyOwned
            })
            .collect::<Vec<_>>();

        assert!(!family_owned_descriptors.is_empty());
        for descriptor in family_owned_descriptors {
            assert_eq!(
                family_owned_runtime_tensor_contract_family(
                    descriptor.pack_contract.runtime_tensor_contract_id
                ),
                Some(descriptor.identity.model_family)
            );
        }
        assert_eq!(
            family_owned_runtime_tensor_contract_family("unknown-contract"),
            None
        );
    }

    #[test]
    fn family_owned_runtime_contracts_fail_closed_without_unknown_contract() {
        for descriptor in OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .filter(|descriptor| {
                descriptor.execution_contract.prepared_runtime
                    == OpenAsrPreparedRuntimeStrategy::FamilyOwned
            })
        {
            let contract_id = descriptor.pack_contract.runtime_tensor_contract_id;
            let error = resolve_builtin_runtime_tensor_contract_descriptors(
                contract_id,
                qwen_contract_metadata(),
            )
            .expect_err("family-owned contract should not materialize shared composer descriptors");

            assert_eq!(
                error,
                RuntimeTensorContractRegistryError::FamilyOwnedRuntimeContract {
                    contract_id: contract_id.to_string(),
                    family: descriptor.identity.model_family,
                }
            );
        }
    }
}
