//! Tensor-backed audio-frontend registry for the **data-driven composer**
//! families (Cohere Transcribe + Qwen3-ASR). These two materialize their mel
//! frontend from GGUF tensors via `build_builtin_runtime_component_bootstrap`,
//! so they need a central place to map `architecture -> frontend plan`.
//!
//! Dedicated-executor families deliberately do *not* go through this registry:
//! each owns its frontend + weight loading in its own family module and never
//! calls `build_builtin_runtime_component_bootstrap`. The canonical architecture
//! inventory distinguishes a known dedicated frontend from an unknown id, so
//! this component registry does not maintain a second family/frontend list.

use thiserror::Error;

use crate::GgufTensorDataReader;
use crate::arch::OpenAsrArchitectureRegistry;
use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;

use super::cohere::{
    CohereTranscribeFrontendPlan, load_cohere_transcribe_frontend_plan_from_reader,
};
use super::qwen::{Qwen3AsrMelFrontendPlan, load_qwen3_mel_frontend_plan_from_reader};
use super::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BuiltinAudioFrontendComponent {
    CohereTranscribe(CohereTranscribeFrontendPlan),
    Qwen3Asr(Qwen3AsrMelFrontendPlan),
}

impl BuiltinAudioFrontendComponent {
    pub(crate) fn into_cohere_transcribe(self) -> Option<CohereTranscribeFrontendPlan> {
        match self {
            Self::CohereTranscribe(plan) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrMelFrontendPlan> {
        match self {
            Self::Qwen3Asr(plan) => Some(plan),
            _ => None,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinAudioFrontendComponentRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown builtin audio frontend '{frontend_id}'")]
    UnknownAudioFrontend { frontend_id: String },
    #[error(
        "builtin audio frontend '{frontend_id}' expected metadata for '{expected_kind}', got '{found_kind}'"
    )]
    MetadataKindMismatch {
        frontend_id: String,
        expected_kind: &'static str,
        found_kind: &'static str,
    },
    #[error("builtin audio frontend '{frontend_id}' materialization is unsupported: {reason}")]
    UnsupportedMaterialization { frontend_id: String, reason: String },
    #[error("builtin audio frontend '{frontend_id}' materialization failed: {reason}")]
    MaterializationFailed { frontend_id: String, reason: String },
}

pub(crate) fn materialize_builtin_audio_frontend_for_architecture(
    model_architecture: &str,
    reader: &GgufTensorDataReader,
    metadata: RuntimeTensorContractMetadata,
) -> Result<BuiltinAudioFrontendComponent, BuiltinAudioFrontendComponentRegistryError> {
    let descriptor = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .ok_or_else(
            || BuiltinAudioFrontendComponentRegistryError::UnknownArchitecture {
                model_architecture: model_architecture.to_string(),
            },
        )?;
    materialize_builtin_audio_frontend(descriptor.pack_contract.audio_frontend_id, reader, metadata)
}

pub(crate) fn materialize_builtin_audio_frontend(
    frontend_id: &str,
    reader: &GgufTensorDataReader,
    metadata: RuntimeTensorContractMetadata,
) -> Result<BuiltinAudioFrontendComponent, BuiltinAudioFrontendComponentRegistryError> {
    match (frontend_id, metadata) {
        (
            crate::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
            RuntimeTensorContractMetadata::CohereTranscribe(metadata),
        ) => load_cohere_transcribe_frontend_plan_from_reader(reader, metadata)
            .map(BuiltinAudioFrontendComponent::CohereTranscribe)
            .map_err(|error| materialization_failed(frontend_id, error)),
        (crate::QWEN3_ASR_AUDIO_FRONTEND_ID, RuntimeTensorContractMetadata::Qwen3Asr(metadata)) => {
            load_qwen3_mel_frontend_plan_from_reader(reader, metadata)
                .map(BuiltinAudioFrontendComponent::Qwen3Asr)
                .map_err(|error| materialization_failed(frontend_id, error))
        }
        (crate::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID, metadata) => Err(
            BuiltinAudioFrontendComponentRegistryError::MetadataKindMismatch {
                frontend_id: frontend_id.to_string(),
                expected_kind: "cohere-transcribe",
                found_kind: metadata_kind_label(metadata),
            },
        ),
        (crate::QWEN3_ASR_AUDIO_FRONTEND_ID, metadata) => Err(
            BuiltinAudioFrontendComponentRegistryError::MetadataKindMismatch {
                frontend_id: frontend_id.to_string(),
                expected_kind: QWEN3_ASR_MODEL_FAMILY,
                found_kind: metadata_kind_label(metadata),
            },
        ),
        _ if OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .any(|descriptor| descriptor.pack_contract.audio_frontend_id == frontend_id) =>
        {
            Err(
                BuiltinAudioFrontendComponentRegistryError::UnsupportedMaterialization {
                    frontend_id: frontend_id.to_string(),
                    reason: "frontend belongs to a dedicated executor; it loads via its family module, not the tensor-backed composer registry".to_string(),
                },
            )
        }
        _ => Err(BuiltinAudioFrontendComponentRegistryError::UnknownAudioFrontend {
            frontend_id: frontend_id.to_string(),
        }),
    }
}

fn metadata_kind_label(metadata: RuntimeTensorContractMetadata) -> &'static str {
    match metadata {
        RuntimeTensorContractMetadata::CohereTranscribe(_) => "cohere-transcribe",
        RuntimeTensorContractMetadata::Qwen3Asr(_) => QWEN3_ASR_MODEL_FAMILY,
    }
}

fn materialization_failed(
    frontend_id: &str,
    error: impl std::fmt::Display,
) -> BuiltinAudioFrontendComponentRegistryError {
    BuiltinAudioFrontendComponentRegistryError::MaterializationFailed {
        frontend_id: frontend_id.to_string(),
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::{NamedTempFile, TempPath};

    use super::*;
    use crate::ggml_runtime::GgufRuntimeSourcePreflight;
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::models::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::{
        read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
        validate_ggml_runtime_source_path,
    };

    fn write_cohere_preflight() -> (TempPath, GgufRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-frontend-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgufRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    fn qwen_frontend_fixture_spec() -> TinyGgufFixtureSpec {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        metadata.insert("qwen3-asr.sample_rate".to_string(), "16000".to_string());
        metadata.insert("qwen3-asr.n_mels".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.n_fft".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.win_length".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.hop_length".to_string(), "160".to_string());
        metadata.insert("qwen3-asr.audio.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.audio.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.audio.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.llm.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.n_kv_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.head_dim".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.llm.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.vocab_size".to_string(), "32".to_string());
        metadata.insert("qwen3-asr.llm.max_pos".to_string(), "256".to_string());
        metadata.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            "2".to_string(),
        );
        metadata.insert("qwen3-asr.audio_end_token_id".to_string(), "3".to_string());
        metadata.insert("qwen3-asr.audio_pad_token_id".to_string(), "4".to_string());
        metadata.insert("qwen3-asr.eos_token_id".to_string(), "0".to_string());
        metadata.insert("qwen3-asr.pad_token_id".to_string(), "6".to_string());
        TinyGgufFixtureSpec::new(metadata)
            .with_tensor_shape("audio.mel_filters", [8_u64, 201_u64])
            .with_tensor_shape("audio.mel_window", [400_u64])
    }

    fn write_qwen_preflight() -> (TempPath, GgufRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = qwen_frontend_fixture_spec();
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgufRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn materializes_cohere_frontend_plan_for_architecture() {
        let (_runtime_path, preflight) = write_cohere_preflight();
        let metadata = RuntimeTensorContractMetadata::CohereTranscribe(
            crate::models::cohere::runtime_contract::parse_cohere_transcribe_execution_metadata(
                &preflight.metadata,
            )
            .expect("metadata"),
        );
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");

        let plan = materialize_builtin_audio_frontend_for_architecture(
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            &reader,
            metadata,
        )
        .expect("frontend plan")
        .into_cohere_transcribe()
        .expect("cohere variant");

        assert_eq!(plan.n_mels, 32);
        assert_eq!(plan.win_length, 400);
    }

    #[test]
    fn materializes_qwen_frontend_plan_for_architecture() {
        let (_runtime_path, preflight) = write_qwen_preflight();
        let metadata = RuntimeTensorContractMetadata::Qwen3Asr(
            crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata(
                &preflight.metadata,
            )
            .expect("metadata"),
        );
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");

        let plan = materialize_builtin_audio_frontend_for_architecture(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &reader,
            metadata,
        )
        .expect("frontend plan")
        .into_qwen3_asr()
        .expect("qwen variant");

        assert_eq!(plan.n_mels, 8);
        assert_eq!(plan.n_fft, 400);
    }

    #[test]
    fn dedicated_executor_frontends_fail_closed_outside_composer_registry() {
        // Derive the expected set from the inventory. A new dedicated family
        // is covered without adding its frontend id to this test or registry.
        let (_runtime_path, preflight) = write_cohere_preflight();
        let base_metadata = RuntimeTensorContractMetadata::CohereTranscribe(
            crate::models::cohere::runtime_contract::parse_cohere_transcribe_execution_metadata(
                &preflight.metadata,
            )
            .expect("metadata"),
        );
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");

        let dedicated_frontend_ids = OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .map(|descriptor| descriptor.pack_contract.audio_frontend_id)
            .filter(|frontend_id| {
                !matches!(
                    *frontend_id,
                    crate::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID | crate::QWEN3_ASR_AUDIO_FRONTEND_ID
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert!(!dedicated_frontend_ids.is_empty());
        for frontend_id in dedicated_frontend_ids {
            let error = materialize_builtin_audio_frontend(frontend_id, &reader, base_metadata)
                .expect_err("dedicated-executor frontend must not materialize here");
            assert!(
                matches!(
                    error,
                    BuiltinAudioFrontendComponentRegistryError::UnsupportedMaterialization { .. }
                ),
                "{frontend_id} should report UnsupportedMaterialization, got {error:?}"
            );
        }
    }
}
