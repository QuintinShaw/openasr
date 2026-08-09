//! Tensor-backed audio-frontend registry for the **data-driven composer**
//! families (Qwen3-ASR). Qwen materializes its mel
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

use super::qwen::{Qwen3AsrMelFrontendPlan, load_qwen3_mel_frontend_plan_from_reader};
use super::runtime_tensor_contract_registry::RuntimeTensorContractMetadata;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BuiltinAudioFrontendComponent {
    Qwen3Asr(Qwen3AsrMelFrontendPlan),
}

impl BuiltinAudioFrontendComponent {
    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrMelFrontendPlan> {
        match self {
            Self::Qwen3Asr(plan) => Some(plan),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinAudioFrontendComponentRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown builtin audio frontend '{frontend_id}'")]
    UnknownAudioFrontend { frontend_id: String },
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
    let RuntimeTensorContractMetadata::Qwen3Asr { metadata, .. } = metadata;
    match frontend_id {
        crate::QWEN3_ASR_AUDIO_FRONTEND_ID => {
            load_qwen3_mel_frontend_plan_from_reader(reader, metadata)
                .map(BuiltinAudioFrontendComponent::Qwen3Asr)
                .map_err(|error| materialization_failed(frontend_id, error))
        }
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
    fn materializes_qwen_frontend_plan_for_architecture() {
        let (_runtime_path, preflight) = write_qwen_preflight();
        let execution = crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata(
            &preflight.metadata,
        )
        .expect("metadata");
        let decoder_contract = crate::models::qwen::QwenDecoderContract::bind(
            crate::models::qwen::QwenDecoderContractGeometry {
                n_layers: execution.llm_layers,
                d_model: execution.llm_d_model,
                n_heads: execution.llm_heads,
                n_kv_heads: execution.llm_kv_heads,
                head_dim: execution.llm_head_dim,
                ffn_dim: execution.llm_d_model,
                vocab_size: execution.vocab_size,
            },
            crate::models::qwen::runtime_contract::qwen3_asr_decoder_profile(),
        )
        .expect("decoder contract");
        let metadata = RuntimeTensorContractMetadata::Qwen3Asr {
            metadata: execution,
            decoder_contract,
        };
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
        let (_runtime_path, preflight) = write_qwen_preflight();
        let execution = crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata(
            &preflight.metadata,
        )
        .expect("metadata");
        let decoder_contract = crate::models::qwen::QwenDecoderContract::bind(
            crate::models::qwen::QwenDecoderContractGeometry {
                n_layers: execution.llm_layers,
                d_model: execution.llm_d_model,
                n_heads: execution.llm_heads,
                n_kv_heads: execution.llm_kv_heads,
                head_dim: execution.llm_head_dim,
                ffn_dim: execution.llm_d_model,
                vocab_size: execution.vocab_size,
            },
            crate::models::qwen::runtime_contract::qwen3_asr_decoder_profile(),
        )
        .expect("decoder contract");
        let base_metadata = RuntimeTensorContractMetadata::Qwen3Asr {
            metadata: execution,
            decoder_contract,
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");

        let dedicated_frontend_ids = OpenAsrArchitectureRegistry::with_builtins()
            .descriptors()
            .iter()
            .map(|descriptor| descriptor.pack_contract.audio_frontend_id)
            .filter(|frontend_id| !matches!(*frontend_id, crate::QWEN3_ASR_AUDIO_FRONTEND_ID))
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
