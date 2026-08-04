//! Stable, machine-readable projection of the builtin model-family inventory.
//!
//! The runtime descriptors remain crate-private on purpose: they contain
//! function pointers and implementation details that are not a public API.
//! This module exposes only the versioned, data-only projection consumed by
//! tooling (for example `cargo xtask family export-inventory`).  The builtin
//! descriptor registry is the sole source of truth; this projection must not
//! grow a second hand-maintained family table.

use serde::{Deserialize, Serialize};

use crate::arch::{
    OpenAsrArchitectureDescriptor, OpenAsrArchitectureRegistry, OpenAsrBlockStackStrategy,
    OpenAsrDecodeDriverStrategy, OpenAsrDecoderStateTopology, OpenAsrDialectCapability,
    OpenAsrEncoderAttentionSpan, SpeakerSegmentationSource, StreamingPartialGranularity,
};
use crate::ggml_runtime::AutoGpuPolicy;
use crate::models::ggml_family_adapter::{GgmlExecutionCapability, LanguageFamilyHint};

/// Version identifier for the generated inventory file.
pub const MODEL_FAMILY_INVENTORY_SCHEMA_V1: &str = "openasr.model-family-inventory.v1";

/// The complete versioned inventory document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFamilyInventoryV1 {
    pub schema: String,
    pub families: Vec<ModelFamilyInventoryEntryV1>,
}

/// Data-only projection of one [`OpenAsrArchitectureDescriptor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFamilyInventoryEntryV1 {
    pub catalog_family_id: String,
    pub model_family: String,
    pub model_architecture: String,
    pub runtime_architecture_aliases: Vec<String>,
    pub adapter_id: String,
    pub module_slug: String,
    pub language: LanguageInventoryV1,
    pub pack: PackInventoryV1,
    pub execution: ExecutionInventoryV1,
    pub topology: TopologyInventoryV1,
    pub optimization: OptimizationInventoryV1,
    pub quantization: QuantizationInventoryV1,
    pub conformance: ConformanceInventoryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageInventoryV1 {
    pub policy: String,
    pub default_language: Option<String>,
    pub reject_reason: Option<String>,
    pub languages: Vec<String>,
    pub dialect_mode: String,
    pub selectable_dialect_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackInventoryV1 {
    pub audio_frontend_id: String,
    pub decode_policy_id: String,
    pub runtime_tensor_contract_id: String,
    pub tokenizer_id: String,
    pub hparam_schema: Vec<String>,
    pub importer: PackImporterInventoryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackImporterInventoryV1 {
    pub kind: String,
    pub symbol: Option<String>,
    pub relative_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionInventoryV1 {
    pub executor_component_id: String,
    pub executor: String,
    pub execution_capabilities: ExecutionCapabilitiesInventoryV1,
    pub streaming_partial_granularity: String,
    pub speaker_segmentation: String,
    pub emits_punctuation: Option<bool>,
    pub supports_phrase_bias: bool,
    pub phrase_bias_strategy: String,
    pub phrase_bias_required_tensor: Option<String>,
    pub supports_translation_task: bool,
    pub supports_source_language_hint: bool,
    pub adapter_binding: String,
    pub prepared_runtime: String,
    pub word_timestamp_strategy: String,
    pub invocation_span: InvocationSpanInventoryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCapabilitiesInventoryV1 {
    pub cpu: bool,
    pub providers: Vec<ExecutionProviderInventoryV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProviderInventoryV1 {
    pub provider: String,
    pub full_device: bool,
    pub hybrid: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvocationSpanInventoryV1 {
    pub policy: String,
    pub max_seconds: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyInventoryV1 {
    pub decode_driver: String,
    pub decode_driver_reason: Option<String>,
    pub block_stack: String,
    pub block_stack_reason: Option<String>,
    pub decoder_state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationInventoryV1 {
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    pub auto_gpu_policy: String,
    pub encoder_attention_span: String,
    pub encoder_attention_max_safe_chunk_seconds: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationInventoryV1 {
    pub tensor_classification: String,
    pub quantized_axis: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformanceInventoryV1 {
    pub profile_id: String,
    pub reference_dumper_source: Option<String>,
}

/// Project the crate-private builtin registry into the public v1 inventory.
///
/// The returned entries are sorted by `(catalog_family_id, model_architecture)`
/// so that JSON output remains byte-for-byte deterministic even if the source
/// registry is regrouped while preserving its semantics.
pub fn builtin_model_family_inventory() -> ModelFamilyInventoryV1 {
    let mut families = OpenAsrArchitectureRegistry::with_builtins()
        .descriptors()
        .iter()
        .copied()
        .map(project_descriptor)
        .collect::<Vec<_>>();
    families.sort_by(|left, right| {
        left.catalog_family_id
            .cmp(&right.catalog_family_id)
            .then_with(|| left.model_architecture.cmp(&right.model_architecture))
    });

    ModelFamilyInventoryV1 {
        schema: MODEL_FAMILY_INVENTORY_SCHEMA_V1.to_string(),
        families,
    }
}

fn project_descriptor(descriptor: OpenAsrArchitectureDescriptor) -> ModelFamilyInventoryEntryV1 {
    let identity = descriptor.identity;
    let pack_contract = descriptor.pack_contract;
    let execution_contract = descriptor.execution_contract;
    let topology_contract = descriptor.topology_contract;
    let optimization_contract = descriptor.optimization_contract;
    let quantization_contract = descriptor.quantization_contract;
    let conformance_contract = descriptor.conformance_contract;

    let importer = match pack_contract.pack_import {
        crate::arch::OpenAsrPackImportSurface::CoreConvert { symbol, .. } => {
            PackImporterInventoryV1 {
                kind: "core-convert".to_string(),
                symbol: Some(symbol.to_string()),
                relative_path: None,
            }
        }
        crate::arch::OpenAsrPackImportSurface::ExternalTooling { relative_path } => {
            PackImporterInventoryV1 {
                kind: "external-tooling".to_string(),
                symbol: None,
                relative_path: Some(relative_path.to_string()),
            }
        }
    };

    let (decode_driver, decode_driver_reason) = match topology_contract.decode_driver {
        OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy { .. } => {
            ("shared-seq2seq-greedy".to_string(), None)
        }
        OpenAsrDecodeDriverStrategy::SharedCtcGreedy { .. } => {
            ("shared-ctc-greedy".to_string(), None)
        }
        OpenAsrDecodeDriverStrategy::Dedicated { reason, .. } => {
            ("dedicated".to_string(), Some(reason.to_string()))
        }
    };
    let (block_stack, block_stack_reason) = match topology_contract.block_stack {
        OpenAsrBlockStackStrategy::Shared(_) => ("shared".to_string(), None),
        OpenAsrBlockStackStrategy::ArchitectureGraph { reason } => {
            ("architecture-graph".to_string(), Some(reason.to_string()))
        }
    };

    let (encoder_attention_span, encoder_attention_max_safe_chunk_seconds) =
        match optimization_contract.encoder_attention_span {
            OpenAsrEncoderAttentionSpan::GlobalQuadratic {
                max_safe_chunk_seconds,
            } => ("global-quadratic".to_string(), Some(max_safe_chunk_seconds)),
            OpenAsrEncoderAttentionSpan::FixedWindow => ("fixed-window".to_string(), None),
            OpenAsrEncoderAttentionSpan::LocalChunked => ("local-chunked".to_string(), None),
        };

    ModelFamilyInventoryEntryV1 {
        catalog_family_id: identity.catalog_family_id.to_string(),
        model_family: identity.model_family.to_string(),
        model_architecture: identity.model_architecture.to_string(),
        runtime_architecture_aliases: identity
            .runtime_architecture_aliases
            .iter()
            .map(|alias| (*alias).to_string())
            .collect(),
        adapter_id: identity.adapter_id.to_string(),
        module_slug: identity.module_slug.to_string(),
        language: project_language(
            identity.language_family_hint,
            identity.dialect_capability,
            identity.recognized_languages,
        ),
        pack: PackInventoryV1 {
            audio_frontend_id: pack_contract.audio_frontend_id.to_string(),
            decode_policy_id: topology_contract
                .decode_driver
                .decode_policy_id()
                .to_string(),
            runtime_tensor_contract_id: pack_contract.runtime_tensor_contract_id.to_string(),
            tokenizer_id: pack_contract.tokenizer_id.to_string(),
            hparam_schema: pack_contract
                .hparam_schema
                .iter()
                .map(|key| (*key).to_string())
                .collect(),
            importer,
        },
        execution: ExecutionInventoryV1 {
            executor_component_id: execution_contract.executor_component_id.to_string(),
            executor: match execution_contract.execution_capability {
                GgmlExecutionCapability::DedicatedRuntimeExecutorV1 => {
                    "dedicated-runtime-executor-v1"
                }
                GgmlExecutionCapability::NativeGraphLoweringV1 => "native-graph-lowering-v1",
            }
            .to_string(),
            execution_capabilities: project_execution_capabilities(
                execution_contract.execution_capabilities,
            ),
            streaming_partial_granularity: match execution_contract.streaming_partial_granularity {
                StreamingPartialGranularity::FrameSync => "frame-sync",
                StreamingPartialGranularity::Buffered => "buffered",
            }
            .to_string(),
            speaker_segmentation: match execution_contract.speaker_segmentation {
                SpeakerSegmentationSource::InDecoder => "in-decoder",
                SpeakerSegmentationSource::External => "external",
            }
            .to_string(),
            emits_punctuation: execution_contract.emits_punctuation,
            supports_phrase_bias: execution_contract.phrase_bias.is_structurally_supported(),
            phrase_bias_strategy: match execution_contract.phrase_bias {
                crate::arch::OpenAsrPhraseBiasStrategy::Unsupported => "unsupported",
                crate::arch::OpenAsrPhraseBiasStrategy::Always => "always",
                crate::arch::OpenAsrPhraseBiasStrategy::RequiresTensor { .. } => "requires-tensor",
            }
            .to_string(),
            phrase_bias_required_tensor: match execution_contract.phrase_bias {
                crate::arch::OpenAsrPhraseBiasStrategy::RequiresTensor { tensor_name } => {
                    Some(tensor_name.to_string())
                }
                crate::arch::OpenAsrPhraseBiasStrategy::Unsupported
                | crate::arch::OpenAsrPhraseBiasStrategy::Always => None,
            },
            supports_translation_task: execution_contract.supports_translation_task,
            supports_source_language_hint: execution_contract.supports_source_language_hint,
            adapter_binding: execution_contract.adapter_binding.label().to_string(),
            prepared_runtime: match execution_contract.prepared_runtime {
                crate::arch::OpenAsrPreparedRuntimeStrategy::FamilyOwned => "family-owned",
                crate::arch::OpenAsrPreparedRuntimeStrategy::SharedCohereTranscribeV1 => {
                    "shared-cohere-transcribe-v1"
                }
                crate::arch::OpenAsrPreparedRuntimeStrategy::SharedQwen3AsrV1 => {
                    "shared-qwen3-asr-v1"
                }
            }
            .to_string(),
            word_timestamp_strategy: match execution_contract.word_timestamps {
                crate::arch::OpenAsrWordTimestampStrategy::DecodeInvariant => "decode-invariant",
                crate::arch::OpenAsrWordTimestampStrategy::DecodeSensitive => "decode-sensitive",
            }
            .to_string(),
            invocation_span: match descriptor.max_single_invocation_seconds() {
                Some(max_seconds) => InvocationSpanInventoryV1 {
                    policy: "bounded".to_string(),
                    max_seconds: Some(max_seconds),
                },
                None => InvocationSpanInventoryV1 {
                    policy: "elastic".to_string(),
                    max_seconds: None,
                },
            },
        },
        topology: TopologyInventoryV1 {
            decode_driver,
            decode_driver_reason,
            block_stack,
            block_stack_reason,
            decoder_state: match topology_contract.decoder_state_topology {
                OpenAsrDecoderStateTopology::None => "none",
                OpenAsrDecoderStateTopology::CausalSelfAttentionKv => "causal-self-attention-kv",
                OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv => {
                    "encoder-decoder-self-and-cross-attention-kv"
                }
                OpenAsrDecoderStateTopology::FamilyDefinedTokenScaledPersistent => {
                    "family-defined-token-scaled-persistent"
                }
            }
            .to_string(),
        },
        optimization: OptimizationInventoryV1 {
            prefer_cpu_decoder_for_multichunk_metal: optimization_contract
                .prefer_cpu_decoder_for_multichunk_metal,
            auto_gpu_policy: match optimization_contract.auto_gpu_policy {
                AutoGpuPolicy::AllBackends => "all-backends",
                AutoGpuPolicy::ExceptMetal => "except-metal",
                AutoGpuPolicy::Never => "never",
            }
            .to_string(),
            encoder_attention_span,
            encoder_attention_max_safe_chunk_seconds,
        },
        quantization: QuantizationInventoryV1 {
            tensor_classification: match quantization_contract.tensor_classification {
                crate::models::pack_quant::TensorQuantizationContract::SemanticRolesV1 {
                    ..
                } => "semantic-roles-v1",
                crate::models::pack_quant::TensorQuantizationContract::EntireAcousticPack {
                    ..
                } => "entire-acoustic-pack",
                crate::models::pack_quant::TensorQuantizationContract::NotApplicable { .. } => {
                    "not-applicable"
                }
            }
            .to_string(),
            quantized_axis: match quantization_contract.tensor_classification {
                crate::models::pack_quant::TensorQuantizationContract::SemanticRolesV1 {
                    quantized_axis,
                    ..
                } => Some(
                    match quantized_axis {
                        crate::models::pack_quant::QuantizedAxis::First => "first",
                        crate::models::pack_quant::QuantizedAxis::Last => "last",
                    }
                    .to_string(),
                ),
                crate::models::pack_quant::TensorQuantizationContract::EntireAcousticPack {
                    ..
                }
                | crate::models::pack_quant::TensorQuantizationContract::NotApplicable { .. } => {
                    None
                }
            },
        },
        conformance: ConformanceInventoryV1 {
            profile_id: conformance_contract.profile_id.to_string(),
            reference_dumper_source: conformance_contract
                .reference_dumper_source
                .map(str::to_string),
        },
    }
}

fn project_language(
    hint: LanguageFamilyHint,
    dialect_capability: OpenAsrDialectCapability,
    recognized_languages: &'static [&'static str],
) -> LanguageInventoryV1 {
    let (dialect_mode, selectable_dialect_codes) = match dialect_capability {
        OpenAsrDialectCapability::NotAdvertised => ("not-advertised", Vec::new()),
        OpenAsrDialectCapability::RecognizesCatalogDeclared => {
            ("recognizes-catalog-declared", Vec::new())
        }
        OpenAsrDialectCapability::SelectsViaPrompt { codes } => (
            "selects-via-prompt",
            codes.iter().map(|code| (*code).to_string()).collect(),
        ),
    };
    let mut language = match hint {
        LanguageFamilyHint::WhisperVocabGated => LanguageInventoryV1 {
            policy: "whisper-vocab-gated".to_string(),
            default_language: None,
            reject_reason: None,
            languages: Vec::new(),
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
        LanguageFamilyHint::SelfDetectsRejectsHint { reject_reason } => LanguageInventoryV1 {
            policy: "self-detects-rejects-hint".to_string(),
            default_language: None,
            reject_reason: Some(reject_reason.to_string()),
            languages: Vec::new(),
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
        LanguageFamilyHint::SelectsViaPrompt { default_language } => LanguageInventoryV1 {
            policy: "selects-via-prompt".to_string(),
            default_language: Some(default_language.to_string()),
            reject_reason: None,
            languages: Vec::new(),
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
        LanguageFamilyHint::DetectAndSelectsViaPrompt => LanguageInventoryV1 {
            policy: "detect-and-selects-via-prompt".to_string(),
            default_language: None,
            reject_reason: None,
            languages: Vec::new(),
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
        LanguageFamilyHint::FixedMonolingual { language } => LanguageInventoryV1 {
            policy: "fixed-monolingual".to_string(),
            default_language: Some(language.to_string()),
            reject_reason: None,
            languages: vec![language.to_string()],
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
        LanguageFamilyHint::FixedMultilingual { languages } => LanguageInventoryV1 {
            policy: "fixed-multilingual".to_string(),
            default_language: None,
            reject_reason: None,
            languages: languages
                .iter()
                .map(|language| (*language).to_string())
                .collect(),
            dialect_mode: String::new(),
            selectable_dialect_codes: Vec::new(),
        },
    };
    language.languages = recognized_languages
        .iter()
        .map(|language| (*language).to_string())
        .collect();
    language.dialect_mode = dialect_mode.to_string();
    language.selectable_dialect_codes = selectable_dialect_codes;
    language
}

fn project_execution_capabilities(
    capabilities: crate::device::execution_policy::ExecutionCapabilities,
) -> ExecutionCapabilitiesInventoryV1 {
    // This DTO intentionally carries a stable summary instead of exposing the
    // backend bitset internals.  The descriptor itself remains authoritative;
    // tooling only needs to know whether CPU and accelerated placement are
    // available for onboarding/catalog display.
    use crate::device::execution_policy::ExecutionPlacement;
    use crate::device::execution_route::ExecutionProvider;

    let mut providers = Vec::new();
    for provider in [
        ExecutionProvider::Accelerator,
        ExecutionProvider::Cuda,
        ExecutionProvider::Hip,
        ExecutionProvider::Metal,
        ExecutionProvider::Unknown,
        ExecutionProvider::Vulkan,
    ] {
        let full_device = capabilities.supports(provider, ExecutionPlacement::FullDevice);
        let hybrid = capabilities.supports(provider, ExecutionPlacement::Hybrid);
        if full_device || hybrid {
            providers.push(ExecutionProviderInventoryV1 {
                provider: provider.as_str().to_string(),
                full_device,
                hybrid,
            });
        }
    }
    ExecutionCapabilitiesInventoryV1 {
        cpu: capabilities.supports_cpu(),
        providers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_is_sorted_and_has_unique_architectures() {
        let inventory = builtin_model_family_inventory();
        assert_eq!(inventory.schema, MODEL_FAMILY_INVENTORY_SCHEMA_V1);
        assert!(inventory.families.windows(2).all(|pair| (
            pair[0].catalog_family_id.as_str(),
            pair[0].model_architecture.as_str()
        ) <= (
            pair[1].catalog_family_id.as_str(),
            pair[1].model_architecture.as_str()
        )));
        let mut architectures = inventory
            .families
            .iter()
            .map(|family| family.model_architecture.as_str())
            .collect::<Vec<_>>();
        architectures.sort_unstable();
        architectures.dedup();
        assert_eq!(architectures.len(), inventory.families.len());

        let mut catalog_family_ids = inventory
            .families
            .iter()
            .map(|family| family.catalog_family_id.as_str())
            .collect::<Vec<_>>();
        catalog_family_ids.sort_unstable();
        catalog_family_ids.dedup();
        assert_eq!(catalog_family_ids.len(), inventory.families.len());
    }

    #[test]
    fn inventory_json_is_deterministic() {
        let first = serde_json::to_string_pretty(&builtin_model_family_inventory()).unwrap();
        let second = serde_json::to_string_pretty(&builtin_model_family_inventory()).unwrap();
        assert_eq!(first, second);
        assert!(first.ends_with('}'));
    }

    #[test]
    fn inventory_projects_task_capabilities_from_descriptors() {
        let registry = OpenAsrArchitectureRegistry::with_builtins();
        for family in builtin_model_family_inventory().families {
            let descriptor = registry
                .find_by_adapter_id(&family.adapter_id)
                .expect("inventory adapter must resolve to its canonical descriptor");
            assert_eq!(
                family.execution.supports_translation_task,
                descriptor.execution_contract.supports_translation_task,
                "translation capability drifted for {}",
                family.adapter_id
            );
            assert_eq!(
                family.execution.supports_source_language_hint,
                descriptor.execution_contract.supports_source_language_hint,
                "source-language capability drifted for {}",
                family.adapter_id
            );
            assert_eq!(
                family.execution.adapter_binding,
                descriptor.execution_contract.adapter_binding.label(),
                "adapter binding strategy drifted for {}",
                family.adapter_id
            );
        }
    }
}
