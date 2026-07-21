use std::collections::BTreeMap;

use crate::arch::OpenAsrArchitectureRegistry;

use super::ggml_family_adapter::{
    GgmlFamilyAdapterDescriptor, GgmlFamilyAdapterSelectionFields, GgmlFamilyAdapterSelectionSpec,
    OasrV1MetadataError,
};
use super::oasr_metadata::OASR_PACKAGE_VERSION_V1;
pub const COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID: &str =
    crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID;
pub const COHERE_TRANSCRIBE_GGML_ADAPTER_ID: &str = crate::arch::COHERE_TRANSCRIBE_GGML_ADAPTER_ID;
pub const COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID: &str =
    crate::arch::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID;
pub const COHERE_TRANSCRIBE_TOKENIZER_ID: &str = crate::arch::COHERE_TRANSCRIBE_TOKENIZER_ID;
pub const COHERE_TRANSCRIBE_DECODE_POLICY_ID: &str =
    crate::arch::COHERE_TRANSCRIBE_DECODE_POLICY_ID;

pub const WHISPER_GGML_ARCHITECTURE_ID: &str = crate::arch::WHISPER_GGML_ARCHITECTURE_ID;
pub const WHISPER_GGML_ADAPTER_ID: &str = crate::arch::WHISPER_GGML_ADAPTER_ID;
pub const WHISPER_AUDIO_FRONTEND_ID: &str = crate::arch::WHISPER_AUDIO_FRONTEND_ID;
pub const WHISPER_TOKENIZER_ID: &str = crate::arch::WHISPER_TOKENIZER_ID;
pub const WHISPER_DECODE_POLICY_ID: &str = crate::arch::WHISPER_DECODE_POLICY_ID;

pub const QWEN3_ASR_GGML_ARCHITECTURE_ID: &str = crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID;
pub const QWEN3_ASR_GGML_ADAPTER_ID: &str = crate::arch::QWEN3_ASR_GGML_ADAPTER_ID;
pub const QWEN3_ASR_AUDIO_FRONTEND_ID: &str = crate::arch::QWEN3_ASR_AUDIO_FRONTEND_ID;
pub const QWEN3_ASR_TOKENIZER_ID: &str = crate::arch::QWEN3_ASR_TOKENIZER_ID;
pub const QWEN3_ASR_DECODE_POLICY_ID: &str = crate::arch::QWEN3_ASR_DECODE_POLICY_ID;

pub const SENSEVOICE_GGML_ARCHITECTURE_ID: &str = crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID;
pub const SENSEVOICE_GGML_ADAPTER_ID: &str = crate::arch::SENSEVOICE_GGML_ADAPTER_ID;
pub const SENSEVOICE_AUDIO_FRONTEND_ID: &str = crate::arch::SENSEVOICE_AUDIO_FRONTEND_ID;
pub const SENSEVOICE_TOKENIZER_ID: &str = crate::arch::SENSEVOICE_TOKENIZER_ID;
pub const SENSEVOICE_DECODE_POLICY_ID: &str = crate::arch::SENSEVOICE_DECODE_POLICY_ID;
pub const PARAKEET_CTC_GGML_ARCHITECTURE_ID: &str = crate::arch::PARAKEET_CTC_GGML_ARCHITECTURE_ID;
pub const PARAKEET_CTC_GGML_ADAPTER_ID: &str = crate::arch::PARAKEET_CTC_GGML_ADAPTER_ID;
pub const PARAKEET_CTC_AUDIO_FRONTEND_ID: &str = crate::arch::PARAKEET_CTC_AUDIO_FRONTEND_ID;
pub const PARAKEET_CTC_TOKENIZER_ID: &str = crate::arch::PARAKEET_CTC_TOKENIZER_ID;
pub const PARAKEET_CTC_DECODE_POLICY_ID: &str = crate::arch::PARAKEET_CTC_DECODE_POLICY_ID;

pub const PARAKEET_TDT_GGML_ARCHITECTURE_ID: &str = crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID;
pub const PARAKEET_TDT_GGML_ADAPTER_ID: &str = crate::arch::PARAKEET_TDT_GGML_ADAPTER_ID;
pub const PARAKEET_TDT_AUDIO_FRONTEND_ID: &str = crate::arch::PARAKEET_TDT_AUDIO_FRONTEND_ID;
pub const PARAKEET_TDT_TOKENIZER_ID: &str = crate::arch::PARAKEET_TDT_TOKENIZER_ID;
pub const PARAKEET_TDT_DECODE_POLICY_ID: &str = crate::arch::PARAKEET_TDT_DECODE_POLICY_ID;

pub const WAV2VEC2_CTC_GGML_ARCHITECTURE_ID: &str = crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID;
pub const WAV2VEC2_CTC_GGML_ADAPTER_ID: &str = crate::arch::WAV2VEC2_CTC_GGML_ADAPTER_ID;
pub const WAV2VEC2_CTC_AUDIO_FRONTEND_ID: &str = crate::arch::WAV2VEC2_CTC_AUDIO_FRONTEND_ID;
pub const WAV2VEC2_CTC_TOKENIZER_ID: &str = crate::arch::WAV2VEC2_CTC_TOKENIZER_ID;
pub const WAV2VEC2_CTC_DECODE_POLICY_ID: &str = crate::arch::WAV2VEC2_CTC_DECODE_POLICY_ID;

pub const XASR_ZIPFORMER_GGML_ARCHITECTURE_ID: &str =
    crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID;
pub const XASR_ZIPFORMER_GGML_ADAPTER_ID: &str = crate::arch::XASR_ZIPFORMER_GGML_ADAPTER_ID;
pub const XASR_ZIPFORMER_AUDIO_FRONTEND_ID: &str = crate::arch::XASR_ZIPFORMER_AUDIO_FRONTEND_ID;
pub const XASR_ZIPFORMER_TOKENIZER_ID: &str = crate::arch::XASR_ZIPFORMER_TOKENIZER_ID;
pub const XASR_ZIPFORMER_DECODE_POLICY_ID: &str = crate::arch::XASR_ZIPFORMER_DECODE_POLICY_ID;

pub const MOONSHINE_GGML_ARCHITECTURE_ID: &str = crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID;
pub const MOONSHINE_GGML_ADAPTER_ID: &str = crate::arch::MOONSHINE_GGML_ADAPTER_ID;
pub const MOONSHINE_AUDIO_FRONTEND_ID: &str = crate::arch::MOONSHINE_AUDIO_FRONTEND_ID;
pub const MOONSHINE_TOKENIZER_ID: &str = crate::arch::MOONSHINE_TOKENIZER_ID;
pub const MOONSHINE_DECODE_POLICY_ID: &str = crate::arch::MOONSHINE_DECODE_POLICY_ID;

pub const DOLPHIN_GGML_ARCHITECTURE_ID: &str = crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID;
pub const DOLPHIN_GGML_ADAPTER_ID: &str = crate::arch::DOLPHIN_GGML_ADAPTER_ID;
pub const DOLPHIN_AUDIO_FRONTEND_ID: &str = crate::arch::DOLPHIN_AUDIO_FRONTEND_ID;
pub const DOLPHIN_TOKENIZER_ID: &str = crate::arch::DOLPHIN_TOKENIZER_ID;
pub const DOLPHIN_DECODE_POLICY_ID: &str = crate::arch::DOLPHIN_DECODE_POLICY_ID;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgmlFamilyRegistrySelectionError {
    InvalidMetadata(OasrV1MetadataError),
    UnsupportedPackageVersion {
        expected: &'static str,
        found: String,
    },
    UnknownFamily {
        model_family: String,
    },
    NoMatchingAdapter {
        model_family: String,
        model_architecture: String,
        audio_frontend_id: String,
        decode_policy_id: String,
        tokenizer_id: Option<String>,
    },
    Ambiguous {
        adapter_ids: Vec<&'static str>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct GgmlFamilyRegistry {
    descriptors: Vec<GgmlFamilyAdapterDescriptor>,
}

impl GgmlFamilyRegistry {
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    pub fn with_builtin_adapters() -> Self {
        let mut registry = Self::new();
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            registry.register(descriptor.ggml_family_adapter_descriptor());
        }
        registry
    }

    pub fn register(&mut self, descriptor: GgmlFamilyAdapterDescriptor) {
        if let Some(existing) = self
            .descriptors
            .iter_mut()
            .find(|existing| existing.adapter_id == descriptor.adapter_id)
        {
            *existing = descriptor;
            return;
        }
        self.descriptors.push(descriptor);
    }

    pub fn descriptors(&self) -> &[GgmlFamilyAdapterDescriptor] {
        &self.descriptors
    }

    pub fn find_by_adapter_id(&self, adapter_id: &str) -> Option<&GgmlFamilyAdapterDescriptor> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.adapter_id == adapter_id)
    }

    pub fn select<'a>(
        &'a self,
        spec: &GgmlFamilyAdapterSelectionSpec<'_>,
    ) -> Result<&'a GgmlFamilyAdapterDescriptor, GgmlFamilyRegistrySelectionError> {
        let fields = spec
            .parse_selection_fields()
            .map_err(GgmlFamilyRegistrySelectionError::InvalidMetadata)?;
        self.select_from_fields(&fields)
    }

    pub fn select_from_gguf_metadata_v1<'a>(
        &'a self,
        metadata: &BTreeMap<String, String>,
    ) -> Result<&'a GgmlFamilyAdapterDescriptor, GgmlFamilyRegistrySelectionError> {
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(metadata);
        self.select(&spec)
    }

    pub fn select_from_oasr_metadata_v1<'a>(
        &'a self,
        metadata: &BTreeMap<String, String>,
    ) -> Result<&'a GgmlFamilyAdapterDescriptor, GgmlFamilyRegistrySelectionError> {
        let spec = GgmlFamilyAdapterSelectionSpec::from_oasr_metadata_v1(metadata);
        self.select(&spec)
    }

    pub fn select_from_fields<'a>(
        &'a self,
        fields: &GgmlFamilyAdapterSelectionFields<'_>,
    ) -> Result<&'a GgmlFamilyAdapterDescriptor, GgmlFamilyRegistrySelectionError> {
        if fields.package_version != OASR_PACKAGE_VERSION_V1 {
            return Err(
                GgmlFamilyRegistrySelectionError::UnsupportedPackageVersion {
                    expected: OASR_PACKAGE_VERSION_V1,
                    found: fields.package_version.to_string(),
                },
            );
        }

        if !self
            .descriptors
            .iter()
            .any(|descriptor| descriptor.model_family == fields.model_family)
        {
            return Err(GgmlFamilyRegistrySelectionError::UnknownFamily {
                model_family: fields.model_family.to_string(),
            });
        }

        let matches: Vec<_> = self
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.matches_selection_fields(fields))
            .collect();

        match matches.as_slice() {
            [descriptor] => Ok(*descriptor),
            [] => Err(GgmlFamilyRegistrySelectionError::NoMatchingAdapter {
                model_family: fields.model_family.to_string(),
                model_architecture: fields.model_architecture.to_string(),
                audio_frontend_id: fields.audio_frontend_id.to_string(),
                decode_policy_id: fields.decode_policy_id.to_string(),
                tokenizer_id: fields.tokenizer_id.map(str::to_string),
            }),
            _ => Err(GgmlFamilyRegistrySelectionError::Ambiguous {
                adapter_ids: matches
                    .iter()
                    .map(|descriptor| descriptor.adapter_id)
                    .collect(),
            }),
        }
    }
}

pub fn cohere_transcribe_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID)
        .expect("builtin cohere architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn whisper_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(WHISPER_GGML_ARCHITECTURE_ID)
        .expect("builtin whisper architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn qwen3_asr_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID)
        .expect("builtin qwen architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn sensevoice_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID)
        .expect("builtin sensevoice architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn parakeet_ctc_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(PARAKEET_CTC_GGML_ARCHITECTURE_ID)
        .expect("builtin parakeet-ctc architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn parakeet_tdt_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(PARAKEET_TDT_GGML_ARCHITECTURE_ID)
        .expect("builtin parakeet-tdt architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn wav2vec2_ctc_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(WAV2VEC2_CTC_GGML_ARCHITECTURE_ID)
        .expect("builtin wav2vec2-ctc architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn xasr_zipformer_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(XASR_ZIPFORMER_GGML_ARCHITECTURE_ID)
        .expect("builtin xasr-zipformer architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn moonshine_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(MOONSHINE_GGML_ARCHITECTURE_ID)
        .expect("builtin moonshine architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn dolphin_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(DOLPHIN_GGML_ARCHITECTURE_ID)
        .expect("builtin dolphin architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn firered_aed_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID)
        .expect("builtin firered-aed architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn firered_llm_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID)
        .expect("builtin firered-llm architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn mimo_asr_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID)
        .expect("builtin mimo-asr architecture must exist")
        .ggml_family_adapter_descriptor()
}

pub fn moss_transcribe_diarize_runtime_descriptor_v1() -> GgmlFamilyAdapterDescriptor {
    OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID)
        .expect("builtin moss-transcribe-diarize architecture must exist")
        .ggml_family_adapter_descriptor()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::models::cohere::COHERE_TRANSCRIBE_MODEL_FAMILY;
    use crate::models::ggml_family_adapter::GgmlExecutionCapability;
    use crate::models::oasr_metadata::{
        OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
        OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
    };
    use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;
    use crate::models::whisper::WHISPER_MODEL_FAMILY;

    use super::*;

    fn metadata_for(
        family: &str,
        architecture: &str,
        frontend: &str,
        decode_policy: &str,
    ) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            family.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            architecture.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            frontend.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            decode_policy.to_string(),
        );
        metadata
    }

    #[test]
    fn builtin_registry_registers_whisper_runtime() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        assert!(
            registry
                .find_by_adapter_id(COHERE_TRANSCRIBE_GGML_ADAPTER_ID)
                .is_some()
        );
        assert!(
            registry
                .find_by_adapter_id(WHISPER_GGML_ADAPTER_ID)
                .is_some()
        );
        assert!(
            registry
                .find_by_adapter_id(QWEN3_ASR_GGML_ADAPTER_ID)
                .is_some()
        );
        assert!(
            registry
                .find_by_adapter_id(XASR_ZIPFORMER_GGML_ADAPTER_ID)
                .is_some()
        );
    }

    #[test]
    fn selects_cohere_runtime_from_oasr_v1_metadata() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let mut metadata = metadata_for(
            COHERE_TRANSCRIBE_MODEL_FAMILY,
            COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
            COHERE_TRANSCRIBE_DECODE_POLICY_ID,
        );
        metadata.insert(
            "openasr.tokenizer.id".to_string(),
            COHERE_TRANSCRIBE_TOKENIZER_ID.to_string(),
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let selected = registry.select(&spec).expect("must select");

        assert_eq!(selected.adapter_id, COHERE_TRANSCRIBE_GGML_ADAPTER_ID);
        assert_eq!(
            selected.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
    }

    #[test]
    fn selects_whisper_runtime_from_oasr_v1_metadata() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let selected = registry.select(&spec).expect("must select");

        assert_eq!(selected.adapter_id, WHISPER_GGML_ADAPTER_ID);
        assert_eq!(
            selected.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
    }

    #[test]
    fn selects_qwen3_asr_runtime_from_oasr_v1_metadata() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let metadata = metadata_for(
            QWEN3_ASR_MODEL_FAMILY,
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            QWEN3_ASR_AUDIO_FRONTEND_ID,
            QWEN3_ASR_DECODE_POLICY_ID,
        );
        let mut metadata = metadata;
        metadata.insert(
            "openasr.tokenizer.id".to_string(),
            QWEN3_ASR_TOKENIZER_ID.to_string(),
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let selected = registry.select(&spec).expect("must select");

        assert_eq!(selected.adapter_id, QWEN3_ASR_GGML_ADAPTER_ID);
        assert_eq!(
            selected.execution_capability,
            GgmlExecutionCapability::NativeGraphLoweringV1
        );
    }

    #[test]
    fn selects_xasr_zipformer_runtime_from_oasr_v1_metadata() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let mut metadata = metadata_for(
            crate::arch::XASR_ZIPFORMER_MODEL_FAMILY,
            XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
            XASR_ZIPFORMER_DECODE_POLICY_ID,
        );
        metadata.insert(
            "openasr.tokenizer.id".to_string(),
            XASR_ZIPFORMER_TOKENIZER_ID.to_string(),
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let selected = registry.select(&spec).expect("must select");

        assert_eq!(selected.adapter_id, XASR_ZIPFORMER_GGML_ADAPTER_ID);
        assert_eq!(
            selected.execution_capability,
            GgmlExecutionCapability::DedicatedRuntimeExecutorV1
        );
    }

    #[test]
    fn returns_no_match_when_tokenizer_id_conflicts() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let mut metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        metadata.insert("openasr.tokenizer.id".to_string(), "wrong.id".to_string());
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);

        let error = registry.select(&spec).expect_err("must fail closed");
        assert_eq!(
            error,
            GgmlFamilyRegistrySelectionError::NoMatchingAdapter {
                model_family: WHISPER_MODEL_FAMILY.to_string(),
                model_architecture: WHISPER_GGML_ARCHITECTURE_ID.to_string(),
                audio_frontend_id: WHISPER_AUDIO_FRONTEND_ID.to_string(),
                decode_policy_id: WHISPER_DECODE_POLICY_ID.to_string(),
                tokenizer_id: Some("wrong.id".to_string()),
            }
        );
    }

    #[test]
    fn returns_ambiguous_when_multiple_descriptors_match() {
        let mut registry = GgmlFamilyRegistry::with_builtin_adapters();
        let mut duplicate = whisper_runtime_descriptor_v1();
        duplicate.adapter_id = "ggml-family-whisper-duplicate-runtime-v1";
        registry.register(duplicate);

        let metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        let spec = GgmlFamilyAdapterSelectionSpec::from_gguf_metadata_v1(&metadata);
        let error = registry
            .select(&spec)
            .expect_err("ambiguous must fail closed");

        match error {
            GgmlFamilyRegistrySelectionError::Ambiguous { adapter_ids } => {
                assert_eq!(adapter_ids.len(), 2);
                assert!(adapter_ids.contains(&WHISPER_GGML_ADAPTER_ID));
                assert!(adapter_ids.contains(&"ggml-family-whisper-duplicate-runtime-v1"));
            }
            other => panic!("expected ambiguous error, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_reports_unknown_family() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let metadata = metadata_for(
            "unknown-family",
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );

        let error = registry
            .select_from_gguf_metadata_v1(&metadata)
            .expect_err("unknown family must fail closed");

        assert_eq!(
            error,
            GgmlFamilyRegistrySelectionError::UnknownFamily {
                model_family: "unknown-family".to_string(),
            }
        );
    }

    #[test]
    fn dispatch_reports_unsupported_package_version() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let mut metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        metadata.insert(
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            "2".to_string(),
        );

        let error = registry
            .select_from_gguf_metadata_v1(&metadata)
            .expect_err("package version must fail closed");

        assert_eq!(
            error,
            GgmlFamilyRegistrySelectionError::UnsupportedPackageVersion {
                expected: OASR_PACKAGE_VERSION_V1,
                found: "2".to_string(),
            }
        );
    }

    #[test]
    fn dispatch_reports_no_matching_adapter() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            "whisper.invalid.policy.v0",
        );

        let error = registry
            .select_from_gguf_metadata_v1(&metadata)
            .expect_err("no matching adapter must fail closed");

        assert_eq!(
            error,
            GgmlFamilyRegistrySelectionError::NoMatchingAdapter {
                model_family: WHISPER_MODEL_FAMILY.to_string(),
                model_architecture: WHISPER_GGML_ARCHITECTURE_ID.to_string(),
                audio_frontend_id: WHISPER_AUDIO_FRONTEND_ID.to_string(),
                decode_policy_id: "whisper.invalid.policy.v0".to_string(),
                tokenizer_id: None,
            }
        );
    }

    #[test]
    fn supports_whisper_dispatch_from_oasr_package_metadata_map() {
        let registry = GgmlFamilyRegistry::with_builtin_adapters();
        let metadata = metadata_for(
            WHISPER_MODEL_FAMILY,
            WHISPER_GGML_ARCHITECTURE_ID,
            WHISPER_AUDIO_FRONTEND_ID,
            WHISPER_DECODE_POLICY_ID,
        );
        let selected = registry
            .select_from_oasr_metadata_v1(&metadata)
            .expect("must dispatch");

        assert_eq!(selected.adapter_id, WHISPER_GGML_ADAPTER_ID);
    }
}
