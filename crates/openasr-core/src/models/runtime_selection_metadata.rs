use std::collections::BTreeMap;

use crate::arch::OpenAsrArchitectureRegistry;
use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};

pub(crate) fn selection_metadata_from_gguf(
    gguf_metadata: &GgufMetadata,
) -> BTreeMap<String, String> {
    let mut metadata: BTreeMap<String, String> = gguf_metadata
        .values()
        .iter()
        .filter_map(|(key, value)| {
            let text = scalar_metadata_value_to_string(value)?;
            let normalized = text.trim();
            if normalized.is_empty() {
                None
            } else {
                Some((key.clone(), normalized.to_string()))
            }
        })
        .collect();
    synthesize_oasr_v1_selection_metadata_from_architecture(&mut metadata);
    metadata
}

pub(crate) fn synthesize_oasr_v1_selection_metadata_from_architecture(
    metadata: &mut BTreeMap<String, String>,
) {
    OpenAsrArchitectureRegistry::with_builtins().synthesize_selection_metadata_defaults(metadata);
}

fn scalar_metadata_value_to_string(value: &GgufMetadataValue) -> Option<String> {
    match value {
        GgufMetadataValue::String(text) => Some(text.clone()),
        GgufMetadataValue::U32(number) => Some(number.to_string()),
        GgufMetadataValue::U64(number) => Some(number.to_string()),
        GgufMetadataValue::Bool(flag) => Some(flag.to_string()),
        GgufMetadataValue::F32(number) => Some(number.to_string()),
        GgufMetadataValue::StringArray(_) | GgufMetadataValue::U32Array(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};
    use crate::models::cohere::COHERE_TRANSCRIBE_MODEL_FAMILY;
    use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
    use crate::models::oasr_metadata::{
        OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
        OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
    };
    use crate::models::qwen::QWEN3_ASR_MODEL_FAMILY;
    use crate::models::whisper::WHISPER_MODEL_FAMILY;

    #[test]
    fn synthesizes_cohere_selection_metadata_from_general_architecture() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            crate::arch::GENERAL_ARCHITECTURE_KEY.to_string(),
            "cohere-transcribe".to_string(),
        );
        synthesize_oasr_v1_selection_metadata_from_architecture(&mut metadata);

        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&COHERE_TRANSCRIBE_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(&crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID.to_string())
        );
        assert_eq!(
            metadata.get(GGML_TOKENIZER_ID_KEY),
            Some(&crate::arch::COHERE_TRANSCRIBE_TOKENIZER_ID.to_string())
        );
    }

    #[test]
    fn synthesizes_whisper_selection_metadata_from_general_architecture() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            crate::arch::GENERAL_ARCHITECTURE_KEY.to_string(),
            "whisper".to_string(),
        );
        synthesize_oasr_v1_selection_metadata_from_architecture(&mut metadata);

        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&WHISPER_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(&crate::arch::WHISPER_GGML_ARCHITECTURE_ID.to_string())
        );
    }

    #[test]
    fn synthesizes_qwen_selection_metadata_from_general_architecture() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            crate::arch::GENERAL_ARCHITECTURE_KEY.to_string(),
            "qwen3-asr".to_string(),
        );
        synthesize_oasr_v1_selection_metadata_from_architecture(&mut metadata);

        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&QWEN3_ASR_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(&crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID.to_string())
        );
        assert_eq!(
            metadata.get(GGML_TOKENIZER_ID_KEY),
            Some(&crate::arch::QWEN3_ASR_TOKENIZER_ID.to_string())
        );
    }

    #[test]
    fn ignores_array_values_when_projecting_from_gguf() {
        let mut values = BTreeMap::new();
        values.insert(
            crate::arch::GENERAL_ARCHITECTURE_KEY.to_string(),
            GgufMetadataValue::String("whisper".to_string()),
        );
        values.insert(
            "general.tags".to_string(),
            GgufMetadataValue::StringArray(vec!["a".to_string(), "b".to_string()]),
        );
        let metadata = selection_metadata_from_gguf(&GgufMetadata::from_values_for_test(values));

        assert!(!metadata.contains_key("general.tags"));
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_PACKAGE_VERSION),
            Some(&OASR_PACKAGE_VERSION_V1.to_string())
        );
        assert_eq!(
            metadata.get(GGML_TOKENIZER_ID_KEY),
            Some(&crate::arch::WHISPER_TOKENIZER_ID.to_string())
        );
    }
}
