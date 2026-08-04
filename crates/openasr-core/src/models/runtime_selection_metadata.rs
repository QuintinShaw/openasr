use std::collections::BTreeMap;

use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};

pub(crate) fn selection_metadata_from_gguf(
    gguf_metadata: &GgufMetadata,
) -> BTreeMap<String, String> {
    gguf_metadata
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
        .collect()
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
    use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
    use crate::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION;
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
        assert_eq!(metadata.len(), 1);
        assert_eq!(
            metadata.get(crate::arch::GENERAL_ARCHITECTURE_KEY),
            Some(&"whisper".to_string())
        );
        assert!(!metadata.contains_key(OASR_METADATA_KEY_PACKAGE_VERSION));
        assert!(!metadata.contains_key(GGML_TOKENIZER_ID_KEY));
    }
}
