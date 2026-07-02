use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};

const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_LLAMA: &str = "llama";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const SENTENCEPIECE_WORD_START: &str = "\u{2581}";

#[derive(Debug, Clone)]
pub(crate) struct CohereTranscribeTokenizer {
    id_to_token: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl PhraseBiasTokenEncoder for CohereTranscribeTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_sentencepiece_phrase_bias_tokens(phrase, &self.token_to_id, "Cohere Transcribe")
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for CohereTranscribeTokenizer {}

impl CohereTranscribeTokenizer {
    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_LLAMA) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Cohere Transcribe GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_LLAMA, tokenizer_model
                ),
            });
        }
        let tokens = required_metadata_string_array(metadata, TOKENIZER_GGML_TOKENS_KEY)?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Cohere Transcribe GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_TOKENS_KEY
                ),
            });
        }
        let mut token_to_id = BTreeMap::new();
        for (index, token) in tokens.iter().enumerate() {
            let token_id =
                u32::try_from(index).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Cohere Transcribe tokenizer token index {index} does not fit u32"
                    ),
                })?;
            if token_to_id.insert(token.clone(), token_id).is_some() {
                return Err(NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "Cohere Transcribe GGUF tokenizer contains duplicate token '{}'",
                        token
                    ),
                });
            }
        }
        Ok(Self {
            id_to_token: tokens.to_vec(),
            token_to_id,
        })
    }

    pub(crate) fn token_id_by_content(&self, content: &str) -> Option<u32> {
        self.token_to_id.get(content).copied()
    }

    pub(crate) fn token_content_by_id(&self, token_id: u32) -> Option<&str> {
        self.id_to_token.get(token_id as usize).map(String::as_str)
    }

    pub(crate) fn decode_text_token_ids(
        &self,
        token_ids: &[u32],
    ) -> Result<String, NativeAsrError> {
        let mut output = String::new();
        let mut pending_bytes = Vec::new();

        let flush_pending_bytes = |output: &mut String, pending_bytes: &mut Vec<u8>| {
            if pending_bytes.is_empty() {
                return;
            }
            if let Ok(text) = String::from_utf8(pending_bytes.clone()) {
                output.push_str(&text);
            }
            pending_bytes.clear();
        };

        for token_id in token_ids {
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("Cohere tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(token) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Cohere tokenizer id {token_id} is not in vocab"),
                });
            };

            if let Some(byte) = parse_sentencepiece_byte_fallback(token) {
                pending_bytes.push(byte);
                continue;
            }

            flush_pending_bytes(&mut output, &mut pending_bytes);
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }
            output.push_str(&token.replace(SENTENCEPIECE_WORD_START, " "));
        }
        flush_pending_bytes(&mut output, &mut pending_bytes);
        Ok(output.strip_prefix(' ').unwrap_or(&output).to_string())
    }
}

fn parse_sentencepiece_byte_fallback(token: &str) -> Option<u8> {
    if !token.starts_with("<0x") || !token.ends_with('>') || token.len() != 6 {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

fn required_metadata_string<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a str, NativeAsrError> {
    let value = metadata
        .get_string(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Cohere Transcribe GGUF tokenizer is missing required key '{key}'"),
        })?;
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!("Cohere Transcribe GGUF tokenizer key '{key}' cannot be empty"),
        });
    }
    Ok(normalized)
}

fn required_metadata_string_array<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a [String], NativeAsrError> {
    metadata
        .get_string_array(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!(
                "Cohere Transcribe GGUF tokenizer requires key '{key}' as array[string]"
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgufMetadataValue;

    fn metadata_with_tokens(tokens: Vec<String>) -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_LLAMA.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_looks_up_prompt_tokens() {
        let metadata = metadata_with_tokens(vec![
            "<|startoftranscript|>".to_string(),
            "<|en|>".to_string(),
            "<|endoftext|>".to_string(),
        ]);
        let tokenizer =
            CohereTranscribeTokenizer::from_gguf_metadata(&metadata).expect("tokenizer");
        assert_eq!(tokenizer.token_id_by_content("<|en|>"), Some(1));
    }

    #[test]
    fn tokenizer_decodes_sentencepiece_and_byte_fallback() {
        let metadata = metadata_with_tokens(vec![
            "▁hello".to_string(),
            "<0xE4>".to_string(),
            "<0xBD>".to_string(),
            "<0xA0>".to_string(),
            "▁world".to_string(),
            "<|endoftext|>".to_string(),
        ]);
        let tokenizer =
            CohereTranscribeTokenizer::from_gguf_metadata(&metadata).expect("tokenizer");
        let text = tokenizer
            .decode_text_token_ids(&[0, 1, 2, 3, 4, 5])
            .expect("decode");
        assert_eq!(text, "hello你 world");
    }
}
