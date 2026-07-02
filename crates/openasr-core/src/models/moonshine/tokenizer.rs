use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};

const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_LLAMA: &str = "llama";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const SENTENCEPIECE_WORD_START: &str = "\u{2581}";

/// SentencePiece-style ▁/byte-fallback BPE tokenizer used by Moonshine.
///
/// Detokenization mirrors HF `Sequence[Replace(▁→space), ByteFallback, Fuse, Strip(start=1)]`:
/// `▁` becomes a space, `<0xXX>` byte-fallback tokens are fused into UTF-8, special
/// `<...>` tokens are dropped, and a single leading space is stripped.
#[derive(Debug, Clone)]
pub(crate) struct MoonshineTokenizer {
    id_to_token: Vec<String>,
    token_to_id: BTreeMap<String, u32>,
}

impl PhraseBiasTokenEncoder for MoonshineTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        encode_sentencepiece_phrase_bias_tokens(phrase, &self.token_to_id, "Moonshine")
    }
}

impl MoonshineTokenizer {
    pub(crate) fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_LLAMA) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Moonshine GGUF tokenizer key '{TOKENIZER_GGML_MODEL_KEY}' must be '{TOKENIZER_GGML_MODEL_VALUE_LLAMA}', got '{tokenizer_model}'"
                ),
            });
        }
        let tokens = required_metadata_string_array(metadata, TOKENIZER_GGML_TOKENS_KEY)?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Moonshine GGUF tokenizer key '{TOKENIZER_GGML_TOKENS_KEY}' cannot be empty"
                ),
            });
        }
        let mut token_to_id = BTreeMap::new();
        for (index, token) in tokens.iter().enumerate() {
            let token_id =
                u32::try_from(index).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!("Moonshine tokenizer token index {index} does not fit u32"),
                })?;
            // Vocab can legitimately contain duplicate string content (e.g. placeholder
            // unused slots); keep the first id, like the cohere tokenizer.
            token_to_id.entry(token.clone()).or_insert(token_id);
        }
        Ok(Self {
            id_to_token: tokens.to_vec(),
            token_to_id,
        })
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
                message: format!("Moonshine tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(token) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Moonshine tokenizer id {token_id} is not in vocab"),
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
    metadata
        .get_string(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Moonshine GGUF metadata is missing required key '{key}'"),
        })
}

fn required_metadata_string_array<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a [String], NativeAsrError> {
    metadata
        .get_string_array(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Moonshine GGUF metadata is missing required string array '{key}'"),
        })
}
