use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::phrase_bias_decode::{
    PhraseBiasTokenEncoder, encode_sentencepiece_phrase_bias_tokens,
};
use crate::models::spm_decoder::{SpmDecoderConfig, decode_spm_pieces};

const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_LLAMA: &str = "llama";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

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
        let mut pieces = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("Moonshine tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(token) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Moonshine tokenizer id {token_id} is not in vocab"),
                });
            };
            pieces.push(token.as_str());
        }
        Ok(decode_spm_pieces(
            pieces,
            SpmDecoderConfig::BYTE_FALLBACK_BPE,
        ))
    }
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
