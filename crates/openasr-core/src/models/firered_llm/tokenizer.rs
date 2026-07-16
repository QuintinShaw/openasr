//! firered-llm tokenizer: the official Qwen2-7B-Instruct byte-level BPE
//! vocabulary (`tokenizer.ggml.{model,tokens,merges}`, baked in verbatim by
//! `package_import`, including the literal ChatML special-token strings --
//! see `package_import`'s `patch_added_tokens`), reusing the same shared
//! `models::gpt2_bpe` engine `qwen::tokenizer` uses (there is nothing
//! Qwen2/Qwen3-specific about byte-level BPE encode/decode).

use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    required_metadata_string, required_metadata_string_array, required_metadata_u32,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

use super::runtime_contract::{
    FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY, FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY,
    FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY, FIRERED_LLM_LLM_VOCAB_SIZE_KEY,
    FIRERED_LLM_SPEECH_TOKEN_ID_KEY,
};

const FIRERED_LLM_TOKENIZER_FAMILY: &str = "FireRedASR2-LLM";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

#[derive(Debug, Clone)]
pub(crate) struct FireRedLlmTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    pub speech_token_id: u32,
    pub chatml_im_start_token_id: u32,
    pub chatml_im_end_token_id: u32,
    pub endoftext_token_id: u32,
}

impl FireRedLlmTokenizer {
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(
            metadata,
            TOKENIZER_GGML_MODEL_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "FireRedASR2-LLM GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }

        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "FireRedASR2-LLM GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_TOKENS_KEY
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        if merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "FireRedASR2-LLM GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_MERGES_KEY
                ),
            });
        }

        let vocab_size = required_metadata_u32(
            metadata,
            FIRERED_LLM_LLM_VOCAB_SIZE_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        let token_count =
            u32::try_from(tokens.len()).map_err(|_| NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "FireRedASR2-LLM GGUF tokenizer token count {} exceeds u32",
                    tokens.len()
                ),
            })?;
        if token_count != vocab_size {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "FireRedASR2-LLM GGUF tokenizer token count {} does not match '{}'={}",
                    token_count, FIRERED_LLM_LLM_VOCAB_SIZE_KEY, vocab_size
                ),
            });
        }

        let speech_token_id = required_metadata_u32(
            metadata,
            FIRERED_LLM_SPEECH_TOKEN_ID_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        let chatml_im_start_token_id = required_metadata_u32(
            metadata,
            FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        let chatml_im_end_token_id = required_metadata_u32(
            metadata,
            FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;
        let endoftext_token_id = required_metadata_u32(
            metadata,
            FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )?;

        let id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let token_to_id = build_token_to_id(tokens, FIRERED_LLM_TOKENIZER_FAMILY)?;
        let merge_rank = build_merge_rank(merges);

        for token_id in [
            speech_token_id,
            chatml_im_start_token_id,
            chatml_im_end_token_id,
            endoftext_token_id,
        ] {
            validate_token_id_in_range(&id_to_token, token_id)?;
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            speech_token_id,
            chatml_im_start_token_id,
            chatml_im_end_token_id,
            endoftext_token_id,
        })
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if *token_id == self.speech_token_id
                || *token_id == self.chatml_im_start_token_id
                || *token_id == self.chatml_im_end_token_id
                || *token_id == self.endoftext_token_id
            {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("FireRedASR2-LLM tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("FireRedASR2-LLM tokenizer id {token_id} is not in vocab"),
                });
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn encode_prompt_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        encode_prompt_text(
            text,
            &self.token_to_id,
            &self.merge_rank,
            FIRERED_LLM_TOKENIZER_FAMILY,
        )
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for FireRedLlmTokenizer {}

impl PhraseBiasTokenEncoder for FireRedLlmTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        self.encode_prompt_text(phrase)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        encode_bpe_phrase_bias_variants(phrase, |text| self.encode_prompt_text(text)).map(Some)
    }
}

fn validate_token_id_in_range(
    id_to_token: &[Option<String>],
    token_id: u32,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!("FireRedASR2-LLM tokenizer token id {token_id} does not fit into usize"),
    })?;
    if index < id_to_token.len() {
        return Ok(());
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "FireRedASR2-LLM tokenizer token id {token_id} is out of range for vocab size {}",
            id_to_token.len()
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{GgufMetadata, GgufMetadataValue};

    use super::*;

    fn base_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
        );
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|im_start|>".to_string(),
                "user".to_string(),
                "assistant".to_string(),
                "\u{0120}hi".to_string(),
                "\u{010A}there".to_string(),
                "\u{010A}".to_string(),
                "<speech>".to_string(),
                "<|im_end|>".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(vec![
                "u s".to_string(),
                "us e".to_string(),
                "use r".to_string(),
            ]),
        );
        values.insert(
            FIRERED_LLM_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(9),
        );
        values.insert(
            FIRERED_LLM_SPEECH_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(6),
        );
        values.insert(
            FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(0),
        );
        values.insert(
            FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(7),
        );
        values.insert(
            FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(8),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_loads_and_decodes_gpt2_bytes_skipping_control_tokens() {
        let metadata = base_metadata();
        let tokenizer = FireRedLlmTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        let text = tokenizer
            .decode_text_token_ids(&[0, 3, 4, 7])
            .expect("decode tokens");
        assert_eq!(text, " hi\nthere");
    }

    #[test]
    fn tokenizer_encodes_chatml_prompt_text() {
        // The `<speech>` placeholder is never text-encoded in production
        // (`decode_prompt::build_firered_llm_decode_prompt` splices
        // `speech_token_id` directly into the id sequence, exactly like
        // qwen3-asr's `<|audio_pad|>` span -- see that module's doc comment),
        // so this only exercises the ChatML boilerplate around it.
        let metadata = base_metadata();
        let tokenizer = FireRedLlmTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        let token_ids = tokenizer
            .encode_prompt_text("<|im_start|>user\n<|im_end|>\n")
            .expect("encode prompt");
        assert_eq!(token_ids, vec![0, 1, 5, 7, 5]);
    }

    #[test]
    fn tokenizer_rejects_vocab_size_mismatch() {
        let mut values = base_metadata().values().clone();
        values.insert(
            FIRERED_LLM_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(3),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = FireRedLlmTokenizer::from_gguf_metadata(&metadata)
            .expect_err("mismatch should fail")
            .to_string();
        assert!(error.contains("token count"), "{error}");
    }
}
