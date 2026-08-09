//! funasr-nano tokenizer: the stock Qwen3-0.6B byte-level BPE vocabulary
//! (`tokenizer.ggml.{model,tokens,merges}`, baked in verbatim by the pack
//! importer), reusing the same shared `models::gpt2_bpe` engine `qwen`,
//! `firered_llm`, and `moss_transcribe_diarize` use -- there is nothing
//! Qwen3-specific about byte-level BPE encode/decode.

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{Gpt2BpeTable, validate_gpt2_bpe_table_admission};
use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    required_metadata_string, required_metadata_string_array, required_metadata_u32,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

use super::runtime_contract::{
    LLM_CHATML_IM_END_TOKEN_ID_KEY, LLM_CHATML_IM_START_TOKEN_ID_KEY, LLM_ENDOFTEXT_TOKEN_ID_KEY,
    LLM_VOCAB_SIZE_KEY,
};

const FUNASR_NANO_TOKENIZER_FAMILY: &str = "Fun-ASR-Nano";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

#[derive(Debug, Clone)]
pub(crate) struct FunasrNanoTokenizer {
    bpe: Gpt2BpeTable,
    pub chatml_im_start_token_id: u32,
    pub chatml_im_end_token_id: u32,
    pub endoftext_token_id: u32,
}

impl FunasrNanoTokenizer {
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(
            metadata,
            TOKENIZER_GGML_MODEL_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Fun-ASR-Nano GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }

        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;
        let vocab_size =
            required_metadata_u32(metadata, LLM_VOCAB_SIZE_KEY, FUNASR_NANO_TOKENIZER_FAMILY)?;
        validate_gpt2_bpe_table_admission(
            tokens,
            merges,
            Some(vocab_size),
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;

        let chatml_im_start_token_id = required_metadata_u32(
            metadata,
            LLM_CHATML_IM_START_TOKEN_ID_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;
        let chatml_im_end_token_id = required_metadata_u32(
            metadata,
            LLM_CHATML_IM_END_TOKEN_ID_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;
        let endoftext_token_id = required_metadata_u32(
            metadata,
            LLM_ENDOFTEXT_TOKEN_ID_KEY,
            FUNASR_NANO_TOKENIZER_FAMILY,
        )?;

        let bpe = Gpt2BpeTable::from_admitted_tables(tokens, merges, FUNASR_NANO_TOKENIZER_FAMILY)?;

        for token_id in [
            chatml_im_start_token_id,
            chatml_im_end_token_id,
            endoftext_token_id,
        ] {
            bpe.validate_token_id(token_id)?;
        }

        Ok(Self {
            bpe,
            chatml_im_start_token_id,
            chatml_im_end_token_id,
            endoftext_token_id,
        })
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        self.bpe.decode_token_ids(token_ids, |token_id| {
            token_id == self.chatml_im_start_token_id
                || token_id == self.chatml_im_end_token_id
                || token_id == self.endoftext_token_id
        })
    }

    pub fn encode_prompt_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        self.bpe.encode_prompt_text(text)
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for FunasrNanoTokenizer {}

impl PhraseBiasTokenEncoder for FunasrNanoTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        self.encode_prompt_text(phrase)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        encode_bpe_phrase_bias_variants(phrase, |text| self.encode_prompt_text(text)).map(Some)
    }
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
        values.insert(LLM_VOCAB_SIZE_KEY.to_string(), GgufMetadataValue::U32(8));
        values.insert(
            LLM_CHATML_IM_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(0),
        );
        values.insert(
            LLM_CHATML_IM_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(6),
        );
        values.insert(
            LLM_ENDOFTEXT_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(7),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_loads_and_decodes_gpt2_bytes_skipping_control_tokens() {
        let tokenizer =
            FunasrNanoTokenizer::from_gguf_metadata(&base_metadata()).expect("load tokenizer");
        let text = tokenizer
            .decode_text_token_ids(&[0, 3, 4, 6])
            .expect("decode tokens");
        assert_eq!(text, " hi\nthere");
    }

    #[test]
    fn tokenizer_rejects_vocab_size_mismatch() {
        let mut values = base_metadata().values().clone();
        values.insert(LLM_VOCAB_SIZE_KEY.to_string(), GgufMetadataValue::U32(3));
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = FunasrNanoTokenizer::from_gguf_metadata(&metadata)
            .expect_err("mismatch should fail")
            .to_string();
        assert!(error.contains("token count"), "{error}");
    }
}
