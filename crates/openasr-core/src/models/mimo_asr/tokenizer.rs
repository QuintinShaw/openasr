//! mimo-asr tokenizer: the official Qwen2 byte-level BPE vocabulary plus
//! MiMo's own added special tokens (`<|sosp|>`/`<|eosp|>`/`<|empty|>`/
//! `<|eot|>`/`<|eostm|>`), baked verbatim by
//! `tooling/mimo-asr/convert_mimo_asr.py` (`tokenizer.ggml.{model,tokens,merges}`).
//! Reuses the shared `models::gpt2_bpe` engine -- byte-level BPE encode/decode
//! has nothing family-specific about it (same precedent as
//! `firered_llm::tokenizer`/`qwen::tokenizer`).

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{Gpt2BpeTable, validate_gpt2_bpe_table_admission};
use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    required_metadata_string, required_metadata_string_array, required_metadata_u32,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

use super::runtime_contract::MimoSpecialTokens;

const MIMO_ASR_TOKENIZER_FAMILY: &str = "MiMo-V2.5-ASR";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

#[derive(Debug, Clone)]
pub(crate) struct MimoAsrTokenizer {
    bpe: Gpt2BpeTable,
    pub special: MimoSpecialTokens,
}

impl MimoAsrTokenizer {
    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: &GgufMetadata,
    ) -> Result<u64, String> {
        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )
        .map_err(|error| error.to_string())?;
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )
        .map_err(|error| error.to_string())?;
        Gpt2BpeTable::quoted_retained_system_memory_bytes(tokens, merges, "mimo-asr")
            .map_err(|error| error.to_string())
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        self.bpe.retained_system_memory_bytes_with_label("mimo-asr")
    }

    pub fn from_gguf_metadata(
        metadata: &GgufMetadata,
        special: MimoSpecialTokens,
    ) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(
            metadata,
            TOKENIZER_GGML_MODEL_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "mimo-asr GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }
        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )?;
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )?;
        let vocab_size =
            required_metadata_u32(metadata, "mimo.llm.vocab_size", MIMO_ASR_TOKENIZER_FAMILY)?;
        validate_gpt2_bpe_table_admission(
            tokens,
            merges,
            Some(vocab_size),
            MIMO_ASR_TOKENIZER_FAMILY,
        )?;

        let bpe = Gpt2BpeTable::from_admitted_tables_with_error_family(
            tokens,
            merges,
            MIMO_ASR_TOKENIZER_FAMILY,
            "mimo-asr",
        )?;

        for token_id in [
            special.eos_id,
            special.im_start_id,
            special.im_end_id,
            special.sosp_id,
            special.eosp_id,
            special.empty_id,
            special.eot_id,
            special.eostm_id,
        ] {
            bpe.validate_token_id(token_id)?;
        }

        Ok(Self { bpe, special })
    }

    /// Decode generated token ids to text, dropping the audio-boundary and
    /// speech-slot placeholder tokens the greedy decoder may still emit
    /// (`<|empty|>` in particular is a legitimate, if rare, argmax hit even
    /// with the 16L speech-gen `local_transformer` dropped -- P2.0 findings
    /// SS1 point 4/SS3's `asr_sft` postprocess strips it defensively too).
    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        self.bpe.decode_token_ids(token_ids, |token_id| {
            token_id == self.special.eos_id
                || token_id == self.special.im_start_id
                || token_id == self.special.im_end_id
                || token_id == self.special.sosp_id
                || token_id == self.special.eosp_id
                || token_id == self.special.empty_id
                || token_id == self.special.eot_id
                || token_id == self.special.eostm_id
        })
    }

    pub fn encode_prompt_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        self.bpe.encode_prompt_text(text)
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for MimoAsrTokenizer {}

impl PhraseBiasTokenEncoder for MimoAsrTokenizer {
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

    fn special_tokens() -> MimoSpecialTokens {
        MimoSpecialTokens {
            eos_id: 8,
            im_start_id: 0,
            im_end_id: 7,
            sosp_id: 9,
            eosp_id: 10,
            empty_id: 11,
            eot_id: 12,
            eostm_id: 13,
        }
    }

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
                "x".to_string(),
                "<|im_end|>".to_string(),
                "<|endoftext|>".to_string(),
                "<|sosp|>".to_string(),
                "<|eosp|>".to_string(),
                "<|empty|>".to_string(),
                "<|eot|>".to_string(),
                "<|eostm|>".to_string(),
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
            "mimo.llm.vocab_size".to_string(),
            GgufMetadataValue::U32(14),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_loads_and_decodes_skipping_control_and_audio_tokens() {
        let metadata = base_metadata();
        let tokenizer = MimoAsrTokenizer::from_gguf_metadata(&metadata, special_tokens())
            .expect("load tokenizer");
        let text = tokenizer
            .decode_text_token_ids(&[0, 3, 4, 7, 9, 11, 10])
            .expect("decode tokens");
        assert_eq!(text, " hi\nthere");
    }

    #[test]
    fn tokenizer_rejects_out_of_range_special_token() {
        let metadata = base_metadata();
        let mut special = special_tokens();
        special.eot_id = 999;
        let error =
            MimoAsrTokenizer::from_gguf_metadata(&metadata, special).expect_err("must fail");
        assert!(matches!(error, NativeAsrError::UnsupportedModelPack { .. }));
    }
}
