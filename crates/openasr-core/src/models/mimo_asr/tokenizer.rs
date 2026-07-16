//! mimo-asr tokenizer: the official Qwen2 byte-level BPE vocabulary plus
//! MiMo's own added special tokens (`<|sosp|>`/`<|eosp|>`/`<|empty|>`/
//! `<|eot|>`/`<|eostm|>`), baked verbatim by
//! `tooling/mimo-asr/convert_mimo_asr.py` (`tokenizer.ggml.{model,tokens,merges}`).
//! Reuses the shared `models::gpt2_bpe` engine -- byte-level BPE encode/decode
//! has nothing family-specific about it (same precedent as
//! `firered_llm::tokenizer`/`qwen::tokenizer`).

use std::collections::BTreeMap;

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::oasr_metadata::{
    TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    required_metadata_string, required_metadata_string_array,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

use super::runtime_contract::MimoSpecialTokens;

const MIMO_ASR_TOKENIZER_FAMILY: &str = "MiMo-V2.5-ASR";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

#[derive(Debug, Clone)]
pub(crate) struct MimoAsrTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    pub special: MimoSpecialTokens,
}

impl MimoAsrTokenizer {
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
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "mimo-asr GGUF tokenizer key '{TOKENIZER_GGML_TOKENS_KEY}' cannot be empty"
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            MIMO_ASR_TOKENIZER_FAMILY,
        )?;
        if merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "mimo-asr GGUF tokenizer key '{TOKENIZER_GGML_MERGES_KEY}' cannot be empty"
                ),
            });
        }

        let id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let token_to_id = build_token_to_id(tokens, MIMO_ASR_TOKENIZER_FAMILY)?;
        let merge_rank = build_merge_rank(merges);

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
            validate_token_id_in_range(&id_to_token, token_id)?;
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special,
        })
    }

    /// Decode generated token ids to text, dropping the audio-boundary and
    /// speech-slot placeholder tokens the greedy decoder may still emit
    /// (`<|empty|>` in particular is a legitimate, if rare, argmax hit even
    /// with the 16L speech-gen `local_transformer` dropped -- P2.0 findings
    /// SS1 point 4/SS3's `asr_sft` postprocess strips it defensively too).
    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if *token_id == self.special.eos_id
                || *token_id == self.special.im_start_id
                || *token_id == self.special.im_end_id
                || *token_id == self.special.sosp_id
                || *token_id == self.special.eosp_id
                || *token_id == self.special.empty_id
                || *token_id == self.special.eot_id
                || *token_id == self.special.eostm_id
            {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("mimo-asr tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("mimo-asr tokenizer id {token_id} is not in vocab"),
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
            MIMO_ASR_TOKENIZER_FAMILY,
        )
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

fn validate_token_id_in_range(
    id_to_token: &[Option<String>],
    token_id: u32,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!("mimo-asr tokenizer token id {token_id} does not fit into usize"),
    })?;
    if index < id_to_token.len() {
        return Ok(());
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "mimo-asr tokenizer token id {token_id} is out of range for vocab size {}",
            id_to_token.len()
        ),
    })
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
