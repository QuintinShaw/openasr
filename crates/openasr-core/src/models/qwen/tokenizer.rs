use std::collections::{BTreeMap, BTreeSet};

use crate::NativeAsrError;
use crate::ggml_runtime::GgufMetadata;
use crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource;
use crate::models::gpt2_bpe::{
    build_merge_rank, build_token_to_id, encode_prompt_text, token_to_bytes,
};
use crate::models::phrase_bias_decode::{PhraseBiasTokenEncoder, encode_bpe_phrase_bias_variants};

use super::runtime_contract::{
    QWEN3_AUDIO_END_TOKEN_ID_KEY, QWEN3_AUDIO_PAD_TOKEN_ID_KEY, QWEN3_AUDIO_START_TOKEN_ID_KEY,
    QWEN3_EOS_TOKEN_ID_KEY, QWEN3_LLM_VOCAB_SIZE_KEY, QWEN3_PAD_TOKEN_ID_KEY,
};

const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    special_token_ids: BTreeSet<u32>,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
    pub eos_token_id: u32,
}

impl Qwen3AsrTokenizer {
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model = required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }

        let tokens = required_metadata_string_array(metadata, TOKENIZER_GGML_TOKENS_KEY)?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_TOKENS_KEY
                ),
            });
        }
        let merges = required_metadata_string_array(metadata, TOKENIZER_GGML_MERGES_KEY)?;
        if merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_MERGES_KEY
                ),
            });
        }

        let vocab_size = required_metadata_u32(metadata, QWEN3_LLM_VOCAB_SIZE_KEY)?;
        let token_count =
            u32::try_from(tokens.len()).map_err(|_| NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer token count {} exceeds u32",
                    tokens.len()
                ),
            })?;
        if token_count != vocab_size {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer token count {} does not match '{}'={}",
                    token_count, QWEN3_LLM_VOCAB_SIZE_KEY, vocab_size
                ),
            });
        }

        let audio_start_token_id = required_metadata_u32(metadata, QWEN3_AUDIO_START_TOKEN_ID_KEY)?;
        let audio_end_token_id = required_metadata_u32(metadata, QWEN3_AUDIO_END_TOKEN_ID_KEY)?;
        let audio_pad_token_id = required_metadata_u32(metadata, QWEN3_AUDIO_PAD_TOKEN_ID_KEY)?;
        let eos_token_id = required_metadata_u32(metadata, QWEN3_EOS_TOKEN_ID_KEY)?;
        let pad_token_id = required_metadata_u32(metadata, QWEN3_PAD_TOKEN_ID_KEY)?;

        let mut id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let mut token_to_id = build_token_to_id(tokens, "Qwen3-ASR")?;
        let merge_rank = build_merge_rank(merges);

        let mut special_token_ids = BTreeSet::new();
        for token_id in [
            audio_start_token_id,
            audio_end_token_id,
            audio_pad_token_id,
            eos_token_id,
            pad_token_id,
        ] {
            validate_token_id_in_range(&id_to_token, token_id)?;
            special_token_ids.insert(token_id);
        }
        patch_known_special_tokens(&mut id_to_token, &mut token_to_id);

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            special_token_ids,
            audio_start_token_id,
            audio_end_token_id,
            audio_pad_token_id,
            eos_token_id,
        })
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if self.special_token_ids.contains(token_id) {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!("Qwen3-ASR tokenizer id {token_id} does not fit into usize"),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!("Qwen3-ASR tokenizer id {token_id} is not in vocab"),
                });
            };
            bytes.extend(token_to_bytes(token));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn audio_prompt_token_triplet(&self) -> (u32, u32, u32) {
        (
            self.audio_start_token_id,
            self.audio_end_token_id,
            self.audio_pad_token_id,
        )
    }

    pub fn encode_prompt_text(&self, text: &str) -> Result<Vec<u32>, NativeAsrError> {
        encode_prompt_text(text, &self.token_to_id, &self.merge_rank, "Qwen3-ASR")
    }
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for Qwen3AsrTokenizer {
    fn audio_end_token_id(&self) -> Option<u32> {
        Some(self.audio_end_token_id)
    }

    fn audio_pad_token_id(&self) -> Option<u32> {
        Some(self.audio_pad_token_id)
    }
}

impl PhraseBiasTokenEncoder for Qwen3AsrTokenizer {
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
        self.encode_prompt_text(phrase)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        // Byte-level BPE: also match the leading-space form the model emits
        // mid-sentence, not just the standalone tokenization.
        encode_bpe_phrase_bias_variants(phrase, |text| self.encode_prompt_text(text)).map(Some)
    }
}

const KNOWN_SPECIAL_TOKEN_PATCHES: &[(u32, &str)] = &[
    (151643, "<|endoftext|>"),
    (151644, "<|im_start|>"),
    (151645, "<|im_end|>"),
    (151646, "<|object_ref_start|>"),
    (151647, "<|object_ref_end|>"),
    (151648, "<|box_start|>"),
    (151649, "<|box_end|>"),
    (151650, "<|quad_start|>"),
    (151651, "<|quad_end|>"),
    (151652, "<|vision_start|>"),
    (151653, "<|vision_end|>"),
    (151654, "<|vision_pad|>"),
    (151655, "<|image_pad|>"),
    (151656, "<|video_pad|>"),
    (151669, "<|audio_start|>"),
    (151670, "<|audio_end|>"),
    (151676, "<|audio_pad|>"),
];

fn patch_known_special_tokens(
    id_to_token: &mut [Option<String>],
    token_to_id: &mut BTreeMap<String, u32>,
) {
    for &(token_id, token_text) in KNOWN_SPECIAL_TOKEN_PATCHES {
        let Ok(index) = usize::try_from(token_id) else {
            continue;
        };
        let Some(slot) = id_to_token.get_mut(index) else {
            continue;
        };
        if let Some(previous) = slot.replace(token_text.to_string()) {
            token_to_id.remove(&previous);
        }
        token_to_id.insert(token_text.to_string(), token_id);
    }
}

fn validate_token_id_in_range(
    id_to_token: &[Option<String>],
    token_id: u32,
) -> Result<(), NativeAsrError> {
    let index = usize::try_from(token_id).map_err(|_| NativeAsrError::UnsupportedModelPack {
        reason: format!("Qwen3-ASR tokenizer token id {token_id} does not fit into usize"),
    })?;
    if index < id_to_token.len() {
        return Ok(());
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "Qwen3-ASR tokenizer token id {token_id} is out of range for vocab size {}",
            id_to_token.len()
        ),
    })
}

fn required_metadata_string<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a str, NativeAsrError> {
    let value = metadata
        .get_string(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Qwen3-ASR GGUF tokenizer is missing required key '{key}'"),
        })?;
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(NativeAsrError::UnsupportedModelPack {
            reason: format!("Qwen3-ASR GGUF tokenizer key '{key}' cannot be empty"),
        });
    }
    Ok(normalized)
}

fn required_metadata_u32(
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<u32, NativeAsrError> {
    if let Some(value) = metadata.get_u32(key) {
        return Ok(value);
    }
    if let Some(value) = metadata.get_u64(key) {
        return u32::try_from(value).map_err(|_| NativeAsrError::UnsupportedModelPack {
            reason: format!("Qwen3-ASR GGUF tokenizer key '{key}' value {value} does not fit u32"),
        });
    }
    if let Some(value) = metadata.get_string(key) {
        let parsed = value.trim().parse::<u32>().map_err(|error| {
            NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "Qwen3-ASR GGUF tokenizer key '{key}' cannot parse '{value}' as u32: {error}"
                ),
            }
        })?;
        return Ok(parsed);
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!("Qwen3-ASR GGUF tokenizer is missing required key '{key}'"),
    })
}

fn required_metadata_string_array<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a [String], NativeAsrError> {
    metadata
        .get_string_array(key)
        .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
            reason: format!("Qwen3-ASR GGUF tokenizer requires key '{key}' as array[string]"),
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
                "system".to_string(),
                "user".to_string(),
                "assistant".to_string(),
                "\u{0120}hi".to_string(),
                "\u{010A}there".to_string(),
                "\u{010A}".to_string(),
                "<|audio_start|>".to_string(),
                "<|audio_end|>".to_string(),
                "<|audio_pad|>".to_string(),
                "<|im_end|>".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(vec![
                "s y".to_string(),
                "sy s".to_string(),
                "sys t".to_string(),
                "syst e".to_string(),
                "syste m".to_string(),
            ]),
        );
        values.insert(
            QWEN3_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(12),
        );
        values.insert(
            QWEN3_AUDIO_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(7),
        );
        values.insert(
            QWEN3_AUDIO_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(8),
        );
        values.insert(
            QWEN3_AUDIO_PAD_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(9),
        );
        values.insert(
            QWEN3_EOS_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(10),
        );
        values.insert(
            QWEN3_PAD_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(11),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_loads_and_decodes_gpt2_bytes() {
        let metadata = base_metadata();
        let tokenizer = Qwen3AsrTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        let text = tokenizer
            .decode_text_token_ids(&[7, 4, 5, 11])
            .expect("decode tokens");
        assert_eq!(text, " hi\nthere");
    }

    #[test]
    fn tokenizer_encodes_chatml_prompt_with_special_tokens() {
        let metadata = base_metadata();
        let tokenizer = Qwen3AsrTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        let token_ids = tokenizer
            .encode_prompt_text("<|im_start|>system\n<|audio_start|><|audio_end|>\n")
            .expect("encode prompt");
        assert_eq!(token_ids, vec![0, 1, 6, 7, 8, 6]);
    }

    #[test]
    fn tokenizer_rejects_vocab_size_mismatch() {
        let mut values = base_metadata().values().clone();
        values.insert(
            QWEN3_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(6),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = Qwen3AsrTokenizer::from_gguf_metadata(&metadata)
            .expect_err("mismatch should fail")
            .to_string();
        assert!(error.contains("token count"), "{error}");
        assert!(error.contains(QWEN3_LLM_VOCAB_SIZE_KEY), "{error}");
    }
}
