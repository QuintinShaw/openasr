//! moss-transcribe-diarize tokenizer: the official Qwen3 byte-level BPE
//! vocabulary (`tokenizer.ggml.{model,tokens,merges}`, baked in verbatim by
//! `package_import`, including the `<|audio_start|>` / `<|audio_end|>` /
//! `<|audio_pad|>` special tokens from `tokenizer.json`'s `added_tokens`),
//! reusing the same shared `models::gpt2_bpe` engine every other BPE family
//! in this crate uses (there is nothing MOSS-specific about byte-level BPE
//! encode/decode).

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
use crate::models::runtime_memory::{checked_sum, element_bytes};
use crate::models::system_memory_owner::{SystemMemoryCapacity, SystemMemoryOwnerError};

use super::runtime_contract::{
    LLM_AUDIO_END_TOKEN_ID_KEY, LLM_AUDIO_PAD_TOKEN_ID_KEY, LLM_AUDIO_START_TOKEN_ID_KEY,
    LLM_VOCAB_SIZE_KEY,
};

const MOSS_TD_TOKENIZER_FAMILY: &str = "MOSS-Transcribe-Diarize";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";

#[derive(Debug, Clone)]
pub(crate) struct MossTdTokenizer {
    id_to_token: Vec<Option<String>>,
    token_to_id: BTreeMap<String, u32>,
    merge_rank: BTreeMap<String, usize>,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_pad_token_id: u32,
    /// `im_start`/`im_end` (Qwen ChatML turn delimiters) resolved by literal
    /// text lookup rather than a dedicated metadata key -- both are ordinary
    /// vocab entries already present in `tokenizer.ggml.tokens`. The decoder
    /// never actually emits `im_start` mid-generation, but
    /// `decode_text_token_ids` strips it defensively anyway, same precedent
    /// as `firered_llm::tokenizer`'s `chatml_im_start_token_id`.
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    /// Single-token ids for `'0'..'9'`, used by `decode_prompt`'s time-anchor
    /// markers (mirrors upstream `MossTranscribeDiarizeProcessor::_get_digit_token_ids`).
    pub digit_token_ids: [u32; 10],
}

impl MossTdTokenizer {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = SystemMemoryCapacity::default();
        bytes.add_vec(
            &self.id_to_token,
            "moss-transcribe-diarize tokenizer id table",
        )?;
        for token in self.id_to_token.iter().flatten() {
            bytes.add_string(token, "moss-transcribe-diarize tokenizer id token")?;
        }
        for (label, len, entry_size) in [
            (
                "moss-transcribe-diarize tokenizer token map",
                self.token_to_id.len(),
                std::mem::size_of::<(String, u32)>(),
            ),
            (
                "moss-transcribe-diarize tokenizer merge map",
                self.merge_rank.len(),
                std::mem::size_of::<(String, usize)>(),
            ),
        ] {
            bytes.add_usize(
                len.checked_mul(entry_size)
                    .ok_or_else(|| format!("{label} byte count overflowed"))?,
                label,
            )?;
        }
        for key in self.token_to_id.keys().chain(self.merge_rank.keys()) {
            bytes.add_string(key, "moss-transcribe-diarize tokenizer map key")?;
        }
        Ok(bytes.finish())
    }

    pub(crate) fn quoted_retained_system_memory_bytes(
        metadata: &GgufMetadata,
    ) -> Result<u64, SystemMemoryOwnerError> {
        let tokens = metadata
            .get_string_array(TOKENIZER_GGML_TOKENS_KEY)
            .ok_or_else(|| {
                tokenizer_capacity_error(format!(
                    "metadata is missing '{TOKENIZER_GGML_TOKENS_KEY}'"
                ))
            })?;
        let merges = metadata
            .get_string_array(TOKENIZER_GGML_MERGES_KEY)
            .ok_or_else(|| {
                tokenizer_capacity_error(format!(
                    "metadata is missing '{TOKENIZER_GGML_MERGES_KEY}'"
                ))
            })?;
        let token_text_bytes = tokenizer_text_bytes(tokens, "tokenizer token text")?;
        let merge_text_bytes = tokenizer_text_bytes(merges, "tokenizer merge text")?;
        let token_text_copies = token_text_bytes.checked_mul(2).ok_or_else(|| {
            tokenizer_capacity_error("tokenizer token text copies byte count overflowed")
        })?;

        checked_sum(
            [
                element_bytes::<Option<String>>(
                    tokens.len(),
                    MOSS_TD_TOKENIZER_FAMILY,
                    "tokenizer id table",
                )?,
                element_bytes::<(String, u32)>(
                    tokens.len(),
                    MOSS_TD_TOKENIZER_FAMILY,
                    "tokenizer reverse map",
                )?,
                token_text_copies,
                element_bytes::<(String, usize)>(
                    merges.len(),
                    MOSS_TD_TOKENIZER_FAMILY,
                    "tokenizer merge map",
                )?,
                merge_text_bytes,
            ],
            MOSS_TD_TOKENIZER_FAMILY,
            "tokenizer retained bytes",
        )
    }

    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self, NativeAsrError> {
        let tokenizer_model =
            required_metadata_string(metadata, TOKENIZER_GGML_MODEL_KEY, MOSS_TD_TOKENIZER_FAMILY)?;
        if !tokenizer_model.eq_ignore_ascii_case(TOKENIZER_GGML_MODEL_VALUE_GPT2) {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "moss-transcribe-diarize GGUF tokenizer key '{}' must be '{}', got '{}'",
                    TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2, tokenizer_model
                ),
            });
        }

        let tokens = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_TOKENS_KEY,
            MOSS_TD_TOKENIZER_FAMILY,
        )?;
        if tokens.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "moss-transcribe-diarize GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_TOKENS_KEY
                ),
            });
        }
        let merges = required_metadata_string_array(
            metadata,
            TOKENIZER_GGML_MERGES_KEY,
            MOSS_TD_TOKENIZER_FAMILY,
        )?;
        if merges.is_empty() {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "moss-transcribe-diarize GGUF tokenizer key '{}' cannot be empty",
                    TOKENIZER_GGML_MERGES_KEY
                ),
            });
        }

        let vocab_size =
            required_metadata_u32(metadata, LLM_VOCAB_SIZE_KEY, MOSS_TD_TOKENIZER_FAMILY)?;
        let token_count =
            u32::try_from(tokens.len()).map_err(|_| NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "moss-transcribe-diarize GGUF tokenizer token count {} exceeds u32",
                    tokens.len()
                ),
            })?;
        if token_count != vocab_size {
            return Err(NativeAsrError::UnsupportedModelPack {
                reason: format!(
                    "moss-transcribe-diarize GGUF tokenizer token count {} does not match '{}'={}",
                    token_count, LLM_VOCAB_SIZE_KEY, vocab_size
                ),
            });
        }

        let audio_start_token_id = required_metadata_u32(
            metadata,
            LLM_AUDIO_START_TOKEN_ID_KEY,
            MOSS_TD_TOKENIZER_FAMILY,
        )?;
        let audio_end_token_id = required_metadata_u32(
            metadata,
            LLM_AUDIO_END_TOKEN_ID_KEY,
            MOSS_TD_TOKENIZER_FAMILY,
        )?;
        let audio_pad_token_id = required_metadata_u32(
            metadata,
            LLM_AUDIO_PAD_TOKEN_ID_KEY,
            MOSS_TD_TOKENIZER_FAMILY,
        )?;

        let id_to_token = tokens
            .iter()
            .map(|token| Some(token.clone()))
            .collect::<Vec<_>>();
        let token_to_id = build_token_to_id(tokens, MOSS_TD_TOKENIZER_FAMILY)?;
        let merge_rank = build_merge_rank(merges);

        let im_start_token_id = *token_to_id.get("<|im_start|>").ok_or_else(|| {
            NativeAsrError::UnsupportedModelPack {
                reason: "moss-transcribe-diarize tokenizer vocab has no '<|im_start|>' entry"
                    .to_string(),
            }
        })?;
        let im_end_token_id =
            *token_to_id
                .get("<|im_end|>")
                .ok_or_else(|| NativeAsrError::UnsupportedModelPack {
                    reason: "moss-transcribe-diarize tokenizer vocab has no '<|im_end|>' entry"
                        .to_string(),
                })?;

        let mut digit_token_ids = [0u32; 10];
        for (digit, slot) in "0123456789".chars().zip(digit_token_ids.iter_mut()) {
            *slot = *token_to_id.get(digit.encode_utf8(&mut [0; 4]) as &str).ok_or_else(|| {
                NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "moss-transcribe-diarize tokenizer vocab is missing a single-token entry for digit '{digit}'"
                    ),
                }
            })?;
        }

        for token_id in [
            audio_start_token_id,
            audio_end_token_id,
            audio_pad_token_id,
            im_start_token_id,
            im_end_token_id,
        ] {
            validate_token_id_in_range(&id_to_token, token_id)?;
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            audio_start_token_id,
            audio_end_token_id,
            audio_pad_token_id,
            im_start_token_id,
            im_end_token_id,
            digit_token_ids,
        })
    }

    pub fn decode_text_token_ids(&self, token_ids: &[u32]) -> Result<String, NativeAsrError> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if *token_id == self.audio_start_token_id
                || *token_id == self.audio_end_token_id
                || *token_id == self.audio_pad_token_id
                || *token_id == self.im_start_token_id
                || *token_id == self.im_end_token_id
            {
                continue;
            }
            let index = usize::try_from(*token_id).map_err(|_| NativeAsrError::SessionFailed {
                message: format!(
                    "moss-transcribe-diarize tokenizer id {token_id} does not fit into usize"
                ),
            })?;
            let Some(Some(token)) = self.id_to_token.get(index) else {
                return Err(NativeAsrError::SessionFailed {
                    message: format!(
                        "moss-transcribe-diarize tokenizer id {token_id} is not in vocab"
                    ),
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
            MOSS_TD_TOKENIZER_FAMILY,
        )
    }
}

fn tokenizer_text_bytes(values: &[String], label: &str) -> Result<u64, SystemMemoryOwnerError> {
    values.iter().try_fold(0_u64, |total, value| {
        let length = u64::try_from(value.len())
            .map_err(|_| tokenizer_capacity_error(format!("{label} length does not fit u64")))?;
        total
            .checked_add(length)
            .ok_or_else(|| tokenizer_capacity_error(format!("{label} byte count overflowed")))
    })
}

fn tokenizer_capacity_error(reason: impl Into<String>) -> SystemMemoryOwnerError {
    SystemMemoryOwnerError::capacity_failure(
        "model_runtime_memory",
        format!("{MOSS_TD_TOKENIZER_FAMILY}: {}", reason.into()),
    )
}

impl BuiltinSeq2SeqDecodePolicyTokenSource for MossTdTokenizer {}

impl PhraseBiasTokenEncoder for MossTdTokenizer {
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
        reason: format!(
            "moss-transcribe-diarize tokenizer token id {token_id} does not fit into usize"
        ),
    })?;
    if index < id_to_token.len() {
        return Ok(());
    }
    Err(NativeAsrError::UnsupportedModelPack {
        reason: format!(
            "moss-transcribe-diarize tokenizer token id {token_id} is out of range for vocab size {}",
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
        let mut tokens: Vec<String> = "0123456789".chars().map(|c| c.to_string()).collect();
        tokens.extend([
            "<|im_start|>".to_string(),
            "user".to_string(),
            "assistant".to_string(),
            "\u{0120}hi".to_string(),
            "\u{010A}".to_string(),
            "<|audio_start|>".to_string(),
            "<|audio_end|>".to_string(),
            "<|audio_pad|>".to_string(),
            "<|im_end|>".to_string(),
            "<|endoftext|>".to_string(),
        ]);
        let vocab_size = tokens.len() as u32;
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            // A real pack's merge list is never empty (byte-level BPE always
            // has real merges); this fixture only needs one placeholder entry
            // to satisfy `MossTdTokenizer::from_gguf_metadata`'s non-empty
            // check, not the actual pairing behavior these tests exercise.
            GgufMetadataValue::StringArray(vec!["\u{0120} \u{010A}".to_string()]),
        );
        values.insert(
            LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(vocab_size),
        );
        // Indices into the 20-entry `tokens` vec above: 0-9=digits,
        // 10=<|im_start|>, 11=user, 12=assistant, 13=hi, 14=\n,
        // 15=<|audio_start|>, 16=<|audio_end|>, 17=<|audio_pad|>,
        // 18=<|im_end|>, 19=<|endoftext|>.
        values.insert(
            LLM_AUDIO_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(15),
        );
        values.insert(
            LLM_AUDIO_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(16),
        );
        values.insert(
            LLM_AUDIO_PAD_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(17),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn tokenizer_resolves_digit_ids_and_audio_ids() {
        let metadata = base_metadata();
        let tokenizer = MossTdTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        assert_eq!(tokenizer.digit_token_ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(tokenizer.audio_start_token_id, 15);
        assert_eq!(tokenizer.audio_pad_token_id, 17);
        assert_eq!(tokenizer.im_end_token_id, 18);
    }

    #[test]
    fn tokenizer_decodes_skipping_control_tokens() {
        let metadata = base_metadata();
        let tokenizer = MossTdTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        // 10=<|im_start|>, 13=hi, 15=<|audio_start|>, 18=<|im_end|>
        let text = tokenizer
            .decode_text_token_ids(&[10, 13, 15, 18])
            .expect("decode");
        assert_eq!(text, " hi");
    }

    #[test]
    fn tokenizer_rejects_vocab_size_mismatch() {
        let mut values = base_metadata().values().clone();
        values.insert(LLM_VOCAB_SIZE_KEY.to_string(), GgufMetadataValue::U32(3));
        let metadata = GgufMetadata::from_values_for_test(values);
        let error = MossTdTokenizer::from_gguf_metadata(&metadata)
            .expect_err("mismatch should fail")
            .to_string();
        assert!(error.contains("token count"), "{error}");
    }

    #[test]
    fn retained_system_memory_is_within_metadata_quote() {
        let metadata = base_metadata();
        let tokenizer = MossTdTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        let actual = tokenizer
            .retained_system_memory_bytes()
            .expect("count retained tokenizer memory");
        let quote = MossTdTokenizer::quoted_retained_system_memory_bytes(&metadata)
            .expect("quote retained tokenizer memory");
        assert!(actual <= quote, "actual {actual} exceeds quote {quote}");
    }

    #[test]
    fn duplicate_tokens_and_merges_still_fit_the_count_based_quote() {
        let mut values = base_metadata().values().clone();
        let mut tokens = match values.remove(TOKENIZER_GGML_TOKENS_KEY) {
            Some(GgufMetadataValue::StringArray(tokens)) => tokens,
            _ => panic!("base fixture must contain tokenizer tokens"),
        };
        tokens.push("user".to_string());
        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(tokens.clone()),
        );
        values.insert(
            LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(tokens.len() as u32),
        );

        let mut merges = match values.remove(TOKENIZER_GGML_MERGES_KEY) {
            Some(GgufMetadataValue::StringArray(merges)) => merges,
            _ => panic!("base fixture must contain tokenizer merges"),
        };
        merges.push(merges[0].clone());
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(merges.clone()),
        );

        let metadata = GgufMetadata::from_values_for_test(values);
        let tokenizer = MossTdTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
        assert!(tokenizer.token_to_id.len() < tokens.len());
        assert!(tokenizer.merge_rank.len() < merges.len());
        let actual = tokenizer
            .retained_system_memory_bytes()
            .expect("count retained tokenizer memory");
        let quote = MossTdTokenizer::quoted_retained_system_memory_bytes(&metadata)
            .expect("quote retained tokenizer memory");
        assert!(actual <= quote, "actual {actual} exceeds quote {quote}");
    }

    #[test]
    fn tokenizer_quote_fails_closed_when_metadata_is_missing() {
        let mut values = base_metadata().values().clone();
        values.remove(TOKENIZER_GGML_TOKENS_KEY);
        let missing_tokens = GgufMetadata::from_values_for_test(values);
        assert!(MossTdTokenizer::quoted_retained_system_memory_bytes(&missing_tokens).is_err());

        let mut values = base_metadata().values().clone();
        values.remove(TOKENIZER_GGML_MERGES_KEY);
        let missing_merges = GgufMetadata::from_values_for_test(values);
        assert!(MossTdTokenizer::quoted_retained_system_memory_bytes(&missing_merges).is_err());
    }
}
