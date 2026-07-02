use crate::GgufMetadata;
use crate::models::runtime_contract::{
    MetadataContractError, optional_u64_scalar as optional_u64_scalar_contract,
    required_string_scalar as required_string_scalar_contract,
    required_u64_scalar as required_u64_scalar_contract, u64_to_u32 as u64_to_u32_contract,
    u64_to_usize as u64_to_usize_contract,
    validate_positive_usize as validate_positive_usize_contract,
};

use super::tokenizer::{TOKENIZER_GGML_EOT_TOKEN_ID_KEY, TOKENIZER_GGML_SOT_TOKEN_ID_KEY};
use crate::arch::{
    GENERAL_ARCHITECTURE_KEY,
    hparams::{
        WHISPER_DECODER_BLOCK_COUNT_KEY, WHISPER_DECODER_CONTEXT_LENGTH_KEY,
        WHISPER_DECODER_EMBEDDING_LENGTH_KEY, WHISPER_DECODER_HEAD_COUNT_KEY,
        WHISPER_ENCODER_BLOCK_COUNT_KEY, WHISPER_ENCODER_CONTEXT_LENGTH_KEY,
        WHISPER_ENCODER_EMBEDDING_LENGTH_KEY, WHISPER_ENCODER_HEAD_COUNT_KEY,
        WHISPER_ENCODER_MELS_COUNT_KEY, WHISPER_VOCAB_SIZE_KEY,
    },
};

const WHISPER_DEFAULT_SOT_TOKEN_ID: u32 = 50258;
const WHISPER_DEFAULT_EOT_TOKEN_ID: u32 = 50257;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperGgmlExecutionMetadata {
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_hidden_size: usize,
    pub encoder_attention_heads: usize,
    pub encoder_context_length: usize,
    pub decoder_attention_heads: usize,
    pub max_target_positions: usize,
    pub decoder_hidden_size: usize,
    pub vocab_size: usize,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub encoder_mels_count: usize,
}

pub(crate) fn validate_whisper_execution_metadata(
    metadata: &GgufMetadata,
) -> Result<WhisperGgmlExecutionMetadata, MetadataContractError> {
    let architecture = required_string_scalar_contract(metadata, GENERAL_ARCHITECTURE_KEY)?;
    if architecture != "whisper" {
        return Err(MetadataContractError::InvalidValue {
            key: GENERAL_ARCHITECTURE_KEY,
            reason: format!("expected 'whisper', got '{architecture}'"),
        });
    }

    let encoder_layers = required_u64_scalar_contract(metadata, WHISPER_ENCODER_BLOCK_COUNT_KEY)?;
    let decoder_layers = required_u64_scalar_contract(metadata, WHISPER_DECODER_BLOCK_COUNT_KEY)?;
    let encoder_hidden_size =
        required_u64_scalar_contract(metadata, WHISPER_ENCODER_EMBEDDING_LENGTH_KEY)?;
    let encoder_attention_heads =
        required_u64_scalar_contract(metadata, WHISPER_ENCODER_HEAD_COUNT_KEY)?;
    let encoder_context_length =
        required_u64_scalar_contract(metadata, WHISPER_ENCODER_CONTEXT_LENGTH_KEY)?;
    let decoder_hidden_size =
        required_u64_scalar_contract(metadata, WHISPER_DECODER_EMBEDDING_LENGTH_KEY)?;
    let decoder_attention_heads =
        required_u64_scalar_contract(metadata, WHISPER_DECODER_HEAD_COUNT_KEY)?;
    let max_target_positions =
        required_u64_scalar_contract(metadata, WHISPER_DECODER_CONTEXT_LENGTH_KEY)?;
    let vocab_size = required_u64_scalar_contract(metadata, WHISPER_VOCAB_SIZE_KEY)?;
    let decoder_start_token_id =
        optional_u64_scalar_contract(metadata, TOKENIZER_GGML_SOT_TOKEN_ID_KEY)?
            .unwrap_or(WHISPER_DEFAULT_SOT_TOKEN_ID as u64);
    let eos_token_id = optional_u64_scalar_contract(metadata, TOKENIZER_GGML_EOT_TOKEN_ID_KEY)?
        .unwrap_or(WHISPER_DEFAULT_EOT_TOKEN_ID as u64);
    let encoder_mels_count =
        required_u64_scalar_contract(metadata, WHISPER_ENCODER_MELS_COUNT_KEY)?;

    let encoder_layers = u64_to_usize_contract(encoder_layers, WHISPER_ENCODER_BLOCK_COUNT_KEY)?;
    let decoder_layers = u64_to_usize_contract(decoder_layers, WHISPER_DECODER_BLOCK_COUNT_KEY)?;
    let encoder_hidden_size =
        u64_to_usize_contract(encoder_hidden_size, WHISPER_ENCODER_EMBEDDING_LENGTH_KEY)?;
    let encoder_attention_heads =
        u64_to_usize_contract(encoder_attention_heads, WHISPER_ENCODER_HEAD_COUNT_KEY)?;
    let encoder_context_length =
        u64_to_usize_contract(encoder_context_length, WHISPER_ENCODER_CONTEXT_LENGTH_KEY)?;
    let decoder_hidden_size =
        u64_to_usize_contract(decoder_hidden_size, WHISPER_DECODER_EMBEDDING_LENGTH_KEY)?;
    let decoder_attention_heads =
        u64_to_usize_contract(decoder_attention_heads, WHISPER_DECODER_HEAD_COUNT_KEY)?;
    let max_target_positions =
        u64_to_usize_contract(max_target_positions, WHISPER_DECODER_CONTEXT_LENGTH_KEY)?;
    let vocab_size = u64_to_usize_contract(vocab_size, WHISPER_VOCAB_SIZE_KEY)?;
    let decoder_start_token_id =
        u64_to_u32_contract(decoder_start_token_id, TOKENIZER_GGML_SOT_TOKEN_ID_KEY)?;
    let eos_token_id = u64_to_u32_contract(eos_token_id, TOKENIZER_GGML_EOT_TOKEN_ID_KEY)?;
    let encoder_mels_count =
        u64_to_usize_contract(encoder_mels_count, WHISPER_ENCODER_MELS_COUNT_KEY)?;

    validate_positive_usize_contract(encoder_layers, WHISPER_ENCODER_BLOCK_COUNT_KEY)?;
    validate_positive_usize_contract(decoder_layers, WHISPER_DECODER_BLOCK_COUNT_KEY)?;
    validate_positive_usize_contract(encoder_hidden_size, WHISPER_ENCODER_EMBEDDING_LENGTH_KEY)?;
    validate_positive_usize_contract(encoder_attention_heads, WHISPER_ENCODER_HEAD_COUNT_KEY)?;
    validate_positive_usize_contract(encoder_context_length, WHISPER_ENCODER_CONTEXT_LENGTH_KEY)?;
    validate_positive_usize_contract(decoder_hidden_size, WHISPER_DECODER_EMBEDDING_LENGTH_KEY)?;
    validate_positive_usize_contract(decoder_attention_heads, WHISPER_DECODER_HEAD_COUNT_KEY)?;
    validate_positive_usize_contract(max_target_positions, WHISPER_DECODER_CONTEXT_LENGTH_KEY)?;
    validate_positive_usize_contract(vocab_size, WHISPER_VOCAB_SIZE_KEY)?;
    validate_positive_usize_contract(encoder_mels_count, WHISPER_ENCODER_MELS_COUNT_KEY)?;

    if !encoder_hidden_size.is_multiple_of(encoder_attention_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: WHISPER_ENCODER_HEAD_COUNT_KEY,
            reason: format!(
                "{WHISPER_ENCODER_EMBEDDING_LENGTH_KEY} {encoder_hidden_size} must be divisible by {WHISPER_ENCODER_HEAD_COUNT_KEY} {encoder_attention_heads}"
            ),
        });
    }
    if !decoder_hidden_size.is_multiple_of(decoder_attention_heads) {
        return Err(MetadataContractError::InvalidValue {
            key: WHISPER_DECODER_HEAD_COUNT_KEY,
            reason: format!(
                "{WHISPER_DECODER_EMBEDDING_LENGTH_KEY} {decoder_hidden_size} must be divisible by {WHISPER_DECODER_HEAD_COUNT_KEY} {decoder_attention_heads}"
            ),
        });
    }

    Ok(WhisperGgmlExecutionMetadata {
        encoder_layers,
        decoder_layers,
        encoder_hidden_size,
        encoder_attention_heads,
        encoder_context_length,
        decoder_attention_heads,
        max_target_positions,
        decoder_hidden_size,
        vocab_size,
        decoder_start_token_id,
        eos_token_id,
        encoder_mels_count,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::GgufMetadataValue;

    use super::*;

    fn base_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("whisper".to_string()),
        );
        values.insert(
            "whisper.encoder.block_count".to_string(),
            GgufMetadataValue::U64(2),
        );
        values.insert(
            "whisper.encoder.embedding_length".to_string(),
            GgufMetadataValue::U64(16),
        );
        values.insert(
            "whisper.encoder.attention.head_count".to_string(),
            GgufMetadataValue::U64(2),
        );
        values.insert(
            "whisper.encoder.context_length".to_string(),
            GgufMetadataValue::U64(8),
        );
        values.insert(
            "whisper.encoder.mels_count".to_string(),
            GgufMetadataValue::U64(80),
        );
        values.insert(
            "whisper.decoder.block_count".to_string(),
            GgufMetadataValue::U64(2),
        );
        values.insert(
            "whisper.decoder.embedding_length".to_string(),
            GgufMetadataValue::U64(16),
        );
        values.insert(
            "whisper.decoder.attention.head_count".to_string(),
            GgufMetadataValue::U64(2),
        );
        values.insert(
            "whisper.decoder.context_length".to_string(),
            GgufMetadataValue::U64(8),
        );
        values.insert(
            "whisper.vocab_size".to_string(),
            GgufMetadataValue::U64(128),
        );
        values.insert(
            "tokenizer.ggml.sot_token_id".to_string(),
            GgufMetadataValue::U64(7),
        );
        values.insert(
            "tokenizer.ggml.eot_token_id".to_string(),
            GgufMetadataValue::U64(8),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn validates_minimal_whisper_metadata_contract() {
        let metadata = base_metadata();
        let parsed = validate_whisper_execution_metadata(&metadata).expect("metadata must parse");
        assert_eq!(parsed.encoder_layers, 2);
        assert_eq!(parsed.decoder_start_token_id, 7);
        assert_eq!(parsed.eos_token_id, 8);
    }

    #[test]
    fn rejects_legacy_naked_hparam_keys() {
        let mut metadata = base_metadata();
        let mut values = metadata.values().clone();
        values.remove("whisper.encoder.block_count");
        values.insert("n_audio_layer".to_string(), GgufMetadataValue::U64(3));
        metadata = GgufMetadata::from_values_for_test(values);
        let error = validate_whisper_execution_metadata(&metadata).expect_err("must fail");
        assert!(error.to_string().contains("whisper.encoder.block_count"));
    }
}
