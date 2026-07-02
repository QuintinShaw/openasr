use thiserror::Error;

use crate::NativeAsrError;
use crate::arch::OpenAsrArchitectureRegistry;
use crate::ggml_runtime::GgufMetadata;

use super::cohere::CohereTranscribeTokenizer;
use super::qwen::Qwen3AsrTokenizer;
use super::whisper::WhisperTokenizer;

#[derive(Debug, Clone)]
pub(crate) enum BuiltinTokenizerComponent {
    CohereTranscribe(CohereTranscribeTokenizer),
    Qwen3Asr(Qwen3AsrTokenizer),
    Whisper(WhisperTokenizer),
}

impl BuiltinTokenizerComponent {
    pub(crate) fn into_cohere_transcribe(self) -> Option<CohereTranscribeTokenizer> {
        match self {
            Self::CohereTranscribe(tokenizer) => Some(tokenizer),
            _ => None,
        }
    }

    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrTokenizer> {
        match self {
            Self::Qwen3Asr(tokenizer) => Some(tokenizer),
            _ => None,
        }
    }

    pub(crate) fn into_whisper(self) -> Option<WhisperTokenizer> {
        match self {
            Self::Whisper(tokenizer) => Some(tokenizer),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum BuiltinTokenizerComponentRegistryError {
    #[error("unknown builtin model architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("unknown builtin tokenizer '{tokenizer_id}'")]
    UnknownTokenizer { tokenizer_id: String },
    #[error("builtin tokenizer '{tokenizer_id}' materialization failed: {source}")]
    MaterializationFailed {
        tokenizer_id: String,
        #[source]
        source: NativeAsrError,
    },
}

pub(crate) fn materialize_builtin_tokenizer_for_architecture(
    model_architecture: &str,
    metadata: &GgufMetadata,
) -> Result<BuiltinTokenizerComponent, BuiltinTokenizerComponentRegistryError> {
    let descriptor = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .ok_or_else(
            || BuiltinTokenizerComponentRegistryError::UnknownArchitecture {
                model_architecture: model_architecture.to_string(),
            },
        )?;
    materialize_builtin_tokenizer(descriptor.tokenizer_id, metadata)
}

pub(crate) fn materialize_builtin_tokenizer(
    tokenizer_id: &str,
    metadata: &GgufMetadata,
) -> Result<BuiltinTokenizerComponent, BuiltinTokenizerComponentRegistryError> {
    match tokenizer_id {
        crate::COHERE_TRANSCRIBE_TOKENIZER_ID => {
            CohereTranscribeTokenizer::from_gguf_metadata(metadata)
                .map(BuiltinTokenizerComponent::CohereTranscribe)
                .map_err(
                    |source| BuiltinTokenizerComponentRegistryError::MaterializationFailed {
                        tokenizer_id: tokenizer_id.to_string(),
                        source,
                    },
                )
        }
        crate::QWEN3_ASR_TOKENIZER_ID => Qwen3AsrTokenizer::from_gguf_metadata(metadata)
            .map(BuiltinTokenizerComponent::Qwen3Asr)
            .map_err(
                |source| BuiltinTokenizerComponentRegistryError::MaterializationFailed {
                    tokenizer_id: tokenizer_id.to_string(),
                    source,
                },
            ),
        crate::WHISPER_TOKENIZER_ID => WhisperTokenizer::from_gguf_metadata(metadata)
            .map(BuiltinTokenizerComponent::Whisper)
            .map_err(
                |source| BuiltinTokenizerComponentRegistryError::MaterializationFailed {
                    tokenizer_id: tokenizer_id.to_string(),
                    source,
                },
            ),
        _ => Err(BuiltinTokenizerComponentRegistryError::UnknownTokenizer {
            tokenizer_id: tokenizer_id.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::ggml_runtime::GgufMetadataValue;

    fn cohere_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|startoftranscript|>".to_string(),
                "<|en|>".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        GgufMetadata::from_values_for_test(values)
    }

    fn qwen_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("gpt2".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|endoftext|>".to_string(),
                "hello".to_string(),
                "<|audio_start|>".to_string(),
                "<|audio_end|>".to_string(),
                "<|audio_pad|>".to_string(),
                "world".to_string(),
                "<|pad|>".to_string(),
            ]),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::StringArray(vec!["h e".to_string()]),
        );
        values.insert(
            "qwen3-asr.llm.vocab_size".to_string(),
            GgufMetadataValue::String("7".to_string()),
        );
        values.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            GgufMetadataValue::String("2".to_string()),
        );
        values.insert(
            "qwen3-asr.audio_end_token_id".to_string(),
            GgufMetadataValue::String("3".to_string()),
        );
        values.insert(
            "qwen3-asr.audio_pad_token_id".to_string(),
            GgufMetadataValue::String("4".to_string()),
        );
        values.insert(
            "qwen3-asr.eos_token_id".to_string(),
            GgufMetadataValue::String("0".to_string()),
        );
        values.insert(
            "qwen3-asr.pad_token_id".to_string(),
            GgufMetadataValue::String("6".to_string()),
        );
        GgufMetadata::from_values_for_test(values)
    }

    fn whisper_metadata() -> GgufMetadata {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("gpt2".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|endoftext|>".to_string(),
                "<|startoftranscript|>".to_string(),
                "<|transcribe|>".to_string(),
                "<|notimestamps|>".to_string(),
                " hello".to_string(),
            ]),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::StringArray(vec!["h e".to_string()]),
        );
        values.insert(
            "tokenizer.ggml.special_token_ids".to_string(),
            GgufMetadataValue::U32Array(vec![0, 1, 2, 3]),
        );
        values.insert(
            "tokenizer.ggml.sot_token_id".to_string(),
            GgufMetadataValue::String("1".to_string()),
        );
        values.insert(
            "tokenizer.ggml.eot_token_id".to_string(),
            GgufMetadataValue::String("0".to_string()),
        );
        values.insert(
            "tokenizer.ggml.transcribe_token_id".to_string(),
            GgufMetadataValue::String("2".to_string()),
        );
        values.insert(
            "tokenizer.ggml.no_timestamps_token_id".to_string(),
            GgufMetadataValue::String("3".to_string()),
        );
        values.insert(
            "whisper.decoder.vocab_size".to_string(),
            GgufMetadataValue::String("5".to_string()),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn materializes_builtin_cohere_tokenizer_for_architecture() {
        let tokenizer = materialize_builtin_tokenizer_for_architecture(
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            &cohere_metadata(),
        )
        .expect("cohere tokenizer")
        .into_cohere_transcribe()
        .expect("cohere variant");
        assert_eq!(tokenizer.token_id_by_content("<|en|>"), Some(1));
    }

    #[test]
    fn materializes_builtin_qwen_tokenizer_for_architecture() {
        let tokenizer = materialize_builtin_tokenizer_for_architecture(
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            &qwen_metadata(),
        )
        .expect("qwen tokenizer")
        .into_qwen3_asr()
        .expect("qwen variant");
        assert_eq!(tokenizer.audio_prompt_token_triplet(), (2, 3, 4));
        assert_eq!(tokenizer.eos_token_id, 0);
    }

    #[test]
    fn materializes_builtin_whisper_tokenizer_for_architecture() {
        let tokenizer = materialize_builtin_tokenizer_for_architecture(
            crate::WHISPER_GGML_ARCHITECTURE_ID,
            &whisper_metadata(),
        )
        .expect("whisper tokenizer")
        .into_whisper()
        .expect("whisper variant");
        assert_eq!(tokenizer.start_of_transcript_token_id(), Some(1));
        assert_eq!(tokenizer.end_of_text_token_id(), Some(0));
    }

    #[test]
    fn rejects_unknown_builtin_architecture() {
        let error = materialize_builtin_tokenizer_for_architecture(
            "not-a-builtin-arch",
            &cohere_metadata(),
        )
        .expect_err("unknown architecture must fail");
        assert!(matches!(
            error,
            BuiltinTokenizerComponentRegistryError::UnknownArchitecture { .. }
        ));
    }
}
