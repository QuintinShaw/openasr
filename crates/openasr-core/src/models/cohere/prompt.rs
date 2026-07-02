use crate::GgmlAsrExecutionOptions;

use super::tokenizer::CohereTranscribeTokenizer;
use thiserror::Error;

const COHERE_DIARIZE_TOKEN: &str = "<|diarize|>";
const COHERE_NO_DIARIZE_TOKEN: &str = "<|nodiarize|>";
const COHERE_TIMESTAMP_TOKEN: &str = "<|timestamp|>";
const COHERE_NO_TIMESTAMP_TOKEN: &str = "<|notimestamp|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CohereTranscribeDecodePrompt {
    pub token_ids: Vec<u32>,
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereTranscribeDecodePromptError {
    #[error(
        "cohere decode prompt requires token '{token}' for diarization but the tokenizer does not contain it"
    )]
    MissingRequiredControlToken { token: &'static str },
    #[error(
        "cohere decode prompt requested language '{language}' but the tokenizer has no '<|{language}|>' token"
    )]
    UnsupportedLanguage { language: String },
}

pub(crate) fn build_cohere_transcribe_decode_prompt(
    tokenizer: &CohereTranscribeTokenizer,
    _decoder_start_token_id: u32,
    language: Option<&str>,
    options: &GgmlAsrExecutionOptions,
) -> Result<CohereTranscribeDecodePrompt, CohereTranscribeDecodePromptError> {
    let mut token_ids = Vec::with_capacity(9);
    let requested_language = language
        .or(options.language.as_deref())
        .unwrap_or("en")
        .trim()
        .to_lowercase();
    let language_token = format!("<|{}|>", requested_language);
    // Fail closed when the pack vocab has no control token for the requested
    // language, instead of silently dropping it and transcribing in a different
    // language than the caller asked for.
    if tokenizer.token_id_by_content(&language_token).is_none() {
        return Err(CohereTranscribeDecodePromptError::UnsupportedLanguage {
            language: requested_language,
        });
    }
    let punctuation_token = if options
        .prompt
        .as_deref()
        .is_some_and(|prompt| prompt.contains("<|nopnc|>"))
    {
        "<|nopnc|>"
    } else {
        "<|pnc|>"
    };
    let diarization_token = if options.diarize {
        COHERE_DIARIZE_TOKEN
    } else {
        COHERE_NO_DIARIZE_TOKEN
    };
    let timestamp_token = if options.diarize {
        COHERE_TIMESTAMP_TOKEN
    } else {
        COHERE_NO_TIMESTAMP_TOKEN
    };

    for token in [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        language_token.as_str(),
        language_token.as_str(),
        punctuation_token,
        "<|noitn|>",
        timestamp_token,
        diarization_token,
    ] {
        if let Some(token_id) = tokenizer.token_id_by_content(token) {
            token_ids.push(token_id);
        } else if options.diarize && token == COHERE_TIMESTAMP_TOKEN {
            return Err(
                CohereTranscribeDecodePromptError::MissingRequiredControlToken {
                    token: COHERE_TIMESTAMP_TOKEN,
                },
            );
        } else if options.diarize && token == COHERE_DIARIZE_TOKEN {
            return Err(
                CohereTranscribeDecodePromptError::MissingRequiredControlToken {
                    token: COHERE_DIARIZE_TOKEN,
                },
            );
        }
    }

    Ok(CohereTranscribeDecodePrompt {
        token_ids,
        eos_token_id: tokenizer.token_id_by_content("<|endoftext|>"),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::ggml_runtime::{GgufMetadata, GgufMetadataValue};

    use super::*;

    fn tokenizer() -> CohereTranscribeTokenizer {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "<|startofcontext|>".to_string(),
                "<|startoftranscript|>".to_string(),
                "<|emo:undefined|>".to_string(),
                "<|en|>".to_string(),
                "<|pnc|>".to_string(),
                "<|noitn|>".to_string(),
                "<|notimestamp|>".to_string(),
                "<|timestamp|>".to_string(),
                "<|nodiarize|>".to_string(),
                "<|diarize|>".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        CohereTranscribeTokenizer::from_gguf_metadata(&GgufMetadata::from_values_for_test(values))
            .expect("tokenizer")
    }

    #[test]
    fn builds_default_prompt_with_language_and_eos() {
        let tokenizer = tokenizer();
        let prompt = build_cohere_transcribe_decode_prompt(
            &tokenizer,
            13764,
            Some("en"),
            &GgmlAsrExecutionOptions::default(),
        )
        .expect("prompt");
        assert_eq!(prompt.token_ids, vec![0, 1, 2, 3, 3, 4, 5, 6, 8]);
        assert_eq!(prompt.eos_token_id, Some(10));
    }

    #[test]
    fn builds_diarization_prompt_when_requested() {
        let tokenizer = tokenizer();
        let options = GgmlAsrExecutionOptions {
            diarize: true,
            ..GgmlAsrExecutionOptions::default()
        };
        let prompt = build_cohere_transcribe_decode_prompt(&tokenizer, 13764, Some("en"), &options)
            .expect("prompt");
        assert_eq!(prompt.token_ids, vec![0, 1, 2, 3, 3, 4, 5, 7, 9]);
        assert_eq!(prompt.eos_token_id, Some(10));
    }

    #[test]
    fn rejects_diarization_prompt_when_required_token_is_missing() {
        let tokenizer = CohereTranscribeTokenizer::from_gguf_metadata(&{
            let mut values = BTreeMap::new();
            values.insert(
                "tokenizer.ggml.model".to_string(),
                GgufMetadataValue::String("llama".to_string()),
            );
            values.insert(
                "tokenizer.ggml.tokens".to_string(),
                GgufMetadataValue::StringArray(vec![
                    "<|startofcontext|>".to_string(),
                    "<|startoftranscript|>".to_string(),
                    "<|emo:undefined|>".to_string(),
                    "<|en|>".to_string(),
                    "<|pnc|>".to_string(),
                    "<|noitn|>".to_string(),
                    "<|notimestamp|>".to_string(),
                    "<|timestamp|>".to_string(),
                    "<|nodiarize|>".to_string(),
                    "<|endoftext|>".to_string(),
                ]),
            );
            GgufMetadata::from_values_for_test(values)
        })
        .expect("tokenizer");
        let options = GgmlAsrExecutionOptions {
            diarize: true,
            ..GgmlAsrExecutionOptions::default()
        };
        let error = build_cohere_transcribe_decode_prompt(&tokenizer, 13764, Some("en"), &options)
            .expect_err("missing diarization token must fail closed")
            .to_string();
        assert!(error.contains("<|diarize|>"), "{error}");
    }

    #[test]
    fn rejects_unsupported_language_when_token_is_missing() {
        let tokenizer = tokenizer();
        let error = build_cohere_transcribe_decode_prompt(
            &tokenizer,
            13764,
            Some("fr"),
            &GgmlAsrExecutionOptions::default(),
        )
        .expect_err("a language without a control token must fail closed")
        .to_string();
        assert!(error.contains("fr"), "{error}");
    }
}
