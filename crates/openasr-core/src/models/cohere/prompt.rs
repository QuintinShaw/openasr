use crate::GgmlAsrExecutionOptions;

use super::tokenizer::CohereTranscribeTokenizer;
use thiserror::Error;

// This family declares `arch::SpeakerSegmentationSource::External`: its
// decoder does have a `<|diarize|>` / `<|spltoken0|>` speaker-token mode, but
// no published cohere pack can run it (the packs would have to be re-converted
// and re-published for it), so the runtime never asks for it and reports the
// capability as unsupported rather than half-promising it. The prompt
// therefore pins the plain-transcript control tokens unconditionally; when the
// in-decoder mode is actually enabled, this is where the switch comes back.
const COHERE_NO_DIARIZE_TOKEN: &str = "<|nodiarize|>";
const COHERE_NO_TIMESTAMP_TOKEN: &str = "<|notimestamp|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CohereTranscribeDecodePrompt {
    pub token_ids: Vec<u32>,
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Error)]
pub(crate) enum CohereTranscribeDecodePromptError {
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
    for token in [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        language_token.as_str(),
        language_token.as_str(),
        punctuation_token,
        "<|noitn|>",
        COHERE_NO_TIMESTAMP_TOKEN,
        COHERE_NO_DIARIZE_TOKEN,
    ] {
        if let Some(token_id) = tokenizer.token_id_by_content(token) {
            token_ids.push(token_id);
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

    /// The plain-transcript control tokens are unconditional: asking for
    /// in-decoder speakers must not change this family's prompt, because it
    /// does not offer that mode (see the constants' comment above).
    #[test]
    fn asking_for_in_decoder_speakers_does_not_change_the_prompt() {
        let tokenizer = tokenizer();
        let plain = build_cohere_transcribe_decode_prompt(
            &tokenizer,
            13764,
            Some("en"),
            &GgmlAsrExecutionOptions::default(),
        )
        .expect("prompt");
        let requested = build_cohere_transcribe_decode_prompt(
            &tokenizer,
            13764,
            Some("en"),
            &GgmlAsrExecutionOptions {
                in_decoder_speakers: true,
                ..GgmlAsrExecutionOptions::default()
            },
        )
        .expect("prompt");
        assert_eq!(plain.token_ids, requested.token_ids);
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
