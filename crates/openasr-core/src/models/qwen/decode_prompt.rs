use thiserror::Error;

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tokenizer::Qwen3AsrTokenizer;
use crate::models::ggml_asr_executor::GgmlAsrExecutionOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3AsrDecodePrompt {
    pub token_ids: Vec<u32>,
    pub audio_pad_start_index: usize,
    pub audio_pad_count: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum Qwen3AsrDecodePromptError {
    #[error("qwen3-asr decode prompt requires at least one audio frame token")]
    EmptyAudioFrames,
    #[error(
        "qwen3-asr decode prompt currently does not support request option '{option}': {reason}"
    )]
    UnsupportedRequestOption {
        option: &'static str,
        reason: &'static str,
    },
}

pub(crate) fn build_qwen3_decode_prompt(
    metadata: Qwen3AsrExecutionMetadata,
    tokenizer: Option<&Qwen3AsrTokenizer>,
    audio_frame_count: usize,
    request_options: &GgmlAsrExecutionOptions,
) -> Result<Qwen3AsrDecodePrompt, Qwen3AsrDecodePromptError> {
    if audio_frame_count == 0 {
        return Err(Qwen3AsrDecodePromptError::EmptyAudioFrames);
    }
    if request_options
        .language
        .as_deref()
        .is_some_and(|language| !language.trim().is_empty())
    {
        // Defense in depth: the post-family gate already rejects an explicit
        // language for Qwen3-ASR. To wire it here later, inject the language as
        // chat-prompt text (Qwen3-ASR has no language tokens in its vocab) and
        // verify the prompt against the reference inference on a real pack before
        // claiming support -- otherwise a wrong format would silently no-op.
        return Err(Qwen3AsrDecodePromptError::UnsupportedRequestOption {
            option: "language",
            reason: "Qwen3-ASR auto-detects the source language and does not accept an explicit selection; use a multilingual Whisper pack to force or report a language.",
        });
    }
    let (audio_start_token_id, audio_end_token_id, audio_pad_token_id) = tokenizer
        .map(Qwen3AsrTokenizer::audio_prompt_token_triplet)
        .unwrap_or((
            metadata.audio_start_token_id,
            metadata.audio_end_token_id,
            metadata.audio_pad_token_id,
        ));

    let normalized_prompt = request_options
        .prompt
        .as_deref()
        .map(str::trim)
        .unwrap_or("");
    if let Some(tokenizer) = tokenizer {
        let prefix = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
        let mut suffix = String::from("<|audio_end|>");
        if !normalized_prompt.is_empty() {
            suffix.push('\n');
            suffix.push_str(normalized_prompt);
        }
        suffix.push_str("<|im_end|>\n<|im_start|>assistant\n");
        let prefix_ids = tokenizer.encode_prompt_text(prefix).map_err(|_| {
            Qwen3AsrDecodePromptError::UnsupportedRequestOption {
                option: "prompt",
                reason: "ChatML prompt tokenization failed",
            }
        })?;
        let suffix_ids = tokenizer.encode_prompt_text(&suffix).map_err(|_| {
            Qwen3AsrDecodePromptError::UnsupportedRequestOption {
                option: "prompt",
                reason: "ChatML prompt tokenization failed",
            }
        })?;
        let mut token_ids = Vec::with_capacity(
            prefix_ids
                .len()
                .saturating_add(audio_frame_count)
                .saturating_add(suffix_ids.len()),
        );
        token_ids.extend(prefix_ids.iter().copied());
        let audio_pad_start_index = token_ids.len();
        token_ids.extend(std::iter::repeat_n(audio_pad_token_id, audio_frame_count));
        token_ids.extend(suffix_ids.iter().copied());
        return Ok(Qwen3AsrDecodePrompt {
            token_ids,
            audio_pad_start_index,
            audio_pad_count: audio_frame_count,
        });
    }

    if !normalized_prompt.is_empty() {
        return Err(Qwen3AsrDecodePromptError::UnsupportedRequestOption {
            option: "prompt",
            reason: "text prompt tokenization requires GGUF tokenizer metadata",
        });
    }

    let mut token_ids = Vec::with_capacity(audio_frame_count.saturating_add(2));
    token_ids.push(audio_start_token_id);
    token_ids.extend(std::iter::repeat_n(audio_pad_token_id, audio_frame_count));
    token_ids.push(audio_end_token_id);

    Ok(Qwen3AsrDecodePrompt {
        token_ids,
        audio_pad_start_index: 1,
        audio_pad_count: audio_frame_count,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::models::ggml_asr_executor::GgmlAsrExecutionOptions;
    use crate::{GgufMetadata, GgufMetadataValue};

    use super::super::runtime_contract::*;
    use super::*;

    fn metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 18,
            audio_d_model: 896,
            audio_heads: 14,
            llm_layers: 28,
            llm_d_model: 1024,
            llm_heads: 16,
            llm_kv_heads: 8,
            llm_head_dim: 128,
            vocab_size: 151_936,
            llm_max_positions: 40_960,
            audio_start_token_id: 11,
            audio_end_token_id: 12,
            audio_pad_token_id: 13,
            eos_token_id: 14,
            pad_token_id: 15,
        }
    }

    fn tokenizer_fixture() -> Qwen3AsrTokenizer {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("gpt2".to_string()),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec![
                "s".to_string(),
                "y".to_string(),
                "t".to_string(),
                "e".to_string(),
                "m".to_string(),
                "u".to_string(),
                "r".to_string(),
                "a".to_string(),
                "i".to_string(),
                "n".to_string(),
                "\u{010A}".to_string(),
                "sy".to_string(),
                "sys".to_string(),
                "syst".to_string(),
                "syste".to_string(),
                "us".to_string(),
                "use".to_string(),
                "ser".to_string(),
                "as".to_string(),
                "ass".to_string(),
                "assi".to_string(),
                "assis".to_string(),
                "assist".to_string(),
                "assista".to_string(),
                "assistan".to_string(),
                "assistant".to_string(),
                "<|im_start|>".to_string(),
                "system".to_string(),
                "<|im_end|>".to_string(),
                "<|audio_start|>".to_string(),
                "user".to_string(),
                "<|audio_end|>".to_string(),
                "<|audio_pad|>".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::StringArray(vec![
                "s y".to_string(),
                "sy s".to_string(),
                "sys t".to_string(),
                "syst e".to_string(),
                "syste m".to_string(),
                "u s".to_string(),
                "us e".to_string(),
                "use r".to_string(),
                "a s".to_string(),
                "as s".to_string(),
                "ass i".to_string(),
                "assi s".to_string(),
                "assis t".to_string(),
                "assist a".to_string(),
                "assista n".to_string(),
                "assistan t".to_string(),
            ]),
        );
        values.insert(
            QWEN3_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(34),
        );
        values.insert(
            QWEN3_AUDIO_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(29),
        );
        values.insert(
            QWEN3_AUDIO_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(31),
        );
        values.insert(
            QWEN3_AUDIO_PAD_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(32),
        );
        values.insert(
            QWEN3_EOS_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(28),
        );
        values.insert(
            QWEN3_PAD_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(33),
        );
        Qwen3AsrTokenizer::from_gguf_metadata(&GgufMetadata::from_values_for_test(values))
            .expect("tokenizer")
    }

    #[test]
    fn build_decode_prompt_uses_tokenizer_when_available() {
        let tokenizer = tokenizer_fixture();
        let prompt = build_qwen3_decode_prompt(
            metadata(),
            Some(&tokenizer),
            3,
            &GgmlAsrExecutionOptions::default(),
        )
        .expect("prompt");
        assert_eq!(
            prompt.token_ids,
            vec![
                26, 27, 10, 28, 10, 26, 30, 10, 29, 32, 32, 32, 31, 28, 10, 26, 25, 10
            ]
        );
        assert_eq!(prompt.audio_pad_start_index, 9);
        assert_eq!(prompt.audio_pad_count, 3);
    }

    #[test]
    fn build_decode_prompt_falls_back_to_runtime_metadata_tokens() {
        let prompt =
            build_qwen3_decode_prompt(metadata(), None, 2, &GgmlAsrExecutionOptions::default())
                .expect("prompt");
        assert_eq!(prompt.token_ids, vec![11, 13, 13, 12]);
    }

    #[test]
    fn build_decode_prompt_rejects_non_empty_prompt_without_tokenizer_metadata() {
        let options = GgmlAsrExecutionOptions {
            prompt: Some("hello".to_string()),
            ..GgmlAsrExecutionOptions::default()
        };
        let error = build_qwen3_decode_prompt(metadata(), None, 2, &options)
            .expect_err("must fail")
            .to_string();
        assert!(error.contains("request option 'prompt'"), "{error}");
    }

    #[test]
    fn build_decode_prompt_rejects_explicit_language() {
        // Qwen3-ASR auto-detects the language and does not accept an explicit
        // selection; the prompt builder fails closed instead of silently ignoring
        // the hint and transcribing in some other language.
        let tokenizer = tokenizer_fixture();
        let options = GgmlAsrExecutionOptions {
            language: Some("fr".to_string()),
            ..GgmlAsrExecutionOptions::default()
        };
        let error = build_qwen3_decode_prompt(metadata(), Some(&tokenizer), 2, &options)
            .expect_err("an explicit language must be rejected, not silently ignored")
            .to_string();
        assert!(error.contains("language"), "{error}");
        assert!(error.contains("Whisper"), "{error}");
    }
}
