//! Whisper prompt topology shared by execution and decoder-state planning.
//!
//! Keeping tokenization, wrapper ordering, long-form tail trimming, and the
//! stable carry bound in one oracle prevents the runtime prompt from growing
//! beyond the resident self-KV span selected before allocation.

use thiserror::Error;

use crate::GgmlAsrExecutionOptions;
use crate::models::decode_token_history::trim_prompt_token_tail;

use super::WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT;
use super::ggml_executor::WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE;
use super::runtime_contract::WhisperGgmlExecutionMetadata;
use super::tokenizer::{WhisperPrefixError, WhisperPrefixSpec, WhisperTokenizer};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum WhisperPromptError {
    #[error("this whisper pack has no <|{language}|> language token")]
    LanguageTokenMissing { language: String },
    #[error("this whisper pack has no <|translate|> task token")]
    TranslateTokenMissing,
    #[error("whisper tokenizer returned an empty decoder prefix")]
    EmptyDecoderPrefix,
    #[error("could not encode whisper request prompt: {reason}")]
    PromptEncodingFailed { reason: String },
    #[error(
        "whisper prompt prefix length {prompt_positions} leaves no generation budget in decoder context {decoder_position_cap}"
    )]
    PromptExhaustsContext {
        prompt_positions: usize,
        decoder_position_cap: usize,
    },
    #[error("whisper stable prompt position arithmetic overflowed")]
    PositionOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperPromptPositionBounds {
    pub(crate) logical: usize,
    pub(crate) stable: usize,
}

fn map_prefix_error(error: WhisperPrefixError) -> WhisperPromptError {
    match error {
        WhisperPrefixError::LanguageTokenMissing { language } => {
            WhisperPromptError::LanguageTokenMissing { language }
        }
        WhisperPrefixError::TranslateTokenMissing => WhisperPromptError::TranslateTokenMissing,
    }
}

/// Build the exact token sequence consumed by decoder prefill.
pub(crate) fn build_whisper_initial_prompt_tokens(
    execution: &WhisperGgmlExecutionMetadata,
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
    // A language detected by LID takes precedence over the request language.
    override_language: Option<&str>,
) -> Result<Vec<u32>, WhisperPromptError> {
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix_spec = WhisperPrefixSpec {
        language: override_language.or(request_options.language.as_deref()),
        task: request_options.task,
        is_multilingual: execution.vocab_size > WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE,
    };
    let prompt_init_tokens = tokenizer
        .decoder_prefix(decoder_start_token_id, &prefix_spec)
        .map_err(map_prefix_error)?;
    if prompt_init_tokens.is_empty() {
        return Err(WhisperPromptError::EmptyDecoderPrefix);
    }

    let mut prompt_tokens = if let Some(token_ids) = request_options.prompt_token_ids.as_ref() {
        token_ids.clone()
    } else {
        let Some(prompt) = request_options.prompt.as_deref().map(str::trim) else {
            return Ok(prompt_init_tokens);
        };
        if prompt.is_empty() {
            return Ok(prompt_init_tokens);
        }
        tokenizer.encode_prompt_text(prompt).map_err(|error| {
            WhisperPromptError::PromptEncodingFailed {
                reason: error.to_string(),
            }
        })?
    };
    if prompt_tokens.is_empty() {
        return Ok(prompt_init_tokens);
    }

    let longform_enabled = request_options.longform_mode_enabled();
    let prev_token = longform_enabled
        .then(|| tokenizer.token_id_by_content("<|startofprev|>"))
        .flatten();
    let max_prompt_tokens = execution
        .max_target_positions
        .saturating_sub(prompt_init_tokens.len())
        .saturating_sub(usize::from(prev_token.is_some()))
        .saturating_sub(1);
    if max_prompt_tokens == 0 {
        return Err(WhisperPromptError::PromptExhaustsContext {
            prompt_positions: prompt_init_tokens.len(),
            decoder_position_cap: execution.max_target_positions,
        });
    }
    prompt_tokens = trim_prompt_token_tail(
        prompt_tokens,
        max_prompt_tokens,
        longform_enabled,
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT,
    );

    let mut initial_prompt_tokens = Vec::with_capacity(
        prompt_init_tokens.len() + prompt_tokens.len() + usize::from(prev_token.is_some()),
    );
    if let Some(prev_token) = prev_token {
        initial_prompt_tokens.push(prev_token);
        initial_prompt_tokens.extend(prompt_tokens);
        initial_prompt_tokens.extend(prompt_init_tokens);
    } else {
        initial_prompt_tokens.extend(prompt_init_tokens);
        initial_prompt_tokens.extend(prompt_tokens);
    }
    Ok(initial_prompt_tokens)
}

/// Exact current prompt positions plus the smallest proven upper bound for
/// every prompt/carry token sequence admitted by the session envelope.
pub(crate) fn whisper_prompt_position_bounds(
    execution: &WhisperGgmlExecutionMetadata,
    tokenizer: &WhisperTokenizer,
    request_options: &GgmlAsrExecutionOptions,
    max_variable_prompt_tokens: usize,
) -> Result<WhisperPromptPositionBounds, WhisperPromptError> {
    let current_prompt_positions =
        build_whisper_initial_prompt_tokens(execution, tokenizer, request_options, None)?.len();
    // LID runs after session allocation. For a malformed-but-readable
    // multilingual pack that omits `<|en|>` while retaining another language
    // token, the detected-language prefix can be one position wider than the
    // unset prefix. Compare the only two possible prefix widths now so the
    // arena remains the unique minimum safe bound for either runtime branch.
    let logical = if request_options
        .language
        .as_deref()
        .map(str::trim)
        .filter(|language| !language.is_empty())
        .is_none()
        && execution.vocab_size > super::ggml_executor::WHISPER_ENGLISH_ONLY_MAX_VOCAB_SIZE
    {
        tokenizer
            .first_present_language_code()
            .map(|language| {
                build_whisper_initial_prompt_tokens(
                    execution,
                    tokenizer,
                    request_options,
                    Some(language),
                )
                .map(|tokens| tokens.len().max(current_prompt_positions))
            })
            .transpose()?
            .unwrap_or(current_prompt_positions)
    } else {
        current_prompt_positions
    };
    if max_variable_prompt_tokens == 0 || !request_options.longform_prompt_carry_enabled() {
        return Ok(WhisperPromptPositionBounds {
            logical,
            stable: logical,
        });
    }

    let mut base_options = request_options.clone();
    base_options.prompt = None;
    base_options.prompt_token_ids = None;
    let base_positions =
        build_whisper_initial_prompt_tokens(execution, tokenizer, &base_options, None)?.len();
    let marker_positions = usize::from(
        request_options.longform_mode_enabled()
            && tokenizer.token_id_by_content("<|startofprev|>").is_some(),
    );
    let context_tail_cap = execution
        .max_target_positions
        .checked_sub(base_positions)
        .and_then(|positions| positions.checked_sub(marker_positions))
        .and_then(|positions| positions.checked_sub(1))
        .ok_or(WhisperPromptError::PromptExhaustsContext {
            prompt_positions: base_positions + marker_positions,
            decoder_position_cap: execution.max_target_positions,
        })?;
    let family_tail_cap = if request_options.longform_mode_enabled() {
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT
    } else {
        usize::MAX
    };
    let variable_positions = max_variable_prompt_tokens
        .min(family_tail_cap)
        .min(context_tail_cap);
    let stable = base_positions
        .checked_add(marker_positions)
        .and_then(|positions| positions.checked_add(variable_positions))
        .map(|positions| positions.max(logical))
        .ok_or(WhisperPromptError::PositionOverflow)?;
    Ok(WhisperPromptPositionBounds { logical, stable })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execution_metadata() -> WhisperGgmlExecutionMetadata {
        WhisperGgmlExecutionMetadata {
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_hidden_size: 16,
            encoder_attention_heads: 2,
            encoder_context_length: 1_500,
            decoder_attention_heads: 2,
            max_target_positions: 448,
            decoder_hidden_size: 16,
            vocab_size: 51_865,
            decoder_start_token_id: 60,
            eos_token_id: 61,
            encoder_mels_count: 80,
        }
    }

    fn multilingual_tokenizer_without_english() -> WhisperTokenizer {
        WhisperTokenizer::from_tokenizer_payload_bytes(
            br#"{
                "version":"1.0",
                "added_tokens":[
                    {"id":60,"content":"<|startoftranscript|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":61,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":62,"content":"<|fr|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":63,"content":"<|transcribe|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":64,"content":"<|notimestamps|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
                ],
                "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0},"merges":[]}
            }"#,
        )
        .expect("test tokenizer")
    }

    #[test]
    fn auto_language_detection_width_is_included_before_arena_allocation() {
        let tokenizer = multilingual_tokenizer_without_english();
        let options = GgmlAsrExecutionOptions::default();
        assert_eq!(
            build_whisper_initial_prompt_tokens(&execution_metadata(), &tokenizer, &options, None,)
                .unwrap()
                .len(),
            3,
            "the malformed pack's unset prefix has no language token"
        );
        assert_eq!(
            whisper_prompt_position_bounds(&execution_metadata(), &tokenizer, &options, 0,)
                .unwrap(),
            WhisperPromptPositionBounds {
                logical: 4,
                stable: 4,
            },
            "planning must cover the wider post-LID prefix"
        );
    }
}
