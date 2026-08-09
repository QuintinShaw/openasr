//! Audio-embedding splice + prompt assembly, mirroring HF
//! `GraniteSpeechModel.get_merged_audio_embeddings` (the token-id side) and
//! `GraniteSpeechProcessor.__call__` (the placeholder-expansion side):
//!
//! 1. The processor expands the *single* literal `<|audio|>` token in the
//!    chat-formatted prompt text into `audio_embed_sizes[i]` (here: the
//!    Q-Former projector's token count) repeated copies of that same literal
//!    string, so tokenization produces exactly that many consecutive
//!    `audio_token_id` (`100352`) tokens.
//! 2. `get_merged_audio_embeddings` embeds every token id (audio positions
//!    embed as token id `0`, a dummy -- HF zeroes them before the embedding
//!    lookup specifically so they never index out of range) then
//!    `masked_scatter`s the projector's audio embeddings over those
//!    positions, in order. `GraniteModel.forward` applies
//!    `embedding_multiplier` to the whole assembled sequence *after* this
//!    splice, so the audio embeddings get scaled by it too --
//!    `decoder_graph::prefill_logits_from_embeddings` already does that
//!    scaling on whatever it's handed, so this module hands it the
//!    un-scaled splice.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;

use super::decoder_graph::GraniteSpeechDecoderError;
use super::decoder_graph::{GraniteSpeechDecoderConfig, embed_token_row};
use super::tokenizer::{GraniteSpeechTokenizer, GraniteSpeechTokenizerError};

pub(crate) const GRANITE_SPEECH_AUDIO_TOKEN: &str = "<|audio|>";
pub(crate) const GRANITE_SPEECH_AUDIO_TOKEN_ID: u32 = 100_352;
pub(crate) const GRANITE_SPEECH_DEFAULT_QUESTION: &str =
    "can you transcribe the speech into a written format?";

/// Family-owned prompt-text oracle shared by capacity planning and execution.
/// Keyword bias is part of the actual prompt topology, never an unmodelled
/// runtime suffix or a reason to disable the feature under admission.
pub(crate) fn build_granite_speech_prompt_text(
    phrase_bias: Option<&crate::PhraseBiasConfig>,
) -> String {
    let question = match phrase_bias.filter(|bias| !bias.is_empty()) {
        Some(phrase_bias) => {
            let keywords = phrase_bias
                .entries()
                .iter()
                .map(|entry| entry.phrase())
                .collect::<Vec<_>>()
                .join(", ");
            format!("transcribe the speech to text. Keywords: {keywords}")
        }
        None => GRANITE_SPEECH_DEFAULT_QUESTION.to_string(),
    };
    format!("USER: {GRANITE_SPEECH_AUDIO_TOKEN}{question}\n ASSISTANT:")
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GraniteSpeechPromptError {
    #[error(
        "granite-speech prompt text must contain exactly one '<|audio|>' placeholder, found {count}"
    )]
    PlaceholderCount { count: usize },
    #[error(
        "granite-speech prompt token count mismatch: tokenized text has {token_audio_count} '<|audio|>' \
         tokens, expected {expected_audio_tokens} (the projector's output token count)"
    )]
    AudioTokenCountMismatch {
        token_audio_count: usize,
        expected_audio_tokens: usize,
    },
    #[error("granite-speech prompt tokenizer error: {0}")]
    Tokenizer(#[from] GraniteSpeechTokenizerError),
    #[error("granite-speech prompt embedding lookup error: {0}")]
    Decoder(#[from] GraniteSpeechDecoderError),
    #[error("granite-speech mapped token embedding lookup failed: {reason}")]
    MappedTokenEmbedding { reason: String },
    #[error("granite-speech prompt audio embeddings shape error: {reason}")]
    Shape { reason: String },
}

/// Expands the single `<|audio|>` placeholder in `prompt_text` into
/// `audio_token_count` repeated copies (the processor's placeholder
/// expansion, step 1 in the module doc).
fn expand_audio_placeholder(
    prompt_text: &str,
    audio_token_count: usize,
) -> Result<String, GraniteSpeechPromptError> {
    let count = prompt_text.matches(GRANITE_SPEECH_AUDIO_TOKEN).count();
    if count != 1 {
        return Err(GraniteSpeechPromptError::PlaceholderCount { count });
    }
    Ok(prompt_text.replacen(
        GRANITE_SPEECH_AUDIO_TOKEN,
        &GRANITE_SPEECH_AUDIO_TOKEN.repeat(audio_token_count),
        1,
    ))
}

/// Exact prompt-token oracle. The runtime embedding builder delegates to this
/// same function, so capacity and execution cannot disagree about BPE merges,
/// keyword suffixes, or repeated audio placeholders.
pub(crate) fn build_audio_prompt_token_ids(
    tokenizer: &GraniteSpeechTokenizer,
    prompt_text: &str,
    audio_token_count: usize,
) -> Result<Vec<u32>, GraniteSpeechPromptError> {
    let expanded_text = expand_audio_placeholder(prompt_text, audio_token_count)?;
    let token_ids = tokenizer.encode_prompt_text(&expanded_text)?;
    let token_audio_count = token_ids
        .iter()
        .filter(|&&id| id == GRANITE_SPEECH_AUDIO_TOKEN_ID)
        .count();
    if token_audio_count != audio_token_count {
        return Err(GraniteSpeechPromptError::AudioTokenCountMismatch {
            token_audio_count,
            expected_audio_tokens: audio_token_count,
        });
    }
    Ok(token_ids)
}

/// Builds the full `[n_tokens, hidden_size]` embedding sequence for a prompt
/// containing one audio placeholder, ready for
/// `decoder_graph::prefill_logits_from_embeddings`. `audio_embeddings` is the
/// Q-Former projector's `[audio_token_count, hidden_size]` output (already
/// projected to the LLM's `hidden_size`, see `qformer::project`).
pub(crate) fn build_audio_prompt_embeddings(
    config: &GraniteSpeechDecoderConfig,
    provider: &HashMap<String, Vec<f32>>,
    tokenizer: &GraniteSpeechTokenizer,
    prompt_text: &str,
    audio_embeddings: &[f32],
    audio_token_count: usize,
) -> Result<(Vec<u32>, Vec<f32>), GraniteSpeechPromptError> {
    build_audio_prompt_embeddings_with(
        config,
        tokenizer,
        prompt_text,
        audio_embeddings,
        audio_token_count,
        |token_ids| {
            let mut rows = Vec::with_capacity(token_ids.len() * config.hidden_size);
            for &token_id in token_ids {
                rows.extend_from_slice(embed_token_row(config, provider, token_id)?);
            }
            Ok(rows)
        },
    )
}

pub(crate) fn build_audio_prompt_embeddings_from_mapped_table(
    config: &GraniteSpeechDecoderConfig,
    table: &MappedTokenEmbeddingTable,
    tokenizer: &GraniteSpeechTokenizer,
    prompt_text: &str,
    audio_embeddings: &[f32],
    audio_token_count: usize,
) -> Result<(Vec<u32>, Vec<f32>), GraniteSpeechPromptError> {
    build_audio_prompt_embeddings_with(
        config,
        tokenizer,
        prompt_text,
        audio_embeddings,
        audio_token_count,
        |token_ids| {
            table.gather_rows(token_ids).map_err(|error| {
                GraniteSpeechPromptError::MappedTokenEmbedding {
                    reason: error.to_string(),
                }
            })
        },
    )
}

fn build_audio_prompt_embeddings_with(
    config: &GraniteSpeechDecoderConfig,
    tokenizer: &GraniteSpeechTokenizer,
    prompt_text: &str,
    audio_embeddings: &[f32],
    audio_token_count: usize,
    gather_text_rows: impl FnOnce(&[u32]) -> Result<Vec<f32>, GraniteSpeechPromptError>,
) -> Result<(Vec<u32>, Vec<f32>), GraniteSpeechPromptError> {
    if audio_embeddings.len() != audio_token_count * config.hidden_size {
        return Err(GraniteSpeechPromptError::Shape {
            reason: format!(
                "audio_embeddings has {} values, expected {audio_token_count}x{}",
                audio_embeddings.len(),
                config.hidden_size
            ),
        });
    }

    let token_ids = build_audio_prompt_token_ids(tokenizer, prompt_text, audio_token_count)?;
    let text_token_ids = token_ids
        .iter()
        .copied()
        .filter(|token_id| *token_id != GRANITE_SPEECH_AUDIO_TOKEN_ID)
        .collect::<Vec<_>>();
    let text_rows = gather_text_rows(&text_token_ids)?;
    let expected_text_values = text_token_ids
        .len()
        .checked_mul(config.hidden_size)
        .ok_or_else(|| GraniteSpeechPromptError::Shape {
            reason: "text embedding row count overflowed".to_string(),
        })?;
    if text_rows.len() != expected_text_values {
        return Err(GraniteSpeechPromptError::Shape {
            reason: format!(
                "text embedding gather returned {} values, expected {expected_text_values}",
                text_rows.len(),
            ),
        });
    }

    let mut embeddings = Vec::with_capacity(token_ids.len() * config.hidden_size);
    let mut next_audio_slot = 0usize;
    let mut next_text_slot = 0usize;
    for &token_id in &token_ids {
        if token_id == GRANITE_SPEECH_AUDIO_TOKEN_ID {
            let start = next_audio_slot * config.hidden_size;
            embeddings.extend_from_slice(&audio_embeddings[start..start + config.hidden_size]);
            next_audio_slot += 1;
        } else {
            let start = next_text_slot * config.hidden_size;
            embeddings.extend_from_slice(&text_rows[start..start + config.hidden_size]);
            next_text_slot += 1;
        }
    }

    Ok((token_ids, embeddings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_audio_placeholder_replaces_single_marker() {
        let text = format!("USER: {GRANITE_SPEECH_AUDIO_TOKEN}transcribe this\n ASSISTANT:");
        let expanded = expand_audio_placeholder(&text, 3).expect("expand");
        assert_eq!(
            expanded,
            format!(
                "USER: {a}{a}{a}transcribe this\n ASSISTANT:",
                a = GRANITE_SPEECH_AUDIO_TOKEN
            )
        );
    }

    #[test]
    fn expand_audio_placeholder_rejects_wrong_count() {
        let text = "no placeholder here";
        assert!(matches!(
            expand_audio_placeholder(text, 3),
            Err(GraniteSpeechPromptError::PlaceholderCount { count: 0 })
        ));
        let two = format!("{a}{a}", a = GRANITE_SPEECH_AUDIO_TOKEN);
        assert!(matches!(
            expand_audio_placeholder(&two, 3),
            Err(GraniteSpeechPromptError::PlaceholderCount { count: 2 })
        ));
    }

    #[test]
    fn prompt_text_oracle_preserves_the_default_and_keyword_shapes() {
        let default = build_granite_speech_prompt_text(None);
        assert!(default.contains(GRANITE_SPEECH_DEFAULT_QUESTION));
        assert_eq!(default.matches(GRANITE_SPEECH_AUDIO_TOKEN).count(), 1);

        let phrase_bias =
            crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0), ("Granite", 2.0)])
                .expect("valid phrase bias");
        let biased = build_granite_speech_prompt_text(Some(&phrase_bias));
        assert!(biased.contains("Keywords: OpenASR, Granite"));
        assert_eq!(biased.matches(GRANITE_SPEECH_AUDIO_TOKEN).count(), 1);
    }
}
