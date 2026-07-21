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

use super::decoder_graph::GraniteSpeechDecoderError;
use super::decoder_graph::{
    GraniteSpeechDecoderConfig, GraniteSpeechDecoderWeightProvider, embed_token_row,
};
use super::tokenizer::{GraniteSpeechTokenizer, GraniteSpeechTokenizerError};

pub(crate) const GRANITE_SPEECH_AUDIO_TOKEN: &str = "<|audio|>";
pub(crate) const GRANITE_SPEECH_AUDIO_TOKEN_ID: u32 = 100_352;

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

/// Builds the full `[n_tokens, hidden_size]` embedding sequence for a prompt
/// containing one audio placeholder, ready for
/// `decoder_graph::prefill_logits_from_embeddings`. `audio_embeddings` is the
/// Q-Former projector's `[audio_token_count, hidden_size]` output (already
/// projected to the LLM's `hidden_size`, see `qformer::project`).
pub(crate) fn build_audio_prompt_embeddings(
    config: &GraniteSpeechDecoderConfig,
    provider: &dyn GraniteSpeechDecoderWeightProvider,
    tokenizer: &GraniteSpeechTokenizer,
    prompt_text: &str,
    audio_embeddings: &[f32],
    audio_token_count: usize,
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

    let mut embeddings = Vec::with_capacity(token_ids.len() * config.hidden_size);
    let mut next_audio_slot = 0usize;
    for &token_id in &token_ids {
        if token_id == GRANITE_SPEECH_AUDIO_TOKEN_ID {
            let start = next_audio_slot * config.hidden_size;
            embeddings.extend_from_slice(&audio_embeddings[start..start + config.hidden_size]);
            next_audio_slot += 1;
        } else {
            embeddings.extend_from_slice(embed_token_row(config, provider, token_id)?);
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
}
