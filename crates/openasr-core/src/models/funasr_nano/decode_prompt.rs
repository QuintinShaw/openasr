//! Fun-ASR-Nano ChatML decode prompt + audio-embedding splice span, ported
//! one-for-one from the official funasr-nano llama.cpp runtime's prompt (and
//! the model.pt-derived reference oracle):
//!
//! ```text
//! <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//! <|im_start|>user\n语音转写：{audio}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! `{audio}` is a contiguous run of one placeholder token per kept adaptor
//! output frame (`n_aud`, the low-frame-rate-truncated audio-token count),
//! standing in for the audio embeddings the executor splices in afterward --
//! the exact same contiguous-span shape as qwen3-asr's `<|audio_pad|>` run, so
//! this reuses `qwen::Qwen3AsrDecodePrompt` /
//! `build_qwen3_prompt_embeddings_with_audio_splice` directly (mirrors
//! `firered_llm::decode_prompt`). The placeholder token id is irrelevant to the
//! forward pass -- its embedding is overwritten by the spliced audio rows -- so
//! any in-vocab id serves; the ChatML `<|im_start|>` marker is used.

use thiserror::Error;

use crate::models::qwen::Qwen3AsrDecodePrompt;

use super::tokenizer::FunasrNanoTokenizer;

/// The fixed ChatML system + user-instruction wrapper the checkpoint was
/// fine-tuned against ("please transcribe the speech": the literal Mandarin
/// instruction "语音转写:"). Not user-configurable -- substituting a different
/// instruction is an unverified, unrequested capability.
const FUNASR_NANO_PROMPT_PREFIX: &str =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：";
const FUNASR_NANO_PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FunasrNanoDecodePromptError {
    #[error("funasr-nano decode prompt requires at least one audio frame")]
    EmptyAudioFrames,
    #[error("funasr-nano decode prompt tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
}

pub(crate) fn build_funasr_nano_decode_prompt(
    tokenizer: &FunasrNanoTokenizer,
    audio_frame_count: usize,
) -> Result<Qwen3AsrDecodePrompt, FunasrNanoDecodePromptError> {
    if audio_frame_count == 0 {
        return Err(FunasrNanoDecodePromptError::EmptyAudioFrames);
    }
    let encode = |text: &str| -> Result<Vec<u32>, FunasrNanoDecodePromptError> {
        tokenizer.encode_prompt_text(text).map_err(|error| {
            FunasrNanoDecodePromptError::TokenizationFailed {
                reason: error.to_string(),
            }
        })
    };
    let prefix_ids = encode(FUNASR_NANO_PROMPT_PREFIX)?;
    let suffix_ids = encode(FUNASR_NANO_PROMPT_SUFFIX)?;
    let mut token_ids = Vec::with_capacity(
        prefix_ids
            .len()
            .saturating_add(audio_frame_count)
            .saturating_add(suffix_ids.len()),
    );
    token_ids.extend(prefix_ids.iter().copied());
    let audio_pad_start_index = token_ids.len();
    token_ids.extend(std::iter::repeat_n(
        tokenizer.chatml_im_start_token_id,
        audio_frame_count,
    ));
    token_ids.extend(suffix_ids.iter().copied());
    Ok(Qwen3AsrDecodePrompt {
        token_ids,
        audio_pad_start_index,
        audio_pad_count: audio_frame_count,
    })
}

/// Fun-ASR-Nano low-frame-rate audio-token count: the number of adaptor output
/// frames kept as real audio tokens (the leading `n_aud` frames), derived from
/// the encoder's LFR frame count `t` via the official runtime's fake-token
/// formula (`ol = 1 + (t - 3 + 2) / 2` applied twice, then `(ol - 1) / 2 + 1`,
/// all integer division). Keeping only the leading `n_aud` frames is what stops
/// the LLM from repeating over the adaptor's trailing padded frames.
pub(crate) fn funasr_nano_audio_token_count(lfr_frame_count: usize) -> usize {
    let conv = |t: usize| 1 + (t.saturating_sub(3) + 2) / 2;
    let ol = conv(conv(lfr_frame_count));
    (ol.saturating_sub(1)) / 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_token_count_matches_reference_formula() {
        // The two committed golden clips: t=120 -> n_aud=15, t=94 -> n_aud=12
        // (funasr-golden meta_en/meta_zh).
        assert_eq!(funasr_nano_audio_token_count(120), 15);
        assert_eq!(funasr_nano_audio_token_count(94), 12);
    }

    #[test]
    fn audio_token_count_is_monotonic_and_positive() {
        assert!(funasr_nano_audio_token_count(1) >= 1);
        assert!(funasr_nano_audio_token_count(1000) > funasr_nano_audio_token_count(100));
    }
}
