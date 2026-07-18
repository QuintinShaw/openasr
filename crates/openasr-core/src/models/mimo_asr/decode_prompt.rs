//! ChatML + `<|sosp|>`/`<|eosp|>` audio-frame splice span, mirroring
//! `mimo_audio.py::get_asr_sft_prompt`'s inference prompt:
//! `<|im_start|>user\n<|sosp|>[audio]<|eosp|>{instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n`.
//!
//! P2.0 findings SS3 point 7 established that, for the ASR input side, the
//! audio-boundary tokens (`<|sosp|>`/`<|eosp|>`) behave as PLAIN ChatML text
//! tokens: at those two boundary groups `is_speech` is false (only the
//! literal `<|empty|>` text-channel value marks a group as audio), so their
//! contribution is `embed_tokens(sosp_id/eosp_id)` exactly like any other
//! text token, with zero speech-embedding contribution. That means this
//! family's audio splice has the SAME shape as qwen3-asr's
//! `<|audio_start|>...<|audio_pad|>...<|audio_end|>` prompt and reuses
//! `qwen::Qwen3AsrDecodePrompt`/`build_qwen3_prompt_embeddings_with_audio_splice`
//! directly (see `firered_llm::decode_prompt`'s identical reasoning) -- the
//! only per-frame audio-channel-specific work (8-codebook sum, input-local
//! transformer, group downcast) happens upstream in
//! [`super::input_local_graph`] and is what fills the pad span's rows.

use thiserror::Error;

use crate::models::qwen::Qwen3AsrDecodePrompt;

use super::tokenizer::MimoAsrTokenizer;

/// Fixed instruction text (one of the upstream `asr_zh_templates`, picked
/// deterministically rather than the reference's per-call `random.choice`):
/// the model was trained against many paraphrases of this instruction so a
/// fixed pick is a config choice, not a correctness requirement, and a fixed
/// prompt keeps golden-diff tests deterministic (same reasoning as
/// `firered_llm::decode_prompt`'s hard-coded instruction).
const MIMO_ASR_INSTRUCTION_TEXT: &str = "请将这段语音转换为文字";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum MimoDecodePromptError {
    #[error("mimo-asr decode prompt requires at least one audio frame group")]
    EmptyAudioGroups,
    #[error("mimo-asr decode prompt tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
}

pub(crate) fn build_mimo_asr_decode_prompt(
    tokenizer: &MimoAsrTokenizer,
    audio_group_count: usize,
) -> Result<Qwen3AsrDecodePrompt, MimoDecodePromptError> {
    if audio_group_count == 0 {
        return Err(MimoDecodePromptError::EmptyAudioGroups);
    }
    let prefix = "<|im_start|>user\n";
    let suffix = format!(
        "{MIMO_ASR_INSTRUCTION_TEXT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    );
    let prefix_ids = tokenizer.encode_prompt_text(prefix).map_err(|error| {
        MimoDecodePromptError::TokenizationFailed {
            reason: error.to_string(),
        }
    })?;
    let suffix_ids = tokenizer.encode_prompt_text(&suffix).map_err(|error| {
        MimoDecodePromptError::TokenizationFailed {
            reason: error.to_string(),
        }
    })?;

    let mut token_ids = Vec::with_capacity(
        prefix_ids
            .len()
            .saturating_add(1)
            .saturating_add(audio_group_count)
            .saturating_add(1)
            .saturating_add(suffix_ids.len()),
    );
    token_ids.extend(prefix_ids.iter().copied());
    token_ids.push(tokenizer.special.sosp_id);
    let audio_pad_start_index = token_ids.len();
    // Placeholder id for the pad span -- overwritten by the input-local
    // transformer's group embeddings before the prompt ever reaches the
    // backbone (see `build_qwen3_prompt_embeddings_with_audio_splice`); using
    // `empty_id` here is the closest analog to the reference's own
    // text-channel filler for audio groups.
    token_ids.extend(std::iter::repeat_n(
        tokenizer.special.empty_id,
        audio_group_count,
    ));
    token_ids.push(tokenizer.special.eosp_id);
    token_ids.extend(suffix_ids.iter().copied());

    Ok(Qwen3AsrDecodePrompt {
        token_ids,
        audio_pad_start_index,
        audio_pad_count: audio_group_count,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::models::gpt2_bpe::bytes_to_unicode;
    use crate::models::mimo_asr::runtime_contract::MimoSpecialTokens;
    use crate::{GgufMetadata, GgufMetadataValue};

    use super::*;

    /// Byte-level BPE with no merges at all: every distinct raw byte across
    /// the whole prompt (prefix + instruction + suffix) gets its own vocab
    /// entry (one token per byte on encode), so this fixture only has to
    /// enumerate bytes, not reason about which multi-byte merges would fire.
    fn tokenizer_fixture() -> MimoAsrTokenizer {
        let mut values = BTreeMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("gpt2".to_string()),
        );
        let mut tokens = vec![
            "<|im_start|>".to_string(),  // 0
            "<|im_end|>".to_string(),    // 1
            "<|endoftext|>".to_string(), // 2
            "<|sosp|>".to_string(),      // 3
            "<|eosp|>".to_string(),      // 4
            "<|empty|>".to_string(),     // 5
            "<|eot|>".to_string(),       // 6
            "<|eostm|>".to_string(),     // 7
        ];
        let whole_prompt = format!(
            "<|im_start|>user\n<|sosp|><|eosp|>{MIMO_ASR_INSTRUCTION_TEXT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        );
        let mut seen_bytes = std::collections::BTreeSet::new();
        for byte in whole_prompt.as_bytes() {
            if seen_bytes.insert(*byte) {
                tokens.push(bytes_to_unicode(&[*byte]));
            }
        }
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            // A merge rule that can never match this prompt's bytes (no 'z'
            // appears anywhere in it): byte-level BPE then emits one token
            // per input byte, which is exactly what this fixture wants.
            GgufMetadataValue::StringArray(vec!["z z".to_string()]),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        let special = MimoSpecialTokens {
            eos_id: 2,
            im_start_id: 0,
            im_end_id: 1,
            sosp_id: 3,
            eosp_id: 4,
            empty_id: 5,
            eot_id: 6,
            eostm_id: 7,
        };
        MimoAsrTokenizer::from_gguf_metadata(&metadata, special).expect("tokenizer")
    }

    #[test]
    fn builds_chatml_prompt_with_sosp_eosp_pad_span() {
        let tokenizer = tokenizer_fixture();
        let prompt = build_mimo_asr_decode_prompt(&tokenizer, 5).expect("prompt");

        let prefix_ids = tokenizer
            .encode_prompt_text("<|im_start|>user\n")
            .expect("encode prefix");
        assert_eq!(&prompt.token_ids[..prefix_ids.len()], prefix_ids.as_slice());
        let after_prefix = prefix_ids.len();
        assert_eq!(prompt.token_ids[after_prefix], tokenizer.special.sosp_id);
        let pad_start = after_prefix + 1;
        assert_eq!(
            &prompt.token_ids[pad_start..pad_start + 5],
            &[tokenizer.special.empty_id; 5]
        );
        assert_eq!(prompt.token_ids[pad_start + 5], tokenizer.special.eosp_id);
        assert_eq!(prompt.audio_pad_start_index, pad_start);
        assert_eq!(prompt.audio_pad_count, 5);

        // The fixed "<think>\n\n</think>\n" continuation prefix is the tail
        // of the prompt (the model continues generating from right after it).
        let think_ids = tokenizer
            .encode_prompt_text("<think>\n\n</think>\n")
            .expect("encode think");
        assert_eq!(
            &prompt.token_ids[prompt.token_ids.len() - think_ids.len()..],
            think_ids.as_slice()
        );
    }

    #[test]
    fn rejects_zero_audio_groups() {
        let tokenizer = tokenizer_fixture();
        let error = build_mimo_asr_decode_prompt(&tokenizer, 0).expect_err("must fail");
        assert_eq!(error, MimoDecodePromptError::EmptyAudioGroups);
    }
}
