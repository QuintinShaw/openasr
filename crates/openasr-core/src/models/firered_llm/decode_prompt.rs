//! ChatML decode prompt + `<speech>` audio-frame splice span, mirroring
//! `fireredasr2/asr.py`'s inference prompt:
//! `<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n`
//! with `<speech>` (repeated once per adapter output frame) standing in for
//! the audio embeddings the executor splices in afterward -- the exact same
//! shape as qwen3-asr's `<|audio_start|>...<|audio_pad|>...<|audio_end|>`
//! prompt, reusing `qwen::Qwen3AsrDecodePrompt` /
//! `build_qwen3_prompt_embeddings_with_audio_splice` directly (see this
//! module's `decode_prompt` construction below): splice-a-run-of-pad-tokens
//! has no Qwen2/Qwen3-specific shape.

use thiserror::Error;

use crate::models::qwen::Qwen3AsrDecodePrompt;

use super::tokenizer::FireRedLlmTokenizer;

/// The upstream default instruction (`fireredasr2/asr.py`'s hard-coded
/// Mandarin prompt text, verified in `scratchpad/fr2/fireredasr_llm.py`):
/// "please transcribe the speech into text". Not user-configurable in this
/// stage -- FireRedASR2-LLM's LLM decoder was fine-tuned against this exact
/// instruction wording (see LoRA training args), so substituting a different
/// instruction string is an unverified, unrequested capability.
const FIRERED_LLM_INSTRUCTION_TEXT: &str = "请转写音频为文字";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FireRedLlmDecodePromptError {
    #[error("firered-llm decode prompt requires at least one audio frame")]
    EmptyAudioFrames,
    #[error("firered-llm decode prompt tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
}

pub(crate) fn build_firered_llm_decode_prompt(
    tokenizer: &FireRedLlmTokenizer,
    audio_frame_count: usize,
) -> Result<Qwen3AsrDecodePrompt, FireRedLlmDecodePromptError> {
    if audio_frame_count == 0 {
        return Err(FireRedLlmDecodePromptError::EmptyAudioFrames);
    }
    let prefix = "<|im_start|>user\n";
    let suffix = format!("{FIRERED_LLM_INSTRUCTION_TEXT}<|im_end|>\n<|im_start|>assistant\n");
    let prefix_ids = tokenizer.encode_prompt_text(prefix).map_err(|error| {
        FireRedLlmDecodePromptError::TokenizationFailed {
            reason: error.to_string(),
        }
    })?;
    let suffix_ids = tokenizer.encode_prompt_text(&suffix).map_err(|error| {
        FireRedLlmDecodePromptError::TokenizationFailed {
            reason: error.to_string(),
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
    token_ids.extend(std::iter::repeat_n(
        tokenizer.speech_token_id,
        audio_frame_count,
    ));
    token_ids.extend(suffix_ids.iter().copied());
    Ok(Qwen3AsrDecodePrompt {
        token_ids,
        audio_pad_start_index,
        audio_pad_count: audio_frame_count,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::models::gpt2_bpe::bytes_to_unicode;
    use crate::{GgufMetadata, GgufMetadataValue};

    use super::*;
    use crate::models::firered_llm::runtime_contract::{
        FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY, FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY,
        FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY, FIRERED_LLM_LLM_VOCAB_SIZE_KEY,
        FIRERED_LLM_SPEECH_TOKEN_ID_KEY,
    };
    use crate::models::oasr_metadata::{
        TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_TOKENS_KEY,
    };

    fn tokenizer_fixture() -> FireRedLlmTokenizer {
        let mut values = BTreeMap::new();
        values.insert(
            TOKENIZER_GGML_MODEL_KEY.to_string(),
            GgufMetadataValue::String("gpt2".to_string()),
        );
        // Minimal vocab covering the ChatML prefix/suffix + speech
        // placeholder: "user"/"assistant" are reachable via real merge
        // chains (byte-level BPE cannot match a whole word absent the merges
        // that build up to it), and the Chinese instruction text's raw UTF-8
        // bytes are each their own single-byte vocab entry (no merges
        // needed -- BPE with zero applicable merges just emits one token per
        // input byte, and this test only cares that encoding succeeds and
        // the audio-pad span lands where expected, not the suffix's exact
        // token ids).
        let mut tokens = vec![
            "<|im_start|>".to_string(),  // 0
            "u".to_string(),             // 1
            "s".to_string(),             // 2
            "e".to_string(),             // 3
            "r".to_string(),             // 4
            "us".to_string(),            // 5
            "use".to_string(),           // 6
            "user".to_string(),          // 7
            "\u{010A}".to_string(),      // 8
            "<speech>".to_string(),      // 9
            "a".to_string(),             // 10
            "i".to_string(),             // 11
            "t".to_string(),             // 12
            "n".to_string(),             // 13
            "as".to_string(),            // 14
            "ass".to_string(),           // 15
            "assi".to_string(),          // 16
            "assis".to_string(),         // 17
            "assist".to_string(),        // 18
            "assista".to_string(),       // 19
            "assistan".to_string(),      // 20
            "assistant".to_string(),     // 21
            "<|im_end|>".to_string(),    // 22
            "<|endoftext|>".to_string(), // 23
        ];
        let merges = vec![
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
        ];
        // One vocab entry per unique raw byte in the upstream instruction
        // text -- byte-level BPE with no applicable merge for these bytes
        // emits them as individual tokens.
        let mut seen_bytes = std::collections::BTreeSet::new();
        for byte in FIRERED_LLM_INSTRUCTION_TEXT.as_bytes() {
            if seen_bytes.insert(*byte) {
                tokens.push(bytes_to_unicode(&[*byte]));
            }
        }
        let speech_token_id = 9;
        let im_start_token_id = 0;
        let im_end_token_id = 22;
        let endoftext_token_id = 23;
        let vocab_size = tokens.len() as u32;

        values.insert(
            TOKENIZER_GGML_TOKENS_KEY.to_string(),
            GgufMetadataValue::StringArray(tokens),
        );
        values.insert(
            TOKENIZER_GGML_MERGES_KEY.to_string(),
            GgufMetadataValue::StringArray(merges),
        );
        values.insert(
            FIRERED_LLM_LLM_VOCAB_SIZE_KEY.to_string(),
            GgufMetadataValue::U32(vocab_size),
        );
        values.insert(
            FIRERED_LLM_SPEECH_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(speech_token_id),
        );
        values.insert(
            FIRERED_LLM_CHATML_IM_START_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(im_start_token_id),
        );
        values.insert(
            FIRERED_LLM_CHATML_IM_END_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(im_end_token_id),
        );
        values.insert(
            FIRERED_LLM_ENDOFTEXT_TOKEN_ID_KEY.to_string(),
            GgufMetadataValue::U32(endoftext_token_id),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        FireRedLlmTokenizer::from_gguf_metadata(&metadata).expect("tokenizer")
    }

    #[test]
    fn builds_chatml_prompt_with_speech_pad_span() {
        let tokenizer = tokenizer_fixture();
        let prompt = build_firered_llm_decode_prompt(&tokenizer, 4).expect("prompt");
        // <|im_start|> user \n [speech x4] instruction <|im_end|> \n <|im_start|> assistant \n
        assert_eq!(prompt.token_ids[0], 0); // <|im_start|>
        assert_eq!(prompt.token_ids[1], 7); // user
        assert_eq!(prompt.token_ids[2], 8); // \n
        assert_eq!(&prompt.token_ids[3..7], &[9, 9, 9, 9]); // <speech> x4
        assert_eq!(prompt.audio_pad_start_index, 3);
        assert_eq!(prompt.audio_pad_count, 4);
    }

    #[test]
    fn rejects_zero_audio_frames() {
        let tokenizer = tokenizer_fixture();
        let error = build_firered_llm_decode_prompt(&tokenizer, 0).expect_err("must fail");
        assert_eq!(error, FireRedLlmDecodePromptError::EmptyAudioFrames);
    }
}
