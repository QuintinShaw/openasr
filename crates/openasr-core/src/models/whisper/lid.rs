//! Whisper source-language auto-detection (LID).
//!
//! A single decoder step over the bare `[<sot>]` prefix, reusing the already-run
//! encoder and the populated cross-attention cache, then an argmax over only the
//! language-token logits. This mirrors whisper.cpp's `whisper_lang_auto_detect`.
//!
//! Fail-open by design: any failure (no language tokens in vocab, a graph error,
//! a width mismatch) returns `None`, so the caller leaves the request language
//! unset and the decode falls back to the byte-identical default path.

use std::collections::HashMap;

use crate::ggml_runtime::GgmlCpuGraphRunner;
use crate::models::language::{WHISPER_LANGUAGE_CODES, language_control_token};

use super::ggml_decoder_graph::{
    WhisperDecoderExecutionTensorCache, WhisperDecoderGraphExecutionConfig,
    WhisperDecoderGraphExecutionInput, WhisperDecoderGraphPlan, WhisperDecoderHiddenStateLayout,
    WhisperDecoderPersistentWeightCache, WhisperDecoderSelfKvCacheState,
    WhisperDecoderTensorSource, run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0,
};
use super::tokenizer::WhisperTokenizer;

/// Map every `<|code|>` language token actually present in this pack's vocab to
/// its code. Empty for an English-only pack (no language-token block), which
/// makes detection a no-op there.
fn language_token_id_map(tokenizer: &WhisperTokenizer) -> HashMap<u32, &'static str> {
    let mut map = HashMap::with_capacity(WHISPER_LANGUAGE_CODES.len());
    for &code in WHISPER_LANGUAGE_CODES {
        if let Some(id) = tokenizer.token_id_by_content(&language_control_token(code)) {
            map.insert(id, code);
        }
    }
    map
}

/// Run one decoder step over `[<sot>]` against the populated cross-attention
/// cache and return the argmax language code over the language-token id set.
///
/// Returns `None` on any error or when the pack carries no language tokens
/// (fail-open: the caller leaves the language unset). The step uses its own
/// throwaway self-KV and tensor caches and must never share them with the main
/// decode loop, which starts from a fresh self-KV of its own.
#[allow(clippy::too_many_arguments)]
pub(super) fn detect_whisper_language_sot_step(
    runner: &mut GgmlCpuGraphRunner,
    persistent_weights: &WhisperDecoderPersistentWeightCache,
    plan: &WhisperDecoderGraphPlan,
    tensor_source: &dyn WhisperDecoderTensorSource,
    config: WhisperDecoderGraphExecutionConfig,
    tokenizer: &WhisperTokenizer,
    encoder_hidden_f32: &[f32],
    vocab_size: usize,
) -> Option<String> {
    let sot = tokenizer.start_of_transcript_token_id()?;
    let language_ids = language_token_id_map(tokenizer);
    if language_ids.is_empty() {
        return None;
    }
    let input = WhisperDecoderGraphExecutionInput {
        decoder_prefix_tokens: vec![sot],
        encoder_hidden_state: encoder_hidden_f32.to_vec(),
        encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
    };
    let self_kv = WhisperDecoderSelfKvCacheState::new();
    let mut tensor_cache = WhisperDecoderExecutionTensorCache::default();
    let output = run_whisper_decoder_greedy_step_with_cache_and_runner_ggml_v0(
        runner,
        Some(persistent_weights),
        Some(&self_kv),
        0,
        plan,
        &input,
        tensor_source,
        config,
        &mut tensor_cache,
    )
    .ok()?;
    if output.logits.len() != vocab_size {
        return None;
    }
    // Argmax over only the language-token ids (not the full vocab, which is what
    // `output.greedy_token` would give).
    let mut best: Option<(u32, f32)> = None;
    for &id in language_ids.keys() {
        let Some(&logit) = output.logits.get(id as usize) else {
            continue;
        };
        if logit.is_finite() && best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((id, logit));
        }
    }
    let (id, _) = best?;
    language_ids.get(&id).map(|code| (*code).to_string())
}
