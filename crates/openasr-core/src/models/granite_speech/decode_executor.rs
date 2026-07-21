//! Granite Speech decode-step executor: wires `decoder_graph::prefill_logits`
//! into the shared greedy-decode driver
//! (`seq2seq_greedy_decode::run_seq2seq_greedy_decode_loop_with_adapter_v0`,
//! reached the same way every other builtin family reaches it -- see
//! `AGENTS.md`'s "one greedy decode driver" invariant). This module never
//! picks a token itself; `decode_step_logits` only returns a logits row, and
//! the shared driver owns argmax, suppression, stop-token, and the
//! degenerate-loop guard.
//!
//! KV-cache note (explicitly scoped this way, see the coordinator note this
//! was written against): another in-flight change is reworking the shared
//! `nn::decoder` graph-reuse/KV-cache mechanism that qwen's production decode
//! depends on. Rather than build a second, competing incremental-KV
//! mechanism inside this family while that lands, this executor recomputes
//! the *entire* prefix from scratch every step via `prefill_logits` (a plain
//! non-incremental forward, see that module's doc) and reads off the last
//! position's logits. This is the "use the current mechanism" instruction
//! taken literally: `prefill_logits` IS the current (only) mechanism this
//! family has, so decode-step-N is just prefill over
//! `initial_prompt_tokens ++ generated_tokens_so_far`. It is O(n^2) in total
//! decoded length and does not share this family's future incremental
//! KV-cache session; once the shared graph-reuse mechanism lands, swapping
//! this executor to it is a local, non-invasive change (the
//! `Seq2SeqGreedyDecodeStepExecutor` boundary is exactly where that swap
//! happens, nothing above this module needs to change).

#![allow(dead_code)]

use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};

use super::decoder_graph::{
    GraniteSpeechDecoderConfig, GraniteSpeechDecoderWeightProvider, prefill_logits,
};

pub(crate) struct GraniteSpeechDecodeStepExecutor<'p> {
    config: GraniteSpeechDecoderConfig,
    provider: &'p dyn GraniteSpeechDecoderWeightProvider,
    backend: GgmlCpuGraphBackend,
}

impl<'p> GraniteSpeechDecodeStepExecutor<'p> {
    pub(crate) fn new(
        config: GraniteSpeechDecoderConfig,
        provider: &'p dyn GraniteSpeechDecoderWeightProvider,
        backend: GgmlCpuGraphBackend,
    ) -> Self {
        Self {
            config,
            provider,
            backend,
        }
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for GraniteSpeechDecodeStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        let mut token_ids: Vec<u32> =
            Vec::with_capacity(input.initial_prompt_tokens.len() + input.generated_tokens.len());
        token_ids.extend_from_slice(input.initial_prompt_tokens);
        token_ids.extend_from_slice(input.generated_tokens);

        let output = prefill_logits(&self.config, self.provider, &token_ids, self.backend)
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: format!("granite-speech decoder step {}: {error}", input.step_index),
            })?;

        let vocab_size = output.vocab_size;
        let last_start = (output.n_tokens - 1) * vocab_size;
        let logits = output.logits[last_start..last_start + vocab_size].to_vec();

        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::models::seq2seq_greedy_decode::{
        Seq2SeqGreedyDecodeConfig, run_seq2seq_greedy_decode_loop_with_adapter_v0,
    };

    const SOURCE_ROOT: &str =
        "/Volumes/QuintinDocument/openasr-dev/tmp/granite-work/granite-speech-4.1-2b-src";
    const GOLDEN_ROOT: &str = "/Volumes/QuintinDocument/openasr-dev/tmp/granite-work/golden";

    fn load_safetensors_prefixed(dir: &Path, prefix: &str) -> HashMap<String, Vec<f32>> {
        let index_path = dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path).expect("read safetensors index");
        let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("parse index");
        let weight_map = index["weight_map"].as_object().expect("weight_map object");
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shard_names.sort();
        shard_names.dedup();

        let mut out = HashMap::new();
        for shard in shard_names {
            let path = dir.join(&shard);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header_end = 8 + header_len;
            let header: serde_json::Value =
                serde_json::from_slice(&bytes[8..header_end]).expect("parse safetensors header");
            let obj = header.as_object().expect("header object");
            for (name, meta) in obj {
                if name == "__metadata__" || !name.starts_with(prefix) {
                    continue;
                }
                let dtype = meta["dtype"].as_str().expect("dtype");
                let offsets = meta["data_offsets"].as_array().expect("data_offsets");
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;
                let raw = &bytes[header_end + start..header_end + end];
                let values: Vec<f32> = match dtype {
                    "F32" => raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                    "BF16" => raw
                        .chunks_exact(2)
                        .map(|c| {
                            f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
                        })
                        .collect(),
                    _ => continue,
                };
                out.insert(name.clone(), values);
            }
        }
        out
    }

    fn load_npy_i64(path: &Path) -> Vec<i64> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let data_start = header_start + header_len;
        bytes[data_start..]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    #[ignore = "requires local 4.6GB granite-speech-4.1-2b weights + golden fixtures under tmp/ (not committed)"]
    fn granite_speech_greedy_decode_matches_hf_reference() {
        let source_root = PathBuf::from(SOURCE_ROOT);
        if !source_root.join("model.safetensors.index.json").exists() {
            eprintln!("skip: {SOURCE_ROOT} not present");
            return;
        }
        let golden_root = PathBuf::from(GOLDEN_ROOT);

        let weights = load_safetensors_prefixed(&source_root, "language_model.");
        let config = GraniteSpeechDecoderConfig::granite_speech_4_1_2b();

        let prompt: Vec<u32> = vec![
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
        ];
        let golden_continuation =
            load_npy_i64(&golden_root.join("decoder_greedy_continuation.npy"));
        // HF's `generate` includes the EOS token in its output; the shared
        // driver's `eot_token_id` stop check does the same (the generated
        // list it returns also includes the token that triggered the stop).
        let eot_token_id = 100_257u32;

        let decode_config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: prompt,
            eot_token_id,
            stop_token_ids: vec![],
            vocab_size: config.vocab_size,
            max_generated_tokens: 16,
            suppress_first_step_token_ids: vec![],
            suppress_token_ids: vec![],
            phrase_biases: vec![],
        };

        let mut executor =
            GraniteSpeechDecodeStepExecutor::new(config, &weights, GgmlCpuGraphBackend::Cpu);

        let decode_text_token_ids =
            |_token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> { Ok(String::new()) };

        let result = run_seq2seq_greedy_decode_loop_with_adapter_v0(
            &decode_config,
            &mut executor,
            &decode_text_token_ids,
            |error| error,
            |error| error,
            &|text| text,
            &mut |_step, _token, _eot| {},
            &mut |_step, _logits| {},
        )
        .expect("greedy decode");

        println!("== Granite Speech greedy decode (registry-shared driver) ==");
        println!("actual generated tokens: {:?}", result.generated_tokens);
        println!("golden   generated tokens: {:?}", golden_continuation);

        // HF's `generate()` includes the terminating EOT token in its output;
        // the shared driver treats EOT as a stop signal and excludes it from
        // `generated_tokens` (an ASR-decode convention, not a bug -- the
        // content tokens before it are what matters). Strip it before
        // comparing.
        let mut golden_u32: Vec<u32> = golden_continuation.iter().map(|&id| id as u32).collect();
        if golden_u32.last() == Some(&eot_token_id) {
            golden_u32.pop();
        }
        assert_eq!(
            result.generated_tokens, golden_u32,
            "greedy-decoded token sequence must match the HF reference exactly"
        );
    }
}
