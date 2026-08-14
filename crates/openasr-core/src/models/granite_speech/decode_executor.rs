//! Granite Speech decode-step executors: wire an incremental-KV
//! `decode_session::GraniteSpeechDecodeSession` into the shared greedy-decode
//! driver (`seq2seq_greedy_decode::run_seq2seq_greedy_decode_loop_with_adapter_v0`,
//! reached the same way every other builtin family reaches it -- see
//! `AGENTS.md`'s "one greedy decode driver" invariant). This module never
//! picks a token itself; `decode_step_logits` only returns a logits row, and
//! the shared driver owns argmax, suppression, stop-token, and the
//! degenerate-loop guard.
//!
//! Incremental KV cache: the first `decode_step_logits` call prefills the whole
//! prompt into a persistent session (seeding every layer's K/V); each later call
//! computes Q/K/V for only the newly generated token, appends its K/V, and
//! attends the single new query against the full cached history (see
//! `decode_session`). This is the same incremental-decode shape every other
//! autoregressive family here already uses (qwen `Qwen3AsrLayerKvCacheState`,
//! firered-llm, cohere), and it replaces the earlier
//! recompute-the-entire-prefix-every-step path that was `O(n^2)` in decoded
//! length (~430x realtime for the 2B Granite decoder). The session's outputs are
//! bit-identical to that full recompute -- proven by
//! `decode_session`'s `granite_incremental_decode_matches_full_recompute_bit_exact`.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::mapped_token_embedding::{MappedTokenEmbeddingError, MappedTokenEmbeddingTable};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};

use super::decode_session::{GraniteSpeechDecodeSession, GraniteSpeechKvCacheCapacity};
use super::decoder_graph::{
    GraniteSpeechDecoderConfig, GraniteSpeechDecoderError, embed_token_row,
};
use super::prompt::{
    GRANITE_SPEECH_AUDIO_TOKEN_ID, materialize_audio_prompt_embeddings_from_mapped_table,
};

fn map_step_error(
    step_index: usize,
    label: &'static str,
) -> impl Fn(GraniteSpeechDecoderError) -> Seq2SeqGreedyDecodeError {
    move |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: format!("granite-speech {label} decoder step {step_index}: {error}"),
    }
}

fn map_embedding_error(
    step_index: usize,
) -> impl Fn(MappedTokenEmbeddingError) -> Seq2SeqGreedyDecodeError {
    move |error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: format!(
            "granite-speech audio decoder step {step_index} token embedding gather failed: {error}"
        ),
    }
}

/// Assert the shared greedy driver is calling us in the strict incremental
/// order the KV cache session assumes: step 0 with no generated tokens (prefill
/// covers the prompt), then exactly one new token per step. Fails closed rather
/// than silently returning stale logits if that invariant is ever violated.
fn incremental_new_token(
    session: &GraniteSpeechDecodeSession,
    prompt_len: usize,
    input: &Seq2SeqGreedyDecodeStepInput<'_>,
) -> Result<u32, Seq2SeqGreedyDecodeError> {
    let expected_cached = prompt_len + input.generated_tokens.len().saturating_sub(1);
    let new_token = input.generated_tokens.last().copied();
    match new_token {
        Some(token) if session.cached_positions() == expected_cached => Ok(token),
        _ => Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
            reason: format!(
                "granite-speech decoder step {} out of incremental order: cached {} positions, \
                 {} generated tokens, prompt {}",
                input.step_index,
                session.cached_positions(),
                input.generated_tokens.len(),
                prompt_len
            ),
        }),
    }
}

/// Incremental-KV greedy step executor for a text-token prompt. Builds a
/// persistent [`GraniteSpeechDecodeSession`] on the first call (prefilling the
/// prompt), then advances it one token per step -- eliminating the historical
/// per-step full-prefix recompute (`O(n^2)` decode).
pub(crate) struct GraniteSpeechDecodeStepExecutor<'p> {
    config: GraniteSpeechDecoderConfig,
    provider: &'p HashMap<String, Vec<f32>>,
    backend: GgmlCpuGraphBackend,
    session: Option<GraniteSpeechDecodeSession>,
    prompt_len: usize,
    capacity: GraniteSpeechKvCacheCapacity,
}

impl<'p> GraniteSpeechDecodeStepExecutor<'p> {
    pub(crate) fn new(
        config: GraniteSpeechDecoderConfig,
        provider: &'p HashMap<String, Vec<f32>>,
        backend: GgmlCpuGraphBackend,
        capacity: GraniteSpeechKvCacheCapacity,
    ) -> Self {
        Self {
            config,
            provider,
            backend,
            session: None,
            prompt_len: 0,
            capacity,
        }
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for GraniteSpeechDecodeStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        // Steps after the first advance the existing session by one token.
        if let Some(session) = self.session.as_mut() {
            let new_token = incremental_new_token(session, self.prompt_len, &input)?;
            let logits = session
                .decode_step(new_token, self.provider)
                .map_err(map_step_error(input.step_index, "text"))?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            });
        }

        // First call: prefill the text-token prompt into a fresh session.
        let hidden = self.config.hidden_size;
        let mut embeddings = Vec::with_capacity(input.initial_prompt_tokens.len() * hidden);
        for &token_id in input.initial_prompt_tokens {
            let row = embed_token_row(&self.config, self.provider, token_id)
                .map_err(map_step_error(input.step_index, "text"))?;
            embeddings.extend_from_slice(row);
        }
        let mut session = GraniteSpeechDecodeSession::new(self.config, self.provider, self.backend)
            .map_err(map_step_error(input.step_index, "text"))?;
        let logits = session
            .prefill(
                &embeddings,
                input.initial_prompt_tokens.len(),
                self.capacity,
            )
            .map_err(map_step_error(input.step_index, "text"))?;
        self.prompt_len = input.initial_prompt_tokens.len();
        self.session = Some(session);
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

/// Same incremental-KV session as [`GraniteSpeechDecodeStepExecutor`], but for a
/// prompt containing a spliced-in audio embedding sequence (see
/// `prompt::build_audio_prompt_embeddings`): the fixed prompt prefix's
/// embeddings (audio + text) are prefilled once; each subsequent step embeds
/// only the single newly generated (always plain-text) token and advances the
/// cache. `input.initial_prompt_tokens` must be the exact token-id sequence
/// `initial_prompt_embeddings` was built from (used for its length, and so the
/// shared driver's phrase-bias/stop-token bookkeeping still sees real token
/// ids) -- this executor never re-derives the prompt embeddings from it.
pub(crate) struct GraniteSpeechAudioDecodeStepExecutor<'p> {
    config: GraniteSpeechDecoderConfig,
    provider: &'p HashMap<String, Vec<f32>>,
    backend: GgmlCpuGraphBackend,
    initial_prompt_embeddings: Vec<f32>,
    session: Option<GraniteSpeechDecodeSession>,
    prompt_len: usize,
    capacity: GraniteSpeechKvCacheCapacity,
}

impl<'p> GraniteSpeechAudioDecodeStepExecutor<'p> {
    /// f32-arena audio-prompt step executor: builds a fresh session from
    /// `provider` (the safetensors/`HashMap` test path) on the first step. The
    /// runtime keep-quantized path is served by
    /// [`GraniteSpeechResidentAudioDecodeStepExecutor`] against a cross-request
    /// resident session instead.
    pub(crate) fn new(
        config: GraniteSpeechDecoderConfig,
        provider: &'p HashMap<String, Vec<f32>>,
        backend: GgmlCpuGraphBackend,
        initial_prompt_embeddings: Vec<f32>,
        capacity: GraniteSpeechKvCacheCapacity,
    ) -> Self {
        Self {
            config,
            provider,
            backend,
            initial_prompt_embeddings,
            session: None,
            prompt_len: 0,
            capacity,
        }
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for GraniteSpeechAudioDecodeStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        // Steps after the first advance the existing session by one token.
        if let Some(session) = self.session.as_mut() {
            let new_token = incremental_new_token(session, self.prompt_len, &input)?;
            let logits = session
                .decode_step(new_token, self.provider)
                .map_err(map_step_error(input.step_index, "audio"))?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            });
        }

        // First call: prefill the audio-spliced prompt embeddings into a fresh
        // session (never re-derived from the token ids -- see the struct doc).
        let mut session = GraniteSpeechDecodeSession::new(self.config, self.provider, self.backend)
            .map_err(map_step_error(input.step_index, "audio"))?;
        let logits = session
            .prefill(
                &self.initial_prompt_embeddings,
                input.initial_prompt_tokens.len(),
                self.capacity,
            )
            .map_err(map_step_error(input.step_index, "audio"))?;
        self.prompt_len = input.initial_prompt_tokens.len();
        self.session = Some(session);
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

/// Keep-quantized audio-prompt step executor that drives a **cross-request
/// resident** [`GraniteSpeechDecodeSession`] (owned by
/// `executor::GraniteSpeechPreparedRuntime`, checked out from the admitted
/// resident actor pool) instead of building one per request. The session's heavy
/// state -- the graph runner, the mmap'd loaded weight context, and its
/// zero-copy bound decoder weights -- is already built and reused; this
/// executor only prefills the per-request audio-spliced prompt on the first
/// step (the session was released to `prefilled = false` with empty K/V before
/// being cached, so `prefill` starts clean) and advances the KV cache one token
/// per subsequent step. `embedding_table` is the compact mmap-backed row
/// gatherer borrowed disjointly from the same prepared runtime as `session`.
/// Only one generated-token row is materialized per step.
pub(crate) struct GraniteSpeechResidentAudioDecodeStepExecutor<'s> {
    session: &'s mut GraniteSpeechDecodeSession,
    embedding_table: &'s MappedTokenEmbeddingTable,
    initial_prompt_token_ids: Vec<u32>,
    initial_audio_embeddings: Vec<f32>,
    prompt_len: usize,
    prefilled: bool,
    capacity: GraniteSpeechKvCacheCapacity,
}

impl<'s> GraniteSpeechResidentAudioDecodeStepExecutor<'s> {
    pub(crate) fn new(
        session: &'s mut GraniteSpeechDecodeSession,
        embedding_table: &'s MappedTokenEmbeddingTable,
        initial_prompt_token_ids: Vec<u32>,
        initial_audio_embeddings: Vec<f32>,
        capacity: GraniteSpeechKvCacheCapacity,
    ) -> Self {
        Self {
            session,
            embedding_table,
            initial_prompt_token_ids,
            initial_audio_embeddings,
            prompt_len: 0,
            prefilled: false,
            capacity,
        }
    }
}

impl Seq2SeqGreedyDecodeStepExecutor for GraniteSpeechResidentAudioDecodeStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        // Steps after the first advance the resident session by one token.
        if self.prefilled {
            let new_token = incremental_new_token(self.session, self.prompt_len, &input)?;
            if let Some(output) = self
                .session
                .decode_step_from_token_id(new_token)
                .map_err(map_step_error(input.step_index, "audio"))?
            {
                return Ok(output);
            }
            let embedding = self
                .embedding_table
                .gather_rows(&[new_token])
                .map_err(map_embedding_error(input.step_index))?;
            let logits = self
                .session
                .decode_step_from_embedding(&embedding)
                .map_err(map_step_error(input.step_index, "audio"))?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            });
        }

        // First call: the accelerated resident path gathers token rows and
        // splices audio rows in the same graph that seeds K/V. CPU/scheduler
        // runners retain the exact mapped-table host oracle as a lazy fallback.
        if input.initial_prompt_tokens != self.initial_prompt_token_ids {
            return Err(Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "granite-speech initial prompt tokens changed before prefill".to_string(),
            });
        }
        let audio_positions = self
            .initial_prompt_token_ids
            .iter()
            .enumerate()
            .filter_map(|(position, &token_id)| {
                (token_id == GRANITE_SPEECH_AUDIO_TOKEN_ID).then_some(position)
            })
            .collect::<Vec<_>>();
        let output = match self
            .session
            .prefill_token_ids_with_audio(
                &self.initial_prompt_token_ids,
                &self.initial_audio_embeddings,
                &audio_positions,
                self.capacity,
            )
            .map_err(map_step_error(input.step_index, "audio"))?
        {
            Some(output) => output,
            None => {
                let prompt_embeddings = materialize_audio_prompt_embeddings_from_mapped_table(
                    self.session.config(),
                    self.embedding_table,
                    &self.initial_prompt_token_ids,
                    &self.initial_audio_embeddings,
                )
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: format!(
                        "granite-speech audio decoder step {} prompt embedding fallback failed: {error}",
                        input.step_index
                    ),
                })?;
                let logits = self
                    .session
                    .prefill(
                        &prompt_embeddings,
                        self.initial_prompt_token_ids.len(),
                        self.capacity,
                    )
                    .map_err(map_step_error(input.step_index, "audio"))?;
                Seq2SeqGreedyDecodeStepLogitsOutput {
                    logits,
                    greedy_token_hint: None,
                }
            }
        };
        self.prompt_len = self.initial_prompt_token_ids.len();
        self.prefilled = true;
        Ok(output)
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

    const SOURCE_ROOT_ENV: &str = "OPENASR_GRANITE_SPEECH_SOURCE_ROOT";
    const GOLDEN_ROOT_ENV: &str = "OPENASR_GRANITE_SPEECH_GOLDEN_ROOT";
    const SAMPLES_ROOT_ENV: &str = "OPENASR_GRANITE_SPEECH_SAMPLES_ROOT";

    fn source_root() -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var_os(SOURCE_ROOT_ENV)?);
        path.join("model.safetensors.index.json")
            .exists()
            .then_some(path)
    }

    fn golden_root() -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var_os(GOLDEN_ROOT_ENV)?);
        path.is_dir().then_some(path)
    }

    fn sample_wav(name: &str) -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var_os(SAMPLES_ROOT_ENV)?).join(name);
        path.is_file().then_some(path)
    }

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
        let Some(source_root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let Some(golden_root) = golden_root() else {
            eprintln!("skip: set {GOLDEN_ROOT_ENV} to a local granite-speech golden dir");
            return;
        };

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
        let logical_positions =
            crate::capacity::decode_schedule::greedy_self_kv_positions(prompt.len(), 16)
                .expect("test decode schedule");
        let kv_capacity = GraniteSpeechKvCacheCapacity::new(logical_positions, logical_positions)
            .expect("test KV capacity");

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

        let mut executor = GraniteSpeechDecodeStepExecutor::new(
            config,
            &weights,
            GgmlCpuGraphBackend::Cpu,
            kv_capacity,
        );

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
            &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
            None,
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

    fn load_wav_pcm16_mono_f32(path: &Path) -> Vec<f32> {
        let wav_bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let mut cursor = 12usize;
        let mut data_offset = None;
        let mut data_len = 0usize;
        while cursor + 8 <= wav_bytes.len() {
            let chunk_id = &wav_bytes[cursor..cursor + 4];
            let chunk_len =
                u32::from_le_bytes(wav_bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
            if chunk_id == b"data" {
                data_offset = Some(cursor + 8);
                data_len = chunk_len;
                break;
            }
            cursor += 8 + chunk_len + (chunk_len % 2);
        }
        let data_offset = data_offset.expect("wav data chunk");
        wav_bytes[data_offset..data_offset + data_len]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32 / 32768.0)
            .collect()
    }

    /// Full audio-to-text pipeline: frontend -> encoder -> Q-Former projector
    /// -> audio-splice prompt assembly -> greedy decode through the shared
    /// driver -> BPE text decode. `question` is the ASR instruction text
    /// (optionally with a `Keywords: ...` suffix for KWB, matching the HF
    /// model card's documented prompt format).
    fn transcribe_wav(
        source_root: &Path,
        wav_path: &Path,
        question: &str,
        max_generated_tokens: usize,
    ) -> String {
        let samples = load_wav_pcm16_mono_f32(wav_path);

        let frontend = super::super::frontend::GraniteSpeechMelFrontend::new();
        let (features, frames) = frontend.extract(&samples).expect("frontend");

        let encoder_weights = load_safetensors_prefixed(source_root, "encoder.");
        let encoder_config =
            super::super::encoder_graph::GraniteSpeechEncoderConfig::granite_speech_4_1_2b();
        let encoder_output = super::super::encoder_graph::encode(
            &encoder_config,
            &encoder_weights,
            &features,
            frames,
            GgmlCpuGraphBackend::Cpu,
            false,
        )
        .expect("encode");

        let projector_weights = load_safetensors_prefixed(source_root, "projector.");
        let projector_config =
            super::super::qformer::GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let projector_output = super::super::qformer::project(
            &projector_config,
            &projector_weights,
            &encoder_output.encoder_out,
            encoder_output.frames,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("project");

        let decoder_weights = load_safetensors_prefixed(source_root, "language_model.");
        let decoder_config = GraniteSpeechDecoderConfig::granite_speech_4_1_2b();

        let tokenizer =
            super::super::tokenizer::GraniteSpeechTokenizer::from_source_files(source_root)
                .expect("tokenizer");

        let prompt_text = format!(
            "USER: {}{question}\n ASSISTANT:",
            super::super::prompt::GRANITE_SPEECH_AUDIO_TOKEN
        );
        let (prompt_token_ids, prompt_embeddings) =
            super::super::prompt::build_audio_prompt_embeddings(
                &decoder_config,
                &decoder_weights,
                &tokenizer,
                &prompt_text,
                &projector_output.projected,
                projector_output.tokens,
            )
            .expect("build prompt");

        let eot_token_id = 100_257u32;
        let logical_positions = crate::capacity::decode_schedule::greedy_self_kv_positions(
            prompt_token_ids.len(),
            max_generated_tokens,
        )
        .expect("test decode schedule");
        let kv_capacity = GraniteSpeechKvCacheCapacity::new(logical_positions, logical_positions)
            .expect("test KV capacity");
        let decode_config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: prompt_token_ids,
            eot_token_id,
            stop_token_ids: vec![],
            vocab_size: decoder_config.vocab_size,
            max_generated_tokens,
            suppress_first_step_token_ids: vec![],
            suppress_token_ids: vec![],
            phrase_biases: vec![],
        };

        let mut executor = GraniteSpeechAudioDecodeStepExecutor::new(
            decoder_config,
            &decoder_weights,
            GgmlCpuGraphBackend::Cpu,
            prompt_embeddings,
            kv_capacity,
        );

        let decode_text_token_ids =
            |token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> {
                tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                })
            };

        let result = run_seq2seq_greedy_decode_loop_with_adapter_v0(
            &decode_config,
            &mut executor,
            &decode_text_token_ids,
            |error| error,
            |error| error,
            &|text| text,
            &mut |_step, _token, _eot| {},
            &mut |_step, _logits| {},
            &std::sync::Arc::new(crate::api::backend::TranscriptionControl::new()),
            None,
        )
        .expect("greedy decode");

        result.text.trim().to_string()
    }

    /// Compares the final transcribed text against the official llama.cpp
    /// reference (see the granite-speech integration notes): en_short.wav ->
    /// "the quick brown fox jumps over the lazy dog near the old bridge".
    /// `#[ignore]`: needs the local 4.6GB checkpoint.
    #[test]
    #[ignore = "requires local 4.6GB granite-speech-4.1-2b weights under tmp/ (not committed)"]
    fn granite_speech_end_to_end_transcribes_en_short() {
        let Some(source_root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let Some(wav_path) = sample_wav("en_short.wav") else {
            eprintln!("skip: set {SAMPLES_ROOT_ENV} to a local granite-speech samples dir");
            return;
        };
        let text = transcribe_wav(
            &source_root,
            &wav_path,
            "can you transcribe the speech into a written format?",
            40,
        );
        println!("== Granite Speech end-to-end (en_short.wav) ==");
        println!("transcribed text: {text:?}");
        assert_eq!(
            text, "the quick brown fox jumps over the lazy dog near the old bridge",
            "end-to-end transcription must match the llama.cpp Q8 reference text"
        );
    }

    /// Japanese sample (kanji + katakana): ja_short.wav -> llama.cpp reference
    /// "東京タワーの近くにあるカフェでコーヒーを飲みながら新聞を読みました"
    /// (Q8 reference dropped the input's punctuation; text otherwise exact).
    #[test]
    #[ignore = "requires local 4.6GB granite-speech-4.1-2b weights under tmp/ (not committed)"]
    fn granite_speech_end_to_end_transcribes_ja_short() {
        let Some(source_root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let Some(wav_path) = sample_wav("ja_short.wav") else {
            eprintln!("skip: set {SAMPLES_ROOT_ENV} to a local granite-speech samples dir");
            return;
        };
        let text = transcribe_wav(
            &source_root,
            &wav_path,
            "can you transcribe the speech into a written format?",
            60,
        );
        println!("== Granite Speech end-to-end (ja_short.wav) ==");
        println!("transcribed text: {text:?}");
        assert_eq!(
            text, "東京タワーの近くにあるカフェでコーヒーを飲みながら新聞を読みました",
            "end-to-end Japanese transcription must match the llama.cpp Q8 reference text"
        );
    }

    /// Keyword-list-biasing (KWB) pair: kwb_test.wav, "Xiomara Okonkwo-Yeltsin"
    /// / "Zylenthrax". Without the `Keywords:` prompt suffix the name garbles
    /// (matches the llama.cpp reference exactly); with it, the name comes out
    /// right -- reproducing the "add keywords -> name corrects" behavior the
    /// coordinator asked to confirm.
    #[test]
    #[ignore = "requires local 4.6GB granite-speech-4.1-2b weights under tmp/ (not committed)"]
    fn granite_speech_keyword_list_biasing_corrects_name() {
        let Some(source_root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let Some(wav_path) = sample_wav("kwb_test.wav") else {
            eprintln!("skip: set {SAMPLES_ROOT_ENV} to a local granite-speech samples dir");
            return;
        };

        let without_kwb = transcribe_wav(
            &source_root,
            &wav_path,
            "transcribe the speech to text.",
            40,
        );
        println!("== Granite Speech KWB pair (kwb_test.wav) ==");
        println!("without Keywords: {without_kwb:?}");
        assert_eq!(
            without_kwb,
            "please schedule a meeting with ceo mara akonkwoyeltsin about the xylenthrax project",
            "no-KWB baseline must match the llama.cpp Q8 reference text"
        );

        let with_kwb = transcribe_wav(
            &source_root,
            &wav_path,
            "transcribe the speech to text. Keywords: Xiomara Okonkwo-Yeltsin, Zylenthrax",
            40,
        );
        println!("with Keywords:    {with_kwb:?}");
        assert!(
            with_kwb.contains("xiomara okonkwo-yeltsin"),
            "KWB prompt must correct the garbled name (got {with_kwb:?})"
        );
    }
}
