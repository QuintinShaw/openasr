//! firered-llm dedicated executor: fbank+CMVN [`frontend`](super::super::firered_aed::frontend)
//! -> the parity-verified Conformer encoder
//! [`encoder_graph`](super::super::firered_aed::encoder_graph) (both reused
//! byte-for-byte from `firered_aed` -- architecturally identical, see
//! `package_import`'s module doc) -> the 2x frame-stacking [`adapter_graph`]
//! -> ChatML+`<speech>` splice ([`decode_prompt`] +
//! `qwen::build_qwen3_prompt_embeddings_with_audio_splice`) -> Qwen2
//! [`llm_transformer`] prefill/decode, driven through the ONE shared greedy
//! decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a
//! [`Seq2SeqGreedyDecodeStepExecutor`] impl below -- never a hand-rolled
//! argmax loop (the repo's `model-integration-shared-driver` invariant, see
//! `AGENTS.md`).

// Module-wide (not narrowed to individual items): matches every other model
// family's dedicated executor in this crate (e.g. `firered_aed::executor`).
// `FireRedLlmGgmlExecutor` is reached only through the registries in
// `executor_component_registry.rs` / `builtin_execution_dispatch.rs` and its
// error variants only through `#[cfg(test)]` fixtures, both invisible to
// per-item `dead_code` analysis. Narrowing this file alone would diverge from
// the established per-family convention without a matching crate-wide pass.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_LLM_DECODE_POLICY_ID;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendOverrideGuard, RequestBackendPreference,
    install_request_backend_override, request_backend_override,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::firered_aed::encoder_graph::FireRedEncoderGraphRuntime;
use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
use crate::models::ggml_asr_executor::{
    GgmlAsrBackendPreference, GgmlAsrExecutionError, GgmlAsrExecutionRequest,
    GgmlAsrExecutionResult, GgmlAsrExecutor, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, build_qwen3_prompt_embeddings_with_audio_splice,
};
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, current_unload_generation, take_generation_tagged,
};

use super::adapter_graph::FireRedLlmAdapterGraphRuntime;
use super::decode_prompt::build_firered_llm_decode_prompt;
use super::llm_transformer::FireRedLlmDecoderRuntime;
use super::runtime_contract::{
    parse_firered_llm_adapter_metadata, parse_firered_llm_decoder_metadata,
    parse_firered_llm_encoder_metadata,
};
use super::tokenizer::FireRedLlmTokenizer;

/// Resident decoder cache (mirrors qwen's `QWEN_WHOLE_DECODER_BY_KEY` design
/// S4, `models::qwen::ggml_executor`): `FireRedLlmDecoderRuntime` owns the
/// Qwen2 whole-decoder graph runner, its device-uploaded layer weights, the
/// logits head, and the token-embedding table -- all identical across every
/// `execute()` call against the same pack on the same backend. Without this,
/// every request paid a full decoder-runtime rebuild (~1.8-2.0s measured,
/// `docs/model-audits/firered2-llm.md` SS3) purely to re-derive state that
/// does not change between requests. Keyed by (pack path, backend): unlike
/// qwen this family has no LoRA/adapter input, so the key omits qwen's third
/// (adapter fingerprint) component.
///
/// This reuses the same session/model split transcribe.cpp uses for its own
/// Qwen-family LLM decoder (`references/transcribe.cpp@b6a6acad`,
/// `src/arch/qwen3_asr/model.cpp`: weights load once into `transcribe_model`,
/// a session's KV cache grows-to-fit and is reused across `run()` calls
/// instead of being rebuilt) -- transcribe.cpp keeps the resident state in a
/// caller-owned session object rather than a thread-local cache; this crate's
/// dispatch is a stateless `&self` executor called per request, so the
/// thread-local + generation-tag mechanism below is what plays that role
/// here.
///
/// A plain `HashMap` (not the shared bounded LRU): the key does not explode
/// per audio chunk (one entry is built and reused across a whole longform
/// run for a given pack), so there is no unbounded-growth hazard to bound.
/// Entries are tagged with the idle-unload generation they were built under
/// (`thread_local_runtime_cache`): the `idle_unload` reaper cannot reach this
/// TLS from its own thread, so `take_cached_decoder_runtime` discards any
/// decoder built before the last unload instead of handing it back out --
/// this is how the resident 8B decoder becomes evictable under memory
/// pressure without a bespoke eviction policy.
type FireRedLlmDecoderCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static FIRERED_LLM_DECODER_BY_KEY: RefCell<HashMap<FireRedLlmDecoderCacheKey, (u64, FireRedLlmDecoderRuntime)>> =
        RefCell::new(HashMap::new());
}

fn take_cached_decoder_runtime(
    key: &FireRedLlmDecoderCacheKey,
    unload_generation: u64,
) -> Option<FireRedLlmDecoderRuntime> {
    FIRERED_LLM_DECODER_BY_KEY
        .with(|cache| take_generation_tagged(&mut cache.borrow_mut(), key, unload_generation))
}

fn store_cached_decoder_runtime(
    key: FireRedLlmDecoderCacheKey,
    unload_generation: u64,
    decoder: FireRedLlmDecoderRuntime,
) {
    FIRERED_LLM_DECODER_BY_KEY.with(|cache| {
        cache.borrow_mut().insert(key, (unload_generation, decoder));
    });
}

const FIRERED_LLM_EXECUTOR_ID: &str = "firered-llm-ggml-executor-v1";
const FIRERED_LLM_STREAMING_EXECUTOR_ID: &str = "firered-llm-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";
/// Upstream single-utterance hard cap (`fireredasr2` README: "single 40s max
/// input"). The executor fails closed rather than silently truncating or
/// running an out-of-distribution multi-minute prefill; longer audio is the
/// longform slicing orchestrator's job (see `FIRERED_LLM_DECODE_POLICY_ID`'s
/// `ConservativeSeq2SeqV1` longform profile registration).
const FIRERED_LLM_MAX_INPUT_SECONDS: f32 = 40.0;
/// Generous upper bound on generated tokens per utterance -- greedy decode
/// stops at `<|im_end|>` well before this in practice; this is only the
/// fail-closed backstop against a runaway (non-terminating) decode.
const FIRERED_LLM_MAX_GENERATED_TOKENS: usize = 512;

#[derive(Debug, Error)]
enum FireRedLlmExecutorError {
    #[error("firered-llm executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("firered-llm executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("firered-llm runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("firered-llm tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("firered-llm cmvn vectors failed: {reason}")]
    CmvnBuildFailed { reason: String },
    #[error("firered-llm frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("firered-llm audio duration {seconds:.1}s exceeds the upstream {limit:.0}s hard cap")]
    AudioTooLong { seconds: f32, limit: f32 },
    #[error("firered-llm encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("firered-llm adapter failed: {reason}")]
    AdapterGraphFailed { reason: String },
    #[error("firered-llm decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("firered-llm prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("firered-llm decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("firered-llm greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FireRedLlmGgmlExecutor;

/// A no-op phrase-bias/token-source shim: firered-llm's decode policy never
/// consults these (no phrase bias, `seq2seq_stop_token_kind: None` -- eot is
/// supplied directly via `BuiltinSeq2SeqDecodePolicyConfigInput`), so a real
/// implementation would be dead weight. `resolve_builtin_decode_policy`'s
/// config builder still requires the trait object, matching `()`'s existing
/// blanket impl of `BuiltinSeq2SeqDecodePolicyTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

/// Drives `FireRedLlmDecoderRuntime` through the shared greedy loop: the
/// first step (index 0, no generated tokens yet) consumes the pre-built
/// prompt embeddings via one prefill pass; every step after that embeds the
/// last generated token and runs one incremental decode step. Mirrors
/// `qwen::ggml_executor::Qwen3AsrPrefillOnlyGreedyStepExecutor`'s shape.
struct FireRedLlmGreedyStepExecutor<'a> {
    decoder: &'a mut FireRedLlmDecoderRuntime,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    prompt_embeddings: Option<crate::models::qwen::Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
}

impl Seq2SeqGreedyDecodeStepExecutor for FireRedLlmGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if let Some(prompt_embeddings) = self.prompt_embeddings.take() {
            self.cache_prompt_tokens = prompt_embeddings.token_count;
            let logits = self
                .decoder
                .prefill(&prompt_embeddings, &mut self.layer_kv_caches)
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let last_token = input.generated_tokens.last().copied().ok_or_else(|| {
            Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "firered-llm generated token history is unexpectedly empty".to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "firered-llm decode cache position underflowed".to_string(),
            })?;
        let logits = self
            .decoder
            .decode_step(last_token, cache_position, &mut self.layer_kv_caches)
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

impl FireRedLlmGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedLlmExecutorError> {
        let expected_adapter = crate::arch::FIRERED_LLM_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(FireRedLlmExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| FireRedLlmExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;

        let encoder_metadata =
            parse_firered_llm_encoder_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let adapter_metadata =
            parse_firered_llm_adapter_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let decoder_metadata =
            parse_firered_llm_decoder_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokenizer = FireRedLlmTokenizer::from_gguf_metadata(&preflight.metadata).map_err(
            |error: NativeAsrError| FireRedLlmExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            },
        )?;

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > FIRERED_LLM_MAX_INPUT_SECONDS {
            return Err(FireRedLlmExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: FIRERED_LLM_MAX_INPUT_SECONDS,
            });
        }

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
            FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            }
        })?;
        let feature_dim_shape = [encoder_metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;

        let frontend = FireRedFbankFrontend::new();
        let mut features =
            frontend
                .compute(samples)
                .map_err(|error| FireRedLlmExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).map_err(
            |error| FireRedLlmExecutorError::FrontendFailed {
                reason: error.to_string(),
            },
        )?;

        let runtime_path = preflight.runtime_source.path();
        let mut encoder_runtime = FireRedEncoderGraphRuntime::new(runtime_path, encoder_metadata)
            .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;
        let encoder_output = encoder_runtime
            .encode(&features.data, features.n_frames)
            .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;

        let adapter_profile_started_at = std::time::Instant::now();
        let mut adapter_runtime =
            FireRedLlmAdapterGraphRuntime::new(runtime_path).map_err(|error| {
                FireRedLlmExecutorError::AdapterGraphFailed {
                    reason: error.to_string(),
                }
            })?;
        let (speech_rows, speech_frame_count) = adapter_runtime
            .run(
                &encoder_output.rows,
                encoder_output.frame_count,
                encoder_metadata.d_model,
                adapter_metadata.downsample_rate,
                adapter_metadata.llm_dim,
            )
            .map_err(|error| FireRedLlmExecutorError::AdapterGraphFailed {
                reason: error.to_string(),
            })?;
        // Opt-in perf diagnostic, same gate/shape as the decoder_backend line
        // below (mirrors the qwen `OPENASR_HYMT2_PROFILE` precedent): the
        // adapter stage regressed to 2868ms/18.4% of `execute` on the naive
        // scalar-dequant host implementation this ggml graph replaced (see
        // this module's doc comment), so it earns the same always-available
        // (opt-in) timing visibility as the decoder backend choice.
        if std::env::var_os("OPENASR_FIRERED_LLM_PROFILE").is_some() {
            eprintln!(
                "OPENASR_FIRERED_LLM_PROFILE stage=adapter ms={:.2}",
                adapter_profile_started_at.elapsed().as_secs_f64() * 1000.0
            );
        }

        let decode_prompt = build_firered_llm_decode_prompt(&tokenizer, speech_frame_count)
            .map_err(|error| FireRedLlmExecutorError::DecodePromptFailed {
                reason: error.to_string(),
            })?;

        // Pin the memory-dominant 7B decoder to a backend that fits this host
        // (see `resolve_decoder_backend_override`). Held for the whole decode so
        // the graph runner built here -- and reused every step -- stays on the
        // chosen backend; the encoder/adapter above already ran and are
        // unaffected.
        let _decoder_backend_override =
            resolve_decoder_backend_override(runtime_path, request.backend_preference);
        // Resolved AFTER installing the override guard, so the cache key
        // reflects the backend the decoder actually builds on (matches
        // qwen's `WholeDecoderCacheKey` ordering).
        let decoder_cache_key: FireRedLlmDecoderCacheKey = (
            canonical_runtime_cache_path(runtime_path),
            GgmlCpuGraphConfig::resolve_runtime_backend(),
        );
        // Sampled before the cache take and reused for the store-back below:
        // if the idle-unload reaper bumps the generation while this decode is
        // in flight, the decoder goes back tagged with the pre-unload
        // generation and the *next* take discards it, so an unload can never
        // be lost to an overlapping decode (mirrors qwen's
        // `ggml_executor::execute_inner`).
        let unload_generation = current_unload_generation();
        let firered_llm_profile_enabled = std::env::var_os("OPENASR_FIRERED_LLM_PROFILE").is_some();
        let mut decoder = match take_cached_decoder_runtime(&decoder_cache_key, unload_generation) {
            // Resident hit: layer weights already uploaded to the device and
            // the reuse graph already built -- skip the ~1.8-2.0s rebuild.
            Some(decoder) => {
                if firered_llm_profile_enabled {
                    eprintln!("OPENASR_FIRERED_LLM_PROFILE stage=decoder_cache_hit");
                }
                decoder
            }
            None => {
                let decoder = FireRedLlmDecoderRuntime::new(runtime_path, decoder_metadata)
                    .map_err(|error| FireRedLlmExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    })?;
                if firered_llm_profile_enabled {
                    eprintln!("OPENASR_FIRERED_LLM_PROFILE stage=decoder_cache_miss_init");
                }
                decoder
            }
        };
        // Opt-in perf diagnostic (mirrors the qwen `OPENASR_HYMT2_PROFILE`
        // backend log line): confirms which backend the 7B Qwen2 decoder graph
        // actually resolved to (Metal vs CPU) without asserting a host-dependent
        // timing number in shipping code.
        if firered_llm_profile_enabled {
            eprintln!(
                "OPENASR_FIRERED_LLM_PROFILE decoder_backend={}",
                decoder.backend_label()
            );
        }
        let token_rows_len = decode_prompt.token_ids.len() * decoder_metadata.d_model;
        let mut token_rows = Vec::with_capacity(token_rows_len);
        for &token_id in &decode_prompt.token_ids {
            let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                FireRedLlmExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
            token_rows.extend_from_slice(&row);
        }
        let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
            &decode_prompt,
            decoder_metadata.d_model,
            &token_rows,
            &speech_rows,
        )
        .map_err(|error| FireRedLlmExecutorError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;

        // Request-sized, not the decoder's native 32768-token context: see
        // `FireRedLlmDecoderRuntime::new_kv_caches`'s doc comment for why the
        // fixed reuse-graph span this sizes must stay tight to what this
        // utterance actually needs.
        let layer_kv_caches = decoder.new_kv_caches(
            decode_prompt
                .token_ids
                .len()
                .saturating_add(FIRERED_LLM_MAX_GENERATED_TOKENS),
        );
        let mut step_executor = FireRedLlmGreedyStepExecutor {
            decoder: &mut decoder,
            layer_kv_caches,
            prompt_embeddings: Some(prompt_embeddings),
            cache_prompt_tokens: 0,
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.chatml_im_end_token_id,
            vocab_size: decoder_metadata.vocab_size,
            max_generated_tokens: FIRERED_LLM_MAX_GENERATED_TOKENS,
        };
        let decode_result = run_builtin_seq2seq_decode_policy(
            FIRERED_LLM_DECODE_POLICY_ID,
            &config,
            &NoPhraseBiasTokenSource,
            None,
            &mut step_executor,
            &|token_ids: &[u32]| {
                tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                })
            },
            |error: Seq2SeqGreedyDecodeError| error,
            |error: Seq2SeqGreedyDecodeError| error,
            map_registry_error,
        );
        // Return the resident decoder to the cache for the next chunk /
        // execute() regardless of decode outcome -- its weights + reuse graph
        // stay valid either way (mirrors qwen's `store_cached_whole_decoder`
        // call site). `step_executor` is not used again after this point, so
        // its `&mut decoder` borrow ends here under NLL and `decoder` (owned
        // locally) is free to move into the cache.
        store_cached_decoder_runtime(decoder_cache_key, unload_generation, decoder);
        let result =
            decode_result.map_err(|error| FireRedLlmExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?;

        let text = result.text.trim().to_string();
        let transcription = Transcription {
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            text,
            longform: None,
            language: None,
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
        })
    }
}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

/// Decide which backend the 7B Qwen2 decoder stage should build on, and return
/// an override guard (kept alive across the decoder's construction + decode)
/// that pins it there.
///
/// The decoder is the memory-dominant stage: at fp16/q8_0 its weights plus the
/// f16 embedding, the growing KV cache and the ggml compute buffers overrun a
/// small unified-memory GPU working set, so a Metal command buffer fails with
/// `kIOGPUCommandBufferCallbackErrorOutOfMemory` partway through decode (M1/16GB
/// measured: q8_0 OOMs, q4_k -- ~4.9GB peak RSS -- fits comfortably). Rather
/// than let Auto crash mid-decode or silently fall back to a 30x-slower path, we
/// pick the backend up front:
///
/// - An explicit `Accelerated`/`CpuOnly` request always wins (the product rule
///   that a user's explicit hardware choice is never second-guessed) -- we just
///   honor it, which also makes the request preference authoritative on the
///   direct-`execute` path that does not go through the dispatch wrapper.
/// - Under `Auto`, follow the process/env default, but if the pack cannot fit
///   this host (a conservative `pack_bytes * 2 <= total_RAM` unified-memory
///   budget: weights resident on the GPU plus comparable headroom for the host
///   embedding, KV, compute buffers and the OS) and the default would be a
///   GPU-class backend, fall the decoder back to CPU and print a one-line
///   not-real-time notice instead of OOM-ing.
///
/// Returns `None` when no override is needed (Auto that fits / already CPU).
fn resolve_decoder_backend_override(
    runtime_path: &std::path::Path,
    backend_preference: GgmlAsrBackendPreference,
) -> Option<RequestBackendOverrideGuard> {
    match backend_preference {
        GgmlAsrBackendPreference::CpuOnly => Some(install_request_backend_override(Some(
            RequestBackendPreference::CpuOnly,
        ))),
        GgmlAsrBackendPreference::Accelerated => Some(install_request_backend_override(Some(
            RequestBackendPreference::Accelerated,
        ))),
        GgmlAsrBackendPreference::Auto => {
            // An explicit accelerate request installed upstream still wins.
            if matches!(
                request_backend_override(),
                Some(RequestBackendPreference::Accelerated)
            ) {
                return None;
            }
            if !GgmlCpuGraphConfig::resolve_runtime_backend().is_gpu_class() {
                return None;
            }
            // Unknown RAM -> trust the default rather than force CPU.
            let total_ram = crate::host::host_total_memory_bytes()?;
            let pack_bytes = std::fs::metadata(runtime_path).ok()?.len();
            if pack_bytes.saturating_mul(2) <= total_ram {
                return None;
            }
            eprintln!(
                "openasr: the FireRedASR2-LLM 7B decoder pack ({:.1} GB) does not fit this host's \
                 GPU memory budget ({:.1} GB RAM); running the decoder on CPU (transcription will \
                 be slower than real time). Install a lower-quant pack (q4_k) for GPU-accelerated \
                 decode.",
                pack_bytes as f64 / 1.0e9,
                total_ram as f64 / 1.0e9,
            );
            Some(install_request_backend_override(Some(
                RequestBackendPreference::CpuOnly,
            )))
        }
    }
}

impl GgmlAsrExecutor for FireRedLlmGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_LLM_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }
}

impl GgmlAsrStreamingExecutor for FireRedLlmGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_LLM_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FIRERED_LLM_STREAMING_EXECUTOR_ID,
            crate::arch::FIRERED_LLM_GGML_ADAPTER_ID,
            "firered-llm",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FireRedLlmGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;
    use crate::models::ggml_family_registry::firered_llm_runtime_descriptor_v1;

    use super::*;

    /// Points at the real converted pack from T2
    /// (`scratchpad/fr2/T2-report.md`), an ~8.9GB q8_0 `.oasr` NOT committed
    /// to the repo (dev-only artifact, same convention as firered-aed's own
    /// `tmp/firered-out/firered-aed-l-fp16.oasr` golden pack). Loading it
    /// mmaps + touches most of an 8.9GB file plus materializes the ~1GB f16
    /// token-embedding table -- a real memory commitment, not a network
    /// fetch, so this stays `#[ignore]`d and skips silently when absent
    /// (matches firered-aed's own dev-pack test convention) rather than
    /// gating CI on a multi-GB private artifact.
    fn dev_pack_path() -> PathBuf {
        PathBuf::from(
            "/Volumes/QuintinDocument/openasr-dev/tmp-weights/fr2/out/firered2-llm-q8_0.oasr",
        )
    }

    // Pinned to the real dev-pack decode. CPU is the deterministic reference
    // backend, so these goldens request `CpuOnly`; the decode is byte-identical
    // on Metal (verified against the q4_k pack on an M1: Metal:MTL0 vs Cpu:CPU
    // produce the same transcript). The q4_k pack fits Metal comfortably (~4.9GB
    // peak RSS on a 16GB Mac); only the larger fp16/q8_0 packs overrun a small
    // unified-memory GPU, which `resolve_decoder_backend_override` now handles by
    // falling the decoder back to CPU under Auto. JFK is word-for-word correct;
    // the Mandarin sentence is the same non-copyrighted `say -v Tingting`
    // synthesis firered-aed's own golden uses (see that family's `zh_sample.wav`
    // doc comment).
    const GOLDEN_JFK_TEXT: &str = "and so my fellow americans ask not what your country can do \
        for you ask what you can do for your country";

    const GOLDEN_ZH_TEXT: &str = "今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的\
        川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下";

    // Code-switch coverage (first 5s of jfk.wav + first 8s of zh_sample.wav,
    // single <=40s utterance, no longform slicing involved): both languages'
    // ChatML/tokenizer/decode paths share one prefill+decode call here.
    const GOLDEN_EN_ZH_MIXED_TEXT: &str = "and so my fellow americans ask not 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开";

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn en_zh_mixed_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/en_zh_mixed.wav")
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<(String, std::time::Duration, f32)> {
        transcribe_with_pack(dev_pack_path(), wav_path, GgmlAsrBackendPreference::CpuOnly)
    }

    fn transcribe_with_pack(
        pack_path: PathBuf,
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, std::time::Duration, f32)> {
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "firered-llm e2e test",
            "firered-llm e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: firered_llm_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
        };

        let executor = FireRedLlmGgmlExecutor;
        let started_at = Instant::now();
        let result = executor.execute(&request).expect("firered-llm transcribe");
        let elapsed = started_at.elapsed();
        Some((result.transcription.text, elapsed, audio_duration_seconds))
    }

    /// M1 CPU-vs-Metal RTF + peak-RSS AB harness for the FireRedASR2-LLM 7B
    /// decoder across quant rungs. One config per invocation (env-selected) so
    /// `peak_rss_bytes` (process-global `ru_maxrss` high-water) stays isolated;
    /// prints a machine-greppable `FR2_LLM_AB ...` line. Never asserts a timing
    /// number (host-dependent) -- it only measures. Mirrors dolphin's
    /// `dolphin_perf_ab`.
    ///
    /// Env: `OPENASR_FR2_AB_BACKEND=cpu|metal|auto` (default auto),
    /// `OPENASR_FR2_AB_QUANT=q4_k|q8_0|fp16` (default q4_k),
    /// `OPENASR_FR2_AB_CLIP=<wav path>` (default fixtures/zh_sample.wav).
    /// Set `OPENASR_FIRERED_LLM_PROFILE=1` too to also log the resolved
    /// decoder backend.
    #[test]
    #[ignore = "perf AB harness: requires the private dev-only firered2-llm-<quant>.oasr packs \
                under tmp-weights/fr2/out; env-selected backend/quant, prints FR2_LLM_AB + peak RSS"]
    fn firered_llm_perf_ab() {
        let quant = std::env::var("OPENASR_FR2_AB_QUANT").unwrap_or_else(|_| "q4_k".to_string());
        let pack_path = PathBuf::from(format!(
            "/Volumes/QuintinDocument/openasr-dev/tmp-weights/fr2/out/firered2-llm-{quant}.oasr"
        ));
        let backend = match std::env::var("OPENASR_FR2_AB_BACKEND").as_deref() {
            Ok("cpu") => GgmlAsrBackendPreference::CpuOnly,
            Ok("metal") | Ok("gpu") => GgmlAsrBackendPreference::Accelerated,
            _ => GgmlAsrBackendPreference::Auto,
        };
        let clip = std::env::var("OPENASR_FR2_AB_CLIP")
            .map(PathBuf::from)
            .unwrap_or_else(|_| zh_wav_path());

        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_pack(pack_path, clip, backend)
        else {
            return;
        };
        let rtf = elapsed.as_secs_f32() / audio_duration_seconds.max(0.001);
        let peak_rss_mb = crate::metrics::peak_rss_bytes()
            .map(|bytes| bytes as f64 / 1.0e6)
            .unwrap_or(0.0);
        eprintln!(
            "FR2_LLM_AB quant={quant} backend={backend:?} audio={audio_duration_seconds:.2}s \
             elapsed={elapsed:?} RTF={rtf:.3} peak_rss={peak_rss_mb:.0}MB text={text}"
        );
    }

    // T5: promoted from the Stage-4 "prints transcript for manual judgement"
    // probe once a human read the printed transcripts and confirmed JFK is
    // word-for-word correct and the Mandarin sentence is coherent (see the T5
    // report's parity + e2e sections) -- mirrors firered-aed's own
    // `golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_*`
    // promotion history. RTF/elapsed are still logged to stderr (not asserted:
    // wall-clock varies with shared-machine load) so a maintainer re-running
    // this locally still gets the performance signal the old probe printed.
    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; runs the \
                deterministic CPU reference decode (requested via CpuOnly; q8_0 overruns a 16GB \
                Mac's GPU so Auto falls it back to CPU anyway -- q4_k fits Metal, see \
                firered_llm_perf_ab)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_jfk_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(jfk_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; runs the \
                deterministic CPU reference decode (requested via CpuOnly; q8_0 overruns a 16GB \
                Mac's GPU so Auto falls it back to CPU anyway -- q4_k fits Metal, see \
                firered_llm_perf_ab)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_zh_sample_wav() {
        let Some((text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack(zh_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [zh_sample.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_ZH_TEXT);
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; runs the \
                deterministic CPU reference decode (requested via CpuOnly; q8_0 overruns a 16GB \
                Mac's GPU so Auto falls it back to CPU anyway -- q4_k fits Metal, see \
                firered_llm_perf_ab)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_en_zh_mixed_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(en_zh_mixed_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_EN_ZH_MIXED_TEXT);
    }

    /// Resident decoder cache regression: calling `execute()` twice in a row
    /// on the same thread (same pack + backend) must hit the thread-local
    /// `FIRERED_LLM_DECODER_BY_KEY` cache on the second call and still
    /// produce a byte-identical transcript to the first call and to the
    /// dedicated single-call goldens above -- the resident decoder carries no
    /// per-request state across calls that could leak into a later
    /// transcript. Run with `OPENASR_FIRERED_LLM_PROFILE=1 cargo test ...
    /// -- --ignored --nocapture` to also see the
    /// `stage=decoder_cache_miss_init` / `stage=decoder_cache_hit` lines this
    /// test exercises but does not itself assert on (stderr capture of a
    /// specific line is not a stable test signal; the byte-identical output
    /// plus the manual profile run together are the evidence this cache is
    /// wired correctly).
    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; see \
                golden_diff_end_to_end_transcribe_matches_reference_decode_on_jfk_wav for why \
                CpuOnly is the deterministic reference backend here"]
    fn resident_decoder_cache_reuse_across_consecutive_calls_stays_byte_identical() {
        let Some((first_text, _, _)) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        let Some((second_text, _, _)) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        assert_eq!(first_text, GOLDEN_JFK_TEXT);
        assert_eq!(
            second_text, GOLDEN_JFK_TEXT,
            "second execute() (a resident-decoder cache hit) must match the first \
             (cache-miss/build) call byte-for-byte"
        );
    }
}
