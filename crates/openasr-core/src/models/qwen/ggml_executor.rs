use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use thiserror::Error;

use super::Qwen3AsrTokenizer;
use super::audio_encoder::{
    Qwen3AsrAudioEncoderError, Qwen3AsrAudioEncoderRuntime, Qwen3AsrAudioEncoderWeights,
};
use super::batched_decode::{
    Qwen3AsrServeBatchConfig, Qwen3AsrServeBatchJob, submit_qwen_serve_batch_job,
};
use super::decode_prompt::{Qwen3AsrDecodePromptError, build_qwen3_decode_prompt};
use super::frontend::{
    Qwen3AsrMelFeatures, Qwen3AsrMelFrontendError, Qwen3AsrMelFrontendPlan,
    qwen3_mel_features_from_prepared_audio,
};
use super::greedy_decode::{Qwen3AsrGreedyDecodeError, run_qwen3_greedy_decode_loop};
use super::kv_cache::Qwen3AsrLayerKvCacheState;
use super::llm_prefill::{Qwen3AsrLlmPrefillInputError, build_qwen3_llm_prefill_input};
use super::llm_transformer::{
    Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmWholeDecoderGraphExecutor,
};
use super::lora::{qwen_adapter_cache_fingerprint, resolve_qwen_lora_adapter};
use super::prepared_runtime::{Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError};
use super::prompt_embedding::{
    Qwen3AsrPromptEmbeddingError, build_qwen3_prompt_embeddings_with_audio_splice,
};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::token_embedding::Qwen3AsrTokenEmbeddingError;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::hparams::{QWEN3_AUDIO_LAYERS_KEY, QWEN3_LLM_LAYERS_KEY};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{OpenAsrArchitectureRegistry, QWEN3_ASR_GGML_ARCHITECTURE_ID};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, env_var_truthy};
use crate::models::decode_policy_component_registry::{
    BuiltinSeq2SeqDecodePolicyConfigInput, build_builtin_seq2seq_decode_policy_config,
    resolve_builtin_decode_policy,
};
use crate::models::decode_token_history::context_window_budget;
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ, build_seq2seq_streaming_session,
};
use crate::models::runtime_prepared_registry::{
    BuiltinPreparedRuntimeCache, BuiltinPreparedRuntimeRegistryError,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::seq2seq_word_timestamps::seq2seq_word_timestamps_from_generated_tokens;
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, with_thread_local_cached_mut_by_key,
};
use crate::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrPreparedAudio, GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
    NativeAsrSession, QWEN3_ASR_GGML_ADAPTER_ID, Segment, Transcription,
};

#[cfg(test)]
use super::runtime_contract::parse_qwen3_execution_metadata;
#[cfg(test)]
use crate::GgmlAsrRuntimeSourcePreflight;
#[cfg(test)]
use crate::models::runtime_prepared_registry::build_builtin_prepared_runtime;

/// Resident whole-decoder cache (design S4): the LLM whole-decoder owns the
/// runner + the device-uploaded layer weights + the reuse graph, all of which
/// are identical across every `execute()` for the same pack. Without this, the
/// longform path rebuilt the decoder and re-uploaded every layer's weights to
/// the GPU on every chunk (the "~1GB re-upload per chunk" cost). Keyed by
/// (pack path, backend, adapter fingerprint) so a CPU-validate then GPU-run in
/// the same process does not reuse a backend-mismatched decoder, and a run with
/// an adapter does not reuse a graph built without one (correctness: the LoRA
/// tensors are baked into the arena at construction time).
type WholeDecoderCacheKey = (PathBuf, GgmlCpuGraphBackend, String);
type AudioEncoderCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static QWEN_WHOLE_DECODER_BY_KEY: RefCell<HashMap<WholeDecoderCacheKey, Qwen3AsrLlmWholeDecoderGraphExecutor>> =
        RefCell::new(HashMap::new());
    static QWEN_AUDIO_ENCODER_BY_KEY: RefCell<HashMap<AudioEncoderCacheKey, Qwen3AsrAudioEncoderRuntime>> =
        RefCell::new(HashMap::new());
}

fn take_cached_whole_decoder(
    key: &WholeDecoderCacheKey,
) -> Option<Qwen3AsrLlmWholeDecoderGraphExecutor> {
    QWEN_WHOLE_DECODER_BY_KEY.with(|cache| cache.borrow_mut().remove(key))
}

fn store_cached_whole_decoder(
    key: WholeDecoderCacheKey,
    decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
) {
    QWEN_WHOLE_DECODER_BY_KEY.with(|cache| {
        cache.borrow_mut().insert(key, decoder);
    });
}

fn encode_qwen_audio_embeddings_cached(
    key: AudioEncoderCacheKey,
    runtime_source_path: &std::path::Path,
    audio_encoder_weights: &Qwen3AsrAudioEncoderWeights,
    metadata: Qwen3AsrExecutionMetadata,
    mel_features: &Qwen3AsrMelFeatures,
) -> Result<super::audio_encoder::Qwen3AsrAudioEncoderOutput, Qwen3AsrAudioEncoderError> {
    with_thread_local_cached_mut_by_key(
        &QWEN_AUDIO_ENCODER_BY_KEY,
        key,
        || Qwen3AsrAudioEncoderRuntime::new(Some(runtime_source_path)),
        |runtime| runtime.encode(audio_encoder_weights, metadata, mel_features),
    )
}

const QWEN3_EXECUTOR_ID: &str = "qwen3-asr-ggml-executor-v1";
const QWEN3_STREAMING_EXECUTOR_ID: &str = "qwen3-asr-ggml-snapshot-streaming-executor-v1";
const QWEN3_DECODE_MIN_GENERATED_TOKENS: usize = 128;
const QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND: usize = 12;
const QWEN3_DECODE_TOKEN_BUDGET_MARGIN: usize = 32;
const QWEN3_DECODE_PROFILE_ENV: &str = "OPENASR_QWEN_DECODE_PROFILE";

fn qwen_decode_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_var_truthy(QWEN3_DECODE_PROFILE_ENV))
}

fn qwen_decode_profile_start() -> Option<Instant> {
    qwen_decode_profile_enabled().then(Instant::now)
}

fn qwen_decode_profile_log(stage: &str, started_at: Instant) {
    eprintln!(
        "openasr_qwen_decode_profile: stage={} total_us={}",
        stage,
        started_at.elapsed().as_micros()
    );
}

fn qwen_decode_profile_log_opt(stage: &str, started_at: Option<Instant>) {
    if let Some(started_at) = started_at {
        qwen_decode_profile_log(stage, started_at);
    }
}

fn qwen_decode_profile_log_prefill_chunk(
    position_offset: usize,
    chunk_len: usize,
    started_at: Option<Instant>,
) {
    if let Some(started_at) = started_at {
        eprintln!(
            "openasr_qwen_decode_profile: stage=prefill_chunk position_offset={} chunk_len={} total_us={}",
            position_offset,
            chunk_len,
            started_at.elapsed().as_micros()
        );
    }
}

#[derive(Debug, Error)]
enum Qwen3AsrGgmlExecutorError {
    #[error("qwen3-asr ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("qwen3-asr runtime contract check failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("qwen3-asr runtime metadata read failed: {reason}")]
    RuntimeMetadataReadFailed { reason: String },
    #[error("qwen3-asr decode prompt construction failed: {reason}")]
    DecodePromptConstructionFailed { reason: String },
    #[error("qwen3-asr mel frontend failed: {reason}")]
    MelFrontendFailed { reason: String },
    #[error("qwen3-asr audio encoder failed: {reason}")]
    AudioEncoderFailed { reason: String },
    #[error("qwen3-asr token embedding prefill failed: {reason}")]
    TokenEmbeddingPrefillFailed { reason: String },
    #[error("qwen3-asr prompt embedding assembly failed: {reason}")]
    PromptEmbeddingAssemblyFailed { reason: String },
    #[error("qwen3-asr llm prefill input assembly failed: {reason}")]
    LlmPrefillInputAssemblyFailed { reason: String },
    #[error("qwen3-asr greedy decode loop failed: {reason}")]
    GreedyDecodeFailed { reason: String },
    #[error("qwen3-asr decode token budget is unavailable: {reason}")]
    DecodeBudgetUnavailable { reason: String },
    #[error("qwen3-asr llm logits head failed: {reason}")]
    LlmLogitsHeadFailed { reason: String },
    #[error("qwen3-asr llm transformer decode step failed: {reason}")]
    LlmTransformerDecodeStepFailed { reason: String },
    #[error(
        "qwen3-asr ggml executor currently supports only {expected_sample_rate_hz}Hz mono input, got sample_rate={sample_rate_hz} channels={channels}"
    )]
    UnsupportedInputShape {
        expected_sample_rate_hz: u32,
        sample_rate_hz: u32,
        channels: u16,
    },
    /// Carries a transient serve-batch failure (queue full / owner gone / reply
    /// timeout) through to the `execute` trait boundary so it can become a
    /// retryable HTTP status instead of a generic 500.
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Qwen3AsrGgmlExecutor {
    runtime_cache_by_path: BuiltinPreparedRuntimeCache,
}

impl Qwen3AsrGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        if request.selected_family.adapter_id != QWEN3_ASR_GGML_ADAPTER_ID {
            return Err(Qwen3AsrGgmlExecutorError::AdapterMismatch {
                expected: QWEN3_ASR_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let profile_started_at = qwen_decode_profile_start();
        let preflight_started_at = qwen_decode_profile_start();
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(
                |error| Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed {
                    reason: error.to_string(),
                },
            )?;
        qwen_decode_profile_log_opt("runtime_preflight", preflight_started_at);
        let prepared_runtime_started_at = qwen_decode_profile_start();
        let result = self
            .runtime_cache_by_path
            .with_qwen3_asr_runtime_for_preflight(
                request.selected_family.model_architecture,
                preflight.as_ref(),
                map_prepared_runtime_registry_error,
                qwen_runtime_cache_poisoned,
                || Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                    reason: format!(
                        "prepared runtime registry returned non-qwen runtime for architecture '{}'",
                        request.selected_family.model_architecture
                    ),
                },
                |prepared_runtime| {
                    self.execute_with_prepared_runtime(request, prepared_runtime, skip_serve_batch)
                },
            );
        qwen_decode_profile_log_opt("prepared_runtime_and_execute", prepared_runtime_started_at);
        qwen_decode_profile_log_opt("execute_inner_total", profile_started_at);
        result
    }

    fn execute_with_prepared_runtime(
        &self,
        request: &GgmlAsrExecutionRequest,
        prepared_runtime: &Qwen3AsrPreparedRuntime,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        self.execute_with_runtime_assets(
            request,
            prepared_runtime.metadata,
            prepared_runtime.tokenizer.as_ref(),
            &prepared_runtime.mel_frontend_plan,
            &prepared_runtime.audio_encoder_weights,
            prepared_runtime.token_embedding_table.clone(),
            prepared_runtime.logits_head.clone(),
            prepared_runtime.layer_attention_projections.clone(),
            skip_serve_batch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with_runtime_assets(
        &self,
        request: &GgmlAsrExecutionRequest,
        metadata: Qwen3AsrExecutionMetadata,
        tokenizer: Option<&Qwen3AsrTokenizer>,
        mel_frontend_plan: &Qwen3AsrMelFrontendPlan,
        audio_encoder_weights: &Qwen3AsrAudioEncoderWeights,
        token_embedding_table: super::token_embedding::Qwen3AsrTokenEmbeddingTable,
        logits_head: super::logits_head::Qwen3AsrLlmLogitsHead,
        layer_attention_projections: Vec<Qwen3AsrLlmLayerAttentionProjection>,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        let profile_started_at = qwen_decode_profile_start();
        let validate_shape_started_at = qwen_decode_profile_start();
        self.validate_prepared_audio_shape(metadata, &request.prepared_audio)?;
        qwen_decode_profile_log_opt("validate_prepared_audio_shape", validate_shape_started_at);
        let mel_started_at = qwen_decode_profile_start();
        let mel_features =
            qwen3_mel_features_from_prepared_audio(&request.prepared_audio, mel_frontend_plan)
                .map_err(map_mel_frontend_error)?;
        qwen_decode_profile_log_opt("mel_frontend", mel_started_at);
        let result = self.decode_with_runtime_assets(
            request,
            metadata,
            tokenizer,
            token_embedding_table,
            audio_encoder_weights,
            &mel_features,
            logits_head,
            layer_attention_projections,
            skip_serve_batch,
        );
        qwen_decode_profile_log_opt("execute_with_runtime_assets_total", profile_started_at);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_runtime_assets(
        &self,
        request: &GgmlAsrExecutionRequest,
        metadata: Qwen3AsrExecutionMetadata,
        tokenizer: Option<&Qwen3AsrTokenizer>,
        token_embedding_table: super::token_embedding::Qwen3AsrTokenEmbeddingTable,
        audio_encoder_weights: &Qwen3AsrAudioEncoderWeights,
        mel_features: &Qwen3AsrMelFeatures,
        logits_head: super::logits_head::Qwen3AsrLlmLogitsHead,
        layer_attention_projections: Vec<Qwen3AsrLlmLayerAttentionProjection>,
        skip_serve_batch: bool,
    ) -> Result<GgmlAsrExecutionResult, Qwen3AsrGgmlExecutorError> {
        let profile_started_at = qwen_decode_profile_start();
        let runtime_cache_path = canonical_runtime_cache_path(&request.runtime_source_path);
        let audio_encoder_cache_key: AudioEncoderCacheKey = (
            runtime_cache_path.clone(),
            GgmlCpuGraphConfig::resolve_runtime_backend(),
        );
        let audio_encoder_started_at = qwen_decode_profile_start();
        let audio_embeddings = encode_qwen_audio_embeddings_cached(
            audio_encoder_cache_key,
            &request.runtime_source_path,
            audio_encoder_weights,
            metadata,
            mel_features,
        )
        .map_err(map_audio_encoder_error)?;
        qwen_decode_profile_log_opt("audio_encoder_cached", audio_encoder_started_at);
        let decode_prompt_started_at = qwen_decode_profile_start();
        let decode_prompt = build_qwen3_decode_prompt(
            metadata,
            tokenizer,
            audio_embeddings.row_count,
            &request.request_options,
        )
        .map_err(map_decode_prompt_error)?;
        qwen_decode_profile_log_opt("decode_prompt", decode_prompt_started_at);
        let token_embedding_started_at = qwen_decode_profile_start();
        let token_rows = token_embedding_table
            .gather_rows(&decode_prompt.token_ids)
            .map_err(map_token_embedding_error)?;
        qwen_decode_profile_log_opt("token_embedding_gather", token_embedding_started_at);
        let prompt_embedding_started_at = qwen_decode_profile_start();
        let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
            &decode_prompt,
            token_embedding_table.d_model(),
            &token_rows,
            &audio_embeddings.rows,
        )
        .map_err(map_prompt_embedding_error)?;
        qwen_decode_profile_log_opt("prompt_embedding_splice", prompt_embedding_started_at);
        let llm_prefill_started_at = qwen_decode_profile_start();
        let llm_prefill_input =
            build_qwen3_llm_prefill_input(&prompt_embeddings).map_err(map_llm_prefill_error)?;
        qwen_decode_profile_log_opt("llm_prefill_input", llm_prefill_started_at);
        if layer_attention_projections.is_empty() {
            return Err(Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: "qwen3-asr runtime exposes zero llm layers; at least 1 is required"
                    .to_string(),
            });
        }
        let token_budget_started_at = qwen_decode_profile_start();
        let max_generated_tokens = qwen3_generated_token_budget(
            &request.prepared_audio,
            decode_prompt.token_ids.len(),
            metadata,
        )?;
        qwen_decode_profile_log_opt("decode_token_budget", token_budget_started_at);
        let decode_config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer
                .map(|tokenizer| tokenizer.eos_token_id)
                .unwrap_or(metadata.eos_token_id),
            vocab_size: metadata.vocab_size,
            max_generated_tokens,
        };
        let token_source: &dyn crate::models::decode_policy_component_registry::BuiltinSeq2SeqDecodePolicyTokenSource =
            tokenizer
                .map(|tokenizer| tokenizer as _)
                .unwrap_or(&metadata);
        let validate_stacks_started_at = qwen_decode_profile_start();
        self.validate_materialized_block_stacks(
            metadata,
            audio_encoder_weights.layer_count(),
            layer_attention_projections.len(),
        )?;
        qwen_decode_profile_log_opt(
            "validate_materialized_block_stacks",
            validate_stacks_started_at,
        );
        let serve_batch_graph_config = super::graph_config::qwen_runtime_graph_config();
        if let Some(serve_batch_config) = Qwen3AsrServeBatchConfig::from_env()
            .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?
            // Serve-batch needs the unified GPU lane on the direct reusable graph.
            // Fall through to direct decode on CPU / scheduler-backed nodes instead
            // of hard-failing with UnsupportedBackend, so a globally-set
            // OPENASR_SERVE_BATCH degrades gracefully (mirrors whisper's gate).
            .filter(|_| {
                // Streaming bypasses the batch worker so live sessions stay on the
                // direct greedy loop below.
                !skip_serve_batch
                    && serve_batch_graph_config.backend.is_gpu_class()
                    && !serve_batch_graph_config.use_scheduler
            })
        {
            let serve_batch_started_at = qwen_decode_profile_start();
            let decode_policy = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
                .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                })?;
            let seq2seq_decode_config = build_builtin_seq2seq_decode_policy_config(
                decode_policy,
                &decode_config,
                token_source,
                request.request_options.phrase_bias.as_ref(),
            )
            .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?;
            let result = submit_qwen_serve_batch_job(
                serve_batch_config,
                Qwen3AsrServeBatchJob {
                    runtime_source_path: request.runtime_source_path.clone(),
                    runtime_cache_path,
                    backend: GgmlCpuGraphConfig::resolve_runtime_backend(),
                    metadata,
                    tokenizer: tokenizer.cloned(),
                    token_embedding_table,
                    logits_head,
                    layer_attention_projections: Arc::new(layer_attention_projections),
                    llm_prefill_input,
                    decode_config: seq2seq_decode_config,
                    text_postprocess_kind: decode_policy.seq2seq_text_postprocess_kind,
                    word_timestamps: request.request_options.word_timestamps,
                    audio_duration_seconds: audio_duration_seconds(&request.prepared_audio),
                },
            )
            .map_err(|error| match error.unavailable_retryable() {
                Some(retryable) => Qwen3AsrGgmlExecutorError::ServeBatchUnavailable {
                    reason: error.to_string(),
                    retryable,
                },
                None => Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                },
            });
            qwen_decode_profile_log_opt("serve_batch_submit", serve_batch_started_at);
            return result;
        }
        // OADP Phase 0: resolve the active LoRA adapter (request-level path,
        // env fallback). Any mismatch (base binding, content change, non-target
        // tensor) is fail-closed — adapters are never silently ignored.
        let preflight_for_adapter =
            request
                .resolve_runtime_source_preflight()
                .map_err(
                    |error| Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed {
                        reason: error.to_string(),
                    },
                )?;
        let adapter = resolve_qwen_lora_adapter(
            request.request_options.adapter_path.as_deref(),
            preflight_for_adapter.as_ref(),
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!("qwen3-asr lora adapter rejected: {error}"),
            },
        )?;
        let adapter_fingerprint = qwen_adapter_cache_fingerprint(adapter.as_deref());
        let decoder_cache_key: WholeDecoderCacheKey = (
            runtime_cache_path,
            GgmlCpuGraphConfig::resolve_runtime_backend(),
            adapter_fingerprint,
        );
        let whole_decoder_started_at = qwen_decode_profile_start();
        let whole_decoder = match take_cached_whole_decoder(&decoder_cache_key) {
            // Resident hit: layer weights already uploaded to the device and the
            // reuse graph already built — skip the per-chunk rebuild + re-upload.
            Some(decoder) => {
                qwen_decode_profile_log_opt("whole_decoder_cache_hit", whole_decoder_started_at);
                decoder
            }
            None => {
                let decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_lora(
                    &layer_attention_projections,
                    Some(request.runtime_source_path.as_path()),
                    adapter.as_deref(),
                )
                .map_err(|error| {
                    Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                        reason: format!("qwen3-asr whole-decoder graph init failed: {error}"),
                    }
                })?;
                qwen_decode_profile_log_opt(
                    "whole_decoder_cache_miss_init",
                    whole_decoder_started_at,
                );
                decoder
            }
        };
        drop(layer_attention_projections);

        let kv_cache_started_at = qwen_decode_profile_start();
        let layer_kv_caches = (0..metadata.llm_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    decode_prompt
                        .token_ids
                        .len()
                        .saturating_add(decode_config.max_generated_tokens),
                    metadata.llm_kv_heads,
                    metadata.llm_head_dim,
                )
            })
            .collect();
        qwen_decode_profile_log_opt("layer_kv_cache_alloc", kv_cache_started_at);
        let mut step_executor = Qwen3AsrPrefillOnlyGreedyStepExecutor {
            metadata,
            prefill_input: llm_prefill_input,
            logits_head,
            token_embedding_table,
            layer_kv_caches,
            whole_decoder,
            cache_prompt_tokens: 1,
            consumed_prefill_step: false,
        };
        let decode_text_token_ids = |token_ids: &[u32]| {
            if let Some(tokenizer) = tokenizer {
                return tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Qwen3AsrGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                });
            }
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "))
        };
        let greedy_decode_started_at = qwen_decode_profile_start();
        let decode_result = run_qwen3_greedy_decode_loop(
            &decode_config,
            token_source,
            request.request_options.phrase_bias.as_ref(),
            &mut step_executor,
            &decode_text_token_ids,
        );
        qwen_decode_profile_log_opt("greedy_decode_loop", greedy_decode_started_at);
        // Return the resident whole-decoder to the cache for the next chunk /
        // execute(); its weights + reuse graph stay valid regardless of outcome.
        let store_decoder_started_at = qwen_decode_profile_start();
        store_cached_whole_decoder(decoder_cache_key, step_executor.whole_decoder);
        qwen_decode_profile_log_opt("store_cached_whole_decoder", store_decoder_started_at);
        // Hitting the token budget without EOT degrades to the generated prefix
        // (mirrors cohere/moonshine) rather than failing the call — so a no-EOT
        // partial cannot kill a live streaming session. The FINAL re-decodes the
        // whole buffer the same way, so it stays consistent with offline `execute()`.
        let postprocess_started_at = qwen_decode_profile_start();
        let (text, generated_tokens, generated_probabilities) = match decode_result {
            Ok(output) => (
                output.text.trim().to_string(),
                output.generated_tokens,
                output.generated_probabilities,
            ),
            Err(Qwen3AsrGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                generated_tokens,
                generated_probabilities,
                ..
            }) => {
                let text = decode_text_token_ids(&generated_tokens)
                    .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                        reason: error.to_string(),
                    })?
                    .trim()
                    .to_string();
                (text, generated_tokens, generated_probabilities)
            }
            Err(error) => {
                return Err(Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                });
            }
        };
        qwen_decode_profile_log_opt("decode_text_postprocess", postprocess_started_at);
        let audio_duration_seconds = audio_duration_seconds(&request.prepared_audio);
        let word_timestamps_started_at = qwen_decode_profile_start();
        let words = if request.request_options.word_timestamps {
            let decode_policy = resolve_builtin_decode_policy(crate::QWEN3_ASR_DECODE_POLICY_ID)
                .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                })?;
            seq2seq_word_timestamps_from_generated_tokens(
                &generated_tokens,
                &generated_probabilities,
                0.0,
                audio_duration_seconds,
                decode_policy.seq2seq_text_postprocess_kind,
                &decode_text_token_ids,
            )
            .map_err(|error| Qwen3AsrGgmlExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?
        } else {
            Vec::new()
        };
        qwen_decode_profile_log_opt("word_timestamps", word_timestamps_started_at);
        let segments = if words.is_empty() || text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: audio_duration_seconds,
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words,
            }]
        };
        let result = Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text,
                segments,
                longform: None,
                language: None,
            },
            carry_context: None,
        });
        qwen_decode_profile_log_opt("decode_with_runtime_assets_total", profile_started_at);
        result
    }

    fn validate_prepared_audio_shape(
        &self,
        metadata: Qwen3AsrExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudio,
    ) -> Result<(), Qwen3AsrGgmlExecutorError> {
        if prepared_audio.sample_rate_hz != metadata.sample_rate_hz || prepared_audio.channels != 1
        {
            return Err(Qwen3AsrGgmlExecutorError::UnsupportedInputShape {
                expected_sample_rate_hz: metadata.sample_rate_hz,
                sample_rate_hz: prepared_audio.sample_rate_hz,
                channels: prepared_audio.channels,
            });
        }
        Ok(())
    }

    fn validate_materialized_block_stacks(
        &self,
        metadata: Qwen3AsrExecutionMetadata,
        audio_layer_count: usize,
        llm_layer_count: usize,
    ) -> Result<(), Qwen3AsrGgmlExecutorError> {
        let qwen_descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(QWEN3_ASR_GGML_ARCHITECTURE_ID);
        let qwen_block_stack = qwen_descriptor
            .as_ref()
            .and_then(|descriptor| descriptor.block_stack.as_ref());
        let layer_resolver = Qwen3AsrLayerCountResolver {
            audio_layers: metadata.audio_layers,
            llm_layers: metadata.llm_layers,
        };
        validate_stage_against_descriptor(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            qwen_block_stack,
            OpenAsrStageRole::Encoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::TransformerEncoderLayer,
                tensor_name_scope: "audio.blk",
                family_layer_count: audio_layer_count,
            },
            &layer_resolver,
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!(
                    "qwen3-asr audio-encoder block-stack descriptor mismatch: {error:?}"
                ),
            },
        )?;
        validate_stage_against_descriptor(
            QWEN3_ASR_GGML_ARCHITECTURE_ID,
            qwen_block_stack,
            OpenAsrStageRole::Decoder,
            OpenAsrOrchestrationShape::LlmDecoder,
            StageBuildPlan {
                block_kind: OpenAsrBlockKind::LlmDecoderLayer,
                tensor_name_scope: "blk",
                family_layer_count: llm_layer_count,
            },
            &layer_resolver,
        )
        .map_err(
            |error| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!("qwen3-asr decoder block-stack descriptor mismatch: {error:?}"),
            },
        )?;
        Ok(())
    }

    #[cfg(test)]
    fn build_prepared_runtime(
        &self,
        model_architecture: &str,
        preflight: &GgmlAsrRuntimeSourcePreflight,
    ) -> Result<Qwen3AsrPreparedRuntime, Qwen3AsrGgmlExecutorError> {
        build_builtin_prepared_runtime(model_architecture, preflight)
            .map_err(map_prepared_runtime_registry_error)?
            .into_qwen3_asr()
            .ok_or_else(|| Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
                reason: format!(
                    "prepared runtime registry returned non-qwen runtime for architecture '{model_architecture}'"
                ),
            })
    }
}

fn qwen_runtime_cache_poisoned() -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed {
        reason: "qwen runtime cache mutex is poisoned".to_string(),
    }
}

fn qwen3_generated_token_budget(
    prepared_audio: &GgmlAsrPreparedAudio,
    prompt_tokens: usize,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<usize, Qwen3AsrGgmlExecutorError> {
    let context_remaining = context_window_budget(metadata.llm_max_positions, prompt_tokens)
        .ok_or_else(|| Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
            reason: format!(
                "prompt_tokens={prompt_tokens} exhausts llm_max_positions={}",
                metadata.llm_max_positions
            ),
        })?;
    let sample_rate = usize::try_from(prepared_audio.sample_rate_hz).map_err(|_| {
        Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
            reason: format!(
                "sample_rate_hz={} does not fit usize",
                prepared_audio.sample_rate_hz
            ),
        }
    })?;
    if sample_rate == 0 {
        return Err(Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
            reason: "sample_rate_hz must be greater than zero".to_string(),
        });
    }
    let audio_rate_budget = prepared_audio
        .samples_f32
        .len()
        .checked_mul(QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND)
        .and_then(|value| value.checked_add(sample_rate - 1))
        .and_then(|value| value.checked_div(sample_rate))
        .ok_or_else(|| Qwen3AsrGgmlExecutorError::DecodeBudgetUnavailable {
            reason: "audio duration token budget overflowed".to_string(),
        })?;
    let desired = audio_rate_budget
        .saturating_add(QWEN3_DECODE_TOKEN_BUDGET_MARGIN)
        .max(QWEN3_DECODE_MIN_GENERATED_TOKENS);
    Ok(desired.min(context_remaining))
}

fn audio_duration_seconds(prepared_audio: &GgmlAsrPreparedAudio) -> f32 {
    prepared_audio.samples_f32.len() as f32 / prepared_audio.sample_rate_hz.max(1) as f32
}

fn map_decode_prompt_error(error: Qwen3AsrDecodePromptError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::DecodePromptConstructionFailed {
        reason: error.to_string(),
    }
}

fn map_mel_frontend_error(error: Qwen3AsrMelFrontendError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::MelFrontendFailed {
        reason: error.to_string(),
    }
}

fn map_audio_encoder_error(error: Qwen3AsrAudioEncoderError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::AudioEncoderFailed {
        reason: error.to_string(),
    }
}

fn map_token_embedding_error(error: Qwen3AsrTokenEmbeddingError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::TokenEmbeddingPrefillFailed {
        reason: error.to_string(),
    }
}

fn map_prepared_runtime_error(error: Qwen3AsrPreparedRuntimeError) -> Qwen3AsrGgmlExecutorError {
    match error {
        Qwen3AsrPreparedRuntimeError::RuntimeContractViolation { reason } => {
            Qwen3AsrGgmlExecutorError::RuntimeContractViolation { reason }
        }
        Qwen3AsrPreparedRuntimeError::RuntimeMetadataReadFailed { reason } => {
            Qwen3AsrGgmlExecutorError::RuntimeMetadataReadFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::MelFrontendFailed { reason } => {
            Qwen3AsrGgmlExecutorError::MelFrontendFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::AudioEncoderFailed { reason } => {
            Qwen3AsrGgmlExecutorError::AudioEncoderFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::TokenEmbeddingPrefillFailed { reason } => {
            Qwen3AsrGgmlExecutorError::TokenEmbeddingPrefillFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::LlmLogitsHeadFailed { reason } => {
            Qwen3AsrGgmlExecutorError::LlmLogitsHeadFailed { reason }
        }
        Qwen3AsrPreparedRuntimeError::LlmTransformerDecodeStepFailed { reason } => {
            Qwen3AsrGgmlExecutorError::LlmTransformerDecodeStepFailed { reason }
        }
    }
}

fn map_prepared_runtime_registry_error(
    error: BuiltinPreparedRuntimeRegistryError,
) -> Qwen3AsrGgmlExecutorError {
    match error {
        BuiltinPreparedRuntimeRegistryError::Qwen3AsrBuild { source } => {
            map_prepared_runtime_error(source)
        }
        other => Qwen3AsrGgmlExecutorError::RuntimeContractViolation {
            reason: other.to_string(),
        },
    }
}

fn map_prompt_embedding_error(error: Qwen3AsrPromptEmbeddingError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::PromptEmbeddingAssemblyFailed {
        reason: error.to_string(),
    }
}

fn map_llm_prefill_error(error: Qwen3AsrLlmPrefillInputError) -> Qwen3AsrGgmlExecutorError {
    Qwen3AsrGgmlExecutorError::LlmPrefillInputAssemblyFailed {
        reason: error.to_string(),
    }
}

/// Resolves a qwen block-stack stage's `layer_count_hparam` to the count parsed
/// from the GGUF hparams (NOT `layers.len()` — see the [`LayerCountResolver`]
/// honesty contract), so `validate_stage_against_descriptor` can cross-check the
/// materialized layer count against the descriptor's declared key. Carries both
/// stages' counts so one resolver serves the audio-encoder and LLM-decoder gates.
struct Qwen3AsrLayerCountResolver {
    audio_layers: usize,
    llm_layers: usize,
}

impl LayerCountResolver for Qwen3AsrLayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            QWEN3_AUDIO_LAYERS_KEY => Some(self.audio_layers),
            QWEN3_LLM_LAYERS_KEY => Some(self.llm_layers),
            _ => None,
        }
    }
}

struct Qwen3AsrPrefillOnlyGreedyStepExecutor {
    metadata: Qwen3AsrExecutionMetadata,
    prefill_input: super::llm_prefill::Qwen3AsrLlmPrefillInput,
    logits_head: super::logits_head::Qwen3AsrLlmLogitsHead,
    token_embedding_table: super::token_embedding::Qwen3AsrTokenEmbeddingTable,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    cache_prompt_tokens: usize,
    consumed_prefill_step: bool,
}

impl Seq2SeqGreedyDecodeStepExecutor for Qwen3AsrPrefillOnlyGreedyStepExecutor {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<
        Seq2SeqGreedyDecodeStepLogitsOutput,
        crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError,
    > {
        if !self.consumed_prefill_step && input.step_index == 0 && input.generated_tokens.is_empty()
        {
            let logits = self.prefill_prompt_and_compute_last_logits().map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;
            self.consumed_prefill_step = true;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits,
                greedy_token_hint: None,
            });
        }

        if input.generated_tokens.is_empty() {
            return Err(
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr generated token history is unexpectedly empty".to_string(),
                },
            );
        }

        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr decode cache position underflowed".to_string(),
                }
            })?;
        let mut hidden = self
            .gather_last_generated_token_hidden(input.generated_tokens)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;
        hidden = self
            .run_llm_layers_with_kv(hidden, cache_position)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;

        let logits = self
            .logits_head
            .compute_logits_for_last_hidden(&hidden)
            .map_err(|error| {
                crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                }
            })?;
        Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
            logits,
            greedy_token_hint: None,
        })
    }
}

impl Qwen3AsrPrefillOnlyGreedyStepExecutor {
    fn prefill_prompt_and_compute_last_logits(
        &mut self,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        let token_count = self.prefill_input.token_count;
        if token_count == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill token count is zero".to_string(),
            });
        }
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        let Some(chunk_size) = self
            .whole_decoder
            .safe_host_cache_prefill_chunk_size_for(token_count)
        else {
            let result = self.prefill_prompt_serial_and_compute_last_logits();
            qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
            return result;
        };
        let result = self.prefill_prompt_chunked_and_compute_last_logits(chunk_size);
        qwen_decode_profile_log_opt("prefill_prompt_total", profile_started_at);
        result
    }

    fn prefill_prompt_chunked_and_compute_last_logits(
        &mut self,
        chunk_size: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        if chunk_size == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill chunk size is zero".to_string(),
            });
        }
        let token_count = self.prefill_input.token_count;
        if token_count <= chunk_size {
            let chunk_started_at = qwen_decode_profile_start();
            let step = self
                .whole_decoder
                .run_prefill(
                    &self.prefill_input.token_major_embeddings,
                    token_count,
                    1_000_000.0,
                )
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            qwen_decode_profile_log_prefill_chunk(0, token_count, chunk_started_at);
            let result = self.write_prefill_step_outputs_and_compute_last_logits(token_count, step);
            qwen_decode_profile_log_opt("prefill_prompt_chunked", profile_started_at);
            return result;
        }
        let hidden_size = self.prefill_input.hidden_size;
        let require_even_chunks = self.whole_decoder.prefill_chunks_require_even_width();
        let mut position_offset = 0usize;
        let mut final_hidden = None;
        while position_offset < token_count {
            let remaining = token_count - position_offset;
            let chunk_len = if require_even_chunks {
                super::even_prefill_chunk_len(remaining, chunk_size)
            } else {
                remaining.min(chunk_size)
            };
            let hidden_start = position_offset.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden offset overflowed".to_string(),
                }
            })?;
            let hidden_len = chunk_len.checked_mul(hidden_size).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden width overflowed".to_string(),
                }
            })?;
            let hidden_end = hidden_start.checked_add(hidden_len).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk hidden end overflowed".to_string(),
                }
            })?;
            let total_token_count = position_offset.checked_add(chunk_len).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill chunk span overflowed".to_string(),
                }
            })?;
            let chunk_started_at = qwen_decode_profile_start();
            let step = self
                .whole_decoder
                .run_prefill_chunk(
                    &self.prefill_input.token_major_embeddings[hidden_start..hidden_end],
                    chunk_len,
                    position_offset,
                    total_token_count,
                    &self.layer_kv_caches,
                    1_000_000.0,
                )
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            qwen_decode_profile_log_prefill_chunk(position_offset, chunk_len, chunk_started_at);
            final_hidden =
                Some(self.write_prefill_chunk_outputs(position_offset, chunk_len, step)?);
            position_offset = total_token_count;
        }
        self.cache_prompt_tokens = token_count;
        let result = self
            .logits_head
            .compute_logits_for_last_hidden(&final_hidden.ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill produced no final hidden state".to_string(),
                }
            })?)
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            });
        qwen_decode_profile_log_opt("prefill_prompt_chunked", profile_started_at);
        result
    }

    fn prefill_prompt_serial_and_compute_last_logits(
        &mut self,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let profile_started_at = qwen_decode_profile_start();
        let token_count = self.prefill_input.token_count;
        let mut final_hidden = None;
        for token_position in 0..token_count {
            let hidden = self.prefill_prompt_hidden_at(token_position)?;
            let hidden = self.run_llm_layers_with_kv(hidden, token_position)?;
            final_hidden = Some(hidden);
        }
        self.cache_prompt_tokens = token_count;
        let result = self
            .logits_head
            .compute_logits_for_last_hidden(&final_hidden.ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill produced no final hidden state".to_string(),
                }
            })?)
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            });
        qwen_decode_profile_log_opt("prefill_prompt_serial", profile_started_at);
        result
    }

    fn write_prefill_step_outputs_and_compute_last_logits(
        &mut self,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let final_hidden = self.write_prefill_chunk_outputs(0, token_count, step)?;
        self.cache_prompt_tokens = token_count;
        self.logits_head
            .compute_logits_for_last_hidden(&final_hidden)
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })
    }

    fn write_prefill_chunk_outputs(
        &mut self,
        position_offset: usize,
        token_count: usize,
        step: super::llm_transformer::Qwen3AsrLlmWholeStepOutput,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        if step.layer_kv.len() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill layer-KV count mismatch".to_string(),
            });
        }
        let kv_row_width = self
            .metadata
            .llm_kv_heads
            .checked_mul(self.metadata.llm_head_dim)
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill KV row width overflowed".to_string(),
            })?;
        for token_position in 0..token_count {
            let absolute_position =
                position_offset.checked_add(token_position).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill absolute row overflowed".to_string(),
                    }
                })?;
            let row_start = token_position.checked_mul(kv_row_width).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill KV row offset overflowed".to_string(),
                }
            })?;
            let row_end = row_start.checked_add(kv_row_width).ok_or_else(|| {
                Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: "qwen3-asr prefill KV row end overflowed".to_string(),
                }
            })?;
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill K row out of bounds".to_string(),
                    }
                })?;
                let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                    Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                        reason: "qwen3-asr prefill V row out of bounds".to_string(),
                    }
                })?;
                self.layer_kv_caches[layer_index]
                    .write(absolute_position, key_row, value_row)
                    .map_err(|reason| Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason })?;
            }
        }
        let hidden_size = self.prefill_input.hidden_size;
        let final_hidden_start = token_count
            .checked_sub(1)
            .and_then(|position| position.checked_mul(hidden_size))
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final-hidden offset overflowed".to_string(),
            })?;
        let final_hidden_end = final_hidden_start.checked_add(hidden_size).ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final-hidden end overflowed".to_string(),
            }
        })?;
        let final_hidden = step
            .hidden
            .get(final_hidden_start..final_hidden_end)
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill final hidden out of bounds".to_string(),
            })?
            .to_vec();
        Ok(final_hidden)
    }

    fn prefill_prompt_hidden_at(
        &self,
        token_position: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let hidden_size = self.prefill_input.hidden_size;
        let start = token_position.checked_mul(hidden_size).ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill hidden-state indexing overflowed".to_string(),
            }
        })?;
        let end = start.checked_add(hidden_size).ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill hidden-state indexing overflowed".to_string(),
            }
        })?;
        self.prefill_input
            .token_major_embeddings
            .get(start..end)
            .ok_or_else(|| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr prefill hidden-state slice is out of bounds".to_string(),
            })
            .map(<[f32]>::to_vec)
    }

    fn run_llm_layers_with_kv(
        &mut self,
        hidden: Vec<f32>,
        cache_position: usize,
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        if self.whole_decoder.layer_count() != self.layer_kv_caches.len() {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr layer/cache mismatch: layers={} caches={}",
                    self.whole_decoder.layer_count(),
                    self.layer_kv_caches.len()
                ),
            });
        }
        if self.metadata.llm_heads == 0 || self.metadata.llm_kv_heads == 0 {
            return Err(Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: format!(
                    "qwen3-asr invalid llm head metadata: llm_heads={} llm_kv_heads={}",
                    self.metadata.llm_heads, self.metadata.llm_kv_heads
                ),
            });
        }

        let started_at = if qwen_decode_profile_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        // Reuse the built decode graph across tokens only on the direct single-GPU
        // lane (`is_gpu_class && !scheduler`). CPU compute and multi-backend
        // scheduler paths rebuild the growing-KV graph each token because their
        // in-place-KV reuse path is not byte-identical.
        let reuse_max_positions = self
            .layer_kv_caches
            .first()
            .map(|cache| cache.max_positions())
            .filter(|_| self.whole_decoder.supports_graph_reuse());
        let step = if let Some(max_positions) = reuse_max_positions {
            self.whole_decoder
                .run_step_reused(
                    &hidden,
                    cache_position,
                    &self.layer_kv_caches,
                    1_000_000.0,
                    max_positions,
                )
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?
        } else {
            self.whole_decoder
                .run_step(&hidden, cache_position, &self.layer_kv_caches, 1_000_000.0)
                .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?
        };
        for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
            self.layer_kv_caches[layer_index]
                .write(cache_position, projected_k, projected_v)
                .map_err(|reason| Qwen3AsrGreedyDecodeError::DecoderStepFailed { reason })?;
        }
        if let Some(started_at) = started_at {
            eprintln!(
                "openasr_qwen_decode_profile: cache_position={} layers={} total_us={} build_us={} compute_us={}",
                cache_position,
                step.layer_kv.len(),
                started_at.elapsed().as_micros(),
                step.build_micros,
                step.compute_micros,
            );
        }
        Ok(step.hidden)
    }

    fn gather_last_generated_token_hidden(
        &self,
        generated_tokens: &[u32],
    ) -> Result<Vec<f32>, Qwen3AsrGreedyDecodeError> {
        let last_token = *generated_tokens.last().ok_or_else(|| {
            Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: "qwen3-asr generated token history is unexpectedly empty".to_string(),
            }
        })?;
        self.token_embedding_table
            .gather_rows(&[last_token])
            .map_err(|error| Qwen3AsrGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })
    }
}

impl GgmlAsrExecutor for Qwen3AsrGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        QWEN3_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        // Offline decode: no token observer, batch worker allowed.
        self.execute_inner(request, false)
            .map_err(|error| qwen_execute_error_to_ggml(error, request.selected_family.adapter_id))
    }
}

impl Qwen3AsrGgmlExecutor {
    /// Streaming decode bypasses the batch worker so live sessions stay on the
    /// direct greedy loop. The FINAL transcript remains byte-identical to `execute`.
    pub(crate) fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request, true)
            .map_err(|error| qwen_execute_error_to_ggml(error, request.selected_family.adapter_id))
    }
}

fn qwen_execute_error_to_ggml(
    error: Qwen3AsrGgmlExecutorError,
    adapter_id: &'static str,
) -> GgmlAsrExecutionError {
    match error {
        Qwen3AsrGgmlExecutorError::ServeBatchUnavailable { reason, retryable } => {
            GgmlAsrExecutionError::ServeBatchUnavailable { reason, retryable }
        }
        error => GgmlAsrExecutionError::ExecutorFailed {
            executor_id: QWEN3_EXECUTOR_ID,
            adapter_id,
            reason: error.to_string(),
        },
    }
}

impl GgmlAsrStreamingExecutor for Qwen3AsrGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        QWEN3_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            QWEN3_STREAMING_EXECUTOR_ID,
            QWEN3_ASR_GGML_ADAPTER_ID,
            "qwen3-asr",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            Qwen3AsrGgmlExecutor::execute_streaming,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use super::super::runtime_contract::QWEN3_LLM_VOCAB_SIZE_KEY;
    use super::super::tensor_names::{
        AUDIO_CONV_OUT_BIAS, AUDIO_CONV_OUT_WEIGHT, AUDIO_CONV1_BIAS, AUDIO_CONV1_WEIGHT,
        AUDIO_CONV2_BIAS, AUDIO_CONV2_WEIGHT, AUDIO_CONV3_BIAS, AUDIO_CONV3_WEIGHT,
        AUDIO_LN_POST_BIAS, AUDIO_LN_POST_WEIGHT, AUDIO_MEL_FILTERS, AUDIO_MEL_WINDOW,
        AUDIO_PROJ1_BIAS, AUDIO_PROJ1_WEIGHT, AUDIO_PROJ2_BIAS, AUDIO_PROJ2_WEIGHT,
        OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT, TOKEN_EMBD_WEIGHT, audio_layer_tensor_names,
        llm_layer_tensor_names,
    };

    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, GgmlAsrExecutionRequest,
        GgmlAsrPreparedAudio, LongFormOptions, qwen3_asr_runtime_descriptor_v1,
        whisper_runtime_descriptor_v1,
    };

    use super::*;

    fn qwen_metadata_with_llm_layers(llm_layers: usize) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".to_string(), "qwen3-asr".to_string());
        metadata.insert("qwen3-asr.sample_rate".to_string(), "16000".to_string());
        metadata.insert("qwen3-asr.n_mels".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.n_fft".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.win_length".to_string(), "400".to_string());
        metadata.insert("qwen3-asr.hop_length".to_string(), "160".to_string());
        metadata.insert("qwen3-asr.audio.n_layers".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.audio.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.audio.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.d_model".to_string(), "16".to_string());
        metadata.insert("qwen3-asr.llm.n_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.n_kv_heads".to_string(), "2".to_string());
        metadata.insert("qwen3-asr.llm.head_dim".to_string(), "8".to_string());
        metadata.insert("qwen3-asr.llm.n_layers".to_string(), llm_layers.to_string());
        metadata.insert("qwen3-asr.llm.vocab_size".to_string(), "32".to_string());
        metadata.insert("qwen3-asr.llm.max_pos".to_string(), "256".to_string());
        metadata.insert(
            "qwen3-asr.audio_start_token_id".to_string(),
            "2".to_string(),
        );
        metadata.insert("qwen3-asr.audio_end_token_id".to_string(), "3".to_string());
        metadata.insert("qwen3-asr.audio_pad_token_id".to_string(), "4".to_string());
        metadata.insert("qwen3-asr.eos_token_id".to_string(), "0".to_string());
        metadata.insert("qwen3-asr.pad_token_id".to_string(), "6".to_string());
        metadata
    }

    fn qwen_metadata() -> BTreeMap<String, String> {
        qwen_metadata_with_llm_layers(2)
    }

    fn add_audio_layer_shapes(spec: TinyGgufFixtureSpec, layer_idx: usize) -> TinyGgufFixtureSpec {
        let names = audio_layer_tensor_names(layer_idx);
        spec.with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_norm_bias, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_bias, [16_u64])
            .with_tensor_shape(names.attn_k_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_k_bias, [16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_bias, [16_u64])
            .with_tensor_shape(names.attn_out_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_out_bias, [16_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            .with_tensor_shape(names.ffn_norm_bias, [16_u64])
            .with_tensor_shape(names.ffn_up_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_up_bias, [32_u64])
            .with_tensor_shape(names.ffn_down_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_down_bias, [16_u64])
    }

    fn add_llm_layer_shapes(spec: TinyGgufFixtureSpec, layer_idx: usize) -> TinyGgufFixtureSpec {
        let names = llm_layer_tensor_names(layer_idx);
        spec.with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_k_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_output_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_norm_weight, [8_u64])
            .with_tensor_shape(names.attn_k_norm_weight, [8_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            .with_tensor_shape(names.ffn_gate_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_up_weight, [32_u64, 16_u64])
            .with_tensor_shape(names.ffn_down_weight, [16_u64, 32_u64])
    }

    fn qwen_tensor_ready_fixture_spec_with_llm_layers(llm_layers: usize) -> TinyGgufFixtureSpec {
        let mut spec = TinyGgufFixtureSpec::new(qwen_metadata_with_llm_layers(llm_layers))
            .with_tensor_shape(AUDIO_MEL_FILTERS, [8_u64, 201_u64])
            .with_tensor_shape(AUDIO_MEL_WINDOW, [400_u64])
            .with_tensor_shape(AUDIO_CONV1_WEIGHT, [3_u64, 3_u64, 1_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV1_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV2_WEIGHT, [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV2_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV3_WEIGHT, [3_u64, 3_u64, 4_u64, 4_u64])
            .with_tensor_shape(AUDIO_CONV3_BIAS, [4_u64])
            .with_tensor_shape(AUDIO_CONV_OUT_WEIGHT, [4_u64, 16_u64])
            .with_tensor_shape(AUDIO_CONV_OUT_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_LN_POST_WEIGHT, [16_u64])
            .with_tensor_shape(AUDIO_LN_POST_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_PROJ1_WEIGHT, [16_u64, 16_u64])
            .with_tensor_shape(AUDIO_PROJ1_BIAS, [16_u64])
            .with_tensor_shape(AUDIO_PROJ2_WEIGHT, [16_u64, 16_u64])
            .with_tensor_shape(AUDIO_PROJ2_BIAS, [16_u64])
            .with_tensor_shape(TOKEN_EMBD_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_WEIGHT, [16_u64, 32_u64])
            .with_tensor_shape(OUTPUT_NORM_WEIGHT, [16_u64]);
        for layer_idx in 0..2 {
            spec = add_audio_layer_shapes(spec, layer_idx);
        }
        for layer_idx in 0..llm_layers {
            spec = add_llm_layer_shapes(spec, layer_idx);
        }
        spec
    }

    fn qwen_tensor_ready_fixture_spec() -> TinyGgufFixtureSpec {
        qwen_tensor_ready_fixture_spec_with_llm_layers(2)
    }

    fn qwen_request(runtime_source_path: PathBuf) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            runtime_source_path,
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0; 160]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        }
    }

    #[test]
    fn decode_token_budget_scales_with_audio_and_context() {
        let metadata = parse_qwen3_execution_metadata(&qwen_metadata()).expect("metadata");
        let short_audio = GgmlAsrPreparedAudio::mono_16khz(vec![0.0; 16_000]);
        let long_audio = GgmlAsrPreparedAudio::mono_16khz(vec![0.0; 240_000]);

        let short_budget =
            qwen3_generated_token_budget(&short_audio, 32, metadata).expect("short budget");
        let long_budget =
            qwen3_generated_token_budget(&long_audio, 32, metadata).expect("long budget");
        let context_limited =
            qwen3_generated_token_budget(&long_audio, 240, metadata).expect("limited budget");

        assert_eq!(short_budget, QWEN3_DECODE_MIN_GENERATED_TOKENS);
        assert!(long_budget > short_budget);
        assert_eq!(context_limited, 16);
    }

    #[test]
    fn decode_token_budget_rejects_full_prompt_context() {
        let metadata = parse_qwen3_execution_metadata(&qwen_metadata()).expect("metadata");
        let audio = GgmlAsrPreparedAudio::mono_16khz(vec![0.0; 16_000]);

        let error = qwen3_generated_token_budget(&audio, metadata.llm_max_positions, metadata)
            .expect_err("full context should fail");

        assert!(error.to_string().contains("exhausts llm_max_positions"));
    }

    #[test]
    fn qwen_executor_rejects_non_qwen_adapter() {
        let mut request = qwen_request(PathBuf::from("/tmp/qwen3-asr.gguf"));
        request.selected_family = whisper_runtime_descriptor_v1();
        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute(&request)
            .expect_err("wrong adapter must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains("requires adapter"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn qwen_executor_fails_closed_when_required_metadata_missing() {
        let mut metadata = qwen_metadata();
        metadata.remove(QWEN3_LLM_VOCAB_SIZE_KEY);
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-missing-metadata.gguf");
        let fixture_spec = TinyGgufFixtureSpec::new(metadata);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);
        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute(&request)
            .expect_err("missing metadata must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains(QWEN3_LLM_VOCAB_SIZE_KEY), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn qwen_executor_fails_closed_when_required_tensor_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec().without_tensor(OUTPUT_NORM_WEIGHT);
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);

        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute(&request)
            .expect_err("missing required tensor must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains("runtime contract check failed"), "{reason}");
                assert!(reason.contains(OUTPUT_NORM_WEIGHT), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    fn assert_qwen_executor_runs(runtime_path: PathBuf) {
        let request = qwen_request(runtime_path);
        let executor = Qwen3AsrGgmlExecutor::default();
        with_forced_cpu_backend_for_test(|| match executor.execute(&request) {
            Ok(_) => {}
            Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                if reason.contains("reached max_generated_tokens") => {}
            Err(error) => panic!("qwen executor should reach decode boundary, got {error:?}"),
        });
    }

    #[test]
    fn qwen_executor_runs_full_stack_with_base_fixture() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        assert_qwen_executor_runs(runtime_path);
    }

    #[test]
    fn qwen_executor_reuses_runtime_assets_across_repeated_runs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");

        let request = qwen_request(runtime_path);
        let executor = Qwen3AsrGgmlExecutor::default();

        with_forced_cpu_backend_for_test(|| {
            for _ in 0..2 {
                match executor.execute(&request) {
                    Ok(_) => {}
                    Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                        if reason.contains("reached max_generated_tokens") => {}
                    Err(error) => {
                        panic!(
                            "qwen cached runtime path should reach decode boundary, got {error:?}"
                        )
                    }
                }
            }
        });
    }

    #[test]
    fn qwen_executor_reuses_runtime_assets_for_longform_runs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");

        let mut request = qwen_request(runtime_path);
        request.request_options.longform = Some(LongFormOptions::default());
        let executor = Qwen3AsrGgmlExecutor::default();

        with_forced_cpu_backend_for_test(|| {
            for _ in 0..2 {
                match executor.execute(&request) {
                    Ok(_) => {}
                    Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                        if reason.contains("reached max_generated_tokens") => {}
                    Err(error) => {
                        panic!(
                            "qwen longform cached runtime path should reach decode boundary, got {error:?}"
                        )
                    }
                }
            }
        });
    }

    #[test]
    fn qwen_prepared_runtime_builder_accepts_deeper_layer_fixtures() {
        let executor = Qwen3AsrGgmlExecutor::default();
        for llm_layers in 3..=9 {
            let temp = tempfile::tempdir().expect("tempdir");
            let runtime_path = temp
                .path()
                .join(format!("qwen3-asr-0.6b-q4_k-layer{llm_layers}.gguf"));
            let fixture_spec = qwen_tensor_ready_fixture_spec_with_llm_layers(llm_layers);
            write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec)
                .expect("write gguf fixture");
            let request = qwen_request(runtime_path);
            let preflight = request
                .resolve_runtime_source_preflight()
                .expect("runtime preflight");
            executor
                .build_prepared_runtime(
                    request.selected_family.model_architecture,
                    preflight.as_ref(),
                )
                .expect("prepared runtime should build");
        }
    }

    #[test]
    fn qwen_prepared_runtime_drops_zero_copy_audio_projection_payloads() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let request = qwen_request(runtime_path);
        let preflight = request
            .resolve_runtime_source_preflight()
            .expect("runtime preflight");
        let executor = Qwen3AsrGgmlExecutor::default();
        let prepared = executor
            .build_prepared_runtime(
                request.selected_family.model_architecture,
                preflight.as_ref(),
            )
            .expect("prepared runtime should build");
        assert!(
            prepared
                .audio_encoder_weights
                .zero_copy_audio_projection_payloads_dropped_for_test()
        );
    }

    #[test]
    fn qwen_executor_fails_closed_when_runtime_metadata_cannot_be_read() {
        let request = qwen_request(PathBuf::from("/tmp/does-not-exist-qwen3.gguf"));
        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute(&request)
            .expect_err("missing runtime source must fail");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(reason.contains("runtime metadata read failed"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn qwen_executor_rejects_non_empty_prompt_option_until_tokenization_lands() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen3-asr-0.6b-q4_k.gguf");
        let fixture_spec = qwen_tensor_ready_fixture_spec();
        write_tiny_gguf_runtime_source(&runtime_path, &fixture_spec).expect("write gguf fixture");
        let mut request = qwen_request(runtime_path);
        request.request_options.prompt = Some("test".to_string());

        let executor = Qwen3AsrGgmlExecutor::default();
        let error = executor
            .execute(&request)
            .expect_err("non-empty prompt must fail closed");
        match error {
            GgmlAsrExecutionError::ExecutorFailed { reason, .. } => {
                assert!(
                    reason.contains("decode prompt construction failed"),
                    "{reason}"
                );
                assert!(reason.contains("request option 'prompt'"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
