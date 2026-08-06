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

use std::sync::Arc;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_LLM_DECODE_POLICY_ID;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::firered_aed::encoder_graph::FireRedEncoderGraphRuntime;
use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrKvCacheCapacity, Qwen3AsrKvCacheCapacityError,
    build_qwen3_prompt_embeddings_with_audio_splice,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use super::adapter_graph::FireRedLlmAdapterGraphRuntime;
use super::decode_prompt::build_firered_llm_decode_prompt;
use super::llm_transformer::FireRedLlmDecoderRuntime;
use super::runtime_contract::{
    FIRERED_LLM_CMVN_INV_STDDEV_TENSOR, FIRERED_LLM_CMVN_NEG_MEAN_TENSOR,
    parse_firered_llm_adapter_metadata, parse_firered_llm_decoder_metadata,
    parse_firered_llm_encoder_metadata,
};
use super::tokenizer::FireRedLlmTokenizer;

/// Resident decoder actor pool. `FireRedLlmDecoderRuntime` owns the
/// Qwen2 whole-decoder graph runner, its device-uploaded layer weights, the
/// logits head, and the token-embedding table -- all identical across every
/// `execute()` call against the same pack on the same backend. Without this,
/// every request paid a full decoder-runtime rebuild (~1.8-2.0s measured,
/// `docs/model-audits/firered2-llm.md` SS3) purely to re-derive state that
/// does not change between requests. Keyed by pack content, concrete execution
/// lane, and stable resident KV span. Unlike qwen this family has no
/// LoRA/adapter input. The pack half is a
/// [`PackContentKey`] built from the same already-open source the request
/// preflight resolved: an in-place `.oasr` replacement at the same path
/// resolves to a different id, so the next lookup misses and rebuilds instead
/// of reusing a decoder whose device-uploaded weights came from the old
/// bytes.
///
/// Each finite checkout owns a dedicated thread because ggml runtime objects
/// are thread-affine. The accompanying [`SystemMemoryOwner`] accounts for the
/// host logits/embedding representation, while native graph construction
/// accounts backend buffers in their physical memory domain.
type FireRedLlmDecoderCacheKey = (PackContentKey, ExecutionLaneKey, usize);

struct FireRedLlmDecoderActorState {
    runtime: FireRedLlmDecoderRuntime,
}

impl std::fmt::Debug for FireRedLlmDecoderActorState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FireRedLlmDecoderActorState")
            .finish_non_exhaustive()
    }
}

type FireRedLlmDecoderRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<FireRedLlmDecoderCacheKey, FireRedLlmDecoderActorState>;
type FireRedLlmDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<FireRedLlmDecoderCacheKey, FireRedLlmDecoderActorState>;

const FIRERED_LLM_EXECUTOR_ID: &str = crate::arch::FIRERED_LLM_EXECUTOR_COMPONENT_ID;
const FIRERED_LLM_STREAMING_EXECUTOR_ID: &str = "firered-llm-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = FIRERED_LLM_CMVN_NEG_MEAN_TENSOR;
const CMVN_INV_STDDEV_TENSOR: &str = FIRERED_LLM_CMVN_INV_STDDEV_TENSOR;
/// Upstream single-utterance hard cap (`fireredasr2` README: "single 40s max
/// input"). The executor fails closed rather than silently truncating or
/// running an out-of-distribution multi-minute prefill; longer audio is the
/// longform slicing orchestrator's job (see `FIRERED_LLM_DECODE_POLICY_ID`'s
/// `ConservativeSeq2SeqV1` longform profile registration).
pub(crate) const FIRERED_LLM_MAX_INPUT_SECONDS: f32 = 40.0;
/// Generous upper bound on generated tokens per utterance -- greedy decode
/// stops at `<|im_end|>` well before this in practice; this is only the
/// fail-closed backstop against a runaway (non-terminating) decode.
pub(crate) const FIRERED_LLM_MAX_GENERATED_TOKENS: usize = 512;

#[derive(Debug, Error)]
enum FireRedLlmExecutorError {
    #[error("firered-llm executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
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
    #[error("firered-llm decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("firered-llm decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("firered-llm {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error("firered-llm greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

const FIRERED_LLM_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const FIRERED_LLM_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

#[derive(Debug, Clone)]
pub(crate) struct FireRedLlmGgmlExecutor {
    decoder_runtimes: Arc<FireRedLlmDecoderRuntimePool>,
}

impl Default for FireRedLlmGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            FIRERED_LLM_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            FIRERED_LLM_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-firered-llm-decoder-owner",
                limits,
            )),
        }
    }
}

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
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    prompt_embeddings: Option<crate::models::qwen::Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
    /// Explicit cancel/pause/resume control for this decode -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    control: Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for FireRedLlmGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if let Some(prompt_embeddings) = self.prompt_embeddings.take() {
            self.cache_prompt_tokens = prompt_embeddings.token_count;
            let prefill = self
                .decoder
                .prefill(
                    &prompt_embeddings,
                    &mut self.layer_kv_caches,
                    self.kv_capacity,
                    &self.control,
                )
                .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                    reason: error.to_string(),
                })?;
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: prefill.logits,
                greedy_token_hint: prefill.greedy_token_hint,
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
        if let Some(token_id) = self
            .decoder
            .decode_step_reused_top1(
                last_token,
                cache_position,
                &self.layer_kv_caches,
                self.kv_capacity,
            )
            .map_err(|error| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: error.to_string(),
            })?
        {
            return Ok(Seq2SeqGreedyDecodeStepLogitsOutput {
                logits: Vec::new(),
                greedy_token_hint: Some(token_id),
            });
        }
        let logits = self
            .decoder
            .decode_step(
                last_token,
                cache_position,
                &mut self.layer_kv_caches,
                self.kv_capacity,
            )
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
    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> FireRedLlmExecutorError {
        FireRedLlmExecutorError::DecoderFailed {
            reason: format!("{stage} runtime ownership failed: {error}"),
        }
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        metadata: super::runtime_contract::FireRedLlmDecoderMetadata,
        kv_capacity: Qwen3AsrKvCacheCapacity,
        backend: GgmlCpuGraphBackend,
    ) -> Result<FireRedLlmDecoderRuntimeActor, FireRedLlmExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
            kv_capacity.resident_positions(),
        );
        let quote_preflight = preflight.clone();
        let build_preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let reader =
                    crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(
                        &quote_preflight,
                    )
                    .map_err(|error| {
                        FireRedLlmExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason: format!("decoder quote tensor reader failed: {error}"),
                        }
                    })?;
                let (peak_bytes, retained_bytes) =
                    super::llm_transformer::quoted_firered_llm_decoder_system_memory_bytes(
                        &reader, &metadata, backend,
                    )
                    .map_err(|reason| {
                        FireRedLlmExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason,
                        }
                    })?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!(
                        "firered-llm-decoder-runtime:{content_id}:positions={}",
                        kv_capacity.resident_positions()
                    ),
                    peak_bytes,
                    retained_bytes,
                )
                .map_err(|error| {
                    FireRedLlmExecutorError::RuntimeOwnershipFailed {
                        stage: "decoder",
                        reason: error.to_string(),
                    }
                })?;
                Ok((retained_bytes, (build_preflight, metadata, backend, quote)))
            },
            move |(preflight, metadata, backend, quote)| {
                match SystemMemoryOwner::try_allocate_transaction(quote, || {
                    let runtime =
                        FireRedLlmDecoderRuntime::new_from_preflight(&preflight, metadata, backend)
                            .map_err(|error| FireRedLlmExecutorError::DecoderFailed {
                                reason: error.to_string(),
                            })?;
                    let retained = runtime.retained_system_memory_bytes().map_err(|reason| {
                        FireRedLlmExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason,
                        }
                    })?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        FireRedLlmDecoderActorState { runtime },
                        retained,
                        retained,
                    ))
                }) {
                    Ok(owner) => Ok(owner),
                    Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        Err(FireRedLlmExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason: error.to_string(),
                        })
                    }
                }
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
    }

    fn clear_runtime_actors(&self) {
        self.decoder_runtimes.clear();
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedLlmExecutorError> {
        let expected_adapter = crate::arch::FIRERED_LLM_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(FireRedLlmExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request.runtime_source_preflight();

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

        let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
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

        let mut encoder_runtime = FireRedEncoderGraphRuntime::new_from_preflight(
            preflight,
            encoder_metadata,
            request.resolved_runtime.backend(),
        )
        .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;
        let encoder_output = encoder_runtime
            .encode(&features.data, features.n_frames)
            .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;

        let adapter_profile_started_at = std::time::Instant::now();
        let mut adapter_runtime = FireRedLlmAdapterGraphRuntime::new_from_preflight(
            preflight,
            request.resolved_runtime.backend(),
        )
        .map_err(|error| FireRedLlmExecutorError::AdapterGraphFailed {
            reason: error.to_string(),
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

        // The process-root policy resolver has already selected and admitted
        // this candidate. Family executors must not invent a second heuristic
        // fallback: doing so would make allocation ownership and the reported
        // execution route disagree.
        let decoder_backend = request.resolved_runtime.backend();
        let measured_positions =
            crate::capacity::topology::causal_prefix_positions_with_context_cap(
                super::capacity::FIRERED_LLM_SELF_KV_STATE_ID,
                decode_prompt.token_ids.len(),
                FIRERED_LLM_MAX_GENERATED_TOKENS,
                decoder_metadata.max_positions,
            )
            .map_err(|error| FireRedLlmExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            })?;
        let kv_capacity = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::FIRERED_LLM_SELF_KV_STATE_ID,
        )
        .and_then(|capacity| capacity.validate_measured_logical_positions(measured_positions))
        .map_err(|source| FireRedLlmExecutorError::DecoderStateCapacity { source })?;
        let decoder_actor = self.checkout_decoder_runtime(
            preflight,
            decoder_metadata,
            kv_capacity,
            decoder_backend,
        )?;
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.chatml_im_end_token_id,
            vocab_size: decoder_metadata.vocab_size,
            max_generated_tokens: FIRERED_LLM_MAX_GENERATED_TOKENS,
        };
        let decoder_tokenizer = tokenizer.clone();
        let decoder_control = Arc::clone(&request.execution_context.control);
        let profile_enabled = std::env::var_os("OPENASR_FIRERED_LLM_PROFILE").is_some();
        let result = decoder_actor
            .call_mut(move |state| {
                let decoder = &mut state.runtime;
                if profile_enabled {
                    eprintln!(
                        "OPENASR_FIRERED_LLM_PROFILE decoder_backend={}",
                        decoder.backend_label()
                    );
                }
                let token_rows_len = decode_prompt
                    .token_ids
                    .len()
                    .checked_mul(decoder_metadata.d_model)
                    .ok_or_else(|| FireRedLlmExecutorError::PromptEmbeddingFailed {
                        reason: "token embedding row allocation overflowed".to_string(),
                    })?;
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
                    token_rows,
                    &speech_rows,
                )
                .map_err(|error| {
                    FireRedLlmExecutorError::PromptEmbeddingFailed {
                        reason: error.to_string(),
                    }
                })?;
                let layer_kv_caches = decoder
                    .new_kv_caches(kv_capacity)
                    .map_err(|reason| FireRedLlmExecutorError::DecoderFailed { reason })?;
                let mut step_executor = FireRedLlmGreedyStepExecutor {
                    decoder,
                    layer_kv_caches,
                    kv_capacity,
                    prompt_embeddings: Some(prompt_embeddings),
                    cache_prompt_tokens: 0,
                    control: Arc::clone(&decoder_control),
                };
                let decode_result = run_builtin_seq2seq_decode_policy(
                    FIRERED_LLM_DECODE_POLICY_ID,
                    &config,
                    &NoPhraseBiasTokenSource,
                    None,
                    &mut step_executor,
                    &|token_ids: &[u32]| {
                        decoder_tokenizer
                            .decode_text_token_ids(token_ids)
                            .map_err(|error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                                reason: error.to_string(),
                            })
                    },
                    |error: Seq2SeqGreedyDecodeError| error,
                    |error: Seq2SeqGreedyDecodeError| error,
                    map_registry_error,
                    &decoder_control,
                );
                // CPU step buffers are invocation-scoped. Native weights and
                // the stable resident arena remain with this actor.
                state.runtime.release_session_scoped_buffers();
                decode_result.map_err(|error| FireRedLlmExecutorError::GreedyDecodeFailed {
                    reason: error.to_string(),
                })
            })
            .map_err(|error| Self::map_actor_error("decoder", error))??;

        let text = result.text.trim().to_string();
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            segments: vec![Segment {
                start: 0.0,
                end: audio_duration_seconds.max(0.0),
                text: text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            text,
            longform: None,
            language: None,
            ..Default::default()
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
            // No intra-decode timestamps -- the single segment spans the whole
            // buffer -- so the cut point has no honest second to name. See
            // `DecodeTruncation::transcript_covers_up_to_seconds`.
            decode_truncation: result.stop_reason.into_decode_truncation(None),
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

impl GgmlAsrViewExecutor for FireRedLlmGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        FireRedLlmGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        FIRERED_LLM_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_firered_llm_decoder_state,
                super::capacity::FIRERED_LLM_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: GgmlAsrViewExecutor::executor_id(self),
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_actors();
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
            FireRedLlmGgmlExecutor::execute_view,
        )
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_actors();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::arch::builtin_adapter_descriptor;
    use crate::models::ggml_asr_executor::GgmlAsrBackendPreference;
    use crate::models::ggml_asr_executor::GgmlAsrPreparedAudioView;

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
    fn dev_pack_path() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_LLM_PACK",
            "FireRed2 LLM .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    // Pinned to the real dev-pack decode. CPU is the deterministic reference
    // backend, so these goldens request `CpuOnly`; the decode is byte-identical
    // on Metal (verified against the q4_k pack on an M1: Metal:MTL0 vs Cpu:CPU
    // produce the same transcript). The q4_k pack fits Metal comfortably (~4.9GB
    // peak RSS on a 16GB Mac); only the larger fp16/q8_0 packs overrun a small
    // unified-memory GPU; the unified execution policy resolves those packs to
    // another admitted candidate before this family executor runs. JFK is word-for-word correct;
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
        let pack_path = dev_pack_path()?;
        transcribe_with_pack(pack_path, wav_path, GgmlAsrBackendPreference::CpuOnly)
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
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("firered-llm test runtime must pass preflight");

        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID,
            ),
        );
        let request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        // Bypasses the dispatch, so the request-level backend preference
        // must still be installed here for the few remaining thread-local
        // readers unrelated to this family's own resolution (dispatch
        // normally does this too).
        let _backend_guard = crate::ggml_runtime::install_request_backend_override(
            request.backend_preference.request_backend_override(),
        );

        let executor = FireRedLlmGgmlExecutor::default();
        let started_at = Instant::now();
        let result = executor
            .execute_view(&request)
            .expect("firered-llm transcribe");
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
        let pack_path = match crate::testing::external_test_fixture_path(
            "OPENASR_FR2_AB_PACK",
            "FireRed2 LLM benchmark .oasr pack",
        ) {
            Ok(path) => path,
            Err(skip) => {
                eprintln!("skipping: {skip}");
                return;
            }
        };
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
