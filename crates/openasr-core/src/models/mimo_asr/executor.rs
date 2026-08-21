//! mimo-asr dedicated executor: mel [`mel_frontend`] -> the P2.0 blood-lesson
//! audio-tokenizer encoder [`audio_tokenizer_graph`] (skip@L3, conv1 stride 1
//! / conv2 stride 2) -> [`rvq`] (first 8 codebooks, residual argmin) -> 8-way
//! embedding sum + 6L input-local transformer + group downcast
//! [`input_local_graph`] -> ChatML/`<|sosp|>`/`<|eosp|>` splice
//! ([`decode_prompt`] + `qwen::build_qwen3_prompt_embeddings_with_audio_splice`)
//! -> 36L Qwen2 [`llm_transformer`] prefill/decode, driven through the ONE
//! shared greedy decode loop
//! (`decode_policy_component_registry::run_builtin_seq2seq_decode_policy`) --
//! never a hand-rolled argmax loop (the repo's
//! `model-integration-shared-driver` invariant).

#![allow(dead_code)]

use std::sync::Arc;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::MIMO_ASR_DECODE_POLICY_ID;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlNativeGqaCapability, RequestBackendPreference,
    request_backend_override,
};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrKvCacheCapacity, Qwen3AsrKvCacheCapacityError,
    Qwen3AsrPromptTokenInput, qwen_llm_effective_native_gqa_capability,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use super::audio_tokenizer_graph::{MimoAudiotokEncoderOutput, MimoAudiotokEncoderRuntime};
use super::decode_prompt::build_mimo_asr_decode_prompt;
use super::input_local_graph::{
    MimoInputLocalInput, MimoInputLocalRuntime, MimoSpeechEmbeddingTables,
    load_speech_embedding_tables_from_reader, sum_speech_embeddings,
};
use super::llm_transformer::MimoLlmDecoderRuntime;
use super::mel_frontend::{
    MimoMelFrontendPlan, load_mimo_mel_frontend_plan_from_reader, mimo_mel_features_from_samples,
    resample_mono,
};
use super::runtime_contract::{
    MimoInlocalMetadata, MimoLlmMetadata, parse_mimo_audiotok_metadata,
    parse_mimo_inlocal_metadata, parse_mimo_llm_metadata, parse_mimo_mel_metadata,
    parse_mimo_special_tokens,
};
use super::rvq::{MimoRvqCodebooks, encode_rvq_codes, load_mimo_rvq_codebooks_from_reader};
use super::tokenizer::MimoAsrTokenizer;

const MIMO_ASR_EXECUTOR_ID: &str = crate::arch::MIMO_ASR_EXECUTOR_COMPONENT_ID;
const MIMO_ASR_STREAMING_EXECUTOR_ID: &str = "mimo-asr-ggml-snapshot-streaming-executor-v1";
/// The reference `preprocess_input` re-chunks internally at 30s (`chunk_samples
/// = 30 * sampling_rate`); this executor instead fails closed above that same
/// bound and leaves multi-chunk orchestration to the shared longform slicer
/// (mirrors `firered_llm`'s upstream-hard-cap precedent).
pub(crate) const MIMO_ASR_MAX_INPUT_SECONDS: f32 = 30.0;
pub(crate) const MIMO_ASR_MAX_GENERATED_TOKENS: usize = 512;

#[derive(Debug, Error)]
enum MimoAsrExecutorError {
    #[error("mimo-asr executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("mimo-asr runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("mimo-asr tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("mimo-asr audio duration {seconds:.1}s exceeds the {limit:.0}s per-chunk cap")]
    AudioTooLong { seconds: f32, limit: f32 },
    #[error("mimo-asr mel frontend failed: {reason}")]
    MelFrontendFailed { reason: String },
    #[error("mimo-asr audio-tokenizer encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("mimo-asr RVQ encode failed: {reason}")]
    RvqFailed { reason: String },
    #[error("mimo-asr input-local transformer failed: {reason}")]
    InputLocalFailed { reason: String },
    #[error("mimo-asr decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("mimo-asr prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("mimo-asr decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("mimo-asr backbone decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("mimo-asr greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
    #[error("mimo-asr {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
}

/// Everything mimo-asr materializes from a pack that does NOT depend on the
/// per-request audio: all three graph runtimes (audio-tokenizer encoder,
/// input-local transformer, 36L Qwen2 backbone decoder), their
/// device-uploaded / arena-resident weights, plus the immutable derived data
/// read once from the pack. CPU lanes retain scalar-oracle RVQ codebooks and
/// speech-embedding tables; accelerated lanes bind those native tensors in
/// the encoder/input-local graphs and deliberately retain no host-f32 copy.
/// Both retain the mel front-end plan, tokenizer, and the two metadata groups
/// the per-request path still consults. Before this cache, mimo-asr was the only
/// family that rebuilt this ENTIRE set on every `execute()` -- three
/// `Runtime::new()` calls plus a full re-read of the pack's codebooks/tables
/// -- purely to re-derive state that never changes between requests against
/// the same pack. Mirrors `firered_llm`'s resident-decoder cache (`FireRedLlm
/// DecoderRuntime` there is one stage; here the whole prepared pipeline is
/// resident because all three mimo stages are equally per-request-invariant).
struct MimoAsrPreparedRuntime {
    encoder_runtime: MimoAudiotokEncoderRuntime,
    inlocal_runtime: MimoInputLocalRuntime,
    decoder: MimoLlmDecoderRuntime,
    tokenizer: MimoAsrTokenizer,
    /// CPU exact-oracle RVQ tables. Accelerated lanes bind the native GGUF
    /// tensors in `encoder_runtime` and deliberately retain no host f32 copy.
    codebooks: Option<MimoRvqCodebooks>,
    /// CPU exact-oracle speech tables. Accelerated lanes gather their native
    /// f16 tensors inside `inlocal_runtime` and retain no host f32 copy.
    speech_embedding_tables: Option<MimoSpeechEmbeddingTables>,
    mel_plan: MimoMelFrontendPlan,
    llm_metadata: MimoLlmMetadata,
    inlocal_metadata: MimoInlocalMetadata,
}

/// Resident prepared-runtime actor pool keyed by content id and execution lane.
/// Request-sized KV storage is allocated after checkout and released before the
/// actor returns to this pool, so KV capacity is deliberately not part of the
/// resident model identity. The
/// pack half is a [`PackContentKey`] from the request's already-open source,
/// so an in-place `.oasr` replacement at the same path resolves a different id
/// and the next lookup rebuilds instead of reusing runtimes built from the old
/// bytes. Entries are tagged with the idle-unload generation they were built
/// service root can clear or target-evict every actor directly; each runtime is
/// destroyed on the same owner thread that constructed its native contexts.
type MimoAsrPreparedRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, GgmlNativeGqaCapability);

type MimoAsrPreparedRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MimoAsrPreparedRuntimeCacheKey, MimoAsrPreparedRuntime>;
type MimoAsrPreparedRuntimeActor =
    PinnedRuntimeActorCheckout<MimoAsrPreparedRuntimeCacheKey, MimoAsrPreparedRuntime>;

const MIMO_ASR_RUNTIME_MAX_IDLE_ENTRIES: usize = 2;
const MIMO_ASR_RUNTIME_MAX_INSTANCES_PER_KEY: usize = 2;

#[derive(Clone)]
pub(crate) struct MimoAsrGgmlExecutor {
    prepared_runtimes: Arc<MimoAsrPreparedRuntimePool>,
}

impl std::fmt::Debug for MimoAsrGgmlExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MimoAsrGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for MimoAsrGgmlExecutor {
    fn default() -> Self {
        Self {
            prepared_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-mimo-asr-runtime-owner",
                AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                    MIMO_ASR_RUNTIME_MAX_IDLE_ENTRIES,
                    crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    MIMO_ASR_RUNTIME_MAX_INSTANCES_PER_KEY,
                ),
            )),
        }
    }
}

/// Drives `MimoLlmDecoderRuntime` through the shared greedy loop: step 0
/// consumes the pre-built (audio-spliced) prompt embeddings via one prefill
/// pass; every later step embeds the last generated token and decodes
/// incrementally. Mirrors `firered_llm::executor::FireRedLlmGreedyStepExecutor`.
struct MimoAsrGreedyStepExecutor<'a> {
    decoder: &'a mut MimoLlmDecoderRuntime,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    prompt_input: Option<Qwen3AsrPromptTokenInput>,
    cache_prompt_tokens: usize,
    /// Explicit cancel/pause/resume control for this decode -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    control: std::sync::Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for MimoAsrGreedyStepExecutor<'_> {
    fn decode_step_logits(
        &mut self,
        input: Seq2SeqGreedyDecodeStepInput<'_>,
    ) -> Result<Seq2SeqGreedyDecodeStepLogitsOutput, Seq2SeqGreedyDecodeError> {
        if let Some(prompt_input) = self.prompt_input.take() {
            self.cache_prompt_tokens = prompt_input.token_ids.len();
            let prefill = self
                .decoder
                .prefill_token_ids_with_audio(
                    &prompt_input.token_ids,
                    &prompt_input.audio_rows,
                    &prompt_input.audio_positions,
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
                reason: "mimo-asr generated token history is unexpectedly empty".to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "mimo-asr decode cache position underflowed".to_string(),
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

impl MimoAsrPreparedRuntime {
    fn quoted_system_memory_bytes(
        preflight: &crate::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
    ) -> Result<(u64, u64), MimoAsrExecutorError> {
        let llm_metadata = parse_mimo_llm_metadata(&preflight.metadata)
            .map_err(|error| contract_error(error.to_string()))?;
        let inlocal_metadata = parse_mimo_inlocal_metadata(&preflight.metadata)
            .map_err(|error| contract_error(error.to_string()))?;
        let audiotok_metadata = parse_mimo_audiotok_metadata(&preflight.metadata)
            .map_err(|error| contract_error(error.to_string()))?;
        let mel_metadata = parse_mimo_mel_metadata(&preflight.metadata)
            .map_err(|error| contract_error(error.to_string()))?;
        let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
            MimoAsrExecutorError::RuntimeOwnershipFailed {
                stage: "prepared-runtime-quote",
                reason: error.to_string(),
            }
        })?;
        let speech_vocab_sizes = audiotok_metadata
            .codebook_sizes
            .iter()
            .map(|size| {
                size.checked_add(1)
                    .ok_or_else(|| capacity_error("mimo speech vocabulary size overflowed"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tokenizer = MimoAsrTokenizer::quoted_retained_system_memory_bytes(&preflight.metadata)
            .map_err(capacity_error)?;
        let mel = MimoMelFrontendPlan::quoted_retained_system_memory_bytes(&mel_metadata)
            .map_err(capacity_error)?;
        let encoder = MimoAudiotokEncoderRuntime::quoted_retained_system_memory_bytes(
            &audiotok_metadata,
            backend,
        )
        .map_err(capacity_error)?;
        let (codebook_peak, codebook_retained, speech_embedding_retained) = if backend
            .is_gpu_class()
        {
            (
                MimoRvqCodebooks::quoted_device_construction_peak_system_memory_bytes(
                    &audiotok_metadata,
                )
                .map_err(capacity_error)?,
                0,
                0,
            )
        } else {
            let codebooks =
                MimoRvqCodebooks::quoted_retained_system_memory_bytes(&audiotok_metadata)
                    .map_err(capacity_error)?;
            let speech_embeddings = MimoSpeechEmbeddingTables::quoted_retained_system_memory_bytes(
                inlocal_metadata.d_model,
                &speech_vocab_sizes,
            )
            .map_err(capacity_error)?;
            (codebooks, codebooks, speech_embeddings)
        };
        let inlocal =
            MimoInputLocalRuntime::quoted_retained_system_memory_bytes(&inlocal_metadata, backend)
                .map_err(capacity_error)?;
        let (decoder_peak, decoder_retained) =
            super::llm_transformer::quoted_mimo_llm_decoder_system_memory_bytes(
                &reader,
                &llm_metadata,
                backend,
            )
            .map_err(capacity_error)?;

        phase_aware_quote([
            (tokenizer, tokenizer),
            (mel, mel),
            (encoder, encoder),
            (codebook_peak, codebook_retained),
            (speech_embedding_retained, speech_embedding_retained),
            (inlocal, inlocal),
            (decoder_peak, decoder_retained),
        ])
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, MimoAsrExecutorError> {
        checked_sum_u64(
            [
                self.encoder_runtime
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                self.inlocal_runtime
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                self.decoder
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                self.tokenizer
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                self.codebooks
                    .as_ref()
                    .map(MimoRvqCodebooks::retained_system_memory_bytes)
                    .transpose()
                    .map_err(capacity_error)?
                    .unwrap_or(0),
                self.speech_embedding_tables
                    .as_ref()
                    .map(MimoSpeechEmbeddingTables::retained_system_memory_bytes)
                    .transpose()
                    .map_err(capacity_error)?
                    .unwrap_or(0),
                self.mel_plan
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
            ],
            "mimo prepared measured retained bytes",
        )
    }

    /// Materializes every per-request-invariant piece of the mimo-asr pipeline
    /// from an already-resolved preflight: the three graph runtimes and their
    /// resident weights, plus lane-appropriate RVQ/speech-table state, the mel
    /// plan, tokenizer, and the two metadata groups the request path consults.
    /// CPU materializes the scalar-oracle tables; accelerated lanes bind the
    /// native GGUF tensors instead. This is the whole cost the resident cache
    /// exists to pay exactly once per (pack, execution lane).
    fn build(
        preflight: &crate::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        native_gqa: GgmlNativeGqaCapability,
    ) -> Result<Self, MimoAsrExecutorError> {
        let llm_metadata = parse_mimo_llm_metadata(&preflight.metadata).map_err(|error| {
            MimoAsrExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let inlocal_metadata =
            parse_mimo_inlocal_metadata(&preflight.metadata).map_err(|error| {
                MimoAsrExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let audiotok_metadata =
            parse_mimo_audiotok_metadata(&preflight.metadata).map_err(|error| {
                MimoAsrExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let mel_metadata = parse_mimo_mel_metadata(&preflight.metadata).map_err(|error| {
            MimoAsrExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let special_tokens = parse_mimo_special_tokens(&preflight.metadata).map_err(|error| {
            MimoAsrExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let tokenizer = MimoAsrTokenizer::from_gguf_metadata(&preflight.metadata, special_tokens)
            .map_err(|error: NativeAsrError| {
            MimoAsrExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            }
        })?;

        let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
            MimoAsrExecutorError::EncoderFailed {
                reason: error.to_string(),
            }
        })?;

        let mel_plan =
            load_mimo_mel_frontend_plan_from_reader(&reader, &mel_metadata).map_err(|error| {
                MimoAsrExecutorError::MelFrontendFailed {
                    reason: error.to_string(),
                }
            })?;

        let encoder_runtime = MimoAudiotokEncoderRuntime::new_from_preflight(
            preflight,
            audiotok_metadata.clone(),
            backend,
        )
        .map_err(|error| MimoAsrExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;

        let accelerated = backend.is_gpu_class();
        let codebooks = (!accelerated)
            .then(|| load_mimo_rvq_codebooks_from_reader(&reader, &audiotok_metadata))
            .transpose()
            .map_err(|error| MimoAsrExecutorError::RvqFailed {
                reason: error.to_string(),
            })?;

        // `mimo.speech.vocab_size` (LLM-side embedding table sizes) is each
        // RVQ codebook's size +1 (a trailing zeroemb padding row); `mimo.speech.
        // zeroemb_idx` equals the codebook size itself (the last row's index).
        // Reconstruct from `mimo.tok.rvq.codebook_sizes` rather than re-parse
        // a fourth metadata group solely for this (both are baked from the
        // exact same upstream `codebook_size`/`speech_vocab_size` config
        // fields, see GGUF_MANIFEST.md and P2.0 findings SS3 point 7).
        let speech_vocab_sizes: Vec<u32> = audiotok_metadata
            .codebook_sizes
            .iter()
            .map(|size| {
                size.checked_add(1)
                    .ok_or_else(|| MimoAsrExecutorError::RuntimeContractViolation {
                        reason: "speech vocabulary size overflowed".to_string(),
                    })
            })
            .collect::<Result<_, _>>()?;
        let speech_embedding_tables = if accelerated {
            None
        } else {
            let zeroemb_idx: Vec<u32> = audiotok_metadata.codebook_sizes.clone();
            Some(
                load_speech_embedding_tables_from_reader(
                    &reader,
                    inlocal_metadata.d_model,
                    &speech_vocab_sizes,
                    &zeroemb_idx,
                )
                .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
                    reason: error.to_string(),
                })?,
            )
        };

        let inlocal_runtime = MimoInputLocalRuntime::new_from_preflight(
            preflight,
            inlocal_metadata,
            &speech_vocab_sizes,
            backend,
        )
        .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
            reason: error.to_string(),
        })?;

        let decoder =
            MimoLlmDecoderRuntime::new_from_preflight(preflight, llm_metadata, backend, native_gqa)
                .map_err(|error| MimoAsrExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?;

        Ok(Self {
            encoder_runtime,
            inlocal_runtime,
            decoder,
            tokenizer,
            codebooks,
            speech_embedding_tables,
            mel_plan,
            llm_metadata,
            inlocal_metadata,
        })
    }
}

fn contract_error(reason: String) -> MimoAsrExecutorError {
    MimoAsrExecutorError::RuntimeContractViolation { reason }
}

fn capacity_error(reason: impl Into<String>) -> MimoAsrExecutorError {
    MimoAsrExecutorError::RuntimeOwnershipFailed {
        stage: "prepared-runtime",
        reason: reason.into(),
    }
}

fn checked_sum_u64<const N: usize>(
    values: [u64; N],
    label: &'static str,
) -> Result<u64, MimoAsrExecutorError> {
    values.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| capacity_error(format!("{label} overflowed")))
    })
}

fn phase_aware_quote<const N: usize>(
    components: [(u64, u64); N],
) -> Result<(u64, u64), MimoAsrExecutorError> {
    let mut retained = 0u64;
    let mut peak = 0u64;
    for (component_peak, component_retained) in components {
        if component_retained > component_peak {
            return Err(capacity_error(format!(
                "mimo component retained bytes {component_retained} exceed peak {component_peak}"
            )));
        }
        peak = peak.max(
            retained
                .checked_add(component_peak)
                .ok_or_else(|| capacity_error("mimo construction peak quote overflowed"))?,
        );
        retained = retained
            .checked_add(component_retained)
            .ok_or_else(|| capacity_error("mimo retained quote overflowed"))?;
    }
    Ok((peak, retained))
}

fn mimo_native_gqa_candidate(
    backend: GgmlCpuGraphBackend,
    preference: Option<&RequestBackendPreference>,
    resolved: GgmlNativeGqaCapability,
) -> GgmlNativeGqaCapability {
    match backend {
        GgmlCpuGraphBackend::Cpu | GgmlCpuGraphBackend::Metal => resolved,
        GgmlCpuGraphBackend::Gpu => match preference {
            Some(RequestBackendPreference::Exact(route))
                if route.provider == ExecutionProvider::Vulkan =>
            {
                resolved
            }
            Some(RequestBackendPreference::CpuOnly)
            | Some(RequestBackendPreference::Accelerated)
            | Some(RequestBackendPreference::Exact(_))
            | None => GgmlNativeGqaCapability::Unsupported,
        },
    }
}

impl MimoAsrGgmlExecutor {
    fn map_actor_error(error: PinnedRuntimeActorError) -> MimoAsrExecutorError {
        MimoAsrExecutorError::RuntimeOwnershipFailed {
            stage: "prepared-runtime",
            reason: error.to_string(),
        }
    }

    fn checkout_prepared_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        native_gqa: GgmlNativeGqaCapability,
    ) -> Result<MimoAsrPreparedRuntimeActor, MimoAsrExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
            native_gqa,
        );
        let quote_preflight = preflight.clone();
        let build_preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.prepared_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let (peak_bytes, retained_bytes) =
                    MimoAsrPreparedRuntime::quoted_system_memory_bytes(&quote_preflight, backend)?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("mimo-asr-runtime:{content_id}"),
                    peak_bytes,
                    retained_bytes,
                )
                .map_err(|error| capacity_error(error.to_string()))?;
                Ok((retained_bytes, quote))
            },
            move |quote| match SystemMemoryOwner::try_allocate_transaction(quote, || {
                let runtime = MimoAsrPreparedRuntime::build(&build_preflight, backend, native_gqa)?;
                let retained = runtime.retained_system_memory_bytes()?;
                Ok(SystemMemoryAllocationOutcome::new(
                    runtime, retained, retained,
                ))
            }) {
                Ok(owner) => Ok(owner),
                Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(capacity_error(error.to_string()))
                }
            },
            Self::map_actor_error,
        )
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.prepared_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
    }

    fn clear_runtime_actors(&self) {
        self.prepared_runtimes.clear();
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, MimoAsrExecutorError> {
        let expected_adapter = crate::arch::MIMO_ASR_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(MimoAsrExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request.runtime_source_preflight();

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > MIMO_ASR_MAX_INPUT_SECONDS {
            return Err(MimoAsrExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: MIMO_ASR_MAX_INPUT_SECONDS,
            });
        }

        let backend = request.resolved_runtime.backend();
        let native_gqa = qwen_llm_effective_native_gqa_capability(mimo_native_gqa_candidate(
            backend,
            request_backend_override().as_ref(),
            request.resolved_runtime.native_gqa_capability(),
        ));
        let kv_capacity = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::MIMO_ASR_SELF_KV_STATE_ID,
        )
        .map_err(|source| MimoAsrExecutorError::DecoderStateCapacity { source })?;
        let actor = self.checkout_prepared_runtime(preflight, backend, native_gqa)?;
        let samples = samples.to_vec();
        let input_rate = request.prepared_audio.sample_rate_hz;
        let control = Arc::clone(&request.execution_context.control);
        let decode_work_progress = request
            .execution_context
            .decode_work_progress_observer()
            .cloned();
        let unstable_decode_text = request
            .execution_context
            .unstable_decode_text_observer()
            .cloned();
        let result = actor
            .call_mut(move |runtime| {
                let result = Self::transcribe_with_prepared(
                    runtime,
                    samples,
                    input_rate,
                    kv_capacity,
                    control,
                    decode_work_progress,
                    unstable_decode_text,
                );
                runtime.decoder.release_session_scoped_buffers();
                result
            })
            .map_err(Self::map_actor_error)??;
        let text = strip_mimo_language_tags(&result.text);
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

    /// The per-request path: everything that depends on the audio input, run
    /// against an already-built resident [`MimoAsrPreparedRuntime`]. Nothing
    /// here reads the pack or constructs a graph runtime -- it only feeds this
    /// utterance's samples through the resident encoder / input-local / decoder
    /// graphs and returns the greedy decode result. Fields are borrowed
    /// disjointly (`&mut` encoder, then `&mut` input-local, then `&mut` decoder
    /// alongside `&` tokenizer) so the resident runtime is reused in place.
    fn transcribe_with_prepared(
        prepared: &mut MimoAsrPreparedRuntime,
        samples: Vec<f32>,
        input_rate: u32,
        kv_capacity: Qwen3AsrKvCacheCapacity,
        control: Arc<crate::api::backend::TranscriptionControl>,
        decode_work_progress: Option<crate::api::backend::WorkProgressObserver>,
        unstable_decode_text: Option<crate::api::backend::UnstableDecodeTextObserver>,
    ) -> Result<Seq2SeqGreedyDecodeResult, MimoAsrExecutorError> {
        // The OpenASR pipeline delivers 16kHz mono to every executor, but
        // MiMo's audio tokenizer (and its baked mel filterbank/window) is
        // trained at 24kHz -- resample up before the mel front-end, matching
        // the reference `preprocess_input`'s own resample-to-tokenizer-rate.
        let target_rate = prepared.mel_plan.sample_rate_hz as u32;
        let resampled = resample_mono(&samples, input_rate, target_rate).ok_or(
            MimoAsrExecutorError::MelFrontendFailed {
                reason: format!("failed to resample {input_rate}Hz -> {target_rate}Hz"),
            },
        )?;
        let mel_features =
            mimo_mel_features_from_samples(&resampled, &prepared.mel_plan).map_err(|error| {
                MimoAsrExecutorError::MelFrontendFailed {
                    reason: error.to_string(),
                }
            })?;

        let encode_result = prepared.encoder_runtime.encode(&mel_features);
        let release_result = prepared.encoder_runtime.release_transient_compute_memory();
        let encoder_output = match (encode_result, release_result) {
            (Ok(output), Ok(())) => output,
            (Err(error), _) | (Ok(_), Err(error)) => {
                return Err(MimoAsrExecutorError::EncoderFailed {
                    reason: error.to_string(),
                });
            }
        };

        let mut codes =
            match encoder_output {
                MimoAudiotokEncoderOutput::HostHiddenRows {
                    frame_count, rows, ..
                } => {
                    let codebooks = prepared.codebooks.as_ref().ok_or_else(|| {
                        MimoAsrExecutorError::RvqFailed {
                            reason: "host encoder output has no resident CPU RVQ codebooks"
                                .to_string(),
                        }
                    })?;
                    encode_rvq_codes(codebooks, &rows, frame_count).map_err(|error| {
                        MimoAsrExecutorError::RvqFailed {
                            reason: error.to_string(),
                        }
                    })?
                }
                MimoAudiotokEncoderOutput::DeviceCodes(codes) => codes,
            };

        // Truncate to the nearest group_size multiple (drop up to
        // group_size-1 trailing 25Hz frames = well under 200ms of audio) --
        // the reference asserts exact divisibility rather than padding.
        let group_size = prepared.inlocal_metadata.group_size;
        let usable_frames = (codes.frame_count() / group_size) * group_size;
        if usable_frames == 0 {
            return Err(MimoAsrExecutorError::RvqFailed {
                reason: format!(
                    "audio too short: {} RVQ frames produced, need at least {group_size}",
                    codes.frame_count()
                ),
            });
        }
        codes
            .truncate_frames(usable_frames)
            .map_err(|error| MimoAsrExecutorError::RvqFailed {
                reason: error.to_string(),
            })?;

        let llm_d_model = prepared.llm_metadata.d_model;
        let inlocal_result = if let Some(tables) = &prepared.speech_embedding_tables {
            let summed = sum_speech_embeddings(tables, &codes).map_err(|error| {
                MimoAsrExecutorError::InputLocalFailed {
                    reason: error.to_string(),
                }
            })?;
            prepared.inlocal_runtime.run(
                MimoInputLocalInput::HostSummedEmbeddings(&summed),
                usable_frames,
                llm_d_model,
            )
        } else {
            prepared.inlocal_runtime.run(
                MimoInputLocalInput::DeviceRvqCodes(&codes),
                usable_frames,
                llm_d_model,
            )
        };
        let inlocal_release = prepared.inlocal_runtime.release_transient_compute_memory();
        let speech_rows = match (inlocal_result, inlocal_release) {
            (Ok(output), Ok(())) => output,
            (Err(error), _) | (Ok(_), Err(error)) => {
                return Err(MimoAsrExecutorError::InputLocalFailed {
                    reason: error.to_string(),
                });
            }
        };
        let audio_group_count = usable_frames / group_size;

        // Disjoint field borrows: `&mut decoder` for the decode graph and
        // `&tokenizer` for prompt/text decoding are distinct fields of the
        // same resident runtime, so both are live at once without conflict.
        let tokenizer = &prepared.tokenizer;
        let llm_metadata = prepared.llm_metadata;
        let decoder = &mut prepared.decoder;

        let decode_prompt =
            build_mimo_asr_decode_prompt(tokenizer, audio_group_count).map_err(|error| {
                MimoAsrExecutorError::DecodePromptFailed {
                    reason: error.to_string(),
                }
            })?;

        let audio_pad_end = decode_prompt
            .audio_pad_start_index
            .checked_add(decode_prompt.audio_pad_count)
            .ok_or_else(|| MimoAsrExecutorError::PromptEmbeddingFailed {
                reason: "audio pad position overflowed".to_string(),
            })?;
        let prompt_input = Qwen3AsrPromptTokenInput {
            token_ids: decode_prompt.token_ids.clone(),
            audio_rows: speech_rows,
            audio_positions: (decode_prompt.audio_pad_start_index..audio_pad_end).collect(),
        };

        let measured_positions =
            crate::capacity::topology::causal_prefix_positions_with_context_cap(
                super::capacity::MIMO_ASR_SELF_KV_STATE_ID,
                decode_prompt.token_ids.len(),
                MIMO_ASR_MAX_GENERATED_TOKENS,
                llm_metadata.max_positions,
            )
            .map_err(|error| MimoAsrExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            })?;
        let kv_capacity = kv_capacity
            .validate_measured_logical_positions(measured_positions)
            .map_err(|source| MimoAsrExecutorError::DecoderStateCapacity { source })?;
        let layer_kv_caches = decoder
            .new_kv_caches(kv_capacity)
            .map_err(|reason| MimoAsrExecutorError::DecoderFailed { reason })?;
        let mut step_executor = MimoAsrGreedyStepExecutor {
            decoder,
            layer_kv_caches,
            kv_capacity,
            prompt_input: Some(prompt_input),
            cache_prompt_tokens: 0,
            control: Arc::clone(&control),
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.special.im_end_id,
            vocab_size: llm_metadata.vocab_size,
            max_generated_tokens: MIMO_ASR_MAX_GENERATED_TOKENS,
        };
        run_builtin_seq2seq_decode_policy(
            MIMO_ASR_DECODE_POLICY_ID,
            &config,
            &(),
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
            &control,
            decode_work_progress.as_ref(),
            unstable_decode_text.as_ref(),
        )
        .map_err(|error| MimoAsrExecutorError::GreedyDecodeFailed {
            reason: error.to_string(),
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

/// Strip the `<chinese>`/`<english>` language-detection tags MiMo auto-emits
/// as leading text under this family's automatic-language ASR mode.
///
/// We build the decode prompt without an explicit `audio_tag` (see
/// [`super::decode_prompt`]), i.e. the reference's auto mode, so the model
/// self-emits the detected language as a leading `<chinese>`/`<english>` marker
/// (analogous to Whisper's `<|zh|>` tag). These are ordinary decoded *text* --
/// not vocab special tokens -- so [`super::tokenizer::MimoAsrTokenizer::decode_text_token_ids`]'s
/// special-token filter never removes them. The reference
/// `mimo_audio.py::asr_sft` strips them from the returned transcript as a final
/// per-utterance postprocess step (`result.replace('<chinese>', '')
/// .replace('<english>', '').strip()`); this mirrors that exactly, applied to
/// each single-utterance result *before* any longform segment join, so both the
/// direct and longform paths match the reference's user-visible output.
fn strip_mimo_language_tags(text: &str) -> String {
    text.replace("<chinese>", "")
        .replace("<english>", "")
        .trim()
        .to_string()
}

impl GgmlAsrViewExecutor for MimoAsrGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        MimoAsrGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        MIMO_ASR_EXECUTOR_ID
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
                super::capacity::plan_mimo_asr_decoder_state,
                super::capacity::MIMO_ASR_DECODER_STATE_STREAMS,
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

impl GgmlAsrStreamingExecutor for MimoAsrGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MIMO_ASR_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            MIMO_ASR_STREAMING_EXECUTOR_ID,
            crate::arch::MIMO_ASR_GGML_ADAPTER_ID,
            "mimo-asr",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            MimoAsrGgmlExecutor::execute_view,
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
    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudioView};

    use super::*;

    fn exactly_addressable_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(crate::device::execution_route::ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
            addressability:
                crate::device::execution_route::DeviceAddressability::ExactlyAddressable {
                    physical_key: crate::device::execution_route::PhysicalResourceKey::new(
                        "0000:01:00.0",
                    )
                    .expect("physical key"),
                },
        })
    }

    #[test]
    fn native_gqa_candidate_is_vulkan_only_on_discrete_gpu() {
        let validated = GgmlNativeGqaCapability::Validated;
        for (provider, expected) in [
            (
                ExecutionProvider::Cuda,
                GgmlNativeGqaCapability::Unsupported,
            ),
            (ExecutionProvider::Vulkan, validated),
            (ExecutionProvider::Hip, GgmlNativeGqaCapability::Unsupported),
            (
                ExecutionProvider::Unknown,
                GgmlNativeGqaCapability::Unsupported,
            ),
        ] {
            let preference = exactly_addressable_preference(provider);
            assert_eq!(
                mimo_native_gqa_candidate(GgmlCpuGraphBackend::Gpu, Some(&preference), validated,),
                expected,
            );
        }
        assert_eq!(
            mimo_native_gqa_candidate(GgmlCpuGraphBackend::Cpu, None, validated),
            validated,
        );
        assert_eq!(
            mimo_native_gqa_candidate(GgmlCpuGraphBackend::Metal, None, validated),
            validated,
        );
    }

    #[test]
    fn strip_mimo_language_tags_matches_reference_asr_sft_postprocess() {
        // Leading auto-tag (the common single-utterance case) is removed and
        // the exposed leading space trimmed.
        assert_eq!(
            strip_mimo_language_tags("<chinese> 今天天气非常好。"),
            "今天天气非常好。"
        );
        assert_eq!(
            strip_mimo_language_tags("<english> And so, my fellow Americans."),
            "And so, my fellow Americans."
        );
        // Global replace (mirrors Python `str.replace`): every occurrence goes,
        // and `.trim()` only touches the ends -- an interior tag leaves the
        // surrounding spaces exactly as the reference's `.strip()` would.
        assert_eq!(strip_mimo_language_tags("a <chinese> b"), "a  b");
        // No tag -> only the outer trim applies (a plain no-op replace).
        assert_eq!(strip_mimo_language_tags("  hello  "), "hello");
    }

    /// Real converted dev pack from P2.1+P2.2 (`tooling/mimo-asr/convert_mimo_asr.py`),
    /// NOT committed to the repo (dev-only artifact, same convention as
    /// firered2-llm's own `tmp-weights/fr2/out/firered2-llm-q8_0.oasr`).
    fn dev_pack_path() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_MIMO_ASR_PACK",
            "MiMo ASR .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    // Pinned to the real dev-pack decode. CPU is the deterministic transcript
    // oracle; dedicated tests below validate the fused Metal RVQ bridge and
    // the full explicit-Metal product route with placement telemetry. JFK is
    // word-for-word correct; the Mandarin sentence is sentence-correct
    // (matches firered-llm/firered-aed's own `zh_sample.wav` reference
    // meaning, with MiMo additionally emitting punctuation).
    //
    // These are the post-`strip_mimo_language_tags` transcripts: the raw decode
    // leads with the model's auto `<chinese>`/`<english>` language marker (see
    // that function's doc comment), which the executor strips per-utterance to
    // match the reference `mimo_audio.py::asr_sft`. `concat!` keeps the literals
    // robust to line wrapping (a trailing-`\` continuation would silently eat a
    // significant leading space on the next line).
    //
    // Confirmed byte-for-byte against a clean-window re-run of these tests
    // against the real pack (all three asserted equal below).
    const GOLDEN_JFK_TEXT: &str = concat!(
        "And so, my fellow Americans, ask not what your country can do for you. ",
        "Ask what you can do for your country.",
    );

    const GOLDEN_ZH_TEXT: &str = concat!(
        "今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，",
        "听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影放松一下。",
    );

    // Code-switch coverage: `en_zh_mixed.wav` is first 5s of jfk.wav + first
    // 8s of zh_sample.wav concatenated (see firered-llm's identical fixture
    // doc comment), a single <=40s utterance -- both languages' tokenizer/
    // decode paths run in one prefill+decode call, no longform slicing
    // involved. The transcript correctly switches languages mid-utterance
    // and both halves truncate exactly where their source clip was cut
    // (English stops at "ask not", the Mandarin half at the truncated word
    // "新[开]"). Post-strip like the single-language goldens above.
    const GOLDEN_EN_ZH_MIXED_TEXT: &str = concat!(
        "And so, my fellow Americans, ask not. ",
        "今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新",
    );

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
        transcribe_with_dev_pack_backend(wav_path, GgmlAsrBackendPreference::CpuOnly)
    }

    fn transcribe_with_dev_pack_backend(
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, std::time::Duration, f32)> {
        let pack_path = dev_pack_path()?;
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "mimo-asr e2e test",
            "mimo-asr e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("mimo runtime must pass preflight");

        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID,
            ),
        );
        let execution_services =
            crate::models::native_execution_services::test_native_execution_services();
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services: Arc::clone(&execution_services),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };

        let intent = match backend_preference {
            GgmlAsrBackendPreference::CpuOnly => {
                crate::device::execution_policy::ExecutionIntent::CpuOnly
            }
            GgmlAsrBackendPreference::Accelerated => {
                crate::device::execution_policy::ExecutionIntent::AcceleratedOnly
            }
            GgmlAsrBackendPreference::Auto => {
                crate::device::execution_policy::ExecutionIntent::Auto
            }
        };
        let inventory =
            crate::enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
        let execution_plan = execution_services
            .policy_resolver()
            .resolve(
                intent,
                crate::arch::family_auto_gpu_policy_for_model_architecture(
                    request.selected_family.model_architecture,
                ),
                request.selected_family.execution_capabilities,
                &inventory,
            )
            .expect("resolve MiMo test execution plan");
        let candidate = execution_plan
            .candidates()
            .first()
            .expect("MiMo test execution plan has a candidate")
            .clone();

        let started_at = Instant::now();
        let attempt = crate::models::native_execution_services::run_execution_candidate_attempt(
            execution_services.as_ref(),
            &candidate,
            || {
                let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput::for_offline_view_request(
                        request.runtime_source_preflight(),
                        &request.prepared_audio,
                        &request.request_options,
                        request.resolved_runtime.backend(),
                    )?;
                request.decoder_state = execution_services
                    .offline_dispatch()
                    .plan_decoder_state(&request.selected_family, &planning_input)?;
                execution_services.offline_dispatch().execute_view(&request)
            },
        );
        assert!(
            attempt.candidate_failure.is_none(),
            "MiMo product-route candidate failed: {:?}",
            attempt.candidate_failure
        );
        let result = attempt.result.expect("mimo-asr transcribe");
        let elapsed = started_at.elapsed();
        Some((result.transcription.text, elapsed, audio_duration_seconds))
    }

    #[test]
    #[ignore = "requires a private converted MiMo-V2.5-ASR .oasr pack"]
    fn golden_diff_end_to_end_transcribe_jfk_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(jfk_wav_path())
        else {
            return;
        };
        eprintln!(
            "mimo-asr e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    #[test]
    #[ignore = "requires the private ~9.6GB dev-only mimo-v2.5-asr-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu recommended"]
    fn golden_diff_end_to_end_transcribe_zh_sample_wav() {
        let Some((text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack(zh_wav_path())
        else {
            return;
        };
        eprintln!(
            "mimo-asr e2e [zh_sample.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_ZH_TEXT);
    }

    // Code-switch coverage: a single <=40s utterance mixing both languages
    // (no longform slicing involved), reusing the same `en_zh_mixed.wav`
    // fixture firered-llm's own golden test built (first 5s of jfk.wav +
    // first 8s of zh_sample.wav) so both families exercise identical
    // code-switch audio.
    #[test]
    #[ignore = "requires the private ~9.6GB dev-only mimo-v2.5-asr-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu recommended"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(en_zh_mixed_wav_path())
        else {
            return;
        };
        eprintln!(
            "mimo-asr e2e [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_EN_ZH_MIXED_TEXT);
    }

    /// Exact bridge gate for the accelerated rewrite: a diagnostic Metal
    /// encoder emits hidden rows and the original scalar RVQ oracle quantizes
    /// them; production Metal fuses that same sequential residual quantizer
    /// into the encoder graph and returns only compact codes. Holding the
    /// encoder backend fixed isolates the new RVQ math from normal CPU/Metal
    /// reduction-order drift.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "host-local: needs OPENASR_MIMO_ASR_PACK and Metal"]
    fn metal_rvq_fusion_matches_scalar_oracle_on_same_metal_hidden_rows() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("mimo runtime preflight");
        let audiotok_metadata =
            parse_mimo_audiotok_metadata(&preflight.metadata).expect("audio-tokenizer metadata");
        let mel_metadata = parse_mimo_mel_metadata(&preflight.metadata).expect("mel metadata");
        let reader =
            build_runtime_tensor_reader_from_preflight(&preflight).expect("runtime tensor reader");
        let mel_plan =
            load_mimo_mel_frontend_plan_from_reader(&reader, &mel_metadata).expect("mel plan");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            en_zh_mixed_wav_path(),
            "mimo RVQ bridge parity",
            "mimo RVQ bridge parity",
        )
        .expect("load audio");
        let resampled = resample_mono(&samples, 16_000, mel_plan.sample_rate_hz as u32)
            .expect("resample audio");
        let mel = mimo_mel_features_from_samples(&resampled, &mel_plan).expect("mel features");

        let placement = crate::GgmlExecutionTelemetryCollector::new();
        let _placement_guard = placement.install();
        let expected = {
            let mut diagnostic =
                MimoAudiotokEncoderRuntime::new_from_preflight_without_rvq_fusion_for_test(
                    &preflight,
                    audiotok_metadata.clone(),
                    GgmlCpuGraphBackend::Metal,
                )
                .expect("diagnostic Metal audio-tokenizer");
            let output = diagnostic
                .encode(&mel)
                .expect("diagnostic Metal hidden rows");
            diagnostic
                .release_transient_compute_memory()
                .expect("release diagnostic Metal transient memory");
            let MimoAudiotokEncoderOutput::HostHiddenRows {
                frame_count, rows, ..
            } = output
            else {
                panic!("non-fused Metal encoder must expose scalar-oracle hidden rows");
            };
            let codebooks = load_mimo_rvq_codebooks_from_reader(&reader, &audiotok_metadata)
                .expect("scalar RVQ codebooks");
            encode_rvq_codes(&codebooks, &rows, frame_count).expect("scalar RVQ")
        };

        let mut metal = MimoAudiotokEncoderRuntime::new_from_preflight(
            &preflight,
            audiotok_metadata,
            GgmlCpuGraphBackend::Metal,
        )
        .expect("Metal audio-tokenizer");
        let output = metal.encode(&mel).expect("Metal encoder plus RVQ output");
        metal
            .release_transient_compute_memory()
            .expect("release Metal transient memory");
        let MimoAudiotokEncoderOutput::DeviceCodes(actual) = output else {
            panic!("Metal encoder must fuse RVQ and return device codes");
        };
        let observed = placement.snapshot();
        let mismatches = actual
            .values()
            .iter()
            .zip(expected.values())
            .enumerate()
            .filter_map(|(index, (&actual, &expected))| {
                (actual != expected).then_some((index, actual, expected))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.frame_count(), expected.frame_count());
        assert_eq!(actual.channels(), expected.channels());
        assert!(
            mismatches.is_empty(),
            "Metal RVQ fusion changed {} code(s); first mismatches: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(16)]
        );
        assert!(
            !observed.observed_compute_nodes_by_backend.is_empty()
                && observed
                    .observed_compute_nodes_by_backend
                    .keys()
                    .all(|backend| {
                        let backend = backend.to_ascii_lowercase();
                        backend.starts_with("mtl") || backend.contains("metal")
                    }),
            "explicit Metal MiMo encoder/RVQ observed non-Metal compute: {:?}",
            observed.observed_compute_nodes_by_backend
        );
    }

    /// Full product-route acceptance for the host-table removal: explicit
    /// accelerated execution must retain the CPU transcript exactly while
    /// every observed ggml compute node stays on Metal. The lower-level RVQ
    /// gate above localizes code selection; this gate covers device speech
    /// lookup, input-local transformer, prompt splice and decoder together.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "host-local: needs OPENASR_MIMO_ASR_PACK and Metal"]
    fn explicit_metal_product_route_matches_cpu_transcript_and_uses_only_metal() {
        let Some((cpu_text, _, _)) = transcribe_with_dev_pack_backend(
            en_zh_mixed_wav_path(),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };

        let placement = crate::GgmlExecutionTelemetryCollector::new();
        let _placement_guard = placement.install();
        let Some((metal_text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            en_zh_mixed_wav_path(),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        let observed = placement.snapshot();
        eprintln!(
            "MIMO_METAL_PRODUCT_PARITY elapsed={elapsed:?} audio_duration={audio_duration_seconds:.3}s rtf={:.4} observed_compute_nodes={:?}",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001),
            observed.observed_compute_nodes_by_backend,
        );
        assert_eq!(cpu_text, GOLDEN_EN_ZH_MIXED_TEXT);
        assert_eq!(metal_text, cpu_text);
        assert!(
            !observed.observed_compute_nodes_by_backend.is_empty()
                && observed
                    .observed_compute_nodes_by_backend
                    .keys()
                    .all(|backend| {
                        let backend = backend.to_ascii_lowercase();
                        backend.starts_with("mtl") || backend.contains("metal")
                    }),
            "explicit Metal MiMo route observed non-Metal compute: {:?}",
            observed.observed_compute_nodes_by_backend
        );
    }

    /// Resident prepared-runtime pool regression: `execute()` twice in a row
    /// against the same pack content and execution lane must reuse the
    /// executor-owned actor on the second call and still produce a
    /// transcript byte-identical to the first call (a cold cache-miss build) and
    /// to the dedicated single-call golden above -- the resident encoder /
    /// input-local / decoder carry no per-request state across calls that could
    /// leak into a later transcript. This is the mimo mirror of firered-llm's
    /// `resident_decoder_cache_reuse_across_consecutive_calls_stays_byte_identical`.
    /// Run with `OPENASR_GGML_BACKEND=cpu` for the deterministic reference decode.
    #[test]
    #[ignore = "requires the private ~9.6GB dev-only mimo-v2.5-asr-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu recommended (see the single-call goldens)"]
    fn resident_prepared_runtime_reuse_across_consecutive_calls_stays_byte_identical() {
        let Some((first_text, _, _)) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        let Some((second_text, _, _)) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        assert_eq!(first_text, GOLDEN_JFK_TEXT);
        assert_eq!(
            second_text, GOLDEN_JFK_TEXT,
            "second execute() (a resident prepared-runtime cache hit) must match the first \
             (cache-miss/build) call byte-for-byte"
        );
    }
}
