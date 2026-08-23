//! funasr-nano dedicated executor: fbank+LFR frontend (`sensevoice` WavFrontend,
//! NO CMVN -- Fun-ASR-Nano runs directly on fbank+LFR) -> the SAN-M
//! [`encoder_graph`] (eps 1e-5, hidden-state output) -> the 2-layer transformer
//! [`adapter_graph`] -> low-frame-rate audio-token truncation -> ChatML+audio
//! splice ([`decode_prompt`] + `qwen::build_qwen3_prompt_embeddings_with_audio_splice`)
//! -> Qwen3-0.6B [`llm_transformer`] prefill/decode, driven through the ONE
//! shared greedy decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a [`Seq2SeqGreedyDecodeStepExecutor`]
//! impl below -- never a hand-rolled argmax loop (the repo's
//! `model-integration-shared-driver` invariant, see `AGENTS.md`).

#![allow(dead_code)]

use std::sync::Arc;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FUNASR_NANO_DECODE_POLICY_ID;
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlDecodeOutputPlan, RequestBackendPreference, request_backend_override,
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
use crate::models::native_execution_services::{
    ExecutionLaneKey, current_execution_lane_key, current_execution_placement,
};
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrKvCacheCapacity, Qwen3AsrKvCacheCapacityError,
    Qwen3AsrPromptTokenInput,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::sensevoice::encoder_graph::build_sensevoice_encoder_input;
use crate::models::sensevoice::frontend::{SenseVoiceFbankFrontend, apply_lfr};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use super::adapter_graph::FunasrNanoAdapterGraph;
use super::decode_prompt::{build_funasr_nano_decode_prompt, funasr_nano_audio_token_count};
use super::encoder_graph::FunasrNanoEncoderGraph;
use super::llm_transformer::FunasrNanoDecoderRuntime;
use super::runtime_contract::{
    FunasrNanoDecoderMetadata, parse_funasr_nano_adapter_metadata,
    parse_funasr_nano_decoder_metadata, parse_funasr_nano_encoder_metadata,
};
use super::tokenizer::FunasrNanoTokenizer;

const FUNASR_NANO_EXECUTOR_ID: &str = crate::arch::FUNASR_NANO_EXECUTOR_COMPONENT_ID;
const FUNASR_NANO_STREAMING_EXECUTOR_ID: &str = "funasr-nano-ggml-snapshot-streaming-executor-v1";
/// Upstream single-utterance hard cap (the official runtime warns that a single
/// clip beyond ~40s greedily repeats out of distribution; `--chunk 15` fixes
/// it). The executor fails closed rather than silently running an OOD
/// multi-minute prefill; longer audio is the shared longform slicing
/// orchestrator's job (see the `ConservativeSeq2SeqV1` longform profile).
pub(crate) const FUNASR_NANO_MAX_INPUT_SECONDS: f32 = 40.0;
/// Fail-closed backstop against a non-terminating decode -- greedy decode stops
/// at `<|im_end|>` well before this in practice. The decoder-state topology
/// reserves the same limit.
pub(crate) const FUNASR_NANO_MAX_GENERATED_TOKENS: usize = 512;

type FunasrNanoEncoderAdapterRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
type FunasrNanoDecoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, GgmlDecodeOutputPlan);

/// Resident encoder-side runtime: the SAN-M encoder graph + transformer
/// adaptor with their weights already uploaded to (or bound zero-copy in)
/// backend memory. It is owned by a finite service-root actor pool so each
/// graph remains on the thread that created it while repeat requests reuse
/// the immutable weights and only rebuild transient forward graphs.
struct FunasrNanoEncoderAdapterRuntime {
    encoder: FunasrNanoEncoderGraph,
    adapter: FunasrNanoAdapterGraph,
}

type FunasrNanoEncoderAdapterRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    FunasrNanoEncoderAdapterRuntimeCacheKey,
    FunasrNanoEncoderAdapterRuntime,
>;
type FunasrNanoDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    FunasrNanoDecoderRuntimeCacheKey,
    FunasrNanoDecoderRuntime,
>;
type FunasrNanoEncoderAdapterRuntimeActor = PinnedRuntimeActorCheckout<
    FunasrNanoEncoderAdapterRuntimeCacheKey,
    FunasrNanoEncoderAdapterRuntime,
>;
type FunasrNanoDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<FunasrNanoDecoderRuntimeCacheKey, FunasrNanoDecoderRuntime>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunasrNanoUnifiedRuntimeCacheKey {
    content: PackContentKey,
    lane: ExecutionLaneKey,
    output_plan: GgmlDecodeOutputPlan,
}

struct FunasrNanoUnifiedRuntime {
    encoder: FunasrNanoEncoderGraph,
    adapter: FunasrNanoAdapterGraph,
    // Keep the decoder's existing SystemMemory lease nested inside the unified
    // owner. Encoder/adapter graph objects already use native-domain admission;
    // co-location must not weaken the decoder host-state transaction.
    decoder: SystemMemoryOwner<FunasrNanoDecoderRuntime>,
}

type FunasrNanoUnifiedRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    FunasrNanoUnifiedRuntimeCacheKey,
    FunasrNanoUnifiedRuntime,
>;
type FunasrNanoUnifiedRuntimeActor =
    PinnedRuntimeActorCheckout<FunasrNanoUnifiedRuntimeCacheKey, FunasrNanoUnifiedRuntime>;

#[derive(Debug, Error)]
enum FunasrNanoExecutorError {
    #[error("funasr-nano executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("funasr-nano runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("funasr-nano tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("funasr-nano audio duration {seconds:.1}s exceeds the upstream {limit:.0}s hard cap")]
    AudioTooLong { seconds: f32, limit: f32 },
    #[error("funasr-nano frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("funasr-nano encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("funasr-nano adapter failed: {reason}")]
    AdapterFailed { reason: String },
    #[error("funasr-nano decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("funasr-nano prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("funasr-nano decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("funasr-nano decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("funasr-nano {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error("funasr-nano greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

const FUNASR_NANO_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const FUNASR_NANO_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;
const FUNASR_NANO_UNIFIED_GPU_MAX_INSTANCES_PER_KEY: usize = 2;

fn funasr_nano_unified_runtime_enabled(
    allow_unified_runtime: bool,
    backend: GgmlCpuGraphBackend,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> bool {
    allow_unified_runtime
        && backend == GgmlCpuGraphBackend::Gpu
        && placement == Some(ExecutionPlacement::FullDevice)
        && matches!(
            backend_preference,
            Some(RequestBackendPreference::Exact(route))
                if route.addressability.is_exactly_addressable()
                    && matches!(route.provider, ExecutionProvider::Cuda | ExecutionProvider::Vulkan)
        )
}

fn validate_funasr_nano_unified_runtime(
    runtime: &FunasrNanoUnifiedRuntime,
) -> Result<(), FunasrNanoExecutorError> {
    let direct_gpu_lane = (GgmlCpuGraphBackend::Gpu, false);
    if runtime.encoder.graph_lane() != direct_gpu_lane
        || runtime.adapter.graph_lane() != direct_gpu_lane
        || runtime.decoder.graph_lane() != direct_gpu_lane
    {
        return Err(FunasrNanoExecutorError::RuntimeContractViolation {
            reason: "unified FunASR-Nano runtime requires direct GPU encoder, adapter, and decoder lanes"
                .to_string(),
        });
    }
    let encoder_binding = runtime
        .encoder
        .loaded_weight_binding_identity()
        .ok_or_else(|| FunasrNanoExecutorError::RuntimeContractViolation {
            reason: "unified FunASR-Nano encoder did not retain a loaded weight binding"
                .to_string(),
        })?;
    let adapter_binding = runtime.adapter.loaded_weight_binding_identity();
    let decoder_binding = runtime
        .decoder
        .loaded_weight_binding_identity()
        .ok_or_else(|| FunasrNanoExecutorError::RuntimeContractViolation {
            reason: "unified FunASR-Nano decoder did not retain a loaded weight binding"
                .to_string(),
        })?;
    if encoder_binding != adapter_binding || encoder_binding != decoder_binding {
        return Err(FunasrNanoExecutorError::RuntimeContractViolation {
            reason: "unified FunASR-Nano runtime did not coalesce its pack-wide weight binding"
                .to_string(),
        });
    }
    Ok(())
}

fn funasr_nano_decoder_system_memory_quote(
    preflight: &crate::GgufRuntimeSourcePreflight,
    metadata: FunasrNanoDecoderMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<SystemMemoryAllocationQuote, FunasrNanoExecutorError> {
    let reader =
        crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
            .map_err(|error| FunasrNanoExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason: format!("decoder quote tensor reader failed: {error}"),
            })?;
    let (peak_bytes, retained_bytes) =
        super::llm_transformer::quoted_funasr_nano_decoder_system_memory_bytes(
            &reader, &metadata, backend,
        )
        .map_err(|reason| FunasrNanoExecutorError::RuntimeOwnershipFailed {
            stage: "decoder",
            reason,
        })?;
    SystemMemoryAllocationQuote::new(
        format!(
            "funasr-nano-decoder-runtime:{}",
            preflight.runtime_source.content_id()
        ),
        peak_bytes,
        retained_bytes,
    )
    .map_err(|error| FunasrNanoExecutorError::RuntimeOwnershipFailed {
        stage: "decoder",
        reason: error.to_string(),
    })
}

fn allocate_funasr_nano_decoder_runtime(
    preflight: &crate::GgufRuntimeSourcePreflight,
    metadata: FunasrNanoDecoderMetadata,
    resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    quote: SystemMemoryAllocationQuote,
) -> Result<SystemMemoryOwner<FunasrNanoDecoderRuntime>, FunasrNanoExecutorError> {
    match SystemMemoryOwner::try_allocate_transaction(quote, || {
        let runtime =
            FunasrNanoDecoderRuntime::new_from_preflight(preflight, metadata, resolved_runtime)
                .map_err(|error| FunasrNanoExecutorError::DecoderFailed {
                    reason: error.to_string(),
                })?;
        let retained = runtime.retained_system_memory_bytes().map_err(|reason| {
            FunasrNanoExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason,
            }
        })?;
        Ok(SystemMemoryAllocationOutcome::new(
            runtime, retained, retained,
        ))
    }) {
        Ok(owner) => Ok(owner),
        Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
        Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
            Err(FunasrNanoExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason: error.to_string(),
            })
        }
    }
}

fn encode_funasr_nano_with_runtime(
    encoder: &mut FunasrNanoEncoderGraph,
    adapter: &mut FunasrNanoAdapterGraph,
    encoder_input: &crate::models::sensevoice::encoder_graph::SenseVoiceEncoderInput,
    adapter_metadata: super::runtime_contract::FunasrNanoAdapterMetadata,
) -> Result<(Vec<f32>, usize), FunasrNanoExecutorError> {
    let encode_result = encoder.encode(
        &encoder_input.data,
        encoder_input.n_frames,
        encoder_input.feature_dim,
    );
    let encoder_release = encoder.release_transient_compute_memory();
    let encoder_output = match (encode_result, encoder_release) {
        (Ok(output), Ok(())) => output,
        (Err(error), _) | (Ok(_), Err(error)) => {
            return Err(FunasrNanoExecutorError::EncoderFailed {
                reason: error.to_string(),
            });
        }
    };
    let adapter_result = adapter.run(
        &encoder_output.rows,
        encoder_output.frame_count,
        encoder_output.d_model,
    );
    let adapter_release = adapter.release_transient_compute_memory();
    let (adapter_rows, adapter_frames) = match (adapter_result, adapter_release) {
        (Ok(output), Ok(())) => output,
        (Err(error), _) | (Ok(_), Err(error)) => {
            return Err(FunasrNanoExecutorError::AdapterFailed {
                reason: error.to_string(),
            });
        }
    };
    let audio_token_count =
        funasr_nano_audio_token_count(encoder_output.frame_count).min(adapter_frames);
    if audio_token_count == 0 {
        return Err(FunasrNanoExecutorError::AdapterFailed {
            reason: "no audio tokens produced".to_string(),
        });
    }
    let speech_value_count = audio_token_count
        .checked_mul(adapter_metadata.llm_dim)
        .ok_or_else(|| FunasrNanoExecutorError::AdapterFailed {
            reason: "audio token row length overflowed".to_string(),
        })?;
    let speech_rows = adapter_rows
        .get(..speech_value_count)
        .ok_or_else(|| FunasrNanoExecutorError::AdapterFailed {
            reason: format!(
                "adapter returned {} values, expected at least {speech_value_count}",
                adapter_rows.len()
            ),
        })?
        .to_vec();
    Ok((speech_rows, audio_token_count))
}

#[derive(Clone)]
pub(crate) struct FunasrNanoGgmlExecutor {
    encoder_adapter_runtimes: Arc<FunasrNanoEncoderAdapterRuntimePool>,
    decoder_runtimes: Arc<FunasrNanoDecoderRuntimePool>,
    unified_gpu_runtimes: Arc<FunasrNanoUnifiedRuntimePool>,
}

impl std::fmt::Debug for FunasrNanoGgmlExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FunasrNanoGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for FunasrNanoGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = || {
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                FUNASR_NANO_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
                max_committed_requested_bytes,
                FUNASR_NANO_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
            )
        };
        Self {
            encoder_adapter_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-funasr-nano-encoder-adapter-owner",
                limits(),
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-funasr-nano-decoder-owner",
                limits(),
            )),
            unified_gpu_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-funasr-nano-unified-gpu-owner",
                AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                    FUNASR_NANO_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
                    max_committed_requested_bytes,
                    FUNASR_NANO_UNIFIED_GPU_MAX_INSTANCES_PER_KEY,
                ),
            )),
        }
    }
}

/// Drives `FunasrNanoDecoderRuntime` through the shared greedy loop: step 0
/// consumes the pre-built (audio-spliced) prompt embeddings via one prefill
/// pass; every later step embeds the last generated token and decodes
/// incrementally (device-side top-1 on the Metal reuse graph, full host logits
/// on CPU). Mirrors `moss_transcribe_diarize::executor::MossTdGreedyStepExecutor`.
struct FunasrNanoGreedyStepExecutor<'a> {
    decoder: &'a mut FunasrNanoDecoderRuntime,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    prompt_input: Option<Qwen3AsrPromptTokenInput>,
    cache_prompt_tokens: usize,
    control: Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for FunasrNanoGreedyStepExecutor<'_> {
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
                reason: "funasr-nano generated token history is unexpectedly empty".to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "funasr-nano decode cache position underflowed".to_string(),
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

impl FunasrNanoGgmlExecutor {
    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> FunasrNanoExecutorError {
        FunasrNanoExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn checkout_encoder_adapter_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        encoder_metadata: super::runtime_contract::FunasrNanoEncoderMetadata,
        adapter_metadata: super::runtime_contract::FunasrNanoAdapterMetadata,
        backend: GgmlCpuGraphBackend,
    ) -> Result<FunasrNanoEncoderAdapterRuntimeActor, FunasrNanoExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
        );
        let preflight = preflight.clone();
        self.encoder_adapter_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, preflight)),
            move |preflight| {
                let encoder = FunasrNanoEncoderGraph::new_from_preflight(
                    &preflight,
                    encoder_metadata,
                    backend,
                )
                .map_err(|error| FunasrNanoExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
                let adapter = FunasrNanoAdapterGraph::new_from_preflight(
                    &preflight,
                    adapter_metadata,
                    backend,
                )
                .map_err(|error| FunasrNanoExecutorError::AdapterFailed {
                    reason: error.to_string(),
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    FunasrNanoEncoderAdapterRuntime { encoder, adapter },
                ))
            },
            |error| Self::map_actor_error("encoder-adapter", error),
        )
    }

    fn encode_with_owned_encoder_adapter_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        encoder_metadata: super::runtime_contract::FunasrNanoEncoderMetadata,
        adapter_metadata: super::runtime_contract::FunasrNanoAdapterMetadata,
        encoder_input: crate::models::sensevoice::encoder_graph::SenseVoiceEncoderInput,
        backend: GgmlCpuGraphBackend,
    ) -> Result<(Vec<f32>, usize), FunasrNanoExecutorError> {
        let actor = self.checkout_encoder_adapter_runtime(
            preflight,
            encoder_metadata,
            adapter_metadata,
            backend,
        )?;
        actor
            .call_mut(move |runtime| {
                encode_funasr_nano_with_runtime(
                    &mut runtime.encoder,
                    &mut runtime.adapter,
                    &encoder_input,
                    adapter_metadata,
                )
            })
            .map_err(|error| Self::map_actor_error("encoder-adapter", error))?
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        metadata: FunasrNanoDecoderMetadata,
        resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    ) -> Result<FunasrNanoDecoderRuntimeActor, FunasrNanoExecutorError> {
        let backend = resolved_runtime.backend();
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
            resolved_runtime.output_plan(),
        );
        let quote_preflight = preflight.clone();
        let build_preflight = preflight.clone();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let quote =
                    funasr_nano_decoder_system_memory_quote(&quote_preflight, metadata, backend)?;
                Ok((quote.retained_bytes, (build_preflight, quote)))
            },
            move |(preflight, quote)| {
                allocate_funasr_nano_decoder_runtime(&preflight, metadata, resolved_runtime, quote)
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn checkout_unified_gpu_runtime(
        &self,
        preflight: &crate::GgufRuntimeSourcePreflight,
        encoder_metadata: super::runtime_contract::FunasrNanoEncoderMetadata,
        adapter_metadata: super::runtime_contract::FunasrNanoAdapterMetadata,
        decoder_metadata: FunasrNanoDecoderMetadata,
        resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    ) -> Result<FunasrNanoUnifiedRuntimeActor, FunasrNanoExecutorError> {
        let backend = resolved_runtime.backend();
        let key = FunasrNanoUnifiedRuntimeCacheKey {
            content: PackContentKey::for_runtime_source(&preflight.runtime_source),
            lane: current_execution_lane_key(backend),
            output_plan: resolved_runtime.output_plan(),
        };
        let quote_preflight = preflight.clone();
        let build_preflight = preflight.clone();
        self.unified_gpu_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let decoder_quote = funasr_nano_decoder_system_memory_quote(
                    &quote_preflight,
                    decoder_metadata,
                    backend,
                )?;
                // The nested decoder owner performs the actual SystemMemory
                // transaction on the owner thread. The outer actor adds only
                // native-domain graph owners already admitted by their runtime
                // constructors, so it must not reserve the decoder bytes twice.
                Ok((0, (build_preflight, decoder_quote, resolved_runtime)))
            },
            move |(preflight, decoder_quote, resolved_runtime)| {
                let encoder = FunasrNanoEncoderGraph::new_from_preflight(
                    &preflight,
                    encoder_metadata,
                    backend,
                )
                .map_err(|error| FunasrNanoExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
                let adapter = FunasrNanoAdapterGraph::new_from_preflight(
                    &preflight,
                    adapter_metadata,
                    backend,
                )
                .map_err(|error| FunasrNanoExecutorError::AdapterFailed {
                    reason: error.to_string(),
                })?;
                let decoder = allocate_funasr_nano_decoder_runtime(
                    &preflight,
                    decoder_metadata,
                    resolved_runtime,
                    decoder_quote,
                )?;
                let runtime = FunasrNanoUnifiedRuntime {
                    encoder,
                    adapter,
                    decoder,
                };
                validate_funasr_nano_unified_runtime(&runtime)?;
                Ok(SystemMemoryOwner::without_allocation(runtime))
            },
            |error| Self::map_actor_error("unified-runtime", error),
        )
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_adapter_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.unified_gpu_runtimes
            .evict_where(|key| key.content.pack_content_id == pack_content_id);
    }

    fn clear_runtime_actors(&self) {
        self.encoder_adapter_runtimes.clear();
        self.decoder_runtimes.clear();
        self.unified_gpu_runtimes.clear();
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, FunasrNanoExecutorError> {
        self.execute_inner_with_runtime_mode(request, true)
    }

    fn execute_streaming_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner_with_runtime_mode(request, false)
            .map_err(|error| GgmlAsrExecutionError::ExecutorFailed {
                executor_id: FUNASR_NANO_STREAMING_EXECUTOR_ID,
                adapter_id: request.selected_family.adapter_id,
                reason: error.to_string(),
            })
    }

    fn execute_inner_with_runtime_mode(
        &self,
        request: &GgmlAsrExecutionViewRequest,
        allow_unified_runtime: bool,
    ) -> Result<GgmlAsrExecutionResult, FunasrNanoExecutorError> {
        let expected_adapter = crate::arch::FUNASR_NANO_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(FunasrNanoExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request.runtime_source_preflight();

        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let decoder_metadata =
            parse_funasr_nano_decoder_metadata(&*preflight.metadata).map_err(|error| {
                FunasrNanoExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokenizer = FunasrNanoTokenizer::from_gguf_metadata(&preflight.metadata).map_err(
            |error: NativeAsrError| FunasrNanoExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            },
        )?;

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > FUNASR_NANO_MAX_INPUT_SECONDS {
            return Err(FunasrNanoExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: FUNASR_NANO_MAX_INPUT_SECONDS,
            });
        }

        // Frontend: kaldi fbank + FunASR LFR stacking, NO CMVN (Fun-ASR-Nano's
        // config carries `cmvn_file: null`; the official runtime runs directly
        // on fbank+LFR).
        let fbank = SenseVoiceFbankFrontend::new()
            .compute(samples)
            .map_err(|error| FunasrNanoExecutorError::FrontendFailed {
                reason: error.to_string(),
            })?;
        let lfr = apply_lfr(&fbank.data, fbank.n_mels).map_err(|error| {
            FunasrNanoExecutorError::FrontendFailed {
                reason: error.to_string(),
            }
        })?;
        if lfr.feature_dim != encoder_metadata.feature_dim {
            return Err(FunasrNanoExecutorError::FrontendFailed {
                reason: format!(
                    "LFR feature dim {} does not match encoder feature dim {}",
                    lfr.feature_dim, encoder_metadata.feature_dim
                ),
            });
        }
        let encoder_input = build_sensevoice_encoder_input(
            &[],
            &lfr.data,
            encoder_metadata.feature_dim,
            encoder_metadata.d_model,
        )
        .map_err(|error| FunasrNanoExecutorError::FrontendFailed {
            reason: error.to_string(),
        })?;

        let backend = request.resolved_runtime.backend();
        let backend_preference = request_backend_override();
        let placement = current_execution_placement();
        let unified_gpu_runtime = if funasr_nano_unified_runtime_enabled(
            allow_unified_runtime,
            backend,
            backend_preference.as_ref(),
            placement,
        ) {
            Some(self.checkout_unified_gpu_runtime(
                preflight,
                encoder_metadata,
                adapter_metadata,
                decoder_metadata,
                request.resolved_runtime,
            )?)
        } else {
            None
        };
        let (speech_rows, audio_token_count) = if let Some(runtime) = unified_gpu_runtime.as_ref() {
            runtime
                .call_mut_fallible(move |state| {
                    encode_funasr_nano_with_runtime(
                        &mut state.encoder,
                        &mut state.adapter,
                        &encoder_input,
                        adapter_metadata,
                    )
                })
                .map_err(|error| Self::map_actor_error("unified-encoder-adapter", error))??
        } else {
            self.encode_with_owned_encoder_adapter_runtime(
                preflight,
                encoder_metadata,
                adapter_metadata,
                encoder_input,
                backend,
            )?
        };
        if speech_rows.iter().any(|value| !value.is_finite()) {
            return Err(FunasrNanoExecutorError::AdapterFailed {
                reason: "adapter output contains non-finite values".to_string(),
            });
        }

        let decode_prompt = build_funasr_nano_decode_prompt(&tokenizer, audio_token_count)
            .map_err(|error| FunasrNanoExecutorError::DecodePromptFailed {
                reason: error.to_string(),
            })?;

        let measured_positions =
            crate::capacity::topology::causal_prefix_positions_with_context_cap(
                super::capacity::FUNASR_NANO_SELF_KV_STATE_ID,
                decode_prompt.token_ids.len(),
                FUNASR_NANO_MAX_GENERATED_TOKENS,
                decoder_metadata.max_positions,
            )
            .map_err(|error| FunasrNanoExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            })?;
        let kv_capacity = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::FUNASR_NANO_SELF_KV_STATE_ID,
        )
        .and_then(|capacity| capacity.validate_measured_logical_positions(measured_positions))
        .map_err(|source| FunasrNanoExecutorError::DecoderStateCapacity { source })?;
        let decoder_control = Arc::clone(&request.execution_context.control);
        let decoder_decode_work_progress = request
            .execution_context
            .decode_work_progress_observer()
            .cloned();
        let decoder_unstable_decode_text = request
            .execution_context
            .unstable_decode_text_observer()
            .cloned();
        let result = if let Some(runtime) = unified_gpu_runtime.as_ref() {
            runtime
                .call_mut_fallible(move |state| {
                    let result = decode_with_decoder(
                        &mut state.decoder,
                        &decoder_metadata,
                        &decode_prompt,
                        &speech_rows,
                        &tokenizer,
                        kv_capacity,
                        &decoder_control,
                        decoder_decode_work_progress.as_ref(),
                        decoder_unstable_decode_text.as_ref(),
                    );
                    state.decoder.release_session_scoped_buffers();
                    result
                })
                .map_err(|error| Self::map_actor_error("unified-decoder", error))??
        } else {
            let decoder_actor = self.checkout_decoder_runtime(
                preflight,
                decoder_metadata,
                request.resolved_runtime,
            )?;
            decoder_actor
                .call_mut(move |runtime| {
                    let result = decode_with_decoder(
                        runtime,
                        &decoder_metadata,
                        &decode_prompt,
                        &speech_rows,
                        &tokenizer,
                        kv_capacity,
                        &decoder_control,
                        decoder_decode_work_progress.as_ref(),
                        decoder_unstable_decode_text.as_ref(),
                    );
                    runtime.release_session_scoped_buffers();
                    result
                })
                .map_err(|error| Self::map_actor_error("decoder", error))??
        };
        let decode_truncation = result.stop_reason.into_decode_truncation(None);

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
            decode_truncation,
        })
    }
}

fn decode_with_decoder(
    decoder: &mut FunasrNanoDecoderRuntime,
    decoder_metadata: &FunasrNanoDecoderMetadata,
    decode_prompt: &crate::models::qwen::Qwen3AsrDecodePrompt,
    speech_rows: &[f32],
    tokenizer: &FunasrNanoTokenizer,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    control: &Arc<crate::api::backend::TranscriptionControl>,
    decode_work_progress: Option<&crate::api::backend::WorkProgressObserver>,
    unstable_decode_text: Option<&crate::api::backend::UnstableDecodeTextObserver>,
) -> Result<Seq2SeqGreedyDecodeResult, FunasrNanoExecutorError> {
    let audio_pad_end = decode_prompt
        .audio_pad_start_index
        .checked_add(decode_prompt.audio_pad_count)
        .ok_or_else(|| FunasrNanoExecutorError::PromptEmbeddingFailed {
            reason: "audio pad position overflowed".to_string(),
        })?;
    let prompt_input = Qwen3AsrPromptTokenInput {
        token_ids: decode_prompt.token_ids.clone(),
        audio_rows: speech_rows.to_vec(),
        audio_positions: (decode_prompt.audio_pad_start_index..audio_pad_end).collect(),
    };

    let layer_kv_caches = decoder
        .new_kv_caches(kv_capacity)
        .map_err(|reason| FunasrNanoExecutorError::DecoderFailed { reason })?;
    let mut step_executor = FunasrNanoGreedyStepExecutor {
        decoder,
        layer_kv_caches,
        kv_capacity,
        prompt_input: Some(prompt_input),
        cache_prompt_tokens: 0,
        control: Arc::clone(control),
    };
    let config = BuiltinSeq2SeqDecodePolicyConfigInput {
        initial_prompt_tokens: decode_prompt.token_ids.clone(),
        eot_token_id: tokenizer.chatml_im_end_token_id,
        vocab_size: decoder_metadata.vocab_size,
        max_generated_tokens: FUNASR_NANO_MAX_GENERATED_TOKENS,
    };
    let result = run_builtin_seq2seq_decode_policy(
        FUNASR_NANO_DECODE_POLICY_ID,
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
        control,
        decode_work_progress,
        unstable_decode_text,
    )
    .map_err(|error| FunasrNanoExecutorError::GreedyDecodeFailed {
        reason: error.to_string(),
    })?;
    Ok(result)
}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

impl GgmlAsrViewExecutor for FunasrNanoGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        FunasrNanoGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        FUNASR_NANO_EXECUTOR_ID
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
                super::capacity::plan_funasr_nano_decoder_state,
                super::capacity::FUNASR_NANO_DECODER_STATE_STREAMS,
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

impl GgmlAsrStreamingExecutor for FunasrNanoGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FUNASR_NANO_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FUNASR_NANO_STREAMING_EXECUTOR_ID,
            crate::arch::FUNASR_NANO_GGML_ADAPTER_ID,
            "funasr-nano",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FunasrNanoGgmlExecutor::execute_streaming_view,
        )
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_actors();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

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
    fn unified_owner_is_limited_to_offline_exact_cuda_and_vulkan_full_device() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let preference = exactly_addressable_preference(provider);
            assert!(funasr_nano_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
            assert!(!funasr_nano_unified_runtime_enabled(
                false,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
            assert!(!funasr_nano_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::Hybrid),
            ));
        }
        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Hip,
            ExecutionProvider::Accelerator,
            ExecutionProvider::Unknown,
        ] {
            let preference = exactly_addressable_preference(provider);
            assert!(!funasr_nano_unified_runtime_enabled(
                true,
                GgmlCpuGraphBackend::Gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
        }
    }

    #[test]
    fn output_plan_partitions_unified_runtime_cache_identity() {
        let content = PackContentKey::new("sha256:funasr-nano-output-plan-fixture");
        let lane = current_execution_lane_key(GgmlCpuGraphBackend::Cpu);
        let full_logits = FunasrNanoUnifiedRuntimeCacheKey {
            content: content.clone(),
            lane: lane.clone(),
            output_plan: GgmlDecodeOutputPlan::FullLogits,
        };
        let compact = FunasrNanoUnifiedRuntimeCacheKey {
            content: content.clone(),
            lane: lane.clone(),
            output_plan: GgmlDecodeOutputPlan::NativeFirstMaxToken,
        };
        assert_ne!(full_logits, compact);

        let full_decoder: FunasrNanoDecoderRuntimeCacheKey = (
            content.clone(),
            lane.clone(),
            GgmlDecodeOutputPlan::FullLogits,
        );
        let compact_decoder: FunasrNanoDecoderRuntimeCacheKey =
            (content, lane, GgmlDecodeOutputPlan::NativeFirstMaxToken);
        assert_ne!(full_decoder, compact_decoder);
    }

    /// Bring-up golden: reads the committed reference LFR features + adaptor
    /// output + reference transcript for the two clips the model.pt-derived
    /// oracle produced (`OPENASR_FUNASR_NANO_GOLDEN_DIR`), plus the fp16 `.oasr`
    /// pack (`OPENASR_FUNASR_NANO_PACK`, ~1.97GB dev-only artifact, NOT
    /// committed). Runs the SAN-M encoder + transformer adaptor against the
    /// reference LFR and asserts a near-1.0 cosine similarity vs the reference
    /// adaptor output, then drives the Qwen3-0.6B decoder through the shared
    /// greedy loop and asserts the decoded transcript matches the reference
    /// text. Stays `#[ignore]`d (multi-GB pack) like every other builtin
    /// family's real-weights golden.
    fn golden_dir() -> Option<PathBuf> {
        std::env::var_os("OPENASR_FUNASR_NANO_GOLDEN_DIR").map(PathBuf::from)
    }

    fn pack_path() -> Option<PathBuf> {
        std::env::var_os("OPENASR_FUNASR_NANO_PACK").map(PathBuf::from)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        (dot / (na * nb + 1e-12)) as f32
    }

    #[test]
    #[ignore = "requires OPENASR_FUNASR_NANO_GOLDEN_DIR + the ~1.97GB dev-only \
                OPENASR_FUNASR_NANO_PACK fp16 .oasr; runs encoder+adaptor cosine parity vs the \
                model.pt oracle and end-to-end greedy decode vs the reference transcript"]
    fn golden_encoder_adapter_cosine_and_end_to_end_text() {
        let (Some(dir), Some(pack)) = (golden_dir(), pack_path()) else {
            eprintln!("skipping: set OPENASR_FUNASR_NANO_GOLDEN_DIR and OPENASR_FUNASR_NANO_PACK");
            return;
        };
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("runtime source");
        let preflight = crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
            &runtime_source,
        )
        .expect("runtime preflight");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&gguf_metadata).expect("encoder metadata");
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&gguf_metadata).expect("adapter metadata");
        let decoder_metadata =
            parse_funasr_nano_decoder_metadata(&gguf_metadata).expect("decoder metadata");
        let tokenizer = FunasrNanoTokenizer::from_gguf_metadata(&gguf_metadata).expect("tokenizer");

        for (tag, expected_text) in [
            (
                "en",
                "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.",
            ),
            ("zh", "开饭时间早上九点至下午五点。"),
        ] {
            let lfr = read_f32(&dir.join(format!("lfr_{tag}.bin")));
            let ref_adp = read_f32(&dir.join(format!("adp_{tag}.bin")));
            let n_frames = lfr.len() / encoder_metadata.feature_dim;

            let encoder_input = build_sensevoice_encoder_input(
                &[],
                &lfr,
                encoder_metadata.feature_dim,
                encoder_metadata.d_model,
            )
            .expect("encoder input");
            let (speech_rows_full, _) = {
                let mut encoder = FunasrNanoEncoderGraph::new_from_preflight(
                    &preflight,
                    encoder_metadata,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("encoder");
                let out = encoder
                    .encode(
                        &encoder_input.data,
                        encoder_input.n_frames,
                        encoder_input.feature_dim,
                    )
                    .expect("encode");
                let mut adapter = FunasrNanoAdapterGraph::new_from_preflight(
                    &preflight,
                    adapter_metadata,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("adapter");
                adapter
                    .run(&out.rows, out.frame_count, out.d_model)
                    .expect("adapter")
            };
            assert_eq!(
                speech_rows_full.len(),
                ref_adp.len(),
                "[{tag}] adaptor shape"
            );
            let cos = cosine(&speech_rows_full, &ref_adp);
            eprintln!("[{tag}] adaptor cosine = {cos:.6} (frames={n_frames})");
            assert!(cos > 0.999, "[{tag}] adaptor cosine {cos} below 0.999");

            // End-to-end greedy decode from the reference-derived audio rows.
            let n_aud = funasr_nano_audio_token_count(n_frames);
            let speech_rows = speech_rows_full[..n_aud * adapter_metadata.llm_dim].to_vec();
            let decode_prompt =
                build_funasr_nano_decode_prompt(&tokenizer, n_aud).expect("decode prompt");
            let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            );
            let mut decoder = FunasrNanoDecoderRuntime::new_from_preflight(
                &preflight,
                decoder_metadata,
                resolved_runtime,
            )
            .expect("decoder");
            let control = std::sync::Arc::new(crate::api::backend::TranscriptionControl::new());
            let result = decode_with_decoder(
                &mut decoder,
                &decoder_metadata,
                &decode_prompt,
                &speech_rows,
                &tokenizer,
                Qwen3AsrKvCacheCapacity::new(
                    decode_prompt.token_ids.len() + FUNASR_NANO_MAX_GENERATED_TOKENS,
                    decode_prompt.token_ids.len() + FUNASR_NANO_MAX_GENERATED_TOKENS,
                )
                .expect("test KV capacity"),
                &control,
                None,
                None,
            )
            .expect("decode");
            eprintln!("[{tag}] text = {}", result.text);
            assert_eq!(
                result.text.trim(),
                expected_text,
                "[{tag}] transcript mismatch"
            );
        }
    }

    /// Residency must not change output: the resident encoder+adaptor actor
    /// path (pool miss on the first call, then actor reuse at both a different
    /// and a previously seen frame count) must
    /// produce bit-for-bit the same audio-token rows as a freshly built
    /// one-shot encoder + adaptor over the same reference LFR features
    /// (the dolphin prepared-runtime bit-identity pinning pattern).
    #[test]
    #[ignore = "requires OPENASR_FUNASR_NANO_GOLDEN_DIR + the ~1.97GB dev-only \
                OPENASR_FUNASR_NANO_PACK fp16 .oasr; pins bit-identity of the resident \
                cached encoder+adaptor runtime vs a fresh one-shot build"]
    fn cached_encoder_adapter_matches_fresh_build_bit_for_bit() {
        let (Some(dir), Some(pack)) = (golden_dir(), pack_path()) else {
            eprintln!("skipping: set OPENASR_FUNASR_NANO_GOLDEN_DIR and OPENASR_FUNASR_NANO_PACK");
            return;
        };
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack).expect("runtime source");
        let preflight = crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
            &runtime_source,
        )
        .expect("runtime preflight");
        let gguf_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let encoder_metadata =
            parse_funasr_nano_encoder_metadata(&gguf_metadata).expect("encoder metadata");
        let adapter_metadata =
            parse_funasr_nano_adapter_metadata(&gguf_metadata).expect("adapter metadata");
        let executor = FunasrNanoGgmlExecutor::default();

        // en = cache miss (build + insert), zh = cache hit at a different
        // frame count, en again = cache hit at a previously seen frame count.
        for tag in ["en", "zh", "en"] {
            let lfr = read_f32(&dir.join(format!("lfr_{tag}.bin")));
            let encoder_input = build_sensevoice_encoder_input(
                &[],
                &lfr,
                encoder_metadata.feature_dim,
                encoder_metadata.d_model,
            )
            .expect("encoder input");

            // Fresh one-shot reference: a brand-new encoder + adaptor per call.
            let mut encoder = FunasrNanoEncoderGraph::new_from_preflight(
                &preflight,
                encoder_metadata,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("fresh encoder");
            let out = encoder
                .encode(
                    &encoder_input.data,
                    encoder_input.n_frames,
                    encoder_input.feature_dim,
                )
                .expect("fresh encode");
            let mut adapter = FunasrNanoAdapterGraph::new_from_preflight(
                &preflight,
                adapter_metadata,
                GgmlCpuGraphBackend::Cpu,
            )
            .expect("fresh adapter");
            let (full_rows, adapter_frames) = adapter
                .run(&out.rows, out.frame_count, out.d_model)
                .expect("fresh adapter run");
            let fresh_n_aud = funasr_nano_audio_token_count(out.frame_count).min(adapter_frames);
            let fresh_rows = &full_rows[..fresh_n_aud * adapter_metadata.llm_dim];

            // Resident cached path (what execute_inner runs).
            let (cached_rows, cached_n_aud) = executor
                .encode_with_owned_encoder_adapter_runtime(
                    &preflight,
                    encoder_metadata,
                    adapter_metadata,
                    encoder_input,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("cached encoder+adapter");

            assert_eq!(cached_n_aud, fresh_n_aud, "[{tag}] audio token count");
            assert_eq!(cached_rows.len(), fresh_rows.len(), "[{tag}] row length");
            for (index, (cached, fresh)) in cached_rows.iter().zip(fresh_rows).enumerate() {
                assert_eq!(
                    cached.to_bits(),
                    fresh.to_bits(),
                    "[{tag}] audio-token value {index} differs: cached {cached} vs fresh {fresh}"
                );
            }
            eprintln!("[{tag}] cached == fresh bit-for-bit ({cached_n_aud} audio tokens)");
        }
    }
}
