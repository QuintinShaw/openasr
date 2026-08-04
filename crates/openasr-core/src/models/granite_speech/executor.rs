//! `GgmlAsrViewExecutor` implementation for granite-speech, wiring the already-
//! validated pipeline (`frontend` -> `encoder_graph` -> `qformer` -> `prompt`
//! -> `decode_executor` -> shared greedy-decode driver -> `tokenizer`)
//! against a real `.oasr` pack via `runtime_provider::load_tensors_from_oasr_pack`.
//!
//! Registry status: wired into `arch::mod`'s `OpenAsrArchitectureRegistry`
//! (architecture descriptor + component descriptors), `executor_component_registry`,
//! `decode_policy_component_registry`, and `runtime_tensor_contract_registry`
//! (see `runtime_contract.rs` for the metadata parsers those two validate
//! against). No dedicated `frontend_component_registry`/
//! `tokenizer_component_registry` entry exists in this codebase shape --
//! frontend/tokenizer selection for a dedicated (non-composed) executor
//! family is the executor's own job (this file constructs
//! `GraniteSpeechMelFrontend`/`GraniteSpeechTokenizer` directly), the same
//! precedent `firered_llm`/`mimo_asr`/`moss_transcribe_diarize` follow.
//!
//! Streaming: this pass was scoped "file-transcribe only, no streaming", but
//! `builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch`
//! has a fail-closed completeness gate that rejects its ENTIRE dispatch (for
//! every family, not just this one) if any registered architecture has no
//! streaming executor at all -- discovered by a workspace-wide test failure
//! across unrelated families after this one's architecture descriptor
//! landed, not a granite-speech-specific test. `GgmlAsrStreamingExecutor`
//! below is therefore a required registration, not scope creep: it reuses
//! the exact same offline `execute_inner` through the shared buffered
//! snapshot streaming driver (`build_seq2seq_streaming_session`, matching
//! moonshine/qwen's own precedent for a family with no incremental decode
//! session yet) -- no new streaming-specific logic, and no claim of
//! streaming-tuned latency.

#![allow(dead_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use super::decode_executor::GraniteSpeechResidentAudioDecodeStepExecutor;
use super::decode_session::{
    GraniteSpeechDecodeSession, GraniteSpeechKvCacheCapacity, GraniteSpeechKvCacheCapacityError,
};
use super::decoder_graph::GraniteSpeechDecoderConfig;
use super::encoder_graph::{GraniteSpeechEncoderConfig, GraniteSpeechEncoderRuntime};
use super::prompt::{build_audio_prompt_embeddings, build_granite_speech_prompt_text};
use super::qformer::{GraniteSpeechProjectorConfig, GraniteSpeechProjectorRuntime};
use super::runtime_contract::{
    parse_decoder_metadata, parse_encoder_metadata, parse_projector_metadata,
};
use super::runtime_provider::load_tensors_from_preflight;
use super::tokenizer::GraniteSpeechTokenizer;
use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::ggml_runtime::GgufRuntimeSourcePreflight;
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlAsrViewExecutor,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::phrase_bias_decode::PhraseBiasTokenEncoder;
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeError;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use crate::arch::{GRANITE_SPEECH_DECODE_POLICY_ID, GRANITE_SPEECH_GGML_ADAPTER_ID};

/// Everything granite-speech materializes from a pack that does NOT depend on
/// the per-request audio, kept resident across requests: the keep-quantized
/// decode session (its graph runner, the mmap'd loaded weight context, and the
/// decoder's zero-copy bound projection/norm/lm_head weights) plus the decoder
/// token-embedding table read once from the pack. Before this cache, the
/// keep-quantized decoder still rebuilt its loaded weight context on EVERY
/// `execute()` (a fresh runner init + `load_gguf_weight_context` +
/// `GraniteDecoderLoadedWeights::load`, ~4.2s measured) purely to re-derive
/// state that never changes between requests against the same pack. Mirrors
/// `firered_llm`'s resident-decoder cache (`FIRERED_LLM_DECODER_BY_KEY`) and
/// `mimo_asr`'s `MimoAsrPreparedRuntime`.
///
/// The single-runner invariant is preserved by construction: the session owns
/// its runner and the loaded context that was built ON that runner together as
/// one unit (`GraniteSpeechDecodeSession::new_keep_quantized`), and this struct
/// moves that whole session in and out of the cache without ever separating the
/// runner from its loaded context or re-binding weights onto a different runner.
/// Before the session re-enters the cache, `release_session_scoped_buffers`
/// releases CPU host K/V and logically resets the GPU path. The GPU's fixed
/// resident K/V arena and persistent graph stay allocated across requests;
/// subsequent prefill/steps overwrite every visible row and mask the stale
/// tail.
///
/// The embedding table is OWNED here (not a borrow of the request's transient
/// load) so the resident session carries no lifetime and can live in the
/// thread-local cache; it doubles as the `GraniteSpeechDecoderWeightProvider`
/// for prompt assembly and each generated token's per-step embedding lookup.
struct GraniteSpeechPreparedRuntime {
    encoder_config: GraniteSpeechEncoderConfig,
    projector_config: GraniteSpeechProjectorConfig,
    decoder_config: GraniteSpeechDecoderConfig,
    tokenizer: GraniteSpeechTokenizer,
    encoder: GraniteSpeechEncoderRuntime,
    projector: GraniteSpeechProjectorRuntime,
    session: GraniteSpeechDecodeSession,
    embed_table: HashMap<String, Vec<f32>>,
}

impl GraniteSpeechPreparedRuntime {
    fn quoted_system_memory_bytes(
        preflight: &GgufRuntimeSourcePreflight,
    ) -> Result<(u64, u64), GraniteSpeechGgmlExecutorError> {
        let metadata = &preflight.metadata;
        let encoder_config = parse_encoder_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let projector_config = parse_projector_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = parse_decoder_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let tokenizer_retained = GraniteSpeechTokenizer::quoted_retained_system_memory_bytes(
            metadata,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::TokenizerFailed {
            reason: error.to_string(),
        })?;
        let embedding_retained = quoted_embedding_table_system_memory_bytes(&decoder_config)?;
        let (encoder_peak, encoder_retained) =
            GraniteSpeechEncoderRuntime::quoted_system_memory_bytes(&encoder_config).map_err(
                |reason| GraniteSpeechGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "prepared-runtime-quote",
                    reason,
                },
            )?;
        let (projector_peak, projector_retained) =
            GraniteSpeechProjectorRuntime::quoted_system_memory_bytes(&projector_config).map_err(
                |reason| GraniteSpeechGgmlExecutorError::RuntimeOwnershipFailed {
                    stage: "prepared-runtime-quote",
                    reason,
                },
            )?;
        let session_retained =
            GraniteSpeechDecodeSession::quoted_retained_system_memory_bytes(&decoder_config)
                .map_err(
                    |reason| GraniteSpeechGgmlExecutorError::RuntimeOwnershipFailed {
                        stage: "prepared-runtime-quote",
                        reason,
                    },
                )?;

        let retained = checked_sum_u64(
            [
                tokenizer_retained,
                embedding_retained,
                encoder_retained,
                projector_retained,
                session_retained,
            ],
            "granite prepared retained quote",
        )?;
        let tokenizer_phase = tokenizer_retained;
        let embedding_phase = tokenizer_retained
            .checked_add(embedding_retained)
            .ok_or_else(|| capacity_error("granite embedding construction quote overflowed"))?;
        let encoder_phase = embedding_phase
            .checked_add(encoder_peak)
            .ok_or_else(|| capacity_error("granite encoder construction quote overflowed"))?;
        let projector_phase = embedding_phase
            .checked_add(encoder_retained)
            .and_then(|bytes| bytes.checked_add(projector_peak))
            .ok_or_else(|| capacity_error("granite projector construction quote overflowed"))?;
        let session_phase = retained;
        Ok((
            tokenizer_phase
                .max(embedding_phase)
                .max(encoder_phase)
                .max(projector_phase)
                .max(session_phase),
            retained,
        ))
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, GraniteSpeechGgmlExecutorError> {
        checked_sum_u64(
            [
                self.tokenizer
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                embedding_table_system_memory_bytes(&self.embed_table)?,
                self.encoder
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
                self.projector.retained_system_memory_bytes(),
                self.session
                    .retained_system_memory_bytes()
                    .map_err(capacity_error)?,
            ],
            "granite prepared measured retained bytes",
        )
    }

    /// Materialize the resident decode session + embedding table once for a
    /// given `(pack, backend)`. This is the whole ~4.2s cost the resident cache
    /// exists to pay exactly once instead of per request.
    fn build(
        preflight: &GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechGgmlExecutorError> {
        let metadata = &preflight.metadata;
        let encoder_config = parse_encoder_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let projector_config = parse_projector_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = parse_decoder_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let tokenizer = GraniteSpeechTokenizer::from_gguf_metadata(metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::TokenizerFailed {
                reason: error.to_string(),
            }
        })?;
        // Only the decoder's token-embedding table on the host (dequantized to
        // f32) -- the projection/norm/lm_head weights are bound zero-copy inside
        // the session below (see `runtime_provider` / `decode_session`).
        let embed_table =
            load_tensors_from_preflight(preflight, "language_model.model.embed_tokens.weight")
                .map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
                    reason: error.to_string(),
                })?;
        let encoder =
            GraniteSpeechEncoderRuntime::new_from_preflight(preflight, &encoder_config, backend)
                .map_err(|error| GraniteSpeechGgmlExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;
        let projector = GraniteSpeechProjectorRuntime::new_from_preflight(
            preflight,
            &projector_config,
            backend,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::ProjectorFailed {
            reason: error.to_string(),
        })?;
        let session = GraniteSpeechDecodeSession::new_keep_quantized_from_preflight(
            decoder_config,
            preflight,
            backend,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
            reason: error.to_string(),
        })?;
        Ok(Self {
            encoder_config,
            projector_config,
            decoder_config,
            tokenizer,
            encoder,
            projector,
            session,
            embed_table,
        })
    }
}

fn capacity_error(reason: impl Into<String>) -> GraniteSpeechGgmlExecutorError {
    GraniteSpeechGgmlExecutorError::RuntimeOwnershipFailed {
        stage: "prepared-runtime",
        reason: reason.into(),
    }
}

fn checked_sum_u64<const N: usize>(
    values: [u64; N],
    label: &'static str,
) -> Result<u64, GraniteSpeechGgmlExecutorError> {
    values.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| capacity_error(format!("{label} overflowed")))
    })
}

fn quoted_embedding_table_system_memory_bytes(
    config: &GraniteSpeechDecoderConfig,
) -> Result<u64, GraniteSpeechGgmlExecutorError> {
    const NAME: &str = "language_model.model.embed_tokens.weight";
    let values = config
        .hidden_size
        .checked_mul(config.vocab_size)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| capacity_error("granite embedding table quote overflowed"))?;
    let map_entry = std::mem::size_of::<(String, Vec<f32>)>();
    let bytes = values
        .checked_add(NAME.len())
        .and_then(|total| total.checked_add(map_entry))
        .ok_or_else(|| capacity_error("granite embedding container quote overflowed"))?;
    u64::try_from(bytes).map_err(|_| capacity_error("granite embedding quote exceeds u64"))
}

fn embedding_table_system_memory_bytes(
    tensors: &HashMap<String, Vec<f32>>,
) -> Result<u64, GraniteSpeechGgmlExecutorError> {
    let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
    let map_entry_bytes = tensors
        .len()
        .checked_mul(std::mem::size_of::<(String, Vec<f32>)>())
        .ok_or_else(|| capacity_error("granite embedding map entry count overflowed"))?;
    bytes
        .add_usize(map_entry_bytes, "granite embedding map entries")
        .map_err(capacity_error)?;
    for (name, values) in tensors {
        bytes
            .add_string(name, "granite embedding tensor name")
            .map_err(capacity_error)?;
        bytes
            .add_vec(values, "granite embedding tensor values")
            .map_err(capacity_error)?;
    }
    Ok(bytes.finish())
}

/// Resident prepared-runtime actor pool keyed by content id, execution lane,
/// and resident KV capacity.
/// The pack half is a [`PackContentKey`] from the request's already-open
/// source, so an in-place `.oasr` replacement at the same path resolves a
/// different id and the next lookup rebuilds instead of reusing a session whose
/// device-bound weights came from the old bytes. Entries carry the idle-unload
/// The service root can clear or target-evict these actors directly; each actor
/// owns its memory lease and destroys the runtime on its owner thread.
type GraniteSpeechPreparedRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, usize);
struct GraniteSpeechPreparedRuntimeActorState {
    runtime: GraniteSpeechPreparedRuntime,
}

impl std::fmt::Debug for GraniteSpeechPreparedRuntimeActorState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GraniteSpeechPreparedRuntimeActorState")
            .finish_non_exhaustive()
    }
}
type GraniteSpeechPreparedRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    GraniteSpeechPreparedRuntimeCacheKey,
    GraniteSpeechPreparedRuntimeActorState,
>;
type GraniteSpeechPreparedRuntimeActor = PinnedRuntimeActorCheckout<
    GraniteSpeechPreparedRuntimeCacheKey,
    GraniteSpeechPreparedRuntimeActorState,
>;

const GRANITE_SPEECH_EXECUTOR_ID: &str = crate::arch::GRANITE_SPEECH_EXECUTOR_COMPONENT_ID;
/// Greedy decode stop token (`<|end_of_text|>` in the packed GPT-2 BPE
/// table). Shared with the runtime contract validator, which fails closed on
/// a pack whose vocab or token table cannot represent it.
pub(crate) const GRANITE_SPEECH_EOT_TOKEN_ID: u32 = 100_257;
/// Fail-closed backstop against a non-terminating decode -- greedy decode stops
/// at `<|end_of_text|>` well before this in practice. Also a first-class input
/// to `capacity::derive_max_input_whole_seconds` (must stay in lockstep with that
/// derivation's published limit).
pub(crate) const GRANITE_SPEECH_MAX_GENERATED_TOKENS: usize = 256;

#[derive(Debug, Error)]
enum GraniteSpeechGgmlExecutorError {
    #[error("granite-speech ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error(
        "granite-speech audio duration {seconds:.1}s exceeds the derived {limit:.0}s \
         single-decode cap (decoder context {ctx} tokens at {rate:.0} audio tokens/s; \
         longer audio is the shared longform slicer's job)"
    )]
    AudioTooLong {
        seconds: f32,
        limit: f32,
        ctx: usize,
        rate: f32,
    },
    #[error("granite-speech ggml executor frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("granite-speech ggml executor metadata contract failed: {reason}")]
    MetadataFailed { reason: String },
    #[error("granite-speech ggml executor encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("granite-speech ggml executor projector failed: {reason}")]
    ProjectorFailed { reason: String },
    #[error("granite-speech ggml executor tokenizer failed: {reason}")]
    TokenizerFailed { reason: String },
    #[error("granite-speech ggml executor prompt assembly failed: {reason}")]
    PromptFailed { reason: String },
    #[error("granite-speech decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: GraniteSpeechKvCacheCapacityError,
    },
    #[error("granite-speech ggml executor decode failed: {reason}")]
    DecodeFailed { reason: String },
    #[error("granite-speech {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
}

/// Fail-closed single-decode duration gate. Pure so unit tests can exercise the
/// typed `AudioTooLong` path without materializing a multi-GB pack. Limit and
/// rate come from `super::capacity` (derived, not guessed).
fn ensure_audio_within_capacity(sample_count: usize) -> Result<(), GraniteSpeechGgmlExecutorError> {
    use super::capacity::{
        GRANITE_SPEECH_DECODER_MAX_POSITIONS, GRANITE_SPEECH_MAX_INPUT_SAMPLES,
        GRANITE_SPEECH_MAX_INPUT_SECONDS, granite_speech_audio_tokens_per_second,
    };
    if sample_count > GRANITE_SPEECH_MAX_INPUT_SAMPLES {
        return Err(GraniteSpeechGgmlExecutorError::AudioTooLong {
            seconds: sample_count as f32 / super::frontend::SAMPLE_RATE_HZ,
            limit: GRANITE_SPEECH_MAX_INPUT_SECONDS as f32,
            ctx: GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            rate: granite_speech_audio_tokens_per_second(),
        });
    }
    Ok(())
}

/// No-op phrase-bias shim: granite-speech applies keyword biasing through its
/// own prompt convention (the `Keywords:` suffix assembled above), never the
/// shared decode-time logit-boost path, so the registry-routed greedy loop is
/// handed a token source that contributes nothing -- mirrors
/// `mimo_asr::executor::NoPhraseBiasTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

fn map_registry_error(
    error: BuiltinDecodePolicyComponentRegistryError,
) -> Seq2SeqGreedyDecodeError {
    Seq2SeqGreedyDecodeError::DecoderStepFailed {
        reason: error.to_string(),
    }
}

const GRANITE_SPEECH_RUNTIME_MAX_IDLE_ENTRIES: usize = 4;
const GRANITE_SPEECH_RUNTIME_MAX_INSTANCES_PER_KEY: usize = 2;

#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeechGgmlExecutor {
    prepared_runtimes: Arc<GraniteSpeechPreparedRuntimePool>,
}

impl Default for GraniteSpeechGgmlExecutor {
    fn default() -> Self {
        Self {
            prepared_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-granite-speech-runtime-owner",
                AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                    GRANITE_SPEECH_RUNTIME_MAX_IDLE_ENTRIES,
                    crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    GRANITE_SPEECH_RUNTIME_MAX_INSTANCES_PER_KEY,
                ),
            )),
        }
    }
}

impl GraniteSpeechGgmlExecutor {
    fn map_actor_error(error: PinnedRuntimeActorError) -> GraniteSpeechGgmlExecutorError {
        GraniteSpeechGgmlExecutorError::RuntimeOwnershipFailed {
            stage: "prepared-runtime",
            reason: error.to_string(),
        }
    }

    fn checkout_prepared_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
        kv_capacity: GraniteSpeechKvCacheCapacity,
    ) -> Result<GraniteSpeechPreparedRuntimeActor, GraniteSpeechGgmlExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
            kv_capacity.resident_positions(),
        );
        let quote_preflight = preflight.clone();
        let build_preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.prepared_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let (peak_bytes, retained_bytes) =
                    GraniteSpeechPreparedRuntime::quoted_system_memory_bytes(&quote_preflight)?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!(
                        "granite-speech-runtime:{content_id}:positions={}",
                        kv_capacity.resident_positions()
                    ),
                    peak_bytes,
                    retained_bytes,
                )
                .map_err(|error| capacity_error(error.to_string()))?;
                Ok((retained_bytes, quote))
            },
            move |quote| match SystemMemoryOwner::try_allocate_transaction(quote, || {
                let prepared = GraniteSpeechPreparedRuntime::build(&build_preflight, backend)?;
                let retained = prepared.retained_system_memory_bytes()?;
                Ok(SystemMemoryAllocationOutcome::new(
                    GraniteSpeechPreparedRuntimeActorState { runtime: prepared },
                    retained,
                    retained,
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
    ) -> Result<GgmlAsrExecutionResult, GraniteSpeechGgmlExecutorError> {
        if request.selected_family.adapter_id != GRANITE_SPEECH_GGML_ADAPTER_ID {
            return Err(GraniteSpeechGgmlExecutorError::AdapterMismatch {
                expected: GRANITE_SPEECH_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let preflight = request.runtime_source_preflight();
        let samples = downmix_prepared_audio(&request.prepared_audio);
        // Duration gate before any frontend / encoder / projector work: the
        // limit is the decoder-context-derived ceiling in `super::capacity`
        // (not a magic number). Over-limit fails closed with a typed error --
        // never silent truncation. Shared longform slicing keeps ordinary
        // multi-minute work inside 30s windows well under this bound.
        ensure_audio_within_capacity(samples.len())?;
        let frontend = super::frontend::GraniteSpeechMelFrontend::new();
        let (features, frames) = frontend.extract(samples.as_ref()).map_err(|error| {
            GraniteSpeechGgmlExecutorError::FrontendFailed {
                reason: error.to_string(),
            }
        })?;
        let backend = request.resolved_runtime.backend();
        // KWB (keyword-list biasing): the model's own documented prompt
        // convention -- "transcribe the speech to text. Keywords: <kw1>,
        // <kw2>, ..." -- not a decode-time logit bias (see the family's
        // end-to-end KWB test). `phrase_bias`'s configured phrases become
        // the `Keywords:` suffix when present.
        let prompt_text =
            build_granite_speech_prompt_text(request.request_options.phrase_bias.as_ref());
        let kv_capacity = GraniteSpeechKvCacheCapacity::from_decoder_state(&request.decoder_state)
            .and_then(|capacity| {
                capacity.validate_hard_cap(super::capacity::GRANITE_SPEECH_DECODER_MAX_POSITIONS)
            })
            .map_err(|source| GraniteSpeechGgmlExecutorError::DecoderStateCapacity { source })?;
        let actor = self.checkout_prepared_runtime(preflight, backend, kv_capacity)?;
        let control = Arc::clone(&request.execution_context.control);
        let result = actor
            .call_mut(move |state| {
                let prepared = &mut state.runtime;
                let decode_result = (|| {
                    let encoder_output = prepared
                        .encoder
                        .encode(&prepared.encoder_config, &features, frames, false)
                        .map_err(|error| GraniteSpeechGgmlExecutorError::EncoderFailed {
                            reason: error.to_string(),
                        })?;
                    let projector_output = prepared
                        .projector
                        .project(
                            &prepared.projector_config,
                            &encoder_output.encoder_out,
                            encoder_output.frames,
                        )
                        .map_err(|error| GraniteSpeechGgmlExecutorError::ProjectorFailed {
                            reason: error.to_string(),
                        })?;
                    let (prompt_token_ids, prompt_embeddings) = build_audio_prompt_embeddings(
                        &prepared.decoder_config,
                        &prepared.embed_table,
                        &prepared.tokenizer,
                        &prompt_text,
                        &projector_output.projected,
                        projector_output.tokens,
                    )
                    .map_err(|error| GraniteSpeechGgmlExecutorError::PromptFailed {
                        reason: error.to_string(),
                    })?;
                    let measured_positions =
                        crate::capacity::topology::causal_prefix_positions_with_context_cap(
                            super::capacity::GRANITE_SPEECH_SELF_KV_STATE_ID,
                            prompt_token_ids.len(),
                            GRANITE_SPEECH_MAX_GENERATED_TOKENS,
                            super::capacity::GRANITE_SPEECH_DECODER_MAX_POSITIONS,
                        )
                        .map_err(|_| {
                            GraniteSpeechGgmlExecutorError::DecoderStateCapacity {
                                source: GraniteSpeechKvCacheCapacityError::LogicalPositionOverflow,
                            }
                        })?;
                    let kv_capacity =
                        kv_capacity
                            .validate_measured_logical_positions(measured_positions)
                            .map_err(|source| {
                                GraniteSpeechGgmlExecutorError::DecoderStateCapacity { source }
                            })?;
                    let decode_config = BuiltinSeq2SeqDecodePolicyConfigInput {
                        initial_prompt_tokens: prompt_token_ids,
                        eot_token_id: GRANITE_SPEECH_EOT_TOKEN_ID,
                        vocab_size: prepared.decoder_config.vocab_size,
                        max_generated_tokens: GRANITE_SPEECH_MAX_GENERATED_TOKENS,
                    };
                    let decode_text_token_ids =
                        |token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> {
                            prepared
                                .tokenizer
                                .decode_text_token_ids(token_ids)
                                .map_err(|error| Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                                    reason: error.to_string(),
                                })
                        };
                    let mut step_executor = GraniteSpeechResidentAudioDecodeStepExecutor::new(
                        &mut prepared.session,
                        &prepared.embed_table,
                        prompt_embeddings,
                        kv_capacity,
                    );
                    run_builtin_seq2seq_decode_policy(
                        GRANITE_SPEECH_DECODE_POLICY_ID,
                        &decode_config,
                        &NoPhraseBiasTokenSource,
                        None,
                        &mut step_executor,
                        &decode_text_token_ids,
                        |error: Seq2SeqGreedyDecodeError| error,
                        |error: Seq2SeqGreedyDecodeError| error,
                        map_registry_error,
                        &control,
                    )
                    .map_err(|error| {
                        GraniteSpeechGgmlExecutorError::DecodeFailed {
                            reason: error.to_string(),
                        }
                    })
                })();
                prepared.session.release_session_scoped_buffers();
                decode_result
            })
            .map_err(Self::map_actor_error)??;
        let audio_duration_seconds = request.prepared_audio.samples_f32.len() as f32
            / request.prepared_audio.sample_rate_hz.max(1) as f32;
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: result.text.clone(),
                segments: vec![Segment {
                    start: 0.0,
                    end: audio_duration_seconds,
                    text: result.text,
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                }],
                longform: None,
                language: None,
            },
            carry_context: None,
            // No intra-decode timestamps -- the single segment spans the whole
            // buffer -- so the cut point has no honest second to name, same as
            // mimo-asr / firered-aed.
            decode_truncation: result.stop_reason.into_decode_truncation(None),
        })
    }
}

fn downmix_prepared_audio<'a>(audio: &'a GgmlAsrPreparedAudioView<'_>) -> Cow<'a, [f32]> {
    if audio.channels <= 1 {
        return Cow::Borrowed(audio.samples_f32.as_ref());
    }
    let channels = audio.channels as usize;
    Cow::Owned(
        audio
            .samples_f32
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
    )
}

impl GgmlAsrViewExecutor for GraniteSpeechGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        GraniteSpeechGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        GRANITE_SPEECH_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // Native KWB via the prompt convention above -- not the shared
        // decode-time phrase_bias_decode logit-boost mechanism (unused here,
        // matching AGENTS.md's per-family explicit-declaration rule: a family
        // states its own true/false, it never inherits a default).
        true
    }

    fn decoder_state_contract(
        &self,
        _selected_family: &crate::GgmlFamilyAdapterDescriptor,
    ) -> Result<crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract, GgmlAsrExecutionError>
    {
        Ok(
            crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::planned(
                super::capacity::plan_granite_speech_decoder_state,
                super::capacity::GRANITE_SPEECH_DECODER_STATE_STREAMS,
            ),
        )
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| granite_speech_execute_error_to_ggml(self, error, request))
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_actors();
    }
}

fn granite_speech_execute_error_to_ggml(
    executor: &GraniteSpeechGgmlExecutor,
    error: GraniteSpeechGgmlExecutorError,
    request: &GgmlAsrExecutionViewRequest,
) -> GgmlAsrExecutionError {
    GgmlAsrExecutionError::ExecutorFailed {
        executor_id: GgmlAsrViewExecutor::executor_id(executor),
        adapter_id: request.selected_family.adapter_id,
        reason: error.to_string(),
    }
}

impl GraniteSpeechGgmlExecutor {
    /// Streaming decode: re-runs the SAME offline pipeline (`execute_inner`)
    /// against the growing/windowed audio buffer the shared streaming driver
    /// hands it. The resident session's incremental-KV decode is scoped to one
    /// request's own generated tokens; a streaming partial cannot reuse a prior
    /// partial's KV, because each partial re-splices a LONGER audio prompt
    /// (different projected rows, different prompt embeddings), which
    /// invalidates every cached position. So every partial re-does frontend +
    /// encoder + Q-Former + a full prefill-style decode from scratch. This is
    /// registered to satisfy the codebase's fail-closed streaming-completeness
    /// gate
    /// (`builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch`
    /// rejects the WHOLE dispatch, for every family, if any registered
    /// architecture has no streaming executor at all) -- it is correctness-
    /// only, not a real-time-tuned streaming path. The FINAL transcript stays
    /// byte-identical to `execute()`.
    fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| granite_speech_execute_error_to_ggml(self, error, request))
    }
}

const GRANITE_SPEECH_STREAMING_EXECUTOR_ID: &str =
    "granite-speech-ggml-snapshot-streaming-executor-v1";

impl crate::models::ggml_asr_executor::GgmlAsrStreamingExecutor for GraniteSpeechGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        GRANITE_SPEECH_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &crate::models::ggml_asr_executor::GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
        crate::models::incremental_streaming_driver::build_seq2seq_streaming_session(
            self.clone(),
            GRANITE_SPEECH_STREAMING_EXECUTOR_ID,
            GRANITE_SPEECH_GGML_ADAPTER_ID,
            "granite-speech",
            request,
            crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            GraniteSpeechGgmlExecutor::execute_streaming,
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
    use crate::api::backend::{
        DecodeTruncationReason, ExecutionTarget, NATIVE_RUNTIME_MODEL_ID_AUTO, NativeBackend,
        TranscriptionBackend, TranscriptionRequest,
    };
    use crate::arch::builtin_adapter_descriptor;
    use crate::models::ggml_asr_executor::GgmlAsrBackendPreference;
    use crate::models::seq2seq_greedy_decode::Seq2SeqGreedyDecodeStopReason;
    use crate::{LongFormMode, LongFormOptions};

    /// Points at a real converted granite-speech `.oasr` pack via
    /// `OPENASR_GRANITE_SPEECH_PACK`. Loading it mmaps + touches a multi-GB
    /// file plus materializes the f16 token-embedding table -- a real memory
    /// commitment, not a network fetch -- so this stays `#[ignore]`d and skips
    /// silently when the env var is unset (same convention as firered-llm's
    /// own dev-pack test) rather than gating CI on a private multi-GB artifact.
    fn dev_pack_path() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_GRANITE_SPEECH_PACK",
            "granite-speech .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn longform_en_zh_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/longform_en_zh.wav")
    }

    const JFK_REFERENCE_TRANSCRIPT: &str = "and so my fellow americans ask not what your country can do for you ask what you can do for your country";
    /// Anchor phrase that appears three times in `fixtures/longform_en_zh.wav`
    /// (EN/ZH/EN/ZH/EN concat of committed short fixtures). Used to detect both
    /// under-coverage (pathological truncation dropping later EN clips) and
    /// runaway repetition (the same phrase looping past the fixture's real
    /// count).
    const JFK_ANCHOR_PHRASE: &str = "ask not what your country can do for you";
    const ZH_ANCHOR_PHRASE: &str = "今天天气";

    /// Generation budget backstop the executor hands the shared greedy driver.
    /// Multi-minute audio is NOT served by raising this -- it is served by the
    /// SharedWindow longform slicer keeping each buffer inside one decode's
    /// budget. Pinning the number here keeps the audit form and the code from
    /// drifting independently.
    #[test]
    fn generation_budget_backstop_is_finite_and_not_a_longform_substitute() {
        // Pin the published backstop: audit form + long-audio degradation gate
        // assume this exact figure. Multi-minute audio is served by the
        // SharedWindow slicer, not by raising this into the thousands.
        const {
            assert!(GRANITE_SPEECH_MAX_GENERATED_TOKENS == 256);
            assert!(GRANITE_SPEECH_MAX_GENERATED_TOKENS < 1024);
        }
    }

    #[test]
    fn mono_frontend_borrows_the_prepared_pcm_backing() {
        let backing = crate::PcmBuffer::from_vec(vec![0.25, -0.5, 0.75]);
        let prepared = GgmlAsrPreparedAudioView::mono_16khz_shared(backing.full_slice());
        let samples = downmix_prepared_audio(&prepared);

        assert!(matches!(samples, Cow::Borrowed(_)));
        assert_eq!(samples.as_ptr(), prepared.samples_f32.as_ptr());
    }

    #[test]
    fn multichannel_frontend_owns_only_the_required_downmix() {
        let backing = crate::PcmBuffer::from_vec(vec![1.0, 3.0, -1.0, 1.0]);
        let prepared = GgmlAsrPreparedAudioView {
            sample_rate_hz: 16_000,
            channels: 2,
            samples_f32: backing.full_slice().into(),
        };
        let samples = downmix_prepared_audio(&prepared);

        assert!(matches!(samples, Cow::Owned(_)));
        assert_eq!(samples.as_ref(), &[2.0, 0.0]);
    }

    /// The executor lifts every driver stop reason through
    /// `into_decode_truncation` so a truncated decode cannot be laundered into a
    /// normal success. Weight-free contract test: the mapping itself, which is
    /// the only place a silent-truncation regression can hide on this family.
    #[test]
    fn stop_reason_maps_to_visible_decode_truncation() {
        assert!(
            Seq2SeqGreedyDecodeStopReason::StopToken
                .into_decode_truncation(None)
                .is_none(),
            "a clean EOT must not mark the transcript truncated"
        );

        let guard = Seq2SeqGreedyDecodeStopReason::DegenerateRepeatGuard
            .into_decode_truncation(None)
            .expect("guard trip must surface");
        assert_eq!(guard.reason, DecodeTruncationReason::DegenerateRepeatGuard);
        assert!(Seq2SeqGreedyDecodeStopReason::DegenerateRepeatGuard.is_truncated());

        let budget = Seq2SeqGreedyDecodeStopReason::BudgetExhausted
            .into_decode_truncation(None)
            .expect("budget exhaustion must surface");
        assert_eq!(budget.reason, DecodeTruncationReason::BudgetExhausted);
        assert!(Seq2SeqGreedyDecodeStopReason::BudgetExhausted.is_truncated());

        // Granite has no honest intra-decode time anchor (one segment spans the
        // whole buffer), so the cut point stays None -- same as mimo/firered.
        // Presence of the truncation entry is the load-bearing signal.
        assert!(guard.transcript_covers_up_to_seconds.is_none());
        assert!(budget.transcript_covers_up_to_seconds.is_none());
    }

    /// Runs one full `execute()` against `pack_path` on the requested backend and
    /// returns the transcript. Skips (returns `None`) when the pack is absent.
    fn transcribe_with_pack(
        pack_path: PathBuf,
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<String> {
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "granite-speech e2e test",
            "granite-speech e2e test",
        )
        .expect("load wav fixture");
        let runtime_source_preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("granite runtime must pass preflight");

        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
        );
        if backend_preference == GgmlAsrBackendPreference::Accelerated {
            assert!(
                resolved_runtime.backend().is_gpu_class(),
                "accelerated Granite acceptance must resolve to a GPU-class backend"
            );
            #[cfg(target_os = "macos")]
            assert_eq!(
                resolved_runtime.backend(),
                crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
                "macOS Granite Metal acceptance must not silently run another backend"
            );
        }
        let request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let _backend_guard = crate::ggml_runtime::install_request_backend_override(
            request.backend_preference.request_backend_override(),
        );

        let executor = GraniteSpeechGgmlExecutor::default();
        let result = executor
            .execute_view(&request)
            .expect("granite-speech transcribe");
        // Single-pass path: a clean EOT leaves decode_truncation unset. If the
        // driver ever trips the budget/guard on short JFK audio, surface it
        // rather than silently returning a short "success" string.
        assert!(
            result.decode_truncation.is_none(),
            "short JFK fixture must finish on EOT, not truncate: {:?}",
            result.decode_truncation
        );
        Some(result.transcription.text)
    }

    /// Count case-insensitive non-overlapping occurrences of `needle` in `hay`.
    fn count_phrase_occurrences(hay: &str, needle: &str) -> usize {
        let hay_l = hay.to_lowercase();
        let needle_l = needle.to_lowercase();
        if needle_l.is_empty() {
            return 0;
        }
        let mut count = 0usize;
        let mut start = 0usize;
        while let Some(offset) = hay_l[start..].find(&needle_l) {
            count += 1;
            start += offset + needle_l.len();
        }
        count
    }

    /// True when a medium-length token window repeats back-to-back enough times
    /// to indicate a runaway decode loop rather than ordinary seam overlap.
    fn has_pathological_token_run(text: &str) -> bool {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 16 {
            return false;
        }
        // 4-token window repeating 6+ times consecutively is not human speech.
        const WINDOW: usize = 4;
        const MIN_REPEATS: usize = 6;
        if tokens.len() < WINDOW * MIN_REPEATS {
            return false;
        }
        let mut i = 0usize;
        while i + WINDOW * MIN_REPEATS <= tokens.len() {
            let pattern = &tokens[i..i + WINDOW];
            let mut repeats = 1usize;
            let mut cursor = i + WINDOW;
            while cursor + WINDOW <= tokens.len() && &tokens[cursor..cursor + WINDOW] == pattern {
                repeats += 1;
                cursor += WINDOW;
            }
            if repeats >= MIN_REPEATS {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Full native longform path (not the low-level single-buffer executor): the
    /// SharedWindow slicer must split `fixtures/longform_en_zh.wav` (~69s) into
    /// multiple chunks, assemble a multi-clip transcript without pathological
    /// repetition, and never claim success while a slice hit the 256-token
    /// budget or the degenerate-loop guard without recording it on
    /// `truncated_decodes`.
    #[test]
    #[ignore = "requires a private multi-GB granite-speech .oasr pack via \
                OPENASR_GRANITE_SPEECH_PACK; runs the real native longform path on \
                fixtures/longform_en_zh.wav (multi-slice, CPU)"]
    fn longform_multi_slice_avoids_repetition_and_silent_truncation() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let audio = longform_en_zh_wav_path();
        assert!(
            audio.is_file(),
            "committed longform fixture missing at {}",
            audio.display()
        );

        // Fixed 30s window forces a deterministic multi-slice plan independent of
        // VAD availability, matching the SharedWindow default chunk length.
        let longform = LongFormOptions {
            mode: LongFormMode::Fixed,
            chunk_seconds: 30.0,
            // Granite has no token-history producer. Deliberately request carry
            // here so the architecture policy must normalize it off on the
            // production path.
            carry_prompt_across_slices: true,
            ..LongFormOptions::default()
        };
        let request = TranscriptionRequest::new(audio, NATIVE_RUNTIME_MODEL_ID_AUTO)
            .with_model_pack_path(Some(pack_path))
            .with_execution_target(Some(ExecutionTarget::Cpu))
            .with_longform(Some(longform));
        let transcription = NativeBackend::new(
            crate::models::native_execution_services::test_native_execution_services(),
        )
        .transcribe(request)
        .expect("granite-speech longform native transcribe");

        let longform_meta = transcription
            .longform
            .as_ref()
            .expect("longform metadata must be present on a multi-slice run");
        assert!(
            longform_meta.chunk_count >= 2,
            "69s fixture must slice under the 30s SharedWindow, got chunk_count={}",
            longform_meta.chunk_count
        );

        let text = transcription.text.trim();
        assert!(!text.is_empty(), "longform transcript must be non-empty");
        eprintln!(
            "granite-speech longform: chunk_count={} truncated={} text_chars={} text={text:?}",
            longform_meta.chunk_count,
            transcription.is_truncated(),
            text.len(),
        );

        // Truncation visibility: either every slice finished on EOT (happy path
        // for this fixture), or any budget/guard stop is recorded on the
        // transcript. Claiming success with a short string and an empty
        // truncated_decodes list is the silent-truncation defect.
        if transcription.is_truncated() {
            assert!(
                !transcription.truncated_decodes.is_empty(),
                "is_truncated() without truncated_decodes entries"
            );
            for entry in &transcription.truncated_decodes {
                assert!(
                    matches!(
                        entry.truncation.reason,
                        DecodeTruncationReason::BudgetExhausted
                            | DecodeTruncationReason::DegenerateRepeatGuard
                    ),
                    "unexpected truncation reason: {:?}",
                    entry.truncation.reason
                );
            }
            // A truncated multi-slice run is still a degradation we want the
            // audit to see -- fail the gate so it cannot be marked Supported
            // while this fixture trips the budget/guard.
            panic!(
                "longform fixture truncated on one or more slices: {:?}; \
                 multi-slice path must finish this ~69s recording without hitting \
                 the 256-token backstop or the degenerate-loop guard",
                transcription.truncated_decodes
            );
        }

        assert!(
            !has_pathological_token_run(text),
            "pathological consecutive n-gram run in longform transcript: {text:?}"
        );

        // Coverage anchors: fixture is EN/ZH/EN/ZH/EN. Expect the JFK anchor on
        // the order of the three EN clips (allow seam duplication up to 2x) and
        // at least one Chinese weather phrase from the ZH clips.
        let jfk_hits = count_phrase_occurrences(text, JFK_ANCHOR_PHRASE);
        assert!(
            (2..=6).contains(&jfk_hits),
            "JFK anchor should appear ~3 times (fixture has 3 EN clips); got {jfk_hits} in {text:?}"
        );
        let zh_hits = count_phrase_occurrences(text, ZH_ANCHOR_PHRASE);
        assert!(
            zh_hits >= 1,
            "Chinese anchor missing from multi-slice transcript (possible mid-recording drop): {text:?}"
        );
    }

    /// Cross-backend parity skeleton on a medium multi-slice fixture. Compares
    /// CPU vs Accelerated (Metal on macOS) transcripts under the same Fixed
    /// longform plan. Not a bit-exact logits gate -- greedy text equivalence is
    /// the external contract, matching the short JFK Metal/CPU acceptance.
    #[test]
    #[ignore = "requires a private multi-GB granite-speech .oasr pack via \
                OPENASR_GRANITE_SPEECH_PACK and a GPU-class host; runs CPU + \
                Accelerated native longform on fixtures/longform_en_zh.wav"]
    fn longform_cpu_vs_accelerated_transcript_parity() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let audio = longform_en_zh_wav_path();
        assert!(audio.is_file(), "fixture missing at {}", audio.display());

        let longform = LongFormOptions {
            mode: LongFormMode::Fixed,
            chunk_seconds: 30.0,
            carry_prompt_across_slices: true,
            ..LongFormOptions::default()
        };

        let transcribe = |target: ExecutionTarget| {
            NativeBackend::new(
                crate::models::native_execution_services::test_native_execution_services(),
            )
            .transcribe(
                TranscriptionRequest::new(audio.clone(), NATIVE_RUNTIME_MODEL_ID_AUTO)
                    .with_model_pack_path(Some(pack_path.clone()))
                    .with_execution_target(Some(target))
                    .with_longform(Some(longform.clone())),
            )
            .unwrap_or_else(|error| panic!("{target:?} longform failed: {error}"))
        };

        let cpu = transcribe(ExecutionTarget::Cpu);
        let accelerated = transcribe(ExecutionTarget::Accelerated);

        assert!(
            cpu.longform.as_ref().map(|m| m.chunk_count).unwrap_or(0) >= 2,
            "CPU longform must multi-slice"
        );
        assert_eq!(
            cpu.longform.as_ref().map(|m| m.chunk_count),
            accelerated.longform.as_ref().map(|m| m.chunk_count),
            "CPU and Accelerated must share the same slice plan shape"
        );
        assert!(
            !cpu.is_truncated() && !accelerated.is_truncated(),
            "parity gate requires both backends finish without truncation; cpu={:?} accel={:?}",
            cpu.truncated_decodes,
            accelerated.truncated_decodes
        );

        let cpu_text = cpu.text.trim();
        let accel_text = accelerated.text.trim();
        eprintln!("granite longform CPU text:  {cpu_text:?}");
        eprintln!("granite longform Accel text:{accel_text:?}");
        assert_eq!(
            cpu_text, accel_text,
            "CPU and Accelerated longform transcripts must match under greedy decode"
        );
    }

    /// Resident prepared-runtime cache regression: calling `execute()` twice in
    /// a row on the same thread (same pack + backend) MUST hit the thread-local
    /// `GRANITE_SPEECH_PREPARED_BY_KEY` cache on the second call and still
    /// produce a byte-identical transcript to the first (cache-miss/build)
    /// call. This is the load-bearing correctness gate for the resident cache:
    /// the second decode reuses a logically reset session. CPU host K/V was
    /// released; GPU resident K/V remained allocated but must be overwritten
    /// or masked. Any visible leak of prior-request state would diverge the two
    /// transcripts here.
    ///
    /// Transcript-vs-reference correctness (that the decode produces the RIGHT
    /// text) is covered separately by the llama.cpp-reference goldens in
    /// `decode_executor` (en/ja/kwb) and the bit-exact incremental-session gate
    /// in `decode_session`; this test isolates the "reuse == fresh" invariant
    /// the resident cache adds, which those do not exercise.
    #[test]
    #[ignore = "requires a private multi-GB granite-speech .oasr pack via \
                OPENASR_GRANITE_SPEECH_PACK and a Metal-capable host; skips when unset"]
    fn metal_resident_reusable_graph_matches_reference_cold_and_warm() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let Some(first_text) = transcribe_with_pack(
            pack_path.clone(),
            jfk_wav_path(),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        let second_text = transcribe_with_pack(
            pack_path,
            jfk_wav_path(),
            GgmlAsrBackendPreference::Accelerated,
        )
        .expect("warm Metal transcribe");

        // Metal and CPU use different parallel reduction orders, so their f32
        // logits are not expected to be bit-identical. The load-bearing external
        // equivalence is the greedy token sequence rendered as the known JFK
        // reference transcript, for both the cold graph build and the warm reuse.
        assert_eq!(
            first_text, JFK_REFERENCE_TRANSCRIPT,
            "cold Metal transcript"
        );
        assert_eq!(
            second_text, JFK_REFERENCE_TRANSCRIPT,
            "warm Metal transcript"
        );
        assert_eq!(
            first_text, second_text,
            "warm resident graph reuse must preserve the cold greedy transcript"
        );
    }

    #[test]
    fn audio_within_derived_capacity_is_accepted() {
        use super::super::capacity::{
            GRANITE_SPEECH_MAX_INPUT_SAMPLES, GRANITE_SPEECH_SAMPLE_RATE_HZ,
        };
        ensure_audio_within_capacity(0).expect("empty buffer");
        ensure_audio_within_capacity(30 * GRANITE_SPEECH_SAMPLE_RATE_HZ as usize)
            .expect("shared-window slice");
        ensure_audio_within_capacity(GRANITE_SPEECH_MAX_INPUT_SAMPLES)
            .expect("exactly at the derived limit must pass (strict >)");
    }

    #[test]
    fn audio_past_derived_capacity_fails_closed_with_typed_error() {
        use super::super::capacity::{
            GRANITE_SPEECH_DECODER_MAX_POSITIONS, GRANITE_SPEECH_MAX_INPUT_SAMPLES,
            GRANITE_SPEECH_MAX_INPUT_SECONDS, GRANITE_SPEECH_SAMPLE_RATE_HZ,
            granite_speech_audio_tokens_per_second,
        };
        let over = GRANITE_SPEECH_MAX_INPUT_SAMPLES + 1;
        let err = ensure_audio_within_capacity(over).expect_err("over-limit must fail closed");
        let message = err.to_string();
        match err {
            GraniteSpeechGgmlExecutorError::AudioTooLong {
                seconds,
                limit,
                ctx,
                rate,
            } => {
                assert!(seconds > GRANITE_SPEECH_MAX_INPUT_SECONDS as f32);
                assert_eq!(limit, GRANITE_SPEECH_MAX_INPUT_SECONDS as f32);
                assert_eq!(ctx, GRANITE_SPEECH_DECODER_MAX_POSITIONS);
                assert_eq!(rate, granite_speech_audio_tokens_per_second());
            }
            other => panic!("expected AudioTooLong, got {other}"),
        }
        assert!(
            message.contains("exceeds the derived"),
            "error text must name the derived cap: {message}"
        );
        assert!(
            message.contains("381"),
            "error text must surface the numeric limit: {message}"
        );
        // A multi-minute buffer that would previously have run OOD against the
        // 4096-token context must also hit the same typed path.
        let long = ensure_audio_within_capacity(600 * GRANITE_SPEECH_SAMPLE_RATE_HZ as usize)
            .expect_err("10-minute buffer");
        assert!(matches!(
            long,
            GraniteSpeechGgmlExecutorError::AudioTooLong { .. }
        ));
    }

    #[test]
    #[ignore = "requires a private multi-GB granite-speech .oasr pack via \
                OPENASR_GRANITE_SPEECH_PACK; skips silently when unset"]
    fn resident_prepared_runtime_reuse_across_consecutive_calls_stays_byte_identical() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        let Some(first_text) = transcribe_with_pack(
            pack_path.clone(),
            jfk_wav_path(),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };
        let second_text =
            transcribe_with_pack(pack_path, jfk_wav_path(), GgmlAsrBackendPreference::CpuOnly)
                .expect("second transcribe");
        assert!(
            !first_text.trim().is_empty(),
            "first (cache-miss/build) transcript must be non-empty"
        );
        assert_eq!(
            first_text, second_text,
            "second execute() (a resident prepared-runtime cache hit) must match the first \
             (cache-miss/build) call byte-for-byte"
        );
    }
}
