//! firered-aed dedicated executor (Stage 4): fbank+CMVN [`frontend`] -> the
//! parity-verified Conformer [`encoder_graph`] -> greedy attention
//! [`decoder_graph`] -> char+SPM [`tokenizer`] detokenize. No CTC branch, no
//! phrase bias (pure autoregressive attention decode). The executor fails
//! closed with typed errors on a bad pack and never fabricates a transcript.
//!
//! Each call here encodes/decodes exactly one audio window ("single-segment"
//! in that sense -- there is no internal multi-slice batching, unlike
//! cohere's `batched_decode`). Long-file transcription is NOT single-shot,
//! though: the architecture-agnostic longform slicer in
//! `api::backend::native_transcribe` calls this executor once per slice for
//! every builtin family, firered-aed included, with each window pre-capped to
//! this architecture's `GlobalQuadratic` safety ceiling (issue #68) -- well
//! under the encoder's PE-table capacity. `execute_inner` still checks that
//! capacity itself and fails closed with a typed error if a window ever
//! arrives oversized (issue #158's defense-in-depth: a caller that bypasses
//! longform, or a future regression in the slicing wiring, must not reach an
//! opaque graph-allocation failure or a silently degraded transcript).
//!
//! [`frontend`]: super::frontend
//! [`encoder_graph`]: super::encoder_graph
//! [`decoder_graph`]: super::decoder_graph
//! [`tokenizer`]: super::tokenizer

#![allow(dead_code)]

use std::sync::Arc;

use thiserror::Error;

use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_AED_GGML_ADAPTER_ID;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufRuntimeSourcePreflight};
use crate::models::admitted_pinned_runtime_actor_pool::{
    AdmittedPinnedRuntimeActorCheckoutPool, AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    PinnedRuntimeActorCheckout, PinnedRuntimeActorError,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::native_execution_services::{ExecutionLaneKey, current_execution_lane_key};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_decoder_state::Seq2SeqResidentCapacity;
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};

use super::decoder_graph::{
    FireRedDecoderGraphRuntime, run_firered_aed_decoder_greedy_with_runtime,
};
use super::encoder_graph::{
    FireRedEncoderGraphRuntime, FireRedEncoderOutput, predicted_encoder_time_frames,
};
use super::frontend::{FireRedFbankFrontend, apply_cmvn};
use super::runtime_contract::{FireRedAedExecutionMetadata, parse_firered_aed_execution_metadata};
use super::tokenizer::FireRedTokenizer;

const FIRERED_AED_EXECUTOR_ID: &str = crate::arch::FIRERED_AED_EXECUTOR_COMPONENT_ID;
const FIRERED_AED_STREAMING_EXECUTOR_ID: &str = "firered-aed-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const FIRERED_AED_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const FIRERED_AED_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

type FireRedAedEncoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
/// (pack content id, backend, resident self/cross spans). The content id
/// ([`PackContentKey::for_runtime_source`]) keeps an in-place pack
/// replacement at the same path from reusing a runtime built from the old
/// bytes. Logical per-chunk shapes do not belong in this key: the runtime
/// activates them inside the planner-reserved spans without reallocating.
type FireRedAedDecoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, Seq2SeqResidentCapacity);

type FireRedAedEncoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    FireRedAedEncoderRuntimeCacheKey,
    FireRedEncoderGraphRuntime,
>;
type FireRedAedDecoderRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<
    FireRedAedDecoderRuntimeCacheKey,
    FireRedDecoderGraphRuntime,
>;
type FireRedAedEncoderRuntime =
    PinnedRuntimeActorCheckout<FireRedAedEncoderRuntimeCacheKey, FireRedEncoderGraphRuntime>;
type FireRedAedDecoderRuntime =
    PinnedRuntimeActorCheckout<FireRedAedDecoderRuntimeCacheKey, FireRedDecoderGraphRuntime>;

#[derive(Debug, Error)]
enum FireRedAedExecutorError {
    #[error("firered-aed executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("firered-aed runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("firered-aed tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("firered-aed cmvn vectors failed: {reason}")]
    CmvnBuildFailed { reason: String },
    #[error("firered-aed frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("firered-aed encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("firered-aed decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("firered-aed {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error("firered-aed audio window ({window_seconds:.1}s) is too long for this pack: {reason}")]
    AudioWindowTooLong { window_seconds: f32, reason: String },
}

fn window_seconds(n_samples: usize, sample_rate_hz: u32) -> f32 {
    n_samples as f32 / sample_rate_hz.max(1) as f32
}

#[derive(Clone)]
pub(crate) struct FireRedAedGgmlExecutor {
    encoder_runtimes: Arc<FireRedAedEncoderRuntimePool>,
    decoder_runtimes: Arc<FireRedAedDecoderRuntimePool>,
}

impl std::fmt::Debug for FireRedAedGgmlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FireRedAedGgmlExecutor")
            .finish_non_exhaustive()
    }
}

impl Default for FireRedAedGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            FIRERED_AED_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            FIRERED_AED_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-firered-aed-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-firered-aed-decoder-owner",
                limits,
            )),
        }
    }
}

fn allocate_encoder_runtime_owner(
    preflight: GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    backend: GgmlCpuGraphBackend,
    quote: SystemMemoryAllocationQuote,
) -> Result<SystemMemoryOwner<FireRedEncoderGraphRuntime>, FireRedAedExecutorError> {
    match SystemMemoryOwner::try_allocate_transaction(quote, || {
        let runtime = FireRedEncoderGraphRuntime::new_from_preflight(&preflight, metadata, backend)
            .map_err(|error| FireRedAedExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;
        let retained = runtime.retained_system_memory_bytes().map_err(|reason| {
            FireRedAedExecutorError::RuntimeOwnershipFailed {
                stage: "encoder",
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
            Err(FireRedAedExecutorError::RuntimeOwnershipFailed {
                stage: "encoder",
                reason: error.to_string(),
            })
        }
    }
}

fn allocate_decoder_runtime_owner(
    preflight: GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
    backend: GgmlCpuGraphBackend,
    quote: SystemMemoryAllocationQuote,
) -> Result<SystemMemoryOwner<FireRedDecoderGraphRuntime>, FireRedAedExecutorError> {
    match SystemMemoryOwner::try_allocate_transaction(quote, || {
        let runtime = FireRedDecoderGraphRuntime::new(&preflight, metadata, decoder_state, backend)
            .map_err(|error| FireRedAedExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;
        let retained = runtime.retained_system_memory_bytes().map_err(|reason| {
            FireRedAedExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason,
            }
        })?;
        let peak = retained
            .checked_add(
                runtime
                    .construction_transient_system_memory_bytes()
                    .map_err(|reason| FireRedAedExecutorError::RuntimeOwnershipFailed {
                        stage: "decoder",
                        reason,
                    })?,
            )
            .ok_or_else(|| FireRedAedExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason: "post-build SystemMemory peak overflowed".to_string(),
            })?;
        Ok(SystemMemoryAllocationOutcome::new(runtime, peak, retained))
    }) {
        Ok(owner) => Ok(owner),
        Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
        Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
            Err(FireRedAedExecutorError::RuntimeOwnershipFailed {
                stage: "decoder",
                reason: error.to_string(),
            })
        }
    }
}

impl FireRedAedGgmlExecutor {
    fn map_actor_error(
        stage: &'static str,
        error: PinnedRuntimeActorError,
    ) -> FireRedAedExecutorError {
        FireRedAedExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn map_runtime_ownership_error(
        stage: &'static str,
        reason: impl Into<String>,
    ) -> FireRedAedExecutorError {
        FireRedAedExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: reason.into(),
        }
    }

    fn checkout_encoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        backend: GgmlCpuGraphBackend,
    ) -> Result<FireRedAedEncoderRuntime, FireRedAedExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
        );
        let preflight = preflight.clone();
        let pack_content_id = preflight.runtime_source.content_id().to_string();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let quote =
                    FireRedEncoderGraphRuntime::system_memory_quote(metadata, &pack_content_id)
                        .map_err(|reason| Self::map_runtime_ownership_error("encoder", reason))?;
                Ok((quote.retained_bytes, (preflight, quote)))
            },
            move |(preflight, quote)| {
                allocate_encoder_runtime_owner(preflight, metadata, backend, quote)
            },
            |error| Self::map_actor_error("encoder", error),
        )
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        backend: GgmlCpuGraphBackend,
    ) -> Result<FireRedAedDecoderRuntime, FireRedAedExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(backend),
            decoder_state.resident_capacity(),
        );
        let preflight = preflight.clone();
        let pack_content_id = preflight.runtime_source.content_id().to_string();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let quote = FireRedDecoderGraphRuntime::system_memory_quote(
                    metadata,
                    decoder_state,
                    backend,
                    &pack_content_id,
                )
                .map_err(|reason| Self::map_runtime_ownership_error("decoder", reason))?;
                Ok((quote.retained_bytes, (preflight, quote)))
            },
            move |(preflight, quote)| {
                allocate_decoder_runtime_owner(preflight, metadata, decoder_state, backend, quote)
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn encode_with_owned_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        cmvn_features: Vec<f32>,
        n_frames: usize,
        backend: GgmlCpuGraphBackend,
    ) -> Result<FireRedEncoderOutput, FireRedAedExecutorError> {
        let runtime = self.checkout_encoder_runtime(preflight, metadata, backend)?;
        runtime
            .call_mut(move |runtime| {
                let encode_result = runtime.encode(&cmvn_features, n_frames);
                let release_result = runtime.release_transient_compute_memory();
                match (encode_result, release_result) {
                    (Ok(output), Ok(())) => Ok(output),
                    (Err(error), _) => Err(error),
                    (Ok(_), Err(error)) => Err(error),
                }
            })
            .map_err(|error| Self::map_actor_error("encoder", error))?
            .map_err(|error| FireRedAedExecutorError::EncoderFailed {
                reason: error.to_string(),
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_owned_runtime(
        &self,
        preflight: &GgufRuntimeSourcePreflight,
        metadata: FireRedAedExecutionMetadata,
        encoder_rows: Vec<f32>,
        encoder_frame_count: usize,
        decoder_state: crate::models::seq2seq_decoder_state::Seq2SeqDecoderState,
        tokenizer: FireRedTokenizer,
        control: Arc<crate::api::backend::TranscriptionControl>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<super::decoder_graph::FireRedAedGreedyDecodeOutput, FireRedAedExecutorError> {
        let runtime = self.checkout_decoder_runtime(preflight, metadata, decoder_state, backend)?;
        runtime
            .call_mut(move |runtime| {
                runtime.activate_decoder_state(decoder_state)?;
                run_firered_aed_decoder_greedy_with_runtime(
                    runtime,
                    metadata,
                    &encoder_rows,
                    encoder_frame_count,
                    |ids| tokenizer.decode(ids).map_err(|error| error.to_string()),
                    &control,
                )
            })
            .map_err(|error| Self::map_actor_error("decoder", error))?
            .map_err(|error| FireRedAedExecutorError::DecoderFailed {
                reason: error.to_string(),
            })
    }

    fn clear_runtime_actors(&self) {
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
    }

    /// Evicts only actors prepared from the replaced content generation.
    /// Other packs in this service root stay warm.
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_runtimes
            .evict_where(|(pack, _)| pack.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|(pack, _, _)| pack.pack_content_id == pack_content_id);
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedAedExecutorError> {
        if request.selected_family.adapter_id != FIRERED_AED_GGML_ADAPTER_ID {
            return Err(FireRedAedExecutorError::AdapterMismatch {
                expected: FIRERED_AED_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let decoder_state =
            crate::models::seq2seq_decoder_state::Seq2SeqDecoderState::from_request_state(
                &request.decoder_state,
                super::capacity::FIRERED_AED_DECODER_STATE_IDS,
            )
            .map_err(|error| FireRedAedExecutorError::DecoderFailed {
                reason: error.to_string(),
            })?;
        let preflight = request.runtime_source_preflight();
        let metadata =
            parse_firered_aed_execution_metadata(&preflight.metadata).map_err(|error| {
                FireRedAedExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokens = preflight
            .metadata
            .get_string_array(TOKENIZER_TOKENS_KEY)
            .ok_or_else(|| FireRedAedExecutorError::TokenizerBuildFailed {
                reason: "pack missing tokenizer.ggml.tokens".to_string(),
            })?
            .to_vec();
        let tokenizer = FireRedTokenizer::new(tokens);

        let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
            FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            }
        })?;
        let feature_dim_shape = [metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedAedExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;

        let samples = &request.prepared_audio.samples_f32;
        let frontend = FireRedFbankFrontend::new();
        let mut features =
            frontend
                .compute(samples)
                .map_err(|error| FireRedAedExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).map_err(
            |error| FireRedAedExecutorError::FrontendFailed {
                reason: error.to_string(),
            },
        )?;

        // Defense in depth (issue #158): the generic longform slicer in
        // `native_transcribe` already caps every window at this architecture's
        // declared `GlobalQuadratic` safety ceiling (issue #68), which is well
        // inside the encoder's baked rel-pos-table capacity below -- so this
        // should never trip in the normal request path. But this executor is
        // also reachable directly (a caller that skips longform, a future
        // regression in the slicing wiring, an oversized fixed/manual chunk
        // request), and a window past the PE table's capacity is a quality/
        // correctness problem even when it happens to fit in memory: reject it
        // with a typed, actionable error up front rather than let a caller
        // silently degrade or hit an opaque graph-allocation failure deep in
        // `encoder_graph`.
        let predicted_encoder_frames =
            predicted_encoder_time_frames(features.n_frames).map_err(|error| {
                FireRedAedExecutorError::AudioWindowTooLong {
                    window_seconds: window_seconds(
                        samples.len(),
                        request.prepared_audio.sample_rate_hz,
                    ),
                    reason: error.to_string(),
                }
            })?;
        let max_encoder_frames = metadata.encoder_max_frames();
        if predicted_encoder_frames > max_encoder_frames {
            return Err(FireRedAedExecutorError::AudioWindowTooLong {
                window_seconds: window_seconds(
                    samples.len(),
                    request.prepared_audio.sample_rate_hz,
                ),
                reason: format!(
                    "encoder frame count {predicted_encoder_frames} exceeds this pack's \
                     positional-encoding capacity of {max_encoder_frames} frames \
                     (~{:.0}s at 25 fps); this window should already be capped by \
                     longform slicing before reaching the executor",
                    max_encoder_frames as f32 / 25.0
                ),
            });
        }

        let backend = request.resolved_runtime.backend();
        let feature_frames = features.n_frames;
        let encoder_output = self.encode_with_owned_runtime(
            preflight,
            metadata,
            features.data,
            feature_frames,
            backend,
        )?;

        let decode = self.decode_with_owned_runtime(
            preflight,
            metadata,
            encoder_output.rows,
            encoder_output.frame_count,
            decoder_state,
            tokenizer,
            Arc::clone(&request.execution_context.control),
            backend,
        )?;

        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        let text = decode.text.trim().to_string();
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
            decode_truncation: decode.stop_reason.into_decode_truncation(None),
        })
    }
}

impl GgmlAsrViewExecutor for FireRedAedGgmlExecutor {
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        FireRedAedGgmlExecutor::evict_prepared_runtime_content_id(self, pack_content_id);
    }

    fn executor_id(&self) -> &'static str {
        FIRERED_AED_EXECUTOR_ID
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
                super::capacity::plan_firered_aed_decoder_state,
                super::capacity::FIRERED_AED_DECODER_STATE_STREAMS,
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

impl GgmlAsrStreamingExecutor for FireRedAedGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_AED_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FIRERED_AED_STREAMING_EXECUTOR_ID,
            FIRERED_AED_GGML_ADAPTER_ID,
            "firered-aed",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FireRedAedGgmlExecutor::execute_view,
        )
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_actors();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use crate::arch::builtin_adapter_descriptor;
    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudioView};

    use super::*;

    // Pinned to the reference PyTorch decode captured by the dev-only
    // `tmp/firered-ref-src` harness (see the Stage 1-2 module docs); the
    // fp16 pack itself is a private, non-committed dev artifact.
    const GOLDEN_JFK_TEXT: &str = "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO \
         FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY";

    // Pinned to the reference PyTorch decode of `fixtures/zh_sample.wav` (a
    // macOS `say -v Tingting` synthesis of an original, non-copyrighted
    // Mandarin sentence written for this test) via the same
    // `tmp/firered-ref-src` harness. The reference tokenizer's `dict.txt` has
    // no punctuation/`<space>` entries, so the golden text is intentionally
    // punctuation-free.
    const GOLDEN_ZH_TEXT: &str = "今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的\
         川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下";

    struct TestPinnedRuntime {
        id: usize,
        _thread_affine: Rc<Cell<usize>>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for TestPinnedRuntime {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    type TestRuntimePool = AdmittedPinnedRuntimeActorCheckoutPool<String, TestPinnedRuntime>;
    type TestRuntimeCheckout = PinnedRuntimeActorCheckout<String, TestPinnedRuntime>;

    fn test_runtime_pool() -> TestRuntimePool {
        AdmittedPinnedRuntimeActorCheckoutPool::new(
            "firered-aed-runtime-owner-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                FIRERED_AED_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
                1_024,
                FIRERED_AED_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
            ),
        )
    }

    fn checkout_test_runtime(
        pool: &TestRuntimePool,
        key: &str,
        builds: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    ) -> Result<TestRuntimeCheckout, String> {
        pool.checkout_or_try_build_with(
            key.to_string(),
            || Ok((32, ())),
            move |()| {
                let id = builds.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(SystemMemoryOwner::with_committed_requested_bytes_for_test(
                    TestPinnedRuntime {
                        id,
                        _thread_affine: Rc::new(Cell::new(id)),
                        drops,
                    },
                    32,
                ))
            },
            |error| error.to_string(),
        )
    }

    #[test]
    fn executor_owned_actor_pool_reuses_a_returned_runtime() {
        let pool = test_runtime_pool();
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));

        let first = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .expect("first checkout");
        assert_eq!(first.call_mut(|runtime| runtime.id).unwrap(), 1);
        drop(first);
        assert_eq!(pool.usage_for_test(), (1, 32));

        let second = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .expect("warm checkout");
        assert_eq!(second.call_mut(|runtime| runtime.id).unwrap(), 1);
        drop(second);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        pool.clear();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn executor_owned_actor_pool_runs_two_same_key_sessions_concurrently() {
        const {
            assert!(
                FIRERED_AED_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY >= 2,
                "FireRed-AED server sessions must not collapse onto one actor"
            );
        }
        let pool = test_runtime_pool();
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let first = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .unwrap();
        let second = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 2);

        let barrier = Arc::new(Barrier::new(3));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let run = |checkout: TestRuntimeCheckout| {
            let barrier = Arc::clone(&barrier);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            thread::spawn(move || {
                checkout
                    .call_mut(move |_| {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(now, Ordering::SeqCst);
                        barrier.wait();
                        active.fetch_sub(1, Ordering::SeqCst);
                    })
                    .unwrap();
                drop(checkout);
            })
        };
        let first = run(first);
        let second = run(second);
        barrier.wait();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(maximum.load(Ordering::SeqCst), 2);
        pool.clear();
    }

    #[test]
    fn clear_does_not_resurrect_an_in_flight_firered_actor() {
        let pool = test_runtime_pool();
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let in_flight = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .unwrap();

        pool.clear();
        drop(in_flight);
        assert_eq!(pool.usage_for_test(), (0, 0));
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let rebuilt = checkout_test_runtime(
            &pool,
            "sha256:same",
            Arc::clone(&builds),
            Arc::clone(&drops),
        )
        .unwrap();
        assert_eq!(rebuilt.call_mut(|runtime| runtime.id).unwrap(), 2);
        drop(rebuilt);
        pool.clear();
    }

    #[test]
    fn targeted_content_eviction_preserves_other_pack_and_rebuilds_only_old_pack() {
        let pool = test_runtime_pool();
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let old =
            checkout_test_runtime(&pool, "sha256:old", Arc::clone(&builds), Arc::clone(&drops))
                .unwrap();
        assert_eq!(old.call_mut(|runtime| runtime.id).unwrap(), 1);
        drop(old);

        let replacement =
            checkout_test_runtime(&pool, "sha256:new", Arc::clone(&builds), Arc::clone(&drops))
                .unwrap();
        assert_eq!(replacement.call_mut(|runtime| runtime.id).unwrap(), 2);
        drop(replacement);
        assert_eq!(builds.load(Ordering::SeqCst), 2);

        pool.evict_where(|key| key == "sha256:old");
        assert_eq!(pool.usage_for_test(), (1, 32));
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let still_warm =
            checkout_test_runtime(&pool, "sha256:new", Arc::clone(&builds), Arc::clone(&drops))
                .unwrap();
        assert_eq!(still_warm.call_mut(|runtime| runtime.id).unwrap(), 2);
        drop(still_warm);

        let rebuilt_old =
            checkout_test_runtime(&pool, "sha256:old", Arc::clone(&builds), Arc::clone(&drops))
                .unwrap();
        assert_eq!(rebuilt_old.call_mut(|runtime| runtime.id).unwrap(), 3);
        drop(rebuilt_old);
        assert_eq!(builds.load(Ordering::SeqCst), 3);
        pool.clear();
    }

    fn dev_pack_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/firered-out/firered-aed-l-fp16.oasr")
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn test_runtime_preflight(path: &Path) -> GgufRuntimeSourcePreflight {
        crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(path)
            .expect("firered-aed test runtime must pass preflight")
    }

    fn decoder_runtime_state(
        metadata: FireRedAedExecutionMetadata,
        cross_positions: usize,
    ) -> crate::models::seq2seq_decoder_state::Seq2SeqDecoderState {
        use crate::models::seq2seq_decoder_state::{Seq2SeqDecoderState, Seq2SeqStateAxis};

        let self_positions = super::super::decode_budget::firered_aed_decode_budget(
            cross_positions,
            metadata.decoder_pe_len,
        )
        .expect("test decoder budget")
        .self_kv_positions;

        Seq2SeqDecoderState {
            self_attention: Seq2SeqStateAxis {
                logical_positions: self_positions,
                resident_positions: self_positions,
                hard_position_cap: metadata.decoder_pe_len,
            },
            cross_attention: Seq2SeqStateAxis {
                logical_positions: cross_positions,
                resident_positions: cross_positions,
                hard_position_cap: metadata.encoder_max_frames(),
            },
        }
    }

    fn plan_request_decoder_state(
        request: &mut GgmlAsrExecutionViewRequest<'_>,
        envelope_samples: Option<usize>,
    ) -> Result<(), String> {
        use std::num::NonZeroU32;

        let preflight = request.runtime_source_preflight();
        let sample_rate = NonZeroU32::new(request.prepared_audio.sample_rate_hz)
            .ok_or_else(|| "test sample rate is zero".to_string())?;
        let invocation = crate::capacity::topology::InvocationShapeInput::new(
            sample_rate,
            request.prepared_audio.samples_f32.len(),
        )
        .map_err(|error| error.to_string())?;
        let envelope = crate::capacity::topology::InvocationEnvelope::new(
            sample_rate,
            envelope_samples
                .unwrap_or(invocation.samples())
                .max(invocation.samples()),
        )
        .map_err(|error| error.to_string())?;
        let planning_input = crate::models::ggml_asr_executor::GgmlAsrDecoderStatePlanningInput {
            preflight,
            invocation,
            envelope,
            request_options: &request.request_options,
            backend: request.resolved_runtime.backend(),
        };
        let plan = super::super::capacity::plan_firered_aed_decoder_state(&planning_input)
            .map_err(|error| error.to_string())?;
        request.decoder_state =
            crate::models::ggml_asr_executor::GgmlAsrDecoderState::planned_for_test(plan, envelope);
        Ok(())
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<String> {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "firered-aed golden test",
            "firered-aed golden test",
        )
        .expect("load wav fixture");

        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                test_runtime_preflight(&pack_path),
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_request_decoder_state(&mut request, None).expect("plan firered-aed decoder state");

        let executor = FireRedAedGgmlExecutor::default();
        let result = executor
            .execute_view(&request)
            .expect("firered-aed transcribe");
        Some(result.transcription.text)
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_jfk_wav() {
        let Some(text) = transcribe_with_dev_pack(jfk_wav_path()) else {
            return;
        };
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_zh_sample_wav() {
        let Some(text) = transcribe_with_dev_pack(zh_wav_path()) else {
            return;
        };
        assert_eq!(text, GOLDEN_ZH_TEXT);
    }

    /// Proof that the Metal reusable single-token decode graph is
    /// output-preserving: drive a full greedy decode twice against the same
    /// encoder output -- one runtime on the default path (which must take the
    /// persistent reuse graph for every incremental step) and one runtime
    /// forced onto the rebuild-per-step path -- and require the logits of
    /// every step to match BIT FOR BIT, not just the argmax token. Skips
    /// (with a message) when the dev pack is absent or Metal is unavailable;
    /// on Metal it also asserts the reuse graph actually engaged, so this
    /// cannot silently pass by both sides falling back to the same path.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack and a Metal device; see module docs"]
    fn metal_reused_decode_graph_logits_match_rebuilt_graph_bit_for_bit() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            jfk_wav_path(),
            "firered-aed reuse-parity test",
            "firered-aed reuse-parity test",
        )
        .expect("load jfk.wav");
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                test_runtime_preflight(&pack_path),
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples.clone()),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_request_decoder_state(&mut request, None).expect("plan firered-aed decoder state");
        let preflight = request.runtime_source_preflight();
        let metadata = parse_firered_aed_execution_metadata(&preflight.metadata)
            .expect("parse execution metadata");

        // Frontend + CMVN + CPU encoder: one shared encoder output feeds both
        // decoder runtimes, so any divergence below is decode-path-only.
        let reader = build_runtime_tensor_reader_from_preflight(preflight)
            .expect("build runtime tensor reader");
        let feature_dim_shape = [metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .expect("cmvn neg_mean");
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .expect("cmvn inv_stddev");
        let frontend = FireRedFbankFrontend::new();
        let mut features = frontend.compute(&samples).expect("fbank features");
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev)
            .expect("apply cmvn");
        let mut encoder_runtime =
            FireRedEncoderGraphRuntime::new(preflight, metadata, GgmlCpuGraphBackend::Cpu)
                .expect("build cpu encoder runtime");
        let encoder_output = encoder_runtime
            .encode(&features.data, features.n_frames)
            .expect("encode");

        // Two sequential passes over the SAME prefix sequence (the fresh
        // pass's own greedy argmax drives both), so only one 2+ GiB Metal
        // weight upload is resident at a time. Divergence cannot hide in the
        // replay: every reused-pass step is asserted bitwise against the
        // recorded fresh-pass logits for the identical prefix.
        let max_steps = 256usize;
        let mut fresh_runtime = match FireRedDecoderGraphRuntime::new(
            preflight,
            metadata,
            decoder_runtime_state(metadata, encoder_output.frame_count),
            GgmlCpuGraphBackend::Metal,
        ) {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("skipping: Metal decoder runtime unavailable: {error}");
                return;
            }
        };
        fresh_runtime
            .populate_cross_attention_cache(&encoder_output.rows, encoder_output.frame_count)
            .expect("populate cross cache (fresh runtime)");
        let mut prefix = vec![metadata.sos_token_id];
        let mut fresh_logits_by_step = Vec::new();
        for _ in 0..max_steps {
            let fresh_logits = fresh_runtime
                .compute_step_logits_forcing_fresh_graph(&prefix)
                .expect("fresh-path step logits");
            let next = fresh_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index as u32)
                .expect("non-empty logits");
            fresh_logits_by_step.push(fresh_logits);
            if next == metadata.eos_token_id {
                break;
            }
            prefix.push(next);
        }
        assert!(
            !fresh_runtime.has_active_reuse_graph(),
            "the forced-fresh runtime must never have built a reuse graph"
        );
        drop(fresh_runtime);
        let generated = prefix.len() - 1;
        assert!(
            generated >= 8,
            "expected a non-trivial greedy decode to exercise many incremental steps, got {generated} tokens"
        );

        let mut reused_runtime = FireRedDecoderGraphRuntime::new(
            preflight,
            metadata,
            decoder_runtime_state(metadata, encoder_output.frame_count),
            GgmlCpuGraphBackend::Metal,
        )
        .expect("build Metal decoder runtime (reused pass)");
        reused_runtime
            .populate_cross_attention_cache(&encoder_output.rows, encoder_output.frame_count)
            .expect("populate cross cache (reused runtime)");
        for (step, fresh_logits) in fresh_logits_by_step.iter().enumerate() {
            let step_prefix = &prefix[..(step + 1).min(prefix.len())];
            let reused_logits = reused_runtime
                .compute_step_logits(step_prefix)
                .expect("reused-path step logits");
            assert_eq!(
                reused_logits.len(),
                fresh_logits.len(),
                "step {step}: logits length mismatch"
            );
            for (index, (reused, fresh)) in
                reused_logits.iter().zip(fresh_logits.iter()).enumerate()
            {
                assert_eq!(
                    reused.to_bits(),
                    fresh.to_bits(),
                    "step {step}: logit {index} differs bitwise: reused={reused} fresh={fresh}"
                );
            }
        }
        assert!(
            reused_runtime.has_active_reuse_graph(),
            "the default-path Metal runtime must have engaged the persistent reuse graph"
        );
        // Transcript-level goldenness of the whole Metal decode path
        // (scheduler-off + reuse graph): the greedy token sequence both paths
        // agreed on must detokenize to the same reference transcript the CPU
        // golden test pins.
        let tokenizer = FireRedTokenizer::new(
            preflight
                .metadata
                .get_string_array(TOKENIZER_TOKENS_KEY)
                .expect("pack missing tokenizer.ggml.tokens")
                .to_vec(),
        );
        let text = tokenizer
            .decode(&prefix[1..])
            .expect("detokenize greedy tokens")
            .trim()
            .to_string();
        assert_eq!(
            text, GOLDEN_JFK_TEXT,
            "Metal reused-graph transcript must match the reference golden"
        );
        eprintln!("firered-aed reuse parity: {generated} incremental steps bit-identical on Metal");
    }

    /// Demonstrates the executor-owned encoder/decoder actor pools: the
    /// second sequential transcription of the same pack must be
    /// meaningfully faster than the first, because it skips re-loading the
    /// GGUF weight context (mmap + tensor-metadata construction) for both the
    /// encoder and the decoder. Not a strict regression gate (wall-clock,
    /// shared CI hardware) -- just an executable record of the speedup this
    /// module claims; skips silently without the dev-only pack.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn second_sequential_transcribe_is_faster_than_first_due_to_runtime_reuse() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            jfk_wav_path(),
            "firered-aed perf test",
            "firered-aed perf test",
        )
        .expect("load jfk.wav");

        let build_request = || {
            let mut request = GgmlAsrExecutionViewRequest {
                execution_services:
                    crate::models::native_execution_services::test_native_execution_services(),
                decoder_state:
                    crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
                verified_pack:
                    crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                        test_runtime_preflight(&pack_path),
                        crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
                    ),
                selected_family: builtin_adapter_descriptor(
                    crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
                ),
                prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples.clone()),
                request_options: Default::default(),
                backend_preference: GgmlAsrBackendPreference::CpuOnly,
                resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                    (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                    crate::ggml_runtime::AutoGpuPolicy::AllBackends,
                ),
                execution_context: std::sync::Arc::new(
                    crate::RequestExecutionContext::uncancellable("test fixture"),
                ),
            };
            plan_request_decoder_state(&mut request, None).expect("plan firered-aed decoder state");
            request
        };
        let executor = FireRedAedGgmlExecutor::default();

        let first_start = std::time::Instant::now();
        let first = executor
            .execute_view(&build_request())
            .expect("firered-aed transcribe (first, cold runtime cache)");
        let first_elapsed = first_start.elapsed();

        let second_start = std::time::Instant::now();
        let second = executor
            .execute_view(&build_request())
            .expect("firered-aed transcribe (second, warm runtime cache)");
        let second_elapsed = second_start.elapsed();

        assert_eq!(first.transcription.text, GOLDEN_JFK_TEXT);
        assert_eq!(second.transcription.text, GOLDEN_JFK_TEXT);
        eprintln!("firered-aed runtime cache: first={first_elapsed:?} second={second_elapsed:?}");
        assert!(
            second_elapsed < first_elapsed,
            "expected cached (second) transcribe to be faster: first={first_elapsed:?} second={second_elapsed:?}"
        );
    }

    /// Structural regression test for the VAD/longform 0%-cache-hit bug this
    /// module fixes: chunks with DIFFERENT encoder frame counts on the same
    /// executor scope must still land in the SAME decoder pool slot when both carry
    /// the same 60-second session envelope. `jfk.wav` and `zh_sample.wav` are
    /// different durations (and firered's char+SPM tokenizer makes their
    /// encoder frame counts essentially certain to differ), so this is a real
    /// cross-frame-count reuse, not a same-length coincidence.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn differently_sized_chunks_reuse_the_same_decoder_runtime_cache_slot() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let build_request = |wav_path: PathBuf| {
            let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                wav_path,
                "firered-aed cache-reuse test",
                "firered-aed cache-reuse test",
            )
            .expect("load wav fixture");
            let mut request = GgmlAsrExecutionViewRequest {
                execution_services:
                    crate::models::native_execution_services::test_native_execution_services(),
                decoder_state:
                    crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
                verified_pack:
                    crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                        test_runtime_preflight(&pack_path),
                        crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
                    ),
                selected_family: builtin_adapter_descriptor(
                    crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
                ),
                prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
                request_options: Default::default(),
                backend_preference: GgmlAsrBackendPreference::CpuOnly,
                resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                    (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                    crate::ggml_runtime::AutoGpuPolicy::AllBackends,
                ),
                execution_context: std::sync::Arc::new(
                    crate::RequestExecutionContext::uncancellable("test fixture"),
                ),
            };
            plan_request_decoder_state(&mut request, Some(60 * 16_000))
                .expect("plan firered-aed shared decoder envelope");
            request
        };
        let executor = FireRedAedGgmlExecutor::default();

        let decoder_cache_len = || executor.decoder_runtimes.usage_for_test().0;
        let encoder_cache_len = || executor.encoder_runtimes.usage_for_test().0;
        assert_eq!(decoder_cache_len(), 0, "cache must start empty");

        let jfk = executor
            .execute_view(&build_request(jfk_wav_path()))
            .expect("firered-aed transcribe jfk.wav");
        assert_eq!(jfk.transcription.text, GOLDEN_JFK_TEXT);
        eprintln!(
            "firered-aed cache slots after chunk 1: decoder={} encoder={}",
            decoder_cache_len(),
            encoder_cache_len()
        );
        assert_eq!(decoder_cache_len(), 1, "first chunk must build one slot");

        let zh = executor
            .execute_view(&build_request(zh_wav_path()))
            .expect("firered-aed transcribe zh_sample.wav");
        assert_eq!(zh.transcription.text, GOLDEN_ZH_TEXT);
        eprintln!(
            "firered-aed cache slots after chunk 2 (different frame count): decoder={} encoder={}",
            decoder_cache_len(),
            encoder_cache_len()
        );
        assert_eq!(
            decoder_cache_len(),
            1,
            "a differently-sized second chunk must reuse the SAME decoder cache slot, not mint a second one"
        );
        assert_eq!(
            encoder_cache_len(),
            1,
            "encoder cache was already frame-count-agnostic"
        );
    }

    /// A single window past the pack's PE-table capacity must be rejected by
    /// decoder-state planning, before executor graph construction or memory
    /// admission can begin.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-fp16.oasr pack; see module docs"]
    fn oversized_window_fails_closed_with_typed_error_instead_of_reaching_the_encoder_graph() {
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        // 210s of silence at 16 kHz: cheap to construct, and content does not
        // matter -- the guard trips on shape (predicted encoder frame count)
        // before any real encoding is attempted.
        let samples = vec![0.0_f32; 210 * 16_000];
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                test_runtime_preflight(&pack_path),
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let message = plan_request_decoder_state(&mut request, None)
            .expect_err("a 210s window must fail decoder-state planning");
        assert!(
            message.contains("hard cap") || message.contains("position"),
            "expected a positional-cap planning error, got: {message}"
        );
    }

    /// A 45-second single invocation remains legal when its declared session
    /// envelope is 60 seconds; it must fit the preplanned resident span with
    /// no allocation growth and preserve the recorded transcript.
    #[test]
    #[ignore = "requires the private dev-only firered-aed-l-v2-q4_k.oasr pack; see module docs"]
    fn mode_off_single_window_fits_declared_envelope_and_matches_baseline() {
        let pack_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/firered-out/firered-aed-l-v2-q4_k.oasr");
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/mode-off-regression/longform_en_zh_45s.wav");
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return;
        }
        // Recorded on `origin/main` (pre-capacity-refactor, exact-per-call
        // cross-KV allocation) with the same pack/clip via a temporary
        // baseline test, not committed.
        const BASELINE_TEXT: &str = "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭晚上我们还计划去一家新开的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园";
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "firered-aed mode=off envelope test",
            "firered-aed mode=off envelope test",
        )
        .expect("load wav fixture");
        let mut request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                test_runtime_preflight(&pack_path),
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            selected_family: builtin_adapter_descriptor(
                crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID,
            ),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        plan_request_decoder_state(&mut request, Some(60 * 16_000))
            .expect("plan firered-aed 60-second decoder envelope");
        let executor = FireRedAedGgmlExecutor::default();
        let result = executor
            .execute_view(&request)
            .expect("mode=off single-window transcribe must succeed, not fail closed");
        assert_eq!(result.transcription.text, BASELINE_TEXT);
    }
}
