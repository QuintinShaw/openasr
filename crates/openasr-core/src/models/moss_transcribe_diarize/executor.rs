//! moss-transcribe-diarize dedicated executor: chunked Whisper-Medium
//! encoder (30s windows, each trimmed to its own valid frame count before
//! concatenation -- mirrors upstream `get_audio_features`'s
//! `whisper_features[chunk_idx:chunk_idx+1, :token_len*4]`) -> [`adaptor_graph`]
//! (4x merge + VQAdaptor over the FULL concatenated sequence, numerically
//! identical to merging per-chunk-then-concatenating since each kept
//! chunk length is already a multiple of the merge size) -> ChatML+audio-span
//! prompt ([`decode_prompt`] + [`prompt_embedding`]'s sparse splice, since
//! digit time-anchor tokens interrupt the `<|audio_pad|>` run) -> Qwen3-0.6B
//! [`llm_decoder`] prefill/decode, driven through the ONE shared greedy
//! decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a [`Seq2SeqGreedyDecodeStepExecutor`]
//! impl below -- never a hand-rolled argmax loop (this repo's
//! `model-integration-shared-driver` invariant, see `AGENTS.md`).
//!
//! File-transcribe only: no streaming/realtime session (this family's
//! architecture always needs the full audio to compute time-anchor markers
//! ahead of the prompt, so there is no meaningful "partial" mode yet).

#![allow(dead_code)]

use std::sync::Arc;

use thiserror::Error;

use crate::api::backend::Transcription;
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
use crate::models::prepared_runtime_cache::{
    PreparedRuntimeCache, PreparedRuntimeHandle, PreparedRuntimeQuoteContext,
};
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrKvCacheCapacity, Qwen3AsrKvCacheCapacityError,
    Qwen3AsrPromptEmbeddings,
};
use crate::models::runtime_cache_coordinator::PackContentKey;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
    Seq2SeqGreedyDecodeStopReason,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner,
};
use crate::models::whisper::whisper_log_mel_spectrogram_16khz_mono_v0;

use super::adaptor_graph::run_moss_adaptor;
use super::decode_budget::moss_td_generated_token_budget as moss_td_budget_for_shape;
#[cfg(test)]
use super::decode_budget::{MOSS_TD_MAX_GENERATED_TOKENS, MOSS_TD_MIN_GENERATED_TOKENS};
use super::decode_prompt::build_moss_td_decode_prompt;
use super::encoder_graph::{MossEncoderConfig, MossEncoderRuntime};
use super::graph_config::{moss_td_encoder_graph_config, moss_td_runtime_graph_config};
use super::llm_decoder::MossTdDecoderRuntime;
use super::prepared_runtime::{
    MossTdPreparedRuntime, MossTdPreparedRuntimeError, build_moss_td_prepared_runtime,
};
use super::prompt_embedding::build_moss_td_prompt_embeddings_with_audio_splice;
use super::runtime_contract::{
    MossTdDecoderMetadata, moss_td_kv_cache_positions, moss_td_request_kv_cache_positions,
};
use super::speaker_segments::MossTdDecodeExtent;
use super::tokenizer::MossTdTokenizer;

/// `WhisperFeatureExtractor`'s `chunk_length=30` @ 16kHz (`preprocessor_config.json`,
/// verified against the real checkpoint). `pub(crate)` because the capacity
/// derivation shares the chunk quantum (see `super::capacity`).
pub(crate) const CHUNK_SAMPLES: usize = 480_000;
const MEL_TARGET_FRAMES: usize = 3000;
/// `pub(crate)` for the same reason: the capacity frontend registry states
/// the same architectural facts and is pinned equal to these constants.
pub(crate) const SAMPLE_RATE_HZ: usize = 16_000;
/// `WhisperFeatureExtractor.hop_length` (160) * the Whisper conv stem's 2x
/// stride * `audio_merge_size` -- upstream's
/// `_compute_audio_token_length`'s `stride` (`processing_moss_transcribe_diarize.py`).
pub(crate) const WHISPER_ENCODER_CONV_STRIDE: usize = 2;
pub(crate) const HOP_LENGTH: usize = 160;
/// Audio tokens per second the adaptor emits (`audio_tokens_per_second` in
/// `processing_moss_transcribe_diarize.py`, same value `decode_prompt`'s marker
/// cadence uses). Only used to render the `AudioExceedsContext` limit as an
/// approximate minutes figure; not part of any decode math. `pub(crate)` so
/// `super::capacity`'s drift guard can pin it equal to the capacity frontend
/// registry's derived rate (three copies of one fact, one pinned number).
pub(crate) const AUDIO_TOKENS_PER_SECOND_FOR_LIMIT: f32 =
    super::decode_prompt::AUDIO_TOKENS_PER_SECOND;

#[derive(Debug, Error)]
enum MossTdExecutorError {
    #[error("moss-transcribe-diarize executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("moss-transcribe-diarize executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("moss-transcribe-diarize runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("moss-transcribe-diarize tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("moss-transcribe-diarize requires non-empty audio")]
    EmptyAudio,
    #[error(
        "moss-transcribe-diarize one invocation contains {samples} samples, exceeding the product envelope {max_samples} samples"
    )]
    InvocationTooLong { samples: usize, max_samples: usize },
    #[error("moss-transcribe-diarize decode budget is unavailable: {reason}")]
    DecodeBudgetUnavailable { reason: String },
    #[error(
        "moss-transcribe-diarize audio is too long: its {prompt_tokens}-token audio prompt plus \
         the {generation_budget}-token decode budget needs {required_positions} positions within \
         the {kv_capacity}-position decoder context (about {max_minutes:.0} min of audio); split \
         the input into shorter files"
    )]
    AudioExceedsContext {
        prompt_tokens: usize,
        generation_budget: usize,
        required_positions: usize,
        kv_capacity: usize,
        max_minutes: f32,
    },
    #[error("moss-transcribe-diarize mel frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("moss-transcribe-diarize encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("moss-transcribe-diarize adaptor failed: {reason}")]
    AdaptorFailed { reason: String },
    #[error("moss-transcribe-diarize decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("moss-transcribe-diarize decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("moss-transcribe-diarize {stage} runtime ownership failed: {reason}")]
    RuntimeOwnershipFailed { stage: &'static str, reason: String },
    #[error("moss-transcribe-diarize prepared runtime failed: {reason}")]
    PreparedRuntimeFailed { reason: String },
    #[error("moss-transcribe-diarize prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize decoder-state capacity contract failed: {source}")]
    DecoderStateCapacity {
        #[source]
        source: Qwen3AsrKvCacheCapacityError,
    },
    #[error("moss-transcribe-diarize greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

const MOSS_TD_RUNTIME_ACTOR_MAX_IDLE_ENTRIES: usize = 4;
const MOSS_TD_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY: usize = 2;

type MossTdEncoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey);
type MossTdDecoderRuntimeCacheKey = (PackContentKey, ExecutionLaneKey, usize);

struct MossTdEncoderActorState {
    runtime: MossEncoderRuntime,
    _prepared_owner: PreparedRuntimeHandle<MossTdPreparedRuntime>,
}

struct MossTdDecoderActorState {
    runtime: MossTdDecoderRuntime,
    _prepared_owner: PreparedRuntimeHandle<MossTdPreparedRuntime>,
}

type MossTdEncoderRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MossTdEncoderRuntimeCacheKey, MossTdEncoderActorState>;
type MossTdDecoderRuntimePool =
    AdmittedPinnedRuntimeActorCheckoutPool<MossTdDecoderRuntimeCacheKey, MossTdDecoderActorState>;
type MossTdEncoderRuntimeActor =
    PinnedRuntimeActorCheckout<MossTdEncoderRuntimeCacheKey, MossTdEncoderActorState>;
type MossTdDecoderRuntimeActor =
    PinnedRuntimeActorCheckout<MossTdDecoderRuntimeCacheKey, MossTdDecoderActorState>;

#[derive(Clone)]
pub(crate) struct MossTdGgmlExecutor {
    prepared_runtimes: Arc<PreparedRuntimeCache<MossTdPreparedRuntime>>,
    encoder_runtimes: Arc<MossTdEncoderRuntimePool>,
    decoder_runtimes: Arc<MossTdDecoderRuntimePool>,
}

impl std::fmt::Debug for MossTdGgmlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTdGgmlExecutor").finish_non_exhaustive()
    }
}

impl Default for MossTdGgmlExecutor {
    fn default() -> Self {
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        let limits = AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
            MOSS_TD_RUNTIME_ACTOR_MAX_IDLE_ENTRIES,
            max_committed_requested_bytes,
            MOSS_TD_RUNTIME_ACTOR_MAX_INSTANCES_PER_KEY,
        );
        Self {
            prepared_runtimes: Arc::new(PreparedRuntimeCache::default()),
            encoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moss-td-encoder-owner",
                limits,
            )),
            decoder_runtimes: Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
                "openasr-moss-td-decoder-owner",
                limits,
            )),
        }
    }
}

const MOSS_TD_EXECUTOR_ID: &str = "moss-transcribe-diarize-ggml-executor-v1";
const MOSS_TD_STREAMING_EXECUTOR_ID: &str =
    "moss-transcribe-diarize-ggml-snapshot-streaming-executor-v1";

struct MossTdGreedyStepExecutor<'a> {
    decoder: &'a mut MossTdDecoderRuntime,
    layer_kv_caches: Qwen3AsrHostKvCacheOwner,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    prompt_embeddings: Option<Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
    /// Explicit cancel/pause/resume control for this decode -- never a
    /// thread-local. See [`crate::RequestExecutionContext`].
    control: std::sync::Arc<crate::api::backend::TranscriptionControl>,
}

impl Seq2SeqGreedyDecodeStepExecutor for MossTdGreedyStepExecutor<'_> {
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
                reason: "moss-transcribe-diarize generated token history is unexpectedly empty"
                    .to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "moss-transcribe-diarize decode cache position underflowed".to_string(),
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

/// Upstream `_compute_audio_token_length`'s per-chunk audio-token count: how
/// many post-merge adaptor tokens one Whisper-encoder chunk of `chunk_samples`
/// raw 16kHz samples produces, given `token_stride` (`hop_length` * the
/// Whisper conv stem's 2x stride * the adaptor's merge size). Pure integer
/// arithmetic with no model-pack dependency -- factored out of the encode
/// loop below so the slice-planning math can be pinned by a weight-free unit
/// test (`moss_td_chunk_frame_math_tests`) that runs in every default
/// `cargo nextest run`, unlike the family's real end-to-end `golden_diff_*`
/// tests, which need the private dev-only fp16 pack and stay `#[ignore]`d
/// (same artifact-policy constraint every other builtin family's CI golden
/// coverage works around -- see e.g. firered-aed's weight-free frontend
/// golden).
pub(crate) fn moss_td_chunk_token_length(chunk_samples: usize, token_stride: usize) -> usize {
    (chunk_samples - 1) / token_stride.max(1) + 1
}

/// This chunk's post-merge encoder frames actually kept: `token_length` audio
/// tokens each span `merge_size` pre-merge encoder frames, capped at the
/// encoder's `max_source_positions` (a full un-trimmed 30s chunk can never
/// legitimately need more than that many frames kept).
pub(crate) fn moss_td_chunk_keep_frames(
    token_length: usize,
    merge_size: usize,
    max_source_positions: usize,
) -> usize {
    (token_length * merge_size).min(max_source_positions)
}

/// Upstream's `time_merge` truncation: the total kept frames across every
/// chunk, rounded down to the nearest full `merge_size` group. In practice
/// every chunk's `moss_td_chunk_keep_frames` result is already a multiple of
/// `merge_size` (either `token_length * merge_size` directly, or the
/// `max_source_positions` cap, which is itself merge-size-aligned for every
/// real checkpoint), so summing them keeps the running total aligned too --
/// this is a no-op guard against that invariant, not a silent frame drop.
pub(crate) fn moss_td_aligned_frame_count(total_frames: usize, merge_size: usize) -> usize {
    let merge_size = merge_size.max(1);
    (total_frames / merge_size) * merge_size
}

/// Derive this request's decode budget: audio-proportional, clamped by both
/// the checkpoint's 4096-token runaway backstop and whatever decoder context
/// this request's own prompt left unused.
///
/// The context clamp is what makes the generous rate above safe to state. The
/// KV cache is allocated for exactly `prompt + budget` and the executor
/// rejects a request whose total does not fit, so an allowance the context
/// cannot serve is not a bigger budget -- it is a refused request. Clamping
/// here makes the budget "as much as this context can still serve, up to the
/// backstop": the largest honest answer available, and never a promise the
/// cache cannot keep.
fn moss_td_generated_token_budget(
    sample_count: usize,
    prompt_tokens: usize,
    kv_capacity: usize,
) -> Result<usize, MossTdExecutorError> {
    moss_td_budget_for_shape(sample_count, SAMPLE_RATE_HZ, prompt_tokens, kv_capacity).map_err(
        |error| MossTdExecutorError::DecodeBudgetUnavailable {
            reason: error.to_string(),
        },
    )
}

/// Weight-free, always-on coverage for the executor's chunk/slice-planning
/// arithmetic: pure integer math with no model pack involved, so (unlike the
/// family's `golden_diff_*` end-to-end tests below, which need the private
/// dev-only fp16 pack and stay `#[ignore]`d) these run in every default
/// `cargo nextest run --workspace`. Constants are pinned against the real
/// checkpoint's shape (`runtime_contract::tests::parses_adaptor_metadata_matching_real_checkpoint`'s
/// `merge_size == 4`, `package_import`'s `audio_merge_size: 4`, and
/// `parses_encoder_metadata_matching_real_checkpoint`'s
/// `max_source_positions == 1500` -- the standard Whisper-Medium 30s ->
/// 1500-frame shape).
#[cfg(test)]
mod moss_td_chunk_frame_math_tests {
    use super::*;

    const MERGE_SIZE: usize = 4;
    const MAX_SOURCE_POSITIONS: usize = 1500;
    const TOKEN_STRIDE: usize = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * MERGE_SIZE;

    fn budget_for(window_seconds: f32) -> usize {
        let samples = (window_seconds * SAMPLE_RATE_HZ as f32) as usize;
        // These tests exercise the audio-proportional rule itself. Exact
        // prompt+budget coverage is tested by the family topology module.
        moss_td_generated_token_budget(
            samples,
            0,
            crate::models::moss_transcribe_diarize::runtime_contract::MOSS_TD_MAX_KV_CACHE_POSITIONS,
        )
        .expect("budget")
    }

    /// The whole point of the declared slice window: at the family's maximum
    /// slice length, the audio prompt plus this call's generation budget must
    /// still fit inside the decoder's KV context, or the executor fails the
    /// request closed instead of decoding it. Pins the arithmetic that ties
    /// `OpenAsrLongformSliceShape::ScopedSlices` on the moss architecture
    /// descriptor to the budget rule -- the two are a pair, and widening the
    /// window alone silently eats the headroom.
    #[test]
    fn product_envelope_is_30s_target_and_60s_maximum() {
        let crate::arch::OpenAsrLongformSliceShape::ScopedSlices {
            integral_seconds,
            target_seconds,
            max_seconds,
        } = crate::arch::longform_slice_shape_for_model_architecture(
            crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
        )
        else {
            panic!("moss-transcribe-diarize must declare ScopedSlices");
        };
        assert_eq!(
            target_seconds,
            crate::arch::MOSS_TD_TARGET_INVOCATION_SECONDS as f32
        );
        assert_eq!(
            max_seconds,
            crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as f32
        );
        assert_eq!(integral_seconds, max_seconds);
        assert_eq!(budget_for(30.0), 818);
        assert_eq!(budget_for(60.0), 1_508);
    }

    /// A short clip keeps a small, audio-proportional budget: reserving the
    /// full backstop for ten seconds of speech would size its persistent Metal
    /// reuse graph for a transcript that cannot exist.
    #[test]
    fn a_short_clip_keeps_a_small_proportional_budget() {
        let budget = budget_for(11.0);
        assert!(
            budget < MOSS_TD_MAX_GENERATED_TOKENS / 4,
            "an 11s clip must not reserve the runaway backstop, got {budget}"
        );
        assert!(budget >= MOSS_TD_MIN_GENERATED_TOKENS);
    }

    /// The budget never outruns the context: a prompt that has already eaten
    /// most of the decoder leaves only what is left, so the executor's
    /// fail-closed capacity check cannot be handed an impossible request.
    #[test]
    fn the_budget_never_exceeds_the_context_the_prompt_left() {
        let kv_capacity = 4_096;
        let prompt_tokens = 3_900;
        let budget =
            moss_td_generated_token_budget(600 * SAMPLE_RATE_HZ, prompt_tokens, kv_capacity)
                .expect("budget");
        assert!(
            prompt_tokens + budget <= kv_capacity,
            "budget {budget} on top of prompt {prompt_tokens} overruns capacity {kv_capacity}"
        );
    }

    #[test]
    fn token_stride_matches_the_real_checkpoints_merge_size() {
        assert_eq!(TOKEN_STRIDE, 1_280);
    }

    #[test]
    fn short_clip_single_partial_chunk_keeps_the_expected_frame_count() {
        // A ~10s clip (jfk.wav-shaped): one partial 30s chunk, well under
        // `CHUNK_SAMPLES`, never hits the `max_source_positions` cap.
        let chunk_samples = 160_000; // 10s @ 16kHz
        let token_length = moss_td_chunk_token_length(chunk_samples, TOKEN_STRIDE);
        assert_eq!(token_length, 125);
        let keep_frames = moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        assert_eq!(keep_frames, 500);
    }

    #[test]
    fn full_chunk_saturates_max_source_positions_without_truncating() {
        // A full un-trimmed 30s chunk (`CHUNK_SAMPLES`) always keeps exactly
        // `max_source_positions` frames -- the encoder always outputs that
        // many for a full chunk, so the `.min()` cap lands exactly on it
        // rather than truncating away real content.
        let token_length = moss_td_chunk_token_length(CHUNK_SAMPLES, TOKEN_STRIDE);
        assert_eq!(token_length, 375);
        let keep_frames = moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        assert_eq!(keep_frames, MAX_SOURCE_POSITIONS);
    }

    #[test]
    fn multi_chunk_long_file_sums_every_chunks_kept_frames() {
        // A ~76s file (longform-shaped, like the other builtin families'
        // committed `fixtures/longform_en_zh.wav` golden): splits into three
        // `CHUNK_SAMPLES`-bounded chunks -- two full 30s chunks plus a ~16s
        // tail -- exercising the same multi-chunk accumulation the
        // executor's real encode loop runs across every chunk of a longform
        // request, all the way through the final merge-size-alignment
        // truncation, without needing a real pack/weights.
        let chunk_lens = [CHUNK_SAMPLES, CHUNK_SAMPLES, 256_000];
        let mut total_frames = 0usize;
        for &chunk_samples in &chunk_lens {
            let token_length = moss_td_chunk_token_length(chunk_samples, TOKEN_STRIDE);
            total_frames +=
                moss_td_chunk_keep_frames(token_length, MERGE_SIZE, MAX_SOURCE_POSITIONS);
        }
        assert_eq!(total_frames, 1_500 + 1_500 + 800);
        // Every chunk's kept-frame count is already a multiple of
        // `MERGE_SIZE`, so the running total across all three chunks stays
        // aligned and the final truncation is a no-op (see
        // `moss_td_aligned_frame_count`'s doc comment).
        assert_eq!(
            moss_td_aligned_frame_count(total_frames, MERGE_SIZE),
            total_frames
        );
    }

    #[test]
    fn aligned_frame_count_truncates_a_synthetic_misaligned_total() {
        // Real per-chunk totals are always already merge-size-aligned (see
        // the test above), so this never fires in production -- but the
        // truncation function itself must still behave correctly as
        // defense-in-depth if that invariant is ever violated by a future
        // change.
        assert_eq!(moss_td_aligned_frame_count(3_803, MERGE_SIZE), 3_800);
        assert_eq!(moss_td_aligned_frame_count(3_800, MERGE_SIZE), 3_800);
    }

    #[test]
    fn decode_budget_scales_to_the_real_moss_golden_lengths() {
        // The private-reference goldens emit 71 tokens for JFK (11s), 76 for
        // the mixed clip (13s), and 920 for the three-minute AISHELL-4 clip.
        // The two short clips stay on the proportional floor (no fixed
        // 4096-token Metal reuse-graph reservation for a few seconds of
        // speech); the historical three-minute direct-call fixture still
        // demonstrates the independent runaway backstop. Product slicing now
        // keeps legal invocations at or below 60s.
        assert_eq!(budget_for(11.0), 381);
        assert_eq!(budget_for(13.0), 427);
        assert_eq!(budget_for(180.0), MOSS_TD_MAX_GENERATED_TOKENS);
        // Every one of them still clears the golden's real token count with
        // room to spare.
        for (window_seconds, golden_tokens) in [(11.0_f32, 71), (13.0, 76), (180.0, 920)] {
            assert!(budget_for(window_seconds) > golden_tokens);
        }
    }
}

/// Encodes every 30s chunk of `samples` against the checked-out resident encoder
/// runtime for this pack+backend, returning the concatenated (already
/// merge-size-aligned) encoder rows and the number of kept frames -- the same
/// computation the executor's per-chunk loop always did, just routed through
/// the executor-owned actor pool instead of building a fresh
/// [`MossEncoderRuntime`] (and re-reading every encoder tensor from disk) on
/// every call.
fn encode_moss_td_chunks_with_cached_runtime(
    actor: &MossTdEncoderRuntimeActor,
    encoder_config: MossEncoderConfig,
    merge_size: usize,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), MossTdExecutorError> {
    // Upstream `_compute_audio_token_length`'s stride: hop_length * the
    // Whisper conv stem's 2x stride * audio_merge_size.
    let token_stride = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * merge_size;
    let samples = samples.to_vec();
    actor
        .call_mut(move |state| {
            let mut concatenated_rows: Vec<f32> = Vec::new();
            let mut total_frames = 0usize;
            for chunk in samples.chunks(CHUNK_SAMPLES) {
                let mel = whisper_log_mel_spectrogram_16khz_mono_v0(
                    chunk,
                    encoder_config.n_mels,
                    MEL_TARGET_FRAMES,
                )
                .map_err(|error| MossTdExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
                let encoder_out = state
                    .runtime
                    .encode(encoder_config, mel.data(), MEL_TARGET_FRAMES)
                    .map_err(|error| MossTdExecutorError::EncoderFailed {
                        reason: error.to_string(),
                    })?;
                let token_length = moss_td_chunk_token_length(chunk.len(), token_stride);
                let keep_frames = moss_td_chunk_keep_frames(
                    token_length,
                    merge_size,
                    encoder_config.max_source_positions,
                );
                let keep_values = keep_frames * encoder_config.d_model;
                concatenated_rows.extend_from_slice(&encoder_out[..keep_values]);
                total_frames += keep_frames;
            }
            Ok((concatenated_rows, total_frames))
        })
        .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
            stage: "encoder",
            reason: error.to_string(),
        })?
}

/// One decode's text plus how the shared driver ended it. The stop reason is
/// what keeps `speaker_segments` from closing a cut-short decode's final
/// segment at the end of the clip (see [`MossTdDecodeExtent`]) and what the
/// executor lifts into the transcript's truncation signal.
struct MossTdDecodeOutput {
    text: String,
    stop_reason: Seq2SeqGreedyDecodeStopReason,
}

/// Runs the ChatML+audio-splice prompt embedding through the cached, resident
/// decoder runtime for this pack+backend: prefill, then the shared greedy
/// decode driver through to `<|im_end|>` (or the fail-closed token budget),
/// returning the trimmed decode text. Mirrors `firered_aed::executor`'s
/// `decode_with_cached_runtime`: the runtime (loaded weights + the Qwen
/// decode graph's reuse machinery) stays resident across calls. Every
/// per-utterance host KV cache is allocated fresh at the exact logical span;
/// the device arena/graph remains at the stable session-envelope reserve and
/// is logically reset by the next prefill. `release_session_scoped_buffers`
/// below drops poisoned resident state and CPU-only scratch before reuse.
#[allow(clippy::too_many_arguments)]
fn run_moss_td_decoder_with_cached_runtime(
    actor: &MossTdDecoderRuntimeActor,
    decoder_metadata: MossTdDecoderMetadata,
    kv_capacity: Qwen3AsrKvCacheCapacity,
    max_generated_tokens: usize,
    decode_prompt_token_ids: &[u32],
    audio_pad_positions: &[usize],
    audio_rows: &[f32],
    tokenizer: MossTdTokenizer,
    control: Arc<crate::api::backend::TranscriptionControl>,
) -> Result<MossTdDecodeOutput, MossTdExecutorError> {
    let decode_prompt_token_ids = decode_prompt_token_ids.to_vec();
    let audio_pad_positions = audio_pad_positions.to_vec();
    let audio_rows = audio_rows.to_vec();
    actor
        .call_mut(move |state| {
            let decoder = &mut state.runtime;
            if std::env::var_os("OPENASR_MOSS_TD_PROFILE").is_some() {
                eprintln!(
                    "OPENASR_MOSS_TD_PROFILE decoder_backend={}",
                    decoder.backend_label()
                );
            }

            let token_rows_len = decode_prompt_token_ids.len() * decoder_metadata.d_model;
            let mut token_rows = Vec::with_capacity(token_rows_len);
            for &token_id in &decode_prompt_token_ids {
                let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                    MossTdExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    }
                })?;
                token_rows.extend_from_slice(&row);
            }
            let spliced = build_moss_td_prompt_embeddings_with_audio_splice(
                decode_prompt_token_ids.len(),
                &audio_pad_positions,
                decoder_metadata.d_model,
                &token_rows,
                &audio_rows,
            )
            .map_err(|error| MossTdExecutorError::PromptEmbeddingFailed {
                reason: error.to_string(),
            })?;
            let prompt_embeddings = Qwen3AsrPromptEmbeddings {
                hidden_size: spliced.hidden_size,
                token_count: spliced.token_count,
                token_major_values: spliced.token_major_values,
            };

            let layer_kv_caches = decoder
                .new_kv_caches(kv_capacity)
                .map_err(|reason| MossTdExecutorError::DecoderFailed { reason })?;
            let mut step_executor = MossTdGreedyStepExecutor {
                decoder,
                layer_kv_caches,
                kv_capacity,
                prompt_embeddings: Some(prompt_embeddings),
                cache_prompt_tokens: 0,
                control: Arc::clone(&control),
            };
            let config = BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: decode_prompt_token_ids.to_vec(),
                eot_token_id: tokenizer.im_end_token_id,
                vocab_size: decoder_metadata.vocab_size,
                max_generated_tokens,
            };
            let result = run_builtin_seq2seq_decode_policy(
                crate::arch::MOSS_TD_DECODE_POLICY_ID,
                &config,
                &tokenizer,
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
            );
            // Release this request's per-token grow-to-fit host buffer before
            // the runtime goes back into the cache (mirrors qwen3-asr's
            // `ggml_executor`'s `release_session_scoped_buffers` call around
            // its own resident whole-decoder cache) -- unconditionally, on
            // both the success and failure paths, so a failed decode never
            // leaves a session-scoped allocation riding along on the cached
            // runtime.
            step_executor.decoder.release_session_scoped_buffers();
            let result = match result {
                Ok(result) => result,
                // Budget exhausted before `<|im_end|>`: keep the generated
                // prefix instead of failing the whole request closed, matching
                // firered-aed's handling of the same driver error. A partial
                // transcript is a real answer for the audio it covers, and
                // `truncated` below keeps it labelled as one -- the segment
                // assembler will not stretch the last segment over the audio
                // the decode never reached. Discarding it instead returns
                // nothing at all for a recording the model largely transcribed,
                // which is strictly worse for the same underlying shortfall.
                Err(Seq2SeqGreedyDecodeError::EotNotReachedBeforeMaxTokens {
                    generated_tokens,
                    ..
                }) => {
                    let text = tokenizer.decode_text_token_ids(&generated_tokens).map_err(
                        |error| MossTdExecutorError::GreedyDecodeFailed {
                            reason: format!(
                                "tokenizer decode of the budget-exhausted prefix failed: {error}"
                            ),
                        },
                    )?;
                    Seq2SeqGreedyDecodeResult {
                        text,
                        generated_tokens,
                        generated_probabilities: Vec::new(),
                        stop_reason: Seq2SeqGreedyDecodeStopReason::BudgetExhausted,
                    }
                }
                Err(error) => {
                    return Err(MossTdExecutorError::GreedyDecodeFailed {
                        reason: error.to_string(),
                    });
                }
            };
            Ok(MossTdDecodeOutput {
                text: result.text.trim().to_string(),
                stop_reason: result.stop_reason,
            })
        })
        .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
            stage: "decoder",
            reason: error.to_string(),
        })?
}

impl MossTdGgmlExecutor {
    fn map_actor_error(stage: &'static str, error: PinnedRuntimeActorError) -> MossTdExecutorError {
        MossTdExecutorError::RuntimeOwnershipFailed {
            stage,
            reason: error.to_string(),
        }
    }

    fn prepared_runtime_for_preflight(
        &self,
        preflight: &crate::GgmlAsrRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<PreparedRuntimeHandle<MossTdPreparedRuntime>, MossTdExecutorError> {
        self.prepared_runtimes.get_or_try_insert_with(
            &preflight.runtime_source,
            PreparedRuntimeQuoteContext {
                model_architecture: crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
                metadata: &preflight.metadata,
                tensor_index: &preflight.tensor_index,
                backend,
            },
            || {
                build_moss_td_prepared_runtime(preflight, backend).map_err(
                    |error: MossTdPreparedRuntimeError| {
                        MossTdExecutorError::PreparedRuntimeFailed {
                            reason: error.to_string(),
                        }
                    },
                )
            },
            || MossTdExecutorError::PreparedRuntimeFailed {
                reason: "prepared-runtime cache lock poisoned".to_string(),
            },
            |error| MossTdExecutorError::RuntimeOwnershipFailed {
                stage: "prepared",
                reason: error.to_string(),
            },
        )
    }

    fn checkout_encoder_runtime(
        &self,
        preflight: &crate::GgmlAsrRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MossTdPreparedRuntime>,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<MossTdEncoderRuntimeActor, MossTdExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(moss_td_encoder_graph_config(backend).backend),
        );
        let preflight = preflight.clone();
        self.encoder_runtimes.checkout_or_try_build_with(
            key,
            move || Ok((0, (preflight, prepared))),
            move |(preflight, prepared)| {
                let runtime = MossEncoderRuntime::new_with_prepared_weights_from_preflight(
                    &preflight,
                    Arc::clone(&prepared.encoder_weights),
                    backend,
                )
                .map_err(|error| MossTdExecutorError::EncoderFailed {
                    reason: format!("could not initialize encoder runtime: {error}"),
                })?;
                Ok(SystemMemoryOwner::without_allocation(
                    MossTdEncoderActorState {
                        runtime,
                        _prepared_owner: prepared,
                    },
                ))
            },
            |error| Self::map_actor_error("encoder", error),
        )
    }

    fn checkout_decoder_runtime(
        &self,
        preflight: &crate::GgmlAsrRuntimeSourcePreflight,
        prepared: PreparedRuntimeHandle<MossTdPreparedRuntime>,
        kv_capacity: Qwen3AsrKvCacheCapacity,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<MossTdDecoderRuntimeActor, MossTdExecutorError> {
        let key = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            current_execution_lane_key(moss_td_runtime_graph_config(backend).backend),
            kv_capacity.resident_positions(),
        );
        let preflight = preflight.clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        self.decoder_runtimes.checkout_or_try_build_with(
            key,
            move || {
                let retained = MossTdDecoderRuntime::quoted_resident_system_memory_bytes(
                    prepared.decoder_metadata.n_layers,
                )
                .map_err(|reason| MossTdExecutorError::RuntimeOwnershipFailed {
                    stage: "decoder",
                    reason,
                })?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!(
                        "moss-td-decoder-runtime:{content_id}:positions={}",
                        kv_capacity.resident_positions()
                    ),
                    retained,
                    retained,
                )
                .map_err(|error| MossTdExecutorError::RuntimeOwnershipFailed {
                    stage: "decoder",
                    reason: error.to_string(),
                })?;
                Ok((retained, (preflight, prepared, quote)))
            },
            move |(preflight, prepared, quote)| match SystemMemoryOwner::try_allocate_transaction(
                quote,
                || {
                    let runtime = MossTdDecoderRuntime::new_with_prepared_state_from_preflight(
                        &preflight,
                        prepared.decoder_metadata,
                        Arc::clone(&prepared.decoder_plan),
                        Arc::clone(&prepared.logits_head),
                        Arc::clone(&prepared.token_embedding),
                        backend,
                    )
                    .map_err(|error| MossTdExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    })?;
                    let retained = runtime.resident_system_memory_bytes().map_err(|reason| {
                        MossTdExecutorError::RuntimeOwnershipFailed {
                            stage: "decoder",
                            reason,
                        }
                    })?;
                    Ok(SystemMemoryAllocationOutcome::new(
                        MossTdDecoderActorState {
                            runtime,
                            _prepared_owner: prepared,
                        },
                        retained,
                        retained,
                    ))
                },
            ) {
                Ok(owner) => Ok(owner),
                Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                    Err(MossTdExecutorError::RuntimeOwnershipFailed {
                        stage: "decoder",
                        reason: error.to_string(),
                    })
                }
            },
            |error| Self::map_actor_error("decoder", error),
        )
    }

    fn clear_runtime_owners(&self) {
        self.encoder_runtimes.clear();
        self.decoder_runtimes.clear();
        self.prepared_runtimes.clear();
    }

    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        self.encoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.decoder_runtimes
            .evict_where(|key| key.0.pack_content_id == pack_content_id);
        self.prepared_runtimes.evict_content_id(pack_content_id);
    }

    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionViewRequest,
    ) -> Result<GgmlAsrExecutionResult, MossTdExecutorError> {
        let expected_adapter = crate::arch::MOSS_TD_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(MossTdExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| MossTdExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;

        let backend = request.resolved_runtime.backend();
        let prepared = self.prepared_runtime_for_preflight(&preflight, backend)?;
        let encoder_metadata = prepared.encoder_metadata;
        let adaptor_metadata = prepared.adaptor_metadata;
        let decoder_metadata = prepared.decoder_metadata;
        let tokenizer = prepared.tokenizer.clone();

        let samples = &request.prepared_audio.samples_f32;
        if samples.is_empty() {
            return Err(MossTdExecutorError::EmptyAudio);
        }
        let max_samples = SAMPLE_RATE_HZ
            .checked_mul(crate::arch::MOSS_TD_MAX_INVOCATION_SECONDS as usize)
            .ok_or_else(|| MossTdExecutorError::RuntimeContractViolation {
                reason: "MOSS-TD product invocation sample ceiling overflowed".to_string(),
            })?;
        if samples.len() > max_samples {
            return Err(MossTdExecutorError::InvocationTooLong {
                samples: samples.len(),
                max_samples,
            });
        }
        // Derived from THIS call's buffer, never from a request-level "whole
        // recording" duration. Under longform slicing that buffer is one slice,
        // and this value is what `speaker_segments` clamps a truncated decode's
        // final segment to -- so a slice that ends without a stop token can only
        // ever blanket the rest of its own slice, not the rest of the recording.
        let audio_duration_seconds = samples.len() as f32 / SAMPLE_RATE_HZ as f32;

        let encoder_config = MossEncoderConfig {
            n_layers: encoder_metadata.n_layers,
            d_model: encoder_metadata.d_model,
            n_heads: encoder_metadata.n_heads,
            n_mels: encoder_metadata.n_mels,
            max_source_positions: encoder_metadata.max_source_positions,
        };
        let encoder_actor =
            self.checkout_encoder_runtime(&preflight, Arc::clone(&prepared), backend)?;
        let (mut concatenated_rows, total_frames) = encode_moss_td_chunks_with_cached_runtime(
            &encoder_actor,
            encoder_config,
            adaptor_metadata.merge_size,
            samples,
        )?;
        let aligned_frames = moss_td_aligned_frame_count(total_frames, adaptor_metadata.merge_size);
        concatenated_rows.truncate(aligned_frames * encoder_metadata.d_model);

        let (audio_rows, audio_token_count) = run_moss_adaptor(
            &prepared.adaptor_weights,
            &concatenated_rows,
            aligned_frames,
            encoder_metadata.d_model,
            adaptor_metadata.merge_size,
        )
        .map_err(|error| MossTdExecutorError::AdaptorFailed {
            reason: error.to_string(),
        })?;

        let decode_prompt =
            build_moss_td_decode_prompt(&tokenizer, audio_token_count).map_err(|error| {
                MossTdExecutorError::DecodePromptFailed {
                    reason: error.to_string(),
                }
            })?;

        // Fail closed up front when this call's prompt plus the configured
        // decode budget cannot fit the decoder's KV context. The shared native
        // slicer keeps ordinary requests well inside it (the family declares
        // its own slice window via `OpenAsrLongformSliceShape::ScopedSlices`),
        // so this is the backstop for a caller that bypasses longform slicing
        // entirely. The request-sized cache must reserve every possible decode
        // position; clamping an over-limit request would defer the failure to a
        // cryptic KV write mid-generation.
        let kv_capacity = moss_td_kv_cache_positions(decoder_metadata.max_positions);
        // Sized once the prompt is known, so the budget can claim the decoder
        // context the prompt did not need (see `moss_td_generated_token_budget`).
        let max_generated_tokens = moss_td_generated_token_budget(
            samples.len(),
            decode_prompt.token_ids.len(),
            kv_capacity,
        )?;
        let semantic_required_positions = decode_prompt
            .token_ids
            .len()
            .checked_add(max_generated_tokens)
            .ok_or_else(|| MossTdExecutorError::DecodeBudgetUnavailable {
                reason: "prompt plus generation position count overflowed".to_string(),
            })?;
        let request_kv_cache_positions = moss_td_request_kv_cache_positions(
            decoder_metadata.max_positions,
            decode_prompt.token_ids.len(),
            max_generated_tokens,
        )
        .ok_or_else(|| MossTdExecutorError::AudioExceedsContext {
            prompt_tokens: decode_prompt.token_ids.len(),
            generation_budget: max_generated_tokens,
            required_positions: semantic_required_positions,
            kv_capacity,
            max_minutes: (kv_capacity.saturating_sub(max_generated_tokens) as f32
                / AUDIO_TOKENS_PER_SECOND_FOR_LIMIT
                / 60.0)
                .max(0.0),
        })?;
        let kv_capacity_plan = Qwen3AsrKvCacheCapacity::from_decoder_state(
            &request.decoder_state,
            super::capacity::MOSS_TD_SELF_KV_STATE_ID,
        )
        .and_then(|capacity| {
            capacity.validate_measured_logical_positions(request_kv_cache_positions)
        })
        .map_err(|source| MossTdExecutorError::DecoderStateCapacity { source })?;
        if kv_capacity_plan.resident_positions() > kv_capacity {
            return Err(MossTdExecutorError::RuntimeContractViolation {
                reason: format!(
                    "planned resident KV span {} exceeds the validated MOSS cap {kv_capacity}",
                    kv_capacity_plan.resident_positions()
                ),
            });
        }

        let decoder_actor =
            self.checkout_decoder_runtime(&preflight, prepared, kv_capacity_plan, backend)?;
        let decoded = run_moss_td_decoder_with_cached_runtime(
            &decoder_actor,
            decoder_metadata,
            kv_capacity_plan,
            max_generated_tokens,
            &decode_prompt.token_ids,
            &decode_prompt.audio_pad_positions,
            &audio_rows,
            tokenizer,
            Arc::clone(&request.execution_context.control),
        )?;
        // Normalize the model's own inline `[start][end][SNN]` markup into the
        // engine's shared segment representation. The decode prompt is fixed,
        // so the markers are written whether or not the request asked for
        // speakers: stripping them from the transcript is this layer's job, and
        // `in_decoder_speakers` decides only whether the recording-local
        // `SPEAKER_NN` labels survive. See `speaker_segments`'s module doc for
        // the grammar, the fail-closed policy, and the degrade shape.
        let normalized = super::speaker_segments::normalize_moss_td_decode(
            &decoded.text,
            MossTdDecodeExtent {
                audio_duration_seconds,
                truncated: decoded.stop_reason.is_truncated(),
            },
            request.request_options.in_decoder_speakers,
        );
        // moss-td is the one family with decoder-emitted timestamps, so it can
        // name the point the transcript stops describing the audio instead of
        // only reporting that it does.
        let decode_truncation = decoded
            .stop_reason
            .into_decode_truncation(normalized.truncated_at_seconds);
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            segments: normalized.segments,
            text: normalized.text,
            longform: None,
            language: None,
        };
        Ok(GgmlAsrExecutionResult {
            transcription,
            carry_context: None,
            decode_truncation,
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

impl GgmlAsrViewExecutor for MossTdGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MOSS_TD_EXECUTOR_ID
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
                super::capacity::plan_moss_td_decoder_state,
                super::capacity::MOSS_TD_DECODER_STATE_STREAMS,
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
        self.clear_runtime_owners();
    }
}

/// Not a true incremental streaming session -- this family's architecture
/// needs the full audio up front to place its numeric time-anchor markers
/// (see `decode_prompt`'s module doc), so there is no meaningful "partial"
/// mode yet (matches the top-of-file doc's "file-transcribe only" note).
/// Still registers a buffered snapshot-streaming session (mirrors
/// `firered_llm`'s identical precedent: a family with no real partial path
/// still needs SOME streaming executor, or the builtin dispatch's
/// fail-fast completeness gate rejects the whole registry at startup) so a
/// live-caption request degrades to "one final result at end of audio"
/// instead of silently falling back to a broken cadence.
impl GgmlAsrStreamingExecutor for MossTdGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MOSS_TD_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            MOSS_TD_STREAMING_EXECUTOR_ID,
            crate::arch::MOSS_TD_GGML_ADAPTER_ID,
            "moss-transcribe-diarize",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            MossTdGgmlExecutor::execute_view,
        )
    }

    fn unload_idle_state(&self) {
        self.clear_runtime_owners();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::ggml_runtime::install_request_backend_override;
    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudioView};
    use crate::models::ggml_family_registry::moss_transcribe_diarize_runtime_descriptor_v1;

    use crate::api::backend::Segment;

    use super::super::speaker_segments::parse_moss_td_speaker_segments;
    use super::*;

    /// Real converted dev pack (fp16), NOT committed -- same dev-only-artifact
    /// convention as `decode_prompt`'s own `dev_pack_path` and mimo-asr's
    /// `mimo-v2.5-asr-q8_0.oasr`.
    fn dev_pack_path() -> Option<PathBuf> {
        crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_PACK",
            "MOSS Transcribe Diarize .oasr pack",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}"))
        .ok()
    }

    fn dev_sample_path(name: &str) -> PathBuf {
        match crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_SAMPLES",
            "MOSS Transcribe Diarize sample directory",
        ) {
            Ok(path) => path.join(name),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                PathBuf::new()
            }
        }
    }

    // The `GOLDEN_*_TEXT` constants below are the raw *reference decode* -- the
    // tagged string the model itself produces, compared against the HF fp32
    // reference. They are deliberately NOT what the executor returns: the
    // family's inline markup is an internal transport for speaker structure and
    // is normalized away before anything else sees it (see
    // `speaker_segments`'s module doc), so the executor's flat text is the
    // markup-free projection of these, obtained through the same normalizer.
    // Keeping the goldens in reference form is what lets a decode regression
    // (different words, shifted anchors, a lost speaker change) still show up
    // here instead of being hidden by the stripping.
    //
    // Pinned to the real dev-pack CPU decode (backend forced to CPU below).
    // The encoder binds its 2D projection weights zero-copy as native f16 and
    // runs flash attention (see `encoder_graph`), so this decode path is f16
    // weights + flash, NOT the f32-naive path -- do not assert flash == naive or
    // f16 == f32 bit-for-bit. What IS asserted, matching the reference-platform
    // golden policy: the transcript is text-level identical to the HF fp32
    // reference (`tmp/moss-td/golden/*.json`'s `text`), including speaker labels,
    // and every emitted time anchor is within 0.05s of it. In practice jfk and
    // the 3-minute aishell clip come out byte-for-byte equal to the HF golden
    // (time anchors included); en_zh_mixed matches the HF text exactly with two
    // anchors shifted by 0.02s ([2.34]->[2.32], [4.94]->[4.96]), the f16+flash
    // numeric delta.
    const GOLDEN_JFK_TEXT: &str = concat!(
        "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S01] ask not what your ",
        "country can do for you,[7.71][8.12][S01] ask what you can do for your country.[10.59]",
    );

    // Code-switch coverage: `en_zh_mixed.wav` mixes English then Mandarin in a
    // single utterance, exercising both tokenizer/decode paths plus a second
    // speaker label (`[S02]`) in one prefill+decode. Text identical to the HF
    // golden `en_zh_mixed.json`'s `text`; two time anchors sit 0.02s off (see the
    // pinning note above).
    const GOLDEN_EN_ZH_MIXED_TEXT: &str = concat!(
        "[0.27][S01]And so, my fellow Americans,[2.32][3.21][S01]ask not.",
        "[4.44][4.96][S02]今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去伊加新[12.88]",
    );

    /// The flat transcript a caller receives for a given reference decode: the
    /// same words with the family's markup normalized away.
    fn normalized_golden_text(reference_decode: &str, audio_duration_seconds: f32) -> String {
        super::super::speaker_segments::normalize_moss_td_decode(
            reference_decode,
            MossTdDecodeExtent::complete(audio_duration_seconds),
            true,
        )
        .text
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<(String, std::time::Duration, f32)> {
        // Force CPU. This family's Metal path has two open defects (encoder
        // numeric divergence -> empty-shell output, and a per-step wired-memory
        // blow-up -- see the `arch` descriptor's `auto_gpu_policy` note), so the
        // reference decode is CPU-only.
        transcribe_with_dev_pack_backend(wav_path, GgmlAsrBackendPreference::CpuOnly).map(
            |(text, _, elapsed, audio_duration_seconds)| (text, elapsed, audio_duration_seconds),
        )
    }

    /// Same dev-pack e2e path as [`transcribe_with_dev_pack`], but lets the
    /// caller pick the backend preference -- used by the `_accelerated`
    /// variants below to drive an explicit `execution_target=accelerated`
    /// request end to end (encoder AND decode), the same override an
    /// `Accelerated` request installs in production (see
    /// `GgmlAsrBackendPreference::request_backend_override`'s doc and
    /// `graph_config.rs`'s note that an explicit request always wins over
    /// Auto Metal; family policy is AllBackends).
    fn transcribe_with_dev_pack_backend(
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, Vec<Segment>, std::time::Duration, f32)> {
        let executor = MossTdGgmlExecutor::default();
        transcribe_with_dev_pack_backend_using_executor(&executor, wav_path, backend_preference)
    }

    /// Execute one dev-pack request through the caller-provided executor.
    /// Keeping the executor outside this helper lets tests prove that the
    /// owner-actor pools retain and reuse runtimes across requests, while the
    /// ordinary fixture helpers below still get an isolated default executor.
    fn transcribe_with_dev_pack_backend_using_executor(
        executor: &MossTdGgmlExecutor,
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, Vec<Segment>, std::time::Duration, f32)> {
        let pack_path = dev_pack_path()?;
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return None;
        }
        // `backend_preference` alone is inert on a direct `execute()` (it is
        // only consulted via the thread-local override -- see
        // `GgmlAsrExecutionViewRequest::backend_preference`'s doc), so install the
        // override explicitly rather than relying on the ambient backend.
        // Hold the RAII guard for the whole decode: it restores the previous
        // thread-local override on drop at the end of this function.
        let _backend_override_guard =
            install_request_backend_override(backend_preference.request_backend_override());
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            backend_preference.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
        );

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;

        let request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: moss_transcribe_diarize_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            // The goldens pin the reference decode including its speaker
            // structure, so ask for it -- with Voice ID off the normalizer
            // drops the labels by design (see `speaker_segments`).
            request_options: crate::models::ggml_asr_executor::GgmlAsrExecutionOptions {
                in_decoder_speakers: true,
                ..Default::default()
            },
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };

        let started_at = Instant::now();
        let result = executor.execute_view(&request).expect("moss-td transcribe");
        let elapsed = started_at.elapsed();
        Some((
            result.transcription.text,
            result.transcription.segments,
            elapsed,
            audio_duration_seconds,
        ))
    }

    /// Same dev-pack e2e path as [`transcribe_with_dev_pack`], but returns the
    /// full [`Segment`] list instead of only the flat text -- used to check
    /// that the real decode's speaker/time-anchor markup round-trips through
    /// `speaker_segments::parse_moss_td_speaker_segments` (as wired into the
    /// executor) into the same structure the golden `[Sxx]`/`[t]` tags encode.
    fn transcribe_with_dev_pack_segments(wav_path: PathBuf) -> Option<Vec<Segment>> {
        let pack_path = dev_pack_path()?;
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        if !wav_path.exists() {
            eprintln!("skipping: {} not present", wav_path.display());
            return None;
        }
        let _backend_override_guard = install_request_backend_override(
            GgmlAsrBackendPreference::CpuOnly.request_backend_override(),
        );
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            GgmlAsrBackendPreference::CpuOnly.request_backend_override(),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID,
            ),
        );
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: moss_transcribe_diarize_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(samples),
            request_options: crate::models::ggml_asr_executor::GgmlAsrExecutionOptions {
                in_decoder_speakers: true,
                ..Default::default()
            },
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime,
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let executor = MossTdGgmlExecutor::default();
        let result = executor.execute_view(&request).expect("moss-td transcribe");
        Some(result.transcription.segments)
    }

    /// Two-layer comparison for the accelerated e2e smoke tests, run over the
    /// normalized segments rather than the raw tagged string (the family's
    /// markup never leaves its normalizer, see `speaker_segments`): (1) the
    /// segment count, each segment's text/punctuation, and each segment's
    /// speaker label must match the CPU golden's exactly; (2) each segment's
    /// start/end -- the family's time anchors, in normalized form -- only needs
    /// to be within `tolerance_secs` of the golden's, not bit-identical.
    ///
    /// Rationale for tolerating (2) rather than requiring (1)'s strictness
    /// there too: this repo's own firered-aed encoder parity investigation
    /// (`firered_aed::encoder_graph::parity_tests`, see its `dump_...`
    /// harness doc comment) already concluded that cross-backend/cross-
    /// implementation fp32 bit-identical output is not a goal this runtime
    /// has ever held anywhere -- ggml's vs another implementation's non-
    /// bit-identical fp32 reduction order routinely produces small absolute
    /// diffs at numerically delicate positions without either side being
    /// wrong. Time anchors here are exactly such a floating-point-derived
    /// value (not a token id), and the measured 0.02s CPU-vs-accelerated
    /// divergence on `en_zh_mixed.wav` lands the accelerated run on the same
    /// values as the HF fp32 reference (see that test's comment) -- i.e.
    /// both sides are plausible fp32 outcomes, not a defect on either one.
    fn assert_segments_match_golden_within_anchor_tolerance(
        actual: &[Segment],
        golden_reference_decode: &str,
        audio_duration_seconds: f32,
        tolerance_secs: f32,
    ) {
        let golden = super::super::speaker_segments::parse_moss_td_speaker_segments(
            golden_reference_decode,
            MossTdDecodeExtent::complete(audio_duration_seconds),
        )
        .expect("the golden reference decode parses");
        assert_eq!(
            actual.len(),
            golden.len(),
            "segment count diverged from the CPU golden"
        );
        for (index, (actual_segment, golden_segment)) in actual.iter().zip(&golden).enumerate() {
            assert_eq!(
                actual_segment.text, golden_segment.text,
                "segment[{index}] text/punctuation diverged from the CPU golden (strict layer -- \
                 times are compared separately with tolerance)"
            );
            assert_eq!(
                actual_segment.speaker, golden_segment.speaker,
                "segment[{index}] speaker label diverged from the CPU golden"
            );
            for (edge, actual_time, golden_time) in [
                ("start", actual_segment.start, golden_segment.start),
                ("end", actual_segment.end, golden_segment.end),
            ] {
                let diff = (actual_time - golden_time).abs();
                assert!(
                    diff <= tolerance_secs,
                    "segment[{index}].{edge} exceeds tolerance: actual={actual_time} \
                     golden={golden_time} diff={diff:.4}s (tolerance={tolerance_secs}s)"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a local moss-transcribe-diarize .oasr pack and jfk.wav; runs the CPU host-prefill path"]
    fn voice_id_disabled_real_jfk_request_prefills_without_prior_host_history() {
        // A server request with `diarize=false` reaches this native MOSS
        // executor before any optional diarization or Voice ID post-processing.
        // Keep this a real-pack smoke so an empty Q8_0 host KV prefix cannot
        // regress into a cache-count error behind the HTTP boundary.
        let Some((text, _, _)) = transcribe_with_dev_pack(dev_sample_path("jfk.wav")) else {
            return;
        };
        assert!(
            !text.trim().is_empty(),
            "MOSS must return a transcript before optional Voice ID processing"
        );
    }

    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn golden_diff_end_to_end_transcribe_jfk_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(dev_sample_path("jfk.wav"))
        else {
            return;
        };
        eprintln!(
            "moss-td e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, normalized_golden_text(GOLDEN_JFK_TEXT, 10.59));
    }

    /// Pins the resident owner pool's two contracts: (1) a second
    /// `execute()` through the same executor reuses both owner-actor runtimes
    /// rather than creating another cache entry, and (2) reuse changes nothing
    /// observable: the second call's transcript is byte-for-byte identical to
    /// the first (and to `GOLDEN_JFK_TEXT`).
    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn resident_runtime_cache_hits_on_a_second_transcribe_call_for_the_same_pack() {
        let executor = MossTdGgmlExecutor::default();
        let Some((first_text, _, _, _)) = transcribe_with_dev_pack_backend_using_executor(
            &executor,
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };
        assert_eq!(first_text, normalized_golden_text(GOLDEN_JFK_TEXT, 10.59));
        assert_eq!(
            executor.encoder_runtimes.usage_for_test().0,
            1,
            "first call must retain one idle encoder owner"
        );
        assert_eq!(
            executor.decoder_runtimes.usage_for_test().0,
            1,
            "first call must retain one idle decoder owner"
        );

        let Some((second_text, _, _, _)) = transcribe_with_dev_pack_backend_using_executor(
            &executor,
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::CpuOnly,
        ) else {
            return;
        };
        assert_eq!(
            second_text, first_text,
            "reusing the owner-actor runtimes must not change the decode"
        );
        assert_eq!(
            executor.encoder_runtimes.usage_for_test().0,
            1,
            "second call must reuse the same idle encoder owner"
        );
        assert_eq!(
            executor.decoder_runtimes.usage_for_test().0,
            1,
            "second call must reuse the same idle decoder owner"
        );
    }

    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(dev_sample_path("en_zh_mixed.wav"))
        else {
            return;
        };
        eprintln!(
            "moss-td e2e [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, normalized_golden_text(GOLDEN_EN_ZH_MIXED_TEXT, 12.88));
    }

    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn golden_diff_end_to_end_transcribe_jfk_wav_speaker_segments() {
        let Some(segments) = transcribe_with_dev_pack_segments(dev_sample_path("jfk.wav")) else {
            return;
        };
        // Same three speaker turns the flat-text golden's `[Sxx]`/`[t]` tags
        // encode (see `golden_diff_end_to_end_transcribe_jfk_wav` and
        // `GOLDEN_JFK_TEXT`) -- this asserts the executor's real dev-pack
        // decode round-trips through `speaker_segments` into that same
        // structure, not just that the flat string matches.
        let expected =
            parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, MossTdDecodeExtent::complete(10.59))
                .expect("golden text itself must parse");
        assert_eq!(segments, expected);
    }

    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav_speaker_segments() {
        let Some(segments) = transcribe_with_dev_pack_segments(dev_sample_path("en_zh_mixed.wav"))
        else {
            return;
        };
        let expected = parse_moss_td_speaker_segments(
            GOLDEN_EN_ZH_MIXED_TEXT,
            MossTdDecodeExtent::complete(12.88),
        )
        .expect("golden text itself must parse");
        assert_eq!(segments, expected);
    }

    /// Snapshot of the shape `speaker_segments` produces for the two golden
    /// transcripts pinned above, independent of any dev-pack decode -- pins
    /// the exact segment count/speaker-label/start/end/text tuple this PR's
    /// parser derives from the reference HF text, so a future edit to the
    /// grammar (e.g. changing how a back-to-back closing/opening anchor pair
    /// is split) shows up as a diff here even without the private pack.
    #[test]
    fn snapshot_jfk_and_en_zh_mixed_golden_speaker_segments() {
        let jfk =
            parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, MossTdDecodeExtent::complete(10.59))
                .expect("jfk parses");
        let jfk_snapshot: Vec<(&str, f32, f32, &str)> = jfk
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            jfk_snapshot,
            vec![
                ("SPEAKER_01", 0.28, 2.32, "And so, my fellow Americans,"),
                (
                    "SPEAKER_01",
                    3.22,
                    7.71,
                    "ask not what your country can do for you,"
                ),
                (
                    "SPEAKER_01",
                    8.12,
                    10.59,
                    "ask what you can do for your country."
                ),
            ]
        );

        let en_zh_mixed = parse_moss_td_speaker_segments(
            GOLDEN_EN_ZH_MIXED_TEXT,
            MossTdDecodeExtent::complete(12.88),
        )
        .expect("parses");
        let en_zh_mixed_snapshot: Vec<(&str, f32, f32, &str)> = en_zh_mixed
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            en_zh_mixed_snapshot,
            vec![
                ("SPEAKER_01", 0.27, 2.32, "And so, my fellow Americans,"),
                ("SPEAKER_01", 3.21, 4.44, "ask not."),
                (
                    "SPEAKER_02",
                    4.96,
                    12.88,
                    "今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去伊加新"
                ),
            ]
        );
    }

    /// Synthetic (not a real decode) multi-chunk-duration transcript: every
    /// anchor above sits well inside `executor.rs`'s first 30s encoder chunk
    /// (`CHUNK_SAMPLES`), so `snapshot_jfk_and_en_zh_mixed_golden_speaker_segments`
    /// never exercises `speaker_segments` against text spanning more than one
    /// chunk's worth of audio duration. This transcript's anchors straddle
    /// two `CHUNK_SAMPLES` boundaries (30s and 60s) across three speaker
    /// turns and a language switch, covering the shape a real multi-chunk
    /// longform decode would produce -- text parsing itself is chunk-count-
    /// agnostic (it runs once over the final concatenated decode, same as
    /// for a single-chunk clip), so this is a scale/structure regression
    /// check on the parser, not a claim that this exact text was ever
    /// decoded from real audio.
    const SYNTHETIC_MULTI_CHUNK_TEXT: &str = concat!(
        "[0.50][S01] Good morning everyone, let's get started.[29.80][31.20][S01] ",
        "First, a quick recap of last week's numbers.[58.90][61.40][S02] 谢谢，我来补充一下财务方面的情况。",
        "[92.15][93.00][S01] Great, let's move to questions then.[110.75]",
    );

    #[test]
    fn synthetic_multi_chunk_duration_transcript_parses_into_structured_segments() {
        let segments = parse_moss_td_speaker_segments(
            SYNTHETIC_MULTI_CHUNK_TEXT,
            MossTdDecodeExtent::complete(110.75),
        )
        .expect("synthetic multi-chunk transcript parses");
        let snapshot: Vec<(&str, f32, f32, &str)> = segments
            .iter()
            .map(|segment| {
                (
                    segment.speaker.as_deref().unwrap_or(""),
                    segment.start,
                    segment.end,
                    segment.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            snapshot,
            vec![
                (
                    "SPEAKER_01",
                    0.50,
                    29.80,
                    "Good morning everyone, let's get started."
                ),
                (
                    "SPEAKER_01",
                    31.20,
                    58.90,
                    "First, a quick recap of last week's numbers."
                ),
                (
                    "SPEAKER_02",
                    61.40,
                    92.15,
                    "谢谢，我来补充一下财务方面的情况。"
                ),
                (
                    "SPEAKER_01",
                    93.00,
                    110.75,
                    "Great, let's move to questions then."
                ),
            ]
        );
    }

    /// Time anchors are floating-point-derived (see
    /// `assert_segments_match_golden_within_anchor_tolerance`'s doc
    /// comment for why exact cross-backend anchor equality is not the
    /// right bar); 0.03s covers the largest measured CPU-vs-accelerated
    /// anchor divergence on these clips (0.02s on `en_zh_mixed.wav`,
    /// direction-flipped relative to the CPU golden -- see below) with a
    /// small margin, while still catching anything structurally different
    /// (a wrong anchor would fail the strict skeleton check first anyway).
    const ACCELERATED_ANCHOR_TOLERANCE_SECS: f32 = 0.03;

    // Explicit `execution_target=accelerated` e2e smoke: an explicit
    // `Accelerated` request installs the same thread-local override
    // `graph_config.rs` documents as always winning over this family's
    // AllBackends Auto / explicit accelerated, so the encoder graph builds on Metal instead
    // of being downgraded to CPU (the gate only ever pins what *Auto*
    // resolves to -- see `encoder_graph_config_honors_explicit_accelerated_
    // request` in `graph_config.rs`). Decode already runs on Metal under
    // Auto today (the shared qwen decode path is `AllBackends`, and #180
    // fixed its reuse-path graph so Metal decode reuses its graph), so this
    // is the full accelerated-request path: Metal encoder + Metal decode,
    // diffed against the same CPU golden the two tests above pin, via
    // `assert_segments_match_golden_within_anchor_tolerance` (strict on
    // segment count, text/punctuation and speaker labels, tolerant only on
    // each segment's start/end time).
    //
    // jfk.wav: byte-for-byte identical to the CPU golden, anchors included
    // (diff = 0.0 on every anchor).
    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; drives an explicit accelerated request \
                (Metal encoder + Metal decode) and needs a Metal device"]
    fn golden_diff_end_to_end_transcribe_jfk_wav_accelerated() {
        let Some((_, segments, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_segments_match_golden_within_anchor_tolerance(
            &segments,
            GOLDEN_JFK_TEXT,
            10.59,
            ACCELERATED_ANCHOR_TOLERANCE_SECS,
        );
    }

    // MEASURED ANCHOR DIVERGENCE (within tolerance, not a defect): unlike
    // jfk.wav above, this clip's accelerated (Metal encoder + Metal decode)
    // transcript is not byte-identical to the CPU golden. Measured output:
    //
    //   "...Americans,[2.34][3.21][S01]ask not....[4.44][4.94][S02]..."
    //
    // vs. `GOLDEN_EN_ZH_MIXED_TEXT`:
    //
    //   "...Americans,[2.32][3.21][S01]ask not....[4.44][4.96][S02]..."
    //
    // The only differing characters are two digits inside two numeric
    // time-anchor tokens ([2.34] vs [2.32], [4.94] vs [4.96], both a 0.02s
    // shift) -- every word, punctuation mark, speaker label, and the other
    // two anchors are identical, so the strict skeleton layer of
    // `assert_segments_match_golden_within_anchor_tolerance` passes and
    // only the anchor-tolerance layer is exercised here. Notably,
    // [2.34]/[4.94] are the same values the top-of-file
    // `golden_diff_end_to_end_transcribe_en_zh_mixed_wav` comment records
    // for the *HF fp32 reference* (before its own documented 0.02s CPU
    // f16+flash shift to [2.32]/[4.96]) -- i.e. the accelerated path's
    // anchors land on the fp32 reference's values, not the CPU-forced
    // golden's. Both are plausible fp32 outcomes of a numerically delicate
    // computation (see `ACCELERATED_ANCHOR_TOLERANCE_SECS`'s doc comment
    // and the firered-aed parity precedent it cites) -- neither is "the
    // bug".
    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; drives an explicit accelerated request \
                (Metal encoder + Metal decode) and needs a Metal device"]
    fn golden_diff_end_to_end_transcribe_en_zh_mixed_wav_accelerated() {
        let Some((_, segments, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("en_zh_mixed.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_segments_match_golden_within_anchor_tolerance(
            &segments,
            GOLDEN_EN_ZH_MIXED_TEXT,
            12.88,
            ACCELERATED_ANCHOR_TOLERANCE_SECS,
        );
    }

    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack and the 3-minute AISHELL-4 fixture; drives an explicit accelerated request"]
    fn accelerated_aishell4_three_minute_smoke_completes_with_structured_transcript() {
        let Some((text, segments, _, _)) = transcribe_with_dev_pack_backend(
            dev_sample_path("aishell4_multispeaker_3min.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        assert!(
            !text.trim().is_empty(),
            "accelerated AISHELL-4 decode must emit a non-empty transcript"
        );
        let Ok(golden_root) = crate::testing::external_test_fixture_path(
            "OPENASR_MOSS_TRANSCRIBE_DIARIZE_GOLDEN",
            "MOSS Transcribe Diarize development golden directory",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}")) else {
            return;
        };
        let golden_path = golden_root.join("aishell4_multispeaker_3min.json");
        if !golden_path.exists() {
            eprintln!("skipping: {} not present", golden_path.display());
            return;
        }
        let golden: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&golden_path).expect("read AISHELL-4 development golden"),
        )
        .expect("parse AISHELL-4 development golden");
        // The pinned reference text is the raw tagged decode; what a caller
        // gets is its markup-free projection (see `speaker_segments`).
        assert_eq!(
            text,
            normalized_golden_text(
                golden["text"].as_str().expect("AISHELL-4 golden text"),
                180.0
            ),
            "accelerated AISHELL-4 transcript must match the pinned reference text"
        );
        assert!(
            !segments.is_empty(),
            "AISHELL-4 must emit structured segments"
        );
        assert!(
            segments.iter().all(|segment| {
                segment.speaker.is_some()
                    && segment.start.is_finite()
                    && segment.end.is_finite()
                    && segment.start <= segment.end
                    && !segment.text.trim().is_empty()
            }),
            "AISHELL-4 segments must retain speaker labels and valid time ranges"
        );
        assert!(
            segments
                .windows(2)
                .all(|pair| pair[0].start <= pair[1].start),
            "AISHELL-4 segment starts must be ordered"
        );
        assert!(
            !text.contains("[S"),
            "the family's speaker markup must never reach the caller"
        );
    }
}
