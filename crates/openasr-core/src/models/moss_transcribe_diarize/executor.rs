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

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::NativeAsrError;
use crate::api::backend::Transcription;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::qwen::{Qwen3AsrLayerKvCacheState, Qwen3AsrPromptEmbeddings};
use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, canonical_runtime_cache_path,
    with_thread_local_cached_mut_by_key,
};
use crate::models::whisper::whisper_log_mel_spectrogram_16khz_mono_v0;

use super::adaptor_graph::{load_moss_adaptor_weights_from_reader, run_moss_adaptor};
use super::decode_prompt::build_moss_td_decode_prompt;
use super::encoder_graph::{MossEncoderConfig, MossEncoderRuntime};
use super::graph_config::{moss_td_encoder_graph_config, moss_td_runtime_graph_config};
use super::llm_decoder::MossTdDecoderRuntime;
use super::prompt_embedding::build_moss_td_prompt_embeddings_with_audio_splice;
use super::runtime_contract::{
    MOSS_TD_ADAPTOR_NORM_EPSILON, MossTdDecoderMetadata, moss_td_kv_cache_positions,
    parse_adaptor_metadata, parse_decoder_metadata, parse_encoder_metadata,
};
use super::tokenizer::MossTdTokenizer;

/// `WhisperFeatureExtractor`'s `chunk_length=30` @ 16kHz (`preprocessor_config.json`,
/// verified against the real checkpoint).
const CHUNK_SAMPLES: usize = 480_000;
const MEL_TARGET_FRAMES: usize = 3000;
const SAMPLE_RATE_HZ: usize = 16_000;
/// `WhisperFeatureExtractor.hop_length` (160) * the Whisper conv stem's 2x
/// stride * `audio_merge_size` -- upstream's
/// `_compute_audio_token_length`'s `stride` (`processing_moss_transcribe_diarize.py`).
const WHISPER_ENCODER_CONV_STRIDE: usize = 2;
const HOP_LENGTH: usize = 160;
/// Generous upper bound on generated tokens; greedy decode stops at
/// `<|im_end|>` well before this in practice (the real checkpoint's own
/// reference generation config used this exact cap -- verified against
/// `tmp/moss-td/golden/*.json`'s `max_new_tokens`). Only the fail-closed
/// backstop against a runaway (non-terminating) decode.
const MOSS_TD_MAX_GENERATED_TOKENS: usize = 4096;
/// Audio tokens per second the adaptor emits (`audio_tokens_per_second` in
/// `processing_moss_transcribe_diarize.py`, same value `decode_prompt`'s marker
/// cadence uses). Only used to render the `AudioExceedsContext` limit as an
/// approximate minutes figure; not part of any decode math.
const AUDIO_TOKENS_PER_SECOND_FOR_LIMIT: f32 = 12.5;

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
        "moss-transcribe-diarize audio is too long: its {prompt_tokens}-token audio prompt \
         needs at least one free position within the {kv_capacity}-position decoder context \
         (about {max_minutes:.0} min of audio); split the input into shorter files"
    )]
    AudioExceedsContext {
        prompt_tokens: usize,
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
    #[error("moss-transcribe-diarize prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct MossTdGgmlExecutor;

const MOSS_TD_EXECUTOR_ID: &str = "moss-transcribe-diarize-ggml-executor-v1";
const MOSS_TD_STREAMING_EXECUTOR_ID: &str =
    "moss-transcribe-diarize-ggml-snapshot-streaming-executor-v1";

struct MossTdGreedyStepExecutor<'a> {
    decoder: &'a mut MossTdDecoderRuntime,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    prompt_embeddings: Option<Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
}

impl Seq2SeqGreedyDecodeStepExecutor for MossTdGreedyStepExecutor<'_> {
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

/// Thread-local resident cache for the family's two heavy per-pack runtimes
/// (the Whisper-Medium-style [`MossEncoderRuntime`] and the Qwen3
/// [`MossTdDecoderRuntime`]), keyed by `(canonical pack path, resolved
/// backend)`. Mirrors `firered_aed::executor`'s
/// `FIRERED_AED_ENCODER_RUNTIME_BY_KEY`/`FIRERED_AED_DECODER_RUNTIME_BY_KEY`
/// exactly -- same shared `BoundedRuntimeCache` + `with_thread_local_cached_mut_by_key`
/// machinery, same key shape, same lazy idle-unload-generation eviction. Before
/// this, every `execute()` rebuilt both runtimes from scratch: re-mmapped the
/// pack, re-read every encoder tensor off disk, and re-uploaded every decoder
/// layer's weights, on every single call (including every chunk of the same
/// longform request).
type MossTdEncoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);
type MossTdDecoderRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static MOSS_TD_ENCODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<MossTdEncoderRuntimeCacheKey, MossEncoderRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
    static MOSS_TD_DECODER_RUNTIME_BY_KEY: RefCell<BoundedRuntimeCache<MossTdDecoderRuntimeCacheKey, MossTdDecoderRuntime>> =
        RefCell::new(BoundedRuntimeCache::new());
}

// Test-only build counters, incremented from inside the two caches' `build`
// closures below -- lets a same-thread test pin "a second call reuses the
// cached runtime" as a structural fact (build count stays 1 across two
// calls) rather than inferring cache-hit behavior from wall-clock timing.
#[cfg(test)]
thread_local! {
    static MOSS_TD_ENCODER_RUNTIME_BUILD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static MOSS_TD_DECODER_RUNTIME_BUILD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_moss_td_runtime_build_counts_for_test() {
    MOSS_TD_ENCODER_RUNTIME_BUILD_COUNT.with(|count| count.set(0));
    MOSS_TD_DECODER_RUNTIME_BUILD_COUNT.with(|count| count.set(0));
}

/// `(encoder builds, decoder builds)` recorded on the calling thread since
/// the last [`reset_moss_td_runtime_build_counts_for_test`].
#[cfg(test)]
fn moss_td_runtime_build_counts_for_test() -> (usize, usize) {
    (
        MOSS_TD_ENCODER_RUNTIME_BUILD_COUNT.with(std::cell::Cell::get),
        MOSS_TD_DECODER_RUNTIME_BUILD_COUNT.with(std::cell::Cell::get),
    )
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
fn moss_td_chunk_token_length(chunk_samples: usize, token_stride: usize) -> usize {
    (chunk_samples - 1) / token_stride.max(1) + 1
}

/// This chunk's post-merge encoder frames actually kept: `token_length` audio
/// tokens each span `merge_size` pre-merge encoder frames, capped at the
/// encoder's `max_source_positions` (a full un-trimmed 30s chunk can never
/// legitimately need more than that many frames kept).
fn moss_td_chunk_keep_frames(
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
fn moss_td_aligned_frame_count(total_frames: usize, merge_size: usize) -> usize {
    let merge_size = merge_size.max(1);
    (total_frames / merge_size) * merge_size
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
}

/// Encodes every 30s chunk of `samples` against the cached, resident encoder
/// runtime for this pack+backend, returning the concatenated (already
/// merge-size-aligned) encoder rows and the number of kept frames -- the same
/// computation the executor's per-chunk loop always did, just routed through
/// the shared resident-runtime cache instead of building a fresh
/// [`MossEncoderRuntime`] (and re-reading every encoder tensor from disk) on
/// every call.
fn encode_moss_td_chunks_with_cached_runtime(
    runtime_path: &Path,
    encoder_config: MossEncoderConfig,
    merge_size: usize,
    samples: &[f32],
) -> Result<(Vec<f32>, usize), MossTdExecutorError> {
    let key = (
        canonical_runtime_cache_path(runtime_path),
        moss_td_encoder_graph_config().backend,
    );
    // Upstream `_compute_audio_token_length`'s stride: hop_length * the
    // Whisper conv stem's 2x stride * audio_merge_size.
    let token_stride = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * merge_size;
    with_thread_local_cached_mut_by_key(
        &MOSS_TD_ENCODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            #[cfg(test)]
            MOSS_TD_ENCODER_RUNTIME_BUILD_COUNT.with(|count| count.set(count.get() + 1));
            MossEncoderRuntime::new(runtime_path, encoder_config).map_err(|error| {
                MossTdExecutorError::EncoderFailed {
                    reason: format!("could not initialize encoder runtime: {error}"),
                }
            })
        },
        |runtime| {
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
                let encoder_out = runtime
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
        },
    )
}

/// Runs the ChatML+audio-splice prompt embedding through the cached, resident
/// decoder runtime for this pack+backend: prefill, then the shared greedy
/// decode driver through to `<|im_end|>` (or the fail-closed token budget),
/// returning the trimmed decode text. Mirrors `firered_aed::executor`'s
/// `decode_with_cached_runtime`: the runtime (loaded weights + the Qwen
/// decode graph's reuse machinery) stays resident across calls, while every
/// per-utterance KV cache is still allocated fresh right here
/// (`MossTdDecoderRuntime::new_kv_caches`) -- unlike firered-aed's decoder,
/// this family's `MossTdDecoderRuntime` carries no cross-request KV state of
/// its own between calls, so no cache-reset step is needed before reuse.
fn run_moss_td_decoder_with_cached_runtime(
    runtime_path: &Path,
    decoder_metadata: MossTdDecoderMetadata,
    decode_prompt_token_ids: &[u32],
    audio_pad_positions: &[usize],
    audio_rows: &[f32],
    tokenizer: &MossTdTokenizer,
) -> Result<String, MossTdExecutorError> {
    let key = (
        canonical_runtime_cache_path(runtime_path),
        moss_td_runtime_graph_config().backend,
    );
    with_thread_local_cached_mut_by_key(
        &MOSS_TD_DECODER_RUNTIME_BY_KEY,
        key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            #[cfg(test)]
            MOSS_TD_DECODER_RUNTIME_BUILD_COUNT.with(|count| count.set(count.get() + 1));
            MossTdDecoderRuntime::new(runtime_path, decoder_metadata).map_err(|error| {
                MossTdExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })
        },
        |decoder| {
            if std::env::var_os("OPENASR_MOSS_TD_PROFILE").is_some() {
                eprintln!(
                    "OPENASR_MOSS_TD_PROFILE decoder_backend={}",
                    decoder.backend_label()
                );
            }

            let token_rows_len = decode_prompt_token_ids.len() * decoder_metadata.d_model;
            let mut token_rows = Vec::with_capacity(token_rows_len);
            for &token_id in decode_prompt_token_ids {
                let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                    MossTdExecutorError::DecoderFailed {
                        reason: error.to_string(),
                    }
                })?;
                token_rows.extend_from_slice(&row);
            }
            let spliced = build_moss_td_prompt_embeddings_with_audio_splice(
                decode_prompt_token_ids.len(),
                audio_pad_positions,
                decoder_metadata.d_model,
                &token_rows,
                audio_rows,
            )
            .map_err(|error| MossTdExecutorError::PromptEmbeddingFailed {
                reason: error.to_string(),
            })?;
            let prompt_embeddings = Qwen3AsrPromptEmbeddings {
                hidden_size: spliced.hidden_size,
                token_count: spliced.token_count,
                token_major_values: spliced.token_major_values,
            };

            // Request-sized, not the checkpoint's native 131072-token RoPE
            // context: see `MossTdDecoderRuntime::new_kv_caches`'s doc comment
            // for why the fixed reuse-graph span this sizes must stay tight
            // to what this utterance actually needs.
            let layer_kv_caches = decoder.new_kv_caches(
                spliced
                    .token_count
                    .saturating_add(MOSS_TD_MAX_GENERATED_TOKENS),
            );
            let mut step_executor = MossTdGreedyStepExecutor {
                decoder,
                layer_kv_caches,
                prompt_embeddings: Some(prompt_embeddings),
                cache_prompt_tokens: 0,
            };
            let config = BuiltinSeq2SeqDecodePolicyConfigInput {
                initial_prompt_tokens: decode_prompt_token_ids.to_vec(),
                eot_token_id: tokenizer.im_end_token_id,
                vocab_size: decoder_metadata.vocab_size,
                max_generated_tokens: MOSS_TD_MAX_GENERATED_TOKENS,
            };
            let result = run_builtin_seq2seq_decode_policy(
                crate::arch::MOSS_TD_DECODE_POLICY_ID,
                &config,
                tokenizer,
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
            // Release this request's per-token grow-to-fit host buffer before
            // the runtime goes back into the cache (mirrors qwen3-asr's
            // `ggml_executor`'s `release_session_scoped_buffers` call around
            // its own resident whole-decoder cache) -- unconditionally, on
            // both the success and failure paths, so a failed decode never
            // leaves a session-scoped allocation riding along on the cached
            // runtime.
            step_executor.decoder.release_session_scoped_buffers();
            let result = result.map_err(|error| MossTdExecutorError::GreedyDecodeFailed {
                reason: error.to_string(),
            })?;
            Ok(result.text.trim().to_string())
        },
    )
}

impl MossTdGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
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

        let encoder_metadata = parse_encoder_metadata(&*preflight.metadata).map_err(|error| {
            MossTdExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let adaptor_metadata = parse_adaptor_metadata(&*preflight.metadata).map_err(|error| {
            MossTdExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let decoder_metadata = parse_decoder_metadata(&*preflight.metadata).map_err(|error| {
            MossTdExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let tokenizer = MossTdTokenizer::from_gguf_metadata(&preflight.metadata).map_err(
            |error: NativeAsrError| MossTdExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            },
        )?;

        let samples = &request.prepared_audio.samples_f32;
        if samples.is_empty() {
            return Err(MossTdExecutorError::EmptyAudio);
        }
        let audio_duration_seconds = samples.len() as f32 / SAMPLE_RATE_HZ as f32;

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
            MossTdExecutorError::RuntimeContractViolation {
                reason: error.to_string(),
            }
        })?;
        let encoder_config = MossEncoderConfig {
            n_layers: encoder_metadata.n_layers,
            d_model: encoder_metadata.d_model,
            n_heads: encoder_metadata.n_heads,
            n_mels: encoder_metadata.n_mels,
            max_source_positions: encoder_metadata.max_source_positions,
        };
        let adaptor_weights = load_moss_adaptor_weights_from_reader(
            &reader,
            encoder_metadata.d_model,
            adaptor_metadata.merge_size,
            decoder_metadata.d_model,
            MOSS_TD_ADAPTOR_NORM_EPSILON,
        )
        .map_err(|error| MossTdExecutorError::AdaptorFailed {
            reason: error.to_string(),
        })?;

        // Routed through the resident, thread-local encoder-runtime cache
        // (mirrors `firered_aed::executor`'s cached encoder): the loaded
        // weights + mmap'd zero-copy context stay resident across calls to
        // this pack+backend instead of being rebuilt from scratch on every
        // `execute()`.
        let (mut concatenated_rows, total_frames) = encode_moss_td_chunks_with_cached_runtime(
            preflight.runtime_source.path(),
            encoder_config,
            adaptor_metadata.merge_size,
            samples,
        )?;
        let aligned_frames = moss_td_aligned_frame_count(total_frames, adaptor_metadata.merge_size);
        concatenated_rows.truncate(aligned_frames * encoder_metadata.d_model);

        let (audio_rows, audio_token_count) = run_moss_adaptor(
            &adaptor_weights,
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

        // Fail closed up front when the whole-audio prompt cannot fit the
        // decoder's KV context. This family ingests the full audio in one
        // decode (native longform slicing is disabled for it, see the
        // decode-policy `SelfChunkingExecutorV1`), so a very long file grows
        // the prompt until it exceeds the KV-cache capacity. `kv_capacity`
        // positions (~one every 12.5 audio tokens/sec plus the fixed template
        // and generated tokens) works out to roughly 7-10 minutes of audio;
        // beyond that, fail with a clear message instead of a cryptic mid-
        // decode KV-bounds error (or worse, silent truncation).
        let kv_capacity = moss_td_kv_cache_positions(decoder_metadata.max_positions);
        if decode_prompt.token_ids.len() >= kv_capacity {
            let max_minutes =
                (kv_capacity as f32 / AUDIO_TOKENS_PER_SECOND_FOR_LIMIT / 60.0).max(0.0);
            return Err(MossTdExecutorError::AudioExceedsContext {
                prompt_tokens: decode_prompt.token_ids.len(),
                kv_capacity,
                max_minutes,
            });
        }

        // Routed through the resident, thread-local decoder-runtime cache
        // (mirrors `firered_aed::executor`'s cached decoder): the loaded
        // decoder weights + reuse-graph machinery stay resident across calls
        // to this pack+backend, while the KV cache for this one utterance is
        // still allocated fresh inside the helper.
        let runtime_path = preflight.runtime_source.path();
        let text = run_moss_td_decoder_with_cached_runtime(
            runtime_path,
            decoder_metadata,
            &decode_prompt.token_ids,
            &decode_prompt.audio_pad_positions,
            &audio_rows,
            &tokenizer,
        )?;
        // Parse the model's own inline `[start][end][SNN]` markup into real
        // speaker segments, degrading fail-closed to the single speaker-less
        // segment carrying the untouched raw text when the tag stream is
        // malformed or empty (see `speaker_segments`'s module doc for the
        // grammar, the fail-closed policy, and the degrade shape's tests).
        // `text` itself is never rewritten either way.
        let segments =
            super::speaker_segments::moss_td_segments_or_degrade(&text, audio_duration_seconds);
        let transcription = Transcription {
            segments,
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

impl GgmlAsrExecutor for MossTdGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MOSS_TD_EXECUTOR_ID
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
            MossTdGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::ggml_runtime::install_request_backend_override;
    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudio};
    use crate::models::ggml_family_registry::moss_transcribe_diarize_runtime_descriptor_v1;

    use crate::api::backend::Segment;

    use super::super::speaker_segments::parse_moss_td_speaker_segments;
    use super::*;

    /// Real converted dev pack (fp16), NOT committed -- same dev-only-artifact
    /// convention as `decode_prompt`'s own `dev_pack_path` and mimo-asr's
    /// `mimo-v2.5-asr-q8_0.oasr`.
    fn dev_pack_path() -> PathBuf {
        PathBuf::from(
            "/Volumes/QuintinDocument/openasr-dev/tmp/moss-td/moss-transcribe-diarize-fp16.oasr",
        )
    }

    fn dev_sample_path(name: &str) -> PathBuf {
        PathBuf::from("/Volumes/QuintinDocument/openasr-dev/tmp/moss-td/samples").join(name)
    }

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

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<(String, std::time::Duration, f32)> {
        // Force CPU. This family's Metal path has two open defects (encoder
        // numeric divergence -> empty-shell output, and a per-step wired-memory
        // blow-up -- see the `arch` descriptor's `auto_gpu_policy` note), so the
        // reference decode is CPU-only.
        transcribe_with_dev_pack_backend(wav_path, GgmlAsrBackendPreference::CpuOnly)
    }

    /// Same dev-pack e2e path as [`transcribe_with_dev_pack`], but lets the
    /// caller pick the backend preference -- used by the `_accelerated`
    /// variants below to drive an explicit `execution_target=accelerated`
    /// request end to end (encoder AND decode), the same override an
    /// `Accelerated` request installs in production (see
    /// `GgmlAsrBackendPreference::request_backend_override`'s doc and
    /// `graph_config.rs`'s note that an explicit request always wins over
    /// the family's `ExceptMetal` Auto gate).
    fn transcribe_with_dev_pack_backend(
        wav_path: PathBuf,
        backend_preference: GgmlAsrBackendPreference,
    ) -> Option<(String, std::time::Duration, f32)> {
        let pack_path = dev_pack_path();
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
        // `GgmlAsrExecutionRequest::backend_preference`'s doc), so install the
        // override explicitly rather than relying on the ambient backend.
        // Hold the RAII guard for the whole decode: it restores the previous
        // thread-local override on drop at the end of this function.
        let _backend_override_guard =
            install_request_backend_override(backend_preference.request_backend_override());

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: moss_transcribe_diarize_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference,
        };

        let executor = MossTdGgmlExecutor;
        let started_at = Instant::now();
        let result = executor.execute(&request).expect("moss-td transcribe");
        let elapsed = started_at.elapsed();
        Some((result.transcription.text, elapsed, audio_duration_seconds))
    }

    /// Same dev-pack e2e path as [`transcribe_with_dev_pack`], but returns the
    /// full [`Segment`] list instead of only the flat text -- used to check
    /// that the real decode's speaker/time-anchor markup round-trips through
    /// `speaker_segments::parse_moss_td_speaker_segments` (as wired into the
    /// executor) into the same structure the golden `[Sxx]`/`[t]` tags encode.
    fn transcribe_with_dev_pack_segments(wav_path: PathBuf) -> Option<Vec<Segment>> {
        let pack_path = dev_pack_path();
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
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "moss-td e2e test",
            "moss-td e2e test",
        )
        .expect("load wav fixture");
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: moss_transcribe_diarize_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };
        let executor = MossTdGgmlExecutor;
        let result = executor.execute(&request).expect("moss-td transcribe");
        Some(result.transcription.segments)
    }

    /// Splits a moss-td transcript into (a) its "skeleton" -- every literal
    /// character with each numeric time-anchor token's digits blanked out to
    /// `[]` (leaving non-numeric bracketed tokens like `[S01]` untouched) --
    /// and (b) the anchors' parsed float values in order. Used by
    /// [`assert_transcript_matches_golden_within_anchor_tolerance`] to split
    /// "does the text/structure match" from "do the anchors match" into two
    /// independently-checked layers.
    fn parse_transcript_skeleton_and_anchors(text: &str) -> (String, Vec<f32>) {
        let mut skeleton = String::with_capacity(text.len());
        let mut anchors = Vec::new();
        let mut rest = text;
        while let Some(open_rel) = rest.find('[') {
            skeleton.push_str(&rest[..open_rel]);
            let after_open = &rest[open_rel + 1..];
            let Some(close_rel) = after_open.find(']') else {
                // Unterminated '[': copy the rest verbatim and stop.
                skeleton.push_str(&rest[open_rel..]);
                rest = "";
                break;
            };
            let inner = &after_open[..close_rel];
            if let Ok(value) = inner.trim().parse::<f32>() {
                anchors.push(value);
                skeleton.push_str("[]");
            } else {
                skeleton.push('[');
                skeleton.push_str(inner);
                skeleton.push(']');
            }
            rest = &after_open[close_rel + 1..];
        }
        skeleton.push_str(rest);
        (skeleton, anchors)
    }

    /// Two-layer transcript comparison for the accelerated e2e smoke tests:
    /// (1) text, punctuation, speaker labels, and anchor count/order must
    /// match the CPU golden byte-for-byte (asserted via the anchor-blanked
    /// "skeleton"); (2) each numeric time-anchor's value only needs to be
    /// within `tolerance_secs` of the golden's, not bit-identical.
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
    fn assert_transcript_matches_golden_within_anchor_tolerance(
        actual: &str,
        golden: &str,
        tolerance_secs: f32,
    ) {
        let (actual_skeleton, actual_anchors) = parse_transcript_skeleton_and_anchors(actual);
        let (golden_skeleton, golden_anchors) = parse_transcript_skeleton_and_anchors(golden);
        assert_eq!(
            actual_skeleton, golden_skeleton,
            "transcript text/punctuation/speaker-labels/anchor-count-and-order diverged from \
             the CPU golden (strict layer -- anchor *values* are compared separately with \
             tolerance, this only checks everything else)"
        );
        assert_eq!(
            actual_anchors.len(),
            golden_anchors.len(),
            "anchor count mismatch (should already have failed the skeleton check above)"
        );
        for (idx, (actual_anchor, golden_anchor)) in
            actual_anchors.iter().zip(golden_anchors.iter()).enumerate()
        {
            let diff = (actual_anchor - golden_anchor).abs();
            assert!(
                diff <= tolerance_secs,
                "anchor[{idx}] exceeds tolerance: actual={actual_anchor} golden={golden_anchor} \
                 diff={diff:.4}s (tolerance={tolerance_secs}s)"
            );
        }
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
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    /// Pins the resident-runtime cache's two contracts introduced by this
    /// PR: (1) a second `execute()` on the same thread against the same pack
    /// reuses both the cached encoder and decoder runtimes rather than
    /// rebuilding them (asserted structurally via the build counters, not by
    /// timing -- see `moss_td_runtime_build_counts_for_test`'s doc comment),
    /// and (2) reuse changes nothing observable: the second call's transcript
    /// is byte-for-byte identical to the first (and to `GOLDEN_JFK_TEXT`).
    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; CPU-only (Metal path has known defects)"]
    fn resident_runtime_cache_hits_on_a_second_transcribe_call_for_the_same_pack() {
        reset_moss_td_runtime_build_counts_for_test();
        let Some((first_text, _, _)) = transcribe_with_dev_pack(dev_sample_path("jfk.wav")) else {
            return;
        };
        assert_eq!(first_text, GOLDEN_JFK_TEXT);
        let (encoder_builds, decoder_builds) = moss_td_runtime_build_counts_for_test();
        assert_eq!(
            encoder_builds, 1,
            "first call must build the encoder runtime exactly once"
        );
        assert_eq!(
            decoder_builds, 1,
            "first call must build the decoder runtime exactly once"
        );

        let Some((second_text, _, _)) = transcribe_with_dev_pack(dev_sample_path("jfk.wav")) else {
            return;
        };
        assert_eq!(
            second_text, first_text,
            "reusing the cached runtimes must not change the decode"
        );
        let (encoder_builds, decoder_builds) = moss_td_runtime_build_counts_for_test();
        assert_eq!(
            encoder_builds, 1,
            "second call must hit the cached encoder runtime, not rebuild it"
        );
        assert_eq!(
            decoder_builds, 1,
            "second call must hit the cached decoder runtime, not rebuild it"
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
        assert_eq!(text, GOLDEN_EN_ZH_MIXED_TEXT);
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
        let expected = parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, 10.59)
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
        let expected = parse_moss_td_speaker_segments(GOLDEN_EN_ZH_MIXED_TEXT, 12.88)
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
        let jfk = parse_moss_td_speaker_segments(GOLDEN_JFK_TEXT, 10.59).expect("jfk parses");
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

        let en_zh_mixed =
            parse_moss_td_speaker_segments(GOLDEN_EN_ZH_MIXED_TEXT, 12.88).expect("parses");
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
        let segments = parse_moss_td_speaker_segments(SYNTHETIC_MULTI_CHUNK_TEXT, 110.75)
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
    /// `assert_transcript_matches_golden_within_anchor_tolerance`'s doc
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
    // `ExceptMetal` Auto gate, so the encoder graph builds on Metal instead
    // of being downgraded to CPU (the gate only ever pins what *Auto*
    // resolves to -- see `encoder_graph_config_honors_explicit_accelerated_
    // request` in `graph_config.rs`). Decode already runs on Metal under
    // Auto today (the shared qwen decode path is `AllBackends`, and #180
    // fixed its reuse-path graph so Metal decode reuses its graph), so this
    // is the full accelerated-request path: Metal encoder + Metal decode,
    // diffed against the same CPU golden the two tests above pin, via
    // `assert_transcript_matches_golden_within_anchor_tolerance` (strict on
    // text/punctuation/speaker-labels/anchor-count-and-order, tolerant only
    // on each anchor's numeric value).
    //
    // jfk.wav: byte-for-byte identical to the CPU golden, anchors included
    // (diff = 0.0 on every anchor).
    #[test]
    #[ignore = "requires the private dev-only moss-transcribe-diarize-fp16.oasr pack \
                and tmp/moss-td/samples/*.wav; drives an explicit accelerated request \
                (Metal encoder + Metal decode) and needs a Metal device"]
    fn golden_diff_end_to_end_transcribe_jfk_wav_accelerated() {
        let Some((text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("jfk.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_transcript_matches_golden_within_anchor_tolerance(
            &text,
            GOLDEN_JFK_TEXT,
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
    // `assert_transcript_matches_golden_within_anchor_tolerance` passes and
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
        let Some((text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack_backend(
            dev_sample_path("en_zh_mixed.wav"),
            GgmlAsrBackendPreference::Accelerated,
        ) else {
            return;
        };
        eprintln!(
            "moss-td e2e accelerated [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_transcript_matches_golden_within_anchor_tolerance(
            &text,
            GOLDEN_EN_ZH_MIXED_TEXT,
            ACCELERATED_ANCHOR_TOLERANCE_SECS,
        );
    }
}
