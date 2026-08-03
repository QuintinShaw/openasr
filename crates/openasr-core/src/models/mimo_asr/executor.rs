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

use std::cell::RefCell;
use std::collections::HashMap;

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::MIMO_ASR_DECODE_POLICY_ID;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest, GgmlAsrViewExecutor,
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
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeResult, Seq2SeqGreedyDecodeStepExecutor,
    Seq2SeqGreedyDecodeStepInput, Seq2SeqGreedyDecodeStepLogitsOutput,
};
use crate::models::thread_local_runtime_cache::{
    PackContentKey, current_unload_generation, take_generation_tagged,
};

use super::audio_tokenizer_graph::MimoAudiotokEncoderRuntime;
use super::decode_prompt::build_mimo_asr_decode_prompt;
use super::input_local_graph::{
    MimoInputLocalRuntime, MimoSpeechEmbeddingTables, load_speech_embedding_tables_from_reader,
    sum_speech_embeddings,
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

const MIMO_ASR_EXECUTOR_ID: &str = "mimo-asr-ggml-executor-v1";
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
    #[error("mimo-asr executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
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
    #[error("mimo-asr backbone decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("mimo-asr greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct MimoAsrGgmlExecutor;

/// Everything mimo-asr materializes from a pack that does NOT depend on the
/// per-request audio: all three graph runtimes (audio-tokenizer encoder,
/// input-local transformer, 36L Qwen2 backbone decoder), their
/// device-uploaded / arena-resident weights, plus the immutable derived data
/// read once from the pack (RVQ codebooks, the 8 speech-embedding tables, the
/// mel front-end plan, the tokenizer, and the two metadata groups the
/// per-request path still consults). Before this cache, mimo-asr was the only
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
    codebooks: MimoRvqCodebooks,
    speech_embedding_tables: MimoSpeechEmbeddingTables,
    mel_plan: MimoMelFrontendPlan,
    llm_metadata: MimoLlmMetadata,
    inlocal_metadata: MimoInlocalMetadata,
}

/// Resident prepared-runtime cache keyed by (pack content id, backend), the
/// same design + idle-unload-generation tagging `firered_llm` uses for its
/// resident decoder (see that module's `FIRERED_LLM_DECODER_BY_KEY` doc). The
/// pack half is a [`PackContentKey`] from the request's already-open source,
/// so an in-place `.oasr` replacement at the same path resolves a different id
/// and the next lookup rebuilds instead of reusing runtimes built from the old
/// bytes. Entries are tagged with the idle-unload generation they were built
/// under: `take_generation_tagged` discards any prepared runtime built before
/// the last idle unload (the reaper cannot reach this worker thread's TLS from
/// its own thread), so the resident ~8B pipeline stays evictable under memory
/// pressure through the shared central unload clock -- no bespoke policy.
///
/// A plain `HashMap` (not the bounded LRU): the key does not explode per audio
/// chunk (one entry is built and reused across a whole longform run for a given
/// pack/backend), so there is no unbounded-growth hazard to bound.
type MimoAsrPreparedRuntimeCacheKey = (PackContentKey, GgmlCpuGraphBackend);

thread_local! {
    static MIMO_ASR_PREPARED_BY_KEY: RefCell<
        HashMap<MimoAsrPreparedRuntimeCacheKey, (u64, MimoAsrPreparedRuntime)>,
    > = RefCell::new(HashMap::new());
}

fn take_cached_prepared_runtime(
    key: &MimoAsrPreparedRuntimeCacheKey,
    unload_generation: u64,
) -> Option<MimoAsrPreparedRuntime> {
    MIMO_ASR_PREPARED_BY_KEY
        .with(|cache| take_generation_tagged(&mut cache.borrow_mut(), key, unload_generation))
}

fn store_cached_prepared_runtime(
    key: MimoAsrPreparedRuntimeCacheKey,
    unload_generation: u64,
    prepared: MimoAsrPreparedRuntime,
) {
    MIMO_ASR_PREPARED_BY_KEY.with(|cache| {
        cache
            .borrow_mut()
            .insert(key, (unload_generation, prepared));
    });
}

/// No-op phrase-bias shim: mimo-asr's decode policy never consults these (no
/// phrase bias, single config-supplied eot token) -- mirrors
/// `firered_llm::executor::NoPhraseBiasTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

/// Drives `MimoLlmDecoderRuntime` through the shared greedy loop: step 0
/// consumes the pre-built (audio-spliced) prompt embeddings via one prefill
/// pass; every later step embeds the last generated token and decodes
/// incrementally. Mirrors `firered_llm::executor::FireRedLlmGreedyStepExecutor`.
struct MimoAsrGreedyStepExecutor<'a> {
    decoder: &'a mut MimoLlmDecoderRuntime,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    prompt_embeddings: Option<crate::models::qwen::Qwen3AsrPromptEmbeddings>,
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
        if let Some(prompt_embeddings) = self.prompt_embeddings.take() {
            self.cache_prompt_tokens = prompt_embeddings.token_count;
            let prefill = self
                .decoder
                .prefill(&prompt_embeddings, &mut self.layer_kv_caches, &self.control)
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
            .decode_step_reused_top1(last_token, cache_position, &self.layer_kv_caches)
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

impl MimoAsrPreparedRuntime {
    /// Materializes every per-request-invariant piece of the mimo-asr pipeline
    /// from an already-resolved preflight: the three graph runtimes and their
    /// resident weights, plus the RVQ codebooks, speech-embedding tables, mel
    /// plan, tokenizer, and the two metadata groups the request path consults.
    /// This is the whole cost the resident cache exists to pay exactly once per
    /// (pack, backend).
    fn build(
        preflight: &crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
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

        let runtime_source = &preflight.runtime_source;
        let encoder_runtime =
            MimoAudiotokEncoderRuntime::new(runtime_source, audiotok_metadata.clone(), backend)
                .map_err(|error| MimoAsrExecutorError::EncoderFailed {
                    reason: error.to_string(),
                })?;

        let codebooks =
            load_mimo_rvq_codebooks_from_reader(&reader, &audiotok_metadata).map_err(|error| {
                MimoAsrExecutorError::RvqFailed {
                    reason: error.to_string(),
                }
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
            .map(|size| size + 1)
            .collect();
        let zeroemb_idx: Vec<u32> = audiotok_metadata.codebook_sizes.clone();
        let speech_embedding_tables = load_speech_embedding_tables_from_reader(
            &reader,
            inlocal_metadata.d_model,
            &speech_vocab_sizes,
            &zeroemb_idx,
        )
        .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
            reason: error.to_string(),
        })?;

        let inlocal_runtime = MimoInputLocalRuntime::new(runtime_source, inlocal_metadata, backend)
            .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
                reason: error.to_string(),
            })?;

        let decoder =
            MimoLlmDecoderRuntime::new(runtime_source, llm_metadata, backend).map_err(|error| {
                MimoAsrExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
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

impl MimoAsrGgmlExecutor {
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
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| MimoAsrExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > MIMO_ASR_MAX_INPUT_SECONDS {
            return Err(MimoAsrExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: MIMO_ASR_MAX_INPUT_SECONDS,
            });
        }

        // Take the resident prepared runtime out of this thread's cache (or
        // build it once on a cold miss). Sampled before the take and reused for
        // the store-back below: if the idle-unload reaper bumps the generation
        // while this decode is in flight, the runtime goes back tagged with the
        // pre-unload generation and the *next* take discards it, so an unload is
        // never lost to an overlapping decode (mirrors `firered_llm`).
        let backend = request.resolved_runtime.backend();
        let cache_key: MimoAsrPreparedRuntimeCacheKey = (
            PackContentKey::for_runtime_source(&preflight.runtime_source),
            backend,
        );
        let unload_generation = current_unload_generation();
        let mut prepared = match take_cached_prepared_runtime(&cache_key, unload_generation) {
            Some(prepared) => prepared,
            None => MimoAsrPreparedRuntime::build(&preflight, backend)?,
        };

        let result = self.transcribe_with_prepared(&mut prepared, request, samples);

        // Return the resident runtime to the cache for the next chunk /
        // execute() regardless of decode outcome (the session-scoped buffer
        // release below already discarded any state a partial compute poisoned;
        // uploaded weights and the arena/graph handles remain valid).
        prepared.decoder.release_session_scoped_buffers();
        store_cached_prepared_runtime(cache_key, unload_generation, prepared);

        let result = result?;
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
        &self,
        prepared: &mut MimoAsrPreparedRuntime,
        request: &GgmlAsrExecutionViewRequest,
        samples: &[f32],
    ) -> Result<Seq2SeqGreedyDecodeResult, MimoAsrExecutorError> {
        // The OpenASR pipeline delivers 16kHz mono to every executor, but
        // MiMo's audio tokenizer (and its baked mel filterbank/window) is
        // trained at 24kHz -- resample up before the mel front-end, matching
        // the reference `preprocess_input`'s own resample-to-tokenizer-rate.
        let input_rate = request.prepared_audio.sample_rate_hz;
        let target_rate = prepared.mel_plan.sample_rate_hz as u32;
        let resampled = resample_mono(samples, input_rate, target_rate).ok_or(
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

        let encoder_output = prepared
            .encoder_runtime
            .encode(&mel_features)
            .map_err(|error| MimoAsrExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;

        let mut codes = encode_rvq_codes(
            &prepared.codebooks,
            &encoder_output.rows,
            encoder_output.frame_count,
        )
        .map_err(|error| MimoAsrExecutorError::RvqFailed {
            reason: error.to_string(),
        })?;

        // Truncate to the nearest group_size multiple (drop up to
        // group_size-1 trailing 25Hz frames = well under 200ms of audio) --
        // the reference asserts exact divisibility rather than padding.
        let group_size = prepared.inlocal_metadata.group_size;
        let usable_frames = (codes.len() / group_size) * group_size;
        codes.truncate(usable_frames);
        if usable_frames == 0 {
            return Err(MimoAsrExecutorError::RvqFailed {
                reason: format!(
                    "audio too short: {} RVQ frames produced, need at least {group_size}",
                    codes.len()
                ),
            });
        }

        let summed = sum_speech_embeddings(&prepared.speech_embedding_tables, &codes);

        let llm_d_model = prepared.llm_metadata.d_model;
        let speech_rows = prepared
            .inlocal_runtime
            .run(&summed, usable_frames, llm_d_model)
            .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
                reason: error.to_string(),
            })?;
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

        let mut token_rows =
            Vec::with_capacity(decode_prompt.token_ids.len() * llm_metadata.d_model);
        for &token_id in &decode_prompt.token_ids {
            let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                MimoAsrExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
            token_rows.extend_from_slice(&row);
        }
        let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
            &decode_prompt,
            llm_metadata.d_model,
            token_rows,
            &speech_rows,
        )
        .map_err(|error| MimoAsrExecutorError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;

        // Request-sized, not the decoder's native context window: see
        // `MimoLlmDecoderRuntime::new_kv_caches`'s doc comment.
        let layer_kv_caches = decoder.new_kv_caches(
            decode_prompt
                .token_ids
                .len()
                .saturating_add(MIMO_ASR_MAX_GENERATED_TOKENS),
        );
        let mut step_executor = MimoAsrGreedyStepExecutor {
            decoder,
            layer_kv_caches,
            prompt_embeddings: Some(prompt_embeddings),
            cache_prompt_tokens: 0,
            control: std::sync::Arc::clone(&request.execution_context.control),
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
            &request.execution_context.control,
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
    fn executor_id(&self) -> &'static str {
        MIMO_ASR_EXECUTOR_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
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
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudioView};
    use crate::models::ggml_family_registry::mimo_asr_runtime_descriptor_v1;

    use super::*;

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

    // Pinned to the real dev-pack decode (q8_0, `OPENASR_GGML_BACKEND=cpu` --
    // CPU is the deterministic reference backend; the default Metal backend's
    // memory fit for this family's ~8B combined weights on a 16GB
    // unified-memory Mac is unverified, see this module's e2e report). JFK is
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

        let request = GgmlAsrExecutionViewRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: mimo_asr_runtime_descriptor_v1(),
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

        let executor = MimoAsrGgmlExecutor;
        let started_at = Instant::now();
        let result = executor
            .execute_view(&request)
            .expect("mimo-asr transcribe");
        let elapsed = started_at.elapsed();
        Some((result.transcription.text, elapsed, audio_duration_seconds))
    }

    #[test]
    #[ignore = "requires the private ~9.6GB dev-only mimo-v2.5-asr-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu recommended (Metal memory fit unverified for this \
                family's ~8B combined weights on a 16GB unified-memory Mac)"]
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

    /// Resident prepared-runtime cache regression: `execute()` twice in a row on
    /// the same thread (same pack + backend) must hit the thread-local
    /// `MIMO_ASR_PREPARED_BY_KEY` cache on the second call and still produce a
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
