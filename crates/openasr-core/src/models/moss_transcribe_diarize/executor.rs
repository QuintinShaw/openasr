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

use thiserror::Error;

use crate::NativeAsrError;
use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::GgmlCpuGraphRunner;
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
use crate::models::whisper::whisper_log_mel_spectrogram_16khz_mono_v0;

use super::adaptor_graph::{load_moss_adaptor_weights_from_reader, run_moss_adaptor};
use super::decode_prompt::build_moss_td_decode_prompt;
use super::encoder_graph::{
    MossEncoderConfig, load_moss_encoder_weights_from_reader, run_moss_encoder_chunk,
};
use super::graph_config::moss_td_encoder_graph_config;
use super::llm_decoder::MossTdDecoderRuntime;
use super::prompt_embedding::build_moss_td_prompt_embeddings_with_audio_splice;
use super::runtime_contract::{
    MOSS_TD_ADAPTOR_NORM_EPSILON, parse_adaptor_metadata, parse_decoder_metadata,
    parse_encoder_metadata,
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
        let encoder_weights = load_moss_encoder_weights_from_reader(&reader, encoder_config)
            .map_err(|error| MossTdExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;
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

        // Upstream `_compute_audio_token_length`'s stride: hop_length * the
        // Whisper conv stem's 2x stride * audio_merge_size.
        let token_stride = HOP_LENGTH * WHISPER_ENCODER_CONV_STRIDE * adaptor_metadata.merge_size;
        let mut encoder_runner =
            GgmlCpuGraphRunner::new(moss_td_encoder_graph_config()).map_err(|error| {
                MossTdExecutorError::EncoderFailed {
                    reason: format!("could not initialize encoder graph runner: {error}"),
                }
            })?;

        let mut concatenated_rows: Vec<f32> = Vec::new();
        let mut total_frames = 0usize;
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let mel = whisper_log_mel_spectrogram_16khz_mono_v0(
                chunk,
                encoder_metadata.n_mels,
                MEL_TARGET_FRAMES,
            )
            .map_err(|error| MossTdExecutorError::FrontendFailed {
                reason: error.to_string(),
            })?;
            let encoder_out = run_moss_encoder_chunk(
                &mut encoder_runner,
                &encoder_weights,
                encoder_config,
                mel.data(),
                MEL_TARGET_FRAMES,
            )
            .map_err(|error| MossTdExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;
            let token_length = (chunk.len() - 1) / token_stride.max(1) + 1;
            let keep_frames = (token_length * adaptor_metadata.merge_size)
                .min(encoder_metadata.max_source_positions);
            let keep_values = keep_frames * encoder_metadata.d_model;
            concatenated_rows.extend_from_slice(&encoder_out[..keep_values]);
            total_frames += keep_frames;
        }
        // Upstream's `time_merge` truncates any remainder below a full
        // merge-size group; concatenating already-merge-size-aligned
        // per-chunk lengths (see above) means the total is already aligned,
        // so this is a no-op guard, not a silent frame drop.
        let aligned_frames =
            (total_frames / adaptor_metadata.merge_size) * adaptor_metadata.merge_size;
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

        let runtime_path = preflight.runtime_source.path();
        let mut decoder =
            MossTdDecoderRuntime::new(runtime_path, decoder_metadata).map_err(|error| {
                MossTdExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
        if std::env::var_os("OPENASR_MOSS_TD_PROFILE").is_some() {
            eprintln!(
                "OPENASR_MOSS_TD_PROFILE decoder_backend={}",
                decoder.backend_label()
            );
        }

        let token_rows_len = decode_prompt.token_ids.len() * decoder_metadata.d_model;
        let mut token_rows = Vec::with_capacity(token_rows_len);
        for &token_id in &decode_prompt.token_ids {
            let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                MossTdExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
            token_rows.extend_from_slice(&row);
        }
        let spliced = build_moss_td_prompt_embeddings_with_audio_splice(
            decode_prompt.token_ids.len(),
            &decode_prompt.audio_pad_positions,
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

        let layer_kv_caches = decoder.new_kv_caches();
        let mut step_executor = MossTdGreedyStepExecutor {
            decoder: &mut decoder,
            layer_kv_caches,
            prompt_embeddings: Some(prompt_embeddings),
            cache_prompt_tokens: 0,
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.im_end_token_id,
            vocab_size: decoder_metadata.vocab_size,
            max_generated_tokens: MOSS_TD_MAX_GENERATED_TOKENS,
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
        )
        .map_err(|error| MossTdExecutorError::GreedyDecodeFailed {
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
