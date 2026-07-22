//! `GgmlAsrExecutor` implementation for granite-speech, wiring the already-
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

use thiserror::Error;

use super::decode_executor::GraniteSpeechAudioDecodeStepExecutor;
use super::prompt::{GRANITE_SPEECH_AUDIO_TOKEN, build_audio_prompt_embeddings};
use super::runtime_contract::{
    parse_decoder_metadata, parse_encoder_metadata, parse_projector_metadata,
};
use super::runtime_provider::load_tensors_from_oasr_pack;
use super::tokenizer::GraniteSpeechTokenizer;
use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::{GgmlCpuGraphConfig, read_gguf_metadata};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrPreparedAudio,
};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeConfig, Seq2SeqGreedyDecodeError,
    run_seq2seq_greedy_decode_loop_with_adapter_v0,
};

use crate::arch::GRANITE_SPEECH_GGML_ADAPTER_ID;

const GRANITE_SPEECH_EXECUTOR_ID: &str = "granite-speech-ggml-executor-v1";
const GRANITE_SPEECH_EOT_TOKEN_ID: u32 = 100_257;
const GRANITE_SPEECH_DEFAULT_QUESTION: &str =
    "can you transcribe the speech into a written format?";
const GRANITE_SPEECH_MAX_GENERATED_TOKENS: usize = 256;

#[derive(Debug, Error)]
enum GraniteSpeechGgmlExecutorError {
    #[error("granite-speech ggml executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("granite-speech ggml executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
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
    #[error("granite-speech ggml executor decode failed: {reason}")]
    DecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GraniteSpeechGgmlExecutor;

impl GraniteSpeechGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GraniteSpeechGgmlExecutorError> {
        if request.selected_family.adapter_id != GRANITE_SPEECH_GGML_ADAPTER_ID {
            return Err(GraniteSpeechGgmlExecutorError::AdapterMismatch {
                expected: GRANITE_SPEECH_GGML_ADAPTER_ID,
                found: request.selected_family.adapter_id.to_string(),
            });
        }

        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(
                |error| GraniteSpeechGgmlExecutorError::RuntimePreflightFailed {
                    reason: error.to_string(),
                },
            )?;
        let pack_path = preflight.runtime_source.path();

        let samples = downmix_prepared_audio(&request.prepared_audio);
        let frontend = super::frontend::GraniteSpeechMelFrontend::new();
        let (features, frames) = frontend.extract(&samples).map_err(|error| {
            GraniteSpeechGgmlExecutorError::FrontendFailed {
                reason: error.to_string(),
            }
        })?;

        let backend = GgmlCpuGraphConfig::resolve_runtime_backend();

        let metadata = read_gguf_metadata(pack_path).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let encoder_config = parse_encoder_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let projector_config = parse_projector_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_config = parse_decoder_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::MetadataFailed {
                reason: error.to_string(),
            }
        })?;

        let encoder_weights =
            load_tensors_from_oasr_pack(pack_path, "encoder.").map_err(|error| {
                GraniteSpeechGgmlExecutorError::EncoderFailed {
                    reason: error.to_string(),
                }
            })?;
        let encoder_output = super::encoder_graph::encode(
            &encoder_config,
            &encoder_weights,
            &features,
            frames,
            backend,
            false,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;

        let projector_weights =
            load_tensors_from_oasr_pack(pack_path, "projector.").map_err(|error| {
                GraniteSpeechGgmlExecutorError::ProjectorFailed {
                    reason: error.to_string(),
                }
            })?;
        let projector_output = super::qformer::project(
            &projector_config,
            &projector_weights,
            &encoder_output.encoder_out,
            encoder_output.frames,
            backend,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::ProjectorFailed {
            reason: error.to_string(),
        })?;

        let decoder_weights =
            load_tensors_from_oasr_pack(pack_path, "language_model.").map_err(|error| {
                GraniteSpeechGgmlExecutorError::DecodeFailed {
                    reason: error.to_string(),
                }
            })?;

        let tokenizer = GraniteSpeechTokenizer::from_gguf_metadata(&metadata).map_err(|error| {
            GraniteSpeechGgmlExecutorError::TokenizerFailed {
                reason: error.to_string(),
            }
        })?;

        // KWB (keyword-list biasing): the model's own documented prompt
        // convention -- "transcribe the speech to text. Keywords: <kw1>,
        // <kw2>, ..." -- not a decode-time logit bias (see the family's
        // end-to-end KWB test). `phrase_bias`'s configured phrases become
        // the `Keywords:` suffix when present.
        let question = match request.request_options.phrase_bias.as_ref() {
            Some(phrase_bias) if !phrase_bias.is_empty() => {
                let keywords = phrase_bias
                    .entries()
                    .iter()
                    .map(|entry| entry.phrase())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("transcribe the speech to text. Keywords: {keywords}")
            }
            _ => GRANITE_SPEECH_DEFAULT_QUESTION.to_string(),
        };
        let prompt_text = format!("USER: {GRANITE_SPEECH_AUDIO_TOKEN}{question}\n ASSISTANT:");
        let (prompt_token_ids, prompt_embeddings) = build_audio_prompt_embeddings(
            &decoder_config,
            &decoder_weights,
            &tokenizer,
            &prompt_text,
            &projector_output.projected,
            projector_output.tokens,
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::PromptFailed {
            reason: error.to_string(),
        })?;

        let decode_config = Seq2SeqGreedyDecodeConfig {
            initial_prompt_tokens: prompt_token_ids,
            eot_token_id: GRANITE_SPEECH_EOT_TOKEN_ID,
            stop_token_ids: vec![],
            vocab_size: decoder_config.vocab_size,
            max_generated_tokens: GRANITE_SPEECH_MAX_GENERATED_TOKENS,
            suppress_first_step_token_ids: vec![],
            suppress_token_ids: vec![],
            phrase_biases: vec![],
        };
        let mut step_executor = GraniteSpeechAudioDecodeStepExecutor::new(
            decoder_config,
            &decoder_weights,
            backend,
            prompt_embeddings,
        );
        let decode_text_token_ids =
            |token_ids: &[u32]| -> Result<String, Seq2SeqGreedyDecodeError> {
                tokenizer.decode_text_token_ids(token_ids).map_err(|error| {
                    Seq2SeqGreedyDecodeError::TokenizerDecodeFailed {
                        reason: error.to_string(),
                    }
                })
            };
        let result = run_seq2seq_greedy_decode_loop_with_adapter_v0(
            &decode_config,
            &mut step_executor,
            &decode_text_token_ids,
            |error| error,
            |error| error,
            &|text| text,
            &mut |_step, _token, _eot| {},
            &mut |_step, _logits| {},
        )
        .map_err(|error| GraniteSpeechGgmlExecutorError::DecodeFailed {
            reason: error.to_string(),
        })?;

        let audio_duration_seconds = request.prepared_audio.samples_f32.len() as f32
            / request.prepared_audio.sample_rate_hz.max(1) as f32;
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text: result.text.clone(),
                segments: vec![Segment {
                    start: 0.0,
                    end: audio_duration_seconds,
                    text: result.text,
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                }],
                longform: None,
                language: None,
            },
            carry_context: None,
        })
    }
}

fn downmix_prepared_audio(audio: &GgmlAsrPreparedAudio) -> Vec<f32> {
    if audio.channels <= 1 {
        return audio.samples_f32.clone();
    }
    let channels = audio.channels as usize;
    audio
        .samples_f32
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

impl GgmlAsrExecutor for GraniteSpeechGgmlExecutor {
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

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        self.execute_inner(request)
            .map_err(|error| granite_speech_execute_error_to_ggml(self, error, request))
    }
}

fn granite_speech_execute_error_to_ggml(
    executor: &GraniteSpeechGgmlExecutor,
    error: GraniteSpeechGgmlExecutorError,
    request: &GgmlAsrExecutionRequest,
) -> GgmlAsrExecutionError {
    GgmlAsrExecutionError::ExecutorFailed {
        executor_id: GgmlAsrExecutor::executor_id(executor),
        adapter_id: request.selected_family.adapter_id,
        reason: error.to_string(),
    }
}

impl GraniteSpeechGgmlExecutor {
    /// Streaming decode: re-runs the SAME offline pipeline (`execute_inner`)
    /// against the growing/windowed audio buffer the shared streaming driver
    /// hands it -- there is no incremental KV-cache session to plug in yet
    /// (see `decode_executor.rs`'s O(n^2) recompute-per-step note), so every
    /// partial re-does frontend + encoder + Q-Former + a full prefill-style
    /// decode from scratch. This is registered to satisfy the codebase's
    /// fail-closed streaming-completeness gate
    /// (`builtin_execution_dispatch::build_builtin_ggml_streaming_execution_dispatch`
    /// rejects the WHOLE dispatch, for every family, if any registered
    /// architecture has no streaming executor at all) -- it is correctness-
    /// only, not a real-time-tuned streaming path. The FINAL transcript stays
    /// byte-identical to `execute()`.
    fn execute_streaming(
        &self,
        request: &GgmlAsrExecutionRequest,
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
            *self,
            GRANITE_SPEECH_STREAMING_EXECUTOR_ID,
            GRANITE_SPEECH_GGML_ADAPTER_ID,
            "granite-speech",
            request,
            crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_HEAVY_SEQ2SEQ,
            GraniteSpeechGgmlExecutor::execute_streaming,
        )
    }
}
