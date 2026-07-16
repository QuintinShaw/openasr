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

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::MIMO_ASR_DECODE_POLICY_ID;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
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
    Seq2SeqGreedyDecodeError, Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
    Seq2SeqGreedyDecodeStepLogitsOutput,
};

use super::audio_tokenizer_graph::MimoAudiotokEncoderRuntime;
use super::decode_prompt::build_mimo_asr_decode_prompt;
use super::input_local_graph::{
    MimoInputLocalRuntime, load_speech_embedding_tables_from_reader, sum_speech_embeddings,
};
use super::llm_transformer::MimoLlmDecoderRuntime;
use super::mel_frontend::{
    load_mimo_mel_frontend_plan_from_reader, mimo_mel_features_from_samples,
};
use super::runtime_contract::{
    parse_mimo_audiotok_metadata, parse_mimo_inlocal_metadata, parse_mimo_llm_metadata,
    parse_mimo_mel_metadata, parse_mimo_special_tokens,
};
use super::rvq::{encode_rvq_codes, load_mimo_rvq_codebooks_from_reader};
use super::tokenizer::MimoAsrTokenizer;

const MIMO_ASR_EXECUTOR_ID: &str = "mimo-asr-ggml-executor-v1";
const MIMO_ASR_STREAMING_EXECUTOR_ID: &str = "mimo-asr-ggml-snapshot-streaming-executor-v1";
/// The reference `preprocess_input` re-chunks internally at 30s (`chunk_samples
/// = 30 * sampling_rate`); this executor instead fails closed above that same
/// bound and leaves multi-chunk orchestration to the shared longform slicer
/// (mirrors `firered_llm`'s upstream-hard-cap precedent).
const MIMO_ASR_MAX_INPUT_SECONDS: f32 = 30.0;
const MIMO_ASR_MAX_GENERATED_TOKENS: usize = 512;

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
}

impl Seq2SeqGreedyDecodeStepExecutor for MimoAsrGreedyStepExecutor<'_> {
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

impl MimoAsrGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
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

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > MIMO_ASR_MAX_INPUT_SECONDS {
            return Err(MimoAsrExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: MIMO_ASR_MAX_INPUT_SECONDS,
            });
        }

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
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
        let mel_features = mimo_mel_features_from_samples(samples, &mel_plan).map_err(|error| {
            MimoAsrExecutorError::MelFrontendFailed {
                reason: error.to_string(),
            }
        })?;

        let runtime_path = preflight.runtime_source.path();
        let mut encoder_runtime =
            MimoAudiotokEncoderRuntime::new(runtime_path, audiotok_metadata.clone()).map_err(
                |error| MimoAsrExecutorError::EncoderFailed {
                    reason: error.to_string(),
                },
            )?;
        let encoder_output = encoder_runtime.encode(&mel_features).map_err(|error| {
            MimoAsrExecutorError::EncoderFailed {
                reason: error.to_string(),
            }
        })?;

        let codebooks =
            load_mimo_rvq_codebooks_from_reader(&reader, &audiotok_metadata).map_err(|error| {
                MimoAsrExecutorError::RvqFailed {
                    reason: error.to_string(),
                }
            })?;
        let mut codes =
            encode_rvq_codes(&codebooks, &encoder_output.rows, encoder_output.frame_count)
                .map_err(|error| MimoAsrExecutorError::RvqFailed {
                    reason: error.to_string(),
                })?;

        // Truncate to the nearest group_size multiple (drop up to
        // group_size-1 trailing 25Hz frames = well under 200ms of audio) --
        // the reference asserts exact divisibility rather than padding.
        let group_size = inlocal_metadata.group_size;
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
        let tables = load_speech_embedding_tables_from_reader(
            &reader,
            inlocal_metadata.d_model,
            &speech_vocab_sizes,
            &zeroemb_idx,
        )
        .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
            reason: error.to_string(),
        })?;
        let summed = sum_speech_embeddings(&tables, &codes);

        let mut inlocal_runtime = MimoInputLocalRuntime::new(runtime_path, inlocal_metadata)
            .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
                reason: error.to_string(),
            })?;
        let speech_rows = inlocal_runtime
            .run(&summed, usable_frames, llm_metadata.d_model)
            .map_err(|error| MimoAsrExecutorError::InputLocalFailed {
                reason: error.to_string(),
            })?;
        let audio_group_count = usable_frames / group_size;

        let decode_prompt =
            build_mimo_asr_decode_prompt(&tokenizer, audio_group_count).map_err(|error| {
                MimoAsrExecutorError::DecodePromptFailed {
                    reason: error.to_string(),
                }
            })?;

        let mut decoder =
            MimoLlmDecoderRuntime::new(runtime_path, llm_metadata).map_err(|error| {
                MimoAsrExecutorError::DecoderFailed {
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
            &token_rows,
            &speech_rows,
        )
        .map_err(|error| MimoAsrExecutorError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;

        let layer_kv_caches = decoder.new_kv_caches();
        let mut step_executor = MimoAsrGreedyStepExecutor {
            decoder: &mut decoder,
            layer_kv_caches,
            prompt_embeddings: Some(prompt_embeddings),
            cache_prompt_tokens: 0,
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.special.im_end_id,
            vocab_size: llm_metadata.vocab_size,
            max_generated_tokens: MIMO_ASR_MAX_GENERATED_TOKENS,
        };
        let result = run_builtin_seq2seq_decode_policy(
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
        )
        .map_err(|error| MimoAsrExecutorError::GreedyDecodeFailed {
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

impl GgmlAsrExecutor for MimoAsrGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        MIMO_ASR_EXECUTOR_ID
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
            MimoAsrGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudio};
    use crate::models::ggml_family_registry::mimo_asr_runtime_descriptor_v1;

    use super::*;

    /// Real converted dev pack from P2.1+P2.2 (`tooling/mimo-asr/convert_mimo_asr.py`),
    /// NOT committed to the repo (dev-only artifact, same convention as
    /// firered2-llm's own `tmp-weights/fr2/out/firered2-llm-q8_0.oasr`).
    fn dev_pack_path() -> PathBuf {
        PathBuf::from(
            "/Volumes/QuintinDocument/openasr-dev/tmp-weights/mimo/out/mimo-v2.5-asr-q8_0.oasr",
        )
    }

    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    fn zh_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zh_sample.wav")
    }

    fn transcribe_with_dev_pack(wav_path: PathBuf) -> Option<(String, std::time::Duration, f32)> {
        let pack_path = dev_pack_path();
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

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: mimo_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };

        let executor = MimoAsrGgmlExecutor;
        let started_at = Instant::now();
        let result = executor.execute(&request).expect("mimo-asr transcribe");
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
            "mimo-asr e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s text={text:?}",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert!(!text.is_empty());
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
            "mimo-asr e2e [zh_sample.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s text={text:?}",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert!(!text.is_empty());
    }
}
