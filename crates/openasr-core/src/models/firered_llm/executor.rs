//! firered-llm dedicated executor: fbank+CMVN [`frontend`](super::super::firered_aed::frontend)
//! -> the parity-verified Conformer encoder
//! [`encoder_graph`](super::super::firered_aed::encoder_graph) (both reused
//! byte-for-byte from `firered_aed` -- architecturally identical, see
//! `package_import`'s module doc) -> the 2x frame-stacking [`adapter_graph`]
//! -> ChatML+`<speech>` splice ([`decode_prompt`] +
//! `qwen::build_qwen3_prompt_embeddings_with_audio_splice`) -> Qwen2
//! [`llm_transformer`] prefill/decode, driven through the ONE shared greedy
//! decode loop (`models::decode_policy_component_registry::
//! run_builtin_seq2seq_decode_policy`) via a
//! [`Seq2SeqGreedyDecodeStepExecutor`] impl below -- never a hand-rolled
//! argmax loop (the repo's `model-integration-shared-driver` invariant, see
//! `AGENTS.md`).

#![allow(dead_code)]

use thiserror::Error;

use crate::NativeAsrError;
use crate::NativeAsrSession;
use crate::api::backend::{Segment, Transcription};
use crate::arch::FIRERED_LLM_DECODE_POLICY_ID;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, BuiltinSeq2SeqDecodePolicyConfigInput,
    BuiltinSeq2SeqDecodePolicyTokenSource, run_builtin_seq2seq_decode_policy,
};
use crate::models::firered_aed::encoder_graph::FireRedEncoderGraphRuntime;
use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
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

use super::adapter_graph::{load_firered_llm_adapter_weights_from_reader, run_firered_llm_adapter};
use super::decode_prompt::build_firered_llm_decode_prompt;
use super::llm_transformer::FireRedLlmDecoderRuntime;
use super::runtime_contract::{
    parse_firered_llm_adapter_metadata, parse_firered_llm_decoder_metadata,
    parse_firered_llm_encoder_metadata,
};
use super::tokenizer::FireRedLlmTokenizer;

const FIRERED_LLM_EXECUTOR_ID: &str = "firered-llm-ggml-executor-v1";
const FIRERED_LLM_STREAMING_EXECUTOR_ID: &str = "firered-llm-ggml-snapshot-streaming-executor-v1";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";
/// Upstream single-utterance hard cap (`fireredasr2` README: "single 40s max
/// input"). The executor fails closed rather than silently truncating or
/// running an out-of-distribution multi-minute prefill; longer audio is the
/// longform slicing orchestrator's job (see `FIRERED_LLM_DECODE_POLICY_ID`'s
/// `ConservativeSeq2SeqV1` longform profile registration).
const FIRERED_LLM_MAX_INPUT_SECONDS: f32 = 40.0;
/// Generous upper bound on generated tokens per utterance -- greedy decode
/// stops at `<|im_end|>` well before this in practice; this is only the
/// fail-closed backstop against a runaway (non-terminating) decode.
const FIRERED_LLM_MAX_GENERATED_TOKENS: usize = 512;

#[derive(Debug, Error)]
enum FireRedLlmExecutorError {
    #[error("firered-llm executor requires adapter '{expected}', got '{found}'")]
    AdapterMismatch {
        expected: &'static str,
        found: String,
    },
    #[error("firered-llm executor runtime preflight failed: {reason}")]
    RuntimePreflightFailed { reason: String },
    #[error("firered-llm runtime metadata contract failed: {reason}")]
    RuntimeContractViolation { reason: String },
    #[error("firered-llm tokenizer materialization failed: {reason}")]
    TokenizerBuildFailed { reason: String },
    #[error("firered-llm cmvn vectors failed: {reason}")]
    CmvnBuildFailed { reason: String },
    #[error("firered-llm frontend failed: {reason}")]
    FrontendFailed { reason: String },
    #[error("firered-llm audio duration {seconds:.1}s exceeds the upstream {limit:.0}s hard cap")]
    AudioTooLong { seconds: f32, limit: f32 },
    #[error("firered-llm encoder failed: {reason}")]
    EncoderFailed { reason: String },
    #[error("firered-llm adapter failed: {reason}")]
    AdapterGraphFailed { reason: String },
    #[error("firered-llm decode prompt failed: {reason}")]
    DecodePromptFailed { reason: String },
    #[error("firered-llm prompt embedding splice failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("firered-llm decoder failed: {reason}")]
    DecoderFailed { reason: String },
    #[error("firered-llm greedy decode failed: {reason}")]
    GreedyDecodeFailed { reason: String },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FireRedLlmGgmlExecutor;

/// A no-op phrase-bias/token-source shim: firered-llm's decode policy never
/// consults these (no phrase bias, `seq2seq_stop_token_kind: None` -- eot is
/// supplied directly via `BuiltinSeq2SeqDecodePolicyConfigInput`), so a real
/// implementation would be dead weight. `resolve_builtin_decode_policy`'s
/// config builder still requires the trait object, matching `()`'s existing
/// blanket impl of `BuiltinSeq2SeqDecodePolicyTokenSource`.
struct NoPhraseBiasTokenSource;
impl PhraseBiasTokenEncoder for NoPhraseBiasTokenSource {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}
impl BuiltinSeq2SeqDecodePolicyTokenSource for NoPhraseBiasTokenSource {}

/// Drives `FireRedLlmDecoderRuntime` through the shared greedy loop: the
/// first step (index 0, no generated tokens yet) consumes the pre-built
/// prompt embeddings via one prefill pass; every step after that embeds the
/// last generated token and runs one incremental decode step. Mirrors
/// `qwen::ggml_executor::Qwen3AsrPrefillOnlyGreedyStepExecutor`'s shape.
struct FireRedLlmGreedyStepExecutor<'a> {
    decoder: &'a mut FireRedLlmDecoderRuntime,
    layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    prompt_embeddings: Option<crate::models::qwen::Qwen3AsrPromptEmbeddings>,
    cache_prompt_tokens: usize,
}

impl Seq2SeqGreedyDecodeStepExecutor for FireRedLlmGreedyStepExecutor<'_> {
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
                reason: "firered-llm generated token history is unexpectedly empty".to_string(),
            }
        })?;
        let cache_position = self
            .cache_prompt_tokens
            .checked_add(input.generated_tokens.len())
            .and_then(|total| total.checked_sub(1))
            .ok_or_else(|| Seq2SeqGreedyDecodeError::DecoderStepFailed {
                reason: "firered-llm decode cache position underflowed".to_string(),
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

impl FireRedLlmGgmlExecutor {
    fn execute_inner(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, FireRedLlmExecutorError> {
        let expected_adapter = crate::arch::FIRERED_LLM_GGML_ADAPTER_ID;
        if request.selected_family.adapter_id != expected_adapter {
            return Err(FireRedLlmExecutorError::AdapterMismatch {
                expected: expected_adapter,
                found: request.selected_family.adapter_id.to_string(),
            });
        }
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| FireRedLlmExecutorError::RuntimePreflightFailed {
                reason: error.to_string(),
            })?;

        let encoder_metadata =
            parse_firered_llm_encoder_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let adapter_metadata =
            parse_firered_llm_adapter_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let decoder_metadata =
            parse_firered_llm_decoder_metadata(&*preflight.metadata).map_err(|error| {
                FireRedLlmExecutorError::RuntimeContractViolation {
                    reason: error.to_string(),
                }
            })?;
        let tokenizer = FireRedLlmTokenizer::from_gguf_metadata(&preflight.metadata).map_err(
            |error: NativeAsrError| FireRedLlmExecutorError::TokenizerBuildFailed {
                reason: error.to_string(),
            },
        )?;

        let samples = &request.prepared_audio.samples_f32;
        let audio_duration_seconds =
            samples.len() as f32 / request.prepared_audio.sample_rate_hz.max(1) as f32;
        if audio_duration_seconds > FIRERED_LLM_MAX_INPUT_SECONDS {
            return Err(FireRedLlmExecutorError::AudioTooLong {
                seconds: audio_duration_seconds,
                limit: FIRERED_LLM_MAX_INPUT_SECONDS,
            });
        }

        let reader = build_runtime_tensor_reader_from_preflight(&preflight).map_err(|error| {
            FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            }
        })?;
        let feature_dim_shape = [encoder_metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim_shape)
            .map_err(|error| FireRedLlmExecutorError::CmvnBuildFailed {
                reason: error.to_string(),
            })?;

        let frontend = FireRedFbankFrontend::new();
        let mut features =
            frontend
                .compute(samples)
                .map_err(|error| FireRedLlmExecutorError::FrontendFailed {
                    reason: error.to_string(),
                })?;
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).map_err(
            |error| FireRedLlmExecutorError::FrontendFailed {
                reason: error.to_string(),
            },
        )?;

        let runtime_path = preflight.runtime_source.path();
        let mut encoder_runtime = FireRedEncoderGraphRuntime::new(runtime_path, encoder_metadata)
            .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
            reason: error.to_string(),
        })?;
        let encoder_output = encoder_runtime
            .encode(&features.data, features.n_frames)
            .map_err(|error| FireRedLlmExecutorError::EncoderFailed {
                reason: error.to_string(),
            })?;

        let adapter_weights = load_firered_llm_adapter_weights_from_reader(
            &reader,
            encoder_metadata.d_model,
            adapter_metadata.downsample_rate,
            adapter_metadata.llm_dim,
        )
        .map_err(|error| FireRedLlmExecutorError::AdapterGraphFailed {
            reason: error.to_string(),
        })?;
        let (speech_rows, speech_frame_count) = run_firered_llm_adapter(
            &adapter_weights,
            &encoder_output.rows,
            encoder_output.frame_count,
            encoder_metadata.d_model,
            adapter_metadata.downsample_rate,
        )
        .map_err(|error| FireRedLlmExecutorError::AdapterGraphFailed {
            reason: error.to_string(),
        })?;

        let decode_prompt = build_firered_llm_decode_prompt(&tokenizer, speech_frame_count)
            .map_err(|error| FireRedLlmExecutorError::DecodePromptFailed {
                reason: error.to_string(),
            })?;

        let mut decoder =
            FireRedLlmDecoderRuntime::new(runtime_path, decoder_metadata).map_err(|error| {
                FireRedLlmExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
        let token_rows_len = decode_prompt.token_ids.len() * decoder_metadata.d_model;
        let mut token_rows = Vec::with_capacity(token_rows_len);
        for &token_id in &decode_prompt.token_ids {
            let row = decoder.gather_token_embedding(token_id).map_err(|error| {
                FireRedLlmExecutorError::DecoderFailed {
                    reason: error.to_string(),
                }
            })?;
            token_rows.extend_from_slice(&row);
        }
        let prompt_embeddings = build_qwen3_prompt_embeddings_with_audio_splice(
            &decode_prompt,
            decoder_metadata.d_model,
            &token_rows,
            &speech_rows,
        )
        .map_err(|error| FireRedLlmExecutorError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;

        let layer_kv_caches = decoder.new_kv_caches();
        let mut step_executor = FireRedLlmGreedyStepExecutor {
            decoder: &mut decoder,
            layer_kv_caches,
            prompt_embeddings: Some(prompt_embeddings),
            cache_prompt_tokens: 0,
        };
        let config = BuiltinSeq2SeqDecodePolicyConfigInput {
            initial_prompt_tokens: decode_prompt.token_ids.clone(),
            eot_token_id: tokenizer.chatml_im_end_token_id,
            vocab_size: decoder_metadata.vocab_size,
            max_generated_tokens: FIRERED_LLM_MAX_GENERATED_TOKENS,
        };
        let result = run_builtin_seq2seq_decode_policy(
            FIRERED_LLM_DECODE_POLICY_ID,
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
        .map_err(|error| FireRedLlmExecutorError::GreedyDecodeFailed {
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

impl GgmlAsrExecutor for FireRedLlmGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_LLM_EXECUTOR_ID
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

impl GgmlAsrStreamingExecutor for FireRedLlmGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        FIRERED_LLM_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        build_seq2seq_streaming_session(
            self.clone(),
            FIRERED_LLM_STREAMING_EXECUTOR_ID,
            crate::arch::FIRERED_LLM_GGML_ADAPTER_ID,
            "firered-llm",
            request,
            STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT,
            FireRedLlmGgmlExecutor::execute,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::models::ggml_asr_executor::{GgmlAsrBackendPreference, GgmlAsrPreparedAudio};
    use crate::models::ggml_family_registry::firered_llm_runtime_descriptor_v1;

    use super::*;

    /// Points at the real converted pack from T2
    /// (`scratchpad/fr2/T2-report.md`), an ~8.3GB q8_0 `.oasr` NOT committed
    /// to the repo (dev-only artifact, same convention as firered-aed's own
    /// `tmp/firered-out/firered-aed-l-fp16.oasr` golden pack). Loading it
    /// mmaps + touches most of an 8.3GB file plus materializes the ~1GB f16
    /// token-embedding table -- a real memory commitment, not a network
    /// fetch, so this stays `#[ignore]`d and skips silently when absent
    /// (matches firered-aed's own dev-pack test convention) rather than
    /// gating CI on a multi-GB private artifact.
    fn dev_pack_path() -> PathBuf {
        PathBuf::from(
            "/Volumes/QuintinDocument/openasr-dev/tmp-weights/fr2/out/firered2-llm-q8_0.oasr",
        )
    }

    // Pinned to the real T5 dev-pack decode (q8_0, this repo's `OPENASR_GGML_BACKEND=cpu`
    // -- CPU is the deterministic reference backend; Metal currently OOMs this
    // family's 7B decoder on a 16GB unified-memory Mac, see the T5 report). JFK
    // is word-for-word correct; the Mandarin sentence is the same non-copyrighted
    // `say -v Tingting` synthesis firered-aed's own golden uses (see that
    // family's `zh_sample.wav` doc comment).
    const GOLDEN_JFK_TEXT: &str = "and so my fellow americans ask not what your country can do \
        for you ask what you can do for your country";

    const GOLDEN_ZH_TEXT: &str = "今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的\
        川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下";

    // Code-switch coverage (first 5s of jfk.wav + first 8s of zh_sample.wav,
    // single <=40s utterance, no longform slicing involved): both languages'
    // ChatML/tokenizer/decode paths share one prefill+decode call here.
    const GOLDEN_EN_ZH_MIXED_TEXT: &str = "and so my fellow americans ask not 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开";

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
        let pack_path = dev_pack_path();
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return None;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav_path,
            "firered-llm e2e test",
            "firered-llm e2e test",
        )
        .expect("load wav fixture");
        let audio_duration_seconds = samples.len() as f32 / 16_000.0;

        let request = GgmlAsrExecutionRequest {
            runtime_source_path: pack_path,
            runtime_source_preflight: None,
            selected_family: firered_llm_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: Default::default(),
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };

        let executor = FireRedLlmGgmlExecutor;
        let started_at = Instant::now();
        let result = executor.execute(&request).expect("firered-llm transcribe");
        let elapsed = started_at.elapsed();
        Some((result.transcription.text, elapsed, audio_duration_seconds))
    }

    // T5: promoted from the Stage-4 "prints transcript for manual judgement"
    // probe once a human read the printed transcripts and confirmed JFK is
    // word-for-word correct and the Mandarin sentence is coherent (see the T5
    // report's parity + e2e sections) -- mirrors firered-aed's own
    // `golden_diff_end_to_end_transcribe_matches_reference_pytorch_decode_on_*`
    // promotion history. RTF/elapsed are still logged to stderr (not asserted:
    // wall-clock varies with shared-machine load) so a maintainer re-running
    // this locally still gets the performance signal the old probe printed.
    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu (Metal currently OOMs this family's 7B decoder \
                on a 16GB unified-memory Mac -- see the T5 report)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_jfk_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(jfk_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [jfk.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_JFK_TEXT);
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu (Metal currently OOMs this family's 7B decoder \
                on a 16GB unified-memory Mac -- see the T5 report)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_zh_sample_wav() {
        let Some((text, elapsed, audio_duration_seconds)) = transcribe_with_dev_pack(zh_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [zh_sample.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_ZH_TEXT);
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; \
                OPENASR_GGML_BACKEND=cpu (Metal currently OOMs this family's 7B decoder \
                on a 16GB unified-memory Mac -- see the T5 report)"]
    fn golden_diff_end_to_end_transcribe_matches_reference_decode_on_en_zh_mixed_wav() {
        let Some((text, elapsed, audio_duration_seconds)) =
            transcribe_with_dev_pack(en_zh_mixed_wav_path())
        else {
            return;
        };
        eprintln!(
            "firered-llm e2e [en_zh_mixed.wav]: rtf={:.3} elapsed={elapsed:?} audio_duration={audio_duration_seconds:.2}s",
            elapsed.as_secs_f32() / audio_duration_seconds.max(0.001)
        );
        assert_eq!(text, GOLDEN_EN_ZH_MIXED_TEXT);
    }
}
