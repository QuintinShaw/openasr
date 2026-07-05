//! sensevoice transcription core: frontend (fbank + LFR + CMVN) -> prompt
//! splice -> SAN-M encoder graph -> CTC greedy decode -> tag-prefix strip.
//!
//! Mirrors `parakeet_ctc::executor` (prepared-runtime cache keyed by
//! `(canonical path, backend)`, shared CTC decode policy, snapshot/incremental
//! streaming driver). SenseVoice-specific: the request language selects the
//! 4-token prompt fail-closed (`language::build_sensevoice_prompt`), and the
//! decoded text's leading `<|lang|><|emotion|><|event|><|itn|>` tags are parsed
//! into structured fields -- emotion/event stay shadowed
//! (`SenseVoiceTagShadow::Shadowed`); only the language read-back is surfaced.
//!
//! Word timestamps: none (dolphin precedent). SenseVoice's CTC frames sit on a
//! 60 ms LFR grid behind 4 prompt frames; deriving per-word times from them
//! would be fabricated precision, so `words` stays empty.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::PhraseBiasConfig;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{OpenAsrArchitectureRegistry, SENSEVOICE_GGML_ARCHITECTURE_ID};
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::ctc_greedy_decode::{CtcGreedyDecodeError, CtcGreedyDecodeResult};
use crate::models::ctc_streaming_driver::build_ctc_streaming_driver;
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentRegistryError, run_builtin_ctc_decode_policy,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutor, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::ggml_streaming_session::GgmlAsrStreamingTranscriptSession;
use crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT;
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, with_thread_local_cached_mut_by_key,
};
use crate::{NativeAsrSession, SENSEVOICE_GGML_ADAPTER_ID};

use super::encoder_graph::{SenseVoiceEncoderGraph, build_sensevoice_encoder_input};
use super::encoder_weights::load_sensevoice_encoder_weights;
use super::frontend::{SenseVoiceFbankFrontend, apply_cmvn, apply_lfr};
use super::language::build_sensevoice_prompt;
use super::runtime_contract::{SenseVoiceExecutionMetadata, parse_sensevoice_execution_metadata};
use super::tokenizer::SenseVoiceTokenizer;
use super::{SenseVoiceTagShadow, strip_sensevoice_tag_prefix};

type SenseVoiceRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static SENSEVOICE_RUNTIME_BY_KEY: RefCell<HashMap<SenseVoiceRuntimeCacheKey, SenseVoicePreparedRuntime>> =
        RefCell::new(HashMap::new());
}

const SENSEVOICE_STREAMING_EXECUTOR_ID: &str = "sensevoice-ggml-snapshot-streaming-executor-v1";

/// Resolves the sensevoice block-stack `layer_count_hparam` against the parsed
/// metadata (reads the named hparam, not the materialized stack length).
struct SenseVoiceLayerCountResolver {
    n_layers: usize,
}

impl LayerCountResolver for SenseVoiceLayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            "sensevoice.n_layers" => Some(self.n_layers),
            _ => None,
        }
    }
}

fn ctc_err_to_string(error: CtcGreedyDecodeError) -> String {
    error.to_string()
}
fn registry_err_to_string(error: BuiltinDecodePolicyComponentRegistryError) -> String {
    error.to_string()
}

struct SenseVoicePreparedRuntime {
    metadata: SenseVoiceExecutionMetadata,
    tokenizer: SenseVoiceTokenizer,
    graph: SenseVoiceEncoderGraph,
    /// 16x560 prompt-embedding rows (host f32).
    prompt_embed: Vec<f32>,
    cmvn_neg_mean: Vec<f32>,
    cmvn_inv_stddev: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SenseVoiceTranscription {
    pub text: String,
    /// Honest language read-back: the requested code when one was selected, else
    /// the model's detected `<|lang|>` tag when it is a code this family
    /// advertises; `None` otherwise (never fabricated).
    pub language: Option<String>,
}

fn build_sensevoice_prepared_runtime(
    pack_path: &Path,
) -> Result<SenseVoicePreparedRuntime, String> {
    let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(pack_path)
        .map_err(|e| e.to_string())?;
    let gguf_metadata =
        crate::ggml_runtime::read_gguf_metadata(pack_path).map_err(|e| e.to_string())?;
    let metadata =
        parse_sensevoice_execution_metadata(&gguf_metadata).map_err(|e| e.to_string())?;
    let tokenizer = SenseVoiceTokenizer::from_metadata(&gguf_metadata)?;
    let weights = load_sensevoice_encoder_weights(&reader, &metadata).map_err(|e| e.to_string())?;
    validate_sensevoice_block_stack(metadata, weights.enc_layers.len())?;
    let prompt_embed = weights.prompt_embed.values.clone();
    let cmvn_neg_mean = weights.cmvn_neg_mean.values.clone();
    let cmvn_inv_stddev = weights.cmvn_inv_stddev.values.clone();
    let graph = SenseVoiceEncoderGraph::new(&weights, metadata, Some(pack_path))
        .map_err(|e| e.to_string())?;
    Ok(SenseVoicePreparedRuntime {
        metadata,
        tokenizer,
        graph,
        prompt_embed,
        cmvn_neg_mean,
        cmvn_inv_stddev,
    })
}

fn validate_sensevoice_block_stack(
    metadata: SenseVoiceExecutionMetadata,
    materialized_enc_layers: usize,
) -> Result<(), String> {
    let block_stack = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(SENSEVOICE_GGML_ARCHITECTURE_ID)
        .and_then(|descriptor| descriptor.block_stack);
    validate_stage_against_descriptor(
        SENSEVOICE_GGML_ARCHITECTURE_ID,
        block_stack.as_ref(),
        OpenAsrStageRole::Encoder,
        OpenAsrOrchestrationShape::Ctc,
        StageBuildPlan {
            block_kind: OpenAsrBlockKind::SanMFsmnEncoderLayer,
            tensor_name_scope: "enc.blk",
            family_layer_count: materialized_enc_layers,
        },
        &SenseVoiceLayerCountResolver {
            n_layers: metadata.n_layers,
        },
    )
    .map_err(|error| format!("sensevoice encoder block-stack descriptor mismatch: {error:?}"))?;
    Ok(())
}

impl SenseVoicePreparedRuntime {
    fn decode_result(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        phrase_bias: Option<&PhraseBiasConfig>,
    ) -> Result<CtcGreedyDecodeResult, String> {
        let prompt = build_sensevoice_prompt(language, false).map_err(|e| e.to_string())?;
        let dim = self.metadata.feature_dim;

        let fbank = SenseVoiceFbankFrontend::new()
            .compute(samples)
            .map_err(|e| e.to_string())?;
        let mut lfr = apply_lfr(&fbank.data, fbank.n_mels).map_err(|e| e.to_string())?;
        apply_cmvn(
            &mut lfr.data,
            lfr.feature_dim,
            &self.cmvn_neg_mean,
            &self.cmvn_inv_stddev,
        )
        .map_err(|e| e.to_string())?;

        let embed_rows = self.prompt_embed.len() / dim;
        let mut prompt_rows: Vec<&[f32]> = Vec::with_capacity(prompt.embed_indices.len());
        for &index in &prompt.embed_indices {
            if index >= embed_rows {
                return Err(format!(
                    "sensevoice prompt embed index {index} out of range (table has {embed_rows} rows)"
                ));
            }
            prompt_rows.push(&self.prompt_embed[index * dim..(index + 1) * dim]);
        }
        let input =
            build_sensevoice_encoder_input(&prompt_rows, &lfr.data, dim, self.metadata.d_model)
                .map_err(|e| e.to_string())?;
        let output = self.graph.encode(&input).map_err(|e| e.to_string())?;

        let frame_logits: Vec<&[f32]> = (0..output.frame_count)
            .map(|f| &output.logits[f * output.vocab_size..(f + 1) * output.vocab_size])
            .collect();
        let tokenizer = &self.tokenizer;
        let detok = |ids: &[u32]| tokenizer.decode(ids);
        run_builtin_ctc_decode_policy(
            crate::SENSEVOICE_DECODE_POLICY_ID,
            &frame_logits,
            output.vocab_size,
            phrase_bias,
            tokenizer,
            &detok,
            ctc_err_to_string,
            registry_err_to_string,
        )
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        phrase_bias: Option<&PhraseBiasConfig>,
    ) -> Result<SenseVoiceTranscription, String> {
        let result = self.decode_result(samples, language, phrase_bias)?;
        let requested = build_sensevoice_prompt(language, false).map_err(|e| e.to_string())?;
        Ok(sensevoice_result_to_transcription(
            &result.text,
            &requested.resolved_language,
        ))
    }
}

/// Strip the tag prefix and derive the honest language read-back. Emotion/event
/// tags are parsed but stay shadowed per the default [`SenseVoiceTagShadow`].
pub(crate) fn sensevoice_result_to_transcription(
    raw_text: &str,
    resolved_language: &str,
) -> SenseVoiceTranscription {
    let (tags, text) = strip_sensevoice_tag_prefix(raw_text);
    debug_assert!(!SenseVoiceTagShadow::default().exposes_emotion_event());
    let language = if resolved_language != "auto" {
        Some(resolved_language.to_string())
    } else {
        tags.language
            .filter(|code| super::language::SENSEVOICE_LANGUAGE_CODES.contains(&code.as_str()))
    };
    SenseVoiceTranscription { text, language }
}

/// Strip the tag prefix from a raw CTC decode result IN PLACE (used by the
/// streaming driver so PARTIAL transcripts never show `<|zh|>...` tags).
fn strip_tags_in_result(mut result: CtcGreedyDecodeResult) -> CtcGreedyDecodeResult {
    let (_tags, text) = strip_sensevoice_tag_prefix(&result.text);
    result.text = text;
    result
}

fn decode_sensevoice_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    language: Option<&str>,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<CtcGreedyDecodeResult, String> {
    let backend = super::graph_config::sensevoice_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &SENSEVOICE_RUNTIME_BY_KEY,
        key,
        || build_sensevoice_prepared_runtime(pack_path),
        |runtime| runtime.decode_result(samples, language, phrase_bias),
    )
}

fn transcribe_sensevoice_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    language: Option<&str>,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<SenseVoiceTranscription, String> {
    let backend = super::graph_config::sensevoice_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &SENSEVOICE_RUNTIME_BY_KEY,
        key,
        || build_sensevoice_prepared_runtime(pack_path),
        |runtime| runtime.transcribe(samples, language, phrase_bias),
    )
}

/// Dedicated GgmlAsrExecutor for sensevoice (DedicatedRuntimeExecutorV1).
#[derive(Debug, Clone, Default)]
pub(crate) struct SenseVoiceGgmlExecutor;

impl SenseVoiceGgmlExecutor {
    fn execute_ctc_result(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::SENSEVOICE_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        decode_sensevoice_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
            request.request_options.language.as_deref(),
            request.request_options.phrase_bias.as_ref(),
        )
        .map(strip_tags_in_result)
        .map_err(fail)
    }
}

impl GgmlAsrExecutor for SenseVoiceGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::SENSEVOICE_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<
        crate::models::ggml_asr_executor::GgmlAsrExecutionResult,
        crate::models::ggml_asr_executor::GgmlAsrExecutionError,
    > {
        use crate::api::backend::{Segment, Transcription};
        use crate::models::ggml_asr_executor::GgmlAsrExecutionResult;
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::SENSEVOICE_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let output = transcribe_sensevoice_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
            request.request_options.language.as_deref(),
            request.request_options.phrase_bias.as_ref(),
        )
        .map_err(fail)?;
        let duration = request.prepared_audio.samples_f32.len() as f32 / 16_000.0_f32;
        let segments = if output.text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: duration,
                text: output.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                // No acoustic word timestamps for this architecture (dolphin
                // precedent): never fabricate times.
                words: Vec::new(),
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text: output.text,
                segments,
                longform: None,
                language: output.language,
            },
            carry_context: None,
        })
    }
}

impl GgmlAsrStreamingExecutor for SenseVoiceGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        SENSEVOICE_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        if request.selected_family.adapter_id != SENSEVOICE_GGML_ADAPTER_ID {
            return Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: SENSEVOICE_STREAMING_EXECUTOR_ID,
                adapter_id: request.selected_family.adapter_id,
                reason: format!(
                    "sensevoice streaming executor requires adapter '{SENSEVOICE_GGML_ADAPTER_ID}', got '{}'",
                    request.selected_family.adapter_id
                ),
            });
        }
        let driver = build_ctc_streaming_driver(
            self.clone(),
            SENSEVOICE_STREAMING_EXECUTOR_ID,
            SENSEVOICE_GGML_ADAPTER_ID,
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            SenseVoiceGgmlExecutor::execute_ctc_result,
            <SenseVoiceGgmlExecutor as GgmlAsrExecutor>::execute,
        );
        let session = GgmlAsrStreamingTranscriptSession::new(
            SENSEVOICE_STREAMING_EXECUTOR_ID,
            request,
            driver,
        )?;
        Ok(Box::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_surfaces_requested_language_and_strips_tags() {
        let out = sensevoice_result_to_transcription(
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>\u{5f00}\u{996d}",
            "zh",
        );
        assert_eq!(out.text, "\u{5f00}\u{996d}");
        assert_eq!(out.language.as_deref(), Some("zh"));
    }

    #[test]
    fn auto_language_surfaces_only_advertised_detected_codes() {
        let detected =
            sensevoice_result_to_transcription("<|en|><|NEUTRAL|><|Speech|><|woitn|>hello", "auto");
        assert_eq!(detected.language.as_deref(), Some("en"));
        // A non-recognition tag (e.g. nospeech) must not be surfaced as a language.
        let unknown = sensevoice_result_to_transcription(
            "<|nospeech|><|NEUTRAL|><|Speech|><|woitn|>",
            "auto",
        );
        assert_eq!(unknown.language, None);
    }

    /// End-to-end transcription gate on the real packs + real clips (zh + en).
    /// Skipped when the local pack/clips are absent; asserted against the
    /// PyTorch reference transcripts produced by the ref.py oracle.
    #[test]
    #[ignore = "requires local sensevoice pack + audio clips (SENSEVOICE_PACK, SENSEVOICE_AUDIO_DIR)"]
    fn sensevoice_transcribes_zh_and_en_clips() {
        let pack = PathBuf::from(std::env::var("SENSEVOICE_PACK").expect("SENSEVOICE_PACK"));
        let audio_dir =
            PathBuf::from(std::env::var("SENSEVOICE_AUDIO_DIR").expect("SENSEVOICE_AUDIO_DIR"));
        let read_wav = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(audio_dir.join(name)).expect("wav");
            let mut i = 12;
            while i + 8 <= bytes.len() {
                let id = &bytes[i..i + 4];
                let size =
                    u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                        as usize;
                if id == b"data" {
                    let start = i + 8;
                    let end = (start + size).min(bytes.len());
                    return bytes[start..end]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                        .collect();
                }
                i += 8 + size + (size & 1);
            }
            panic!("no data chunk in {name}");
        };

        let zh = read_wav("zh.wav");
        let zh_out =
            transcribe_sensevoice_pcm_cached(&zh, &pack, Some("zh"), None).expect("zh transcribe");
        eprintln!(
            "sensevoice zh: {:?} (lang {:?})",
            zh_out.text, zh_out.language
        );
        // The second character is genuinely ambiguous on this clip: the f32
        // PyTorch reference itself emits \u{653e} under the zh prompt and
        // \u{996d} under auto; quantized packs may land on either. Accept both.
        let zh_expected = [
            "\u{5f00}\u{653e}\u{65f6}\u{95f4}\u{65e9}\u{4e0a}\u{4e5d}\u{70b9}\u{81f3}\u{4e0b}\u{5348}\u{4e94}\u{70b9}",
            "\u{5f00}\u{996d}\u{65f6}\u{95f4}\u{65e9}\u{4e0a}\u{4e5d}\u{70b9}\u{81f3}\u{4e0b}\u{5348}\u{4e94}\u{70b9}",
        ];
        assert!(
            zh_expected.contains(&zh_out.text.as_str()),
            "unexpected zh transcript: {:?}",
            zh_out.text
        );
        assert_eq!(zh_out.language.as_deref(), Some("zh"));

        let en = read_wav("en.wav");
        let en_out =
            transcribe_sensevoice_pcm_cached(&en, &pack, Some("en"), None).expect("en transcribe");
        eprintln!(
            "sensevoice en: {:?} (lang {:?})",
            en_out.text, en_out.language
        );
        let en_reference =
            "the tribal chieftain called for the boy and presented him with fifty pieces of gold";
        if pack.to_string_lossy().contains("fp16") {
            // fp16 must reproduce the PyTorch reference transcript exactly.
            assert_eq!(en_out.text, en_reference);
        } else {
            // Quantized packs may differ at homophone level (e.g. "chieftain"
            // vs "chief then"); gate on WER instead of byte equality.
            let wer = crate::metrics::wer(en_reference, &en_out.text);
            assert!(
                wer <= 0.15,
                "quantized en WER {wer:.3} too high: {:?}",
                en_out.text
            );
        }

        // auto (LID): zh clip must detect zh.
        let auto_out =
            transcribe_sensevoice_pcm_cached(&zh, &pack, None, None).expect("auto transcribe");
        eprintln!(
            "sensevoice auto: {:?} (lang {:?})",
            auto_out.text, auto_out.language
        );
        assert_eq!(auto_out.language.as_deref(), Some("zh"));
    }

    /// RTF probe: warm the prepared runtime once, then time a decode of the
    /// en clip. Prints seconds-of-audio / seconds-of-compute. Run with
    /// OPENASR_GGML_BACKEND=metal for the Metal figure.
    #[test]
    #[ignore = "requires local sensevoice pack + audio clips; prints RTF"]
    fn sensevoice_rtf_probe() {
        let pack = PathBuf::from(std::env::var("SENSEVOICE_PACK").expect("SENSEVOICE_PACK"));
        let audio_dir =
            PathBuf::from(std::env::var("SENSEVOICE_AUDIO_DIR").expect("SENSEVOICE_AUDIO_DIR"));
        let bytes = std::fs::read(audio_dir.join("en.wav")).expect("wav");
        let mut samples = Vec::new();
        let mut i = 12;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            if id == b"data" {
                let start = i + 8;
                let end = (start + size).min(bytes.len());
                samples = bytes[start..end]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                break;
            }
            i += 8 + size + (size & 1);
        }
        let duration = samples.len() as f32 / 16_000.0;
        // Warm (load + first decode), then measure steady-state decodes.
        transcribe_sensevoice_pcm_cached(&samples, &pack, Some("en"), None).expect("warm");
        let runs = 3;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            transcribe_sensevoice_pcm_cached(&samples, &pack, Some("en"), None).expect("run");
        }
        let per_run = start.elapsed().as_secs_f32() / runs as f32;
        eprintln!(
            "sensevoice rtf probe: audio {duration:.2}s, decode {per_run:.3}s, RTF = {:.4}",
            per_run / duration
        );
    }
}
