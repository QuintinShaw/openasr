//! wav2vec2-ctc transcription core: raw-waveform frontend → encoder graph → CTC
//! greedy decode → char detokenize. The WER gate test exercises the whole
//! pipeline on the fixed LibriSpeech clip.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::PhraseBiasConfig;
use crate::WAV2VEC2_CTC_DECODE_POLICY_ID;
use crate::api::backend::WordTimestamp;
use crate::arch::block_stack::{OpenAsrBlockKind, OpenAsrOrchestrationShape};
use crate::arch::shape_orchestrator::{
    LayerCountResolver, OpenAsrStageRole, StageBuildPlan, validate_stage_against_descriptor,
};
use crate::arch::{OpenAsrArchitectureRegistry, WAV2VEC2_CTC_GGML_ARCHITECTURE_ID};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufMetadata, GgufTensorDataReader};
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
use crate::{NativeAsrSession, WAV2VEC2_CTC_GGML_ADAPTER_ID};

use super::encoder_graph::Wav2Vec2CtcEncoderGraph;
use super::encoder_weights::{Wav2Vec2EncoderWeights, load_wav2vec2_ctc_encoder_weights};
use super::frontend::Wav2Vec2Frontend;
use super::graph_config::wav2vec2_ctc_encoder_graph_config;
use super::runtime_contract::{
    Wav2Vec2CtcExecutionMetadata, parse_wav2vec2_ctc_execution_metadata,
};
use super::tokenizer::Wav2Vec2Tokenizer;

type Wav2Vec2RuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static WAV2VEC2_CTC_RUNTIME_BY_KEY: RefCell<HashMap<Wav2Vec2RuntimeCacheKey, Wav2Vec2CtcPreparedRuntime>> =
        RefCell::new(HashMap::new());
}

/// Resolves the wav2vec2 block-stack `layer_count_hparam` against the parsed
/// metadata (HONESTY CONTRACT: reads the named hparam, NOT `weights.layers.len()`).
struct Wav2Vec2LayerCountResolver {
    n_layers: usize,
}

impl LayerCountResolver for Wav2Vec2LayerCountResolver {
    fn resolve_layer_count(&self, hparam_key: &'static str) -> Option<usize> {
        match hparam_key {
            "wav2vec2.n_layers" => Some(self.n_layers),
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

struct Wav2Vec2CtcPreparedRuntime {
    tokenizer: Wav2Vec2Tokenizer,
    graph: Wav2Vec2CtcEncoderGraph,
}

const WAV2VEC2_CTC_STREAMING_EXECUTOR_ID: &str = "wav2vec2-ctc-ggml-snapshot-streaming-executor-v1";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Wav2Vec2CtcTranscription {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

/// Transcribe 16 kHz mono f32 PCM through the wav2vec2-ctc pipeline. `pack_path`
/// is the on-disk runtime pack the encoder memory-maps to bind 2-D linears
/// zero-copy.
pub(crate) fn transcribe_wav2vec2_ctc_pcm(
    reader: &GgufTensorDataReader,
    gguf_metadata: &GgufMetadata,
    samples: &[f32],
    pack_path: &std::path::Path,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
) -> Result<Wav2Vec2CtcTranscription, String> {
    let metadata =
        parse_wav2vec2_ctc_execution_metadata(gguf_metadata).map_err(|e| e.to_string())?;
    let tokenizer = Wav2Vec2Tokenizer::from_metadata(gguf_metadata)?;
    let frontend = Wav2Vec2Frontend::new();
    let audio = frontend
        .features_from_samples(samples)
        .map_err(|e| e.to_string())?;
    let weights =
        load_wav2vec2_ctc_encoder_weights(reader, &metadata).map_err(|e| e.to_string())?;
    validate_wav2vec2_block_stack(metadata, &weights)?;
    let mut graph = Wav2Vec2CtcEncoderGraph::new(&weights, metadata, Some(pack_path))
        .map_err(|e| e.to_string())?;
    let output = graph.encode(&audio.samples).map_err(|e| e.to_string())?;
    decode_wav2vec2_output(
        output,
        &tokenizer,
        phrase_bias,
        word_timestamps,
        samples.len() as f32 / 16_000.0_f32,
    )
}

fn transcribe_wav2vec2_ctc_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
) -> Result<Wav2Vec2CtcTranscription, String> {
    let backend = wav2vec2_ctc_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &WAV2VEC2_CTC_RUNTIME_BY_KEY,
        key,
        || build_wav2vec2_prepared_runtime(pack_path),
        |runtime| runtime.transcribe(samples, phrase_bias, word_timestamps),
    )
}

fn decode_wav2vec2_ctc_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<CtcGreedyDecodeResult, String> {
    let backend = wav2vec2_ctc_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &WAV2VEC2_CTC_RUNTIME_BY_KEY,
        key,
        || build_wav2vec2_prepared_runtime(pack_path),
        |runtime| runtime.decode_result(samples, phrase_bias),
    )
}

fn build_wav2vec2_prepared_runtime(pack_path: &Path) -> Result<Wav2Vec2CtcPreparedRuntime, String> {
    let reader = GgufTensorDataReader::from_path(pack_path).map_err(|e| e.to_string())?;
    let gguf_metadata =
        crate::ggml_runtime::read_gguf_metadata(pack_path).map_err(|e| e.to_string())?;
    let metadata =
        parse_wav2vec2_ctc_execution_metadata(&gguf_metadata).map_err(|e| e.to_string())?;
    let tokenizer = Wav2Vec2Tokenizer::from_metadata(&gguf_metadata)?;
    let weights =
        load_wav2vec2_ctc_encoder_weights(&reader, &metadata).map_err(|e| e.to_string())?;
    validate_wav2vec2_block_stack(metadata, &weights)?;
    let graph = Wav2Vec2CtcEncoderGraph::new(&weights, metadata, Some(pack_path))
        .map_err(|e| e.to_string())?;
    Ok(Wav2Vec2CtcPreparedRuntime { tokenizer, graph })
}

impl Wav2Vec2CtcPreparedRuntime {
    fn transcribe(
        &mut self,
        samples: &[f32],
        phrase_bias: Option<&PhraseBiasConfig>,
        word_timestamps: bool,
    ) -> Result<Wav2Vec2CtcTranscription, String> {
        let result = self.decode_result(samples, phrase_bias)?;
        wav2vec2_ctc_result_to_transcription(
            result,
            &self.tokenizer,
            word_timestamps,
            samples.len() as f32 / 16_000.0_f32,
        )
    }

    fn decode_result(
        &mut self,
        samples: &[f32],
        phrase_bias: Option<&PhraseBiasConfig>,
    ) -> Result<CtcGreedyDecodeResult, String> {
        let frontend = Wav2Vec2Frontend::new();
        let audio = frontend
            .features_from_samples(samples)
            .map_err(|e| e.to_string())?;
        let output = self
            .graph
            .encode(&audio.samples)
            .map_err(|e| e.to_string())?;
        decode_wav2vec2_ctc_result(&output, &self.tokenizer, phrase_bias)
    }
}

fn validate_wav2vec2_block_stack(
    metadata: Wav2Vec2CtcExecutionMetadata,
    weights: &Wav2Vec2EncoderWeights,
) -> Result<(), String> {
    // Make the block-stack descriptor LOAD-BEARING (P4 exit-signal honesty): the
    // Ctc encoder stack this pack materialized must agree with the wav2vec2
    // descriptor's declared shape / block kind / tensor scope / layer count.
    let wav2vec2_block_stack = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(WAV2VEC2_CTC_GGML_ARCHITECTURE_ID)
        .and_then(|descriptor| descriptor.block_stack);
    validate_stage_against_descriptor(
        WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
        wav2vec2_block_stack.as_ref(),
        OpenAsrStageRole::Encoder,
        OpenAsrOrchestrationShape::Ctc,
        StageBuildPlan {
            block_kind: OpenAsrBlockKind::Wav2Vec2PostNormEncoderLayer,
            tensor_name_scope: "enc.blk",
            family_layer_count: weights.layers.len(),
        },
        &Wav2Vec2LayerCountResolver {
            n_layers: metadata.n_layers,
        },
    )
    .map_err(|error| format!("wav2vec2-ctc encoder block-stack descriptor mismatch: {error:?}"))?;
    Ok(())
}

fn decode_wav2vec2_output(
    output: super::encoder_graph::Wav2Vec2CtcEncoderOutput,
    tokenizer: &Wav2Vec2Tokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
    duration_seconds: f32,
) -> Result<Wav2Vec2CtcTranscription, String> {
    let result = decode_wav2vec2_ctc_result(&output, tokenizer, phrase_bias)?;
    wav2vec2_ctc_result_to_transcription(result, tokenizer, word_timestamps, duration_seconds)
}

pub(crate) fn decode_wav2vec2_ctc_result(
    output: &super::encoder_graph::Wav2Vec2CtcEncoderOutput,
    tokenizer: &Wav2Vec2Tokenizer,
    phrase_bias: Option<&PhraseBiasConfig>,
) -> Result<CtcGreedyDecodeResult, String> {
    let frame_logits: Vec<&[f32]> = (0..output.frame_count)
        .map(|f| &output.logits[f * output.vocab_size..(f + 1) * output.vocab_size])
        .collect();
    let detok = |ids: &[u32]| tokenizer.decode(ids);
    run_builtin_ctc_decode_policy(
        WAV2VEC2_CTC_DECODE_POLICY_ID,
        &frame_logits,
        output.vocab_size,
        phrase_bias,
        tokenizer,
        &detok,
        ctc_err_to_string,
        registry_err_to_string,
    )
}

pub(crate) fn wav2vec2_ctc_result_to_transcription(
    result: CtcGreedyDecodeResult,
    tokenizer: &Wav2Vec2Tokenizer,
    word_timestamps: bool,
    duration_seconds: f32,
) -> Result<Wav2Vec2CtcTranscription, String> {
    let words = if word_timestamps {
        tokenizer.word_timestamps_from_token_spans(
            &result.token_spans,
            duration_seconds,
            result.frame_count,
        )?
    } else {
        Vec::new()
    };
    Ok(Wav2Vec2CtcTranscription {
        text: result.text,
        words,
    })
}

/// Dedicated GgmlAsrExecutor for wav2vec2-ctc (DedicatedRuntimeExecutorV1).
#[derive(Debug, Clone, Default)]
pub(crate) struct Wav2Vec2CtcGgmlExecutor;

impl Wav2Vec2CtcGgmlExecutor {
    fn execute_ctc_result(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        decode_wav2vec2_ctc_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
            request.request_options.phrase_bias.as_ref(),
        )
        .map_err(fail)
    }
}

impl GgmlAsrExecutor for Wav2Vec2CtcGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        true
    }

    fn execute(
        &self,
        request: &crate::models::ggml_asr_executor::GgmlAsrExecutionRequest,
    ) -> Result<
        crate::models::ggml_asr_executor::GgmlAsrExecutionResult,
        crate::models::ggml_asr_executor::GgmlAsrExecutionError,
    > {
        use crate::api::backend::{Segment, Transcription};
        use crate::models::ggml_asr_executor::{GgmlAsrExecutionError, GgmlAsrExecutionResult};
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::WAV2VEC2_CTC_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Fail-closed: validate the runtime source path before touching the pack
        // (Gate-0 preflight), then run the cached prepared-runtime path.
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let output = transcribe_wav2vec2_ctc_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
            request.request_options.phrase_bias.as_ref(),
            request.request_options.word_timestamps,
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
                words: output.words,
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text: output.text,
                segments,
                longform: None,
                language: None,
            },
            carry_context: None,
        })
    }
}

impl GgmlAsrStreamingExecutor for Wav2Vec2CtcGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        WAV2VEC2_CTC_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        if request.selected_family.adapter_id != WAV2VEC2_CTC_GGML_ADAPTER_ID {
            return Err(GgmlAsrExecutionError::ExecutorFailed {
                executor_id: WAV2VEC2_CTC_STREAMING_EXECUTOR_ID,
                adapter_id: request.selected_family.adapter_id,
                reason: format!(
                    "wav2vec2-ctc streaming executor requires adapter '{WAV2VEC2_CTC_GGML_ADAPTER_ID}', got '{}'",
                    request.selected_family.adapter_id
                ),
            });
        }

        // Gate-off → snapshot driver; gate-on → incremental/windowed
        // driver. The FINAL is byte-identical either way; only partials differ.
        let driver = build_ctc_streaming_driver(
            self.clone(),
            WAV2VEC2_CTC_STREAMING_EXECUTOR_ID,
            WAV2VEC2_CTC_GGML_ADAPTER_ID,
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            Wav2Vec2CtcGgmlExecutor::execute_ctc_result,
            <Wav2Vec2CtcGgmlExecutor as GgmlAsrExecutor>::execute,
        );
        let session = GgmlAsrStreamingTranscriptSession::new(
            WAV2VEC2_CTC_STREAMING_EXECUTOR_ID,
            request,
            driver,
        )?;
        Ok(Box::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn read_wav_mono_16k(path: &Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        let mut i = 12;
        while i + 8 <= bytes.len() {
            let id = &bytes[i..i + 4];
            let size = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
                as usize;
            if id == b"data" {
                let start = i + 8;
                let end = (start + size).min(bytes.len());
                return Some(
                    bytes[start..end]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                        .collect(),
                );
            }
            i += 8 + size + (size & 1);
        }
        None
    }

    fn wer(reference: &str, hypothesis: &str) -> f64 {
        let norm = |s: &str| -> Vec<String> {
            s.to_uppercase()
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
                .split_whitespace()
                .map(str::to_string)
                .collect()
        };
        let r = norm(reference);
        let h = norm(hypothesis);
        if r.is_empty() {
            return if h.is_empty() { 0.0 } else { 1.0 };
        }
        let mut prev: Vec<usize> = (0..=h.len()).collect();
        let mut cur = vec![0usize; h.len() + 1];
        for (i, rw) in r.iter().enumerate() {
            cur[0] = i + 1;
            for (j, hw) in h.iter().enumerate() {
                let cost = usize::from(rw != hw);
                cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[h.len()] as f64 / r.len() as f64
    }

    fn roots() -> [std::path::PathBuf; 1] {
        [Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")]
    }

    fn find_rel(rel: &str) -> Option<std::path::PathBuf> {
        roots().iter().map(|r| r.join(rel)).find(|p| p.exists())
    }

    fn pack_and_clip() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let pack =
            find_rel("tmp/models/wav2vec2-base-960h-source/openasr/wav2vec2-base-960h-q4k.oasr")?;
        let clip = find_rel("tmp/audio/librispeech/237-134500-0000.wav")?;
        Some((pack, clip))
    }

    /// Run the LibriSpeech-clean clip through one sibling pack and assert a low
    /// WER. Skipped when the pack/clip are absent. Exercises the large-variant
    /// branches (feat_extract_norm=layer, do_stable_layer_norm, conv_bias).
    fn assert_sibling_transcribes(pack_rel: &str) {
        let (Some(pack), Some(clip)) = (
            find_rel(pack_rel),
            find_rel("tmp/audio/librispeech/237-134500-0000.wav"),
        ) else {
            eprintln!("skipping: sibling pack or clip absent ({pack_rel})");
            return;
        };
        let reference = "FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS \
                         DIVORCE CASE THE ANGRIER HE GREW";
        let samples = read_wav_mono_16k(&clip).expect("wav");
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");
        let hypothesis =
            transcribe_wav2vec2_ctc_pcm(&reader, &metadata, &samples, &pack, None, false)
                .expect("decode")
                .text;
        let wer = wer(reference, &hypothesis);
        eprintln!("{pack_rel}: hypothesis {hypothesis:?}\nWER = {wer:.3}");
        assert!(
            wer <= 0.15,
            "WER {wer:.3} too high; hypothesis: {hypothesis:?}"
        );
    }

    /// HuBERT-large-ls960-ft: feat_extract_norm=layer, stable (pre-norm) encoder,
    /// conv_bias=true. Same grouped pos-conv as base-960h.
    #[test]
    fn hubert_large_ls960_ft_transcribes_clip_when_present() {
        assert_sibling_transcribes(
            "tmp/models/hubert-large-ls960-ft-source/openasr/hubert-large-ls960-ft-q4k.oasr",
        );
    }

    /// wav2vec2-large-960h-lv60-self: same large config as HuBERT
    /// (layer/stable/conv_bias) under the `wav2vec2.` prefix.
    #[test]
    fn wav2vec2_large_lv60_transcribes_clip_when_present() {
        assert_sibling_transcribes(
            "tmp/models/wav2vec2-large-960h-lv60-self-source/openasr/\
             wav2vec2-large-960h-lv60-self-q4k.oasr",
        );
    }

    /// The WER gate: wav2vec2-base-960h transcribes the fixed clip. Skipped when
    /// the pack/clip are absent. Prints the hypothesis + WER so the per-stage
    /// pipeline can be bisected if WER is high.
    #[test]
    fn wav2vec2_ctc_transcribes_librispeech_clip_when_present() {
        let Some((pack, clip)) = pack_and_clip() else {
            eprintln!("skipping: wav2vec2 pack or clip absent");
            return;
        };
        let reference = "FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS \
                         DIVORCE CASE THE ANGRIER HE GREW";
        let samples = read_wav_mono_16k(&clip).expect("wav");
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        let hypothesis =
            transcribe_wav2vec2_ctc_pcm(&reader, &metadata, &samples, &pack, None, false)
                .expect("decode")
                .text;
        let wer = wer(reference, &hypothesis);
        eprintln!("wav2vec2-ctc hypothesis: {hypothesis:?}\nWER = {wer:.3}");
        assert!(
            wer <= 0.15,
            "WER {wer:.3} too high; hypothesis: {hypothesis:?}"
        );
    }
}
