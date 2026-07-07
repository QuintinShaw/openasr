//! parakeet-tdt transcription core: frontend -> encoder graph (with in-graph
//! joint encoder projection) -> host TDT greedy decode -> detokenize.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutor, GgmlAsrStreamingExecutor,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::incremental_streaming_driver::{
    STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT, build_seq2seq_streaming_session,
};
use crate::models::parakeet_ctc::frontend::ParakeetFrontend;
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, with_thread_local_cached_mut_by_key,
};
use crate::{NativeAsrSession, PARAKEET_TDT_GGML_ADAPTER_ID};

use super::encoder_graph::{ParakeetTdtEncoderGraph, ParakeetTdtMelFeatures};
use super::encoder_weights::{
    load_parakeet_tdt_encoder_weights, load_parakeet_tdt_joint_weights,
    load_parakeet_tdt_predictor_weights,
};
use super::graph_config::parakeet_tdt_encoder_graph_config;
use super::greedy::{ParakeetTdtJoint, tdt_greedy_decode};
use super::predictor::ParakeetTdtPredictor;
use super::runtime_contract::{
    ParakeetTdtExecutionMetadata, parse_parakeet_tdt_execution_metadata,
};
use super::tokenizer::ParakeetTdtTokenizer;

type ParakeetTdtRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static PARAKEET_TDT_RUNTIME_BY_KEY: RefCell<HashMap<ParakeetTdtRuntimeCacheKey, ParakeetTdtPreparedRuntime>> =
        RefCell::new(HashMap::new());
}

const PARAKEET_TDT_STREAMING_EXECUTOR_ID: &str = "parakeet-tdt-ggml-redecode-streaming-executor-v1";

struct ParakeetTdtPreparedRuntime {
    metadata: ParakeetTdtExecutionMetadata,
    tokenizer: ParakeetTdtTokenizer,
    graph: ParakeetTdtEncoderGraph,
    predictor: ParakeetTdtPredictor,
    joint: ParakeetTdtJoint,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParakeetTdtTranscription {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

fn build_parakeet_tdt_prepared_runtime(
    pack_path: &Path,
) -> Result<ParakeetTdtPreparedRuntime, String> {
    let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(pack_path)
        .map_err(|e| e.to_string())?;
    let gguf_metadata =
        crate::ggml_runtime::read_gguf_metadata(pack_path).map_err(|e| e.to_string())?;
    let metadata =
        parse_parakeet_tdt_execution_metadata(&gguf_metadata).map_err(|e| e.to_string())?;
    let tokenizer = ParakeetTdtTokenizer::from_metadata(&gguf_metadata)?;
    let weights =
        load_parakeet_tdt_encoder_weights(&reader, &metadata).map_err(|e| e.to_string())?;
    let graph = ParakeetTdtEncoderGraph::new(&weights, metadata, Some(pack_path))
        .map_err(|e| e.to_string())?;
    let predictor_weights =
        load_parakeet_tdt_predictor_weights(&reader, &metadata).map_err(|e| e.to_string())?;
    let predictor =
        ParakeetTdtPredictor::new(predictor_weights, metadata.pred_hidden, metadata.vocab_size);
    let joint_weights =
        load_parakeet_tdt_joint_weights(&reader, &metadata).map_err(|e| e.to_string())?;
    let joint = ParakeetTdtJoint::new(joint_weights, metadata.joint_hidden);
    Ok(ParakeetTdtPreparedRuntime {
        metadata,
        tokenizer,
        graph,
        predictor,
        joint,
    })
}

impl ParakeetTdtPreparedRuntime {
    fn transcribe(
        &mut self,
        samples: &[f32],
        word_timestamps: bool,
    ) -> Result<ParakeetTdtTranscription, String> {
        let frontend = ParakeetFrontend::with_n_mels(self.metadata.n_mels);
        let features = frontend
            .features_from_samples(samples)
            .map_err(|e| e.to_string())?;
        let output = self
            .graph
            .encode(&ParakeetTdtMelFeatures {
                data: features.data,
                n_frames: features.n_frames,
                n_mels: features.n_mels,
            })
            .map_err(|e| e.to_string())?;
        if output.joint_hidden != self.metadata.joint_hidden {
            return Err(format!(
                "parakeet-tdt encoder emitted joint width {}, metadata declares {}",
                output.joint_hidden, self.metadata.joint_hidden
            ));
        }
        let emitted = tdt_greedy_decode(
            &output.features,
            output.frame_count,
            &self.metadata,
            &self.predictor,
            &self.joint,
        )?;
        let token_ids: Vec<u32> = emitted.iter().map(|token| token.token_id).collect();
        let text = self.tokenizer.decode(&token_ids)?;
        let words = if word_timestamps {
            self.tokenizer.word_timestamps_from_emitted(
                &emitted,
                samples.len() as f32 / 16_000.0_f32,
                output.frame_count,
            )?
        } else {
            Vec::new()
        };
        Ok(ParakeetTdtTranscription { text, words })
    }
}

/// Transcribe 16 kHz mono f32 PCM through a cached prepared runtime keyed by
/// `(canonical pack path, backend)`.
pub(crate) fn transcribe_parakeet_tdt_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    word_timestamps: bool,
) -> Result<ParakeetTdtTranscription, String> {
    let backend = parakeet_tdt_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &PARAKEET_TDT_RUNTIME_BY_KEY,
        key,
        || build_parakeet_tdt_prepared_runtime(pack_path),
        |runtime| runtime.transcribe(samples, word_timestamps),
    )
}

/// Dedicated GgmlAsrExecutor for parakeet-tdt (DedicatedRuntimeExecutorV1).
#[derive(Debug, Clone, Default)]
pub(crate) struct ParakeetTdtGgmlExecutor;

impl GgmlAsrExecutor for ParakeetTdtGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::PARAKEET_TDT_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // The TDT greedy loop does not apply vocab-logit boosts yet (the
        // xasr transducer precedent); keep the capability honest.
        false
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
            executor_id: crate::arch::PARAKEET_TDT_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Fail-closed: validate the runtime source path before touching the
        // pack (Gate-0 preflight), then run the cached prepared-runtime path.
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let output = transcribe_parakeet_tdt_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
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

impl GgmlAsrStreamingExecutor for ParakeetTdtGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        PARAKEET_TDT_STREAMING_EXECUTOR_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        // Partials re-decode the trailing window through the same offline
        // pipeline as the FINAL (the shared re-decode session every
        // non-frame-sync family uses); the FINAL stays byte-identical to
        // `execute()`. TDT's frame-synchronous decode makes a true
        // append-only frame-sync driver possible later; the offline re-decode
        // keeps v1 honest and simple.
        build_seq2seq_streaming_session(
            self.clone(),
            PARAKEET_TDT_STREAMING_EXECUTOR_ID,
            PARAKEET_TDT_GGML_ADAPTER_ID,
            "parakeet-tdt",
            request,
            STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
            <ParakeetTdtGgmlExecutor as GgmlAsrExecutor>::execute,
        )
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

    /// The exit-signal gate: parakeet-tdt-0.6b-v3 transcribes the bundled JFK
    /// clip coherently, with native word timestamps in order. Skipped when
    /// the pack is absent (tmp/ is host-local).
    #[test]
    fn parakeet_tdt_transcribes_jfk_clip_when_pack_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let pack = root
            .join("tmp/models/parakeet-tdt-0.6b-v3-source/openasr/parakeet-tdt-0.6b-v3-fp16.oasr");
        let clip = root.join("fixtures/jfk.wav");
        if !pack.exists() || !clip.exists() {
            eprintln!("skipping: parakeet-tdt pack or jfk clip absent");
            return;
        }
        let samples = read_wav_mono_16k(&clip).expect("wav");
        let output = transcribe_parakeet_tdt_pcm_cached(&samples, &pack, true).expect("transcribe");
        eprintln!("parakeet-tdt hypothesis: {:?}", output.text);
        eprintln!("parakeet-tdt words: {:?}", output.words);
        let lowered = output.text.to_lowercase();
        assert!(
            lowered.contains("ask not what your country can do for you"),
            "unexpected transcript: {:?}",
            output.text
        );
        assert!(!output.words.is_empty(), "native word timestamps expected");
        for pair in output.words.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "word starts must be monotonic: {:?}",
                output.words
            );
        }
    }
}
