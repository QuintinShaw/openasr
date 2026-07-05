//! X-ASR Zipformer transducer runtime: fbank -> cache-aware encoder chunks ->
//! stateless RNN-T greedy decode.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::NativeAsrSession;
use crate::PhraseBiasConfig;
use crate::api::backend::{Segment, Transcription, WordTimestamp};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufMetadata, GgufTensorDataReader};
use crate::models::frame_sync_streaming_driver::FrameSyncStreamingTranscriptDriver;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
    GgmlAsrStreamingExecutor, GgmlAsrStreamingSessionRequest,
};
use crate::models::ggml_streaming_session::GgmlAsrStreamingTranscriptSession;
use crate::models::thread_local_runtime_cache::{
    canonical_runtime_cache_path, with_thread_local_cached_mut_by_key,
};

use super::frontend::{XASR_FINAL_FLUSH_TAIL_PAD_SAMPLES, XASR_SAMPLE_RATE_HZ};
use super::graph_config::xasr_zipformer_encoder_graph_config;
use super::runtime::{XasrZipformerPreparedRuntime, checkout_prepared_runtime};
use super::streaming_decoder::XasrIncrementalDecoder;

type XasrRuntimeCacheKey = (PathBuf, GgmlCpuGraphBackend);

thread_local! {
    static XASR_ZIPFORMER_RUNTIME_BY_KEY: RefCell<HashMap<XasrRuntimeCacheKey, XasrZipformerPreparedRuntime>> =
        RefCell::new(HashMap::new());
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrZipformerTranscription {
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

fn transcription_from_decode(
    runtime: &XasrZipformerPreparedRuntime,
    result: crate::models::xasr_zipformer::greedy::XasrGreedyDecodeResult,
    word_timestamps: bool,
    duration_seconds: f32,
) -> Result<XasrZipformerTranscription, String> {
    let words = if word_timestamps {
        // `encoder_frames` covers the tail-padded audio, so map frames against
        // the padded duration and clamp back into the real clip: words inside
        // real speech keep their true times, and a token emitted in the pad
        // region (terminal punctuation) lands at the audio end.
        let padded_duration_seconds = if duration_seconds > 0.0 {
            duration_seconds + XASR_FINAL_FLUSH_TAIL_PAD_SAMPLES as f32 / XASR_SAMPLE_RATE_HZ as f32
        } else {
            duration_seconds
        };
        let mut words = runtime.tokenizer().word_timestamps_from_emission_frames(
            &result.token_ids,
            &result.emit_frames,
            &result.emit_probabilities,
            result.encoder_frames,
            padded_duration_seconds,
        )?;
        for word in &mut words {
            word.start = word.start.min(duration_seconds);
            word.end = word.end.min(duration_seconds);
        }
        words
    } else {
        Vec::new()
    };
    Ok(XasrZipformerTranscription {
        text: result.text,
        words,
    })
}

pub(crate) fn transcribe_xasr_zipformer_pcm(
    reader: &GgufTensorDataReader,
    gguf_metadata: &GgufMetadata,
    samples: &[f32],
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
) -> Result<XasrZipformerTranscription, String> {
    if phrase_bias.is_some() {
        return Err("xasr-zipformer phrase bias is not supported".to_string());
    }
    let mut runtime = XasrZipformerPreparedRuntime::from_reader_metadata(reader, gguf_metadata)?;
    let result = runtime.transcribe(samples)?;
    transcription_from_decode(
        &runtime,
        result,
        word_timestamps,
        pcm_duration_seconds(samples),
    )
}

fn transcribe_xasr_zipformer_pcm_cached(
    samples: &[f32],
    pack_path: &Path,
    phrase_bias: Option<&PhraseBiasConfig>,
    word_timestamps: bool,
) -> Result<XasrZipformerTranscription, String> {
    if phrase_bias.is_some() {
        return Err("xasr-zipformer phrase bias is not supported".to_string());
    }
    let backend = xasr_zipformer_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    with_thread_local_cached_mut_by_key(
        &XASR_ZIPFORMER_RUNTIME_BY_KEY,
        key,
        || XasrZipformerPreparedRuntime::load(pack_path),
        |runtime| {
            let result = runtime.transcribe(samples)?;
            transcription_from_decode(
                runtime,
                result,
                word_timestamps,
                pcm_duration_seconds(samples),
            )
        },
    )
}

fn pcm_duration_seconds(samples: &[f32]) -> f32 {
    samples.len() as f32 / 16_000.0_f32
}

fn reject_xasr_phrase_bias(
    selected_family: &crate::GgmlFamilyAdapterDescriptor,
) -> Result<(), GgmlAsrExecutionError> {
    Err(GgmlAsrExecutionError::PhraseBiasUnsupported {
        adapter_id: selected_family.adapter_id,
        model_family: selected_family.model_family,
    })
}

#[derive(Debug, Clone, Default)]
pub(crate) struct XasrZipformerGgmlExecutor;

impl GgmlAsrExecutor for XasrZipformerGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        false
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        if request
            .request_options
            .phrase_bias
            .as_ref()
            .is_some_and(|phrase_bias| !phrase_bias.is_empty())
        {
            reject_xasr_phrase_bias(&request.selected_family)?;
        }
        let fail = |reason: String| {
            GgmlAsrExecutionError::executor_failed(
                crate::arch::XASR_ZIPFORMER_EXECUTOR_COMPONENT_ID,
                request.selected_family.adapter_id,
                reason,
            )
        };
        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        let output = transcribe_xasr_zipformer_pcm_cached(
            &request.prepared_audio.samples_f32,
            &request.runtime_source_path,
            request.request_options.phrase_bias.as_ref(),
            request.request_options.word_timestamps,
        )
        .map_err(fail)?;
        let duration = pcm_duration_seconds(&request.prepared_audio.samples_f32);
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

impl GgmlAsrStreamingExecutor for XasrZipformerGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID
    }

    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        let fail = |reason: String| {
            GgmlAsrExecutionError::executor_failed(
                crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
                request.selected_family.adapter_id,
                reason,
            )
        };
        if request.selected_family.adapter_id != crate::XASR_ZIPFORMER_GGML_ADAPTER_ID {
            return Err(fail(format!(
                "xasr-zipformer streaming executor requires adapter '{}', got '{}'",
                crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
                request.selected_family.adapter_id
            )));
        }
        if request
            .request_options
            .phrase_bias
            .as_ref()
            .is_some_and(|phrase_bias| !phrase_bias.is_empty())
        {
            reject_xasr_phrase_bias(&request.selected_family)?;
        }

        request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        // The pool key and the prepared encoder graph bake the backend at
        // checkout, so the session's execution preference must be installed
        // before the runtime is selected.
        let _backend_guard = crate::ggml_runtime::install_request_backend_override(
            request.backend_preference.request_backend_override(),
        );
        let runtime = checkout_prepared_runtime(&request.runtime_source_path).map_err(fail)?;
        let session_suffix = &request.session_context.session_id.0;
        let decoder = XasrIncrementalDecoder::new(
            request,
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            runtime,
        );
        let driver = FrameSyncStreamingTranscriptDriver::new(
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            format!("utt_{session_suffix}"),
            format!("seg_{session_suffix}"),
            1,
            decoder,
        );
        let session = GgmlAsrStreamingTranscriptSession::new(
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            request,
            driver,
        )?;
        Ok(Box::new(session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::read_gguf_metadata;

    #[test]
    fn missing_pack_fails_before_executor_work() {
        let error = XasrZipformerPreparedRuntime::load(Path::new("/tmp/missing-xasr.oasr"))
            .expect_err("missing pack should fail");
        assert!(!error.trim().is_empty());
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn xasr_word_timestamps_align_with_real_speech_when_pack_present() {
        let pack = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr word timestamp test",
            "xasr word timestamp test",
        )
        .expect("sample wav should load");
        let duration_seconds = samples.len() as f32 / 16_000.0;
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let output = transcribe_xasr_zipformer_pcm(&reader, &metadata, &samples, None, true)
            .expect("xasr word timestamps");

        assert!(!output.words.is_empty(), "real speech must yield words");
        let mut previous_start = 0.0_f32;
        for word in &output.words {
            assert!(word.start >= previous_start, "starts must be monotonic");
            assert!(word.end >= word.start);
            assert!(word.end <= duration_seconds + 0.05);
            previous_start = word.start;
            // The transducer path captures a joiner softmax probability for
            // every emission, so every word must carry a sane confidence.
            let confidence = word
                .confidence
                .expect("xasr words must carry confidence from emission probabilities");
            assert!((0.0..=1.0).contains(&confidence), "confidence {confidence}");
        }
        // The words are exactly the non-special decoded pieces, so modulo
        // whitespace they must reproduce the transcript.
        let joined = output
            .words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<String>();
        let despace = |text: &str| {
            text.chars()
                .filter(|ch| !ch.is_whitespace())
                .collect::<String>()
        };
        assert_eq!(despace(&joined), despace(&output.text));
    }

    #[test]
    #[ignore = "host-local: runs X-ASR executor on the local ONNX-derived pack and synthetic audio"]
    fn xasr_zipformer_executor_smoke_when_pack_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr fp16 pack absent at {}", pack.display());
            return;
        }
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let samples = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.05)
            .collect::<Vec<_>>();
        let output = transcribe_xasr_zipformer_pcm(&reader, &metadata, &samples, None, true)
            .expect("xasr executor smoke");
        assert!(output.text.is_char_boundary(output.text.len()));
    }
}
