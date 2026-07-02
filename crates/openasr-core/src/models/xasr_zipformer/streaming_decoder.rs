use crate::models::frame_sync_streaming_driver::IncrementalAudioDecoder;
use crate::models::ggml_asr_executor::{GgmlAsrExecutionError, GgmlAsrStreamingSessionRequest};
use crate::models::graph_runtime_config::install_request_inference_threads_override;

use super::frontend::{
    XASR_N_MELS, XasrFbankFeatures, XasrFbankFrontend, clean_frame_count_for_samples,
    earliest_sample_needed_for_frame, total_frame_count_for_samples,
};
use super::runtime::{PooledRuntime, XasrChunkedDecodeState};
use super::tokenizer::XasrStreamingDetokenizer;

const XASR_STREAMING_BASELINE_LEFT_CONTEXT_TOKENS: usize = 16;

pub(crate) struct XasrIncrementalDecoder {
    executor_id: &'static str,
    adapter_id: &'static str,
    request: GgmlAsrStreamingSessionRequest,
    runtime: PooledRuntime,
    decode_state: XasrChunkedDecodeState,
    audio: Vec<f32>,
    /// Samples drained from the front of `audio`; all sample/frame indices
    /// below stay absolute against the full stream.
    dropped_samples: usize,
    frontend: XasrFbankFrontend,
    /// Cached fbank rows for frames already free of right-edge reflection;
    /// those rows never change as audio grows, so each push only pays for
    /// newly clean frames instead of recomputing the whole buffer (O(n^2)).
    features: XasrFbankFeatures,
    /// Feature rows drained from the front of `features`; together with the
    /// audio drain a session holds O(1) memory however long an utterance runs.
    dropped_frames: usize,
    /// Exact streaming detokenizer state; `decoded_tokens` counts how many of
    /// `decode_state.emitted` have been fed, so each delta only detokenizes
    /// the NEW tokens instead of re-decoding the whole utterance history.
    detokenizer: XasrStreamingDetokenizer,
    decoded_tokens: usize,
}

impl XasrIncrementalDecoder {
    pub(super) fn new(
        request: &GgmlAsrStreamingSessionRequest,
        executor_id: &'static str,
        adapter_id: &'static str,
        runtime: PooledRuntime,
    ) -> Self {
        let decode_state = runtime.new_decode_state();
        Self {
            executor_id,
            adapter_id,
            request: request.clone(),
            runtime,
            decode_state,
            audio: Vec::new(),
            dropped_samples: 0,
            frontend: XasrFbankFrontend::new(),
            features: XasrFbankFeatures {
                data: Vec::new(),
                n_frames: 0,
                n_mels: XASR_N_MELS,
            },
            dropped_frames: 0,
            detokenizer: XasrStreamingDetokenizer::default(),
            decoded_tokens: 0,
        }
    }

    fn failed(&self, reason: impl Into<String>) -> GgmlAsrExecutionError {
        GgmlAsrExecutionError::executor_failed(self.executor_id, self.adapter_id, reason)
    }

    /// Extends the feature cache up to `target_total_frames` (an absolute
    /// frame count against the full stream).
    fn extend_feature_rows(
        &mut self,
        target_total_frames: usize,
    ) -> Result<(), GgmlAsrExecutionError> {
        let cached_total = self.dropped_frames + self.features.n_frames;
        if target_total_frames <= cached_total {
            return Ok(());
        }
        let rows = self
            .frontend
            .features_for_frame_range_from(
                &self.audio,
                self.dropped_samples,
                cached_total,
                target_total_frames,
            )
            .map_err(|error| self.failed(error.to_string()))?;
        self.features.data.extend_from_slice(&rows);
        self.features.n_frames = target_total_frames - self.dropped_frames;
        Ok(())
    }

    /// Drops feature rows the chunk loop consumed and audio samples no future
    /// fbank frame can read, keeping per-session memory constant. Draining is
    /// amortized: it only compacts once a meaningful prefix is dead.
    fn drain_consumed_prefix(&mut self) {
        const DRAIN_SLACK_FRAMES: usize = 96;
        const DRAIN_SLACK_SAMPLES: usize = 16 * 1024;
        let consumed = self.decode_state.consumed_feature_frames();
        if consumed >= DRAIN_SLACK_FRAMES {
            self.features.data.drain(..consumed * self.features.n_mels);
            self.features.n_frames -= consumed;
            self.decode_state.rebase_feature_frames(consumed);
            self.dropped_frames += consumed;
        }
        let next_frame = self.dropped_frames + self.features.n_frames;
        let keep_from = earliest_sample_needed_for_frame(next_frame);
        if keep_from > self.dropped_samples {
            let dead = (keep_from - self.dropped_samples).min(self.audio.len());
            if dead >= DRAIN_SLACK_SAMPLES {
                self.audio.drain(..dead);
                self.dropped_samples += dead;
            }
        }
    }

    fn process_available_chunks(
        &mut self,
        final_flush: bool,
    ) -> Result<String, GgmlAsrExecutionError> {
        if self.audio.is_empty() {
            return Ok(String::new());
        }
        let total_samples = self.dropped_samples + self.audio.len();
        let target_total_frames = if final_flush {
            total_frame_count_for_samples(total_samples)
        } else {
            clean_frame_count_for_samples(total_samples)
        };
        if target_total_frames == 0 {
            return Ok(String::new());
        }
        self.extend_feature_rows(target_total_frames)?;
        let executor_id = self.executor_id;
        let adapter_id = self.adapter_id;
        let new_tokens = self
            .runtime
            .decode_available_chunks(&mut self.decode_state, &self.features, final_flush)
            .map_err(|error| {
                GgmlAsrExecutionError::executor_failed(executor_id, adapter_id, error)
            })?;
        self.drain_consumed_prefix();
        if new_tokens == 0 {
            return Ok(String::new());
        }
        self.text_delta()
    }

    fn text_delta(&mut self) -> Result<String, GgmlAsrExecutionError> {
        let executor_id = self.executor_id;
        let adapter_id = self.adapter_id;
        let emitted = self.decode_state.emitted_token_ids();
        let stable_len = self.detokenizer.text().len();
        for &id in &emitted[self.decoded_tokens..] {
            self.detokenizer
                .push_token(self.runtime.tokenizer(), id)
                .map_err(|error| {
                    GgmlAsrExecutionError::executor_failed(executor_id, adapter_id, error)
                })?;
        }
        self.decoded_tokens = emitted.len();
        Ok(self.detokenizer.text()[stable_len..].to_string())
    }

    fn rebase_decode_baseline(&mut self) {
        let dropped = self.decode_state.rebase_decoded_emitted_history(
            self.decoded_tokens,
            XASR_STREAMING_BASELINE_LEFT_CONTEXT_TOKENS,
        );
        self.decoded_tokens -= dropped;
        self.detokenizer.rebase_preserving_boundary_context();
        debug_assert_eq!(self.decoded_tokens, self.decode_state.emitted_history_len());
    }
}

impl IncrementalAudioDecoder for XasrIncrementalDecoder {
    fn accept_samples(&mut self, samples: &[f32]) -> Result<String, GgmlAsrExecutionError> {
        if samples.iter().any(|value| !value.is_finite()) {
            return Err(self.failed("xasr streaming requires finite audio samples"));
        }
        self.audio.extend_from_slice(samples);
        let _thread_override = install_request_inference_threads_override(
            self.request.request_options.inference_threads,
        );
        self.process_available_chunks(false)
    }

    fn finish(&mut self) -> Result<String, GgmlAsrExecutionError> {
        let _thread_override = install_request_inference_threads_override(
            self.request.request_options.inference_threads,
        );
        self.process_available_chunks(true)
    }

    fn reset(&mut self) {
        self.audio.clear();
        self.dropped_samples = 0;
        self.features.data.clear();
        self.features.n_frames = 0;
        self.dropped_frames = 0;
        self.detokenizer.reset();
        self.decoded_tokens = 0;
        self.decode_state.reset_for_runtime(&self.runtime);
    }

    fn rebase_after_soft_split(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.rebase_decode_baseline();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
    use crate::models::xasr_zipformer::executor::transcribe_xasr_zipformer_pcm;

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn xasr_accelerated_request_engages_gpu_and_matches_cpu_text() {
        use crate::ggml_runtime::{RequestBackendPreference, install_request_backend_override};

        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr accelerated parity test",
            "xasr accelerated parity test",
        )
        .expect("sample wav should load");
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");

        // The encoder gate keys off the request override: CpuOnly/absent must
        // build a CPU config, Accelerated must keep the GPU-class backend.
        let (cpu_text, cpu_elapsed) = {
            let _guard = install_request_backend_override(Some(RequestBackendPreference::CpuOnly));
            assert_eq!(
                super::super::graph_config::xasr_zipformer_encoder_graph_config().backend,
                crate::ggml_runtime::GgmlCpuGraphBackend::Cpu
            );
            let started = std::time::Instant::now();
            let text = transcribe_xasr_zipformer_pcm(&reader, &metadata, &samples, None, false)
                .expect("cpu xasr")
                .text;
            (text, started.elapsed())
        };

        let (gpu_text, gpu_elapsed) = {
            let _guard =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            let backend = super::super::graph_config::xasr_zipformer_encoder_graph_config().backend;
            assert!(
                backend.is_gpu_class(),
                "accelerated request must keep the GPU-class backend, got {backend:?}"
            );
            let started = std::time::Instant::now();
            let text = transcribe_xasr_zipformer_pcm(&reader, &metadata, &samples, None, false)
                .expect("gpu xasr")
                .text;
            (text, started.elapsed())
        };

        eprintln!(
            "xasr accelerated parity: cpu={cpu_elapsed:?} gpu={gpu_elapsed:?} text={cpu_text:?}"
        );
        assert!(!cpu_text.trim().is_empty());
        assert_eq!(cpu_text, gpu_text, "GPU and CPU transcripts must match");
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn xasr_incremental_streaming_matches_batch_on_real_speech() {
        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr streaming parity test",
            "xasr streaming parity test",
        )
        .expect("sample wav should load");
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let batch = transcribe_xasr_zipformer_pcm(&reader, &metadata, &samples, None, false)
            .expect("batch xasr")
            .text;
        let request = GgmlAsrStreamingSessionRequest {
            runtime_source_path: pack,
            runtime_source_preflight: None,
            selected_family: crate::xasr_zipformer_runtime_descriptor_v1(),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: crate::GgmlAsrBackendPreference::CpuOnly,
            session_context: crate::NativeAsrSessionContext::new("rt_xasr_streaming_match"),
            session_config: crate::NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        };
        let mut decoder = XasrIncrementalDecoder::new(
            &request,
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            super::super::runtime::checkout_prepared_runtime(&request.runtime_source_path)
                .expect("streaming runtime"),
        );
        let mut streaming = String::new();
        for chunk in samples.chunks(320) {
            streaming.push_str(&decoder.accept_samples(chunk).expect("stream chunk"));
        }
        streaming.push_str(&decoder.finish().expect("stream finish"));
        eprintln!("xasr real-speech streaming==batch text={streaming:?}");
        assert!(
            !batch.trim().is_empty(),
            "batch transcript must be non-empty for a meaningful parity check"
        );
        assert_eq!(streaming, batch);
        // Prefix draining must have kept the session buffers bounded: the
        // 5.5s sample is ~88k samples / ~555 feature rows, of which only a
        // small working tail may remain resident.
        assert!(
            decoder.dropped_samples > 0 && decoder.audio.len() < 40_000,
            "audio prefix was not drained: dropped={} resident={}",
            decoder.dropped_samples,
            decoder.audio.len()
        );
        assert!(
            decoder.dropped_frames > 0 && decoder.features.n_frames < 256,
            "feature prefix was not drained: dropped={} resident={}",
            decoder.dropped_frames,
            decoder.features.n_frames
        );
    }
}
