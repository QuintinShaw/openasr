//! CTC-specialized streaming driver.
//!
//! Unlike seq2seq families, CTC partials already have a structured frame-sync
//! greedy result: token ids, token spans, and frame count. This driver keeps the
//! authoritative FINAL on the offline full-buffer executor, but partials consume
//! the raw CTC result directly instead of routing through word-level seq2seq
//! normalization.

use std::time::Instant;

use crate::ggml_runtime::install_request_backend_override;
use crate::models::ctc_greedy_decode::CtcGreedyDecodeResult;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrPreparedAudio,
    GgmlAsrStreamingSessionRequest,
};
use crate::models::ggml_streaming_audio::{FrameTimelineError, GgmlStreamingAudioBuffer};
use crate::models::ggml_streaming_session::{
    GgmlAsrStreamingTranscriptDriver, GgmlAsrStreamingTranscriptUpdate,
};
use crate::models::graph_runtime_config::install_request_inference_threads_override;
use crate::models::incremental_streaming_driver::StreamingPartialTuning;
use crate::models::streaming_partial_cadence::PartialDecodeCadence;
use crate::{RealtimeAudioFrame, TranscriptUpdate, Transcription};

const STREAMING_WARM_UP_AUDIO_MS: usize = 1_000;
const SAMPLES_PER_MS_16KHZ: usize = 16;

type CtcPartialTranscriber =
    dyn FnMut(&GgmlAsrPreparedAudio) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError> + Send;
type CtcFinalTranscriber =
    dyn FnMut(&GgmlAsrPreparedAudio) -> Result<Transcription, GgmlAsrExecutionError> + Send;

pub(crate) fn build_ctc_streaming_driver<E, FPartial, FFinal>(
    executor: E,
    executor_id: &'static str,
    adapter_id: &'static str,
    request: &GgmlAsrStreamingSessionRequest,
    tuning: StreamingPartialTuning,
    partial_decode: FPartial,
    final_decode: FFinal,
) -> Box<dyn GgmlAsrStreamingTranscriptDriver>
where
    E: Clone + Send + 'static,
    FPartial: Fn(&E, &GgmlAsrExecutionRequest) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError>
        + Send
        + 'static,
    FFinal: Fn(&E, &GgmlAsrExecutionRequest) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>
        + Send
        + 'static,
{
    let session_suffix = &request.session_context.session_id.0;
    let utterance_id = format!("utt_{session_suffix}");
    let segment_id = format!("seg_{session_suffix}");
    let partial_results = request.session_config.partial_results;
    let partial_floor_ms = request
        .session_config
        .partial_floor_ms(tuning.min_partial_interval_ms());

    let runtime_source_path = request.runtime_source_path.clone();
    let runtime_source_preflight = request.runtime_source_preflight.clone();
    let selected_family = request.selected_family.clone();
    let request_options = request.request_options.clone();
    let inference_threads = request_options.inference_threads;
    let backend_preference = request.backend_preference;
    let make_request = move |audio: &GgmlAsrPreparedAudio| GgmlAsrExecutionRequest {
        runtime_source_path: runtime_source_path.clone(),
        runtime_source_preflight: runtime_source_preflight.clone(),
        selected_family: selected_family.clone(),
        prepared_audio: audio.clone(),
        request_options: request_options.clone(),
        backend_preference,
    };

    let partial_executor = executor.clone();
    let partial_transcribe = Box::new(move |audio: &GgmlAsrPreparedAudio| {
        let _thread_override = install_request_inference_threads_override(inference_threads);
        // Same as the seq2seq incremental driver: this closure calls the
        // per-family decode fn directly instead of going through
        // GgmlAsrExecutionDispatch::execute, so the request's
        // backend_preference must be installed here or an explicit
        // CpuOnly/Accelerated choice is silently dropped for streaming
        // partials.
        let _backend_override =
            install_request_backend_override(backend_preference.request_backend_override());
        partial_decode(&partial_executor, &make_request(audio))
    });

    let final_executor = executor;
    let runtime_source_path = request.runtime_source_path.clone();
    let runtime_source_preflight = request.runtime_source_preflight.clone();
    let selected_family = request.selected_family.clone();
    let request_options = request.request_options.clone();
    let backend_preference = request.backend_preference;
    let make_final_request = move |audio: &GgmlAsrPreparedAudio| GgmlAsrExecutionRequest {
        runtime_source_path: runtime_source_path.clone(),
        runtime_source_preflight: runtime_source_preflight.clone(),
        selected_family: selected_family.clone(),
        prepared_audio: audio.clone(),
        request_options: request_options.clone(),
        backend_preference,
    };
    let final_transcribe = Box::new(move |audio: &GgmlAsrPreparedAudio| {
        let _thread_override = install_request_inference_threads_override(inference_threads);
        let _backend_override =
            install_request_backend_override(backend_preference.request_backend_override());
        final_decode(&final_executor, &make_final_request(audio)).map(|result| result.transcription)
    });

    Box::new(CtcWindowedStreamingTranscriptDriver::new(
        executor_id,
        adapter_id,
        utterance_id,
        segment_id,
        partial_results,
        PartialDecodeCadence::with_floor_ms(partial_floor_ms)
            .with_first_decode_min_audio_ms(u64::from(tuning.first_partial_audio_ms())),
        tuning.window_ms(),
        partial_transcribe,
        final_transcribe,
    ))
}

pub(crate) struct CtcWindowedStreamingTranscriptDriver {
    executor_id: &'static str,
    adapter_id: &'static str,
    utterance_id_prefix: String,
    segment_id_prefix: String,
    utterance_id: String,
    segment_id: String,
    utterance_index: u64,
    partial_results: bool,
    buffer: GgmlStreamingAudioBuffer,
    cadence: PartialDecodeCadence,
    base_cadence: PartialDecodeCadence,
    last_text: Option<String>,
    next_revision: u64,
    final_emitted: bool,
    window_ms: u64,
    partial_transcribe: Box<CtcPartialTranscriber>,
    final_transcribe: Box<CtcFinalTranscriber>,
}

impl CtcWindowedStreamingTranscriptDriver {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        executor_id: &'static str,
        adapter_id: &'static str,
        utterance_id_prefix: String,
        segment_id_prefix: String,
        partial_results: bool,
        cadence: PartialDecodeCadence,
        window_ms: u64,
        partial_transcribe: Box<CtcPartialTranscriber>,
        final_transcribe: Box<CtcFinalTranscriber>,
    ) -> Self {
        let utterance_id = format!("{utterance_id_prefix}_000000");
        let segment_id = format!("{segment_id_prefix}_000000");
        Self {
            executor_id,
            adapter_id,
            utterance_id_prefix,
            segment_id_prefix,
            utterance_id,
            segment_id,
            utterance_index: 0,
            partial_results,
            buffer: GgmlStreamingAudioBuffer::default(),
            cadence: cadence.clone(),
            base_cadence: cadence,
            last_text: None,
            next_revision: 1,
            final_emitted: false,
            window_ms,
            partial_transcribe,
            final_transcribe,
        }
    }

    fn driver_failed(&self, reason: String) -> GgmlAsrExecutionError {
        GgmlAsrExecutionError::ExecutorFailed {
            executor_id: self.executor_id,
            adapter_id: self.adapter_id,
            reason,
        }
    }

    fn map_timeline_error(&self, error: FrameTimelineError) -> GgmlAsrExecutionError {
        self.driver_failed(error.to_string())
    }

    fn decode_warm_up_silence(&mut self) -> Result<(), GgmlAsrExecutionError> {
        let audio = GgmlAsrPreparedAudio::mono_16khz(vec![
            0.0;
            STREAMING_WARM_UP_AUDIO_MS
                * SAMPLES_PER_MS_16KHZ
        ]);
        let _ = (self.partial_transcribe)(&audio)?;
        Ok(())
    }

    fn decode_partial_if_due(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if !self.partial_results || self.final_emitted || self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let audio_end_ms = self.buffer.end_ms().unwrap_or(0);
        if !self.cadence.should_decode(audio_end_ms) {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let audio = self.buffer.prepared_audio_window(self.window_ms);
        let result = (self.partial_transcribe)(&audio)?;
        let update = self.emit_update(&result.text, false);
        self.cadence
            .record_decode(audio_end_ms, started.elapsed().as_millis() as u64);
        Ok(update.into_iter().collect())
    }

    fn reset_current_utterance(&mut self) {
        self.buffer.clear();
        self.last_text = None;
        self.final_emitted = false;
        self.cadence = self.base_cadence.clone();
        self.utterance_index = self.utterance_index.saturating_add(1);
        self.utterance_id = format!("{}_{:06}", self.utterance_id_prefix, self.utterance_index);
        self.segment_id = format!("{}_{:06}", self.segment_id_prefix, self.utterance_index);
    }

    fn decode_full_buffer(&mut self) -> Result<Transcription, GgmlAsrExecutionError> {
        let audio = self.buffer.prepared_audio_snapshot();
        (self.final_transcribe)(&audio)
    }

    fn emit_update(
        &mut self,
        raw_text: &str,
        final_update: bool,
    ) -> Option<GgmlAsrStreamingTranscriptUpdate> {
        let text = raw_text.trim().to_string();
        if text.is_empty() {
            return None;
        }
        if !final_update && self.last_text.as_deref() == Some(text.as_str()) {
            return None;
        }
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        self.last_text = Some(text.clone());
        if final_update {
            self.final_emitted = true;
        }
        let start_ms = self.buffer.start_ms().unwrap_or(0);
        let end_ms = self
            .buffer
            .end_ms()
            .unwrap_or_else(|| start_ms.saturating_add(self.buffer.duration_ms()));
        let update = TranscriptUpdate::new(
            self.utterance_id.clone(),
            self.segment_id.clone(),
            revision,
            text,
            start_ms,
            end_ms,
        );
        Some(if final_update {
            GgmlAsrStreamingTranscriptUpdate::final_(update)
        } else {
            GgmlAsrStreamingTranscriptUpdate::partial(update)
        })
    }
}

impl GgmlAsrStreamingTranscriptDriver for CtcWindowedStreamingTranscriptDriver {
    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.decode_warm_up_silence()
    }

    fn reset_utterance(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.reset_current_utterance();
        Ok(())
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.buffer
            .push_frame(frame)
            .map_err(|error| self.map_timeline_error(error))?;
        Ok(Vec::new())
    }

    fn poll_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.decode_partial_if_due()
    }

    fn finish_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if self.buffer.is_empty() || self.final_emitted {
            return Ok(Vec::new());
        }
        let transcription = self.decode_full_buffer()?;
        Ok(self
            .emit_update(&transcription.text, true)
            .into_iter()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{RealtimeAudioFormat, RealtimeAudioFrame};

    fn frame(seq: u64, start_ms: u64, samples: Vec<i16>) -> RealtimeAudioFrame {
        RealtimeAudioFrame::new(
            seq,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            samples,
        )
        .unwrap()
    }

    fn ctc_result(text: &str, frames: usize) -> CtcGreedyDecodeResult {
        CtcGreedyDecodeResult {
            token_ids: vec![1],
            token_spans: Vec::new(),
            frame_count: frames,
            text: text.to_string(),
        }
    }

    #[test]
    fn ctc_driver_keeps_push_audio_cheap_and_decodes_on_poll() {
        let mut driver = CtcWindowedStreamingTranscriptDriver::new(
            "ctc-test",
            "ctc-adapter",
            "utt".to_string(),
            "seg".to_string(),
            true,
            PartialDecodeCadence::with_floor_ms(20),
            40,
            Box::new(|audio| Ok(ctc_result(&format!("p{}", audio.samples_f32.len()), 1))),
            Box::new(|audio| {
                Ok(Transcription {
                    text: format!("final{}", audio.samples_f32.len()),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                })
            }),
        );

        assert!(
            driver
                .push_audio(frame(0, 0, vec![0; 320]))
                .unwrap()
                .is_empty()
        );
        let partial = driver.poll_updates().unwrap();
        assert_eq!(partial.len(), 1);
        match &partial[0] {
            GgmlAsrStreamingTranscriptUpdate::Partial(update) => {
                assert_eq!(update.text, "p320");
                assert_eq!(update.end_ms, 20);
            }
            other => panic!("expected partial, got {other:?}"),
        }
    }

    #[test]
    fn ctc_driver_final_uses_full_buffer() {
        let mut driver = CtcWindowedStreamingTranscriptDriver::new(
            "ctc-test",
            "ctc-adapter",
            "utt".to_string(),
            "seg".to_string(),
            true,
            PartialDecodeCadence::with_floor_ms(20),
            20,
            Box::new(|audio| Ok(ctc_result(&format!("p{}", audio.samples_f32.len()), 1))),
            Box::new(|audio| {
                Ok(Transcription {
                    text: format!("final{}", audio.samples_f32.len()),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                })
            }),
        );

        driver.push_audio(frame(0, 0, vec![0; 320])).unwrap();
        driver.push_audio(frame(1, 20, vec![0; 320])).unwrap();
        let final_updates = driver.finish_updates().unwrap();
        match &final_updates[0] {
            GgmlAsrStreamingTranscriptUpdate::Final(update) => {
                assert_eq!(update.text, "final640");
                assert_eq!(update.end_ms, 40);
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    /// Regression test for the same streaming backend-override bypass fixed
    /// in `incremental_streaming_driver.rs`: `build_ctc_streaming_driver`'s
    /// `partial_transcribe`/`final_transcribe` closures call the family's
    /// decode fns directly, not through `GgmlAsrExecutionDispatch::execute`,
    /// so they must install `request.backend_preference` themselves or an
    /// explicit choice is silently dropped for CTC (parakeet/wav2vec2)
    /// streaming.
    #[test]
    fn ctc_streaming_closures_install_request_backend_override() {
        use crate::ggml_runtime::{
            GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendPreference,
        };
        use std::path::PathBuf;

        fn session_request(
            backend_preference: crate::GgmlAsrBackendPreference,
        ) -> GgmlAsrStreamingSessionRequest {
            GgmlAsrStreamingSessionRequest {
                runtime_source_path: PathBuf::from("/tmp/openasr-missing-runtime.gguf"),
                runtime_source_preflight: None,
                selected_family: crate::wav2vec2_ctc_runtime_descriptor_v1(),
                request_options: crate::GgmlAsrExecutionOptions::default(),
                configured_diarize: false,
                backend_preference,
                session_context: crate::NativeAsrSessionContext::new(
                    "rt_ctc_backend_override_test",
                ),
                session_config: crate::NativeAsrStreamingSessionConfig::new()
                    .with_partial_results(true)
                    .into(),
            }
        }

        // Drives one warm-up partial decode through the real
        // `build_ctc_streaming_driver` closure and records what the decode
        // fn observed via the thread-local override, plus what a gated
        // family's `resolve_family_runtime_backend` would resolve to at that
        // instant.
        fn observed_backend_during_partial_decode(
            backend_preference: crate::GgmlAsrBackendPreference,
        ) -> (Option<RequestBackendPreference>, GgmlCpuGraphBackend) {
            let request = session_request(backend_preference);
            let observed: Arc<
                Mutex<Option<(Option<RequestBackendPreference>, GgmlCpuGraphBackend)>>,
            > = Arc::new(Mutex::new(None));
            let observed_for_decode = Arc::clone(&observed);
            let mut driver = build_ctc_streaming_driver(
                (),
                "ctc-backend-override-test-executor",
                crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
                &request,
                crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
                move |_executor: &(), _request: &GgmlAsrExecutionRequest| {
                    *observed_for_decode.lock().unwrap() = Some((
                        crate::ggml_runtime::request_backend_override(),
                        GgmlCpuGraphConfig::resolve_family_runtime_backend(false),
                    ));
                    Ok(ctc_result("", 0))
                },
                move |_executor: &(), _request: &GgmlAsrExecutionRequest| {
                    Ok(GgmlAsrExecutionResult {
                        transcription: Transcription {
                            text: String::new(),
                            segments: Vec::new(),
                            longform: None,
                            language: None,
                        },
                        carry_context: None,
                    })
                },
            );
            driver.warm_up().expect("warm up should decode once");
            observed
                .lock()
                .unwrap()
                .take()
                .expect("partial decode closure should have run")
        }

        // Auto: no override installed, so a gated family stays pinned to CPU.
        let (auto_override, auto_backend) =
            observed_backend_during_partial_decode(crate::GgmlAsrBackendPreference::Auto);
        assert_eq!(auto_override, None);
        assert_eq!(auto_backend, GgmlCpuGraphBackend::Cpu);

        // Explicit Accelerated: the partial_transcribe closure must install
        // the override itself, so a gated family's resolver sees Accelerated
        // instead of silently falling back to CPU.
        let (accel_override, accel_backend) =
            observed_backend_during_partial_decode(crate::GgmlAsrBackendPreference::Accelerated);
        assert_eq!(accel_override, Some(RequestBackendPreference::Accelerated));
        assert_ne!(accel_backend, GgmlCpuGraphBackend::Cpu);
    }
}
