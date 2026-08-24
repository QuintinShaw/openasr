//! CTC-specialized streaming driver.
//!
//! Unlike seq2seq families, CTC partials already have a structured frame-sync
//! greedy result: token ids, token spans, and frame count. This driver keeps the
//! authoritative FINAL on the offline full-buffer executor, but partials consume
//! the raw CTC result directly instead of routing through word-level seq2seq
//! normalization.

use std::time::Instant;

use crate::models::ctc_greedy_decode::CtcGreedyDecodeResult;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
    GgmlAsrPreparedAudioView, GgmlAsrStreamingSessionRequest,
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

type CtcPartialTranscriber = dyn FnMut(
        &GgmlAsrPreparedAudioView<'static>,
    ) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError>
    + Send;
type CtcFinalTranscriber = dyn FnMut(&GgmlAsrPreparedAudioView<'static>) -> Result<Transcription, GgmlAsrExecutionError>
    + Send;

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
    FPartial: Fn(&E, &GgmlAsrExecutionViewRequest) -> Result<CtcGreedyDecodeResult, GgmlAsrExecutionError>
        + Send
        + 'static,
    FFinal: Fn(
            &E,
            &GgmlAsrExecutionViewRequest,
        ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>
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

    let partial_execution_services = std::sync::Arc::clone(&request.execution_services);
    let partial_decode_execution_services = std::sync::Arc::clone(&request.execution_services);
    let partial_decoder_state = request.decoder_state.clone();
    let verified_pack = request.verified_pack.clone();
    let selected_family = request.selected_family.clone();
    let request_options = request.request_options.clone();
    let inference_threads = request_options.inference_threads;
    let backend_preference = request.backend_preference;
    // Resolved once for the whole session (this is a `Copy` value carried on
    // the session request, not a thread-local): every per-frame request this
    // driver builds for the life of the session copies it in directly.
    let resolved_runtime = request.resolved_runtime;
    let partial_execution_lane = request.execution_lane.clone();
    let partial_execution_context = request.per_frame_execution_context(
        "per-frame CTC partial decode has no independent cancel/pause surface; the live session \
         ends when its caller drops it",
    );
    let make_request =
        move |audio: &GgmlAsrPreparedAudioView<'static>| GgmlAsrExecutionViewRequest {
            execution_services: std::sync::Arc::clone(&partial_execution_services),
            decoder_state: partial_decoder_state.clone(),
            verified_pack: verified_pack.clone(),
            selected_family: selected_family.clone(),
            prepared_audio: audio.clone(),
            request_options: request_options.clone(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::clone(&partial_execution_context),
        };

    let partial_executor = executor.clone();
    let partial_transcribe = Box::new(move |audio: &GgmlAsrPreparedAudioView<'static>| {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                partial_decode_execution_services.as_ref(),
            );
        let _resolved_lane =
            crate::models::native_execution_services::install_resolved_execution_lane(
                partial_execution_lane.clone(),
            );
        let _thread_override = install_request_inference_threads_override(inference_threads);
        // This closure bypasses dispatch. The exact lane above, not the coarse
        // request preference, remains authoritative for every TLS reader.
        partial_decode(&partial_executor, &make_request(audio))
    });

    let final_executor = executor;
    let final_execution_services = std::sync::Arc::clone(&request.execution_services);
    let final_decode_execution_services = std::sync::Arc::clone(&request.execution_services);
    let final_decoder_state = request.decoder_state.clone();
    let verified_pack = request.verified_pack.clone();
    let selected_family = request.selected_family.clone();
    let request_options = request.request_options.clone();
    let backend_preference = request.backend_preference;
    let final_execution_lane = request.execution_lane.clone();
    let final_execution_context = request.per_frame_execution_context(
        "per-frame CTC final decode has no independent cancel/pause surface; the live session \
         ends when its caller drops it",
    );
    let make_final_request =
        move |audio: &GgmlAsrPreparedAudioView<'static>| GgmlAsrExecutionViewRequest {
            execution_services: std::sync::Arc::clone(&final_execution_services),
            decoder_state: final_decoder_state.clone(),
            verified_pack: verified_pack.clone(),
            selected_family: selected_family.clone(),
            prepared_audio: audio.clone(),
            request_options: request_options.clone(),
            backend_preference,
            resolved_runtime,
            execution_context: std::sync::Arc::clone(&final_execution_context),
        };
    let final_transcribe = Box::new(move |audio: &GgmlAsrPreparedAudioView<'static>| {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                final_decode_execution_services.as_ref(),
            );
        let _resolved_lane =
            crate::models::native_execution_services::install_resolved_execution_lane(
                final_execution_lane.clone(),
            );
        let _thread_override = install_request_inference_threads_override(inference_threads);
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
        let audio = GgmlAsrPreparedAudioView::mono_16khz(vec![
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
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: format!("final{}", audio.samples_f32.len()),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                    ..Default::default()
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
                    truncated_decodes: Vec::new(),
                    unnamed_speakers: Vec::new(),
                    text: format!("final{}", audio.samples_f32.len()),
                    segments: Vec::new(),
                    longform: None,
                    language: None,
                    ..Default::default()
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
    /// so they must install `request.backend_preference` -- and now also the
    /// resolved family runtime input -- themselves, or an explicit choice
    /// (or the family's own `AutoGpuPolicy` gate) is silently dropped for
    /// CTC (parakeet/wav2vec2) streaming.
    #[test]
    fn ctc_streaming_closures_install_request_backend_override() {
        use crate::ggml_runtime::{
            GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendPreference,
        };
        fn session_request(
            backend_preference: crate::GgmlAsrBackendPreference,
        ) -> GgmlAsrStreamingSessionRequest {
            let request_attempt_id =
                crate::RequestAttemptId::parse("ffeeddccbbaa99887766554433221100").unwrap();
            let receipt = crate::NativeExecutionReceiptCollector::new();
            let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::arch::family_auto_gpu_policy_for_model_architecture(
                    crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                ),
            );
            let runtime_source_preflight =
                crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight();
            GgmlAsrStreamingSessionRequest {
                execution_services:
                    crate::models::native_execution_services::test_native_execution_services(),
                decoder_state:
                    crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
                verified_pack:
                    crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                        runtime_source_preflight,
                        crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                    ),
                selected_family: crate::arch::builtin_adapter_descriptor(
                    crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
                ),
                request_options: crate::GgmlAsrExecutionOptions::default(),
                configured_diarize: false,
                backend_preference,
                resolved_runtime,
                execution_lane:
                    crate::models::native_execution_services::current_execution_lane_key(
                        resolved_runtime.backend(),
                    ),
                final_text_processor: None,
                session_context: crate::NativeAsrSessionContext::new(
                    "rt_ctc_backend_override_test",
                )
                .with_request_attempt_id(request_attempt_id)
                .with_native_execution_receipt(receipt),
                session_config: crate::NativeAsrStreamingSessionConfig::new()
                    .with_partial_results(true)
                    .into(),
            }
        }

        // Drives one warm-up partial decode through the real
        // `build_ctc_streaming_driver` closure and records what the decode
        // fn observed via the thread-local override, plus what
        // `_request.resolved_runtime` reports at that instant -- the same
        // field the driver itself copies from the session request, resolved
        // from the request's architecture-declared `AutoGpuPolicy`.
        fn observed_backend_during_partial_decode(
            backend_preference: crate::GgmlAsrBackendPreference,
        ) -> GgmlCpuGraphBackend {
            let request = session_request(backend_preference);
            let expected_lane = request.execution_lane.clone();
            let expected_attempt = request.session_context.request_attempt_id();
            type ObservedDecode = (
                Option<RequestBackendPreference>,
                GgmlCpuGraphBackend,
                Option<crate::models::native_execution_services::ExecutionLaneKey>,
                Option<crate::RequestAttemptId>,
                bool,
            );
            let observed: Arc<Mutex<Option<ObservedDecode>>> = Arc::new(Mutex::new(None));
            let observed_for_decode = Arc::clone(&observed);
            let observed_final: Arc<Mutex<Option<ObservedDecode>>> = Arc::new(Mutex::new(None));
            let observed_for_final = Arc::clone(&observed_final);
            let mut driver = build_ctc_streaming_driver(
                (),
                "ctc-backend-override-test-executor",
                crate::WAV2VEC2_CTC_GGML_ADAPTER_ID,
                &request,
                crate::models::incremental_streaming_driver::STREAMING_PARTIAL_TUNING_FAST_SNAPSHOT,
                move |_executor: &(), _request: &GgmlAsrExecutionViewRequest| {
                    *observed_for_decode.lock().unwrap() = Some((
                        crate::ggml_runtime::request_backend_override(),
                        _request.resolved_runtime.backend(),
                        _request.execution_context.native_execution_lane().cloned(),
                        _request.execution_context.request_attempt_id(),
                        _request
                            .execution_context
                            .native_execution_receipt()
                            .is_some(),
                    ));
                    Ok(ctc_result("", 0))
                },
                move |_executor: &(), _request: &GgmlAsrExecutionViewRequest| {
                    *observed_for_final.lock().unwrap() = Some((
                        crate::ggml_runtime::request_backend_override(),
                        _request.resolved_runtime.backend(),
                        _request.execution_context.native_execution_lane().cloned(),
                        _request.execution_context.request_attempt_id(),
                        _request
                            .execution_context
                            .native_execution_receipt()
                            .is_some(),
                    ));
                    Ok(GgmlAsrExecutionResult {
                        transcription: Transcription {
                            truncated_decodes: Vec::new(),
                            unnamed_speakers: Vec::new(),
                            text: String::new(),
                            segments: Vec::new(),
                            longform: None,
                            language: None,
                            ..Default::default()
                        },
                        carry_context: None,
                        decode_truncation: None,
                    })
                },
            );
            driver.warm_up().expect("warm up should decode once");
            driver
                .push_audio(frame(0, 0, vec![0; 320]))
                .expect("final decode fixture audio");
            driver.finish_updates().expect("final decode should run");
            let partial = observed
                .lock()
                .unwrap()
                .take()
                .expect("partial decode closure should have run");
            let final_decode = observed_final
                .lock()
                .unwrap()
                .take()
                .expect("final decode closure should have run");
            for (backend_override, _, execution_lane, request_attempt_id, has_receipt) in
                [&partial, &final_decode]
            {
                assert_eq!(execution_lane, &Some(expected_lane.clone()));
                assert_eq!(request_attempt_id, &expected_attempt);
                assert!(
                    *has_receipt,
                    "per-frame CTC decode must retain receipt authority"
                );
                assert_eq!(
                    backend_override,
                    &Some(expected_lane.request_backend_preference())
                );
            }
            assert_eq!(partial.1, final_decode.1);
            partial.1
        }

        // Auto: no override installed. wav2vec2-ctc's policy is `AllBackends`
        // (a no-op gate), so the resolved input must match the generic
        // Auto-mode resolution exactly -- host-independent equality, not a
        // fixed value, since this dev machine's own GPU availability decides
        // what "generic Auto" picks.
        let expected_auto_backend = GgmlCpuGraphConfig::runtime_default().backend;
        let auto_backend =
            observed_backend_during_partial_decode(crate::GgmlAsrBackendPreference::Auto);
        assert_eq!(auto_backend, expected_auto_backend);

        // Explicit Accelerated: the partial_transcribe closure must install
        // the override itself, so the resolved input reflects Accelerated
        // instead of silently falling back to whatever Auto would have
        // picked.
        let accel_backend =
            observed_backend_during_partial_decode(crate::GgmlAsrBackendPreference::Accelerated);
        assert_ne!(accel_backend, GgmlCpuGraphBackend::Cpu);
    }
}
