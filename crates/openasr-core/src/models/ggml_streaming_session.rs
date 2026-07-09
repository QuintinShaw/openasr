#![cfg_attr(not(test), allow(dead_code))]

use crate::api::native::NativeStreamingTranscriptEmitter;
use crate::arch::emits_punctuation_for_model_architecture;
use crate::models::firered_punc::pack::resolve_firered_punc_pack_path;
use crate::models::firered_punc::runtime::SendableFireRedPuncRuntime;
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrStreamingSessionConfig, GgmlAsrStreamingSessionRequest,
};
use crate::punctuation::{PunctuationError, should_apply_punctuation};
use crate::realtime::events::realtime_timestamp_now;
use crate::{
    NativeAsrError, NativeAsrRequestOptions, NativeAsrSession, NativeAsrStreamingSessionConfig,
    RealtimeAudioFrame, RealtimeEventEnvelope, RealtimeSessionState, TranscriptUpdate,
};

/// Punctuates one FINAL segment's text before it is emitted. `Err` means
/// "leave the segment text unchanged" (fail-closed, mirroring the batch
/// stage's `punctuate_transcription_segments` contract); partials never go
/// through this (re-punctuating every partial re-runs a bidirectional encoder
/// per revision and reintroduces caption flicker).
pub(crate) type StreamingFinalPunctuator =
    Box<dyn Fn(&str) -> Result<String, PunctuationError> + Send>;

/// Whether the punctuation stage applies to this model architecture: same
/// capability fact the batch path reads from the pack's
/// `general.architecture` (see `model_emits_punctuation` in
/// `native_transcribe`), sourced here from the already-selected adapter
/// descriptor so no pack re-read is needed.
fn streaming_punctuation_stage_applies(model_architecture: &str) -> bool {
    should_apply_punctuation(emits_punctuation_for_model_architecture(model_architecture))
}

/// Streaming counterpart of the batch `apply_punctuation_stage_if_applicable`
/// gate, resolved once at session construction: the stage activates only for
/// a model family the catalog honestly declares unpunctuated AND when the
/// FireRedPunc capability pack is installed ("pack installed => stage on",
/// no protocol field). A missing or unloadable pack deactivates the stage
/// (fail-closed, never an error); the loaded runtime is cached on the session
/// so finals do not reload BERT weights.
fn resolve_streaming_final_punctuator(
    request: &GgmlAsrStreamingSessionRequest,
) -> Option<StreamingFinalPunctuator> {
    if !streaming_punctuation_stage_applies(request.selected_family.model_architecture) {
        return None;
    }
    let pack_path = resolve_firered_punc_pack_path()?;
    let runtime = SendableFireRedPuncRuntime::from_pack(&pack_path).ok()?;
    Some(Box::new(move |text| runtime.punctuate(text)))
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgmlAsrStreamingTranscriptUpdate {
    Partial(TranscriptUpdate),
    Final(TranscriptUpdate),
}

impl GgmlAsrStreamingTranscriptUpdate {
    pub(crate) fn partial(update: TranscriptUpdate) -> Self {
        Self::Partial(update)
    }

    pub(crate) fn final_(update: TranscriptUpdate) -> Self {
        Self::Final(update)
    }
}

pub(crate) trait GgmlAsrStreamingTranscriptDriver: Send {
    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError>;

    fn poll_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        Ok(Vec::new())
    }

    fn flush_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.poll_updates()
    }

    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        Ok(())
    }

    fn reset_utterance(&mut self) -> Result<(), GgmlAsrExecutionError> {
        Ok(())
    }

    fn finish_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.flush_updates()
    }

    /// Whether [`Self::split_updates`] performs a soft segment split that
    /// preserves decode state. Drivers that re-transcribe a window must say
    /// `false` so forced max-duration boundaries keep doing a full
    /// finalize+reset (their quality degrades with utterance length).
    fn supports_soft_split(&self) -> bool {
        false
    }

    /// Finalizes the current transcript segment WITHOUT resetting decode
    /// state. Used at forced (max-utterance-duration) boundaries so the
    /// recognition context survives an arbitrary mid-speech cut; only called
    /// when [`Self::supports_soft_split`] returns true.
    fn split_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<(), GgmlAsrExecutionError> {
        Ok(())
    }
}

impl GgmlAsrStreamingTranscriptDriver for Box<dyn GgmlAsrStreamingTranscriptDriver> {
    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.as_mut().push_audio(frame)
    }

    fn poll_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.as_mut().poll_updates()
    }

    fn flush_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.as_mut().flush_updates()
    }

    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.as_mut().warm_up()
    }

    fn reset_utterance(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.as_mut().reset_utterance()
    }

    fn finish_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.as_mut().finish_updates()
    }

    fn supports_soft_split(&self) -> bool {
        self.as_ref().supports_soft_split()
    }

    fn split_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        self.as_mut().split_updates()
    }

    fn cancel(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.as_mut().cancel()
    }
}

pub(crate) struct GgmlAsrStreamingTranscriptSession<D>
where
    D: GgmlAsrStreamingTranscriptDriver,
{
    executor_id: &'static str,
    adapter_id: &'static str,
    emitter: NativeStreamingTranscriptEmitter,
    driver: D,
    clock: Box<dyn FnMut() -> String + Send>,
    /// FINAL-segment punctuation stage (see [`StreamingFinalPunctuator`]);
    /// `None` when the stage does not apply to this model family or no
    /// FireRedPunc pack is installed.
    final_punctuator: Option<StreamingFinalPunctuator>,
    queued_audio_frames: usize,
    closed: bool,
}

impl From<GgmlAsrStreamingSessionConfig> for NativeAsrStreamingSessionConfig {
    fn from(config: GgmlAsrStreamingSessionConfig) -> Self {
        Self {
            audio_format: config.audio_format,
            backpressure: config.backpressure,
            partial_results: config.partial_results,
            word_timestamps: config.word_timestamps,
            min_partial_interval_ms: config.min_partial_interval_ms,
        }
    }
}

impl<D> GgmlAsrStreamingTranscriptSession<D>
where
    D: GgmlAsrStreamingTranscriptDriver,
{
    pub(crate) fn new(
        executor_id: &'static str,
        request: &GgmlAsrStreamingSessionRequest,
        driver: D,
    ) -> Result<Self, GgmlAsrExecutionError> {
        Self::new_with_clock_and_punctuator(
            executor_id,
            request,
            driver,
            Box::new(realtime_timestamp_now),
            resolve_streaming_final_punctuator(request),
        )
    }

    /// Test constructor: injectable clock, punctuation stage off (no probing
    /// of the environment for an installed FireRedPunc pack).
    pub(crate) fn new_with_clock(
        executor_id: &'static str,
        request: &GgmlAsrStreamingSessionRequest,
        driver: D,
        clock: Box<dyn FnMut() -> String + Send>,
    ) -> Result<Self, GgmlAsrExecutionError> {
        Self::new_with_clock_and_punctuator(executor_id, request, driver, clock, None)
    }

    pub(crate) fn new_with_clock_and_punctuator(
        executor_id: &'static str,
        request: &GgmlAsrStreamingSessionRequest,
        driver: D,
        mut clock: Box<dyn FnMut() -> String + Send>,
        final_punctuator: Option<StreamingFinalPunctuator>,
    ) -> Result<Self, GgmlAsrExecutionError> {
        let adapter_id = request.selected_family.adapter_id;
        let native_options = NativeAsrRequestOptions::new()
            .with_language(request.request_options.language.clone())
            .with_prompt(request.request_options.prompt.clone())
            .with_phrase_bias(request.request_options.phrase_bias.clone())
            .with_inference_threads(
                request
                    .request_options
                    .inference_threads
                    .and_then(|value| u16::try_from(value).ok()),
            )
            .with_diarization(request.configured_diarize)
            .with_partial_results(request.session_config.partial_results)
            .with_word_timestamps(request.session_config.word_timestamps);
        let native_session_config: NativeAsrStreamingSessionConfig =
            request.session_config.clone().into();
        let created_at = clock();
        let configured_at = clock();
        let started_at = clock();
        let emitter = NativeStreamingTranscriptEmitter::new_started(
            request.session_context.clone(),
            request.selected_family.model_family.to_string(),
            native_options,
            native_session_config,
            created_at,
            configured_at,
            started_at,
        )
        .map_err(|error| ggml_streaming_session_failed(executor_id, adapter_id, error))?;

        Ok(Self {
            executor_id,
            adapter_id,
            emitter,
            driver,
            clock,
            final_punctuator,
            queued_audio_frames: 0,
            closed: false,
        })
    }

    fn now(&mut self) -> String {
        (self.clock)()
    }

    /// FINAL-only punctuation stage: a segment's text is punctuated exactly
    /// once, at the moment it stops being revised. Fail-closed per segment --
    /// a classifier error keeps the driver's original text.
    ///
    /// INVARIANT: every update that can surface as a visible FINAL passes
    /// through here first. There are exactly two producers of finals in this
    /// session -- driver-emitted finals ([`Self::emit_update`]'s `Final`
    /// branch) and the session-promoted pending tail partial
    /// ([`Self::finalize_pending_output`]) -- and both call this before
    /// `apply_final`. Never call `emitter.apply_final` or
    /// `emitter.finalize_pending_output_at` from anywhere else in this
    /// session, or a final can bypass the stage.
    fn punctuate_final_update(&self, update: &mut TranscriptUpdate) {
        if let Some(punctuate) = &self.final_punctuator
            && let Ok(punctuated) = punctuate(&update.text)
        {
            update.text = punctuated;
        }
    }

    /// Promotes the emitter's pending tail partial (if any) into a FINAL,
    /// running the punctuation stage on it exactly like a driver-emitted
    /// final. A driver that only ever emits partials (relying on the session
    /// to finalize the tail on flush/finalize/finish) therefore cannot leak
    /// an unpunctuated final.
    fn finalize_pending_output(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let finalized_at = self.now();
        let Some(mut update) = self.emitter.take_pending_partial_update() else {
            return Ok(Vec::new());
        };
        self.punctuate_final_update(&mut update);
        self.emitter.apply_final(update, finalized_at)
    }

    fn emit_update(
        &mut self,
        update: GgmlAsrStreamingTranscriptUpdate,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let created_at = self.now();
        match update {
            GgmlAsrStreamingTranscriptUpdate::Partial(update) => {
                self.emitter.apply_partial(update, created_at)
            }
            GgmlAsrStreamingTranscriptUpdate::Final(mut update) => {
                self.punctuate_final_update(&mut update);
                self.emitter.apply_final(update, created_at)
            }
        }
    }

    fn emit_updates(
        &mut self,
        updates: Vec<GgmlAsrStreamingTranscriptUpdate>,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let mut events = self.emitter.drain_pending_events();
        for update in updates {
            let emitted = self.emit_update(update)?;
            self.emitter
                .ensure_output_capacity(events.len() + emitted.len())?;
            events.extend(emitted);
        }
        Ok(events)
    }

    fn driver_error_to_native(&self, error: GgmlAsrExecutionError) -> NativeAsrError {
        NativeAsrError::SessionFailed {
            message: format!(
                "native ggml streaming driver '{}' failed for adapter '{}': {error}",
                self.executor_id, self.adapter_id
            ),
        }
    }

    fn ensure_running_for_audio(&self) -> Result<(), NativeAsrError> {
        if self.closed {
            return Err(NativeAsrError::SessionClosed);
        }
        if self.emitter.state() != RealtimeSessionState::Running {
            return Err(NativeAsrError::SessionFailed {
                message:
                    "native ggml streaming session requires running audio input before push_audio"
                        .to_string(),
            });
        }
        Ok(())
    }

    fn flush_driver_updates(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let updates = self
            .driver
            .flush_updates()
            .map_err(|error| self.driver_error_to_native(error))?;
        let mut events = self.emit_updates(updates)?;
        let finalized = self.finalize_pending_output()?;
        self.emitter
            .ensure_output_capacity(events.len() + finalized.len())?;
        events.extend(finalized);
        Ok(events)
    }
}

impl<D> NativeAsrSession for GgmlAsrStreamingTranscriptSession<D>
where
    D: GgmlAsrStreamingTranscriptDriver,
{
    fn session_id(&self) -> &str {
        self.emitter.session_id()
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.ensure_running_for_audio()?;
        self.emitter
            .ensure_push_capacity(self.queued_audio_frames)?;
        self.queued_audio_frames += 1;
        let result = self.driver.push_audio(frame);
        self.queued_audio_frames = self.queued_audio_frames.saturating_sub(1);
        let updates = result.map_err(|error| self.driver_error_to_native(error))?;
        self.emit_updates(updates)
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        let updates = self
            .driver
            .poll_updates()
            .map_err(|error| self.driver_error_to_native(error))?;
        self.emit_updates(updates)
    }

    fn flush(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.flush_driver_updates()
    }

    fn warm_up(&mut self) -> Result<(), NativeAsrError> {
        if self.closed {
            return Ok(());
        }
        self.driver
            .warm_up()
            .map_err(|error| self.driver_error_to_native(error))
    }

    fn finalize_utterance(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        let updates = self
            .driver
            .finish_updates()
            .map_err(|error| self.driver_error_to_native(error))?;
        let mut events = self.emit_updates(updates)?;
        let finalized = self.finalize_pending_output()?;
        self.emitter
            .ensure_output_capacity(events.len() + finalized.len())?;
        events.extend(finalized);
        self.driver
            .reset_utterance()
            .map_err(|error| self.driver_error_to_native(error))?;
        let restarted_at = self.now();
        if let Some(restarted) = self.emitter.reset_for_next_utterance(restarted_at)? {
            self.emitter.ensure_output_capacity(events.len() + 1)?;
            events.push(restarted);
        }
        Ok(events)
    }

    /// Segment split at a forced (max-utterance-duration) boundary. Unlike
    /// [`Self::finalize_utterance`] this preserves the driver's decode state
    /// when the driver supports it, so an arbitrary mid-speech cut cannot
    /// degrade recognition on either side of the boundary. Drivers without
    /// soft-split support fall back to the full finalize+reset.
    fn split_utterance(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        if !self.driver.supports_soft_split() {
            return self.finalize_utterance();
        }
        let updates = self
            .driver
            .split_updates()
            .map_err(|error| self.driver_error_to_native(error))?;
        if updates.is_empty() {
            // Nothing decoded since the last boundary: the driver did not
            // advance its segment identity, so skip the emitter
            // finalize/reset cycle too — otherwise the client would receive a
            // spurious audio.input.started with no segment around it.
            return self.emit_updates(Vec::new());
        }
        let mut events = self.emit_updates(updates)?;
        let finalized = self.finalize_pending_output()?;
        self.emitter
            .ensure_output_capacity(events.len() + finalized.len())?;
        events.extend(finalized);
        let restarted_at = self.now();
        if let Some(restarted) = self.emitter.reset_for_next_utterance(restarted_at)? {
            self.emitter.ensure_output_capacity(events.len() + 1)?;
            events.push(restarted);
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        let updates = self
            .driver
            .finish_updates()
            .map_err(|error| self.driver_error_to_native(error))?;
        let mut events = self.emit_updates(updates)?;
        let finalized = self.finalize_pending_output()?;
        self.emitter
            .ensure_output_capacity(events.len() + finalized.len())?;
        events.extend(finalized);
        let stopped_at = self.now();
        if let Some(stopped) = self
            .emitter
            .close_if_running("input_finished", stopped_at)?
        {
            events.push(stopped);
        }
        Ok(events)
    }

    fn close(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.closed = true;
        let mut events = self.flush_driver_updates()?;
        let stopped_at = self.now();
        if let Some(stopped) = self.emitter.close_if_running("client_closed", stopped_at)? {
            events.push(stopped);
        }
        let closed_at = self.now();
        events.push(self.emitter.close_session("client_closed", closed_at)?);
        Ok(events)
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.closed = true;
        self.driver
            .cancel()
            .map_err(|error| self.driver_error_to_native(error))?;
        let error_at = self.now();
        let closed_at = self.now();
        self.emitter.cancel(
            "Native ggml streaming session was cancelled.",
            error_at,
            closed_at,
        )
    }
}

fn ggml_streaming_session_failed(
    executor_id: &'static str,
    adapter_id: &'static str,
    error: NativeAsrError,
) -> GgmlAsrExecutionError {
    GgmlAsrExecutionError::ExecutorFailed {
        executor_id,
        adapter_id,
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, path::PathBuf};

    use super::*;
    use crate::models::ggml_family_registry::firered_aed_runtime_descriptor_v1;
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionOptions, RealtimeEvent, RealtimeTranscriptEvent,
        qwen3_asr_runtime_descriptor_v1,
    };

    enum ScriptStep {
        Partial { revision: u64, text: &'static str },
        Final { revision: u64, text: &'static str },
    }

    struct ScriptDriver {
        steps: VecDeque<ScriptStep>,
        last_frame: Option<RealtimeAudioFrame>,
    }

    impl ScriptDriver {
        fn new(steps: impl IntoIterator<Item = ScriptStep>) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                last_frame: None,
            }
        }

        fn pop_update_for_frame(
            &mut self,
            frame: &RealtimeAudioFrame,
        ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
            let Some(step) = self.steps.pop_front() else {
                return Ok(Vec::new());
            };
            let update = match step {
                ScriptStep::Partial { revision, text } => {
                    GgmlAsrStreamingTranscriptUpdate::partial(test_update(frame, revision, text))
                }
                ScriptStep::Final { revision, text } => {
                    GgmlAsrStreamingTranscriptUpdate::final_(test_update(frame, revision, text))
                }
            };
            Ok(vec![update])
        }
    }

    impl GgmlAsrStreamingTranscriptDriver for ScriptDriver {
        fn push_audio(
            &mut self,
            frame: RealtimeAudioFrame,
        ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
            self.last_frame = Some(frame.clone());
            self.pop_update_for_frame(&frame)
        }

        fn finish_updates(
            &mut self,
        ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
            let frame = self.last_frame.clone().unwrap_or_else(|| test_frame(0, 0));
            self.pop_update_for_frame(&frame)
        }
    }

    fn request(partial_results: bool) -> GgmlAsrStreamingSessionRequest {
        GgmlAsrStreamingSessionRequest {
            runtime_source_path: PathBuf::from("fixtures/qwen.gguf"),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            request_options: GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: GgmlAsrBackendPreference::Auto,
            session_context: crate::NativeAsrSessionContext::new("rt_ggml_transcript_session"),
            session_config: crate::NativeAsrStreamingSessionConfig::new()
                .with_partial_results(partial_results)
                .into(),
        }
    }

    fn test_clock() -> Box<dyn FnMut() -> String + Send> {
        let mut millis = 0_u32;
        Box::new(move || {
            let value = millis;
            millis += 1;
            format!("2026-06-05T00:00:00.{value:03}Z")
        })
    }

    fn test_frame(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
        let format = crate::RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(20).unwrap();
        RealtimeAudioFrame::new(seq, start_ms, format, vec![0; sample_count]).unwrap()
    }

    fn test_update(frame: &RealtimeAudioFrame, revision: u64, text: &str) -> TranscriptUpdate {
        TranscriptUpdate::new(
            "utt_ggml_streaming",
            "seg_ggml_streaming",
            revision,
            text,
            frame.start_ms,
            frame.end_ms(),
        )
    }

    fn assert_event_types(events: &[RealtimeEventEnvelope], expected: &[&str]) {
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type)
                .collect::<Vec<_>>(),
            expected
        );
    }

    fn assert_transcript_text(event: &RealtimeEventEnvelope, expected: &str, revision: u64) {
        match &event.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => {
                assert_eq!(partial.text, expected);
                assert_eq!(partial.revision, revision);
                assert!(!partial.is_final);
            }
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => {
                assert_eq!(final_.text, expected);
                assert_eq!(final_.revision, revision);
                assert!(final_.is_final);
            }
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision_event)) => {
                assert_eq!(revision_event.text, expected);
                assert_eq!(revision_event.revision, revision);
                assert!(revision_event.is_final);
            }
            other => panic!("expected transcript event, got {other:?}"),
        }
    }

    #[test]
    fn transcript_session_emits_partials_final_and_post_final_revision() {
        let request = request(true);
        let driver = ScriptDriver::new([
            ScriptStep::Partial {
                revision: 1,
                text: "hel",
            },
            ScriptStep::Partial {
                revision: 2,
                text: "hello wor",
            },
            ScriptStep::Final {
                revision: 3,
                text: "hello world",
            },
            ScriptStep::Partial {
                revision: 4,
                text: "hello world",
            },
            ScriptStep::Partial {
                revision: 5,
                text: "hello, world",
            },
        ]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
        )
        .unwrap();

        assert_event_types(
            &session.poll_events().unwrap(),
            &[
                "session.created",
                "session.configured",
                "audio.input.started",
            ],
        );

        let first = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&first, &["transcript.partial"]);
        assert_transcript_text(&first[0], "hel", 1);

        let second = session.push_audio(test_frame(2, 20)).unwrap();
        assert_event_types(&second, &["transcript.partial"]);
        assert_transcript_text(&second[0], "hello wor", 2);

        let final_event = session.push_audio(test_frame(3, 40)).unwrap();
        assert_event_types(&final_event, &["transcript.final"]);
        assert_transcript_text(&final_event[0], "hello world", 3);
        let final_event_id = final_event[0].event_id.clone();

        let duplicate = session.push_audio(test_frame(4, 60)).unwrap();
        assert!(duplicate.is_empty());

        let revision = session.push_audio(test_frame(5, 80)).unwrap();
        assert_event_types(&revision, &["transcript.revision"]);
        assert_transcript_text(&revision[0], "hello, world", 5);
        assert!(matches!(
            &revision[0].event,
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision))
                if revision.revises_event_id.as_ref() == Some(&final_event_id)
        ));
    }

    #[test]
    fn transcript_session_finalizes_suppressed_partial_on_flush() {
        let request = request(false);
        let driver = ScriptDriver::new([ScriptStep::Partial {
            revision: 1,
            text: "held until flush",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());
        let flushed = session.flush().unwrap();

        assert_event_types(&flushed, &["transcript.final"]);
        assert_transcript_text(&flushed[0], "held until flush", 1);
    }

    #[test]
    fn transcript_session_finish_stops_audio_after_finalizing_pending_output() {
        let request = request(false);
        let driver = ScriptDriver::new([ScriptStep::Partial {
            revision: 1,
            text: "finish me",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();
        let _ = session.push_audio(test_frame(1, 0)).unwrap();

        let finished = session.finish().unwrap();

        assert_event_types(&finished, &["transcript.final", "audio.input.stopped"]);
        assert_transcript_text(&finished[0], "finish me", 1);
    }

    #[test]
    fn transcript_session_finalize_utterance_resets_for_next_partial() {
        let request = request(true);
        let driver = ScriptDriver::new([
            ScriptStep::Partial {
                revision: 1,
                text: "first partial",
            },
            ScriptStep::Final {
                revision: 2,
                text: "first final",
            },
            ScriptStep::Partial {
                revision: 3,
                text: "second partial",
            },
            ScriptStep::Final {
                revision: 4,
                text: "second final",
            },
        ]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        let first_partial = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&first_partial, &["transcript.partial"]);
        assert_transcript_text(&first_partial[0], "first partial", 1);

        let first_finalized = session.finalize_utterance().unwrap();
        assert_event_types(
            &first_finalized,
            &["transcript.final", "audio.input.started"],
        );
        assert_transcript_text(&first_finalized[0], "first final", 2);

        let second_partial = session.push_audio(test_frame(2, 20)).unwrap();
        assert_event_types(&second_partial, &["transcript.partial"]);
        assert_transcript_text(&second_partial[0], "second partial", 3);

        let second_finalized = session.finalize_utterance().unwrap();
        assert_event_types(
            &second_finalized,
            &["transcript.final", "audio.input.started"],
        );
        assert_transcript_text(&second_finalized[0], "second final", 4);
    }

    #[test]
    fn punctuation_stage_applies_only_to_unpunctuated_architectures() {
        // Same capability fact as the batch gate: firered (catalog
        // emits_punctuation = false) opts in; qwen (already punctuates) and
        // unknown architectures stay off.
        assert!(streaming_punctuation_stage_applies(
            firered_aed_runtime_descriptor_v1().model_architecture
        ));
        assert!(!streaming_punctuation_stage_applies(
            qwen3_asr_runtime_descriptor_v1().model_architecture
        ));
        assert!(!streaming_punctuation_stage_applies("no-such-architecture"));
    }

    #[test]
    fn final_updates_are_punctuated_partials_are_not() {
        // The FINAL-only punctuation stage runs at the single point where a
        // driver update becomes a transcript event, so every entry path
        // (push_audio / poll / flush / finalize / finish) is covered; partial
        // updates must reach the emitter untouched.
        let request = request(true);
        let driver = ScriptDriver::new([
            ScriptStep::Partial {
                revision: 1,
                text: "你好世",
            },
            ScriptStep::Final {
                revision: 2,
                text: "你好世界",
            },
        ]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock_and_punctuator(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
            Some(Box::new(|text: &str| Ok(format!("{text}。")))),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        let partial = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&partial, &["transcript.partial"]);
        assert_transcript_text(&partial[0], "你好世", 1);

        let final_event = session.push_audio(test_frame(2, 20)).unwrap();
        assert_event_types(&final_event, &["transcript.final"]);
        assert_transcript_text(&final_event[0], "你好世界。", 2);
    }

    #[test]
    fn final_punctuator_error_keeps_original_text() {
        // Fail-closed, mirroring the batch stage: a classifier failure leaves
        // the driver's text exactly as produced instead of dropping the final.
        let request = request(true);
        let driver = ScriptDriver::new([ScriptStep::Final {
            revision: 1,
            text: "你好世界",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock_and_punctuator(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
            Some(Box::new(|_: &str| {
                Err(PunctuationError::Classifier("forward failed".to_string()))
            })),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        let final_event = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&final_event, &["transcript.final"]);
        assert_transcript_text(&final_event[0], "你好世界", 1);
    }

    #[test]
    fn promoted_pending_partial_is_punctuated_on_flush() {
        // A driver that only ever emits partials relies on the session to
        // promote the pending tail into a FINAL on flush/finalize/finish
        // (`finalize_pending_output`); that promotion path never goes through
        // `emit_update`'s Final branch, so it must run the punctuation stage
        // itself -- this is the seam the ScriptStep::Final-based tests above
        // do not cover.
        let request = request(false);
        let driver = ScriptDriver::new([ScriptStep::Partial {
            revision: 1,
            text: "你好世界",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock_and_punctuator(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
            Some(Box::new(|text: &str| Ok(format!("{text}。")))),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());
        let flushed = session.flush().unwrap();

        assert_event_types(&flushed, &["transcript.final"]);
        assert_transcript_text(&flushed[0], "你好世界。", 1);
    }

    #[test]
    fn promoted_pending_partial_is_punctuated_on_finalize_utterance() {
        // Same promotion seam via finalize_utterance with visible partials:
        // the emitted partial must stay untouched, and the tail promoted at
        // the utterance boundary must be punctuated.
        let request = request(true);
        let driver = ScriptDriver::new([ScriptStep::Partial {
            revision: 1,
            text: "你好世界",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock_and_punctuator(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
            Some(Box::new(|text: &str| Ok(format!("{text}。")))),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        let partial = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&partial, &["transcript.partial"]);
        assert_transcript_text(&partial[0], "你好世界", 1);

        let finalized = session.finalize_utterance().unwrap();
        assert_event_types(&finalized, &["transcript.final", "audio.input.started"]);
        assert_transcript_text(&finalized[0], "你好世界。", 1);
    }

    #[test]
    fn sessions_without_punctuator_leave_finals_untouched() {
        // `new_with_clock` (and any family the stage does not apply to)
        // carries no punctuator: finals pass through byte-for-byte.
        let request = request(true);
        let driver = ScriptDriver::new([ScriptStep::Final {
            revision: 1,
            text: "hello world",
        }]);
        let mut session = GgmlAsrStreamingTranscriptSession::new_with_clock(
            "script-streaming-executor",
            &request,
            driver,
            test_clock(),
        )
        .unwrap();
        let _ = session.poll_events().unwrap();

        let final_event = session.push_audio(test_frame(1, 0)).unwrap();
        assert_event_types(&final_event, &["transcript.final"]);
        assert_transcript_text(&final_event[0], "hello world", 1);
    }
}
