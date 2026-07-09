use std::collections::VecDeque;

use crate::realtime::{
    RealtimeErrorCode, RealtimeErrorEvent, RealtimeEventEnvelope, RealtimeLifecycleAction,
    RealtimeSessionConfig, RealtimeSessionController, RealtimeSessionState,
    RealtimeTranscriptEvent, TranscriptLifecycleResult, TranscriptUpdate,
};

use super::{
    NativeAsrError, NativeAsrRequestOptions, NativeAsrSessionContext,
    NativeAsrStreamingSessionConfig,
};

pub(crate) struct NativeStreamingTranscriptEmitter {
    controller: RealtimeSessionController,
    session_config: NativeAsrStreamingSessionConfig,
    pending_events: VecDeque<RealtimeEventEnvelope>,
    pending_partial_update: Option<TranscriptUpdate>,
}

impl NativeStreamingTranscriptEmitter {
    pub(crate) fn new_started(
        context: NativeAsrSessionContext,
        model_id: String,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
        created_at: impl Into<String>,
        configured_at: impl Into<String>,
        started_at: impl Into<String>,
    ) -> Result<Self, NativeAsrError> {
        session_config.validate()?;
        let mut realtime_config =
            RealtimeSessionConfig::new(context.session_id.0.clone(), model_id, created_at.into());
        realtime_config.audio_format = session_config.audio_format;
        realtime_config.partial_results = options.partial_results && session_config.partial_results;
        realtime_config.word_timestamps = options.word_timestamps && session_config.word_timestamps;
        realtime_config.diarize = options.diarize;
        realtime_config.trace_id = context.trace_id;
        realtime_config.request_id = context.request_id;

        let mut controller =
            RealtimeSessionController::new(realtime_config).map_err(session_failed)?;
        let mut pending_events = VecDeque::new();
        pending_events
            .push_back(controller.session_created_event(controller.config().created_at.clone()));
        pending_events.push_back(
            controller
                .lifecycle(RealtimeLifecycleAction::Configure, configured_at)
                .map_err(session_failed)?,
        );
        pending_events.push_back(
            controller
                .lifecycle(RealtimeLifecycleAction::StartAudio, started_at)
                .map_err(session_failed)?,
        );

        if pending_events.len() > session_config.backpressure.max_queued_events {
            return Err(NativeAsrError::session_backpressure(
                "initial event queue reached max_queued_events",
            ));
        }

        Ok(Self {
            controller,
            session_config,
            pending_events,
            pending_partial_update: None,
        })
    }

    pub(crate) fn session_id(&self) -> &str {
        &self.controller.config().session_id.0
    }

    pub(crate) fn state(&self) -> RealtimeSessionState {
        self.controller.state()
    }

    pub(crate) fn partial_results_enabled(&self) -> bool {
        self.controller.config().partial_results
    }

    #[cfg(test)]
    pub(crate) fn word_timestamps_enabled(&self) -> bool {
        self.controller.config().word_timestamps
    }

    pub(crate) fn drain_pending_events(&mut self) -> Vec<RealtimeEventEnvelope> {
        self.pending_events.drain(..).collect()
    }

    pub(crate) fn ensure_push_capacity(
        &self,
        queued_audio_frames: usize,
    ) -> Result<(), NativeAsrError> {
        if self.pending_events.len() >= self.session_config.backpressure.max_queued_events {
            return Err(NativeAsrError::session_backpressure(
                "output event queue reached max_queued_events",
            ));
        }
        if queued_audio_frames >= self.session_config.backpressure.max_queued_audio_frames {
            return Err(NativeAsrError::session_backpressure(
                "audio frame queue reached max_queued_audio_frames",
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_output_capacity(
        &self,
        additional_events: usize,
    ) -> Result<(), NativeAsrError> {
        if self.pending_events.len() + additional_events
            > self.session_config.backpressure.max_queued_events
        {
            return Err(NativeAsrError::session_backpressure(
                "output event queue reached max_queued_events",
            ));
        }
        Ok(())
    }

    pub(crate) fn apply_partial(
        &mut self,
        update: TranscriptUpdate,
        created_at: impl Into<String>,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if !self.partial_results_enabled() {
            self.pending_partial_update = Some(update);
            return Ok(Vec::new());
        }
        let result = self.controller.transcript.apply_partial(update.clone());
        if matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(_))
        ) {
            self.pending_partial_update = Some(update);
        }
        self.emit_transcript_lifecycle(result, created_at)
    }

    pub(crate) fn apply_final(
        &mut self,
        update: TranscriptUpdate,
        created_at: impl Into<String>,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let result = self.controller.transcript.apply_final(update, None);
        if matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(_))
        ) {
            self.pending_partial_update = None;
        }
        self.emit_transcript_lifecycle(result, created_at)
    }

    /// Takes the pending tail partial (if any) so the caller can post-process
    /// its text before promoting it into a FINAL with [`Self::apply_final`].
    /// Deliberately the only way to promote the pending tail: a session that
    /// runs a FINAL post-processing stage (e.g. the ggml streaming session's
    /// punctuation stage) gets no emitter-level shortcut that could promote
    /// the text as-is behind that stage's back.
    pub(crate) fn take_pending_partial_update(&mut self) -> Option<TranscriptUpdate> {
        self.pending_partial_update.take()
    }

    pub(crate) fn lifecycle_event(
        &mut self,
        action: RealtimeLifecycleAction,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, NativeAsrError> {
        self.controller
            .lifecycle(action, created_at)
            .map_err(session_failed)
    }

    pub(crate) fn audio_stopped_event(
        &mut self,
        reason: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, NativeAsrError> {
        self.controller
            .lifecycle(
                RealtimeLifecycleAction::StopAudio {
                    reason: reason.into(),
                },
                created_at,
            )
            .map_err(session_failed)
    }

    pub(crate) fn error_event(
        &mut self,
        event: RealtimeErrorEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, NativeAsrError> {
        self.controller
            .error_event(event, created_at)
            .map_err(session_failed)
    }

    pub(crate) fn reset_for_next_utterance(
        &mut self,
        started_at: impl Into<String>,
    ) -> Result<Option<RealtimeEventEnvelope>, NativeAsrError> {
        if matches!(
            self.controller.state(),
            RealtimeSessionState::Closed | RealtimeSessionState::Cancelled
        ) {
            return Ok(None);
        }
        self.pending_partial_update = None;
        self.controller.reset().map_err(session_failed)?;
        self.controller
            .lifecycle(RealtimeLifecycleAction::StartAudio, started_at)
            .map(Some)
            .map_err(session_failed)
    }

    pub(crate) fn cancel(
        &mut self,
        message: impl Into<String>,
        error_at: impl Into<String>,
        closed_at: impl Into<String>,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let error = self.error_event(
            RealtimeErrorEvent {
                code: RealtimeErrorCode::Cancelled,
                message: message.into(),
                recoverable: false,
            },
            error_at,
        )?;
        let (closed, _) = self
            .controller
            .cancel(0, closed_at)
            .map_err(session_failed)?;
        Ok(vec![error, closed])
    }

    pub(crate) fn close_if_running(
        &mut self,
        stop_reason: impl Into<String>,
        stopped_at: impl Into<String>,
    ) -> Result<Option<RealtimeEventEnvelope>, NativeAsrError> {
        if self.controller.state() != RealtimeSessionState::Running {
            return Ok(None);
        }
        let event = self.audio_stopped_event(stop_reason, stopped_at)?;
        Ok(Some(event))
    }

    pub(crate) fn close_session(
        &mut self,
        reason: impl Into<String>,
        closed_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, NativeAsrError> {
        self.lifecycle_event(
            RealtimeLifecycleAction::Close {
                reason: reason.into(),
            },
            closed_at,
        )
    }

    pub(crate) fn emit_transcript_lifecycle(
        &mut self,
        lifecycle_result: TranscriptLifecycleResult,
        created_at: impl Into<String>,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        let TranscriptLifecycleResult::Event(event) = lifecycle_result else {
            return Ok(Vec::new());
        };
        Ok(vec![self.emit_transcript_event(event, created_at)?])
    }

    fn emit_transcript_event(
        &mut self,
        event: RealtimeTranscriptEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, NativeAsrError> {
        let final_segment = match &event {
            RealtimeTranscriptEvent::Final(final_event) => Some((
                final_event.utterance_id.clone(),
                final_event.segment_id.clone(),
                final_event.revision,
            )),
            _ => None,
        };
        let envelope = self
            .controller
            .transcript_event(event, created_at)
            .map_err(session_failed)?;
        if let Some((utterance_id, segment_id, revision)) = final_segment {
            self.controller.transcript.record_final_event_id(
                &utterance_id,
                &segment_id,
                revision,
                envelope.event_id.clone(),
            );
        }
        Ok(envelope)
    }
}

fn session_failed(error: impl ToString) -> NativeAsrError {
    NativeAsrError::SessionFailed {
        message: error.to_string(),
    }
}
