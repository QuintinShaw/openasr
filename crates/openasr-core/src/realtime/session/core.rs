use thiserror::Error;

use super::{
    audio::{RealtimeAudioFormat, RealtimeAudioFrame, RealtimeFrameError},
    buffer::{BufferedUtterance, RealtimeBuffer, RealtimeBufferConfig, RealtimeBufferError},
    events::{
        AudioInputStartedEvent, AudioInputStoppedEvent, RealtimeAudioInputEvent, RealtimeEvent,
        RealtimeEventEnvelope, RealtimeEventSequencer, RealtimeLifecycleEvent, RealtimeSessionId,
        RealtimeTranscriptEvent, RealtimeVadEvent, SessionClosedEvent, SessionConfiguredEvent,
        SessionCreatedEvent, SessionTranslationSummary, SessionVadSummary,
    },
    transcript::{TranscriptLifecycle, TranscriptRevisionPolicy},
    vad::{
        SpeechBoundaryEvent, VadConfig, VadConfigError, VadDecision, VadFrameDecision, VadMode,
        VadStateMachine,
    },
};
use crate::diarize::vad::{FireRedStreamingVad, RealtimeNeuralVadEngine, SileroStreamingVad};

/// The concrete streaming neural detector backing an `ExternalProbability`
/// session: Silero (default) or FireRedVAD's causal Stream-VAD
/// (`OPENASR_VAD=firered-stream`, opt-in -- see
/// [`crate::diarize::vad::realtime_neural_vad_engine`]). Both expose the same
/// `accept_frame`/`reset` contract; this enum is the seam so
/// `RealtimeSessionController` does not need a trait object for two
/// concrete, non-`dyn`-safe-by-necessity implementations.
#[derive(Debug)]
enum NeuralVad {
    // Boxed: `SileroStreamingVad` carries a much larger inline buffer than
    // `FireRedStreamingVad`, and this enum otherwise sizes to its largest
    // variant (clippy::large_enum_variant).
    Silero(Box<SileroStreamingVad>),
    FireRedStream(Box<FireRedStreamingVad>),
}

impl NeuralVad {
    fn accept_frame(&mut self, samples: &[i16]) -> f32 {
        match self {
            NeuralVad::Silero(detector) => detector.accept_frame(samples),
            NeuralVad::FireRedStream(detector) => detector.accept_frame(samples),
        }
    }

    fn reset(&mut self) {
        match self {
            NeuralVad::Silero(detector) => detector.reset(),
            NeuralVad::FireRedStream(detector) => detector.reset(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeSessionConfig {
    pub session_id: RealtimeSessionId,
    pub model_id: String,
    pub audio_format: RealtimeAudioFormat,
    pub vad: VadConfig,
    pub buffer: RealtimeBufferConfig,
    pub partial_results: bool,
    pub word_timestamps: bool,
    pub diarize: bool,
    pub translation: SessionTranslationSummary,
    pub created_at: String,
    pub trace_id: Option<String>,
    pub request_id: Option<String>,
}

impl RealtimeSessionConfig {
    fn validate_required_field(
        value: &str,
        name: &'static str,
    ) -> Result<(), RealtimeSessionError> {
        if value.trim().is_empty() {
            return Err(RealtimeSessionError::InvalidConfig {
                message: format!("{name} must not be empty"),
            });
        }
        Ok(())
    }

    pub fn new(
        session_id: impl Into<String>,
        model_id: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Self {
        Self {
            session_id: RealtimeSessionId(session_id.into()),
            model_id: model_id.into(),
            audio_format: RealtimeAudioFormat::pcm16_mono_16khz(),
            vad: VadConfig::default(),
            buffer: RealtimeBufferConfig::default(),
            partial_results: true,
            word_timestamps: false,
            diarize: false,
            translation: SessionTranslationSummary::disabled(),
            created_at: created_at.into(),
            trace_id: None,
            request_id: None,
        }
    }

    pub fn validate(&self) -> Result<(), RealtimeSessionError> {
        Self::validate_required_field(&self.session_id.0, "session_id")?;
        Self::validate_required_field(&self.model_id, "model_id")?;
        Self::validate_required_field(&self.created_at, "created_at")?;
        self.audio_format.validate_normalized()?;
        self.vad.validate()?;
        RealtimeBuffer::new(self.buffer)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeSessionState {
    Created,
    Configured,
    Running,
    Closed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeLifecycleAction {
    Configure,
    StartAudio,
    StopAudio { reason: String },
    Close { reason: String },
}

#[derive(Debug)]
pub struct RealtimeSessionController {
    config: RealtimeSessionConfig,
    state: RealtimeSessionState,
    sequencer: RealtimeEventSequencer,
    pub vad: VadStateMachine,
    /// Streaming neural detector, present only when the configured VAD mode is
    /// `ExternalProbability` and the model loaded. Feeds probabilities into
    /// `vad`; all endpointing stays in the state machine.
    neural_vad: Option<NeuralVad>,
    pub buffer: RealtimeBuffer,
    pub transcript: TranscriptLifecycle,
}

impl RealtimeSessionController {
    fn lifecycle_transition_spec_v0(
        &self,
        action: &RealtimeLifecycleAction,
    ) -> Option<(RealtimeSessionState, RealtimeSessionState, &'static str, RealtimeEvent)> {
        match action {
            RealtimeLifecycleAction::Configure => Some((
                RealtimeSessionState::Created,
                RealtimeSessionState::Configured,
                "configure",
                RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(
                    SessionConfiguredEvent {
                        model: self.config.model_id.clone(),
                        partial_results: self.config.partial_results,
                        word_timestamps: self.config.word_timestamps,
                        diarize: self.config.diarize,
                        translation: self.config.translation.clone(),
                        vad: SessionVadSummary { enabled: true },
                    },
                )),
            )),
            RealtimeLifecycleAction::StartAudio => Some((
                RealtimeSessionState::Configured,
                RealtimeSessionState::Running,
                "start_audio",
                RealtimeEvent::AudioInput(RealtimeAudioInputEvent::Started(
                    AudioInputStartedEvent {},
                )),
            )),
            RealtimeLifecycleAction::StopAudio { reason } => Some((
                RealtimeSessionState::Running,
                RealtimeSessionState::Configured,
                "stop_audio",
                RealtimeEvent::AudioInput(RealtimeAudioInputEvent::Stopped(
                    AudioInputStoppedEvent {
                        reason: reason.clone(),
                    },
                )),
            )),
            RealtimeLifecycleAction::Close { .. } => None,
        }
    }

    fn close_session(
        &mut self,
        next_state: RealtimeSessionState,
        reason: String,
        reset_runtime_state: bool,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.ensure_not_terminal()?;
        self.vad.close();
        if reset_runtime_state {
            self.buffer.reset();
            self.transcript.reset();
            if let Some(neural) = self.neural_vad.as_mut() {
                neural.reset();
            }
        }
        self.state = next_state;
        Ok(self.sequencer.next(
            RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionClosed(SessionClosedEvent {
                reason,
            })),
            created_at,
        ))
    }

    fn emit_with_transition(
        &mut self,
        expected: RealtimeSessionState,
        next: RealtimeSessionState,
        action: &'static str,
        event: RealtimeEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.transition_to(expected, next, action)?;
        self.ensure_not_terminal()?;
        Ok(self.sequencer.next(event, created_at))
    }

    fn emit_passthrough_event(
        &mut self,
        event: RealtimeEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.ensure_not_terminal()?;
        Ok(self.sequencer.next(event, created_at))
    }

    fn transition_to(
        &mut self,
        expected: RealtimeSessionState,
        next: RealtimeSessionState,
        action: &'static str,
    ) -> Result<(), RealtimeSessionError> {
        self.ensure_not_terminal()?;
        if self.state != expected {
            return Err(RealtimeSessionError::InvalidStateTransition {
                from: self.state,
                action,
            });
        }
        self.state = next;
        Ok(())
    }

    pub fn new(config: RealtimeSessionConfig) -> Result<Self, RealtimeSessionError> {
        let sequencer = RealtimeEventSequencer::new(config.session_id.clone())
            .with_trace_id(config.trace_id.clone())
            .with_request_id(config.request_id.clone());
        Self::new_with_sequencer(config, sequencer)
    }

    pub fn new_with_sequencer(
        mut config: RealtimeSessionConfig,
        sequencer: RealtimeEventSequencer,
    ) -> Result<Self, RealtimeSessionError> {
        config.validate()?;
        let neural_vad = if config.vad.mode == VadMode::ExternalProbability {
            // Which neural model backs this session: Silero by default,
            // FireRedVAD's causal Stream-VAD opt-in via `OPENASR_VAD`. See
            // `crate::diarize::vad::realtime_neural_vad_engine`.
            let detector = match crate::diarize::vad::realtime_neural_vad_engine(None) {
                RealtimeNeuralVadEngine::Silero => {
                    SileroStreamingVad::shared().map(|d| NeuralVad::Silero(Box::new(d)))
                }
                RealtimeNeuralVadEngine::FireRedStream => {
                    FireRedStreamingVad::shared().map(|d| NeuralVad::FireRedStream(Box::new(d)))
                }
            };
            match detector {
                Some(detector) => Some(detector),
                None => {
                    // Neural VAD was requested but the model is unavailable.
                    // Downgrade to the energy gate with energy-appropriate
                    // defaults: the configured threshold is a probability (~0.5)
                    // that, used as an RMS gate, would silence all speech; and the
                    // hangover may have been tuned short for the neural detector,
                    // which would chop words on an RMS gate. Restore both to the
                    // energy-safe defaults so the fallback behaves like a real
                    // energy session.
                    config.vad.mode = VadMode::Energy;
                    config.vad.energy_threshold = VadConfig::default().energy_threshold;
                    config.vad.speech_stop_ms = VadConfig::default().speech_stop_ms;
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            vad: VadStateMachine::new(config.vad)?,
            neural_vad,
            buffer: RealtimeBuffer::new(config.buffer)?,
            transcript: TranscriptLifecycle::new(
                TranscriptRevisionPolicy::ExplicitPostFinalRevision,
            ),
            config,
            state: RealtimeSessionState::Created,
            sequencer,
        })
    }

    pub fn state(&self) -> RealtimeSessionState {
        self.state
    }

    pub fn config(&self) -> &RealtimeSessionConfig {
        &self.config
    }

    /// Process one audio frame through the configured VAD, returning any speech
    /// boundary events. Neural mode (`ExternalProbability`) feeds the streaming
    /// Silero probability into the state machine; every other mode (and the
    /// fallback when the neural model is unavailable) uses the energy gate. All
    /// endpointing/hysteresis lives in the state machine, not here.
    pub fn process_vad_frame(&mut self, frame: &RealtimeAudioFrame) -> Vec<SpeechBoundaryEvent> {
        self.process_vad_frame_with_speech(frame).0
    }

    pub fn process_vad_frame_with_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
    ) -> (Vec<SpeechBoundaryEvent>, bool) {
        if self.vad.config().mode == VadMode::ExternalProbability
            && let Some(neural) = self.neural_vad.as_mut()
        {
            let probability = neural.accept_frame(frame.samples());
            return self.vad.process_decision_with_speech(
                frame,
                VadFrameDecision {
                    decision: VadDecision::Probability(probability),
                    rms: None,
                },
            );
        }
        self.vad.process_energy_frame_with_speech(frame)
    }

    pub fn session_created_event(
        &mut self,
        created_at: impl Into<String>,
    ) -> RealtimeEventEnvelope {
        self.sequencer.next(
            RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionCreated(
                SessionCreatedEvent {
                    audio_format: self.config.audio_format,
                },
            )),
            created_at,
        )
    }

    pub fn lifecycle(
        &mut self,
        action: RealtimeLifecycleAction,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        if let Some((expected, next, transition, event)) = self.lifecycle_transition_spec_v0(&action)
        {
            return self.emit_with_transition(expected, next, transition, event, created_at);
        }
        match action {
            RealtimeLifecycleAction::Close { reason } => {
                self.close_session(RealtimeSessionState::Closed, reason, false, created_at)
            }
            RealtimeLifecycleAction::Configure
            | RealtimeLifecycleAction::StartAudio
            | RealtimeLifecycleAction::StopAudio { .. } => unreachable!(),
        }
    }

    pub fn vad_event(
        &mut self,
        event: RealtimeVadEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.emit_passthrough_event(RealtimeEvent::Vad(event), created_at)
    }

    pub fn transcript_event(
        &mut self,
        event: RealtimeTranscriptEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.emit_passthrough_event(RealtimeEvent::Transcript(event), created_at)
    }

    pub fn error_event(
        &mut self,
        event: super::events::RealtimeErrorEvent,
        created_at: impl Into<String>,
    ) -> Result<RealtimeEventEnvelope, RealtimeSessionError> {
        self.emit_passthrough_event(RealtimeEvent::Error(event), created_at)
    }

    pub fn reset(&mut self) -> Result<(), RealtimeSessionError> {
        self.ensure_not_terminal()?;
        if self.state == RealtimeSessionState::Created {
            return Err(RealtimeSessionError::InvalidStateTransition {
                from: self.state,
                action: "reset",
            });
        }
        self.vad.reset();
        self.buffer.reset();
        self.transcript.reset();
        self.state = RealtimeSessionState::Configured;
        Ok(())
    }

    pub fn cancel(
        &mut self,
        end_ms: u64,
        created_at: impl Into<String>,
    ) -> Result<(RealtimeEventEnvelope, Option<BufferedUtterance>), RealtimeSessionError> {
        self.ensure_not_terminal()?;
        let cancelled_utterance = self.buffer.cancel(end_ms);
        let event = self.close_session(
            RealtimeSessionState::Cancelled,
            "cancelled".to_string(),
            true,
            created_at,
        )?;
        Ok((event, cancelled_utterance))
    }

    fn ensure_not_terminal(&self) -> Result<(), RealtimeSessionError> {
        match self.state {
            RealtimeSessionState::Closed | RealtimeSessionState::Cancelled => {
                Err(RealtimeSessionError::SessionClosed)
            }
            RealtimeSessionState::Created
            | RealtimeSessionState::Configured
            | RealtimeSessionState::Running => Ok(()),
        }
    }
}

#[derive(Debug, Error)]
pub enum RealtimeSessionError {
    #[error("Invalid realtime session config: {message}.")]
    InvalidConfig { message: String },
    #[error("{0}")]
    AudioFormat(#[from] RealtimeFrameError),
    #[error("{0}")]
    VadConfig(#[from] VadConfigError),
    #[error("{0}")]
    Buffer(#[from] RealtimeBufferError),
    #[error("Realtime session is already closed or cancelled.")]
    SessionClosed,
    #[error("Cannot {action} realtime session from state {from:?}.")]
    InvalidStateTransition {
        from: RealtimeSessionState,
        action: &'static str,
    },
}
