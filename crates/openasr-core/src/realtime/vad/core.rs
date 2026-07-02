use thiserror::Error;

#[path = "internal/decision.rs"]
mod decision;
#[path = "internal/state_machine.rs"]
mod state_machine;
#[path = "internal/timing.rs"]
mod timing;

use super::audio::RealtimeAudioFrame;
use super::events::TranscriptUtteranceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMode {
    Energy,
    ExternalProbability,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadConfig {
    pub frame_duration_ms: u32,
    pub speech_start_ms: u32,
    pub speech_stop_ms: u32,
    pub pre_roll_ms: u32,
    pub max_utterance_ms: Option<u32>,
    pub no_speech_timeout_ms: Option<u32>,
    pub mode: VadMode,
    pub energy_threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 20,
            speech_start_ms: 200,
            speech_stop_ms: 600,
            pre_roll_ms: 320,
            max_utterance_ms: Some(30_000),
            no_speech_timeout_ms: Some(10_000),
            mode: VadMode::Energy,
            energy_threshold: 0.02,
        }
    }
}

impl VadConfig {
    pub fn validate(&self) -> Result<(), VadConfigError> {
        if self.frame_duration_ms == 0 {
            return Err(VadConfigError::ZeroFrameDuration);
        }
        if self.speech_start_ms == 0 {
            return Err(VadConfigError::ZeroSpeechStart);
        }
        if self.speech_stop_ms == 0 {
            return Err(VadConfigError::ZeroSpeechStop);
        }
        if !(0.0..=1.0).contains(&self.energy_threshold) {
            return Err(VadConfigError::InvalidEnergyThreshold {
                threshold: self.energy_threshold,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum VadConfigError {
    #[error("Realtime VAD frame duration must be greater than 0 ms.")]
    ZeroFrameDuration,
    #[error("Realtime VAD speech_start_ms must be greater than 0 ms.")]
    ZeroSpeechStart,
    #[error("Realtime VAD speech_stop_ms must be greater than 0 ms.")]
    ZeroSpeechStop,
    #[error("Realtime VAD energy threshold must be between 0.0 and 1.0, got {threshold}.")]
    InvalidEnergyThreshold { threshold: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadDecision {
    Speech,
    Silence,
    Probability(f32),
}

impl VadDecision {
    fn is_speech(self, threshold: f32) -> bool {
        decision::decision_is_speech(self, threshold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrameDecision {
    pub decision: VadDecision,
    pub rms: Option<f32>,
}

impl VadFrameDecision {
    pub fn from_energy(frame: &RealtimeAudioFrame, threshold: f32) -> Self {
        decision::frame_decision_from_energy(frame, threshold)
    }

    pub fn speech() -> Self {
        Self {
            decision: VadDecision::Speech,
            rms: None,
        }
    }

    pub fn silence() -> Self {
        Self {
            decision: VadDecision::Silence,
            rms: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    WaitingForSpeech,
    InSpeech,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechBoundaryEvent {
    SpeechStarted {
        utterance_id: TranscriptUtteranceId,
        start_ms: u64,
    },
    SpeechStopped {
        utterance_id: TranscriptUtteranceId,
        start_ms: u64,
        end_ms: u64,
    },
    MaxUtterance {
        utterance_id: TranscriptUtteranceId,
        start_ms: u64,
        end_ms: u64,
    },
    NoSpeechTimeout {
        timeout_ms: u32,
        at_ms: u64,
    },
}

#[derive(Debug, Clone)]
pub struct VadStateMachine {
    config: VadConfig,
    state: VadState,
    next_utterance_index: u64,
    consecutive_speech_ms: u64,
    consecutive_silence_ms: u64,
    speech_candidate_start_ms: Option<u64>,
    active_utterance_id: Option<TranscriptUtteranceId>,
    active_start_ms: Option<u64>,
    waiting_start_ms: Option<u64>,
    speech_started_once: bool,
    no_speech_timeout_emitted: bool,
}

impl VadStateMachine {
    pub fn new(config: VadConfig) -> Result<Self, VadConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            state: VadState::WaitingForSpeech,
            next_utterance_index: 1,
            consecutive_speech_ms: 0,
            consecutive_silence_ms: 0,
            speech_candidate_start_ms: None,
            active_utterance_id: None,
            active_start_ms: None,
            waiting_start_ms: None,
            speech_started_once: false,
            no_speech_timeout_emitted: false,
        })
    }

    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    pub fn state(&self) -> VadState {
        self.state
    }

    pub fn process_energy_frame(&mut self, frame: &RealtimeAudioFrame) -> Vec<SpeechBoundaryEvent> {
        self.process_energy_frame_with_speech(frame).0
    }

    pub fn process_energy_frame_with_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
    ) -> (Vec<SpeechBoundaryEvent>, bool) {
        let decision = VadFrameDecision::from_energy(frame, self.config.energy_threshold);
        self.process_decision_with_speech(frame, decision)
    }

    pub fn process_decision(
        &mut self,
        frame: &RealtimeAudioFrame,
        decision: VadFrameDecision,
    ) -> Vec<SpeechBoundaryEvent> {
        self.process_decision_with_speech(frame, decision).0
    }

    pub fn process_decision_with_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
        decision: VadFrameDecision,
    ) -> (Vec<SpeechBoundaryEvent>, bool) {
        if self.state == VadState::Closed {
            return (Vec::new(), false);
        }

        let mut events = Vec::new();
        let is_speech = decision.decision.is_speech(self.config.energy_threshold);
        let frame_duration_ms = timing::frame_duration_ms(frame);
        self.process_state(frame, is_speech, frame_duration_ms, &mut events);
        (events, is_speech)
    }

    pub fn reset(&mut self) {
        self.clear_state_after_boundary();
        self.waiting_start_ms = None;
        self.speech_started_once = false;
        self.no_speech_timeout_emitted = false;
    }

    pub fn close(&mut self) {
        self.state = VadState::Closed;
    }

    fn maybe_emit_no_speech_timeout(
        &mut self,
        frame: &RealtimeAudioFrame,
        events: &mut Vec<SpeechBoundaryEvent>,
    ) {
        let Some(timeout_ms) = self.config.no_speech_timeout_ms else {
            return;
        };
        if !self.speech_started_once
            && !self.no_speech_timeout_emitted
            && timing::reached_no_speech_timeout(self.waiting_start_ms, frame, timeout_ms)
        {
            self.no_speech_timeout_emitted = true;
            events.push(SpeechBoundaryEvent::NoSpeechTimeout {
                timeout_ms,
                at_ms: frame.end_ms(),
            });
        }
    }

    fn next_utterance_id(&mut self) -> TranscriptUtteranceId {
        let id = TranscriptUtteranceId(format!("utt_{:06}", self.next_utterance_index));
        self.next_utterance_index += 1;
        id
    }

    fn stop_event(&self, end_ms: u64) -> SpeechBoundaryEvent {
        SpeechBoundaryEvent::SpeechStopped {
            utterance_id: self
                .active_utterance_id
                .clone()
                .unwrap_or_else(|| TranscriptUtteranceId("utt_unknown".to_string())),
            start_ms: self.active_start_ms.unwrap_or(0),
            end_ms,
        }
    }

    fn clear_state_after_boundary(&mut self) {
        self.state = VadState::WaitingForSpeech;
        self.consecutive_speech_ms = 0;
        self.consecutive_silence_ms = 0;
        self.speech_candidate_start_ms = None;
        self.active_utterance_id = None;
        self.active_start_ms = None;
        self.waiting_start_ms = None;
    }
}
