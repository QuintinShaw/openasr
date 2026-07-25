use thiserror::Error;

use super::{
    RealtimeAudioFrame, SpeechBoundaryEvent, VadConfig, VadConfigError, VadDecision,
    VadFrameDecision, VadMode, VadState,
};
use crate::diarize::vad::{
    DEFAULT_NEURAL_SPEECH_START_MS, DEFAULT_NEURAL_VAD_THRESHOLD, FireRedStreamingVad,
    SHORT_NEURAL_SPEECH_STOP_MS,
};

/// The shared endpointing provider for all realtime streaming surfaces.
///
/// The provider owns both the per-session FireRed stream cache and the VAD state
/// machine. The FireRed weights remain process-wide in `diarize::vad`; only the
/// small causal cache is session-local. Replacing the vendored detector therefore
/// only changes this factory, never server or embedding call sites.
#[derive(Debug)]
pub struct StreamingVadEngine {
    state_machine: Option<super::VadStateMachine>,
    neural_detector: Option<FireRedStreamingVad>,
}

impl StreamingVadEngine {
    /// Creates a provider from the resolved config. `ExternalProbability` is the
    /// neural FireRed mode, `Energy` uses the explicit RMS fallback, and
    /// `Disabled` bypasses endpointing entirely.
    pub fn new(config: VadConfig) -> Result<Self, StreamingVadEngineError> {
        if config.mode == VadMode::Disabled {
            return Ok(Self {
                state_machine: None,
                neural_detector: None,
            });
        }
        config.validate()?;
        let neural_detector = if config.mode == VadMode::ExternalProbability {
            Some(FireRedStreamingVad::shared().ok_or(StreamingVadEngineError::NeuralUnavailable)?)
        } else {
            None
        };
        Ok(Self {
            state_machine: Some(super::VadStateMachine::new(config)?),
            neural_detector,
        })
    }

    pub fn mode(&self) -> VadMode {
        self.state_machine
            .as_ref()
            .map(|state_machine| state_machine.config().mode)
            .unwrap_or(VadMode::Disabled)
    }

    pub fn config(&self) -> Option<&VadConfig> {
        self.state_machine
            .as_ref()
            .map(super::VadStateMachine::config)
    }

    pub fn state(&self) -> VadState {
        self.state_machine
            .as_ref()
            .map(super::VadStateMachine::state)
            .unwrap_or(VadState::WaitingForSpeech)
    }

    pub fn is_enabled(&self) -> bool {
        self.state_machine.is_some()
    }

    /// Runs one validated realtime frame through the configured provider.
    pub fn process_frame_with_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
    ) -> (Vec<SpeechBoundaryEvent>, bool) {
        let Some(state_machine) = self.state_machine.as_mut() else {
            return (Vec::new(), false);
        };
        if let Some(detector) = self.neural_detector.as_mut() {
            let probability = detector.accept_frame(frame.samples());
            return state_machine.process_decision_with_speech(
                frame,
                VadFrameDecision {
                    decision: VadDecision::Probability(probability),
                    rms: None,
                },
            );
        }
        state_machine.process_energy_frame_with_speech(frame)
    }

    pub fn process_frame(&mut self, frame: &RealtimeAudioFrame) -> Vec<SpeechBoundaryEvent> {
        self.process_frame_with_speech(frame).0
    }

    pub fn reset(&mut self) {
        if let Some(state_machine) = self.state_machine.as_mut() {
            state_machine.reset();
        }
        if let Some(detector) = self.neural_detector.as_mut() {
            detector.reset();
        }
    }

    pub fn close(&mut self) {
        if let Some(state_machine) = self.state_machine.as_mut() {
            state_machine.close();
        }
    }
}

#[derive(Debug, Error)]
pub enum StreamingVadEngineError {
    #[error("{0}")]
    Config(#[from] VadConfigError),
    #[error(
        "Stream-VAD is unavailable: vendored weights failed to parse (build-integrity problem)."
    )]
    NeuralUnavailable,
}

/// Resolves the realtime VAD choice. `OPENASR_VAD` wins, then an explicit
/// engine string, and neural is the default. Unknown values intentionally fall
/// back to the shared default so existing server/CLI config handling stays
/// compatible.
pub fn resolve_streaming_vad_mode(engine: Option<&str>) -> VadMode {
    let requested = std::env::var("OPENASR_VAD")
        .ok()
        .or_else(|| engine.map(ToOwned::to_owned));
    match requested
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("energy" | "rms") => VadMode::Energy,
        Some("disabled" | "disable" | "off" | "none") => VadMode::Disabled,
        Some("neural" | "firered-stream" | "firered_stream" | "fireredstream") | None => {
            VadMode::ExternalProbability
        }
        Some(_) => VadMode::ExternalProbability,
    }
}

/// Returns the mode-specific endpointing defaults used by every streaming API.
pub fn default_streaming_vad_config(mode: VadMode, frame_duration_ms: u32) -> VadConfig {
    let mut config = VadConfig {
        frame_duration_ms,
        mode,
        ..VadConfig::default()
    };
    if mode == VadMode::ExternalProbability {
        config.speech_start_ms = DEFAULT_NEURAL_SPEECH_START_MS;
        config.speech_stop_ms = SHORT_NEURAL_SPEECH_STOP_MS;
        config.energy_threshold = DEFAULT_NEURAL_VAD_THRESHOLD;
    } else if mode == VadMode::Energy {
        config.speech_start_ms = 200;
        config.speech_stop_ms = 600;
        config.energy_threshold = 0.02;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_defaults_to_neural_and_preserves_explicit_modes() {
        assert_eq!(
            resolve_streaming_vad_mode(Some("neural")),
            VadMode::ExternalProbability
        );
        assert_eq!(resolve_streaming_vad_mode(Some("energy")), VadMode::Energy);
        assert_eq!(
            resolve_streaming_vad_mode(Some("disabled")),
            VadMode::Disabled
        );
    }

    #[test]
    fn neural_defaults_keep_realtime_endpointing_semantics() {
        let config = default_streaming_vad_config(VadMode::ExternalProbability, 20);
        assert_eq!(config.speech_start_ms, DEFAULT_NEURAL_SPEECH_START_MS);
        assert_eq!(config.speech_stop_ms, SHORT_NEURAL_SPEECH_STOP_MS);
        assert_eq!(config.energy_threshold, DEFAULT_NEURAL_VAD_THRESHOLD);
    }

    fn boundary_shape(event: SpeechBoundaryEvent) -> (&'static str, u64, u64) {
        match event {
            SpeechBoundaryEvent::SpeechStarted { start_ms, .. } => ("start", start_ms, 0),
            SpeechBoundaryEvent::SpeechStopped {
                start_ms, end_ms, ..
            } => ("stop", start_ms, end_ms),
            SpeechBoundaryEvent::MaxUtterance {
                start_ms, end_ms, ..
            } => ("max", start_ms, end_ms),
            SpeechBoundaryEvent::NoSpeechTimeout { timeout_ms, at_ms } => {
                ("timeout", timeout_ms as u64, at_ms)
            }
        }
    }

    #[test]
    fn shared_neural_engines_are_session_isolated_and_reset_deterministically() {
        let Some(_) = crate::diarize::vad::shared_model() else {
            return;
        };
        let config = default_streaming_vad_config(VadMode::ExternalProbability, 20);
        let mut first = StreamingVadEngine::new(config).unwrap();
        let mut second = StreamingVadEngine::new(config).unwrap();
        let format = crate::realtime::RealtimeAudioFormat::pcm16_mono_16khz();
        let pcm = crate::diarize::vad::test_fixtures::golden_pcm();
        let frames = pcm
            .chunks(320)
            .enumerate()
            .filter(|(_, samples)| samples.len() == 320)
            .map(|(index, samples)| {
                crate::realtime::RealtimeAudioFrame::new(
                    index as u64,
                    index as u64 * 20,
                    format,
                    samples.to_vec(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let first_events = frames
            .iter()
            .flat_map(|frame| first.process_frame(frame))
            .map(boundary_shape)
            .collect::<Vec<_>>();
        let second_events = frames
            .iter()
            .flat_map(|frame| second.process_frame(frame))
            .map(boundary_shape)
            .collect::<Vec<_>>();
        assert_eq!(first_events, second_events);

        first.reset();
        let reset_events = frames
            .iter()
            .flat_map(|frame| first.process_frame(frame))
            .map(boundary_shape)
            .collect::<Vec<_>>();
        assert_eq!(first_events, reset_events);
    }

    #[test]
    fn disabled_provider_bypasses_state_machine() {
        let mut engine =
            StreamingVadEngine::new(default_streaming_vad_config(VadMode::Disabled, 20)).unwrap();
        assert!(!engine.is_enabled());
        assert_eq!(engine.mode(), VadMode::Disabled);
        engine.reset();
        engine.close();
    }
}
