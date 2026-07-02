use super::*;

impl VadStateMachine {
    pub(super) fn process_state(
        &mut self,
        frame: &RealtimeAudioFrame,
        is_speech: bool,
        frame_duration_ms: u64,
        events: &mut Vec<SpeechBoundaryEvent>,
    ) {
        match self.state {
            VadState::WaitingForSpeech => {
                self.process_waiting_for_speech(frame, is_speech, frame_duration_ms, events)
            }
            VadState::InSpeech => {
                self.process_in_speech(frame, is_speech, frame_duration_ms, events)
            }
            VadState::Closed => {}
        }
    }

    fn process_waiting_for_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
        is_speech: bool,
        frame_duration_ms: u64,
        events: &mut Vec<SpeechBoundaryEvent>,
    ) {
        self.waiting_start_ms.get_or_insert(frame.start_ms);
        if is_speech {
            self.consume_waiting_speech(frame, frame_duration_ms, events);
            return;
        }

        self.consecutive_speech_ms = 0;
        self.speech_candidate_start_ms = None;
        self.maybe_emit_no_speech_timeout(frame, events);
    }

    fn consume_waiting_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
        frame_duration_ms: u64,
        events: &mut Vec<SpeechBoundaryEvent>,
    ) {
        if self.consecutive_speech_ms == 0 {
            self.speech_candidate_start_ms = Some(frame.start_ms);
        }
        self.consecutive_speech_ms += frame_duration_ms;
        self.consecutive_silence_ms = 0;

        if self.consecutive_speech_ms < u64::from(self.config.speech_start_ms) {
            return;
        }

        let utterance_id = self.next_utterance_id();
        let start_ms = self.speech_candidate_start_ms.unwrap_or(frame.start_ms);
        self.state = VadState::InSpeech;
        self.active_utterance_id = Some(utterance_id.clone());
        self.active_start_ms = Some(start_ms);
        self.speech_started_once = true;
        self.no_speech_timeout_emitted = false;
        events.push(SpeechBoundaryEvent::SpeechStarted {
            utterance_id,
            start_ms,
        });
    }

    fn process_in_speech(
        &mut self,
        frame: &RealtimeAudioFrame,
        is_speech: bool,
        frame_duration_ms: u64,
        events: &mut Vec<SpeechBoundaryEvent>,
    ) {
        if is_speech {
            self.consecutive_silence_ms = 0;
        } else {
            self.consecutive_silence_ms += frame_duration_ms;
            if self.consecutive_silence_ms >= u64::from(self.config.speech_stop_ms) {
                self.push_stop_and_reset(frame.end_ms(), events);
                return;
            }
        }

        if let Some(event) = self.max_utterance_event(frame) {
            events.push(event);
            self.clear_state_after_boundary();
        }
    }

    fn push_stop_and_reset(&mut self, end_ms: u64, events: &mut Vec<SpeechBoundaryEvent>) {
        events.push(self.stop_event(end_ms));
        self.clear_state_after_boundary();
    }

    fn max_utterance_event(&self, frame: &RealtimeAudioFrame) -> Option<SpeechBoundaryEvent> {
        let utterance_id = self.active_utterance_id.clone()?;
        if !timing::reached_max_utterance(frame, self.active_start_ms, self.config.max_utterance_ms) {
            return None;
        }
        let start_ms = self.active_start_ms.unwrap_or(frame.start_ms);
        Some(timing::max_utterance_event(
            utterance_id,
            start_ms,
            frame.end_ms(),
        ))
    }
}
