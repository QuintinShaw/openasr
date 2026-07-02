use super::{RealtimeAudioFrame, SpeechBoundaryEvent, TranscriptUtteranceId};

pub(super) fn frame_duration_ms(frame: &RealtimeAudioFrame) -> u64 {
    u64::from(
        frame
            .duration_ms()
            .expect("RealtimeAudioFrame is validated at construction"),
    )
}

pub(super) fn reached_no_speech_timeout(
    waiting_start_ms: Option<u64>,
    frame: &RealtimeAudioFrame,
    timeout_ms: u32,
) -> bool {
    waiting_start_ms
        .map(|start_ms| frame.end_ms().saturating_sub(start_ms) >= timeout_ms as u64)
        .unwrap_or(false)
}

pub(super) fn reached_max_utterance(
    frame: &RealtimeAudioFrame,
    start_ms: Option<u64>,
    max_utterance_ms: Option<u32>,
) -> bool {
    if let (Some(start_ms), Some(max_ms)) = (start_ms, max_utterance_ms) {
        return frame.end_ms().saturating_sub(start_ms) >= max_ms as u64;
    }
    false
}

pub(super) fn max_utterance_event(
    utterance_id: TranscriptUtteranceId,
    start_ms: u64,
    end_ms: u64,
) -> SpeechBoundaryEvent {
    SpeechBoundaryEvent::MaxUtterance {
        utterance_id,
        start_ms,
        end_ms,
    }
}
