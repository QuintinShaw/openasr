use super::super::*;
use super::fixtures::test_time;
use crate::{
    RealtimeAudioFrame, RealtimeEventEnvelope, RealtimeSessionState, RealtimeTranscriptWord,
    TranscriptUpdate,
};
use std::collections::VecDeque;

mod lifecycle;
mod steps;
mod transcript_flow;

use steps::{TestOnlyStreamingStep, initial_script};

pub(super) struct TestOnlyNativeStreamingSession {
    emitter: NativeStreamingTranscriptEmitter,
    queued_audio_frames: usize,
    processed_audio_frames: usize,
    utterance_index: u64,
    script: VecDeque<TestOnlyStreamingStep>,
    closed: bool,
}

enum CloseMode {
    Finish,
    Close,
    Cancel,
}

impl TestOnlyNativeStreamingSession {
    pub(super) fn new(
        context: NativeAsrSessionContext,
        model_id: String,
        options: NativeAsrRequestOptions,
        session_config: NativeAsrStreamingSessionConfig,
    ) -> Result<Self, NativeAsrError> {
        let emitter = NativeStreamingTranscriptEmitter::new_started(
            context,
            model_id,
            options,
            session_config.clone(),
            test_time(0),
            test_time(1),
            test_time(2),
        )?;

        Ok(Self {
            emitter,
            queued_audio_frames: 0,
            processed_audio_frames: 0,
            utterance_index: 1,
            script: initial_script(),
            closed: false,
        })
    }

    fn drain_pending_events(&mut self) -> Vec<RealtimeEventEnvelope> {
        self.emitter.drain_pending_events()
    }

    fn flush_pending_output(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }
        let mut events = self.drain_pending_events();
        let finalized = self.emitter.finalize_pending_output_at(test_time(89))?;
        if !finalized.is_empty() {
            self.utterance_index += 1;
        }
        events.extend(finalized);
        Ok(events)
    }

    fn ensure_push_capacity(&self) -> Result<(), NativeAsrError> {
        self.emitter.ensure_push_capacity(self.queued_audio_frames)
    }
}

impl NativeAsrSession for TestOnlyNativeStreamingSession {
    fn session_id(&self) -> &str {
        self.emitter.session_id()
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Err(NativeAsrError::SessionClosed);
        }
        if self.emitter.state() != RealtimeSessionState::Running {
            return Err(NativeAsrError::SessionFailed {
                message: "test-only fixture requires running audio input before push_audio"
                    .to_string(),
            });
        }
        self.ensure_push_capacity()?;

        self.queued_audio_frames += 1;
        let result = self.process_audio_frame(frame);
        self.queued_audio_frames = self.queued_audio_frames.saturating_sub(1);
        let output = result?;

        self.emitter.ensure_output_capacity(output.len())?;
        let mut events = self.drain_pending_events();
        events.extend(output);
        Ok(events)
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        Ok(self.drain_pending_events())
    }

    fn flush(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.flush_pending_output()
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.close_impl(CloseMode::Finish)
    }

    fn close(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.close_impl(CloseMode::Close)
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.close_impl(CloseMode::Cancel)
    }
}

fn scripted_words(text: &str, start_ms: u64, end_ms: u64) -> Vec<RealtimeTranscriptWord> {
    let words = text
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }
    let duration = end_ms.saturating_sub(start_ms);
    let count = words.len() as u64;
    words
        .into_iter()
        .enumerate()
        .map(|(index, word)| {
            let index = index as u64;
            let word_start = start_ms + duration.saturating_mul(index) / count;
            let word_end = start_ms + duration.saturating_mul(index + 1) / count;
            RealtimeTranscriptWord {
                word: word.to_string(),
                start_ms: word_start,
                end_ms: word_end.max(word_start),
                confidence: None,
            }
        })
        .collect()
}
