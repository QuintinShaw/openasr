use super::*;

impl TestOnlyNativeStreamingSession {
    pub(super) fn scripted_update(
        &self,
        frame: &RealtimeAudioFrame,
        revision: u64,
        text: &'static str,
    ) -> TranscriptUpdate {
        let update = TranscriptUpdate::new(
            format!("utt_test_only_native_streaming_{}", self.utterance_index),
            "seg_test_only_native_streaming",
            revision,
            text,
            frame.start_ms,
            frame.end_ms(),
        );
        if self.emitter.word_timestamps_enabled() {
            update.with_words(scripted_words(text, frame.start_ms, frame.end_ms()))
        } else {
            update
        }
    }

    pub(super) fn process_audio_frame(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        self.processed_audio_frames += 1;
        let Some(step) = self.script.pop_front() else {
            return Ok(Vec::new());
        };

        self.apply_script_step(
            step,
            &frame,
            test_time(10 + self.processed_audio_frames as u64),
        )
    }

    fn apply_script_step(
        &mut self,
        step: TestOnlyStreamingStep,
        frame: &RealtimeAudioFrame,
        created_at: String,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        match step {
            TestOnlyStreamingStep::Partial { revision, text } => {
                let update = self.scripted_update(frame, revision, text);
                self.emitter.apply_partial(update, created_at)
            }
            TestOnlyStreamingStep::Final { revision, text } => self
                .emitter
                .apply_final(self.scripted_update(frame, revision, text), created_at),
            TestOnlyStreamingStep::PostFinalSameText { revision, text }
            | TestOnlyStreamingStep::PostFinalRevision { revision, text } => self
                .emitter
                .apply_partial(self.scripted_update(frame, revision, text), created_at),
        }
    }
}
