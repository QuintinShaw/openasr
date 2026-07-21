use crate::models::ggml_asr_executor::GgmlAsrExecutionError;
use crate::models::ggml_streaming_audio::{FrameTimeline, FrameTimelineError};
use crate::models::ggml_streaming_session::{
    GgmlAsrStreamingTranscriptDriver, GgmlAsrStreamingTranscriptUpdate,
};
use crate::{RealtimeAudioFrame, TranscriptUpdate};

/// Frame-synchronous incremental decoder contract, reusable by any family
/// whose decode is monotonic over a running audio stream (RNN-T greedy,
/// streaming-trained CTC).
///
/// Contract:
/// - Deltas are APPEND-ONLY: every returned string extends the utterance text;
///   the decoder must never revise text it already returned. (Families that
///   revise — window re-transcribers — belong on
///   `IncrementalStreamingTranscriptDriver` instead.)
/// - Decode must be prefix-stable: feeding the same audio in different push
///   granularities yields the same concatenated text, and `finish` returns
///   exactly the missing tail (the streaming==batch parity gate).
/// - `reset` drops all utterance state; the next `accept_samples` starts a
///   fresh utterance.
pub(crate) trait IncrementalAudioDecoder: Send {
    fn accept_samples(&mut self, samples: &[f32]) -> Result<String, GgmlAsrExecutionError>;
    fn finish(&mut self) -> Result<String, GgmlAsrExecutionError>;
    fn reset(&mut self);

    fn rebase_after_soft_split(&mut self) -> Result<(), GgmlAsrExecutionError> {
        Ok(())
    }

    /// Runs whatever real decode work would otherwise land on the first
    /// production chunk (paying its lazy graph/runner init up front), then
    /// restores the decoder to the exact state [`Self::reset`] produces --
    /// silence fed here must never bleed into the next real utterance. The
    /// default no-op keeps decoders that have nothing to warm (stateless,
    /// re-transcribe-per-window families) trivially correct.
    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        Ok(())
    }
}

pub(crate) struct FrameSyncStreamingTranscriptDriver<D>
where
    D: IncrementalAudioDecoder,
{
    executor_id: &'static str,
    adapter_id: &'static str,
    utterance_id: String,
    segment_id: String,
    utterance_id_prefix: String,
    segment_id_prefix: String,
    utterance_index: u64,
    segment_index: u64,
    decoder: D,
    timeline: FrameTimeline,
    accumulated_text: String,
    last_text: Option<String>,
    next_revision: u64,
    final_emitted: bool,
    /// Start timestamp of the current segment after a soft split; the audio
    /// buffer keeps running across splits, so its start_ms only covers the
    /// first segment.
    segment_start_ms: Option<u64>,
}

impl<D> FrameSyncStreamingTranscriptDriver<D>
where
    D: IncrementalAudioDecoder,
{
    pub(crate) fn new(
        executor_id: &'static str,
        adapter_id: &'static str,
        utterance_id: impl Into<String>,
        segment_id: impl Into<String>,
        first_revision: u64,
        decoder: D,
    ) -> Self {
        let mut driver = Self {
            executor_id,
            adapter_id,
            utterance_id: utterance_id.into(),
            segment_id: segment_id.into(),
            utterance_id_prefix: String::new(),
            segment_id_prefix: String::new(),
            utterance_index: 1,
            segment_index: 1,
            decoder,
            timeline: FrameTimeline::default(),
            accumulated_text: String::new(),
            last_text: None,
            next_revision: first_revision,
            final_emitted: false,
            segment_start_ms: None,
        };
        driver.utterance_id_prefix = driver.utterance_id.clone();
        driver.segment_id_prefix = driver.segment_id.clone();
        driver
    }

    fn driver_failed(&self, reason: impl Into<String>) -> GgmlAsrExecutionError {
        GgmlAsrExecutionError::executor_failed(self.executor_id, self.adapter_id, reason)
    }

    fn map_timeline_error(&self, error: FrameTimelineError) -> GgmlAsrExecutionError {
        self.driver_failed(error.to_string())
    }

    fn append_delta(&mut self, delta: String) {
        if !delta.is_empty() {
            self.accumulated_text.push_str(&delta);
        }
    }

    fn emit_accumulated(&mut self, final_: bool) -> Option<GgmlAsrStreamingTranscriptUpdate> {
        if !final_
            && self
                .last_text
                .as_ref()
                .is_some_and(|text| text == &self.accumulated_text)
        {
            return None;
        }
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        self.last_text = Some(self.accumulated_text.clone());
        let update = TranscriptUpdate::new(
            self.utterance_id.clone(),
            self.segment_id.clone(),
            revision,
            self.accumulated_text.clone(),
            self.segment_start_ms
                .or(self.timeline.first_start_ms())
                .unwrap_or(0),
            self.timeline.next_start_ms().unwrap_or(0),
        );
        Some(if final_ {
            GgmlAsrStreamingTranscriptUpdate::final_(update)
        } else {
            GgmlAsrStreamingTranscriptUpdate::partial(update)
        })
    }

    fn reset_current_utterance(&mut self) {
        self.decoder.reset();
        self.timeline = FrameTimeline::default();
        self.accumulated_text.clear();
        self.last_text = None;
        self.final_emitted = false;
        self.segment_start_ms = None;
        self.utterance_index = self.utterance_index.saturating_add(1);
        self.utterance_id = format!("{}_{:06}", self.utterance_id_prefix, self.utterance_index);
        self.segment_index = self.segment_index.saturating_add(1);
        self.segment_id = format!("{}_{:06}", self.segment_id_prefix, self.segment_index);
    }

    /// Advances utterance/segment identity for the text that follows a soft
    /// split. The decoder and audio buffer keep running — only the transcript
    /// accumulation restarts.
    fn advance_segment_identity(&mut self) {
        self.accumulated_text.clear();
        self.last_text = None;
        self.segment_start_ms = self.timeline.next_start_ms();
        self.utterance_index = self.utterance_index.saturating_add(1);
        self.utterance_id = format!("{}_{:06}", self.utterance_id_prefix, self.utterance_index);
        self.segment_index = self.segment_index.saturating_add(1);
        self.segment_id = format!("{}_{:06}", self.segment_id_prefix, self.segment_index);
    }
}

impl<D> GgmlAsrStreamingTranscriptDriver for FrameSyncStreamingTranscriptDriver<D>
where
    D: IncrementalAudioDecoder,
{
    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if self.final_emitted {
            return Ok(Vec::new());
        }
        self.timeline
            .observe(&frame)
            .map_err(|error| self.map_timeline_error(error))?;
        let samples = frame
            .samples()
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect::<Vec<_>>();
        let delta = self.decoder.accept_samples(&samples)?;
        if delta.is_empty() {
            return Ok(Vec::new());
        }
        self.append_delta(delta);
        Ok(self.emit_accumulated(false).into_iter().collect())
    }

    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        // Delegates straight to the decoder: warm-up never touches this
        // driver's own transcript state (accumulated_text/timeline/revision
        // counters), only `push_audio`/`finish_updates`/`reset_utterance` do.
        // The decoder contract requires it leave itself exactly as `reset`
        // would, so nothing here needs cleanup either.
        self.decoder.warm_up()
    }

    fn reset_utterance(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.reset_current_utterance();
        Ok(())
    }

    fn finish_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if self.final_emitted {
            return Ok(Vec::new());
        }
        let delta = self.decoder.finish()?;
        self.append_delta(delta);
        self.final_emitted = true;
        Ok(self.emit_accumulated(true).into_iter().collect())
    }

    fn supports_soft_split(&self) -> bool {
        true
    }

    fn split_updates(
        &mut self,
    ) -> Result<Vec<GgmlAsrStreamingTranscriptUpdate>, GgmlAsrExecutionError> {
        if self.final_emitted {
            return Ok(Vec::new());
        }
        // No decoder flush here: a soft split finalizes only the text already
        // decoded. Audio still inside the decoder's chunk window keeps its
        // full left context and lands in the next segment — an arbitrary
        // mid-speech cut never degrades recognition.
        if self.accumulated_text.is_empty() {
            self.decoder.rebase_after_soft_split()?;
            return Ok(Vec::new());
        }
        let updates = self.emit_accumulated(true).into_iter().collect();
        self.decoder.rebase_after_soft_split()?;
        self.advance_segment_identity();
        Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::{RealtimeAudioFormat, RealtimeAudioFrame};

    #[derive(Default)]
    struct ScriptDecoder {
        accept_deltas: VecDeque<&'static str>,
        finish_delta: &'static str,
        reset_count: usize,
        rebase_count: usize,
        warm_up_count: usize,
    }

    impl ScriptDecoder {
        fn new(
            accept_deltas: impl IntoIterator<Item = &'static str>,
            finish_delta: &'static str,
        ) -> Self {
            Self {
                accept_deltas: accept_deltas.into_iter().collect(),
                finish_delta,
                reset_count: 0,
                rebase_count: 0,
                warm_up_count: 0,
            }
        }
    }

    impl IncrementalAudioDecoder for ScriptDecoder {
        fn accept_samples(&mut self, samples: &[f32]) -> Result<String, GgmlAsrExecutionError> {
            assert!(!samples.is_empty());
            Ok(self.accept_deltas.pop_front().unwrap_or("").to_string())
        }

        fn finish(&mut self) -> Result<String, GgmlAsrExecutionError> {
            Ok(self.finish_delta.to_string())
        }

        fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
            self.warm_up_count += 1;
            Ok(())
        }

        fn reset(&mut self) {
            self.reset_count += 1;
        }

        fn rebase_after_soft_split(&mut self) -> Result<(), GgmlAsrExecutionError> {
            self.rebase_count += 1;
            Ok(())
        }
    }

    fn frame(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        let sample_count = format.sample_count_for_duration_ms(20).unwrap();
        RealtimeAudioFrame::new(seq, start_ms, format, vec![1; sample_count]).unwrap()
    }

    fn update_text(update: &GgmlAsrStreamingTranscriptUpdate) -> &TranscriptUpdate {
        match update {
            GgmlAsrStreamingTranscriptUpdate::Partial(update)
            | GgmlAsrStreamingTranscriptUpdate::Final(update) => update,
        }
    }

    #[test]
    fn emits_accumulated_partials_and_final_with_monotonic_revisions() {
        let decoder = ScriptDecoder::new(["he", "", "llo"], "!");
        let mut driver = FrameSyncStreamingTranscriptDriver::new(
            "script-frame-sync",
            "script-adapter",
            "utt_script",
            "seg_script",
            1,
            decoder,
        );

        let first = driver.push_audio(frame(1, 0)).unwrap();
        assert_eq!(first.len(), 1);
        let first_update = update_text(&first[0]);
        assert_eq!(first_update.text, "he");
        assert_eq!(first_update.revision, 1);
        assert_eq!(first_update.start_ms, 0);
        assert_eq!(first_update.end_ms, 20);

        assert!(driver.push_audio(frame(2, 20)).unwrap().is_empty());

        let second = driver.push_audio(frame(3, 40)).unwrap();
        let second_update = update_text(&second[0]);
        assert_eq!(second_update.text, "hello");
        assert_eq!(second_update.revision, 2);

        let final_ = driver.finish_updates().unwrap();
        let final_update = update_text(&final_[0]);
        assert_eq!(final_update.text, "hello!");
        assert_eq!(final_update.revision, 3);
    }

    #[test]
    fn soft_split_finalizes_segment_without_resetting_the_decoder() {
        let decoder = ScriptDecoder::new(["he", "llo", "wo", "rld"], "!");
        let mut driver = FrameSyncStreamingTranscriptDriver::new(
            "script-frame-sync",
            "script-adapter",
            "utt_script",
            "seg_script",
            1,
            decoder,
        );

        driver.push_audio(frame(1, 0)).unwrap();
        driver.push_audio(frame(2, 20)).unwrap();

        assert!(driver.supports_soft_split());
        let split = driver.split_updates().unwrap();
        assert_eq!(split.len(), 1);
        let split_update = update_text(&split[0]);
        assert!(matches!(
            split[0],
            GgmlAsrStreamingTranscriptUpdate::Final(_)
        ));
        assert_eq!(split_update.text, "hello");
        // The decoder was NOT reset: context survives the forced boundary.
        assert_eq!(driver.decoder.reset_count, 0);
        assert_eq!(driver.decoder.rebase_count, 1);

        // A split with nothing accumulated is a no-op (no empty segments).
        let empty = driver.split_updates().unwrap();
        assert!(empty.is_empty());
        assert_eq!(driver.decoder.rebase_count, 2);

        // Subsequent audio continues decoding into a fresh segment identity.
        let next = driver.push_audio(frame(3, 40)).unwrap();
        let next_update = update_text(&next[0]);
        assert_eq!(next_update.text, "wo");
        assert_eq!(next_update.utterance_id.0, "utt_script_000002");
        assert_eq!(next_update.segment_id.0, "seg_script_000002");
        assert_eq!(next_update.start_ms, 40);
        assert!(next_update.revision > split_update.revision);
    }

    #[test]
    fn reset_utterance_advances_ids_without_resetting_revision() {
        let decoder = ScriptDecoder::new(["a", "b"], "");
        let mut driver = FrameSyncStreamingTranscriptDriver::new(
            "script-frame-sync",
            "script-adapter",
            "utt_script",
            "seg_script",
            10,
            decoder,
        );

        let first = driver.push_audio(frame(7, 100)).unwrap();
        assert_eq!(update_text(&first[0]).revision, 10);

        driver.reset_utterance().unwrap();
        let second = driver.push_audio(frame(1, 0)).unwrap();
        let second_update = update_text(&second[0]);
        assert_eq!(second_update.text, "b");
        assert_eq!(second_update.revision, 11);
        assert_eq!(second_update.utterance_id.0, "utt_script_000002");
        assert_eq!(second_update.segment_id.0, "seg_script_000002");
    }

    #[test]
    fn warm_up_delegates_to_the_decoder_without_touching_driver_state() {
        let decoder = ScriptDecoder::new(["hel", "lo"], "!");
        let mut driver = FrameSyncStreamingTranscriptDriver::new(
            "script-frame-sync",
            "script-adapter",
            "utt_script",
            "seg_script",
            1,
            decoder,
        );

        driver.warm_up().unwrap();
        driver.warm_up().unwrap();
        assert_eq!(driver.decoder.warm_up_count, 2);
        // Warm-up must not have advanced revision/utterance/segment identity
        // or accumulated any text -- the first real push must behave exactly
        // as if warm_up had never been called.
        assert_eq!(driver.next_revision, 1);
        assert_eq!(driver.utterance_id, "utt_script");
        assert_eq!(driver.segment_id, "seg_script");
        assert!(driver.accumulated_text.is_empty());

        let first = driver.push_audio(frame(1, 0)).unwrap();
        let first_update = update_text(&first[0]);
        assert_eq!(first_update.text, "hel");
        assert_eq!(first_update.revision, 1);
        assert_eq!(first_update.utterance_id.0, "utt_script");
    }
}
