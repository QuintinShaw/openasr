#![cfg_attr(not(test), allow(dead_code))]

use thiserror::Error;

use crate::{GgmlAsrPreparedAudio, RealtimeAudioFrame};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FrameTimelineError {
    #[error("realtime audio frame sequence jumped: expected seq={expected}, got seq={got}")]
    NonContiguousSequence { expected: u64, got: u64 },
    #[error(
        "realtime audio frame start time jumped: expected start_ms={expected}, got start_ms={got}"
    )]
    NonContiguousStartMs { expected: u64, got: u64 },
}

/// Frame-contiguity validation + start/end timestamps for a realtime stream.
/// Shared by the audio-retaining buffer below and by drivers that track the
/// timeline without keeping samples (frame-sync streaming).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct FrameTimeline {
    first_start_ms: Option<u64>,
    next_seq: Option<u64>,
    next_start_ms: Option<u64>,
}

impl FrameTimeline {
    pub(crate) fn observe(&mut self, frame: &RealtimeAudioFrame) -> Result<(), FrameTimelineError> {
        if let Some(expected) = self.next_seq
            && frame.seq != expected
        {
            return Err(FrameTimelineError::NonContiguousSequence {
                expected,
                got: frame.seq,
            });
        }
        if let Some(expected) = self.next_start_ms
            && frame.start_ms != expected
        {
            return Err(FrameTimelineError::NonContiguousStartMs {
                expected,
                got: frame.start_ms,
            });
        }
        self.first_start_ms.get_or_insert(frame.start_ms);
        self.next_seq = Some(frame.seq.saturating_add(1));
        self.next_start_ms = Some(frame.end_ms());
        Ok(())
    }

    pub(crate) fn first_start_ms(&self) -> Option<u64> {
        self.first_start_ms
    }

    pub(crate) fn next_start_ms(&self) -> Option<u64> {
        self.next_start_ms
    }

    pub(crate) fn set_first_start_ms(&mut self, value: Option<u64>) {
        self.first_start_ms = value;
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct GgmlStreamingAudioBuffer {
    samples_i16: Vec<i16>,
    timeline: FrameTimeline,
}

impl GgmlStreamingAudioBuffer {
    pub(crate) fn push_frame(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<(), FrameTimelineError> {
        self.timeline.observe(&frame)?;
        self.samples_i16.extend(frame.into_samples());
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples_i16.is_empty()
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.samples_i16.len()
    }

    pub(crate) fn start_ms(&self) -> Option<u64> {
        self.timeline.first_start_ms()
    }

    pub(crate) fn end_ms(&self) -> Option<u64> {
        self.timeline.next_start_ms()
    }

    pub(crate) fn duration_ms(&self) -> u64 {
        match (
            self.timeline.first_start_ms(),
            self.timeline.next_start_ms(),
        ) {
            (Some(start), Some(end)) => end.saturating_sub(start),
            _ => 0,
        }
    }

    pub(crate) fn prepared_audio_snapshot(&self) -> GgmlAsrPreparedAudio {
        GgmlAsrPreparedAudio::mono_16khz(
            self.samples_i16
                .iter()
                .map(|sample| f32::from(*sample) / 32768.0)
                .collect(),
        )
    }

    pub(crate) fn clear(&mut self) {
        self.samples_i16.clear();
        self.timeline = FrameTimeline::default();
    }

    /// Drop all samples before absolute time `ms` and re-anchor the buffer start
    /// there, keeping `next_seq`/`next_start_ms` so subsequent frames stay
    /// contiguous. Used to re-anchor after a sentence segment is finalized so the
    /// PARTIAL decode only ever spans the current (uncommitted) sentence —
    /// Whisper-Streaming buffer trimming. Sample timestamps are absolute, so the
    /// retained audio keeps its original timeline.
    pub(crate) fn drain_before(&mut self, ms: u64) {
        let Some(start) = self.timeline.first_start_ms() else {
            return;
        };
        if ms <= start {
            return;
        }
        let drop_samples = ((ms - start).saturating_mul(SAMPLES_PER_MS_16KHZ) as usize)
            .min(self.samples_i16.len());
        if drop_samples == 0 {
            return;
        }
        self.samples_i16.drain(0..drop_samples);
        if self.samples_i16.is_empty() {
            self.timeline.set_first_start_ms(None);
        } else {
            self.timeline.set_first_start_ms(Some(
                start.saturating_add(drop_samples as u64 / SAMPLES_PER_MS_16KHZ),
            ));
        }
    }

    /// Prepared audio for the trailing `window_ms` of the buffer (or the whole
    /// buffer if shorter). Used by windowed streaming to bound a PARTIAL decode to
    /// O(window) instead of O(buffer); the FINAL still uses
    /// [`prepared_audio_snapshot`] over the whole buffer.
    pub(crate) fn prepared_audio_window(&self, window_ms: u64) -> GgmlAsrPreparedAudio {
        let window_samples = window_ms.saturating_mul(SAMPLES_PER_MS_16KHZ) as usize;
        let start = self.samples_i16.len().saturating_sub(window_samples);
        GgmlAsrPreparedAudio::mono_16khz(
            self.samples_i16[start..]
                .iter()
                .map(|sample| f32::from(*sample) / 32768.0)
                .collect(),
        )
    }
}

/// 16 samples/ms at the buffer's fixed 16 kHz mono rate.
const SAMPLES_PER_MS_16KHZ: u64 = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RealtimeAudioFormat, RealtimeAudioFrame};

    fn frame(seq: u64, start_ms: u64, samples: Vec<i16>) -> RealtimeAudioFrame {
        RealtimeAudioFrame::new(
            seq,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            samples,
        )
        .unwrap()
    }

    #[test]
    fn accumulates_contiguous_frames_into_prepared_audio_snapshot() {
        let mut buffer = GgmlStreamingAudioBuffer::default();

        buffer.push_frame(frame(7, 40, vec![0; 320])).unwrap();
        let mut second = vec![0; 320];
        second[0] = i16::MAX;
        second[1] = i16::MIN;
        buffer.push_frame(frame(8, 60, second)).unwrap();

        let audio = buffer.prepared_audio_snapshot();
        assert_eq!(buffer.sample_count(), 640);
        assert_eq!(buffer.start_ms(), Some(40));
        assert_eq!(buffer.end_ms(), Some(80));
        assert_eq!(buffer.duration_ms(), 40);
        assert_eq!(audio.sample_rate_hz, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples_f32.len(), 640);
        assert_eq!(audio.samples_f32[320], i16::MAX as f32 / 32768.0);
        assert_eq!(audio.samples_f32[321], -1.0);
    }

    #[test]
    fn rejects_non_contiguous_sequence() {
        let mut buffer = GgmlStreamingAudioBuffer::default();
        buffer.push_frame(frame(1, 0, vec![0; 320])).unwrap();

        let error = buffer.push_frame(frame(3, 20, vec![0; 320])).unwrap_err();

        assert_eq!(
            error,
            FrameTimelineError::NonContiguousSequence {
                expected: 2,
                got: 3,
            }
        );
    }

    #[test]
    fn rejects_non_contiguous_start_time() {
        let mut buffer = GgmlStreamingAudioBuffer::default();
        buffer.push_frame(frame(1, 0, vec![0; 320])).unwrap();

        let error = buffer.push_frame(frame(2, 40, vec![0; 320])).unwrap_err();

        assert_eq!(
            error,
            FrameTimelineError::NonContiguousStartMs {
                expected: 20,
                got: 40,
            }
        );
    }

    #[test]
    fn clear_resets_sequence_and_audio() {
        let mut buffer = GgmlStreamingAudioBuffer::default();
        buffer.push_frame(frame(1, 0, vec![0; 320])).unwrap();

        buffer.clear();
        buffer.push_frame(frame(9, 100, vec![1; 160])).unwrap();

        assert_eq!(buffer.sample_count(), 160);
        assert_eq!(buffer.start_ms(), Some(100));
        assert_eq!(buffer.end_ms(), Some(110));
        assert_eq!(buffer.duration_ms(), 10);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn prepared_audio_window_returns_the_trailing_window() {
        let mut buffer = GgmlStreamingAudioBuffer::default();
        buffer.push_frame(frame(1, 0, vec![0; 320])).unwrap(); // 0..20 ms
        let mut second = vec![0; 320];
        second[0] = i16::MAX;
        buffer.push_frame(frame(2, 20, second)).unwrap(); // 20..40 ms

        // Last 20 ms = the second frame's 320 samples; its first sample is i16::MAX.
        let window = buffer.prepared_audio_window(20);
        assert_eq!(window.samples_f32.len(), 320);
        assert_eq!(window.samples_f32[0], i16::MAX as f32 / 32768.0);
        // A window wider than the buffer clamps to the whole buffer.
        assert_eq!(buffer.prepared_audio_window(1_000).samples_f32.len(), 640);
    }
}
