use std::collections::VecDeque;

use thiserror::Error;

#[path = "../buffer_helpers.rs"]
mod buffer_helpers;

use super::audio::RealtimeAudioFrame;
use super::events::TranscriptUtteranceId;
use super::vad::SpeechBoundaryEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeBufferConfig {
    pub frame_duration_ms: u32,
    pub pre_roll_ms: u32,
    pub max_buffered_frames: usize,
    pub max_buffered_samples: usize,
}

impl Default for RealtimeBufferConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 20,
            pre_roll_ms: 200,
            max_buffered_frames: 1_510,
            max_buffered_samples: 483_200,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferedUtterance {
    pub utterance_id: TranscriptUtteranceId,
    pub start_ms: u64,
    pub end_ms: u64,
    pub frames: Vec<RealtimeAudioFrame>,
    pub reason: RealtimeUtteranceEndReason,
}

impl BufferedUtterance {
    pub fn sample_count(&self) -> usize {
        self.frames
            .iter()
            .map(RealtimeAudioFrame::sample_count)
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeUtteranceEndReason {
    VadStop,
    MaxUtterance,
    Flush,
    Cancel,
}

#[derive(Debug)]
pub struct RealtimeBuffer {
    config: RealtimeBufferConfig,
    pre_roll: VecDeque<RealtimeAudioFrame>,
    active: Option<ActiveUtterance>,
    last_overflow: Option<RealtimeBufferError>,
}

#[derive(Debug)]
pub(super) struct ActiveUtterance {
    utterance_id: TranscriptUtteranceId,
    start_ms: u64,
    frames: Vec<RealtimeAudioFrame>,
}

impl RealtimeBuffer {
    pub fn new(config: RealtimeBufferConfig) -> Result<Self, RealtimeBufferError> {
        if config.frame_duration_ms == 0 {
            return Err(RealtimeBufferError::InvalidConfig {
                message: "frame_duration_ms must be greater than 0".to_string(),
            });
        }
        if config.max_buffered_frames == 0 || config.max_buffered_samples == 0 {
            return Err(RealtimeBufferError::InvalidConfig {
                message: "buffer capacities must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            config,
            pre_roll: VecDeque::new(),
            active: None,
            last_overflow: None,
        })
    }

    pub fn push_frame(
        &mut self,
        frame: RealtimeAudioFrame,
        boundaries: &[SpeechBoundaryEvent],
    ) -> Result<Vec<BufferedUtterance>, RealtimeBufferError> {
        self.ingest_frame(frame, boundaries)?;

        let mut completed = Vec::new();
        self.collect_completed(boundaries, &mut completed);
        Ok(completed)
    }

    pub fn flush(&mut self, end_ms: u64) -> Option<BufferedUtterance> {
        let utterance_id = self.active.as_ref()?.utterance_id.clone();
        self.finish(&utterance_id, end_ms, RealtimeUtteranceEndReason::Flush)
    }

    pub fn cancel(&mut self, end_ms: u64) -> Option<BufferedUtterance> {
        let utterance_id = self.active.as_ref()?.utterance_id.clone();
        self.finish(&utterance_id, end_ms, RealtimeUtteranceEndReason::Cancel)
    }

    pub fn reset(&mut self) {
        self.pre_roll.clear();
        self.active = None;
        self.last_overflow = None;
    }

    pub fn last_overflow(&self) -> Option<&RealtimeBufferError> {
        self.last_overflow.as_ref()
    }

    pub fn active_snapshot(&self, max_duration_ms: Option<u32>) -> Option<BufferedUtterance> {
        let active = self.active.as_ref()?;
        let mut frames = active.frames.clone();
        if let Some(limit_ms) = max_duration_ms
            && limit_ms > 0
        {
            frames = buffer_helpers::limited_active_frames(active, Some(limit_ms))?;
        }
        let (start_ms, end_ms) = buffer_helpers::frame_time_range(&frames, active.start_ms);
        Some(BufferedUtterance {
            utterance_id: active.utterance_id.clone(),
            start_ms,
            end_ms,
            frames,
            reason: RealtimeUtteranceEndReason::Flush,
        })
    }

    fn push_pre_roll(&mut self, frame: RealtimeAudioFrame) -> Result<(), RealtimeBufferError> {
        let mut candidate = self.pre_roll.clone();
        candidate.push_back(frame);
        let target_ms = u64::from(self.config.pre_roll_ms);
        if target_ms == 0 {
            self.pre_roll.clear();
            return Ok(());
        }

        candidate = buffer_helpers::trim_pre_roll(candidate, self.config.pre_roll_ms);
        let frames = candidate.iter().cloned().collect::<Vec<_>>();
        self.ensure_capacity(&frames)?;
        self.pre_roll = candidate;
        Ok(())
    }

    fn push_active(&mut self, frame: RealtimeAudioFrame) -> Result<(), RealtimeBufferError> {
        let mut frames = self
            .active
            .as_ref()
            .map(|active| active.frames.clone())
            .unwrap_or_default();
        frames.push(frame.clone());
        self.ensure_capacity(&frames)?;
        if let Some(active) = self.active.as_mut() {
            active.frames.push(frame);
        }
        Ok(())
    }

    fn ensure_capacity(
        &mut self,
        frames: &[RealtimeAudioFrame],
    ) -> Result<(), RealtimeBufferError> {
        if let Err(error) = buffer_helpers::ensure_capacity(self.config, frames) {
            self.last_overflow = Some(error.clone());
            return Err(error);
        }

        Ok(())
    }

    fn finish(
        &mut self,
        expected_utterance_id: &TranscriptUtteranceId,
        end_ms: u64,
        reason: RealtimeUtteranceEndReason,
    ) -> Option<BufferedUtterance> {
        let active = self.active.take()?;
        match buffer_helpers::finish_utterance(active, expected_utterance_id, end_ms, reason) {
            Ok(utterance) => Some(utterance),
            Err(active) => {
                self.active = Some(active);
                None
            }
        }
    }

    fn finish_from_boundary(
        &mut self,
        boundary: &SpeechBoundaryEvent,
        completed: &mut Vec<BufferedUtterance>,
    ) {
        let Some((utterance_id, end_ms, reason)) = Self::boundary_completion(boundary) else {
            return;
        };
        if let Some(utterance) = self.finish(utterance_id, end_ms, reason) {
            completed.push(utterance);
        }
    }

    fn ingest_frame(
        &mut self,
        frame: RealtimeAudioFrame,
        boundaries: &[SpeechBoundaryEvent],
    ) -> Result<(), RealtimeBufferError> {
        if self.active.is_some() {
            return self.push_active(frame);
        }

        if let Some((utterance_id, start_ms)) = Self::boundary_start(boundaries) {
            return self.start_active_from_pre_roll(frame, utterance_id, start_ms);
        }

        self.push_pre_roll(frame)
    }

    fn start_active_from_pre_roll(
        &mut self,
        frame: RealtimeAudioFrame,
        utterance_id: &TranscriptUtteranceId,
        start_ms: u64,
    ) -> Result<(), RealtimeBufferError> {
        let mut frames = self.pre_roll.iter().cloned().collect::<Vec<_>>();
        frames.push(frame);
        self.ensure_capacity(&frames)?;
        self.pre_roll.clear();
        self.active = Some(ActiveUtterance {
            utterance_id: utterance_id.clone(),
            start_ms,
            frames,
        });
        Ok(())
    }

    fn collect_completed(
        &mut self,
        boundaries: &[SpeechBoundaryEvent],
        completed: &mut Vec<BufferedUtterance>,
    ) {
        for boundary in boundaries {
            self.finish_from_boundary(boundary, completed);
        }
    }

    fn boundary_start(boundaries: &[SpeechBoundaryEvent]) -> Option<(&TranscriptUtteranceId, u64)> {
        boundaries.iter().find_map(|event| match event {
            SpeechBoundaryEvent::SpeechStarted {
                utterance_id,
                start_ms,
            } => Some((utterance_id, *start_ms)),
            _ => None,
        })
    }

    fn boundary_completion(
        boundary: &SpeechBoundaryEvent,
    ) -> Option<(&TranscriptUtteranceId, u64, RealtimeUtteranceEndReason)> {
        match boundary {
            SpeechBoundaryEvent::SpeechStopped {
                utterance_id,
                end_ms,
                ..
            } => Some((utterance_id, *end_ms, RealtimeUtteranceEndReason::VadStop)),
            SpeechBoundaryEvent::MaxUtterance {
                utterance_id,
                end_ms,
                ..
            } => Some((
                utterance_id,
                *end_ms,
                RealtimeUtteranceEndReason::MaxUtterance,
            )),
            SpeechBoundaryEvent::SpeechStarted { .. }
            | SpeechBoundaryEvent::NoSpeechTimeout { .. } => None,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RealtimeBufferError {
    #[error("Invalid realtime buffer config: {message}.")]
    InvalidConfig { message: String },
    #[error(
        "The realtime audio buffer reached capacity ({buffered_frames}/{max_buffered_frames} frames, {buffered_samples}/{max_buffered_samples} samples)."
    )]
    AudioBufferOverflow {
        buffered_frames: usize,
        max_buffered_frames: usize,
        buffered_samples: usize,
        max_buffered_samples: usize,
    },
}

