//! Recording-local speaker activity segmentation.
//!
//! The external diarizer depends on the small [`LocalActivitySegmenter`] seam,
//! not on pyannote or any future model family. The seam deliberately returns
//! unaligned window-local activity and an aggregated speaker count: global
//! speaker identity is reconstructed later from ReDim embeddings.

mod ops;
mod pack;
mod pyannet;

#[cfg(test)]
mod tests;

pub use pack::{SEGMENTER_PACK_ID, segmenter_pack_installed, shared_segmenter};
pub(crate) use pack::{SelectedSegmenter, resolve_segmenter};

use pyannet::{NUM_CLASSES, PyannetModel};
use thiserror::Error;

use super::contract::{SpeakerId, SpeakerTurn, TimeRange};
use super::embed::weights::WeightsError;
use crate::config::VoiceIdSegmenterPreference;

const SAMPLE_RATE_HZ: u32 = 16_000;
/// PyanNet's first output frame sees 911 samples; consecutive frames advance
/// by the SincNet stack's total stride of 270 samples.
const FRAME_DURATION_SAMPLES: f64 = 911.0;
const FRAME_STEP_SAMPLES: f64 = 270.0;
const MAX_LOCAL_SPEAKERS: usize = 3;
const DEFAULT_WINDOW_S: f64 = 10.0;
const DEFAULT_STEP_S: f64 = 1.0;

const POWERSET: [&[usize]; NUM_CLASSES] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("local speaker segmenter requires 16 kHz mono audio, got {0} Hz")]
    UnsupportedSampleRate(u32),
    #[error("local speaker segmenter pack is missing for preference {preference:?}")]
    MissingPack {
        preference: VoiceIdSegmenterPreference,
    },
    #[error("local speaker segmenter pack could not be loaded: {0}")]
    LoadFailed(String),
    #[error("local speaker segmentation failed: {0}")]
    Inference(String),
    #[error("local speaker segmentation was canceled")]
    Canceled,
}

/// Global clock used by aggregated local-speaker counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ActivityFrameClock {
    pub start_s: f64,
    pub duration_s: f64,
    pub step_s: f64,
}

impl ActivityFrameClock {
    pub(crate) fn midpoint_s(self, frame: usize) -> f64 {
        self.start_s + frame as f64 * self.step_s + self.duration_s * 0.5
    }

    pub(crate) fn closest_frame(self, timestamp_s: f64) -> usize {
        ((timestamp_s - self.start_s - self.duration_s * 0.5) / self.step_s)
            .round_ties_even()
            .max(0.0) as usize
    }
}

/// One fixed-duration inference window. Each frame is a bitset of active
/// window-local speaker slots; bit `n` means local slot `n` is active.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalActivityWindow {
    pub start_s: f64,
    pub frame_activity: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalActivity {
    pub frame_clock: ActivityFrameClock,
    pub windows: Vec<LocalActivityWindow>,
    /// Overlap-add mean of each window's active-speaker count, rounded to the
    /// nearest integer exactly once after aggregation.
    pub speaker_count: Vec<u8>,
}

impl LocalActivity {
    pub(crate) fn valid_regions(&self, audio_duration_s: f64) -> Vec<TimeRange> {
        let mut regions = Vec::new();
        let mut start_s = None;
        for (index, &count) in self.speaker_count.iter().enumerate() {
            let midpoint_s = self
                .frame_clock
                .midpoint_s(index)
                .clamp(0.0, audio_duration_s);
            if count > 0 && start_s.is_none() {
                start_s = Some(midpoint_s);
            }
            if (count == 0 || index + 1 == self.speaker_count.len())
                && let Some(start) = start_s.take()
            {
                let end_s = if count > 0 {
                    audio_duration_s
                } else {
                    midpoint_s
                };
                if end_s > start {
                    regions.push(TimeRange::new(start, end_s));
                }
            }
        }
        regions
    }
}

/// Internal seam for recording-local activity models. Selection is performed
/// once during preflight; callers retain the selected adapter, so a load or
/// inference failure can never trigger an implicit fallback to another model.
pub(crate) trait LocalActivitySegmenter: Send + Sync {
    fn segment_local_activity(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
    ) -> Result<LocalActivity, SegmentError>;
}

/// segmentation-3.0 adapter using the official 10 s / 1 s sliding protocol.
pub struct PyannoteSegmenter {
    model: PyannetModel,
    protocol: SlidingProtocol,
}

#[derive(Debug, Clone, Copy)]
struct SlidingProtocol {
    window_s: f64,
    step_s: f64,
}

impl Default for SlidingProtocol {
    fn default() -> Self {
        Self {
            window_s: DEFAULT_WINDOW_S,
            step_s: DEFAULT_STEP_S,
        }
    }
}

impl PyannoteSegmenter {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_safetensors(bytes)?,
            protocol: SlidingProtocol::default(),
        })
    }

    pub fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_oasr(path)?,
            protocol: SlidingProtocol::default(),
        })
    }

    /// Compatibility helper for diagnostics. Production external diarization
    /// consumes [`LocalActivity`] and performs local-to-global reconstruction;
    /// this helper only renders each window-local slot independently.
    pub fn segment(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<Vec<SpeakerTurn>, WeightsError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Ok(Vec::new());
        }
        let (logp, frames) = self.model.forward(samples)?;
        Ok(decode_segments(&logp, frames))
    }

    fn infer_window(&self, samples: &[f32]) -> Result<Vec<u8>, SegmentError> {
        let (logp, frames) = self
            .model
            .forward(samples)
            .map_err(|error| SegmentError::Inference(error.to_string()))?;
        Ok(decode_activity(&logp, frames))
    }
}

impl LocalActivitySegmenter for PyannoteSegmenter {
    fn segment_local_activity(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        canceled: &dyn Fn() -> bool,
    ) -> Result<LocalActivity, SegmentError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(SegmentError::UnsupportedSampleRate(sample_rate_hz));
        }
        if samples.is_empty() {
            return Ok(LocalActivity {
                frame_clock: activity_frame_clock(),
                windows: Vec::new(),
                speaker_count: Vec::new(),
            });
        }

        let window_samples = (self.protocol.window_s * sample_rate_hz as f64) as usize;
        let step_samples = (self.protocol.step_s * sample_rate_hz as f64).round() as usize;
        let complete_windows = if samples.len() >= window_samples {
            1 + (samples.len() - window_samples) / step_samples
        } else {
            0
        };
        let has_last = samples.len() < window_samples
            || !(samples.len() - window_samples).is_multiple_of(step_samples);
        let total_windows = complete_windows + usize::from(has_last);
        let mut windows = Vec::with_capacity(total_windows);

        for index in 0..total_windows {
            if canceled() {
                return Err(SegmentError::Canceled);
            }
            let start = index * step_samples;
            let end = (start + window_samples).min(samples.len());
            let activity = if end - start == window_samples {
                self.infer_window(&samples[start..end])?
            } else {
                let mut padded = vec![0.0f32; window_samples];
                padded[..end - start].copy_from_slice(&samples[start..end]);
                self.infer_window(&padded)?
            };
            windows.push(LocalActivityWindow {
                start_s: start as f64 / sample_rate_hz as f64,
                frame_activity: activity,
            });
        }

        let frame_clock = activity_frame_clock();
        let audio_duration_s = samples.len() as f64 / sample_rate_hz as f64;
        for window in &mut windows {
            window.frame_activity.truncate(frame_count_for_duration(
                frame_clock,
                (audio_duration_s - window.start_s).max(0.0),
            ));
        }
        let speaker_count = aggregate_speaker_count(&windows, frame_clock, audio_duration_s);
        Ok(LocalActivity {
            frame_clock,
            windows,
            speaker_count,
        })
    }
}

fn activity_frame_clock() -> ActivityFrameClock {
    ActivityFrameClock {
        start_s: 0.0,
        duration_s: FRAME_DURATION_SAMPLES / SAMPLE_RATE_HZ as f64,
        step_s: FRAME_STEP_SAMPLES / SAMPLE_RATE_HZ as f64,
    }
}

fn aggregate_speaker_count(
    windows: &[LocalActivityWindow],
    frame_clock: ActivityFrameClock,
    audio_duration_s: f64,
) -> Vec<u8> {
    let Some(first) = windows.first() else {
        return Vec::new();
    };
    let frame_count = frame_count_for_duration(frame_clock, audio_duration_s);
    let mut sums = vec![0.0f32; frame_count];
    let mut observations = vec![0u16; frame_count];
    for window in windows {
        let start = frame_clock.closest_frame(window.start_s + frame_clock.duration_s * 0.5);
        for (offset, &activity) in window.frame_activity.iter().enumerate() {
            let index = start + offset;
            if index >= frame_count {
                break;
            }
            sums[index] += activity.count_ones() as f32;
            observations[index] = observations[index].saturating_add(1);
        }
    }
    debug_assert!(
        !first.frame_activity.is_empty() || windows.iter().all(|w| w.frame_activity.is_empty())
    );
    sums.into_iter()
        .zip(observations)
        .map(|(sum, count)| {
            if count == 0 {
                0
            } else {
                (sum / count as f32)
                    .round_ties_even()
                    .clamp(0.0, u8::MAX as f32) as u8
            }
        })
        .collect()
}

fn frame_count_for_duration(clock: ActivityFrameClock, duration_s: f64) -> usize {
    if duration_s < clock.duration_s * 0.5 {
        0
    } else {
        ((duration_s - clock.duration_s * 0.5) / clock.step_s).floor() as usize + 1
    }
}

fn decode_activity(logp: &[f32], frames: usize) -> Vec<u8> {
    logp.chunks_exact(NUM_CLASSES)
        .take(frames)
        .map(|row| {
            let class = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            POWERSET[class]
                .iter()
                .fold(0u8, |mask, &speaker| mask | (1 << speaker))
        })
        .collect()
}

fn decode_segments(logp: &[f32], frames: usize) -> Vec<SpeakerTurn> {
    let active = decode_activity(logp, frames);
    let time = |frame: usize| frame as f64 * FRAME_STEP_SAMPLES / SAMPLE_RATE_HZ as f64;
    let mut turns = Vec::new();
    for speaker in 0..MAX_LOCAL_SPEAKERS {
        let bit = 1u8 << speaker;
        let mut start = None;
        let mut overlapped = false;
        for (frame, &slots) in active.iter().enumerate() {
            if slots & bit != 0 {
                start.get_or_insert(frame);
                overlapped |= slots.count_ones() > 1;
            } else if let Some(begin) = start.take() {
                turns.push(SpeakerTurn {
                    range: TimeRange::new(time(begin), time(frame)),
                    speaker: SpeakerId(speaker as u32),
                    overlap: overlapped,
                });
                overlapped = false;
            }
        }
        if let Some(begin) = start {
            turns.push(SpeakerTurn {
                range: TimeRange::new(time(begin), time(frames)),
                speaker: SpeakerId(speaker as u32),
                overlap: overlapped,
            });
        }
    }
    turns
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    fn frame(class: usize) -> Vec<f32> {
        let mut row = vec![-10.0f32; NUM_CLASSES];
        row[class] = 0.0;
        row
    }

    #[test]
    fn decodes_speaker_change_and_overlap() {
        let logp: Vec<f32> = [0usize, 1, 4, 2, 0].into_iter().flat_map(frame).collect();
        let turns = decode_segments(&logp, 5);
        let s0: Vec<_> = turns
            .iter()
            .filter(|turn| turn.speaker == SpeakerId(0))
            .collect();
        let s1: Vec<_> = turns
            .iter()
            .filter(|turn| turn.speaker == SpeakerId(1))
            .collect();
        assert_eq!(s0.len(), 1);
        assert_eq!(s1.len(), 1);
        assert!(s0[0].overlap);
        assert!(s1[0].overlap);
    }

    #[test]
    fn count_aggregation_rounds_overlapping_windows() {
        let clock = ActivityFrameClock {
            start_s: 0.0,
            duration_s: 1.0,
            step_s: 1.0,
        };
        let windows = vec![
            LocalActivityWindow {
                start_s: 0.0,
                frame_activity: vec![0b01, 0b11],
            },
            LocalActivityWindow {
                start_s: 1.0,
                frame_activity: vec![0b01, 0b01],
            },
        ];
        assert_eq!(aggregate_speaker_count(&windows, clock, 3.0), vec![1, 2, 1]);
    }

    #[test]
    fn valid_regions_are_activity_only() {
        let activity = LocalActivity {
            frame_clock: ActivityFrameClock {
                start_s: 0.0,
                duration_s: 0.2,
                step_s: 0.1,
            },
            windows: Vec::new(),
            speaker_count: vec![0, 1, 1, 0],
        };
        assert_eq!(activity.valid_regions(1.0), vec![TimeRange::new(0.2, 0.4)]);

        let trailing = LocalActivity {
            speaker_count: vec![0, 0, 1],
            ..activity
        };
        let regions = trailing.valid_regions(0.35);
        assert_eq!(regions.len(), 1);
        assert!((regions[0].start_s - 0.3).abs() < 1.0e-9);
        assert_eq!(regions[0].end_s, 0.35);
    }
}
