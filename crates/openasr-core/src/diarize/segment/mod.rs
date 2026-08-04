//! Recording-local speaker activity segmentation.
//!
//! The external diarizer depends on the small [`LocalActivitySegmenter`] seam,
//! not on pyannote or any future model family. The seam deliberately returns
//! unaligned window-local activity and an aggregated speaker count: global
//! speaker identity is reconstructed later from ReDim embeddings.

mod diarizen;
mod ops;
mod pack;
mod policy_runtime;
mod pyannet;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use rayon::prelude::*;

pub(crate) use diarizen::DiariZenRuntime;
pub use diarizen::{
    DIARIZEN_GGML_ARCHITECTURE_ID, DiariZenSegmenter, DiariZenSegmenterError, DiariZenWindowOutput,
    diarizen_pack_installed,
};
pub use pack::{DIARIZEN_PACK_ID, SEGMENTER_PACK_ID, segmenter_pack_installed};
pub(crate) use pack::{PreparedSelectedSegmenter, SegmenterProvider, prepare_segmenter};
pub use policy_runtime::{PolicyResolvedPyannoteSegmenterRuntime, PolicyResolvedSegmenterRuntime};

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
const PYANNOTE_MAX_WINDOW_WORKERS: usize = 4;

static PYANNOTE_WINDOW_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmenterWorkingSetGeometry {
    activity_frame_clock: ActivityFrameClock,
    window_samples: usize,
    pub window_step_samples: usize,
    pub frames_per_window: usize,
    pub local_speaker_slots: usize,
    /// Provider-owned Rust payload that may coexist for one model window.
    /// Native backend buffers are admitted by their graph owner instead.
    pub inference_peak_bytes_per_window: u64,
    pub max_parallel_windows: usize,
    /// Per-result header retained by the bounded parallel collection batch.
    /// Serial providers return zero because they write directly to `windows`.
    pub parallel_batch_slot_bytes: usize,
    pub retain_starts_through_aggregation: bool,
}

impl SegmenterWorkingSetGeometry {
    pub(crate) fn activity_frame_count(self, samples: usize) -> usize {
        self.activity_frame_clock.frame_count_for_samples(samples)
    }

    pub(crate) const fn window_count(self, samples: usize) -> usize {
        sliding_window_count(samples, self.window_samples, self.window_step_samples)
    }

    pub(crate) const fn padded_tail_bytes(self, samples: usize) -> u64 {
        let windows = self.window_count(samples);
        if windows == 0 {
            return 0;
        }
        let last_start = (windows - 1).saturating_mul(self.window_step_samples);
        if samples.saturating_sub(last_start) >= self.window_samples {
            0
        } else {
            (self.window_samples as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
        }
    }
}

pub(crate) fn segmenter_working_set_geometry(
    provider: SegmenterProvider,
) -> SegmenterWorkingSetGeometry {
    match provider {
        SegmenterProvider::Segmentation3_0 => {
            let window_samples = 10 * SAMPLE_RATE_HZ as usize;
            SegmenterWorkingSetGeometry {
                activity_frame_clock: activity_frame_clock(),
                window_samples,
                window_step_samples: SAMPLE_RATE_HZ as usize,
                frames_per_window: pyannet::output_frame_count(window_samples),
                local_speaker_slots: MAX_LOCAL_SPEAKERS,
                inference_peak_bytes_per_window: pyannet::quoted_forward_peak_bytes(window_samples),
                max_parallel_windows: pyannote_window_worker_count(),
                parallel_batch_slot_bytes: std::mem::size_of::<
                    Result<LocalActivityWindow, SegmentError>,
                >(),
                retain_starts_through_aggregation: true,
            }
        }
        SegmenterProvider::DiariZen => {
            let frames_per_window = 1
                + (diarizen::DIARIZEN_WINDOW_SAMPLES
                    - diarizen::DIARIZEN_FRAME_DURATION_SAMPLES as usize)
                    / diarizen::DIARIZEN_FRAME_STEP_SAMPLES as usize;
            SegmenterWorkingSetGeometry {
                activity_frame_clock: ActivityFrameClock::new(
                    0,
                    diarizen::DIARIZEN_FRAME_DURATION_SAMPLES,
                    diarizen::DIARIZEN_FRAME_STEP_SAMPLES,
                    diarizen::DIARIZEN_SAMPLE_RATE_HZ,
                ),
                window_samples: diarizen::DIARIZEN_WINDOW_SAMPLES,
                window_step_samples: diarizen::DIARIZEN_WINDOW_STEP_SAMPLES,
                frames_per_window,
                local_speaker_slots: diarizen::DIARIZEN_LOCAL_SPEAKERS,
                // logits + class row + raw activity + median-filtered activity
                // overlap inside postprocess; the final window mask is then
                // allocated before that model output is dropped.
                inference_peak_bytes_per_window: frames_per_window as u64
                    * (diarizen::DIARIZEN_POWERSET_CLASSES as u64
                        * std::mem::size_of::<f32>() as u64
                        + 2 * diarizen::DIARIZEN_LOCAL_SPEAKERS as u64
                        + 2),
                max_parallel_windows: 1,
                parallel_batch_slot_bytes: 0,
                retain_starts_through_aggregation: false,
            }
        }
    }
}

fn pyannote_window_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(PYANNOTE_MAX_WINDOW_WORKERS)
}

fn pyannote_window_pool() -> &'static Result<rayon::ThreadPool, String> {
    PYANNOTE_WINDOW_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(pyannote_window_worker_count())
            .thread_name(|index| format!("openasr-pyannote-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
}

fn bounded_pyannote_window_map<T, F>(
    starts: &[usize],
    canceled: &dyn Fn() -> bool,
    map: F,
) -> Result<Vec<T>, SegmentError>
where
    T: Send,
    F: Fn(usize) -> Result<T, SegmentError> + Sync,
{
    let pool = pyannote_window_pool().as_ref().map_err(|error| {
        SegmentError::Inference(format!(
            "could not create bounded segmentation-3.0 window pool: {error}"
        ))
    })?;
    let mut output = Vec::with_capacity(starts.len());
    for starts_batch in starts.chunks(pool.current_num_threads().max(1)) {
        if canceled() {
            return Err(SegmentError::Canceled);
        }
        let batch: Vec<Result<T, SegmentError>> =
            pool.install(|| starts_batch.par_iter().map(|&start| map(start)).collect());
        for item in batch {
            output.push(item?);
        }
    }
    Ok(output)
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActivityFrameClock {
    start_samples: u64,
    duration_samples: u32,
    step_samples: u32,
    sample_rate_hz: u32,
}

impl ActivityFrameClock {
    pub(crate) const fn new(
        start_samples: u64,
        duration_samples: u32,
        step_samples: u32,
        sample_rate_hz: u32,
    ) -> Self {
        Self {
            start_samples,
            duration_samples,
            step_samples,
            sample_rate_hz,
        }
    }

    pub(crate) fn duration_s(self) -> f64 {
        self.duration_samples as f64 / self.sample_rate_hz as f64
    }

    pub(crate) fn midpoint_s(self, frame: usize) -> f64 {
        self.midpoint_samples_x2(frame) as f64 / (2.0 * self.sample_rate_hz as f64)
    }

    pub(crate) fn closest_frame(self, timestamp_s: f64) -> usize {
        ((timestamp_s - self.midpoint_s(0))
            / (self.step_samples as f64 / self.sample_rate_hz as f64))
            .round_ties_even()
            .max(0.0) as usize
    }

    pub(crate) fn closest_frame_for_window_start(self, start_sample: usize) -> usize {
        let target = (start_sample as u128)
            .saturating_mul(2)
            .saturating_add(self.duration_samples as u128);
        self.closest_frame_to_samples_x2(target)
    }

    fn midpoint_samples_x2(self, frame: usize) -> u128 {
        (self.start_samples as u128)
            .saturating_mul(2)
            .saturating_add(self.duration_samples as u128)
            .saturating_add(
                (frame as u128)
                    .saturating_mul(self.step_samples as u128)
                    .saturating_mul(2),
            )
    }

    fn closest_frame_to_samples_x2(self, target: u128) -> usize {
        let first = self.midpoint_samples_x2(0);
        if target <= first {
            return 0;
        }
        let step_x2 = (self.step_samples as u128) * 2;
        let delta = target - first;
        let quotient = delta / step_x2;
        let remainder = delta % step_x2;
        let rounded = match (remainder * 2).cmp(&step_x2) {
            std::cmp::Ordering::Less => quotient,
            std::cmp::Ordering::Greater => quotient + 1,
            std::cmp::Ordering::Equal if quotient % 2 == 1 => quotient + 1,
            std::cmp::Ordering::Equal => quotient,
        };
        rounded.min(usize::MAX as u128) as usize
    }

    fn frame_count_for_samples(self, samples: usize) -> usize {
        let audio_end_x2 = (samples as u128).saturating_mul(2);
        let first = self.midpoint_samples_x2(0);
        if audio_end_x2 < first {
            0
        } else {
            let count = (audio_end_x2 - first) / ((self.step_samples as u128) * 2) + 1;
            count.min(usize::MAX as u128) as usize
        }
    }
}

/// One fixed-duration inference window. Each frame is a bitset of active
/// window-local speaker slots; bit `n` means local slot `n` is active.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalActivityWindow {
    pub start_sample: usize,
    pub frame_activity: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalActivity {
    pub frame_clock: ActivityFrameClock,
    pub windows: Vec<LocalActivityWindow>,
    /// Number of window-local speaker slots emitted by the selected model.
    /// This is part of the activity contract: segmentation-3.0 emits three,
    /// while DiariZen emits four.
    pub local_speaker_slots: u8,
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
        samples: crate::PcmSlice,
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
    #[cfg(test)]
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_safetensors(bytes)?,
            protocol: SlidingProtocol::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_oasr(path)?,
            protocol: SlidingProtocol::default(),
        })
    }

    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_preflight(preflight)?,
            protocol: SlidingProtocol::default(),
        })
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, WeightsError> {
        PyannetModel::quoted_persistent_host_commitment_bytes(tensor_index)
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, WeightsError> {
        self.model.persistent_host_commitment_bytes()
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
        samples: crate::PcmSlice,
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
                local_speaker_slots: MAX_LOCAL_SPEAKERS as u8,
                speaker_count: Vec::new(),
            });
        }

        let window_samples = (self.protocol.window_s * sample_rate_hz as f64) as usize;
        let step_samples = (self.protocol.step_s * sample_rate_hz as f64).round() as usize;
        let starts = sliding_window_starts(samples.len(), window_samples, step_samples);
        let mut windows = bounded_pyannote_window_map(&starts, canceled, |start| {
            let end = (start + window_samples).min(samples.len());
            let activity = if end - start == window_samples {
                self.infer_window(&samples[start..end])?
            } else {
                let mut padded = vec![0.0f32; window_samples];
                padded[..end - start].copy_from_slice(&samples[start..end]);
                self.infer_window(&padded)?
            };
            Ok(LocalActivityWindow {
                start_sample: start,
                frame_activity: activity,
            })
        })?;

        let frame_clock = activity_frame_clock();
        for window in &mut windows {
            window.frame_activity.truncate(
                frame_clock
                    .frame_count_for_samples(samples.len().saturating_sub(window.start_sample)),
            );
        }
        let speaker_count = aggregate_speaker_count(&windows, frame_clock, samples.len());
        Ok(LocalActivity {
            frame_clock,
            windows,
            local_speaker_slots: MAX_LOCAL_SPEAKERS as u8,
            speaker_count,
        })
    }
}

pub(super) fn segment_diarizen_local_activity(
    samples: crate::PcmSlice,
    sample_rate_hz: u32,
    canceled: &dyn Fn() -> bool,
    mut infer_window: impl FnMut(
        crate::PcmSlice,
    ) -> Result<diarizen::DiariZenWindowOutput, SegmentError>,
) -> Result<LocalActivity, SegmentError> {
    use diarizen::{
        DIARIZEN_FRAME_DURATION_SAMPLES, DIARIZEN_FRAME_STEP_SAMPLES, DIARIZEN_LOCAL_SPEAKERS,
        DIARIZEN_SAMPLE_RATE_HZ, DIARIZEN_WINDOW_SAMPLES, DIARIZEN_WINDOW_STEP_SAMPLES,
    };

    if sample_rate_hz != DIARIZEN_SAMPLE_RATE_HZ {
        return Err(SegmentError::UnsupportedSampleRate(sample_rate_hz));
    }
    let frame_clock = ActivityFrameClock::new(
        0,
        DIARIZEN_FRAME_DURATION_SAMPLES,
        DIARIZEN_FRAME_STEP_SAMPLES,
        DIARIZEN_SAMPLE_RATE_HZ,
    );
    if samples.is_empty() {
        return Ok(LocalActivity {
            frame_clock,
            windows: Vec::new(),
            local_speaker_slots: DIARIZEN_LOCAL_SPEAKERS as u8,
            speaker_count: Vec::new(),
        });
    }

    let starts = sliding_window_starts(
        samples.len(),
        DIARIZEN_WINDOW_SAMPLES,
        DIARIZEN_WINDOW_STEP_SAMPLES,
    );
    let mut windows = Vec::with_capacity(starts.len());
    for start in starts {
        if canceled() {
            return Err(SegmentError::Canceled);
        }
        let end = (start + DIARIZEN_WINDOW_SAMPLES).min(samples.len());
        let output = if end - start == DIARIZEN_WINDOW_SAMPLES {
            infer_window(samples.slice(start..end))?
        } else {
            let mut padded = vec![0.0f32; DIARIZEN_WINDOW_SAMPLES];
            padded[..end - start].copy_from_slice(&samples[start..end]);
            infer_window(padded.into())?
        };
        if canceled() {
            return Err(SegmentError::Canceled);
        }
        if output.activity.len() != output.frame_count * DIARIZEN_LOCAL_SPEAKERS {
            return Err(SegmentError::Inference(format!(
                "DiariZen activity shape mismatch: {} values for {} frames x {} speakers",
                output.activity.len(),
                output.frame_count,
                DIARIZEN_LOCAL_SPEAKERS
            )));
        }
        let frame_activity = output
            .activity
            .chunks_exact(DIARIZEN_LOCAL_SPEAKERS)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold(0u8, |mask, (speaker, &active)| {
                        mask | (u8::from(active != 0) << speaker)
                    })
            })
            .collect();
        windows.push(LocalActivityWindow {
            start_sample: start,
            frame_activity,
        });
    }

    for window in &mut windows {
        window.frame_activity.truncate(
            frame_clock.frame_count_for_samples(samples.len().saturating_sub(window.start_sample)),
        );
    }
    let speaker_count = aggregate_speaker_count(&windows, frame_clock, samples.len());
    Ok(LocalActivity {
        frame_clock,
        windows,
        local_speaker_slots: DIARIZEN_LOCAL_SPEAKERS as u8,
        speaker_count,
    })
}

const fn sliding_window_count(
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
) -> usize {
    debug_assert!(window_samples > 0);
    debug_assert!(step_samples > 0);
    if sample_count == 0 {
        return 0;
    }
    let complete_windows = if sample_count >= window_samples {
        1 + (sample_count - window_samples) / step_samples
    } else {
        0
    };
    let has_last = sample_count < window_samples
        || !(sample_count - window_samples).is_multiple_of(step_samples);
    complete_windows + has_last as usize
}

fn sliding_window_starts(
    sample_count: usize,
    window_samples: usize,
    step_samples: usize,
) -> Vec<usize> {
    (0..sliding_window_count(sample_count, window_samples, step_samples))
        .map(|index| index * step_samples)
        .collect()
}

fn activity_frame_clock() -> ActivityFrameClock {
    ActivityFrameClock::new(
        0,
        FRAME_DURATION_SAMPLES as u32,
        FRAME_STEP_SAMPLES as u32,
        SAMPLE_RATE_HZ,
    )
}

fn aggregate_speaker_count(
    windows: &[LocalActivityWindow],
    frame_clock: ActivityFrameClock,
    audio_samples: usize,
) -> Vec<u8> {
    let Some(first) = windows.first() else {
        return Vec::new();
    };
    let frame_count = frame_clock.frame_count_for_samples(audio_samples);
    let mut sums = vec![0.0f32; frame_count];
    let mut observations = vec![0u16; frame_count];
    for window in windows {
        let start = frame_clock.closest_frame_for_window_start(window.start_sample);
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
        let clock = ActivityFrameClock::new(0, 1, 1, 1);
        let windows = vec![
            LocalActivityWindow {
                start_sample: 0,
                frame_activity: vec![0b01, 0b11],
            },
            LocalActivityWindow {
                start_sample: 1,
                frame_activity: vec![0b01, 0b01],
            },
        ];
        assert_eq!(aggregate_speaker_count(&windows, clock, 3), vec![1, 2, 1]);
    }

    #[test]
    fn valid_regions_are_activity_only() {
        let activity = LocalActivity {
            frame_clock: ActivityFrameClock::new(0, 2, 1, 10),
            windows: Vec::new(),
            local_speaker_slots: 3,
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

    #[test]
    fn frame_count_uses_exact_sample_boundaries() {
        let diarizen = ActivityFrameClock::new(0, 400, 320, 16_000);
        assert_eq!(diarizen.frame_count_for_samples(199), 0);
        assert_eq!(diarizen.frame_count_for_samples(200), 1);
        assert_eq!(diarizen.frame_count_for_samples(519), 1);
        assert_eq!(diarizen.frame_count_for_samples(520), 2);
        assert_eq!(diarizen.frame_count_for_samples(839), 2);
        assert_eq!(diarizen.frame_count_for_samples(840), 3);

        let pyannote = activity_frame_clock();
        assert_eq!(pyannote.frame_count_for_samples(455), 0);
        assert_eq!(pyannote.frame_count_for_samples(456), 1);
    }

    #[test]
    fn diarizen_adapter_uses_official_geometry_and_four_slot_masks() {
        let samples: crate::PcmSlice = vec![0.0f32; diarizen::DIARIZEN_WINDOW_SAMPLES].into();
        let activity = segment_diarizen_local_activity(samples, 16_000, &|| false, |window| {
            assert_eq!(window.len(), diarizen::DIARIZEN_WINDOW_SAMPLES);
            Ok(diarizen::DiariZenWindowOutput {
                frame_count: 4,
                logits: Vec::new(),
                powerset_class: Vec::new(),
                activity: vec![
                    1, 0, 0, 0, // local 0
                    0, 1, 0, 1, // local 1 + 3 overlap
                    0, 0, 1, 0, // local 2
                    0, 0, 0, 0,
                ],
            })
        })
        .expect("adapter");

        assert_eq!(activity.frame_clock.midpoint_s(0), 0.0125);
        assert_eq!(activity.frame_clock.midpoint_s(1), 0.0325);
        assert_eq!(activity.local_speaker_slots, 4);
        assert_eq!(activity.windows.len(), 1);
        assert_eq!(
            activity.windows[0].frame_activity,
            vec![0b0001, 0b1010, 0b0100, 0]
        );
    }

    #[test]
    fn diarizen_adapter_pads_and_truncates_the_orphan_window() {
        let samples: crate::PcmSlice = vec![1.0f32; 17 * 16_000].into();
        let mut calls = 0;
        let activity = segment_diarizen_local_activity(samples, 16_000, &|| false, |window| {
            calls += 1;
            assert_eq!(window.len(), diarizen::DIARIZEN_WINDOW_SAMPLES);
            if calls == 2 {
                assert_eq!(window[15 * 16_000 + 6_399], 1.0);
                assert_eq!(window[15 * 16_000 + 6_400], 0.0);
                assert_eq!(window[diarizen::DIARIZEN_WINDOW_SAMPLES - 1], 0.0);
            }
            Ok(diarizen::DiariZenWindowOutput {
                frame_count: 799,
                logits: Vec::new(),
                powerset_class: Vec::new(),
                activity: vec![0; 799 * diarizen::DIARIZEN_LOCAL_SPEAKERS],
            })
        })
        .expect("adapter");

        assert_eq!(calls, 2);
        assert_eq!(activity.windows[0].start_sample, 0);
        assert_eq!(
            activity.windows[1].start_sample,
            diarizen::DIARIZEN_WINDOW_STEP_SAMPLES
        );
        assert_eq!(activity.windows[0].frame_activity.len(), 799);
        assert_eq!(activity.windows[1].frame_activity.len(), 770);
    }

    #[test]
    fn diarizen_adapter_empty_and_cancellation_are_explicit() {
        let empty =
            segment_diarizen_local_activity(Vec::<f32>::new().into(), 16_000, &|| false, |_| {
                panic!("empty audio must not run inference")
            })
            .expect("empty");
        assert!(empty.windows.is_empty());
        assert_eq!(empty.local_speaker_slots, 4);

        let calls = std::cell::Cell::new(0);
        let error = segment_diarizen_local_activity(
            vec![0.0f32; 17 * 16_000].into(),
            16_000,
            &|| calls.get() > 0,
            |_| {
                calls.set(calls.get() + 1);
                Ok(diarizen::DiariZenWindowOutput {
                    frame_count: 1,
                    logits: Vec::new(),
                    powerset_class: Vec::new(),
                    activity: vec![0; diarizen::DIARIZEN_LOCAL_SPEAKERS],
                })
            },
        )
        .expect_err("cancel after first window");
        assert!(matches!(error, SegmentError::Canceled));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn sliding_window_geometry_does_not_duplicate_exact_tail() {
        assert_eq!(sliding_window_starts(0, 160, 16), Vec::<usize>::new());
        assert_eq!(sliding_window_starts(159, 160, 16), vec![0]);
        assert_eq!(sliding_window_starts(160, 160, 16), vec![0]);
        assert_eq!(sliding_window_starts(176, 160, 16), vec![0, 16]);
        assert_eq!(sliding_window_starts(177, 160, 16), vec![0, 16, 32]);
    }
}
