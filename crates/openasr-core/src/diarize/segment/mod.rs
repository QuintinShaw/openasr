//! Speaker segmentation: pyannote segmentation-3.0.
//!
//! A pure-Rust forward pass of the PyanNet model that turns a waveform window
//! into per-frame powerset speaker-activity probabilities — finer than the VAD's
//! one-speaker-per-region heuristic, and the basis for overlap-aware diarization.

mod ops;
mod pack;
mod pyannet;

#[cfg(test)]
mod tests;

pub use pack::shared_segmenter;

use pyannet::{NUM_CLASSES, PyannetModel};

use super::contract::{SpeakerId, SpeakerTurn, TimeRange};
use super::embed::weights::WeightsError;

/// Sample rate the segmenter requires.
const SAMPLE_RATE_HZ: u32 = 16_000;
/// Output frame step in samples (SincNet stride 10 × three maxpool/3 = 270).
const FRAME_STEP_SAMPLES: f64 = 270.0;
/// Max concurrent local speakers the powerset head models.
const MAX_LOCAL_SPEAKERS: usize = 3;

/// Powerset class → the set of active local speakers (pyannote seg-3.0 ordering:
/// silence, the three singletons, then the three pairs).
const POWERSET: [&[usize]; NUM_CLASSES] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

/// pyannote segmentation-3.0 speaker segmenter.
pub struct PyannoteSegmenter {
    model: PyannetModel,
}

impl PyannoteSegmenter {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_safetensors(bytes)?,
        })
    }

    /// Load from a pulled `.oasr` (GGUF-v0) pack.
    pub fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        Ok(Self {
            model: PyannetModel::from_oasr(path)?,
        })
    }

    /// Segment `samples` into overlap-aware local-speaker turns. The returned
    /// `SpeakerId`s are window-local (0..3); the batch pipeline re-embeds and
    /// globally clusters each turn, so they need only separate speakers here.
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
}

/// Decode per-frame powerset log-probs `[frames, 7]` into contiguous per-speaker
/// turns, flagging turns that overlap another speaker.
fn decode_segments(logp: &[f32], frames: usize) -> Vec<SpeakerTurn> {
    // Per-frame active local speakers via argmax over the powerset classes.
    let active: Vec<&[usize]> = logp
        .chunks_exact(NUM_CLASSES)
        .map(|row| {
            let class = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            POWERSET[class]
        })
        .collect();

    let time = |frame: usize| frame as f64 * FRAME_STEP_SAMPLES / SAMPLE_RATE_HZ as f64;
    let mut turns = Vec::new();
    for speaker in 0..MAX_LOCAL_SPEAKERS {
        let mut start: Option<usize> = None;
        let mut overlapped = false;
        for (f, slots) in active.iter().enumerate() {
            if slots.contains(&speaker) {
                if start.is_none() {
                    start = Some(f);
                    overlapped = false;
                }
                if slots.len() > 1 {
                    overlapped = true;
                }
            } else if let Some(s) = start.take() {
                turns.push(SpeakerTurn {
                    range: TimeRange::new(time(s), time(f)),
                    speaker: SpeakerId(speaker as u32),
                    overlap: overlapped,
                });
            }
        }
        if let Some(s) = start.take() {
            turns.push(SpeakerTurn {
                range: TimeRange::new(time(s), time(frames)),
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
        // frames: silence, spk0, spk0+1 (overlap), spk1, silence.
        let logp: Vec<f32> = [0usize, 1, 4, 2, 0].into_iter().flat_map(frame).collect();
        let turns = decode_segments(&logp, 5);
        // speaker 0 active frames 1-2, speaker 1 active frames 2-3.
        let s0: Vec<_> = turns.iter().filter(|t| t.speaker == SpeakerId(0)).collect();
        let s1: Vec<_> = turns.iter().filter(|t| t.speaker == SpeakerId(1)).collect();
        assert_eq!(s0.len(), 1);
        assert_eq!(s1.len(), 1);
        assert!(s0[0].overlap, "speaker 0 overlapped at frame 2");
        assert!(s1[0].overlap, "speaker 1 overlapped at frame 2");
    }

    #[test]
    fn silence_yields_no_turns() {
        let logp: Vec<f32> = (0..3).flat_map(|_| frame(0)).collect();
        assert!(decode_segments(&logp, 3).is_empty());
    }
}
