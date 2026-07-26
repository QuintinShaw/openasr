//! Independent audibility judgement used to *validate* a long-form slice plan.
//!
//! # The invariant this module exists to enforce
//!
//! Long-form planning has two distinct kinds of level decision, and they must
//! never share a number:
//!
//! 1. **Decision**: which audio the pipeline keeps, and where it cuts. That is
//!    the energy VAD's gate (`vad.rs`) and the silence-aware split search
//!    (`choose_forced_cut`), both anchored on the configurable
//!    [`LongFormOptions::energy_silence_threshold_db`](super::LongFormOptions)
//!    (-38 dBFS by default).
//! 2. **Validation**: did the plan that came out of (1) throw away content a
//!    human would have heard? That is this module.
//!
//! If (2) measures "audible" against the same line (1) used to decide, the
//! check is a closed loop: the energy VAD elides exactly what falls under the
//! absolute silence floor, so a validator reading that same floor back is
//! structurally incapable of ever disagreeing with it. That is not a
//! mis-tuned constant -- no value of the constant fixes it -- and it shipped:
//! on a far-field meeting whose speech sits at -44..-50 dBFS, entirely below
//! the -38 dBFS floor, the auto planner elided 47% of a 360s recording and
//! the coverage guard called every dropped second silence, while the *neural*
//! VAD candidates (which decide by a different quantity) were the ones the
//! guard disqualified.
//!
//! **INVARIANT: nothing in this module may read `energy_silence_threshold_db`,
//! and no caller may pass a threshold derived from it.** "Unifying the
//! constants" reinstates the bug. The audibility threshold here is
//! deliberately *not* configurable for the same reason: a knob is one edit
//! away from being wired to the VAD's knob.
//!
//! # What replaces it
//!
//! A recording-relative reference. A dropped region counts as audible when a
//! short window inside it is within [`AUDIBLE_MARGIN_BELOW_SPEECH_LEVEL_DB`]
//! of the *speech level of this very recording* -- estimated as a high
//! percentile of windowed RMS, taken as the lower of the whole recording's
//! level and the level of the audio the plan chose to keep (so a plan that
//! keeps only the loudest passages cannot raise its own bar, and a plan that
//! keeps only quiet passages while dropping loud ones is still caught). The
//! only absolute constant is [`NEVER_AUDIBLE_FLOOR_DBFS`], far below any
//! usable capture level, which exists so a recording of pure digital silence
//! does not read as "all of it is speech".

use super::slicing::{rms, seconds_to_samples};

/// Window size for level statistics and for the audible-content scan.
/// Deliberately much coarser than the VAD's own 20ms analysis frame
/// (`vad::DEFAULT_FRAME_MS`) -- this scan exists to catch a burst of real
/// speech that a *whole-region* average would dilute into apparent silence,
/// not to re-derive frame-level VAD. 0.5s is short enough to isolate a few
/// words of speech from surrounding silence, long enough to stay cheap and to
/// avoid tripping on a single loud click or breath.
pub(super) const AUDIBILITY_WINDOW_SECONDS: f32 = 0.5;

/// How far below the recording's own speech level a dropped window may sit
/// before it stops counting as possibly-audible content.
///
/// Speech is not level within a recording: quiet syllables, a talker turning
/// away from the mic, and a far-field participant all run well under the
/// loudest talker, and every one of those is content a transcript must keep.
/// 20 dB is wide enough to cover that spread -- on the AliMeeting far-field
/// sessions the 5th and 90th percentile 0.5s windows are 13-23 dB apart, and
/// the quiet talkers land inside the band -- while still leaving room for the
/// gaps a normally levelled recording actually elides: measured on four real
/// and synthetic recordings whose pauses are true room tone (25-50 dB under
/// their speech), the packed layout still wins at this margin.
pub(super) const AUDIBLE_MARGIN_BELOW_SPEECH_LEVEL_DB: f32 = 20.0;

/// Absolute floor under which nothing is ever called audible, regardless of
/// the relative rule. Not a silence threshold for any *decision* -- it is far
/// below the level of any recording a user would try to transcribe, and only
/// keeps the relative rule from declaring the contents of an all-silent (or
/// dither-only) recording to be speech.
pub(super) const NEVER_AUDIBLE_FLOOR_DBFS: f32 = -60.0;

/// Percentile of windowed RMS taken as "this recording's speech level".
/// High enough to track the talkers rather than the pauses, below the peak so
/// a single transient cannot set the reference.
const SPEECH_LEVEL_PERCENTILE: f32 = 0.90;

/// Short label naming the criterion in logs, so a dropped-audio report says
/// *how* the call was made and not just that it was made.
pub(super) const AUDIBILITY_CRITERION_LABEL: &str = "relative_to_recording_speech_level";

/// The audibility reference for one candidate plan over one recording.
///
/// Built from the audio itself and the plan's kept ranges; see the module
/// docs for why it must not be built from `energy_silence_threshold_db`.
#[derive(Debug, Clone, Copy)]
pub(super) struct AudibilityReference {
    speech_level_linear: f32,
    threshold_linear: f32,
    window_samples: usize,
}

impl AudibilityReference {
    /// `kept_ranges` are the plan's kept original-sample ranges (empty is
    /// allowed and means "this plan keeps nothing", which makes the reference
    /// fall back to the absolute floor so any real content reads as dropped).
    pub(super) fn for_plan(
        samples: &[f32],
        sample_rate_hz: u32,
        kept_ranges: &[(usize, usize)],
    ) -> Self {
        let window_samples = seconds_to_samples(AUDIBILITY_WINDOW_SECONDS, sample_rate_hz).max(1);
        let floor_linear = 10.0_f32.powf(NEVER_AUDIBLE_FLOOR_DBFS / 20.0);
        let recording_level = speech_level_of(samples, 0, samples.len(), window_samples);
        let kept_level = kept_ranges
            .iter()
            .flat_map(|(start, end)| window_levels(samples, *start, *end, window_samples))
            .collect::<Vec<_>>();
        let kept_level = percentile(kept_level, SPEECH_LEVEL_PERCENTILE);
        // The lower of the two: a plan cannot raise its own bar by keeping
        // only the loudest passages, and a plan that keeps only quiet audio
        // is still measured against the quiet level it kept.
        let speech_level_linear = if kept_ranges.is_empty() {
            recording_level
        } else {
            recording_level.min(kept_level)
        };
        let relative_threshold =
            speech_level_linear * 10.0_f32.powf(-AUDIBLE_MARGIN_BELOW_SPEECH_LEVEL_DB / 20.0);
        Self {
            speech_level_linear,
            threshold_linear: relative_threshold.max(floor_linear),
            window_samples,
        }
    }

    pub(super) fn threshold_dbfs(&self) -> f32 {
        linear_to_dbfs(self.threshold_linear)
    }

    pub(super) fn speech_level_dbfs(&self) -> f32 {
        linear_to_dbfs(self.speech_level_linear)
    }

    /// Scans `[start, end)` in fixed windows and returns the first window
    /// whose RMS exceeds the reference, if any. Windowed rather than a single
    /// average over the whole range so a short loud passage inside an
    /// otherwise quiet dropped region is still caught (see
    /// [`AUDIBILITY_WINDOW_SECONDS`]).
    pub(super) fn find_audible_window(
        &self,
        samples: &[f32],
        start: usize,
        end: usize,
    ) -> Option<(usize, usize, f32)> {
        let start = start.min(samples.len());
        let end = end.min(samples.len());
        if end <= start {
            return None;
        }
        let mut cursor = start;
        while cursor < end {
            let window_end = (cursor + self.window_samples).min(end);
            let window_rms = rms(&samples[cursor..window_end]);
            if window_rms > self.threshold_linear {
                return Some((cursor, window_end, window_rms));
            }
            cursor = window_end;
        }
        None
    }

    /// How far a level sits above the audibility reference, as a ratio >= 0.
    /// Used to scale the soft elision/gap-edge penalties so their charge is
    /// expressed relative to this recording's own speech level rather than to
    /// an absolute dBFS line.
    pub(super) fn excess_ratio(&self, level_linear: f32) -> f32 {
        if self.threshold_linear <= 0.0 {
            return 0.0;
        }
        (level_linear / self.threshold_linear).max(1.0) - 1.0
    }

    pub(super) fn is_audible(&self, level_linear: f32) -> bool {
        level_linear > self.threshold_linear
    }
}

pub(super) fn linear_to_dbfs(value: f32) -> f32 {
    20.0 * value.max(f32::MIN_POSITIVE).log10()
}

fn speech_level_of(samples: &[f32], start: usize, end: usize, window_samples: usize) -> f32 {
    percentile(
        window_levels(samples, start, end, window_samples),
        SPEECH_LEVEL_PERCENTILE,
    )
}

fn window_levels(samples: &[f32], start: usize, end: usize, window_samples: usize) -> Vec<f32> {
    let start = start.min(samples.len());
    let end = end.min(samples.len());
    if end <= start || window_samples == 0 {
        return Vec::new();
    }
    let mut levels = Vec::with_capacity((end - start) / window_samples + 1);
    let mut cursor = start;
    while cursor < end {
        let window_end = (cursor + window_samples).min(end);
        levels.push(rms(&samples[cursor..window_end]));
        cursor = window_end;
    }
    levels
}

fn percentile(mut values: Vec<f32>, fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clamped = fraction.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f32 * clamped).round() as usize;
    values[index.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(samples: usize, amplitude: f32, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..samples)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = (state >> 40) as f32 / 8_388_608.0 - 1.0;
                value * amplitude
            })
            .collect()
    }

    /// The point of the whole module: on a recording whose speech sits below
    /// the pipeline's absolute silence floor, the reference tracks the
    /// recording down instead of calling all of it silence.
    #[test]
    fn threshold_tracks_a_quiet_recording_instead_of_an_absolute_floor() {
        // ~-38.3 dBFS speech, i.e. just below the -38 dBFS energy silence
        // floor and far enough above `NEVER_AUDIBLE_FLOOR_DBFS` that the
        // relative rule, not the absolute clamp, is what sets the threshold.
        let samples = noise(16_000 * 20, 0.021, 0x1234_5678);
        let speech_level = rms(&samples[..16_000]);
        assert!(
            speech_level < 10.0_f32.powf(-38.0 / 20.0),
            "fixture must sit below the pipeline's absolute silence floor, got {:.1} dBFS",
            linear_to_dbfs(speech_level)
        );
        let reference = AudibilityReference::for_plan(&samples, 16_000, &[(0, samples.len())]);
        assert!(
            reference.threshold_dbfs() < -50.0
                && reference.threshold_dbfs() > NEVER_AUDIBLE_FLOOR_DBFS,
            "threshold must track this recording's own speech level, got {:.1} dBFS",
            reference.threshold_dbfs()
        );
        assert!(
            reference.is_audible(speech_level),
            "sub-floor speech must read as audible"
        );
    }

    /// ... but a normally levelled recording keeps a threshold far above its
    /// room tone, so packing out true silence stays available.
    #[test]
    fn threshold_stays_above_room_tone_of_a_normal_recording() {
        let mut samples = noise(16_000 * 20, 0.2, 0xabcd_ef01);
        let room_tone = noise(16_000 * 10, 0.006, 0x0fed_cba9);
        samples.extend(room_tone.iter().copied());
        let reference = AudibilityReference::for_plan(&samples, 16_000, &[(0, 16_000 * 20)]);
        assert!(
            !reference.is_audible(rms(&room_tone)),
            "room tone {:.1} dBFS must stay under the threshold {:.1} dBFS",
            linear_to_dbfs(rms(&room_tone)),
            reference.threshold_dbfs()
        );
        assert!(
            reference
                .find_audible_window(&samples, 16_000 * 20, samples.len())
                .is_none()
        );
    }

    /// Digital silence must not read as speech just because the relative rule
    /// scales with the recording.
    #[test]
    fn all_silent_recording_is_never_audible() {
        let samples = vec![0.0_f32; 16_000 * 10];
        let reference = AudibilityReference::for_plan(&samples, 16_000, &[(0, 16_000)]);
        assert!(
            reference
                .find_audible_window(&samples, 0, samples.len())
                .is_none()
        );
    }

    /// A plan that keeps only the loudest talker cannot raise its own bar out
    /// of the recording's speech spread: a later, quieter talker it dropped
    /// still reads as audible (this is the long-form code-switch shape -- a
    /// loud opener near -25 dBFS and a quieter tail near -34 dBFS).
    #[test]
    fn keeping_only_the_loud_talker_does_not_raise_the_bar() {
        let mut samples = noise(16_000 * 10, 0.098, 0x2222_3333);
        samples.extend(noise(16_000 * 30, 0.035, 0x4444_5555));
        let reference = AudibilityReference::for_plan(&samples, 16_000, &[(0, 16_000 * 10)]);
        assert!(
            reference
                .find_audible_window(&samples, 16_000 * 10, samples.len())
                .is_some(),
            "quiet dropped speech must stay audible against a loud kept passage \
             (threshold {:.1} dBFS)",
            reference.threshold_dbfs()
        );
    }
}
