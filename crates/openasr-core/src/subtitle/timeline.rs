//! Timeline precision policy and dual-view projection.
//!
//! After decode, optional forced alignment, and speaker attribution, the
//! finished transcript is projected into:
//! - `segments`: speaker-merged reading paragraphs (manuscript)
//! - `subtitle_cues`: short cues for SRT/VTT and on-screen display
//!
//! Both views share the same attributed word timeline.

use serde::{Deserialize, Serialize};

use super::anchors::WordAnchorValidation;
use super::cues::resegment_segments_into_cues;
use super::reading::merge_reading_segments;
use crate::api::backend::{Segment, Transcription};

/// How precise word timestamps must be for this request.
///
/// Request-layer policy (product contract):
/// - [`Auto`](Self::Auto) (default): guarantee a precise timeline only when
///   Voice ID needs word anchors, the response is a subtitle export
///   (SRT/VTT), or the caller explicitly refined.
/// - [`Always`](Self::Always): guarantee precise word timestamps during
///   transcription (skip the aligner when native anchors already validate).
/// - [`Off`](Self::Off): do not run the forced aligner for timeline quality;
///   keep model-native timestamps. Voice ID that still requires word anchors
///   overrides this for the alignment step only (see
///   [`decide_forced_alignment`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelinePrecisionPolicy {
    #[default]
    Auto,
    Always,
    Off,
}

impl TimelinePrecisionPolicy {
    pub const ALL: &'static [&'static str] = &["auto", "always", "off"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Always => "always",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for TimelinePrecisionPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            "off" => Ok(Self::Off),
            other => Err(format!(
                "Unsupported timeline precision '{other}'. Use one of: {}.",
                Self::ALL.join(", ")
            )),
        }
    }
}

/// Provenance of the word timeline attached to a finished transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelineQuality {
    /// Native word timestamps passed runtime validation.
    NativeReliable,
    /// Forced aligner produced (or replaced) the word timestamps.
    ForcedAligned,
    /// Model-native approximate timestamps were kept (policy did not require
    /// a precise timeline, or validation was not demanded).
    NativeApproximate,
}

/// Inputs that decide whether the shared forced aligner must run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForcedAlignmentDecision {
    pub need_align: bool,
    /// True when alignment is mandatory for Voice ID correctness. Combined
    /// with a missing pack this fails the request closed.
    pub required_for_voice_id: bool,
    /// True when alignment is mandatory for the requested precision / export.
    pub required_for_precision: bool,
    /// Native anchors already validate; the aligner can be skipped even when
    /// precision is requested.
    pub native_reliable: bool,
}

/// Decide whether forced alignment must run for this request.
///
/// V1 never partially splices native and aligned words: when alignment is
/// needed and native anchors are unreliable, the whole transcript is realigned.
pub fn decide_forced_alignment(
    policy: TimelinePrecisionPolicy,
    explicit_refine: bool,
    voice_id_requires_word_alignment: bool,
    needs_subtitle_export: bool,
    native_validation: &WordAnchorValidation,
) -> ForcedAlignmentDecision {
    let native_reliable = native_validation.is_reliable();
    let required_for_voice_id = voice_id_requires_word_alignment;
    let required_for_precision = explicit_refine
        || matches!(policy, TimelinePrecisionPolicy::Always)
        || (matches!(policy, TimelinePrecisionPolicy::Auto) && needs_subtitle_export);

    // Off still yields to Voice ID: without word anchors multi-speaker
    // attribution cannot split faithfully. That is fail-closed at the pack
    // boundary, not a silent degradation.
    let want_precise = required_for_voice_id || required_for_precision;

    let need_align = if explicit_refine {
        // Explicit refine always runs the aligner (user asked for aligned words).
        true
    } else if required_for_voice_id && !native_reliable {
        true
    } else if !want_precise {
        false
    } else {
        // Precise timeline wanted: skip only when native anchors already pass.
        !native_reliable
    };

    ForcedAlignmentDecision {
        need_align,
        required_for_voice_id,
        required_for_precision,
        native_reliable,
    }
}

/// Options for projecting a speaker-attributed transcription into reading +
/// subtitle views.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineProjectOptions {
    /// Quality tag written onto the result.
    pub timeline_quality: TimelineQuality,
    /// When true, clear per-word arrays on reading segments and cues after
    /// projection (cue start/end stay correct). Used when words were only
    /// forced on for internal Voice ID / cue packing and the caller did not
    /// request word timestamps.
    pub strip_words: bool,
    /// Recording length in seconds. Used as the hard end for the last subtitle
    /// cue's CPS display stretch so layout never fabricates time past the audio.
    pub audio_duration_s: Option<f32>,
}

/// Project attributed segments into reading paragraphs + subtitle cues.
///
/// Expects `transcription.segments` to already carry speaker attribution and
/// (when available) word timestamps. Replaces `segments` with reading
/// paragraphs and fills `subtitle_cues`.
pub fn project_transcription(
    mut transcription: Transcription,
    options: TimelineProjectOptions,
) -> Transcription {
    let attributed = std::mem::take(&mut transcription.segments);
    let subtitle_cues = resegment_segments_into_cues(attributed.clone(), options.audio_duration_s);
    let reading = merge_reading_segments(attributed);
    transcription.segments = reading;
    transcription.subtitle_cues = subtitle_cues;
    transcription.timeline_quality = Some(options.timeline_quality);
    if options.strip_words {
        strip_unrequested_word_timestamps(&mut transcription);
    }
    transcription
}

/// Clear per-word arrays while keeping segment/cue start and end intact.
pub fn strip_unrequested_word_timestamps(transcription: &mut Transcription) {
    for segment in &mut transcription.segments {
        segment.words.clear();
    }
    for cue in &mut transcription.subtitle_cues {
        cue.words.clear();
    }
}

/// Timed cues used by SRT/VTT renderers: prefer `subtitle_cues`, fall back to
/// reading `segments` for legacy rows that predate the dual-view projection.
pub fn timed_cues_for_export(transcription: &Transcription) -> &[Segment] {
    if transcription.subtitle_cues.is_empty() {
        &transcription.segments
    } else {
        &transcription.subtitle_cues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::backend::{Segment, WordTimestamp};
    use crate::subtitle::anchors::{WordAnchorQuality, WordAnchorValidation};

    fn reliable() -> WordAnchorValidation {
        WordAnchorValidation {
            quality: WordAnchorQuality::Reliable,
            issues: Vec::new(),
        }
    }

    fn unreliable() -> WordAnchorValidation {
        WordAnchorValidation {
            quality: WordAnchorQuality::Unreliable,
            issues: Vec::new(),
        }
    }

    #[test]
    fn auto_skips_aligner_when_native_reliable_and_no_subtitle_need() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Auto,
            false,
            false,
            false,
            &reliable(),
        );
        assert!(!decision.need_align);
    }

    #[test]
    fn auto_aligns_for_subtitle_export_when_native_unreliable() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Auto,
            false,
            false,
            true,
            &unreliable(),
        );
        assert!(decision.need_align);
        assert!(decision.required_for_precision);
    }

    #[test]
    fn auto_skips_aligner_for_subtitle_when_native_reliable() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Auto,
            false,
            false,
            true,
            &reliable(),
        );
        assert!(!decision.need_align);
        assert!(decision.native_reliable);
    }

    #[test]
    fn always_aligns_only_when_native_unreliable() {
        assert!(
            !decide_forced_alignment(
                TimelinePrecisionPolicy::Always,
                false,
                false,
                false,
                &reliable(),
            )
            .need_align
        );
        assert!(
            decide_forced_alignment(
                TimelinePrecisionPolicy::Always,
                false,
                false,
                false,
                &unreliable(),
            )
            .need_align
        );
    }

    #[test]
    fn off_does_not_align_for_export_alone() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Off,
            false,
            false,
            true,
            &unreliable(),
        );
        assert!(!decision.need_align);
    }

    #[test]
    fn voice_id_forces_align_even_when_policy_off() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Off,
            false,
            true,
            false,
            &unreliable(),
        );
        assert!(decision.need_align);
        assert!(decision.required_for_voice_id);
    }

    #[test]
    fn explicit_refine_always_runs_aligner() {
        let decision = decide_forced_alignment(
            TimelinePrecisionPolicy::Off,
            true,
            false,
            false,
            &reliable(),
        );
        assert!(decision.need_align);
    }

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn attributed_two_speakers() -> Transcription {
        Transcription {
            text: "hello world. other speaker".to_string(),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 2.0,
                    text: "hello world.".to_string(),
                    speaker: Some("SPEAKER_00".to_string()),
                    speaker_label: Some("SPEAKER_00".to_string()),
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![word("hello", 0.0, 0.5), word("world.", 0.6, 1.2)],
                },
                Segment {
                    start: 2.0,
                    end: 3.5,
                    text: "other speaker".to_string(),
                    speaker: Some("SPEAKER_01".to_string()),
                    speaker_label: Some("SPEAKER_01".to_string()),
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![word("other", 2.0, 2.4), word("speaker", 2.5, 3.2)],
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn project_fills_reading_and_cues_with_speaker_hard_boundary() {
        let out = project_transcription(
            attributed_two_speakers(),
            TimelineProjectOptions {
                timeline_quality: TimelineQuality::NativeReliable,
                strip_words: false,
                audio_duration_s: Some(3.5),
            },
        );
        assert_eq!(out.timeline_quality, Some(TimelineQuality::NativeReliable));
        assert!(!out.subtitle_cues.is_empty());
        for cue in &out.subtitle_cues {
            let speaker = cue.speaker.as_deref().unwrap();
            if cue.text.contains("other") {
                assert_eq!(speaker, "SPEAKER_01");
            } else {
                assert_eq!(speaker, "SPEAKER_00");
            }
        }
        // Reading view keeps speaker turns separate when speakers differ.
        assert_eq!(out.segments.len(), 2);
    }

    #[test]
    fn strip_clears_words_but_keeps_cue_times() {
        let out = project_transcription(
            attributed_two_speakers(),
            TimelineProjectOptions {
                timeline_quality: TimelineQuality::ForcedAligned,
                strip_words: true,
                audio_duration_s: Some(3.5),
            },
        );
        assert!(out.segments.iter().all(|s| s.words.is_empty()));
        assert!(out.subtitle_cues.iter().all(|c| c.words.is_empty()));
        assert!(out.subtitle_cues.iter().all(|c| c.end > c.start));
    }
}
