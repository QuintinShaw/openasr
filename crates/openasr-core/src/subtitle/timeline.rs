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
/// - [`Always`](Self::Always): run forced alignment during transcription,
///   even when native anchors already validate.
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
    /// a precise timeline, validation was not demanded, or in-process
    /// alignment gates failed — see `Transcription.timeline_degraded_reason`).
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
    /// Native anchors already validate; Auto can skip the aligner. Always and
    /// explicit refine still request alignment regardless of this diagnostic.
    pub native_reliable: bool,
}

/// Decide whether forced alignment must run for this request.
///
/// V1 never partially splices native and aligned words: when alignment is
/// needed, the whole transcript is realigned. Always and explicit refine request
/// alignment independently of native reliability; failure handling is unchanged.
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

    let need_align = if explicit_refine || matches!(policy, TimelinePrecisionPolicy::Always) {
        // An explicit alignment request takes precedence over native reliability.
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
    transcription.subtitle_cues = sanitize_export_cues(&subtitle_cues);
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
/// Zero-length and overlapping cues are dropped or clamped so exporters never
/// write illegal timings even if a projection leaked one.
pub fn timed_cues_for_export(transcription: &Transcription) -> Vec<Segment> {
    let source = if transcription.subtitle_cues.is_empty() {
        &transcription.segments
    } else {
        &transcription.subtitle_cues
    };
    sanitize_export_cues(source)
}

fn sanitize_export_cues(cues: &[Segment]) -> Vec<Segment> {
    let mut previous_end = f32::NEG_INFINITY;
    let mut out = Vec::with_capacity(cues.len());
    for cue in cues {
        let mut cue = cue.clone();
        if cue.end <= cue.start {
            continue;
        }
        if cue.start < previous_end {
            cue.start = previous_end;
        }
        if cue.end <= cue.start {
            continue;
        }
        previous_end = cue.end;
        out.push(cue);
    }
    out
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
    fn alignment_policy_decision_matrix() {
        for policy in [
            TimelinePrecisionPolicy::Auto,
            TimelinePrecisionPolicy::Always,
            TimelinePrecisionPolicy::Off,
        ] {
            for validation in [reliable(), unreliable()] {
                for subtitle_export in [false, true] {
                    for voice_id in [false, true] {
                        for explicit_refine in [false, true] {
                            let precision = explicit_refine
                                || policy == TimelinePrecisionPolicy::Always
                                || (policy == TimelinePrecisionPolicy::Auto && subtitle_export);
                            let expected = ForcedAlignmentDecision {
                                need_align: explicit_refine
                                    || policy == TimelinePrecisionPolicy::Always
                                    || ((voice_id || precision) && !validation.is_reliable()),
                                required_for_voice_id: voice_id,
                                required_for_precision: precision,
                                native_reliable: validation.is_reliable(),
                            };
                            assert_eq!(
                                decide_forced_alignment(
                                    policy,
                                    explicit_refine,
                                    voice_id,
                                    subtitle_export,
                                    &validation,
                                ),
                                expected,
                                "{policy:?}, {validation:?}, export={subtitle_export}, \
                                 voice_id={voice_id}, refine={explicit_refine}"
                            );
                        }
                    }
                }
            }
        }
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

    #[test]
    fn point_word_projection_preserves_text_identity_and_bounded_export_times() {
        let mut input = attributed_two_speakers();
        input.segments[0].text = "We go. Next".to_string();
        input.segments[0].start = 0.0;
        input.segments[0].end = 1.2395;
        input.segments[0].words = vec![
            word("We", 0.0, 0.0395),
            word("go", 0.2, 0.2395),
            word(".", 0.8685, 0.908),
            word("Next", 1.2, 1.2395),
        ];
        input.segments[0].speaker_person_id = Some("person-0".to_string());
        input.segments[0].speaker_snapshot_label = Some("Alice".to_string());
        // This unsplittable burst needs more display time than the tight audio
        // bound permits; a larger known recording end can accommodate it.
        let dense = "abcdefghijabcdefghijabcdefghij";
        input.segments[1].text = dense.to_string();
        input.segments[1].start = 1.4;
        input.segments[1].end = 1.4395;
        input.segments[1].words = vec![word(dense, 1.4, 1.4395)];
        input.text = format!("We go. Next {dense}");
        let original_words = input
            .segments
            .iter()
            .flat_map(|segment| segment.words.clone())
            .collect::<Vec<_>>();

        for audio_duration_s in [None, Some(1.5), Some(4.0)] {
            let out = project_transcription(
                input.clone(),
                TimelineProjectOptions {
                    timeline_quality: TimelineQuality::NativeApproximate,
                    strip_words: false,
                    audio_duration_s,
                },
            );
            assert_eq!(out.text, input.text);
            assert_eq!(out.segments, input.segments);
            assert_eq!(
                out.timeline_quality,
                Some(TimelineQuality::NativeApproximate)
            );
            let export = timed_cues_for_export(&out);
            assert_eq!(export, out.subtitle_cues);
            assert_eq!(export[0].text, "We go.");
            assert_eq!(export[1].text, "Next");
            assert_eq!(
                export
                    .iter()
                    .flat_map(|cue| cue.words.clone())
                    .collect::<Vec<_>>(),
                original_words
            );
            for cue in &export {
                assert!(cue.start >= 0.0 && cue.end > cue.start);
                if let Some(duration) = audio_duration_s {
                    assert!(cue.end <= duration);
                }
                let attributed = input
                    .segments
                    .iter()
                    .find(|segment| segment.speaker == cue.speaker)
                    .unwrap();
                assert_eq!(cue.speaker_label, attributed.speaker_label);
                assert_eq!(cue.speaker_person_id, attributed.speaker_person_id);
                assert_eq!(
                    cue.speaker_snapshot_label,
                    attributed.speaker_snapshot_label
                );
            }
            assert!(export.windows(2).all(|pair| pair[0].end <= pair[1].start));
            let last = export.last().unwrap();
            if audio_duration_s == Some(4.0) {
                assert!(30.0 / (last.end - last.start) <= 21.0 + 1e-3);
            } else {
                assert_eq!(last.end, 1.4395);
                assert!(30.0 / (last.end - last.start) > 21.0);
            }
        }
    }

    #[test]
    fn timeline_quality_wire_values_stay_desktop_compatible() {
        assert_eq!(
            serde_json::to_string(&TimelineQuality::NativeReliable).unwrap(),
            "\"native_reliable\""
        );
        assert_eq!(
            serde_json::to_string(&TimelineQuality::ForcedAligned).unwrap(),
            "\"forced_aligned\""
        );
        assert_eq!(
            serde_json::to_string(&TimelineQuality::NativeApproximate).unwrap(),
            "\"native_approximate\""
        );
    }
}
