use serde::Serialize;

use crate::api::backend::{Transcription, TranscriptionLongFormMetadata, TruncatedDecode};
use crate::diarize::voice_id::SpeakerNamingRefusal;
use crate::subtitle::TimelineQuality;

#[derive(Serialize)]
pub(super) struct JsonTranscription<'a> {
    text: &'a str,
    segments: Vec<JsonSegment<'a>>,
    /// Short subtitle cues for SRT/VTT and on-screen display. Omitted when
    /// empty (legacy rows and paths that have not projected dual views).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    subtitle_cues: Vec<JsonSegment<'a>>,
    /// Provenance of the word timeline. Omitted on legacy data.
    #[serde(skip_serializing_if = "Option::is_none")]
    timeline_quality: Option<TimelineQuality>,
    /// Decodes behind this transcript that stopped before covering their audio.
    ///
    /// Present in the plain `json` format, not only `verbose_json`: "this text
    /// is not all of the recording" is not a diagnostic detail a caller can be
    /// asked to opt into. Omitted entirely on a complete transcript, so an
    /// existing consumer sees no change until something actually went wrong.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    truncated: Vec<JsonTruncatedDecode>,
    /// Speakers this transcript labels anonymously, with why Voice ID did not
    /// name them.
    ///
    /// Present in the plain `json` format for the same reason `truncated` is:
    /// "this speaker is a number because the recording was too short to judge"
    /// is not a diagnostic detail a caller can be asked to opt into -- without
    /// it a bare `SPEAKER_01` is indistinguishable from Voice ID being broken.
    /// Omitted entirely when every speaker was named (and when nothing was
    /// diarized at all), so an existing consumer sees no change until there is
    /// something to explain.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unnamed_speakers: Vec<JsonUnnamedSpeaker<'a>>,
}

#[derive(Serialize)]
pub(super) struct VerboseJsonTranscription<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// Transcribed duration in seconds (the last segment's end), the OpenAI
    /// verbose_json `duration` field. Omitted when there are no timed
    /// segments rather than fabricating a value.
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f32>,
    text: &'a str,
    segments: Vec<JsonSegment<'a>>,
    /// Short subtitle cues for SRT/VTT. Always present on new dual-view
    /// results; omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    subtitle_cues: Vec<JsonSegment<'a>>,
    /// Provenance of the word timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    timeline_quality: Option<TimelineQuality>,
    /// OpenAI verbose_json top-level `words` array: per-word timing flattened
    /// across all reading segments. Present only when word timestamps were
    /// produced; the per-segment `words` arrays stay for existing clients.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    words: Vec<JsonWord<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    longform: Option<VerboseLongFormMetadata<'a>>,
    /// See [`JsonTranscription::truncated`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    truncated: Vec<JsonTruncatedDecode>,
    /// See [`JsonTranscription::unnamed_speakers`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unnamed_speakers: Vec<JsonUnnamedSpeaker<'a>>,
}

#[derive(Serialize)]
struct JsonTruncatedDecode {
    /// 1-based long-form slice index; absent for a single-pass decode.
    #[serde(skip_serializing_if = "Option::is_none")]
    slice: Option<usize>,
    /// `degenerate-repeat-guard` or `budget-exhausted`.
    reason: &'static str,
    /// Second, within the decode's own audio, up to which the transcript still
    /// describes it. Absent when the family emits no intra-decode timestamps:
    /// there is no honest value, and the clip length would read as "nothing was
    /// lost".
    #[serde(skip_serializing_if = "Option::is_none")]
    covers_up_to_seconds: Option<f32>,
}

fn json_truncated_decodes(transcription: &Transcription) -> Vec<JsonTruncatedDecode> {
    transcription
        .truncated_decodes
        .iter()
        .map(|truncated: &TruncatedDecode| JsonTruncatedDecode {
            slice: truncated.slice_index,
            reason: truncated.truncation.reason.as_str(),
            covers_up_to_seconds: truncated.truncation.transcript_covers_up_to_seconds,
        })
        .collect()
}

/// One anonymous speaker and the reason Voice ID left it that way.
///
/// The numbers behind the verdict are on the wire, not only the verdict, so a
/// client can say "2.0s of the 3.0s needed" instead of a bare "not enough" --
/// and so a client that only knows `reason` can still render something useful
/// for a reason it has never heard of.
#[derive(Serialize)]
struct JsonUnnamedSpeaker<'a> {
    /// The `speaker_label` this transcript's segments carry.
    label: &'a str,
    /// `not-enough-speech`, `mixed-voices`, `no-match-in-library`, or
    /// `embedder-unavailable`.
    reason: &'static str,
    /// Embedding windows / seconds of usable voice behind the label, and the
    /// thresholds they were compared against. Present for the evidence
    /// reasons.
    #[serde(skip_serializing_if = "Option::is_none")]
    windows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required_windows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required_seconds: Option<f64>,
    /// How long one uninterrupted turn has to be to clear both gates. The one
    /// figure fit to show a user: `required_seconds` is the smaller,
    /// non-binding gate, so advice derived from it fails on the retry.
    #[serde(skip_serializing_if = "Option::is_none")]
    required_continuous_seconds: Option<f64>,
    /// Best similarity any enrolled person scored and the floor it had to
    /// clear, plus whether anyone is enrolled at all. Present for
    /// `no-match-in-library`.
    #[serde(skip_serializing_if = "Option::is_none")]
    library_empty: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accept_threshold: Option<f32>,
}

fn json_unnamed_speakers(transcription: &Transcription) -> Vec<JsonUnnamedSpeaker<'_>> {
    transcription
        .unnamed_speakers
        .iter()
        .map(|speaker| {
            let mut json = JsonUnnamedSpeaker {
                label: &speaker.label,
                reason: speaker.reason.kind(),
                windows: None,
                required_windows: None,
                seconds: None,
                required_seconds: None,
                required_continuous_seconds: None,
                library_empty: None,
                best_score: None,
                accept_threshold: None,
            };
            match &speaker.reason {
                SpeakerNamingRefusal::EmbedderUnavailable => {}
                SpeakerNamingRefusal::NotEnoughSpeech {
                    windows,
                    required_windows,
                    seconds,
                    required_seconds,
                    required_continuous_seconds,
                } => {
                    json.windows = Some(*windows);
                    json.required_windows = Some(*required_windows);
                    json.seconds = Some(*seconds);
                    json.required_seconds = Some(*required_seconds);
                    json.required_continuous_seconds = Some(*required_continuous_seconds);
                }
                SpeakerNamingRefusal::MixedVoices { windows, seconds } => {
                    json.windows = Some(*windows);
                    json.seconds = Some(*seconds);
                }
                SpeakerNamingRefusal::NoMatchInLibrary {
                    library_empty,
                    best_score,
                    accept_threshold,
                } => {
                    json.library_empty = Some(*library_empty);
                    json.best_score = *best_score;
                    json.accept_threshold = *accept_threshold;
                }
            }
            json
        })
        .collect()
}

#[derive(Serialize)]
struct JsonSegment<'a> {
    /// Zero-based segment index, the OpenAI verbose_json segment `id`. Only
    /// verbose_json sets it; the plain `json` format stays unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<usize>,
    start: f32,
    end: f32,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_label: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_person_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_snapshot_label: Option<&'a str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    words: Vec<JsonWord<'a>>,
}

#[derive(Serialize)]
struct JsonWord<'a> {
    word: &'a str,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
}

#[derive(Serialize)]
struct VerboseLongFormMetadata<'a> {
    chunk_count: usize,
    skipped_silent_chunks: usize,
    duplicate_merge_count: usize,
    provenance: &'a [String],
}

fn json_segments(transcription: &Transcription, with_ids: bool) -> Vec<JsonSegment<'_>> {
    map_json_segments(&transcription.segments, with_ids)
}

fn json_subtitle_cues(transcription: &Transcription, with_ids: bool) -> Vec<JsonSegment<'_>> {
    map_json_segments(&transcription.subtitle_cues, with_ids)
}

fn map_json_segments(
    segments: &[crate::api::backend::Segment],
    with_ids: bool,
) -> Vec<JsonSegment<'_>> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| JsonSegment {
            id: with_ids.then_some(index),
            start: segment.start,
            end: segment.end,
            text: &segment.text,
            speaker: segment.speaker.as_deref(),
            speaker_label: segment.speaker_label.as_deref(),
            speaker_person_id: segment.speaker_person_id.as_deref(),
            speaker_snapshot_label: segment.speaker_snapshot_label.as_deref(),
            words: segment
                .words
                .iter()
                .map(|word| JsonWord {
                    word: &word.word,
                    start: word.start,
                    end: word.end,
                    confidence: word.confidence,
                })
                .collect(),
        })
        .collect()
}

fn transcribed_duration_seconds(transcription: &Transcription) -> Option<f32> {
    transcription
        .segments
        .iter()
        .map(|segment| segment.end)
        .filter(|end| end.is_finite() && *end >= 0.0)
        .max_by(|left, right| left.total_cmp(right))
}

fn flattened_words(transcription: &Transcription) -> Vec<JsonWord<'_>> {
    transcription
        .segments
        .iter()
        .flat_map(|segment| segment.words.iter())
        .map(|word| JsonWord {
            word: &word.word,
            start: word.start,
            end: word.end,
            confidence: word.confidence,
        })
        .collect()
}

impl<'a> From<&'a Transcription> for JsonTranscription<'a> {
    fn from(transcription: &'a Transcription) -> Self {
        Self {
            text: &transcription.text,
            segments: json_segments(transcription, false),
            subtitle_cues: json_subtitle_cues(transcription, false),
            timeline_quality: transcription.timeline_quality,
            truncated: json_truncated_decodes(transcription),
            unnamed_speakers: json_unnamed_speakers(transcription),
        }
    }
}

impl<'a> From<&'a Transcription> for VerboseJsonTranscription<'a> {
    fn from(transcription: &'a Transcription) -> Self {
        Self {
            language: transcription
                .language
                .as_deref()
                .map(crate::models::language::code_to_english_name),
            duration: transcribed_duration_seconds(transcription),
            text: &transcription.text,
            segments: json_segments(transcription, true),
            subtitle_cues: json_subtitle_cues(transcription, true),
            timeline_quality: transcription.timeline_quality,
            words: flattened_words(transcription),
            longform: transcription
                .longform
                .as_ref()
                .map(verbose_longform_metadata),
            truncated: json_truncated_decodes(transcription),
            unnamed_speakers: json_unnamed_speakers(transcription),
        }
    }
}

fn verbose_longform_metadata(
    metadata: &TranscriptionLongFormMetadata,
) -> VerboseLongFormMetadata<'_> {
    VerboseLongFormMetadata {
        chunk_count: metadata.chunk_count,
        skipped_silent_chunks: metadata.skipped_silent_chunks,
        duplicate_merge_count: metadata.duplicate_merge_count,
        provenance: metadata.provenance.as_slice(),
    }
}
