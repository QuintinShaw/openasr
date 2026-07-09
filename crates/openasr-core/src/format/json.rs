use serde::Serialize;

use crate::api::backend::{Transcription, TranscriptionLongFormMetadata};

#[derive(Serialize)]
pub(super) struct JsonTranscription<'a> {
    text: &'a str,
    segments: Vec<JsonSegment<'a>>,
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
    /// OpenAI verbose_json top-level `words` array: per-word timing flattened
    /// across all segments. Present only when word timestamps were produced;
    /// the per-segment `words` arrays stay for existing clients.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    words: Vec<JsonWord<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    longform: Option<VerboseLongFormMetadata<'a>>,
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
    speaker_profile_id: Option<&'a str>,
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
    transcription
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| JsonSegment {
            id: with_ids.then_some(index),
            start: segment.start,
            end: segment.end,
            text: &segment.text,
            speaker: segment.speaker.as_deref(),
            speaker_label: segment.speaker_label.as_deref(),
            speaker_profile_id: segment.speaker_profile_id.as_deref(),
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
            words: flattened_words(transcription),
            longform: transcription
                .longform
                .as_ref()
                .map(verbose_longform_metadata),
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
