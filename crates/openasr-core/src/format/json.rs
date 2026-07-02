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
    text: &'a str,
    segments: Vec<JsonSegment<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    longform: Option<VerboseLongFormMetadata<'a>>,
}

#[derive(Serialize)]
struct JsonSegment<'a> {
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

impl<'a> From<&'a Transcription> for JsonTranscription<'a> {
    fn from(transcription: &'a Transcription) -> Self {
        Self {
            text: &transcription.text,
            segments: transcription
                .segments
                .iter()
                .map(|segment| JsonSegment {
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
                .collect(),
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
            text: &transcription.text,
            segments: transcription
                .segments
                .iter()
                .map(|segment| JsonSegment {
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
                .collect(),
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
