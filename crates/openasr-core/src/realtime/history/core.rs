use std::collections::HashMap;

use serde::Serialize;
use thiserror::Error;

#[path = "../history_export.rs"]
mod history_export;
#[path = "../history_sort.rs"]
mod history_sort;
#[path = "../history_state.rs"]
mod history_state;
#[path = "../history_subtitle.rs"]
mod history_subtitle;
#[path = "../history_text.rs"]
mod history_text;

use super::{
    TranscriptSegmentId, TranscriptUtteranceId,
    events::{
        RealtimeEventEnvelope, RealtimeTranscriptEvent, RealtimeTranscriptFinal,
        RealtimeTranscriptRevision,
    },
};
use history_state::{TranscriptUpdateInput, UpdateReason};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RealtimeHistoryEntry {
    pub utterance_id: String,
    pub segment_id: String,
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub revised: bool,
    pub revision_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_seq: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RealtimeHistoryRevision {
    pub utterance_id: String,
    pub segment_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_revision: Option<u64>,
    pub to_revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_text: Option<String>,
    pub to_text: String,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revises_event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_seq: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealtimeHistoryApplyResult {
    Applied,
    IgnoredDuplicateOrOutOfOrder {
        current_revision: u64,
        incoming_revision: u64,
    },
    IgnoredNoChange {
        current_revision: u64,
    },
    IgnoredNonTranscriptEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeExportFormat {
    Text,
    Json,
    Markdown,
    Srt,
    Vtt,
}

impl RealtimeExportFormat {
    pub fn from_extension(path: &std::path::Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "txt" => Some(Self::Text),
            "json" => Some(Self::Json),
            "md" | "markdown" => Some(Self::Markdown),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimePostProcessor {
    pub trim_whitespace: bool,
    pub collapse_internal_whitespace: bool,
    pub join_segments: bool,
    pub suggest_title: bool,
}

impl Default for RealtimePostProcessor {
    fn default() -> Self {
        Self {
            trim_whitespace: true,
            collapse_internal_whitespace: true,
            join_segments: false,
            suggest_title: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimePostProcessOutput {
    pub lines: Vec<String>,
    pub joined_text: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RealtimeTranscriptHistory {
    entries: Vec<RealtimeHistoryEntry>,
    revisions: Vec<RealtimeHistoryRevision>,
    index: HashMap<(String, String), usize>,
}

#[derive(Debug, Error)]
pub enum RealtimeHistoryExportError {
    #[error(
        "Unsupported live export extension for path '{path}'. Use one of: .txt, .json, .md, .srt, .vtt."
    )]
    UnsupportedExtension { path: String },
    #[error("Could not serialize live transcript history JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "Cannot export subtitle format because line {index} has non-increasing timestamps ({start_ms}ms..{end_ms}ms)."
    )]
    InvalidSubtitleTiming {
        index: usize,
        start_ms: u64,
        end_ms: u64,
    },
}

impl Default for RealtimeTranscriptHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeTranscriptHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            revisions: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn entries(&self) -> &[RealtimeHistoryEntry] {
        &self.entries
    }

    pub fn revisions(&self) -> &[RealtimeHistoryRevision] {
        &self.revisions
    }

    pub fn apply_envelope(
        &mut self,
        envelope: &RealtimeEventEnvelope,
    ) -> RealtimeHistoryApplyResult {
        match &envelope.event {
            super::events::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => {
                self.apply_transcript_update(
                    TranscriptEventRef::Final(event),
                    Some(envelope.event_id.0.clone()),
                    Some(envelope.seq),
                )
            }
            super::events::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(event)) => {
                self.apply_transcript_update(
                    TranscriptEventRef::Revision(event),
                    Some(envelope.event_id.0.clone()),
                    Some(envelope.seq),
                )
            }
            _ => RealtimeHistoryApplyResult::IgnoredNonTranscriptEvent,
        }
    }

    pub fn apply_final(
        &mut self,
        event: &RealtimeTranscriptFinal,
        source_event_id: Option<String>,
        source_seq: Option<u64>,
    ) -> RealtimeHistoryApplyResult {
        self.apply_transcript_update(TranscriptEventRef::Final(event), source_event_id, source_seq)
    }

    pub fn apply_revision(
        &mut self,
        event: &RealtimeTranscriptRevision,
        source_event_id: Option<String>,
        source_seq: Option<u64>,
    ) -> RealtimeHistoryApplyResult {
        self.apply_transcript_update(
            TranscriptEventRef::Revision(event),
            source_event_id,
            source_seq,
        )
    }

    fn apply_transcript_update(
        &mut self,
        event: TranscriptEventRef<'_>,
        source_event_id: Option<String>,
        source_seq: Option<u64>,
    ) -> RealtimeHistoryApplyResult {
        self.apply_update(event.to_update_input(source_event_id, source_seq))
    }

    pub fn post_process(&self, options: &RealtimePostProcessor) -> RealtimePostProcessOutput {
        let sorted_entries = history_sort::sorted_entries(&self.entries);
        let lines = sorted_entries
            .iter()
            .map(|entry| history_text::normalize_whitespace(&entry.text, options))
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        let joined_text = if options.join_segments {
            lines.join(" ")
        } else {
            lines.join("\n")
        };
        let title = if options.suggest_title {
            history_text::suggest_title_from_text(&joined_text)
        } else {
            None
        };
        RealtimePostProcessOutput {
            lines,
            joined_text,
            title,
        }
    }

    pub fn export(
        &self,
        format: RealtimeExportFormat,
        options: &RealtimePostProcessor,
    ) -> Result<String, RealtimeHistoryExportError> {
        let processed = self.post_process(options);
        history_export::export_rendered(format, &self.entries, &self.revisions, processed, options)
    }

    pub fn export_by_path(
        &self,
        path: &std::path::Path,
        options: &RealtimePostProcessor,
    ) -> Result<(RealtimeExportFormat, String), RealtimeHistoryExportError> {
        let format = RealtimeExportFormat::from_extension(path).ok_or_else(|| {
            RealtimeHistoryExportError::UnsupportedExtension {
                path: path.display().to_string(),
            }
        })?;
        let rendered = self.export(format, options)?;
        Ok((format, rendered))
    }

    fn apply_update(&mut self, update: TranscriptUpdateInput) -> RealtimeHistoryApplyResult {
        history_state::apply_transcript_update(
            &mut self.entries,
            &mut self.revisions,
            &mut self.index,
            update,
        )
    }
}

#[derive(Clone, Copy)]
enum TranscriptEventRef<'a> {
    Final(&'a RealtimeTranscriptFinal),
    Revision(&'a RealtimeTranscriptRevision),
}

impl TranscriptEventRef<'_> {
    fn reason(self) -> UpdateReason {
        match self {
            Self::Final(_) => UpdateReason::Final {
                revises_event_id: None,
            },
            Self::Revision(event) => UpdateReason::Revision {
                revises_event_id: event.revises_event_id.as_ref().map(|value| value.0.clone()),
                reason: event.reason.clone(),
            },
        }
    }

    fn to_update_input(
        self,
        source_event_id: Option<String>,
        source_seq: Option<u64>,
    ) -> TranscriptUpdateInput {
        match self {
            Self::Final(event) => TranscriptUpdateInput {
                utterance_id: event.utterance_id.clone(),
                segment_id: event.segment_id.clone(),
                revision: event.revision,
                text: event.text.clone(),
                start_ms: event.start_ms,
                end_ms: event.end_ms,
                language: event.language.clone(),
                reason: self.reason(),
                source_event_id,
                source_seq,
            },
            Self::Revision(event) => TranscriptUpdateInput {
                utterance_id: event.utterance_id.clone(),
                segment_id: event.segment_id.clone(),
                revision: event.revision,
                text: event.text.clone(),
                start_ms: event.start_ms,
                end_ms: event.end_ms,
                language: event.language.clone(),
                reason: self.reason(),
                source_event_id,
                source_seq,
            },
        }
    }
}
