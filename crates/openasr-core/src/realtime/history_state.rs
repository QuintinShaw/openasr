use std::collections::HashMap;

use super::{
    RealtimeHistoryApplyResult, RealtimeHistoryEntry, RealtimeHistoryRevision, TranscriptSegmentId,
    TranscriptUtteranceId,
};

#[derive(Debug, Clone)]
pub(super) enum UpdateReason {
    Final {
        revises_event_id: Option<String>,
    },
    Revision {
        revises_event_id: Option<String>,
        reason: String,
    },
}

impl UpdateReason {
    pub(super) fn export_reason(&self) -> String {
        match self {
            Self::Final { .. } => "transcript.final".to_string(),
            Self::Revision { reason, .. } => reason.clone(),
        }
    }

    pub(super) fn revises_event_id(&self) -> Option<String> {
        match self {
            Self::Final { revises_event_id }
            | Self::Revision {
                revises_event_id, ..
            } => revises_event_id.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TranscriptUpdateInput {
    pub(super) utterance_id: TranscriptUtteranceId,
    pub(super) segment_id: TranscriptSegmentId,
    pub(super) revision: u64,
    pub(super) text: String,
    pub(super) start_ms: u64,
    pub(super) end_ms: u64,
    pub(super) language: Option<String>,
    pub(super) reason: UpdateReason,
    pub(super) source_event_id: Option<String>,
    pub(super) source_seq: Option<u64>,
}

pub(super) fn apply_transcript_update(
    entries: &mut Vec<RealtimeHistoryEntry>,
    revisions: &mut Vec<RealtimeHistoryRevision>,
    index: &mut HashMap<(String, String), usize>,
    input: TranscriptUpdateInput,
) -> RealtimeHistoryApplyResult {
    let TranscriptUpdateInput {
        utterance_id,
        segment_id,
        revision,
        text,
        start_ms,
        end_ms,
        language,
        reason,
        source_event_id,
        source_seq,
    } = input;
    let key = (utterance_id.0.clone(), segment_id.0.clone());
    if let Some(entry_index) = index.get(&key).copied() {
        let current = &entries[entry_index];
        if revision <= current.revision {
            return RealtimeHistoryApplyResult::IgnoredDuplicateOrOutOfOrder {
                current_revision: current.revision,
                incoming_revision: revision,
            };
        }
        if text == current.text {
            return RealtimeHistoryApplyResult::IgnoredNoChange {
                current_revision: current.revision,
            };
        }

        let previous_revision = current.revision;
        let previous_text = current.text.clone();
        let next_language = language.or_else(|| current.language.clone());
        let next_revision_count = current.revision_count.saturating_add(1);

        entries[entry_index] = RealtimeHistoryEntry {
            utterance_id: utterance_id.0.clone(),
            segment_id: segment_id.0.clone(),
            revision,
            text: text.clone(),
            start_ms,
            end_ms,
            language: next_language,
            revised: true,
            revision_count: next_revision_count,
            source_event_id: source_event_id.clone(),
            source_seq,
        };
        revisions.push(RealtimeHistoryRevision {
            utterance_id: utterance_id.0,
            segment_id: segment_id.0,
            from_revision: Some(previous_revision),
            to_revision: revision,
            from_text: Some(previous_text),
            to_text: text,
            reason: reason.export_reason(),
            revises_event_id: reason.revises_event_id(),
            source_event_id,
            source_seq,
        });
        return RealtimeHistoryApplyResult::Applied;
    }

    let revised = matches!(reason, UpdateReason::Revision { .. });
    let revision_count = if revised { 1 } else { 0 };
    index.insert(key, entries.len());
    entries.push(RealtimeHistoryEntry {
        utterance_id: utterance_id.0.clone(),
        segment_id: segment_id.0.clone(),
        revision,
        text: text.clone(),
        start_ms,
        end_ms,
        language,
        revised,
        revision_count,
        source_event_id: source_event_id.clone(),
        source_seq,
    });

    if revised {
        revisions.push(RealtimeHistoryRevision {
            utterance_id: utterance_id.0,
            segment_id: segment_id.0,
            from_revision: None,
            to_revision: revision,
            from_text: None,
            to_text: text,
            reason: reason.export_reason(),
            revises_event_id: reason.revises_event_id(),
            source_event_id,
            source_seq,
        });
    }

    RealtimeHistoryApplyResult::Applied
}
