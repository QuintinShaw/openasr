#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::{
        RealtimeEvent, RealtimeEventEnvelope, RealtimeEventId, RealtimeSessionId,
        RealtimeTranscriptEvent, RealtimeTranscriptFinal, RealtimeTranscriptRevision,
    };

    fn base_final(revision: u64, text: &str) -> RealtimeTranscriptFinal {
        RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision,
            text: text.to_string(),
            start_ms: 0,
            end_ms: 320,
            is_final: true,
            words: Vec::new(),
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        }
    }

    #[test]
    fn appends_history_from_transcript_final() {
        let mut history = RealtimeTranscriptHistory::new();
        let result =
            history.apply_final(&base_final(1, "hello"), Some("evt_1".to_string()), Some(3));
        assert_eq!(result, RealtimeHistoryApplyResult::Applied);
        assert_eq!(history.entries().len(), 1);
        assert_eq!(history.entries()[0].text, "hello");
        assert!(!history.entries()[0].revised);
    }

    #[test]
    fn applies_revision_and_marks_entry_revised() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(&base_final(1, "helo"), Some("evt_1".to_string()), Some(3));
        let revision = RealtimeTranscriptRevision {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revises_event_id: Some(RealtimeEventId("evt_1".to_string())),
            revision: 2,
            text: "hello".to_string(),
            start_ms: 0,
            end_ms: 340,
            is_final: true,
            reason: "post_final_correction".to_string(),
            words: Vec::new(),
            language: None,
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        };
        let result = history.apply_revision(&revision, Some("evt_2".to_string()), Some(4));
        assert_eq!(result, RealtimeHistoryApplyResult::Applied);
        assert_eq!(history.entries()[0].text, "hello");
        assert_eq!(history.entries()[0].revision, 2);
        assert!(history.entries()[0].revised);
        assert_eq!(history.entries()[0].revision_count, 1);
        assert_eq!(history.revisions().len(), 1);
        assert_eq!(history.revisions()[0].reason, "post_final_correction");
    }

    #[test]
    fn ignores_duplicate_or_lower_revision_deterministically() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(&base_final(2, "hello"), None, None);
        let duplicate = history.apply_final(&base_final(2, "hello again"), None, None);
        assert_eq!(
            duplicate,
            RealtimeHistoryApplyResult::IgnoredDuplicateOrOutOfOrder {
                current_revision: 2,
                incoming_revision: 2
            }
        );
        let lower = history.apply_final(&base_final(1, "h"), None, None);
        assert_eq!(
            lower,
            RealtimeHistoryApplyResult::IgnoredDuplicateOrOutOfOrder {
                current_revision: 2,
                incoming_revision: 1
            }
        );
    }

    #[test]
    fn ignores_no_change_revision_without_mutating_entry_metadata() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(&base_final(1, "hello"), None, Some(10));
        let result = history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 2,
                text: "hello".to_string(),
                start_ms: 100,
                end_ms: 500,
                is_final: true,
                words: Vec::new(),
                language: Some("fr".to_string()),
                speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            },
            Some("evt_2".to_string()),
            Some(11),
        );
        assert_eq!(
            result,
            RealtimeHistoryApplyResult::IgnoredNoChange {
                current_revision: 1
            }
        );
        assert_eq!(history.entries()[0].revision, 1);
        assert_eq!(history.entries()[0].start_ms, 0);
        assert_eq!(history.entries()[0].end_ms, 320);
        assert_eq!(history.entries()[0].language.as_deref(), Some("en"));
        assert_eq!(history.entries()[0].source_seq, Some(10));
        assert!(history.revisions().is_empty());
    }

    #[test]
    fn exports_text_json_markdown() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(&base_final(1, " hello   world "), None, None);
        let post = RealtimePostProcessor {
            trim_whitespace: true,
            collapse_internal_whitespace: true,
            join_segments: false,
            suggest_title: true,
        };
        let text = history.export(RealtimeExportFormat::Text, &post).unwrap();
        assert_eq!(text, "hello world\n");
        let json = history.export(RealtimeExportFormat::Json, &post).unwrap();
        assert!(json.contains("\"entries\""));
        assert!(json.contains("\"hello world\""));
        assert!(!json.contains(" hello   world "));
        let markdown = history
            .export(RealtimeExportFormat::Markdown, &post)
            .unwrap();
        assert!(markdown.starts_with("# hello world\n\n"));
    }

    #[test]
    fn exports_follow_timestamp_order_not_arrival_order() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_2".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "second".to_string(),
                start_ms: 500,
                end_ms: 800,
                is_final: true,
                words: Vec::new(),
                language: None,
                speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            },
            Some("evt_2".to_string()),
            Some(2),
        );
        history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "first".to_string(),
                start_ms: 0,
                end_ms: 300,
                is_final: true,
                words: Vec::new(),
                language: None,
                speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        let text = history
            .export(
                RealtimeExportFormat::Text,
                &RealtimePostProcessor::default(),
            )
            .unwrap();
        assert_eq!(text, "first\nsecond\n");
    }

    #[test]
    fn post_processing_cleanup_join_and_title_are_conservative() {
        let mut history = RealtimeTranscriptHistory::new();
        history.apply_final(&base_final(1, "  hello    world  "), None, None);
        history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_2".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "from   openasr".to_string(),
                start_ms: 330,
                end_ms: 660,
                is_final: true,
                words: Vec::new(),
                language: None,
                speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            },
            None,
            None,
        );
        let post = RealtimePostProcessor {
            join_segments: true,
            suggest_title: true,
            ..RealtimePostProcessor::default()
        };
        let output = history.post_process(&post);
        assert_eq!(output.lines, vec!["hello world", "from openasr"]);
        assert_eq!(output.joined_text, "hello world from openasr");
        assert_eq!(output.title.as_deref(), Some("hello world from openasr"));
    }

    #[test]
    fn envelope_apply_uses_transcript_events_only() {
        let mut history = RealtimeTranscriptHistory::new();
        let envelope = RealtimeEventEnvelope {
            event_type: "session.created",
            session_id: RealtimeSessionId("rt_1".to_string()),
            event_id: RealtimeEventId("evt_1".to_string()),
            seq: 1,
            created_at: "2026-05-10T00:00:00.000Z".to_string(),
            trace_id: None,
            request_id: None,
            event: RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(base_final(
                1, "hello",
            ))),
        };
        assert_eq!(
            history.apply_envelope(&envelope),
            RealtimeHistoryApplyResult::Applied
        );
        assert_eq!(
            history.entries()[0].source_event_id.as_deref(),
            Some("evt_1")
        );
        assert_eq!(history.entries()[0].source_seq, Some(1));
    }
}
