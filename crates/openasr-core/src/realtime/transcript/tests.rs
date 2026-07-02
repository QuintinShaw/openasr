#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_partial_final_lifecycle_advances_revisions() {
        let mut lifecycle = TranscriptLifecycle::default();
        let first =
            lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 1, "hel", 0, 100));
        assert!(matches!(
            first,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(_))
        ));

        let second =
            lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 2, "hello", 0, 200));
        assert!(matches!(
            second,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(
                RealtimeTranscriptPartial { revision: 2, .. }
            ))
        ));

        let final_event = lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 3, "hello world", 0, 300)
                .with_language(Some("en".to_string())),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert!(matches!(
            final_event,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(
                RealtimeTranscriptFinal {
                    revision: 3,
                    is_final: true,
                    language: Some(_),
                    ..
                }
            ))
        ));
    }

    #[test]
    fn final_can_stabilize_same_revision_partial() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 1, "hello", 0, 200));
        let result = lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert!(matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(
                RealtimeTranscriptFinal {
                    revision: 1,
                    is_final: true,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn final_can_correct_same_revision_partial_text() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 1, "hel", 0, 200));
        let result = lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert!(matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(
                RealtimeTranscriptFinal {
                    revision: 1,
                    text,
                    is_final: true,
                    ..
                }
            )) if text == "hello"
        ));
    }

    #[test]
    fn post_final_text_change_emits_revision() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello world", 0, 300),
            Some(RealtimeEventId("evt_final".to_string())),
        );

        let result = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_1",
            "seg_1",
            2,
            "hello, world",
            0,
            300,
        ));
        assert!(matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    revision: 2,
                    is_final: true,
                    revises_event_id: Some(_),
                    reason,
                    ..
                }
            )) if reason == TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION
        ));
    }

    #[test]
    fn lower_revision_is_ignored() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 3, "hello", 0, 200));
        assert_eq!(
            lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 2, "hel", 0, 100)),
            TranscriptLifecycleResult::IgnoredOutOfOrder {
                current_revision: 3,
                incoming_revision: 2
            }
        );
    }

    #[test]
    fn separate_utterances_do_not_interfere() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 4, "first", 0, 200));
        let result =
            lifecycle.apply_partial(TranscriptUpdate::new("utt_2", "seg_1", 1, "second", 0, 200));
        assert!(matches!(
            result,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(
                RealtimeTranscriptPartial {
                    utterance_id,
                    revision: 1,
                    ..
                }
            )) if utterance_id == TranscriptUtteranceId("utt_2".to_string())
        ));
    }

    #[test]
    fn final_events_stay_stable_without_explicit_revision() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "stable text", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert_eq!(
            lifecycle.apply_partial(TranscriptUpdate::new(
                "utt_1",
                "seg_1",
                2,
                "stable text",
                0,
                200
            )),
            TranscriptLifecycleResult::IgnoredNoChange {
                current_revision: 1
            }
        );
    }

    #[test]
    fn duplicate_partial_revision_is_ignored() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 2, "hello", 0, 200));
        assert_eq!(
            lifecycle.apply_partial(TranscriptUpdate::new("utt_1", "seg_1", 2, "hello", 0, 200)),
            TranscriptLifecycleResult::IgnoredOutOfOrder {
                current_revision: 2,
                incoming_revision: 2
            }
        );
    }

    #[test]
    fn post_final_same_text_is_ignored_even_with_higher_revision() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "stable text", 0, 200)
                .with_language(Some("en".to_string())),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert_eq!(
            lifecycle.apply_partial(
                TranscriptUpdate::new("utt_1", "seg_1", 2, "stable text", 0, 200)
                    .with_language(Some("fr".to_string()))
            ),
            TranscriptLifecycleResult::IgnoredNoChange {
                current_revision: 1
            }
        );
    }

    #[test]
    fn language_is_preserved_when_update_omits_it() {
        let mut lifecycle = TranscriptLifecycle::default();
        let partial = lifecycle.apply_partial(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello", 0, 200)
                .with_language(Some("en".to_string())),
        );
        assert!(matches!(
            partial,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Partial(
                RealtimeTranscriptPartial {
                    language: Some(language),
                    ..
                }
            )) if language == "en"
        ));

        let final_event = lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );
        assert!(matches!(
            final_event,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Final(
                RealtimeTranscriptFinal {
                    language: Some(language),
                    ..
                }
            )) if language == "en"
        ));

        let revision = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_1",
            "seg_1",
            2,
            "hello world",
            0,
            220,
        ));
        assert!(matches!(
            revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    language: Some(language),
                    ..
                }
            )) if language == "en"
        ));
    }

    #[test]
    fn language_can_be_updated_on_post_final_revision() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello world", 0, 200)
                .with_language(Some("en".to_string())),
            Some(RealtimeEventId("evt_final".to_string())),
        );

        let revision = lifecycle.apply_partial(
            TranscriptUpdate::new("utt_1", "seg_1", 2, "bonjour le monde", 0, 200)
                .with_language(Some("fr".to_string())),
        );
        assert!(matches!(
            revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    language: Some(language),
                    ..
                }
            )) if language == "fr"
        ));
    }

    #[test]
    fn post_final_revision_defaults_to_recorded_final_event_id() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello world", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );

        let revision = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_1",
            "seg_1",
            2,
            "hello, world",
            0,
            200,
        ));
        assert!(matches!(
            revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    revises_event_id: Some(RealtimeEventId(revises_event_id)),
                    ..
                }
            )) if revises_event_id == "evt_final"
        ));
    }

    #[test]
    fn explicit_revises_event_id_overrides_one_event_but_does_not_rebind_final_reference() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello world", 0, 200),
            Some(RealtimeEventId("evt_final".to_string())),
        );

        let explicit_revision = lifecycle.apply_partial(
            TranscriptUpdate::new("utt_1", "seg_1", 2, "hello, world", 0, 200)
                .with_revises_event_id(Some(RealtimeEventId("evt_custom".to_string()))),
        );
        assert!(matches!(
            explicit_revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    revises_event_id: Some(RealtimeEventId(revises_event_id)),
                    ..
                }
            )) if revises_event_id == "evt_custom"
        ));

        let next_revision = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_1",
            "seg_1",
            3,
            "hello world!",
            0,
            200,
        ));
        assert!(matches!(
            next_revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    revises_event_id: Some(RealtimeEventId(revises_event_id)),
                    ..
                }
            )) if revises_event_id == "evt_final"
        ));
    }

    #[test]
    fn revision_reason_list_matches_event_reason() {
        assert_eq!(
            TRANSCRIPT_REVISION_REASONS,
            [TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION]
        );
    }

    #[test]
    fn recorded_final_event_id_is_used_for_later_revision_fallback() {
        let mut lifecycle = TranscriptLifecycle::default();
        lifecycle.apply_final(
            TranscriptUpdate::new("utt_1", "seg_1", 1, "hello world", 0, 200),
            None,
        );
        lifecycle.record_final_event_id(
            &TranscriptUtteranceId("utt_1".to_string()),
            &TranscriptSegmentId("seg_1".to_string()),
            1,
            RealtimeEventId("evt_final".to_string()),
        );

        let revision = lifecycle.apply_partial(TranscriptUpdate::new(
            "utt_1",
            "seg_1",
            2,
            "hello, world",
            0,
            200,
        ));
        assert!(matches!(
            revision,
            TranscriptLifecycleResult::Event(RealtimeTranscriptEvent::Revision(
                RealtimeTranscriptRevision {
                    revises_event_id: Some(RealtimeEventId(revises_event_id)),
                    ..
                }
            )) if revises_event_id == "evt_final"
        ));
    }
}
