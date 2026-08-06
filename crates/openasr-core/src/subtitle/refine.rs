//! Post-hoc precise timeline refine for a finished transcription.
//!
//! The execution entry point lives in the native backend
//! ([`crate::refine_existing_transcription_timeline`]) so Forced Aligner
//! loading stays next to the in-transcription refine path. This module owns
//! product-level tests for the dual-view contract after refine.

#[cfg(test)]
mod tests {
    use crate::api::backend::{Segment, Transcription, WordTimestamp};
    use crate::{ExecutionTarget, NativeExecutionServices, refine_existing_transcription_timeline};

    fn reliable_two_speaker_transcription() -> Transcription {
        Transcription {
            text: "hello world. other speaker".to_string(),
            language: Some("en".into()),
            timeline_quality: Some(super::super::TimelineQuality::NativeReliable),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.5,
                    text: "hello world.".to_string(),
                    speaker: Some("SPEAKER_00".to_string()),
                    speaker_label: Some("SPEAKER_00".to_string()),
                    speaker_person_id: Some("person-a".to_string()),
                    speaker_snapshot_label: Some("Alice".to_string()),
                    words: vec![
                        WordTimestamp {
                            word: "hello".into(),
                            start: 0.0,
                            end: 0.5,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "world.".into(),
                            start: 0.55,
                            end: 1.2,
                            confidence: None,
                        },
                    ],
                },
                Segment {
                    start: 1.5,
                    end: 3.0,
                    text: "other speaker".to_string(),
                    speaker: Some("SPEAKER_01".to_string()),
                    speaker_label: Some("SPEAKER_01".to_string()),
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![
                        WordTimestamp {
                            word: "other".into(),
                            start: 1.5,
                            end: 2.0,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "speaker".into(),
                            start: 2.1,
                            end: 2.8,
                            confidence: None,
                        },
                    ],
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn refine_existing_noop_when_native_reliable_preserves_speakers_and_fills_cues() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let audio = vec![0.0f32; 16_000 * 3];
        let out = refine_existing_transcription_timeline(
            reliable_two_speaker_transcription(),
            &audio,
            &services,
            ExecutionTarget::Cpu,
            Some("en"),
            true,
        )
        .expect("native-reliable refine should no-op without the aligner pack");
        assert_eq!(
            out.timeline_quality,
            Some(super::super::TimelineQuality::NativeReliable)
        );
        assert!(!out.subtitle_cues.is_empty());
        assert!(out.segments.iter().any(|segment| {
            segment.speaker.as_deref() == Some("SPEAKER_00")
                && segment.speaker_person_id.as_deref() == Some("person-a")
        }));
        assert!(
            out.segments
                .iter()
                .any(|segment| segment.speaker.as_deref() == Some("SPEAKER_01"))
        );
        for cue in &out.subtitle_cues {
            assert!(cue.speaker.is_some(), "cue must keep speaker attribution");
        }
    }

    #[test]
    fn refine_existing_fail_closed_when_pack_missing() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let temp = tempfile::tempdir().unwrap();
        let mut transcription = reliable_two_speaker_transcription();
        transcription.timeline_quality = Some(super::super::TimelineQuality::NativeApproximate);
        for segment in &mut transcription.segments {
            segment.words.clear();
        }
        let audio = vec![0.0f32; 16_000 * 3];
        let error = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(temp.path().as_os_str().to_os_string())),
                ("OPENASR_FORCED_ALIGNER_PACK", None),
                ("OPENASR_MODELS_DIR", None),
            ],
            || {
                refine_existing_transcription_timeline(
                    transcription.clone(),
                    &audio,
                    &services,
                    ExecutionTarget::Cpu,
                    Some("en"),
                    true,
                )
                .expect_err("missing forced-aligner pack must fail closed")
            },
        );
        assert!(
            matches!(
                error,
                crate::BackendError::WordTimestampAlignmentPackMissing { backend: "native" }
            ),
            "expected pack-missing, got {error}"
        );
        assert_eq!(
            transcription.segments[0].speaker_person_id.as_deref(),
            Some("person-a")
        );
    }
}
