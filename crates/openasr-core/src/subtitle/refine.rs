//! Post-hoc precise timeline refine for a finished transcription.
//!
//! The execution entry point lives in the native backend
//! ([`crate::refine_existing_transcription_timeline`]) so Forced Aligner
//! loading stays next to the in-transcription refine path. This module owns
//! product-level tests for the dual-view contract after refine.

#[cfg(test)]
mod tests {
    use crate::api::backend::{Segment, Transcription, WordTimestamp};
    use crate::{
        ExecutionTarget, NativeExecutionServices, align_plain_transcript_to_audio,
        refine_existing_transcription_timeline,
    };

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

    #[test]
    fn align_plain_transcript_fails_closed_on_empty_text() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let audio = vec![0.0f32; 16_000];
        let error = align_plain_transcript_to_audio(
            "   \n".into(),
            &audio,
            &services,
            ExecutionTarget::Cpu,
            Some("en"),
            true,
        )
        .expect_err("empty transcript must fail closed");
        assert!(
            matches!(
                error,
                crate::BackendError::WordTimestampAlignmentFailed { ref reason } if reason.contains("empty")
            ),
            "got {error}"
        );
    }

    #[test]
    fn align_plain_transcript_fails_closed_on_punctuation_only() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let audio = vec![0.0f32; 16_000];
        let error = align_plain_transcript_to_audio(
            "... -- !!".into(),
            &audio,
            &services,
            ExecutionTarget::Cpu,
            Some("en"),
            true,
        )
        .expect_err("punctuation-only transcript must fail closed");
        assert!(
            matches!(
                error,
                crate::BackendError::WordTimestampAlignmentFailed { .. }
            ),
            "got {error}"
        );
    }

    #[test]
    fn align_plain_transcript_fails_closed_on_japanese() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let audio = vec![0.0f32; 16_000];
        let error = align_plain_transcript_to_audio(
            "こんにちは".into(),
            &audio,
            &services,
            ExecutionTarget::Cpu,
            Some("ja"),
            true,
        )
        .expect_err("japanese must fail closed");
        assert!(
            error.to_string().contains("does not yet support language"),
            "got {error}"
        );
    }

    #[test]
    fn align_plain_transcript_fails_closed_when_pack_missing() {
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let temp = tempfile::tempdir().unwrap();
        let audio = vec![0.0f32; 16_000];
        let error = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(temp.path().as_os_str().to_os_string())),
                ("OPENASR_FORCED_ALIGNER_PACK", None),
                ("OPENASR_MODELS_DIR", None),
            ],
            || {
                align_plain_transcript_to_audio(
                    "hello world".into(),
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
    }

    #[test]
    fn align_plain_transcript_jfk_end_to_end_when_pack_available() {
        let source_pack =
            crate::models::qwen::forced_aligner_pack::resolve_forced_aligner_pack_path();
        let Some(source_pack) = source_pack else {
            eprintln!("skipping: Qwen3-ForcedAligner pack is not installed");
            return;
        };
        let fixture =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
        if !fixture.exists() {
            eprintln!("skipping: fixtures/jfk.wav is not present");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let copied = temp.path().join("qwen3-forced-aligner.oasr");
        std::fs::copy(&source_pack, &copied).expect("copy forced-aligner pack into isolated home");
        let samples = crate::load_native_wav_16khz_mono_f32_v0(
            &fixture,
            "plain-transcript-align-jfk",
            "plain-transcript-align-jfk",
        )
        .expect("load jfk.wav");
        let services = NativeExecutionServices::for_local_process()
            .expect("native execution services for test");
        let aligned = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(temp.path().as_os_str().to_os_string())),
                (
                    "OPENASR_FORCED_ALIGNER_PACK",
                    Some(copied.as_os_str().to_os_string()),
                ),
                ("OPENASR_MODELS_DIR", None),
            ],
            || {
                align_plain_transcript_to_audio(
                    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.".into(),
                    &samples,
                    &services,
                    ExecutionTarget::Cpu,
                    Some("en"),
                    true,
                )
                .expect("jfk forced alignment")
            },
        );
        assert_eq!(
            aligned.timeline_quality,
            Some(super::super::TimelineQuality::ForcedAligned)
        );
        assert!(
            !aligned.subtitle_cues.is_empty(),
            "subtitle cues must be projected for SRT/VTT export"
        );
        let words: Vec<_> = aligned
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .collect();
        assert!(
            words.len() >= 20,
            "expected the JFK word list, got {}",
            words.len()
        );
        assert_eq!(words.first().map(|word| word.word.as_str()), Some("And"));
        assert_eq!(words.last().map(|word| word.word.as_str()), Some("country"));
        let audio_duration_s = samples.len() as f32 / 16_000.0;
        for window in words.windows(2) {
            assert!(
                window[1].start + 1e-3 >= window[0].start,
                "word starts must be non-decreasing"
            );
        }
        assert!(
            words
                .last()
                .is_some_and(|word| word.end <= audio_duration_s + 0.35),
            "last word must stay inside the audio"
        );
        let srt =
            crate::render_transcription(&aligned, crate::ResponseFormat::Srt).expect("render srt");
        assert!(srt.contains("And so"), "SRT must reuse the shared exporter");
    }
}
