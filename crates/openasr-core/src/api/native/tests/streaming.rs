use super::support::*;
use super::*;
use crate::{BackendKind, RealtimeEvent, RealtimeTranscriptEvent};

#[test]
fn test_only_native_streaming_fixture_emits_mutable_partials_then_final() {
    let mut session = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    assert_event_types(
        session.poll_events().unwrap(),
        &[
            "session.created",
            "session.configured",
            "audio.input.started",
        ],
    );

    let first = session.push_audio(test_frame(1, 0)).unwrap();
    assert_eq!(first[0].event_type, "transcript.partial");
    assert_transcript_text(&first[0], "hel", 1, false);

    let second = session.push_audio(test_frame(2, 20)).unwrap();
    assert_eq!(second[0].event_type, "transcript.partial");
    assert_transcript_text(&second[0], "hello wor", 2, false);

    let final_event = session.push_audio(test_frame(3, 40)).unwrap();
    assert_eq!(final_event[0].event_type, "transcript.final");
    assert_transcript_text(&final_event[0], "hello world", 3, true);

    let final_id = final_event[0].event_id.clone();
    let duplicate_stable_text = session.push_audio(test_frame(4, 60)).unwrap();
    assert!(
        duplicate_stable_text.is_empty(),
        "same-text post-final output must not silently rewrite a final"
    );

    let revision = session.push_audio(test_frame(5, 80)).unwrap();
    assert_eq!(revision[0].event_type, "transcript.revision");
    assert_transcript_text(&revision[0], "hello, world", 5, true);
    assert!(matches!(
        &revision[0].event,
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision))
            if revision.revises_event_id.as_ref() == Some(&final_id)
                && revision.reason == crate::TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION
    ));
}

#[test]
fn native_streaming_executor_carries_requested_word_timestamps() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let options = NativeAsrRequestOptions::new()
        .with_partial_results(true)
        .with_word_timestamps(true);
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            options,
            NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
        )
        .unwrap();

    let configured = session.poll_events().unwrap();
    assert_configured_word_timestamps(&configured, true);

    let first = session.push_audio(test_frame(1, 0)).unwrap();
    assert_eq!(first[0].event_type, "transcript.partial");
    assert_transcript_words(&first[0], &["hel"]);

    let second = session.push_audio(test_frame(2, 20)).unwrap();
    assert_eq!(second[0].event_type, "transcript.partial");
    assert_transcript_words(&second[0], &["hello", "wor"]);
}

#[test]
fn native_streaming_executor_does_not_emit_words_when_request_opts_out() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let options = NativeAsrRequestOptions::new().with_partial_results(true);
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            options,
            NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .with_word_timestamps(true),
        )
        .unwrap();

    let configured = session.poll_events().unwrap();
    assert_configured_word_timestamps(&configured, false);

    let first = session.push_audio(test_frame(1, 0)).unwrap();
    assert_eq!(first[0].event_type, "transcript.partial");
    assert_transcript_words(&first[0], &[]);
}

#[test]
fn test_only_native_streaming_fixture_orders_lifecycle_before_first_audio_output() {
    let mut session = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );

    let events = session.push_audio(test_frame(1, 0)).unwrap();
    assert_event_types(
        events.clone(),
        &[
            "session.created",
            "session.configured",
            "audio.input.started",
            "transcript.partial",
        ],
    );
    assert_transcript_text(&events[3], "hel", 1, false);
    assert!(session.poll_events().unwrap().is_empty());
}

#[test]
fn test_only_native_streaming_fixture_preserves_lifecycle_on_early_close_and_cancel() {
    let mut closed = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    assert_event_types(
        closed.close().unwrap(),
        &[
            "session.created",
            "session.configured",
            "audio.input.started",
            "audio.input.stopped",
            "session.closed",
        ],
    );

    let mut cancelled = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    assert_event_types(
        cancelled.cancel().unwrap(),
        &[
            "session.created",
            "session.configured",
            "audio.input.started",
            "error",
            "session.closed",
        ],
    );
}

#[test]
fn test_only_native_streaming_fixture_cancel_stops_output() {
    let mut session = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    let _ = session.poll_events().unwrap();
    assert_eq!(
        session.push_audio(test_frame(1, 0)).unwrap()[0].event_type,
        "transcript.partial"
    );

    let cancelled = session.cancel().unwrap();
    assert_event_types(cancelled, &["error", "session.closed"]);
    assert!(matches!(
        session.push_audio(test_frame(2, 20)),
        Err(NativeAsrError::SessionClosed)
    ));
    assert!(session.poll_events().unwrap().is_empty());
}

#[test]
fn test_only_native_streaming_fixture_audio_backpressure_is_queue_depth() {
    let mut session = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new()
            .with_partial_results(true)
            .with_backpressure(NativeAsrBackpressurePolicy {
                max_queued_audio_frames: 1,
                max_queued_events: 64,
            }),
    );
    let _ = session.poll_events().unwrap();
    for seq in 1..=5 {
        session.push_audio(test_frame(seq, (seq - 1) * 20)).unwrap();
    }
}

#[test]
fn test_only_fixture_cannot_be_selected_as_product_backend() {
    assert!(!BackendKind::ALL.contains(&TEST_ONLY_STREAMING_FIXTURE_ID));
    assert!(
        TEST_ONLY_STREAMING_FIXTURE_ID
            .parse::<BackendKind>()
            .is_err()
    );
}

#[test]
fn native_streaming_executor_downgrades_to_final_only_when_session_partial_disabled() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let context = NativeAsrSessionContext::new("rt_native_test");
    let options = NativeAsrRequestOptions::new().with_partial_results(true);
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            context,
            options,
            NativeAsrStreamingSessionConfig::new().with_partial_results(false),
        )
        .unwrap();
    let configured = session.poll_events().unwrap();
    assert_configured_partial_results(&configured, false);

    assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());
    assert!(session.push_audio(test_frame(2, 20)).unwrap().is_empty());
    let final_event = session.push_audio(test_frame(3, 40)).unwrap();
    assert_event_types(final_event.clone(), &["transcript.final"]);
    assert_transcript_text(&final_event[0], "hello world", 3, true);
}

#[test]
fn native_streaming_executor_requires_true_streaming_capability() {
    let adapter = TestOnlyOfflineAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();

    assert!(matches!(
        executor.start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        ),
        Err(NativeAsrError::BackendDoesNotSupportTrueStreaming { backend })
            if backend == "test-only-offline-adapter"
    ));
}

#[test]
fn default_start_streaming_session_fails_closed() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyDefaultStreamingExecutor;
    let model_pack = test_model_pack();

    assert!(matches!(
        executor.start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        ),
        Err(NativeAsrError::BackendDoesNotSupportTrueStreaming { backend })
            if backend == "test-only-default-streaming-executor"
    ));
}

#[test]
fn request_partial_opt_out_prevents_test_fixture_partials() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            NativeAsrRequestOptions::new().with_partial_results(false),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        )
        .unwrap();
    let configured = session.poll_events().unwrap();
    assert_configured_partial_results(&configured, false);

    assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());
    assert!(session.push_audio(test_frame(2, 20)).unwrap().is_empty());
    let final_event = session.push_audio(test_frame(3, 40)).unwrap();
    assert_event_types(final_event.clone(), &["transcript.final"]);
    assert_transcript_text(&final_event[0], "hello world", 3, true);
}

#[test]
fn native_streaming_executor_downgrades_when_adapter_lacks_partial_support() {
    let adapter = TestOnlyNativeFinalOnlyStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            NativeAsrRequestOptions::new().with_partial_results(true),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        )
        .unwrap();
    let configured = session.poll_events().unwrap();
    assert_configured_partial_results(&configured, false);

    assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());
    assert!(session.push_audio(test_frame(2, 20)).unwrap().is_empty());
    let final_event = session.push_audio(test_frame(3, 40)).unwrap();
    assert_event_types(final_event.clone(), &["transcript.final"]);
    assert_transcript_text(&final_event[0], "hello world", 3, true);
}

#[test]
fn test_only_native_streaming_fixture_flush_finish_and_close_lifecycle() {
    let mut flushed = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    let _ = flushed.poll_events().unwrap();
    let _ = flushed.push_audio(test_frame(1, 0)).unwrap();
    let flushed_events = flushed.flush().unwrap();
    assert_event_types(flushed_events.clone(), &["transcript.final"]);
    assert_transcript_text(&flushed_events[0], "hel", 1, true);
    let after_flush = flushed.push_audio(test_frame(2, 20)).unwrap();
    assert_event_types(after_flush.clone(), &["transcript.partial"]);
    assert_transcript_text(&after_flush[0], "hello wor", 2, false);
    assert_ne!(
        transcript_utterance_id(&flushed_events[0]),
        transcript_utterance_id(&after_flush[0])
    );
    let finished_after_flush = flushed.finish().unwrap();
    assert_event_types(
        finished_after_flush.clone(),
        &["transcript.final", "audio.input.stopped"],
    );
    assert_transcript_text(&finished_after_flush[0], "hello wor", 2, true);
    assert_event_types(flushed.close().unwrap(), &["session.closed"]);

    let mut finished = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    let _ = finished.poll_events().unwrap();
    let _ = finished.push_audio(test_frame(1, 0)).unwrap();
    let finished_events = finished.finish().unwrap();
    assert_event_types(
        finished_events.clone(),
        &["transcript.final", "audio.input.stopped"],
    );
    assert_transcript_text(&finished_events[0], "hel", 1, true);
    assert_event_types(finished.close().unwrap(), &["session.closed"]);

    let mut closed = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new().with_partial_results(true),
    );
    let _ = closed.poll_events().unwrap();
    assert_event_types(
        closed.close().unwrap(),
        &["audio.input.stopped", "session.closed"],
    );
    assert!(matches!(
        closed.push_audio(test_frame(1, 0)),
        Err(NativeAsrError::SessionClosed)
    ));
}

#[test]
fn test_only_native_streaming_fixture_rejects_event_queue_smaller_than_first_output() {
    assert_session_failed_contains(
        test_only_streaming_session_result(
            NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .with_backpressure(NativeAsrBackpressurePolicy {
                    max_queued_audio_frames: 64,
                    max_queued_events: 3,
                }),
            true,
        ),
        "invalid Native ASR streaming session config",
    );
}

#[test]
fn test_only_native_streaming_fixture_accepts_minimum_event_queue_on_first_audio_output() {
    let mut session = test_only_streaming_session(
        NativeAsrStreamingSessionConfig::new()
            .with_partial_results(true)
            .with_backpressure(NativeAsrBackpressurePolicy {
                max_queued_audio_frames: 64,
                max_queued_events: 4,
            }),
    );

    assert_event_types(
        session.push_audio(test_frame(1, 0)).unwrap(),
        &[
            "session.created",
            "session.configured",
            "audio.input.started",
            "transcript.partial",
        ],
    );
}

#[test]
fn final_only_flush_preserves_suppressed_hypothesis() {
    let adapter = TestOnlyNativeStreamingAdapter;
    let executor = TestOnlyNativeStreamingExecutor;
    let model_pack = test_model_pack();
    let mut session = executor
        .start_streaming_session(
            &adapter,
            &model_pack,
            NativeAsrHardwareTarget::Cpu,
            NativeAsrSessionContext::new("rt_native_test"),
            NativeAsrRequestOptions::new().with_partial_results(false),
            NativeAsrStreamingSessionConfig::new().with_partial_results(true),
        )
        .unwrap();
    let _ = session.poll_events().unwrap();
    assert!(session.push_audio(test_frame(1, 0)).unwrap().is_empty());

    let flushed = session.flush().unwrap();
    assert_event_types(flushed.clone(), &["transcript.final"]);
    assert_transcript_text(&flushed[0], "hel", 1, true);
}
