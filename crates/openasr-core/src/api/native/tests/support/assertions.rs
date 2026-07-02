use super::super::NativeAsrError;
use crate::{
    RealtimeEvent, RealtimeEventEnvelope, RealtimeLifecycleEvent, RealtimeTranscriptEvent,
};

pub(in super::super) fn assert_session_failed_contains<T>(
    result: Result<T, NativeAsrError>,
    expected: &str,
) {
    match result {
        Err(NativeAsrError::SessionFailed { message }) => {
            assert!(message.contains(expected), "{message}");
        }
        _ => panic!("expected SessionFailed containing '{expected}'"),
    }
}

pub(in super::super) fn assert_event_types(events: Vec<RealtimeEventEnvelope>, expected: &[&str]) {
    let actual = events
        .iter()
        .map(|event| event.event_type)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

pub(in super::super) fn assert_transcript_text(
    envelope: &RealtimeEventEnvelope,
    expected_text: &str,
    expected_revision: u64,
    expected_final: bool,
) {
    with_transcript_event(envelope, |event| {
        assert_eq!(event.text(), expected_text);
        assert_eq!(event.revision(), expected_revision);
        assert_eq!(event.is_final(), expected_final);
    });
}

pub(in super::super) fn transcript_utterance_id(envelope: &RealtimeEventEnvelope) -> String {
    with_transcript_event(envelope, |event| event.utterance_id().0.clone())
}

pub(in super::super) fn assert_configured_partial_results(
    events: &[RealtimeEventEnvelope],
    expected: bool,
) {
    let configured = events
        .iter()
        .find(|event| event.event_type == "session.configured")
        .expect("session.configured event");
    assert!(matches!(
        &configured.event,
        RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(configured))
            if configured.partial_results == expected
    ));
}

pub(in super::super) fn assert_configured_word_timestamps(
    events: &[RealtimeEventEnvelope],
    expected: bool,
) {
    let configured = events
        .iter()
        .find(|event| event.event_type == "session.configured")
        .expect("session.configured event");
    assert!(matches!(
        &configured.event,
        RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(configured))
            if configured.word_timestamps == expected
    ));
}

pub(in super::super) fn assert_transcript_words(
    envelope: &RealtimeEventEnvelope,
    expected_words: &[&str],
) {
    with_transcript_event(envelope, |event| {
        let words = event
            .words()
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>();
        assert_eq!(words, expected_words);
    });
}

fn with_transcript_event<T>(
    envelope: &RealtimeEventEnvelope,
    map: impl FnOnce(TranscriptEventRef<'_>) -> T,
) -> T {
    let event = match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => {
            TranscriptEventRef::Partial(partial)
        }
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_event)) => {
            TranscriptEventRef::Final(final_event)
        }
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision)) => {
            TranscriptEventRef::Revision(revision)
        }
        other => panic!("expected transcript event, got {other:?}"),
    };
    map(event)
}

enum TranscriptEventRef<'a> {
    Partial(&'a crate::realtime::RealtimeTranscriptPartial),
    Final(&'a crate::realtime::RealtimeTranscriptFinal),
    Revision(&'a crate::realtime::RealtimeTranscriptRevision),
}

impl TranscriptEventRef<'_> {
    fn text(&self) -> &str {
        match self {
            Self::Partial(event) => &event.text,
            Self::Final(event) => &event.text,
            Self::Revision(event) => &event.text,
        }
    }

    fn revision(&self) -> u64 {
        match self {
            Self::Partial(event) => event.revision,
            Self::Final(event) => event.revision,
            Self::Revision(event) => event.revision,
        }
    }

    fn is_final(&self) -> bool {
        match self {
            Self::Partial(event) => event.is_final,
            Self::Final(event) => event.is_final,
            Self::Revision(event) => event.is_final,
        }
    }

    fn utterance_id(&self) -> &crate::realtime::TranscriptUtteranceId {
        match self {
            Self::Partial(event) => &event.utterance_id,
            Self::Final(event) => &event.utterance_id,
            Self::Revision(event) => &event.utterance_id,
        }
    }

    fn words(&self) -> &[crate::realtime::RealtimeTranscriptWord] {
        match self {
            Self::Partial(event) => &event.words,
            Self::Final(event) => &event.words,
            Self::Revision(event) => &event.words,
        }
    }
}
