use super::{
    RealtimeEventId, RealtimeTranscriptEvent, RealtimeTranscriptFinal, RealtimeTranscriptPartial,
    RealtimeTranscriptRevision, TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION,
    TranscriptSegmentState, TranscriptUpdate,
};

pub(super) fn inherited_language(
    update: &TranscriptUpdate,
    previous_language: Option<String>,
) -> Option<String> {
    update.language.clone().or(previous_language)
}

pub(super) fn inherited_speaker(
    update: &TranscriptUpdate,
    previous_speaker: Option<String>,
) -> Option<String> {
    update.speaker.clone().or(previous_speaker)
}

pub(super) fn inherited_speaker_label(
    update: &TranscriptUpdate,
    previous_speaker_label: Option<String>,
) -> Option<String> {
    update.speaker_label.clone().or(previous_speaker_label)
}

pub(super) fn inherited_speaker_profile_id(
    update: &TranscriptUpdate,
    previous_speaker_profile_id: Option<String>,
) -> Option<String> {
    update
        .speaker_profile_id
        .clone()
        .or(previous_speaker_profile_id)
}

pub(super) fn to_partial_event(
    update: TranscriptUpdate,
    language: Option<String>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
) -> RealtimeTranscriptEvent {
    to_update_event(
        update,
        false,
        language,
        speaker,
        speaker_label,
        speaker_profile_id,
    )
}

pub(super) fn to_final_event(
    update: TranscriptUpdate,
    language: Option<String>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
) -> RealtimeTranscriptEvent {
    to_update_event(
        update,
        true,
        language,
        speaker,
        speaker_label,
        speaker_profile_id,
    )
}

fn to_update_event(
    update: TranscriptUpdate,
    final_update: bool,
    language: Option<String>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
) -> RealtimeTranscriptEvent {
    let TranscriptUpdate {
        utterance_id,
        segment_id,
        revision,
        text,
        start_ms,
        end_ms,
        words,
        ..
    } = update;
    if final_update {
        RealtimeTranscriptEvent::Final(RealtimeTranscriptFinal {
            utterance_id,
            segment_id,
            revision,
            text,
            start_ms,
            end_ms,
            is_final: true,
            words,
            language,
            speaker,
            speaker_label,
            speaker_profile_id,
        })
    } else {
        RealtimeTranscriptEvent::Partial(RealtimeTranscriptPartial {
            utterance_id,
            segment_id,
            revision,
            text,
            start_ms,
            end_ms,
            is_final: false,
            words,
            language,
            speaker,
            speaker_label,
            speaker_profile_id,
        })
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn to_revision_event(
    update: TranscriptUpdate,
    recorded_final_event_id: Option<RealtimeEventId>,
    previous_language: Option<String>,
    previous_speaker: Option<String>,
    previous_speaker_label: Option<String>,
    previous_speaker_profile_id: Option<String>,
) -> (
    RealtimeTranscriptEvent,
    Option<RealtimeEventId>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let revises_event_id = update
        .revises_event_id
        .clone()
        .or(recorded_final_event_id.clone());
    let final_event_id = recorded_final_event_id.or(update.revises_event_id.clone());
    let language = inherited_language(&update, previous_language);
    let speaker = inherited_speaker(&update, previous_speaker);
    let speaker_label = inherited_speaker_label(&update, previous_speaker_label);
    let speaker_profile_id =
        inherited_speaker_profile_id(&update, previous_speaker_profile_id);

    (
        RealtimeTranscriptEvent::Revision(RealtimeTranscriptRevision {
            utterance_id: update.utterance_id,
            segment_id: update.segment_id,
            revises_event_id,
            revision: update.revision,
            text: update.text,
            start_ms: update.start_ms,
            end_ms: update.end_ms,
            is_final: true,
            reason: TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION.to_string(),
            words: update.words,
            language: language.clone(),
            speaker: speaker.clone(),
            speaker_label: speaker_label.clone(),
            speaker_profile_id: speaker_profile_id.clone(),
        }),
        final_event_id,
        language,
        speaker,
        speaker_label,
        speaker_profile_id,
    )
}

pub(super) fn segment_state(
    revision: u64,
    text: String,
    finalized: bool,
    final_event_id: Option<RealtimeEventId>,
    language: Option<String>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
) -> TranscriptSegmentState {
    TranscriptSegmentState {
        revision,
        text,
        finalized,
        final_event_id,
        language,
        speaker,
        speaker_label,
        speaker_profile_id,
    }
}
