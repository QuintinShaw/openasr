use std::collections::HashMap;

#[path = "../transcript_apply.rs"]
mod transcript_apply;
#[path = "../transcript_state.rs"]
mod transcript_state;

use super::events::{
    RealtimeEventId, RealtimeTranscriptEvent, RealtimeTranscriptFinal, RealtimeTranscriptPartial,
    RealtimeTranscriptRevision, RealtimeTranscriptWord, TranscriptSegmentId, TranscriptUtteranceId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptRevisionPolicy {
    ExplicitPostFinalRevision,
}

pub const TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION: &str = "post_final_correction";

pub const TRANSCRIPT_REVISION_REASONS: [&str; 1] =
    [TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION];

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptUpdate {
    pub utterance_id: TranscriptUtteranceId,
    pub segment_id: TranscriptSegmentId,
    pub revision: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub language: Option<String>,
    pub speaker: Option<String>,
    pub speaker_label: Option<String>,
    pub speaker_profile_id: Option<String>,
    pub words: Vec<RealtimeTranscriptWord>,
    pub revises_event_id: Option<RealtimeEventId>,
}

impl TranscriptUpdate {
    pub fn new(
        utterance_id: impl Into<String>,
        segment_id: impl Into<String>,
        revision: u64,
        text: impl Into<String>,
        start_ms: u64,
        end_ms: u64,
    ) -> Self {
        Self {
            utterance_id: TranscriptUtteranceId(utterance_id.into()),
            segment_id: TranscriptSegmentId(segment_id.into()),
            revision,
            text: text.into(),
            start_ms,
            end_ms,
            language: None,
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
            revises_event_id: None,
        }
    }

    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    pub fn with_speaker(mut self, speaker: Option<String>) -> Self {
        self.speaker = speaker;
        self
    }

    pub fn with_speaker_identity(
        mut self,
        speaker: Option<String>,
        speaker_label: Option<String>,
        speaker_profile_id: Option<String>,
    ) -> Self {
        self.speaker = speaker;
        self.speaker_label = speaker_label;
        self.speaker_profile_id = speaker_profile_id;
        self
    }

    pub fn with_words(mut self, words: Vec<RealtimeTranscriptWord>) -> Self {
        self.words = words;
        self
    }

    pub fn with_revises_event_id(mut self, event_id: Option<RealtimeEventId>) -> Self {
        self.revises_event_id = event_id;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
// `Event` is the dominant, hot-path variant (nearly every result); boxing it to
// shrink the rare `Ignored*` variants would add a heap allocation per transcript
// event for no real benefit.
#[allow(clippy::large_enum_variant)]
pub enum TranscriptLifecycleResult {
    Event(RealtimeTranscriptEvent),
    IgnoredOutOfOrder {
        current_revision: u64,
        incoming_revision: u64,
    },
    IgnoredNoChange {
        current_revision: u64,
    },
}

#[derive(Debug, Clone)]
pub struct TranscriptLifecycle {
    policy: TranscriptRevisionPolicy,
    segments: HashMap<(TranscriptUtteranceId, TranscriptSegmentId), TranscriptSegmentState>,
}

#[derive(Debug, Clone)]
pub(super) struct TranscriptSegmentState {
    revision: u64,
    text: String,
    finalized: bool,
    final_event_id: Option<RealtimeEventId>,
    language: Option<String>,
    speaker: Option<String>,
    speaker_label: Option<String>,
    speaker_profile_id: Option<String>,
}

impl Default for TranscriptLifecycle {
    fn default() -> Self {
        Self::new(TranscriptRevisionPolicy::ExplicitPostFinalRevision)
    }
}

impl TranscriptLifecycle {
    pub fn new(policy: TranscriptRevisionPolicy) -> Self {
        Self {
            policy,
            segments: HashMap::new(),
        }
    }

    pub fn apply_partial(&mut self, update: TranscriptUpdate) -> TranscriptLifecycleResult {
        self.apply(update, false, None)
    }

    pub fn apply_final(
        &mut self,
        update: TranscriptUpdate,
        final_event_id: Option<RealtimeEventId>,
    ) -> TranscriptLifecycleResult {
        self.apply(update, true, final_event_id)
    }

    pub fn reset(&mut self) {
        self.segments.clear();
    }

    pub fn record_final_event_id(
        &mut self,
        utterance_id: &TranscriptUtteranceId,
        segment_id: &TranscriptSegmentId,
        revision: u64,
        event_id: RealtimeEventId,
    ) {
        let key = (utterance_id.clone(), segment_id.clone());
        if let Some(state) = self.segments.get_mut(&key)
            && state.finalized
            && state.revision == revision
        {
            state.final_event_id = Some(event_id);
        }
    }

    fn apply(
        &mut self,
        update: TranscriptUpdate,
        final_update: bool,
        final_event_id: Option<RealtimeEventId>,
    ) -> TranscriptLifecycleResult {
        let key = (update.utterance_id.clone(), update.segment_id.clone());
        let previous_state = self.segments.get(&key).cloned();
        if let Some(state) = previous_state.as_ref() {
            if let Some(result) = transcript_apply::reject_out_of_order_or_stable_same_revision(
                state,
                &update,
                final_update,
            ) {
                return result;
            }

            if state.finalized {
                if let Some(result) = transcript_apply::reject_no_change_after_final(state, &update)
                {
                    return result;
                }
                return self.apply_post_final_revision(
                    update,
                    state.final_event_id.clone(),
                    state.language.clone(),
                    state.speaker.clone(),
                    state.speaker_label.clone(),
                    state.speaker_profile_id.clone(),
                );
            }
        }
        let language = transcript_state::inherited_language(
            &update,
            previous_state
                .as_ref()
                .and_then(|state| state.language.clone()),
        );
        let speaker = transcript_state::inherited_speaker(
            &update,
            previous_state
                .as_ref()
                .and_then(|state| state.speaker.clone()),
        );
        let speaker_label = transcript_state::inherited_speaker_label(
            &update,
            previous_state
                .as_ref()
                .and_then(|state| state.speaker_label.clone()),
        );
        let speaker_profile_id = transcript_state::inherited_speaker_profile_id(
            &update,
            previous_state
                .as_ref()
                .and_then(|state| state.speaker_profile_id.clone()),
        );

        let state = transcript_state::segment_state(
            update.revision,
            update.text.clone(),
            final_update,
            final_update.then_some(final_event_id).flatten(),
            language.clone(),
            speaker.clone(),
            speaker_label.clone(),
            speaker_profile_id.clone(),
        );
        self.segments.insert(key, state);
        if final_update {
            TranscriptLifecycleResult::Event(transcript_state::to_final_event(
                update,
                language,
                speaker,
                speaker_label,
                speaker_profile_id,
            ))
        } else {
            TranscriptLifecycleResult::Event(transcript_state::to_partial_event(
                update,
                language,
                speaker,
                speaker_label,
                speaker_profile_id,
            ))
        }
    }

    fn apply_post_final_revision(
        &mut self,
        update: TranscriptUpdate,
        recorded_final_event_id: Option<RealtimeEventId>,
        previous_language: Option<String>,
        previous_speaker: Option<String>,
        previous_speaker_label: Option<String>,
        previous_speaker_profile_id: Option<String>,
    ) -> TranscriptLifecycleResult {
        match self.policy {
            TranscriptRevisionPolicy::ExplicitPostFinalRevision => {
                let key = (update.utterance_id.clone(), update.segment_id.clone());
                let (event, final_event_id, language, speaker, speaker_label, speaker_profile_id) =
                    transcript_state::to_revision_event(
                        update.clone(),
                        recorded_final_event_id,
                        previous_language,
                        previous_speaker,
                        previous_speaker_label,
                        previous_speaker_profile_id,
                    );
                self.segments.insert(
                    key,
                    transcript_state::segment_state(
                        update.revision,
                        update.text.clone(),
                        true,
                        final_event_id,
                        language.clone(),
                        speaker.clone(),
                        speaker_label.clone(),
                        speaker_profile_id.clone(),
                    ),
                );
                TranscriptLifecycleResult::Event(event)
            }
        }
    }
}
