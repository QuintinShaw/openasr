//! Speaker display-assignment type shared by the realtime/native transcription
//! paths.
//!
//! This module used to also hold the legacy single-file `voiceprints.json`
//! store (`SpeakerProfile` / `VoiceprintStore` / `SpeakerProfileMatcher`).
//! That store had no successor migration into Voice ID v2 and no live
//! callers, so it was removed; Voice ID (`super::voice_id`) is the only
//! speaker-identity store now. `SpeakerDisplayAssignment` survives because
//! realtime/native transcription still assemble it as the display-facing
//! projection of a `VoiceIdAssignment`.

use super::contract::SpeakerId;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerDisplayAssignment {
    pub speaker_id: SpeakerId,
    pub speaker: String,
    pub speaker_label: String,
    /// Stable Voice ID person id when a v2 match was accepted.
    pub speaker_person_id: Option<String>,
    /// Display name frozen at assignment time for history.
    pub speaker_snapshot_label: Option<String>,
}

impl SpeakerDisplayAssignment {
    pub fn anonymous(speaker_id: SpeakerId) -> Self {
        let speaker_label = speaker_id.label();
        Self {
            speaker_id,
            speaker: speaker_label.clone(),
            speaker_label,
            speaker_person_id: None,
            speaker_snapshot_label: None,
        }
    }

    pub fn from_voice_id_assignment(
        assignment: crate::diarize::voice_id::VoiceIdAssignment,
    ) -> Self {
        Self {
            speaker_id: assignment.speaker_id,
            speaker: assignment.speaker,
            speaker_label: assignment.speaker_label,
            speaker_person_id: assignment.speaker_person_id,
            speaker_snapshot_label: assignment.speaker_snapshot_label,
        }
    }

    /// True when `speaker` carries an enrolled voice-match display name
    /// rather than the anonymous session label. This is the wire contract's
    /// gate for the `speaker_label` field (see the `RealtimeTranscript*`
    /// ts-rs docs): `speaker_label` is only meaningful once `speaker` has
    /// been replaced, since otherwise it would just duplicate `speaker`.
    /// Deliberately compares `speaker` to `speaker_label` rather than
    /// checking `speaker_person_id.is_some()` -- the wire field is about
    /// whether the *displayed* value changed, not about which internal
    /// mechanism produced the assignment.
    pub fn is_display_name_match(&self) -> bool {
        self.speaker != self.speaker_label
    }
}
