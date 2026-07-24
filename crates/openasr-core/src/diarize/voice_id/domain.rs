//! Voice ID domain model.
//!
//! Stable person_id is the only identity key. Display names are presentation
//! metadata and never group, merge, or decide margin.

use serde::{Deserialize, Serialize};

use super::ids::{PersonId, PrototypeId, SampleId};
use super::space::EmbeddingSpace;
use crate::diarize::contract::SpeakerEmbedding;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonStatus {
    Active,
    ConsentRevoked,
    Deleted,
}

impl PersonStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::ConsentRevoked => "consent_revoked",
            Self::Deleted => "deleted",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "active" => Some(Self::Active),
            "consent_revoked" => Some(Self::ConsentRevoked),
            "deleted" => Some(Self::Deleted),
            _ => None,
        }
    }

    pub fn allows_matching(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub granted_at: String,
    pub notice_version: String,
    pub capture_method: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureContext {
    pub device_class: String,
    pub input_route: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleQuality {
    pub speech_seconds: f32,
    pub snr_estimate: f32,
    pub clipping_ratio: f32,
    pub vad_coverage: f32,
    pub accepted_reason: String,
}

impl SampleQuality {
    /// Relative weight in [0, 1] used by prototype construction and support
    /// bonus. Higher speech duration / SNR / VAD coverage and lower clipping
    /// raise the weight; never invent a biometric score here.
    pub fn weight(&self) -> f32 {
        let speech = (self.speech_seconds / 15.0).clamp(0.25, 1.0);
        let snr = ((self.snr_estimate - 5.0) / 25.0).clamp(0.2, 1.0);
        let clipping = (1.0 - self.clipping_ratio * 8.0).clamp(0.1, 1.0);
        let coverage = self.vad_coverage.clamp(0.2, 1.0);
        (speech * 0.35 + snr * 0.25 + clipping * 0.2 + coverage * 0.2).clamp(0.05, 1.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub person_id: PersonId,
    pub display_name: String,
    pub status: PersonStatus,
    pub created_at: String,
    pub updated_at: String,
    /// Monotonic revision used as an ETag for optimistic concurrency.
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_preference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrollmentSample {
    pub sample_id: SampleId,
    pub person_id: PersonId,
    pub created_at: String,
    pub consent: ConsentRecord,
    pub capture_context: CaptureContext,
    pub quality: SampleQuality,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleEmbedding {
    pub sample_id: SampleId,
    pub space: EmbeddingSpace,
    pub embedding: SpeakerEmbedding,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrototypeMember {
    pub sample_id: SampleId,
    pub quality_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersonPrototype {
    pub prototype_id: PrototypeId,
    pub person_id: PersonId,
    pub space: EmbeddingSpace,
    pub medoid_sample_id: SampleId,
    pub medoid_embedding: SpeakerEmbedding,
    pub policy_version: String,
    pub members: Vec<PrototypeMember>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersonMatch {
    pub person_id: PersonId,
    pub display_name: String,
    pub score: f32,
    pub threshold: f32,
    pub runner_up_score: Option<f32>,
    pub matched_prototype_id: PrototypeId,
    pub matched_sample_id: SampleId,
}

/// Wire/history assignment produced when a diarized anonymous turn is resolved
/// against the Voice ID store. Snapshot label freezes the display name at match
/// time so later renames do not rewrite history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceIdAssignment {
    pub speaker_id: crate::diarize::contract::SpeakerId,
    /// Display name at match time (or the anonymous SPEAKER_NN label).
    pub speaker: String,
    /// Session-local anonymous label (SPEAKER_NN).
    pub speaker_label: String,
    pub speaker_person_id: Option<String>,
    /// Frozen display name written into history. Always set when a Person match
    /// is accepted so deleted persons still render safely.
    pub speaker_snapshot_label: Option<String>,
    /// Legacy v1 profile id when the match came through a migrated alias.
    pub speaker_profile_id: Option<String>,
}

impl VoiceIdAssignment {
    pub fn anonymous(speaker_id: crate::diarize::contract::SpeakerId) -> Self {
        let speaker_label = speaker_id.label();
        Self {
            speaker_id,
            speaker: speaker_label.clone(),
            speaker_label,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            speaker_profile_id: None,
        }
    }

    pub fn from_person_match(
        speaker_id: crate::diarize::contract::SpeakerId,
        person_match: &PersonMatch,
        legacy_profile_id: Option<String>,
    ) -> Self {
        Self {
            speaker_id,
            speaker: person_match.display_name.clone(),
            speaker_label: speaker_id.label(),
            speaker_person_id: Some(person_match.person_id.as_str().to_string()),
            speaker_snapshot_label: Some(person_match.display_name.clone()),
            speaker_profile_id: legacy_profile_id,
        }
    }
}

/// Candidate scope for a single transcription/realtime job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateScope {
    /// All active, consented, space-compatible persons.
    AllCompatible,
    /// Explicit allow-list. Empty means matching is disabled (not "all").
    Explicit(Vec<PersonId>),
}

impl CandidateScope {
    pub fn from_optional_ids(ids: Option<Vec<String>>) -> Result<Self, super::ids::IdError> {
        match ids {
            None => Ok(Self::AllCompatible),
            Some(raw) => {
                let mut parsed = Vec::with_capacity(raw.len());
                for id in raw {
                    parsed.push(PersonId::parse(id)?);
                }
                Ok(Self::Explicit(parsed))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonView {
    pub person_id: String,
    pub display_name: String,
    pub status: PersonStatus,
    pub created_at: String,
    pub updated_at: String,
    pub revision: u64,
    pub sample_count: usize,
    pub needs_reenrollment: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_preference: Option<String>,
    pub samples: Vec<SampleView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleView {
    pub sample_id: String,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_label: Option<String>,
    pub quality: SampleQuality,
    pub capture_context: CaptureContext,
    pub space_compatible: bool,
    pub needs_reenrollment: bool,
}
