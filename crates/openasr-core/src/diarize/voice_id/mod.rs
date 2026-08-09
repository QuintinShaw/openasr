//! Voice ID core.
//!
//! Stable `person_id` identities, multi-sample enrollment, quality-aware medoid
//! prototypes, person-level margin matching, and SQLite WAL storage. Raw
//! enrollment audio is never retained.

mod domain;
mod evidence;
mod identity;
mod ids;
mod matcher;
mod naming;
mod prototypes;
mod quality;
mod service;
mod space;
mod store;

pub use domain::{
    CandidateScope, CaptureContext, ConsentRecord, EnrollmentSample, Person, PersonMatch,
    PersonPrototype, PersonStatus, PersonView, PrototypeMember, SampleEmbedding, SampleQuality,
    SampleView, VOICE_ID_LABEL_MAX_CHARS, VoiceIdAssignment, VoiceIdColor,
};
pub use identity::{
    SpeakerIdentityError, SpeakerScope, name_speakers_across_scopes,
    name_speakers_from_labeled_segments,
};
pub(crate) use identity::{
    name_speakers_across_scopes_with_embedder_and_progress,
    resolve_timeline_identities_with_embedder_and_progress,
};
pub use ids::{IdError, PERSON_ID_PREFIX, PersonId, PrototypeId, SAMPLE_ID_PREFIX, SampleId};
pub use matcher::{MatcherPerson, PersonMatcher};
pub use naming::{SpeakerNamingRefusal, UnnamedSpeaker};
pub use prototypes::{
    DEFAULT_CLUSTER_COSINE_DISTANCE, MAX_PROTOTYPES_PER_PERSON, PrototypeSample,
    build_person_prototypes, score_prototype,
};
pub use quality::{
    MIN_SAMPLE_SPEECH_SECONDS, QualityError, TARGET_SAMPLE_SPEECH_SECONDS,
    assess_enrollment_quality,
};
pub(crate) use service::load_person_matcher_for_embedder;
pub use service::{
    EnrollmentClip, VoiceIdLibraryError, VoiceIdServiceError, add_sample_from_pcm,
    add_sample_from_pcm_idempotent, enroll_person_from_clips, enroll_person_from_clips_idempotent,
    person_library_is_non_empty, prepare_sample_from_pcm, prepare_sample_from_wav_file,
};
pub use space::{
    EmbeddingSpace, LEGACY_UNVERIFIABLE_V1_MARKER, MATCHER_POLICY_VERSION,
    REDIMNET_FRONTEND_VERSION,
};
pub use store::{
    IdempotencyRequest, IdempotentPersonResult, NewSampleInput, PersonMetadataUpdate,
    VOICE_ID_DB_ENV, VOICE_ID_SCHEMA_VERSION, VoiceIdStore, VoiceIdStoreError, timestamp_now,
};
