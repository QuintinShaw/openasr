//! Voice ID core v2.
//!
//! Stable `person_id` identities, multi-sample enrollment, quality-aware medoid
//! prototypes, person-level margin matching, SQLite WAL storage, and conservative
//! v1 JSON migration. Raw enrollment audio is never retained.

mod backend;
mod domain;
mod ids;
mod matcher;
mod migrate;
mod prototypes;
mod quality;
mod service;
mod space;
mod store;

pub use backend::{
    DiarizationOutput, DiarizerBackendKind, EmbeddingEvidence, voice_id_evidence_from_output,
};
pub use domain::{
    CandidateScope, CaptureContext, ConsentRecord, EnrollmentSample, Person, PersonMatch,
    PersonPrototype, PersonStatus, PersonView, PrototypeMember, SampleEmbedding, SampleQuality,
    SampleView, VoiceIdAssignment,
};
pub use ids::{IdError, PERSON_ID_PREFIX, PersonId, PrototypeId, SAMPLE_ID_PREFIX, SampleId};
pub use matcher::{MatcherPerson, PersonMatcher};
pub use migrate::{VoiceIdMigrationError, migrate_v1_json_if_needed, open_store_with_v1_migration};
pub use prototypes::{
    DEFAULT_CLUSTER_COSINE_DISTANCE, MAX_PROTOTYPES_PER_PERSON, PrototypeSample,
    build_person_prototypes, score_prototype,
};
pub use quality::{
    MIN_SAMPLE_SPEECH_SECONDS, QualityError, TARGET_SAMPLE_SPEECH_SECONDS,
    assess_enrollment_quality,
};
pub use service::{
    EnrollmentClip, VoiceIdServiceError, add_sample_from_pcm, enroll_person_from_clips,
    load_person_matcher_for_active_embedder, prepare_sample_from_pcm, prepare_sample_from_wav_file,
};
pub use space::{
    EmbeddingSpace, LEGACY_UNVERIFIABLE_V1_MARKER, MATCHER_POLICY_VERSION,
    REDIMNET_FRONTEND_VERSION, WESPEAKER_FRONTEND_VERSION,
};
pub use store::{
    NewSampleInput, VOICE_ID_DB_ENV, VOICE_ID_SCHEMA_VERSION, VoiceIdStore, VoiceIdStoreError,
    timestamp_now,
};
