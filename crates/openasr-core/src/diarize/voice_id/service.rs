//! High-level Voice ID enrollment service.
//!
//! Embed work happens outside the DB transaction. Writes re-check person status
//! and revision under `BEGIN IMMEDIATE` before committing.

use std::path::Path;

use thiserror::Error;

use super::domain::{CaptureContext, ConsentRecord, PersonView};
use super::ids::{PersonId, SampleId};
use super::matcher::PersonMatcher;
use super::quality::{QualityError, assess_enrollment_quality};
use super::space::EmbeddingSpace;
use super::store::{NewSampleInput, VoiceIdStore, VoiceIdStoreError};
use crate::diarize::contract::SpeakerEmbedding;
use crate::diarize::embed::{
    EmbedError, SpeakerEmbedder, SpeakerEmbedderIdentity, shared_embedder,
};

#[derive(Debug, Error)]
pub enum VoiceIdServiceError {
    #[error("{0}")]
    Store(#[from] VoiceIdStoreError),
    #[error("{0}")]
    Quality(#[from] QualityError),
    #[error("speaker enrollment requires a 16 kHz mono PCM16 WAV: {0}")]
    InvalidAudio(String),
    #[error("{}", crate::diarize::embed::VOICE_ID_EMBEDDER_PACK_MISSING_REASON)]
    EmbedderPackMissing,
    #[error("could not embed enrollment audio: {0}")]
    Embed(#[from] EmbedError),
    #[error("initial enrollment requires between {min} and {max} samples, got {got}")]
    InvalidSampleCount { min: usize, max: usize, got: usize },
}

#[derive(Debug, Error)]
pub enum VoiceIdLibraryError {
    #[error("Voice ID speaker embedder does not expose an embedding-space identity")]
    EmbedderIdentityUnavailable,
    #[error(transparent)]
    Store(#[from] VoiceIdStoreError),
}

const MIN_INITIAL_SAMPLES: usize = 1;
const MAX_INITIAL_SAMPLES: usize = 5;

pub struct EnrollmentClip {
    pub samples: Vec<f32>,
    pub capture_context: CaptureContext,
}

/// Load the live Voice ID matcher for the active embedder pack.
///
/// A missing embedder is represented by `Ok(None)`. Store/home failures stay
/// typed errors; callers must not misreport an unreadable library as empty.
pub fn load_person_matcher_for_active_embedder()
-> Result<Option<PersonMatcher>, VoiceIdLibraryError> {
    let Some(embedder) = shared_embedder() else {
        return Ok(None);
    };
    load_person_matcher_for_embedder(embedder.as_ref()).map(Some)
}

/// Load the person library for the exact embedding-space snapshot held by a
/// caller. This prevents a same-path pack replacement from pairing embeddings
/// produced by the old snapshot with enrollment rows from the new one.
pub(crate) fn load_person_matcher_for_embedder(
    embedder: &dyn SpeakerEmbedder,
) -> Result<PersonMatcher, VoiceIdLibraryError> {
    let Some(identity) = embedder.identity() else {
        return Err(VoiceIdLibraryError::EmbedderIdentityUnavailable);
    };
    let calibration = embedder.calibration_profile();
    let space = EmbeddingSpace::for_active_embedder(&identity, calibration);
    let threshold = calibration.voice_id_accept_threshold();
    let margin = calibration.voice_id_margin();
    let store = VoiceIdStore::open_default()?;
    Ok(store.matcher_for_space(&space, threshold, margin)?)
}

/// Whether any enrolled (non-deleted) person exists, independent of the
/// active embedder's space.
///
/// The naming stage (`identity::name_speakers_across_scopes`) uses this to
/// decide whether a missing embedder is a legitimate no-op (nobody enrolled,
/// so there is nothing naming could have attached) or a real degrade that
/// must fail closed (a person is enrolled and would silently go unmatched).
/// Deliberately not gated on the embedder identity the way
/// [`load_person_matcher_for_active_embedder`] is -- the question here is
/// "does a library exist at all", not "can today's embedder match against
/// it". An unreadable store is not evidence of an empty library, so failures
/// propagate to the request boundary.
pub fn person_library_is_non_empty() -> Result<bool, VoiceIdLibraryError> {
    let store = VoiceIdStore::open_default()?;
    Ok(!store.list_persons(None)?.is_empty())
}

/// Embed + quality-gate one clip. Does not touch the store.
pub fn prepare_sample_from_pcm(
    pcm: &[f32],
    capture_context: CaptureContext,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
) -> Result<NewSampleInput, VoiceIdServiceError> {
    let quality = assess_enrollment_quality(pcm)?;
    let embedding = embed_enrollment(embedder, pcm)?;
    let space = EmbeddingSpace::for_active_embedder(identity, embedder.calibration_profile());
    if embedding.dim() != space.dimension {
        return Err(VoiceIdServiceError::Embed(EmbedError::Unavailable(
            format!(
                "embedding dim {} != space dim {}",
                embedding.dim(),
                space.dimension
            ),
        )));
    }
    Ok(NewSampleInput {
        sample_id: SampleId::generate(),
        capture_context,
        quality,
        space,
        embedding,
    })
}

pub fn prepare_sample_from_wav_file(
    path: &Path,
    capture_context: CaptureContext,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
) -> Result<NewSampleInput, VoiceIdServiceError> {
    let pcm = load_wav(path)?;
    prepare_sample_from_pcm(&pcm, capture_context, embedder, identity)
}

pub fn enroll_person_from_clips(
    store: &VoiceIdStore,
    display_name: impl Into<String>,
    consent: ConsentRecord,
    clips: Vec<EnrollmentClip>,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
    color_preference: Option<String>,
) -> Result<PersonView, VoiceIdServiceError> {
    let n = clips.len();
    if !(MIN_INITIAL_SAMPLES..=MAX_INITIAL_SAMPLES).contains(&n) {
        return Err(VoiceIdServiceError::InvalidSampleCount {
            min: MIN_INITIAL_SAMPLES,
            max: MAX_INITIAL_SAMPLES,
            got: n,
        });
    }
    // Prepare all samples first. Any quality/embed failure aborts with zero writes.
    let mut prepared = Vec::with_capacity(n);
    for clip in clips {
        prepared.push(prepare_sample_from_pcm(
            &clip.samples,
            clip.capture_context,
            embedder,
            identity,
        )?);
    }
    Ok(store.enroll_person(display_name, consent, prepared, color_preference)?)
}

pub fn enroll_person_from_clips_idempotent(
    store: &VoiceIdStore,
    display_name: impl Into<String>,
    consent: ConsentRecord,
    clips: Vec<EnrollmentClip>,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
    color_preference: Option<String>,
    idempotency: super::store::IdempotencyRequest,
) -> Result<super::store::IdempotentPersonResult, VoiceIdServiceError> {
    let n = clips.len();
    if !(MIN_INITIAL_SAMPLES..=MAX_INITIAL_SAMPLES).contains(&n) {
        return Err(VoiceIdServiceError::InvalidSampleCount {
            min: MIN_INITIAL_SAMPLES,
            max: MAX_INITIAL_SAMPLES,
            got: n,
        });
    }
    let mut prepared = Vec::with_capacity(n);
    for clip in clips {
        prepared.push(prepare_sample_from_pcm(
            &clip.samples,
            clip.capture_context,
            embedder,
            identity,
        )?);
    }
    Ok(store.enroll_person_idempotent(
        display_name,
        consent,
        prepared,
        color_preference,
        idempotency,
    )?)
}

pub fn add_sample_from_pcm(
    store: &VoiceIdStore,
    person_id: &PersonId,
    expected_revision: Option<u64>,
    consent: ConsentRecord,
    pcm: &[f32],
    capture_context: CaptureContext,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
) -> Result<PersonView, VoiceIdServiceError> {
    let prepared = prepare_sample_from_pcm(pcm, capture_context, embedder, identity)?;
    Ok(store.add_sample(person_id, expected_revision, consent, prepared)?)
}

pub fn add_sample_from_pcm_idempotent(
    store: &VoiceIdStore,
    person_id: &PersonId,
    expected_revision: Option<u64>,
    consent: ConsentRecord,
    pcm: &[f32],
    capture_context: CaptureContext,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
    idempotency: super::store::IdempotencyRequest,
) -> Result<super::store::IdempotentPersonResult, VoiceIdServiceError> {
    let prepared = prepare_sample_from_pcm(pcm, capture_context, embedder, identity)?;
    Ok(
        store.add_sample_idempotent(
            person_id,
            expected_revision,
            consent,
            prepared,
            idempotency,
        )?,
    )
}

fn embed_enrollment(
    embedder: &dyn SpeakerEmbedder,
    samples: &[f32],
) -> Result<SpeakerEmbedding, VoiceIdServiceError> {
    // Prefer the same diarize-centroid path used by v1 enrollment when speech
    // regions are available; fall back to a direct embed of the whole clip.
    let speech = crate::diarize::pipeline::resolve_speech_regions(samples);
    if let Some(regions) = speech.filter(|r| !r.is_empty()) {
        let clusterer = crate::diarize::clustering::AgglomerativeClusterer::for_embedder(embedder);
        let diarization = crate::diarize::pipeline::BatchDiarizer::new(embedder, &clusterer)
            .diarize(
                samples,
                16_000,
                &regions,
                crate::diarize::contract::DiarizeHint::NumSpeakers(1),
            );
        if let Some((_, centroid)) = diarization.centroids.into_iter().next() {
            return Ok(centroid);
        }
    }
    Ok(embedder.embed(samples, 16_000)?)
}

fn load_wav(path: &Path) -> Result<Vec<f32>, VoiceIdServiceError> {
    crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        path,
        "voice-id enrollment",
        path.to_str().unwrap_or("voice-id enrollment input"),
    )
    .map_err(|e| VoiceIdServiceError::InvalidAudio(e.to_string()))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    struct IdentifiedEmbedder;

    impl SpeakerEmbedder for IdentifiedEmbedder {
        fn embed(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, EmbedError> {
            unreachable!("matcher loading must not run inference")
        }

        fn embedding_dim(&self) -> usize {
            192
        }

        fn identity(&self) -> Option<SpeakerEmbedderIdentity> {
            Some(SpeakerEmbedderIdentity {
                embedding_dim: 192,
                pack_fingerprint: "test-pack".to_string(),
            })
        }
    }

    #[test]
    fn empty_library_is_empty_but_unreadable_library_is_an_error() {
        let home = tempdir().expect("temporary OpenASR home");
        let db = home.path().join("voice-id.db");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(home.path().as_os_str().to_os_string())),
                (
                    super::super::VOICE_ID_DB_ENV,
                    Some(db.as_os_str().to_os_string()),
                ),
            ],
            || {
                assert!(!person_library_is_non_empty().expect("fresh store"));
                assert!(
                    load_person_matcher_for_embedder(&IdentifiedEmbedder)
                        .expect("fresh store matcher")
                        .is_empty()
                );

                std::fs::remove_file(&db).expect("remove fresh database");
                std::fs::create_dir(&db).expect("replace database with a directory");

                assert!(matches!(
                    person_library_is_non_empty(),
                    Err(VoiceIdLibraryError::Store(
                        VoiceIdStoreError::OpenDatabase { .. }
                    ))
                ));
                assert!(matches!(
                    load_person_matcher_for_embedder(&IdentifiedEmbedder),
                    Err(VoiceIdLibraryError::Store(
                        VoiceIdStoreError::OpenDatabase { .. }
                    ))
                ));
            },
        );
    }

    #[test]
    fn identityless_embedder_is_not_reported_as_an_empty_library() {
        struct IdentitylessEmbedder;

        impl SpeakerEmbedder for IdentitylessEmbedder {
            fn embed(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
            ) -> Result<SpeakerEmbedding, EmbedError> {
                unreachable!("matcher loading must not run inference")
            }

            fn embedding_dim(&self) -> usize {
                192
            }
        }

        assert!(matches!(
            load_person_matcher_for_embedder(&IdentitylessEmbedder),
            Err(VoiceIdLibraryError::EmbedderIdentityUnavailable)
        ));
    }
}
