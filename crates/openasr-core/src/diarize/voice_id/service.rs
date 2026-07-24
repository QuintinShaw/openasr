//! High-level Voice ID enrollment service.
//!
//! Embed work happens outside the DB transaction. Writes re-check person status
//! and revision under `BEGIN IMMEDIATE` before committing.

use std::path::Path;

use thiserror::Error;

use super::domain::{CaptureContext, ConsentRecord, PersonView};
use super::ids::{PersonId, SampleId};
use super::matcher::PersonMatcher;
use super::migrate::open_store_with_v1_migration;
use super::quality::{QualityError, assess_enrollment_quality};
use super::space::EmbeddingSpace;
use super::store::{NewSampleInput, VoiceIdStore, VoiceIdStoreError};
use crate::diarize::contract::SpeakerEmbedding;
use crate::diarize::embed::{
    EmbedError, SpeakerEmbedder, SpeakerEmbedderIdentity, shared_embedder, shared_embedder_identity,
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
    #[error("{0}")]
    Migration(String),
}

const MIN_INITIAL_SAMPLES: usize = 1;
const MAX_INITIAL_SAMPLES: usize = 5;

pub struct EnrollmentClip {
    pub samples: Vec<f32>,
    pub capture_context: CaptureContext,
}

/// Load the live Voice ID matcher for the active embedder pack.
///
/// Opens the v2 store (running v1 migration when needed). Returns an empty
/// matcher when the embedder pack, home directory, or store is unavailable so
/// batch/streaming paths stay fail-open toward anonymous labels rather than
/// aborting transcription.
pub fn load_person_matcher_for_active_embedder() -> PersonMatcher {
    let Some(identity) = shared_embedder_identity() else {
        return empty_person_matcher();
    };
    let Some(embedder) = shared_embedder() else {
        return empty_person_matcher();
    };
    let calibration = embedder.calibration_profile();
    let space = EmbeddingSpace::for_active_embedder(identity, calibration);
    let threshold = calibration.voice_id_accept_threshold();
    let margin = calibration.voice_id_margin();
    let Ok(home) = crate::openasr_home() else {
        return PersonMatcher::new(space, Vec::new(), threshold, margin);
    };
    let Ok(store) = open_store_with_v1_migration(home) else {
        return PersonMatcher::new(space, Vec::new(), threshold, margin);
    };
    store
        .matcher_for_space(&space, threshold, margin)
        .unwrap_or_else(|_| PersonMatcher::new(space, Vec::new(), threshold, margin))
}

fn empty_person_matcher() -> PersonMatcher {
    // Unmatchable placeholder space: best_match always returns None.
    PersonMatcher::new(
        EmbeddingSpace::legacy_unverifiable_v1(1, "none"),
        Vec::new(),
        1.0,
        0.0,
    )
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
