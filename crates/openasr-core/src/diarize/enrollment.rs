//! Local speaker voice-match profiles.
//!
//! Profiles persist only L2-normalized embeddings plus the active speaker
//! embedder identity. Raw enrollment audio is never written to the store.
//! Matching is a diarization convenience, not authentication.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::contract::{SpeakerEmbedding, SpeakerId};
use super::embed::{EmbedError, SpeakerEmbedder, SpeakerEmbedderIdentity, shared_embedder};

/// Default display name for the first-person voice-match profile created by the
/// CLI. The REST API accepts any non-empty display name.
pub const DEFAULT_ENROLLED_NAME: &str = "SPEAKER_ME";
/// Default cosine-similarity floor for matching a diarized speaker to a stored
/// voice-match profile.
pub const DEFAULT_MATCH_SIMILARITY: f32 = 0.5;
/// Version of the on-disk multi-profile voiceprint store.
pub const VOICEPRINT_STORE_VERSION: u32 = 1;
/// Store override. This is intentionally not the old single-profile env var.
pub const VOICEPRINT_STORE_ENV: &str = "OPENASR_SPEAKER_PROFILES";
/// Minimum amount of detected non-silent speech required for enrollment.
pub const MIN_ENROLLMENT_SPEECH_SECONDS: f32 = 5.0;
/// Public id prefix emitted in API responses and realtime events.
pub const SPEAKER_PROFILE_ID_PREFIX: &str = "vp_";

static PROFILE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Canonical multi-profile store location: `OPENASR_SPEAKER_PROFILES` when set,
/// otherwise `openasr_home()/diarize/voiceprints.json`.
pub fn voiceprint_store_path() -> Option<PathBuf> {
    std::env::var(VOICEPRINT_STORE_ENV)
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            crate::openasr_home()
                .ok()
                .map(|home| home.join("diarize").join("voiceprints.json"))
        })
}

#[derive(Debug, Error)]
pub enum VoiceprintStoreError {
    #[error("unsupported voiceprint store version {found}; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("could not read voiceprint store {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not parse voiceprint store {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("could not create voiceprint store directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not write voiceprint store {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not serialize voiceprint store: {0}")]
    Serialize(serde_json::Error),
    #[error("speaker profile not found: {0}")]
    NotFound(String),
    #[error("speaker profile name must not be empty")]
    EmptyName,
    #[error("speaker profile id is invalid: {0}")]
    InvalidId(String),
    #[error("speaker profile match similarity must be between 0 and 1")]
    InvalidMatchSimilarity,
}

#[derive(Debug, Error)]
pub enum SpeakerEnrollmentError {
    #[error("speaker enrollment requires a 16 kHz mono PCM16 WAV: {0}")]
    InvalidAudio(String),
    #[error("enrollment audio is silent: no speech was detected")]
    NoSpeech,
    #[error(
        "enrollment audio is too short: need at least {required:.1} seconds of speech, got {actual:.2}"
    )]
    TooShortSpeech { required: f32, actual: f32 },
    #[error(
        "creating a voice match profile requires the WeSpeaker speaker-embedder pack (wespeaker-voxceleb-resnet34-lm); install the pack first"
    )]
    EmbedderPackMissing,
    #[error("could not embed enrollment audio: {0}")]
    Embed(EmbedError),
    #[error("{0}")]
    Store(VoiceprintStoreError),
}

impl From<VoiceprintStoreError> for SpeakerEnrollmentError {
    fn from(error: VoiceprintStoreError) -> Self {
        Self::Store(error)
    }
}

impl From<EmbedError> for SpeakerEnrollmentError {
    fn from(error: EmbedError) -> Self {
        Self::Embed(error)
    }
}

/// Versioned set of voice-match profiles. All compatibility
/// metadata lives per profile so future packs can coexist with older profiles;
/// incompatible profiles are skipped at match time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceprintStore {
    pub version: u32,
    pub profiles: Vec<SpeakerProfile>,
}

impl Default for VoiceprintStore {
    fn default() -> Self {
        Self {
            version: VOICEPRINT_STORE_VERSION,
            profiles: Vec::new(),
        }
    }
}

impl VoiceprintStore {
    pub fn load(path: &Path) -> Result<Self, VoiceprintStoreError> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).map_err(|source| VoiceprintStoreError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let store: Self =
            serde_json::from_slice(&bytes).map_err(|source| VoiceprintStoreError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        store.validate_version()?;
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<(), VoiceprintStoreError> {
        self.validate_version()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| VoiceprintStoreError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
            set_owner_only_dir_permissions(parent);
        }
        let json = serde_json::to_vec_pretty(self).map_err(VoiceprintStoreError::Serialize)?;
        write_owner_only_file(path, &json).map_err(|source| VoiceprintStoreError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn add_profile(&mut self, profile: SpeakerProfile) {
        self.profiles.push(profile);
        self.profiles.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
    }

    pub fn profile(&self, id: &str) -> Option<&SpeakerProfile> {
        self.profiles.iter().find(|profile| profile.id == id)
    }

    pub fn profile_mut(&mut self, id: &str) -> Option<&mut SpeakerProfile> {
        self.profiles.iter_mut().find(|profile| profile.id == id)
    }

    pub fn remove_profile(&mut self, id: &str) -> Option<SpeakerProfile> {
        let index = self.profiles.iter().position(|profile| profile.id == id)?;
        Some(self.profiles.remove(index))
    }

    pub fn compatible_profiles<'a>(
        &'a self,
        identity: &SpeakerEmbedderIdentity,
    ) -> Vec<&'a SpeakerProfile> {
        self.profiles
            .iter()
            .filter(|profile| profile.is_compatible_with(identity))
            .collect()
    }

    fn validate_version(&self) -> Result<(), VoiceprintStoreError> {
        if self.version == VOICEPRINT_STORE_VERSION {
            Ok(())
        } else {
            Err(VoiceprintStoreError::UnsupportedVersion {
                found: self.version,
                expected: VOICEPRINT_STORE_VERSION,
            })
        }
    }
}

/// One stored voice-match profile. `embedding` is the only biometric material.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeakerProfile {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
    pub sample_seconds: f32,
    pub embedding_dim: usize,
    pub pack_fingerprint: String,
    pub match_similarity: f32,
    pub embedding: Vec<f32>,
}

impl SpeakerProfile {
    pub fn is_compatible_with(&self, identity: &SpeakerEmbedderIdentity) -> bool {
        self.embedding_dim == identity.embedding_dim
            && self.embedding.len() == identity.embedding_dim
            && self.pack_fingerprint == identity.pack_fingerprint
    }

    pub fn compatibility_status(&self, identity: &SpeakerEmbedderIdentity) -> ProfileCompatibility {
        if self.embedding_dim != identity.embedding_dim
            || self.embedding.len() != self.embedding_dim
        {
            return ProfileCompatibility::Incompatible {
                reason: format!(
                    "embedding dimension mismatch: profile has {}, active embedder has {}",
                    self.embedding_dim, identity.embedding_dim
                ),
            };
        }
        if self.pack_fingerprint != identity.pack_fingerprint {
            return ProfileCompatibility::Incompatible {
                reason: "embedder pack fingerprint mismatch".to_string(),
            };
        }
        ProfileCompatibility::Compatible
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCompatibility {
    Compatible,
    Incompatible { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerProfileMatch {
    pub profile_id: String,
    pub name: String,
    pub similarity: f32,
    pub threshold: f32,
    pub runner_up_similarity: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerDisplayAssignment {
    pub speaker_id: SpeakerId,
    pub speaker: String,
    pub speaker_label: String,
    pub speaker_profile_id: Option<String>,
}

impl SpeakerDisplayAssignment {
    pub fn anonymous(speaker_id: SpeakerId) -> Self {
        let speaker_label = speaker_id.label();
        Self {
            speaker_id,
            speaker: speaker_label.clone(),
            speaker_label,
            speaker_profile_id: None,
        }
    }

    pub fn from_match(speaker_id: SpeakerId, profile_match: SpeakerProfileMatch) -> Self {
        Self {
            speaker_id,
            speaker: profile_match.name,
            speaker_label: speaker_id.label(),
            speaker_profile_id: Some(profile_match.profile_id),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpeakerProfileMatcher {
    profiles: Vec<SpeakerProfile>,
}

impl SpeakerProfileMatcher {
    pub fn load_for_identity(
        path: &Path,
        identity: &SpeakerEmbedderIdentity,
    ) -> Result<Self, VoiceprintStoreError> {
        let store = VoiceprintStore::load(path)?;
        Ok(Self {
            profiles: store
                .profiles
                .into_iter()
                .filter(|profile| profile.is_compatible_with(identity))
                .collect(),
        })
    }

    pub fn from_profiles(profiles: Vec<SpeakerProfile>) -> Self {
        Self { profiles }
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn best_match(&self, embedding: &SpeakerEmbedding) -> Option<SpeakerProfileMatch> {
        self.best_match_with_policy(embedding, 0.0, 0.0, 0.0)
    }

    /// A conservative realtime anchor match. The per-profile threshold remains
    /// the user-visible match floor, but streaming identity anchoring needs a
    /// higher floor plus a runner-up margin before it creates/reuses a
    /// profile-owned session speaker.
    pub fn strong_unambiguous_match(
        &self,
        embedding: &SpeakerEmbedding,
        threshold_floor: f32,
        margin: f32,
    ) -> Option<SpeakerProfileMatch> {
        self.best_match_with_policy(embedding, threshold_floor, margin, 0.0)
    }

    pub fn strong_unambiguous_match_with_tolerance(
        &self,
        embedding: &SpeakerEmbedding,
        threshold_floor: f32,
        margin: f32,
        threshold_tolerance: f32,
    ) -> Option<SpeakerProfileMatch> {
        self.best_match_with_policy(
            embedding,
            threshold_floor,
            margin,
            threshold_tolerance.max(0.0),
        )
    }

    pub fn best_similarity_and_threshold(
        &self,
        embedding: &SpeakerEmbedding,
        threshold_floor: f32,
    ) -> Option<(f32, f32)> {
        self.profiles
            .iter()
            .filter(|profile| profile.embedding.len() == embedding.dim())
            .map(|profile| {
                let stored = SpeakerEmbedding(profile.embedding.clone());
                (
                    stored.cosine(embedding),
                    profile.match_similarity.max(threshold_floor),
                )
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
    }

    fn best_match_with_policy(
        &self,
        embedding: &SpeakerEmbedding,
        threshold_floor: f32,
        margin: f32,
        threshold_tolerance: f32,
    ) -> Option<SpeakerProfileMatch> {
        let mut best: Option<SpeakerProfileMatch> = None;
        let mut runner_up_similarity: Option<f32> = None;
        for profile in &self.profiles {
            if profile.embedding.len() != embedding.dim() {
                continue;
            }
            let stored = SpeakerEmbedding(profile.embedding.clone());
            let similarity = stored.cosine(embedding);
            let threshold = profile.match_similarity.max(threshold_floor);
            let candidate = SpeakerProfileMatch {
                profile_id: profile.id.clone(),
                name: profile.name.clone(),
                similarity,
                threshold,
                runner_up_similarity: None,
            };
            match &best {
                Some(current) if similarity <= current.similarity => {
                    runner_up_similarity = Some(
                        runner_up_similarity
                            .map(|runner_up| runner_up.max(similarity))
                            .unwrap_or(similarity),
                    );
                }
                Some(current) => {
                    runner_up_similarity = Some(
                        runner_up_similarity
                            .map(|runner_up| runner_up.max(current.similarity))
                            .unwrap_or(current.similarity),
                    );
                    best = Some(candidate);
                }
                None => {
                    best = Some(candidate);
                }
            }
        }

        let mut best = best?;
        if best.similarity + threshold_tolerance < best.threshold {
            return None;
        }
        if let Some(runner_up) = runner_up_similarity {
            best.runner_up_similarity = Some(runner_up);
            if best.similarity - runner_up < margin {
                return None;
            }
        }
        Some(best)
    }
}

pub fn load_store_from_default_path() -> Result<VoiceprintStore, VoiceprintStoreError> {
    let Some(path) = voiceprint_store_path() else {
        return Ok(VoiceprintStore::default());
    };
    VoiceprintStore::load(&path)
}

pub fn save_store_to_default_path(
    store: &VoiceprintStore,
) -> Result<PathBuf, VoiceprintStoreError> {
    let path = voiceprint_store_path().ok_or_else(|| VoiceprintStoreError::Write {
        path: PathBuf::from("<openasr-home-unavailable>"),
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "could not determine OpenASR home",
        ),
    })?;
    store.save(&path)?;
    Ok(path)
}

pub fn create_profile_from_wav_file(
    path: &Path,
    name: impl Into<String>,
    match_similarity: Option<f32>,
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    let samples = load_enrollment_wav(path)?;
    create_profile_from_samples(&samples, name, match_similarity)
}

pub fn create_profile_from_samples(
    samples: &[f32],
    name: impl Into<String>,
    match_similarity: Option<f32>,
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    let sample_seconds = validate_enrollment_speech(samples)?;
    let embedder = shared_embedder().ok_or(SpeakerEnrollmentError::EmbedderPackMissing)?;
    let identity = super::embed::shared_embedder_identity()
        .ok_or(SpeakerEnrollmentError::EmbedderPackMissing)?
        .clone();
    create_profile_from_samples_with_embedder_and_seconds(
        samples,
        name,
        match_similarity,
        sample_seconds,
        embedder,
        &identity,
    )
}

pub fn create_profile_from_samples_with_embedder(
    samples: &[f32],
    name: impl Into<String>,
    match_similarity: Option<f32>,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    let sample_seconds = validate_enrollment_speech(samples)?;
    create_profile_from_samples_with_embedder_and_seconds(
        samples,
        name,
        match_similarity,
        sample_seconds,
        embedder,
        identity,
    )
}

fn create_profile_from_samples_with_embedder_and_seconds(
    samples: &[f32],
    name: impl Into<String>,
    match_similarity: Option<f32>,
    sample_seconds: f32,
    embedder: &dyn SpeakerEmbedder,
    identity: &SpeakerEmbedderIdentity,
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    let name = normalize_profile_name(name.into())?;
    let match_similarity =
        validate_match_similarity(match_similarity.unwrap_or(DEFAULT_MATCH_SIMILARITY))?;
    let embedding = embedding_from_enrollment_audio(embedder, samples)?;
    Ok(profile_from_embedding(
        name,
        sample_seconds,
        match_similarity,
        embedding,
        identity,
    ))
}

pub fn add_profile_to_default_store(
    profile: SpeakerProfile,
) -> Result<PathBuf, VoiceprintStoreError> {
    let path = voiceprint_store_path().ok_or_else(|| VoiceprintStoreError::Write {
        path: PathBuf::from("<openasr-home-unavailable>"),
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "could not determine OpenASR home",
        ),
    })?;
    let mut store = VoiceprintStore::load(&path)?;
    store.add_profile(profile);
    store.save(&path)?;
    Ok(path)
}

pub fn rename_profile_in_store(
    path: &Path,
    id: &str,
    name: impl Into<String>,
) -> Result<SpeakerProfile, VoiceprintStoreError> {
    validate_profile_id(id)?;
    let name = normalize_profile_name(name.into())?;
    let mut store = VoiceprintStore::load(path)?;
    let profile = store
        .profile_mut(id)
        .ok_or_else(|| VoiceprintStoreError::NotFound(id.to_string()))?;
    profile.name = name;
    profile.updated_at = timestamp_now();
    let response = profile.clone();
    store.save(path)?;
    Ok(response)
}

pub fn delete_profile_from_store(
    path: &Path,
    id: &str,
) -> Result<SpeakerProfile, VoiceprintStoreError> {
    validate_profile_id(id)?;
    let mut store = VoiceprintStore::load(path)?;
    let removed = store
        .remove_profile(id)
        .ok_or_else(|| VoiceprintStoreError::NotFound(id.to_string()))?;
    store.save(path)?;
    Ok(removed)
}

pub fn replace_profile_embedding_from_samples(
    path: &Path,
    id: &str,
    samples: &[f32],
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    validate_profile_id(id)?;
    let sample_seconds = validate_enrollment_speech(samples)?;
    let embedder = shared_embedder().ok_or(SpeakerEnrollmentError::EmbedderPackMissing)?;
    let identity = super::embed::shared_embedder_identity()
        .ok_or(SpeakerEnrollmentError::EmbedderPackMissing)?
        .clone();
    let embedding = embedding_from_enrollment_audio(embedder, samples)?;
    let mut store = VoiceprintStore::load(path)?;
    let profile = store
        .profile_mut(id)
        .ok_or_else(|| VoiceprintStoreError::NotFound(id.to_string()))?;
    profile.sample_seconds = sample_seconds;
    profile.embedding_dim = identity.embedding_dim;
    profile.pack_fingerprint = identity.pack_fingerprint;
    profile.embedding = embedding.0;
    profile.updated_at = timestamp_now();
    let response = profile.clone();
    store.save(path)?;
    Ok(response)
}

pub fn replace_profile_embedding_from_wav_file(
    store_path: &Path,
    id: &str,
    wav_path: &Path,
) -> Result<SpeakerProfile, SpeakerEnrollmentError> {
    let samples = load_enrollment_wav(wav_path)?;
    replace_profile_embedding_from_samples(store_path, id, &samples)
}

pub fn load_compatible_profile_matcher_for_active_embedder() -> SpeakerProfileMatcher {
    let Some(path) = voiceprint_store_path() else {
        return SpeakerProfileMatcher::default();
    };
    let Some(identity) = super::embed::shared_embedder_identity() else {
        return SpeakerProfileMatcher::default();
    };
    SpeakerProfileMatcher::load_for_identity(&path, identity).unwrap_or_default()
}

fn load_enrollment_wav(path: &Path) -> Result<Vec<f32>, SpeakerEnrollmentError> {
    crate::api::audio_io::load_wav_16khz_mono_f32_v0(path, "speaker enrollment", path_label(path))
        .map_err(|error| SpeakerEnrollmentError::InvalidAudio(error.to_string()))
}

fn path_label(path: &Path) -> &str {
    path.to_str().unwrap_or("speaker enrollment input")
}

fn embedding_from_enrollment_audio(
    embedder: &dyn SpeakerEmbedder,
    samples: &[f32],
) -> Result<SpeakerEmbedding, SpeakerEnrollmentError> {
    let speech = super::pipeline::resolve_speech_regions(samples)
        .filter(|regions| !regions.is_empty())
        .ok_or(SpeakerEnrollmentError::NoSpeech)?;
    let clusterer = super::clustering::AgglomerativeClusterer::for_embedder(embedder);
    let diarization = super::pipeline::BatchDiarizer::new(embedder, &clusterer).diarize(
        samples,
        16_000,
        &speech,
        super::contract::DiarizeHint::NumSpeakers(1),
    );
    let (_, centroid) =
        diarization
            .centroids
            .into_iter()
            .next()
            .ok_or(SpeakerEnrollmentError::TooShortSpeech {
                required: MIN_ENROLLMENT_SPEECH_SECONDS,
                actual: 0.0,
            })?;
    Ok(centroid)
}

fn profile_from_embedding(
    name: String,
    sample_seconds: f32,
    match_similarity: f32,
    embedding: SpeakerEmbedding,
    identity: &SpeakerEmbedderIdentity,
) -> SpeakerProfile {
    let now = timestamp_now();
    SpeakerProfile {
        id: generate_profile_id(),
        name,
        created_at: now.clone(),
        updated_at: now,
        sample_seconds,
        embedding_dim: identity.embedding_dim,
        pack_fingerprint: identity.pack_fingerprint.clone(),
        match_similarity,
        embedding: embedding.0,
    }
}

fn normalize_profile_name(name: String) -> Result<String, VoiceprintStoreError> {
    let trimmed = name.trim().to_string();
    if trimmed.is_empty() {
        Err(VoiceprintStoreError::EmptyName)
    } else {
        Ok(trimmed)
    }
}

fn validate_profile_id(id: &str) -> Result<(), VoiceprintStoreError> {
    if id.starts_with(SPEAKER_PROFILE_ID_PREFIX)
        && id[SPEAKER_PROFILE_ID_PREFIX.len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        Ok(())
    } else {
        Err(VoiceprintStoreError::InvalidId(id.to_string()))
    }
}

fn validate_match_similarity(value: f32) -> Result<f32, VoiceprintStoreError> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(VoiceprintStoreError::InvalidMatchSimilarity)
    }
}

pub fn validate_enrollment_speech(samples: &[f32]) -> Result<f32, SpeakerEnrollmentError> {
    let speech_seconds = speech_like_seconds(samples);
    if speech_seconds <= f32::EPSILON {
        return Err(SpeakerEnrollmentError::NoSpeech);
    }
    if speech_seconds < MIN_ENROLLMENT_SPEECH_SECONDS {
        return Err(SpeakerEnrollmentError::TooShortSpeech {
            required: MIN_ENROLLMENT_SPEECH_SECONDS,
            actual: speech_seconds,
        });
    }
    Ok(speech_seconds)
}

fn speech_like_seconds(samples: &[f32]) -> f32 {
    const SAMPLE_RATE_HZ: usize = 16_000;
    const FRAME_SAMPLES: usize = SAMPLE_RATE_HZ / 50; // 20 ms.
    const RMS_SPEECH_FLOOR: f32 = 0.01;

    samples
        .chunks(FRAME_SAMPLES)
        .filter(|chunk| {
            if chunk.is_empty() {
                return false;
            }
            let rms = (chunk.iter().map(|sample| sample * sample).sum::<f32>()
                / chunk.len() as f32)
                .sqrt();
            rms >= RMS_SPEECH_FLOOR
        })
        .map(|chunk| chunk.len() as f32 / SAMPLE_RATE_HZ as f32)
        .sum()
}

fn generate_profile_id() -> String {
    let counter = PROFILE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(now.to_le_bytes());
    hasher.update(counter.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("{SPEAKER_PROFILE_ID_PREFIX}{}", &digest[..16])
}

fn timestamp_now() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format_unix_millis(duration.as_secs(), duration.subsec_millis()),
        Err(_) => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

fn format_unix_millis(seconds: u64, millis: u32) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

fn write_owner_only_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    crate::atomic_file::write_owner_only_file_atomically(path, bytes)
}

fn set_owner_only_dir_permissions(#[cfg_attr(not(unix), allow(unused_variables))] path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(dim: usize, fingerprint: &str) -> SpeakerEmbedderIdentity {
        SpeakerEmbedderIdentity {
            embedding_dim: dim,
            pack_fingerprint: fingerprint.to_string(),
        }
    }

    fn profile(id: &str, dim: usize, fingerprint: &str, embedding: Vec<f32>) -> SpeakerProfile {
        SpeakerProfile {
            id: id.to_string(),
            name: "Alice".to_string(),
            created_at: "2026-06-11T00:00:00.000Z".to_string(),
            updated_at: "2026-06-11T00:00:00.000Z".to_string(),
            sample_seconds: 5.2,
            embedding_dim: dim,
            pack_fingerprint: fingerprint.to_string(),
            match_similarity: 0.5,
            embedding,
        }
    }

    #[test]
    fn store_roundtrip_persists_embeddings_only() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("voiceprints.json");
        let mut store = VoiceprintStore::default();
        store.add_profile(profile(
            "vp_aaaaaaaaaaaaaaaa",
            2,
            "sha256:pack",
            vec![0.6, 0.8],
        ));

        store.save(&path).unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"version\""));
        assert!(raw.contains("\"embedding\""));
        assert!(!raw.contains("audio"));
        assert!(!raw.contains("wav"));

        let loaded = VoiceprintStore::load(&path).unwrap();
        assert_eq!(loaded.profiles.len(), 1);
        assert_eq!(loaded.profiles[0].embedding, vec![0.6, 0.8]);
        assert_eq!(loaded.profiles[0].embedding_dim, 2);
        assert_eq!(loaded.profiles[0].pack_fingerprint, "sha256:pack");
    }

    #[cfg(unix)]
    #[test]
    fn store_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested").join("voiceprints.json");
        VoiceprintStore::default().save(&path).unwrap();

        let file_mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        let dir_mode = fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(file_mode, 0o600);
        assert_eq!(dir_mode, 0o700);
    }

    #[test]
    fn store_save_replaces_existing_file_without_leftover_temp() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("diarize").join("voiceprints.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"old voiceprint store").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).unwrap();
        }

        let mut store = VoiceprintStore::default();
        store.add_profile(profile(
            "vp_aaaaaaaaaaaaaaaa",
            2,
            "sha256:pack",
            vec![0.6, 0.8],
        ));
        store.save(&path).unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"id\": \"vp_aaaaaaaaaaaaaaaa\""));
        assert!(!raw.contains("old voiceprint store"));
        let entries: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("voiceprints.json")]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn speech_validation_rejects_silent_and_short_audio() {
        let silent = vec![0.0; 16_000 * 6];
        assert!(matches!(
            validate_enrollment_speech(&silent),
            Err(SpeakerEnrollmentError::NoSpeech)
        ));

        let short = vec![0.1; 16_000 * 4];
        assert!(matches!(
            validate_enrollment_speech(&short),
            Err(SpeakerEnrollmentError::TooShortSpeech { .. })
        ));
    }

    #[test]
    fn profile_matcher_skips_dimension_and_pack_mismatches() {
        let active = identity(2, "sha256:active");
        let store = VoiceprintStore {
            version: VOICEPRINT_STORE_VERSION,
            profiles: vec![
                profile("vp_aaaaaaaaaaaaaaaa", 2, "sha256:active", vec![1.0, 0.0]),
                profile(
                    "vp_bbbbbbbbbbbbbbbb",
                    3,
                    "sha256:active",
                    vec![1.0, 0.0, 0.0],
                ),
                profile("vp_cccccccccccccccc", 2, "sha256:other", vec![1.0, 0.0]),
            ],
        };
        let compatible = store.compatible_profiles(&active);
        assert_eq!(compatible.len(), 1);
        assert_eq!(compatible[0].id, "vp_aaaaaaaaaaaaaaaa");

        let matcher = SpeakerProfileMatcher::from_profiles(
            store
                .profiles
                .into_iter()
                .filter(|profile| profile.is_compatible_with(&active))
                .collect(),
        );
        let matched = matcher
            .best_match(&SpeakerEmbedding::l2_normalized(vec![0.95, 0.05]))
            .unwrap();
        assert_eq!(matched.profile_id, "vp_aaaaaaaaaaaaaaaa");
    }

    #[test]
    fn strong_profile_match_requires_anchor_floor_and_runner_up_margin() {
        let threshold_only = SpeakerProfileMatcher::from_profiles(vec![profile(
            "vp_aaaaaaaaaaaaaaaa",
            2,
            "sha256:active",
            vec![1.0, 0.0],
        )]);
        assert!(
            threshold_only
                .best_match(&SpeakerEmbedding::l2_normalized(vec![0.80, 0.60]))
                .is_some(),
            "the existing match policy still honors the profile threshold"
        );
        assert!(
            threshold_only
                .strong_unambiguous_match(
                    &SpeakerEmbedding::l2_normalized(vec![0.80, 0.60]),
                    0.85,
                    0.08
                )
                .is_none(),
            "streaming anchors require the higher anchor floor"
        );

        let ambiguous = SpeakerProfileMatcher::from_profiles(vec![
            profile("vp_aaaaaaaaaaaaaaaa", 2, "sha256:active", vec![1.0, 0.0]),
            profile("vp_bbbbbbbbbbbbbbbb", 2, "sha256:active", vec![0.96, 0.28]),
        ]);
        let embedding = SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]);
        assert!(
            ambiguous
                .strong_unambiguous_match(&embedding, 0.85, 0.08)
                .is_none(),
            "near-tied profiles are not unambiguous enough to anchor"
        );

        let clear = SpeakerProfileMatcher::from_profiles(vec![
            profile("vp_aaaaaaaaaaaaaaaa", 2, "sha256:active", vec![1.0, 0.0]),
            profile("vp_bbbbbbbbbbbbbbbb", 2, "sha256:active", vec![0.0, 1.0]),
        ]);
        let matched = clear
            .strong_unambiguous_match(&embedding, 0.85, 0.08)
            .expect("clear high-confidence profile match");
        assert_eq!(matched.profile_id, "vp_aaaaaaaaaaaaaaaa");
        assert_eq!(matched.runner_up_similarity, Some(0.0));
    }

    #[test]
    fn strong_profile_match_tolerance_accepts_borderline_threshold_only() {
        let matcher = SpeakerProfileMatcher::from_profiles(vec![profile(
            "vp_aaaaaaaaaaaaaaaa",
            2,
            "sha256:active",
            vec![1.0, 0.0],
        )]);
        let borderline = SpeakerEmbedding::l2_normalized(vec![0.849, 0.5283923]);
        assert!(
            matcher
                .strong_unambiguous_match(&borderline, 0.85, 0.08)
                .is_none()
        );
        let matched = matcher
            .strong_unambiguous_match_with_tolerance(&borderline, 0.85, 0.08, 0.01)
            .expect("small realtime tolerance accepts a just-under-threshold profile");
        assert_eq!(matched.profile_id, "vp_aaaaaaaaaaaaaaaa");

        let below_tolerance = SpeakerEmbedding::l2_normalized(vec![0.835, 0.5502504]);
        assert!(
            matcher
                .strong_unambiguous_match_with_tolerance(&below_tolerance, 0.85, 0.08, 0.01)
                .is_none(),
            "tolerance is bounded and does not lower the anchor floor broadly"
        );
    }

    #[test]
    fn compatibility_status_reports_clear_reason() {
        let active = identity(2, "sha256:active");
        let mismatched = profile("vp_aaaaaaaaaaaaaaaa", 2, "sha256:other", vec![1.0, 0.0]);
        assert_eq!(
            mismatched.compatibility_status(&active),
            ProfileCompatibility::Incompatible {
                reason: "embedder pack fingerprint mismatch".to_string()
            }
        );
    }

    /// Pins the exact behavior a ReDimNet2-B6 cutover relies on: a voiceprint
    /// registered under the legacy 256-dim WeSpeaker embedder must never be
    /// silently reused once the active embedder resolves to the 192-dim
    /// ReDimNet2 pack -- it needs explicit re-registration, not a dimension
    /// mismatch crash or (worse) a wrong-space cosine comparison.
    #[test]
    fn old_wespeaker_profile_is_incompatible_with_new_redimnet_embedder() {
        let legacy_wespeaker_profile = profile(
            "vp_aaaaaaaaaaaaaaaa",
            256,
            "sha256:wespeaker-resnet34-pack",
            vec![0.1; 256],
        );
        let active_redimnet_identity = identity(192, "sha256:redimnet2-b6-pack");

        assert!(
            !legacy_wespeaker_profile.is_compatible_with(&active_redimnet_identity),
            "a 256-dim WeSpeaker profile must not be treated as compatible with a 192-dim ReDimNet2 identity"
        );
        assert_eq!(
            legacy_wespeaker_profile.compatibility_status(&active_redimnet_identity),
            ProfileCompatibility::Incompatible {
                reason: "embedding dimension mismatch: profile has 256, active embedder has 192"
                    .to_string()
            },
            "the reason must name both dimensions so a user/operator understands \
             re-registration (not a bug) is required"
        );

        let mut store = VoiceprintStore::default();
        store.add_profile(legacy_wespeaker_profile);
        assert!(
            store
                .compatible_profiles(&active_redimnet_identity)
                .is_empty(),
            "the voiceprint store must drop the incompatible profile rather than \
             attempt a cross-embedding-space comparison"
        );
    }
}
