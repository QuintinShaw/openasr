//! Operator-only speaker voice-match profile routes.

use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Serialize)]
pub(crate) struct SpeakerProfileView {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub sample_seconds: f32,
    pub compatible: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct SpeakerListResponse {
    pub data: Vec<SpeakerProfileView>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SpeakerDeleteResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RenameSpeakerRequest {
    pub name: String,
}

pub(crate) async fn list_speakers(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<SpeakerListResponse>, ApiError> {
    let path = speaker_store_path(&distribution)?;
    let store =
        openasr_core::diarize::enrollment::VoiceprintStore::load(&path).map_err(speaker_error)?;
    let active = openasr_core::diarize::embed::shared_embedder_identity().cloned();
    let data = store
        .profiles
        .iter()
        .map(|profile| SpeakerProfileView {
            id: profile.id.clone(),
            name: profile.name.clone(),
            created_at: profile.created_at.clone(),
            sample_seconds: profile.sample_seconds,
            compatible: active
                .as_ref()
                .is_some_and(|identity| profile.is_compatible_with(identity)),
        })
        .collect();
    Ok(Json(SpeakerListResponse { data }))
}

pub(crate) async fn create_speaker(
    Extension(distribution): Extension<DistributionContext>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(StatusCode, Json<SpeakerProfileView>), ApiError> {
    let parsed = parse_speaker_enrollment_multipart(multipart, true).await?;
    let path = speaker_store_path(&distribution)?;
    let profile = openasr_core::diarize::enrollment::create_profile_from_wav_file(
        parsed.wav_path.as_ref(),
        parsed.name.expect("name is required for create"),
        None,
    )
    .map_err(enrollment_error)?;
    let mut store =
        openasr_core::diarize::enrollment::VoiceprintStore::load(&path).map_err(speaker_error)?;
    store.add_profile(profile.clone());
    store.save(&path).map_err(speaker_error)?;
    Ok((StatusCode::CREATED, Json(profile_view(&profile, true))))
}

pub(crate) async fn rename_speaker(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<RenameSpeakerRequest>,
) -> Result<Json<SpeakerProfileView>, ApiError> {
    let path = speaker_store_path(&distribution)?;
    let profile =
        openasr_core::diarize::enrollment::rename_profile_in_store(&path, &id, request.name)
            .map_err(speaker_error)?;
    Ok(Json(profile_view_with_active_compatibility(&profile)))
}

pub(crate) async fn delete_speaker(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<SpeakerDeleteResponse>, ApiError> {
    let path = speaker_store_path(&distribution)?;
    let removed = openasr_core::diarize::enrollment::delete_profile_from_store(&path, &id)
        .map_err(speaker_error)?;
    Ok(Json(SpeakerDeleteResponse {
        id: removed.id,
        deleted: true,
    }))
}

pub(crate) async fn reenroll_speaker(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(id): AxumPath<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<SpeakerProfileView>, ApiError> {
    let parsed = parse_speaker_enrollment_multipart(multipart, false).await?;
    let path = speaker_store_path(&distribution)?;
    let profile = openasr_core::diarize::enrollment::replace_profile_embedding_from_wav_file(
        &path,
        &id,
        parsed.wav_path.as_ref(),
    )
    .map_err(enrollment_error)?;
    Ok(Json(profile_view_with_active_compatibility(&profile)))
}

struct ParsedSpeakerEnrollment {
    name: Option<String>,
    wav_path: tempfile::TempPath,
}

async fn parse_speaker_enrollment_multipart(
    multipart: Result<Multipart, MultipartRejection>,
    require_name: bool,
) -> Result<ParsedSpeakerEnrollment, ApiError> {
    let mut multipart = multipart.map_err(ApiError::MultipartRejection)?;
    let mut name: Option<String> = None;
    let mut wav_path: Option<tempfile::TempPath> = None;

    while let Some(field) = multipart.next_field().await.map_err(ApiError::Multipart)? {
        match field.name().unwrap_or_default() {
            "name" => {
                name = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "wav" => {
                let bytes = field.bytes().await.map_err(ApiError::Multipart)?;
                wav_path = Some(write_upload_temp_file(&bytes, ".wav")?);
            }
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
    }

    if require_name
        && name
            .as_deref()
            .map(|value| value.trim().is_empty())
            .unwrap_or(true)
    {
        return Err(ApiError::BadRequest(
            "Missing required form field: name".to_string(),
        ));
    }
    let Some(wav_path) = wav_path else {
        return Err(ApiError::BadRequest(
            "Missing required form field: wav".to_string(),
        ));
    };

    Ok(ParsedSpeakerEnrollment { name, wav_path })
}

fn speaker_store_path(distribution: &DistributionContext) -> Result<PathBuf, ApiError> {
    if let Ok(path) = std::env::var(openasr_core::diarize::enrollment::VOICEPRINT_STORE_ENV) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    Ok(distribution
        .openasr_home()?
        .join("diarize")
        .join("voiceprints.json"))
}

fn profile_view_with_active_compatibility(
    profile: &openasr_core::diarize::enrollment::SpeakerProfile,
) -> SpeakerProfileView {
    let compatible = openasr_core::diarize::embed::shared_embedder_identity()
        .is_some_and(|identity| profile.is_compatible_with(identity));
    profile_view(profile, compatible)
}

fn profile_view(
    profile: &openasr_core::diarize::enrollment::SpeakerProfile,
    compatible: bool,
) -> SpeakerProfileView {
    SpeakerProfileView {
        id: profile.id.clone(),
        name: profile.name.clone(),
        created_at: profile.created_at.clone(),
        sample_seconds: profile.sample_seconds,
        compatible,
    }
}

fn enrollment_error(error: openasr_core::diarize::enrollment::SpeakerEnrollmentError) -> ApiError {
    use openasr_core::diarize::enrollment::SpeakerEnrollmentError;
    match error {
        SpeakerEnrollmentError::Store(error) => speaker_error(error),
        other => ApiError::BadRequest(other.to_string()),
    }
}

fn speaker_error(error: openasr_core::diarize::enrollment::VoiceprintStoreError) -> ApiError {
    use openasr_core::diarize::enrollment::VoiceprintStoreError;
    match error {
        VoiceprintStoreError::NotFound(message) => ApiError::NotFound(message),
        VoiceprintStoreError::EmptyName
        | VoiceprintStoreError::InvalidId(_)
        | VoiceprintStoreError::InvalidMatchSimilarity
        | VoiceprintStoreError::UnsupportedVersion { .. } => {
            ApiError::BadRequest(error.to_string())
        }
        VoiceprintStoreError::Read { .. }
        | VoiceprintStoreError::Parse { .. }
        | VoiceprintStoreError::CreateDir { .. }
        | VoiceprintStoreError::Write { .. }
        | VoiceprintStoreError::Serialize(_) => ApiError::JobStore(error.to_string()),
    }
}
