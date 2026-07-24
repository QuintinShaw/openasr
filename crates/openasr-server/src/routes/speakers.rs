//! Operator-only speaker voice-match profile routes.
//!
//! Compatibility surface over Voice ID v2. List/create/rename/delete project
//! onto the SQLite person store; reenroll stays rejected because multi-sample
//! persons must add samples via `/v1/voice-id`. After v1 migration is DONE the
//! legacy `voiceprints.json` file is never written.

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
    let store = open_voice_id_store(&distribution)?;
    let active = active_space();
    let persons = store
        .list_persons(active.as_ref())
        .map_err(voice_id_store_error)?;
    let mut data = Vec::with_capacity(persons.len());
    for person in persons {
        let person_id = openasr_core::diarize::voice_id::PersonId::parse(&person.person_id)
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let public_id = store
            .preferred_public_id(&person_id)
            .map_err(voice_id_store_error)?;
        data.push(person_to_view(&person, public_id));
    }
    Ok(Json(SpeakerListResponse { data }))
}

pub(crate) async fn create_speaker(
    Extension(distribution): Extension<DistributionContext>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(StatusCode, Json<SpeakerProfileView>), ApiError> {
    let parsed = parse_speaker_enrollment_multipart(multipart, true).await?;
    let pcm = load_enrollment_wav(parsed.wav_path.as_ref())?;
    // Quality-gate before pack lookup so short/silent clips surface the more
    // specific audio error even when the embedder pack is not installed.
    openasr_core::diarize::voice_id::assess_enrollment_quality(&pcm)
        .map_err(|error| ApiError::BadRequest(error.to_string()))?;
    let store = open_voice_id_store(&distribution)?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let name = parsed.name.expect("name is required for create");
    let person = openasr_core::diarize::voice_id::enroll_person_from_clips(
        &store,
        name,
        openasr_core::diarize::voice_id::ConsentRecord {
            granted_at: openasr_core::diarize::voice_id::timestamp_now(),
            notice_version: "legacy-speakers-api-v1".into(),
            capture_method: "speakers-api".into(),
        },
        vec![openasr_core::diarize::voice_id::EnrollmentClip {
            samples: pcm,
            capture_context: openasr_core::diarize::voice_id::CaptureContext {
                device_class: "unknown".into(),
                input_route: "speakers-api".into(),
                environment_hint: None,
                sample_label: Some("speakers-api enrollment".into()),
            },
        }],
        embedder,
        &identity,
        None,
    )
    .map_err(voice_id_service_error)?;
    let person_id = openasr_core::diarize::voice_id::PersonId::parse(&person.person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let public_id = store
        .preferred_public_id(&person_id)
        .map_err(voice_id_store_error)?;
    Ok((
        StatusCode::CREATED,
        Json(person_to_view(&person, public_id)),
    ))
}

pub(crate) async fn rename_speaker(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<RenameSpeakerRequest>,
) -> Result<Json<SpeakerProfileView>, ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let person_id = store
        .resolve_person_ref(&id)
        .map_err(voice_id_store_error)?;
    let person = store
        .rename_person(&person_id, request.name, None)
        .map_err(voice_id_store_error)?;
    let public_id = store
        .preferred_public_id(&person_id)
        .map_err(voice_id_store_error)?;
    Ok(Json(person_to_view(&person, public_id)))
}

pub(crate) async fn delete_speaker(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<SpeakerDeleteResponse>, ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let person_id = store
        .resolve_person_ref(&id)
        .map_err(voice_id_store_error)?;
    let public_id = store
        .preferred_public_id(&person_id)
        .map_err(voice_id_store_error)?;
    store
        .delete_person(&person_id, None, "speakers_api_delete")
        .map_err(voice_id_store_error)?;
    Ok(Json(SpeakerDeleteResponse {
        id: public_id,
        deleted: true,
    }))
}

pub(crate) async fn reenroll_speaker(
    Extension(_distribution): Extension<DistributionContext>,
    AxumPath(_id): AxumPath<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<SpeakerProfileView>, ApiError> {
    // Consume multipart so clients do not hang, then reject: multi-sample
    // Voice ID persons must add samples via /v1/voice-id, not overwrite.
    let _ = parse_speaker_enrollment_multipart(multipart, false).await?;
    Err(ApiError::BadRequest(
        "reenroll is not supported for Voice ID v2 multi-sample persons; POST /v1/voice-id/persons/{id}/samples to add a sample or DELETE a sample and re-add".into(),
    ))
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

fn open_voice_id_store(
    distribution: &DistributionContext,
) -> Result<openasr_core::diarize::voice_id::VoiceIdStore, ApiError> {
    let home = distribution.openasr_home()?;
    openasr_core::diarize::voice_id::open_store_with_v1_migration(home)
        .map_err(|e| ApiError::JobStore(format!("voice-id store open/migration failed: {e}")))
}

fn active_space() -> Option<openasr_core::diarize::voice_id::EmbeddingSpace> {
    let identity = openasr_core::diarize::embed::shared_embedder_identity()?;
    let embedder = openasr_core::diarize::embed::shared_embedder()?;
    Some(
        openasr_core::diarize::voice_id::EmbeddingSpace::for_active_embedder(
            identity,
            embedder.calibration_profile(),
        ),
    )
}

fn active_embedder_and_identity() -> Result<
    (
        &'static dyn openasr_core::diarize::embed::SpeakerEmbedder,
        openasr_core::diarize::embed::SpeakerEmbedderIdentity,
    ),
    ApiError,
> {
    let embedder = openasr_core::diarize::embed::shared_embedder().ok_or_else(|| {
        ApiError::BadRequest(
            openasr_core::diarize::embed::VOICE_ID_EMBEDDER_PACK_MISSING_REASON.into(),
        )
    })?;
    let identity = openasr_core::diarize::embed::shared_embedder_identity()
        .cloned()
        .ok_or_else(|| {
            ApiError::BadRequest(
                openasr_core::diarize::embed::VOICE_ID_EMBEDDER_PACK_MISSING_REASON.into(),
            )
        })?;
    Ok((embedder, identity))
}

fn load_enrollment_wav(path: &std::path::Path) -> Result<Vec<f32>, ApiError> {
    openasr_core::load_native_wav_16khz_mono_f32_v0(
        path,
        "speaker enrollment",
        path.to_str().unwrap_or("speaker enrollment input"),
    )
    .map_err(|e| ApiError::BadRequest(e.to_string()))
}

fn person_to_view(
    person: &openasr_core::diarize::voice_id::PersonView,
    public_id: String,
) -> SpeakerProfileView {
    let sample_seconds = person
        .samples
        .iter()
        .map(|sample| sample.quality.speech_seconds)
        .sum::<f32>();
    SpeakerProfileView {
        id: public_id,
        name: person.display_name.clone(),
        created_at: person.created_at.clone(),
        sample_seconds,
        // Compatible with the active embedder when the person does not need
        // reenrollment for the currently loaded matching space.
        compatible: !person.needs_reenrollment,
    }
}

fn voice_id_store_error(error: openasr_core::diarize::voice_id::VoiceIdStoreError) -> ApiError {
    use openasr_core::diarize::voice_id::VoiceIdStoreError;
    match error {
        VoiceIdStoreError::NotFound(message) | VoiceIdStoreError::SampleNotFound(message) => {
            ApiError::NotFound(message)
        }
        VoiceIdStoreError::RevisionConflict { .. } => ApiError::Conflict(error.to_string()),
        VoiceIdStoreError::EmptyName
        | VoiceIdStoreError::InvalidId(_)
        | VoiceIdStoreError::NotActive(_)
        | VoiceIdStoreError::Migration(_) => ApiError::BadRequest(error.to_string()),
        other => ApiError::JobStore(other.to_string()),
    }
}

fn voice_id_service_error(error: openasr_core::diarize::voice_id::VoiceIdServiceError) -> ApiError {
    use openasr_core::diarize::voice_id::VoiceIdServiceError;
    match error {
        VoiceIdServiceError::Store(error) => voice_id_store_error(error),
        other => ApiError::BadRequest(other.to_string()),
    }
}
