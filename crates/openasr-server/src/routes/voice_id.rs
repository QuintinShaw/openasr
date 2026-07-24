//! Operator-only Voice ID v2 routes (`/v1/voice-id/*`).

use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Serialize)]
pub(crate) struct PersonListResponse {
    pub data: Vec<openasr_core::diarize::voice_id::PersonView>,
    pub revision: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct DeleteResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PatchPersonRequest {
    pub display_name: Option<String>,
    pub color_preference: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PatchSampleRequest {
    #[allow(dead_code)]
    pub sample_label: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RevokeConsentRequest {
    #[serde(default)]
    pub reason: Option<String>,
}

pub(crate) async fn list_persons(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<(HeaderMap, Json<PersonListResponse>), ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let active = active_space();
    let data = store
        .list_persons(active.as_ref())
        .map_err(voice_id_store_error)?;
    let revision = global_revision_etag(&store);
    let mut headers = HeaderMap::new();
    if let Ok(value) = HeaderValue::from_str(&revision) {
        headers.insert(header::ETAG, value);
    }
    Ok((headers, Json(PersonListResponse { data, revision })))
}

pub(crate) async fn get_person(
    Extension(distribution): Extension<DistributionContext>,
    AxumPath(person_id): AxumPath<String>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let person = store
        .get_person(&id, active_space().as_ref())
        .map_err(voice_id_store_error)?;
    let mut headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        headers.insert(header::ETAG, value);
    }
    Ok((headers, Json(person)))
}

pub(crate) async fn enroll_person(
    Extension(distribution): Extension<DistributionContext>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<
    (
        StatusCode,
        HeaderMap,
        Json<openasr_core::diarize::voice_id::PersonView>,
    ),
    ApiError,
> {
    let parsed = parse_enroll_multipart(multipart).await?;
    let store = open_voice_id_store(&distribution)?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let person = openasr_core::diarize::voice_id::enroll_person_from_clips(
        &store,
        parsed.display_name,
        parsed.consent,
        parsed.clips,
        embedder,
        &identity,
        parsed.color_preference,
    )
    .map_err(voice_id_service_error)?;
    let mut headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        headers.insert(header::ETAG, value);
    }
    Ok((StatusCode::CREATED, headers, Json(person)))
}

pub(crate) async fn patch_person(
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(person_id): AxumPath<String>,
    Json(request): Json<PatchPersonRequest>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let Some(display_name) = request.display_name else {
        return Err(ApiError::BadRequest(
            "PATCH currently supports display_name only".into(),
        ));
    };
    let _ = request.color_preference; // reserved for app presentation metadata
    let person = store
        .rename_person(&id, display_name, expected)
        .map_err(voice_id_store_error)?;
    let mut out_headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        out_headers.insert(header::ETAG, value);
    }
    Ok((out_headers, Json(person)))
}

pub(crate) async fn delete_person(
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(person_id): AxumPath<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    store
        .delete_person(&id, expected, "api_delete")
        .map_err(voice_id_store_error)?;
    Ok(Json(DeleteResponse {
        id: person_id,
        deleted: true,
    }))
}

pub(crate) async fn add_sample(
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(person_id): AxumPath<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let parsed = parse_sample_multipart(multipart).await?;
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let person = openasr_core::diarize::voice_id::add_sample_from_pcm(
        &store,
        &id,
        expected,
        parsed.consent,
        &parsed.pcm,
        parsed.capture_context,
        embedder,
        &identity,
    )
    .map_err(voice_id_service_error)?;
    let mut out_headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        out_headers.insert(header::ETAG, value);
    }
    Ok((out_headers, Json(person)))
}

pub(crate) async fn delete_sample(
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(sample_id): AxumPath<String>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::SampleId::parse(&sample_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let person = store
        .delete_sample(&id, expected)
        .map_err(voice_id_store_error)?;
    let mut out_headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        out_headers.insert(header::ETAG, value);
    }
    Ok((out_headers, Json(person)))
}

pub(crate) async fn patch_sample(
    Extension(_distribution): Extension<DistributionContext>,
    AxumPath(_sample_id): AxumPath<String>,
    Json(_request): Json<PatchSampleRequest>,
) -> Result<StatusCode, ApiError> {
    // Sample label metadata edits land with a dedicated store method in a
    // follow-up; reject rather than silently no-op.
    Err(ApiError::BadRequest(
        "sample metadata PATCH is not implemented in this build".into(),
    ))
}

pub(crate) async fn revoke_consent(
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(person_id): AxumPath<String>,
    Json(request): Json<RevokeConsentRequest>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let reason = request
        .reason
        .unwrap_or_else(|| "consent_revoked".to_string());
    store
        .revoke_consent(&id, expected, &reason)
        .map_err(voice_id_store_error)?;
    Ok(Json(DeleteResponse {
        id: person_id,
        deleted: true,
    }))
}

pub(crate) async fn export_metadata(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let json = store.export_metadata_json().map_err(voice_id_store_error)?;
    let value: serde_json::Value =
        serde_json::from_str(&json).map_err(|e| ApiError::JobStore(e.to_string()))?;
    Ok(Json(value))
}

struct ParsedEnroll {
    display_name: String,
    consent: openasr_core::diarize::voice_id::ConsentRecord,
    color_preference: Option<String>,
    clips: Vec<openasr_core::diarize::voice_id::EnrollmentClip>,
}

struct ParsedSample {
    consent: openasr_core::diarize::voice_id::ConsentRecord,
    capture_context: openasr_core::diarize::voice_id::CaptureContext,
    pcm: Vec<f32>,
}

async fn parse_enroll_multipart(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<ParsedEnroll, ApiError> {
    let mut multipart = multipart.map_err(ApiError::MultipartRejection)?;
    let mut display_name: Option<String> = None;
    let mut notice_version = "voice-id-notice-v1".to_string();
    let mut capture_method = "upload".to_string();
    let mut color_preference = None;
    let mut device_class = "unknown".to_string();
    let mut input_route = "unknown".to_string();
    let mut environment_hint = None;
    let mut wav_paths: Vec<tempfile::TempPath> = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(ApiError::Multipart)? {
        match field.name().unwrap_or_default() {
            "display_name" | "name" => {
                display_name = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "notice_version" => {
                notice_version = field.text().await.map_err(ApiError::Multipart)?;
            }
            "capture_method" => {
                capture_method = field.text().await.map_err(ApiError::Multipart)?;
            }
            "color_preference" => {
                color_preference = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "device_class" => {
                device_class = field.text().await.map_err(ApiError::Multipart)?;
            }
            "input_route" => {
                input_route = field.text().await.map_err(ApiError::Multipart)?;
            }
            "environment_hint" => {
                environment_hint = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "wav" | "sample" | "samples" => {
                let bytes = field.bytes().await.map_err(ApiError::Multipart)?;
                wav_paths.push(write_upload_temp_file(&bytes, ".wav")?);
            }
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
    }

    let display_name = display_name
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ApiError::BadRequest("Missing required form field: display_name".into()))?;
    if wav_paths.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing required form field: wav (one or more enrollment samples)".into(),
        ));
    }
    if wav_paths.len() > 5 {
        return Err(ApiError::BadRequest(
            "Initial enrollment accepts at most 5 samples".into(),
        ));
    }

    // Prepare all clips first; any failure leaves zero DB writes.
    let mut clips = Vec::with_capacity(wav_paths.len());
    for (idx, path) in wav_paths.iter().enumerate() {
        // Load via enrollment helper path (public) by reading bytes through
        // the same WAV loader the core enrollment path uses.
        let pcm = load_enrollment_wav(path.as_ref())?;
        clips.push(openasr_core::diarize::voice_id::EnrollmentClip {
            samples: pcm,
            capture_context: openasr_core::diarize::voice_id::CaptureContext {
                device_class: device_class.clone(),
                input_route: input_route.clone(),
                environment_hint: environment_hint.clone(),
                sample_label: Some(format!("enrollment-{}", idx + 1)),
            },
        });
    }

    let consent = openasr_core::diarize::voice_id::ConsentRecord {
        // Server-side clock; do not trust client timestamps for consent.
        granted_at: openasr_core::diarize::voice_id::timestamp_now(),
        notice_version,
        capture_method,
    };
    Ok(ParsedEnroll {
        display_name,
        consent,
        color_preference,
        clips,
    })
}

async fn parse_sample_multipart(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<ParsedSample, ApiError> {
    let mut multipart = multipart.map_err(ApiError::MultipartRejection)?;
    let mut notice_version = "voice-id-notice-v1".to_string();
    let mut capture_method = "upload".to_string();
    let mut device_class = "unknown".to_string();
    let mut input_route = "unknown".to_string();
    let mut environment_hint = None;
    let mut sample_label = None;
    let mut wav_path: Option<tempfile::TempPath> = None;

    while let Some(field) = multipart.next_field().await.map_err(ApiError::Multipart)? {
        match field.name().unwrap_or_default() {
            "notice_version" => {
                notice_version = field.text().await.map_err(ApiError::Multipart)?;
            }
            "capture_method" => {
                capture_method = field.text().await.map_err(ApiError::Multipart)?;
            }
            "device_class" => {
                device_class = field.text().await.map_err(ApiError::Multipart)?;
            }
            "input_route" => {
                input_route = field.text().await.map_err(ApiError::Multipart)?;
            }
            "environment_hint" => {
                environment_hint = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "sample_label" => {
                sample_label = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "wav" | "sample" => {
                let bytes = field.bytes().await.map_err(ApiError::Multipart)?;
                wav_path = Some(write_upload_temp_file(&bytes, ".wav")?);
            }
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
    }
    let Some(wav_path) = wav_path else {
        return Err(ApiError::BadRequest(
            "Missing required form field: wav".into(),
        ));
    };
    let pcm = load_enrollment_wav(wav_path.as_ref())?;
    Ok(ParsedSample {
        consent: openasr_core::diarize::voice_id::ConsentRecord {
            granted_at: openasr_core::diarize::voice_id::timestamp_now(),
            notice_version,
            capture_method,
        },
        capture_context: openasr_core::diarize::voice_id::CaptureContext {
            device_class,
            input_route,
            environment_hint,
            sample_label,
        },
        pcm,
    })
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

fn parse_if_match(headers: &HeaderMap) -> Result<Option<u64>, ApiError> {
    let Some(value) = headers.get(header::IF_MATCH) else {
        return Ok(None);
    };
    let raw = value
        .to_str()
        .map_err(|_| ApiError::BadRequest("Invalid If-Match header".into()))?
        .trim()
        .trim_matches('"');
    let revision = raw
        .parse::<u64>()
        .map_err(|_| ApiError::BadRequest(format!("Invalid If-Match revision '{raw}'")))?;
    Ok(Some(revision))
}

fn global_revision_etag(store: &openasr_core::diarize::voice_id::VoiceIdStore) -> String {
    let rev = store
        .migration_state("global_revision")
        .ok()
        .flatten()
        .unwrap_or_else(|| "0".into());
    format!("\"{rev}\"")
}

fn load_enrollment_wav(path: &std::path::Path) -> Result<Vec<f32>, ApiError> {
    openasr_core::load_native_wav_16khz_mono_f32_v0(
        path,
        "voice-id enrollment",
        path.to_str().unwrap_or("voice-id enrollment input"),
    )
    .map_err(|e| ApiError::BadRequest(e.to_string()))
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
