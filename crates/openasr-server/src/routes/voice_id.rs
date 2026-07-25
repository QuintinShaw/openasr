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
    let person = store
        .update_person_metadata(
            &id,
            expected,
            openasr_core::diarize::voice_id::PersonMetadataUpdate {
                display_name: request.display_name,
                color_preference: request.color_preference.map(Some),
            },
        )
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
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(sample_id): AxumPath<String>,
    Json(request): Json<PatchSampleRequest>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let store = open_voice_id_store(&distribution)?;
    let id = openasr_core::diarize::voice_id::SampleId::parse(&sample_id)
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let sample_label = request
        .sample_label
        .ok_or_else(|| ApiError::BadRequest("PATCH requires sample_label".into()))?;
    let person = store
        .rename_sample(&id, sample_label, expected)
        .map_err(voice_id_store_error)?;
    let mut out_headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        out_headers.insert(header::ETAG, value);
    }
    Ok((out_headers, Json(person)))
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
    let mut sample_labels: Vec<String> = Vec::new();
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
            // Repeat this field once per WAV to label every sample, or provide
            // it once to label just the first sample.
            "sample_label" => {
                sample_labels.push(field.text().await.map_err(ApiError::Multipart)?);
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
    let sample_labels = resolve_initial_sample_labels(sample_labels, wav_paths.len())?;

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
                sample_label: Some(sample_labels[idx].clone()),
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

fn resolve_initial_sample_labels(
    sample_labels: Vec<String>,
    sample_count: usize,
) -> Result<Vec<String>, ApiError> {
    if sample_labels.len() > sample_count
        || (sample_labels.len() > 1 && sample_labels.len() != sample_count)
    {
        return Err(ApiError::BadRequest(
            "Provide one sample_label for the first sample, or one for every enrollment WAV".into(),
        ));
    }
    let mut resolved = (1..=sample_count)
        .map(|index| format!("enrollment-{index}"))
        .collect::<Vec<_>>();
    match sample_labels.as_slice() {
        [] => {}
        [first] => resolved[0] = first.clone(),
        labels => resolved.clone_from_slice(labels),
    }
    Ok(resolved)
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
        | VoiceIdStoreError::EmptySampleLabel
        | VoiceIdStoreError::LabelTooLong { .. }
        | VoiceIdStoreError::InvalidColorPreference(_)
        | VoiceIdStoreError::EmptyPersonMetadataUpdate
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

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        extract::FromRequest,
        http::{Request, header},
    };

    use super::{Multipart, parse_enroll_multipart, resolve_initial_sample_labels};

    #[test]
    fn initial_enrollment_sample_labels_preserve_client_values_and_fallbacks() {
        assert_eq!(
            resolve_initial_sample_labels(Vec::new(), 2).unwrap(),
            vec!["enrollment-1", "enrollment-2"]
        );
        assert_eq!(
            resolve_initial_sample_labels(vec!["First take".into()], 2).unwrap(),
            vec!["First take", "enrollment-2"]
        );
        assert_eq!(
            resolve_initial_sample_labels(vec!["Office".into(), "Car".into()], 2).unwrap(),
            vec!["Office", "Car"]
        );
        assert!(resolve_initial_sample_labels(vec!["one".into(), "two".into()], 3).is_err());
        assert!(resolve_initial_sample_labels(vec!["one".into(), "two".into()], 1).is_err());
    }

    #[tokio::test]
    async fn enrollment_multipart_assigns_first_and_per_sample_labels() {
        let first = parse_enroll(&["First take"]).await;
        assert_eq!(
            first.clips[0].capture_context.sample_label.as_deref(),
            Some("First take")
        );
        assert_eq!(
            first.clips[1].capture_context.sample_label.as_deref(),
            Some("enrollment-2")
        );

        let every = parse_enroll(&["Office", "Car"]).await;
        assert_eq!(
            every.clips[0].capture_context.sample_label.as_deref(),
            Some("Office")
        );
        assert_eq!(
            every.clips[1].capture_context.sample_label.as_deref(),
            Some("Car")
        );
    }

    async fn parse_enroll(sample_labels: &[&str]) -> super::ParsedEnroll {
        let boundary = "voice-id-test-boundary";
        let mut body = Vec::new();
        form_field(&mut body, boundary, "display_name", b"Alice");
        for sample_label in sample_labels {
            form_field(&mut body, boundary, "sample_label", sample_label.as_bytes());
        }
        for name in ["first.wav", "second.wav"] {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"wav\"; filename=\"{name}\"\r\nContent-Type: audio/wav\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(&pcm16_wav());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await;
        parse_enroll_multipart(multipart).await.unwrap()
    }

    fn form_field(body: &mut Vec<u8>, boundary: &str, name: &str, value: &[u8]) {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }

    fn pcm16_wav() -> Vec<u8> {
        let samples = 16_000u32;
        let data_bytes = samples * 2;
        let mut wav = Vec::with_capacity(44 + data_bytes as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&(16_000u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        wav.resize(44 + data_bytes as usize, 0);
        wav
    }
}
