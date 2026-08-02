//! Operator-only Voice ID v2 routes (`/v1/voice-id/*`).

use std::io::Write;

use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::*;

// A Voice ID sample is intended to be a short 16 kHz mono WAV. Eight MiB
// admits over four minutes of PCM16 while bounding a five-sample enrollment to
// forty MiB on disk and O(chunk) memory during upload.
const MAX_VOICE_ID_WAV_BYTES: u64 = 8 * 1024 * 1024;
/// Source-media enrollment accepts the same bounded temporary-upload model as
/// transcription, but never persists the source after this request completes.
const MAX_VOICE_ID_SOURCE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
struct UploadedWavFingerprint {
    sha256: String,
    bytes: u64,
}

struct UploadedVoiceIdWav {
    path: tempfile::TempPath,
    fingerprint: UploadedWavFingerprint,
}

struct UploadedVoiceIdSource {
    path: tempfile::TempPath,
    fingerprint: UploadedWavFingerprint,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SourceInterval {
    start: f32,
    end: f32,
}

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

#[derive(Debug, Default)]
pub(crate) enum PatchField<T> {
    #[default]
    Missing,
    Null,
    Value(T),
}

impl<'de, T> Deserialize<'de> for PatchField<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Value(value),
            None => Self::Null,
        })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct PatchPersonRequest {
    #[serde(default)]
    pub display_name: PatchField<String>,
    #[serde(default)]
    pub color_preference: PatchField<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PatchSampleRequest {
    #[serde(default)]
    pub sample_label: PatchField<String>,
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
    headers: HeaderMap,
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
    let idempotency = idempotency_request(&headers, enroll_request_hash(&parsed))?;
    let store = open_voice_id_store(&distribution)?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let person = match idempotency {
        Some(idempotency) => {
            openasr_core::diarize::voice_id::enroll_person_from_clips_idempotent(
                &store,
                parsed.display_name,
                parsed.consent,
                parsed.clips,
                embedder.as_ref(),
                &identity,
                parsed.color_preference,
                idempotency,
            )
            .map_err(voice_id_service_error)?
            .person
        }
        None => openasr_core::diarize::voice_id::enroll_person_from_clips(
            &store,
            parsed.display_name,
            parsed.consent,
            parsed.clips,
            embedder.as_ref(),
            &identity,
            parsed.color_preference,
        )
        .map_err(voice_id_service_error)?,
    };
    let mut headers = HeaderMap::new();
    let etag = format!("\"{}\"", person.revision);
    if let Ok(value) = HeaderValue::from_str(&etag) {
        headers.insert(header::ETAG, value);
    }
    Ok((StatusCode::CREATED, headers, Json(person)))
}

pub(crate) async fn enroll_person_from_source_audio(
    State(runtime): State<ServerRuntime>,
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<
    (
        StatusCode,
        HeaderMap,
        Json<openasr_core::diarize::voice_id::PersonView>,
    ),
    ApiError,
> {
    let parsed = parse_source_enroll_multipart(&runtime, multipart).await?;
    let idempotency = idempotency_request(&headers, source_enroll_request_hash(&parsed))?;
    let store = open_voice_id_store(&distribution)?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let person = match idempotency {
        Some(idempotency) => {
            openasr_core::diarize::voice_id::enroll_person_from_clips_idempotent(
                &store,
                parsed.display_name,
                parsed.consent,
                vec![parsed.clip],
                embedder.as_ref(),
                &identity,
                parsed.color_preference,
                idempotency,
            )
            .map_err(voice_id_service_error)?
            .person
        }
        None => openasr_core::diarize::voice_id::enroll_person_from_clips(
            &store,
            parsed.display_name,
            parsed.consent,
            vec![parsed.clip],
            embedder.as_ref(),
            &identity,
            parsed.color_preference,
        )
        .map_err(voice_id_service_error)?,
    };
    let mut out_headers = HeaderMap::new();
    out_headers.insert(
        header::ETAG,
        HeaderValue::from_str(&format!("\"{}\"", person.revision)).unwrap(),
    );
    Ok((StatusCode::CREATED, out_headers, Json(person)))
}

pub(crate) async fn add_sample_from_source_audio(
    State(runtime): State<ServerRuntime>,
    Extension(distribution): Extension<DistributionContext>,
    headers: HeaderMap,
    AxumPath(person_id): AxumPath<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<(HeaderMap, Json<openasr_core::diarize::voice_id::PersonView>), ApiError> {
    let parsed = parse_source_sample_multipart(&runtime, multipart).await?;
    let id = openasr_core::diarize::voice_id::PersonId::parse(&person_id)
        .map_err(|error| ApiError::BadRequest(error.to_string()))?;
    let expected = parse_if_match(&headers)?;
    let idempotency =
        idempotency_request(&headers, source_sample_request_hash(&id, expected, &parsed))?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let store = open_voice_id_store(&distribution)?;
    let person = match idempotency {
        Some(idempotency) => {
            openasr_core::diarize::voice_id::add_sample_from_pcm_idempotent(
                &store,
                &id,
                expected,
                parsed.consent,
                &parsed.pcm,
                parsed.capture_context,
                embedder.as_ref(),
                &identity,
                idempotency,
            )
            .map_err(voice_id_service_error)?
            .person
        }
        None => openasr_core::diarize::voice_id::add_sample_from_pcm(
            &store,
            &id,
            expected,
            parsed.consent,
            &parsed.pcm,
            parsed.capture_context,
            embedder.as_ref(),
            &identity,
        )
        .map_err(voice_id_service_error)?,
    };
    let mut out_headers = HeaderMap::new();
    out_headers.insert(
        header::ETAG,
        HeaderValue::from_str(&format!("\"{}\"", person.revision)).unwrap(),
    );
    Ok((out_headers, Json(person)))
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
    let display_name = match request.display_name {
        PatchField::Missing => None,
        PatchField::Null => {
            return Err(ApiError::BadRequest("display_name must be a string".into()));
        }
        PatchField::Value(display_name) => Some(display_name),
    };
    let color_preference = match request.color_preference {
        PatchField::Missing => None,
        PatchField::Null => Some(None),
        PatchField::Value(color_preference) => Some(Some(color_preference)),
    };
    let person = store
        .update_person_metadata(
            &id,
            expected,
            openasr_core::diarize::voice_id::PersonMetadataUpdate {
                display_name,
                color_preference,
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
    let idempotency =
        idempotency_request(&headers, add_sample_request_hash(&id, expected, &parsed))?;
    let (embedder, identity) = active_embedder_and_identity()?;
    let person = match idempotency {
        Some(idempotency) => {
            openasr_core::diarize::voice_id::add_sample_from_pcm_idempotent(
                &store,
                &id,
                expected,
                parsed.consent,
                &parsed.pcm,
                parsed.capture_context,
                embedder.as_ref(),
                &identity,
                idempotency,
            )
            .map_err(voice_id_service_error)?
            .person
        }
        None => openasr_core::diarize::voice_id::add_sample_from_pcm(
            &store,
            &id,
            expected,
            parsed.consent,
            &parsed.pcm,
            parsed.capture_context,
            embedder.as_ref(),
            &identity,
        )
        .map_err(voice_id_service_error)?,
    };
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
    let sample_label = match request.sample_label {
        PatchField::Value(sample_label) => sample_label,
        PatchField::Missing | PatchField::Null => {
            return Err(ApiError::BadRequest("PATCH requires sample_label".into()));
        }
    };
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
    wav_fingerprints: Vec<UploadedWavFingerprint>,
}

struct ParsedSample {
    consent: openasr_core::diarize::voice_id::ConsentRecord,
    capture_context: openasr_core::diarize::voice_id::CaptureContext,
    pcm: Vec<f32>,
    wav_fingerprint: UploadedWavFingerprint,
}

struct ParsedSourceEnroll {
    display_name: String,
    consent: openasr_core::diarize::voice_id::ConsentRecord,
    color_preference: Option<String>,
    clip: openasr_core::diarize::voice_id::EnrollmentClip,
    source_fingerprint: UploadedWavFingerprint,
    intervals: Vec<SourceInterval>,
}

struct ParsedSourceSample {
    consent: openasr_core::diarize::voice_id::ConsentRecord,
    capture_context: openasr_core::diarize::voice_id::CaptureContext,
    pcm: Vec<f32>,
    source_fingerprint: UploadedWavFingerprint,
    intervals: Vec<SourceInterval>,
}

async fn parse_source_enroll_multipart(
    runtime: &ServerRuntime,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<ParsedSourceEnroll, ApiError> {
    let (display_name, consent, color_preference, capture_context, source, intervals) =
        parse_source_audio_multipart(multipart, true).await?;
    let pcm = extract_source_intervals(runtime, &source, &intervals).await?;
    Ok(ParsedSourceEnroll {
        display_name: display_name.ok_or_else(|| {
            ApiError::BadRequest("Missing required form field: display_name".into())
        })?,
        consent,
        color_preference,
        clip: openasr_core::diarize::voice_id::EnrollmentClip {
            samples: pcm,
            capture_context,
        },
        source_fingerprint: source.fingerprint,
        intervals,
    })
}

async fn parse_source_sample_multipart(
    runtime: &ServerRuntime,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<ParsedSourceSample, ApiError> {
    let (_display_name, consent, _color_preference, capture_context, source, intervals) =
        parse_source_audio_multipart(multipart, false).await?;
    let pcm = extract_source_intervals(runtime, &source, &intervals).await?;
    Ok(ParsedSourceSample {
        consent,
        capture_context,
        pcm,
        source_fingerprint: source.fingerprint,
        intervals,
    })
}

async fn parse_source_audio_multipart(
    multipart: Result<Multipart, MultipartRejection>,
    needs_display_name: bool,
) -> Result<
    (
        Option<String>,
        openasr_core::diarize::voice_id::ConsentRecord,
        Option<String>,
        openasr_core::diarize::voice_id::CaptureContext,
        UploadedVoiceIdSource,
        Vec<SourceInterval>,
    ),
    ApiError,
> {
    let mut multipart = multipart.map_err(ApiError::MultipartRejection)?;
    let mut display_name = None;
    let mut notice_version = "voice-id-notice-v1".to_string();
    let mut capture_method = "source_audio".to_string();
    let mut color_preference = None;
    let mut device_class = "unknown".to_string();
    let mut input_route = "source_audio".to_string();
    let mut environment_hint = None;
    let mut sample_label = None;
    let mut intervals = None;
    let mut source = None;
    while let Some(field) = multipart.next_field().await.map_err(ApiError::Multipart)? {
        match field.name().unwrap_or_default() {
            "display_name" | "name" => {
                display_name = Some(
                    field
                        .text()
                        .await
                        .map_err(ApiError::Multipart)?
                        .trim()
                        .to_string(),
                )
            }
            "notice_version" => notice_version = field.text().await.map_err(ApiError::Multipart)?,
            "capture_method" => capture_method = field.text().await.map_err(ApiError::Multipart)?,
            "color_preference" => {
                color_preference = Some(field.text().await.map_err(ApiError::Multipart)?)
            }
            "device_class" => device_class = field.text().await.map_err(ApiError::Multipart)?,
            "input_route" => input_route = field.text().await.map_err(ApiError::Multipart)?,
            "environment_hint" => {
                environment_hint = Some(field.text().await.map_err(ApiError::Multipart)?)
            }
            "sample_label" => sample_label = Some(field.text().await.map_err(ApiError::Multipart)?),
            "intervals" => {
                intervals = Some(
                    serde_json::from_str::<Vec<SourceInterval>>(
                        &field.text().await.map_err(ApiError::Multipart)?,
                    )
                    .map_err(|error| {
                        ApiError::BadRequest(format!("Invalid intervals JSON: {error}"))
                    })?,
                )
            }
            "source_audio" => source = Some(stream_voice_id_source(field).await?),
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
    }
    let display_name = display_name.filter(|value| !value.is_empty());
    if needs_display_name && display_name.is_none() {
        return Err(ApiError::BadRequest(
            "Missing required form field: display_name".into(),
        ));
    }
    let source = source
        .ok_or_else(|| ApiError::BadRequest("Missing required form field: source_audio".into()))?;
    // `intervals` is optional: an omitted or empty selection means "enroll from
    // the whole decoded source", the same clip semantics the wav-only
    // enrollment path uses for an entire recording. A caller that wants a
    // subset still sends explicit, validated ranges.
    let intervals = intervals.unwrap_or_default();
    Ok((
        display_name,
        openasr_core::diarize::voice_id::ConsentRecord {
            granted_at: openasr_core::diarize::voice_id::timestamp_now(),
            notice_version,
            capture_method,
        },
        color_preference,
        openasr_core::diarize::voice_id::CaptureContext {
            device_class,
            input_route,
            environment_hint,
            sample_label,
        },
        source,
        intervals,
    ))
}

/// Decodes `source`'s upload and slices out `intervals` as concatenated PCM.
/// Runs the full decode/resample (and, for a non-wav or non-conformant wav
/// upload, an external ffmpeg/afconvert conversion) on a `spawn_blocking`
/// worker, matching `transcription.rs`'s `transcribe_with_runtime` --
/// `AudioPreparationOptions` here also follows `runtime`'s configured ffmpeg
/// binary and backend instead of hardcoding `Native` with no ffmpeg
/// knowledge, so an operator-configured `media.ffmpeg_bin`/
/// `OPENASR_FFMPEG_BIN` is honored on this path exactly as it is for
/// transcription uploads.
async fn extract_source_intervals(
    runtime: &ServerRuntime,
    source: &UploadedVoiceIdSource,
    intervals: &[SourceInterval],
) -> Result<Vec<f32>, ApiError> {
    let source_path = source.path.to_path_buf();
    let intervals = intervals.to_vec();
    let backend = runtime.backend;
    let ffmpeg_bin = runtime.ffmpeg_bin.clone();
    let ffmpeg_bin_explicit = runtime.ffmpeg_bin_explicit;
    tokio::task::spawn_blocking(move || {
        decode_and_slice_source_intervals(
            &source_path,
            &intervals,
            backend,
            ffmpeg_bin,
            ffmpeg_bin_explicit,
        )
    })
    .await
    .map_err(ApiError::BackendJoin)?
}

/// Synchronous decode + slice body of [`extract_source_intervals`], run
/// inside `spawn_blocking` -- kept as a plain function (not a closure) so it
/// stays testable and readable independent of the tokio wiring around it.
fn decode_and_slice_source_intervals(
    source_path: &std::path::Path,
    intervals: &[SourceInterval],
    backend: openasr_core::BackendKind,
    ffmpeg_bin: Option<std::path::PathBuf>,
    ffmpeg_bin_explicit: bool,
) -> Result<Vec<f32>, ApiError> {
    let prepared = openasr_core::prepare_audio_input(
        source_path,
        &openasr_core::AudioPreparationOptions::new(backend)
            .with_ffmpeg_bin(ffmpeg_bin)
            .with_ffmpeg_bin_explicit(ffmpeg_bin_explicit)
            .with_native_non_wav_conversion(true),
    )
    .map_err(|error| ApiError::BadRequest(format!("Could not decode source_audio: {error}")))?;
    let decoded = match prepared.samples() {
        Some(samples) => samples.to_vec(),
        None => openasr_core::load_native_wav_16khz_mono_f32_v0(
            prepared.path(),
            "voice-id source enrollment",
            "source_audio",
        )
        .map_err(|error| ApiError::BadRequest(error.to_string()))?,
    };
    // No interval selection means "use the whole decoded source" -- return it
    // as-is (no per-clip fade), matching how the wav enrollment path hands the
    // entire recording to the embedder. The too-short/quality floors are
    // enforced downstream in the voice-id service, so this stays a pure slice.
    if intervals.is_empty() {
        return Ok(decoded);
    }
    let duration = decoded.len() as f32 / 16_000.0;
    let mut output = Vec::new();
    let mut previous_end = 0.0_f32;
    for interval in intervals {
        if !interval.start.is_finite()
            || !interval.end.is_finite()
            || interval.start < 0.0
            || interval.end <= interval.start
            || interval.end > duration
            || interval.start < previous_end
        {
            return Err(ApiError::BadRequest(
                "intervals must be ordered, non-overlapping, and within source_audio duration"
                    .into(),
            ));
        }
        let start = (interval.start * 16_000.0).floor() as usize;
        let end = (interval.end * 16_000.0).ceil() as usize;
        let mut clip = decoded[start.min(decoded.len())..end.min(decoded.len())].to_vec();
        apply_interval_fade(&mut clip);
        output.extend(clip);
        previous_end = interval.end;
    }
    Ok(output)
}

fn apply_interval_fade(samples: &mut [f32]) {
    let fade = samples.len().min(160);
    for index in 0..fade {
        let gain = index as f32 / fade.max(1) as f32;
        samples[index] *= gain;
        let tail = samples.len() - 1 - index;
        samples[tail] *= gain;
    }
}

async fn stream_voice_id_source(mut field: Field<'_>) -> Result<UploadedVoiceIdSource, ApiError> {
    // Shares `transcription.rs`'s `safe_extension_suffix` (length-capped,
    // ASCII-alphanumeric, case-normalized) instead of a second, weaker duplicate.
    let suffix = field
        .file_name()
        .and_then(safe_extension_suffix)
        .unwrap_or_default();
    let mut file = tempfile::Builder::new()
        .prefix("openasr-voice-id-source-")
        .suffix(&suffix)
        .tempfile()
        .map_err(ApiError::TempFile)?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    while let Some(chunk) = field.chunk().await.map_err(ApiError::Multipart)? {
        bytes = bytes.saturating_add(chunk.len() as u64);
        if bytes > MAX_VOICE_ID_SOURCE_BYTES {
            return Err(ApiError::BadRequest(format!(
                "Voice ID source_audio exceeds the {} MiB upload limit",
                MAX_VOICE_ID_SOURCE_BYTES / (1024 * 1024)
            )));
        }
        digest.update(&chunk);
        file.write_all(&chunk).map_err(ApiError::TempFile)?;
    }
    file.flush().map_err(ApiError::TempFile)?;
    Ok(UploadedVoiceIdSource {
        path: file.into_temp_path(),
        fingerprint: UploadedWavFingerprint {
            sha256: hex_digest(digest.finalize()),
            bytes,
        },
    })
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
    let mut wavs: Vec<UploadedVoiceIdWav> = Vec::new();

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
                wavs.push(stream_voice_id_wav(field).await?);
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
    if wavs.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing required form field: wav (one or more enrollment samples)".into(),
        ));
    }
    if wavs.len() > 5 {
        return Err(ApiError::BadRequest(
            "Initial enrollment accepts at most 5 samples".into(),
        ));
    }
    let sample_labels = resolve_initial_sample_labels(sample_labels, wavs.len())?;

    // Prepare all clips first; any failure leaves zero DB writes.
    let mut clips = Vec::with_capacity(wavs.len());
    let mut wav_fingerprints = Vec::with_capacity(wavs.len());
    for (idx, wav) in wavs.iter().enumerate() {
        let pcm = load_enrollment_wav(wav.path.as_ref())?;
        wav_fingerprints.push(wav.fingerprint.clone());
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
        wav_fingerprints,
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
    let mut wav: Option<UploadedVoiceIdWav> = None;

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
                wav = Some(stream_voice_id_wav(field).await?);
            }
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
    }
    let Some(wav) = wav else {
        return Err(ApiError::BadRequest(
            "Missing required form field: wav".into(),
        ));
    };
    let pcm = load_enrollment_wav(wav.path.as_ref())?;
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
        wav_fingerprint: wav.fingerprint,
    })
}

async fn stream_voice_id_wav(mut field: Field<'_>) -> Result<UploadedVoiceIdWav, ApiError> {
    let mut file = tempfile::Builder::new()
        .prefix("openasr-voice-id-")
        .suffix(".wav")
        .tempfile()
        .map_err(ApiError::TempFile)?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    while let Some(chunk) = field.chunk().await.map_err(ApiError::Multipart)? {
        bytes = bytes.saturating_add(chunk.len() as u64);
        if bytes > MAX_VOICE_ID_WAV_BYTES {
            return Err(ApiError::BadRequest(format!(
                "Voice ID WAV exceeds the {} MiB upload limit",
                MAX_VOICE_ID_WAV_BYTES / (1024 * 1024)
            )));
        }
        digest.update(&chunk);
        file.write_all(&chunk).map_err(ApiError::TempFile)?;
    }
    file.flush().map_err(ApiError::TempFile)?;
    Ok(UploadedVoiceIdWav {
        path: file.into_temp_path(),
        fingerprint: UploadedWavFingerprint {
            sha256: hex_digest(digest.finalize()),
            bytes,
        },
    })
}

pub(crate) fn open_voice_id_store(
    distribution: &DistributionContext,
) -> Result<openasr_core::diarize::voice_id::VoiceIdStore, ApiError> {
    let home = distribution.openasr_home()?;
    openasr_core::diarize::voice_id::VoiceIdStore::open_checked(home)
        .map_err(|e| ApiError::JobStore(format!("voice-id store open failed: {e}")))
}

pub(crate) fn active_space() -> Option<openasr_core::diarize::voice_id::EmbeddingSpace> {
    let embedder = openasr_core::diarize::embed::shared_embedder()?;
    // Derive both values from one pinned pack snapshot. Resolving the identity
    // independently can observe a path replacement between the two calls and
    // pair one model's calibration with another model's content identity.
    let identity = embedder.identity()?;
    Some(
        openasr_core::diarize::voice_id::EmbeddingSpace::for_active_embedder(
            &identity,
            embedder.calibration_profile(),
        ),
    )
}

fn active_embedder_and_identity() -> Result<
    (
        std::sync::Arc<dyn openasr_core::diarize::embed::SpeakerEmbedder>,
        openasr_core::diarize::embed::SpeakerEmbedderIdentity,
    ),
    ApiError,
> {
    let embedder = openasr_core::diarize::embed::shared_embedder().ok_or_else(|| {
        ApiError::BadRequest(
            openasr_core::diarize::embed::VOICE_ID_EMBEDDER_PACK_MISSING_REASON.into(),
        )
    })?;
    // Keep the Arc and identity coupled to the exact same resolved pack for
    // the full enrollment operation; the configured path may be replaced by
    // an installer while this request is in flight.
    let identity = embedder.identity().ok_or_else(|| {
        ApiError::BadRequest(
            openasr_core::diarize::embed::VOICE_ID_EMBEDDER_PACK_MISSING_REASON.into(),
        )
    })?;
    Ok((embedder, identity))
}

fn idempotency_request(
    headers: &HeaderMap,
    request_hash: String,
) -> Result<Option<openasr_core::diarize::voice_id::IdempotencyRequest>, ApiError> {
    let Some(key) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let key = key
        .to_str()
        .map_err(|_| ApiError::BadRequest("Invalid Idempotency-Key header".into()))?;
    if key.is_empty() || key.len() > 255 || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(ApiError::BadRequest(
            "Idempotency-Key must contain 1-255 visible ASCII characters".into(),
        ));
    }
    Ok(Some(openasr_core::diarize::voice_id::IdempotencyRequest {
        key_hash: sha256_hex(key.as_bytes()),
        request_hash,
    }))
}

fn enroll_request_hash(parsed: &ParsedEnroll) -> String {
    let canonical = serde_json::json!({
        "operation": "enroll_person",
        "display_name": parsed.display_name.trim(),
        // `granted_at` is deliberately omitted: it is server-generated on every
        // receipt and therefore cannot be part of a stable retry identity.
        "notice_version": parsed.consent.notice_version,
        "capture_method": parsed.consent.capture_method,
        "color_preference": parsed.color_preference.as_deref().map(str::trim),
        "clips": parsed.clips.iter().zip(&parsed.wav_fingerprints).map(|(clip, wav)| serde_json::json!({
            "capture_context": clip.capture_context,
            "wav": wav,
        })).collect::<Vec<_>>(),
    });
    canonical_json_hash(&canonical)
}

fn add_sample_request_hash(
    person_id: &openasr_core::diarize::voice_id::PersonId,
    expected_revision: Option<u64>,
    parsed: &ParsedSample,
) -> String {
    let canonical = serde_json::json!({
        "operation": "add_sample",
        "person_id": person_id.as_str(),
        "expected_revision": expected_revision,
        // As with enrollment, preserve the server timestamp in the stored
        // consent record but exclude it from retry identity.
        "notice_version": parsed.consent.notice_version,
        "capture_method": parsed.consent.capture_method,
        "capture_context": parsed.capture_context,
        "wav": parsed.wav_fingerprint,
    });
    canonical_json_hash(&canonical)
}

fn source_enroll_request_hash(parsed: &ParsedSourceEnroll) -> String {
    canonical_json_hash(&serde_json::json!({
        "operation": "enroll_person_from_source_audio",
        "display_name": parsed.display_name.trim(),
        "notice_version": parsed.consent.notice_version,
        "capture_method": parsed.consent.capture_method,
        "color_preference": parsed.color_preference.as_deref().map(str::trim),
        "capture_context": parsed.clip.capture_context,
        "source": parsed.source_fingerprint,
        "intervals": parsed.intervals,
    }))
}

fn source_sample_request_hash(
    person_id: &openasr_core::diarize::voice_id::PersonId,
    expected_revision: Option<u64>,
    parsed: &ParsedSourceSample,
) -> String {
    canonical_json_hash(&serde_json::json!({
        "operation": "add_sample_from_source_audio",
        "person_id": person_id.as_str(),
        "expected_revision": expected_revision,
        "notice_version": parsed.consent.notice_version,
        "capture_method": parsed.consent.capture_method,
        "capture_context": parsed.capture_context,
        "source": parsed.source_fingerprint,
        "intervals": parsed.intervals,
    }))
}

fn canonical_json_hash(value: &serde_json::Value) -> String {
    // Only the digest crosses the HTTP/storage boundary; raw sample bytes are
    // dropped after embedding and are never stored in the idempotency ledger.
    sha256_hex(&serde_json::to_vec(value).expect("voice-id request is serializable"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    use std::fmt::Write;

    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
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
        .metadata_value("global_revision")
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

pub(crate) fn voice_id_store_error(
    error: openasr_core::diarize::voice_id::VoiceIdStoreError,
) -> ApiError {
    use openasr_core::diarize::voice_id::VoiceIdStoreError;
    match error {
        VoiceIdStoreError::NotFound(message) | VoiceIdStoreError::SampleNotFound(message) => {
            ApiError::NotFound(message)
        }
        VoiceIdStoreError::RevisionConflict { .. } | VoiceIdStoreError::IdempotencyConflict => {
            ApiError::Conflict(error.to_string())
        }
        VoiceIdStoreError::EmptyName
        | VoiceIdStoreError::EmptySampleLabel
        | VoiceIdStoreError::LabelTooLong { .. }
        | VoiceIdStoreError::InvalidColorPreference(_)
        | VoiceIdStoreError::EmptyPersonMetadataUpdate
        | VoiceIdStoreError::InvalidId(_)
        | VoiceIdStoreError::NotActive(_)
        | VoiceIdStoreError::InvalidEnrollment(_) => ApiError::BadRequest(error.to_string()),
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

    use super::{
        Multipart, parse_enroll_multipart, parse_source_enroll_multipart,
        resolve_initial_sample_labels,
    };

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

    #[tokio::test]
    async fn idempotency_hash_ignores_server_consent_time_and_multipart_boundary() {
        let wav = pcm16_wav();
        let first = parse_enroll_with_wav("first-boundary", &wav).await;
        std::thread::sleep(std::time::Duration::from_millis(5));
        let replay = parse_enroll_with_wav("second-boundary", &wav).await;

        assert_ne!(first.consent.granted_at, replay.consent.granted_at);
        assert_eq!(
            super::enroll_request_hash(&first),
            super::enroll_request_hash(&replay)
        );
    }

    #[tokio::test]
    async fn idempotency_hash_changes_for_different_raw_wav_bytes() {
        let mut changed = pcm16_wav();
        *changed.last_mut().unwrap() = 1;
        let first = parse_enroll_with_wav("first-boundary", &pcm16_wav()).await;
        let second = parse_enroll_with_wav("second-boundary", &changed).await;
        assert_ne!(
            super::enroll_request_hash(&first),
            super::enroll_request_hash(&second)
        );
    }

    /// Matches production wiring for the voice-id source-audio routes: real
    /// deployments always run these with the native backend, so tests that
    /// exercise `extract_source_intervals`/`parse_source_*_multipart` build
    /// the same `ServerRuntime` shape rather than falling back to
    /// `ServerRuntime::default()`'s `BackendKind::Mock`, which would skip the
    /// already-conformant/ffmpeg-conversion logic entirely.
    fn native_test_runtime() -> super::ServerRuntime {
        super::ServerRuntime {
            backend: openasr_core::BackendKind::Native,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn source_audio_intervals_are_aggregated_and_validated() {
        use std::io::Write;

        // A `.wav` suffix here mirrors what `stream_voice_id_source` actually
        // produces in production (it preserves the uploaded filename's
        // extension) -- without it `extract_source_intervals` cannot tell
        // this is a wav at all, which is exactly the code path
        // `voice_id_registration_accepts_non_wav_and_non_conformant_wav_source_audio`
        // below exercises for formats that are not already-conformant.
        let mut file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        file.write_all(&pcm16_wav()).unwrap();
        let source = super::UploadedVoiceIdSource {
            path: file.into_temp_path(),
            fingerprint: super::UploadedWavFingerprint {
                sha256: "test".into(),
                bytes: 0,
            },
        };
        let runtime = native_test_runtime();
        let clips = super::extract_source_intervals(
            &runtime,
            &source,
            &[
                super::SourceInterval {
                    start: 0.0,
                    end: 0.25,
                },
                super::SourceInterval {
                    start: 0.50,
                    end: 0.75,
                },
            ],
        )
        .await
        .unwrap();
        assert_eq!(clips.len(), 8_000);
        assert!(
            super::extract_source_intervals(
                &runtime,
                &source,
                &[
                    super::SourceInterval {
                        start: 0.0,
                        end: 0.5
                    },
                    super::SourceInterval {
                        start: 0.4,
                        end: 0.75
                    },
                ],
            )
            .await
            .is_err()
        );
    }

    /// Empty intervals mean "enroll from the whole decoded source": the
    /// aggregator must return every decoded sample (here the full 1.0 s /
    /// 16 000-sample `pcm16_wav`), not reject the request, so a client that has
    /// no way to probe a non-wav clip's duration can still register it end to
    /// end by simply omitting the selection.
    #[tokio::test]
    async fn source_audio_empty_intervals_use_the_whole_decoded_source() {
        use std::io::Write;

        let mut file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        file.write_all(&pcm16_wav()).unwrap();
        let source = super::UploadedVoiceIdSource {
            path: file.into_temp_path(),
            fingerprint: super::UploadedWavFingerprint {
                sha256: "test".into(),
                bytes: 0,
            },
        };
        let clips = super::extract_source_intervals(&native_test_runtime(), &source, &[])
            .await
            .expect("empty intervals must enroll from the whole decoded source");
        assert_eq!(clips.len(), 16_000);
    }

    /// Builds and parses a `/v1/voice-id/persons/from-audio`-style multipart
    /// request carrying a single `source_audio` field (with `filename` so
    /// `stream_voice_id_source` preserves its extension, exactly as a real
    /// upload does) plus a short `intervals` entry -- the real path
    /// `enroll_person_from_source_audio` runs in production.
    async fn parse_source_enroll_with_file(
        filename: &str,
        bytes: &[u8],
    ) -> Result<super::ParsedSourceEnroll, crate::ApiError> {
        let boundary = "voice-id-source-test-boundary";
        let mut body = Vec::new();
        form_field(&mut body, boundary, "display_name", b"Alice");
        // 0.3s leaves margin below the encoders' true ~0.5s content for any
        // mp3 encoder priming/padding samples, so this stays valid regardless
        // of the exact decoded sample count.
        form_field(
            &mut body,
            boundary,
            "intervals",
            br#"[{"start":0.0,"end":0.3}]"#,
        );
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"source_audio\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await;
        parse_source_enroll_multipart(&native_test_runtime(), multipart).await
    }

    /// Regression test for the missing `.with_native_non_wav_conversion(true)`
    /// switch on `extract_source_intervals`'s `prepare_audio_input` call
    /// (0.1.24 regression): before the fix, `prepare_audio_input` took the
    /// Native + conversion-disabled passthrough at the very top of
    /// `prepare_external_input` for *any* source_audio, so a 44.1 kHz stereo
    /// wav or an mp3 was handed straight to `load_native_wav_16khz_mono_f32_
    /// v0` untouched and failed with "expected 16 kHz mono PCM16 or float32
    /// WAV input for source_audio" -- a decode error, not an audio problem.
    /// This is the test whose absence let that regression ship; it must
    /// register successfully via the real multipart-parsing path.
    #[tokio::test]
    async fn voice_id_source_audio_registration_accepts_mp3_and_non_conformant_wav() {
        let mp3 = include_bytes!("../../tests/fixtures/tone_stereo_44100.mp3");
        let mp3_enroll = parse_source_enroll_with_file("clip.mp3", mp3)
            .await
            .expect("a 44.1 kHz stereo mp3 source_audio must register successfully");
        assert!(!mp3_enroll.clip.samples.is_empty());

        let wav = include_bytes!("../../tests/fixtures/tone_stereo_44100.wav");
        let wav_enroll = parse_source_enroll_with_file("clip.wav", wav).await.expect(
            "a 44.1 kHz stereo (non-conformant) wav source_audio must register successfully",
        );
        assert!(!wav_enroll.clip.samples.is_empty());
    }

    #[tokio::test]
    async fn voice_id_wav_upload_rejects_oversized_field_while_streaming() {
        let boundary = "oversized-wav-boundary";
        let mut body = Vec::new();
        form_field(&mut body, boundary, "display_name", b"Alice");
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"wav\"; filename=\"oversized.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.resize(body.len() + super::MAX_VOICE_ID_WAV_BYTES as usize + 1, 0);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await;
        let error = match parse_enroll_multipart(multipart).await {
            Ok(_) => panic!("oversized Voice ID upload was accepted"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::ApiError::BadRequest(_) | crate::ApiError::Multipart(_)
        ));
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

    async fn parse_enroll_with_wav(boundary: &str, wav: &[u8]) -> super::ParsedEnroll {
        let mut body = Vec::new();
        form_field(&mut body, boundary, "display_name", b"Alice");
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"wav\"; filename=\"voice.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(wav);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
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
