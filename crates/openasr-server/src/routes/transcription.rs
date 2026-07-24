//! HTTP transcription/translation handlers and all supporting helpers.
//! Pure code-motion from `lib.rs`; shared crate-root items come via `use crate::*`.

use std::{io::Write, path::Path, str::FromStr, sync::Arc};

use openasr_core::config::load_config_document;
use openasr_core::realtime::history::{
    DaemonHistoryKind, DaemonHistoryProvenance, DaemonHistoryRecord, DaemonHistoryStore,
};
use openasr_core::{
    AudioPreparationOptions, BackendKind, CatalogError, ExecutionTarget, LongFormMode,
    LongFormOptions, ModelResolutionError, NativeAsrError, NativeAsrExecutor,
    NativeAsrHardwareTarget, NativeAsrModelAdapter, NativeAsrModelPackRef, NativeAsrOfflineRequest,
    NativeAsrRequestOptions, NativeBackendExecutor, NativeRuntimeModelIdSource, PhraseBiasConfig,
    ResponseFormat, RuntimeModelResolutionError, TranscriptionRequest, TranscriptionTask,
    add_segment_word_timestamps, config::MAX_INFERENCE_THREADS,
    native_runtime_model_adapter_for_path, parse_model_ref, prepare_audio_input,
    render_transcription, resolve_local_native_runtime_model_identity, resolve_runtime_model_ref,
    runtime_registry,
};

use crate::*;

// ── Axum HTTP handlers ────────────────────────────────────────────────────────

pub(crate) async fn transcriptions(
    State(runtime): State<ServerRuntime>,
    Query(query): Query<TranscriptionQuery>,
    headers: HeaderMap,
    Extension(auth): Extension<ServerAuth>,
    Extension(distribution): Extension<DistributionContext>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    if query.stream.unwrap_or(false) {
        return crate::realtime::stream_transcription(
            runtime,
            distribution,
            multipart,
            !is_remote_compute_client_request(&headers, &auth),
        )
        .await;
    }

    run_offline_transcription(runtime, headers, auth, distribution, multipart, None).await
}

/// OpenAI-compatible `/v1/audio/translations`: always X->English translation.
/// Clients of this route send no `task` field (the route implies translate), so
/// it injects `task=translate` and shares the transcription handler. Non-stream
/// only, matching the OpenAI translations contract.
pub(crate) async fn translations(
    State(runtime): State<ServerRuntime>,
    headers: HeaderMap,
    Extension(auth): Extension<ServerAuth>,
    Extension(distribution): Extension<DistributionContext>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    run_offline_transcription(
        runtime,
        headers,
        auth,
        distribution,
        multipart,
        Some(TranscriptionTask::Translate),
    )
    .await
}

/// Fixed denominator for the backward-compatible `done`/`total` ratio: `done /
/// total == fraction`. Only exists so clients that predate the `fraction` field
/// keep working; new clients read `fraction` directly.
const PROGRESS_LEGACY_SCALE: u32 = 1000;

#[derive(serde::Serialize)]
pub(crate) struct TranscriptionProgressBody {
    /// Coarse phase label of the in-flight run (`"decode"`, `"assemble"`, or
    /// `"align"`), or `null` when no native run is in flight. The UI may show this
    /// as phase text (e.g. "Refining word timestamps") next to the bar.
    phase: Option<&'static str>,
    /// Monotonic overall progress in `0.0..=1.0`; `0.0` when idle. The UI progress
    /// bar reads this directly -- it already spans decode, assembly, and the
    /// forced-align refine, so it no longer stalls near the end.
    fraction: f32,
    /// Backward-compatible ratio for clients that predate `fraction`: `done/total`
    /// equals `fraction`. `total` is `0` when idle, so legacy clients still fall
    /// back to a time-based estimate exactly as before.
    done: u32,
    total: u32,
}

/// Progress of the in-flight file transcription, for the UI progress bar. Returns
/// `{phase:null,fraction:0,done:0,total:0}` when nothing is running: short
/// single-pass decodes expose no sub-step, so the client estimates from elapsed
/// time. Auth is enforced by the shared middleware like every other non-operator
/// route.
pub(crate) async fn transcription_progress() -> Json<TranscriptionProgressBody> {
    let body = match openasr_core::api::backend::native_transcription_progress() {
        Some(progress) => {
            let fraction = progress.fraction.clamp(0.0, 1.0);
            TranscriptionProgressBody {
                phase: Some(progress.phase.label()),
                fraction,
                done: (fraction * PROGRESS_LEGACY_SCALE as f32).round() as u32,
                total: PROGRESS_LEGACY_SCALE,
            }
        }
        None => TranscriptionProgressBody {
            phase: None,
            fraction: 0.0,
            done: 0,
            total: 0,
        },
    };
    Json(body)
}

/// Wire status returned by the pause/resume/cancel control endpoints.
#[derive(serde::Serialize)]
pub(crate) struct TranscriptionControlBody {
    /// The client-supplied transcription id the control acted on.
    id: String,
    /// The requested control state: `"paused"`, `"running"` (after resume), or
    /// `"canceled"`. This is the request that was recorded on the in-flight run;
    /// the actual decode observes it at the next long-form slice boundary.
    state: &'static str,
}

/// RAII cleanup that removes an in-flight transcription's control from the
/// registry when the request handler returns (success, error, or cancel), so a
/// finished transcription's id can never be paused/canceled afterward.
///
/// This is also the leak-prevention safety net for a client that disconnects
/// mid-run. A paused (or still-decoding) native transcription holds its
/// `spawn_blocking` worker thread parked on `TranscriptionControl`'s Condvar
/// until a resume/cancel arrives; if the client goes away first (closes the
/// connection, or the app quits), nothing would ever wake that thread. Axum
/// drops the handler's async future when the connection closes, which drops
/// every local still live at the suspended `.await` point -- including this
/// guard. While `armed`, `Drop` fires `control.request_cancel()` so the
/// worker wakes at the next slice boundary (immediately, if it was parked
/// mid-pause) and unwinds through `TranscriptionCanceled` instead of leaking
/// its thread forever. `disarm()` is called right after the decode call
/// returns on its own (success or failure) -- from that point on there is no
/// more worker thread to protect, so a normal completion must never also
/// fire a spurious cancel.
struct ActiveTranscriptionCleanup {
    distribution: DistributionContext,
    transcription_id: String,
    control: Arc<openasr_core::TranscriptionControl>,
    armed: bool,
}

impl ActiveTranscriptionCleanup {
    fn new(
        distribution: DistributionContext,
        transcription_id: String,
        control: Arc<openasr_core::TranscriptionControl>,
    ) -> Self {
        Self {
            distribution,
            transcription_id,
            control,
            armed: true,
        }
    }

    /// Disarms the disconnect-cancel safety net. Call this once the decode
    /// call has returned by itself (success or failure); doing so before then
    /// would let a genuine mid-decode disconnect leak the worker thread.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ActiveTranscriptionCleanup {
    fn drop(&mut self) {
        if self.armed {
            self.control.request_cancel();
        }
        self.distribution
            .clear_transcription(&self.transcription_id);
    }
}

fn control_body_response(id: String, state: &'static str) -> Result<Response, ApiError> {
    Ok((
        StatusCode::ACCEPTED,
        Json(TranscriptionControlBody { id, state }),
    )
        .into_response())
}

fn active_transcription_control(
    distribution: &DistributionContext,
    id: &str,
) -> Result<Arc<openasr_core::TranscriptionControl>, ApiError> {
    distribution.transcription_control(id).ok_or_else(|| {
        ApiError::NotFound(format!(
            "No in-flight transcription with id '{id}'. It may have already finished, been canceled, or never opted into control (missing transcription_id)."
        ))
    })
}

/// `POST /v1/audio/transcriptions/{id}/cancel`: cancel an in-flight file
/// transcription. The decode stops at the next long-form slice boundary and the
/// original transcription request fails closed with a canceled status; the
/// already-decoded portion is discarded (see `BackendError::TranscriptionCanceled`).
pub(crate) async fn cancel_transcription_job(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    active_transcription_control(&distribution, &id)?.request_cancel();
    control_body_response(id, "canceled")
}

/// `POST /v1/audio/transcriptions/{id}/pause`: pause an in-flight file
/// transcription at the next long-form slice boundary. The decode thread (and
/// the original request) block until a matching resume or cancel arrives.
pub(crate) async fn pause_transcription_job(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    active_transcription_control(&distribution, &id)?.request_pause();
    control_body_response(id, "paused")
}

/// `POST /v1/audio/transcriptions/{id}/resume`: resume a paused in-flight file
/// transcription. Decoding continues from the next slice within the same
/// in-flight run, keeping the already-accumulated segments.
pub(crate) async fn resume_transcription_job(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    active_transcription_control(&distribution, &id)?.request_resume();
    control_body_response(id, "running")
}

/// Shared non-streaming transcription/translation core. `task_override` forces
/// the task regardless of the request body (used by the translations alias) and
/// wins over both the multipart field and saved preferences.
async fn run_offline_transcription(
    runtime: ServerRuntime,
    headers: HeaderMap,
    auth: ServerAuth,
    distribution: DistributionContext,
    multipart: Result<Multipart, MultipartRejection>,
    task_override: Option<TranscriptionTask>,
) -> Result<Response, ApiError> {
    let home = distribution.openasr_home()?;
    let catalog = load_runtime_model_catalog(distribution.catalog_source(), &home)?;
    let mut parsed =
        parse_transcription_multipart(multipart, runtime.backend, catalog.as_ref()).await?;
    if parsed.stream_form_field {
        // Fail closed instead of silently returning a JSON body an OpenAI SDK
        // streaming client would hang on (it expects `transcript.text.*` SSE
        // events, which this server does not emit).
        return Err(ApiError::BadRequest(
            "The 'stream' form field is not supported. SSE streaming on this server is the OpenASR realtime protocol, enabled with the '?stream=true' query parameter, and does not emit OpenAI transcript.text.* events -- OpenAI SDK stream=True calls cannot parse it. Retry without 'stream' for a complete response, or POST to /v1/audio/transcriptions?stream=true and handle OpenASR realtime events.".to_string(),
        ));
    }
    // A well-formed transcription request must not fail because the daemon's
    // on-disk preferences are unreadable or hold out-of-range values: degrade to
    // defaults (and log) rather than failing the request. The /v1/config
    // endpoint still surfaces the corruption for the user to fix.
    let preferences = match load_config_document(&home) {
        Ok(document) if document.preferences.validate().is_ok() => Some(document.preferences),
        Ok(_) => {
            eprintln!(
                "openasr-server: ignoring invalid daemon preferences for this transcription; using defaults"
            );
            None
        }
        Err(error) => {
            eprintln!(
                "openasr-server: ignoring unreadable daemon config for this transcription; using defaults: {error}"
            );
            None
        }
    };
    if let Some(preferences) = preferences {
        apply_transcription_preferences(&mut parsed.request, &preferences);
    }
    // The translations alias forces translate over the body/preferences.
    if let Some(task) = task_override {
        parsed.request.task = Some(task);
    }
    parsed.request.source = if task_override == Some(TranscriptionTask::Translate) {
        openasr_core::RequestSource::ServerTranslate
    } else {
        openasr_core::RequestSource::ServerTranscribe
    };
    if runtime.backend == BackendKind::Native {
        parsed.request.serve_batch_max_native_sessions = Some(
            runtime
                .native_execution
                .max_concurrent_sessions_per_model()
                .get(),
        );
    }
    let history_request = parsed.request.clone();
    if runtime.backend == BackendKind::Native {
        validate_native_request_model(&runtime, &parsed.request.model_id)
            .map_err(ApiError::BadRequest)?;
    }
    // Register an in-session pause/cancel control when the client supplied a
    // transcription id and the native backend is in use (control acts at
    // long-form slice boundaries; the mock backend has no such loop). The
    // cleanup guard removes the registry entry on every exit -- success, error,
    // or cancel.
    let control = (runtime.backend == BackendKind::Native)
        .then(|| parsed.transcription_id.clone())
        .flatten()
        .map(|id| {
            let control = Arc::new(openasr_core::TranscriptionControl::new());
            distribution.register_transcription(&id, Arc::clone(&control));
            (id, control)
        });
    // Armed for as long as the decode call below is in flight: if the client
    // disconnects and axum drops this handler future first, `Drop` cancels the
    // control so the (possibly paused) worker thread wakes and exits instead
    // of leaking. Disarmed immediately after that call returns, either way.
    let mut control_cleanup = control.as_ref().map(|(id, control)| {
        ActiveTranscriptionCleanup::new(distribution.clone(), id.clone(), Arc::clone(control))
    });
    let control_handle = control.as_ref().map(|(_, control)| Arc::clone(control));
    let transcription =
        match transcribe_with_runtime(runtime, parsed.request, control_handle.clone()).await {
            Ok(transcription) => {
                if let Some(cleanup) = control_cleanup.as_mut() {
                    cleanup.disarm();
                }
                transcription
            }
            Err(error) => {
                if let Some(cleanup) = control_cleanup.as_mut() {
                    cleanup.disarm();
                }
                // A cancel surfaces from core as a generic fail-closed error (the
                // typed cancel is flattened through the NativeAsrError layer), so
                // consult the control to report it honestly as a 409 canceled result
                // rather than a 400 fail-closed refusal.
                if control_handle.is_some_and(|control| control.is_canceled()) {
                    return Err(ApiError::Backend(
                        openasr_core::BackendError::TranscriptionCanceled,
                    ));
                }
                return Err(error);
            }
        };
    let rendered = render_transcription(&transcription, parsed.response_format)
        .map_err(ApiError::Serialize)?;
    // History is a best-effort audit side-write: a successful transcription must
    // not fail because the history store could not be written (e.g. a read-only
    // or misconfigured OPENASR_HOME). Log and continue; the realtime path already
    // treats history the same way.
    if !is_remote_compute_client_request(&headers, &auth)
        && let Err(error) = record_file_transcription_history(
            &distribution,
            &history_request,
            &transcription,
            parsed.response_format,
        )
    {
        eprintln!(
            "openasr-server: could not record file transcription history (continuing): {error}"
        );
    }

    let content_type = match parsed.response_format {
        ResponseFormat::Json | ResponseFormat::VerboseJson => mime::APPLICATION_JSON.as_ref(),
        ResponseFormat::Text
        | ResponseFormat::Srt
        | ResponseFormat::Vtt
        | ResponseFormat::Markdown => mime::TEXT_PLAIN_UTF_8.as_ref(),
    };

    Ok(([(header::CONTENT_TYPE, content_type)], rendered).into_response())
}

// ── History / auth helpers ────────────────────────────────────────────────────

pub(crate) fn is_remote_compute_client_request(headers: &HeaderMap, auth: &ServerAuth) -> bool {
    headers
        .get(REMOTE_COMPUTE_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case(REMOTE_COMPUTE_CLIENT_VALUE))
        && auth.authorizes_remote_compute_client(headers)
}

pub(crate) fn record_file_transcription_history(
    distribution: &DistributionContext,
    request: &TranscriptionRequest,
    transcription: &openasr_core::Transcription,
    output_format: ResponseFormat,
) -> Result<(), ApiError> {
    let home = distribution.openasr_home()?;
    // History persistence is governed solely by the saved-history scope
    // (`history_retention`). `auto_save` controls transcript-file exports and
    // must not gate history. "Off" retention is fail-fast: never write a
    // transcript we would only prune away on the next sweep.
    let document = load_config_document(&home).unwrap_or_default();
    if !document
        .preferences
        .history_retention
        .persists_new_entries()
    {
        return Ok(());
    }
    let store = DaemonHistoryStore::open(&home);
    store
        .record(DaemonHistoryRecord {
            kind: DaemonHistoryKind::File,
            model: request.model_id.clone(),
            source_name: request.display_file_name.clone().or_else(|| {
                request
                    .input_path
                    .file_name()?
                    .to_str()
                    .map(ToOwned::to_owned)
            }),
            duration_seconds: transcription_duration_seconds(transcription),
            output_format: Some(output_format),
            diarization_active: Some(request.diarize),
            provenance: Some(DaemonHistoryProvenance::Recorded),
            // Persist the per-segment timing so exports can rebuild SRT/VTT/JSON
            // later; the store derives the advertised `formats` from these so we
            // never claim a format the stored transcript cannot render.
            segments: transcription.segments.clone(),
            text: transcription.text.clone(),
        })
        .map_err(ApiError::History)?;
    if let Err(error) = prune_history_store(&store, document.preferences.history_retention) {
        eprintln!("openasr-server: could not prune transcription history (continuing): {error}");
    }
    Ok(())
}

fn transcription_duration_seconds(transcription: &openasr_core::Transcription) -> Option<f32> {
    transcription
        .segments
        .iter()
        .map(|segment| segment.end)
        .filter(|end| end.is_finite() && *end >= 0.0)
        .max_by(|left, right| left.total_cmp(right))
}

// ── Request parsing ───────────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
pub(crate) struct TranscriptionQuery {
    pub(crate) stream: Option<bool>,
}

pub(crate) struct ParsedTranscriptionRequest {
    pub(crate) request: TranscriptionRequest,
    pub(crate) response_format: ResponseFormat,
    /// Optional client-supplied id for this transcription. When present, the
    /// handler registers a pause/cancel control under it so the
    /// `/v1/audio/transcriptions/{id}/{pause,resume,cancel}` endpoints can act on
    /// the in-flight run. Absent (older clients) keeps today's uncontrolled,
    /// run-to-completion behavior.
    pub(crate) transcription_id: Option<String>,
    /// `true` when the client sent `stream=true` as a multipart form field
    /// (OpenAI SDK semantics). The non-streaming handler rejects this fail
    /// closed: our SSE protocol is enabled by the `?stream=true` query
    /// parameter and emits OpenASR realtime events, not OpenAI
    /// `transcript.text.*` events, so silently ignoring the field would leave
    /// an OpenAI SDK streaming client hanging over a JSON body it never parses.
    pub(crate) stream_form_field: bool,
    pub(crate) _uploaded_file: tempfile::TempPath,
}

struct TranscriptionRequestBuilder {
    file_name: Option<String>,
    saw_file: bool,
    uploaded_file: Option<tempfile::TempPath>,
    transcription_id: Option<String>,
    stream: bool,
    model: Option<String>,
    language: Option<String>,
    task: Option<TranscriptionTask>,
    prompt: Option<String>,
    response_format: ResponseFormat,
    timestamp_granularities: Vec<String>,
    diarize: bool,
    speakers: Option<u8>,
    punctuate: bool,
    segment_mode: Option<String>,
    chunk_seconds: Option<f32>,
    segment_overlap_seconds: Option<f32>,
    vad_threshold_db: Option<f32>,
    vad_min_silence_ms: Option<u32>,
    vad_padding_ms: Option<u32>,
    min_segment_seconds: Option<f32>,
    suppress_silent_slices: Option<bool>,
    phrase_bias_phrases: Vec<String>,
    hotword_boost: Option<f32>,
    phrase_bias_boost: Option<f32>,
    inference_threads: Option<u16>,
    execution_target: Option<ExecutionTarget>,
}

impl Default for TranscriptionRequestBuilder {
    fn default() -> Self {
        Self {
            file_name: None,
            saw_file: false,
            uploaded_file: None,
            transcription_id: None,
            stream: false,
            model: None,
            language: None,
            task: None,
            prompt: None,
            response_format: ResponseFormat::Json,
            timestamp_granularities: Vec::new(),
            diarize: false,
            speakers: None,
            // Auto-on, mirroring `TranscriptionRequest::new`'s default: this
            // form field is only a client-facing opt-out (the desktop
            // punctuation preference toggle), not the primary gate -- the
            // stage itself still requires `emits_punctuation == Some(false)`
            // and the FireRedPunc capability pack to be installed.
            punctuate: true,
            segment_mode: None,
            chunk_seconds: None,
            segment_overlap_seconds: None,
            vad_threshold_db: None,
            vad_min_silence_ms: None,
            vad_padding_ms: None,
            min_segment_seconds: None,
            suppress_silent_slices: None,
            phrase_bias_phrases: Vec::new(),
            hotword_boost: None,
            phrase_bias_boost: None,
            inference_threads: None,
            execution_target: None,
        }
    }
}

impl TranscriptionRequestBuilder {
    async fn ingest_field(&mut self, field: Field<'_>) -> Result<(), ApiError> {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                self.saw_file = true;
                self.file_name = field.file_name().map(ToOwned::to_owned);
                let suffix = self
                    .file_name
                    .as_deref()
                    .and_then(safe_extension_suffix)
                    .unwrap_or_default();
                self.uploaded_file = Some(write_upload_temp_file_streaming(field, &suffix).await?);
            }
            "transcription_id" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                let trimmed = value.trim();
                self.transcription_id = (!trimmed.is_empty()).then(|| trimmed.to_string());
            }
            "stream" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.stream = parse_bool_field("stream", &value)?;
            }
            "model" => {
                self.model = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "response_format" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.response_format =
                    ResponseFormat::from_str(&value).map_err(ApiError::Format)?;
            }
            "language" => {
                self.language = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "task" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.task =
                    Some(TranscriptionTask::from_str(&value).map_err(ApiError::BadRequest)?);
            }
            "prompt" => {
                self.prompt = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "diarize" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.diarize = parse_bool_field("diarize", &value)?;
            }
            "punctuate" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.punctuate = parse_bool_field("punctuate", &value)?;
            }
            "speakers" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                let speakers = parse_u32_field("speakers", &value)?;
                if speakers == 0 || speakers > u8::MAX as u32 {
                    return Err(ApiError::BadRequest(format!(
                        "Form field speakers must be between 1 and {}.",
                        u8::MAX
                    )));
                }
                self.speakers = Some(speakers as u8);
            }
            "timestamp_granularities" | "timestamp_granularities[]" => {
                self.timestamp_granularities
                    .push(field.text().await.map_err(ApiError::Multipart)?);
            }
            "segment_mode" => {
                self.segment_mode = Some(field.text().await.map_err(ApiError::Multipart)?);
            }
            "chunk_seconds" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.chunk_seconds = Some(parse_f32_field("chunk_seconds", &value)?);
            }
            "segment_overlap_seconds" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.segment_overlap_seconds =
                    Some(parse_f32_field("segment_overlap_seconds", &value)?);
            }
            "vad_threshold_db" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.vad_threshold_db = Some(parse_f32_field("vad_threshold_db", &value)?);
            }
            "vad_min_silence_ms" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.vad_min_silence_ms = Some(parse_u32_field("vad_min_silence_ms", &value)?);
            }
            "vad_padding_ms" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.vad_padding_ms = Some(parse_u32_field("vad_padding_ms", &value)?);
            }
            "min_segment_seconds" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.min_segment_seconds = Some(parse_f32_field("min_segment_seconds", &value)?);
            }
            "suppress_silent_slices" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.suppress_silent_slices =
                    Some(parse_bool_field("suppress_silent_slices", &value)?);
            }
            "hotword" | "phrase_bias" => {
                self.phrase_bias_phrases
                    .push(field.text().await.map_err(ApiError::Multipart)?);
            }
            "hotword_boost" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.hotword_boost = Some(parse_f32_field("hotword_boost", &value)?);
            }
            "phrase_bias_boost" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.phrase_bias_boost = Some(parse_f32_field("phrase_bias_boost", &value)?);
            }
            "inference_threads" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.inference_threads = Some(parse_inference_threads_field(&value)?);
            }
            "execution_target" => {
                let value = field.text().await.map_err(ApiError::Multipart)?;
                self.execution_target = Some(parse_execution_target_field(&value)?);
            }
            _ => {
                let _ = field.bytes().await.map_err(ApiError::Multipart)?;
            }
        }
        Ok(())
    }

    fn finish(
        self,
        backend: BackendKind,
        catalog: Option<&openasr_core::ModelCatalog>,
    ) -> Result<ParsedTranscriptionRequest, ApiError> {
        let Self {
            file_name,
            saw_file,
            uploaded_file,
            transcription_id,
            stream,
            model,
            language,
            task,
            prompt,
            response_format,
            timestamp_granularities,
            diarize,
            speakers,
            punctuate,
            segment_mode,
            chunk_seconds,
            segment_overlap_seconds,
            vad_threshold_db,
            vad_min_silence_ms,
            vad_padding_ms,
            min_segment_seconds,
            suppress_silent_slices,
            phrase_bias_phrases,
            hotword_boost,
            phrase_bias_boost,
            inference_threads,
            execution_target,
        } = self;

        validate_timestamp_granularities(&timestamp_granularities)?;

        if speakers.is_some() && !diarize {
            return Err(ApiError::BadRequest(
                "Form field speakers requires diarize=true.".to_string(),
            ));
        }

        if !saw_file {
            return Err(ApiError::BadRequest(
                "Missing required form field: file".to_string(),
            ));
        }
        let Some(uploaded_file) = uploaded_file else {
            return Err(ApiError::BadRequest(
                "Missing required form field: file".to_string(),
            ));
        };

        let Some(model) = model else {
            return Err(ApiError::BadRequest(
                "Missing required form field: model".to_string(),
            ));
        };
        let normalized_model = model.trim();
        if normalized_model.is_empty() {
            return Err(ApiError::BadRequest(
                "Model form field must be a non-empty model id.".to_string(),
            ));
        }

        let model_id = resolve_and_validate_form_model_id(normalized_model, backend, catalog)?;
        let has_longform_fields = segment_mode.is_some()
            || chunk_seconds.is_some()
            || segment_overlap_seconds.is_some()
            || vad_threshold_db.is_some()
            || vad_min_silence_ms.is_some()
            || vad_padding_ms.is_some()
            || min_segment_seconds.is_some()
            || suppress_silent_slices.is_some();
        let longform = if backend == BackendKind::Native {
            build_native_longform_options_override(
                segment_mode.as_deref(),
                chunk_seconds,
                segment_overlap_seconds,
                vad_threshold_db,
                vad_min_silence_ms,
                vad_padding_ms,
                min_segment_seconds,
                suppress_silent_slices,
            )?
        } else if has_longform_fields {
            return Err(ApiError::BadRequest(
                "Longform segmentation fields are only supported with backend=native.".to_string(),
            ));
        } else {
            None
        };
        let phrase_bias =
            build_phrase_bias_config(&phrase_bias_phrases, hotword_boost, phrase_bias_boost)?;
        // `word_aligned` opts into the Qwen3-ForcedAligner-0.6B refinement tier
        // (see `--word-timestamps=aligned`); it also implies `word` so callers
        // do not have to pass both. The server never auto-installs the pack --
        // a missing pack fails the request closed (BackendError mapped to 400)
        // rather than silently falling back to approximate timestamps.
        let word_timestamps_refine = timestamp_granularities
            .iter()
            .any(|value| value.as_str() == "word_aligned");
        let word_timestamps = word_timestamps_refine
            || timestamp_granularities
                .iter()
                .any(|value| value.as_str() == "word");
        let uploaded_path: &Path = uploaded_file.as_ref();
        let request = TranscriptionRequest::new(uploaded_path.to_path_buf(), model_id)
            .with_language(language)
            .with_task(task)
            .with_prompt(prompt)
            .with_longform(longform)
            .with_phrase_bias(phrase_bias)
            .with_inference_threads(inference_threads)
            .with_execution_target(execution_target)
            .with_word_timestamps(word_timestamps)
            .with_word_timestamps_refine(word_timestamps_refine)
            .with_display_file_name(file_name)
            .with_diarization(diarize)
            .with_diarize_speakers(speakers)
            .with_punctuation(punctuate);

        Ok(ParsedTranscriptionRequest {
            request,
            response_format,
            transcription_id,
            stream_form_field: stream,
            _uploaded_file: uploaded_file,
        })
    }
}

pub(crate) async fn parse_transcription_multipart(
    multipart: Result<Multipart, MultipartRejection>,
    backend: BackendKind,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<ParsedTranscriptionRequest, ApiError> {
    let mut multipart = multipart.map_err(ApiError::MultipartRejection)?;
    let mut builder = TranscriptionRequestBuilder::default();

    while let Some(field) = multipart.next_field().await.map_err(ApiError::Multipart)? {
        builder.ingest_field(field).await?;
    }

    builder.finish(backend, catalog)
}

// ── Model catalog / resolution helpers ───────────────────────────────────────

pub(crate) fn load_runtime_model_catalog(
    catalog_source: Option<CatalogSource<'_>>,
    home: &Path,
) -> Result<Option<openasr_core::ModelCatalog>, ApiError> {
    catalog_source
        .map(|source| resolve_runtime_catalog_for_source(source, home).map_err(ApiError::Catalog))
        .transpose()
}

pub(crate) fn validate_native_runtime_pack(
    pack_root: &Path,
) -> Result<openasr_core::NativeRuntimeModelIdentity, openasr_core::BackendError> {
    resolve_native_runtime_model_identity(pack_root, None)
}

fn resolve_native_runtime_model_identity(
    pack_root: &Path,
    explicit_model_id_fallback: Option<&str>,
) -> Result<openasr_core::NativeRuntimeModelIdentity, openasr_core::BackendError> {
    let mut identity =
        resolve_local_native_runtime_model_identity(pack_root, explicit_model_id_fallback)
            .map_err(|error| openasr_core::BackendError::NativeFailClosed {
                reason: format!(
                    "could not resolve native model id from ggml runtime source '{}': {error}",
                    pack_root.display()
                ),
            })?;
    if is_retired_native_model_ref(&identity.model_id)
        && matches!(
            identity.source,
            NativeRuntimeModelIdSource::MetadataGgufKey { .. }
        )
        && let Some(stem) = pack_root.file_stem().and_then(|value| value.to_str())
    {
        let normalized_stem = stem.trim();
        if !normalized_stem.is_empty()
            && parse_model_ref(normalized_stem).is_ok()
            && !is_retired_native_model_ref(normalized_stem)
        {
            identity = openasr_core::NativeRuntimeModelIdentity {
                model_id: normalized_stem.to_string(),
                source: NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback,
            };
        }
    }
    if is_retired_native_model_ref(&identity.model_id) {
        return Err(openasr_core::BackendError::NativeFailClosed {
            reason: format!(
                "model '{}' is a retired legacy metadata id and is not executable",
                identity.model_id
            ),
        });
    }
    Ok(identity)
}

pub(crate) fn resolve_and_validate_form_model_id(
    model: &str,
    backend: BackendKind,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<String, ApiError> {
    let registry = runtime_registry(catalog).map_err(ApiError::from)?;

    match backend {
        BackendKind::Mock => {
            let resolved = resolve_runtime_model_ref(&registry, catalog, model)
                .map_err(|error| ApiError::BadRequest(api_runtime_model_resolution_error(error)))?;
            Ok(resolved.model_id)
        }
        BackendKind::Native => {
            parse_model_ref(model).map_err(|error| {
                ApiError::BadRequest(format!(
                    "Native backend requires a valid model id in form field 'model': {error}"
                ))
            })?;
            if is_retired_native_model_ref(model) {
                return Err(ApiError::BadRequest(format!(
                    "Model '{model}' is a retired legacy metadata id and is not executable in native mode."
                )));
            }
            Ok(model.to_string())
        }
    }
}

// Native model handling is intentionally two-phase: form parsing rejects invalid
// or retired ids, then runtime validation checks that the loaded pack matches.
pub(crate) fn validate_native_request_model(
    runtime: &ServerRuntime,
    model: &str,
) -> Result<(), String> {
    let Some(model_pack_path) = runtime.model_pack_path.as_deref() else {
        // No model bound at all: a fresh install with zero pulled models is a
        // normal daemon state (it starts and answers /health fine), but a
        // transcription request needs a model, so this is where that need
        // becomes a fail-closed, structured error.
        return Err(format!(
            "Model '{model}' is not installed. No models are installed on this server yet -- install one first (openasr pull {model}, or via the model market)."
        ));
    };
    let pack_root = openasr_core::validate_local_native_model_pack_path(model_pack_path)
        .map_err(|error| error.to_string())?;
    let identity = resolve_native_runtime_model_identity(&pack_root, Some(model))
        .map_err(|error| error.to_string())?;
    match identity.source {
        NativeRuntimeModelIdSource::ExplicitModelIdFallback => Ok(()),
        NativeRuntimeModelIdSource::MetadataGgufKey { .. }
        | NativeRuntimeModelIdSource::RuntimeSourcePathStemFallback => {
            if !native_model_refs_match(model, &identity.model_id) {
                return Err(format!(
                    "Model '{}' does not match server native local runtime source id '{}' ({}).",
                    model,
                    identity.model_id,
                    openasr_core::describe_native_runtime_model_mismatch(model, &identity.model_id)
                ));
            }
            Ok(())
        }
    }
}

fn native_model_refs_match(requested: &str, runtime_source_id: &str) -> bool {
    // Single source of truth for the bare-id / quant-alias matching contract;
    // the local tests below stay as a regression net over the same semantics.
    openasr_core::native_runtime_model_refs_match(requested, runtime_source_id)
}

/// Stable identity for the actual native runtime that will execute requests.
///
/// It intentionally ignores the client-supplied model spelling: aliases and a
/// bare runtime metadata id must all share one capacity slot. The identity comes
/// from the validated runtime pack, never from an unbounded request value.
pub(crate) fn native_model_session_key(runtime: &ServerRuntime) -> Result<String, ApiError> {
    let model_pack_path = runtime.model_pack_path.as_deref().ok_or_else(|| {
        ApiError::Backend(openasr_core::BackendError::NativeModelPackPathRequired)
    })?;
    let pack_root = openasr_core::validate_local_native_model_pack_path(model_pack_path)
        .map_err(ApiError::Backend)?;
    let identity = validate_native_runtime_pack(&pack_root).map_err(ApiError::Backend)?;
    let model_ref = parse_model_ref(&identity.model_id).map_err(|error| {
        ApiError::Backend(openasr_core::BackendError::NativeFailClosed {
            reason: format!(
                "could not canonicalize native runtime model identity '{}': {error}",
                identity.model_id
            ),
        })
    })?;
    let quant = model_ref
        .tag
        .as_deref()
        .map(openasr_core::canonical_quant_tag)
        .map(|tag| format!(":{tag}"))
        .unwrap_or_default();
    Ok(format!("native:{}{quant}", model_ref.family))
}

// Bare ids of models that are *live* in the current catalog must never be
// listed here: a native pack legitimately carries its bare family id as
// metadata (packs burn no quant tag into `openasr.model.id` -- the "bare id
// contract" enforced by `native_model_refs_match`'s `(Some(_), None) => true`
// arm above), so blacklisting a live family's bare id makes every pack for
// that family fail closed. Only list ids that no longer resolve to a
// supported catalog family/tag combination at all.
pub(crate) fn is_retired_native_model_ref(value: &str) -> bool {
    matches!(
        value,
        "whisper-tiny:q4_0"
            | "whisper-base:q4_0"
            | "whisper-large-v3-turbo:q4_0"
            | "whisper-tiny.en:q5_1"
            | "sense-voice-small"
            | "sense-voice-small:onnx"
            | "whisper-tiny.en-q5_1"
            | "sense-voice-small-onnx"
    )
}

pub(crate) fn api_runtime_model_resolution_error(error: RuntimeModelResolutionError) -> String {
    match error {
        RuntimeModelResolutionError::Registry(ModelResolutionError::UnknownModel(model)) => {
            format!("Model '{model}' was not found in the registry. Run: openasr list")
        }
        RuntimeModelResolutionError::Catalog(CatalogError::UnknownModel { reference }) => {
            format!("Model '{reference}' was not found in the registry. Run: openasr list")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod native_model_ref_tests {
    use super::native_model_refs_match;

    #[test]
    fn native_model_refs_match_catalog_suffix_and_runtime_quant_aliases() {
        assert!(native_model_refs_match(
            "qwen3-asr-0.6b:q8",
            "qwen3-asr-0.6b:q8_0"
        ));
        assert!(native_model_refs_match(
            "qwen3-asr-0.6b:q4",
            "qwen3-asr-0.6b:q4_k"
        ));
        assert!(native_model_refs_match(
            "qwen3-asr-0.6b:q4_k_m",
            "qwen3-asr-0.6b:q4_k"
        ));
        assert!(!native_model_refs_match(
            "qwen3-asr-0.6b",
            "qwen3-asr-0.6b:q8_0"
        ));
        // Quant-pinned request vs the BARE runtime source id (a native pack's
        // openasr.model.id carries no quant): must match — the daemon resolves an
        // installed pull ref to "<id>:<quant>" and the loaded pack is that model.
        // Regression guard for dictation / live captions ("daemon source id" error).
        assert!(native_model_refs_match(
            "qwen3-asr-0.6b:q8_0",
            "qwen3-asr-0.6b"
        ));
    }

    #[test]
    fn native_model_refs_reject_wrong_family_or_tag() {
        assert!(!native_model_refs_match(
            "qwen3-asr-1.7b:q8",
            "qwen3-asr-0.6b:q8_0"
        ));
        assert!(!native_model_refs_match(
            "qwen3-asr-0.6b:typo",
            "qwen3-asr-0.6b:q8_0"
        ));
    }
}

/// Coverage for the disconnect-cancel safety net: `ActiveTranscriptionCleanup`
/// must cancel the control when dropped while still armed (simulating the
/// handler future being dropped mid-decode on a client disconnect), and must
/// not when disarmed first (the normal completion path).
#[cfg(test)]
mod active_transcription_cleanup_tests {
    use std::sync::Arc;

    use super::{ActiveTranscriptionCleanup, DistributionContext, DistributionRuntime};

    fn distribution_for_test() -> DistributionContext {
        DistributionContext::new(DistributionRuntime {
            openasr_home: None,
            catalog_url: None,
            catalog_local_override: None,
        })
    }

    #[test]
    fn drop_while_armed_cancels_control_and_clears_registry() {
        let distribution = distribution_for_test();
        let control = Arc::new(openasr_core::TranscriptionControl::new());
        distribution.register_transcription("txn-disconnect", Arc::clone(&control));

        {
            let _cleanup = ActiveTranscriptionCleanup::new(
                distribution.clone(),
                "txn-disconnect".to_string(),
                Arc::clone(&control),
            );
            assert!(!control.is_canceled());
            // Dropped here without ever calling `disarm()`, exactly as happens
            // when a client disconnects and axum drops the handler future
            // before the decode call above it returns.
        }

        assert!(
            control.is_canceled(),
            "a disconnect before disarm must cancel the control so a paused/decoding worker thread wakes and exits"
        );
        assert!(
            distribution
                .transcription_control("txn-disconnect")
                .is_none()
        );
    }

    #[test]
    fn disarm_then_drop_does_not_cancel_but_still_clears_registry() {
        let distribution = distribution_for_test();
        let control = Arc::new(openasr_core::TranscriptionControl::new());
        distribution.register_transcription("txn-normal", Arc::clone(&control));

        {
            let mut cleanup = ActiveTranscriptionCleanup::new(
                distribution.clone(),
                "txn-normal".to_string(),
                Arc::clone(&control),
            );
            cleanup.disarm();
            // Normal completion path: disarmed right after the decode call
            // returns, before this guard drops at the end of the handler.
        }

        assert!(
            !control.is_canceled(),
            "a normal completion must never fire a spurious cancel"
        );
        assert!(distribution.transcription_control("txn-normal").is_none());
    }
}

// ── Multipart field parsers ───────────────────────────────────────────────────

pub(crate) fn parse_bool_field(name: &str, value: &str) -> Result<bool, ApiError> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(ApiError::BadRequest(format!(
            "Unsupported boolean value '{other}' for field '{name}'. Use true or false."
        ))),
    }
}

pub(crate) fn parse_f32_field(name: &str, value: &str) -> Result<f32, ApiError> {
    value.parse::<f32>().map_err(|error| {
        ApiError::BadRequest(format!(
            "Invalid float value '{value}' for field '{name}': {error}"
        ))
    })
}

pub(crate) fn parse_u32_field(name: &str, value: &str) -> Result<u32, ApiError> {
    value.parse::<u32>().map_err(|error| {
        ApiError::BadRequest(format!(
            "Invalid unsigned integer value '{value}' for field '{name}': {error}"
        ))
    })
}

pub(crate) fn parse_inference_threads_field(raw: &str) -> Result<u16, ApiError> {
    let value = parse_u32_field("inference_threads", raw)?;
    let threads = u16::try_from(value).map_err(|_| {
        ApiError::BadRequest(format!(
            "inference_threads must be between 1 and {MAX_INFERENCE_THREADS}."
        ))
    })?;
    if !(1..=MAX_INFERENCE_THREADS).contains(&threads) {
        return Err(ApiError::BadRequest(format!(
            "inference_threads must be between 1 and {MAX_INFERENCE_THREADS}."
        )));
    }
    Ok(threads)
}

pub(crate) fn parse_execution_target_field(raw: &str) -> Result<ExecutionTarget, ApiError> {
    match raw.trim() {
        "auto" => Ok(ExecutionTarget::Auto),
        "cpu" => Ok(ExecutionTarget::Cpu),
        "accelerated" => Ok(ExecutionTarget::Accelerated),
        other => Err(ApiError::BadRequest(format!(
            "Unsupported execution_target '{other}'. Use one of: auto, cpu, accelerated."
        ))),
    }
}

// ── Preferences / longform / phrase-bias ─────────────────────────────────────

pub(crate) fn apply_transcription_preferences(
    request: &mut TranscriptionRequest,
    preferences: &openasr_core::config::Preferences,
) {
    if request.inference_threads.is_none() {
        request.inference_threads = preferences.inference_threads;
    }
    if request.execution_target.is_none() {
        request.execution_target = Some(preferences.execution_target);
    }
}

pub(crate) fn parse_segment_mode(value: &str) -> Result<LongFormMode, ApiError> {
    match value {
        "off" => Ok(LongFormMode::Off),
        "auto" => Ok(LongFormMode::Auto),
        "fixed" => Ok(LongFormMode::Fixed),
        "energy" => Ok(LongFormMode::Energy),
        "vad" => Ok(LongFormMode::Vad),
        other => Err(ApiError::BadRequest(format!(
            "Unsupported segment_mode '{other}'. Use one of: off, auto, fixed, energy, vad."
        ))),
    }
}

pub(crate) fn build_native_longform_options(
    segment_mode: Option<&str>,
    chunk_seconds: Option<f32>,
    segment_overlap_seconds: Option<f32>,
    vad_threshold_db: Option<f32>,
    vad_min_silence_ms: Option<u32>,
    vad_padding_ms: Option<u32>,
    min_segment_seconds: Option<f32>,
    suppress_silent_slices: Option<bool>,
) -> Result<LongFormOptions, ApiError> {
    let mut options = LongFormOptions::default();
    if let Some(segment_mode) = segment_mode {
        options.mode = parse_segment_mode(segment_mode)?;
    }
    if let Some(chunk_seconds) = chunk_seconds {
        options.chunk_seconds = chunk_seconds;
    }
    if let Some(segment_overlap_seconds) = segment_overlap_seconds {
        options.overlap_seconds = segment_overlap_seconds;
    }
    if let Some(vad_threshold_db) = vad_threshold_db {
        options.energy_silence_threshold_db = vad_threshold_db;
    }
    if let Some(vad_min_silence_ms) = vad_min_silence_ms {
        options.vad.min_silence_duration_ms = vad_min_silence_ms;
    }
    if let Some(vad_padding_ms) = vad_padding_ms {
        options.padding_seconds = vad_padding_ms as f32 / 1000.0;
    }
    if let Some(min_segment_seconds) = min_segment_seconds {
        options.min_chunk_seconds = min_segment_seconds;
    }
    if let Some(suppress_silent_slices) = suppress_silent_slices {
        options.suppress_silent_slices = suppress_silent_slices;
    }
    options.validate().map_err(|error| {
        ApiError::BadRequest(format!(
            "Invalid longform segmentation configuration for native backend: {error}"
        ))
    })?;
    Ok(options)
}

pub(crate) fn build_native_longform_options_override(
    segment_mode: Option<&str>,
    chunk_seconds: Option<f32>,
    segment_overlap_seconds: Option<f32>,
    vad_threshold_db: Option<f32>,
    vad_min_silence_ms: Option<u32>,
    vad_padding_ms: Option<u32>,
    min_segment_seconds: Option<f32>,
    suppress_silent_slices: Option<bool>,
) -> Result<Option<LongFormOptions>, ApiError> {
    if segment_mode.is_none()
        && chunk_seconds.is_none()
        && segment_overlap_seconds.is_none()
        && vad_threshold_db.is_none()
        && vad_min_silence_ms.is_none()
        && vad_padding_ms.is_none()
        && min_segment_seconds.is_none()
        && suppress_silent_slices.is_none()
    {
        return Ok(None);
    }
    build_native_longform_options(
        segment_mode,
        chunk_seconds,
        segment_overlap_seconds,
        vad_threshold_db,
        vad_min_silence_ms,
        vad_padding_ms,
        min_segment_seconds,
        suppress_silent_slices,
    )
    .map(Some)
}

fn build_phrase_bias_config(
    phrases: &[String],
    hotword_boost: Option<f32>,
    phrase_bias_boost: Option<f32>,
) -> Result<Option<PhraseBiasConfig>, ApiError> {
    let boost = match (hotword_boost, phrase_bias_boost) {
        (Some(_), Some(_)) => {
            return Err(ApiError::BadRequest(
                "Use only one phrase bias boost field: hotword_boost or phrase_bias_boost."
                    .to_string(),
            ));
        }
        (Some(boost), None) | (None, Some(boost)) => Some(boost),
        (None, None) => None,
    };

    if phrases.is_empty() {
        if boost.is_some() {
            return Err(ApiError::BadRequest(
                "Phrase bias boost requires at least one hotword or phrase_bias field.".to_string(),
            ));
        }
        return Ok(None);
    }

    PhraseBiasConfig::from_phrases_with_default_boost(phrases.iter().cloned(), boost)
        .map(Some)
        .map_err(|error| {
            ApiError::BadRequest(format!("Invalid phrase bias request fields: {error}"))
        })
}

fn validate_timestamp_granularities(values: &[String]) -> Result<(), ApiError> {
    for value in values {
        match value.as_str() {
            "segment" | "word" | "word_aligned" => {}
            other => {
                return Err(ApiError::BadRequest(format!(
                    "Unsupported timestamp granularity '{other}'. Use one of: segment, word, word_aligned."
                )));
            }
        }
    }

    Ok(())
}

// ── Backend execution ─────────────────────────────────────────────────────────

/// Runs the native-runtime portion of a transcription after all audio-only
/// preparation has completed. The caller owns the blocking task, while this
/// helper keeps the permit scoped to the real native execution closure.
fn run_admitted_native_transcription<R>(
    model_session_permit: ModelSessionPermit,
    decode: impl FnOnce() -> Result<R, TranscriptionRuntimeError>,
) -> Result<R, TranscriptionRuntimeError> {
    let _model_session_permit = model_session_permit;
    decode()
}

#[cfg(test)]
mod admission_tests {
    use std::{
        num::NonZeroUsize,
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    #[tokio::test]
    async fn admitted_native_decode_retains_capacity_after_owner_is_dropped() {
        let supervisor = NativeExecutionSupervisor::new(NonZeroUsize::new(1).unwrap());
        let model_identity = "native:test-decode-lifecycle";
        let permit = supervisor
            .try_acquire(model_identity)
            .expect("first native decode must acquire the model slot");
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let task = tokio::task::spawn_blocking(move || {
            run_admitted_native_transcription(permit, move || {
                started_sender
                    .send(())
                    .expect("test must observe the native execution boundary");
                release_receiver
                    .recv()
                    .expect("test must release the native decode");
                Ok(())
            })
        });

        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("native decode must reach the admitted execution boundary");
        assert!(
            supervisor.try_acquire(model_identity).is_err(),
            "a second same-model request must be rejected while native execution runs"
        );

        // Dropping the async owner detaches `spawn_blocking`; it must not release
        // the permit before the real native decode closure exits.
        drop(task);
        assert!(
            supervisor.try_acquire(model_identity).is_err(),
            "a detached native decode must retain its model slot"
        );

        release_sender
            .send(())
            .expect("release the detached native decode");
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Ok(permit) = supervisor.try_acquire(model_identity) {
                drop(permit);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "native model capacity must return after the decode exits"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}

pub(crate) async fn transcribe_with_runtime(
    runtime: ServerRuntime,
    request: TranscriptionRequest,
    control: Option<Arc<openasr_core::TranscriptionControl>>,
) -> Result<openasr_core::Transcription, ApiError> {
    match runtime.backend {
        BackendKind::Mock => {
            // The mock backend runs a single opaque decode with no slice loop, so
            // there is no boundary to observe a pause/cancel; the control (if any)
            // is simply not installed here.
            let _ = &control;
            let prepared = prepare_audio_input(
                &request.input_path,
                &AudioPreparationOptions::new(runtime.backend),
            )
            .map_err(ApiError::AudioPreparation)?;
            let mut request = request;
            request.input_path = prepared.path().to_path_buf();
            let word_timestamps = request.word_timestamps;
            let mut transcription =
                openasr_core::api::backend::transcribe_with_mock_backend(request)
                    .map_err(ApiError::Backend)?;
            if word_timestamps {
                add_segment_word_timestamps(&mut transcription);
            }
            Ok(transcription)
        }
        BackendKind::Native => {
            tokio::task::spawn_blocking(move || {
                // Audio normalization may run an external converter or decode a
                // full upload in process, but it does not touch a native model
                // runtime. Keep it outside the per-model admission window so
                // upload preparation cannot serialize an unrelated native session
                // for the same model.
                let prepared = prepare_audio_input(
                    &request.input_path,
                    &AudioPreparationOptions::new(runtime.backend)
                        .with_ffmpeg_bin(runtime.ffmpeg_bin.clone())
                        .with_ffmpeg_bin_explicit(runtime.ffmpeg_bin_explicit)
                        .with_native_non_wav_conversion(true),
                )
                .map_err(ApiError::AudioPreparation)?;
                let model_session_permit = runtime.acquire_native_execution()?;
                run_admitted_native_transcription(model_session_permit, move || {
                    // Marks this offline (file-transcription / realtime-per-utterance
                    // backend-job) decode as active for the whole synchronous run, so
                    // the idle_unload reaper never evicts the model runtime cache out
                    // from under it; dropped (any exit path) once the decode returns.
                    let _activity_guard = NativeActivityGuard::enter();
                    // Bind the pause/cancel control to this decode thread for the whole
                    // synchronous run so the long-form slice loop can observe it; the
                    // guard clears the binding on any exit. `None` (no control requested)
                    // leaves the decode byte-identical to before.
                    let _control_guard =
                        control.map(openasr_core::install_active_transcription_control);
                    let model_pack_path = runtime.model_pack_path.clone().ok_or_else(|| {
                        TranscriptionRuntimeError::Backend(
                        openasr_core::BackendError::NativeModelPackPathRejected {
                            reason:
                                "native backend requires an explicit local .oasr runtime pack path"
                                    .to_string(),
                        },
                    )
                    })?;
                    let adapter =
                native_runtime_model_adapter_for_path(&model_pack_path).ok_or_else(|| {
                    TranscriptionRuntimeError::Backend(
                        openasr_core::BackendError::NativeFailClosed {
                            reason: format!(
                                "could not select a native model adapter from runtime source '{}'",
                                model_pack_path.display()
                            ),
                        },
                    )
                })?;
                    let mut request = request;
                    request.input_path = prepared.path().to_path_buf();
                    let word_timestamps = request.word_timestamps;
                    let model_pack = NativeAsrModelPackRef::new(
                        request.model_id.clone(),
                        adapter.model_family(),
                        model_pack_path,
                    );
                    let offline_request = NativeAsrOfflineRequest::new(request.input_path.clone())
                        .with_options(
                            NativeAsrRequestOptions::new()
                                .with_language(request.language.clone())
                                .with_prompt(request.prompt.clone())
                                .with_phrase_bias(request.phrase_bias.clone())
                                .with_inference_threads(request.inference_threads)
                                .with_diarization(request.diarize)
                                .with_word_timestamps(request.word_timestamps)
                                .with_word_timestamps_refine(request.word_timestamps_refine),
                        )
                        .with_longform(request.longform.clone())
                        .with_display_file_name(request.display_file_name.clone())
                        .with_source(request.source)
                        // The source audio's real format for the `stage=request_context`
                        // log line -- `prepared.original()` is the pre-normalization
                        // probe (WAV fmt chunk) or decode (other recognized formats)
                        // result; `None` when this pipeline could not determine it
                        // (unrecognized extension, or a format only the external
                        // ffmpeg/afconvert fallback handles).
                        .with_source_audio_format(
                            prepared.original().sample_rate_hz,
                            prepared.original().channels,
                        )
                        // Extension only, off the upload's own temp file (which
                        // preserves the client's original extension via
                        // `safe_extension_suffix` -- see `ingest_field` above); never
                        // the client-supplied file name/stem itself.
                        .with_source_container(prepared.original().extension.clone())
                        // Lets the native backend decode straight from the in-process
                        // symphonia decode's in-memory samples (uploads are almost
                        // always a non-WAV/non-conformant container) instead of
                        // re-reading `input_path` from disk -- see
                        // `PreparedAudioInput::shared_samples`.
                        .with_prepared_samples(prepared.shared_samples());
                    let executor = NativeBackendExecutor;
                    let mut transcription = NativeAsrExecutor::transcribe(
                        &executor,
                        &adapter,
                        &model_pack,
                        native_hardware_target_from_execution_target(request.execution_target),
                        offline_request,
                    )
                    .map_err(native_asr_error_to_backend)
                    .map_err(TranscriptionRuntimeError::Backend)?;
                    // The decode above only returns `Ok` after the model runtime is
                    // built (or reused) and actually ran, so this is the resident
                    // signal `/health`'s `model_resident` field reads -- see
                    // `idle_activity::native_model_is_resident`.
                    crate::idle_activity::mark_native_model_warm();
                    if word_timestamps {
                        add_segment_word_timestamps(&mut transcription);
                    }
                    drop(prepared);
                    Ok::<_, TranscriptionRuntimeError>(transcription)
                })
                .map_err(ApiError::from)
            })
            .await
            .map_err(ApiError::BackendJoin)?
        }
    }
}

pub(crate) fn native_hardware_target_from_execution_target(
    target: Option<ExecutionTarget>,
) -> NativeAsrHardwareTarget {
    match target.unwrap_or_default() {
        ExecutionTarget::Auto => NativeAsrHardwareTarget::Auto,
        ExecutionTarget::Cpu => NativeAsrHardwareTarget::Cpu,
        ExecutionTarget::Accelerated => NativeAsrHardwareTarget::Accelerated,
    }
}

fn native_asr_error_to_backend(error: NativeAsrError) -> openasr_core::BackendError {
    match error {
        NativeAsrError::PhraseBiasUnsupportedByModel {
            adapter,
            model_family,
        } => openasr_core::BackendError::PhraseBiasUnsupportedByModel {
            adapter,
            model_family,
        },
        error => openasr_core::BackendError::NativeFailClosed {
            reason: error.to_string(),
        },
    }
}

// ── Upload helpers ────────────────────────────────────────────────────────────

pub(crate) fn write_upload_temp_file(
    bytes: &[u8],
    suffix: &str,
) -> Result<tempfile::TempPath, ApiError> {
    let mut file = tempfile::Builder::new()
        .prefix("openasr-upload-")
        .suffix(suffix)
        .tempfile()
        .map_err(ApiError::TempFile)?;
    file.write_all(bytes).map_err(ApiError::TempFile)?;
    file.flush().map_err(ApiError::TempFile)?;
    Ok(file.into_temp_path())
}

/// Streams a multipart `file` field straight to a temp file, one chunk at a
/// time, instead of buffering the whole upload in memory first. This is what
/// lets `/v1/audio/transcriptions` accept multi-gigabyte recordings under
/// `MAX_TRANSCRIPTION_UPLOAD_BYTES` with O(chunk) memory instead of O(file):
/// the previous `field.bytes()` path held the entire upload in a `Bytes`
/// buffer before ever touching disk.
pub(crate) async fn write_upload_temp_file_streaming(
    mut field: Field<'_>,
    suffix: &str,
) -> Result<tempfile::TempPath, ApiError> {
    let mut file = tempfile::Builder::new()
        .prefix("openasr-upload-")
        .suffix(suffix)
        .tempfile()
        .map_err(ApiError::TempFile)?;
    let temp_dir = file.path().parent().map(Path::to_path_buf);

    // Preflight: fail closed before writing a single byte if the temp
    // volume is already below the headroom floor.
    check_temp_dir_headroom(temp_dir.as_deref())?;

    let mut since_last_check: u64 = 0;
    while let Some(chunk) = field.chunk().await.map_err(ApiError::Multipart)? {
        since_last_check = since_last_check.saturating_add(chunk.len() as u64);
        if since_last_check >= DISK_SPACE_CHECK_INTERVAL_BYTES {
            since_last_check = 0;
            check_temp_dir_headroom(temp_dir.as_deref())?;
        }
        file.write_all(&chunk).map_err(ApiError::TempFile)?;
    }
    file.flush().map_err(ApiError::TempFile)?;
    Ok(file.into_temp_path())
}

/// Fails closed with a 507 if the temp directory's volume has dropped below
/// [`MIN_FREE_DISK_HEADROOM_BYTES`] free. `None` (probe unsupported on this
/// platform, or no temp dir to check) stays permissive, matching how
/// `pull.rs`'s `ensure_available_space` treats an unknown probe.
fn check_temp_dir_headroom(temp_dir: Option<&Path>) -> Result<(), ApiError> {
    let Some(dir) = temp_dir else {
        return Ok(());
    };
    check_disk_headroom_bytes(openasr_core::available_disk_space_bytes(dir), dir)
}

/// Pure decision function split out from `check_temp_dir_headroom` so the
/// insufficient-space branch can be unit tested by injecting an `available_bytes`
/// value directly, without needing to actually fill a disk.
fn check_disk_headroom_bytes(available_bytes: Option<u64>, dir: &Path) -> Result<(), ApiError> {
    match available_bytes {
        Some(available) if available < MIN_FREE_DISK_HEADROOM_BYTES => {
            Err(ApiError::InsufficientDiskSpace(format!(
                "Not enough free disk space to receive this upload: {} MB free in '{}', \
                 need at least {} MB headroom. Free up space on that volume and retry.",
                available / (1024 * 1024),
                dir.display(),
                MIN_FREE_DISK_HEADROOM_BYTES / (1024 * 1024),
            )))
        }
        _ => Ok(()),
    }
}

fn safe_extension_suffix(file_name: &str) -> Option<String> {
    let extension = std::path::Path::new(file_name)
        .file_name()
        .map(std::path::Path::new)
        .and_then(std::path::Path::extension)
        .and_then(std::ffi::OsStr::to_str)?
        .to_ascii_lowercase();
    // Single source of truth for the recognized-audio-extension whitelist
    // (was a second hand-kept copy of the list in `openasr_core::audio::types`,
    // which could silently drift from it).
    openasr_core::recognized_audio_extensions()
        .contains(&extension.as_str())
        .then(|| format!(".{extension}"))
}

#[cfg(test)]
mod native_runtime_tests {
    use std::fs;

    use super::{
        check_disk_headroom_bytes, native_asr_error_to_backend, parse_bool_field,
        safe_extension_suffix, write_upload_temp_file,
    };

    #[test]
    fn native_phrase_bias_error_maps_to_specific_backend_error() {
        let error = native_asr_error_to_backend(
            openasr_core::NativeAsrError::PhraseBiasUnsupportedByModel {
                adapter: "ggml-family-xasr-zipformer-runtime-v1".to_string(),
                model_family: "xasr-zipformer".to_string(),
            },
        );

        match error {
            openasr_core::BackendError::PhraseBiasUnsupportedByModel {
                adapter,
                model_family,
            } => {
                assert_eq!(adapter, "ggml-family-xasr-zipformer-runtime-v1");
                assert_eq!(model_family, "xasr-zipformer");
            }
            other => panic!("expected PhraseBiasUnsupportedByModel, got {other:?}"),
        }
    }

    #[test]
    fn upload_temp_file_preserves_safe_audio_extension_and_bytes() {
        let temp_path = write_upload_temp_file(b"mock wav bytes", ".wav").unwrap();
        let path = temp_path.to_path_buf();

        assert_eq!(path.extension().and_then(|ext| ext.to_str()), Some("wav"));
        assert_eq!(fs::read(&path).unwrap(), b"mock wav bytes");
        drop(temp_path);
        assert!(!path.exists());
    }

    #[test]
    fn upload_temp_file_is_readable_while_delete_guard_is_alive() {
        let temp_path = write_upload_temp_file(b"backend readable bytes", ".wav").unwrap();
        let path: &std::path::Path = temp_path.as_ref();

        assert_eq!(fs::read(path).unwrap(), b"backend readable bytes");
    }

    #[test]
    fn safe_extension_suffix_allows_known_audio_extensions_case_insensitively() {
        assert_eq!(safe_extension_suffix("sample.WAV").as_deref(), Some(".wav"));
        assert_eq!(
            safe_extension_suffix("recording.final.FlAc").as_deref(),
            Some(".flac")
        );
        assert_eq!(safe_extension_suffix("clip.webm").as_deref(), Some(".webm"));
    }

    #[test]
    fn safe_extension_suffix_rejects_unknown_or_missing_extensions() {
        assert_eq!(safe_extension_suffix("sample.exe"), None);
        assert_eq!(safe_extension_suffix("sample"), None);
        assert_eq!(safe_extension_suffix("sample."), None);
    }

    #[test]
    fn safe_extension_suffix_uses_only_the_client_file_basename() {
        assert_eq!(
            safe_extension_suffix("..\\..\\nested\\sample.wav").as_deref(),
            Some(".wav")
        );
        assert_eq!(
            safe_extension_suffix("../../nested/sample.mp3").as_deref(),
            Some(".mp3")
        );
    }

    #[test]
    fn parse_bool_field_accepts_true_false_values() {
        assert!(parse_bool_field("diarize", "true").unwrap());
        assert!(parse_bool_field("diarize", "1").unwrap());
        assert!(!parse_bool_field("diarize", "false").unwrap());
        assert!(!parse_bool_field("diarize", "0").unwrap());
    }

    #[test]
    fn parse_bool_field_rejects_unknown_values() {
        let error = parse_bool_field("diarize", "yes").unwrap_err();

        match error {
            super::ApiError::BadRequest(message) => {
                assert!(message.contains("Unsupported boolean value 'yes'"));
                assert!(message.contains("diarize"));
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    // Disk-headroom checks below inject `available_bytes` directly (rather
    // than filling a real disk) per `check_disk_headroom_bytes`'s doc comment.

    #[test]
    fn disk_headroom_check_fails_closed_when_available_space_is_below_the_floor() {
        let dir = std::path::Path::new("/tmp/openasr-upload-test");
        let error = check_disk_headroom_bytes(Some(1024), dir).unwrap_err();

        match error {
            super::ApiError::InsufficientDiskSpace(message) => {
                assert!(message.contains("Not enough free disk space"), "{message}");
                assert!(message.contains("/tmp/openasr-upload-test"), "{message}");
            }
            other => panic!("expected InsufficientDiskSpace, got {other:?}"),
        }
    }

    #[test]
    fn disk_headroom_check_passes_when_available_space_is_ample() {
        let dir = std::path::Path::new("/tmp/openasr-upload-test");
        assert!(check_disk_headroom_bytes(Some(64 * 1024 * 1024 * 1024), dir).is_ok());
    }

    #[test]
    fn disk_headroom_check_stays_permissive_when_probe_is_unsupported() {
        // `None` means the platform/probe couldn't tell -- must not block
        // uploads on that basis, matching pull.rs's `ensure_available_space`.
        let dir = std::path::Path::new("/tmp/openasr-upload-test");
        assert!(check_disk_headroom_bytes(None, dir).is_ok());
    }

    // Locks the wire shape of GET /v1/audio/transcriptions/progress. No native run
    // is in flight in this unit test, so the idle body must stay backward
    // compatible: `total == 0` keeps legacy clients on their time-based estimate,
    // and the new `phase`/`fraction` fields are present (null / 0.0) for clients
    // that read them.
    #[tokio::test]
    async fn transcription_progress_idle_body_is_backward_compatible() {
        let axum::Json(body) = super::transcription_progress().await;
        let value = serde_json::to_value(&body).expect("progress body serializes");
        assert_eq!(value["phase"], serde_json::Value::Null);
        assert_eq!(value["fraction"], serde_json::json!(0.0));
        assert_eq!(value["done"], serde_json::json!(0));
        assert_eq!(value["total"], serde_json::json!(0));
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) enum TranscriptionRuntimeError {
    Backend(openasr_core::BackendError),
}

impl From<TranscriptionRuntimeError> for ApiError {
    fn from(error: TranscriptionRuntimeError) -> Self {
        match error {
            TranscriptionRuntimeError::Backend(error) => Self::Backend(error),
        }
    }
}
