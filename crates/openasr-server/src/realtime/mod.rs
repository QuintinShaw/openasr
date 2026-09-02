use std::{
    collections::{HashMap, HashSet, VecDeque},
    convert::Infallible,
    io::{self, Write},
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{
        Multipart, State,
        ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade},
    },
    http::HeaderMap,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::{SinkExt, StreamExt};
use openasr_core::realtime::REALTIME_VOICE_ID_UNSUPPORTED_REASON;
use openasr_core::{
    BufferedUtterance, MAX_INFERENCE_THREADS, PhraseBiasConfig, RealtimeAudioEncoding,
    RealtimeAudioFormat, RealtimeAudioFrame, RealtimeBackendCapabilities, RealtimeBufferConfig,
    RealtimeErrorCode, RealtimeErrorEvent, RealtimeEvent, RealtimeEventEnvelope, RealtimeEventId,
    RealtimeEventSequencer, RealtimeLifecycleAction, RealtimeLifecycleEvent, RealtimeSessionConfig,
    RealtimeSessionController, RealtimeSessionId, RealtimeSessionState, RealtimeTranscriptEvent,
    RealtimeTranscriptFinal, RealtimeTranscriptRevision, RealtimeTranscriptWord,
    RealtimeUtteranceEndReason, RealtimeVadEvent, ResponseFormat, SessionCapabilitiesEvent,
    SpeechBoundaryEvent, TranscriptLifecycleResult, TranscriptSegmentId, TranscriptUpdate,
    TranscriptUtteranceId, Transcription, VadConfig, VadMode, VadSpeechStartedEvent,
    VadSpeechStoppedEvent, VadState, WordTimestamp, native_runtime_model_adapter_for_path,
    parse_model_ref,
    realtime::history::{
        DaemonHistoryKind, DaemonHistoryProvenance, DaemonHistoryRecord, DaemonHistoryStore,
    },
    resolve_runtime_model_ref, runtime_registry,
};
use openasr_core::{
    NativeAsrExecutor, NativeAsrModelAdapter, NativeAsrRequestOptions, NativeAsrSession,
    NativeAsrSessionContext, NativeAsrStreamingSessionConfig, NativeBackendExecutor,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    ApiError, DistributionContext, PENDING_IDLE_SWITCH_MESSAGE, SERVER_BUSY_MESSAGE, ServerAuth,
    ServerRuntime, apply_remote_compute_client_request_policy, file_slot_occupied,
    is_remote_compute_client_request, native_hardware_target_from_execution_target,
    parse_transcription_multipart, realtime_capabilities_for_runtime_and_distribution,
    record_file_transcription_history, transcribe_parsed_file, transcribe_with_runtime,
};

mod native_worker;
mod ws_session;

pub(crate) use native_worker::*;
pub(crate) use ws_session::*;

const DEFAULT_FRAME_DURATION_MS: u32 = 20;
const MAX_WS_MESSAGE_BYTES: usize = 64 * 1024;
const OUTGOING_EVENT_QUEUE_CAPACITY: usize = 64;
const AUDIO_FRAME_QUEUE_CAPACITY: usize = 64;
/// Native streaming decodes can run slower than the 20 ms audio cadence. Keep a
/// bounded per-session command/outcome buffer so the WS task can continue reading
/// frames while bounded Poll checkpoints are still decoding.
const NATIVE_STREAMING_COMMAND_QUEUE_CAPACITY: usize = 1024;
const NATIVE_STREAMING_OUTCOME_QUEUE_CAPACITY: usize = 1024;
const NATIVE_STREAMING_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Poll is latest-only at the worker boundary: one heavy decode may be in
/// flight, while newer audio keeps buffering cheaply. Extra queued Polls only
/// re-decode stale checkpoints, waste compute, and can delay VAD-driven
/// Finalize behind partial work.
const NATIVE_STREAMING_MAX_OUTSTANDING_POLLS: usize = 1;
/// Give startup speech/audio a chance to reach the worker before spending many
/// seconds on an opportunistic, non-interruptible runtime warm-up. If real
/// commands arrive during this grace window, Warm is skipped and the first Poll
/// owns the cold bind instead of queueing behind warm-up.
const NATIVE_STREAMING_WORKER_HARD_RELEASE_AFTER: Duration = Duration::from_secs(60);
const NATIVE_STREAMING_WORKER_REAPER_INTERVAL: Duration = Duration::from_secs(10);
const BACKEND_JOB_QUEUE_CAPACITY: usize = 4;
const SHARED_BACKEND_WORKER_QUEUE_CAPACITY: usize = BACKEND_JOB_QUEUE_CAPACITY * 64;
const SHARED_BACKEND_WORKER_COLLECT_WINDOW: Duration = Duration::from_millis(2);
const BACKPRESSURE_TIMEOUT: Duration = Duration::from_secs(1);
/// Watchdog window for the realtime backend-result wait. If a session has
/// backend jobs in flight but the loop sees neither a backend result nor any
/// socket activity for this long, the backend step is treated as hung and the
/// session fails closed instead of waiting forever and pinning a worker.
/// Deliberately generous so a legitimately slow or shared-worker-queued
/// utterance under load is not mistaken for a hang; overridable via
/// `OPENASR_REALTIME_BACKEND_RESULT_TIMEOUT_SECS`. Only the backend-result wait
/// is bounded — the socket recv side is left untimed so idle clients keep their
/// session.
const DEFAULT_BACKEND_RESULT_TIMEOUT: Duration = Duration::from_secs(300);
const BACKEND_RESULT_TIMEOUT_SECS_ENV: &str = "OPENASR_REALTIME_BACKEND_RESULT_TIMEOUT_SECS";

fn parse_backend_result_timeout(raw: Option<&str>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_BACKEND_RESULT_TIMEOUT)
}

fn backend_result_timeout() -> Duration {
    parse_backend_result_timeout(
        std::env::var(BACKEND_RESULT_TIMEOUT_SECS_ENV)
            .ok()
            .as_deref(),
    )
}
const DEFAULT_MAX_BUFFERED_FRAMES: usize = 1_510;
const MAX_VAD_BUFFER_WINDOW_MS: u32 = 120_000;
const DEFAULT_REALTIME_HOTWORD_BOOST: f32 = openasr_core::DEFAULT_PHRASE_BIAS_BOOST;
const WS_TEMP_PREFIX: &str = "openasr-ws-utterance-";
const DICTATION_SOURCE_NAME: &str = "Dictation";
const DICTATION_SPEAKERS_UNSUPPORTED_MESSAGE: &str = "Dictation does not support speaker labels. Use Live or Meeting for anonymous speaker separation.";
const DICTATION_FALLBACK_RMS_THRESHOLD: f32 = 0.001;
const DICTATION_FALLBACK_PEAK_THRESHOLD: f32 = 0.006;

static SHARED_BACKEND_WORKERS: OnceLock<
    Mutex<HashMap<RealtimeBackendWorkerKey, mpsc::Sender<RealtimeBackendWorkerMessage>>>,
> = OnceLock::new();
static SHARED_NATIVE_STREAMING_WORKERS: OnceLock<
    Mutex<HashMap<NativeStreamingWorkerKey, NativeStreamingWorkerEntry>>,
> = OnceLock::new();
static NATIVE_STREAMING_WORKER_REAPER_STARTED: OnceLock<()> = OnceLock::new();

pub(crate) async fn websocket(
    State(runtime): State<ServerRuntime>,
    axum::Extension(distribution): axum::Extension<DistributionContext>,
    axum::Extension(auth): axum::Extension<ServerAuth>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let remote_compute_client = is_remote_compute_client_request(&headers, &auth);
    let record_history = !remote_compute_client;
    let pairing_device_id = auth.pairing_device_id_for_headers(&headers);
    let caller_is_operator = auth.authorizes_pairing_admin(&headers) || !remote_compute_client;
    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .write_buffer_size(MAX_WS_MESSAGE_BYTES)
        .max_write_buffer_size(MAX_WS_MESSAGE_BYTES * 2)
        .on_upgrade(move |socket| {
            handle_websocket(
                socket,
                runtime,
                distribution,
                record_history,
                pairing_device_id,
                caller_is_operator,
                remote_compute_client,
            )
        })
}

pub(crate) async fn stream_transcription(
    runtime: ServerRuntime,
    distribution: DistributionContext,
    headers: HeaderMap,
    auth: ServerAuth,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
    remote_compute_client: bool,
) -> Result<Response, ApiError> {
    let home = distribution.openasr_home()?;
    let catalog = super::load_runtime_model_catalog(distribution.catalog_source(), &home)?;
    let mut parsed =
        parse_transcription_multipart(multipart, runtime.backend, catalog.as_ref()).await?;
    if remote_compute_client {
        apply_remote_compute_client_request_policy(&mut parsed.request);
    }
    super::reject_device_token_speaker_embeddings(&parsed.request, remote_compute_client)?;
    super::reject_streaming_speaker_embeddings(&parsed.request)?;
    if matches!(
        parsed.response_format,
        ResponseFormat::Srt | ResponseFormat::Vtt
    ) {
        return Err(ApiError::BadRequest(
            "Streaming transcription does not support SRT/VTT response_format. Use the non-streaming transcription endpoint for subtitle output.".to_string(),
        ));
    }
    let record_history = !remote_compute_client;
    if !runtime.native_execution.remote_policy().admits_new_tasks() {
        return Err(ApiError::Conflict(PENDING_IDLE_SWITCH_MESSAGE.to_string()));
    }
    if file_slot_occupied(&runtime, None) {
        return Err(ApiError::Busy(SERVER_BUSY_MESSAGE.to_string()));
    }
    let model_id = parsed.request.model_id.clone();
    let stream_started_at = Instant::now();
    let outcome = match transcribe_parsed_file(
        runtime,
        headers,
        auth,
        distribution.clone(),
        parsed,
        None,
        None,
    )
    .await
    {
        Err(error) if stream_file_job_returns_http_error(&error) => return Err(error),
        outcome => outcome,
    };

    let (sender, receiver) =
        mpsc::channel::<Result<Event, Infallible>>(OUTGOING_EVENT_QUEUE_CAPACITY);
    emit_one_shot_file_stream_events(
        &sender,
        &distribution,
        record_history,
        model_id,
        stream_started_at,
        outcome,
    )
    .await;

    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response())
}

fn stream_file_job_returns_http_error(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::Busy(_)
            | ApiError::Conflict(_)
            | ApiError::Forbidden(_)
            | ApiError::Backend(openasr_core::BackendError::TranscriptionCanceled)
    )
}

async fn emit_one_shot_file_stream_events(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    distribution: &DistributionContext,
    record_history: bool,
    model_id: String,
    stream_started_at: Instant,
    outcome: Result<super::FileTranscriptionOutcome, ApiError>,
) {
    let mut controller = one_shot_controller(&model_id);
    let mut protocol_event_seq = 1_u64;
    let mut protocol_status = "ok";
    let mut first_final_latency_ms: Option<u64> = None;
    send_sse(sender, controller.session_created_event(timestamp_now())).await;
    if let Ok(configured) =
        controller.lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
    {
        send_sse(sender, configured).await;
    }
    if let Ok(started) = controller.lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
    {
        send_sse(sender, started).await;
    }

    match outcome {
        Ok(outcome) => {
            let transcription = outcome.transcription;
            let end_ms = transcription_end_ms(&transcription);
            let history_transcription = transcription.clone();
            let words = realtime_words_from_transcription(&transcription);
            let update = TranscriptUpdate {
                utterance_id: TranscriptUtteranceId("utt_file_000001".to_string()),
                segment_id: TranscriptSegmentId("seg_file_000001".to_string()),
                revision: 1,
                text: transcription.text,
                start_ms: 0,
                end_ms,
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words,
                revises_event_id: None,
            };
            send_sse_named(
                sender,
                "segment_start",
                protocol_event_id(protocol_event_seq),
                json!({
                    "utteranceId": update.utterance_id.0,
                    "segmentId": update.segment_id.0,
                    "startMs": update.start_ms
                }),
            )
            .await;
            protocol_event_seq += 1;
            if let TranscriptLifecycleResult::Event(event) =
                controller.transcript.apply_final(update, None)
            {
                let final_segment = match &event {
                    RealtimeTranscriptEvent::Final(final_event) => Some((
                        final_event.utterance_id.clone(),
                        final_event.segment_id.clone(),
                        final_event.revision,
                    )),
                    _ => None,
                };
                if let Ok(envelope) = controller.transcript_event(event, timestamp_now()) {
                    if let Some((utterance_id, segment_id, revision)) = final_segment {
                        controller.transcript.record_final_event_id(
                            &utterance_id,
                            &segment_id,
                            revision,
                            envelope.event_id.clone(),
                        );
                    }
                    send_sse(sender, envelope).await;
                    send_sse_named(
                        sender,
                        "final",
                        protocol_event_id(protocol_event_seq),
                        json!({
                            "utteranceId": "utt_file_000001",
                            "segmentId": "seg_file_000001",
                            "startMs": 0,
                            "endMs": end_ms
                        }),
                    )
                    .await;
                    if first_final_latency_ms.is_none() {
                        first_final_latency_ms =
                            Some(stream_started_at.elapsed().as_millis() as u64);
                    }
                    protocol_event_seq += 1;
                    send_sse_named(
                        sender,
                        "segment_end",
                        protocol_event_id(protocol_event_seq),
                        json!({
                            "utteranceId": "utt_file_000001",
                            "segmentId": "seg_file_000001",
                            "endMs": end_ms
                        }),
                    )
                    .await;
                    protocol_event_seq += 1;
                }
            }
            if record_history
                && let Err(error) = record_file_transcription_history(
                    distribution,
                    &outcome.history_request,
                    &history_transcription,
                    ResponseFormat::Text,
                )
            {
                eprintln!(
                    "openasr-server: could not record streaming transcription history (continuing): {error}"
                );
            }
        }
        Err(error) => {
            protocol_status = "error";
            if let Ok(envelope) = controller.error_event(
                RealtimeErrorEvent {
                    code: realtime_error_code_for_api_error(&error),
                    message: error.to_string(),
                    recoverable: matches!(
                        error,
                        ApiError::ModelSessionCapacity(_)
                            | ApiError::Busy(_)
                            | ApiError::Conflict(_)
                    ),
                },
                timestamp_now(),
            ) {
                send_sse(sender, envelope).await;
            }
            send_sse_named(
                sender,
                "error",
                protocol_event_id(protocol_event_seq),
                json!({
                    "message": error.to_string()
                }),
            )
            .await;
            protocol_event_seq += 1;
        }
    }

    if controller.state() == RealtimeSessionState::Running
        && let Ok(stopped) = controller.lifecycle(
            RealtimeLifecycleAction::StopAudio {
                reason: "file_complete".to_string(),
            },
            timestamp_now(),
        )
    {
        send_sse(sender, stopped).await;
    }
    if !matches!(
        controller.state(),
        RealtimeSessionState::Closed | RealtimeSessionState::Cancelled
    ) && let Ok(closed) = controller.lifecycle(
        RealtimeLifecycleAction::Close {
            reason: "file_complete".to_string(),
        },
        timestamp_now(),
    ) {
        send_sse(sender, closed).await;
    }
    send_sse_named(
        sender,
        "done",
        protocol_event_id(protocol_event_seq),
        json!({
            "status": protocol_status,
            "firstFinalLatencyMs": first_final_latency_ms,
            "totalLatencyMs": stream_started_at.elapsed().as_millis() as u64
        }),
    )
    .await;
}

async fn handle_websocket(
    socket: WebSocket,
    runtime: ServerRuntime,
    distribution: DistributionContext,
    record_history: bool,
    pairing_device_id: Option<String>,
    caller_is_operator: bool,
    remote_compute_client: bool,
) {
    let (mut socket_sender, mut socket_receiver) = socket.split();
    let (event_sender, mut event_receiver) =
        mpsc::channel::<RealtimeEventEnvelope>(OUTGOING_EVENT_QUEUE_CAPACITY);
    let writer = tokio::spawn(async move {
        // Default to a clean close; a terminal (non-recoverable) error event
        // upgrades the close code so the client can distinguish an abnormal end
        // (internal / protocol / unsupported) from a normal session close.
        let mut close_code = 1000_u16;
        while let Some(envelope) = event_receiver.recv().await {
            if let RealtimeEvent::Error(error) = &envelope.event
                && !error.recoverable
            {
                close_code = ws_close_code_for_error(error.code);
            }
            let text = match serde_json::to_string(&envelope) {
                Ok(text) => text,
                Err(_) => break,
            };
            if socket_sender
                .send(Message::Text(text.into()))
                .await
                .is_err()
            {
                break;
            }
        }
        let _ = socket_sender.send(ws_close(close_code)).await;
    });

    let mut session = WsSession::new_with_remote_identity(
        runtime,
        distribution,
        event_sender,
        record_history,
        pairing_device_id,
        caller_is_operator,
        remote_compute_client,
    );
    if session.emit_capabilities().await.is_err() {
        return;
    }
    let result_timeout = backend_result_timeout();
    let mut native_poll_interval = tokio::time::interval(NATIVE_STREAMING_POLL_INTERVAL);
    native_poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if session.drain_native_streaming_outcomes().await.is_err() {
            break;
        }
        // Dispatch the next native partial Poll as soon as the previous decode's
        // outcome has drained (single-flight frees up) and there is new speech,
        // instead of waiting up to a full NATIVE_STREAMING_POLL_INTERVAL tick. This
        // keeps partial cadence decode-bound rather than tick-quantized; the timer
        // branch below stays as a heartbeat. poll_native_streaming is a no-op when
        // outstanding>=1 or no speech arrived since the last poll, so this is cheap.
        if session.is_native_streaming() && session.poll_native_streaming().await.is_err() {
            break;
        }
        if session.has_backend_results() {
            // Only arm the watchdog while a job is actually in flight, so an idle
            // session (backend wired, nothing pending) never trips it.
            let has_pending = session.has_pending_backend_jobs();
            tokio::select! {
                result = session.recv_backend_result() => {
                    if let Some(result) = result
                        && session.apply_backend_result(result).await.is_err() {
                            break;
                        }
                }
                message = socket_receiver.next() => {
                    if !session.handle_incoming_message(message).await {
                        break;
                    }
                }
                _ = tokio::time::sleep(result_timeout), if has_pending => {
                    // A backend job is in flight but neither its result nor any
                    // socket activity arrived within the watchdog window: treat
                    // the backend step as hung and fail the session closed
                    // (reusing the normal backend-error path) instead of waiting
                    // forever and pinning the worker.
                    let message = format!(
                        "backend transcription did not return a result within {}s; the backend step may be hung",
                        result_timeout.as_secs()
                    );
                    let _ = session
                        .apply_backend_result(BackendResult::Error(ApiError::Backend(
                            openasr_core::BackendError::NativeFailClosed { reason: message },
                        )))
                        .await;
                    break;
                }
            }
        } else {
            tokio::select! {
                _ = native_poll_interval.tick(), if session.is_native_streaming() => {
                    if session.poll_native_streaming().await.is_err() {
                        break;
                    }
                }
                message = socket_receiver.next() => {
                    if !session.handle_incoming_message(message).await {
                        break;
                    }
                }
            }
        }
    }

    let _ = session.finish("transport_closed", true).await;
    session.observe_idle_for_pending_switch();
    drop(session.event_sender);
    let _ = writer.await;
}

fn held_native_realtime() -> &'static Mutex<HashMap<String, NativeStreamingDecodeWorker>> {
    static HELD: OnceLock<Mutex<HashMap<String, NativeStreamingDecodeWorker>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) struct ParkedRealtimeControl {
    pub(crate) controller: Option<RealtimeSessionController>,
    pub(crate) streaming_diarizer: Option<openasr_core::diarize::streaming::StreamingDiarizer>,
    pub(crate) native_speaker_change_detector:
        Option<openasr_core::diarize::streaming::StreamingSpeakerChangeDetector>,
}

fn parked_realtime_control() -> &'static Mutex<HashMap<String, ParkedRealtimeControl>> {
    static HELD: OnceLock<Mutex<HashMap<String, ParkedRealtimeControl>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn park_realtime_control(session_id: String, parked: ParkedRealtimeControl) {
    parked_realtime_control()
        .lock()
        .expect("parked realtime control mutex poisoned")
        .insert(session_id, parked);
}

pub(crate) fn take_parked_realtime_control(session_id: &str) -> Option<ParkedRealtimeControl> {
    parked_realtime_control()
        .lock()
        .expect("parked realtime control mutex poisoned")
        .remove(session_id)
}

pub(crate) fn park_native_realtime_worker(session_id: String, worker: NativeStreamingDecodeWorker) {
    held_native_realtime()
        .lock()
        .expect("held native realtime registry mutex poisoned")
        .insert(session_id, worker);
}

pub(crate) fn take_parked_native_realtime_worker(
    session_id: &str,
) -> Option<NativeStreamingDecodeWorker> {
    held_native_realtime()
        .lock()
        .expect("held native realtime registry mutex poisoned")
        .remove(session_id)
}

pub(crate) fn spawn_held_realtime_expiry(
    policy: crate::RemoteRuntimePolicy,
    runtime: ServerRuntime,
    distribution: DistributionContext,
    _session_id: String,
) {
    let grace = policy.reconnect_grace();
    tokio::spawn(async move {
        tokio::time::sleep(grace).await;
        for (id, control) in policy.expire_held_realtime(SystemTime::now()) {
            control.request_cancel();
            if let Some(worker) = take_parked_native_realtime_worker(&id) {
                worker.detach_cancel();
            }
            let _ = take_parked_realtime_control(&id);
        }
        crate::schedule_apply_pending_idle_switch_when_native_idle(runtime, distribution);
    });
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    #[serde(rename = "session.start")]
    SessionStart { session: Box<StartSession> },
    #[serde(rename = "audio.input.configure")]
    AudioInputConfigure {
        format: Option<ClientAudioFormat>,
        frame_duration_ms: Option<u32>,
    },
    #[serde(rename = "session.cancel")]
    SessionCancel { reason: Option<String> },
    #[serde(rename = "session.close")]
    SessionClose,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartSession {
    #[serde(default)]
    session_id: Option<String>,
    model: Option<String>,
    language: Option<String>,
    task: Option<openasr_core::TranscriptionTask>,
    prompt: Option<String>,
    source_name: Option<String>,
    hotwords: Option<Vec<String>>,
    phrase_bias: Option<ClientPhraseBias>,
    inference_threads: Option<u16>,
    execution_target: Option<openasr_core::ExecutionTarget>,
    audio_format: Option<ClientAudioFormat>,
    vad: Option<ClientVadConfig>,
    partial_results: Option<bool>,
    word_timestamps: Option<bool>,
    diarize: Option<bool>,
    /// Enrolled Voice ID matching. Realtime never supports this; anonymous
    /// speaker separation is `diarize`.
    #[serde(default)]
    voice_id: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct ClientPhraseBias {
    #[serde(default)]
    phrases: Vec<String>,
    boost: Option<f32>,
}

fn start_session_requests_phrase_bias(session: &StartSession) -> bool {
    session
        .hotwords
        .as_ref()
        .is_some_and(|hotwords| !hotwords.is_empty())
        || session.phrase_bias.as_ref().is_some_and(|phrase_bias| {
            !phrase_bias.phrases.is_empty() || phrase_bias.boost.is_some()
        })
}

fn should_use_native_streaming_session(
    _source_name: Option<&str>,
    capabilities: RealtimeBackendCapabilities,
) -> bool {
    // Native true-streaming owns per-utterance Finalize/reset, so Dictation and
    // Live Captions can share the same low-latency path whenever the selected
    // pack declares true streaming support.
    capabilities.is_true_streaming
}

fn effective_session_partial_results(
    requested: bool,
    capabilities: RealtimeBackendCapabilities,
    use_native_streaming: bool,
) -> bool {
    use_native_streaming && capabilities.effective_partial_results(requested)
}

fn build_realtime_phrase_bias_config(
    session: &StartSession,
) -> Result<Option<PhraseBiasConfig>, String> {
    let Some(phrase_bias) = session.phrase_bias.as_ref() else {
        let Some(hotwords) = session
            .hotwords
            .as_ref()
            .filter(|hotwords| !hotwords.is_empty())
        else {
            return Ok(None);
        };
        return PhraseBiasConfig::from_phrases(
            hotwords
                .iter()
                .cloned()
                .map(|phrase| (phrase, DEFAULT_REALTIME_HOTWORD_BOOST)),
        )
        .map(Some)
        .map_err(|error| format!("Invalid realtime hotword fields: {error}"));
    };

    if phrase_bias.phrases.is_empty() {
        if let Some(hotwords) = session
            .hotwords
            .as_ref()
            .filter(|hotwords| !hotwords.is_empty())
        {
            let boost = phrase_bias.boost.unwrap_or(DEFAULT_REALTIME_HOTWORD_BOOST);
            return PhraseBiasConfig::from_phrases(
                hotwords.iter().cloned().map(|phrase| (phrase, boost)),
            )
            .map(Some)
            .map_err(|error| format!("Invalid realtime hotword fields: {error}"));
        }
        if phrase_bias.boost.is_some() {
            return Err(
                "Realtime phrase_bias.boost requires at least one phrase_bias phrase.".to_string(),
            );
        }
        return Ok(None);
    }

    let boost = phrase_bias.boost.unwrap_or(DEFAULT_REALTIME_HOTWORD_BOOST);
    PhraseBiasConfig::from_phrases(
        phrase_bias
            .phrases
            .iter()
            .cloned()
            .map(|phrase| (phrase, boost)),
    )
    .map(Some)
    .map_err(|error| format!("Invalid realtime phrase_bias fields: {error}"))
}

fn validate_realtime_inference_threads(value: Option<u16>) -> Result<Option<u16>, String> {
    let Some(threads) = value else {
        return Ok(None);
    };
    if !(1..=MAX_INFERENCE_THREADS).contains(&threads) {
        return Err(format!(
            "inference_threads must be between 1 and {MAX_INFERENCE_THREADS}."
        ));
    }
    Ok(Some(threads))
}

fn realtime_words_from_transcription(transcription: &Transcription) -> Vec<RealtimeTranscriptWord> {
    transcription
        .segments
        .iter()
        .flat_map(|segment| segment.words.iter())
        .map(realtime_word_from_timestamp)
        .collect()
}

fn realtime_word_from_timestamp(word: &WordTimestamp) -> RealtimeTranscriptWord {
    let start_ms = seconds_to_millis(word.start);
    RealtimeTranscriptWord {
        word: word.word.clone(),
        start_ms,
        end_ms: seconds_to_millis(word.end).max(start_ms),
        confidence: word.confidence,
    }
}

fn seconds_to_millis(value: f32) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    (value as f64 * 1000.0).round() as u64
}

/// Reads the user's saved `inference_threads` preference from `home`'s config
/// document. Takes a resolved home directory rather than a
/// [`DistributionContext`] so it can be shared by both a real WS attach
/// (which already has one) and the boot warm-up (which runs before the
/// per-request `DistributionContext` exists) -- see
/// `warm_up_default_native_streaming_worker` in `native_worker.rs`.
fn realtime_inference_threads_preference(home: &Path) -> Option<u16> {
    openasr_core::config::load_config_document(home)
        .ok()
        .and_then(|document| document.preferences.inference_threads)
}

/// Reads the user's saved `execution_target` preference from `home`'s config
/// document. See [`realtime_inference_threads_preference`] for why this takes
/// a resolved home directory instead of a [`DistributionContext`].
pub(crate) fn realtime_execution_target_preference(
    home: &Path,
) -> Option<openasr_core::ExecutionTarget> {
    openasr_core::config::load_config_document(home)
        .ok()
        .map(|document| document.preferences.execution_target)
}

#[derive(Debug, Deserialize)]
struct ClientAudioFormat {
    encoding: Option<String>,
    sample_rate_hz: Option<u32>,
    channels: Option<u16>,
}

impl Default for ClientAudioFormat {
    fn default() -> Self {
        Self {
            encoding: Some("pcm_s16le".to_string()),
            sample_rate_hz: Some(16_000),
            channels: Some(1),
        }
    }
}

impl ClientAudioFormat {
    fn try_into_realtime(self) -> Result<RealtimeAudioFormat, String> {
        let encoding = self.encoding.as_deref().unwrap_or("pcm_s16le");
        if encoding != "pcm_s16le" {
            return Err(format!(
                "Unsupported realtime audio encoding '{encoding}'. Use pcm_s16le."
            ));
        }
        let format = RealtimeAudioFormat {
            encoding: RealtimeAudioEncoding::PcmS16Le,
            sample_rate_hz: self.sample_rate_hz.unwrap_or(16_000),
            channels: self.channels.unwrap_or(1),
        };
        format
            .validate_normalized()
            .map_err(|error| error.to_string())?;
        Ok(format)
    }
}

#[derive(Debug, Default, Deserialize)]
struct ClientVadConfig {
    enabled: Option<bool>,
    /// `"neural"` selects the neural (Stream-VAD) detector; `"energy"`/`"rms"`
    /// the energy gate. Unset defaults to neural. `OPENASR_VAD` overrides this.
    engine: Option<String>,
    speech_start_ms: Option<u32>,
    speech_stop_ms: Option<u32>,
    pre_roll_ms: Option<u32>,
    max_utterance_ms: Option<u32>,
    no_speech_timeout_ms: Option<u32>,
    energy_threshold: Option<f32>,
}

impl ClientVadConfig {
    fn into_vad_config(self, frame_duration_ms: u32) -> VadConfig {
        // `enabled == Some(false)` is rejected earlier (start_session), so the
        // disabled path does not exist here; reintroduce with a test if it lands.
        let default = VadConfig::default();
        let mode = resolve_realtime_vad_mode(self.engine.as_deref());
        let energy_threshold = self.energy_threshold.unwrap_or(match mode {
            VadMode::ExternalProbability => {
                openasr_core::diarize::vad::DEFAULT_NEURAL_VAD_THRESHOLD
            }
            _ => default.energy_threshold,
        });
        // Debounce defaults are mode-conditional: neural sessions can start
        // sooner and stop a little sooner than the RMS energy gate because the
        // probability stream is less noisy. A client-supplied value wins
        // regardless of mode.
        let default_speech_start_ms = match mode {
            VadMode::ExternalProbability => {
                openasr_core::diarize::vad::DEFAULT_NEURAL_SPEECH_START_MS
            }
            _ => default.speech_start_ms,
        };
        let default_speech_stop_ms = match mode {
            VadMode::ExternalProbability => openasr_core::diarize::vad::SHORT_NEURAL_SPEECH_STOP_MS,
            _ => default.speech_stop_ms,
        };
        VadConfig {
            frame_duration_ms,
            speech_start_ms: self.speech_start_ms.unwrap_or(default_speech_start_ms),
            speech_stop_ms: self.speech_stop_ms.unwrap_or(default_speech_stop_ms),
            pre_roll_ms: self.pre_roll_ms.unwrap_or(default.pre_roll_ms),
            max_utterance_ms: self.max_utterance_ms.or(default.max_utterance_ms),
            no_speech_timeout_ms: self.no_speech_timeout_ms.or(default.no_speech_timeout_ms),
            mode,
            energy_threshold,
        }
    }
}

/// Resolve the realtime VAD mode: `OPENASR_VAD` wins, then the client's `engine`,
/// else **default to the neural detector** (it is more accurate at endpointing and
/// unlocks the shorter neural hangover; an explicit `energy`/`rms` opts out).
/// Delegates to the shared `openasr-core` resolver so the server and CLI never
/// diverge.
fn resolve_realtime_vad_mode(engine: Option<&str>) -> VadMode {
    if openasr_core::diarize::vad::realtime_vad_prefers_neural(engine) {
        VadMode::ExternalProbability
    } else {
        VadMode::Energy
    }
}

async fn send_event(
    sender: &mpsc::Sender<RealtimeEventEnvelope>,
    envelope: RealtimeEventEnvelope,
) -> Result<(), ()> {
    match tokio::time::timeout(BACKPRESSURE_TIMEOUT, sender.send(envelope)).await {
        Ok(Ok(())) => Ok(()),
        _ => Err(()),
    }
}

async fn send_sse(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    envelope: RealtimeEventEnvelope,
) {
    let _ = sender.send(Ok(sse_event(envelope))).await;
}

async fn send_sse_named(
    sender: &mpsc::Sender<Result<Event, Infallible>>,
    event_name: &str,
    event_id: String,
    payload: serde_json::Value,
) {
    let event = Event::default()
        .event(event_name)
        .id(event_id)
        .json_data(payload)
        .expect("SSE payload serializes to JSON");
    let _ = sender.send(Ok(event)).await;
}

fn protocol_event_id(sequence: u64) -> String {
    format!("proto_{sequence:06}")
}

fn sse_event(envelope: RealtimeEventEnvelope) -> Event {
    let event_name = envelope.event_type;
    let event_id = envelope.event_id.0.clone();
    Event::default()
        .event(event_name)
        .id(event_id)
        .json_data(envelope)
        .expect("RealtimeEventEnvelope serializes to JSON")
}

fn ws_close(code: u16) -> Message {
    let reason = if code == 1000 {
        Utf8Bytes::from_static("openasr_session_closed")
    } else {
        Utf8Bytes::from_static("openasr_session_error")
    };
    Message::Close(Some(CloseFrame { code, reason }))
}

/// Map a terminal realtime error to a WebSocket close code: 1008 for a
/// protocol/policy violation, 1003 for unsupported data, 1001 for the client
/// going away, 1011 for internal server conditions, 1000 for a clean cancel.
fn ws_close_code_for_error(code: RealtimeErrorCode) -> u16 {
    match code {
        RealtimeErrorCode::StartupConfigError => 1008,
        RealtimeErrorCode::UnsupportedBackend | RealtimeErrorCode::UnsupportedAudioFormat => 1003,
        RealtimeErrorCode::ClientDisconnected => 1001,
        RealtimeErrorCode::Cancelled => 1000,
        _ => 1011,
    }
}

fn one_shot_controller(model_id: &str) -> RealtimeSessionController {
    let mut config =
        RealtimeSessionConfig::new(next_session_id("rt_sse").0, model_id, timestamp_now());
    config.partial_results = false;
    RealtimeSessionController::new(config).expect("one-shot realtime session config is valid")
}

fn resolve_model(
    runtime: &ServerRuntime,
    distribution: &DistributionContext,
    model: &str,
) -> Result<String, String> {
    let catalog = if distribution.catalog_source().is_some() {
        let home = distribution
            .openasr_home()
            .map_err(|error| error.to_string())?;
        super::load_runtime_model_catalog(distribution.catalog_source(), &home)
            .map_err(|error| error.to_string())?
    } else {
        None
    };
    let registry = runtime_registry(catalog.as_ref()).map_err(|error| error.to_string())?;
    match runtime.backend {
        openasr_core::BackendKind::Mock => {
            let resolved = resolve_runtime_model_ref(&registry, catalog.as_ref(), model)
                .map_err(super::api_runtime_model_resolution_error)?;
            Ok(resolved.model_id)
        }
        openasr_core::BackendKind::Native => {
            parse_model_ref(model).map_err(|error| {
                format!("Native backend requires a valid model id in session.model: {error}")
            })?;
            if super::is_retired_native_model_ref(model) {
                return Err(format!(
                    "Model '{model}' is a retired legacy metadata id and is not executable in native mode."
                ));
            }
            Ok(model.to_string())
        }
    }
}

fn realtime_error_code_for_api_error(error: &ApiError) -> RealtimeErrorCode {
    match error {
        ApiError::ModelSessionCapacity(_) | ApiError::Busy(_) | ApiError::Conflict(_) => {
            RealtimeErrorCode::BackendNotReady
        }
        ApiError::Backend(_) | ApiError::BackendJoin(_) => RealtimeErrorCode::BackendCrashed,
        ApiError::AudioPreparation(_) => RealtimeErrorCode::UnsupportedAudioFormat,
        _ => RealtimeErrorCode::StartupConfigError,
    }
}

fn realtime_buffer_config(
    frame_duration_ms: u32,
    vad: VadConfig,
) -> Result<RealtimeBufferConfig, String> {
    let max_utterance_ms = vad.max_utterance_ms.unwrap_or(30_000);
    let buffered_ms = max_utterance_ms
        .checked_add(vad.pre_roll_ms)
        .ok_or_else(|| {
            "Realtime VAD max_utterance_ms plus pre_roll_ms is too large.".to_string()
        })?;
    if buffered_ms > MAX_VAD_BUFFER_WINDOW_MS {
        return Err(format!(
            "Realtime VAD max_utterance_ms plus pre_roll_ms must be at most {MAX_VAD_BUFFER_WINDOW_MS} ms."
        ));
    }
    let sample_window_ms = buffered_ms
        .checked_add(1_000)
        .ok_or_else(|| "Realtime VAD buffered audio window is too large.".to_string())?;
    let max_buffered_frames = (buffered_ms / frame_duration_ms)
        .checked_add(2)
        .ok_or_else(|| "Realtime VAD buffered frame capacity is too large.".to_string())?
        as usize;
    let max_buffered_samples = 16_000usize
        .checked_mul(sample_window_ms as usize)
        .and_then(|samples_per_second_ms| samples_per_second_ms.checked_div(1_000))
        .ok_or_else(|| "Realtime VAD buffered sample capacity is too large.".to_string())?;

    Ok(RealtimeBufferConfig {
        frame_duration_ms,
        pre_roll_ms: vad.pre_roll_ms,
        max_buffered_frames: max_buffered_frames.max(DEFAULT_MAX_BUFFERED_FRAMES),
        max_buffered_samples,
    })
}

fn transcription_end_ms(transcription: &openasr_core::Transcription) -> u64 {
    transcription
        .segments
        .iter()
        .map(|segment| (segment.end * 1_000.0).round().max(0.0) as u64)
        .max()
        .unwrap_or(0)
}

fn dictation_fallback_has_audible_audio(frames: &[RealtimeAudioFrame]) -> bool {
    let mut sum_square = 0.0_f64;
    let mut sample_count = 0_usize;
    let mut peak = 0.0_f32;
    for frame in frames {
        for sample in frame.samples() {
            let normalized = *sample as f32 / i16::MAX as f32;
            peak = peak.max(normalized.abs());
            sum_square += f64::from(normalized * normalized);
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        return false;
    }
    let rms = (sum_square / sample_count as f64).sqrt() as f32;
    rms >= DICTATION_FALLBACK_RMS_THRESHOLD && peak >= DICTATION_FALLBACK_PEAK_THRESHOLD
}

fn contains_backend_field(value: &serde_json::Value) -> bool {
    value.get("backend").is_some()
        || value
            .get("session")
            .and_then(|session| session.get("backend"))
            .is_some()
}

#[cfg(test)]
fn should_record_history_for_headers(headers: &HeaderMap, auth: &ServerAuth) -> bool {
    !is_remote_compute_client_request(headers, auth)
}

fn write_temp_utterance_wav(utterance: &BufferedUtterance) -> io::Result<tempfile::NamedTempFile> {
    let mut file = tempfile::Builder::new()
        .prefix(WS_TEMP_PREFIX)
        .suffix(".wav")
        .tempfile()?;
    let samples = utterance
        .frames
        .iter()
        .flat_map(|frame| frame.samples().iter().copied())
        .collect::<Vec<_>>();
    write_pcm16_mono_16khz_wav(file.as_file_mut(), &samples)?;
    file.as_file_mut().flush()?;
    Ok(file)
}

fn write_pcm16_mono_16khz_wav(mut writer: impl Write, samples: &[i16]) -> io::Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let riff_len = 36u32 + data_len;
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&16_000u32.to_le_bytes())?;
    writer.write_all(&32_000u32.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}

fn next_session_id(prefix: &str) -> RealtimeSessionId {
    let mut bytes = [0u8; 16];
    if getrandom::fill(&mut bytes).is_err() {
        let pid = std::process::id().to_le_bytes();
        let nanos = std::time::Instant::now().elapsed().as_nanos().to_le_bytes();
        for (index, slot) in bytes.iter_mut().enumerate() {
            *slot = pid[index % pid.len()] ^ nanos[index % nanos.len()] ^ (index as u8);
        }
    }
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    RealtimeSessionId(format!("{prefix}_{hex}"))
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
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
