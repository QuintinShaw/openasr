//! Native streaming + shared backend worker plumbing for realtime sessions.
//!
//! Pure code-motion from `realtime.rs`; no behavior changes.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::sync::mpsc;

use super::*;

pub(crate) struct BackendJob {
    pub(crate) utterance_id: TranscriptUtteranceId,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) segment_id: TranscriptSegmentId,
    pub(crate) model_id: String,
    pub(crate) language: Option<String>,
    pub(crate) task: Option<openasr_core::TranscriptionTask>,
    pub(crate) prompt: Option<String>,
    pub(crate) phrase_bias: Option<openasr_core::PhraseBiasConfig>,
    pub(crate) inference_threads: Option<u16>,
    pub(crate) execution_target: Option<openasr_core::ExecutionTarget>,
    pub(crate) word_timestamps: bool,
    pub(crate) display_name: String,
    pub(crate) temp_wav: tempfile::NamedTempFile,
}

pub(crate) struct BackendSuccess {
    pub(crate) utterance_id: TranscriptUtteranceId,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) segment_id: TranscriptSegmentId,
    pub(crate) text: String,
    pub(crate) language: Option<String>,
    pub(crate) words: Vec<RealtimeTranscriptWord>,
}

pub(crate) enum BackendResult {
    Final(BackendSuccess),
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NativeStreamingWorkerKey {
    pub(crate) model_pack_path: PathBuf,
    pub(crate) hardware_target: String,
    pub(crate) inference_threads: Option<u16>,
}

impl NativeStreamingWorkerKey {
    pub(crate) fn new(
        model_pack_path: impl Into<PathBuf>,
        hardware_target: openasr_core::NativeAsrHardwareTarget,
        inference_threads: Option<u16>,
    ) -> Self {
        let model_pack_path = model_pack_path.into();
        let model_pack_path = model_pack_path
            .canonicalize()
            .unwrap_or_else(|_| model_pack_path.clone());
        Self {
            model_pack_path,
            hardware_target: hardware_target.to_string(),
            inference_threads,
        }
    }
}

#[derive(Clone)]
pub(crate) struct NativeStreamingWorkerEntry {
    pub(crate) sender: mpsc::Sender<NativeStreamingWorkerMessage>,
    pub(crate) state: Arc<NativeStreamingWorkerState>,
}

pub(crate) struct NativeStreamingWorkerHandle {
    pub(crate) sender: mpsc::Sender<NativeStreamingWorkerMessage>,
    pub(crate) state: Arc<NativeStreamingWorkerState>,
}

pub(crate) struct NativeStreamingWorkerState {
    pub(crate) active_or_attaching: AtomicUsize,
    pub(crate) idle_since: Mutex<Instant>,
}

impl NativeStreamingWorkerState {
    pub(crate) fn new_acquired(now: Instant) -> Self {
        Self {
            active_or_attaching: AtomicUsize::new(1),
            idle_since: Mutex::new(now),
        }
    }

    pub(crate) fn acquire(&self) {
        self.active_or_attaching.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn release(&self) {
        let mut current = self.active_or_attaching.load(Ordering::Acquire);
        loop {
            if current == 0 {
                debug_assert!(
                    false,
                    "native streaming worker state released too many times"
                );
                return;
            }
            match self.active_or_attaching.compare_exchange(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(previous) => {
                    if previous == 1 {
                        let mut idle_since = self
                            .idle_since
                            .lock()
                            .expect("native streaming worker idle mutex poisoned");
                        *idle_since = Instant::now();
                    }
                    return;
                }
                Err(next) => current = next,
            }
        }
    }

    pub(crate) fn is_idle_for(&self, now: Instant, idle_for: Duration) -> bool {
        if self.active_or_attaching.load(Ordering::Acquire) != 0 {
            return false;
        }
        let idle_since = *self
            .idle_since
            .lock()
            .expect("native streaming worker idle mutex poisoned");
        now.checked_duration_since(idle_since).unwrap_or_default() >= idle_for
    }
}

/// Command sent to a native-streaming decode worker for one attached session.
pub(crate) enum NativeStreamingCommand {
    /// Pre-bind the family runtime on the decode thread without emitting events.
    Warm,
    PushAudio(RealtimeAudioFrame),
    Poll,
    Finalize,
    /// Forced max-duration segment split: `session.split_utterance()`, which
    /// preserves decode state when the session supports it.
    SplitUtterance,
    /// Finalize: `session.finish()`, then `session.close()` when `close` is set.
    Finish {
        close: bool,
    },
    /// Abort: `session.cancel()`.
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeStreamingCommandKind {
    Warm,
    PushAudio,
    Poll,
    Finalize,
    SplitUtterance,
    Finish,
    Cancel,
}

impl NativeStreamingCommand {
    pub(crate) fn kind(&self) -> NativeStreamingCommandKind {
        match self {
            NativeStreamingCommand::Warm => NativeStreamingCommandKind::Warm,
            NativeStreamingCommand::PushAudio(_) => NativeStreamingCommandKind::PushAudio,
            NativeStreamingCommand::Poll => NativeStreamingCommandKind::Poll,
            NativeStreamingCommand::Finalize => NativeStreamingCommandKind::Finalize,
            NativeStreamingCommand::SplitUtterance => NativeStreamingCommandKind::SplitUtterance,
            NativeStreamingCommand::Finish { .. } => NativeStreamingCommandKind::Finish,
            NativeStreamingCommand::Cancel => NativeStreamingCommandKind::Cancel,
        }
    }
}

pub(crate) struct NativeStreamingCommandEnvelope {
    pub(crate) kind: NativeStreamingCommandKind,
    pub(crate) command: NativeStreamingCommand,
}

/// Outcome the decode thread returns for one command.
pub(crate) enum NativeStreamingOutcome {
    Events {
        kind: NativeStreamingCommandKind,
        events: Vec<RealtimeEventEnvelope>,
    },
    Error {
        kind: NativeStreamingCommandKind,
        message: String,
    },
}

impl NativeStreamingOutcome {
    pub(crate) fn kind(&self) -> NativeStreamingCommandKind {
        match self {
            NativeStreamingOutcome::Events { kind, .. }
            | NativeStreamingOutcome::Error { kind, .. } => *kind,
        }
    }
}

pub(crate) enum NativeStreamingWorkerMessage {
    Attach {
        session: Box<dyn NativeAsrSession>,
        commands: mpsc::Receiver<NativeStreamingCommandEnvelope>,
        outcomes: mpsc::Sender<NativeStreamingOutcome>,
        finalize_requested: Arc<AtomicBool>,
        cancel_requested: Arc<AtomicBool>,
    },
}

/// Session-local handle to a process-shared native streaming worker. The
/// underlying OS thread is keyed by runtime identity and intentionally survives
/// session teardown, preserving thread-local ggml/Qwen decoder caches across
/// dictation sessions. The WS task still drives request/response one command at a
/// time (bounded by the existing watchdog), so frame order and emitted events
/// remain deterministic.
pub(crate) struct NativeStreamingDecodeWorker {
    pub(crate) commands: mpsc::Sender<NativeStreamingCommandEnvelope>,
    pub(crate) outcomes: mpsc::Receiver<NativeStreamingOutcome>,
    pub(crate) finalize_requested: Arc<AtomicBool>,
    pub(crate) cancel_requested: Arc<AtomicBool>,
}

impl NativeStreamingDecodeWorker {
    pub(crate) async fn attach(
        key: NativeStreamingWorkerKey,
        session: Box<dyn NativeAsrSession>,
    ) -> Result<Self, String> {
        let (command_tx, command_rx) = mpsc::channel::<NativeStreamingCommandEnvelope>(
            NATIVE_STREAMING_COMMAND_QUEUE_CAPACITY,
        );
        let (outcome_tx, outcome_rx) =
            mpsc::channel::<NativeStreamingOutcome>(NATIVE_STREAMING_OUTCOME_QUEUE_CAPACITY);
        let finalize_requested = Arc::new(AtomicBool::new(false));
        let cancel_requested = Arc::new(AtomicBool::new(false));
        let worker = native_streaming_worker_for_key(key);
        if worker
            .sender
            .send(NativeStreamingWorkerMessage::Attach {
                session,
                commands: command_rx,
                outcomes: outcome_tx,
                finalize_requested: Arc::clone(&finalize_requested),
                cancel_requested: Arc::clone(&cancel_requested),
            })
            .await
            .is_err()
        {
            worker.state.release();
            return Err("native streaming worker stopped before session attach".to_string());
        }
        Ok(Self {
            commands: command_tx,
            outcomes: outcome_rx,
            finalize_requested,
            cancel_requested,
        })
    }

    pub(crate) fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    /// Release this session's command channel. The shared decode thread observes
    /// the closed receiver, cancels/drops the active session if needed, then stays
    /// alive for the next session so thread-local runtime caches remain resident.
    pub(crate) fn join(self) {
        drop(self.commands);
    }

    pub(crate) fn detach_cancel(self) {
        self.request_cancel();
        drop(self.commands);
    }
}

pub(crate) fn native_streaming_worker_for_key(
    key: NativeStreamingWorkerKey,
) -> NativeStreamingWorkerHandle {
    spawn_native_streaming_worker_reaper();
    let registry = SHARED_NATIVE_STREAMING_WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut workers = registry
        .lock()
        .expect("native streaming worker registry mutex poisoned");
    if let Some(entry) = workers.get(&key)
        && !entry.sender.is_closed()
    {
        entry.state.acquire();
        return NativeStreamingWorkerHandle {
            sender: entry.sender.clone(),
            state: entry.state.clone(),
        };
    }

    let (sender, receiver) =
        mpsc::channel::<NativeStreamingWorkerMessage>(SHARED_BACKEND_WORKER_QUEUE_CAPACITY);
    let state = Arc::new(NativeStreamingWorkerState::new_acquired(Instant::now()));
    spawn_native_streaming_worker(receiver, state.clone());
    workers.insert(
        key,
        NativeStreamingWorkerEntry {
            sender: sender.clone(),
            state: state.clone(),
        },
    );
    NativeStreamingWorkerHandle { sender, state }
}

pub(crate) fn spawn_native_streaming_worker_reaper() {
    NATIVE_STREAMING_WORKER_REAPER_STARTED.get_or_init(|| {
        std::thread::Builder::new()
            .name("openasr-rt-decode-reaper".to_string())
            .spawn(|| {
                loop {
                    std::thread::sleep(NATIVE_STREAMING_WORKER_REAPER_INTERVAL);
                    let _ = prune_idle_native_streaming_workers(
                        Instant::now(),
                        NATIVE_STREAMING_WORKER_HARD_RELEASE_AFTER,
                    );
                }
            })
            .expect("spawn native streaming decode worker reaper");
    });
}

pub(crate) fn prune_idle_native_streaming_workers(now: Instant, idle_for: Duration) -> usize {
    let Some(registry) = SHARED_NATIVE_STREAMING_WORKERS.get() else {
        return 0;
    };
    let mut workers = registry
        .lock()
        .expect("native streaming worker registry mutex poisoned");
    let before = workers.len();
    workers.retain(|_, entry| !entry.sender.is_closed() && !entry.state.is_idle_for(now, idle_for));
    before - workers.len()
}

pub(crate) fn spawn_native_streaming_worker(
    mut receiver: mpsc::Receiver<NativeStreamingWorkerMessage>,
    state: Arc<NativeStreamingWorkerState>,
) {
    std::thread::Builder::new()
        .name("openasr-rt-decode".to_string())
        .spawn(move || {
            while let Some(message) = receiver.blocking_recv() {
                match message {
                    NativeStreamingWorkerMessage::Attach {
                        session,
                        commands,
                        outcomes,
                        finalize_requested,
                        cancel_requested,
                    } => {
                        run_native_streaming_session_on_worker(
                            session,
                            commands,
                            outcomes,
                            finalize_requested,
                            cancel_requested,
                        );
                        state.release();
                    }
                }
            }
        })
        .expect("spawn native streaming decode worker");
}

pub(crate) fn run_native_streaming_session_on_worker(
    mut session: Box<dyn NativeAsrSession>,
    mut commands: mpsc::Receiver<NativeStreamingCommandEnvelope>,
    outcomes: mpsc::Sender<NativeStreamingOutcome>,
    finalize_requested: Arc<AtomicBool>,
    cancel_requested: Arc<AtomicBool>,
) {
    session.set_cancellation_token(Arc::clone(&cancel_requested));
    let mut terminal_received = false;
    while let Some(envelope) = commands.blocking_recv() {
        let kind = envelope.kind;
        if cancel_requested.load(Ordering::Acquire) && kind != NativeStreamingCommandKind::Cancel {
            break;
        }
        let (result, terminal) = match envelope.command {
            NativeStreamingCommand::Warm => (
                warm_up_native_streaming_session_once(session.as_mut()).map(|()| Vec::new()),
                false,
            ),
            NativeStreamingCommand::PushAudio(frame) => (session.push_audio(frame), false),
            NativeStreamingCommand::Poll if finalize_requested.load(Ordering::Acquire) => {
                (Ok(Vec::new()), false)
            }
            NativeStreamingCommand::Poll => (session.poll_events(), false),
            NativeStreamingCommand::Finalize => {
                let result = session.finalize_utterance();
                finalize_requested.store(false, Ordering::Release);
                (result, false)
            }
            NativeStreamingCommand::SplitUtterance => (session.split_utterance(), false),
            NativeStreamingCommand::Finish { close } => (
                finish_native_streaming_session_in_worker(session.as_mut(), close),
                true,
            ),
            NativeStreamingCommand::Cancel => (session.cancel(), true),
        };
        if cancel_requested.load(Ordering::Acquire) && !terminal {
            break;
        }
        terminal_received |= terminal;
        let outcome = match result {
            Ok(events) => NativeStreamingOutcome::Events { kind, events },
            Err(error) => NativeStreamingOutcome::Error {
                kind,
                message: error.to_string(),
            },
        };
        let send_failed = outcomes.blocking_send(outcome).is_err();
        if terminal || send_failed {
            break;
        }
    }
    if !terminal_received {
        let _ = session.cancel();
    }
    // Drop only the per-session state. Thread-local decoder/audio-encoder caches
    // live on this worker thread and remain available to the next attachment.
}

/// Pays the cold runtime-build cost exactly once per worker thread (worker
/// threads are keyed by backend+pack and persist across sessions). The old
/// "warm only if idle for 5s" gate skipped warm-up the moment live audio
/// frames queued — which is exactly the case where the cold build then landed
/// on the first real decode and delayed the first partial by many seconds.
/// Warm-up is enqueued before any audio, so paying it immediately moves the
/// cold build ahead of speech; on an already-warm thread it is a no-op.
pub(crate) fn warm_up_native_streaming_session_once(
    session: &mut dyn NativeAsrSession,
) -> Result<(), openasr_core::NativeAsrError> {
    thread_local! {
        static WARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if WARMED.with(std::cell::Cell::get) {
        return Ok(());
    }
    session.warm_up()?;
    WARMED.with(|warmed| warmed.set(true));
    Ok(())
}

pub(crate) fn finish_native_streaming_session_in_worker(
    session: &mut dyn NativeAsrSession,
    close: bool,
) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
    let mut events = session.finish()?;
    if close {
        events.extend(session.close()?);
    }
    Ok(events)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RealtimeBackendWorkerKey {
    pub(crate) backend: String,
    pub(crate) ffmpeg_bin: Option<std::path::PathBuf>,
    pub(crate) model_pack_path: Option<std::path::PathBuf>,
}

impl RealtimeBackendWorkerKey {
    pub(crate) fn from_runtime(runtime: &ServerRuntime) -> Self {
        Self {
            backend: runtime.backend.to_string(),
            ffmpeg_bin: runtime.ffmpeg_bin.clone(),
            model_pack_path: runtime.model_pack_path.clone(),
        }
    }
}

pub(crate) struct RealtimeBackendWorkItem {
    pub(crate) session_key: String,
    pub(crate) job: BackendJob,
    pub(crate) result_sender: mpsc::Sender<BackendResult>,
    pub(crate) cancelled: Arc<AtomicBool>,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum RealtimeBackendWorkerMessage {
    Job(RealtimeBackendWorkItem),
    Completed { session_key: String },
}

pub(crate) fn realtime_backend_worker_for_runtime(
    runtime: ServerRuntime,
) -> mpsc::Sender<RealtimeBackendWorkerMessage> {
    let key = RealtimeBackendWorkerKey::from_runtime(&runtime);
    let registry = SHARED_BACKEND_WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut workers = registry
        .lock()
        .expect("realtime backend worker registry mutex poisoned");
    if let Some(sender) = workers.get(&key)
        && !sender.is_closed()
    {
        return sender.clone();
    }

    let (sender, receiver) =
        mpsc::channel::<RealtimeBackendWorkerMessage>(SHARED_BACKEND_WORKER_QUEUE_CAPACITY);
    spawn_realtime_backend_worker(runtime, receiver, sender.clone());
    workers.insert(key, sender.clone());
    sender
}

pub(crate) fn spawn_realtime_backend_worker(
    runtime: ServerRuntime,
    mut receiver: mpsc::Receiver<RealtimeBackendWorkerMessage>,
    worker_sender: mpsc::Sender<RealtimeBackendWorkerMessage>,
) {
    tokio::spawn(async move {
        let mut pending_by_session: HashMap<String, VecDeque<RealtimeBackendWorkItem>> =
            HashMap::new();
        let mut active_sessions: HashSet<String> = HashSet::new();

        while let Some(message) = receiver.recv().await {
            let collect_more = matches!(message, RealtimeBackendWorkerMessage::Job(_));
            handle_realtime_backend_worker_message(
                message,
                &mut pending_by_session,
                &mut active_sessions,
            );
            if collect_more {
                let deadline = tokio::time::Instant::now() + SHARED_BACKEND_WORKER_COLLECT_WINDOW;
                loop {
                    tokio::select! {
                        maybe_message = receiver.recv() => {
                            let Some(message) = maybe_message else {
                                break;
                            };
                            handle_realtime_backend_worker_message(
                                message,
                                &mut pending_by_session,
                                &mut active_sessions,
                            );
                        }
                        _ = tokio::time::sleep_until(deadline) => break,
                    }
                    if tokio::time::Instant::now() >= deadline {
                        break;
                    }
                }
            }
            launch_ready_realtime_backend_jobs(
                &runtime,
                &worker_sender,
                &mut pending_by_session,
                &mut active_sessions,
            );
        }
    });
}

pub(crate) fn handle_realtime_backend_worker_message(
    message: RealtimeBackendWorkerMessage,
    pending_by_session: &mut HashMap<String, VecDeque<RealtimeBackendWorkItem>>,
    active_sessions: &mut HashSet<String>,
) {
    match message {
        RealtimeBackendWorkerMessage::Job(item) => {
            pending_by_session
                .entry(item.session_key.clone())
                .or_default()
                .push_back(item);
        }
        RealtimeBackendWorkerMessage::Completed { session_key } => {
            active_sessions.remove(&session_key);
        }
    }
}

pub(crate) fn launch_ready_realtime_backend_jobs(
    runtime: &ServerRuntime,
    worker_sender: &mpsc::Sender<RealtimeBackendWorkerMessage>,
    pending_by_session: &mut HashMap<String, VecDeque<RealtimeBackendWorkItem>>,
    active_sessions: &mut HashSet<String>,
) {
    let ready_items = take_ready_realtime_backend_items(pending_by_session, active_sessions);
    for item in ready_items {
        launch_realtime_backend_work_item(runtime.clone(), worker_sender.clone(), item);
    }
}

pub(crate) fn take_ready_realtime_backend_items(
    pending_by_session: &mut HashMap<String, VecDeque<RealtimeBackendWorkItem>>,
    active_sessions: &mut HashSet<String>,
) -> Vec<RealtimeBackendWorkItem> {
    let ready_sessions = pending_by_session
        .keys()
        .filter(|session_key| !active_sessions.contains(*session_key))
        .cloned()
        .collect::<Vec<_>>();
    let mut ready_items = Vec::with_capacity(ready_sessions.len());

    for session_key in ready_sessions {
        loop {
            let (item, remove_queue) = match pending_by_session.get_mut(&session_key) {
                Some(queue) => {
                    let item = queue.pop_front();
                    (item, queue.is_empty())
                }
                None => (None, false),
            };
            if remove_queue {
                pending_by_session.remove(&session_key);
            }
            let Some(item) = item else {
                break;
            };
            if item.cancelled.load(Ordering::Relaxed) {
                continue;
            }
            active_sessions.insert(session_key.clone());
            ready_items.push(item);
            break;
        }
    }
    ready_items
}

pub(crate) fn launch_realtime_backend_work_item(
    runtime: ServerRuntime,
    worker_sender: mpsc::Sender<RealtimeBackendWorkerMessage>,
    item: RealtimeBackendWorkItem,
) {
    let session_key = item.session_key.clone();
    tokio::spawn(async move {
        let cancelled = Arc::clone(&item.cancelled);
        let result_sender = item.result_sender.clone();
        let result = run_realtime_backend_job(runtime, item.job).await;
        if !cancelled.load(Ordering::Relaxed) {
            let _ = result_sender.send(result).await;
        }
        let _ = worker_sender
            .send(RealtimeBackendWorkerMessage::Completed { session_key })
            .await;
    });
}

async fn run_realtime_backend_job(runtime: ServerRuntime, job: BackendJob) -> BackendResult {
    // Echo the requested language into the final realtime result; the core
    // Transcription carries no detected language, so the request value is the
    // only source. Capture it before job.language is moved into the builder.
    let response_language = job.language.clone();
    let request = openasr_core::TranscriptionRequest::new(job.temp_wav.path(), job.model_id)
        .with_language(job.language)
        .with_task(job.task)
        .with_prompt(job.prompt)
        .with_phrase_bias(job.phrase_bias)
        .with_inference_threads(job.inference_threads)
        .with_execution_target(job.execution_target)
        .with_word_timestamps(job.word_timestamps)
        .with_display_file_name(Some(job.display_name));
    match transcribe_with_runtime(runtime, request, None).await {
        Ok(transcription) => {
            let words = realtime_words_from_transcription(&transcription);
            BackendResult::Final(BackendSuccess {
                utterance_id: job.utterance_id,
                start_ms: job.start_ms,
                end_ms: job.end_ms,
                segment_id: job.segment_id,
                text: transcription.text,
                language: response_language,
                words,
            })
        }
        Err(error) => BackendResult::Error(format!(
            "Could not transcribe completed realtime utterance: {error}"
        )),
    }
}
