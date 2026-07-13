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
    /// Owns this attach attempt's `idle_activity` accounting. Constructed once,
    /// here, and then handed off atomically: either it rides along inside the
    /// `Attach` message and the worker thread becomes its owner once the
    /// message is actually received (see `spawn_native_streaming_worker`), or
    /// -- if the handle is dropped without ever being attached (e.g. `send`
    /// fails) -- its `Drop` fires right here and retires the activity count.
    /// There is no code path that can enter without a guard that will
    /// eventually drop exactly once, unlike the previous bare
    /// `native_activity_enter`/`native_activity_exit` pairing across two call
    /// sites, which a send failure could skip.
    ///
    /// Shared (not a bare `NativeActivityGuard`) so a stuck worker's decode
    /// watchdog or a same-key preemption can force an early release without
    /// waiting for (or being able to interrupt) the worker OS thread -- see
    /// `SharedNativeActivityGuard` and `abandon_native_streaming_worker_state`.
    pub(crate) activity: crate::idle_activity::SharedNativeActivityGuard,
}

pub(crate) struct NativeStreamingWorkerState {
    pub(crate) active_or_attaching: AtomicUsize,
    pub(crate) idle_since: Mutex<Instant>,
    /// Cancellation flag of whichever session is *currently* attached to this
    /// worker's OS thread (the one `run_native_streaming_session_on_worker` is
    /// presently driving), if any -- `None` while the thread is idle between
    /// attaches. Set (and cleared again once that attach finishes, whether
    /// normally or abandoned) by `NativeStreamingDecodeWorker::attach` and
    /// `spawn_native_streaming_worker`'s loop.
    ///
    /// A same-key attach that finds the worker still occupied checks this
    /// before deciding to queue behind it (see
    /// `native_streaming_worker_for_key`): if the occupant's owning WS
    /// connection already called `detach_cancel` (transport close, or an
    /// explicit session cancel -- the only two ways `cancel_requested` gets
    /// set in this codebase, and both already drop that attach's command
    /// channel in the same call), continuing to wait cannot help -- a
    /// cancelled attach does no further useful decode work, it is only
    /// waiting for whatever single in-flight command it was mid-processing
    /// to return. In that case the new attach immediately abandons this
    /// worker (the same eviction path the decode watchdog uses) instead of
    /// queueing behind an occupant nobody is waiting on anymore.
    pub(crate) current_cancel_requested: Mutex<Option<Arc<AtomicBool>>>,
    /// The current occupant's shared activity guard, mirroring
    /// `current_cancel_requested` above -- so the same-key preemption path
    /// can force-release it without the registry needing a live reference to
    /// the specific `WsSession` that owns it.
    pub(crate) current_activity: Mutex<Option<crate::idle_activity::SharedNativeActivityGuard>>,
    /// Set once by `abandon_native_streaming_worker_state` when this worker
    /// instance is abandoned (decode watchdog timeout, or same-key
    /// preemption). The worker OS thread cannot be interrupted, so if it is
    /// currently stuck it keeps running; this flag is what makes its
    /// eventual (possibly much later) completion harmless -- see
    /// `run_native_streaming_session_on_worker` and
    /// `warm_up_native_streaming_session_once`, which both check it after a
    /// blocking call returns, before performing any process-wide side effect.
    pub(crate) abandoned: AtomicBool,
}

impl NativeStreamingWorkerState {
    pub(crate) fn new_acquired(now: Instant) -> Self {
        Self {
            active_or_attaching: AtomicUsize::new(1),
            idle_since: Mutex::new(now),
            current_cancel_requested: Mutex::new(None),
            current_activity: Mutex::new(None),
            abandoned: AtomicBool::new(false),
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
        /// Transfers ownership of this attach attempt's `idle_activity` guard
        /// to whichever side ends up retiring it: the worker thread once it
        /// has fully processed the session (see
        /// `spawn_native_streaming_worker`), or the sender if the message
        /// never gets received at all (the `mpsc::Sender::send` error path
        /// returns the whole message, guard included, so it drops there). A
        /// clone of this same guard also lives in `NativeStreamingWorkerState
        /// ::current_activity` for as long as this attach is the worker's
        /// occupant, so a decode watchdog timeout or a same-key preemption can
        /// force an early release; see `SharedNativeActivityGuard`.
        activity: crate::idle_activity::SharedNativeActivityGuard,
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
    /// The worker-key and shared registry state this attach's OS thread is
    /// keyed under. Retained so the WS-session-level decode watchdog
    /// (`ws_session.rs`'s `enforce_native_streaming_watchdog` /
    /// `native_streaming_command`) can evict a stuck worker via
    /// `abandon_stuck_native_streaming_worker` without a second registry
    /// lookup.
    pub(crate) key: NativeStreamingWorkerKey,
    pub(crate) state: Arc<NativeStreamingWorkerState>,
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
        let worker = native_streaming_worker_for_key(key.clone());
        // `worker.activity` moves into the message: on success the worker
        // thread now owns retiring it (exactly once, after it finishes
        // processing this session, or earlier if a watchdog/preemption
        // trigger force-releases it first -- see `SharedNativeActivityGuard`);
        // on failure `send` hands the whole message -- guard included -- back
        // in its `Err`, so letting that error value drop here retires the
        // activity count without any separate manual call. There is no path
        // that both enters and never exits.
        if let Err(send_error) = worker
            .sender
            .send(NativeStreamingWorkerMessage::Attach {
                session,
                commands: command_rx,
                outcomes: outcome_tx,
                finalize_requested: Arc::clone(&finalize_requested),
                cancel_requested: Arc::clone(&cancel_requested),
                activity: worker.activity.clone(),
            })
            .await
        {
            drop(send_error);
            worker.state.release();
            return Err("native streaming worker stopped before session attach".to_string());
        }
        // Only now, after the worker thread is guaranteed to (eventually)
        // receive this Attach message, record it as the worker's current
        // occupant: a failed `send` above means no occupant change actually
        // happened, and must not stomp whatever occupant record (if any)
        // already belonged to the previous attach.
        *worker
            .state
            .current_cancel_requested
            .lock()
            .expect("native streaming worker current-cancel mutex poisoned") =
            Some(Arc::clone(&cancel_requested));
        *worker
            .state
            .current_activity
            .lock()
            .expect("native streaming worker current-activity mutex poisoned") =
            Some(worker.activity.clone());
        Ok(Self {
            commands: command_tx,
            outcomes: outcome_rx,
            finalize_requested,
            cancel_requested,
            key,
            state: worker.state,
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
    if let Some(entry) = workers.get(&key) {
        // Fix B (same-key preemption): the worker OS thread is alive, but if
        // its current occupant's owning WS connection already disconnected
        // (`detach_cancel` already set `cancel_requested` -- the same signal
        // a transport close or an explicit session cancel already produce,
        // no new protocol), waiting for it cannot help this new attach. A
        // cancelled occupant does no further useful decode work; it can only
        // still be mid-processing whatever single command it happened to be
        // running when its client disappeared, which may never return (a
        // stuck Metal `waitUntilCompleted` cannot be aborted). Abandon it now
        // instead of queueing behind it.
        let occupant_abandoned_by_client = entry
            .state
            .current_cancel_requested
            .lock()
            .expect("native streaming worker current-cancel mutex poisoned")
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        if !entry.sender.is_closed() && !occupant_abandoned_by_client {
            entry.state.acquire();
            // idle_unload must never fire while a native session is attached to
            // (or attaching to) any worker. `SharedNativeActivityGuard::new()`
            // starts that window; ownership rides in
            // `NativeStreamingWorkerHandle` (and then the `Attach` message)
            // until whichever side ends up retiring it -- see the doc
            // comments on those types.
            return NativeStreamingWorkerHandle {
                sender: entry.sender.clone(),
                state: entry.state.clone(),
                activity: crate::idle_activity::SharedNativeActivityGuard::new(),
            };
        }
        if occupant_abandoned_by_client {
            abandon_native_streaming_worker_state(
                &key,
                &entry.state,
                "same_key_preemption: previous occupant's client already disconnected",
            );
        }
        // Either the OS thread already exited (`sender.is_closed()`) or its
        // occupant was just abandoned above: fall through and spawn a fresh
        // worker. `workers.insert` a few lines down replaces this entry,
        // dropping the registry's last handle to the old thread's message
        // sender.
    }

    let (sender, receiver) =
        mpsc::channel::<NativeStreamingWorkerMessage>(SHARED_BACKEND_WORKER_QUEUE_CAPACITY);
    let state = Arc::new(NativeStreamingWorkerState::new_acquired(Instant::now()));
    let activity = crate::idle_activity::SharedNativeActivityGuard::new();
    spawn_native_streaming_worker(receiver, state.clone());
    workers.insert(
        key,
        NativeStreamingWorkerEntry {
            sender: sender.clone(),
            state: state.clone(),
        },
    );
    NativeStreamingWorkerHandle {
        sender,
        state,
        activity,
    }
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

/// Native decode worker watchdog timeouts, one budget per operation class.
/// The desktop client gives up on a stuck finalize after ~8s and force-
/// restarts the daemon (see the investigation that motivated this: the WS
/// session used to wait on the shared, HTTP-job-oriented 300s
/// `OPENASR_REALTIME_BACKEND_RESULT_TIMEOUT_SECS` default, far past that 8s
/// deadline). Every budget below is deliberately smaller so the *server*
/// fails closed with a typed `BackendCrashed` error, and evicts the stuck
/// worker (see `abandon_stuck_native_streaming_worker`), before the client
/// ever gives up and force-restarts the daemon -- an explainable error beats
/// a bare timeout plus a lost daemon.
///
/// `Warm`/`Finalize`/`SplitUtterance`/`Finish`/`Cancel` can each pay a cold
/// runtime rebuild (observed 1.5-2.1s) or a full re-decode of the buffered
/// utterance (observed sub-second for realistic dictation utterances); 6s
/// leaves more than 2x headroom over the slowest observed legitimate case
/// while still finishing a full 2s before the client's ~8s deadline.
const NATIVE_STREAMING_HEAVY_COMMAND_TIMEOUT: Duration = Duration::from_secs(6);
/// `PushAudio`/`Poll` push or pull one ~20ms audio frame's worth of
/// incremental decode state and are never expected to approach the
/// cold-rebuild budget; a much tighter 3s budget catches a stuck per-frame
/// step long before it could ever compound into a client-visible stall.
const NATIVE_STREAMING_LIGHT_COMMAND_TIMEOUT: Duration = Duration::from_secs(3);

/// Production default decode-watchdog budget for one command `kind`. Tests
/// that need a different budget (a tight one, to exercise the watchdog
/// without a real multi-second sleep, or -- for the real-model smoke test --
/// a much longer one to absorb real hardware variance) set
/// `WsSession::native_decode_timeout_override` instead of calling this
/// directly.
pub(crate) fn native_streaming_command_timeout(kind: NativeStreamingCommandKind) -> Duration {
    match kind {
        NativeStreamingCommandKind::Warm
        | NativeStreamingCommandKind::Finalize
        | NativeStreamingCommandKind::SplitUtterance
        | NativeStreamingCommandKind::Finish
        | NativeStreamingCommandKind::Cancel => NATIVE_STREAMING_HEAVY_COMMAND_TIMEOUT,
        NativeStreamingCommandKind::PushAudio | NativeStreamingCommandKind::Poll => {
            NATIVE_STREAMING_LIGHT_COMMAND_TIMEOUT
        }
    }
}

/// The state-mutating half of worker abandonment: shared by
/// `abandon_stuck_native_streaming_worker` (the decode-watchdog trigger,
/// which does not already hold the registry lock and so also evicts the
/// registry entry itself) and `native_streaming_worker_for_key`'s same-key
/// preemption check (which already holds the registry lock and lets its own
/// following `workers.insert` replace the entry instead of removing it here).
///
/// Marks the worker instance abandoned and force-releases whatever activity
/// accounting its current occupant still holds. Does **not** touch the OS
/// thread itself -- a stuck Metal `waitUntilCompleted` cannot be aborted from
/// outside its own thread, so the thread is simply left to finish (or never
/// finish) on its own; see `run_native_streaming_session_on_worker` and
/// `warm_up_native_streaming_session_once` for how its eventual late
/// completion is made harmless.
fn abandon_native_streaming_worker_state(
    key: &NativeStreamingWorkerKey,
    state: &NativeStreamingWorkerState,
    reason: &str,
) {
    state.abandoned.store(true, Ordering::Release);
    let released_activity = state
        .current_activity
        .lock()
        .expect("native streaming worker current-activity mutex poisoned")
        .take();
    if let Some(activity) = released_activity {
        activity.release();
    }
    openasr_core::stage_timing::log_event(
        "native_streaming_watchdog",
        format_args!(
            "model_pack_path={} hardware_target={} inference_threads={:?} reason={reason} \
             action=abandon_worker",
            key.model_pack_path.display(),
            key.hardware_target,
            key.inference_threads,
        ),
    );
}

/// Decode-watchdog trigger: `key`'s worker did not return a command outcome
/// within its budget (see `ws_session.rs`'s `enforce_native_streaming_watchdog`
/// / `native_streaming_command`). Evicts `key`'s registry entry -- but only if
/// it still points at exactly `state` (`Arc::ptr_eq`), guarding against a
/// race with a *different* trigger (the same-key preemption path, or another
/// watchdog firing concurrently) that already evicted or replaced it, which
/// would otherwise remove a different, perfectly healthy worker that has
/// since taken this key -- then delegates the rest of the abandonment to
/// [`abandon_native_streaming_worker_state`].
pub(crate) fn abandon_stuck_native_streaming_worker(
    key: &NativeStreamingWorkerKey,
    state: &Arc<NativeStreamingWorkerState>,
    reason: &str,
) {
    if let Some(registry) = SHARED_NATIVE_STREAMING_WORKERS.get() {
        let mut workers = registry
            .lock()
            .expect("native streaming worker registry mutex poisoned");
        if workers
            .get(key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.state, state))
        {
            workers.remove(key);
        }
    }
    abandon_native_streaming_worker_state(key, state, reason);
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
                        activity,
                    } => {
                        run_native_streaming_session_on_worker(
                            session,
                            commands,
                            outcomes,
                            finalize_requested,
                            cancel_requested,
                            &state.abandoned,
                        );
                        state.release();
                        // `activity` was handed off from
                        // `native_streaming_worker_for_key` once this message
                        // was actually received. Calling `release()` (rather
                        // than relying on `Drop`) is what makes the
                        // enter/exit pairing unconditional -- there is no
                        // longer a code path that enters without a guard
                        // whose release retires it exactly once -- and it is
                        // idempotent, so this is a no-op if a decode watchdog
                        // or same-key preemption already force-released this
                        // same guard while this call was stuck.
                        activity.release();
                        // Clear the occupant record: whoever attaches next
                        // (reusing this thread, or -- if this attach was
                        // abandoned -- a brand new one spawned under the same
                        // key) must not inherit a stale cancelled/activity
                        // record that belonged to this now-finished attach.
                        *state
                            .current_cancel_requested
                            .lock()
                            .expect("native streaming worker current-cancel mutex poisoned") = None;
                        *state
                            .current_activity
                            .lock()
                            .expect("native streaming worker current-activity mutex poisoned") =
                            None;
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
    abandoned: &AtomicBool,
) {
    session.set_cancellation_token(Arc::clone(&cancel_requested));
    let mut terminal_received = false;
    while let Some(envelope) = commands.blocking_recv() {
        let kind = envelope.kind;
        if abandoned.load(Ordering::Acquire) {
            // This attach was abandoned (decode watchdog timeout, or a
            // same-key preemption) while this thread was stuck processing a
            // previous command, or idle between commands: stop taking
            // further commands for a session the rest of the process has
            // already forgotten about (its registry entry and activity
            // accounting are already gone). Whatever was in flight when the
            // abandonment happened is not affected by this check -- it can
            // only run to completion or hang forever, per the usual
            // un-abortable-Metal-call caveat -- this only stops the *next*
            // command from starting.
            break;
        }
        if cancel_requested.load(Ordering::Acquire) && kind != NativeStreamingCommandKind::Cancel {
            break;
        }
        let (result, terminal) = match envelope.command {
            NativeStreamingCommand::Warm => (
                warm_up_native_streaming_session_once(session.as_mut(), abandoned)
                    .map(|()| Vec::new()),
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

/// Pays the cold runtime-build cost exactly once per worker thread *per
/// resident runtime generation* (worker threads are keyed by backend+pack and
/// persist across sessions, well past any `idle_unload` eviction of the
/// runtime they built -- see `idle_activity::native_unload_generation`). The
/// old "warm only if idle for 5s" gate skipped warm-up the moment live audio
/// frames queued — which is exactly the case where the cold build then landed
/// on the first real decode and delayed the first partial by many seconds.
/// Warm-up is enqueued before any audio, so paying it immediately moves the
/// cold build ahead of speech; on an already-warm thread whose runtime is
/// still resident it is a no-op.
///
/// The gate is keyed by unload generation, not a bare bool: under an opt-in
/// `idle_unload` policy shorter than the worker thread's own hard-release
/// threshold, the thread stays alive after its runtime has been evicted. A
/// bare "warmed once" bool would keep reading true post-eviction and skip
/// re-warm, silently pushing the cold rebuild back onto the first real decode
/// of the next attach -- the exact first-frame-latency regression this gate
/// exists to avoid.
pub(crate) fn warm_up_native_streaming_session_once(
    session: &mut dyn NativeAsrSession,
    abandoned: &AtomicBool,
) -> Result<(), openasr_core::NativeAsrError> {
    thread_local! {
        static WARMED_AT_GENERATION: std::cell::Cell<Option<u64>> =
            const { std::cell::Cell::new(None) };
    }
    let current_generation = crate::idle_activity::native_unload_generation();
    if WARMED_AT_GENERATION.with(std::cell::Cell::get) == Some(current_generation) {
        return Ok(());
    }
    openasr_core::stage_timing::log_event("realtime_warmup", "stage=start");
    let warmup_started = Instant::now();
    session.warm_up()?;
    openasr_core::stage_timing::log_stage("realtime_warmup", "complete", warmup_started.elapsed());
    WARMED_AT_GENERATION.with(|warmed| warmed.set(Some(current_generation)));
    // A decode watchdog or same-key preemption may have already abandoned
    // this attach (evicted its registry entry, force-released its activity
    // guard) while `session.warm_up()` above was still stuck -- this thread
    // has no way to know that until the blocking call above actually
    // returns. If so, this warm-up finished too late to matter to anything
    // still watching, and must not mark the model resident here: `/health`
    // would otherwise report a model resident on the strength of a worker
    // instance the process has already abandoned, possibly while a fresh
    // worker for the same key has its own, unrelated warm state in
    // progress -- see `idle_activity::mark_native_model_warm`.
    if !abandoned.load(Ordering::Acquire) {
        // Process-wide counterpart of the thread-local gate above, so
        // `/health` can answer "is the model resident" without reaching into
        // any worker thread's TLS -- see
        // `idle_activity::native_model_is_resident`.
        crate::idle_activity::mark_native_model_warm();
    }
    Ok(())
}

/// Warms the worker thread for the daemon's default bound native model pack
/// in the background, right after `serve_with_launch_options` finishes
/// binding the listener -- so the very first real dictation session does not
/// pay the cold model-pack-load cost (observed 1.7-2.1s) before its first
/// partial. Fire-and-forget: never blocks bind/serve/health, and any failure
/// here (bad pack, no adapter, ...) is swallowed silently -- a real request
/// still fails closed with a proper error through the normal request path.
/// Derives its `hardware_target`/`inference_threads` from the user's saved
/// preferences the same way a real WS attach without an explicit per-session
/// override does (see `realtime_execution_target_preference` /
/// `realtime_inference_threads_preference`), so a user who changed their
/// default execution target or thread count still lands this warm-up on the
/// same [`NativeStreamingWorkerKey`] their next real attach will use -- a
/// worker warmed under the *bare* default key would otherwise sit unused
/// while the real attach pays the cold-build cost on a different key. The
/// existing WS-attach `Warm` command (see `start_native_streaming_session` in
/// `ws_session.rs`) remains the fallback for whatever this boot warm-up does
/// not cover: no pack bound yet at boot, an explicit per-session
/// `inference_threads`/`execution_target` override that differs from the
/// saved preference, or the bound pack having changed since boot.
///
/// Dedup with a concurrent real attach is structural, not a flag: this
/// attaches its own short-lived "session" to the same
/// [`NativeStreamingWorkerKey`] a matching real WS attach would use, sends
/// `Warm`, waits for it to finish, and only then releases the worker -- the
/// worker thread processes one attached session's commands at a time, so a
/// real attach for the same key that arrives mid-warm-up queues behind this
/// one instead of racing a second cold build. Whichever attach's `Warm`
/// actually runs first pays the cost once (see the thread-local
/// `WARMED_AT_GENERATION` gate in `warm_up_native_streaming_session_once`);
/// every later attach on that thread reuses the now-warm state, until an
/// `idle_unload` eviction bumps the generation and forces a re-warm.
pub(crate) fn spawn_boot_native_warmup(runtime: ServerRuntime) {
    tokio::spawn(async move {
        warm_up_default_native_streaming_worker(runtime).await;
    });
}

async fn warm_up_default_native_streaming_worker(runtime: ServerRuntime) {
    if runtime.backend != openasr_core::BackendKind::Native {
        return;
    }
    let Some(model_pack_path) = runtime.model_pack_path.clone() else {
        // Fresh install / no model installed yet: nothing to warm. The daemon
        // still serves `/health`; a bound pack arrives on a future restart.
        return;
    };
    let Some(adapter) = openasr_core::native_runtime_model_adapter_for_path(&model_pack_path)
    else {
        return;
    };
    let model_pack =
        NativeAsrModelPackRef::new("native-default", adapter.model_family(), model_pack_path);
    let context = NativeAsrSessionContext::new("boot-warmup");
    // Same saved-preferences fallback a real WS attach applies when a session
    // does not set an explicit `execution_target`/`inference_threads`
    // override (`realtime_execution_target_preference` /
    // `realtime_inference_threads_preference` in `realtime/mod.rs`), so a
    // user who changed their default hardware target or thread count still
    // gets a worker warmed at the key their next attach will actually use.
    // `openasr_home()` resolution failing here (unreadable env, race) just
    // means "no preference found" -- same graceful fallback to defaults the
    // request-time paths use.
    let preferences_home = openasr_core::openasr_home().ok();
    let inference_threads = preferences_home
        .as_deref()
        .and_then(realtime_inference_threads_preference);
    let execution_target_preference = preferences_home
        .as_deref()
        .and_then(realtime_execution_target_preference);
    let options = NativeAsrRequestOptions::new().with_inference_threads(inference_threads);
    let session_config = NativeAsrStreamingSessionConfig::new()
        .with_audio_format(RealtimeAudioFormat::pcm16_mono_16khz());
    let executor = NativeBackendExecutor;
    let hardware_target = native_hardware_target_from_execution_target(execution_target_preference);
    let session = match NativeAsrExecutor::start_streaming_session(
        &executor,
        &adapter,
        &model_pack,
        hardware_target,
        context,
        options,
        session_config,
    ) {
        Ok(session) => session,
        Err(_) => return,
    };
    let key =
        NativeStreamingWorkerKey::new(model_pack.root.clone(), hardware_target, inference_threads);
    attach_and_run_boot_warmup(key, session).await;
}

/// The generic (session-agnostic) half of the boot warm-up: attach `session`
/// under `key`, send `Warm`, wait for it to finish, then release. Split out
/// from `warm_up_default_native_streaming_worker` so the async-scheduling
/// property that actually matters -- this runs to completion in the
/// background without the caller (`spawn_boot_native_warmup`, called from
/// `serve_with_launch_options` right after bind) ever awaiting it -- is
/// testable with a fake, injectable-latency [`NativeAsrSession`] instead of a
/// real model pack.
pub(crate) async fn attach_and_run_boot_warmup(
    key: NativeStreamingWorkerKey,
    session: Box<dyn NativeAsrSession>,
) {
    let Ok(mut worker) = NativeStreamingDecodeWorker::attach(key, session).await else {
        return;
    };
    let envelope = NativeStreamingCommandEnvelope {
        kind: NativeStreamingCommandKind::Warm,
        command: NativeStreamingCommand::Warm,
    };
    if worker.commands.send(envelope).await.is_ok() {
        // Wait for the Warm outcome so this session -- and the worker-key
        // acquisition it holds -- does not release before warm-up actually
        // finishes; otherwise a concurrent real attach would not meaningfully
        // "queue behind" this one.
        let _ = worker.outcomes.recv().await;
    }
    // No Finish/Cancel is sent, so the worker thread cancels this session on
    // its behalf (see `run_native_streaming_session_on_worker`) and loops
    // straight back to accept the next real Attach -- the thread-local warm
    // state this call just primed stays resident for it.
    worker.join();
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
