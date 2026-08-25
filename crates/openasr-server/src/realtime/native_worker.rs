//! Native streaming + shared backend worker plumbing for realtime sessions.
//!
//! Pure code-motion from `realtime.rs`; no behavior changes.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::sync::{Notify, mpsc};

use crate::ModelSessionPermit;

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
    Error(ApiError),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NativeStreamingWorkerKey {
    pub(crate) model_pack_path: PathBuf,
    pub(crate) residency_key: crate::idle_activity::NativeRuntimeResidencyKey,
    pub(crate) hardware_target: String,
    /// Resolved execution-route isolation key (provider/stable_id[/pci:...]).
    /// Coarse targets without a resolved device keep the public spelling
    /// (`auto`/`cpu`/`accelerated`) so existing single-device hosts behave as
    /// before; Exact / preferred-accelerated pins isolate per device.
    pub(crate) execution_route_key: String,
    pub(crate) inference_threads: Option<u16>,
}

impl NativeStreamingWorkerKey {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(
        model_pack_path: impl Into<PathBuf>,
        hardware_target: openasr_core::NativeAsrHardwareTarget,
        inference_threads: Option<u16>,
    ) -> Self {
        Self::with_route(model_pack_path, hardware_target, None, inference_threads)
    }

    pub(crate) fn with_route(
        model_pack_path: impl Into<PathBuf>,
        hardware_target: openasr_core::NativeAsrHardwareTarget,
        resolved_route: Option<&openasr_core::ResolvedExecutionRoute>,
        inference_threads: Option<u16>,
    ) -> Self {
        let model_pack_path = model_pack_path.into();
        let residency_key =
            crate::idle_activity::NativeRuntimeResidencyKey::legacy_path(&model_pack_path);
        Self::with_route_and_residency(
            model_pack_path,
            residency_key,
            hardware_target,
            resolved_route,
            inference_threads,
        )
    }

    pub(crate) fn with_route_and_residency(
        model_pack_path: impl Into<PathBuf>,
        residency_key: crate::idle_activity::NativeRuntimeResidencyKey,
        hardware_target: openasr_core::NativeAsrHardwareTarget,
        resolved_route: Option<&openasr_core::ResolvedExecutionRoute>,
        inference_threads: Option<u16>,
    ) -> Self {
        let model_pack_path = model_pack_path.into();
        let model_pack_path = model_pack_path
            .canonicalize()
            .unwrap_or_else(|_| model_pack_path.clone());
        let hardware_target = hardware_target.to_string();
        let execution_route_key =
            openasr_core::worker_route_isolation_key(&hardware_target, resolved_route);
        Self {
            model_pack_path,
            residency_key,
            hardware_target,
            execution_route_key,
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
    /// This attach attempt's `idle_unload` accounting guard. Constructed once,
    /// in `native_streaming_worker_for_key`, then moved into the attach's
    /// [`AttachToken`] by `NativeStreamingDecodeWorker::attach`; from there a
    /// clone rides in the `Attach` message and the worker thread retires it
    /// once it finishes the session, while `attach`'s own `send`-failure path
    /// releases it explicitly. There is no code path that enters the activity
    /// count without a guard that is eventually retired exactly once, unlike
    /// the previous bare `native_activity_enter`/`native_activity_exit`
    /// pairing across two call sites, which a send failure could skip.
    ///
    /// Shared (not a bare `NativeActivityGuard`) so a stuck worker's decode
    /// watchdog, a same-key preemption, or the owning WS's own disconnect path
    /// can force an early release without waiting for (or being able to
    /// interrupt) the worker OS thread -- see `SharedNativeActivityGuard` and
    /// `AttachToken`.
    pub(crate) activity: crate::idle_activity::SharedNativeActivityGuard,
}

/// Per-attach supervision token. One is minted by
/// [`NativeStreamingDecodeWorker::attach`] for each attach attempt and travels
/// with that attach's `Attach` message to the worker thread. Everything a
/// decode watchdog or a same-key preemption needs to act on a *single* attach
/// -- without touching a queued sibling that merely shares the same per-key
/// worker OS thread -- lives here, not on the shared
/// [`NativeStreamingWorkerState`]:
///
/// - `cancel_requested`: set by the owning WS connection on transport close or
///   an explicit session cancel; the worker checks it between commands and it
///   is what `session.set_cancellation_token` receives.
/// - `activity`: this attach's `idle_unload` accounting guard. Idempotently
///   releasable from either the worker thread (normal finish) or an external
///   supervisor (watchdog / preemption / the owning WS's own disconnect path),
///   so a stuck decode never pins `idle_unload` waiting on an OS thread that
///   cannot be interrupted.
/// - `abandoned`: set when *this specific* attach is abandoned. Makes the
///   worker skip this attach if it has not started it yet, and makes a
///   late-returning `warm_up` inert (it must not mark the model resident on
///   behalf of an attach the rest of the process has already forgotten).
///   Being per-attach rather than a single per-worker flag is the core
///   structural fix (BLOCKER 1): abandoning one attach can no longer poison a
///   healthy queued sibling on the same worker thread.
/// - `permit`: this attach's model-capacity admission permit (limit-1 per
///   model identity). Held here rather than as a bare local on the worker
///   thread's stack so that when a decode watchdog abandons a wedged worker it
///   can reclaim the permit immediately, instead of waiting on an OS thread
///   stuck in an uninterruptible decode that will never drop its stack local
///   (and would otherwise leak the admission slot, locking every later attach
///   for the same identity out at limit-1 until the daemon restarts). The
///   `Mutex<Option<...>>` is a single-owner hand-off: whichever of the worker
///   thread (normal finish) and `abandon` (watchdog / same-key preemption)
///   calls `take_permit` first gets `Some` and drops it, releasing the slot
///   exactly once; the loser gets `None`, so a late-returning wedged decode can
///   never double-release the slot nor touch a permit a fresh session has since
///   acquired. Nothing else takes the permit -- the owning WS's own disconnect
///   paths (`join` / `detach_cancel`) deliberately leave it with the worker so
///   a merely-disconnected transport cannot release capacity while the model is
///   still executing.
pub(crate) struct AttachToken {
    pub(crate) cancel_requested: Arc<AtomicBool>,
    pub(crate) activity: crate::idle_activity::SharedNativeActivityGuard,
    pub(crate) abandoned: AtomicBool,
    pub(crate) permit: Mutex<Option<ModelSessionPermit>>,
}

impl AttachToken {
    fn new(
        activity: crate::idle_activity::SharedNativeActivityGuard,
        permit: Option<ModelSessionPermit>,
    ) -> Arc<Self> {
        Arc::new(Self {
            cancel_requested: Arc::new(AtomicBool::new(false)),
            activity,
            abandoned: AtomicBool::new(false),
            permit: Mutex::new(permit),
        })
    }

    /// Take this attach's model-capacity permit out of the token, if it is
    /// still here. Returns it so the caller drops it where it chooses; dropping
    /// the returned permit is what actually releases the admission slot.
    /// Idempotent by construction -- the `Option::take` under the mutex hands
    /// the permit to exactly one caller, so whichever of the worker thread
    /// (normal completion) and `abandon` (watchdog / same-key preemption)
    /// reaches it first frees the slot once and the loser gets `None`; never a
    /// double-release, even if a wedged decode thread returns long after the
    /// watchdog reclaimed its permit.
    fn take_permit(&self) -> Option<ModelSessionPermit> {
        self.permit
            .lock()
            .expect("native streaming attach permit mutex poisoned")
            .take()
    }

    /// Abandon this one attach: mark it (so the worker stops servicing it and a
    /// late warm-up stays inert), force-release its `idle_unload` guard (so a
    /// stuck, uninterruptible decode thread stops pinning the reaper), and
    /// reclaim its model-capacity permit (so that same stuck thread stops
    /// pinning the admission slot). Idempotent -- the guard release is
    /// idempotent, the flag is a monotonic set, and `take_permit` releases the
    /// slot at most once no matter how many triggers fire or how late the
    /// wedged thread eventually returns.
    fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
        self.activity.release();
        drop(self.take_permit());
    }
}

pub(crate) struct NativeStreamingWorkerState {
    pub(crate) active_or_attaching: AtomicUsize,
    pub(crate) idle_since: Mutex<Instant>,
    /// The token of the attach the worker OS thread is *currently* processing,
    /// set by the worker thread itself the moment it begins an attach (see
    /// `spawn_native_streaming_worker`) and cleared when that attach finishes.
    /// `None` while the thread sits idle in `blocking_recv` between attaches.
    ///
    /// Recorded by the worker thread on start -- never by `attach()` at enqueue
    /// time -- so it always names the attach that actually holds the thread,
    /// not whichever attach most recently queued behind it. Both external
    /// supervisors (the decode watchdog's `abandon_stuck_native_streaming_worker`
    /// and the same-key preemption check in `native_streaming_worker_for_key`)
    /// act through this pointer, so they only ever touch the one attach
    /// occupying the thread and can never poison a queued sibling.
    pub(crate) current_token: Mutex<Option<Arc<AttachToken>>>,
}

impl NativeStreamingWorkerState {
    pub(crate) fn new_acquired(now: Instant) -> Self {
        Self {
            active_or_attaching: AtomicUsize::new(1),
            idle_since: Mutex::new(now),
            current_token: Mutex::new(None),
        }
    }

    /// Records `token` as the attach the worker thread is now driving. Called
    /// by the worker thread itself right before it starts processing an attach.
    fn begin_occupant(&self, token: Arc<AttachToken>) {
        *self
            .current_token
            .lock()
            .expect("native streaming worker current-token mutex poisoned") = Some(token);
    }

    /// Clears the current-occupant record once the worker thread finishes an
    /// attach (whether normally, cancelled, or after abandonment).
    fn clear_occupant(&self) {
        *self
            .current_token
            .lock()
            .expect("native streaming worker current-token mutex poisoned") = None;
    }

    /// Abandon whatever attach currently occupies the worker thread, if any:
    /// releasing the *real* occupant's `idle_unload` guard (freeing the reaper)
    /// and marking that occupant so its eventual late completion is inert.
    /// Takes the token out, so a second trigger firing concurrently is a no-op
    /// and a queued sibling that later becomes the occupant starts clean.
    fn abandon_current_occupant(&self) {
        let token = self
            .current_token
            .lock()
            .expect("native streaming worker current-token mutex poisoned")
            .take();
        if let Some(token) = token {
            token.abandon();
        }
    }

    /// Whether the current occupant's owning WS connection has already
    /// requested cancel (transport close or explicit session cancel). `false`
    /// when the thread is idle between attaches.
    fn current_occupant_cancelled(&self) -> bool {
        self.current_token
            .lock()
            .expect("native streaming worker current-token mutex poisoned")
            .as_ref()
            .is_some_and(|token| token.cancel_requested.load(Ordering::Acquire))
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
        /// This attach's supervision token (cancel flag, `idle_unload` guard,
        /// per-attach `abandoned` flag, and this attach's model-capacity permit
        /// -- see [`AttachToken`]). Carried by value so, if this message is
        /// never received (the `mpsc::Sender::send` error path returns the whole
        /// message), the token -- and the guard and permit inside it -- drop on
        /// the sender side and both the activity count and the admission slot
        /// still retire. Once the worker thread receives it, the worker records
        /// it as the current occupant (`begin_occupant`) and drives the session
        /// against it, then reclaims the permit (`take_permit`) once the session
        /// has actually returned; the WS session and any external supervisor
        /// hold their own clones of the same `Arc<AttachToken>`. The worker
        /// effectively owns the permit until this attached session returns,
        /// including after the WS transport has disconnected, so a detached
        /// blocking decode cannot be overlapped -- only a decode watchdog or a
        /// same-key preemption may reclaim it earlier, via `AttachToken::abandon`.
        token: Arc<AttachToken>,
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
    /// This attach's supervision token -- see [`AttachToken`]. The WS session
    /// drives cancel through it (`request_cancel`) and, on its own disconnect
    /// paths (`join`/`detach_cancel`), releases its `idle_unload` guard
    /// immediately rather than waiting on the (possibly stuck) worker thread.
    pub(crate) token: Arc<AttachToken>,
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
    #[cfg(test)]
    pub(crate) async fn attach(
        key: NativeStreamingWorkerKey,
        session: Box<dyn NativeAsrSession>,
    ) -> Result<Self, String> {
        Self::attach_admitted(key, session, None).await
    }

    pub(crate) async fn attach_admitted(
        key: NativeStreamingWorkerKey,
        session: Box<dyn NativeAsrSession>,
        model_session_permit: Option<ModelSessionPermit>,
    ) -> Result<Self, String> {
        let (command_tx, command_rx) = mpsc::channel::<NativeStreamingCommandEnvelope>(
            NATIVE_STREAMING_COMMAND_QUEUE_CAPACITY,
        );
        let (outcome_tx, outcome_rx) =
            mpsc::channel::<NativeStreamingOutcome>(NATIVE_STREAMING_OUTCOME_QUEUE_CAPACITY);
        let finalize_requested = Arc::new(AtomicBool::new(false));
        let worker = native_streaming_worker_for_key(key.clone());
        // Mint this attach's supervision token, taking ownership of the
        // handle's `idle_unload` guard. A clone rides in the `Attach` message
        // (the worker thread's copy); the returned worker keeps `token` so the
        // WS session can cancel and release its own guard. Nothing is recorded
        // as the worker's current occupant here -- the worker thread does that
        // itself when it actually begins processing this attach, so a queued
        // attach never stomps the occupant record of the one still running.
        let token = AttachToken::new(worker.activity, model_session_permit);
        // On success the worker thread owns retiring the guard and the permit
        // (once, after it finishes this session, or earlier if a
        // watchdog/preemption trigger releases them first -- both releases are
        // idempotent). On failure `send` hands the whole message -- token
        // included -- back in its `Err`; we release this attach's guard and
        // reclaim its permit explicitly so the activity count and the admission
        // slot retire now, without waiting on the token clones to all drop.
        if let Err(send_error) = worker
            .sender
            .send(NativeStreamingWorkerMessage::Attach {
                session,
                commands: command_rx,
                outcomes: outcome_tx,
                finalize_requested: Arc::clone(&finalize_requested),
                token: Arc::clone(&token),
            })
            .await
        {
            drop(send_error);
            token.activity.release();
            drop(token.take_permit());
            worker.state.release();
            return Err("native streaming worker stopped before session attach".to_string());
        }
        Ok(Self {
            commands: command_tx,
            outcomes: outcome_rx,
            finalize_requested,
            token,
            key,
            state: worker.state,
        })
    }

    pub(crate) fn request_cancel(&self) {
        self.token.cancel_requested.store(true, Ordering::Release);
    }

    /// Release this session's command channel. The shared decode thread observes
    /// the closed receiver, cancels/drops the active session if needed, then stays
    /// alive for the next session so thread-local runtime caches remain resident.
    ///
    /// Also releases this attach's own `idle_unload` guard immediately
    /// (idempotent): on a normal finish the worker thread has already retired
    /// it, but on any path where the worker is still mid-decode this is what
    /// keeps `idle_unload` from staying pinned on an OS thread that cannot be
    /// interrupted -- releasing only *this* attach's guard, which tokenization
    /// makes safe (see `AttachToken`).
    pub(crate) fn join(self) {
        self.token.activity.release();
        drop(self.commands);
    }

    pub(crate) fn detach_cancel(self) {
        self.request_cancel();
        // Disconnect path: free this attach's `idle_unload` guard now, without
        // waiting on the worker thread -- it may be stuck deep in an
        // uninterruptible decode. Safe because the token scopes the release to
        // exactly this session's own accounting (BLOCKER 2's "the disconnect
        // path releases the guard, not the big-budget watchdog").
        self.token.activity.release();
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
        // the attach it is *currently* processing (its `current_token`, set by
        // the worker thread itself, never by a queued attach) belongs to a WS
        // connection that already disconnected (`detach_cancel` already set
        // `cancel_requested` -- the same signal a transport close or explicit
        // session cancel already produce, no new protocol), waiting for it
        // cannot help this new attach. A cancelled occupant does no further
        // useful decode work; it can only still be mid-processing whatever
        // single command it happened to be running when its client
        // disappeared, which may never return (a stuck Metal
        // `waitUntilCompleted` cannot be aborted). Abandon that one occupant
        // now instead of queueing behind it -- queued siblings are untouched.
        let occupant_abandoned_by_client = entry.state.current_occupant_cancelled();
        if !entry.sender.is_closed() && !occupant_abandoned_by_client {
            entry.state.acquire();
            // idle_unload must never fire while a native session is attached to
            // (or attaching to) any worker. `SharedNativeActivityGuard::new()`
            // starts that window; ownership moves into this attach's
            // `AttachToken` (and a clone into the `Attach` message) until
            // whichever side ends up retiring it -- see the doc comments on
            // those types.
            return NativeStreamingWorkerHandle {
                sender: entry.sender.clone(),
                state: entry.state.clone(),
                activity: crate::idle_activity::SharedNativeActivityGuard::new(),
            };
        }
        if occupant_abandoned_by_client {
            // Preemption abandons only the current occupant token (freeing the
            // real culprit's guard); it does NOT go through
            // `abandon_stuck_native_streaming_worker` -- a disconnected client
            // is normal cleanup, not evidence of a hung decode thread, so it
            // must not count toward the fail-loud abandonment budget (S1). We
            // already hold the registry lock, so the entry is replaced by the
            // `workers.insert` below rather than removed here.
            entry.state.abandon_current_occupant();
            openasr_core::stage_timing::log_event(
                "native_streaming_watchdog",
                format_args!(
                    "model_pack=resident hardware_target={} execution_route_key={} inference_threads={:?} \
                     reason=same_key_preemption_client_disconnected action=abandon_occupant",
                    key.hardware_target, key.execution_route_key, key.inference_threads,
                ),
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
    spawn_native_streaming_worker(receiver, state.clone(), key.clone());
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

/// Native decode-watchdog budget: a single, command-agnostic "the decode
/// thread is genuinely hung" bound. Its only job is to catch a decode that
/// will *never* return (e.g. a Metal `waitUntilCompleted` that never
/// completes) so the server can reap the wedged worker -- not to enforce a
/// per-command UX deadline. That deadline is the desktop client's job: it
/// gives up on a stuck finalize after ~8s and self-heals by restarting the
/// daemon, so the server's role here is *post-hoc recovery* (evict the worker,
/// free the guard, keep the next attach clean), not racing to fail before 8s.
///
/// Why command-agnostic rather than the old heavy/light split: every command
/// kind can drive a real, whole-window decode in some model family, so no kind
/// is safely "light". `Poll` runs a cadence-driven partial decode over the
/// whole retained window (`IncrementalStreamingTranscriptDriver::poll_updates`
/// -> `decode_partial_if_due`); `PushAudio` decodes frame-synchronously in
/// frame-sync families (`FrameSyncStreamingTranscriptDriver::push_audio` ->
/// `decoder.accept_samples`); `Finalize`/`SplitUtterance`/`Finish`/`Warm` all
/// decode. The previous 3s "light" budget for `Poll`/`PushAudio` would have
/// false-killed a perfectly healthy long-utterance decode.
///
/// Derivation of the bound. The largest audio a single decode is ever handed
/// is the incremental window cap, `DEFAULT_TOKEN_INCREMENTAL_WINDOW_MS` = 30s
/// (forced segment trimming at `FORCE_SEGMENT_TRIM_MS` = 12s keeps committed
/// segments well under that; `finish` only re-decodes the current trailing
/// window). Worst committed CPU real-time factor on the macos-aarch64 perf
/// baselines is ~0.72, so a worst-case legitimate single decode on that
/// hardware is ~30s * 0.72 ~= 22s; allow ~2.5x for a slower/older/thermally
/// throttled low-end machine -> ~54s. Round to 60s. That is ~7x the ~4-9s a
/// real long-sentence finalize actually takes on shipped models, so a genuine
/// decode is never mistaken for a hang, while a truly wedged thread is still
/// reaped in bounded time (turning the "idle_unload pinned for 11 minutes"
/// symptom into an at-most-60s recovery -- and the disconnect path frees the
/// guard even sooner, see `NativeStreamingDecodeWorker::detach_cancel`).
const NATIVE_STREAMING_DECODE_STUCK_TIMEOUT: Duration = Duration::from_secs(60);

/// Production default decode-watchdog budget for one command `kind`. Currently
/// command-agnostic (every kind can decode -- see
/// `NATIVE_STREAMING_DECODE_STUCK_TIMEOUT`), but kept keyed by `kind` so a
/// future provably-decode-free command could carry a tighter budget without
/// re-plumbing every call site. Tests that need a different budget (a tight
/// one, to exercise the watchdog without a real multi-second sleep, or -- for
/// the real-model smoke test -- their own value) set
/// `WsSession::native_decode_timeout_override` instead of calling this
/// directly.
pub(crate) fn native_streaming_command_timeout(kind: NativeStreamingCommandKind) -> Duration {
    match kind {
        NativeStreamingCommandKind::Warm
        | NativeStreamingCommandKind::PushAudio
        | NativeStreamingCommandKind::Poll
        | NativeStreamingCommandKind::Finalize
        | NativeStreamingCommandKind::SplitUtterance
        | NativeStreamingCommandKind::Finish
        | NativeStreamingCommandKind::Cancel => NATIVE_STREAMING_DECODE_STUCK_TIMEOUT,
    }
}

/// Process-wide count of workers abandoned by the decode watchdog because a
/// decode never returned within `NATIVE_STREAMING_DECODE_STUCK_TIMEOUT`. Each
/// such abandonment leaks one OS thread that is (presumed) permanently wedged
/// inside an uninterruptible decode -- and that thread pins its thread-local
/// ggml/Metal model runtime resident, so the leaked memory is a whole model's
/// worth, not a trickle. A handful of these means the process is in an
/// unrecoverable state that only a restart can clear (S1). Same-key
/// preemption of a merely-disconnected client does NOT count here -- that path
/// (`native_streaming_worker_for_key`) usually lets the old thread exit
/// cleanly and is normal cleanup, not evidence of a hang.
static ABANDONED_STUCK_WORKER_COUNT: AtomicUsize = AtomicUsize::new(0);

/// How many watchdog-abandoned (wedged, leaked) workers the daemon tolerates
/// before it deliberately fails loud and exits (S1). One or two can be a
/// transient GPU/driver stall; by the third distinct hung decode the leaked
/// model runtimes are a real, unrecoverable memory leak, and a clean restart
/// (the desktop supervisor self-heals via `serve --parent-pid`, and a bare
/// daemon's operator restarts it) is strictly better than limping along
/// leaking a model's worth of RAM per incident.
pub(crate) const MAX_ABANDONED_STUCK_WORKERS_BEFORE_EXIT: usize = 3;

/// Exit code the daemon uses when it fails loud on too many abandoned workers
/// (S1). Distinct from a crash/panic so a supervisor/log can tell this
/// deliberate self-termination apart from an unexpected fault.
const ABANDONED_STUCK_WORKER_EXIT_CODE: i32 = 87;

/// Process-wide count of watchdog-abandoned (presumed-wedged) native streaming
/// workers. Surfaced by `/health` as `abandoned_worker_count` next to
/// `native_active_count`, so an operator/support session can see the daemon
/// accumulating hung decodes before the fail-loud exit threshold (S1).
pub(crate) fn abandoned_stuck_worker_count() -> usize {
    ABANDONED_STUCK_WORKER_COUNT.load(Ordering::Acquire)
}

/// Pure threshold predicate for the fail-loud decision, split out so it is
/// unit-testable without triggering a real `process::exit` (S1/S2).
pub(crate) fn abandonment_count_requires_fail_loud(count: usize) -> bool {
    count >= MAX_ABANDONED_STUCK_WORKERS_BEFORE_EXIT
}

/// Decode-watchdog trigger: `key`'s worker did not return a command outcome
/// within its budget (see `ws_session.rs`'s `enforce_native_streaming_watchdog`
/// / `native_streaming_command`). Evicts `key`'s registry entry -- but only if
/// it still points at exactly `state` (`Arc::ptr_eq`), guarding against a
/// race with a *different* trigger (the same-key preemption path, or another
/// watchdog firing concurrently) that already evicted or replaced it, which
/// would otherwise remove a different, perfectly healthy worker that has
/// since taken this key -- then abandons the one attach that occupies the
/// stuck thread (freeing the real culprit's `idle_unload` guard) and records
/// the abandonment against the fail-loud budget (S1).
///
/// Does **not** touch the OS thread itself -- a stuck Metal
/// `waitUntilCompleted` cannot be aborted from outside its own thread, so the
/// thread is simply left to finish (or never finish) on its own; see
/// `run_native_streaming_session_on_worker` and
/// `warm_up_native_streaming_session_once` for how its eventual late
/// completion is made harmless.
pub(crate) fn abandon_stuck_native_streaming_worker(
    key: &NativeStreamingWorkerKey,
    state: &Arc<NativeStreamingWorkerState>,
    reason: &str,
) {
    let evicted_this_worker = if let Some(registry) = SHARED_NATIVE_STREAMING_WORKERS.get() {
        let mut workers = registry
            .lock()
            .expect("native streaming worker registry mutex poisoned");
        if workers
            .get(key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.state, state))
        {
            workers.remove(key);
            true
        } else {
            false
        }
    } else {
        false
    };
    // Abandon the occupant on every trigger (idempotent), but count -- and
    // possibly fail loud -- only on the one that actually evicted this worker.
    // Two WS sessions racing the same wedged worker (its stuck occupant plus a
    // sibling queued behind it, both timing out) must count as one leaked
    // thread, not two, or the fail-loud budget would trip too early.
    state.abandon_current_occupant();
    if !evicted_this_worker {
        return;
    }
    let abandoned_count = ABANDONED_STUCK_WORKER_COUNT.fetch_add(1, Ordering::AcqRel) + 1;
    openasr_core::stage_timing::log_event(
        "native_streaming_watchdog",
        format_args!(
            "model_pack=resident hardware_target={} execution_route_key={} inference_threads={:?} reason={reason} \
             action=abandon_worker abandoned_workers={abandoned_count}",
            key.hardware_target, key.execution_route_key, key.inference_threads,
        ),
    );
    if abandonment_count_requires_fail_loud(abandoned_count) {
        eprintln!(
            "openasr-server: {abandoned_count} native streaming decode workers have hung and been \
             abandoned; each leaks a resident model runtime, so this is now an unrecoverable \
             memory leak. Exiting (code {ABANDONED_STUCK_WORKER_EXIT_CODE}) so the supervising \
             process can restart the daemon with a clean slate."
        );
        openasr_core::stage_timing::log_event(
            "native_streaming_watchdog",
            format_args!(
                "action=fail_loud_exit abandoned_workers={abandoned_count} \
                 exit_code={ABANDONED_STUCK_WORKER_EXIT_CODE}"
            ),
        );
        // Never inside the test binary: the counter is process-wide, so
        // several watchdog-exercising tests sharing one `cargo test` process
        // (not `cargo nextest`, which isolates each test) could otherwise trip
        // this and kill the whole run. The threshold logic itself is covered
        // by `abandonment_count_requires_fail_loud`'s unit test.
        #[cfg(not(test))]
        std::process::exit(ABANDONED_STUCK_WORKER_EXIT_CODE);
    }
}

pub(crate) fn spawn_native_streaming_worker(
    mut receiver: mpsc::Receiver<NativeStreamingWorkerMessage>,
    state: Arc<NativeStreamingWorkerState>,
    key: NativeStreamingWorkerKey,
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
                        token,
                    } => {
                        // Record this attach as the current occupant only now,
                        // as the thread actually begins driving it -- so
                        // external supervisors always act on the attach that
                        // holds the thread, never one still queued behind it.
                        // This attach's model-capacity permit lives in `token`
                        // and stays owned effectively by this worker thread
                        // until `run_native_streaming_session_on_worker`
                        // returns; a disconnected WS cannot release it early
                        // (only a decode watchdog / same-key preemption may, via
                        // `AttachToken::abandon`).
                        state.begin_occupant(Arc::clone(&token));
                        run_native_streaming_session_on_worker(
                            session,
                            commands,
                            outcomes,
                            finalize_requested,
                            &token,
                            &key.residency_key,
                        );
                        // Clear the occupant record first, then retire the
                        // per-key acquire count, this attach's activity guard,
                        // and its model-capacity permit.
                        // `token.activity.release()` and `token.take_permit()`
                        // (rather than relying on `Drop`) make the enter/exit
                        // and acquire/release pairings unconditional and are
                        // idempotent -- each is a no-op if a decode watchdog,
                        // same-key preemption, or the owning WS's own disconnect
                        // path already released this same guard / reclaimed this
                        // permit while this call was stuck. This is what keeps a
                        // late-returning wedged decode from double-releasing the
                        // admission slot.
                        state.clear_occupant();
                        state.release();
                        token.activity.release();
                        drop(token.take_permit());
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
    token: &AttachToken,
    residency_key: &crate::idle_activity::NativeRuntimeResidencyKey,
) {
    let cancel_requested = &token.cancel_requested;
    let abandoned = &token.abandoned;
    session.set_cancellation_token(Arc::clone(cancel_requested));
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
                warm_up_native_streaming_session_once(session.as_mut(), abandoned, residency_key)
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
/// runtime they built -- see `idle_activity::native_runtime_generation`). The
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
    residency_key: &crate::idle_activity::NativeRuntimeResidencyKey,
) -> Result<(), openasr_core::NativeAsrError> {
    thread_local! {
        static WARMED_AT_GENERATION: std::cell::Cell<Option<u64>> =
            const { std::cell::Cell::new(None) };
    }
    let current_generation = crate::idle_activity::native_runtime_generation();
    if WARMED_AT_GENERATION.with(std::cell::Cell::get) == Some(current_generation) {
        if !abandoned.load(Ordering::Acquire) {
            crate::idle_activity::mark_native_model_warm(residency_key);
        }
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
        crate::idle_activity::mark_native_model_warm(residency_key);
    }
    Ok(())
}

/// Warms the worker thread for the daemon's default bound native model pack
/// in the background, right after `serve_with_launch_options` finishes
/// binding the listener -- so the very first real dictation session does not
/// pay the cold model-pack-load cost (observed 1.7-2.1s) before its first
/// partial. Fire-and-forget: never blocks bind/serve/health, and a failure here
/// (bad pack, no adapter, ...) is reported as a one-line diagnostic without
/// changing daemon readiness -- a real request still fails closed through the
/// normal request path.
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
/// Dedup with a concurrent real attach is a process-wide warmup lease, not a
/// user session permit: this never takes `max-native-sessions-per-model`. A
/// real WS/transcription acquire waits until the lease drops, then takes the
/// user slot and attaches. If a user session already holds a slot, warmup is
/// opportunistic and returns without attaching -- the live request's own `Warm`
/// pays the load. Whichever `Warm` actually runs first pays the cost once
/// (see the thread-local `WARMED_AT_GENERATION` gate in
/// `warm_up_native_streaming_session_once`); every later attach on that thread
/// reuses the now-warm state, until an `idle_unload` eviction bumps the
/// generation and forces a re-warm.
/// Whether any native streaming worker currently has an attached (or attaching)
/// session occupying its slot. Used by default-model rebind to fail closed
/// rather than tearing down in-flight realtime work.
pub(crate) fn native_streaming_workers_occupy_slot() -> bool {
    let Some(registry) = SHARED_NATIVE_STREAMING_WORKERS.get() else {
        return false;
    };
    let workers = registry
        .lock()
        .expect("native streaming worker registry mutex poisoned");
    workers
        .values()
        .any(|entry| entry.state.active_or_attaching.load(Ordering::Acquire) > 0)
}

static NATIVE_WARMUP_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

fn native_warmup_notify() -> &'static Notify {
    static NOTIFY: OnceLock<Notify> = OnceLock::new();
    NOTIFY.get_or_init(Notify::new)
}

pub(crate) fn native_warmup_in_flight() -> bool {
    NATIVE_WARMUP_IN_FLIGHT.load(Ordering::Acquire)
}

pub(crate) struct NativeWarmupLease {
    _private: (),
}

pub(crate) fn try_begin_native_warmup() -> Option<NativeWarmupLease> {
    NATIVE_WARMUP_IN_FLIGHT
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .ok()
        .map(|_| NativeWarmupLease { _private: () })
}

impl Drop for NativeWarmupLease {
    fn drop(&mut self) {
        NATIVE_WARMUP_IN_FLIGHT.store(false, Ordering::Release);
        native_warmup_notify().notify_waiters();
    }
}

pub(crate) async fn wait_while_native_warmup_in_flight() {
    let started = Instant::now();
    let notify = native_warmup_notify();
    while native_warmup_in_flight() && started.elapsed() < Duration::from_secs(60) {
        let _ = tokio::time::timeout(Duration::from_millis(50), notify.notified()).await;
    }
}

/// Blocking counterpart for `spawn_blocking` transcription: same 60s cap, no
/// tokio runtime. Call immediately before `acquire_native_execution`.
pub(crate) fn wait_while_native_warmup_in_flight_blocking() {
    let started = Instant::now();
    while native_warmup_in_flight() && started.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(50));
    }
}

pub(crate) fn spawn_boot_native_warmup(
    runtime: ServerRuntime,
    home: PathBuf,
) -> tokio::task::JoinHandle<()> {
    // Reactivation walks the same verify -> resolve -> reserve -> materialize
    // -> attest -> reconcile transaction as set-default. Its journal only
    // validates that the durable V2 request is still current; it never writes
    // a new generation during boot.
    tokio::task::spawn_blocking(move || {
        let Some(requested_path) = runtime.model_pack_path.requested_path() else {
            return;
        };
        let record = match openasr_core::default_selection::read_active_model_selection_v2(&home) {
            Ok(Some(record)) => record,
            Ok(None) => {
                log_default_reactivation_failed(
                    &home,
                    requested_path.as_path(),
                    "durable V2 default-selection is missing",
                );
                return;
            }
            Err(error) => {
                log_default_reactivation_failed(
                    &home,
                    requested_path.as_path(),
                    &error.to_string(),
                );
                return;
            }
        };
        let packs = match openasr_core::list_installed_packs(&home) {
            Ok(packs) => packs,
            Err(error) => {
                log_default_reactivation_failed(
                    &home,
                    requested_path.as_path(),
                    &error.to_string(),
                );
                return;
            }
        };
        let Some(pack) = packs.into_iter().find(|pack| pack.path == requested_path) else {
            log_default_reactivation_failed(
                &home,
                requested_path.as_path(),
                "requested pack is not in InstalledModelStore",
            );
            return;
        };
        let intent = match openasr_core::default_selection::execution_intent_from_v2_wire(
            &record.execution_intent,
        ) {
            Ok(intent) => intent,
            Err(error) => {
                log_default_reactivation_failed(
                    &home,
                    requested_path.as_path(),
                    &error.to_string(),
                );
                return;
            }
        };
        if let Err(error) = crate::activate_default_model_blocking(
            &runtime,
            &home,
            &pack,
            record.quant_preference,
            intent,
            crate::DefaultModelActivationMode::ReactivateDurableSelection,
        ) {
            log_default_reactivation_failed(&home, requested_path.as_path(), &error.to_string());
        }
    })
}

fn log_default_reactivation_failed(home: &Path, requested_path: &Path, reason: &str) {
    let mut reason = single_line_log_value(reason);
    for sensitive_path in [home, requested_path] {
        let rendered = sensitive_path.to_string_lossy();
        if !rendered.is_empty() {
            reason = reason.replace(rendered.as_ref(), "<redacted-path>");
        }
    }
    let reason = reason.chars().take(512).collect::<String>();
    openasr_core::stage_timing::log_event(
        "server_boot",
        format_args!("stage=default_reactivation_failed reason={reason}"),
    );
}

/// Attest a candidate pack without publishing it as the live default.
/// `candidate_pack` is loaded for warmup/probe only; live `model_pack_path`
/// is left unchanged. `None` probes the currently bound pack (boot warmup).
#[derive(Clone)]
pub(crate) struct NativeWarmupTarget {
    path: PathBuf,
    residency_key: crate::idle_activity::NativeRuntimeResidencyKey,
}

impl NativeWarmupTarget {
    pub(crate) fn attested(path: PathBuf, pack_content_id: &str) -> Self {
        Self {
            path,
            residency_key: crate::idle_activity::NativeRuntimeResidencyKey::attested(
                pack_content_id,
            ),
        }
    }
}

pub(crate) async fn probe_native_activation(
    runtime: ServerRuntime,
    candidate_pack: Option<NativeWarmupTarget>,
    preferences_home: Option<PathBuf>,
    reservation_context: Option<openasr_core::ActivationReservationContext>,
) -> Result<Option<openasr_core::NativeExecutionReceiptSnapshot>, ApiError> {
    if let Some(failpoint) = runtime.model_pack_path.activation_probe_failpoint() {
        return failpoint.map(|()| None).map_err(ApiError::BadRequest);
    }
    warm_up_native_pack(
        runtime,
        candidate_pack,
        preferences_home,
        reservation_context,
    )
    .await
    .map(Some)
    .map_err(|reason| {
        ApiError::BadRequest(format!("default model activation probe failed: {reason}"))
    })
}

/// Sync entry for the typestate attestation contract. Injection failpoints
/// complete without awaiting warmup, so current-thread tests still enter this
/// shipped function. Real warmup uses `block_in_place` on a multi-thread
/// runtime.
pub(crate) fn probe_native_activation_blocking(
    runtime: ServerRuntime,
    candidate_pack: Option<NativeWarmupTarget>,
    preferences_home: Option<PathBuf>,
    reservation_context: Option<openasr_core::ActivationReservationContext>,
) -> Result<Option<openasr_core::NativeExecutionReceiptSnapshot>, ApiError> {
    let future = probe_native_activation(
        runtime,
        candidate_pack,
        preferences_home,
        reservation_context,
    );
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread => {
            futures_util::FutureExt::now_or_never(future).ok_or_else(|| {
                ApiError::BadRequest(
                    "activation probe required async warmup on the current-thread runtime"
                        .to_string(),
                )
            })?
        }
        Ok(_) => tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future)),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| ApiError::BadRequest(error.to_string()))?
            .block_on(future),
    }
}

#[cfg(test)]
pub(in crate::realtime) async fn warm_up_default_native_streaming_worker(
    runtime: ServerRuntime,
) -> Result<(), String> {
    warm_up_native_pack(runtime, None, None, None)
        .await
        .map(|_| ())
}

async fn warm_up_native_pack(
    runtime: ServerRuntime,
    candidate_pack: Option<NativeWarmupTarget>,
    preferences_home: Option<PathBuf>,
    reservation_context: Option<openasr_core::ActivationReservationContext>,
) -> Result<openasr_core::NativeExecutionReceiptSnapshot, String> {
    let receipt = openasr_core::NativeExecutionReceiptCollector::new();
    let request_attempt_id = openasr_core::RequestAttemptId::generate()
        .map_err(|_| "could not allocate warmup request attempt identity".to_string())?;
    receipt.bind_request_attempt(request_attempt_id);
    if runtime.backend != openasr_core::BackendKind::Native {
        return Ok(receipt.snapshot());
    }
    // A boot warmup must claim the warmup lease while holding the same barrier
    // used by activation and new-session admission. Reading the active path
    // first and claiming the lease later leaves a window in which activation
    // can publish a different model and evict the old runtime before this
    // warmup starts. Candidate probes already run under the caller's activation
    // barrier; they use the exact candidate path rather than the active slot.
    let (target, _lease) = match candidate_pack {
        Some(target) => {
            if runtime.native_execution.has_active_sessions() {
                return Err("native session is already active".to_string());
            }
            let Some(lease) = try_begin_native_warmup() else {
                return Err("native warmup already in flight".to_string());
            };
            (target, lease)
        }
        None => {
            {
                let _activation_barrier = runtime
                    .begin_native_activation()
                    .map_err(|error| error.to_string())?;
                let Some(active_model) = runtime.model_pack_path.current_snapshot() else {
                    // Fresh install / no model installed yet: nothing to warm.
                    // The daemon still serves `/health`; a later activation
                    // binds and probes a verified pack transactionally.
                    return Ok(receipt.snapshot());
                };
                if runtime.native_execution.has_active_sessions() {
                    return Err("native session is already active".to_string());
                }
                let Some(lease) = try_begin_native_warmup() else {
                    return Err("native warmup already in flight".to_string());
                };
                (
                    NativeWarmupTarget {
                        path: active_model.path().to_path_buf(),
                        residency_key: active_model.residency_key().clone(),
                    },
                    lease,
                )
            }
        }
    };
    let model_pack_path = target.path;
    let preferences_home = preferences_home.or_else(|| openasr_core::openasr_home().ok());
    let inference_threads = preferences_home
        .as_deref()
        .and_then(realtime_inference_threads_preference);
    let execution_target_preference = preferences_home
        .as_deref()
        .and_then(realtime_execution_target_preference);
    let resolved_route = crate::routes::transcription::resolve_execution_route_for_target(
        execution_target_preference,
    )
    .ok()
    .flatten();
    let Some(adapter) = openasr_core::native_runtime_model_adapter_for_path(&model_pack_path)
    else {
        return Err("no native runtime adapter for selected model pack".to_string());
    };
    let model_pack = adapter
        .model_pack_ref("native-default")
        .map_err(|_| "could not open selected native default pack".to_string())?;
    let context = NativeAsrSessionContext::new("boot-warmup")
        .with_request_attempt_id(request_attempt_id)
        .with_native_execution_receipt(receipt.clone());
    let context = match reservation_context {
        Some(reservation_context) => {
            context.with_activation_reservation_context(reservation_context)
        }
        None => context,
    };
    // Same saved-preferences fallback a real WS attach applies when a session
    // does not set an explicit `execution_target`/`inference_threads`
    // override (`realtime_execution_target_preference` /
    // `realtime_inference_threads_preference` in `realtime/mod.rs`), so a
    // user who changed their default hardware target or thread count still
    // gets a worker warmed at the key their next attach will actually use.
    // `openasr_home()` resolution failing here (unreadable env, race) just
    // means "no preference found" -- same graceful fallback to defaults the
    // request-time paths use.
    let options = NativeAsrRequestOptions::new().with_inference_threads(inference_threads);
    let session_config = NativeAsrStreamingSessionConfig::new()
        .with_audio_format(RealtimeAudioFormat::pcm16_mono_16khz());
    let executor =
        NativeBackendExecutor::new(Arc::clone(runtime.native_execution.execution_services()));
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
        Err(error) => {
            receipt.record_terminal(openasr_core::RequestExecutionTerminal::Failed);
            openasr_core::stage_timing::log_event(
                "realtime_warmup",
                format_args!(
                    "stage=session_start_failed model_pack=selected hardware_target={} failure=session_start",
                    hardware_target,
                ),
            );
            return Err(error.to_string());
        }
    };
    let key = NativeStreamingWorkerKey::with_route_and_residency(
        model_pack.root.clone(),
        target.residency_key,
        hardware_target,
        resolved_route.as_ref(),
        inference_threads,
    );
    let diagnostic_key = key.clone();
    if let Err(error) = attach_and_run_boot_warmup(key, session, None).await {
        receipt.record_terminal(openasr_core::RequestExecutionTerminal::Failed);
        openasr_core::stage_timing::log_event(
            "realtime_warmup",
            format_args!(
                "stage=failed model_pack=resident hardware_target={} execution_route_key={} failure=worker_attach_or_warmup",
                diagnostic_key.hardware_target, diagnostic_key.execution_route_key,
            ),
        );
        return Err(error);
    }
    receipt.record_terminal(openasr_core::RequestExecutionTerminal::Succeeded);
    Ok(receipt.snapshot())
}

pub(crate) fn single_line_log_value(value: &str) -> String {
    value.replace('\r', "\\r").replace('\n', "\\n")
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
    model_session_permit: Option<ModelSessionPermit>,
) -> Result<(), String> {
    let mut worker =
        NativeStreamingDecodeWorker::attach_admitted(key, session, model_session_permit)
            .await
            .map_err(|error| format!("worker attach failed: {error}"))?;
    let envelope = NativeStreamingCommandEnvelope {
        kind: NativeStreamingCommandKind::Warm,
        command: NativeStreamingCommand::Warm,
    };
    let result = match worker.commands.send(envelope).await {
        Ok(()) => {
            // Wait for the Warm outcome so this session -- and the worker-key
            // acquisition it holds -- does not release before warm-up actually
            // finishes; otherwise a concurrent real attach would not meaningfully
            // "queue behind" this one.
            match worker.outcomes.recv().await {
                Some(NativeStreamingOutcome::Events {
                    kind: NativeStreamingCommandKind::Warm,
                    ..
                }) => Ok(()),
                Some(NativeStreamingOutcome::Error {
                    kind: NativeStreamingCommandKind::Warm,
                    message,
                }) => Err(message),
                Some(outcome) => Err(format!(
                    "worker returned unexpected {:?} outcome for boot warm-up",
                    outcome.kind()
                )),
                None => Err("worker stopped before returning the boot warm-up outcome".to_string()),
            }
        }
        Err(_) => Err("worker stopped before accepting the boot warm-up command".to_string()),
    };
    // No Finish/Cancel is sent, so the worker thread cancels this session on
    // its behalf (see `run_native_streaming_session_on_worker`) and loops
    // straight back to accept the next real Attach -- the thread-local warm
    // state this call just primed stays resident for it.
    worker.join();
    result
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
            model_pack_path: runtime.model_pack_path.current(),
        }
    }
}

pub(crate) struct RealtimeBackendWorkItem {
    pub(crate) session_key: String,
    pub(crate) job: BackendJob,
    pub(crate) result_sender: mpsc::Sender<BackendResult>,
    pub(crate) cancelled: Arc<AtomicBool>,
    /// Explicit cancel/pause/resume context for this job's decode -- never a
    /// thread-local. Required (not `Option`): a work item with nothing to
    /// cancel still carries a real, detached context. See
    /// [`openasr_core::RequestExecutionContext`].
    pub(crate) execution_context: Arc<openasr_core::RequestExecutionContext>,
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
        let result = run_realtime_backend_job(runtime, item.job, item.execution_context).await;
        if !cancelled.load(Ordering::Relaxed) {
            let _ = result_sender.send(result).await;
        }
        let _ = worker_sender
            .send(RealtimeBackendWorkerMessage::Completed { session_key })
            .await;
    });
}

async fn run_realtime_backend_job(
    runtime: ServerRuntime,
    job: BackendJob,
    execution_context: Arc<openasr_core::RequestExecutionContext>,
) -> BackendResult {
    // Echo the requested language into the final realtime result; the core
    // Transcription carries no detected language, so the request value is the
    // only source. Capture it before job.language is moved into the builder.
    let response_language = job.language.clone();
    let request = openasr_core::TranscriptionRequest::new(job.temp_wav.path(), job.model_id)
        .with_source(openasr_core::RequestSource::ServerRealtime)
        // Not a normalization-pipeline guess: `write_temp_utterance_wav`
        // (see `realtime/mod.rs`) always writes PCM16 mono 16 kHz WAV -- this
        // is the realtime session's real fixed capture format/container,
        // unlike an uploaded file whose source format is unknown until
        // probed/decoded.
        .with_source_audio_format(Some(16_000), Some(1))
        .with_source_container(Some("wav".to_string()))
        .with_language(job.language)
        .with_task(job.task)
        .with_prompt(job.prompt)
        .with_phrase_bias(job.phrase_bias)
        .with_inference_threads(job.inference_threads)
        .with_execution_target(job.execution_target)
        .with_word_timestamps(job.word_timestamps)
        .with_display_file_name(Some(job.display_name));
    match transcribe_with_runtime(runtime, request, execution_context).await {
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
        Err(error) => BackendResult::Error(error),
    }
}
