use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use thiserror::Error;

use super::clause::ClauseId;
use super::session::{TranslationOutput, TranslationRequest, TranslationTimings};

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationWorkerOutput {
    pub text: String,
    pub timings: TranslationTimings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranslationQueueSubmit {
    pub replaced_pending: bool,
}

#[derive(Debug, Error)]
pub enum TranslationQueueError {
    #[error("translation queue is closed")]
    Closed,
    #[error("translation queue backpressure: {reason}")]
    Backpressure { reason: String },
    #[error("translation worker failed: {reason}")]
    Worker { reason: String },
}

pub struct LatestOnlyTranslationQueue {
    shared: Arc<QueueShared>,
    results: mpsc::Receiver<Result<TranslationOutput, TranslationQueueError>>,
    worker: Option<thread::JoinHandle<()>>,
}

struct QueueShared {
    state: Mutex<QueueState>,
    changed: Condvar,
}

#[derive(Debug)]
struct QueueState {
    pending_final: VecDeque<QueuedTranslationRequest>,
    pending_provisional: Option<QueuedTranslationRequest>,
    latest_versions: BTreeMap<ClauseId, u64>,
    retired_clause_ids: BTreeSet<ClauseId>,
    next_translation_version: u64,
    running: bool,
    running_clause_id: Option<ClauseId>,
    unread_results: usize,
    shutdown: bool,
    worker_failed: Option<String>,
    /// True once the worker closure is initialized and processing requests.
    /// `spawn` workers are ready at birth; `spawn_thread_local` workers become
    /// ready when their thread-local initialization completes.
    worker_ready: bool,
}

#[derive(Debug)]
struct QueuedTranslationRequest {
    request: TranslationRequest,
    enqueued_at: Instant,
}

impl LatestOnlyTranslationQueue {
    pub fn spawn(
        worker: impl FnMut(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
        + Send
        + 'static,
    ) -> Self {
        let shared = Arc::new(QueueShared {
            state: Mutex::new(QueueState {
                pending_final: VecDeque::new(),
                pending_provisional: None,
                latest_versions: BTreeMap::new(),
                retired_clause_ids: BTreeSet::new(),
                next_translation_version: 1,
                running: false,
                running_clause_id: None,
                unread_results: 0,
                shutdown: false,
                worker_failed: None,
                worker_ready: true,
            }),
            changed: Condvar::new(),
        });
        let (sender, results) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || run_worker(worker_shared, sender, worker));
        Self {
            shared,
            results,
            worker: Some(worker),
        }
    }

    /// Spawns a worker whose (potentially slow) initialization runs on the
    /// worker thread, OFF the caller's critical path. The queue is usable
    /// immediately: requests enqueued before initialization completes are
    /// buffered and processed once the worker is ready. If initialization
    /// fails (or panics), the failure is delivered through the results channel
    /// as a `TranslationQueueError::Worker` on the next `try_recv`.
    pub fn spawn_thread_local<W>(
        init_worker: impl FnOnce() -> Result<W, TranslationQueueError> + Send + 'static,
    ) -> Self
    where
        W: FnMut(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
            + 'static,
    {
        let shared = Arc::new(QueueShared {
            state: Mutex::new(QueueState {
                pending_final: VecDeque::new(),
                pending_provisional: None,
                latest_versions: BTreeMap::new(),
                retired_clause_ids: BTreeSet::new(),
                next_translation_version: 1,
                running: false,
                running_clause_id: None,
                unread_results: 0,
                shutdown: false,
                worker_failed: None,
                worker_ready: false,
            }),
            changed: Condvar::new(),
        });
        let (sender, results) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || {
            let init_result =
                catch_unwind(AssertUnwindSafe(init_worker)).unwrap_or_else(|payload| {
                    Err(TranslationQueueError::Worker {
                        reason: panic_payload_to_reason(payload),
                    })
                });
            match init_result {
                Ok(worker) => {
                    mark_worker_ready(&worker_shared);
                    run_worker(worker_shared, sender, worker);
                }
                Err(error) => {
                    let reason = match error {
                        TranslationQueueError::Worker { reason } => reason,
                        other => other.to_string(),
                    };
                    mark_worker_failed(&worker_shared, reason.clone());
                    let result = mark_result_ready(
                        &worker_shared,
                        Err(TranslationQueueError::Worker { reason }),
                    );
                    if sender.send(result).is_err() {
                        mark_result_discarded(&worker_shared);
                    }
                }
            }
        });
        Self {
            shared,
            results,
            worker: Some(worker),
        }
    }

    /// True once the worker finished initialization and is processing
    /// requests. Stays false forever if initialization failed; the failure
    /// itself surfaces through `try_recv`.
    pub fn worker_ready(&self) -> bool {
        self.shared
            .state
            .lock()
            .map(|state| state.worker_ready)
            .unwrap_or(false)
    }

    pub fn enqueue(
        &self,
        request: TranslationRequest,
    ) -> Result<TranslationQueueSubmit, TranslationQueueError> {
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| TranslationQueueError::Closed)?;
        if let Some(reason) = state.worker_failed.clone() {
            return Err(TranslationQueueError::Worker { reason });
        }
        if state.shutdown {
            return Err(TranslationQueueError::Closed);
        }
        let mut replaced_pending = false;
        let queued = QueuedTranslationRequest {
            request,
            enqueued_at: Instant::now(),
        };
        let clause_id = queued.request.clause_id;
        let source_version = queued.request.source_version;
        if queued.request.finalized {
            if state.pending_final.len() >= MAX_PENDING_FINAL_TRANSLATIONS {
                return Err(TranslationQueueError::Backpressure {
                    reason: format!(
                        "pending final translation cap ({MAX_PENDING_FINAL_TRANSLATIONS}) reached; rejecting the newest final instead of buffering unbounded work"
                    ),
                });
            }
            if let Some(pending) = state.pending_provisional.as_ref()
                && pending.request.clause_id == clause_id
                && pending.request.source_version <= source_version
            {
                state.pending_provisional = None;
                replaced_pending = true;
            }
            let latest = state
                .latest_versions
                .entry(clause_id)
                .or_insert(source_version);
            *latest = (*latest).max(source_version);
            state.pending_final.push_back(queued);
        } else {
            let latest = state
                .latest_versions
                .entry(clause_id)
                .or_insert(source_version);
            *latest = (*latest).max(source_version);
            replaced_pending = state.pending_provisional.replace(queued).is_some();
        }
        self.shared.changed.notify_one();
        Ok(TranslationQueueSubmit { replaced_pending })
    }

    pub fn try_recv(&self) -> Result<Option<TranslationOutput>, TranslationQueueError> {
        match self.results.try_recv() {
            Ok(Ok(output)) => {
                mark_result_consumed(&self.shared, Some(&output));
                Ok(Some(output))
            }
            Ok(Err(error)) => {
                mark_result_consumed(&self.shared, None);
                Err(error)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(TranslationQueueError::Closed),
        }
    }

    pub fn retire_clause_ids(
        &self,
        clause_ids: impl IntoIterator<Item = ClauseId>,
    ) -> Result<(), TranslationQueueError> {
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| TranslationQueueError::Closed)?;
        for clause_id in clause_ids {
            state.latest_versions.remove(&clause_id);
            state
                .pending_final
                .retain(|queued| queued.request.clause_id != clause_id);
            if state
                .pending_provisional
                .as_ref()
                .is_some_and(|queued| queued.request.clause_id == clause_id)
            {
                state.pending_provisional = None;
            }
            if state.running_clause_id == Some(clause_id) || state.unread_results > 0 {
                state.retired_clause_ids.insert(clause_id);
            } else {
                state.retired_clause_ids.remove(&clause_id);
            }
        }
        prune_retired_clause_ids(&mut state);
        self.shared.changed.notify_one();
        Ok(())
    }

    pub fn has_pending_or_running(&self) -> bool {
        self.shared
            .state
            .lock()
            .map(|state| {
                state.running
                    || state.unread_results > 0
                    || !state.pending_final.is_empty()
                    || state.pending_provisional.is_some()
            })
            .unwrap_or(false)
    }
}

impl Drop for LatestOnlyTranslationQueue {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.shutdown = true;
            self.shared.changed.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = thread::Builder::new()
                .name("openasr-translation-reaper".to_string())
                .spawn(move || {
                    let _ = worker.join();
                });
        }
    }
}

fn run_worker(
    shared: Arc<QueueShared>,
    sender: mpsc::Sender<Result<TranslationOutput, TranslationQueueError>>,
    mut worker: impl FnMut(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>,
) {
    loop {
        let Some(queued) = take_next_request(&shared) else {
            return;
        };
        let started_at = Instant::now();
        let request = queued.request.clone();
        let result = match catch_unwind(AssertUnwindSafe(|| worker(request.clone()))) {
            Ok(result) => result.map(|worker_output| {
                let queue_wait = started_at.saturating_duration_since(queued.enqueued_at);
                complete_output(&shared, request, worker_output, queue_wait)
            }),
            Err(payload) => {
                let reason = panic_payload_to_reason(payload);
                mark_worker_failed(&shared, reason.clone());
                Err(TranslationQueueError::Worker { reason })
            }
        };
        let result = mark_result_ready(&shared, result);
        let send_failed = sender.send(result).is_err();
        if send_failed {
            mark_result_discarded(&shared);
        }
        mark_worker_idle(&shared);
        if send_failed {
            return;
        }
        if let Ok(state) = shared.state.lock()
            && state.worker_failed.is_some()
        {
            return;
        }
    }
}

fn take_next_request(shared: &QueueShared) -> Option<QueuedTranslationRequest> {
    let mut state = shared.state.lock().ok()?;
    loop {
        if state.shutdown {
            return None;
        }
        if let Some(queued) = state.pending_final.pop_front() {
            state.running = true;
            state.running_clause_id = Some(queued.request.clause_id);
            return Some(queued);
        }
        if let Some(queued) = state.pending_provisional.take() {
            state.running = true;
            state.running_clause_id = Some(queued.request.clause_id);
            return Some(queued);
        }
        state = shared.changed.wait(state).ok()?;
    }
}

fn mark_worker_ready(shared: &QueueShared) {
    if let Ok(mut state) = shared.state.lock() {
        state.worker_ready = true;
        shared.changed.notify_all();
    }
}

fn mark_worker_failed(shared: &QueueShared, reason: String) {
    if let Ok(mut state) = shared.state.lock() {
        state.worker_failed = Some(reason);
        state.running = false;
        state.running_clause_id = None;
        state.shutdown = true;
        shared.changed.notify_all();
    }
}

fn mark_worker_idle(shared: &QueueShared) {
    if let Ok(mut state) = shared.state.lock() {
        state.running = false;
        state.running_clause_id = None;
    }
}

fn mark_result_ready(
    shared: &QueueShared,
    result: Result<TranslationOutput, TranslationQueueError>,
) -> Result<TranslationOutput, TranslationQueueError> {
    match shared.state.lock() {
        Ok(mut state) => {
            if state.unread_results >= MAX_UNREAD_TRANSLATION_RESULTS {
                let reason = format!(
                    "unread translation result cap ({MAX_UNREAD_TRANSLATION_RESULTS}) reached; closing the translation queue instead of buffering unbounded results"
                );
                state.worker_failed = Some(reason.clone());
                state.shutdown = true;
                state.unread_results = state.unread_results.saturating_add(1);
                shared.changed.notify_all();
                return Err(TranslationQueueError::Backpressure { reason });
            }
            state.unread_results = state.unread_results.saturating_add(1);
        }
        Err(_) => return Err(TranslationQueueError::Closed),
    }
    result
}

fn mark_result_consumed(shared: &QueueShared, output: Option<&TranslationOutput>) {
    if let Ok(mut state) = shared.state.lock() {
        state.unread_results = state.unread_results.saturating_sub(1);
        if let Some(output) = output {
            state.retired_clause_ids.remove(&output.clause_id);
            if output.finalized
                && state
                    .latest_versions
                    .get(&output.clause_id)
                    .is_some_and(|latest| *latest <= output.source_version)
            {
                state.latest_versions.remove(&output.clause_id);
            }
        }
        prune_retired_clause_ids(&mut state);
        shared.changed.notify_all();
    }
}

fn mark_result_discarded(shared: &QueueShared) {
    if let Ok(mut state) = shared.state.lock() {
        state.unread_results = state.unread_results.saturating_sub(1);
        shared.changed.notify_all();
    }
}

fn prune_retired_clause_ids(state: &mut QueueState) {
    while state.retired_clause_ids.len() > MAX_RETIRED_CLAUSE_IDS {
        let Some(oldest) = state.retired_clause_ids.iter().next().copied() else {
            return;
        };
        state.retired_clause_ids.remove(&oldest);
    }
}

fn panic_payload_to_reason(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "translation worker panicked".to_string()
}

fn complete_output(
    shared: &QueueShared,
    request: TranslationRequest,
    worker_output: TranslationWorkerOutput,
    queue_wait: std::time::Duration,
) -> TranslationOutput {
    let (translation_version, dropped_stale) = match shared.state.lock() {
        Ok(mut state) => {
            let latest = state
                .latest_versions
                .get(&request.clause_id)
                .copied()
                .unwrap_or(request.source_version);
            let retired = state.retired_clause_ids.contains(&request.clause_id);
            let translation_version = state.next_translation_version;
            state.next_translation_version = state.next_translation_version.saturating_add(1);
            (
                translation_version,
                retired || latest > request.source_version,
            )
        }
        Err(_) => (0, true),
    };
    let mut timings = worker_output.timings;
    timings.queue_wait = queue_wait;
    TranslationOutput {
        clause_id: request.clause_id,
        replaces_clause_id: request.replaces_clause_id,
        source_version: request.source_version,
        translation_version,
        text: worker_output.text,
        source_text: request.source_text,
        finalized: request.finalized,
        revised: request.revised,
        target_lang: request.target_lang,
        dropped_stale,
        timings,
    }
}

const MAX_PENDING_FINAL_TRANSLATIONS: usize = 64;
const MAX_UNREAD_TRANSLATION_RESULTS: usize = 64;
const MAX_RETIRED_CLAUSE_IDS: usize = 128;

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::translation::{TargetLang, TranslationTimings};

    fn request(version: u64, text: &str, finalized: bool) -> TranslationRequest {
        request_with_clause(ClauseId::new(1), version, text, finalized, None)
    }

    fn request_with_clause(
        clause_id: ClauseId,
        version: u64,
        text: &str,
        finalized: bool,
        replaces_clause_id: Option<ClauseId>,
    ) -> TranslationRequest {
        TranslationRequest {
            clause_id,
            replaces_clause_id,
            source_version: version,
            source_text: text.to_string(),
            finalized,
            revised: false,
            target_lang: TargetLang::En,
            finalized_context: Vec::new(),
        }
    }

    #[test]
    fn provisional_pending_is_latest_only() {
        let (seen_sender, seen_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            seen_sender
                .send((request.clause_id, request.source_version))
                .expect("seen send");
            if request.clause_id == ClauseId::new(99) {
                release_receiver.recv().expect("release blocker");
            }
            Ok(TranslationWorkerOutput {
                text: request.source_text,
                timings: TranslationTimings::default(),
            })
        });

        queue
            .enqueue(request_with_clause(
                ClauseId::new(99),
                1,
                "阻塞",
                true,
                None,
            ))
            .unwrap();
        assert_eq!(
            seen_receiver.recv().expect("blocker started"),
            (ClauseId::new(99), 1)
        );
        assert!(
            !queue
                .enqueue(request(1, "旧", false))
                .unwrap()
                .replaced_pending
        );
        assert!(
            queue
                .enqueue(request(2, "新", false))
                .unwrap()
                .replaced_pending
        );
        release_sender.send(()).expect("release blocker");

        let blocker = wait_for_output(&queue);
        let output = wait_for_output(&queue);
        assert_eq!(blocker.clause_id, ClauseId::new(99));
        assert_eq!(
            seen_receiver.try_recv().expect("one request"),
            (ClauseId::new(1), 2)
        );
        assert!(seen_receiver.try_recv().is_err());
        assert_eq!(output.source_version, 2);
        assert!(!output.dropped_stale);
    }

    #[test]
    fn running_provisional_completes_as_stale_when_superseded() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            started_sender
                .send(request.source_version)
                .expect("started send");
            if request.source_version == 1 {
                release_receiver.recv().expect("release first");
            }
            Ok(TranslationWorkerOutput {
                text: request.source_text,
                timings: TranslationTimings::default(),
            })
        });

        queue.enqueue(request(1, "旧", false)).unwrap();
        assert_eq!(started_receiver.recv().expect("started"), 1);
        queue.enqueue(request(2, "新", false)).unwrap();
        release_sender.send(()).expect("release");

        let stale = wait_for_output(&queue);
        let fresh = wait_for_output(&queue);
        assert_eq!(stale.source_version, 1);
        assert!(stale.dropped_stale);
        assert_eq!(fresh.source_version, 2);
        assert!(!fresh.dropped_stale);
    }

    #[test]
    fn running_final_completes_as_stale_when_clause_is_retired() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            started_sender
                .send(request.clause_id)
                .expect("started send");
            if request.clause_id == ClauseId::new(1) {
                release_receiver.recv().expect("release retired final");
            }
            Ok(TranslationWorkerOutput {
                text: request.source_text,
                timings: TranslationTimings::default(),
            })
        });

        queue
            .enqueue(request_with_clause(
                ClauseId::new(1),
                1,
                "旧句子。",
                true,
                None,
            ))
            .unwrap();
        assert_eq!(started_receiver.recv().expect("started"), ClauseId::new(1));
        queue.retire_clause_ids([ClauseId::new(1)]).unwrap();
        queue
            .enqueue(request_with_clause(
                ClauseId::new(2),
                1,
                "新句子。",
                true,
                Some(ClauseId::new(1)),
            ))
            .unwrap();
        release_sender.send(()).expect("release");

        let retired = wait_for_output(&queue);
        let replacement = wait_for_output(&queue);
        assert_eq!(retired.clause_id, ClauseId::new(1));
        assert!(retired.dropped_stale);
        assert_eq!(replacement.clause_id, ClauseId::new(2));
        assert_eq!(replacement.replaces_clause_id, Some(ClauseId::new(1)));
        assert!(!replacement.dropped_stale);
    }

    #[test]
    fn unread_result_keeps_queue_non_idle() {
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            Ok(TranslationWorkerOutput {
                text: request.source_text,
                timings: TranslationTimings::default(),
            })
        });

        queue.enqueue(request(1, "完成但还没读", true)).unwrap();
        wait_for_unread_result(&queue);

        assert!(queue.has_pending_or_running());
        let output = wait_for_output(&queue);
        assert_eq!(output.source_text, "完成但还没读");
        assert!(!queue.has_pending_or_running());
    }

    #[test]
    fn final_job_has_priority_over_pending_provisional() {
        // Gate the worker on a blocker job first: priority is only defined for
        // jobs that are PENDING together, so both must be enqueued before the
        // worker is free to pick (otherwise the worker can race the second
        // enqueue and legitimately process the provisional first).
        let (release_sender, release_receiver) = mpsc::channel::<()>();
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            if request.source_version == 0 {
                release_receiver.recv().expect("release blocker");
            }
            Ok(TranslationWorkerOutput {
                text: request.source_text,
                timings: TranslationTimings::default(),
            })
        });

        queue
            .enqueue(request_with_clause(
                ClauseId::new(9),
                0,
                "占位。",
                true,
                None,
            ))
            .unwrap();
        queue.enqueue(request(1, "临时", false)).unwrap();
        queue.enqueue(request(2, "最终", true)).unwrap();
        release_sender.send(()).expect("release");

        let blocker = wait_for_output(&queue);
        assert_eq!(blocker.source_version, 0);
        let output = wait_for_output(&queue);
        assert!(output.finalized);
        assert_eq!(output.source_version, 2);
    }

    #[test]
    fn enqueue_after_worker_panic_returns_worker_error() {
        let (started_sender, started_receiver) = mpsc::channel();
        let queue = LatestOnlyTranslationQueue::spawn(move |request| {
            started_sender
                .send(request.source_version)
                .expect("started send");
            panic!("simulated translation worker panic");
        });

        queue.enqueue(request(1, "会崩溃", false)).unwrap();
        assert_eq!(started_receiver.recv().expect("started"), 1);
        let error = wait_for_error(&queue);
        assert!(matches!(
            error,
            TranslationQueueError::Worker { reason } if reason.contains("simulated translation worker panic")
        ));

        let error = queue
            .enqueue(request(2, "不会进入队列", false))
            .expect_err("dead worker must reject enqueue");
        assert!(matches!(
            error,
            TranslationQueueError::Worker { reason } if reason.contains("simulated translation worker panic")
        ));
    }

    #[test]
    fn thread_local_spawn_does_not_block_on_slow_init_and_buffers_requests() {
        let (release_init_sender, release_init_receiver) = mpsc::channel::<()>();
        let spawn_started = Instant::now();
        let queue = LatestOnlyTranslationQueue::spawn_thread_local(move || {
            release_init_receiver
                .recv()
                .map_err(|_| TranslationQueueError::Worker {
                    reason: "init release channel closed".to_string(),
                })?;
            Ok(move |request: TranslationRequest| {
                Ok(TranslationWorkerOutput {
                    text: request.source_text,
                    timings: TranslationTimings::default(),
                })
            })
        });
        // Spawn must return immediately even though init is still blocked.
        assert!(spawn_started.elapsed() < Duration::from_millis(500));
        assert!(!queue.worker_ready());

        // Requests enqueued before init completes are buffered, not rejected.
        queue.enqueue(request(1, "排队等待加载", true)).unwrap();

        release_init_sender.send(()).expect("release init");
        let output = wait_for_output(&queue);
        assert_eq!(output.source_text, "排队等待加载");
        assert!(queue.worker_ready());
    }

    #[test]
    fn thread_local_spawn_init_failure_surfaces_through_try_recv() {
        let queue = LatestOnlyTranslationQueue::spawn_thread_local::<
            fn(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>,
        >(|| {
            Err(TranslationQueueError::Worker {
                reason: "模型加载失败".to_string(),
            })
        });

        let error = wait_for_error(&queue);
        assert!(matches!(
            error,
            TranslationQueueError::Worker { reason } if reason.contains("模型加载失败")
        ));
        assert!(!queue.worker_ready());

        let error = queue
            .enqueue(request(1, "不会进入队列", false))
            .expect_err("failed init must reject enqueue");
        assert!(matches!(
            error,
            TranslationQueueError::Worker { reason } if reason.contains("模型加载失败")
        ));
    }

    #[test]
    fn thread_local_spawn_init_panic_surfaces_through_try_recv() {
        let queue = LatestOnlyTranslationQueue::spawn_thread_local::<
            fn(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>,
        >(|| panic!("simulated init panic"));

        let error = wait_for_error(&queue);
        assert!(matches!(
            error,
            TranslationQueueError::Worker { reason } if reason.contains("simulated init panic")
        ));
        assert!(!queue.worker_ready());
    }

    fn wait_for_output(queue: &LatestOnlyTranslationQueue) -> TranslationOutput {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(output) = queue.try_recv().expect("queue recv") {
                return output;
            }
            assert!(Instant::now() < deadline, "timed out waiting for output");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_for_error(queue: &LatestOnlyTranslationQueue) -> TranslationQueueError {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match queue.try_recv() {
                Ok(None) => {}
                Ok(Some(output)) => panic!("expected queue error, got output {output:?}"),
                Err(error) => return error,
            }
            assert!(Instant::now() < deadline, "timed out waiting for error");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_for_unread_result(queue: &LatestOnlyTranslationQueue) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if queue
                .shared
                .state
                .lock()
                .expect("queue state")
                .unread_results
                > 0
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for unread result"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
