//! In-session pause / resume / cancel control for a running native file
//! transcription.
//!
//! Mirrors the pull-job control model (an `Arc` of shared control flags held by
//! both the request handler and the worker), but the signal reaches the deep
//! long-form decode loop through a thread-local install guard -- the same trick
//! [`super::native_transcribe::native_transcription_progress`] uses to avoid
//! threading a handle through the whole executor API surface. The native decode
//! runs synchronously on one thread (the server's `spawn_blocking` worker or the
//! CLI's calling thread), so a thread-local is enough for the slice loop to find
//! its control.
//!
//! ggml's CPU abort_callback may run on a worker-pool thread that does **not**
//! share that TLS. Cancel therefore also dual-writes a heap
//! [`Arc`]`<`[`AtomicBool`]`>` per job. Install publishes that flag on the owner
//! thread; each ggml graph compute clones it and binds `Arc::as_ptr` only for the
//! FFI call. The trampoline wait-free-loads that pointer (never a process-wide
//! slot, never TLS). Parallel server jobs therefore hold distinct data pointers,
//! so canceling job B cannot abort job A's graph.
//!
//! Scope is deliberately in-session: the control lives only for one in-flight
//! transcription. Cross-request or cross-restart resume (which would need
//! persisted partial decode state) is out of scope.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::ggml_runtime::{arm_thread_job_cancel_flag, disarm_thread_job_cancel_flag_if_current};

/// Outcome of a slice-boundary control check inside the long-form decode loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceBoundaryControl {
    /// Keep decoding the next slice.
    Continue,
    /// A cancel was requested; stop decoding and unwind cleanly.
    Canceled,
}

#[derive(Default)]
struct ControlState {
    cancel: bool,
    pause: bool,
}

/// Shared pause / cancel control for one in-flight native transcription.
///
/// The server registers one of these per in-flight file transcription (keyed by
/// a client-supplied job id) so its pause/resume/cancel HTTP handlers can flip
/// the flags while the blocking decode runs on a `spawn_blocking` worker. The
/// worker reads the same handle at each long-form slice boundary via
/// [`current_transcription_control`]. Cancel wins over pause.
pub struct TranscriptionControl {
    state: Mutex<ControlState>,
    // Signaled on resume or cancel so a worker blocked at a paused slice
    // boundary wakes promptly instead of busy-waiting.
    resumed_or_canceled: Condvar,
    /// Wait-free cancel bit dual-written by [`Self::request_cancel`]. Shared with
    /// the ggml abort_callback trampoline (which cannot use thread-locals) as the
    /// compute-scoped callback `data` pointer published by
    /// [`install_active_transcription_control`]. Pause never touches this bit.
    cancel_flag: Arc<AtomicBool>,
}

impl TranscriptionControl {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ControlState::default()),
            resumed_or_canceled: Condvar::new(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation at the next cooperative checkpoint (slice boundary,
    /// token step, native CPU node, or segmented GPU graph view). Idempotent.
    /// Wakes a paused worker so it observes the cancel instead of staying blocked.
    pub fn request_cancel(&self) {
        // Atomic first so a concurrent ggml abort_callback poll observes cancel
        // even if it races the mutex write below.
        self.cancel_flag.store(true, Ordering::SeqCst);
        let mut state = self.lock();
        state.cancel = true;
        self.resumed_or_canceled.notify_all();
    }

    /// Request a pause at the next slice boundary. Idempotent, and a no-op once
    /// canceled (cancel wins and must not be masked by a late pause). Pause does
    /// **not** set the atomic observed by the ggml abort_callback.
    pub fn request_pause(&self) {
        let mut state = self.lock();
        if !state.cancel {
            state.pause = true;
        }
    }

    /// Clear a pending pause and wake a worker blocked at a slice boundary.
    pub fn request_resume(&self) {
        let mut state = self.lock();
        state.pause = false;
        self.resumed_or_canceled.notify_all();
    }

    /// Whether a cancel has been requested.
    pub fn is_canceled(&self) -> bool {
        // Atomic is the source of truth shared with the ggml trampoline; the
        // mutex bit is kept in lockstep by request_cancel.
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Whether a pause is pending (and not superseded by a cancel).
    pub fn is_paused(&self) -> bool {
        let state = self.lock();
        state.pause && !state.cancel && !self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Heap cancel flag for compute-scoped per-job abort_callback data.
    pub(crate) fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    /// Called by the decode loop at each slice boundary. Returns immediately with
    /// `Canceled` when a cancel is pending; otherwise blocks while paused until a
    /// resume or cancel arrives, then returns `Continue` (or `Canceled` if the
    /// wait ended in a cancel).
    ///
    /// Holds the worker thread while paused. That is acceptable for the
    /// single-file desktop scenario this targets: a paused transcription keeps
    /// its `spawn_blocking` worker and its open HTTP request until it is resumed
    /// or canceled. Releasing and re-entering the decode would require persisting
    /// partial decode state, which is the out-of-scope cross-request resume.
    pub fn wait_at_slice_boundary(&self) -> SliceBoundaryControl {
        let mut state = self.lock();
        loop {
            if state.cancel || self.cancel_flag.load(Ordering::SeqCst) {
                return SliceBoundaryControl::Canceled;
            }
            if !state.pause {
                return SliceBoundaryControl::Continue;
            }
            state = self
                .resumed_or_canceled
                .wait(state)
                .expect("transcription control mutex poisoned");
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ControlState> {
        self.state
            .lock()
            .expect("transcription control mutex poisoned")
    }
}

impl Default for TranscriptionControl {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TranscriptionControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (cancel, pause) = match self.state.lock() {
            Ok(state) => (
                state.cancel || self.cancel_flag.load(Ordering::SeqCst),
                state.pause,
            ),
            Err(_) => (self.cancel_flag.load(Ordering::SeqCst), false),
        };
        f.debug_struct("TranscriptionControl")
            .field("cancel", &cancel)
            .field("pause", &pause)
            .finish()
    }
}

thread_local! {
    // The control bound to the run currently executing on *this* thread, set by
    // `install_active_transcription_control` and read by the long-form decode
    // loop. Native transcription runs synchronously on a single thread, so this
    // is enough to attribute the slice-boundary checks to the right control
    // without threading a handle through the executor API.
    static CURRENT_TRANSCRIPTION_CONTROL: RefCell<Option<Arc<TranscriptionControl>>> =
        const { RefCell::new(None) };
}

/// RAII guard that binds a [`TranscriptionControl`] to the current thread for the
/// duration of one native transcription and restores the previous binding on
/// drop (normal return, early `?`, or panic), so a control never leaks into an
/// unrelated later run on the same pooled worker thread. Graph compute reads
/// the published job flag and owns the shorter backend callback lifetime.
#[must_use = "the control binding is cleared when this guard is dropped"]
pub struct ActiveTranscriptionControlGuard {
    previous: Option<Arc<TranscriptionControl>>,
    previous_job_cancel: Option<Arc<AtomicBool>>,
    published_flag: Arc<AtomicBool>,
}

impl Drop for ActiveTranscriptionControlGuard {
    fn drop(&mut self) {
        CURRENT_TRANSCRIPTION_CONTROL.with(|cell| {
            *cell.borrow_mut() = self.previous.take();
        });
        // Restore previous job cancel (nested) or clear. Graph compute owns a
        // scoped Arc clone and the compute API retains no callback data, so
        // cached runtimes never retain this guard's raw pointer.
        let _ = disarm_thread_job_cancel_flag_if_current(
            &self.published_flag,
            self.previous_job_cancel.take(),
        );
    }
}

/// Bind `control` to the current thread so the in-flight native transcription's
/// long-form slice loop observes pause/cancel requests. Returns a guard that
/// restores the previous binding on drop. Install this at the top of the
/// synchronous decode (e.g. inside the server's `spawn_blocking` closure).
///
/// Also publishes `control`'s cancel atomic for compute-scoped callback binding,
/// so ggml's abort_callback (which may run off this thread) observes the same
/// cancel bit via a wait-free pointer load. Pause is never written to that
/// atomic -- the abort trampoline only recognizes cancel.
pub fn install_active_transcription_control(
    control: Arc<TranscriptionControl>,
) -> ActiveTranscriptionControlGuard {
    let published_flag = control.cancel_flag();
    let previous_job_cancel = arm_thread_job_cancel_flag(Some(Arc::clone(&published_flag)));
    let previous = CURRENT_TRANSCRIPTION_CONTROL.with(|cell| cell.replace(Some(control)));
    ActiveTranscriptionControlGuard {
        previous,
        previous_job_cancel,
        published_flag,
    }
}

/// The control bound to the current thread, if any. Read by the long-form decode
/// loop at each slice boundary.
pub(crate) fn current_transcription_control() -> Option<Arc<TranscriptionControl>> {
    CURRENT_TRANSCRIPTION_CONTROL.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{
        cancel_flag_requested_from_data, thread_job_cancel_flag_data, thread_job_cancel_requested,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn cancel_before_boundary_returns_canceled() {
        let control = TranscriptionControl::new();
        control.request_cancel();
        assert!(control.is_canceled());
        assert_eq!(
            control.wait_at_slice_boundary(),
            SliceBoundaryControl::Canceled
        );
    }

    #[test]
    fn no_control_boundary_continues() {
        let control = TranscriptionControl::new();
        assert_eq!(
            control.wait_at_slice_boundary(),
            SliceBoundaryControl::Continue
        );
    }

    #[test]
    fn pause_blocks_until_resume_then_continues() {
        let control = Arc::new(TranscriptionControl::new());
        control.request_pause();
        assert!(control.is_paused());

        let entered = Arc::new(AtomicBool::new(false));
        let worker_control = Arc::clone(&control);
        let worker_entered = Arc::clone(&entered);
        let worker = thread::spawn(move || {
            worker_entered.store(true, Ordering::SeqCst);
            worker_control.wait_at_slice_boundary()
        });

        // Give the worker time to reach the blocking wait, then confirm it has
        // not returned yet (it is parked on the paused boundary).
        while !entered.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        thread::sleep(Duration::from_millis(50));
        assert!(!worker.is_finished(), "worker returned before resume");

        control.request_resume();
        assert_eq!(worker.join().unwrap(), SliceBoundaryControl::Continue);
    }

    #[test]
    fn cancel_while_paused_wakes_and_returns_canceled() {
        let control = Arc::new(TranscriptionControl::new());
        control.request_pause();

        let worker_control = Arc::clone(&control);
        let worker = thread::spawn(move || worker_control.wait_at_slice_boundary());

        thread::sleep(Duration::from_millis(20));
        control.request_cancel();
        assert_eq!(worker.join().unwrap(), SliceBoundaryControl::Canceled);
        // Cancel wins: a pause requested afterward must not clear the cancel.
        control.request_pause();
        assert!(control.is_canceled());
        assert!(!control.is_paused());
    }

    #[test]
    fn install_guard_binds_and_clears_thread_local() {
        assert!(current_transcription_control().is_none());
        let control = Arc::new(TranscriptionControl::new());
        {
            let _guard = install_active_transcription_control(Arc::clone(&control));
            let bound = current_transcription_control().expect("control bound while guard alive");
            assert!(Arc::ptr_eq(&bound, &control));
        }
        assert!(
            current_transcription_control().is_none(),
            "control binding must clear when the guard drops"
        );
    }

    #[test]
    fn cancel_atomic_visible_to_job_abort_data_without_slice_wait() {
        // L2: ggml abort_callback reads the per-job heap atomic via callback data,
        // not TLS and not the slice Condvar path.
        let control = Arc::new(TranscriptionControl::new());
        {
            let _guard = install_active_transcription_control(Arc::clone(&control));
            let data = thread_job_cancel_flag_data();
            assert!(
                !data.is_null(),
                "install must publish abort data on this thread"
            );
            assert!(
                !cancel_flag_requested_from_data(data),
                "armed but not canceled => trampoline must stay false"
            );
            assert!(!thread_job_cancel_requested());
            // Pause must not trip the abort bit.
            control.request_pause();
            assert!(
                !cancel_flag_requested_from_data(data),
                "pause must not trip ggml abort_callback"
            );
            control.request_cancel();
            assert!(
                cancel_flag_requested_from_data(data),
                "request_cancel must dual-write the job atomic behind abort data"
            );
            assert!(thread_job_cancel_requested());
        }
        // After Drop, this thread's published abort data is cleared.
        assert!(
            thread_job_cancel_flag_data().is_null(),
            "drop must disarm this thread's abort data"
        );
        assert!(!thread_job_cancel_requested());
        // The control's own flag stays true after disarm (harmless).
        assert!(control.is_canceled());
    }

    #[test]
    fn no_control_path_leaves_job_cancel_idle() {
        // Bit-identical CLI path: without install_active_transcription_control,
        // no abort callback data is published and callback-free compute uses the
        // original backend API.
        assert!(thread_job_cancel_flag_data().is_null());
        assert!(!thread_job_cancel_requested());
        let orphan = TranscriptionControl::new();
        orphan.request_cancel();
        assert!(orphan.is_canceled());
        assert!(
            thread_job_cancel_flag_data().is_null(),
            "cancel without install must not arm abort data on this thread"
        );
        assert!(!thread_job_cancel_requested());
    }

    #[test]
    fn parallel_installs_cancel_b_does_not_abort_a() {
        // Two fake server jobs on distinct worker threads. Cancel B must not make
        // A's abort data read true -- the process-wide single-slot bug this guards.
        let control_a = Arc::new(TranscriptionControl::new());
        let control_b = Arc::new(TranscriptionControl::new());
        let barrier = Arc::new(Barrier::new(3));

        let thread_a = {
            let control_a = Arc::clone(&control_a);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let _guard = install_active_transcription_control(control_a);
                let data = thread_job_cancel_flag_data();
                barrier.wait();
                barrier.wait();
                assert!(
                    !cancel_flag_requested_from_data(data),
                    "cancel B must not abort job A trampoline data"
                );
            })
        };

        let thread_b = {
            let control_b = Arc::clone(&control_b);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let _guard = install_active_transcription_control(Arc::clone(&control_b));
                let data = thread_job_cancel_flag_data();
                barrier.wait();
                control_b.request_cancel();
                assert!(
                    cancel_flag_requested_from_data(data),
                    "cancel B must abort job B"
                );
                barrier.wait();
            })
        };

        barrier.wait();
        barrier.wait();
        thread_a.join().expect("job A");
        thread_b.join().expect("job B");
        assert!(!control_a.is_canceled());
        assert!(control_b.is_canceled());
    }
}
