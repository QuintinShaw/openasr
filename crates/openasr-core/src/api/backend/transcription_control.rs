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
//! [`Arc`]`<`[`AtomicBool`]`>` that is published into
//! [`crate::ggml_runtime::publish_job_cancel_flag`] for the duration of the
//! install guard; the FFI trampoline reads only that atomic (never TLS).
//!
//! Scope is deliberately in-session: the control lives only for one in-flight
//! transcription. Cross-request or cross-restart resume (which would need
//! persisted partial decode state) is out of scope.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::ggml_runtime::{publish_job_cancel_flag, unpublish_job_cancel_flag_if_current};

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
    /// the ggml abort_callback trampoline (which cannot use thread-locals) via
    /// the job-scoped publish slot armed by
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
    /// token step, or ggml CPU graph node). Idempotent. Wakes a paused worker so
    /// it observes the cancel instead of staying blocked.
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
    /// **not** arm the ggml abort_callback -- only cancel does.
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

    /// Heap cancel flag for job-scoped publish into the ggml abort trampoline.
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
/// unrelated later run on the same pooled worker thread. Also arms the
/// process-wide job cancel atomic used by the ggml abort trampoline.
#[must_use = "the control binding is cleared when this guard is dropped"]
pub struct ActiveTranscriptionControlGuard {
    previous: Option<Arc<TranscriptionControl>>,
    previous_job_cancel: Option<Arc<AtomicBool>>,
    published_flag: Arc<AtomicBool>,
    /// Under `cfg(test)`, holds the job-cancel slot test lock for the full
    /// install lifetime so parallel libtest threads cannot clobber the
    /// process-wide publish slot mid-assertion. Declared last so `Drop` can
    /// unpublish while the lock is still held, then the lock releases.
    #[cfg(test)]
    _job_cancel_slot_test_guard: crate::ggml_runtime::JobCancelSlotTestGuard,
}

impl Drop for ActiveTranscriptionControlGuard {
    fn drop(&mut self) {
        CURRENT_TRANSCRIPTION_CONTROL.with(|cell| {
            *cell.borrow_mut() = self.previous.take();
        });
        // Disarm only if we still own the ggml job-cancel slot (nested installs
        // restore the previous flag rather than clearing a deeper owner's publish).
        unpublish_job_cancel_flag_if_current(&self.published_flag, self.previous_job_cancel.take());
    }
}

/// Bind `control` to the current thread so the in-flight native transcription's
/// long-form slice loop observes pause/cancel requests. Returns a guard that
/// restores the previous binding on drop. Install this at the top of the
/// synchronous decode (e.g. inside the server's `spawn_blocking` closure).
///
/// Also publishes `control`'s cancel atomic into the ggml runtime job slot so
/// ggml's abort_callback (which may run off this thread) observes the same
/// cancel bit. Pause is never published there -- the abort trampoline only
/// recognizes cancel.
pub fn install_active_transcription_control(
    control: Arc<TranscriptionControl>,
) -> ActiveTranscriptionControlGuard {
    #[cfg(test)]
    let job_cancel_slot_test_guard = crate::ggml_runtime::lock_job_cancel_slot_for_test();
    let published_flag = control.cancel_flag();
    let previous_job_cancel = publish_job_cancel_flag(Some(Arc::clone(&published_flag)));
    let previous = CURRENT_TRANSCRIPTION_CONTROL.with(|cell| cell.replace(Some(control)));
    ActiveTranscriptionControlGuard {
        previous,
        previous_job_cancel,
        published_flag,
        #[cfg(test)]
        _job_cancel_slot_test_guard: job_cancel_slot_test_guard,
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
    use crate::ggml_runtime::job_cancel_requested;
    use std::sync::atomic::{AtomicBool, Ordering};
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
    fn cancel_atomic_visible_to_job_cancel_slot_without_slice_wait() {
        // L2: ggml abort_callback reads the published heap atomic, not TLS and
        // not the slice Condvar path. install_active_transcription_control holds
        // the test slot lock for the guard lifetime so parallel libtests cannot
        // clobber the process-wide publish mid-assertion.
        let control = Arc::new(TranscriptionControl::new());
        {
            let _guard = install_active_transcription_control(Arc::clone(&control));
            assert!(
                !job_cancel_requested(),
                "armed but not canceled => trampoline must stay false"
            );
            // Pause must not trip the abort bit.
            control.request_pause();
            assert!(
                !job_cancel_requested(),
                "pause must not arm ggml abort_callback"
            );
            control.request_cancel();
            assert!(
                job_cancel_requested(),
                "request_cancel must dual-write the published atomic"
            );
            // Drop path unpublishes while still holding the test slot lock
            // (field drop order). Post-drop global-slot idle is covered by
            // job_cancel::publish_cancel_and_unpublish_are_visible under its
            // own exclusive lock -- not asserted here (parallel install race).
        }
        // The control's own flag stays true after disarm (harmless); only the
        // process-wide publish is cleared by Drop.
        assert!(control.is_canceled());
    }

    #[test]
    fn no_control_path_leaves_job_cancel_idle() {
        // Bit-identical CLI path: without install_active_transcription_control,
        // the abort trampoline always sees false. Take the test lock so a
        // concurrent install in another libtest thread cannot publish under us.
        let _exclusive = crate::ggml_runtime::lock_job_cancel_slot_for_test();
        let _ = crate::ggml_runtime::publish_job_cancel_flag(None);
        assert!(!job_cancel_requested());
        let orphan = TranscriptionControl::new();
        orphan.request_cancel();
        assert!(orphan.is_canceled());
        assert!(
            !job_cancel_requested(),
            "cancel without install must not publish into the ggml slot"
        );
    }
}
