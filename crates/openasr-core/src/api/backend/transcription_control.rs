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
//! Scope is deliberately in-session: the control lives only for one in-flight
//! transcription. Cross-request or cross-restart resume (which would need
//! persisted partial decode state) is out of scope.

use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex};

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
}

impl TranscriptionControl {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ControlState::default()),
            resumed_or_canceled: Condvar::new(),
        }
    }

    /// Request cancellation at the next slice boundary. Idempotent. Wakes a
    /// paused worker so it observes the cancel instead of staying blocked.
    pub fn request_cancel(&self) {
        let mut state = self.lock();
        state.cancel = true;
        self.resumed_or_canceled.notify_all();
    }

    /// Request a pause at the next slice boundary. Idempotent, and a no-op once
    /// canceled (cancel wins and must not be masked by a late pause).
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
        self.lock().cancel
    }

    /// Whether a pause is pending (and not superseded by a cancel).
    pub fn is_paused(&self) -> bool {
        let state = self.lock();
        state.pause && !state.cancel
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
            if state.cancel {
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
            Ok(state) => (state.cancel, state.pause),
            Err(_) => (false, false),
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
/// unrelated later run on the same pooled worker thread.
#[must_use = "the control binding is cleared when this guard is dropped"]
pub struct ActiveTranscriptionControlGuard {
    previous: Option<Arc<TranscriptionControl>>,
}

impl Drop for ActiveTranscriptionControlGuard {
    fn drop(&mut self) {
        CURRENT_TRANSCRIPTION_CONTROL.with(|cell| {
            *cell.borrow_mut() = self.previous.take();
        });
    }
}

/// Bind `control` to the current thread so the in-flight native transcription's
/// long-form slice loop observes pause/cancel requests. Returns a guard that
/// restores the previous binding on drop. Install this at the top of the
/// synchronous decode (e.g. inside the server's `spawn_blocking` closure).
pub fn install_active_transcription_control(
    control: Arc<TranscriptionControl>,
) -> ActiveTranscriptionControlGuard {
    let previous = CURRENT_TRANSCRIPTION_CONTROL.with(|cell| cell.replace(Some(control)));
    ActiveTranscriptionControlGuard { previous }
}

/// The control bound to the current thread, if any. Read by the long-form decode
/// loop at each slice boundary.
pub(crate) fn current_transcription_control() -> Option<Arc<TranscriptionControl>> {
    CURRENT_TRANSCRIPTION_CONTROL.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
