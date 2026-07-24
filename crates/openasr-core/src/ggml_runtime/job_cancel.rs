//! Per-job cancel atomic for the ggml abort_callback trampoline.
//!
//! ggml CPU may invoke abort_callback on a worker-pool thread that does not
//! share the transcription owner thread's TLS. Cancel is therefore carried as
//! a heap [`Arc`]`<`[`AtomicBool`]`>` **per job**, and the backend callback's
//! `data` pointer is set to that atomic (`Arc::as_ptr`) for the duration of the
//! job on the worker thread that owns the backends.
//!
//! There is intentionally **no** process-wide publish slot: server multi-model
//! parallel jobs each arm their own thread-local backends with their own flag
//! pointer, so canceling job B cannot make job A's trampoline return true.
//!
//! Lives under `ggml_runtime` (not `api::backend`) so the graph runner can read
//! it without a crate-internal module cycle.

use std::cell::RefCell;
use std::os::raw::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    /// Cancel flag for the transcription currently running on this worker thread.
    /// Used only to supply `abort_callback` data when backends are created or
    /// re-armed on this thread. The trampoline itself never reads this TLS -- it
    /// loads through the `data` pointer installed on the backend.
    static ACTIVE_JOB_CANCEL: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Install or replace this thread's job cancel flag. Returns the previous flag
/// (if any) so nested install guards can restore it on drop.
pub(crate) fn arm_thread_job_cancel_flag(flag: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    ACTIVE_JOB_CANCEL.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), flag))
}

/// Disarm only if the thread slot still points at `flag` (nested installs restore
/// their previous publish rather than clearing a deeper owner's flag). Returns
/// whether the slot was ours and was updated.
pub(crate) fn disarm_thread_job_cancel_flag_if_current(
    flag: &Arc<AtomicBool>,
    previous: Option<Arc<AtomicBool>>,
) -> bool {
    ACTIVE_JOB_CANCEL.with(|cell| {
        let mut slot = cell.borrow_mut();
        let still_ours = slot
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, flag));
        if still_ours {
            *slot = previous;
        }
        still_ours
    })
}

/// Raw `abort_callback` data pointer for backends on this thread: `Arc::as_ptr`
/// of the armed job cancel atomic, or null when no control is installed.
///
/// Backend arm paths treat null as "clear the callback entirely" so the CLI /
/// no-control path stays bit-identical to pre-L2 (no per-node abort poll).
///
/// SAFETY for consumers: the pointer remains valid while the corresponding
/// [`Arc`] is held in this thread's slot and/or by the install guard that armed
/// it. Callers must re-arm backends to null (or a still-live previous flag)
/// before the last owning [`Arc`] drops.
pub(crate) fn thread_job_cancel_flag_data() -> *mut c_void {
    ACTIVE_JOB_CANCEL.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|flag| Arc::as_ptr(flag) as *mut c_void)
            .unwrap_or(std::ptr::null_mut())
    })
}

/// Wait-free cancel check used by the ggml abort trampoline.
///
/// `data` is either null (defensive; production installs a null *callback* when
/// disarmed, so the trampoline is not invoked) or `Arc::as_ptr` of the job's
/// [`AtomicBool`]. Pause never writes that atomic, so pause cannot trip abort.
/// Panic-free (null check + atomic load) for direct use from `extern "C"`.
#[inline]
pub(crate) fn cancel_flag_requested_from_data(data: *mut c_void) -> bool {
    if data.is_null() {
        return false;
    }
    // SAFETY: install paths only pass Arc::as_ptr of a job AtomicBool kept alive
    // for the full backend-callback arm window (guard + thread slot).
    unsafe { (*(data as *const AtomicBool)).load(Ordering::SeqCst) }
}

/// True when this thread's armed job cancel atomic is set. False when no control
/// is installed. Test/helper path only -- production trampoline reads `data`.
#[cfg(test)]
pub(crate) fn thread_job_cancel_requested() -> bool {
    ACTIVE_JOB_CANCEL.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn arm_cancel_and_disarm_are_visible_on_thread_data_pointer() {
        let _ = arm_thread_job_cancel_flag(None);
        assert!(thread_job_cancel_flag_data().is_null());
        assert!(!cancel_flag_requested_from_data(std::ptr::null_mut()));
        assert!(!thread_job_cancel_requested());

        let flag = Arc::new(AtomicBool::new(false));
        let prev = arm_thread_job_cancel_flag(Some(Arc::clone(&flag)));
        assert!(prev.is_none());
        let data = thread_job_cancel_flag_data();
        assert!(!data.is_null());
        assert!(!cancel_flag_requested_from_data(data));
        assert!(!thread_job_cancel_requested());

        flag.store(true, Ordering::SeqCst);
        assert!(cancel_flag_requested_from_data(data));
        assert!(thread_job_cancel_requested());

        assert!(disarm_thread_job_cancel_flag_if_current(&flag, None));
        assert!(thread_job_cancel_flag_data().is_null());
        // Stale data pointer would dangle after disarm+drop; only null is safe
        // once the Arc is gone. While `flag` still lives, the old data still
        // reads true -- backends must be re-armed to null on disarm (cpu_graph).
        assert!(cancel_flag_requested_from_data(data));
        assert!(!thread_job_cancel_requested());
    }

    #[test]
    fn stale_disarm_does_not_clear_newer_thread_arm() {
        let flag = Arc::new(AtomicBool::new(false));
        let other = Arc::new(AtomicBool::new(true));
        let _ = arm_thread_job_cancel_flag(Some(Arc::clone(&other)));
        assert!(!disarm_thread_job_cancel_flag_if_current(&flag, None));
        assert!(
            thread_job_cancel_requested(),
            "newer arm must survive stale disarm"
        );
        assert!(cancel_flag_requested_from_data(
            thread_job_cancel_flag_data()
        ));
        assert!(disarm_thread_job_cancel_flag_if_current(&other, None));
        assert!(!thread_job_cancel_requested());
    }

    #[test]
    fn cancel_job_b_does_not_abort_job_a_via_distinct_data_pointers() {
        // Structural guarantee: each job's backends hold that job's AtomicBool
        // pointer as callback data. Cancel B must not make A's trampoline true.
        let flag_a = Arc::new(AtomicBool::new(false));
        let flag_b = Arc::new(AtomicBool::new(false));
        let data_a = Arc::as_ptr(&flag_a) as *mut c_void;
        let data_b = Arc::as_ptr(&flag_b) as *mut c_void;

        assert!(!cancel_flag_requested_from_data(data_a));
        assert!(!cancel_flag_requested_from_data(data_b));

        flag_b.store(true, Ordering::SeqCst);
        assert!(
            !cancel_flag_requested_from_data(data_a),
            "cancel B must not abort A via trampoline data"
        );
        assert!(
            cancel_flag_requested_from_data(data_b),
            "cancel B must abort B"
        );

        flag_a.store(true, Ordering::SeqCst);
        assert!(cancel_flag_requested_from_data(data_a));
        assert!(cancel_flag_requested_from_data(data_b));
    }

    #[test]
    fn interleaved_install_on_two_threads_keeps_cancel_isolated() {
        // Fake parallel server jobs: each worker thread arms its own flag. Cancel
        // on B must leave A's data-pointer read false without any global lock.
        let flag_a = Arc::new(AtomicBool::new(false));
        let flag_b = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(3));

        let thread_a = {
            let flag_a = Arc::clone(&flag_a);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let prev = arm_thread_job_cancel_flag(Some(Arc::clone(&flag_a)));
                assert!(prev.is_none());
                let data = thread_job_cancel_flag_data();
                barrier.wait(); // both armed
                barrier.wait(); // after B canceled
                assert!(
                    !cancel_flag_requested_from_data(data),
                    "job A trampoline must stay false after cancel B"
                );
                assert!(!thread_job_cancel_requested());
                assert!(disarm_thread_job_cancel_flag_if_current(&flag_a, None));
            })
        };

        let thread_b = {
            let flag_b = Arc::clone(&flag_b);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let prev = arm_thread_job_cancel_flag(Some(Arc::clone(&flag_b)));
                assert!(prev.is_none());
                let data = thread_job_cancel_flag_data();
                barrier.wait(); // both armed
                flag_b.store(true, Ordering::SeqCst);
                assert!(
                    cancel_flag_requested_from_data(data),
                    "job B trampoline must observe its own cancel"
                );
                assert!(thread_job_cancel_requested());
                barrier.wait(); // release A to check isolation
                assert!(disarm_thread_job_cancel_flag_if_current(&flag_b, None));
            })
        };

        barrier.wait();
        barrier.wait();
        thread_a.join().expect("job A thread");
        thread_b.join().expect("job B thread");
        assert!(!flag_a.load(Ordering::SeqCst));
        assert!(flag_b.load(Ordering::SeqCst));
    }
}
