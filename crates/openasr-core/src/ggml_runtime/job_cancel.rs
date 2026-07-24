//! Job-scoped cancel atomic for the ggml abort_callback trampoline.
//!
//! ggml CPU may invoke abort_callback on a worker-pool thread that does not
//! share the transcription owner thread's TLS. Cancel is therefore published as
//! a heap [`Arc`]`<`[`AtomicBool`]`>` for the duration of one native transcription
//! (armed by `install_active_transcription_control`, disarmed on guard Drop).
//!
//! Lives under `ggml_runtime` (not `api::backend`) so the graph runner can read
//! it without a crate-internal module cycle.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

static CURRENT_JOB_CANCEL: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);

/// Replace the published job cancel flag. Returns the previous flag (if any)
/// so nested install guards can restore it on drop.
pub(crate) fn publish_job_cancel_flag(flag: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    match CURRENT_JOB_CANCEL.lock() {
        Ok(mut slot) => std::mem::replace(&mut *slot, flag),
        Err(_) => None,
    }
}

/// True when the currently published job cancel atomic is set. False when no
/// control is installed (CLI / no-control path stays bit-identical: trampoline
/// always false). Panic-free for use under `catch_unwind` in the C trampoline.
pub(crate) fn job_cancel_requested() -> bool {
    match CURRENT_JOB_CANCEL.lock() {
        Ok(slot) => slot
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst)),
        Err(_) => false,
    }
}

/// Disarm the slot only if it still points at `flag` (nested installs restore
/// their previous publish rather than clearing a deeper owner's flag).
pub(crate) fn unpublish_job_cancel_flag_if_current(
    flag: &Arc<AtomicBool>,
    previous: Option<Arc<AtomicBool>>,
) {
    if let Ok(mut slot) = CURRENT_JOB_CANCEL.lock() {
        let still_ours = slot
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, flag));
        if still_ours {
            *slot = previous;
        }
    }
}

/// Serializes unit tests that publish into the process-wide job-cancel slot so
/// parallel libtest threads do not clobber each other's assertions. Held for the
/// lifetime of an installed transcription control under `cfg(test)`. Production
/// install/drop paths never take this lock.
#[cfg(test)]
pub(crate) type JobCancelSlotTestGuard = std::sync::MutexGuard<'static, ()>;

#[cfg(test)]
static JOB_CANCEL_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the test-only job-cancel slot lock.
#[cfg(test)]
pub(crate) fn lock_job_cancel_slot_for_test() -> JobCancelSlotTestGuard {
    JOB_CANCEL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_cancel_and_unpublish_are_visible() {
        let _exclusive = lock_job_cancel_slot_for_test();
        let _ = publish_job_cancel_flag(None);
        assert!(!job_cancel_requested());
        let flag = Arc::new(AtomicBool::new(false));
        let prev = publish_job_cancel_flag(Some(Arc::clone(&flag)));
        assert!(prev.is_none());
        assert!(!job_cancel_requested());
        flag.store(true, Ordering::SeqCst);
        assert!(job_cancel_requested());
        unpublish_job_cancel_flag_if_current(&flag, None);
        assert!(!job_cancel_requested());
        let other = Arc::new(AtomicBool::new(true));
        let _ = publish_job_cancel_flag(Some(Arc::clone(&other)));
        unpublish_job_cancel_flag_if_current(&flag, None);
        assert!(
            job_cancel_requested(),
            "newer publish must survive stale unpublish"
        );
        unpublish_job_cancel_flag_if_current(&other, None);
        assert!(!job_cancel_requested());
    }
}
