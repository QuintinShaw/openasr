//! Explicit, `Arc`-cloneable per-request execution context.
//!
//! Replaces the thread-local [`super::TranscriptionControl`] binding
//! (`install_active_transcription_control` / `current_transcription_control`,
//! now removed) as the way a decode boundary observes cancellation. A
//! thread-local only works when the decode that owns a request's cancel
//! control also owns the thread checking it; once a request can be admitted
//! onto a serve-batch owner thread it did not submit from (or a realtime
//! worker that picked it up from a queue), the submitting thread's TLS
//! binding is invisible to the thread actually running the decode, and a
//! cancel silently stops meaning anything.
//!
//! [`RequestExecutionContext`] fixes that by traveling as explicit, ordinary
//! data: it is captured once when a request is admitted and carried inside
//! the job/request struct itself (never installed into TLS), so whichever
//! thread ends up running the decode already has it in hand.
//!
//! Every dispatch surface that can run a decode requires one (never
//! `Option`): [`crate::models::ggml_asr_executor::GgmlAsrExecutionRequest`],
//! the generic seq2seq serve-batch `Envelope`, each family's serve-batch job,
//! and [`crate::realtime::RealtimeBackendWorkItem`]. A caller with nothing to
//! cancel (a CLI single-shot transcribe, an internal test) still constructs a
//! concrete context via [`RequestExecutionContext::detached`] rather than
//! omitting one -- there is no "no context" code path in production.

use std::sync::Arc;

use super::TranscriptionControl;

/// Per-request execution context threaded explicitly through every decode
/// dispatch surface. See the module docs for why this replaced the
/// thread-local control binding.
#[derive(Debug, Clone)]
pub struct RequestExecutionContext {
    /// Client-visible transcription/request id, when the caller registered
    /// one (the server's pause/resume/cancel control endpoints key on this).
    /// `None` for callers that never opted in -- most CLI and realtime
    /// utterance requests.
    pub request_id: Option<String>,
    /// Cancel/pause/resume control for this request's decode.
    pub control: Arc<TranscriptionControl>,
}

// Manual, not derived: `TranscriptionControl` holds a `Mutex`/`Condvar` and
// has no meaningful field-by-field equality. Two contexts are equal when they
// name the same request and share the exact same control instance -- the
// comparison callers of the (derived, request/job-struct-level) `PartialEq`
// actually care about is "is this still the same in-flight request", not
// "do these two independently-constructed controls happen to be in the same
// state".
impl PartialEq for RequestExecutionContext {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id && Arc::ptr_eq(&self.control, &other.control)
    }
}

impl RequestExecutionContext {
    /// Build a context for a request that registered `request_id` and
    /// `control` with the server's in-session control registry.
    pub fn new(request_id: Option<String>, control: Arc<TranscriptionControl>) -> Self {
        Self {
            request_id,
            control,
        }
    }

    /// A context with no external owner: nothing can ever cancel or pause
    /// it. For call paths that have no request id or control to carry (CLI
    /// single-shot transcribe, internal tests) but still need a concrete,
    /// well-formed context to satisfy the required-field contract -- this is
    /// not a "no context" escape hatch, it is a real, valid context whose
    /// control simply has no other holder.
    pub fn detached() -> Self {
        Self {
            request_id: None,
            control: Arc::new(TranscriptionControl::new()),
        }
    }

    /// Whether this request's control has an active cancel request.
    pub fn is_canceled(&self) -> bool {
        self.control.is_canceled()
    }
}

impl Default for RequestExecutionContext {
    fn default() -> Self {
        Self::detached()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detached_context_has_no_request_id_and_never_reports_canceled() {
        let context = RequestExecutionContext::detached();
        assert!(context.request_id.is_none());
        assert!(!context.is_canceled());
    }

    #[test]
    fn new_context_carries_the_given_id_and_control() {
        let control = Arc::new(TranscriptionControl::new());
        let context = RequestExecutionContext::new(Some("job-1".to_string()), Arc::clone(&control));
        assert_eq!(context.request_id.as_deref(), Some("job-1"));
        control.request_cancel();
        assert!(context.is_canceled());
    }
}
