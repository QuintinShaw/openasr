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
//! `Option`): [`crate::models::ggml_asr_executor::GgmlAsrExecutionViewRequest`],
//! the generic seq2seq serve-batch `Envelope`, each family's serve-batch job,
//! and [`crate::realtime::RealtimeBackendWorkItem`]. A caller with nothing to
//! cancel (a CLI single-shot transcribe, an internal test) still constructs a
//! concrete context via [`RequestExecutionContext::uncancellable`] rather
//! than omitting one -- there is no "no context" code path in production.
//! `uncancellable` takes a `reason` argument (never a no-argument escape
//! hatch) -- see its doc comment for why.

use std::sync::Arc;
use std::{fmt, str::FromStr};

use thiserror::Error;

use super::TranscriptionControl;
use crate::models::native_execution_services::ExecutionLaneKey;
use crate::models::request_execution_receipt::NativeExecutionReceiptCollector;

/// Non-secret, process-independent identity for one submitted request attempt.
///
/// This is deliberately distinct from `request_id` (pause/cancel control) and
/// `ExecutionCacheAttemptId` (candidate/cache publication). A managed client
/// may mint it before dispatch; a server mints one when the caller omits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestAttemptId([u8; 16]);

impl RequestAttemptId {
    pub fn generate() -> Result<Self, getrandom::Error> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes)?;
        Ok(Self(bytes))
    }

    pub fn parse(value: &str) -> Result<Self, RequestAttemptIdError> {
        value.parse()
    }
}

impl fmt::Display for RequestAttemptId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl serde::Serialize for RequestAttemptId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for RequestAttemptId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl FromStr for RequestAttemptId {
    type Err = RequestAttemptIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 32
            || !value
                .as_bytes()
                .iter()
                .all(|byte| byte.is_ascii_digit() || (*byte >= b'a' && *byte <= b'f'))
        {
            return Err(RequestAttemptIdError::InvalidFormat);
        }
        let mut bytes = [0_u8; 16];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_lower_hex(pair[0]).ok_or(RequestAttemptIdError::InvalidFormat)?;
            let low = decode_lower_hex(pair[1]).ok_or(RequestAttemptIdError::InvalidFormat)?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

fn decode_lower_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RequestAttemptIdError {
    #[error("request attempt id must be exactly 32 lowercase hexadecimal characters")]
    InvalidFormat,
}

/// Cloneable request-local completed-work observer.
///
/// Unlike a thread-local callback, this value travels with the request into
/// resident actors, serve-batch workers, and auxiliary-model pipelines. The
/// callback is immutable; aggregation belongs to its captured reporter, so
/// sharing it is safe and does not serialize hot loops behind an extra mutex.
#[derive(Clone)]
pub(crate) struct WorkProgressObserver(Arc<dyn Fn(usize, usize) + Send + Sync + 'static>);

impl fmt::Debug for WorkProgressObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WorkProgressObserver(..)")
    }
}

impl WorkProgressObserver {
    pub(crate) fn new(observer: impl Fn(usize, usize) + Send + Sync + 'static) -> Self {
        Self(Arc::new(observer))
    }

    pub(crate) fn report(&self, completed_work: usize, total_work: usize) {
        (self.0)(completed_work, total_work);
    }
}

/// Cloneable request-local observer for postprocessed unstable decode text.
///
/// The shared greedy loop reports each new displayable prefix before EOT so a
/// streaming Poll can emit `transcript.partial` without waiting for the stop
/// token. Empty postprocessed prefixes (for example FunASR `/sil` stripped to
/// nothing) are not reported.
#[derive(Clone)]
pub(crate) struct UnstableDecodeTextObserver(Arc<dyn Fn(&str) + Send + Sync + 'static>);

impl fmt::Debug for UnstableDecodeTextObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UnstableDecodeTextObserver(..)")
    }
}

impl UnstableDecodeTextObserver {
    #[allow(dead_code)] // installed only by tests; live Poll must not fan prefixes
    pub(crate) fn new(observer: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self(Arc::new(observer))
    }

    #[allow(dead_code)] // paired with `new`; production Poll leaves the observer unset
    pub(crate) fn report(&self, text: &str) {
        (self.0)(text);
    }
}

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
    /// Correlation identity for this submitted attempt. It never authorizes
    /// cancellation, replay, daemon access, or cache publication.
    request_attempt_id: Option<RequestAttemptId>,
    /// Cancel/pause/resume control for this request's decode.
    pub control: Arc<TranscriptionControl>,
    /// Optional per-slice decode-work progress. Private so every producer must
    /// use the typed constructor below instead of inventing another callback
    /// transport or process-global registry.
    decode_work_progress: Option<WorkProgressObserver>,
    /// Optional mid-greedy unstable text sink. Streaming partials install this
    /// so the shared decode driver can emit revisable prefixes before EOT;
    /// FINAL / offline paths leave it unset.
    unstable_decode_text: Option<UnstableDecodeTextObserver>,
    /// Explicit opt-in native receipt authority. It is propagated with this
    /// context into candidate attempts and worker-owned decode loops; normal
    /// product requests leave it absent.
    native_execution_receipt: Option<NativeExecutionReceiptCollector>,
    /// Exact candidate lane captured at request dispatch and propagated into
    /// family-owned runtime/cache keys. Absent only for low-level fixtures.
    native_execution_lane: Option<ExecutionLaneKey>,
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
        self.request_id == other.request_id
            && Arc::ptr_eq(&self.control, &other.control)
            && self.request_attempt_id == other.request_attempt_id
    }
}

impl RequestExecutionContext {
    /// Build a context for a request that registered `request_id` and
    /// `control` with the server's in-session control registry.
    pub fn new(request_id: Option<String>, control: Arc<TranscriptionControl>) -> Self {
        Self {
            request_id,
            request_attempt_id: None,
            control,
            decode_work_progress: None,
            unstable_decode_text: None,
            native_execution_receipt: None,
            native_execution_lane: None,
        }
    }

    /// Clone this request context with a decode observer scoped to one slice.
    /// Parallel slices share cancellation identity while retaining independent
    /// progress windows.
    pub(crate) fn with_decode_work_progress_observer(
        &self,
        observer: WorkProgressObserver,
    ) -> Self {
        Self {
            request_id: self.request_id.clone(),
            request_attempt_id: self.request_attempt_id,
            control: Arc::clone(&self.control),
            decode_work_progress: Some(observer),
            unstable_decode_text: self.unstable_decode_text.clone(),
            native_execution_receipt: self.native_execution_receipt.clone(),
            native_execution_lane: self.native_execution_lane.clone(),
        }
    }

    pub(crate) fn decode_work_progress_observer(&self) -> Option<&WorkProgressObserver> {
        self.decode_work_progress.as_ref()
    }

    #[allow(dead_code)] // reserved for a live mid-decode flush; snapshot Poll must not fan out prefixes
    pub(crate) fn with_unstable_decode_text_observer(
        &self,
        observer: UnstableDecodeTextObserver,
    ) -> Self {
        Self {
            request_id: self.request_id.clone(),
            request_attempt_id: self.request_attempt_id,
            control: Arc::clone(&self.control),
            decode_work_progress: self.decode_work_progress.clone(),
            unstable_decode_text: Some(observer),
            native_execution_receipt: self.native_execution_receipt.clone(),
            native_execution_lane: self.native_execution_lane.clone(),
        }
    }

    pub(crate) fn unstable_decode_text_observer(&self) -> Option<&UnstableDecodeTextObserver> {
        self.unstable_decode_text.as_ref()
    }

    pub fn with_request_attempt_id(mut self, attempt_id: RequestAttemptId) -> Self {
        self.request_attempt_id = Some(attempt_id);
        if let Some(receipt) = self.native_execution_receipt.as_ref() {
            receipt.bind_request_attempt(attempt_id);
        }
        self
    }

    pub fn request_attempt_id(&self) -> Option<RequestAttemptId> {
        self.request_attempt_id
    }

    /// Attach the one explicit request-scoped authority that can receive native
    /// execution facts and decode trace events. Receipt consumers must use this
    /// value rather than re-resolving backend policy after the request returns.
    pub fn with_native_execution_receipt(
        mut self,
        receipt: NativeExecutionReceiptCollector,
    ) -> Self {
        if let Some(attempt_id) = self.request_attempt_id {
            receipt.bind_request_attempt(attempt_id);
        }
        self.native_execution_receipt = Some(receipt);
        self
    }

    pub fn native_execution_receipt(&self) -> Option<NativeExecutionReceiptCollector> {
        self.native_execution_receipt.clone()
    }

    /// Attach the exact candidate lane selected for this request attempt.
    pub(crate) fn with_native_execution_lane(mut self, lane: ExecutionLaneKey) -> Self {
        self.native_execution_lane = Some(lane);
        self
    }

    pub(crate) fn native_execution_lane(&self) -> Option<&ExecutionLaneKey> {
        self.native_execution_lane.as_ref()
    }

    /// A context with no external owner: nothing can ever cancel or pause
    /// it. For call paths that have no request id or control to carry (CLI
    /// single-shot transcribe, a public request builder's pre-opt-in
    /// default, an internal test) but still need a concrete, well-formed
    /// context to satisfy the required-field contract -- this is not a "no
    /// context" escape hatch, it is a real, valid context whose control is
    /// explicitly detached so native graph execution stays callback-free.
    ///
    /// `reason` must name, in the caller's own words, *why* this particular
    /// call site has no cancel surface -- never a placeholder like `"n/a"`.
    /// There is deliberately no no-argument form: every opt-out reads inline
    /// at its call site instead of being silently inherited by whoever next
    /// copies the line, and the full list of opt-outs is one
    /// `rg -n 'RequestExecutionContext::uncancellable'` away from being a
    /// reviewable, shrinkable list rather than something discovered by
    /// accident.
    ///
    /// `reason` is a `&'static str`, not a closed enum: the real reasons are
    /// heterogeneous prose ("no request id was ever registered", "this
    /// streaming request type carries no context field yet", "a public
    /// builder's value before a caller opts in") that do not cluster into a
    /// small, stable taxonomy. Closing them into an enum would either force
    /// awkward buckets or grow its own `Other(&'static str)` variant --
    /// ceremony without adding any enforcement an inline string doesn't
    /// already give a reviewer. `reason` is not stored on the context
    /// (nothing at runtime consults it -- `control` alone determines
    /// cancellation behavior either way); it is a compile-time/code-review
    /// aid only, so it costs this frequently-cloned struct nothing.
    ///
    /// As of this writing exactly two classes of *production* call site are
    /// real gaps -- each needs product work, not a context-plumbing change,
    /// to close:
    ///   - `openasr-server`'s one-shot SSE file-transcription response path
    ///     has no client-disconnect signal to observe yet (closing it needs
    ///     a disconnect hook on that stream).
    ///   - `ctc_streaming_driver.rs` / `incremental_streaming_driver.rs`'s
    ///     per-frame streaming partial/final requests carry no
    ///     execution-context field of their own yet, unlike the offline
    ///     request path (closing it needs those streaming request types to
    ///     grow the field and thread it from the session's own cancel
    ///     source).
    ///
    /// Every other production call site opts out because the caller
    /// genuinely has no request id / control to carry (never registered one)
    /// or is a public request builder's value before a caller opts in via
    /// `with_execution_context` -- those are not gaps, they are correctly
    /// detached.
    pub fn uncancellable(reason: &'static str) -> Self {
        debug_assert!(
            !reason.trim().is_empty(),
            "RequestExecutionContext::uncancellable() requires a real reason, not a placeholder"
        );
        let _ = reason;
        Self {
            request_id: None,
            request_attempt_id: None,
            control: Arc::new(TranscriptionControl::detached()),
            decode_work_progress: None,
            unstable_decode_text: None,
            native_execution_receipt: None,
            native_execution_lane: None,
        }
    }

    /// Whether this request's control has an active cancel request.
    pub fn is_canceled(&self) -> bool {
        self.control.is_canceled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncancellable_context_has_no_request_id_and_never_reports_canceled() {
        let context = RequestExecutionContext::uncancellable("test fixture");
        assert!(context.request_id.is_none());
        assert!(!context.is_canceled());
        assert!(!context.control.has_cancel_source());
    }

    #[test]
    fn new_context_carries_the_given_id_and_control() {
        let control = Arc::new(TranscriptionControl::new());
        let context = RequestExecutionContext::new(Some("job-1".to_string()), Arc::clone(&control));
        assert!(context.control.has_cancel_source());
        assert_eq!(context.request_id.as_deref(), Some("job-1"));
        control.request_cancel();
        assert!(context.is_canceled());
    }

    #[test]
    fn request_attempt_is_lower_hex_roundtrippable_and_not_a_control_id() {
        let attempt = RequestAttemptId::parse("00112233445566778899aabbccddeeff").unwrap();
        assert_eq!(attempt.to_string(), "00112233445566778899aabbccddeeff");
        assert!(RequestAttemptId::parse("00112233445566778899AABBCCDDEEFF").is_err());
        assert!(RequestAttemptId::parse("../00112233445566778899aabbccddeeff").is_err());
        let json = serde_json::to_string(&attempt).unwrap();
        assert_eq!(json, "\"00112233445566778899aabbccddeeff\"");
        assert_eq!(
            serde_json::from_str::<RequestAttemptId>(&json).unwrap(),
            attempt
        );

        let context = RequestExecutionContext::uncancellable("request attempt test")
            .with_request_attempt_id(attempt);
        assert_eq!(context.request_attempt_id(), Some(attempt));
        assert!(context.request_id.is_none());
    }

    #[test]
    fn slice_context_preserves_request_attempt_identity() {
        let attempt = RequestAttemptId::parse("ffeeddccbbaa99887766554433221100").unwrap();
        let context = RequestExecutionContext::uncancellable("request attempt propagation")
            .with_request_attempt_id(attempt)
            .with_decode_work_progress_observer(WorkProgressObserver::new(|_, _| {}));
        assert_eq!(context.request_attempt_id(), Some(attempt));
    }

    #[test]
    fn native_receipt_authority_propagates_through_slice_contexts() {
        let receipt = NativeExecutionReceiptCollector::new();
        let context = RequestExecutionContext::uncancellable("receipt propagation test")
            .with_native_execution_receipt(receipt.clone())
            .with_decode_work_progress_observer(WorkProgressObserver::new(|_, _| {}));
        let propagated = context
            .native_execution_receipt()
            .expect("slice context retains receipt authority");
        propagated.record_token(0, 7, false);
        assert_eq!(receipt.snapshot().trace.event_count, 1);
    }

    #[test]
    fn decode_work_observers_cross_threads_without_cross_request_leakage() {
        let observed_a = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_b = Arc::new(std::sync::Mutex::new(Vec::new()));
        let context_a = RequestExecutionContext::uncancellable("progress request A")
            .with_decode_work_progress_observer(WorkProgressObserver::new({
                let observed = Arc::clone(&observed_a);
                move |completed, total| {
                    observed
                        .lock()
                        .expect("request A progress")
                        .push((completed, total));
                }
            }));
        let context_b = RequestExecutionContext::uncancellable("progress request B")
            .with_decode_work_progress_observer(WorkProgressObserver::new({
                let observed = Arc::clone(&observed_b);
                move |completed, total| {
                    observed
                        .lock()
                        .expect("request B progress")
                        .push((completed, total));
                }
            }));

        let worker_a = std::thread::spawn(move || {
            context_a
                .decode_work_progress_observer()
                .expect("request A observer")
                .report(3, 8);
        });
        let worker_b = std::thread::spawn(move || {
            context_b
                .decode_work_progress_observer()
                .expect("request B observer")
                .report(5, 13);
        });
        worker_a.join().expect("request A worker");
        worker_b.join().expect("request B worker");

        assert_eq!(
            *observed_a.lock().expect("request A progress"),
            vec![(3, 8)]
        );
        assert_eq!(
            *observed_b.lock().expect("request B progress"),
            vec![(5, 13)]
        );
    }
}
