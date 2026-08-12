//! Explicit process-owned services for native model execution.
//!
//! A host constructs one service root and injects the same [`Arc`] into every
//! offline backend and streaming session. The root owns both dispatch tables,
//! the stateful family executors shared by those tables, execution-policy
//! resolution, and device-memory accounting. Keeping these resources under the
//! same explicit lifetime prevents a cached model from outliving the broker
//! that admitted it.

use std::{
    cell::{Cell, RefCell},
    fmt,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use thiserror::Error;

use crate::device::{
    execution_memory::{DeviceMemoryBrokerSet, DeviceMemoryPolicy, MemoryReservationCohortId},
    execution_policy::{
        DefaultExecutionPolicyResolver, ExecutionCandidate, ExecutionCandidateFailure,
        ExecutionPlacement, ExecutionPolicyResolver,
    },
    execution_route::{ExecutionProvider, ExecutionRouteCacheKey, ResolvedExecutionRoute},
};
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlExecutionPlacementSummary, GgmlExecutionTelemetryCollector,
    GgmlExecutionTelemetryGuard, RequestBackendOverrideGuard, RequestBackendPreference,
    current_execution_telemetry_collector, install_execution_telemetry_collector,
    install_request_backend_override, request_backend_override, resolve_request_execution_route,
};

use super::{
    builtin_execution_dispatch::{
        build_builtin_ggml_execution_dispatch, build_builtin_ggml_streaming_execution_dispatch,
    },
    executor_component_registry::BuiltinStatefulExecutorScope,
    ggml_asr_executor::GgmlAsrExecutionDispatch,
};

static NEXT_EXECUTION_SCOPE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EXECUTION_CACHE_ATTEMPT_ID: AtomicU64 = AtomicU64::new(1);
/// One byte ledger for every default service root in this process. Multiple
/// hosts (CLI + embedded API, or several server roots in tests) must not each
/// admit against the same live-free snapshot independently.
static PROCESS_MEMORY_BROKER: OnceLock<Arc<DeviceMemoryBrokerSet>> = OnceLock::new();

fn process_memory_broker() -> Arc<DeviceMemoryBrokerSet> {
    Arc::clone(
        PROCESS_MEMORY_BROKER
            .get_or_init(|| Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()))),
    )
}

thread_local! {
    /// Dynamically scoped identity for model-runtime cache construction.
    ///
    /// The service object itself is never ambient: callers still inject the
    /// required `Arc<NativeExecutionServices>` on every request. This TLS only
    /// carries its small identity through legacy family helpers whose cache-key
    /// APIs do not yet accept a request argument.
    static CURRENT_EXECUTION_SCOPE_ID: Cell<Option<NativeExecutionScopeId>> = const {
        Cell::new(None)
    };
    /// Dynamically scoped transport for the explicitly injected process-wide
    /// broker. This is not an ambient owner: the `Arc` originates at the
    /// request's [`NativeExecutionServices`] and is installed only while that
    /// request (or an explicitly propagated worker) is inside native code.
    static CURRENT_EXECUTION_MEMORY_BROKER: RefCell<Option<Arc<DeviceMemoryBrokerSet>>> = const {
        RefCell::new(None)
    };
    /// Placement selected by the active policy candidate. Graph-runtime
    /// configuration consumes this value to make `FullDevice` and `Hybrid`
    /// executable contracts rather than diagnostic labels.
    static CURRENT_EXECUTION_PLACEMENT: Cell<Option<ExecutionPlacement>> = const {
        Cell::new(None)
    };
    /// Typed, attempt-local failure channel. Low-level allocators record only
    /// candidate-local resource/device failures here; business/decode/input
    /// failures never touch it and therefore can never trigger fallback.
    static CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK:
        RefCell<Option<ExecutionCandidateFailureSink>> = const { RefCell::new(None) };
    /// Attempt-local publication journal for resident backend owners. Family
    /// caches may construct an owner while trying a candidate, but the owner
    /// is not visible to later requests until the complete attempt succeeds
    /// without a typed candidate failure. Failed attempts drop staged owners
    /// in reverse construction order.
    static CURRENT_EXECUTION_CACHE_JOURNAL:
        RefCell<Option<ExecutionCacheJournal>> = const { RefCell::new(None) };
    /// Transaction identity shared by nested policy attempts. Owner caches use
    /// it to expose staged values to their own attempt without leaking them to
    /// concurrent candidates before the outermost journal commits.
    static CURRENT_EXECUTION_CACHE_ATTEMPT_ID: Cell<Option<ExecutionCacheAttemptId>> = const {
        Cell::new(None)
    };
}

type DeferredCacheCommit = Box<dyn FnOnce() + 'static>;
type DeferredCacheRollback = Box<dyn FnOnce() + 'static>;

struct ExecutionCacheJournal {
    attempt_id: ExecutionCacheAttemptId,
    commits: Vec<DeferredCacheCommit>,
    rollbacks: Vec<DeferredCacheRollback>,
}

impl ExecutionCacheJournal {
    fn new(attempt_id: ExecutionCacheAttemptId) -> Self {
        Self {
            attempt_id,
            commits: Vec::new(),
            rollbacks: Vec::new(),
        }
    }
}

impl ExecutionCacheJournal {
    fn commit(mut self) {
        // A successful candidate makes rollback-only invalidations obsolete.
        self.rollbacks.clear();
        for commit in self.commits.drain(..) {
            commit();
        }
    }

    fn rollback(mut self) {
        // Captured owners are destroyed in reverse construction order, which
        // mirrors ordinary stack unwinding and releases dependent graph state
        // before the resources it was built from.
        while let Some(commit) = self.commits.pop() {
            drop(commit);
        }
        while let Some(rollback) = self.rollbacks.pop() {
            rollback();
        }
    }
}

struct ExecutionCacheJournalScope {
    previous: Option<ExecutionCacheJournal>,
    previous_attempt_id: Option<ExecutionCacheAttemptId>,
    active: bool,
}

impl ExecutionCacheJournalScope {
    fn begin() -> Self {
        let attempt_id = CURRENT_EXECUTION_CACHE_JOURNAL.with(|current| {
            current
                .borrow()
                .as_ref()
                .map(|journal| journal.attempt_id)
                .unwrap_or_else(ExecutionCacheAttemptId::next)
        });
        let previous = CURRENT_EXECUTION_CACHE_JOURNAL
            .with(|current| current.replace(Some(ExecutionCacheJournal::new(attempt_id))));
        let previous_attempt_id =
            CURRENT_EXECUTION_CACHE_ATTEMPT_ID.with(|current| current.replace(Some(attempt_id)));
        Self {
            previous,
            previous_attempt_id,
            active: true,
        }
    }

    fn finish(mut self, commit: bool) {
        let journal = CURRENT_EXECUTION_CACHE_JOURNAL
            .with(|current| current.replace(self.previous.take()))
            .expect("candidate attempt installed a cache journal");
        CURRENT_EXECUTION_CACHE_ATTEMPT_ID
            .with(|current| current.set(self.previous_attempt_id.take()));
        self.active = false;
        if !commit {
            journal.rollback();
            return;
        }

        // A nested attempt must remain transactional with its parent: move
        // its publications into the parent journal instead of exposing them
        // early.
        let mut journal = Some(journal);
        let merged = CURRENT_EXECUTION_CACHE_JOURNAL.with(|current| {
            let mut current = current.borrow_mut();
            let Some(parent) = current.as_mut() else {
                return false;
            };
            parent
                .commits
                .append(&mut journal.as_mut().expect("journal available").commits);
            parent
                .rollbacks
                .append(&mut journal.as_mut().expect("journal available").rollbacks);
            true
        });
        if !merged {
            journal
                .expect("unmerged journal remains available")
                .commit();
        }
    }
}

impl Drop for ExecutionCacheJournalScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let journal =
            CURRENT_EXECUTION_CACHE_JOURNAL.with(|current| current.replace(self.previous.take()));
        CURRENT_EXECUTION_CACHE_ATTEMPT_ID
            .with(|current| current.set(self.previous_attempt_id.take()));
        if let Some(journal) = journal {
            journal.rollback();
        }
    }
}

/// Physical execution identity for a resident backend owner.
///
/// `GgmlCpuGraphBackend::Gpu` deliberately is not sufficient: it folds CUDA,
/// HIP, Vulkan and every visible card together. The route key retains the
/// provider-local stable id plus PCI identity when ggml exposes it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResolvedDeviceKey {
    route: ExecutionRouteCacheKey,
}

/// Cache key shared by every resident object that owns a ggml backend, device
/// buffer, scheduler, graph, or uploaded tensor arena.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionLaneKey {
    device: ResolvedDeviceKey,
    placement: ExecutionPlacement,
    backend: GgmlCpuGraphBackend,
}

impl ExecutionLaneKey {
    pub(crate) fn backend(&self) -> GgmlCpuGraphBackend {
        self.backend
    }

    #[cfg(test)]
    fn placement(&self) -> ExecutionPlacement {
        self.placement
    }
}

/// Stable identity of one explicitly constructed execution-service root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeExecutionScopeId(u64);

impl NativeExecutionScopeId {
    fn next() -> Self {
        Self(NEXT_EXECUTION_SCOPE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Identity of the outermost transactional cache-publication attempt. Nested
/// auxiliary candidates inherit it and merge their journals into the parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ExecutionCacheAttemptId(u64);

impl ExecutionCacheAttemptId {
    fn next() -> Self {
        Self(NEXT_EXECUTION_CACHE_ATTEMPT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Restores the prior dynamically scoped execution identity on drop.
pub(crate) struct NativeExecutionScopeGuard {
    previous: Option<NativeExecutionScopeId>,
}

impl Drop for NativeExecutionScopeGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_SCOPE_ID.with(|current| current.set(self.previous));
    }
}

/// Cloneable value used to propagate one request's explicitly injected
/// execution services into a worker thread. It intentionally carries only
/// the cache namespace and memory broker required below the dispatch layer,
/// not the dispatch tables themselves.
#[derive(Clone)]
pub(crate) struct NativeExecutionContext {
    scope_id: NativeExecutionScopeId,
    memory_broker: Arc<DeviceMemoryBrokerSet>,
    backend_preference: Option<RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
    failure_sink: Option<ExecutionCandidateFailureSink>,
    cache_attempt_id: Option<ExecutionCacheAttemptId>,
    execution_telemetry: Option<GgmlExecutionTelemetryCollector>,
}

impl NativeExecutionContext {
    /// Stable execution-lane equality for a shared worker/engine key. The
    /// request-local failure sink is intentionally excluded: two jobs may
    /// share an engine only when their scope, broker, backend and placement
    /// agree, while each still retains its own sink for failure fan-out.
    pub(crate) fn shares_execution_lane_with(&self, other: &Self) -> bool {
        self.scope_id == other.scope_id
            && Arc::ptr_eq(&self.memory_broker, &other.memory_broker)
            && self.backend_preference == other.backend_preference
            && self.placement == other.placement
    }

    /// Builds a temporary worker context for one shared graph operation.
    ///
    /// Engine identity is deliberately stricter than family-level
    /// `can_batch_with`: requests may share a graph only when they belong to
    /// the same injected service root, broker, backend route, and placement.
    /// Their attempt-local failure sinks remain independent; the returned
    /// context fans a low-level typed failure out to every request that is
    /// active for this operation.
    pub(crate) fn shared_lane(
        contexts: &[Self],
    ) -> Result<Option<Self>, NativeExecutionContextError> {
        let Some(first) = contexts.first() else {
            return Ok(None);
        };
        for (index, context) in contexts.iter().enumerate().skip(1) {
            if !first.shares_execution_lane_with(context) {
                return Err(NativeExecutionContextError::IncompatibleSharedLane { index });
            }
        }

        let failure_sink = ExecutionCandidateFailureSink::fanout(
            contexts
                .iter()
                .filter_map(|context| context.failure_sink.as_ref()),
        );
        let cache_attempt_id = contexts
            .iter()
            .map(|context| context.cache_attempt_id)
            .reduce(|left, right| (left == right).then_some(left).flatten())
            .flatten();
        let execution_telemetry = GgmlExecutionTelemetryCollector::fanout(
            contexts
                .iter()
                .filter_map(|context| context.execution_telemetry.as_ref()),
        );
        Ok(Some(Self {
            scope_id: first.scope_id,
            memory_broker: Arc::clone(&first.memory_broker),
            backend_preference: first.backend_preference.clone(),
            placement: first.placement,
            failure_sink,
            cache_attempt_id,
            execution_telemetry,
        }))
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum NativeExecutionContextError {
    #[error(
        "request at index {index} does not share the batch execution scope, broker, backend, and placement"
    )]
    IncompatibleSharedLane { index: usize },
}

/// Restores both dynamically scoped values when a request/worker exits.
pub(crate) struct NativeExecutionContextGuard {
    scope: NativeExecutionScopeGuard,
    previous_memory_broker: Option<Arc<DeviceMemoryBrokerSet>>,
    previous_placement: Option<ExecutionPlacement>,
    previous_failure_sink: Option<ExecutionCandidateFailureSink>,
    previous_cache_attempt_id: Option<ExecutionCacheAttemptId>,
    execution_telemetry: GgmlExecutionTelemetryGuard,
    backend: RequestBackendOverrideGuard,
}

impl Drop for NativeExecutionContextGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_MEMORY_BROKER.with(|current| {
            *current.borrow_mut() = self.previous_memory_broker.take();
        });
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.set(self.previous_placement));
        CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK.with(|current| {
            *current.borrow_mut() = self.previous_failure_sink.take();
        });
        CURRENT_EXECUTION_CACHE_ATTEMPT_ID
            .with(|current| current.set(self.previous_cache_attempt_id.take()));
        // `scope` restores itself after this `Drop` returns.
        let _ = &self.scope;
        // `backend` restores itself after this `Drop` returns.
        let _ = &self.backend;
        // `execution_telemetry` restores itself after this `Drop` returns.
        let _ = &self.execution_telemetry;
    }
}

/// Cloneable, request-scoped typed failure recorder for one candidate attempt.
/// The first recorded failure wins: it is the closest causal fact to the
/// allocation/device boundary, while later wrapper failures are consequences.
type ExecutionCandidateFailureSlot = Arc<Mutex<Option<ExecutionCandidateFailure>>>;

#[derive(Clone)]
pub(crate) struct ExecutionCandidateFailureSink {
    targets: Arc<[ExecutionCandidateFailureSlot]>,
}

impl Default for ExecutionCandidateFailureSink {
    fn default() -> Self {
        Self {
            targets: Arc::from([Arc::new(Mutex::new(None))]),
        }
    }
}

impl fmt::Debug for ExecutionCandidateFailureSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionCandidateFailureSink")
            .field("failure", &self.failure())
            .field("target_count", &self.targets.len())
            .finish()
    }
}

impl ExecutionCandidateFailureSink {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn fanout<'a>(
        sinks: impl IntoIterator<Item = &'a ExecutionCandidateFailureSink>,
    ) -> Option<Self> {
        let mut targets = Vec::new();
        for sink in sinks {
            for target in sink.targets.iter() {
                if !targets.iter().any(|existing| Arc::ptr_eq(existing, target)) {
                    targets.push(Arc::clone(target));
                }
            }
        }
        (!targets.is_empty()).then(|| Self {
            targets: Arc::from(targets),
        })
    }

    pub(crate) fn record(&self, failure: ExecutionCandidateFailure) {
        for target in self.targets.iter() {
            let mut slot = target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(failure.clone());
            }
        }
    }

    pub(crate) fn failure(&self) -> Option<ExecutionCandidateFailure> {
        self.targets.iter().find_map(|target| {
            target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        })
    }
}

/// Installs one explicitly injected service root's identity on the current
/// thread while a family executor constructs or looks up runtime resources.
pub(crate) fn install_native_execution_scope(
    scope_id: NativeExecutionScopeId,
) -> NativeExecutionScopeGuard {
    let previous = CURRENT_EXECUTION_SCOPE_ID.with(|current| current.replace(Some(scope_id)));
    NativeExecutionScopeGuard { previous }
}

/// Identity visible to legacy runtime-cache key constructors on this thread.
/// `None` is reserved for tests and internal helpers invoked outside a native
/// request; production dispatch and streaming decode paths always install the
/// required request service before entering family code.
pub(crate) fn current_native_execution_scope_id() -> Option<NativeExecutionScopeId> {
    CURRENT_EXECUTION_SCOPE_ID.with(Cell::get)
}

/// Captures the request context for an explicitly spawned native worker.
/// `None` is reserved for low-level tests/internal helpers outside dispatch.
pub(crate) fn current_native_execution_context() -> Option<NativeExecutionContext> {
    let scope_id = current_native_execution_scope_id()?;
    let memory_broker = current_native_execution_memory_broker()?;
    Some(NativeExecutionContext {
        scope_id,
        memory_broker,
        backend_preference: request_backend_override(),
        placement: current_execution_placement(),
        failure_sink: current_execution_candidate_failure_sink(),
        cache_attempt_id: current_execution_cache_attempt_id(),
        execution_telemetry: current_execution_telemetry_collector(),
    })
}

/// Returns the broker injected by the active request without extending any
/// native allocation's lifetime. Allocation owners clone it only indirectly
/// through the committed reservation they retain.
pub(crate) fn current_native_execution_memory_broker() -> Option<Arc<DeviceMemoryBrokerSet>> {
    CURRENT_EXECUTION_MEMORY_BROKER.with(|current| current.borrow().clone())
}

pub(crate) fn current_execution_placement() -> Option<ExecutionPlacement> {
    CURRENT_EXECUTION_PLACEMENT.with(Cell::get)
}

pub(crate) fn current_execution_candidate_failure_sink() -> Option<ExecutionCandidateFailureSink> {
    CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK.with(|current| current.borrow().clone())
}

/// Reads the first typed failure recorded for the current candidate without
/// changing sink ownership or first-failure-wins semantics.
pub(crate) fn current_execution_candidate_failure() -> Option<ExecutionCandidateFailure> {
    current_execution_candidate_failure_sink().and_then(|sink| sink.failure())
}

pub(crate) fn current_execution_cache_attempt_id() -> Option<ExecutionCacheAttemptId> {
    CURRENT_EXECUTION_CACHE_ATTEMPT_ID.with(Cell::get)
}

/// Physical-memory reservations created while one transactional execution
/// candidate is active share its cohort. This lets nested host/native owners
/// enter the provisional domain gate held by their own candidate without
/// weakening exclusion between independent candidates.
pub(crate) fn current_memory_reservation_cohort_id() -> Option<MemoryReservationCohortId> {
    current_execution_cache_attempt_id().map(|attempt| MemoryReservationCohortId::new(attempt.0))
}

/// Resolves the complete resident-owner cache lane for the active request.
/// Production native entry points always install an Exact candidate (CPU is
/// represented by its resolved CPU route), so CUDA/HIP/Vulkan and individual
/// cards never collapse into the coarse `Gpu` enum variant.
///
/// Standalone low-level tests and legacy internal helpers may execute without
/// a policy attempt. For those callers we resolve the same live route that the
/// ggml backend selector uses. A backend that can actually initialize must be
/// present in that inventory; the synthetic final branch exists only for CPU
/// unit fixtures that intentionally run without a linked device registry.
pub(crate) fn current_execution_lane_key(backend: GgmlCpuGraphBackend) -> ExecutionLaneKey {
    let preference = request_backend_override();
    let route = match preference.as_ref() {
        Some(RequestBackendPreference::Exact(route)) => route.clone(),
        Some(RequestBackendPreference::CpuOnly) => ResolvedExecutionRoute::cpu(),
        Some(RequestBackendPreference::Accelerated) | None => {
            resolve_request_execution_route(preference.as_ref())
                .ok()
                .flatten()
                .unwrap_or_else(|| fallback_route_for_unscoped_backend(backend))
        }
    };
    let placement = current_execution_placement().unwrap_or(match backend {
        GgmlCpuGraphBackend::Cpu => ExecutionPlacement::CpuOnly,
        GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu => ExecutionPlacement::FullDevice,
    });
    ExecutionLaneKey {
        device: ResolvedDeviceKey {
            route: route.cache_key(),
        },
        placement,
        backend,
    }
}

fn fallback_route_for_unscoped_backend(backend: GgmlCpuGraphBackend) -> ResolvedExecutionRoute {
    match backend {
        GgmlCpuGraphBackend::Cpu => ResolvedExecutionRoute::cpu(),
        GgmlCpuGraphBackend::Metal => ResolvedExecutionRoute {
            provider: ExecutionProvider::Metal,
            stable_id: "Metal".to_string(),
            registry_ordinal: 0,
            kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
            addressability:
                crate::device::execution_route::DeviceAddressability::NotExactlyAddressable {
                    reason: "unscoped Metal test route",
                },
        },
        GgmlCpuGraphBackend::Gpu => ResolvedExecutionRoute {
            provider: ExecutionProvider::Unknown,
            stable_id: "unscoped-gpu-test-route".to_string(),
            registry_ordinal: 0,
            kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
            addressability:
                crate::device::execution_route::DeviceAddressability::NotExactlyAddressable {
                    reason: "unscoped generic-GPU test route",
                },
        },
    }
}

/// Defers publication of a newly constructed resident owner until the active
/// candidate attempt commits. Outside a candidate attempt (unit-level family
/// calls), publication remains immediate.
pub(crate) fn stage_execution_cache_commit(commit: impl FnOnce() + 'static) {
    let mut commit = Some(Box::new(commit) as DeferredCacheCommit);
    let staged = CURRENT_EXECUTION_CACHE_JOURNAL.with(|current| {
        let mut current = current.borrow_mut();
        let Some(journal) = current.as_mut() else {
            return false;
        };
        journal
            .commits
            .push(commit.take().expect("cache commit is staged once"));
        true
    });
    if !staged && current_execution_cache_attempt_id().is_none() {
        commit.expect("unstaged cache commit remains available")();
    }
    // An explicitly propagated worker can inherit the attempt id without
    // owning the parent thread's non-Send journal. In that case fail closed:
    // dropping the callback rolls the staged slot back instead of publishing
    // outside the transaction. Candidate construction normally occurs on the
    // journal-owning thread; worker graph execution should not materialize a
    // resident cache owner.
}

/// Registers cache invalidation that runs only if the active candidate rolls
/// back. This is used for already-published owners that participated in a
/// failed attempt: a placement violation is synthesized after the model call
/// returns, so model-local code cannot reliably observe the failure side
/// channel before returning.
pub(crate) fn stage_execution_cache_rollback(rollback: impl FnOnce() + 'static) {
    let mut rollback = Some(Box::new(rollback) as DeferredCacheRollback);
    CURRENT_EXECUTION_CACHE_JOURNAL.with(|current| {
        let mut current = current.borrow_mut();
        let Some(journal) = current.as_mut() else {
            return;
        };
        journal
            .rollbacks
            .push(rollback.take().expect("cache rollback is staged once"));
    });
}

/// Low-level memory/backend code calls this at the point where a typed
/// candidate-local failure is first known. A call outside a policy attempt is
/// intentionally a no-op, preserving standalone low-level tests.
pub(crate) fn record_current_execution_candidate_failure(failure: ExecutionCandidateFailure) {
    if let Some(sink) = current_execution_candidate_failure_sink() {
        sink.record(failure);
    }
}

/// Installs a previously captured request context on a worker thread.
pub(crate) fn install_native_execution_context(
    context: NativeExecutionContext,
) -> NativeExecutionContextGuard {
    let scope = install_native_execution_scope(context.scope_id);
    let previous_memory_broker = CURRENT_EXECUTION_MEMORY_BROKER
        .with(|current| current.replace(Some(context.memory_broker)));
    let previous_placement =
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.replace(context.placement));
    let previous_failure_sink = CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK
        .with(|current| current.replace(context.failure_sink));
    let previous_cache_attempt_id = CURRENT_EXECUTION_CACHE_ATTEMPT_ID
        .with(|current| current.replace(context.cache_attempt_id));
    let execution_telemetry = install_execution_telemetry_collector(context.execution_telemetry);
    let backend = install_request_backend_override(context.backend_preference);
    NativeExecutionContextGuard {
        scope,
        previous_memory_broker,
        previous_placement,
        previous_failure_sink,
        previous_cache_attempt_id,
        execution_telemetry,
        backend,
    }
}

/// Installs the context sourced from one explicitly injected service root.
pub(crate) fn install_native_execution_services(
    services: &NativeExecutionServices,
) -> NativeExecutionContextGuard {
    install_native_execution_context(NativeExecutionContext {
        scope_id: services.scope_id,
        memory_broker: Arc::clone(&services.memory_broker),
        // Preserve an enclosing policy attempt. Legacy direct callers have no
        // enclosing values and continue to install `None` for all three.
        backend_preference: request_backend_override(),
        placement: current_execution_placement(),
        failure_sink: current_execution_candidate_failure_sink(),
        cache_attempt_id: current_execution_cache_attempt_id(),
        execution_telemetry: current_execution_telemetry_collector(),
    })
}

/// Installs one transactional policy candidate around the complete family
/// dispatch. The returned guard also propagates through
/// [`current_native_execution_context`] into explicitly spawned native worker
/// threads (serve-batch included).
pub(crate) fn install_execution_candidate_attempt(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    failure_sink: ExecutionCandidateFailureSink,
) -> NativeExecutionContextGuard {
    let backend_preference = if candidate.placement == ExecutionPlacement::CpuOnly {
        Some(RequestBackendPreference::CpuOnly)
    } else {
        Some(RequestBackendPreference::Exact(
            candidate.device.route.clone(),
        ))
    };
    install_native_execution_context(NativeExecutionContext {
        scope_id: services.scope_id,
        memory_broker: Arc::clone(&services.memory_broker),
        backend_preference,
        placement: Some(candidate.placement),
        failure_sink: Some(failure_sink),
        cache_attempt_id: current_execution_cache_attempt_id(),
        execution_telemetry: current_execution_telemetry_collector(),
    })
}

/// Result of one complete candidate-local operation. The operation's ordinary
/// error remains opaque to policy; only the separately recorded typed failure
/// authorizes a caller to advance to another candidate.
pub(crate) struct ExecutionCandidateAttemptOutcome<T, E> {
    pub(crate) result: Result<T, E>,
    pub(crate) candidate_failure: Option<ExecutionCandidateFailure>,
}

/// Extracts an ordinary error from a failed candidate attempt while ensuring
/// a value returned alongside the typed failure is destroyed transactionally.
///
/// A successful value can own an exclusive actor checkout. Its `Drop` stages
/// an idle-cache return, so dropping it after the original attempt scope has
/// ended would accidentally publish a runtime whose placement/admission was
/// just rejected. The nested rollback journal keeps that destructor from
/// resurrecting candidate-local cache state.
pub(crate) fn execution_candidate_failure_source<T, E>(result: Result<T, E>) -> Option<E> {
    match result {
        Err(error) => Some(error),
        Ok(value) => {
            drop_execution_candidate_value_without_cache_publication(value);
            None
        }
    }
}

/// Destroys candidate-owned state inside a rollback-only cache journal.
///
/// Exclusive checkouts publish themselves back to their idle pool from
/// `Drop`. Candidate failure invalidates that owner, so every persistent
/// runtime/session wrapper must use this helper before discarding its active
/// lane.
pub(crate) fn drop_execution_candidate_value_without_cache_publication<T>(value: T) {
    let rollback = ExecutionCacheJournalScope::begin();
    drop(value);
    rollback.finish(false);
}

fn observed_backend_matches_provider(expected: ExecutionProvider, backend_name: &str) -> bool {
    let observed = ExecutionProvider::from_backend_name(backend_name);
    match expected {
        ExecutionProvider::Cpu
        | ExecutionProvider::Metal
        | ExecutionProvider::Cuda
        | ExecutionProvider::Hip
        | ExecutionProvider::Vulkan => observed == expected,
        // Generic/unknown routes cannot prove that compute stayed on the
        // selected physical accelerator. Fail closed instead of treating a
        // CPU BLAS label as sufficient evidence of GPU execution.
        ExecutionProvider::Accelerator | ExecutionProvider::Unknown => false,
    }
}

fn observed_placement_violation(
    candidate: &ExecutionCandidate,
    observed: &GgmlExecutionPlacementSummary,
) -> Option<ExecutionCandidateFailure> {
    if candidate.placement == ExecutionPlacement::CpuOnly {
        return None;
    }
    let graph_compute_calls = observed
        .direct_graph_computes
        .saturating_add(observed.scheduler_graph_computes);
    // Lazy streaming-session construction is allowed to defer proof until its
    // first warmed compute. Once ggml reports a compute call, however, an
    // empty node map is missing placement evidence and must fail closed.
    if graph_compute_calls == 0 {
        return None;
    }
    let expected = candidate.device.route.provider;
    let selected_nodes = observed
        .observed_compute_nodes_by_backend
        .iter()
        .filter(|(backend, _)| observed_backend_matches_provider(expected, backend))
        .map(|(_, nodes)| *nodes)
        .sum::<u64>();
    let mismatched = observed
        .observed_compute_nodes_by_backend
        .iter()
        .filter(|(backend, nodes)| {
            if **nodes == 0 || observed_backend_matches_provider(expected, backend) {
                return false;
            }
            candidate.placement == ExecutionPlacement::FullDevice
                || ExecutionProvider::from_backend_name(backend) != ExecutionProvider::Cpu
        })
        .map(|(backend, nodes)| format!("{backend}={nodes}"))
        .collect::<Vec<_>>();
    (selected_nodes == 0 || !mismatched.is_empty()).then(|| {
        let observation = if observed.observed_compute_nodes_by_backend.is_empty() {
            "no backend nodes".to_string()
        } else {
            observed
                .observed_compute_nodes_by_backend
                .iter()
                .filter(|(_, nodes)| **nodes > 0)
                .map(|(backend, nodes)| format!("{backend}={nodes}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        ExecutionCandidateFailure::placement(
            "execution-placement",
            format!(
                "selected provider {expected} with {:?} placement observed {observation}",
                candidate.placement
            ),
        )
    })
}

/// Runs a complete allocation/execution operation inside one candidate's
/// dynamic context and captures its typed failure side channel before the
/// context is restored. This is shared by offline dispatch and streaming
/// session construction/warm-up so both surfaces enforce identical retry
/// semantics.
pub(crate) fn run_execution_candidate_attempt<T, E>(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    operation: impl FnOnce() -> Result<T, E>,
) -> ExecutionCandidateAttemptOutcome<T, E> {
    let failure_sink = ExecutionCandidateFailureSink::new();
    let placement_collector = (candidate.placement != ExecutionPlacement::CpuOnly)
        .then(GgmlExecutionTelemetryCollector::new);
    let outer_collector = current_execution_telemetry_collector();
    let combined_collector = GgmlExecutionTelemetryCollector::fanout(
        outer_collector.iter().chain(placement_collector.iter()),
    );
    let (result, candidate_failure) = {
        let _telemetry = install_execution_telemetry_collector(combined_collector);
        let _attempt =
            install_execution_candidate_attempt(services, candidate, failure_sink.clone());
        let journal_scope = ExecutionCacheJournalScope::begin();
        let result = operation();
        let mut candidate_failure = failure_sink.failure();
        if result.is_ok() && candidate_failure.is_none() {
            candidate_failure = placement_collector.as_ref().and_then(|collector| {
                observed_placement_violation(candidate, &collector.snapshot())
            });
        }
        journal_scope.finish(result.is_ok() && candidate_failure.is_none());
        (result, candidate_failure)
    };
    ExecutionCandidateAttemptOutcome {
        result,
        candidate_failure,
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NativeExecutionServicesError {
    #[error("could not build builtin {dispatch_kind} execution dispatch: {reason}")]
    DispatchBuild {
        dispatch_kind: &'static str,
        reason: String,
    },
}

struct NativeExecutionDispatches {
    offline: GgmlAsrExecutionDispatch,
    streaming: GgmlAsrExecutionDispatch,
}

/// Process-owned native execution state.
///
/// There is deliberately no `Default` implementation. Public-library users
/// construct one root with [`Self::for_local_process`] and pass
/// `Arc::clone(&services)` to every native backend/session. Separate roots
/// still share the process ledger, so accidental host duplication cannot
/// defeat atomic memory admission.
pub struct NativeExecutionServices {
    scope_id: NativeExecutionScopeId,
    policy_resolver: Arc<dyn ExecutionPolicyResolver>,
    memory_broker: Arc<DeviceMemoryBrokerSet>,
    auxiliary_runtime_owners: super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache,
    hymt2_translation_actors:
        super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
            super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
            super::hymt2::Hymt2TranslationCandidate,
        >,
    firered_punc_actors: super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        super::firered_punc::runtime::FireRedPuncRuntime,
    >,
    diarizen_segmenter_actors:
        super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
            super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
            crate::diarize::segment::DiariZenRuntime,
        >,
    pyannote_segmenter_actors:
        super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
            super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
            crate::diarize::segment::PyannetGgmlRuntime,
        >,
    redimnet_runtime_actors:
        super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool<
            super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
            crate::diarize::embed::RedimNetResidentRuntime,
        >,
    firered_stream_vad_realtime_actors:
        super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool<
            super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
            crate::diarize::vad::FireRedRealtimeVadRuntime,
        >,
    dispatches: NativeExecutionDispatches,
}

impl NativeExecutionServices {
    pub fn for_local_process() -> Result<Self, NativeExecutionServicesError> {
        Self::new_with_broker(
            Arc::new(DefaultExecutionPolicyResolver),
            process_memory_broker(),
        )
    }

    /// Internal constructor for deterministic broker/policy tests. Production
    /// callers cannot replace the process ledger; doing so would make two
    /// service roots race the same physical memory independently.
    pub(crate) fn new_with_broker(
        policy_resolver: Arc<dyn ExecutionPolicyResolver>,
        memory_broker: Arc<DeviceMemoryBrokerSet>,
    ) -> Result<Self, NativeExecutionServicesError> {
        let executor_scope = BuiltinStatefulExecutorScope::new().map_err(|error| {
            NativeExecutionServicesError::DispatchBuild {
                dispatch_kind: "executor-scope",
                reason: error.to_string(),
            }
        })?;
        let offline = build_builtin_ggml_execution_dispatch(&executor_scope).map_err(|error| {
            NativeExecutionServicesError::DispatchBuild {
                dispatch_kind: "offline",
                reason: error.to_string(),
            }
        })?;
        let streaming =
            build_builtin_ggml_streaming_execution_dispatch(&executor_scope).map_err(|error| {
                NativeExecutionServicesError::DispatchBuild {
                    dispatch_kind: "streaming",
                    reason: error.to_string(),
                }
            })?;

        Ok(Self {
            scope_id: NativeExecutionScopeId::next(),
            policy_resolver,
            memory_broker,
            auxiliary_runtime_owners:
                super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache::default(),
            hymt2_translation_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool::new(
                    "openasr-hymt2-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPoolLimits::new(
                        4,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    ),
                ),
            firered_punc_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool::new(
                    "openasr-firered-punc-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPoolLimits::new(
                        4,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    ),
                ),
            diarizen_segmenter_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool::new(
                    "openasr-diarizen-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPoolLimits::new(
                        4,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    ),
                ),
            pyannote_segmenter_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool::new(
                    "openasr-pyannote-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPoolLimits::new(
                        4,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                    ),
                ),
            redimnet_runtime_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool::new(
                    "openasr-redimnet-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                        crate::diarize::embed::REDIMNET_MAX_BATCH_WORKERS,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                        crate::diarize::embed::REDIMNET_MAX_BATCH_WORKERS,
                    ),
                ),
            firered_stream_vad_realtime_actors:
                super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool::new(
                    "openasr-firered-vad-realtime-owner",
                    super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(
                        1,
                        crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
                        4,
                    ),
                ),
            dispatches: NativeExecutionDispatches { offline, streaming },
        })
    }

    pub fn scope_id(&self) -> NativeExecutionScopeId {
        self.scope_id
    }

    pub fn policy_resolver(&self) -> &Arc<dyn ExecutionPolicyResolver> {
        &self.policy_resolver
    }

    pub fn memory_broker(&self) -> &Arc<DeviceMemoryBrokerSet> {
        &self.memory_broker
    }

    pub(crate) fn auxiliary_runtime_owners(
        &self,
    ) -> &super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache {
        &self.auxiliary_runtime_owners
    }

    pub(crate) fn hymt2_translation_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        super::hymt2::Hymt2TranslationCandidate,
    > {
        &self.hymt2_translation_actors
    }

    pub(crate) fn firered_punc_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        super::firered_punc::runtime::FireRedPuncRuntime,
    > {
        &self.firered_punc_actors
    }

    pub(crate) fn diarizen_segmenter_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        crate::diarize::segment::DiariZenRuntime,
    > {
        &self.diarizen_segmenter_actors
    }

    pub(crate) fn pyannote_segmenter_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        crate::diarize::segment::PyannetGgmlRuntime,
    > {
        &self.pyannote_segmenter_actors
    }

    pub(crate) fn redimnet_runtime_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        crate::diarize::embed::RedimNetResidentRuntime,
    > {
        &self.redimnet_runtime_actors
    }

    pub(crate) fn firered_stream_vad_realtime_actors(
        &self,
    ) -> &super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorCheckoutPool<
        super::policy_resolved_aux_runtime::AuxiliaryPinnedRuntimeCacheKey,
        crate::diarize::vad::FireRedRealtimeVadRuntime,
    > {
        &self.firered_stream_vad_realtime_actors
    }

    pub(crate) fn offline_dispatch(&self) -> &GgmlAsrExecutionDispatch {
        &self.dispatches.offline
    }

    pub(crate) fn streaming_dispatch(&self) -> &GgmlAsrExecutionDispatch {
        &self.dispatches.streaming
    }

    /// Evicts model-resident state owned by this service root.
    pub fn unload_idle_native_model_runtime_caches(&self) {
        let _execution_scope = install_native_execution_services(self);
        self.dispatches.offline.unload_all();
        self.dispatches.streaming.unload_all();
        self.auxiliary_runtime_owners.clear();
        self.hymt2_translation_actors.clear();
        self.firered_punc_actors.clear();
        self.diarizen_segmenter_actors.clear();
        self.pyannote_segmenter_actors.clear();
        self.redimnet_runtime_actors.clear();
        self.firered_stream_vad_realtime_actors.clear();
    }

    /// Evicts one replaced pack identity from this root's prepared-runtime
    /// caches. Pull/install callers must pass the service root explicitly.
    pub fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        let _execution_scope = install_native_execution_services(self);
        self.dispatches
            .offline
            .evict_prepared_runtime_content_id(pack_content_id);
        self.auxiliary_runtime_owners
            .evict_content_id(pack_content_id);
        self.hymt2_translation_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.firered_punc_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.diarizen_segmenter_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.pyannote_segmenter_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.redimnet_runtime_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
    }
}

impl fmt::Debug for NativeExecutionServices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeExecutionServices")
            .field("scope_id", &self.scope_id)
            .field("policy_resolver", &"dyn ExecutionPolicyResolver")
            .field("memory_broker", &self.memory_broker)
            .finish_non_exhaustive()
    }
}

impl PartialEq for NativeExecutionServices {
    fn eq(&self, other: &Self) -> bool {
        self.scope_id == other.scope_id
    }
}

impl Eq for NativeExecutionServices {}

#[cfg(test)]
pub(crate) fn test_native_execution_services() -> Arc<NativeExecutionServices> {
    Arc::new(
        NativeExecutionServices::new_with_broker(
            Arc::new(DefaultExecutionPolicyResolver),
            Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default())),
        )
        .expect("builtin native execution services must construct for tests"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::device::{
        execution_policy::ExecutionDeviceSnapshot,
        execution_route::{
            DeviceAddressability, ExecutionProvider, PhysicalResourceKey, ResolvedExecutionRoute,
            RouteDeviceKind,
        },
    };
    use crate::ggml_runtime::{
        GgmlBackendKind, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
    };

    fn cpu_candidate() -> ExecutionCandidate {
        ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider: ExecutionProvider::Cpu,
                    stable_id: "CPU".to_string(),
                    registry_ordinal: 0,
                    kind: RouteDeviceKind::Cpu,
                    addressability: DeviceAddressability::NotExactlyAddressable {
                        reason: "test CPU",
                    },
                },
                ggml_kind: GgmlBackendKind::Cpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: ExecutionPlacement::CpuOnly,
        }
    }

    fn gpu_candidate(
        provider: ExecutionProvider,
        stable_id: &str,
        physical_id: &str,
        placement: ExecutionPlacement,
    ) -> ExecutionCandidate {
        ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider,
                    stable_id: stable_id.to_string(),
                    registry_ordinal: 0,
                    kind: RouteDeviceKind::Accelerated,
                    addressability: DeviceAddressability::ExactlyAddressable {
                        physical_key: PhysicalResourceKey::new(physical_id).unwrap(),
                    },
                },
                ggml_kind: GgmlBackendKind::Gpu,
                memory: None,
                buffer_alignment: None,
            },
            placement,
        }
    }

    fn record_test_graph_placements(backends: &[&str]) {
        let collector = current_execution_telemetry_collector().expect("candidate collector");
        collector.record_graph_compute(false);
        let observed = backends
            .iter()
            .map(|backend| ((*backend).to_string(), (3, 96)))
            .collect::<std::collections::BTreeMap<_, _>>();
        let compute = backends
            .iter()
            .map(|backend| ((*backend).to_string(), 2))
            .collect::<std::collections::BTreeMap<_, _>>();
        collector.record_observed_graph(7, &observed, &compute, &std::collections::BTreeMap::new());
    }

    fn record_test_graph_placement(backend: &str) {
        record_test_graph_placements(&[backend]);
    }

    #[test]
    fn accelerated_candidate_fails_closed_on_observed_cpu_compute() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::Hybrid,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            record_test_graph_placement("CPU/BLAS");
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        let failure = outcome
            .candidate_failure
            .expect("CPU graph under Metal candidate must fail closed");
        assert_eq!(
            failure.kind,
            crate::device::execution_policy::ExecutionCandidateFailureKind::PlacementViolation
        );
        assert_eq!(failure.operation, "execution-placement");
    }

    #[test]
    fn accelerated_candidate_accepts_compute_on_selected_provider() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::Hybrid,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            record_test_graph_placement("MTL0");
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
    }

    #[test]
    fn hybrid_candidate_accepts_cpu_and_selected_device_compute() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::Hybrid,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            record_test_graph_placements(&["CPU/BLAS", "MTL0"]);
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
    }

    #[test]
    fn full_device_candidate_rejects_any_cpu_compute() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::FullDevice,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            record_test_graph_placements(&["CPU/BLAS", "MTL0"]);
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert_eq!(
            outcome.candidate_failure.unwrap().kind,
            crate::device::execution_policy::ExecutionCandidateFailureKind::PlacementViolation
        );
    }

    #[test]
    fn full_device_candidate_rejects_cpu_graph_before_backend_construction() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::FullDevice,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default()).map(|_| ())
        });
        assert!(matches!(
            outcome.result,
            Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "FullDevice execution requires a GPU-class graph backend",
            })
        ));
    }

    #[test]
    fn accelerated_candidate_rejects_compute_without_backend_node_evidence() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Metal,
            "MTL0",
            "0000:00:02.0",
            ExecutionPlacement::FullDevice,
        );
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            current_execution_telemetry_collector()
                .expect("candidate collector")
                .record_graph_compute(false);
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert_eq!(
            outcome.candidate_failure.unwrap().kind,
            crate::device::execution_policy::ExecutionCandidateFailureKind::PlacementViolation
        );
    }

    #[test]
    fn candidate_context_propagates_request_local_sink_to_worker() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let context = current_native_execution_context().expect("candidate context");
            std::thread::spawn(move || {
                let _guard = install_native_execution_context(context);
                record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
                    "worker-allocation",
                    "worker request-local failure",
                ));
            })
            .join()
            .unwrap();
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert_eq!(
            outcome.candidate_failure.unwrap().operation,
            "worker-allocation"
        );
    }

    #[test]
    fn nested_workers_share_one_memory_reservation_cohort_per_candidate_attempt() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let outer = current_memory_reservation_cohort_id().expect("candidate cohort");
            let context = current_native_execution_context().expect("candidate context");
            let worker = std::thread::spawn(move || {
                let _guard = install_native_execution_context(context);
                current_memory_reservation_cohort_id().expect("worker cohort")
            })
            .join()
            .unwrap();
            assert_eq!(outer, worker);
            let nested = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
                Ok::<_, ()>(
                    current_memory_reservation_cohort_id().expect("nested candidate cohort"),
                )
            });
            assert_eq!(outer, nested.result.unwrap());
            Ok::<_, ()>(outer)
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
        let outer = outcome.result.unwrap();

        let next = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            Ok::<_, ()>(current_memory_reservation_cohort_id().expect("next cohort"))
        });
        assert!(next.result.is_ok());
        // A completed attempt must never reopen the previous cohort gate.
        assert_ne!(outer, next.result.unwrap());
    }

    #[test]
    fn independently_constructed_local_roots_share_the_process_memory_ledger() {
        let first = NativeExecutionServices::for_local_process().unwrap();
        let second = NativeExecutionServices::for_local_process().unwrap();
        assert_ne!(first.scope_id(), second.scope_id());
        assert!(Arc::ptr_eq(first.memory_broker(), second.memory_broker()));
    }

    #[test]
    fn shared_lane_rejects_a_different_service_root() {
        let first_services = test_native_execution_services();
        let second_services = test_native_execution_services();
        let candidate = cpu_candidate();
        let capture = |services: &NativeExecutionServices| {
            let sink = ExecutionCandidateFailureSink::new();
            {
                let _guard =
                    install_execution_candidate_attempt(services, &candidate, sink.clone());
                current_native_execution_context().unwrap()
            }
        };
        let first = capture(first_services.as_ref());
        let second = capture(second_services.as_ref());

        assert!(matches!(
            NativeExecutionContext::shared_lane(&[first, second]),
            Err(NativeExecutionContextError::IncompatibleSharedLane { index: 1 })
        ));
    }

    #[test]
    fn shared_lane_tls_record_fans_out_to_every_active_request() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let capture = || {
            let sink = ExecutionCandidateFailureSink::new();
            let context = {
                let _guard = install_execution_candidate_attempt(
                    services.as_ref(),
                    &candidate,
                    sink.clone(),
                );
                current_native_execution_context().unwrap()
            };
            (context, sink)
        };
        let (first_context, first_sink) = capture();
        let (second_context, second_sink) = capture();
        let (third_context, third_sink) = capture();
        let shared =
            NativeExecutionContext::shared_lane(&[first_context, second_context, third_context])
                .unwrap()
                .unwrap();

        {
            let _guard = install_native_execution_context(shared);
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "shared-graph",
                "shared device failure",
            ));
        }

        assert_eq!(first_sink.failure().unwrap().operation, "shared-graph");
        assert_eq!(second_sink.failure().unwrap().operation, "shared-graph");
        assert_eq!(third_sink.failure().unwrap().operation, "shared-graph");
    }

    #[test]
    fn a_request_that_left_the_lane_is_not_polluted_by_a_later_failure() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let capture = || {
            let sink = ExecutionCandidateFailureSink::new();
            let context = {
                let _guard = install_execution_candidate_attempt(
                    services.as_ref(),
                    &candidate,
                    sink.clone(),
                );
                current_native_execution_context().unwrap()
            };
            (context, sink)
        };
        let (completed_context, completed_sink) = capture();
        let (active_context, active_sink) = capture();
        let (refill_context, refill_sink) = capture();
        let initial =
            NativeExecutionContext::shared_lane(&[completed_context, active_context.clone()])
                .unwrap()
                .unwrap();
        drop(initial);
        let current = NativeExecutionContext::shared_lane(&[active_context, refill_context])
            .unwrap()
            .unwrap();

        {
            let _guard = install_native_execution_context(current);
            record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
                "late-shared-graph",
                "failure after the first request completed",
            ));
        }

        assert!(completed_sink.failure().is_none());
        assert_eq!(
            active_sink.failure().unwrap().operation,
            "late-shared-graph"
        );
        assert_eq!(
            refill_sink.failure().unwrap().operation,
            "late-shared-graph"
        );
    }

    #[test]
    fn candidate_failure_sink_preserves_first_causal_failure() {
        let sink = ExecutionCandidateFailureSink::new();
        sink.record(ExecutionCandidateFailure::capacity("quote", "first"));
        sink.record(ExecutionCandidateFailure::device_lost("compute", "second"));
        let failure = sink.failure().unwrap();
        assert_eq!(failure.operation, "quote");
        assert_eq!(failure.detail, "first");
    }

    #[test]
    fn execution_lane_key_separates_provider_card_and_placement() {
        let services = test_native_execution_services();
        let cuda0 = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::FullDevice,
        );
        let cuda1 = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA1",
            "0000:02:00.0",
            ExecutionPlacement::FullDevice,
        );
        let hip0 = gpu_candidate(
            ExecutionProvider::Hip,
            "HIP0",
            "0000:01:00.0",
            ExecutionPlacement::FullDevice,
        );
        let hybrid = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::Hybrid,
        );
        let lane_for = |candidate: &ExecutionCandidate| {
            let sink = ExecutionCandidateFailureSink::new();
            let _guard = install_execution_candidate_attempt(services.as_ref(), candidate, sink);
            current_execution_lane_key(GgmlCpuGraphBackend::Gpu)
        };
        let cuda0_lane = lane_for(&cuda0);
        assert_ne!(cuda0_lane, lane_for(&cuda1));
        assert_ne!(cuda0_lane, lane_for(&hip0));
        assert_ne!(cuda0_lane, lane_for(&hybrid));
        assert_eq!(cuda0_lane.backend(), GgmlCpuGraphBackend::Gpu);
        assert_eq!(cuda0_lane.placement(), ExecutionPlacement::FullDevice);
    }

    #[test]
    fn candidate_cache_journal_publishes_only_clean_success() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let published = Arc::new(Mutex::new(Vec::new()));

        let clean_target = Arc::clone(&published);
        let clean = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_commit(move || clean_target.lock().unwrap().push("clean"));
            Ok::<_, ()>(())
        });
        assert!(clean.result.is_ok());
        assert!(clean.candidate_failure.is_none());
        assert_eq!(*published.lock().unwrap(), vec!["clean"]);

        let error_target = Arc::clone(&published);
        let ordinary_error = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_commit(move || error_target.lock().unwrap().push("error"));
            Err::<(), _>(())
        });
        assert!(ordinary_error.result.is_err());
        assert_eq!(*published.lock().unwrap(), vec!["clean"]);

        let typed_target = Arc::clone(&published);
        let typed_success = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_commit(move || typed_target.lock().unwrap().push("typed"));
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "test-owner",
                "device disappeared after construction",
            ));
            Ok::<_, ()>(())
        });
        assert!(typed_success.result.is_ok());
        assert!(typed_success.candidate_failure.is_some());
        assert_eq!(*published.lock().unwrap(), vec!["clean"]);

        let rollback_target = Arc::clone(&published);
        let rollback = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_rollback(move || {
                rollback_target.lock().unwrap().push("rolled-back")
            });
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "test-rollback-action",
                "invalidate an already-published owner",
            ));
            Ok::<_, ()>(())
        });
        assert!(rollback.candidate_failure.is_some());
        assert_eq!(*published.lock().unwrap(), vec!["clean", "rolled-back"]);

        let discarded_target = Arc::clone(&published);
        let success = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_rollback(move || {
                discarded_target.lock().unwrap().push("must-not-run")
            });
            Ok::<_, ()>(())
        });
        assert!(success.candidate_failure.is_none());
        assert_eq!(*published.lock().unwrap(), vec!["clean", "rolled-back"]);
    }

    #[test]
    fn failed_candidate_evicts_the_exact_published_pinned_actor() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let pool = super::super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPool::new(
            "candidate-rollback-pinned-actor-test",
            super::super::admitted_pinned_runtime_actor_pool::AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let get_actor = || {
            pool.get_or_try_insert_with(
                "same",
                || Ok::<_, String>((16, ())),
                {
                    let builds = Arc::clone(&builds);
                    move |()| {
                        let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                        Ok(super::super::system_memory_owner::SystemMemoryOwner::with_committed_requested_bytes_for_test(value, 16))
                    }
                },
                |error| error.to_string(),
            )
        };

        let first = run_execution_candidate_attempt(services.as_ref(), &candidate, get_actor);
        assert!(first.candidate_failure.is_none());
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        let persistent_actor = first.result.expect("first actor");

        let failed = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let value = persistent_actor
                .call_mut(|runtime| *runtime)
                .map_err(|error| error.to_string())?;
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "candidate-rollback-pinned-actor-test",
                "invalidate the published owner",
            ));
            Ok::<_, String>(value)
        });
        assert!(failed.candidate_failure.is_some());
        drop(failed.result);

        let rebuilt = run_execution_candidate_attempt(services.as_ref(), &candidate, get_actor);
        assert!(rebuilt.candidate_failure.is_none());
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        drop(persistent_actor);
    }

    #[test]
    fn value_returned_with_candidate_failure_cannot_publish_from_drop() {
        struct PublishesOnDrop(Arc<Mutex<Vec<&'static str>>>);

        impl Drop for PublishesOnDrop {
            fn drop(&mut self) {
                let target = Arc::clone(&self.0);
                stage_execution_cache_commit(move || target.lock().unwrap().push("resurrected"));
            }
        }

        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let published = Arc::new(Mutex::new(Vec::new()));
        let value_target = Arc::clone(&published);
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "test-owner",
                "placement rejected after the owner was returned",
            ));
            Ok::<_, ()>(PublishesOnDrop(value_target))
        });

        assert!(outcome.candidate_failure.is_some());
        assert!(execution_candidate_failure_source(outcome.result).is_none());
        assert!(published.lock().unwrap().is_empty());
    }

    #[test]
    fn candidate_cache_journal_rolls_back_on_unwind_and_restores_the_next_scope() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let published = Arc::new(Mutex::new(Vec::new()));

        let panic_target = Arc::clone(&published);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: ExecutionCandidateAttemptOutcome<(), ()> =
                run_execution_candidate_attempt(services.as_ref(), &candidate, || {
                    stage_execution_cache_commit(move || {
                        panic_target.lock().unwrap().push("panicked")
                    });
                    panic!("candidate construction panic");
                });
        }));
        assert!(panicked.is_err());
        assert!(published.lock().unwrap().is_empty());

        let clean_target = Arc::clone(&published);
        let clean = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            stage_execution_cache_commit(move || clean_target.lock().unwrap().push("next"));
            Ok::<_, ()>(())
        });
        assert!(clean.result.is_ok());
        assert!(clean.candidate_failure.is_none());
        assert_eq!(*published.lock().unwrap(), vec!["next"]);
    }
}
