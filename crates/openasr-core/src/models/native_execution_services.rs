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
    collections::BTreeMap,
    ffi::{CStr, c_char, c_void},
    fmt,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use thiserror::Error;

use crate::device::{
    execution_memory::{
        AllocationLifetime, DeviceMemoryBrokerSet, DeviceMemoryPolicy,
        DeviceMemoryReservationBatch, MemoryReservationCohortId, PhaseSet, PhysicalDeviceKey,
    },
    execution_policy::{
        DefaultExecutionPolicyResolver, ExecutionCandidate, ExecutionCandidateFailure,
        ExecutionIntent, ExecutionPlacement, ExecutionPolicyResolver,
    },
    execution_route::{
        ExecutionProvider, ExecutionRouteCacheKey, ResolvedExecutionRoute,
        enumerate_compute_devices_from_ggml,
    },
};
use crate::ggml_runtime::backend_memory::BackendMemoryAbi;
use crate::ggml_runtime::backend_memory_admission::{
    NativeMemoryAdmissionPlan, NativeMemoryClaimSemantics, NativeQuotedBackendGroup,
};
use crate::ggml_runtime::ffi;
use crate::ggml_runtime::{
    BackendMemoryBytes, BackendMemoryLifecyclePoint, BackendMemoryStatsSnapshot,
    BackendMemoryUnknownReason, SafeBackendMemoryReceipt,
};
use crate::ggml_runtime::{
    GgmlBackend, GgmlBackendDevice, GgmlBackendKind, GgmlCpuGraphBackend, GgmlDeviceMemory,
    GgmlExecutionPlacementSummary, GgmlExecutionTelemetryCollector, GgmlExecutionTelemetryGuard,
    GgmlGraphLifecycleGuard, RequestBackendOverrideGuard, RequestBackendPreference,
    current_execution_telemetry_collector, ensure_backends_loaded, ggml_available_devices,
    install_execution_telemetry_collector, install_graph_lifecycle_collector,
    install_request_backend_override, request_backend_override, resolve_request_execution_route,
};
use crate::models::pack_verifier::{PackRoute, VerifiedPack};

use super::{
    builtin_execution_dispatch::{
        build_builtin_ggml_execution_dispatch, build_builtin_ggml_streaming_execution_dispatch,
    },
    candidate_activation_transaction::{
        ActivationReservation, ActivationStage, AttestationFailure, AttestationOutcome,
        DefaultModelActivationFacts, DefaultModelActivationIdentity, DefaultModelActivationLane,
        DefaultModelActivationPlan, DefaultModelResidentComponentPlan,
        DefaultModelResidentTopologyPlan, ExecutionCandidateAttemptEvidence,
        ExecutionCandidateAttemptJournalFactory, ExecutionCandidateAttemptOwner,
        NativeCandidateAttemptFacts, ResolvedExecutionFacts, TypedAttestation,
    },
    executor_component_registry::BuiltinStatefulExecutorScope,
    ggml_asr_executor::GgmlAsrExecutionDispatch,
    request_execution_receipt::NativeExecutionReceiptCollector,
    runtime_receipts::{
        RuntimeOwnerPlacement, RuntimeReceiptCollector, SafeExecutionLaneProjection,
    },
    system_memory_owner::SystemMemoryAllocationQuote,
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
    /// Explicitly propagated diagnostic collector for one service root. It is
    /// never consulted by admission or fallback.
    static CURRENT_EXECUTION_RECEIPTS: RefCell<Option<RuntimeReceiptCollector>> = const {
        RefCell::new(None)
    };
    /// Placement selected by the active policy candidate. Graph-runtime
    /// configuration consumes this value to make `FullDevice` and `Hybrid`
    /// executable contracts rather than diagnostic labels.
    static CURRENT_EXECUTION_PLACEMENT: Cell<Option<ExecutionPlacement>> = const {
        Cell::new(None)
    };
    /// Exact candidate lane selected before backend initialization. Native
    /// admission and receipt adapters consume this typed value instead of
    /// reconstructing provider/device identity from backend labels.
    static CURRENT_EXECUTION_LANE: RefCell<Option<ExecutionLaneKey>> = const {
        RefCell::new(None)
    };
    /// Optional request-scoped observation channel used by exact native
    /// runtime audits. It is inert unless an explicit caller installs it, and
    /// rides the same context propagation path as the route and placement so
    /// worker-owned graphs cannot turn an Exact request into an unobservable
    /// one.
    static CURRENT_EXECUTION_OBSERVATION_SINK:
        RefCell<Option<ExecutionObservationSink>> = const { RefCell::new(None) };
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
    /// Explicit outer activation cohort shared by every nested reservation
    /// during staged materialization. It takes precedence over the inner cache
    /// journal attempt id and propagates through worker contexts.
    static CURRENT_ACTIVATION_RESERVATION_COHORT: Cell<Option<MemoryReservationCohortId>> = const {
        Cell::new(None)
    };
    /// Strict receipt collection is opt-in and follows the concrete request
    /// context through policy candidates and worker contexts. It is never
    /// reconstructed from process state after native execution.
    static CURRENT_EXECUTION_RECEIPT:
        RefCell<Option<NativeExecutionReceiptCollector>> = const { RefCell::new(None) };
    /// Scope-local slot for the admitted embedded Stream-VAD weights. This is
    /// not a process-global model cache: it is installed from one
    /// [`NativeExecutionServices`] root and restored when that context exits.
    static CURRENT_STREAM_VAD_EMBEDDED: RefCell<Option<crate::diarize::vad::StreamVadEmbeddedSlot>> =
        const { RefCell::new(None) };
    /// NES-owned loaded-weight owner cache. Production publication of
    /// `GgmlLoadedWeightContext` goes through this handle, not a process-global
    /// TLS owner map.
    static CURRENT_LOADED_WEIGHT_OWNERS: RefCell<Option<crate::ggml_runtime::LoadedWeightOwnerCache>> =
        const { RefCell::new(None) };
    /// Quote identity for the current candidate attempt. Nested auxiliary
    /// attempts must install their own pack or declared resident bytes; they
    /// must not inherit an outer ASR mapping or another family's blob.
    static CURRENT_CANDIDATE_ACTIVATION_QUOTE: RefCell<Option<CandidateActivationQuoteSource>> =
        const { RefCell::new(None) };
}

/// A backend that was actually constructed for one policy-scoped graph.
///
/// `requested_route` and `placement` are the policy facts in force while the
/// runner was built; `backend_kind` and `backend_name` come from the live ggml
/// backend handle after initialization. This is deliberately diagnostic data:
/// graph policy continues to use typed capability data rather than parsing a
/// provider label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionBackendObservation {
    pub(crate) requested_route: ResolvedExecutionRoute,
    pub(crate) placement: ExecutionPlacement,
    pub(crate) backend_kind: GgmlCpuGraphBackend,
    pub(crate) backend_name: String,
    pub(crate) actual_provider: ExecutionProvider,
    pub(crate) actual_stable_id: String,
    pub(crate) actual_device: crate::GgmlActualDeviceFacts,
    pub(crate) use_scheduler: bool,
    /// In-process join key only. It is never emitted by the smoke receipt.
    backend_identity: usize,
    pub(crate) memory_receipts: Vec<SafeBackendMemoryReceipt>,
}

/// Explicitly installed sink for exact-route audit evidence. Normal production
/// execution leaves this absent, so collecting observations never changes the
/// default runtime path or retains per-request graph state.
#[derive(Clone, Default)]
pub(crate) struct ExecutionObservationSink {
    observations: Arc<Mutex<Vec<ExecutionBackendObservation>>>,
}

#[allow(dead_code)] // Constructed by ignored host-local true-pack tests only.
impl ExecutionObservationSink {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn observations(&self) -> Vec<ExecutionBackendObservation> {
        self.observations
            .lock()
            .expect("execution observation sink lock must not be poisoned")
            .clone()
    }

    fn record(&self, observation: ExecutionBackendObservation) {
        self.observations
            .lock()
            .expect("execution observation sink lock must not be poisoned")
            .push(observation);
    }

    fn append(&self, observations: impl IntoIterator<Item = ExecutionBackendObservation>) {
        self.observations
            .lock()
            .expect("execution observation sink lock must not be poisoned")
            .extend(observations);
    }

    fn record_memory(&self, backend_identity: usize, mut receipts: Vec<SafeBackendMemoryReceipt>) {
        let mut observations = self
            .observations
            .lock()
            .expect("execution observation sink lock must not be poisoned");
        let Some(observation) = observations
            .iter_mut()
            .rev()
            .find(|observation| observation.backend_identity == backend_identity)
        else {
            return;
        };
        for receipt in &mut receipts {
            let prior = observation
                .memory_receipts
                .iter()
                .filter(|prior| {
                    prior.domain_kind == receipt.domain_kind
                        && prior.heap_index == receipt.heap_index
                })
                .filter_map(
                    |prior| match prior.backend_owned_observed_high_water_bytes {
                        BackendMemoryBytes::Known(bytes) => Some(bytes),
                        BackendMemoryBytes::Unknown(_) => None,
                    },
                )
                .max();
            if let (Some(prior), BackendMemoryBytes::Known(current)) =
                (prior, receipt.backend_owned_observed_high_water_bytes)
            {
                receipt.backend_owned_observed_high_water_bytes =
                    BackendMemoryBytes::Known(prior.max(current));
            }
        }
        for receipt in receipts {
            if let Some(existing) = observation.memory_receipts.iter_mut().find(|existing| {
                existing.lifecycle == receipt.lifecycle
                    && existing.domain_kind == receipt.domain_kind
                    && existing.heap_index == receipt.heap_index
            }) {
                *existing = receipt;
            } else {
                observation.memory_receipts.push(receipt);
            }
        }
    }
}

/// Installs an observation sink around one native request. Callers must retain
/// the returned guard for the complete request so candidate attempts and any
/// worker contexts inherit the same sink.
#[allow(dead_code)] // Installed by ignored host-local true-pack tests only.
pub(crate) fn install_execution_observation_sink(
    sink: ExecutionObservationSink,
) -> ExecutionObservationSinkGuard {
    let previous = CURRENT_EXECUTION_OBSERVATION_SINK.with(|current| current.replace(Some(sink)));
    ExecutionObservationSinkGuard { previous }
}

#[allow(dead_code)] // Returned by the ignored host-local true-pack test seam.
pub(crate) struct ExecutionObservationSinkGuard {
    previous: Option<ExecutionObservationSink>,
}

impl Drop for ExecutionObservationSinkGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_OBSERVATION_SINK.with(|current| {
            *current.borrow_mut() = self.previous.take();
        });
    }
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
#[derive(Debug, Clone)]
pub(crate) struct ResolvedDeviceKey {
    route: ExecutionRouteCacheKey,
    resolved_route: ResolvedExecutionRoute,
}

impl ResolvedDeviceKey {
    fn new(resolved_route: ResolvedExecutionRoute) -> Self {
        Self {
            route: resolved_route.cache_key(),
            resolved_route,
        }
    }
}

// Runtime isolation deliberately ignores registry ordinal; the full route is
// retained only so an immutable lane can reinstall exact backend selection on
// another worker without enumerating devices again.
impl PartialEq for ResolvedDeviceKey {
    fn eq(&self, other: &Self) -> bool {
        self.route == other.route
    }
}

impl Eq for ResolvedDeviceKey {}

impl Hash for ResolvedDeviceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.route.hash(state);
    }
}

/// Cache key shared by every resident object that owns a ggml backend, device
/// buffer, scheduler, graph, or uploaded tensor arena.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionLaneKey {
    device: ResolvedDeviceKey,
    placement: ExecutionPlacement,
    backend: GgmlCpuGraphBackend,
}

/// Live memory observation joined to the same provider + stable device id as
/// an [`ExecutionLaneKey`]. Consumers may project this fact into telemetry or
/// batching policy, but must not repeat device-enumeration identity joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExecutionLaneMemorySample {
    pub(crate) provider: ExecutionProvider,
    pub(crate) stable_device_id: String,
    pub(crate) device_kind: GgmlBackendKind,
    pub(crate) memory: GgmlDeviceMemory,
}

impl ExecutionLaneKey {
    #[allow(dead_code)]
    pub(crate) fn receipt_projection(
        &self,
        collector: &RuntimeReceiptCollector,
    ) -> Option<SafeExecutionLaneProjection> {
        collector.lane_projection(
            self.device.route.provider,
            &self.device.route.stable_id,
            self.placement,
            self.backend,
        )
    }

    pub(crate) fn backend(&self) -> GgmlCpuGraphBackend {
        self.backend
    }

    pub(crate) fn provider(&self) -> ExecutionProvider {
        self.device.route.provider
    }

    pub(crate) fn stable_device_id(&self) -> &str {
        &self.device.route.stable_id
    }

    pub(crate) fn placement(&self) -> ExecutionPlacement {
        self.placement
    }

    pub(crate) fn request_backend_preference(&self) -> RequestBackendPreference {
        match self.provider() {
            ExecutionProvider::Cpu => RequestBackendPreference::CpuOnly,
            ExecutionProvider::Metal
            | ExecutionProvider::Cuda
            | ExecutionProvider::Hip
            | ExecutionProvider::Vulkan => {
                RequestBackendPreference::Exact(self.device.resolved_route.clone())
            }
            // These providers cannot occur in a production candidate lane;
            // retain the coarse fallback only for legacy low-level fixtures.
            ExecutionProvider::Accelerator | ExecutionProvider::Unknown => {
                RequestBackendPreference::Accelerated
            }
        }
    }

    /// Observe memory for this exact lane. Enumeration order never selects a
    /// different device: both provider and provider-local stable id must
    /// match, and a missing observation remains unknown.
    pub(crate) fn live_memory_sample(&self) -> Option<ExecutionLaneMemorySample> {
        let devices = ggml_available_devices()
            .into_iter()
            .map(|device| (device.name, device.kind, device.memory))
            .collect::<Vec<_>>();
        exact_lane_memory_sample_from_device_infos(self, &devices)
    }

    /// Derive a stage-specific lane from the request's already-resolved exact
    /// device. This never consults backend preference or route discovery.
    pub(crate) fn for_stage(
        &self,
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Self {
        let device = if matches!(backend, GgmlCpuGraphBackend::Cpu)
            && matches!(placement, ExecutionPlacement::CpuOnly)
        {
            ResolvedDeviceKey::new(ResolvedExecutionRoute::cpu())
        } else {
            self.device.clone()
        };
        Self {
            device,
            placement,
            backend,
        }
    }

    /// Construct an exact resident-owner lane from the immutable policy candidate.
    /// Backend/provider and placement mismatches fail closed; no ambient request
    /// override or device discovery is consulted.
    pub(crate) fn from_candidate(
        candidate: &ExecutionCandidate,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, &'static str> {
        let provider = candidate.device.route.provider;
        let backend_matches = match backend {
            GgmlCpuGraphBackend::Cpu => provider == ExecutionProvider::Cpu,
            GgmlCpuGraphBackend::Metal => provider == ExecutionProvider::Metal,
            GgmlCpuGraphBackend::Gpu => matches!(
                provider,
                ExecutionProvider::Cuda | ExecutionProvider::Hip | ExecutionProvider::Vulkan
            ),
        };
        let placement_matches = match candidate.placement {
            ExecutionPlacement::CpuOnly => provider == ExecutionProvider::Cpu,
            ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => {
                provider != ExecutionProvider::Cpu
            }
        };
        if !backend_matches || !placement_matches {
            return Err("execution candidate lane is incompatible with resolved backend/placement");
        }
        Ok(Self {
            device: ResolvedDeviceKey::new(candidate.device.route.clone()),
            placement: candidate.placement,
            backend,
        })
    }
    /// Test/internal fallback for callers outside a request candidate. Native
    /// production dispatch attaches an exact lane to its request context.
    #[cfg(test)]
    pub(crate) fn unscoped_for_backend(backend: GgmlCpuGraphBackend) -> Self {
        let route = fallback_route_for_unscoped_backend(backend);
        Self {
            device: ResolvedDeviceKey::new(route),
            placement: match backend {
                GgmlCpuGraphBackend::Cpu => ExecutionPlacement::CpuOnly,
                GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu => {
                    ExecutionPlacement::FullDevice
                }
            },
            backend,
        }
    }
}

fn exact_lane_memory_sample_from_device_infos(
    lane: &ExecutionLaneKey,
    devices: &[(String, GgmlBackendKind, Option<GgmlDeviceMemory>)],
) -> Option<ExecutionLaneMemorySample> {
    let (stable_device_id, device_kind, memory) = devices.iter().find(|(name, _, _)| {
        name == lane.stable_device_id()
            && ExecutionProvider::from_backend_name(name) == lane.provider()
    })?;
    Some(ExecutionLaneMemorySample {
        provider: lane.provider(),
        stable_device_id: stable_device_id.clone(),
        device_kind: *device_kind,
        memory: (*memory)?,
    })
}

/// Stable identity of one explicitly constructed execution-service root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct NativeExecutionScopeId(u64);

impl NativeExecutionScopeId {
    pub(crate) fn next() -> Self {
        Self(NEXT_EXECUTION_SCOPE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Identity of the outermost transactional cache-publication attempt. Nested
/// auxiliary candidates inherit it and merge their journals into the parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ExecutionCacheAttemptId(u64);

impl ExecutionCacheAttemptId {
    fn next() -> Self {
        Self(NEXT_EXECUTION_CACHE_ATTEMPT_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn ordinal(self) -> u64 {
        self.0
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
    runtime_receipts: RuntimeReceiptCollector,
    stream_vad_embedded: crate::diarize::vad::StreamVadEmbeddedSlot,
    loaded_weight_owners: crate::ggml_runtime::LoadedWeightOwnerCache,
    backend_preference: Option<RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
    execution_lane: Option<ExecutionLaneKey>,
    observation_sink: Option<ExecutionObservationSink>,
    failure_sink: Option<ExecutionCandidateFailureSink>,
    cache_attempt_id: Option<ExecutionCacheAttemptId>,
    activation_reservation_cohort: Option<MemoryReservationCohortId>,
    execution_telemetry: Option<GgmlExecutionTelemetryCollector>,
    receipt: Option<NativeExecutionReceiptCollector>,
}

impl NativeExecutionContext {
    /// Stable execution-lane equality for a shared worker/engine key. The
    /// request-local failure sink is intentionally excluded: two jobs may
    /// share an engine only when their scope, broker, backend and placement
    /// agree, while each still retains its own sink for failure fan-out. An
    /// installed observation sink intentionally makes the lane request-local:
    /// audit evidence must never be silently attributed to another request.
    pub(crate) fn shares_execution_lane_with(&self, other: &Self) -> bool {
        self.scope_id == other.scope_id
            && Arc::ptr_eq(&self.memory_broker, &other.memory_broker)
            && self.backend_preference == other.backend_preference
            && self.placement == other.placement
            && self.execution_lane == other.execution_lane
            && self.activation_reservation_cohort == other.activation_reservation_cohort
            && match (&self.observation_sink, &other.observation_sink) {
                (Some(left), Some(right)) => Arc::ptr_eq(&left.observations, &right.observations),
                (None, None) => true,
                _ => false,
            }
            && match (&self.receipt, &other.receipt) {
                (Some(_), Some(_)) => false,
                (None, None) => true,
                _ => false,
            }
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
            runtime_receipts: first.runtime_receipts.clone(),
            stream_vad_embedded: Arc::clone(&first.stream_vad_embedded),
            loaded_weight_owners: first.loaded_weight_owners,
            backend_preference: first.backend_preference.clone(),
            placement: first.placement,
            execution_lane: first.execution_lane.clone(),
            observation_sink: first.observation_sink.clone(),
            failure_sink,
            cache_attempt_id,
            activation_reservation_cohort: first.activation_reservation_cohort,
            execution_telemetry,
            receipt: first.receipt.clone(),
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
    previous_receipts: Option<RuntimeReceiptCollector>,
    previous_stream_vad_embedded: Option<crate::diarize::vad::StreamVadEmbeddedSlot>,
    previous_loaded_weight_owners: Option<crate::ggml_runtime::LoadedWeightOwnerCache>,
    previous_placement: Option<ExecutionPlacement>,
    previous_execution_lane: Option<ExecutionLaneKey>,
    previous_observation_sink: Option<ExecutionObservationSink>,
    previous_failure_sink: Option<ExecutionCandidateFailureSink>,
    previous_cache_attempt_id: Option<ExecutionCacheAttemptId>,
    previous_activation_reservation_cohort: Option<MemoryReservationCohortId>,
    execution_telemetry: GgmlExecutionTelemetryGuard,
    _graph_lifecycle: GgmlGraphLifecycleGuard,
    previous_receipt: Option<NativeExecutionReceiptCollector>,
    backend: RequestBackendOverrideGuard,
}

impl Drop for NativeExecutionContextGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_MEMORY_BROKER.with(|current| {
            *current.borrow_mut() = self.previous_memory_broker.take();
        });
        CURRENT_EXECUTION_RECEIPTS.with(|current| {
            *current.borrow_mut() = self.previous_receipts.take();
        });
        CURRENT_STREAM_VAD_EMBEDDED.with(|current| {
            *current.borrow_mut() = self.previous_stream_vad_embedded.take();
        });
        CURRENT_LOADED_WEIGHT_OWNERS.with(|current| {
            *current.borrow_mut() = self.previous_loaded_weight_owners.take();
        });
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.set(self.previous_placement));
        CURRENT_EXECUTION_LANE.with(|current| {
            *current.borrow_mut() = self.previous_execution_lane.take();
        });
        CURRENT_EXECUTION_OBSERVATION_SINK.with(|current| {
            *current.borrow_mut() = self.previous_observation_sink.take();
        });
        CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK.with(|current| {
            *current.borrow_mut() = self.previous_failure_sink.take();
        });
        CURRENT_EXECUTION_CACHE_ATTEMPT_ID
            .with(|current| current.set(self.previous_cache_attempt_id.take()));
        CURRENT_ACTIVATION_RESERVATION_COHORT
            .with(|current| current.set(self.previous_activation_reservation_cohort.take()));
        CURRENT_EXECUTION_RECEIPT.with(|current| {
            *current.borrow_mut() = self.previous_receipt.take();
        });
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
    let runtime_receipts = current_runtime_receipts()?;
    Some(NativeExecutionContext {
        scope_id,
        memory_broker,
        runtime_receipts,
        stream_vad_embedded: current_stream_vad_embedded_slot()?,
        loaded_weight_owners: current_loaded_weight_owners()?,
        backend_preference: request_backend_override(),
        placement: current_execution_placement(),
        execution_lane: current_execution_lane(),
        observation_sink: current_execution_observation_sink(),
        failure_sink: current_execution_candidate_failure_sink(),
        cache_attempt_id: current_execution_cache_attempt_id(),
        activation_reservation_cohort: current_activation_reservation_cohort_id(),
        execution_telemetry: current_execution_telemetry_collector(),
        receipt: current_execution_receipt_collector(),
    })
}

/// Returns the broker injected by the active request without extending any
/// native allocation's lifetime. Allocation owners clone it only indirectly
/// through the committed reservation they retain.
pub(crate) fn current_native_execution_memory_broker() -> Option<Arc<DeviceMemoryBrokerSet>> {
    CURRENT_EXECUTION_MEMORY_BROKER.with(|current| current.borrow().clone())
}

/// Returns the explicitly propagated owner-receipt collector, if native code is
/// executing under a service root.
pub(crate) fn current_runtime_receipts() -> Option<RuntimeReceiptCollector> {
    CURRENT_EXECUTION_RECEIPTS.with(|current| current.borrow().clone())
}

pub(crate) fn current_stream_vad_embedded_slot()
-> Option<crate::diarize::vad::StreamVadEmbeddedSlot> {
    CURRENT_STREAM_VAD_EMBEDDED.with(|current| current.borrow().clone())
}

/// NES-owned loaded-weight owner cache for the active request. Absent outside
/// an installed service root; production loaders then skip coalescing rather
/// than writing a process-global TLS owner table.
pub(crate) fn current_loaded_weight_owners() -> Option<crate::ggml_runtime::LoadedWeightOwnerCache>
{
    CURRENT_LOADED_WEIGHT_OWNERS.with(|current| *current.borrow())
}

pub(crate) fn current_execution_placement() -> Option<ExecutionPlacement> {
    CURRENT_EXECUTION_PLACEMENT.with(Cell::get)
}

pub(crate) fn current_execution_lane() -> Option<ExecutionLaneKey> {
    CURRENT_EXECUTION_LANE.with(|current| current.borrow().clone())
}

/// Install one already-resolved lane as the sole backend/placement authority.
/// The full route travels with the lane, so worker handoff never enumerates a
/// first/best device again. Other native execution services remain untouched.
pub(crate) struct ResolvedExecutionLaneGuard {
    previous_placement: Option<ExecutionPlacement>,
    previous_execution_lane: Option<ExecutionLaneKey>,
    backend: RequestBackendOverrideGuard,
}

impl Drop for ResolvedExecutionLaneGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.set(self.previous_placement));
        CURRENT_EXECUTION_LANE.with(|current| {
            *current.borrow_mut() = self.previous_execution_lane.take();
        });
        // `backend` restores itself after this `Drop` returns.
        let _ = &self.backend;
    }
}

#[must_use = "the resolved execution lane is uninstalled when the guard drops"]
pub(crate) fn install_resolved_execution_lane(
    lane: ExecutionLaneKey,
) -> ResolvedExecutionLaneGuard {
    let backend = install_request_backend_override(Some(lane.request_backend_preference()));
    let previous_placement =
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.replace(Some(lane.placement())));
    let previous_execution_lane =
        CURRENT_EXECUTION_LANE.with(|current| current.replace(Some(lane)));
    ResolvedExecutionLaneGuard {
        previous_placement,
        previous_execution_lane,
        backend,
    }
}

pub(crate) fn current_execution_observation_sink() -> Option<ExecutionObservationSink> {
    CURRENT_EXECUTION_OBSERVATION_SINK.with(|current| current.borrow().clone())
}

/// Returns the explicit request-local receipt collector, when a strict native
/// evidence producer installed one. Normal inference never creates it.
pub(crate) fn current_execution_receipt_collector() -> Option<NativeExecutionReceiptCollector> {
    CURRENT_EXECUTION_RECEIPT.with(|current| current.borrow().clone())
}

pub(crate) fn current_request_attempt_id() -> Option<crate::RequestAttemptId> {
    current_execution_receipt_collector()?.request_attempt_id()
}

pub(crate) struct ExecutionReceiptCollectorGuard {
    previous: Option<NativeExecutionReceiptCollector>,
    _graph_lifecycle: GgmlGraphLifecycleGuard,
}

impl Drop for ExecutionReceiptCollectorGuard {
    fn drop(&mut self) {
        CURRENT_EXECUTION_RECEIPT.with(|current| {
            *current.borrow_mut() = self.previous.take();
        });
    }
}

/// Install the receipt authority carried by a concrete request context before
/// candidate selection. Candidate/worker propagation then uses the existing
/// `NativeExecutionContext` path.
pub(crate) fn install_execution_receipt_collector(
    collector: Option<NativeExecutionReceiptCollector>,
) -> ExecutionReceiptCollectorGuard {
    let graph_lifecycle = install_graph_lifecycle_collector(
        collector
            .as_ref()
            .map(NativeExecutionReceiptCollector::graph_lifecycle_collector),
    );
    let previous = CURRENT_EXECUTION_RECEIPT.with(|current| current.replace(collector));
    ExecutionReceiptCollectorGuard {
        previous,
        _graph_lifecycle: graph_lifecycle,
    }
}

/// Records a runner-backed observation only while an explicitly instrumented
/// policy request is active. The route is recovered from the same Exact
/// backend preference that the runner receives, never from an environment
/// label.
pub(crate) fn record_current_execution_backend_observation(
    backend_identity: usize,
    backend_kind: GgmlCpuGraphBackend,
    backend_name: &str,
    actual_provider: ExecutionProvider,
    actual_stable_id: &str,
    actual_device: &crate::GgmlActualDeviceFacts,
    use_scheduler: bool,
) {
    if let Some(receipt) = current_execution_receipt_collector() {
        receipt.record_backend_observation(
            backend_identity,
            actual_provider,
            actual_stable_id,
            actual_device,
            use_scheduler,
        );
    }
    let Some(sink) = current_execution_observation_sink() else {
        return;
    };
    let Some(placement) = current_execution_placement() else {
        return;
    };
    let route = match request_backend_override() {
        Some(RequestBackendPreference::Exact(route)) => route,
        Some(RequestBackendPreference::CpuOnly) => ResolvedExecutionRoute::cpu(),
        Some(RequestBackendPreference::Accelerated) | None => return,
    };
    sink.record(ExecutionBackendObservation {
        requested_route: route,
        placement,
        backend_kind,
        backend_name: backend_name.to_string(),
        actual_provider,
        actual_stable_id: actual_stable_id.to_string(),
        actual_device: actual_device.clone(),
        use_scheduler,
        backend_identity,
        memory_receipts: Vec::new(),
    });
}

pub(crate) fn record_current_execution_backend_memory_stats(
    backend_identity: usize,
    lifecycle: BackendMemoryLifecyclePoint,
    snapshot: &BackendMemoryStatsSnapshot,
) {
    let Some(sink) = current_execution_observation_sink() else {
        return;
    };
    let provider = {
        let observations = sink.observations();
        observations
            .iter()
            .rev()
            .find(|observation| observation.backend_identity == backend_identity)
            .map(|observation| observation.actual_provider)
    };
    let Some(provider) = provider else {
        return;
    };
    sink.record_memory(
        backend_identity,
        snapshot.safe_receipts(provider, lifecycle),
    );
}

pub(crate) fn record_current_execution_backend_memory_unavailable(
    backend_identity: usize,
    lifecycle: BackendMemoryLifecyclePoint,
    reason: BackendMemoryUnknownReason,
) {
    let Some(sink) = current_execution_observation_sink() else {
        return;
    };
    sink.record_memory(
        backend_identity,
        vec![SafeBackendMemoryReceipt::unknown(lifecycle, reason)],
    );
}

/// Fail closed when an Exact runner violates its selected placement or is not
/// backed by that precise live ggml device. FullDevice is intentionally
/// stronger than merely "some GPU observation": every constructed runner
/// must be a direct GPU backend. Hybrid retains its explicit CPU helper path.
/// Unscoped low-level Exact tests have no placement contract, but still prove
/// the actual provider-local route identity.
pub(crate) fn attest_current_exact_accelerated_backend(
    backend_kind: GgmlCpuGraphBackend,
    actual_provider: ExecutionProvider,
    actual_stable_id: &str,
    use_scheduler: bool,
) -> Result<(), crate::device::execution_route::ExecutionRouteError> {
    let Some(RequestBackendPreference::Exact(requested)) = request_backend_override() else {
        return Ok(());
    };
    let placement = current_execution_placement();
    if placement == Some(ExecutionPlacement::FullDevice) && !backend_kind.is_gpu_class() {
        return Err(
            crate::device::execution_route::ExecutionRouteError::init_failed(
                "Exact FullDevice initialized a non-GPU runner",
            ),
        );
    }
    if placement == Some(ExecutionPlacement::FullDevice) && use_scheduler {
        return Err(
            crate::device::execution_route::ExecutionRouteError::init_failed(
                "Exact FullDevice initialized a scheduler-backed runner",
            ),
        );
    }
    if !backend_kind.is_gpu_class() {
        return Ok(());
    }
    if requested.provider == actual_provider && requested.stable_id == actual_stable_id {
        return Ok(());
    }
    Err(
        crate::device::execution_route::ExecutionRouteError::init_failed(format!(
            "exact backend mismatch: requested provider={} stable_id={}, actual provider={} stable_id={}",
            requested.provider.as_str(),
            requested.stable_id,
            actual_provider.as_str(),
            actual_stable_id,
        )),
    )
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
    current_activation_reservation_cohort_id().or_else(|| {
        current_execution_cache_attempt_id()
            .map(|attempt| MemoryReservationCohortId::new(attempt.0))
    })
}

fn current_activation_reservation_cohort_id() -> Option<MemoryReservationCohortId> {
    CURRENT_ACTIVATION_RESERVATION_COHORT.with(Cell::get)
}

pub(crate) struct ActivationReservationContextGuard {
    previous: Option<MemoryReservationCohortId>,
}

impl Drop for ActivationReservationContextGuard {
    fn drop(&mut self) {
        CURRENT_ACTIVATION_RESERVATION_COHORT.with(|current| current.set(self.previous.take()));
    }
}

pub(crate) fn install_activation_reservation_context(
    context: Option<ActivationReservationContext>,
) -> ActivationReservationContextGuard {
    let previous = CURRENT_ACTIVATION_RESERVATION_COHORT
        .with(|current| current.replace(context.map(|context| context.cohort_id)));
    ActivationReservationContextGuard { previous }
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
    if let Some(lane) = current_execution_lane() {
        let placement = match (lane.placement(), backend) {
            (ExecutionPlacement::Hybrid, GgmlCpuGraphBackend::Cpu) => ExecutionPlacement::CpuOnly,
            (placement, _) => placement,
        };
        return lane.for_stage(backend, placement);
    }
    let preference = request_backend_override();
    let mut route = match preference.as_ref() {
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
    if matches!(backend, GgmlCpuGraphBackend::Cpu)
        && matches!(placement, ExecutionPlacement::CpuOnly)
    {
        route = ResolvedExecutionRoute::cpu();
    }
    ExecutionLaneKey {
        device: ResolvedDeviceKey::new(route),
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
    let previous_receipts =
        CURRENT_EXECUTION_RECEIPTS.with(|current| current.replace(Some(context.runtime_receipts)));
    let previous_stream_vad_embedded = CURRENT_STREAM_VAD_EMBEDDED
        .with(|current| current.replace(Some(context.stream_vad_embedded)));
    let previous_loaded_weight_owners = CURRENT_LOADED_WEIGHT_OWNERS
        .with(|current| current.replace(Some(context.loaded_weight_owners)));
    let previous_placement =
        CURRENT_EXECUTION_PLACEMENT.with(|current| current.replace(context.placement));
    let previous_execution_lane =
        CURRENT_EXECUTION_LANE.with(|current| current.replace(context.execution_lane));
    let previous_observation_sink = CURRENT_EXECUTION_OBSERVATION_SINK
        .with(|current| current.replace(context.observation_sink));
    let previous_failure_sink = CURRENT_EXECUTION_CANDIDATE_FAILURE_SINK
        .with(|current| current.replace(context.failure_sink));
    let previous_cache_attempt_id = CURRENT_EXECUTION_CACHE_ATTEMPT_ID
        .with(|current| current.replace(context.cache_attempt_id));
    let previous_activation_reservation_cohort = CURRENT_ACTIVATION_RESERVATION_COHORT
        .with(|current| current.replace(context.activation_reservation_cohort));
    let execution_telemetry = install_execution_telemetry_collector(context.execution_telemetry);
    let graph_lifecycle = install_graph_lifecycle_collector(
        context
            .receipt
            .as_ref()
            .map(NativeExecutionReceiptCollector::graph_lifecycle_collector),
    );
    let previous_receipt =
        CURRENT_EXECUTION_RECEIPT.with(|current| current.replace(context.receipt));
    let backend = install_request_backend_override(context.backend_preference);
    NativeExecutionContextGuard {
        scope,
        previous_memory_broker,
        previous_receipts,
        previous_stream_vad_embedded,
        previous_loaded_weight_owners,
        previous_placement,
        previous_execution_lane,
        previous_observation_sink,
        previous_failure_sink,
        previous_cache_attempt_id,
        previous_activation_reservation_cohort,
        execution_telemetry,
        _graph_lifecycle: graph_lifecycle,
        previous_receipt,
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
        runtime_receipts: services.runtime_receipts.clone(),
        stream_vad_embedded: Arc::clone(&services.firered_stream_vad_embedded),
        loaded_weight_owners: services.loaded_weight_owners,
        // Preserve an enclosing policy attempt. Legacy direct callers have no
        // enclosing values and continue to install `None` for all three.
        backend_preference: request_backend_override(),
        placement: current_execution_placement(),
        execution_lane: current_execution_lane(),
        observation_sink: current_execution_observation_sink(),
        failure_sink: current_execution_candidate_failure_sink(),
        cache_attempt_id: current_execution_cache_attempt_id(),
        activation_reservation_cohort: current_activation_reservation_cohort_id(),
        execution_telemetry: current_execution_telemetry_collector(),
        receipt: current_execution_receipt_collector(),
    })
}

fn execution_candidate_context_identity(
    candidate: &ExecutionCandidate,
) -> (Option<RequestBackendPreference>, Option<ExecutionLaneKey>) {
    let backend_preference = if candidate.placement == ExecutionPlacement::CpuOnly {
        Some(RequestBackendPreference::CpuOnly)
    } else {
        Some(RequestBackendPreference::Exact(
            candidate.device.route.clone(),
        ))
    };
    let backend = match candidate.device.route.provider {
        ExecutionProvider::Cpu => GgmlCpuGraphBackend::Cpu,
        ExecutionProvider::Metal => GgmlCpuGraphBackend::Metal,
        ExecutionProvider::Cuda
        | ExecutionProvider::Hip
        | ExecutionProvider::Vulkan
        | ExecutionProvider::Accelerator
        | ExecutionProvider::Unknown => GgmlCpuGraphBackend::Gpu,
    };
    let execution_lane = ExecutionLaneKey::from_candidate(candidate, backend).ok();
    (backend_preference, execution_lane)
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
    let (backend_preference, execution_lane) = execution_candidate_context_identity(candidate);
    install_native_execution_context(NativeExecutionContext {
        scope_id: services.scope_id,
        memory_broker: Arc::clone(&services.memory_broker),
        runtime_receipts: services.runtime_receipts.clone(),
        stream_vad_embedded: Arc::clone(&services.firered_stream_vad_embedded),
        loaded_weight_owners: services.loaded_weight_owners,
        backend_preference,
        placement: Some(candidate.placement),
        execution_lane,
        observation_sink: current_execution_observation_sink(),
        failure_sink: Some(failure_sink),
        cache_attempt_id: current_execution_cache_attempt_id(),
        activation_reservation_cohort: current_activation_reservation_cohort_id(),
        execution_telemetry: current_execution_telemetry_collector(),
        receipt: current_execution_receipt_collector(),
    })
}

/// Runs control-only session cleanup in the already-selected candidate lane.
///
/// This deliberately installs the service root and exact backend identity but
/// none of the mutable attempt authorities: no cache journal, activation
/// reservation, failure/observation sink, placement telemetry, or execution
/// receipt. Cancellation may release candidate-owned resources, but it cannot
/// begin a new inference attempt or overwrite the last completed receipt.
pub(crate) fn run_execution_candidate_control_scope<T>(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    operation: impl FnOnce() -> T,
) -> T {
    let (backend_preference, execution_lane) = execution_candidate_context_identity(candidate);
    let _control_scope = install_native_execution_context(NativeExecutionContext {
        scope_id: services.scope_id,
        memory_broker: Arc::clone(&services.memory_broker),
        runtime_receipts: services.runtime_receipts.clone(),
        stream_vad_embedded: Arc::clone(&services.firered_stream_vad_embedded),
        loaded_weight_owners: services.loaded_weight_owners,
        backend_preference,
        placement: Some(candidate.placement),
        execution_lane,
        observation_sink: None,
        failure_sink: None,
        cache_attempt_id: None,
        activation_reservation_cohort: None,
        execution_telemetry: None,
        receipt: None,
    });
    operation()
}

/// Result of one complete candidate-local operation. The operation's ordinary
/// error remains opaque to policy; only the separately recorded typed failure
/// authorizes a caller to advance to another candidate.
pub(crate) struct ExecutionCandidateAttemptOutcome<T, E> {
    pub(crate) result: Result<T, E>,
    pub(crate) candidate_failure: Option<ExecutionCandidateFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeCandidateAttemptAttestError;

struct NativeCandidateAttemptAttestation<T, E, F> {
    identity: NativeCandidateAttemptFacts,
    candidate: ExecutionCandidate,
    operation: RefCell<Option<F>>,
    failure_sink: ExecutionCandidateFailureSink,
    placement_collector: Option<GgmlExecutionTelemetryCollector>,
    outcome: Rc<RefCell<Option<ExecutionCandidateAttemptOutcome<T, E>>>>,
}

impl<T, E, F> TypedAttestation<NativeCandidateAttemptFacts, NativeCandidateAttemptFacts>
    for NativeCandidateAttemptAttestation<T, E, F>
where
    F: FnOnce() -> Result<T, E>,
{
    type Identity = NativeCandidateAttemptFacts;
    type Evidence = ExecutionCandidateAttemptEvidence;
    type Error = NativeCandidateAttemptAttestError;

    fn identity(&self) -> &Self::Identity {
        &self.identity
    }

    fn attest(
        &self,
        facts: &ResolvedExecutionFacts<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            Self::Identity,
        >,
    ) -> Result<Self::Evidence, AttestationFailure<Self::Error>> {
        debug_assert_eq!(facts.identity(), &self.identity);
        let operation = self
            .operation
            .borrow_mut()
            .take()
            .expect("candidate attempt attestation runs once");
        let result = operation();
        let mut candidate_failure = self.failure_sink.failure();
        if result.is_ok() && candidate_failure.is_none() {
            candidate_failure = self.placement_collector.as_ref().and_then(|collector| {
                observed_placement_violation(&self.candidate, &collector.snapshot())
            });
        }
        let committed = result.is_ok() && candidate_failure.is_none();
        *self.outcome.borrow_mut() = Some(ExecutionCandidateAttemptOutcome {
            result,
            candidate_failure,
        });
        if committed {
            Ok(ExecutionCandidateAttemptEvidence::new(
                self.identity.clone(),
            ))
        } else {
            Err(AttestationFailure::Rejected(
                NativeCandidateAttemptAttestError,
            ))
        }
    }
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

/// Broker-backed reservation for one candidate activation. Quote is obtained
/// separately; this token is the atomic `try_reserve*` result.
/// Opaque identity shared by an outer activation reservation and every
/// nested owner allocation performed while materializing that candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationReservationContext {
    cohort_id: MemoryReservationCohortId,
}

impl ActivationReservationContext {
    /// Mint a request-scoped cohort so concurrent longform slices and nested
    /// host owners share one exclusive system-memory gate. Independent
    /// candidate fallbacks still mint their own attempt journals underneath.
    pub(crate) fn mint() -> Self {
        Self {
            cohort_id: MemoryReservationCohortId::new(ExecutionCacheAttemptId::next().0),
        }
    }
}

pub struct BrokerActivationReservation {
    batch: Option<DeviceMemoryReservationBatch>,
    context: ActivationReservationContext,
}

impl BrokerActivationReservation {
    fn from_batch(
        batch: DeviceMemoryReservationBatch,
        cohort_id: MemoryReservationCohortId,
    ) -> Result<Self, String> {
        if batch.is_empty() {
            return Err(
                "candidate activation quote produced no physical-domain reservation".to_string(),
            );
        }
        Ok(Self {
            batch: Some(batch),
            context: ActivationReservationContext { cohort_id },
        })
    }

    pub const fn context(&self) -> ActivationReservationContext {
        self.context
    }
}

impl ActivationReservation for BrokerActivationReservation {
    type Error = String;

    fn release(&mut self) -> Result<(), Self::Error> {
        drop(self.batch.take());
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        if let Some(mut batch) = self.batch.take() {
            batch.quarantine();
        }
        Ok(())
    }
}

/// Quote identity for one candidate activation. A nested auxiliary attempt
/// replaces the outer source and restores it on drop.
#[derive(Clone)]
pub(crate) enum CandidateActivationQuoteSource {
    Pack(VerifiedPack),
    Declared(SystemMemoryAllocationQuote),
}

pub(crate) struct CandidateActivationQuoteGuard {
    previous: Option<CandidateActivationQuoteSource>,
}

impl Drop for CandidateActivationQuoteGuard {
    fn drop(&mut self) {
        CURRENT_CANDIDATE_ACTIVATION_QUOTE.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub(crate) fn install_candidate_activation_quote(
    source: CandidateActivationQuoteSource,
) -> CandidateActivationQuoteGuard {
    let previous = CURRENT_CANDIDATE_ACTIVATION_QUOTE.with(|slot| slot.replace(Some(source)));
    CandidateActivationQuoteGuard { previous }
}

pub(crate) fn install_candidate_activation_pack(
    pack: VerifiedPack,
) -> CandidateActivationQuoteGuard {
    install_candidate_activation_quote(CandidateActivationQuoteSource::Pack(pack))
}

pub(crate) fn current_candidate_activation_quote() -> Option<CandidateActivationQuoteSource> {
    CURRENT_CANDIDATE_ACTIVATION_QUOTE.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
fn current_candidate_activation_pack() -> Option<VerifiedPack> {
    match current_candidate_activation_quote() {
        Some(CandidateActivationQuoteSource::Pack(pack)) => Some(pack),
        _ => None,
    }
}

fn architecture_id_from_pack(pack: &VerifiedPack) -> Result<&str, String> {
    match pack.route() {
        PackRoute::Asr {
            model_architecture, ..
        } => Ok(*model_architecture),
        PackRoute::Aux {
            model_architecture, ..
        } => Ok(model_architecture.as_str()),
    }
}

/// Resolves the first policy candidate for the pack being activated. This is
/// the live lane, not a dummy CPU route.
pub fn resolve_candidate_activation_lane(
    services: &NativeExecutionServices,
    pack: &VerifiedPack,
    intent: ExecutionIntent,
) -> Result<ExecutionCandidate, String> {
    let architecture_id = architecture_id_from_pack(pack)?;
    let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(architecture_id)
        .ok_or_else(|| {
            format!("candidate activation has no architecture descriptor for {architecture_id}")
        })?;
    let inventory = enumerate_compute_devices_from_ggml(&ggml_available_devices());
    let plan = services
        .policy_resolver()
        .resolve(
            intent,
            crate::arch::family_auto_gpu_policy_for_model_architecture(architecture_id),
            descriptor.execution_contract.execution_capabilities,
            &inventory,
        )
        .map_err(|error| error.to_string())?;
    plan.candidates()
        .first()
        .cloned()
        .ok_or_else(|| format!("execution policy produced no candidate lane for {architecture_id}"))
}

/// Fully resolved default-model activation facts. The server consumes this
/// value but cannot independently recombine family policy, provider identity,
/// output-plan evidence, or reuse evidence.
#[derive(Debug, Clone)]
pub struct ResolvedDefaultModelActivation {
    candidate: ExecutionCandidate,
    verified_pack: VerifiedPack,
    facts: DefaultModelActivationFacts,
}

impl ResolvedDefaultModelActivation {
    pub const fn facts(&self) -> &DefaultModelActivationFacts {
        &self.facts
    }

    pub fn quote(&self) -> Result<DefaultModelActivationQuote, String> {
        let (_backends, _mapping, plan) = quote_candidate_activation_plan(
            &self.verified_pack,
            &self.candidate,
            self.facts.plan().resident_topology(),
        )?;
        Ok(DefaultModelActivationQuote {
            plan,
            candidate: self.candidate.clone(),
            content_id: self.verified_pack.content_id().to_string(),
        })
    }

    pub fn quote_and_reserve(
        &self,
        services: &NativeExecutionServices,
    ) -> Result<BrokerActivationReservation, String> {
        self.quote()?.reserve(services)
    }
}

/// Opaque, observation-bound activation quote. The server can place a fault
/// or audit checkpoint between quote/stat observation and broker mutation, but
/// cannot inspect or rewrite physical-domain rows.
pub struct DefaultModelActivationQuote {
    plan: NativeMemoryAdmissionPlan,
    candidate: ExecutionCandidate,
    content_id: String,
}

impl DefaultModelActivationQuote {
    pub fn reserve(
        self,
        services: &NativeExecutionServices,
    ) -> Result<BrokerActivationReservation, String> {
        reserve_activation_plan(
            services,
            &self.plan,
            &self.candidate,
            Some(&self.content_id),
        )
    }
}

fn resolve_resident_topology_plan(
    descriptor: crate::arch::OpenAsrArchitectureDescriptor,
    pack: &VerifiedPack,
    candidate: &ExecutionCandidate,
    intent: &ExecutionIntent,
    allow_unified_runtime: bool,
) -> Result<DefaultModelResidentTopologyPlan, String> {
    let session = crate::arch::runtime_footprint::ResidentSessionEnvelope::activation_prepare();
    let topology = descriptor
        .build_resident_topology(pack, candidate, intent, &session, allow_unified_runtime)
        .map_err(|error| format!("candidate activation resident topology failed: {error:?}"))?;
    Ok(DefaultModelResidentTopologyPlan {
        architecture: topology.architecture(),
        components: topology
            .components()
            .iter()
            .map(|component| {
                let verified = component.verified();
                let spec = verified.spec();
                DefaultModelResidentComponentPlan {
                    component: spec.component(),
                    variant: spec.variant(),
                    phase: spec.phase(),
                    lifetime: spec.lifetime(),
                    dependencies: spec.dependencies().to_vec(),
                    representations: spec.representations().to_vec(),
                    checkout: spec.checkout(),
                    placement: verified.resolved_variant().placement(),
                }
            })
            .collect(),
        dependency_order: topology.dependency_order().to_vec(),
    })
}

/// Resolve the verified pack, exact candidate lane, output plan, reuse mode,
/// and activation identity exactly once before owner acquisition.
pub fn resolve_default_model_activation(
    services: &NativeExecutionServices,
    pack: &VerifiedPack,
    intent: ExecutionIntent,
    pull: String,
    path: std::path::PathBuf,
) -> Result<ResolvedDefaultModelActivation, String> {
    if pack.preflight().runtime_source().path() != path {
        return Err("default-model activation path does not match the verified source".to_string());
    }
    let architecture_id = architecture_id_from_pack(pack)?;
    let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(architecture_id)
        .ok_or_else(|| {
            format!("candidate activation has no architecture descriptor for {architecture_id}")
        })?;
    let candidate = resolve_candidate_activation_lane(services, pack, intent.clone())?;
    let resident_topology =
        resolve_resident_topology_plan(descriptor, pack, &candidate, &intent, true)?;
    let logits_consumers = super::device_greedy_token::decode_logits_consumers_for_request(
        descriptor.identity.adapter_id,
        false,
        false,
        false,
    );
    let resolved_runtime = super::device_greedy_token::resolved_runtime_for_family_candidate(
        &candidate,
        crate::arch::family_auto_gpu_policy_for_model_architecture(architecture_id),
        descriptor.identity.adapter_id,
        logits_consumers,
    );
    let pack_content_id = pack.content_id().to_string();
    let plan = DefaultModelActivationPlan::new(
        path.clone(),
        pack_content_id.clone(),
        architecture_id.to_string(),
        intent.clone(),
        resolved_runtime,
        resident_topology.clone(),
    );
    let lane = DefaultModelActivationLane::new(candidate.clone());
    let identity = DefaultModelActivationIdentity::new(
        pull,
        path,
        pack_content_id,
        architecture_id.to_string(),
        intent,
        candidate.clone(),
        plan.output_plan(),
        plan.reuse_mode(),
        resident_topology,
    );
    Ok(ResolvedDefaultModelActivation {
        candidate,
        verified_pack: pack.clone(),
        facts: ResolvedExecutionFacts::new(plan, lane, identity),
    })
}

fn cstr_ptr_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn ggml_backend_physical_identity(
    backend: ffi::GgmlBackendRaw,
) -> Result<PhysicalDeviceKey, String> {
    if backend.is_null() {
        return Err("candidate activation quote received a null ggml backend".to_string());
    }
    let device = unsafe { ffi::ggml_backend_get_device(backend) };
    ggml_device_physical_identity(device)
}

fn ggml_device_physical_identity(
    device: ffi::GgmlBackendDevRaw,
) -> Result<PhysicalDeviceKey, String> {
    if device.is_null() {
        return Err("candidate activation backend has no device for physical identity".to_string());
    }
    let mut props = ffi::GgmlBackendDevProps {
        name: std::ptr::null(),
        description: std::ptr::null(),
        memory_free: 0,
        memory_total: 0,
        type_: 0,
        device_id: std::ptr::null(),
        caps: ffi::GgmlBackendDevCaps::default(),
    };
    unsafe { ffi::ggml_backend_dev_get_props(device, &mut props) };
    let name = cstr_ptr_lossy(props.name);
    let provider = ExecutionProvider::from_backend_name(&name);
    let stable_id = if props.device_id.is_null() {
        name
    } else {
        let value = cstr_ptr_lossy(props.device_id);
        if value.trim().is_empty() { name } else { value }
    };
    PhysicalDeviceKey::new(format!("{}:{stable_id}", provider.as_str()))
        .map_err(|error| error.to_string())
}

fn host_backend_for_activation() -> Result<GgmlBackend, String> {
    ensure_backends_loaded();
    GgmlBackend::cpu().map_err(|error| error.to_string())
}

fn discrete_device_for_candidate(
    candidate: &ExecutionCandidate,
) -> Result<Option<GgmlBackendDevice>, String> {
    match candidate.device.route.provider {
        ExecutionProvider::Cuda | ExecutionProvider::Hip | ExecutionProvider::Vulkan
            if candidate.placement != ExecutionPlacement::CpuOnly => {}
        _ => return Ok(None),
    }
    ensure_backends_loaded();
    let devices = ggml_available_devices();
    let wanted = candidate.device.route.provider;
    let stable = candidate.device.route.stable_id.as_str();
    Ok(devices.into_iter().find(|device| {
        ExecutionProvider::from_backend_name(&device.name) == wanted
            && (device.name == stable
                || device
                    .device_id
                    .as_deref()
                    .is_some_and(|id| id.eq_ignore_ascii_case(stable)))
    }))
}

fn quote_activation_group(
    group_id: &str,
    identity: PhysicalDeviceKey,
    abi: BackendMemoryAbi,
    request: ffi::GgmlBackendMemoryRequestV1,
) -> Result<NativeQuotedBackendGroup, String> {
    let semantics = NativeMemoryClaimSemantics {
        resource_id: group_id.to_owned(),
        lifetime: AllocationLifetime::PackShared,
        phases: PhaseSet::ALL,
    };
    NativeQuotedBackendGroup::quote(
        group_id,
        identity,
        abi,
        vec![request],
        BTreeMap::from([(request.request_id, semantics.clone())]),
        semantics,
    )
    .map_err(|error| format!("candidate activation ggml quote: {error}"))
}

fn quote_host_activation_group(
    group_id: &str,
    host: &GgmlBackend,
    request: ffi::GgmlBackendMemoryRequestV1,
) -> Result<NativeQuotedBackendGroup, String> {
    let abi = unsafe { BackendMemoryAbi::from_backend(host.as_ptr()) }
        .map_err(|error| format!("candidate activation ABI: {error}"))?;
    quote_activation_group(
        group_id,
        ggml_backend_physical_identity(host.as_ptr())?,
        abi,
        request,
    )
}

fn quote_discrete_activation_group(
    device: &GgmlBackendDevice,
    activation_resource: &str,
    requested_bytes: u64,
) -> Result<(NativeQuotedBackendGroup, Option<GgmlBackend>), String> {
    let device_raw = device.as_ptr();
    let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device_raw) };
    if buft.is_null() {
        return Err("discrete activation device has no buffer type to quote".to_string());
    }
    let group_id = format!("candidate-activation-device-copy:{activation_resource}");
    let identity = ggml_device_physical_identity(device_raw)?;
    // Vulkan BUFFER quotes accept a registry `buft` with a null backend. CUDA
    // and HIP still require a live backend context to mint the quote token.
    if ExecutionProvider::from_backend_name(&device.name) == ExecutionProvider::Vulkan {
        let request = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
            usage: ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS as u32,
            request_id: 1,
            backend: std::ptr::null_mut(),
            buft,
            requested_bytes,
            currently_allocated_bytes: 0,
            ..Default::default()
        };
        let abi = unsafe { BackendMemoryAbi::from_device(device_raw) }
            .map_err(|error| format!("candidate activation device ABI: {error}"))?;
        return Ok((
            quote_activation_group(&group_id, identity, abi, request)?,
            None,
        ));
    }
    let backend = device.initialize().map_err(|error| error.to_string())?;
    let request = ffi::GgmlBackendMemoryRequestV1 {
        kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
        usage: ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS as u32,
        request_id: 1,
        backend: backend.as_ptr(),
        buft,
        requested_bytes,
        currently_allocated_bytes: 0,
        ..Default::default()
    };
    let abi = unsafe { BackendMemoryAbi::from_backend(backend.as_ptr()) }
        .map_err(|error| format!("candidate activation ABI: {error}"))?;
    Ok((
        quote_activation_group(&group_id, identity, abi, request)?,
        Some(backend),
    ))
}

fn quote_candidate_activation_plan(
    pack: &VerifiedPack,
    candidate: &ExecutionCandidate,
    resident_topology: &DefaultModelResidentTopologyPlan,
) -> Result<
    (
        Vec<GgmlBackend>,
        std::sync::Arc<memmap2::Mmap>,
        NativeMemoryAdmissionPlan,
    ),
    String,
> {
    let prepared_components = resident_topology
        .components
        .iter()
        .filter(|component| {
            matches!(
                component.phase,
                crate::arch::runtime_footprint::ResidentPhase::Prepare
                    | crate::arch::runtime_footprint::ResidentPhase::Load
            )
        })
        .map(|component| component.component)
        .collect::<Vec<_>>();
    if prepared_components.is_empty() {
        return Err(
            "architecture resident topology has no Prepare/Load component to reserve".to_string(),
        );
    }
    let topology_resource = format!(
        "{}:{}",
        resident_topology.architecture,
        prepared_components.join(",")
    );
    quote_pack_activation_plan(pack, candidate, &topology_resource)
}

/// Quote the verified pack bytes for one already-resolved activation resource.
/// ASR callers derive that identity from their resident topology. Auxiliary
/// callers derive it from the canonical aux registry instead of masquerading
/// as an ASR architecture descriptor.
fn quote_pack_activation_plan(
    pack: &VerifiedPack,
    candidate: &ExecutionCandidate,
    activation_resource: &str,
) -> Result<
    (
        Vec<GgmlBackend>,
        std::sync::Arc<memmap2::Mmap>,
        NativeMemoryAdmissionPlan,
    ),
    String,
> {
    let mmap = pack.preflight().runtime_source().backing_mmap();
    let requested_bytes = pack.preflight().runtime_source().byte_len();
    if requested_bytes == 0 || mmap.is_empty() {
        return Err("verified pack mapping is empty and cannot be reserved as zero".to_string());
    }
    let host = host_backend_for_activation()?;
    let host_import = ffi::GgmlBackendMemoryRequestV1 {
        kind: ffi::GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT,
        usage: ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS as u32,
        request_id: 1,
        backend: host.as_ptr(),
        host_ptr: mmap.as_ptr().cast::<c_void>(),
        requested_bytes,
        currently_allocated_bytes: 0,
        ..Default::default()
    };
    let host_group_id = format!("candidate-activation-host-import:{activation_resource}");
    let host_group =
        quote_host_activation_group(&host_group_id, &host, host_import).or_else(|_| {
            let device = unsafe { ffi::ggml_backend_get_device(host.as_ptr()) };
            if device.is_null() {
                return Err("host activation backend has no device".to_string());
            }
            let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device) };
            if buft.is_null() {
                return Err("host activation backend has no buffer type to quote".to_string());
            }
            let host_copy = ffi::GgmlBackendMemoryRequestV1 {
                kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
                usage: ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS as u32,
                request_id: 1,
                backend: host.as_ptr(),
                buft,
                requested_bytes,
                currently_allocated_bytes: 0,
                ..Default::default()
            };
            quote_host_activation_group(
                &format!("candidate-activation-host-copy:{activation_resource}"),
                &host,
                host_copy,
            )
        })?;
    let mut groups = vec![host_group];
    let mut backends = vec![host];
    if let Some(device) = discrete_device_for_candidate(candidate)? {
        let (group, live_backend) =
            quote_discrete_activation_group(&device, activation_resource, requested_bytes)?;
        groups.push(group);
        if let Some(live_backend) = live_backend {
            backends.push(live_backend);
        }
    }
    let plan = admission_plan_from_quoted_groups(groups)?;
    Ok((backends, mmap, plan))
}

fn admission_plan_from_quoted_groups(
    groups: Vec<NativeQuotedBackendGroup>,
) -> Result<NativeMemoryAdmissionPlan, String> {
    let plan = NativeMemoryAdmissionPlan::from_groups(groups)
        .map_err(|error| format!("candidate activation admission plan: {error}"))?;
    if plan.reservation_requests().is_empty() {
        return Err("candidate activation quote produced no physical-domain requests".to_string());
    }
    for request in plan.reservation_requests() {
        if request.peak_bytes == 0 {
            return Err(format!(
                "candidate activation quote for {} is unknown and cannot be reserved as zero",
                request.domain
            ));
        }
    }
    Ok(plan)
}

fn quote_cpu_buffer_plan(
    group_id: &str,
    requested_bytes: u64,
) -> Result<(GgmlBackend, NativeMemoryAdmissionPlan), String> {
    if requested_bytes == 0 {
        return Err(
            "declared resident bytes are unknown and cannot be reserved as zero".to_string(),
        );
    }
    let host = host_backend_for_activation()?;
    let device = unsafe { ffi::ggml_backend_get_device(host.as_ptr()) };
    if device.is_null() {
        return Err("host activation backend has no device".to_string());
    }
    let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device) };
    if buft.is_null() {
        return Err("host activation backend has no buffer type to quote".to_string());
    }
    let request = ffi::GgmlBackendMemoryRequestV1 {
        kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
        usage: ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS as u32,
        request_id: 1,
        backend: host.as_ptr(),
        buft,
        requested_bytes,
        currently_allocated_bytes: 0,
        ..Default::default()
    };
    let group = quote_host_activation_group(group_id, &host, request)?;
    let plan = admission_plan_from_quoted_groups(vec![group])?;
    Ok((host, plan))
}

fn reserve_activation_plan(
    services: &NativeExecutionServices,
    plan: &NativeMemoryAdmissionPlan,
    candidate: &ExecutionCandidate,
    content_id: Option<&str>,
) -> Result<BrokerActivationReservation, String> {
    let mut requests = plan.reservation_requests().to_vec();
    let cohort_id = current_memory_reservation_cohort_id()
        .unwrap_or_else(|| MemoryReservationCohortId::new(ExecutionCacheAttemptId::next().0));
    for request in &mut requests {
        request.cohort_id = Some(cohort_id);
    }
    let collector = services.runtime_receipts();
    let backend = match candidate.device.route.provider {
        ExecutionProvider::Cpu => GgmlCpuGraphBackend::Cpu,
        ExecutionProvider::Metal => GgmlCpuGraphBackend::Metal,
        ExecutionProvider::Cuda
        | ExecutionProvider::Hip
        | ExecutionProvider::Vulkan
        | ExecutionProvider::Accelerator
        | ExecutionProvider::Unknown => GgmlCpuGraphBackend::Gpu,
    };
    let lane = collector.lane_projection(
        candidate.device.route.provider,
        &candidate.device.route.stable_id,
        candidate.placement,
        backend,
    );
    let owner_placement = lane.map_or(
        RuntimeOwnerPlacement::Unknown,
        RuntimeOwnerPlacement::LaneBound,
    );
    let mut batch = services
        .memory_broker()
        .try_reserve_batch_for_scope_and_placement(
            requests,
            Some(services.scope_id),
            owner_placement,
        )
        .map_err(|error| format!("candidate activation reserve: {error}"))?;
    if let Some(lane) = lane
        && let Some(owner) = collector.owner_descriptor(
            "candidate-activation-reservation",
            content_id,
            plan.reservation_requests()
                .first()
                .map(|request| request.resource_id.as_str()),
            Some(lane),
        )
    {
        let resources = plan
            .reservation_requests()
            .iter()
            .filter_map(|request| {
                collector
                    .resource_descriptor(
                        &request.resource_id,
                        &request.domain,
                        request.peak_bytes,
                        request.peak_bytes,
                        request.retained_bytes,
                        plan.quote_confidence_for_domain(&request.domain),
                        Some(request.snapshot.confidence),
                    )
                    .map(|descriptor| (request.domain.clone(), descriptor))
            })
            .collect();
        batch.attach_receipt(collector.clone(), owner, resources);
    }
    BrokerActivationReservation::from_batch(batch, cohort_id)
}

fn quote_and_reserve_declared_host_resident(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    quote: &SystemMemoryAllocationQuote,
) -> Result<BrokerActivationReservation, String> {
    let (_backend, plan) = quote_cpu_buffer_plan(
        &quote.resource_id,
        quote.peak_bytes.max(quote.retained_bytes),
    )?;
    reserve_activation_plan(services, &plan, candidate, None)
}

fn quote_and_reserve_current_candidate_activation(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
) -> Result<BrokerActivationReservation, String> {
    match current_candidate_activation_quote() {
        Some(CandidateActivationQuoteSource::Pack(pack)) => {
            quote_and_reserve_candidate_activation(services, candidate, &pack)
        }
        Some(CandidateActivationQuoteSource::Declared(quote)) => {
            quote_and_reserve_declared_host_resident(services, candidate, &quote)
        }
        None => Err(
            "candidate activation cannot quote without a verified pack or the current owner's declared resident bytes"
                .to_string(),
        ),
    }
}

/// Quotes known physical domains for one candidate lane from the architecture
/// resident footprint and the current execution candidate, then atomically
/// reserves them. Unknown cost is never treated as zero.
pub(crate) fn quote_and_reserve_candidate_activation(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    pack: &VerifiedPack,
) -> Result<BrokerActivationReservation, String> {
    let architecture_id = architecture_id_from_pack(pack)?;
    let (_backends, _mapping, plan) = match pack.route() {
        PackRoute::Asr { .. } => {
            let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(architecture_id)
                .ok_or_else(|| {
                    format!(
                        "candidate activation has no architecture descriptor for {architecture_id}"
                    )
                })?;
            let intent = ExecutionIntent::Exact(
                crate::device::execution_route::ExactDeviceSelector::StableId {
                    provider: Some(candidate.device.route.provider),
                    stable_id: candidate.device.route.stable_id.clone(),
                },
            );
            let topology =
                resolve_resident_topology_plan(descriptor, pack, candidate, &intent, true)?;
            quote_candidate_activation_plan(pack, candidate, &topology)?
        }
        PackRoute::Aux { .. } => {
            let policy = super::aux_pack_registry::auxiliary_execution_policy(architecture_id)
                .ok_or_else(|| {
                    format!(
                        "candidate activation has no auxiliary execution descriptor for {architecture_id}"
                    )
                })?;
            let super::aux_pack_registry::AuxiliaryExecutionPolicy::RequestScoped {
                capabilities,
                ..
            } = policy;
            if !capabilities.supports(candidate.device.route.provider, candidate.placement) {
                return Err(format!(
                    "candidate activation lane is not permitted by auxiliary architecture {architecture_id}"
                ));
            }
            let ownership =
                super::aux_pack_registry::auxiliary_runtime_ownership(architecture_id)
                    .ok_or_else(|| {
                        format!(
                            "candidate activation has no auxiliary ownership descriptor for {architecture_id}"
                        )
                    })?;
            let activation_resource = format!("aux:{architecture_id}:{}", ownership.as_str());
            quote_pack_activation_plan(pack, candidate, &activation_resource)?
        }
    };
    reserve_activation_plan(services, &plan, candidate, Some(pack.content_id()))
}

/// Runs a complete allocation/execution operation inside one candidate's
/// dynamic context and captures its typed failure side channel before the
/// context is restored. This is the single production runner: it walks
/// [`super::candidate_activation_transaction::CandidateActivationTransaction`]
/// internally so offline, streaming, serve-batch, and auxiliary callers share
/// prepare -> reserve -> materialize -> AttestationPending -> attest -> commit.
/// Quote is obtained first; reservation is a broker batch, not a Noop.
pub(crate) fn run_execution_candidate_attempt<T, E>(
    services: &NativeExecutionServices,
    candidate: &ExecutionCandidate,
    operation: impl FnOnce() -> Result<T, E>,
) -> ExecutionCandidateAttemptOutcome<T, E> {
    let failure_sink = ExecutionCandidateFailureSink::new();
    let placement_collector = (candidate.placement != ExecutionPlacement::CpuOnly)
        .then(GgmlExecutionTelemetryCollector::new);
    let receipt = current_execution_receipt_collector();
    // This collector belongs to precisely one candidate attempt. It never
    // consults the caller's cumulative telemetry when publishing receipt facts.
    let receipt_collector = receipt
        .as_ref()
        .map(|_| GgmlExecutionTelemetryCollector::new());
    let outer_collector = current_execution_telemetry_collector();
    let combined_collector = GgmlExecutionTelemetryCollector::fanout(
        outer_collector
            .iter()
            .chain(placement_collector.iter())
            .chain(receipt_collector.iter()),
    );
    // Audit observations follow the same candidate transaction as resident
    // cache publication. A failed FullDevice attempt must not contaminate the
    // evidence for a succeeding Hybrid attempt on the same Exact device.
    let observation_transaction = current_execution_observation_sink()
        .map(|parent| (parent, ExecutionObservationSink::new()));
    if let Some(receipt) = &receipt {
        receipt.begin_candidate_attempt();
    }
    let facts = NativeCandidateAttemptFacts::new(candidate.clone());
    let outcome_slot = Rc::new(RefCell::new(None));
    let (result, candidate_failure, committed, receipt_placement) = {
        let _observation_guard = observation_transaction
            .as_ref()
            .map(|(_, attempt)| install_execution_observation_sink(attempt.clone()));
        let _telemetry = install_execution_telemetry_collector(combined_collector);
        let _attempt =
            install_execution_candidate_attempt(services, candidate, failure_sink.clone());
        if receipt.is_some() {
            services.runtime_receipts.begin_request_event_window();
        }
        let journal_scope = ExecutionCacheJournalScope::begin();
        let reservation = match quote_and_reserve_current_candidate_activation(services, candidate)
        {
            Ok(reservation) => reservation,
            Err(reason) => {
                let failure =
                    ExecutionCandidateFailure::capacity("candidate-activation-reserve", reason);
                record_current_execution_candidate_failure(failure.clone());
                let result = operation();
                let mut candidate_failure = failure_sink.failure().or(Some(failure));
                if result.is_ok() && candidate_failure.is_none() {
                    candidate_failure = placement_collector.as_ref().and_then(|collector| {
                        observed_placement_violation(candidate, &collector.snapshot())
                    });
                }
                journal_scope.finish(false);
                if let Some(receipt) = &receipt {
                    receipt.finish_candidate_attempt(false);
                }
                return ExecutionCandidateAttemptOutcome {
                    result,
                    candidate_failure,
                };
            }
        };
        let contract = NativeCandidateAttemptAttestation {
            identity: facts.clone(),
            candidate: candidate.clone(),
            operation: RefCell::new(Some(operation)),
            failure_sink: failure_sink.clone(),
            placement_collector: placement_collector.clone(),
            outcome: Rc::clone(&outcome_slot),
        };
        let resolved = ResolvedExecutionFacts::new(facts.clone(), facts.clone(), facts.clone());
        let prepared = ExecutionCandidateAttemptJournalFactory::bind(move |commit| {
            journal_scope.finish(commit);
        })
        .prepare(facts, resolved);
        debug_assert_eq!(prepared.stage(), ActivationStage::Prepared);
        let reserved = prepared.reserve(reservation);
        debug_assert_eq!(reserved.stage(), ActivationStage::Reserved);
        let materialized = reserved.materialize(std::iter::once(ExecutionCandidateAttemptOwner));
        debug_assert_eq!(materialized.stage(), ActivationStage::Materialized);
        let pending = materialized.begin_attestation(contract);
        debug_assert_eq!(pending.stage(), ActivationStage::AttestationPending);
        match pending.attest() {
            AttestationOutcome::Attested(attested) => {
                debug_assert_eq!(attested.stage(), ActivationStage::Attested);
                attested
                    .commit_attempt()
                    .expect("candidate attempt journal publication cannot fail");
            }
            AttestationOutcome::Rejected { transaction, .. } => {
                let _ = transaction.rollback_attempt();
            }
            AttestationOutcome::MustQuarantine { transaction, .. } => {
                let _ = transaction.quarantine_attempt();
            }
        }
        let outcome = outcome_slot
            .borrow_mut()
            .take()
            .expect("candidate attempt attestation stored an outcome");
        let committed = outcome.result.is_ok() && outcome.candidate_failure.is_none();
        let receipt_placement = receipt_collector
            .as_ref()
            .map(GgmlExecutionTelemetryCollector::snapshot);
        (
            outcome.result,
            outcome.candidate_failure,
            committed,
            receipt_placement,
        )
    };
    if committed && let Some((parent, attempt)) = observation_transaction {
        parent.append(attempt.observations());
    }
    if let Some(receipt) = receipt {
        receipt.finish_candidate_attempt(committed);
        if committed {
            receipt.record_placement(receipt_placement.unwrap_or_default());
        }
    }
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
    runtime_receipts: RuntimeReceiptCollector,
    auxiliary_runtime_owners: super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache,
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
    firered_stream_vad_embedded: crate::diarize::vad::StreamVadEmbeddedSlot,
    loaded_weight_owners: crate::ggml_runtime::LoadedWeightOwnerCache,
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
        Self::new_with_broker_and_receipt_factory(
            policy_resolver,
            memory_broker,
            RuntimeReceiptCollector::new,
        )
    }

    fn new_with_broker_and_receipt_factory(
        policy_resolver: Arc<dyn ExecutionPolicyResolver>,
        memory_broker: Arc<DeviceMemoryBrokerSet>,
        receipt_factory: impl FnOnce(NativeExecutionScopeId) -> RuntimeReceiptCollector,
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

        let scope_id = NativeExecutionScopeId::next();
        Ok(Self {
            scope_id,
            policy_resolver,
            memory_broker: Arc::clone(&memory_broker),
            runtime_receipts: receipt_factory(scope_id),
            auxiliary_runtime_owners:
                super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache::default(),
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
            firered_stream_vad_embedded: Arc::new(Mutex::new(None)),
            loaded_weight_owners: crate::ggml_runtime::LoadedWeightOwnerCache::new(scope_id),
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

    pub fn runtime_receipts(&self) -> &RuntimeReceiptCollector {
        &self.runtime_receipts
    }

    pub(crate) fn auxiliary_runtime_owners(
        &self,
    ) -> &super::policy_resolved_aux_runtime::AuxiliaryRuntimeOwnerCache {
        &self.auxiliary_runtime_owners
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
        self.firered_punc_actors.clear();
        self.diarizen_segmenter_actors.clear();
        self.pyannote_segmenter_actors.clear();
        self.redimnet_runtime_actors.clear();
        self.firered_stream_vad_realtime_actors.clear();
        *self
            .firered_stream_vad_embedded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.loaded_weight_owners.clear();
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
        self.firered_punc_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.diarizen_segmenter_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.pyannote_segmenter_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.redimnet_runtime_actors
            .evict_where(|key| key.has_content_id(pack_content_id));
        self.loaded_weight_owners.evict_content_id(pack_content_id);
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
pub(crate) fn test_native_execution_services_with_entropy_failure() -> Arc<NativeExecutionServices>
{
    Arc::new(
        NativeExecutionServices::new_with_broker_and_receipt_factory(
            Arc::new(DefaultExecutionPolicyResolver),
            Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default())),
            RuntimeReceiptCollector::new_with_entropy_failure_for_test,
        )
        .expect("receipt entropy failure must not block service construction"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::device::{
        execution_memory::{
            DeviceMemorySnapshot, DeviceMemoryUsage, MemoryDomainKey, MemoryObservationConfidence,
        },
        execution_policy::ExecutionDeviceSnapshot,
        execution_route::{
            DeviceAddressability, ExecutionProvider, PhysicalResourceKey, ResolvedExecutionRoute,
            RouteDeviceKind,
        },
    };
    use crate::ggml_runtime::{
        GgmlActualDeviceFacts, GgmlBackendKind, GgmlCpuGraphConfig, GgmlCpuGraphError,
        GgmlCpuGraphRunner,
    };

    fn test_cuda_device() -> GgmlActualDeviceFacts {
        GgmlActualDeviceFacts {
            device_type: "gpu".to_string(),
            name: "CUDA0".to_string(),
            description: "test CUDA device".to_string(),
            provider_device_id: Some("0000:01:00.0".to_string()),
            pci_vendor_id: Some(0x10de),
        }
    }

    fn nes_unit_test_declared_resident() -> SystemMemoryAllocationQuote {
        SystemMemoryAllocationQuote::new("nes-unit-test.declared-resident", 64 * 1024, 64 * 1024)
            .expect("nes unit-test declared resident")
    }

    fn run_execution_candidate_attempt<T, E>(
        services: &NativeExecutionServices,
        candidate: &ExecutionCandidate,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> ExecutionCandidateAttemptOutcome<T, E> {
        let _guard = current_candidate_activation_quote().is_none().then(|| {
            install_candidate_activation_quote(CandidateActivationQuoteSource::Declared(
                nes_unit_test_declared_resident(),
            ))
        });
        super::run_execution_candidate_attempt(services, candidate, operation)
    }

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
                memory: Some(DeviceMemorySnapshot {
                    free_bytes: 8 * 1024 * 1024 * 1024,
                    total_bytes: 8 * 1024 * 1024 * 1024,
                    confidence: MemoryObservationConfidence::DeviceSnapshot,
                }),
                buffer_alignment: None,
            },
            placement,
        }
    }

    #[test]
    fn candidate_lane_ignores_ambient_override_and_rejects_backend_drift() {
        let candidate = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:00:02.0",
            ExecutionPlacement::FullDevice,
        );
        let ambient = gpu_candidate(
            ExecutionProvider::Vulkan,
            "Vulkan0",
            "0000:00:03.0",
            ExecutionPlacement::Hybrid,
        );
        let _backend_guard = install_request_backend_override(Some(
            RequestBackendPreference::Exact(ambient.device.route.clone()),
        ));

        let lane = ExecutionLaneKey::from_candidate(&candidate, GgmlCpuGraphBackend::Gpu)
            .expect("candidate route must construct the lane");
        let context = crate::RequestExecutionContext::uncancellable("candidate lane test")
            .with_native_execution_lane(lane.clone());
        assert_eq!(lane.provider(), ExecutionProvider::Cuda);
        assert_eq!(lane.stable_device_id(), "CUDA0");
        assert_eq!(lane.placement(), ExecutionPlacement::FullDevice);
        assert_eq!(context.native_execution_lane(), Some(&lane));
        assert!(
            ExecutionLaneKey::from_candidate(&candidate, GgmlCpuGraphBackend::Metal).is_err(),
            "a CUDA candidate must not be relabeled as Metal"
        );
    }

    #[test]
    fn resolved_lane_guard_restores_backend_placement_and_lane_as_one_scope() {
        let outer = ExecutionLaneKey::from_candidate(
            &gpu_candidate(
                ExecutionProvider::Vulkan,
                "Vulkan0",
                "0000:00:03.0",
                ExecutionPlacement::Hybrid,
            ),
            GgmlCpuGraphBackend::Gpu,
        )
        .expect("outer lane");
        let inner = ExecutionLaneKey::from_candidate(
            &gpu_candidate(
                ExecutionProvider::Cuda,
                "CUDA1",
                "0000:00:04.0",
                ExecutionPlacement::FullDevice,
            ),
            GgmlCpuGraphBackend::Gpu,
        )
        .expect("inner lane");
        let original_backend = request_backend_override();
        let original_placement = current_execution_placement();
        let original_lane = current_execution_lane();

        {
            let _outer = install_resolved_execution_lane(outer.clone());
            assert_eq!(
                request_backend_override(),
                Some(outer.request_backend_preference())
            );
            assert_eq!(current_execution_placement(), Some(outer.placement()));
            assert_eq!(current_execution_lane(), Some(outer.clone()));

            {
                let _inner = install_resolved_execution_lane(inner.clone());
                assert_eq!(
                    request_backend_override(),
                    Some(inner.request_backend_preference())
                );
                assert_eq!(current_execution_placement(), Some(inner.placement()));
                assert_eq!(current_execution_lane(), Some(inner));
            }

            assert_eq!(
                request_backend_override(),
                Some(outer.request_backend_preference())
            );
            assert_eq!(current_execution_placement(), Some(outer.placement()));
            assert_eq!(current_execution_lane(), Some(outer));
        }

        assert_eq!(request_backend_override(), original_backend);
        assert_eq!(current_execution_placement(), original_placement);
        assert_eq!(current_execution_lane(), original_lane);
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
            let scope_id = services.scope_id();
            std::thread::spawn(move || {
                let _guard = install_native_execution_context(context);
                assert_eq!(
                    current_runtime_receipts()
                        .expect("propagated receipt collector")
                        .snapshot()
                        .scope_id,
                    scope_id
                );
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
    fn candidate_context_propagates_observation_sink_to_worker() {
        let services = test_native_execution_services();
        let candidate = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::FullDevice,
        );
        let observations = ExecutionObservationSink::new();
        let _observation_guard = install_execution_observation_sink(observations.clone());
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let context = current_native_execution_context().expect("candidate context");
            std::thread::spawn(move || {
                let _guard = install_native_execution_context(context);
                record_current_execution_backend_observation(
                    1,
                    GgmlCpuGraphBackend::Gpu,
                    "CUDA0",
                    ExecutionProvider::Cuda,
                    "CUDA0",
                    &test_cuda_device(),
                    true,
                );
            })
            .join()
            .unwrap();
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert_eq!(
            observations.observations(),
            vec![ExecutionBackendObservation {
                requested_route: candidate.device.route,
                placement: ExecutionPlacement::FullDevice,
                backend_kind: GgmlCpuGraphBackend::Gpu,
                backend_name: "CUDA0".to_string(),
                actual_provider: ExecutionProvider::Cuda,
                actual_stable_id: "CUDA0".to_string(),
                actual_device: test_cuda_device(),
                use_scheduler: true,
                backend_identity: 1,
                memory_receipts: Vec::new(),
            }]
        );
    }

    #[test]
    fn observation_sink_publishes_only_the_committed_candidate_attempt() {
        let services = test_native_execution_services();
        let full_device = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::FullDevice,
        );
        let hybrid = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::Hybrid,
        );
        let observations = ExecutionObservationSink::new();
        let _observation_guard = install_execution_observation_sink(observations.clone());

        let failed = run_execution_candidate_attempt(services.as_ref(), &full_device, || {
            record_current_execution_backend_observation(
                1,
                GgmlCpuGraphBackend::Gpu,
                "CUDA0",
                ExecutionProvider::Cuda,
                "CUDA0",
                &test_cuda_device(),
                false,
            );
            record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
                "full-device-allocation",
                "synthetic capacity rejection",
            ));
            Err::<(), _>("failed")
        });
        assert!(failed.result.is_err());
        assert!(failed.candidate_failure.is_some());
        assert!(observations.observations().is_empty());

        let succeeded = run_execution_candidate_attempt(services.as_ref(), &hybrid, || {
            record_current_execution_backend_observation(
                2,
                GgmlCpuGraphBackend::Gpu,
                "CUDA0",
                ExecutionProvider::Cuda,
                "CUDA0",
                &test_cuda_device(),
                false,
            );
            Ok::<_, &str>(())
        });
        assert!(succeeded.result.is_ok());
        assert!(succeeded.candidate_failure.is_none());
        assert_eq!(observations.observations().len(), 1);
        assert_eq!(
            observations.observations()[0].placement,
            ExecutionPlacement::Hybrid
        );
    }

    #[test]
    fn exact_accelerated_backend_attestation_enforces_full_device_and_route_contracts() {
        let services = test_native_execution_services();
        let full_device = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::FullDevice,
        );
        let hybrid = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA0",
            "0000:01:00.0",
            ExecutionPlacement::Hybrid,
        );
        let attest =
            |candidate: &ExecutionCandidate, backend_kind, provider, stable_id, use_scheduler| {
                let sink = ExecutionCandidateFailureSink::new();
                let _guard =
                    install_execution_candidate_attempt(services.as_ref(), candidate, sink);
                attest_current_exact_accelerated_backend(
                    backend_kind,
                    provider,
                    stable_id,
                    use_scheduler,
                )
            };

        assert!(
            attest(
                &full_device,
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Cuda,
                "CUDA0",
                false,
            )
            .is_ok()
        );
        assert!(
            attest(
                &full_device,
                GgmlCpuGraphBackend::Cpu,
                ExecutionProvider::Cpu,
                "CPU",
                false,
            )
            .is_err()
        );
        assert!(
            attest(
                &full_device,
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Cuda,
                "CUDA0",
                true,
            )
            .is_err()
        );
        assert!(
            attest(
                &full_device,
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Vulkan,
                "Vulkan0",
                false,
            )
            .is_err()
        );
        assert!(
            attest(
                &full_device,
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Cuda,
                "CUDA1",
                false,
            )
            .is_err()
        );
        assert!(
            attest(
                &hybrid,
                GgmlCpuGraphBackend::Cpu,
                ExecutionProvider::Cpu,
                "CPU",
                false,
            )
            .is_ok()
        );

        let _backend_guard = install_request_backend_override(Some(
            RequestBackendPreference::Exact(full_device.device.route.clone()),
        ));
        assert!(
            attest_current_exact_accelerated_backend(
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Cuda,
                "CUDA0",
                false,
            )
            .is_ok(),
            "unscoped Exact GPU runner must retain identity attestation"
        );
        assert!(
            attest_current_exact_accelerated_backend(
                GgmlCpuGraphBackend::Gpu,
                ExecutionProvider::Cuda,
                "CUDA1",
                false,
            )
            .is_err(),
            "unscoped Exact GPU runner must reject a route mismatch"
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
    fn nested_candidate_attempts_keep_parent_activation_cohort() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let parent = ActivationReservationContext::mint();
        let parent_id = parent.cohort_id;
        let _guard = install_activation_reservation_context(Some(parent));
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            assert_eq!(
                current_memory_reservation_cohort_id().expect("attempt cohort"),
                parent_id
            );
            let nested = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
                Ok::<_, ()>(current_memory_reservation_cohort_id().expect("nested cohort"))
            });
            assert_eq!(nested.result.unwrap(), parent_id);
            let context = current_native_execution_context().expect("attempt context");
            let worker = std::thread::spawn(move || {
                let _guard = install_native_execution_context(context);
                current_memory_reservation_cohort_id().expect("worker cohort")
            })
            .join()
            .unwrap();
            assert_eq!(worker, parent_id);
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
    }

    #[test]
    fn native_root_receipts_capture_system_memory_owner_lifecycle() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let cache = super::super::admitted_host_object_cache::AdmittedHostObjectCache::new(
                super::super::admitted_host_object_cache::AdmittedHostObjectCacheLimits::new(1, 8),
            );
            let first = cache
                .get_or_try_insert_with(
                    "receipt-system-memory",
                    || Ok::<_, String>((1, ())),
                    |()| {
                        super::super::system_memory_owner::SystemMemoryOwner::try_allocate(
                            super::super::system_memory_owner::SystemMemoryAllocationQuote::new(
                                "receipt-system-memory-fixture",
                                1,
                                1,
                            )
                            .expect("fixture quote"),
                            || {
                                Ok(super::super::system_memory_owner::SystemMemoryAllocationOutcome::new(
                                    vec![0_u8; 1],
                                    1,
                                    1,
                                ))
                            },
                        )
                        .map(Arc::new)
                        .map_err(|error| error.to_string())
                    },
                    || "cache lock poisoned".to_string(),
                )
                .expect("one-byte host owner should be admitted");
            let second = cache
                .ready(&"receipt-system-memory")
                .expect("published host owner should be reusable");
            assert!(Arc::ptr_eq(&first, &second));
            let live = services.runtime_receipts().summary();
            // One owner is the candidate-level forecast reservation; the
            // other is the committed SystemMemory allocation. Both must be
            // visible while the attempt is active.
            assert_eq!(live.live_owner_count, 2);
            assert_eq!(live.live_resource_count, 2);
            assert!(
                services
                    .runtime_receipts()
                    .snapshot()
                    .events
                    .iter()
                    .any(|event| matches!(
                        event,
                        super::super::runtime_receipts::RuntimeReceiptEvent::OwnerReused { .. }
                    ))
            );
            drop(first);
            drop(second);
            cache.clear();
            let released = services.runtime_receipts().summary();
            assert_eq!(released.live_owner_count, 1);
            assert_eq!(released.live_resource_count, 1);
            Ok::<_, String>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
    }
    #[test]
    fn entropy_failure_keeps_service_execution_alive_and_reports_unavailable() {
        let services = test_native_execution_services_with_entropy_failure();
        let snapshot = services.runtime_receipts().snapshot();
        assert!(matches!(
            snapshot.availability,
            super::super::runtime_receipts::RuntimeReceiptAvailability::Unavailable {
                reason: super::super::runtime_receipts::RuntimeReceiptUnavailableReason::EntropyUnavailable,
            }
        ));
        assert!(!snapshot.completeness.complete);
        assert_eq!(snapshot.live_owners.len(), 0);

        let candidate = cpu_candidate();
        let outcome = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let owner =
                super::super::system_memory_owner::SystemMemoryOwner::try_reserve_invocation(
                    "entropy-failure-execution-fixture",
                    1,
                )
                .map_err(|error| error.to_string())?;
            assert_eq!(owner.committed_requested_bytes(), 1);
            drop(owner);
            Ok::<_, String>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
        assert!(matches!(
            services.runtime_receipts().summary().completeness.reason,
            Some(super::super::runtime_receipts::ReceiptCompletenessReason::Unavailable(
                super::super::runtime_receipts::RuntimeReceiptUnavailableReason::EntropyUnavailable
            ))
        ));
    }

    #[test]
    fn independently_constructed_local_roots_share_the_process_memory_ledger() {
        let first = NativeExecutionServices::for_local_process().unwrap();
        let second = NativeExecutionServices::for_local_process().unwrap();
        assert_ne!(first.scope_id(), second.scope_id());
        assert!(Arc::ptr_eq(first.memory_broker(), second.memory_broker()));
        assert_eq!(
            first.runtime_receipts().snapshot().scope_id,
            first.scope_id()
        );
        assert_eq!(
            second.runtime_receipts().snapshot().scope_id,
            second.scope_id()
        );
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
            assert_eq!(
                current_execution_lane(),
                ExecutionLaneKey::from_candidate(candidate, GgmlCpuGraphBackend::Gpu).ok()
            );
            current_execution_lane_key(GgmlCpuGraphBackend::Gpu)
        };
        let cuda0_lane = lane_for(&cuda0);
        assert_ne!(cuda0_lane, lane_for(&cuda1));
        assert_ne!(cuda0_lane, lane_for(&hip0));
        assert_ne!(cuda0_lane, lane_for(&hybrid));
        assert_eq!(cuda0_lane.backend(), GgmlCpuGraphBackend::Gpu);
        assert_eq!(cuda0_lane.placement(), ExecutionPlacement::FullDevice);
        let projection = cuda0_lane
            .receipt_projection(services.runtime_receipts())
            .expect("entropy-backed lane projection");
        assert_eq!(projection.provider, ExecutionProvider::Cuda);
        assert_eq!(projection.placement, ExecutionPlacement::FullDevice);
    }

    #[test]
    fn execution_lane_memory_sample_is_exact_and_enumeration_order_invariant() {
        let candidate = gpu_candidate(
            ExecutionProvider::Cuda,
            "CUDA1",
            "0000:02:00.0",
            ExecutionPlacement::FullDevice,
        );
        let lane = ExecutionLaneKey::from_candidate(&candidate, GgmlCpuGraphBackend::Gpu)
            .expect("exact CUDA lane");
        let vulkan = (
            "Vulkan0".to_string(),
            GgmlBackendKind::Gpu,
            Some(GgmlDeviceMemory {
                free_bytes: 3 * 1024 * 1024,
                total_bytes: 7 * 1024 * 1024,
            }),
        );
        let cuda0 = (
            "CUDA0".to_string(),
            GgmlBackendKind::Gpu,
            Some(GgmlDeviceMemory {
                free_bytes: 5 * 1024 * 1024,
                total_bytes: 8 * 1024 * 1024,
            }),
        );
        let cuda1 = (
            "CUDA1".to_string(),
            GgmlBackendKind::Gpu,
            Some(GgmlDeviceMemory {
                free_bytes: 11 * 1024 * 1024,
                total_bytes: 16 * 1024 * 1024,
            }),
        );

        for devices in [
            vec![vulkan.clone(), cuda0.clone(), cuda1.clone()],
            vec![cuda1.clone(), cuda0.clone(), vulkan.clone()],
        ] {
            let sample = exact_lane_memory_sample_from_device_infos(&lane, &devices)
                .expect("exact CUDA1 memory sample");
            assert_eq!(sample.provider, ExecutionProvider::Cuda);
            assert_eq!(sample.stable_device_id, "CUDA1");
            assert_eq!(sample.memory.free_bytes, 11 * 1024 * 1024);
            assert_eq!(sample.memory.total_bytes, 16 * 1024 * 1024);
        }

        assert!(
            exact_lane_memory_sample_from_device_infos(
                &lane,
                &[("CUDA1".to_string(), GgmlBackendKind::Gpu, None)],
            )
            .is_none()
        );
        assert!(exact_lane_memory_sample_from_device_infos(&lane, &[vulkan, cuda0]).is_none());
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

    fn shipped_activation_verified_pack() -> VerifiedPack {
        static PACK: OnceLock<VerifiedPack> = OnceLock::new();
        PACK.get_or_init(|| {
            let directory = Box::leak(Box::new(tempfile::tempdir().expect("activation pack dir")));
            let path = directory.path().join("activation-quote.gguf");
            crate::testing::write_tiny_gguf_runtime_source(
                &path,
                &crate::testing::TinyGgufFixtureSpec::whisper_oasr_v1_graph_ready_for_tokenizer_fail_closed(
                    "whisper-tiny",
                ),
            )
            .expect("write activation quote fixture");
            let preflight = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
                .expect("activation quote fixture must pass preflight");
            assert!(
                preflight.runtime_source().byte_len() > 4096,
                "activation quote fixture must exceed a placeholder page, got {}",
                preflight.runtime_source().byte_len()
            );
            crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                preflight,
                crate::WHISPER_GGML_ARCHITECTURE_ID,
            )
        })
        .clone()
    }

    fn auxiliary_activation_verified_pack() -> (tempfile::TempDir, VerifiedPack) {
        let directory = tempfile::tempdir().expect("aux activation pack dir");
        let path = directory.path().join("redimnet2-activation-quote.oasr");
        let tensor = crate::ggml_runtime::GgufWriteTensor {
            name: "fixture.tensor".to_string(),
            dims: vec![1],
            tensor_type: crate::ggml_runtime::GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        };
        crate::models::oasr_metadata::OasrPackWriter::write(
            &path,
            crate::models::oasr_metadata::PackEnvelope::aux(
                crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID,
            ),
            std::collections::BTreeMap::new(),
            &[tensor],
        )
        .expect("write aux activation quote fixture");
        let preflight = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
            .expect("aux activation quote fixture preflight");
        let pack = VerifiedPack::from_unverified_aux_preflight_for_test(
            preflight,
            crate::models::aux_pack_registry::AuxPackKind::Diarization,
            crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID,
        );
        (directory, pack)
    }

    #[test]
    fn auxiliary_pack_activation_uses_aux_registry_without_asr_descriptor() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let (_directory, pack) = auxiliary_activation_verified_pack();
        assert!(
            crate::arch::OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(pack.model_architecture())
                .is_none(),
            "the aux pack must not masquerade as an ASR descriptor"
        );
        let broker = Arc::clone(services.memory_broker());
        let before = broker.usage(&MemoryDomainKey::SystemMemory);
        let reservation =
            quote_and_reserve_candidate_activation(services.as_ref(), &candidate, &pack)
                .expect("registered auxiliary pack must receive a physical-domain quote");
        let during = broker.usage(&MemoryDomainKey::SystemMemory);
        assert!(
            during.pending_bytes > before.pending_bytes
                || during.committed_bytes > before.committed_bytes,
            "auxiliary activation must reserve nonzero physical bytes"
        );
        drop(reservation);
        assert_eq!(broker.usage(&MemoryDomainKey::SystemMemory), before);
    }

    #[test]
    fn shipped_candidate_attempt_reserves_then_releases_on_rollback() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let pack = shipped_activation_verified_pack();
        let quoted_peak = {
            let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(architecture_id_from_pack(&pack).unwrap())
                .unwrap();
            let topology = resolve_resident_topology_plan(
                descriptor,
                &pack,
                &candidate,
                &ExecutionIntent::CpuOnly,
                true,
            )
            .unwrap();
            let (_backends, _mapping, plan) =
                quote_candidate_activation_plan(&pack, &candidate, &topology)
                    .expect("activation footprint must be quotable");
            let peak = plan
                .reservation_requests()
                .iter()
                .map(|request| request.peak_bytes)
                .sum::<u64>();
            assert!(
                peak > 4096,
                "quoted activation peak must exceed a placeholder page, got {peak}"
            );
            assert!(
                plan.reservation_requests()
                    .iter()
                    .all(|request| request.peak_bytes > 0),
                "quoted activation domains must not be reserved as zero"
            );
            peak
        };
        let _pack = install_candidate_activation_pack(pack);
        let broker = Arc::clone(services.memory_broker());
        let before = broker.usage(&MemoryDomainKey::SystemMemory);
        let during = run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let usage = broker.usage(&MemoryDomainKey::SystemMemory);
            let charged = usage
                .pending_bytes
                .saturating_sub(before.pending_bytes)
                .max(usage.committed_bytes.saturating_sub(before.committed_bytes));
            assert_eq!(
                charged, quoted_peak,
                "broker pending/committed must match the ggml quote, got {usage:?} before {before:?}"
            );
            assert!(
                charged > 4096,
                "reserved activation peak must exceed a placeholder page, got {charged}"
            );
            record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
                "candidate-activation-reserve-test",
                "force rollback after reservation",
            ));
            Ok::<_, ()>(usage)
        });
        assert!(during.result.is_ok());
        assert!(during.candidate_failure.is_some());
        let after = broker.usage(&MemoryDomainKey::SystemMemory);
        assert_eq!(
            after,
            DeviceMemoryUsage {
                pending_bytes: before.pending_bytes,
                committed_bytes: before.committed_bytes,
                unreclaimable_bytes: before.unreclaimable_bytes,
                exclusive_pending: before.exclusive_pending,
                quarantined: before.quarantined,
            },
            "rollback must release the candidate reservation"
        );
    }

    #[test]
    fn packless_stream_vad_aux_attempt_reserves_declared_weight_bytes() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        assert!(
            current_candidate_activation_pack().is_none(),
            "this path must not inherit an ASR VerifiedPack"
        );
        let declared = crate::diarize::vad::FireRedStreamVadModel::system_memory_quote()
            .expect("Stream-VAD embedded quote");
        assert!(
            declared.peak_bytes > 4096 && declared.retained_bytes > 4096,
            "declared Stream-VAD WEIGHTS quote must exceed a placeholder page, got peak={} retained={}",
            declared.peak_bytes,
            declared.retained_bytes
        );
        let quoted_peak = {
            let (_backend, plan) = quote_cpu_buffer_plan(
                &declared.resource_id,
                declared.peak_bytes.max(declared.retained_bytes),
            )
            .expect("declared Stream-VAD bytes must be ggml-quotable");
            let peak = plan
                .reservation_requests()
                .iter()
                .map(|request| request.peak_bytes)
                .sum::<u64>();
            assert!(
                peak > 4096,
                "ggml quote of declared Stream-VAD bytes must exceed a placeholder page, got {peak}"
            );
            peak
        };
        let _quote = install_candidate_activation_quote(CandidateActivationQuoteSource::Declared(
            declared.clone(),
        ));
        let broker = Arc::clone(services.memory_broker());
        let before = broker.usage(&MemoryDomainKey::SystemMemory);
        let outcome = super::run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let usage = broker.usage(&MemoryDomainKey::SystemMemory);
            let charged = usage
                .pending_bytes
                .saturating_sub(before.pending_bytes)
                .max(usage.committed_bytes.saturating_sub(before.committed_bytes));
            assert_eq!(
                charged, quoted_peak,
                "packless Stream-VAD reserve must match its declared WEIGHTS ggml quote, got {usage:?} before {before:?}"
            );
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert!(
            outcome.candidate_failure.is_none(),
            "packless Stream-VAD/aux must not capacity-exhaust, got {:?}",
            outcome.candidate_failure
        );
    }

    #[test]
    fn packless_punc_declared_quote_is_not_stream_vad_blob() {
        let services = test_native_execution_services();
        let candidate = cpu_candidate();
        let vad = crate::diarize::vad::FireRedStreamVadModel::system_memory_quote()
            .expect("Stream-VAD embedded quote");
        let punc = SystemMemoryAllocationQuote::new(
            "aux.firered-punc.test.declared-resident",
            128 * 1024,
            96 * 1024,
        )
        .expect("punc declared resident");
        assert_ne!(
            punc.peak_bytes, vad.peak_bytes,
            "punc declared peak must not steal the Stream-VAD WEIGHTS blob"
        );
        let punc_quoted = {
            let (_backend, plan) =
                quote_cpu_buffer_plan(&punc.resource_id, punc.peak_bytes.max(punc.retained_bytes))
                    .expect("punc declared bytes must be ggml-quotable");
            plan.reservation_requests()
                .iter()
                .map(|request| request.peak_bytes)
                .sum::<u64>()
        };
        let vad_quoted = {
            let (_backend, plan) =
                quote_cpu_buffer_plan(&vad.resource_id, vad.peak_bytes.max(vad.retained_bytes))
                    .expect("vad declared bytes must be ggml-quotable");
            plan.reservation_requests()
                .iter()
                .map(|request| request.peak_bytes)
                .sum::<u64>()
        };
        assert_ne!(punc_quoted, vad_quoted);
        let _quote =
            install_candidate_activation_quote(CandidateActivationQuoteSource::Declared(punc));
        let broker = Arc::clone(services.memory_broker());
        let before = broker.usage(&MemoryDomainKey::SystemMemory);
        let outcome = super::run_execution_candidate_attempt(services.as_ref(), &candidate, || {
            let usage = broker.usage(&MemoryDomainKey::SystemMemory);
            let charged = usage
                .pending_bytes
                .saturating_sub(before.pending_bytes)
                .max(usage.committed_bytes.saturating_sub(before.committed_bytes));
            assert_eq!(charged, punc_quoted);
            assert_ne!(charged, vad_quoted);
            Ok::<_, ()>(())
        });
        assert!(outcome.result.is_ok());
        assert!(
            outcome.candidate_failure.is_none(),
            "packless punc declared quote must not capacity-exhaust, got {:?}",
            outcome.candidate_failure
        );
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
