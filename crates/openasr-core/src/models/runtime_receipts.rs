//! Bounded, production-safe diagnostics for native runtime ownership.
//!
//! Receipts are diagnostic evidence only. Admission, candidate ordering, and
//! fallback never read this module's event stream or snapshot.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt::{self, Write as _},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::RequestAttemptId;
use crate::device::execution_memory::{
    DeviceMemoryBrokerSet, MemoryDomainKey, MemoryObservationConfidence, QuoteConfidence,
};
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::native_execution_services::{
    ExecutionCacheAttemptId, NativeExecutionScopeId, current_execution_cache_attempt_id,
    current_request_attempt_id,
};

/// Schema marker for the phase-0 in-process ownership evidence.
pub const RUNTIME_RECEIPT_SCHEMA: &str = "openasr.runtime-ownership-receipt.v1";
/// One advertised short-audio family decode currently emits ~14k owner/resource
/// events at ggml-context granularity. The ring must hold that request window;
/// overflow remains fail-closed for qualification.
const DEFAULT_EVENT_CAPACITY: usize = 16_384;
const MAX_EVENT_CAPACITY: usize = 32_768;
const MAX_LIVE_OWNERS: usize = 1024;
const MAX_RESOURCES_PER_OWNER: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeReceiptUnavailableReason {
    EntropyUnavailable,
    IdentityExhausted,
}

impl RuntimeReceiptUnavailableReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EntropyUnavailable => "entropy-unavailable",
            Self::IdentityExhausted => "identity-exhausted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeReceiptAvailability {
    Available,
    Unavailable {
        reason: RuntimeReceiptUnavailableReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeReceiptMetric {
    Known(u64),
    /// The provider cannot supply this metric.
    Unavailable,
    /// The component has not been physically priced yet.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeBackendOwnedReliability {
    Complete,
    Incomplete,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeResourceState {
    Reserved,
    Reconciled,
    Committed,
    Quarantined,
    Released,
}

/// Safe native evidence attached to a reservation resource. Values are
/// projections only: unavailable fields remain typed unavailable and no raw
/// backend identity, pointer, or path reaches this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeNativeMemoryEvidence {
    pub domain_kind: Option<SafeMemoryDomainKind>,
    pub provider: Option<ExecutionProvider>,
    pub backend_owned_reliability: RuntimeBackendOwnedReliability,
    pub heap_index: Option<u32>,
    pub total_bytes: RuntimeReceiptMetric,
    pub budget_bytes: RuntimeReceiptMetric,
    pub free_bytes: RuntimeReceiptMetric,
    pub used_bytes: RuntimeReceiptMetric,
    pub backend_owned_live_bytes: RuntimeReceiptMetric,
    pub backend_owned_cached_bytes: RuntimeReceiptMetric,
    pub backend_owned_workspace_bytes: RuntimeReceiptMetric,
    pub backend_owned_high_water_bytes: RuntimeReceiptMetric,
    pub stats_generation: RuntimeReceiptMetric,
    pub quote_generation: RuntimeReceiptMetric,
    pub claim_flags: u32,
    pub observation_confidence: Option<MemoryObservationConfidence>,
    pub broker_pending_bytes: RuntimeReceiptMetric,
    pub broker_committed_bytes: RuntimeReceiptMetric,
    pub broker_unreclaimable_bytes: RuntimeReceiptMetric,
}

impl RuntimeReceiptAvailability {
    pub(crate) const fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReceiptCompletenessReason {
    Unavailable(RuntimeReceiptUnavailableReason),
    IdentityExhausted,
    EventCapacityExceeded,
    OwnerCapacityExceeded,
    ResourceCapacityExceeded,
    NotificationCapacityExceeded,
    InvalidLifecycle,
}

impl ReceiptCompletenessReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unavailable(reason) => reason.as_str(),
            Self::IdentityExhausted => "identity-exhausted",
            Self::EventCapacityExceeded => "event-capacity-exceeded",
            Self::OwnerCapacityExceeded => "owner-capacity-exceeded",
            Self::ResourceCapacityExceeded => "resource-capacity-exceeded",
            Self::NotificationCapacityExceeded => "notification-capacity-exceeded",
            Self::InvalidLifecycle => "invalid-lifecycle",
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeReceiptError {
    #[error("runtime receipt event capacity {requested} exceeds maximum {maximum}")]
    CapacityTooLarge { requested: usize, maximum: usize },
    #[error("runtime receipt event capacity must be non-zero")]
    ZeroCapacity,
}

/// A keyed, 128-bit projection of an identity. The key is random per service
/// root and is never retained in a snapshot or exported through this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct RedactedIdentity([u8; 16]);

impl RedactedIdentity {
    pub fn to_hex(self) -> String {
        let mut encoded = String::with_capacity(32);
        for byte in self.0 {
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }
}

/// The safe domain vocabulary used by receipt snapshots. The physical device
/// identity is represented only by `join_id`; the original domain is never
/// stored in a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum SafeMemoryDomainKind {
    SystemMemory,
    DedicatedDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SafeMemoryDomainProjection {
    pub kind: SafeMemoryDomainKind,
    pub heap: Option<u32>,
    pub join_id: RedactedIdentity,
}

/// Redacted execution-lane identity. Provider, placement, and backend reuse
/// the runtime's typed vocabulary; the provider-local device name is keyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SafeExecutionLaneProjection {
    pub provider: ExecutionProvider,
    pub placement: ExecutionPlacement,
    pub backend: GgmlCpuGraphBackend,
    pub device: RedactedIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "kind", content = "lane", rename_all = "kebab-case")]
pub enum RuntimeOwnerPlacement {
    HostNeutral,
    LaneBound(SafeExecutionLaneProjection),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "domain", rename_all = "kebab-case")]
pub enum RuntimeResourceLedgerBinding {
    Brokered(SafeMemoryDomainProjection),
    NoBrokerLease,
    Unknown,
}

/// Stable owner identity within one service root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct RuntimeOwnerId {
    pub scope_id: NativeExecutionScopeId,
    pub ordinal: u64,
}

/// Stable resource identity within one service root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct RuntimeResourceId {
    pub scope_id: NativeExecutionScopeId,
    pub ordinal: u64,
}

/// Safe owner metadata. All free-form identifiers are projected with the
/// service root's keyed digest before they reach this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RuntimeOwnerDescriptor {
    pub component: RedactedIdentity,
    pub content: Option<RedactedIdentity>,
    pub source: Option<RedactedIdentity>,
    pub placement: RuntimeOwnerPlacement,
}

/// Safe resource metadata. Domain and confidence use the existing admission
/// vocabulary where those values are meaningful; the domain identity itself is
/// projected into [`SafeMemoryDomainProjection`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeResourceDescriptor {
    pub kind: RedactedIdentity,
    /// Diagnostic attribution captured on the owner/resource relationship.
    /// Broker admission never reads this field.
    pub placement: RuntimeOwnerPlacement,
    pub domain: Option<SafeMemoryDomainProjection>,
    pub ledger_binding: RuntimeResourceLedgerBinding,
    pub requested: RuntimeReceiptMetric,
    pub peak: RuntimeReceiptMetric,
    pub retained: RuntimeReceiptMetric,
    pub quote_confidence: QuoteConfidence,
    pub observation_confidence: Option<MemoryObservationConfidence>,
    pub native: Option<RuntimeNativeMemoryEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum RuntimeReceiptEvent {
    OwnerCreated {
        owner_id: RuntimeOwnerId,
        descriptor: RuntimeOwnerDescriptor,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
    OwnerReused {
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
    OwnerReleased {
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
    ResourceAcquired {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        descriptor: RuntimeResourceDescriptor,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
    ResourceStateChanged {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        state: RuntimeResourceState,
        descriptor: RuntimeResourceDescriptor,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
    ResourceReleased {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveRuntimeResource {
    pub id: RuntimeResourceId,
    pub descriptor: RuntimeResourceDescriptor,
    pub state: RuntimeResourceState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveRuntimeOwner {
    pub id: RuntimeOwnerId,
    pub descriptor: RuntimeOwnerDescriptor,
    pub resources: BTreeMap<RuntimeResourceId, LiveRuntimeResource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReceiptCompleteness {
    pub complete: bool,
    pub reason: Option<ReceiptCompletenessReason>,
    pub live_state_complete: bool,
    pub live_state_reason: Option<ReceiptCompletenessReason>,
    pub event_history_complete: bool,
    pub event_history_reason: Option<ReceiptCompletenessReason>,
    pub dropped_events: u64,
    pub dropped_owners: u64,
    pub rejected_resources: u64,
    pub dropped_notifications: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RuntimeReceiptSummary {
    pub scope_id: NativeExecutionScopeId,
    pub availability: RuntimeReceiptAvailability,
    pub live_owner_count: usize,
    pub live_resource_count: usize,
    pub event_count: usize,
    pub event_capacity: usize,
    pub completeness: ReceiptCompleteness,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeReceiptSnapshot {
    pub schema: &'static str,
    pub scope_id: NativeExecutionScopeId,
    pub availability: RuntimeReceiptAvailability,
    pub live_owners: Vec<LiveRuntimeOwner>,
    pub events: Vec<RuntimeReceiptEvent>,
    pub event_capacity: usize,
    pub completeness: ReceiptCompleteness,
}

struct RuntimeReceiptState {
    next_owner_ordinal: AtomicU64,
    next_resource_ordinal: AtomicU64,
    identity_exhausted: bool,
    live_owners: BTreeMap<RuntimeOwnerId, LiveRuntimeOwner>,
    events: VecDeque<RuntimeReceiptEvent>,
    dropped_events: u64,
    dropped_owners: u64,
    rejected_resources: u64,
    dropped_notifications: u64,
    live_state_complete: bool,
    live_state_reason: Option<ReceiptCompletenessReason>,
    event_history_complete: bool,
    event_history_reason: Option<ReceiptCompletenessReason>,
}

/// Bounded collector owned by one [`NativeExecutionServices`] root.
#[derive(Clone)]
pub struct RuntimeReceiptCollector {
    scope_id: NativeExecutionScopeId,
    key: Option<[u8; 32]>,
    availability: RuntimeReceiptAvailability,
    event_capacity: usize,
    state: Arc<Mutex<RuntimeReceiptState>>,
}

impl RuntimeReceiptCollector {
    pub(crate) fn new(scope_id: NativeExecutionScopeId) -> Self {
        Self::new_with_capacity_and_entropy(scope_id, DEFAULT_EVENT_CAPACITY, |key| {
            getrandom::fill(key).map_err(|_| ())
        })
        .expect("fixed runtime receipt capacity must be valid")
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        scope_id: NativeExecutionScopeId,
        event_capacity: usize,
    ) -> Result<Self, RuntimeReceiptError> {
        Self::new_with_capacity_and_entropy(scope_id, event_capacity, |key| {
            getrandom::fill(key).map_err(|_| ())
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_entropy_failure_for_test(scope_id: NativeExecutionScopeId) -> Self {
        Self::new_with_capacity_and_entropy(scope_id, DEFAULT_EVENT_CAPACITY, |_| Err(()))
            .expect("fixed runtime receipt capacity must be valid")
    }

    fn new_with_capacity_and_entropy(
        scope_id: NativeExecutionScopeId,
        event_capacity: usize,
        fill_entropy: impl FnOnce(&mut [u8; 32]) -> Result<(), ()>,
    ) -> Result<Self, RuntimeReceiptError> {
        Self::new_with_capacity_entropy_and_ordinals(scope_id, event_capacity, 1, 1, fill_entropy)
    }

    #[cfg(test)]
    fn new_for_test_with_ordinals(
        scope_id: NativeExecutionScopeId,
        event_capacity: usize,
        next_owner_ordinal: u64,
        next_resource_ordinal: u64,
    ) -> Result<Self, RuntimeReceiptError> {
        Self::new_with_capacity_entropy_and_ordinals(
            scope_id,
            event_capacity,
            next_owner_ordinal,
            next_resource_ordinal,
            |key| getrandom::fill(key).map_err(|_| ()),
        )
    }

    fn new_with_capacity_entropy_and_ordinals(
        scope_id: NativeExecutionScopeId,
        event_capacity: usize,
        next_owner_ordinal: u64,
        next_resource_ordinal: u64,
        fill_entropy: impl FnOnce(&mut [u8; 32]) -> Result<(), ()>,
    ) -> Result<Self, RuntimeReceiptError> {
        if event_capacity == 0 {
            return Err(RuntimeReceiptError::ZeroCapacity);
        }
        if event_capacity > MAX_EVENT_CAPACITY {
            return Err(RuntimeReceiptError::CapacityTooLarge {
                requested: event_capacity,
                maximum: MAX_EVENT_CAPACITY,
            });
        }
        let mut key = [0_u8; 32];
        let availability = match fill_entropy(&mut key) {
            Ok(()) => RuntimeReceiptAvailability::Available,
            Err(()) => RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::EntropyUnavailable,
            },
        };
        let live_state_reason =
            (!availability.is_available()).then_some(ReceiptCompletenessReason::Unavailable(
                RuntimeReceiptUnavailableReason::EntropyUnavailable,
            ));
        Ok(Self {
            scope_id,
            key: availability.is_available().then_some(key),
            availability,
            event_capacity,
            state: Arc::new(Mutex::new(RuntimeReceiptState {
                next_owner_ordinal: AtomicU64::new(next_owner_ordinal),
                next_resource_ordinal: AtomicU64::new(next_resource_ordinal),
                identity_exhausted: false,
                live_owners: BTreeMap::new(),
                // Bound is enforced on push; pre-filling 16k large events
                // would commit a constant ~host-MiB tax on every NES root
                // even when the request emits far fewer events.
                events: VecDeque::new(),
                dropped_events: 0,
                dropped_owners: 0,
                rejected_resources: 0,
                dropped_notifications: 0,
                live_state_complete: availability.is_available(),
                live_state_reason,
                event_history_complete: true,
                event_history_reason: None,
            })),
        })
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RuntimeReceiptState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn digest(&self, domain: &[u8], value: &str) -> Option<RedactedIdentity> {
        if !self.is_available() {
            return None;
        }
        let key = self.key?;
        let mut hasher = Sha256::new();
        hasher.update(b"openasr.runtime-receipt.v1");
        hasher.update((domain.len() as u64).to_be_bytes());
        hasher.update(domain);
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
        hasher.update(key);
        let digest = hasher.finalize();
        let mut redacted = [0_u8; 16];
        redacted.copy_from_slice(&digest[..16]);
        Some(RedactedIdentity(redacted))
    }

    pub(crate) fn owner_descriptor(
        &self,
        component: &str,
        content: Option<&str>,
        source: Option<&str>,
        lane: Option<SafeExecutionLaneProjection>,
    ) -> Option<RuntimeOwnerDescriptor> {
        Some(RuntimeOwnerDescriptor {
            component: self.digest(b"component", component)?,
            content: content.and_then(|value| self.digest(b"content", value)),
            source: source.and_then(|value| self.digest(b"source", value)),
            placement: lane.map_or(
                RuntimeOwnerPlacement::Unknown,
                RuntimeOwnerPlacement::LaneBound,
            ),
        })
    }

    pub(crate) fn host_neutral_owner_descriptor(
        &self,
        component: &str,
        content: Option<&str>,
        source: Option<&str>,
    ) -> Option<RuntimeOwnerDescriptor> {
        Some(RuntimeOwnerDescriptor {
            component: self.digest(b"component", component)?,
            content: content.and_then(|value| self.digest(b"content", value)),
            source: source.and_then(|value| self.digest(b"source", value)),
            placement: RuntimeOwnerPlacement::HostNeutral,
        })
    }

    pub(crate) fn resource_descriptor(
        &self,
        kind: &str,
        domain: &MemoryDomainKey,
        requested_bytes: u64,
        peak_bytes: u64,
        retained_bytes: u64,
        quote_confidence: QuoteConfidence,
        observation_confidence: Option<MemoryObservationConfidence>,
    ) -> Option<RuntimeResourceDescriptor> {
        let domain = self.domain_projection(domain)?;
        Some(RuntimeResourceDescriptor {
            kind: self.digest(b"resource-kind", kind)?,
            placement: RuntimeOwnerPlacement::Unknown,
            domain: Some(domain),
            ledger_binding: RuntimeResourceLedgerBinding::Brokered(domain),
            requested: RuntimeReceiptMetric::Known(requested_bytes),
            peak: RuntimeReceiptMetric::Known(peak_bytes),
            retained: RuntimeReceiptMetric::Known(retained_bytes),
            quote_confidence,
            observation_confidence,
            native: None,
        })
    }

    pub(crate) fn with_native_evidence(
        mut descriptor: RuntimeResourceDescriptor,
        native: RuntimeNativeMemoryEvidence,
    ) -> RuntimeResourceDescriptor {
        descriptor.native = Some(native);
        descriptor
    }

    /// Test fixture for proving that unknown memory never compares as zero.
    /// Production owner shapes must use a brokered descriptor or an explicit
    /// semantic [`RuntimeResourceLedgerBinding::NoBrokerLease`] marker.
    #[cfg(test)]
    pub(crate) fn unpriced_resource_descriptor(
        &self,
        kind: &str,
    ) -> Option<RuntimeResourceDescriptor> {
        Some(RuntimeResourceDescriptor {
            kind: self.digest(b"resource-kind", kind)?,
            placement: RuntimeOwnerPlacement::Unknown,
            domain: None,
            ledger_binding: RuntimeResourceLedgerBinding::Unknown,
            requested: RuntimeReceiptMetric::Unknown,
            peak: RuntimeReceiptMetric::Unknown,
            retained: RuntimeReceiptMetric::Unknown,
            quote_confidence: QuoteConfidence::Unknown,
            observation_confidence: None,
            native: None,
        })
    }

    /// Semantic lifetime marker for an object whose physical allocations are
    /// owned and receipted by child broker leases. This descriptor contributes
    /// no bytes and is never a substitute for pricing a native/system-memory
    /// owner; live reconciliation simply skips it after validating the typed
    /// `NoBrokerLease` binding.
    pub(crate) fn no_broker_resource_descriptor(
        &self,
        kind: &str,
    ) -> Option<RuntimeResourceDescriptor> {
        Some(RuntimeResourceDescriptor {
            kind: self.digest(b"resource-kind", kind)?,
            placement: RuntimeOwnerPlacement::Unknown,
            domain: None,
            ledger_binding: RuntimeResourceLedgerBinding::NoBrokerLease,
            requested: RuntimeReceiptMetric::Known(0),
            peak: RuntimeReceiptMetric::Known(0),
            retained: RuntimeReceiptMetric::Known(0),
            quote_confidence: QuoteConfidence::ExactCommitted,
            observation_confidence: None,
            native: None,
        })
    }

    pub(crate) fn lane_projection(
        &self,
        provider: ExecutionProvider,
        stable_device_id: &str,
        placement: ExecutionPlacement,
        backend: GgmlCpuGraphBackend,
    ) -> Option<SafeExecutionLaneProjection> {
        Some(SafeExecutionLaneProjection {
            provider,
            placement,
            backend,
            device: self.digest(b"lane-device", stable_device_id)?,
        })
    }

    fn domain_projection(&self, domain: &MemoryDomainKey) -> Option<SafeMemoryDomainProjection> {
        match domain {
            MemoryDomainKey::SystemMemory => Some(SafeMemoryDomainProjection {
                kind: SafeMemoryDomainKind::SystemMemory,
                heap: None,
                join_id: self.digest(b"domain-system-memory", "system-memory")?,
            }),
            MemoryDomainKey::DedicatedDevice {
                physical_device,
                heap_index,
            } => Some(SafeMemoryDomainProjection {
                kind: SafeMemoryDomainKind::DedicatedDevice,
                heap: Some(*heap_index),
                join_id: self.digest(b"domain-physical-device", physical_device.as_str())?,
            }),
        }
    }

    fn mark_incomplete(state: &mut RuntimeReceiptState, reason: ReceiptCompletenessReason) {
        match reason {
            ReceiptCompletenessReason::EventCapacityExceeded
            | ReceiptCompletenessReason::NotificationCapacityExceeded => {
                state.event_history_complete = false;
                if state.event_history_reason.is_none() {
                    state.event_history_reason = Some(reason);
                }
            }
            ReceiptCompletenessReason::Unavailable(_)
            | ReceiptCompletenessReason::IdentityExhausted
            | ReceiptCompletenessReason::OwnerCapacityExceeded
            | ReceiptCompletenessReason::ResourceCapacityExceeded
            | ReceiptCompletenessReason::InvalidLifecycle => {
                state.live_state_complete = false;
                if state.live_state_reason.is_none() {
                    state.live_state_reason = Some(reason);
                }
            }
        }
    }

    fn completeness_for_state(state: &RuntimeReceiptState) -> ReceiptCompleteness {
        ReceiptCompleteness {
            complete: state.live_state_complete && state.event_history_complete,
            reason: state.live_state_reason.or(state.event_history_reason),
            live_state_complete: state.live_state_complete,
            live_state_reason: state.live_state_reason,
            event_history_complete: state.event_history_complete,
            event_history_reason: state.event_history_reason,
            dropped_events: state.dropped_events,
            dropped_owners: state.dropped_owners,
            rejected_resources: state.rejected_resources,
            dropped_notifications: state.dropped_notifications,
        }
    }

    fn state_is_available(&self, state: &RuntimeReceiptState) -> bool {
        self.availability.is_available() && !state.identity_exhausted
    }

    fn availability_for_state(&self, state: &RuntimeReceiptState) -> RuntimeReceiptAvailability {
        if state.identity_exhausted {
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::IdentityExhausted,
            }
        } else {
            self.availability
        }
    }

    fn mark_identity_exhausted(state: &mut RuntimeReceiptState) {
        state.identity_exhausted = true;
        Self::mark_incomplete(state, ReceiptCompletenessReason::IdentityExhausted);
    }

    fn allocate_ordinal(counter: &AtomicU64) -> Option<u64> {
        // 0 is the exhausted sentinel, never a legal identity. The last legal
        // ordinal (`u64::MAX`) is handed out once; the counter then stores 0 so
        // the next allocation cannot wrap or reuse that identity.
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ordinal| {
                if ordinal == 0 {
                    return None;
                }
                Some(ordinal.checked_add(1).unwrap_or(0))
            })
            .ok()
    }

    fn append_event(state: &mut RuntimeReceiptState, capacity: usize, event: RuntimeReceiptEvent) {
        if state.events.len() == capacity {
            state.events.pop_front();
            state.dropped_events = state.dropped_events.saturating_add(1);
            Self::mark_incomplete(state, ReceiptCompletenessReason::EventCapacityExceeded);
        }
        state.events.push_back(event);
    }

    pub(crate) fn is_available(&self) -> bool {
        let state = self.lock_state();
        self.state_is_available(&state)
    }

    pub(crate) fn start_owner(
        &self,
        descriptor: RuntimeOwnerDescriptor,
        attempt_id: Option<ExecutionCacheAttemptId>,
    ) -> RuntimeOwnerGuard {
        let request_attempt_id = current_request_attempt_id();
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return RuntimeOwnerGuard::empty();
        }
        if state.live_owners.len() >= MAX_LIVE_OWNERS {
            state.dropped_owners = state.dropped_owners.saturating_add(1);
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::OwnerCapacityExceeded);
            return RuntimeOwnerGuard::empty();
        }
        let Some(ordinal) = Self::allocate_ordinal(&state.next_owner_ordinal) else {
            Self::mark_identity_exhausted(&mut state);
            return RuntimeOwnerGuard::empty();
        };
        let owner_id = RuntimeOwnerId {
            scope_id: self.scope_id,
            ordinal,
        };
        let owner_descriptor = descriptor;
        state.live_owners.insert(
            owner_id,
            LiveRuntimeOwner {
                id: owner_id,
                descriptor: owner_descriptor,
                resources: BTreeMap::new(),
            },
        );
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::OwnerCreated {
                owner_id,
                descriptor: owner_descriptor,
                attempt_id,
                request_attempt_id,
            },
        );
        RuntimeOwnerGuard {
            collector: Some(self.clone()),
            owner_id: Some(owner_id),
            attempt_id,
            request_attempt_id,
        }
    }

    pub(crate) fn record_owner_reused(
        &self,
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
    ) -> bool {
        let request_attempt_id = current_request_attempt_id();
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return false;
        }
        if !state.live_owners.contains_key(&owner_id) {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        }
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::OwnerReused {
                owner_id,
                attempt_id,
                request_attempt_id,
            },
        );
        true
    }

    pub(crate) fn record_notification_coalesced(&self) {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return;
        }
        state.dropped_notifications = state.dropped_notifications.saturating_add(1);
        Self::mark_incomplete(
            &mut state,
            ReceiptCompletenessReason::NotificationCapacityExceeded,
        );
    }

    pub(crate) fn acquire_resource(
        &self,
        owner_id: RuntimeOwnerId,
        mut descriptor: RuntimeResourceDescriptor,
    ) -> Option<RuntimeResourceGuard> {
        let attempt_id = current_execution_cache_attempt_id();
        let request_attempt_id = current_request_attempt_id();
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return None;
        }
        let Some(owner) = state.live_owners.get(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return None;
        };
        // Resource placement is bound to the already-published owner inside
        // the collector, rather than trusted from an independently-built
        // descriptor. Broker reconciliation later compares this placement to
        // the attribution captured before admission.
        descriptor.placement = owner.descriptor.placement;
        if owner.resources.len() >= MAX_RESOURCES_PER_OWNER {
            state.rejected_resources = state.rejected_resources.saturating_add(1);
            Self::mark_incomplete(
                &mut state,
                ReceiptCompletenessReason::ResourceCapacityExceeded,
            );
            return None;
        }
        let Some(ordinal) = Self::allocate_ordinal(&state.next_resource_ordinal) else {
            Self::mark_identity_exhausted(&mut state);
            return None;
        };
        let resource_id = RuntimeResourceId {
            scope_id: self.scope_id,
            ordinal,
        };
        state
            .live_owners
            .get_mut(&owner_id)
            .expect("owner was checked above")
            .resources
            .insert(
                resource_id,
                LiveRuntimeResource {
                    id: resource_id,
                    descriptor: descriptor.clone(),
                    state: RuntimeResourceState::Reserved,
                },
            );
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::ResourceAcquired {
                owner_id,
                resource_id,
                descriptor,
                attempt_id,
                request_attempt_id,
            },
        );
        Some(RuntimeResourceGuard {
            collector: Some(self.clone()),
            owner_id,
            resource_id,
            attempt_id,
            request_attempt_id,
        })
    }

    pub(crate) fn update_resource(
        &self,
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        mut descriptor: RuntimeResourceDescriptor,
    ) -> bool {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return false;
        }
        let Some(owner) = state.live_owners.get_mut(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        let Some(resource) = owner.resources.get_mut(&resource_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        // Placement is immutable for a live resource. Updates may refresh
        // byte/native evidence, but cannot silently move ownership lanes.
        descriptor.placement = resource.descriptor.placement;
        resource.descriptor = descriptor;
        true
    }

    pub(crate) fn transition_resource(
        &self,
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        next_state: RuntimeResourceState,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    ) -> bool {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return false;
        }
        let Some(owner) = state.live_owners.get_mut(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        let Some(resource) = owner.resources.get_mut(&resource_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        resource.state = next_state;
        let descriptor = resource.descriptor.clone();
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::ResourceStateChanged {
                owner_id,
                resource_id,
                state: next_state,
                descriptor,
                attempt_id,
                request_attempt_id,
            },
        );
        true
    }

    fn release_resource(
        &self,
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    ) {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return;
        }
        let Some(owner) = state.live_owners.get_mut(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return;
        };
        if owner.resources.remove(&resource_id).is_none() {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return;
        }
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::ResourceReleased {
                owner_id,
                resource_id,
                attempt_id,
                request_attempt_id,
            },
        );
    }

    fn release_owner(
        &self,
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
        request_attempt_id: Option<RequestAttemptId>,
    ) {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return;
        }
        let Some(owner) = state.live_owners.remove(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return;
        };
        if !owner.resources.is_empty() {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
        }
        Self::append_event(
            &mut state,
            self.event_capacity,
            RuntimeReceiptEvent::OwnerReleased {
                owner_id,
                attempt_id,
                request_attempt_id,
            },
        );
    }

    /// Start a request-local event window. Live owners stay; the bounded ring
    /// and drop counters reset so one request cannot inherit another request's
    /// overflow. Qualification still fails if this window itself overflows.
    pub(crate) fn begin_request_event_window(&self) {
        let mut state = self.lock_state();
        if !self.state_is_available(&state) {
            return;
        }
        state.events.clear();
        state.dropped_events = 0;
        state.dropped_notifications = 0;
        state.event_history_complete = true;
        state.event_history_reason = None;
    }

    /// Returns a bounded immutable diagnostic snapshot. It has no effect on
    /// admission, fallback, or owner lifetime.
    pub fn snapshot(&self) -> RuntimeReceiptSnapshot {
        let state = self.lock_state();
        RuntimeReceiptSnapshot {
            schema: RUNTIME_RECEIPT_SCHEMA,
            scope_id: self.scope_id,
            availability: self.availability_for_state(&state),
            live_owners: state.live_owners.values().cloned().collect(),
            events: state.events.iter().cloned().collect(),
            event_capacity: self.event_capacity,
            completeness: Self::completeness_for_state(&state),
        }
    }

    /// Returns a bounded read-only summary without copying live descriptors.
    pub fn summary(&self) -> RuntimeReceiptSummary {
        let state = self.lock_state();
        RuntimeReceiptSummary {
            scope_id: self.scope_id,
            availability: self.availability_for_state(&state),
            live_owner_count: state.live_owners.len(),
            live_resource_count: state
                .live_owners
                .values()
                .map(|owner| owner.resources.len())
                .sum(),
            event_count: state.events.len(),
            event_capacity: self.event_capacity,
            completeness: Self::completeness_for_state(&state),
        }
    }

    /// Live-owner comparison against the process-wide broker ledger.
    ///
    /// This never mutates admission. Incomplete or unavailable receipts cannot
    /// prove coverage and return [`LeaseReceiptShadow::Incomparable`]. Event
    /// ring truncation does not invalidate the live owner table.
    pub fn reconcile_live_leases(&self, broker: &DeviceMemoryBrokerSet) -> LeaseReceiptShadow {
        let before = self.snapshot();
        let ledger_before = broker.ledger_snapshot_for_scope_by_placement(self.scope_id);
        let snapshot = self.snapshot();
        let ledger = broker.ledger_snapshot_for_scope_by_placement(self.scope_id);
        if before.availability != snapshot.availability
            || before.live_owners != snapshot.live_owners
            || before.completeness.live_state_complete != snapshot.completeness.live_state_complete
            || before.completeness.live_state_reason != snapshot.completeness.live_state_reason
            || ledger_before != ledger
        {
            return LeaseReceiptShadow::Incomparable {
                reason: LeaseReceiptShadowIncomparable::SnapshotChanged,
            };
        }
        if !matches!(snapshot.availability, RuntimeReceiptAvailability::Available) {
            return LeaseReceiptShadow::Incomparable {
                reason: LeaseReceiptShadowIncomparable::ReceiptsUnavailable,
            };
        }
        if !snapshot.completeness.live_state_complete {
            return LeaseReceiptShadow::Incomparable {
                reason: LeaseReceiptShadowIncomparable::ReceiptsIncomplete(
                    snapshot
                        .completeness
                        .live_state_reason
                        .unwrap_or(ReceiptCompletenessReason::InvalidLifecycle),
                ),
            };
        }

        let mut receipt_bytes = HashMap::<
            (RuntimeOwnerPlacement, SafeMemoryDomainProjection),
            ReceiptDomainBytes,
        >::new();
        for owner in &snapshot.live_owners {
            if matches!(owner.descriptor.placement, RuntimeOwnerPlacement::Unknown) {
                return LeaseReceiptShadow::Incomparable {
                    reason: LeaseReceiptShadowIncomparable::OwnerPlacementUnknown,
                };
            }
            for resource in owner.resources.values() {
                if matches!(
                    resource.descriptor.placement,
                    RuntimeOwnerPlacement::Unknown
                ) {
                    return LeaseReceiptShadow::Incomparable {
                        reason: LeaseReceiptShadowIncomparable::ResourcePlacementUnknown,
                    };
                }
                if resource.descriptor.placement != owner.descriptor.placement {
                    return LeaseReceiptShadow::Incomparable {
                        reason: LeaseReceiptShadowIncomparable::ResourceOwnerPlacementMismatch,
                    };
                }
                let domain = match resource.descriptor.ledger_binding {
                    RuntimeResourceLedgerBinding::Brokered(domain)
                        if resource.descriptor.domain == Some(domain) =>
                    {
                        domain
                    }
                    RuntimeResourceLedgerBinding::NoBrokerLease
                        if resource.descriptor.domain.is_none() =>
                    {
                        continue;
                    }
                    RuntimeResourceLedgerBinding::Unknown => {
                        return LeaseReceiptShadow::Incomparable {
                            reason: LeaseReceiptShadowIncomparable::UnpricedLiveResource,
                        };
                    }
                    RuntimeResourceLedgerBinding::Brokered(_)
                    | RuntimeResourceLedgerBinding::NoBrokerLease => {
                        return LeaseReceiptShadow::Incomparable {
                            reason: LeaseReceiptShadowIncomparable::InvalidLiveLifecycle,
                        };
                    }
                };
                let slot = receipt_bytes
                    .entry((resource.descriptor.placement, domain))
                    .or_default();
                match resource.state {
                    RuntimeResourceState::Reserved => {
                        match known_metric(resource.descriptor.peak) {
                            Ok(bytes) => {
                                slot.reserved_peak = slot.reserved_peak.saturating_add(bytes)
                            }
                            Err(reason) => {
                                return LeaseReceiptShadow::Incomparable { reason };
                            }
                        }
                    }
                    RuntimeResourceState::Committed | RuntimeResourceState::Reconciled => {
                        match known_metric(resource.descriptor.retained) {
                            Ok(bytes) => {
                                slot.committed_retained =
                                    slot.committed_retained.saturating_add(bytes)
                            }
                            Err(reason) => {
                                return LeaseReceiptShadow::Incomparable { reason };
                            }
                        }
                    }
                    RuntimeResourceState::Quarantined => {
                        match known_metric(resource.descriptor.retained) {
                            Ok(bytes) => {
                                slot.quarantined_retained =
                                    slot.quarantined_retained.saturating_add(bytes)
                            }
                            Err(reason) => {
                                return LeaseReceiptShadow::Incomparable { reason };
                            }
                        }
                    }
                    RuntimeResourceState::Released => {
                        return LeaseReceiptShadow::Incomparable {
                            reason: LeaseReceiptShadowIncomparable::InvalidLiveLifecycle,
                        };
                    }
                }
            }
        }

        let mut seen = HashMap::<(RuntimeOwnerPlacement, SafeMemoryDomainProjection), ()>::new();
        for ((domain, placement), usage) in &ledger {
            if matches!(placement, RuntimeOwnerPlacement::Unknown)
                && (usage.pending_bytes != 0
                    || usage.committed_bytes != 0
                    || usage.unreclaimable_bytes != 0)
            {
                return LeaseReceiptShadow::Incomparable {
                    reason: LeaseReceiptShadowIncomparable::LedgerPlacementUnknown,
                };
            }
            let Some(projection) = self.domain_projection(domain) else {
                return LeaseReceiptShadow::Incomparable {
                    reason: LeaseReceiptShadowIncomparable::ReceiptsUnavailable,
                };
            };
            let key = (*placement, projection);
            seen.insert(key, ());
            let receipts = receipt_bytes.get(&key).copied().unwrap_or_default();
            if receipts.reserved_peak != usage.pending_bytes
                || receipts.committed_retained != usage.committed_bytes
                || receipts.quarantined_retained != usage.unreclaimable_bytes
            {
                return LeaseReceiptShadow::Mismatch(LeaseReceiptShadowMismatch {
                    placement: *placement,
                    domain: projection,
                    broker_pending: usage.pending_bytes,
                    broker_committed: usage.committed_bytes,
                    broker_unreclaimable: usage.unreclaimable_bytes,
                    receipt_reserved: receipts.reserved_peak,
                    receipt_committed: receipts.committed_retained,
                    receipt_quarantined: receipts.quarantined_retained,
                });
            }
        }
        for ((placement, projection), receipts) in &receipt_bytes {
            if seen.contains_key(&(*placement, *projection)) {
                continue;
            }
            if receipts.reserved_peak != 0
                || receipts.committed_retained != 0
                || receipts.quarantined_retained != 0
            {
                return LeaseReceiptShadow::Mismatch(LeaseReceiptShadowMismatch {
                    placement: *placement,
                    domain: *projection,
                    broker_pending: 0,
                    broker_committed: 0,
                    broker_unreclaimable: 0,
                    receipt_reserved: receipts.reserved_peak,
                    receipt_committed: receipts.committed_retained,
                    receipt_quarantined: receipts.quarantined_retained,
                });
            }
        }
        LeaseReceiptShadow::Matched
    }

    /// Waits for the short broker/receipt publication window to quiesce, then
    /// returns the same fail-closed comparison as [`Self::reconcile_live_leases`].
    ///
    /// Broker accounting and diagnostic receipt guards intentionally use
    /// separate locks so diagnostics can never participate in admission. A
    /// reservation therefore has a tiny transition window while its receipt
    /// is attached or released. Activation is the only path that needs a
    /// stable pre-publication verdict; it may retry transient `Mismatch` or
    /// `SnapshotChanged`, but never retries an unavailable, incomplete,
    /// unknown-placement, or unpriced state.
    pub fn reconcile_live_leases_quiescent(
        &self,
        broker: &DeviceMemoryBrokerSet,
    ) -> LeaseReceiptShadow {
        const QUIESCENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);
        const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(1);

        let deadline = std::time::Instant::now() + QUIESCENCE_TIMEOUT;
        loop {
            let reconciliation = self.reconcile_live_leases(broker);
            let retryable = matches!(
                reconciliation,
                LeaseReceiptShadow::Mismatch(_)
                    | LeaseReceiptShadow::Incomparable {
                        reason: LeaseReceiptShadowIncomparable::SnapshotChanged,
                    }
            );
            if !retryable || std::time::Instant::now() >= deadline {
                return reconciliation;
            }
            std::thread::sleep(RETRY_DELAY);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ReceiptDomainBytes {
    reserved_peak: u64,
    committed_retained: u64,
    quarantined_retained: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "detail", rename_all = "kebab-case")]
pub enum LeaseReceiptShadow {
    Matched,
    Incomparable {
        reason: LeaseReceiptShadowIncomparable,
    },
    Mismatch(LeaseReceiptShadowMismatch),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeaseReceiptShadowIncomparable {
    ReceiptsUnavailable,
    ReceiptsIncomplete(ReceiptCompletenessReason),
    UnpricedLiveResource,
    OwnerPlacementUnknown,
    ResourcePlacementUnknown,
    ResourceOwnerPlacementMismatch,
    LedgerPlacementUnknown,
    InvalidLiveLifecycle,
    SnapshotChanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LeaseReceiptShadowMismatch {
    pub placement: RuntimeOwnerPlacement,
    pub domain: SafeMemoryDomainProjection,
    pub broker_pending: u64,
    pub broker_committed: u64,
    pub broker_unreclaimable: u64,
    pub receipt_reserved: u64,
    pub receipt_committed: u64,
    pub receipt_quarantined: u64,
}

fn known_metric(metric: RuntimeReceiptMetric) -> Result<u64, LeaseReceiptShadowIncomparable> {
    match metric {
        RuntimeReceiptMetric::Known(bytes) => Ok(bytes),
        RuntimeReceiptMetric::Unavailable | RuntimeReceiptMetric::Unknown => {
            Err(LeaseReceiptShadowIncomparable::UnpricedLiveResource)
        }
    }
}

/// Drop guard for one live owner. It is diagnostic-only and never owns the
/// underlying native object, so receipt teardown cannot alter execution.
pub(crate) struct RuntimeOwnerGuard {
    collector: Option<RuntimeReceiptCollector>,
    owner_id: Option<RuntimeOwnerId>,
    attempt_id: Option<ExecutionCacheAttemptId>,
    request_attempt_id: Option<RequestAttemptId>,
}

impl RuntimeOwnerGuard {
    fn empty() -> Self {
        Self {
            collector: None,
            owner_id: None,
            attempt_id: None,
            request_attempt_id: None,
        }
    }

    pub(crate) fn owner_id(&self) -> Option<RuntimeOwnerId> {
        self.owner_id
    }

    pub(crate) fn record_reuse(&self, attempt_id: Option<ExecutionCacheAttemptId>) {
        if let (Some(collector), Some(owner_id)) = (&self.collector, self.owner_id) {
            collector.record_owner_reused(owner_id, attempt_id);
        }
    }

    /// Leave the live owner row in the collector after this guard is dropped.
    /// This is reserved for a broker quarantine: the underlying native owner
    /// and its physical charge are intentionally unreclaimable, so deleting
    /// the diagnostic row would make the live receipt contradict the ledger.
    pub(crate) fn persist_for_quarantine(mut self) {
        self.collector = None;
    }

    fn release_inner(&mut self) {
        let Some(owner_id) = self.owner_id.take() else {
            return;
        };
        if let Some(collector) = self.collector.as_ref() {
            collector.release_owner(owner_id, self.attempt_id, self.request_attempt_id);
        }
        self.collector = None;
    }
}

impl fmt::Debug for RuntimeOwnerGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeOwnerGuard")
            .field("owner_id", &self.owner_id)
            .finish_non_exhaustive()
    }
}

impl Drop for RuntimeOwnerGuard {
    fn drop(&mut self) {
        self.release_inner();
    }
}

pub(crate) struct RuntimeResourceGuard {
    collector: Option<RuntimeReceiptCollector>,
    owner_id: RuntimeOwnerId,
    resource_id: RuntimeResourceId,
    attempt_id: Option<ExecutionCacheAttemptId>,
    request_attempt_id: Option<RequestAttemptId>,
}

impl fmt::Debug for RuntimeResourceGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeResourceGuard")
            .field("owner_id", &self.owner_id)
            .field("resource_id", &self.resource_id)
            .finish_non_exhaustive()
    }
}

impl RuntimeResourceGuard {
    pub(crate) fn set_state(&self, state: RuntimeResourceState) {
        if let Some(collector) = self.collector.as_ref() {
            collector.transition_resource(
                self.owner_id,
                self.resource_id,
                state,
                self.attempt_id,
                self.request_attempt_id,
            );
        }
    }

    pub(crate) fn update_descriptor(&self, descriptor: RuntimeResourceDescriptor) {
        if let Some(collector) = self.collector.as_ref() {
            collector.update_resource(self.owner_id, self.resource_id, descriptor);
        }
    }

    /// Preserve the live resource row after the guard is dropped. See
    /// [`RuntimeOwnerGuard::persist_for_quarantine`].
    pub(crate) fn persist_for_quarantine(mut self) {
        self.collector = None;
    }
}

impl Drop for RuntimeResourceGuard {
    fn drop(&mut self) {
        if let Some(collector) = self.collector.as_ref() {
            collector.release_resource(
                self.owner_id,
                self.resource_id,
                self.attempt_id,
                self.request_attempt_id,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_memory::{
        DeviceMemoryBrokerSet, DeviceMemoryPolicy, DeviceMemorySnapshot, DomainReservationRequest,
        MemoryDomainKey, MemoryObservationConfidence, PhysicalDeviceKey, QuoteConfidence,
    };

    fn scope() -> NativeExecutionScopeId {
        NativeExecutionScopeId::next()
    }

    fn collector(capacity: usize) -> RuntimeReceiptCollector {
        RuntimeReceiptCollector::new_for_test(scope(), capacity).unwrap()
    }

    fn collector_with_ordinals(
        capacity: usize,
        next_owner_ordinal: u64,
        next_resource_ordinal: u64,
    ) -> RuntimeReceiptCollector {
        RuntimeReceiptCollector::new_for_test_with_ordinals(
            scope(),
            capacity,
            next_owner_ordinal,
            next_resource_ordinal,
        )
        .unwrap()
    }

    fn owner(collector: &RuntimeReceiptCollector) -> RuntimeOwnerGuard {
        let descriptor = collector
            .owner_descriptor(
                "/tmp/audio/prompt-token-owner",
                Some("/private/models/pack.oasr"),
                Some("/private/source/generation"),
                collector.lane_projection(
                    ExecutionProvider::Cpu,
                    "CPU",
                    ExecutionPlacement::CpuOnly,
                    GgmlCpuGraphBackend::Cpu,
                ),
            )
            .expect("entropy-backed descriptor");
        collector.start_owner(descriptor, None)
    }

    #[test]
    fn owner_create_reuse_release_and_live_table_are_bounded() {
        let collector = collector(8);
        let guard = owner(&collector);
        guard.record_reuse(None);
        assert_eq!(collector.snapshot().live_owners.len(), 1);
        drop(guard);
        let snapshot = collector.snapshot();
        assert!(snapshot.live_owners.is_empty());
        assert_eq!(snapshot.events.len(), 3);
    }

    #[test]
    fn owner_and_resource_events_keep_the_request_attempt_join() {
        let runtime = collector(16);
        let request_receipt =
            crate::models::request_execution_receipt::NativeExecutionReceiptCollector::new();
        let request_attempt =
            crate::RequestAttemptId::parse("00112233445566778899aabbccddeeff").unwrap();
        request_receipt.bind_request_attempt(request_attempt);
        let _request =
            crate::models::native_execution_services::install_execution_receipt_collector(Some(
                request_receipt,
            ));
        let owner = owner(&runtime);
        let resource = runtime
            .acquire_resource(
                owner.owner_id().unwrap(),
                runtime
                    .no_broker_resource_descriptor("request-attempt-marker")
                    .unwrap(),
            )
            .unwrap();
        resource.set_state(RuntimeResourceState::Committed);
        drop(resource);
        drop(owner);

        for event in runtime.snapshot().events {
            let observed = match event {
                RuntimeReceiptEvent::OwnerCreated {
                    request_attempt_id, ..
                }
                | RuntimeReceiptEvent::OwnerReused {
                    request_attempt_id, ..
                }
                | RuntimeReceiptEvent::OwnerReleased {
                    request_attempt_id, ..
                }
                | RuntimeReceiptEvent::ResourceAcquired {
                    request_attempt_id, ..
                }
                | RuntimeReceiptEvent::ResourceStateChanged {
                    request_attempt_id, ..
                }
                | RuntimeReceiptEvent::ResourceReleased {
                    request_attempt_id, ..
                } => request_attempt_id,
            };
            assert_eq!(observed, Some(request_attempt));
        }
    }

    #[test]
    fn resource_lifecycle_uses_safe_domain_projection_and_confidence_types() {
        let collector = collector(8);
        let guard = owner(&collector);
        let owner_id = guard.owner_id().unwrap();
        let domain = MemoryDomainKey::DedicatedDevice {
            physical_device: PhysicalDeviceKey::new("550e8400-e29b-41d4-a716-446655440000")
                .unwrap(),
            heap_index: 7,
        };
        let descriptor = collector
            .resource_descriptor(
                "pack-weight-buffer",
                &domain,
                10,
                20,
                20,
                QuoteConfidence::CommittedUpperBound,
                Some(MemoryObservationConfidence::DeviceSnapshot),
            )
            .expect("entropy-backed resource descriptor");
        let resource = collector.acquire_resource(owner_id, descriptor).unwrap();
        drop(resource);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.live_owners[0].resources.len(), 0);
        assert!(snapshot.completeness.complete);
        let rendered = format!("{snapshot:?}");
        assert!(!rendered.contains("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!rendered.contains("PhysicalDeviceKey"));
    }

    #[test]
    fn keyed_projection_is_domain_separated_collision_resistant_and_root_local() {
        let first = collector(8);
        let second = collector(8);
        let path = "/private/models/pack.oasr";
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        assert_ne!(
            first.digest(b"content", path),
            first.digest(b"source", path)
        );
        assert_ne!(first.digest(b"content", "a"), first.digest(b"content", "b"));
        assert_ne!(
            first.digest(b"content", uuid),
            second.digest(b"content", uuid)
        );
        let descriptor = first
            .owner_descriptor(path, Some(uuid), Some(path), None)
            .expect("entropy-backed owner descriptor");
        let snapshot = {
            let _guard = first.start_owner(descriptor, None);
            first.snapshot()
        };
        let rendered = format!("{snapshot:?}");
        assert!(!rendered.contains(path));
        assert!(!rendered.contains(uuid));
    }

    #[test]
    fn request_event_window_clears_overflow_without_dropping_live_owners() {
        let collector = collector(2);
        let first = owner(&collector);
        drop(first);
        let live = owner(&collector);
        live.record_reuse(None);
        assert!(!collector.snapshot().completeness.event_history_complete);
        collector.begin_request_event_window();
        let snapshot = collector.snapshot();
        assert!(snapshot.completeness.event_history_complete);
        assert_eq!(snapshot.completeness.dropped_events, 0);
        assert!(snapshot.events.is_empty());
        assert_eq!(snapshot.live_owners.len(), 1);
        drop(live);
    }

    #[test]
    fn ring_overflow_marks_snapshot_incomplete_without_unbounded_growth() {
        let collector = collector(2);
        let first = owner(&collector);
        drop(first);
        let second = owner(&collector);
        let second_id = second.owner_id().unwrap();
        second.record_reuse(None);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.events.len(), 2);
        assert_eq!(snapshot.event_capacity, 2);
        assert!(!snapshot.completeness.complete);
        assert!(snapshot.completeness.live_state_complete);
        assert!(!snapshot.completeness.event_history_complete);
        assert!(snapshot.completeness.dropped_events > 0);
        assert_eq!(second_id.scope_id, snapshot.scope_id);
        let broker =
            DeviceMemoryBrokerSet::new(crate::device::execution_memory::DeviceMemoryPolicy {
                maximum_owned_basis_points: 10_000,
                minimum_headroom_bytes: 0,
            });
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Matched,
            "bounded event history must not invalidate the authoritative live owner table"
        );
    }

    #[test]
    fn event_ring_does_not_preallocate_the_advertised_capacity() {
        let collector =
            RuntimeReceiptCollector::new_for_test(scope(), DEFAULT_EVENT_CAPACITY).unwrap();
        assert_eq!(
            collector.lock_state().events.capacity(),
            0,
            "empty collector must not commit DEFAULT_EVENT_CAPACITY event slots"
        );
        assert_eq!(collector.summary().event_capacity, DEFAULT_EVENT_CAPACITY);
        assert_eq!(collector.summary().event_count, 0);
    }

    #[test]
    fn oversized_capacity_is_rejected_and_fixed_constructor_is_bounded() {
        assert!(matches!(
            RuntimeReceiptCollector::new_for_test(scope(), MAX_EVENT_CAPACITY + 1),
            Err(RuntimeReceiptError::CapacityTooLarge { .. })
        ));
        assert!(matches!(
            RuntimeReceiptCollector::new_for_test(scope(), 0),
            Err(RuntimeReceiptError::ZeroCapacity)
        ));
        let collector = RuntimeReceiptCollector::new(scope());
        assert!(collector.summary().event_capacity <= MAX_EVENT_CAPACITY);
    }

    #[test]
    fn entropy_failure_reports_unavailable_without_fake_completeness() {
        let collector = RuntimeReceiptCollector::new_with_entropy_failure_for_test(scope());
        assert_eq!(
            collector.availability,
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::EntropyUnavailable,
            }
        );
        assert!(
            collector
                .owner_descriptor("/private/path", None, None, None)
                .is_none()
        );
        let snapshot = collector.snapshot();
        assert!(!snapshot.completeness.complete);
        assert_eq!(
            snapshot.completeness.reason,
            Some(ReceiptCompletenessReason::Unavailable(
                RuntimeReceiptUnavailableReason::EntropyUnavailable
            ))
        );
        assert_eq!(snapshot.live_owners.len(), 0);
        assert_eq!(collector.summary().availability, snapshot.availability);
    }

    #[test]
    fn ordinal_exhaustion_is_fail_closed_and_stops_owner_evidence() {
        let collector = collector_with_ordinals(8, u64::MAX, 1);
        let first = owner(&collector);
        let first_id = first.owner_id().expect("last legal owner ordinal");
        assert_eq!(first_id.ordinal, u64::MAX);
        assert_ne!(first_id.ordinal, 0);
        let before = collector.snapshot();

        let exhausted = owner(&collector);
        assert!(exhausted.owner_id().is_none());
        let exhausted_snapshot = collector.snapshot();
        let live_ordinals = exhausted_snapshot
            .live_owners
            .iter()
            .map(|owner| owner.id.ordinal)
            .collect::<Vec<_>>();
        assert_eq!(live_ordinals, vec![u64::MAX]);
        assert_eq!(
            exhausted_snapshot.availability,
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::IdentityExhausted,
            }
        );
        assert_eq!(
            exhausted_snapshot.completeness.reason,
            Some(ReceiptCompletenessReason::IdentityExhausted)
        );
        assert_eq!(exhausted_snapshot.events, before.events);
        assert!(
            collector
                .owner_descriptor("/private/path", None, None, None)
                .is_none()
        );

        first.record_reuse(None);
        collector.record_notification_coalesced();
        drop(first);
        assert_eq!(collector.snapshot().events, before.events);
    }

    #[test]
    fn ordinal_exhaustion_is_fail_closed_and_stops_resource_evidence() {
        let collector = collector_with_ordinals(8, 1, u64::MAX);
        let owner_guard = owner(&collector);
        let owner_id = owner_guard.owner_id().unwrap();
        let descriptor = collector
            .unpriced_resource_descriptor("pack-weight-buffer")
            .expect("entropy-backed resource descriptor");
        let first = collector
            .acquire_resource(owner_id, descriptor.clone())
            .expect("last legal resource ordinal is allocated once");
        let first_snapshot = collector.snapshot();
        let resource_ordinals = first_snapshot.live_owners[0]
            .resources
            .keys()
            .map(|resource_id| resource_id.ordinal)
            .collect::<Vec<_>>();
        assert_eq!(resource_ordinals, vec![u64::MAX]);
        assert!(!resource_ordinals.contains(&0));

        assert!(collector.acquire_resource(owner_id, descriptor).is_none());
        let exhausted_snapshot = collector.snapshot();
        assert_eq!(
            exhausted_snapshot.availability,
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::IdentityExhausted,
            }
        );
        assert_eq!(
            exhausted_snapshot.completeness.reason,
            Some(ReceiptCompletenessReason::IdentityExhausted)
        );
        assert_eq!(exhausted_snapshot.events.len(), first_snapshot.events.len());
        assert!(
            collector
                .unpriced_resource_descriptor("another-resource")
                .is_none()
        );

        drop(first);
        drop(owner_guard);
        assert_eq!(
            collector.snapshot().events.len(),
            first_snapshot.events.len()
        );
    }

    #[test]
    fn last_two_owner_ordinals_are_unique_then_identity_exhausts() {
        let collector = collector_with_ordinals(8, u64::MAX - 1, 1);
        let first = owner(&collector);
        let second = owner(&collector);
        let third = owner(&collector);
        let first_id = first.owner_id().expect("penultimate owner ordinal");
        let second_id = second.owner_id().expect("last legal owner ordinal");
        assert_eq!(first_id.ordinal, u64::MAX - 1);
        assert_eq!(second_id.ordinal, u64::MAX);
        assert_ne!(first_id, second_id);
        assert_ne!(first_id.ordinal, 0);
        assert_ne!(second_id.ordinal, 0);
        assert!(third.owner_id().is_none());
        let live_ordinals = collector
            .snapshot()
            .live_owners
            .iter()
            .map(|owner| owner.id.ordinal)
            .collect::<Vec<_>>();
        assert_eq!(live_ordinals.len(), 2);
        assert!(!live_ordinals.contains(&0));
        assert_ne!(live_ordinals[0], live_ordinals[1]);
        assert_eq!(
            collector.snapshot().availability,
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::IdentityExhausted,
            }
        );
    }

    #[test]
    fn zero_ordinal_is_never_allocated_as_a_known_identity() {
        let collector = collector_with_ordinals(8, 0, 0);
        let owner_guard = owner(&collector);
        assert!(owner_guard.owner_id().is_none());
        let snapshot = collector.snapshot();
        assert!(snapshot.live_owners.is_empty());
        assert!(snapshot.events.is_empty());
        assert_eq!(
            snapshot.availability,
            RuntimeReceiptAvailability::Unavailable {
                reason: RuntimeReceiptUnavailableReason::IdentityExhausted,
            }
        );
        assert_eq!(
            snapshot.completeness.reason,
            Some(ReceiptCompletenessReason::IdentityExhausted)
        );
    }

    #[test]
    fn roots_isolate_owner_tables_and_ids() {
        let first = collector(8);
        let second = collector(8);
        let first_guard = owner(&first);
        let second_guard = owner(&second);
        assert_ne!(
            first_guard.owner_id().unwrap(),
            second_guard.owner_id().unwrap()
        );
        assert_eq!(first.snapshot().live_owners.len(), 1);
        assert_eq!(second.snapshot().live_owners.len(), 1);
    }

    #[test]
    fn empty_collector_matches_an_empty_broker_ledger() {
        let collector = collector(8);
        let broker =
            DeviceMemoryBrokerSet::new(crate::device::execution_memory::DeviceMemoryPolicy {
                maximum_owned_basis_points: 10_000,
                minimum_headroom_bytes: 0,
            });
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Matched
        );
    }

    #[test]
    fn equal_domain_totals_cannot_hide_cross_lane_receipt_misattribution() {
        let collector = collector(64);
        let scope_id = collector.scope_id;
        let domain = MemoryDomainKey::DedicatedDevice {
            physical_device: PhysicalDeviceKey::new("0000:01:00.0").unwrap(),
            heap_index: 0,
        };
        let lane_a = collector
            .lane_projection(
                ExecutionProvider::Cuda,
                "cuda:0",
                ExecutionPlacement::FullDevice,
                GgmlCpuGraphBackend::Gpu,
            )
            .unwrap();
        let lane_b = collector
            .lane_projection(
                ExecutionProvider::Vulkan,
                "vulkan:0",
                ExecutionPlacement::FullDevice,
                GgmlCpuGraphBackend::Gpu,
            )
            .unwrap();
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let request = |resource_id: &str| DomainReservationRequest {
            domain: domain.clone(),
            snapshot: DeviceMemorySnapshot {
                free_bytes: 100,
                total_bytes: 100,
                confidence: MemoryObservationConfidence::DeviceSnapshot,
            },
            peak_bytes: 10,
            retained_bytes: 10,
            observed_peak_bytes: None,
            requires_reconciliation: false,
            resource_id: resource_id.to_string(),
            cohort_id: None,
        };
        let mut leases = broker
            .try_reserve_partitioned_for_scope_and_placements(
                vec![vec![request("lane-a")], vec![request("lane-b")]],
                Some(scope_id),
                vec![
                    RuntimeOwnerPlacement::LaneBound(lane_a),
                    RuntimeOwnerPlacement::LaneBound(lane_b),
                ],
            )
            .unwrap();
        for lease in &mut leases {
            lease.commit_quoted().unwrap();
        }

        // Deliberately attribute both receipt resources to lane A. A legacy
        // domain-only comparison sees 20 == 20; the exact-lane comparison must
        // reject it because lane B has no receipt coverage.
        let owner_descriptor = collector
            .owner_descriptor("misattributed-owner", None, None, Some(lane_a))
            .unwrap();
        let owner = collector.start_owner(owner_descriptor, None);
        let owner_id = owner.owner_id().unwrap();
        let mut resources = Vec::new();
        for label in ["first", "second"] {
            let descriptor = collector
                .resource_descriptor(
                    label,
                    &domain,
                    10,
                    10,
                    10,
                    QuoteConfidence::CommittedUpperBound,
                    Some(MemoryObservationConfidence::DeviceSnapshot),
                )
                .unwrap();
            let resource = collector.acquire_resource(owner_id, descriptor).unwrap();
            resource.set_state(RuntimeResourceState::Committed);
            resources.push(resource);
        }

        assert!(matches!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Mismatch(LeaseReceiptShadowMismatch {
                placement: RuntimeOwnerPlacement::LaneBound(lane),
                ..
            }) if lane == lane_a || lane == lane_b
        ));
        drop(resources);
        drop(owner);
        drop(leases);
    }

    #[test]
    fn unpriced_live_resource_is_incomparable_not_silent_zero() {
        let collector = collector(8);
        let owner = owner(&collector);
        let resource = collector
            .acquire_resource(
                owner.owner_id().unwrap(),
                collector
                    .unpriced_resource_descriptor("legacy-unpriced")
                    .unwrap(),
            )
            .expect("unpriced row is retained for diagnosis");
        let broker =
            DeviceMemoryBrokerSet::new(crate::device::execution_memory::DeviceMemoryPolicy {
                maximum_owned_basis_points: 10_000,
                minimum_headroom_bytes: 0,
            });
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Incomparable {
                reason: LeaseReceiptShadowIncomparable::UnpricedLiveResource,
            }
        );
        drop(resource);
        drop(owner);
    }

    #[test]
    fn semantic_child_marker_does_not_duplicate_its_broker_owned_children() {
        let collector = collector(8);
        let owner = owner(&collector);
        let resource = collector
            .acquire_resource(
                owner.owner_id().unwrap(),
                collector
                    .no_broker_resource_descriptor("serve-batch.runtime-width=4")
                    .unwrap(),
            )
            .expect("semantic marker receipt");
        let broker =
            DeviceMemoryBrokerSet::new(crate::device::execution_memory::DeviceMemoryPolicy {
                maximum_owned_basis_points: 10_000,
                minimum_headroom_bytes: 0,
            });
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Matched
        );
        drop(resource);
        drop(owner);
    }

    #[test]
    fn unavailable_receipts_cannot_claim_lease_coverage() {
        let collector = RuntimeReceiptCollector::new_with_entropy_failure_for_test(scope());
        let broker =
            DeviceMemoryBrokerSet::new(crate::device::execution_memory::DeviceMemoryPolicy {
                maximum_owned_basis_points: 10_000,
                minimum_headroom_bytes: 0,
            });
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            LeaseReceiptShadow::Incomparable {
                reason: LeaseReceiptShadowIncomparable::ReceiptsUnavailable,
            }
        );
    }
}
