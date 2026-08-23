//! Bounded, production-safe diagnostics for native runtime ownership.
//!
//! Receipts are diagnostic evidence only. Admission, candidate ordering, and
//! fallback never read this module's event stream or snapshot.

use std::{
    collections::{BTreeMap, VecDeque},
    fmt::{self, Write as _},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::device::execution_memory::{
    MemoryDomainKey, MemoryObservationConfidence, QuoteConfidence,
};
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::native_execution_services::{ExecutionCacheAttemptId, NativeExecutionScopeId};

/// Schema marker for the phase-0 in-process ownership evidence.
pub const RUNTIME_RECEIPT_SCHEMA: &str = "openasr.runtime-ownership-receipt.v1";
const DEFAULT_EVENT_CAPACITY: usize = 256;
const MAX_EVENT_CAPACITY: usize = 4096;
const MAX_LIVE_OWNERS: usize = 1024;
const MAX_RESOURCES_PER_OWNER: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RuntimeReceiptUnavailableReason {
    EntropyUnavailable,
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
    EventCapacityExceeded,
    OwnerCapacityExceeded,
    ResourceCapacityExceeded,
    NotificationCapacityExceeded,
    InvalidLifecycle,
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
    pub lane: Option<SafeExecutionLaneProjection>,
}

/// Safe resource metadata. Domain and confidence use the existing admission
/// vocabulary where those values are meaningful; the domain identity itself is
/// projected into [`SafeMemoryDomainProjection`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeResourceDescriptor {
    pub kind: RedactedIdentity,
    pub domain: Option<SafeMemoryDomainProjection>,
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
    },
    OwnerReused {
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
    },
    OwnerReleased {
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
    },
    ResourceAcquired {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        descriptor: RuntimeResourceDescriptor,
    },
    ResourceStateChanged {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        state: RuntimeResourceState,
        descriptor: RuntimeResourceDescriptor,
    },
    ResourceReleased {
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
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
    live_owners: BTreeMap<RuntimeOwnerId, LiveRuntimeOwner>,
    events: VecDeque<RuntimeReceiptEvent>,
    dropped_events: u64,
    dropped_owners: u64,
    rejected_resources: u64,
    dropped_notifications: u64,
    complete: bool,
    completeness_reason: Option<ReceiptCompletenessReason>,
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
        let completeness_reason =
            (!availability.is_available()).then_some(ReceiptCompletenessReason::Unavailable(
                RuntimeReceiptUnavailableReason::EntropyUnavailable,
            ));
        Ok(Self {
            scope_id,
            key: availability.is_available().then_some(key),
            availability,
            event_capacity,
            state: Arc::new(Mutex::new(RuntimeReceiptState {
                next_owner_ordinal: AtomicU64::new(1),
                next_resource_ordinal: AtomicU64::new(1),
                live_owners: BTreeMap::new(),
                events: VecDeque::with_capacity(event_capacity),
                dropped_events: 0,
                dropped_owners: 0,
                rejected_resources: 0,
                dropped_notifications: 0,
                complete: availability.is_available(),
                completeness_reason,
            })),
        })
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RuntimeReceiptState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn digest(&self, domain: &[u8], value: &str) -> Option<RedactedIdentity> {
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
            lane,
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
        Some(RuntimeResourceDescriptor {
            kind: self.digest(b"resource-kind", kind)?,
            domain: Some(self.domain_projection(domain)?),
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

    /// Serve-batch and other legacy components whose memory footprint is not
    /// priced yet use typed Unknown metrics and no fabricated memory domain.
    pub(crate) fn unpriced_resource_descriptor(
        &self,
        kind: &str,
    ) -> Option<RuntimeResourceDescriptor> {
        Some(RuntimeResourceDescriptor {
            kind: self.digest(b"resource-kind", kind)?,
            domain: None,
            requested: RuntimeReceiptMetric::Unknown,
            peak: RuntimeReceiptMetric::Unknown,
            retained: RuntimeReceiptMetric::Unknown,
            quote_confidence: QuoteConfidence::Unknown,
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
        state.complete = false;
        if state.completeness_reason.is_none() {
            state.completeness_reason = Some(reason);
        }
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
        self.availability.is_available()
    }

    pub(crate) fn start_owner(
        &self,
        descriptor: RuntimeOwnerDescriptor,
        attempt_id: Option<ExecutionCacheAttemptId>,
    ) -> RuntimeOwnerGuard {
        if !self.is_available() {
            return RuntimeOwnerGuard::empty();
        }
        let mut state = self.lock_state();
        if state.live_owners.len() >= MAX_LIVE_OWNERS {
            state.dropped_owners = state.dropped_owners.saturating_add(1);
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::OwnerCapacityExceeded);
            return RuntimeOwnerGuard::empty();
        }
        let owner_id = RuntimeOwnerId {
            scope_id: self.scope_id,
            ordinal: state.next_owner_ordinal.fetch_add(1, Ordering::Relaxed),
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
            },
        );
        RuntimeOwnerGuard {
            collector: Some(self.clone()),
            owner_id: Some(owner_id),
            attempt_id,
        }
    }

    pub(crate) fn record_owner_reused(
        &self,
        owner_id: RuntimeOwnerId,
        attempt_id: Option<ExecutionCacheAttemptId>,
    ) -> bool {
        if !self.is_available() {
            return false;
        }
        let mut state = self.lock_state();
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
            },
        );
        true
    }

    pub(crate) fn record_notification_coalesced(&self) {
        if !self.is_available() {
            return;
        }
        let mut state = self.lock_state();
        state.dropped_notifications = state.dropped_notifications.saturating_add(1);
        Self::mark_incomplete(
            &mut state,
            ReceiptCompletenessReason::NotificationCapacityExceeded,
        );
    }

    pub(crate) fn acquire_resource(
        &self,
        owner_id: RuntimeOwnerId,
        descriptor: RuntimeResourceDescriptor,
    ) -> Option<RuntimeResourceGuard> {
        if !self.is_available() {
            return None;
        }
        let mut state = self.lock_state();
        let Some(owner) = state.live_owners.get(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return None;
        };
        if owner.resources.len() >= MAX_RESOURCES_PER_OWNER {
            state.rejected_resources = state.rejected_resources.saturating_add(1);
            Self::mark_incomplete(
                &mut state,
                ReceiptCompletenessReason::ResourceCapacityExceeded,
            );
            return None;
        }
        let resource_id = RuntimeResourceId {
            scope_id: self.scope_id,
            ordinal: state.next_resource_ordinal.fetch_add(1, Ordering::Relaxed),
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
            },
        );
        Some(RuntimeResourceGuard {
            collector: self.clone(),
            owner_id,
            resource_id,
        })
    }

    pub(crate) fn update_resource(
        &self,
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        descriptor: RuntimeResourceDescriptor,
    ) -> bool {
        if !self.is_available() {
            return false;
        }
        let mut state = self.lock_state();
        let Some(owner) = state.live_owners.get_mut(&owner_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        let Some(resource) = owner.resources.get_mut(&resource_id) else {
            Self::mark_incomplete(&mut state, ReceiptCompletenessReason::InvalidLifecycle);
            return false;
        };
        resource.descriptor = descriptor;
        true
    }

    pub(crate) fn transition_resource(
        &self,
        owner_id: RuntimeOwnerId,
        resource_id: RuntimeResourceId,
        next_state: RuntimeResourceState,
    ) -> bool {
        if !self.is_available() {
            return false;
        }
        let mut state = self.lock_state();
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
            },
        );
        true
    }

    fn release_resource(&self, owner_id: RuntimeOwnerId, resource_id: RuntimeResourceId) {
        if !self.is_available() {
            return;
        }
        let mut state = self.lock_state();
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
            },
        );
    }

    fn release_owner(&self, owner_id: RuntimeOwnerId, attempt_id: Option<ExecutionCacheAttemptId>) {
        if !self.is_available() {
            return;
        }
        let mut state = self.lock_state();
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
            },
        );
    }

    /// Returns a bounded immutable diagnostic snapshot. It has no effect on
    /// admission, fallback, or owner lifetime.
    pub fn snapshot(&self) -> RuntimeReceiptSnapshot {
        let state = self.lock_state();
        RuntimeReceiptSnapshot {
            schema: RUNTIME_RECEIPT_SCHEMA,
            scope_id: self.scope_id,
            availability: self.availability,
            live_owners: state.live_owners.values().cloned().collect(),
            events: state.events.iter().cloned().collect(),
            event_capacity: self.event_capacity,
            completeness: ReceiptCompleteness {
                complete: state.complete,
                reason: state.completeness_reason,
                dropped_events: state.dropped_events,
                dropped_owners: state.dropped_owners,
                rejected_resources: state.rejected_resources,
                dropped_notifications: state.dropped_notifications,
            },
        }
    }

    /// Returns a bounded read-only summary without copying live descriptors.
    pub fn summary(&self) -> RuntimeReceiptSummary {
        let state = self.lock_state();
        RuntimeReceiptSummary {
            scope_id: self.scope_id,
            availability: self.availability,
            live_owner_count: state.live_owners.len(),
            live_resource_count: state
                .live_owners
                .values()
                .map(|owner| owner.resources.len())
                .sum(),
            event_count: state.events.len(),
            event_capacity: self.event_capacity,
            completeness: ReceiptCompleteness {
                complete: state.complete,
                reason: state.completeness_reason,
                dropped_events: state.dropped_events,
                dropped_owners: state.dropped_owners,
                rejected_resources: state.rejected_resources,
                dropped_notifications: state.dropped_notifications,
            },
        }
    }
}

/// Drop guard for one live owner. It is diagnostic-only and never owns the
/// underlying native object, so receipt teardown cannot alter execution.
pub(crate) struct RuntimeOwnerGuard {
    collector: Option<RuntimeReceiptCollector>,
    owner_id: Option<RuntimeOwnerId>,
    attempt_id: Option<ExecutionCacheAttemptId>,
}

impl RuntimeOwnerGuard {
    fn empty() -> Self {
        Self {
            collector: None,
            owner_id: None,
            attempt_id: None,
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

    fn release_inner(&mut self) {
        let Some(owner_id) = self.owner_id.take() else {
            return;
        };
        if let Some(collector) = self.collector.as_ref() {
            collector.release_owner(owner_id, self.attempt_id);
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
    collector: RuntimeReceiptCollector,
    owner_id: RuntimeOwnerId,
    resource_id: RuntimeResourceId,
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
        self.collector
            .transition_resource(self.owner_id, self.resource_id, state);
    }

    pub(crate) fn update_descriptor(&self, descriptor: RuntimeResourceDescriptor) {
        self.collector
            .update_resource(self.owner_id, self.resource_id, descriptor);
    }
}

impl Drop for RuntimeResourceGuard {
    fn drop(&mut self) {
        self.collector
            .release_resource(self.owner_id, self.resource_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_memory::{MemoryDomainKey, PhysicalDeviceKey};

    fn scope() -> NativeExecutionScopeId {
        NativeExecutionScopeId::next()
    }

    fn collector(capacity: usize) -> RuntimeReceiptCollector {
        RuntimeReceiptCollector::new_for_test(scope(), capacity).unwrap()
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
        assert!(snapshot.completeness.dropped_events > 0);
        assert_eq!(second_id.scope_id, snapshot.scope_id);
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
}
