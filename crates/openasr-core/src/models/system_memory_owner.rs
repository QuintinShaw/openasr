//! Owner-bound admission for ordinary Rust system-memory allocations.
//!
//! The decoder topology is semantic and must never own a physical-memory
//! lease. A concrete runtime owner uses this module to reserve provisional
//! engine-requested capacity, perform one fallible allocation transaction,
//! measure requested container capacities against that quote and a fresh live
//! host snapshot, reconcile upward when allocator rounding exceeds the initial
//! estimate, and finally retain the committed lease beside the allocation.
//! Rust container capacity is not allocator usable-size: allocator metadata,
//! size classes, and fragmentation remain covered by policy headroom.

use std::{
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::device::{
    execution_memory::{
        DeviceMemoryBrokerSet, DeviceMemoryReservationBatch, DeviceMemorySnapshot,
        DomainMemoryReconciliation, DomainReservationRequest, MemoryDomainKey,
        MemoryObservationConfidence,
    },
    execution_policy::ExecutionCandidateFailure,
};

use super::native_execution_services::{
    current_memory_reservation_cohort_id, current_native_execution_memory_broker,
    current_native_execution_scope_id, record_current_execution_candidate_failure,
};

/// Checked accumulator for post-build engine-requested Rust heap capacity.
/// `Vec` storage is measured from container capacity and `size_of::<T>()`, never
/// logical length. This deliberately does not claim allocator physical bytes.
/// Callers recurse through initialized elements to add nested Vec/String
/// payloads; the outer Vec's capacity already accounts for every inline element
/// slot, including spare capacity.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SystemMemoryCapacity {
    bytes: u64,
}

impl SystemMemoryCapacity {
    pub(crate) fn add_vec<T>(&mut self, values: &Vec<T>, label: &str) -> Result<(), String> {
        let bytes = values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| format!("{label} Vec capacity byte count overflowed"))?;
        self.add_usize(bytes, label)
    }

    pub(crate) fn add_string(&mut self, value: &String, label: &str) -> Result<(), String> {
        self.add_usize(value.capacity(), label)
    }

    pub(crate) fn add_usize(&mut self, bytes: usize, label: &str) -> Result<(), String> {
        let bytes =
            u64::try_from(bytes).map_err(|_| format!("{label} byte count does not fit u64"))?;
        self.add(bytes, label)
    }

    pub(crate) fn add(&mut self, bytes: u64, label: &str) -> Result<(), String> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("{label} retained system-memory byte sum overflowed"))?;
        Ok(())
    }

    pub(crate) const fn finish(self) -> u64 {
        self.bytes
    }
}

/// Provisional engine-requested heap-capacity quote submitted before an
/// allocation closure runs.
///
/// `peak_bytes` may exceed `retained_bytes` when construction needs temporary
/// readbacks/staging. The closure reports the corresponding measured values
/// after its transient allocations have been released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SystemMemoryAllocationQuote {
    pub(crate) resource_id: String,
    pub(crate) peak_bytes: u64,
    pub(crate) retained_bytes: u64,
}

impl SystemMemoryAllocationQuote {
    pub(crate) fn new(
        resource_id: impl Into<String>,
        peak_bytes: u64,
        retained_bytes: u64,
    ) -> Result<Self, SystemMemoryOwnerError> {
        let resource_id = resource_id.into();
        if resource_id.trim().is_empty() {
            return Err(SystemMemoryOwnerError::new(
                "host_state_quote",
                "system-memory resource id must not be empty",
            ));
        }
        if retained_bytes > peak_bytes {
            return Err(SystemMemoryOwnerError::new(
                "host_state_quote",
                format!(
                    "system-memory retained bytes {retained_bytes} exceed peak bytes {peak_bytes}"
                ),
            ));
        }
        Ok(Self {
            resource_id,
            peak_bytes,
            retained_bytes,
        })
    }
}

/// Result of the fallible allocation closure after construction transients
/// have been dropped. These are engine-requested capacity bytes, not allocator
/// usable-size or physical RSS.
#[derive(Debug)]
pub(crate) struct SystemMemoryAllocationOutcome<T> {
    owner: T,
    requested_peak_bytes: u64,
    requested_retained_bytes: u64,
}

/// Separates a materializer's own typed failure from admission/reconciliation
/// failures owned by this module. A prepared-runtime parser or contract error
/// must remain a build error; only the latter arm is a candidate-capacity
/// failure and may advance execution policy.
#[derive(Debug)]
pub(crate) enum SystemMemoryAllocationTransactionError<E> {
    Allocation(E),
    Capacity(SystemMemoryOwnerError),
}

impl<T> SystemMemoryAllocationOutcome<T> {
    pub(crate) const fn new(
        owner: T,
        requested_peak_bytes: u64,
        requested_retained_bytes: u64,
    ) -> Self {
        Self {
            owner,
            requested_peak_bytes,
            requested_retained_bytes,
        }
    }
}

/// A Rust allocation and its committed engine-requested SystemMemory-capacity
/// lease, which follows the allocation's lifetime.
///
/// Field order is intentional: Rust drops fields in declaration order, so the
/// allocation is destroyed before the lease refunds its committed bytes.
#[derive(Debug)]
pub(crate) struct SystemMemoryOwner<T> {
    owner: T,
    _lease: Option<DeviceMemoryReservationBatch>,
    committed_requested_bytes: u64,
}

/// Cache-neutral handle for any host object whose admission lease must follow
/// every in-flight clone, independently of the cache keying strategy.
pub(crate) type AdmittedHostObject<T> = Arc<SystemMemoryOwner<T>>;

impl<T> SystemMemoryOwner<T> {
    /// Construct a value which provably owns no admitted system-memory payload
    /// (for example an empty host-cache shape on a resident-only route).
    pub(crate) const fn without_allocation(owner: T) -> Self {
        Self {
            owner,
            _lease: None,
            committed_requested_bytes: 0,
        }
    }

    pub(crate) const fn committed_requested_bytes(&self) -> u64 {
        self.committed_requested_bytes
    }

    #[cfg(test)]
    pub(crate) const fn with_committed_requested_bytes_for_test(
        owner: T,
        committed_requested_bytes: u64,
    ) -> Self {
        Self {
            owner,
            _lease: None,
            committed_requested_bytes,
        }
    }

    /// Reserve, allocate, measure, reconcile, then bind the committed lease to
    /// the actual owner. Production native scopes fail closed if their injected
    /// broker or live host observation is unavailable. Low-level tests outside
    /// a native scope still exercise the allocator without global accounting.
    pub(crate) fn try_allocate(
        quote: SystemMemoryAllocationQuote,
        allocate: impl FnOnce() -> Result<SystemMemoryAllocationOutcome<T>, String>,
    ) -> Result<Self, SystemMemoryOwnerError> {
        match Self::try_allocate_transaction(quote, allocate) {
            Ok(owner) => Ok(owner),
            Err(SystemMemoryAllocationTransactionError::Capacity(error)) => Err(error),
            Err(SystemMemoryAllocationTransactionError::Allocation(reason)) => Err(
                SystemMemoryOwnerError::capacity_failure("host_state_allocate", reason),
            ),
        }
    }

    /// The typed transaction used by shared materialization caches. The
    /// caller's build error is never reclassified as an OOM/capacity failure;
    /// provisional admission and post-build quote-validation failures are.
    pub(crate) fn try_allocate_transaction<E>(
        quote: SystemMemoryAllocationQuote,
        allocate: impl FnOnce() -> Result<SystemMemoryAllocationOutcome<T>, E>,
    ) -> Result<Self, SystemMemoryAllocationTransactionError<E>> {
        let broker = current_native_execution_memory_broker();
        let broker_required = current_native_execution_scope_id().is_some();
        Self::try_allocate_transaction_with(
            quote,
            broker,
            broker_required,
            observe_host_memory,
            allocate,
        )
    }

    #[cfg(test)]
    fn try_allocate_with(
        quote: SystemMemoryAllocationQuote,
        broker: Option<Arc<DeviceMemoryBrokerSet>>,
        broker_required: bool,
        observe: impl FnMut() -> Result<DeviceMemorySnapshot, String>,
        allocate: impl FnOnce() -> Result<SystemMemoryAllocationOutcome<T>, String>,
    ) -> Result<Self, SystemMemoryOwnerError> {
        match Self::try_allocate_transaction_with(quote, broker, broker_required, observe, allocate)
        {
            Ok(owner) => Ok(owner),
            Err(SystemMemoryAllocationTransactionError::Capacity(error)) => Err(error),
            Err(SystemMemoryAllocationTransactionError::Allocation(reason)) => Err(
                SystemMemoryOwnerError::capacity_failure("host_state_allocate", reason),
            ),
        }
    }

    fn try_allocate_transaction_with<E>(
        quote: SystemMemoryAllocationQuote,
        broker: Option<Arc<DeviceMemoryBrokerSet>>,
        broker_required: bool,
        mut observe: impl FnMut() -> Result<DeviceMemorySnapshot, String>,
        allocate: impl FnOnce() -> Result<SystemMemoryAllocationOutcome<T>, E>,
    ) -> Result<Self, SystemMemoryAllocationTransactionError<E>> {
        let Some(broker) = broker else {
            if broker_required || !cfg!(test) {
                return Err(SystemMemoryAllocationTransactionError::Capacity(
                    SystemMemoryOwnerError::capacity_failure(
                        "host_state_admission",
                        "system-memory allocation has no injected process-wide memory broker",
                    ),
                ));
            }
            let outcome = allocate().map_err(SystemMemoryAllocationTransactionError::Allocation)?;
            validate_outcome(&outcome).map_err(SystemMemoryAllocationTransactionError::Capacity)?;
            return Ok(Self::without_allocation(outcome.owner));
        };

        let snapshot_before = observe().map_err(|reason| {
            SystemMemoryAllocationTransactionError::Capacity(
                SystemMemoryOwnerError::capacity_failure("host_state_observe_before", reason),
            )
        })?;
        let quoted_peak_bytes = quote.peak_bytes;
        let quoted_retained_bytes = quote.retained_bytes;
        let reservation_cohort = current_memory_reservation_cohort_id();
        let resource_id = quote.resource_id;
        let wait_deadline = Instant::now() + Duration::from_secs(30);
        let mut retry_delay = Duration::from_millis(1);
        let mut snapshot_before = snapshot_before;
        let mut reservation = loop {
            match broker.try_reserve_batch(vec![DomainReservationRequest {
                domain: MemoryDomainKey::SystemMemory,
                snapshot: snapshot_before,
                peak_bytes: quoted_peak_bytes,
                retained_bytes: quoted_retained_bytes,
                // Rust allocator capacity is measured only after construction;
                // the provisional/exclusive path is what makes reconciliation
                // safe even when `reserve_exact` rounds upward.
                requires_reconciliation: true,
                resource_id: resource_id.clone(),
                cohort_id: reservation_cohort,
            }]) {
                Ok(reservation) => break reservation,
                Err(crate::device::execution_memory::MemoryPlanningError::DeviceDomainBusy {
                    ..
                }) if Instant::now() < wait_deadline => {
                    // A concurrent cold runtime is still measuring its actual
                    // allocation in this physical domain. That is transient
                    // serialization imposed by the broker, not evidence that
                    // this candidate is over capacity. Wait briefly, refresh
                    // the live snapshot, and retry the same candidate instead
                    // of spuriously falling back to a different backend.
                    thread::sleep(retry_delay);
                    retry_delay = (retry_delay * 2).min(Duration::from_millis(32));
                    snapshot_before = observe().map_err(|reason| {
                        SystemMemoryAllocationTransactionError::Capacity(
                            SystemMemoryOwnerError::capacity_failure(
                                "host_state_observe_before",
                                reason,
                            ),
                        )
                    })?;
                }
                Err(error) => {
                    let reason = error.to_string();
                    return Err(SystemMemoryAllocationTransactionError::Capacity(
                        SystemMemoryOwnerError::capacity_failure("host_state_admission", reason),
                    ));
                }
            }
        };

        let outcome = allocate().map_err(SystemMemoryAllocationTransactionError::Allocation)?;
        validate_outcome(&outcome).map_err(SystemMemoryAllocationTransactionError::Capacity)?;
        let reconciled_peak_bytes = quoted_peak_bytes.max(outcome.requested_peak_bytes);
        let reconciled_retained_bytes = quoted_retained_bytes.max(outcome.requested_retained_bytes);
        let snapshot_after = observe().map_err(|reason| {
            SystemMemoryAllocationTransactionError::Capacity(
                SystemMemoryOwnerError::capacity_failure("host_state_observe_after", reason),
            )
        })?;
        reservation
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: MemoryDomainKey::SystemMemory,
                // Container capacities are only observable after construction.
                // Never shrink below the shape quote, but let the provisional
                // transaction reconcile allocator rounding or a conservative
                // family counter that was smaller than the measured capacity.
                actual_peak_bytes: reconciled_peak_bytes,
                actual_retained_bytes: reconciled_retained_bytes,
                snapshot_after,
            }])
            .map_err(|error| {
                let reason = error.to_string();
                SystemMemoryAllocationTransactionError::Capacity(
                    SystemMemoryOwnerError::capacity_failure("host_state_reconcile", reason),
                )
            })?;
        Ok(Self {
            owner: outcome.owner,
            _lease: Some(reservation),
            committed_requested_bytes: reconciled_retained_bytes,
        })
    }
}

impl SystemMemoryOwner<()> {
    /// Atomically reserves SystemMemory capacity for fallible allocations made
    /// by an invocation whose individual containers cannot own the broker
    /// lease. The returned guard must outlive every covered allocation.
    pub(crate) fn try_reserve_invocation(
        resource_id: impl Into<String>,
        bytes: u64,
    ) -> Result<Self, SystemMemoryOwnerError> {
        if bytes == 0 {
            return Ok(Self::without_allocation(()));
        }
        let quote = SystemMemoryAllocationQuote::new(resource_id, bytes, bytes)?;
        Self::try_allocate(quote, || {
            Ok(SystemMemoryAllocationOutcome::new((), bytes, bytes))
        })
    }
}

impl<T> Deref for SystemMemoryOwner<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.owner
    }
}

impl<T> DerefMut for SystemMemoryOwner<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.owner
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{stage}: {reason}")]
pub(crate) struct SystemMemoryOwnerError {
    stage: &'static str,
    reason: String,
}

impl SystemMemoryOwnerError {
    fn new(stage: &'static str, reason: impl Into<String>) -> Self {
        Self {
            stage,
            reason: reason.into(),
        }
    }

    pub(crate) fn capacity_failure(stage: &'static str, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        record_capacity_failure(stage, &reason);
        Self::new(stage, reason)
    }
}

fn validate_outcome<T>(
    outcome: &SystemMemoryAllocationOutcome<T>,
) -> Result<(), SystemMemoryOwnerError> {
    if outcome.requested_retained_bytes > outcome.requested_peak_bytes {
        return fail(
            "host_state_measure",
            format!(
                "requested retained bytes {} exceed requested peak bytes {}",
                outcome.requested_retained_bytes, outcome.requested_peak_bytes
            ),
        );
    }
    Ok(())
}

fn observe_host_memory() -> Result<DeviceMemorySnapshot, String> {
    let total_bytes = crate::host::host_total_memory_bytes()
        .ok_or_else(|| "host total-memory observation is unavailable".to_string())?;
    let free_bytes = crate::host::host_available_memory_bytes()
        .ok_or_else(|| "host available-memory observation is unavailable".to_string())?;
    Ok(DeviceMemorySnapshot {
        free_bytes,
        total_bytes,
        confidence: MemoryObservationConfidence::DeviceSnapshot,
    })
}

fn fail<T>(stage: &'static str, reason: impl Into<String>) -> Result<T, SystemMemoryOwnerError> {
    Err(SystemMemoryOwnerError::capacity_failure(stage, reason))
}

fn record_capacity_failure(stage: &'static str, reason: &str) {
    record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
        stage,
        reason.to_string(),
    ));
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::thread;

    use crate::device::execution_memory::{
        DeviceMemoryPolicy, DeviceMemoryUsage, MemoryPlanningError,
    };

    use super::*;

    fn snapshot(free_bytes: u64) -> DeviceMemorySnapshot {
        DeviceMemorySnapshot {
            free_bytes,
            total_bytes: 256,
            confidence: MemoryObservationConfidence::DeviceSnapshot,
        }
    }

    fn test_broker() -> Arc<DeviceMemoryBrokerSet> {
        Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }))
    }

    #[test]
    fn pending_allocation_excludes_a_competing_system_memory_candidate() {
        let broker = test_broker();
        let competing_broker = Arc::clone(&broker);
        let quote = SystemMemoryAllocationQuote::new("test.first", 96, 96).unwrap();
        let owner = SystemMemoryOwner::try_allocate_with(
            quote,
            Some(Arc::clone(&broker)),
            true,
            || Ok(snapshot(200)),
            move || {
                let rejected = competing_broker.try_reserve_batch(vec![DomainReservationRequest {
                    domain: MemoryDomainKey::SystemMemory,
                    snapshot: snapshot(200),
                    peak_bytes: 96,
                    retained_bytes: 96,
                    requires_reconciliation: true,
                    resource_id: "test.competing".to_string(),
                    cohort_id: None,
                }]);
                assert!(matches!(
                    rejected,
                    Err(MemoryPlanningError::DeviceDomainBusy { .. })
                ));
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(96)
                    .map_err(|error| error.to_string())?;
                bytes.resize(96, 0_u8);
                let actual = bytes.capacity() as u64;
                Ok(SystemMemoryAllocationOutcome::new(bytes, actual, actual))
            },
        )
        .unwrap();
        let usage = broker.usage(&MemoryDomainKey::SystemMemory);
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.committed_bytes, owner.capacity() as u64);
        drop(owner);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }

    #[test]
    fn pending_allocation_excludes_a_concurrent_system_memory_candidate() {
        let broker = test_broker();
        let allocating_broker = Arc::clone(&broker);
        let (pending_tx, pending_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let allocating = thread::spawn(move || {
            let owner = SystemMemoryOwner::try_allocate_with(
                SystemMemoryAllocationQuote::new("test.concurrent-first", 96, 96).unwrap(),
                Some(allocating_broker),
                true,
                || Ok(snapshot(200)),
                move || {
                    pending_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 96], 96, 96))
                },
            )
            .unwrap();
            drop(owner);
        });

        pending_rx.recv().unwrap();
        let rejected = broker.try_reserve_batch(vec![DomainReservationRequest {
            domain: MemoryDomainKey::SystemMemory,
            snapshot: snapshot(200),
            peak_bytes: 96,
            retained_bytes: 96,
            requires_reconciliation: true,
            resource_id: "test.concurrent-competing".to_string(),
            cohort_id: None,
        }]);
        assert!(matches!(
            rejected,
            Err(MemoryPlanningError::DeviceDomainBusy { .. })
        ));
        release_tx.send(()).unwrap();
        allocating.join().unwrap();
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }

    #[test]
    fn concurrent_owner_waits_for_provisional_gate_then_reobserves_and_succeeds() {
        let broker = test_broker();
        let allocating_broker = Arc::clone(&broker);
        let waiting_broker = Arc::clone(&broker);
        let (pending_tx, pending_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (allocated_tx, allocated_rx) = mpsc::channel();

        let allocating = thread::spawn(move || {
            let owner = SystemMemoryOwner::try_allocate_with(
                SystemMemoryAllocationQuote::new("test.wait-first", 64, 64).unwrap(),
                Some(allocating_broker),
                true,
                || Ok(snapshot(256)),
                move || {
                    pending_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 64], 64, 64))
                },
            )
            .unwrap();
            drop(owner);
        });

        pending_rx.recv().unwrap();
        let waiting = thread::spawn(move || {
            let owner = SystemMemoryOwner::try_allocate_with(
                SystemMemoryAllocationQuote::new("test.wait-second", 64, 64).unwrap(),
                Some(waiting_broker),
                true,
                || Ok(snapshot(256)),
                || Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 64], 64, 64)),
            )
            .unwrap();
            allocated_tx.send(()).unwrap();
            drop(owner);
        });

        assert!(matches!(
            allocated_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release_tx.send(()).unwrap();
        allocated_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("the waiting owner should retry after the provisional gate clears");
        allocating.join().unwrap();
        waiting.join().unwrap();
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }

    struct DropProbe {
        broker: Arc<DeviceMemoryBrokerSet>,
        allocation_dropped_before_lease: Arc<AtomicBool>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.allocation_dropped_before_lease.store(
                self.broker
                    .usage(&MemoryDomainKey::SystemMemory)
                    .committed_bytes
                    != 0,
                Ordering::SeqCst,
            );
        }
    }

    #[test]
    fn allocation_drops_before_its_committed_lease_refunds_bytes() {
        let broker = test_broker();
        let observed = Arc::new(AtomicBool::new(false));
        let probe = DropProbe {
            broker: Arc::clone(&broker),
            allocation_dropped_before_lease: Arc::clone(&observed),
        };
        let owner = SystemMemoryOwner::try_allocate_with(
            SystemMemoryAllocationQuote::new("test.drop-order", 1, 1).unwrap(),
            Some(Arc::clone(&broker)),
            true,
            || Ok(snapshot(200)),
            || Ok(SystemMemoryAllocationOutcome::new(probe, 1, 1)),
        )
        .unwrap();
        drop(owner);
        assert!(observed.load(Ordering::SeqCst));
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }

    #[test]
    fn measured_peak_may_exceed_retained_after_transients_drop() {
        let broker = test_broker();
        let owner = SystemMemoryOwner::try_allocate_with(
            SystemMemoryAllocationQuote::new("test.transient", 128, 64).unwrap(),
            Some(Arc::clone(&broker)),
            true,
            || Ok(snapshot(128)),
            || Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 64], 128, 64)),
        )
        .unwrap();
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            64
        );
        drop(owner);
    }

    #[test]
    fn smaller_requested_capacity_does_not_shrink_the_committed_quote() {
        let broker = test_broker();
        let owner = SystemMemoryOwner::try_allocate_with(
            SystemMemoryAllocationQuote::new("test.no-shrink", 128, 96).unwrap(),
            Some(Arc::clone(&broker)),
            true,
            || Ok(snapshot(200)),
            || Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 32], 32, 32)),
        )
        .unwrap();
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            96
        );
        drop(owner);
    }

    #[test]
    fn requested_capacity_above_the_provisional_quote_reconciles_upward() {
        let broker = test_broker();
        let owner = SystemMemoryOwner::try_allocate_with(
            SystemMemoryAllocationQuote::new("test.quote-violation", 64, 64).unwrap(),
            Some(Arc::clone(&broker)),
            true,
            || Ok(snapshot(200)),
            || Ok(SystemMemoryAllocationOutcome::new(vec![0_u8; 65], 65, 65)),
        )
        .expect("provisional owner reconciles allocator rounding");
        assert_eq!(owner.committed_requested_bytes(), 65);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            65
        );
        drop(owner);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }
}
