//! Process-wide shared residency for file-backed pack weight mappings.
//!
//! A combination pack (encoder + adapter + decoder in one `.oasr`) is mmap'd
//! once and then bound by several stage-local `GgmlLoadedWeightContext`s. Each
//! stage asks the backend for a HOST_IMPORT of the *same* mapping. Charging
//! `mmap.len()` into [`super::execution_memory::MemoryDomainKey::SystemMemory`]
//! once per stage double-counts one physical mapping; charging zero for every
//! FILE_BACKED claim under-counts concurrent distinct packs and real working
//! sets.
//!
//! This ledger keys on `(physical domain, open mapping identity)`:
//!
//! - first live owner of a mapping reserves and commits the quoted byte size
//!   on the **policy** ledger (distinct packs still add);
//! - further owners of the *same* mapping share that charge (zero incremental);
//! - the last owner drop refunds **only** if the table entry's generation still
//!   matches the handle (prevents ABA: a concurrent re-acquire must not have
//!   its reservation refunded by a stale Drop);
//! - already-open file-backed mappings do not require `observed free >= pack
//!   size` (clean pages are reclaimable); policy still tracks full size so two
//!   concurrent distinct packs cannot both admit against the full RAM budget.
//!
//! Callers still perform the native host-import; they must set
//! `currently_allocated_bytes = requested_bytes` on the HOST_IMPORT quote so
//! the backend reports a reuse (zero incremental) and this lease remains the
//! sole SystemMemory policy charge for the mapping.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use super::execution_memory::{
    DeviceMemoryBrokerSet, DeviceMemoryReservationBatch, DeviceMemorySnapshot, MemoryDomainKey,
    MemoryPlanningError, MemoryReservationCohortId,
};
use crate::models::native_execution_services::{
    current_execution_cache_attempt_id, current_native_execution_scope_id, current_runtime_receipts,
};
use crate::models::runtime_receipts::{
    RuntimeOwnerGuard, RuntimeOwnerPlacement, RuntimeResourceGuard, RuntimeResourceState,
};

/// Process-local identity of one already-open pack weight mapping.
///
/// Construct only from the live `Arc<Mmap>` that owns the host import. The
/// identity is the Arc allocation pointer: clones of the same open mapping
/// share it, and a separately admitted file (even at the same path) gets a
/// fresh Arc and therefore a distinct identity. Generation tokens on the
/// residency table handle ABA across drop/re-acquire of the same key; this
/// type only names *which* open mapping is being charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PackWeightMappingIdentity(usize);

impl PackWeightMappingIdentity {
    /// Identity of the open mapping owned by `mmap`.
    fn from_open_mmap(mmap: &Arc<memmap2::Mmap>) -> Self {
        Self(std::sync::Arc::as_ptr(mmap) as usize)
    }

    #[cfg(test)]
    pub(crate) fn from_raw_for_test(id: usize) -> Self {
        Self(id)
    }

    pub(crate) fn as_raw(self) -> usize {
        self.0
    }
}

/// Identity of one already-open pack mapping in one physical memory domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PackWeightResidencyKey {
    pub domain: MemoryDomainKey,
    pub mapping_identity: PackWeightMappingIdentity,
}

#[derive(Debug)]
pub(crate) struct PackWeightResidencyEntry {
    charged_bytes: u64,
    /// Monotonic token assigned at insert. Drop only refunds when this still
    /// matches the live entry (stale Drop after concurrent re-acquire is a no-op).
    generation: u64,
    /// Committed broker batch while any handle is live. Taken on last matching drop.
    reservation: Option<DeviceMemoryReservationBatch>,
    /// Strong count is tracked via [`Arc`] handles; this Weak lets a new
    /// acquirer join an existing entry without racing a concurrent last drop.
    live: Weak<PackWeightResidencyInner>,
}

#[derive(Debug)]
struct PackWeightResidencyInner {
    broker: Arc<DeviceMemoryBrokerSet>,
    key: PackWeightResidencyKey,
    generation: u64,
    charged_bytes: u64,
    /// The unique pack-level receipt follows this same Arc as the sole broker
    /// reservation. It is attached only after native host-import succeeds.
    receipt: Mutex<Option<(RuntimeResourceGuard, RuntimeOwnerGuard)>>,
    /// Owns the exact mapping whose Arc allocation address forms `key`.
    /// Keeping this clone until the last residency handle drops makes address
    /// reuse impossible while the table can still upgrade `live`.
    #[allow(dead_code)]
    mapping_owner: Option<Arc<memmap2::Mmap>>,
}

/// One owner of a shared pack-weight residency charge. Clone freely; the
/// underlying SystemMemory commitment is released when the last clone drops
/// **and** the table entry generation still matches.
#[derive(Debug, Clone)]
pub(crate) struct PackWeightResidencyHandle {
    /// Load-bearing: keeps the shared residency Arc alive until the last stage
    /// that bound this mapping drops. Not read outside Drop of the Arc.
    #[allow(dead_code)]
    inner: Arc<PackWeightResidencyInner>,
    #[cfg(test)]
    charged_bytes: u64,
}

impl PackWeightResidencyHandle {
    /// Attach the one pack-level receipt after the native host-import has
    /// succeeded. Concurrent stage/backend contexts share this Arc and only
    /// the first successful attach creates an owner/resource pair.
    pub(crate) fn attach_receipt(&self) {
        self.inner.attach_receipt();
    }

    pub(crate) fn record_receipt_reuse(&self) {
        self.inner.record_receipt_reuse();
    }

    #[cfg(test)]
    pub(crate) fn charged_bytes(&self) -> u64 {
        self.charged_bytes
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.inner.generation
    }
}

impl PackWeightResidencyInner {
    fn attach_receipt(&self) {
        let Some(collector) =
            current_runtime_receipts().filter(|collector| collector.is_available())
        else {
            return;
        };
        let mut receipt = self
            .receipt
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if receipt.is_some() {
            return;
        }
        let Some(descriptor) =
            collector.host_neutral_owner_descriptor("pack-weight-residency", None, None)
        else {
            return;
        };
        let owner = collector.start_owner(descriptor, current_execution_cache_attempt_id());
        let Some(owner_id) = owner.owner_id() else {
            return;
        };
        let Some(descriptor) = collector.resource_descriptor(
            "pack-weight-residency",
            &self.key.domain,
            self.charged_bytes,
            self.charged_bytes,
            self.charged_bytes,
            super::execution_memory::QuoteConfidence::CommittedUpperBound,
            Some(super::execution_memory::MemoryObservationConfidence::DeviceSnapshot),
        ) else {
            return;
        };
        let Some(resource) = collector.acquire_resource(owner_id, descriptor) else {
            return;
        };
        // Receipts are attached after the broker lease has already committed.
        resource.set_state(RuntimeResourceState::Committed);
        *receipt = Some((resource, owner));
    }

    fn record_receipt_reuse(&self) {
        let receipt = self
            .receipt
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some((_, owner)) = receipt.as_ref() {
            owner.record_reuse(current_execution_cache_attempt_id());
        }
    }
}

impl Drop for PackWeightResidencyInner {
    fn drop(&mut self) {
        self.broker
            .release_pack_weight_residency(&self.key, self.generation);
    }
}

impl DeviceMemoryBrokerSet {
    /// Acquire shared residency for an exact, already-open pack mapping.
    ///
    /// Identity and byte size are derived inside this Interface from the same
    /// owning `Arc<Mmap>` that the returned handle retains. Callers therefore
    /// cannot pair an unrelated identity/size with a lease, and the allocator
    /// cannot reuse the identity address while a table entry remains live.
    pub(crate) fn acquire_open_pack_weight_residency(
        self: &Arc<Self>,
        domain: MemoryDomainKey,
        mapping: Arc<memmap2::Mmap>,
        snapshot: DeviceMemorySnapshot,
        cohort_id: Option<MemoryReservationCohortId>,
    ) -> Result<(PackWeightResidencyHandle, u64), MemoryPlanningError> {
        let bytes = u64::try_from(mapping.len()).map_err(|_| {
            MemoryPlanningError::ReservationLedgerCorrupted {
                domain: domain.clone(),
            }
        })?;
        let key = PackWeightResidencyKey {
            domain,
            mapping_identity: PackWeightMappingIdentity::from_open_mmap(&mapping),
        };
        self.acquire_pack_weight_residency_inner(key, bytes, snapshot, cohort_id, Some(mapping))
    }

    /// Acquire shared residency for one open pack mapping.
    ///
    /// Returns `(handle, incremental_bytes_charged_now)`. `incremental` is the
    /// full `bytes` on the first live owner and `0` when joining an existing
    /// charge for the same mapping identity.
    ///
    /// `snapshot` must reflect **live** host free/total. Already-open file-backed
    /// residency does not need `free >= bytes` (observed peak is 0); the policy
    /// ledger still charges `bytes` so concurrent distinct packs fail closed.
    #[cfg(test)]
    pub(crate) fn acquire_pack_weight_residency(
        self: &Arc<Self>,
        key: PackWeightResidencyKey,
        bytes: u64,
        snapshot: DeviceMemorySnapshot,
        cohort_id: Option<MemoryReservationCohortId>,
    ) -> Result<(PackWeightResidencyHandle, u64), MemoryPlanningError> {
        self.acquire_pack_weight_residency_inner(key, bytes, snapshot, cohort_id, None)
    }

    fn acquire_pack_weight_residency_inner(
        self: &Arc<Self>,
        key: PackWeightResidencyKey,
        bytes: u64,
        snapshot: DeviceMemorySnapshot,
        cohort_id: Option<MemoryReservationCohortId>,
        mapping_owner: Option<Arc<memmap2::Mmap>>,
    ) -> Result<(PackWeightResidencyHandle, u64), MemoryPlanningError> {
        if bytes == 0 {
            return Err(MemoryPlanningError::EmptyResourceId);
        }
        let mut table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = table.get(&key)
            && let Some(inner) = entry.live.upgrade()
        {
            if entry.charged_bytes != bytes {
                // Same mapping must not be re-quoted at a different size;
                // that would mean two readers disagreed on mmap.len().
                return Err(MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: key.domain.clone(),
                });
            }
            inner.record_receipt_reuse();
            return Ok((
                PackWeightResidencyHandle {
                    inner,
                    #[cfg(test)]
                    charged_bytes: bytes,
                },
                0,
            ));
        }

        // Dead weak (last owner Drop in flight or finished without removing yet)
        // or missing key: refund any stale reservation **before** quoting a new
        // one. Otherwise last-drop/reacquire overlap double-counts policy peak
        // and can fail-closed while physical residency is only one mapping.
        if let Some(mut stale) = table.remove(&key) {
            drop(stale.reservation.take());
        }

        // First live owner (or re-acquire after last drop): reserve under the
        // ordinary domain ledger. Policy peak = full mapping size. Observed
        // peak = 0 because the mmap is already open at preflight and host-import
        // does not allocate a second anonymous copy of those bytes.
        //
        // Hold the residency table lock across reserve+insert so a concurrent
        // last Drop cannot observe a half-published generation, and so the
        // stale refund above is atomic with the new charge relative to other
        // acquirers of this key.
        let mut request = super::execution_memory::DomainReservationRequest {
            domain: key.domain.clone(),
            snapshot,
            peak_bytes: bytes,
            retained_bytes: bytes,
            // Already-resident file-backed pages are reclaimable; do not require
            // live free == pack size. Policy ledger still prevents oversell.
            observed_peak_bytes: Some(0),
            requires_reconciliation: false,
            resource_id: format!(
                "pack-weight-residency:{}:{:#x}",
                key.domain,
                key.mapping_identity.as_raw()
            ),
            cohort_id: None,
        };
        request.cohort_id = cohort_id;
        let owner_scope_id = current_native_execution_scope_id();
        let owner_placement = if owner_scope_id.is_some() {
            RuntimeOwnerPlacement::HostNeutral
        } else {
            RuntimeOwnerPlacement::Unknown
        };
        let mut batch = self.try_reserve_batch_for_scope_and_placement(
            vec![request],
            owner_scope_id,
            owner_placement,
        )?;
        batch.commit_quoted()?;

        let generation = self
            .next_pack_weight_residency_generation
            .fetch_add(1, Ordering::Relaxed);
        let inner = Arc::new(PackWeightResidencyInner {
            broker: Arc::clone(self),
            key: key.clone(),
            generation,
            charged_bytes: bytes,
            receipt: Mutex::new(None),
            mapping_owner,
        });
        table.insert(
            key,
            PackWeightResidencyEntry {
                charged_bytes: bytes,
                generation,
                reservation: Some(batch),
                live: Arc::downgrade(&inner),
            },
        );
        Ok((
            PackWeightResidencyHandle {
                inner,
                #[cfg(test)]
                charged_bytes: bytes,
            },
            bytes,
        ))
    }

    /// Refund residency only when `generation` still owns the table entry.
    /// A stale Drop after a concurrent re-acquire of the same mapping key is
    /// a deliberate no-op so the new reservation cannot be refunded early.
    fn release_pack_weight_residency(&self, key: &PackWeightResidencyKey, generation: u64) {
        let mut table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(entry) = table.get(key) else {
            return;
        };
        if entry.generation != generation {
            return;
        }
        let Some(mut entry) = table.remove(key) else {
            return;
        };
        // Dropping the committed batch refunds SystemMemory.
        drop(entry.reservation.take());
    }

    #[cfg(test)]
    pub(crate) fn pack_weight_residency_live_count(&self) -> usize {
        let table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        table
            .values()
            .filter(|entry| entry.live.strong_count() > 0)
            .count()
    }

    #[cfg(test)]
    pub(crate) fn pack_weight_residency_generation(
        &self,
        key: &PackWeightResidencyKey,
    ) -> Option<u64> {
        let table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        table.get(key).map(|entry| entry.generation)
    }

    /// Test-only: install a **dead** table entry that still holds a live
    /// SystemMemory reservation. Models the last-drop/reacquire window where
    /// the previous owner's `Weak` is already dead but refund has not run (or
    /// was skipped). Production `acquire_pack_weight_residency` must refund this
    /// stale charge **before** quoting a new one; without that step a single
    /// reacquire against a one-slot budget fails closed.
    #[cfg(test)]
    pub(crate) fn inject_dead_pack_weight_residency_with_live_reservation_for_test(
        self: &Arc<Self>,
        key: PackWeightResidencyKey,
        bytes: u64,
        snapshot: DeviceMemorySnapshot,
    ) -> Result<(), MemoryPlanningError> {
        if bytes == 0 {
            return Err(MemoryPlanningError::EmptyResourceId);
        }
        let mut table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if table.contains_key(&key) {
            return Err(MemoryPlanningError::ReservationLedgerCorrupted {
                domain: key.domain.clone(),
            });
        }
        let mut request = super::execution_memory::DomainReservationRequest {
            domain: key.domain.clone(),
            snapshot,
            peak_bytes: bytes,
            retained_bytes: bytes,
            observed_peak_bytes: Some(0),
            requires_reconciliation: false,
            resource_id: format!(
                "pack-weight-residency-dead-inject:{}:{:#x}",
                key.domain,
                key.mapping_identity.as_raw()
            ),
            cohort_id: None,
        };
        request.cohort_id = None;
        let mut batch = self.try_reserve_batch(vec![request])?;
        batch.commit_quoted()?;
        let generation = self
            .next_pack_weight_residency_generation
            .fetch_add(1, Ordering::Relaxed);
        table.insert(
            key,
            PackWeightResidencyEntry {
                charged_bytes: bytes,
                generation,
                reservation: Some(batch),
                // No live owner: upgrade() returns None, matching the race window.
                live: Weak::new(),
            },
        );
        Ok(())
    }
}

pub(crate) fn empty_pack_weight_residency_table()
-> Mutex<HashMap<PackWeightResidencyKey, PackWeightResidencyEntry>> {
    Mutex::new(HashMap::new())
}

pub(crate) fn new_pack_weight_residency_generation_counter() -> AtomicU64 {
    // Start at 1 so generation 0 is never a live token (easier debug).
    AtomicU64::new(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_memory::{
        DeviceMemoryPolicy, DeviceMemorySnapshot, MemoryDomainKey, MemoryObservationConfidence,
    };
    use std::sync::Barrier;
    use std::thread;

    const GIB: u64 = 1 << 30;

    fn snapshot(free: u64, total: u64) -> DeviceMemorySnapshot {
        DeviceMemorySnapshot {
            free_bytes: free,
            total_bytes: total,
            confidence: MemoryObservationConfidence::DeviceSnapshot,
        }
        .normalized()
        .expect("snapshot")
    }

    fn key(id: usize) -> PackWeightResidencyKey {
        PackWeightResidencyKey {
            domain: MemoryDomainKey::SystemMemory,
            mapping_identity: PackWeightMappingIdentity::from_raw_for_test(id),
        }
    }

    #[test]
    fn same_mapping_is_charged_once_across_handles() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(16 * GIB, 16 * GIB);
        let (a, charged_a) = broker
            .acquire_pack_weight_residency(key(0xA), 4 * GIB, snap, None)
            .expect("first");
        assert_eq!(charged_a, 4 * GIB);
        let (b, charged_b) = broker
            .acquire_pack_weight_residency(key(0xA), 4 * GIB, snap, None)
            .expect("second share");
        assert_eq!(charged_b, 0);
        assert_eq!(a.charged_bytes(), b.charged_bytes());
        assert_eq!(a.generation(), b.generation());
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            4 * GIB
        );
        drop(a);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            4 * GIB,
            "still held by b"
        );
        drop(b);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
        assert_eq!(broker.pack_weight_residency_live_count(), 0);
    }

    #[test]
    fn shared_receipt_is_one_owner_until_the_last_mapping_handle_drops() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope =
            crate::models::native_execution_services::install_native_execution_services(&services);
        let broker = Arc::clone(services.memory_broker());
        let snap = snapshot(16 * GIB, 16 * GIB);
        let mapping = key(0xFACE);
        let (first, charged) = broker
            .acquire_pack_weight_residency(mapping.clone(), 4 * GIB, snap, None)
            .expect("first mapping owner");
        assert_eq!(charged, 4 * GIB);
        first.attach_receipt();
        let (second, shared) = broker
            .acquire_pack_weight_residency(mapping, 4 * GIB, snap, None)
            .expect("second mapping owner");
        assert_eq!(shared, 0);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 1);
        assert_eq!(services.runtime_receipts().summary().live_resource_count, 1);

        drop(first);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 1);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            4 * GIB
        );
        drop(second);
        assert_eq!(services.runtime_receipts().summary().live_owner_count, 0);
        assert_eq!(services.runtime_receipts().summary().live_resource_count, 0);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    #[test]
    fn production_lease_owns_mapping_until_the_last_handle_drops() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("temp mapping");
        file.write_all(&[0_u8; 4096]).expect("seed mapping");
        file.flush().expect("flush mapping");
        let mapping = Arc::new(unsafe {
            memmap2::MmapOptions::new()
                .map(file.as_file())
                .expect("map file")
        });
        let weak = Arc::downgrade(&mapping);
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(16 * GIB, 16 * GIB);
        let (first, charged) = broker
            .acquire_open_pack_weight_residency(
                MemoryDomainKey::SystemMemory,
                Arc::clone(&mapping),
                snap,
                None,
            )
            .expect("first production lease");
        assert_eq!(charged, 4096);
        let (second, shared) = broker
            .acquire_open_pack_weight_residency(
                MemoryDomainKey::SystemMemory,
                Arc::clone(&mapping),
                snap,
                None,
            )
            .expect("shared production lease");
        assert_eq!(shared, 0);

        drop(mapping);
        assert!(
            weak.upgrade().is_some(),
            "residency lease must retain the identity allocation"
        );
        drop(first);
        assert!(weak.upgrade().is_some(), "second handle still owns mapping");
        drop(second);
        assert!(
            weak.upgrade().is_none(),
            "last lease drop must release the mapping owner"
        );
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    #[test]
    fn distinct_mappings_add() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(16 * GIB, 16 * GIB);
        let (_a, ca) = broker
            .acquire_pack_weight_residency(key(1), 3 * GIB, snap, None)
            .expect("a");
        let (_b, cb) = broker
            .acquire_pack_weight_residency(key(2), 5 * GIB, snap, None)
            .expect("b");
        assert_eq!(ca, 3 * GIB);
        assert_eq!(cb, 5 * GIB);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            8 * GIB
        );
    }

    #[test]
    fn second_mapping_fails_closed_when_budget_exhausted() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        // total 6 GiB, first takes 4, second wants 4 -> fail on policy ledger
        let snap = snapshot(6 * GIB, 6 * GIB);
        let _a = broker
            .acquire_pack_weight_residency(key(1), 4 * GIB, snap, None)
            .expect("first");
        let err = broker
            .acquire_pack_weight_residency(key(2), 4 * GIB, snap, None)
            .expect_err("second must fail closed");
        assert!(matches!(
            err,
            MemoryPlanningError::DeviceBudgetExceeded { .. }
        ));
    }

    #[test]
    fn already_open_mapping_admits_when_live_free_is_below_pack_size() {
        // Policy still has headroom (total 16 GiB, nothing committed). Live free
        // is only 1 GiB — below the 4 GiB pack — but file-backed residency of an
        // already-open mmap must not require free >= pack size.
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(GIB, 16 * GIB);
        let (_h, charged) = broker
            .acquire_pack_weight_residency(key(7), 4 * GIB, snap, None)
            .expect("already-open file-backed pack must admit on policy alone");
        assert_eq!(charged, 4 * GIB);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            4 * GIB
        );
    }

    #[test]
    fn stale_drop_does_not_refund_concurrent_reacquire() {
        // Deterministic ABA without lock timing games:
        // 1) acquire gen1, drop it (refunds).
        // 2) re-acquire gen2 under the same key.
        // 3) call release with the stale gen1 token — must be a no-op.
        // Without generation tokens step 3 would remove gen2's reservation.
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(16 * GIB, 16 * GIB);
        let k = key(0xABA);
        let (h1, _) = broker
            .acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None)
            .expect("gen1");
        let gen1 = h1.generation();
        drop(h1);

        let (h2, charged2) = broker
            .acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None)
            .expect("gen2");
        assert_eq!(charged2, 3 * GIB);
        let gen2 = h2.generation();
        assert_ne!(gen1, gen2);

        broker.release_pack_weight_residency(&k, gen1);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            3 * GIB,
            "stale gen1 release must not refund gen2"
        );
        assert_eq!(broker.pack_weight_residency_generation(&k), Some(gen2));

        drop(h2);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    #[test]
    fn concurrent_share_and_reacquire_keeps_ledger_consistent() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            ..DeviceMemoryPolicy::default()
        }));
        let snap = snapshot(32 * GIB, 32 * GIB);
        let k = key(0xC0FFEE);
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let broker = Arc::clone(&broker);
            let barrier = Arc::clone(&barrier);
            let k = k.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    let (h, _) = broker
                        .acquire_pack_weight_residency(k.clone(), 2 * GIB, snap, None)
                        .expect("acquire");
                    drop(h);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker");
        }
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0,
            "all handles dropped"
        );
        assert_eq!(broker.pack_weight_residency_live_count(), 0);
    }

    #[test]
    fn dead_entry_with_live_reservation_is_refunded_before_first_reacquire() {
        // Deterministic proof of refund-before-reserve (no retry loop):
        //
        // Budget fits exactly one 3 GiB residency. Inject a dead-weak table entry
        // that still holds a live 3 GiB reservation -- the race window after the
        // last owner's Arc died but before Drop refunded (or after a stale Drop
        // left the entry). A single `acquire` call must:
        //   1. observe the dead weak,
        //   2. refund the stale reservation under the table lock,
        //   3. reserve+commit the new generation,
        //   4. return Ok on the first try.
        // Without step 2 the same call fails closed with DeviceBudgetExceeded
        // because the ledger still shows 3 GiB committed against a 3 GiB ceiling.
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            maximum_owned_basis_points: 10_000,
        }));
        let snap = snapshot(3 * GIB, 3 * GIB);
        let k = key(0xDEAD_E077);

        broker
            .inject_dead_pack_weight_residency_with_live_reservation_for_test(
                k.clone(),
                3 * GIB,
                snap,
            )
            .expect("inject dead entry with live reservation");
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            3 * GIB,
            "precondition: stale reservation still charges the policy ledger"
        );
        assert_eq!(
            broker.pack_weight_residency_live_count(),
            0,
            "precondition: injected entry has no live owner"
        );

        // One call. No retry. Old code (reserve without refunding dead entry)
        // fails here with DeviceBudgetExceeded.
        let (h2, charged) = broker
            .acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None)
            .expect("first reacquire must succeed without retry when dead entry is refunded first");
        assert_eq!(charged, 3 * GIB, "new generation takes the full charge");
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            3 * GIB,
            "policy must show exactly one residency, not double-count"
        );
        assert_eq!(broker.pack_weight_residency_live_count(), 1);
        drop(h2);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
        assert_eq!(broker.pack_weight_residency_live_count(), 0);
    }

    #[test]
    fn concurrent_last_drop_and_reacquire_never_exceeds_one_mapping_charge() {
        // Stress companion to the deterministic inject test: drop the last owner
        // while another thread reacquires. Policy committed bytes must never
        // exceed one mapping; final state is exactly one live charge.
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            maximum_owned_basis_points: 10_000,
        }));
        let snap = snapshot(3 * GIB, 3 * GIB);
        let k = key(0xC0_C12E);
        let (h1, charged1) = broker
            .acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None)
            .expect("initial owner");
        assert_eq!(charged1, 3 * GIB);

        // Dual barrier: both threads arm, then drop and reacquire overlap.
        let armed = Arc::new(Barrier::new(2));
        let dropper = {
            let armed = Arc::clone(&armed);
            thread::spawn(move || {
                armed.wait();
                drop(h1);
            })
        };
        let reacquirer = {
            let broker = Arc::clone(&broker);
            let armed = Arc::clone(&armed);
            let k = k.clone();
            thread::spawn(move || {
                armed.wait();
                // First successful acquire wins; DeviceBudgetExceeded is only
                // tolerated while the dropper has not yet refunded. The
                // deterministic inject test above proves the no-retry path; this
                // stress only checks the ledger never overshoots one charge.
                for _ in 0..256 {
                    match broker.acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None) {
                        Ok((h, charged)) => {
                            assert!(
                                charged == 0 || charged == 3 * GIB,
                                "incremental charge must be share-or-full, got {charged}"
                            );
                            let committed =
                                broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes;
                            assert!(
                                committed <= 3 * GIB,
                                "policy must never double-count one mapping, committed={committed}"
                            );
                            return Ok(h);
                        }
                        Err(MemoryPlanningError::DeviceBudgetExceeded { .. }) => {
                            thread::yield_now();
                        }
                        Err(other) => return Err(other),
                    }
                }
                Err(MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: MemoryDomainKey::SystemMemory,
                })
            })
        };

        dropper.join().expect("dropper");
        let h2 = reacquirer
            .join()
            .expect("reacquirer thread")
            .expect("reacquire must eventually succeed under overlap");
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            3 * GIB
        );
        drop(h2);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }
}
