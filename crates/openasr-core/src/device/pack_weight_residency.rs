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

/// Identity of one already-open pack mapping in one physical memory domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PackWeightResidencyKey {
    pub domain: MemoryDomainKey,
    /// [`std::sync::Arc::as_ptr`] of the process-local `Arc<Mmap>` (stable for
    /// clones of the same open generation; distinct for separately admitted
    /// files even at the same path).
    pub mapping_identity: usize,
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
    #[cfg(test)]
    pub(crate) fn charged_bytes(&self) -> u64 {
        self.charged_bytes
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.inner.generation
    }
}

impl Drop for PackWeightResidencyInner {
    fn drop(&mut self) {
        self.broker
            .release_pack_weight_residency(&self.key, self.generation);
    }
}

impl DeviceMemoryBrokerSet {
    /// Acquire shared residency for one open pack mapping.
    ///
    /// Returns `(handle, incremental_bytes_charged_now)`. `incremental` is the
    /// full `bytes` on the first live owner and `0` when joining an existing
    /// charge for the same mapping identity.
    ///
    /// `snapshot` must reflect **live** host free/total. Already-open file-backed
    /// residency does not need `free >= bytes` (observed peak is 0); the policy
    /// ledger still charges `bytes` so concurrent distinct packs fail closed.
    pub(crate) fn acquire_pack_weight_residency(
        self: &Arc<Self>,
        key: PackWeightResidencyKey,
        bytes: u64,
        snapshot: DeviceMemorySnapshot,
        cohort_id: Option<MemoryReservationCohortId>,
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
                key.domain, key.mapping_identity
            ),
            cohort_id: None,
        };
        request.cohort_id = cohort_id;
        let mut batch = self.try_reserve_batch(vec![request])?;
        batch.commit_quoted()?;

        let generation = self
            .next_pack_weight_residency_generation
            .fetch_add(1, Ordering::Relaxed);
        let inner = Arc::new(PackWeightResidencyInner {
            broker: Arc::clone(self),
            key: key.clone(),
            generation,
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
            mapping_identity: id,
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
    fn last_drop_reacquire_overlap_does_not_double_charge_policy() {
        // Controllable barrier race:
        // - total budget fits exactly one 4 GiB residency (not two).
        // - Thread A holds the last live handle; Thread B waits to reacquire.
        // - A drops while B acquires. Without dead-entry refund-before-reserve,
        //   B can see a dead weak, reserve a second 4 GiB while A's Drop has not
        //   refunded yet, and fail-closed even though only one mapping exists.
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            minimum_headroom_bytes: 0,
            maximum_owned_basis_points: 10_000,
        }));
        // Exactly one 3 GiB residency fits; two would exceed. Use 3 not 4 so the
        // arithmetic stays obvious under a full-ownership policy ceiling.
        let snap = snapshot(3 * GIB, 3 * GIB);
        let k = key(0xDEAD_E077);
        let (h1, charged1) = broker
            .acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None)
            .expect("initial owner");
        assert_eq!(charged1, 3 * GIB);

        let start = Arc::new(Barrier::new(2));
        let dropper = {
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                drop(h1);
            })
        };
        let reacquirer = {
            let broker = Arc::clone(&broker);
            let start = Arc::clone(&start);
            let k = k.clone();
            thread::spawn(move || {
                start.wait();
                // Spin briefly so drop and reacquire truly overlap under load.
                for _ in 0..64 {
                    match broker.acquire_pack_weight_residency(k.clone(), 3 * GIB, snap, None) {
                        Ok((h, charged)) => {
                            assert!(
                                charged == 0 || charged == 3 * GIB,
                                "incremental charge must be share-or-full, got {charged}"
                            );
                            assert!(
                                broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes
                                    <= 3 * GIB,
                                "policy must never double-count one mapping"
                            );
                            return Ok(h);
                        }
                        Err(MemoryPlanningError::DeviceBudgetExceeded { .. }) => {
                            // Transient only if still racing; retry while drop completes.
                            thread::yield_now();
                        }
                        Err(other) => return Err(other),
                    }
                }
                broker
                    .acquire_pack_weight_residency(k, 3 * GIB, snap, None)
                    .map(|(h, _)| h)
            })
        };

        dropper.join().expect("dropper");
        let h2 = reacquirer
            .join()
            .expect("reacquirer thread")
            .expect("reacquire must succeed without false budget exceed");
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
