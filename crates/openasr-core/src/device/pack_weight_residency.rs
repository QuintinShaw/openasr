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
//! - first live owner of a mapping reserves and commits the quoted byte size;
//! - further owners of the *same* mapping share that charge (zero incremental);
//! - distinct mappings add;
//! - the last owner drop refunds the committed bytes.
//!
//! Callers still perform the native host-import; they must set
//! `currently_allocated_bytes = requested_bytes` on the HOST_IMPORT quote so
//! the backend reports a reuse (zero incremental) and this lease remains the
//! sole SystemMemory charge for the mapping.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
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
    /// Committed broker batch while any handle is live. Taken on last drop.
    reservation: Option<DeviceMemoryReservationBatch>,
    /// Strong count is tracked via [`Arc`] handles; this Weak lets a new
    /// acquirer join an existing entry without racing a concurrent last drop.
    live: Weak<PackWeightResidencyInner>,
}

#[derive(Debug)]
struct PackWeightResidencyInner {
    broker: Arc<DeviceMemoryBrokerSet>,
    key: PackWeightResidencyKey,
}

/// One owner of a shared pack-weight residency charge. Clone freely; the
/// underlying SystemMemory commitment is released when the last clone drops.
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
}

impl Drop for PackWeightResidencyInner {
    fn drop(&mut self) {
        self.broker.release_pack_weight_residency(&self.key);
    }
}

impl DeviceMemoryBrokerSet {
    /// Acquire shared residency for one open pack mapping.
    ///
    /// Returns `(handle, incremental_bytes_charged_now)`. `incremental` is the
    /// full `bytes` on the first live owner and `0` when joining an existing
    /// charge for the same mapping identity.
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

        // First live owner: reserve + commit under the ordinary domain ledger.
        let mut request = super::execution_memory::DomainReservationRequest {
            domain: key.domain.clone(),
            snapshot,
            peak_bytes: bytes,
            retained_bytes: bytes,
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

        let inner = Arc::new(PackWeightResidencyInner {
            broker: Arc::clone(self),
            key: key.clone(),
        });
        table.insert(
            key,
            PackWeightResidencyEntry {
                charged_bytes: bytes,
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

    fn release_pack_weight_residency(&self, key: &PackWeightResidencyKey) {
        let mut table = self
            .pack_weight_residencies
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
}

pub(crate) fn empty_pack_weight_residency_table()
-> Mutex<HashMap<PackWeightResidencyKey, PackWeightResidencyEntry>> {
    Mutex::new(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_memory::{
        DeviceMemoryPolicy, DeviceMemorySnapshot, MemoryDomainKey, MemoryObservationConfidence,
    };

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
        // total 6 GiB, first takes 4, second wants 4 -> fail
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
}
