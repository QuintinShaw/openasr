//! Physical-memory planning and process-wide byte reservations.
//!
//! Decoder topology answers *what state is semantically required*.  This
//! module answers a different question: whether one concrete execution
//! candidate can commit the backend buffers for that state, its weights, and
//! the largest simultaneously-live workspace without OpenASR racing itself.
//!
//! A reservation never promises that an open desktop GPU cannot still OOM:
//! another process may allocate after the memory snapshot, and backend-private
//! pools may only provide an upper bound.  Allocation failures therefore stay
//! typed and recoverable by the execution-policy layer.  What the broker does
//! guarantee is that two OpenASR sessions sharing a physical memory domain
//! cannot both pass admission against the same bytes.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use thiserror::Error;

use crate::models::native_execution_services::NativeExecutionScopeId;
use crate::models::runtime_receipts::{
    RuntimeOwnerDescriptor, RuntimeOwnerPlacement, RuntimeReceiptCollector, RuntimeReceiptMetric,
    RuntimeResourceDescriptor, RuntimeResourceGuard, RuntimeResourceState,
};

/// Physical budget identity. Multiple APIs exposing the same PCI device must
/// resolve to the same key before asking the broker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryDomainKey {
    DedicatedDevice {
        physical_device: PhysicalDeviceKey,
        heap_index: u32,
    },
    /// Ordinary host allocations and every integrated accelerator drawing
    /// from the same physical RAM. Keeping one key is essential: a CPU
    /// session and a Metal/UMA session must not each admit against the full
    /// system-memory budget independently.
    SystemMemory,
}

impl fmt::Display for MemoryDomainKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DedicatedDevice {
                physical_device,
                heap_index,
            } => write!(f, "device/{physical_device}/heap-{heap_index}"),
            Self::SystemMemory => f.write_str("system-memory"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalDeviceKey(String);

impl PhysicalDeviceKey {
    pub fn new(value: impl Into<String>) -> Result<Self, MemoryPlanningError> {
        let value = value.into().trim().to_ascii_lowercase();
        if value.is_empty() {
            return Err(MemoryPlanningError::EmptyPhysicalDeviceKey);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identity shared by every nested memory transaction in one execution
/// candidate attempt. It is not a budget namespace: all cohorts still charge
/// the same physical-domain ledger. The identity only proves that a nested
/// reservation belongs to the provisional transaction currently holding that
/// domain's exclusive reconciliation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MemoryReservationCohortId(u64);

impl MemoryReservationCohortId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }
}

impl fmt::Display for PhysicalDeviceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Quality of a live memory observation supplied by a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum MemoryObservationConfidence {
    /// Backend/driver reports current free and total bytes for the target heap.
    DeviceSnapshot,
    /// A working-set budget (for example Metal), not raw physical free pages.
    WorkingSetBudget,
    /// Only a total heap size is known; `free_bytes` is a heuristic.
    Heuristic,
    Unknown,
}

impl MemoryObservationConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeviceSnapshot => "device-snapshot",
            Self::WorkingSetBudget => "working-set-budget",
            Self::Heuristic => "heuristic",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemorySnapshot {
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub confidence: MemoryObservationConfidence,
}

impl DeviceMemorySnapshot {
    pub fn normalized(self) -> Result<Self, MemoryPlanningError> {
        if self.total_bytes == 0 {
            return Err(MemoryPlanningError::InvalidMemorySnapshot {
                free_bytes: self.free_bytes,
                total_bytes: self.total_bytes,
            });
        }
        Ok(Self {
            // Several backend APIs have historically underflowed their
            // working-set subtraction. Never let an impossible free > total
            // observation inflate admission.
            free_bytes: self.free_bytes.min(self.total_bytes),
            ..self
        })
    }
}

/// Whether a quote describes backend-requested bytes or a physical commitment
/// upper bound.  The distinction is part of the type so diagnostics and tests
/// cannot accidentally relabel requested Vulkan/CUDA bytes as exact VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum QuoteConfidence {
    ExactCommitted,
    CommittedUpperBound,
    /// The backend can price every engine-controlled allocation, but some
    /// backend/driver-private commitment is only an estimate. Admission may
    /// use the estimate transactionally, but the reservation cannot become a
    /// committed lease until live post-allocation statistics reconcile it.
    Provisional,
    Unknown,
}

impl QuoteConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactCommitted => "exact-committed",
            Self::CommittedUpperBound => "committed-upper-bound",
            Self::Provisional => "provisional",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationLifetime {
    BackendGlobal,
    PackShared,
    RunnerRetainedHighWater,
    SessionResident,
    PhaseTransient,
    StepTransient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutionPhase {
    ModelLoad = 0,
    Encoder = 1,
    Adaptor = 2,
    DecoderPrefill = 3,
    DecoderStep = 4,
    SpeakerAttribution = 5,
}

impl ExecutionPhase {
    const ALL: [Self; 6] = [
        Self::ModelLoad,
        Self::Encoder,
        Self::Adaptor,
        Self::DecoderPrefill,
        Self::DecoderStep,
        Self::SpeakerAttribution,
    ];

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Compact phase membership for one allocation. Persistent resources simply
/// include every phase in which they remain alive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseSet(u8);

impl PhaseSet {
    pub const ALL: Self = Self((1 << 6) - 1);

    pub const fn one(phase: ExecutionPhase) -> Self {
        Self(phase.bit())
    }

    pub const fn range(first: ExecutionPhase, last: ExecutionPhase) -> Self {
        let first = first as u8;
        let last = last as u8;
        if first > last {
            return Self(0);
        }
        let width = last - first + 1;
        Self((((1_u16 << width) - 1) << first) as u8)
    }

    pub const fn contains(self, phase: ExecutionPhase) -> bool {
        self.0 & phase.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryClaim {
    pub resource_id: String,
    pub domain: MemoryDomainKey,
    /// Logical payload OpenASR asks the backend to make addressable. This is a
    /// diagnostic quantity, not a physical-memory estimate: alignment,
    /// allocator blocks, imports, and cache reuse can all make it differ from
    /// the incremental commitment below.
    pub requested_bytes: u64,
    /// Maximum *additional* physical commitment while this resource is being
    /// established. Existing cached ownership is excluded and remains charged
    /// to its existing lease.
    pub incremental_peak_bytes: Option<u64>,
    /// Additional physical commitment retained by this resource's owner after
    /// its allocation phase completes. Transient workspaces use zero.
    pub incremental_retained_bytes: Option<u64>,
    pub confidence: QuoteConfidence,
    pub lifetime: AllocationLifetime,
    pub phases: PhaseSet,
}

impl MemoryClaim {
    fn validated_bytes(&self) -> Result<(u64, u64), MemoryPlanningError> {
        if self.resource_id.trim().is_empty() {
            return Err(MemoryPlanningError::EmptyResourceId);
        }
        if self.phases.is_empty() {
            return Err(MemoryPlanningError::EmptyPhaseSet {
                resource_id: self.resource_id.clone(),
            });
        }
        if self.confidence == QuoteConfidence::Unknown {
            return Err(MemoryPlanningError::CapacityUnproven {
                resource_id: self.resource_id.clone(),
            });
        }
        let peak = self.incremental_peak_bytes.ok_or_else(|| {
            MemoryPlanningError::InvalidCommitmentBound {
                resource_id: self.resource_id.clone(),
                incremental_peak_bytes: self.incremental_peak_bytes,
                incremental_retained_bytes: self.incremental_retained_bytes,
            }
        })?;
        let retained = self.incremental_retained_bytes.ok_or_else(|| {
            MemoryPlanningError::InvalidCommitmentBound {
                resource_id: self.resource_id.clone(),
                incremental_peak_bytes: self.incremental_peak_bytes,
                incremental_retained_bytes: self.incremental_retained_bytes,
            }
        })?;
        if retained > peak {
            return Err(MemoryPlanningError::InvalidCommitmentBound {
                resource_id: self.resource_id.clone(),
                incremental_peak_bytes: self.incremental_peak_bytes,
                incremental_retained_bytes: self.incremental_retained_bytes,
            });
        }
        Ok((peak, retained))
    }
}

/// One physical domain's phase-aware incremental requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainFootprint {
    pub domain: MemoryDomainKey,
    pub peak_bytes: u64,
    pub retained_bytes: u64,
    pub requires_reconciliation: bool,
    pub resource_ids: Vec<String>,
}

/// Phase-aware footprint. Non-overlapping encoder/decode workspaces are never
/// summed; retained resources appear in every phase in which their owner is
/// alive and therefore naturally contribute to the right peak.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocationFootprint {
    claims: Vec<MemoryClaim>,
}

impl AllocationFootprint {
    pub fn new(claims: Vec<MemoryClaim>) -> Self {
        Self { claims }
    }

    pub fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub fn domain_footprints(&self) -> Result<Vec<DomainFootprint>, MemoryPlanningError> {
        let mut by_domain: BTreeMap<MemoryDomainKey, Vec<&MemoryClaim>> = BTreeMap::new();
        for claim in &self.claims {
            // Validate even claims in a phase that never becomes the maximum;
            // malformed quotes must not hide behind a larger valid claim.
            claim.validated_bytes()?;
            by_domain
                .entry(claim.domain.clone())
                .or_default()
                .push(claim);
        }

        let mut footprints = Vec::with_capacity(by_domain.len());
        for (domain, claims) in by_domain {
            let mut peak = 0_u64;
            let mut retained = 0_u64;
            let mut requires_reconciliation = false;
            let mut resource_ids = Vec::with_capacity(claims.len());
            for claim in &claims {
                requires_reconciliation |= claim.confidence == QuoteConfidence::Provisional;
                resource_ids.push(claim.resource_id.clone());
            }
            resource_ids.sort();
            resource_ids.dedup();

            for phase in ExecutionPhase::ALL {
                let mut phase_peak = 0_u64;
                let mut phase_retained = 0_u64;
                for claim in claims
                    .iter()
                    .copied()
                    .filter(|claim| claim.phases.contains(phase))
                {
                    let (claim_peak, claim_retained) = claim.validated_bytes()?;
                    phase_peak = phase_peak.checked_add(claim_peak).ok_or(
                        MemoryPlanningError::ArithmeticOverflow {
                            operation: "phase footprint peak sum",
                        },
                    )?;
                    phase_retained = phase_retained.checked_add(claim_retained).ok_or(
                        MemoryPlanningError::ArithmeticOverflow {
                            operation: "phase footprint retained sum",
                        },
                    )?;
                }
                peak = peak.max(phase_peak);
                retained = retained.max(phase_retained);
            }
            if retained > peak {
                return Err(MemoryPlanningError::InvalidDomainFootprint {
                    domain,
                    peak_bytes: peak,
                    retained_bytes: retained,
                });
            }
            footprints.push(DomainFootprint {
                domain,
                peak_bytes: peak,
                retained_bytes: retained,
                requires_reconciliation,
                resource_ids,
            });
        }
        Ok(footprints)
    }

    pub fn peak_bytes(&self, domain: &MemoryDomainKey) -> Result<u64, MemoryPlanningError> {
        Ok(self
            .domain_footprints()?
            .into_iter()
            .find(|footprint| &footprint.domain == domain)
            .map_or(0, |footprint| footprint.peak_bytes))
    }

    pub fn retained_bytes(&self, domain: &MemoryDomainKey) -> Result<u64, MemoryPlanningError> {
        Ok(self
            .domain_footprints()?
            .into_iter()
            .find(|footprint| &footprint.domain == domain)
            .map_or(0, |footprint| footprint.retained_bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryPolicy {
    /// Fraction of the reported total the engine may ever own, in basis points.
    pub maximum_owned_basis_points: u16,
    /// Absolute driver/external-process reserve.
    pub minimum_headroom_bytes: u64,
}

impl Default for DeviceMemoryPolicy {
    fn default() -> Self {
        Self {
            maximum_owned_basis_points: 9_500,
            minimum_headroom_bytes: 256 * 1024 * 1024,
        }
    }
}

impl DeviceMemoryPolicy {
    fn limits(self, snapshot: DeviceMemorySnapshot) -> Result<(u64, u64), MemoryPlanningError> {
        if self.maximum_owned_basis_points == 0 || self.maximum_owned_basis_points > 10_000 {
            return Err(MemoryPlanningError::InvalidOwnedFraction {
                basis_points: self.maximum_owned_basis_points,
            });
        }
        let snapshot = snapshot.normalized()?;
        let policy_ceiling = u128::from(snapshot.total_bytes)
            .checked_mul(u128::from(self.maximum_owned_basis_points))
            .ok_or(MemoryPlanningError::ArithmeticOverflow {
                operation: "device policy ceiling",
            })?
            / 10_000;
        let policy_ceiling =
            u64::try_from(policy_ceiling).map_err(|_| MemoryPlanningError::ArithmeticOverflow {
                operation: "device policy ceiling conversion",
            })?;
        let observed_ceiling = snapshot
            .free_bytes
            .saturating_sub(self.minimum_headroom_bytes);
        Ok((policy_ceiling, observed_ceiling))
    }
}

fn merge_candidate_snapshots(
    left: DeviceMemorySnapshot,
    right: DeviceMemorySnapshot,
) -> DeviceMemorySnapshot {
    let confidence_rank = |confidence| match confidence {
        MemoryObservationConfidence::DeviceSnapshot => 3,
        MemoryObservationConfidence::WorkingSetBudget => 2,
        MemoryObservationConfidence::Heuristic => 1,
        MemoryObservationConfidence::Unknown => 0,
    };
    DeviceMemorySnapshot {
        free_bytes: left.free_bytes.min(right.free_bytes),
        total_bytes: left.total_bytes.min(right.total_bytes),
        confidence: if confidence_rank(left.confidence) <= confidence_rank(right.confidence) {
            left.confidence
        } else {
            right.confidence
        },
    }
}

#[derive(Debug, Default)]
struct DomainAccount {
    pending_bytes: u64,
    pending_bytes_by_cohort: HashMap<ReservationCohortKey, u64>,
    committed_bytes: u64,
    unreclaimable_bytes: u64,
    /// Number of child reservations from the one provisional candidate that
    /// still hold this domain's admission gate. While non-zero no unrelated
    /// candidate may enter the domain, even with a zero-byte request.
    exclusive_pending_children: u32,
    exclusive_pending_cohort: Option<ReservationCohortKey>,
    quarantined: bool,
    by_scope: HashMap<NativeExecutionScopeId, ScopedDomainAccount>,
}

#[derive(Debug, Default)]
struct ScopedDomainAccount {
    pending_bytes: u64,
    committed_bytes: u64,
    unreclaimable_bytes: u64,
    /// Diagnostic-only attribution. The aggregate fields above remain the
    /// scope-local accounting authority; admission never branches on this map.
    by_placement: HashMap<RuntimeOwnerPlacement, PlacementDomainAccount>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct PlacementDomainAccount {
    pending_bytes: u64,
    committed_bytes: u64,
    unreclaimable_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ReservationCohortKey {
    Explicit(MemoryReservationCohortId),
    Anonymous(u64),
}

/// One domain row submitted to the broker as part of an atomic candidate
/// admission. Callers obtain these rows by joining a backend quote's native
/// domain identifiers with a fresh backend memory observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainReservationRequest {
    pub domain: MemoryDomainKey,
    pub snapshot: DeviceMemorySnapshot,
    pub peak_bytes: u64,
    pub retained_bytes: u64,
    /// Bytes that must fit in **live** free/observed capacity for this row.
    ///
    /// `None` (default) means the observed check uses [`Self::peak_bytes`] —
    /// ordinary anonymous allocations. `Some(0)` is for already-open reclaimable
    /// file-backed residency: the policy ledger still charges `peak_bytes` so
    /// concurrent distinct packs fail closed, but live free is not required to
    /// cover the mapping size again (clean file pages are reclaimable and often
    /// still counted as free by the host).
    pub observed_peak_bytes: Option<u64>,
    pub requires_reconciliation: bool,
    pub resource_id: String,
    pub(crate) cohort_id: Option<MemoryReservationCohortId>,
}

impl DomainReservationRequest {
    pub fn from_footprint(footprint: DomainFootprint, snapshot: DeviceMemorySnapshot) -> Self {
        Self {
            domain: footprint.domain,
            snapshot,
            peak_bytes: footprint.peak_bytes,
            retained_bytes: footprint.retained_bytes,
            observed_peak_bytes: None,
            requires_reconciliation: footprint.requires_reconciliation,
            resource_id: footprint.resource_ids.join("+"),
            cohort_id: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_cohort_id(mut self, cohort_id: Option<MemoryReservationCohortId>) -> Self {
        self.cohort_id = cohort_id;
        self
    }
}

/// Live post-allocation evidence used to turn a provisional reservation into
/// an owner-bound committed lease. The allocation owner must remain alive
/// until this method either commits or the caller tears it down; otherwise the
/// snapshot and physical delta no longer describe the candidate being judged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainMemoryReconciliation {
    pub domain: MemoryDomainKey,
    pub actual_peak_bytes: u64,
    pub actual_retained_bytes: u64,
    pub snapshot_after: DeviceMemorySnapshot,
}

/// One process-wide ledger, internally partitioned by physical memory domain.
/// Clone/pass it as an `Arc`; do not instantiate one per request in a server.
#[derive(Debug)]
pub struct DeviceMemoryBrokerSet {
    policy: DeviceMemoryPolicy,
    accounts: Mutex<HashMap<MemoryDomainKey, DomainAccount>>,
    next_anonymous_cohort: AtomicU64,
    /// Shared FILE_BACKED pack-weight charges keyed by open mapping identity.
    /// See [`super::pack_weight_residency`].
    pub(crate) pack_weight_residencies: Mutex<
        HashMap<
            super::pack_weight_residency::PackWeightResidencyKey,
            super::pack_weight_residency::PackWeightResidencyEntry,
        >,
    >,
    /// Monotonic generation for pack-weight residency entries (ABA guard).
    pub(crate) next_pack_weight_residency_generation: AtomicU64,
}

impl DeviceMemoryBrokerSet {
    pub fn new(policy: DeviceMemoryPolicy) -> Self {
        Self {
            policy,
            accounts: Mutex::new(HashMap::new()),
            next_anonymous_cohort: AtomicU64::new(1),
            pack_weight_residencies:
                super::pack_weight_residency::empty_pack_weight_residency_table(),
            next_pack_weight_residency_generation:
                super::pack_weight_residency::new_pack_weight_residency_generation_counter(),
        }
    }

    /// Absolute bytes deliberately left outside OpenASR ownership in every
    /// physical domain. Native providers use this policy reserve for opaque
    /// command-buffer/driver commitments that cannot be attributed to an
    /// engine-visible allocation claim.
    pub fn minimum_headroom_bytes(&self) -> u64 {
        self.policy.minimum_headroom_bytes
    }

    /// Atomically reserves every physical domain used by one candidate.
    ///
    /// A discrete-GPU candidate often has both device-local and system-memory
    /// rows. Checking them one by one would permit another session to consume
    /// the second domain after the first had passed, creating a partial
    /// admission and a classic check-then-act race. This method validates all
    /// rows under one process-wide lock and mutates none unless all fit.
    pub fn try_reserve_batch(
        self: &Arc<Self>,
        requests: Vec<DomainReservationRequest>,
    ) -> Result<DeviceMemoryReservationBatch, MemoryPlanningError> {
        self.try_reserve_batch_for_scope_and_placement(
            requests,
            crate::models::native_execution_services::current_native_execution_scope_id(),
            RuntimeOwnerPlacement::Unknown,
        )
    }

    /// Scoped admission with a diagnostic ownership attribution captured
    /// before any allocation. The placement never changes physical capacity
    /// policy; it only makes later receipt reconciliation lane-exact.
    pub(crate) fn try_reserve_batch_for_scope_and_placement(
        self: &Arc<Self>,
        requests: Vec<DomainReservationRequest>,
        owner_scope_id: Option<NativeExecutionScopeId>,
        owner_placement: RuntimeOwnerPlacement,
    ) -> Result<DeviceMemoryReservationBatch, MemoryPlanningError> {
        self.try_reserve_partitioned_for_scope_and_placements(
            vec![requests],
            owner_scope_id,
            vec![owner_placement],
        )?
        .pop()
        .ok_or(MemoryPlanningError::ReservationLedgerCorrupted {
            domain: MemoryDomainKey::SystemMemory,
        })
    }

    /// Atomically admits one candidate while preserving separate native-owner
    /// leases. Each partition becomes one child batch, but capacity is checked
    /// against the sum of every partition under the same ledger lock.
    ///
    /// If any child in a physical domain is provisional, that domain must have
    /// no pre-existing pending allocation and is made candidate-exclusive until
    /// every child touching the domain commits, quarantines, or releases. This
    /// is the concurrency proof that permits a reconciled physical delta to be
    /// larger than a provider's non-upper-bound estimate.
    pub fn try_reserve_partitioned(
        self: &Arc<Self>,
        partitions: Vec<Vec<DomainReservationRequest>>,
    ) -> Result<Vec<DeviceMemoryReservationBatch>, MemoryPlanningError> {
        let placements = vec![RuntimeOwnerPlacement::Unknown; partitions.len()];
        self.try_reserve_partitioned_for_scope_and_placements(
            partitions,
            crate::models::native_execution_services::current_native_execution_scope_id(),
            placements,
        )
    }

    pub(crate) fn try_reserve_partitioned_for_scope_and_placements(
        self: &Arc<Self>,
        partitions: Vec<Vec<DomainReservationRequest>>,
        owner_scope_id: Option<NativeExecutionScopeId>,
        owner_placements: Vec<RuntimeOwnerPlacement>,
    ) -> Result<Vec<DeviceMemoryReservationBatch>, MemoryPlanningError> {
        if owner_placements.len() != partitions.len() {
            return Err(MemoryPlanningError::ReservationPlacementSetMismatch {
                expected: partitions.len(),
                actual: owner_placements.len(),
            });
        }
        if owner_scope_id.is_none()
            && owner_placements
                .iter()
                .any(|placement| !matches!(placement, RuntimeOwnerPlacement::Unknown))
        {
            return Err(MemoryPlanningError::ReservationPlacementWithoutScope);
        }

        #[derive(Clone)]
        struct Aggregate {
            snapshot: DeviceMemorySnapshot,
            peak_bytes: u64,
            /// Live free/observed capacity required for this aggregate. Defaults
            /// to [`Self::peak_bytes`] per row unless a request overrides with
            /// [`DomainReservationRequest::observed_peak_bytes`] (e.g. already-
            /// open reclaimable file-backed residency uses 0).
            observed_peak_bytes: u64,
            retained_bytes: u64,
            requires_reconciliation: bool,
            resource_ids: Vec<String>,
            child_count: u32,
        }

        let mut explicit_cohort = None;
        let mut saw_unscoped_request = false;
        for request in partitions.iter().flatten() {
            match request.cohort_id {
                Some(cohort) => match explicit_cohort {
                    Some(existing) if existing != cohort => {
                        return Err(MemoryPlanningError::MixedReservationCohorts);
                    }
                    None => explicit_cohort = Some(cohort),
                    Some(_) => {}
                },
                None => saw_unscoped_request = true,
            }
        }
        if explicit_cohort.is_some() && saw_unscoped_request {
            return Err(MemoryPlanningError::MixedReservationCohorts);
        }
        let cohort = explicit_cohort.map_or_else(
            || {
                ReservationCohortKey::Anonymous(
                    self.next_anonymous_cohort.fetch_add(1, Ordering::Relaxed),
                )
            },
            ReservationCohortKey::Explicit,
        );

        let mut aggregates = BTreeMap::<MemoryDomainKey, Aggregate>::new();
        for partition in &partitions {
            let mut seen_domains = HashSet::with_capacity(partition.len());
            for request in partition {
                if request.resource_id.trim().is_empty() {
                    return Err(MemoryPlanningError::EmptyResourceId);
                }
                if request.retained_bytes > request.peak_bytes {
                    return Err(MemoryPlanningError::InvalidDomainFootprint {
                        domain: request.domain.clone(),
                        peak_bytes: request.peak_bytes,
                        retained_bytes: request.retained_bytes,
                    });
                }
                let row_observed_peak = request.observed_peak_bytes.unwrap_or(request.peak_bytes);
                if row_observed_peak > request.peak_bytes {
                    return Err(MemoryPlanningError::InvalidDomainFootprint {
                        domain: request.domain.clone(),
                        peak_bytes: request.peak_bytes,
                        retained_bytes: row_observed_peak,
                    });
                }
                if !seen_domains.insert(request.domain.clone()) {
                    return Err(MemoryPlanningError::DuplicateMemoryDomain {
                        domain: request.domain.clone(),
                    });
                }
                if request.snapshot.confidence == MemoryObservationConfidence::Unknown {
                    return Err(MemoryPlanningError::MemoryObservationUnavailable {
                        domain: request.domain.clone(),
                        resource_id: request.resource_id.clone(),
                    });
                }
                let normalized = request.snapshot.normalized()?;
                if let Some(aggregate) = aggregates.get_mut(&request.domain) {
                    aggregate.snapshot = merge_candidate_snapshots(aggregate.snapshot, normalized);
                    aggregate.peak_bytes = aggregate
                        .peak_bytes
                        .checked_add(request.peak_bytes)
                        .ok_or(MemoryPlanningError::ArithmeticOverflow {
                            operation: "partitioned candidate peak sum",
                        })?;
                    aggregate.observed_peak_bytes = aggregate
                        .observed_peak_bytes
                        .checked_add(row_observed_peak)
                        .ok_or(MemoryPlanningError::ArithmeticOverflow {
                            operation: "partitioned candidate observed peak sum",
                        })?;
                    aggregate.retained_bytes = aggregate
                        .retained_bytes
                        .checked_add(request.retained_bytes)
                        .ok_or(MemoryPlanningError::ArithmeticOverflow {
                            operation: "partitioned candidate retained sum",
                        })?;
                    aggregate.requires_reconciliation |= request.requires_reconciliation;
                    aggregate.resource_ids.push(request.resource_id.clone());
                    aggregate.child_count = aggregate.child_count.checked_add(1).ok_or(
                        MemoryPlanningError::ArithmeticOverflow {
                            operation: "exclusive child count",
                        },
                    )?;
                } else {
                    aggregates.insert(
                        request.domain.clone(),
                        Aggregate {
                            snapshot: normalized,
                            peak_bytes: request.peak_bytes,
                            observed_peak_bytes: row_observed_peak,
                            retained_bytes: request.retained_bytes,
                            requires_reconciliation: request.requires_reconciliation,
                            resource_ids: vec![request.resource_id.clone()],
                            child_count: 1,
                        },
                    );
                }
            }
        }

        // Keep diagnostic placement sums separate from physical-domain
        // aggregates. The latter remain the only values used for admission.
        let mut placement_peaks = HashMap::<(MemoryDomainKey, RuntimeOwnerPlacement), u64>::new();
        for (partition, placement) in partitions.iter().zip(owner_placements.iter().copied()) {
            for request in partition {
                let slot = placement_peaks
                    .entry((request.domain.clone(), placement))
                    .or_default();
                *slot = slot.checked_add(request.peak_bytes).ok_or(
                    MemoryPlanningError::ArithmeticOverflow {
                        operation: "scoped placement pending reservation sum",
                    },
                )?;
            }
        }

        let mut accounts = self.lock_accounts();
        let empty_account = DomainAccount::default();
        // Read-only validation first: no account is mutated unless the complete
        // multi-owner candidate fits every physical domain.
        for (domain, aggregate) in &aggregates {
            let (policy_ceiling, observed_ceiling) = self.policy.limits(aggregate.snapshot)?;
            let account = accounts.get(domain).unwrap_or(&empty_account);
            if account.quarantined {
                return Err(MemoryPlanningError::DeviceQuarantined {
                    domain: domain.clone(),
                });
            }
            if !domain_account_is_consistent(account) {
                accounts.entry(domain.clone()).or_default().quarantined = true;
                return Err(MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: domain.clone(),
                });
            }
            let same_cohort_pending = account
                .pending_bytes_by_cohort
                .get(&cohort)
                .copied()
                .unwrap_or(0);
            let other_cohort_pending = account
                .pending_bytes
                .checked_sub(same_cohort_pending)
                .ok_or_else(|| MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: domain.clone(),
                })?;
            if account
                .exclusive_pending_cohort
                .is_some_and(|exclusive| exclusive != cohort)
                || (aggregate.requires_reconciliation && other_cohort_pending != 0)
            {
                return Err(MemoryPlanningError::DeviceDomainBusy {
                    domain: domain.clone(),
                    resource_id: aggregate.resource_ids.join("+"),
                    pending_bytes: account.pending_bytes,
                    exclusive_pending_children: account.exclusive_pending_children,
                });
            }
            let policy_remaining = policy_ceiling.saturating_sub(
                account
                    .committed_bytes
                    .saturating_add(account.pending_bytes)
                    .saturating_add(account.unreclaimable_bytes),
            );
            // Pending reservations are not necessarily reflected in the
            // driver's free snapshot yet. Committed allocations normally are,
            // so subtracting committed bytes here would count them twice.
            //
            // Policy check uses peak_bytes (full ownership charge). Observed
            // check uses observed_peak_bytes so reclaimable already-open
            // file-backed residency can charge policy without requiring live
            // free == pack size a second time.
            let policy_ok = aggregate.peak_bytes <= policy_remaining;
            let observed_remaining = observed_ceiling.saturating_sub(account.pending_bytes);
            let observed_ok = aggregate.observed_peak_bytes <= observed_remaining;
            if !policy_ok || !observed_ok {
                let available_bytes = if !policy_ok {
                    policy_remaining
                } else {
                    observed_remaining
                };
                return Err(MemoryPlanningError::DeviceBudgetExceeded {
                    domain: domain.clone(),
                    resource_id: aggregate.resource_ids.join("+"),
                    requested_bytes: aggregate.peak_bytes,
                    pending_bytes: account.pending_bytes,
                    committed_bytes: account.committed_bytes,
                    unreclaimable_bytes: account.unreclaimable_bytes,
                    policy_ceiling,
                    observed_ceiling,
                    available_bytes,
                });
            }
            account
                .pending_bytes
                .checked_add(aggregate.peak_bytes)
                .ok_or(MemoryPlanningError::ArithmeticOverflow {
                    operation: "pending reservation sum",
                })?;
            if let Some(owner_scope_id) = owner_scope_id {
                let scoped = account.by_scope.get(&owner_scope_id);
                scoped
                    .map(|account| account.pending_bytes)
                    .unwrap_or(0)
                    .checked_add(aggregate.peak_bytes)
                    .ok_or(MemoryPlanningError::ArithmeticOverflow {
                        operation: "scoped pending reservation sum",
                    })?;
                for ((placement_domain, placement), bytes) in &placement_peaks {
                    if placement_domain != domain {
                        continue;
                    }
                    scoped
                        .and_then(|account| account.by_placement.get(placement))
                        .map(|account| account.pending_bytes)
                        .unwrap_or(0)
                        .checked_add(*bytes)
                        .ok_or(MemoryPlanningError::ArithmeticOverflow {
                            operation: "scoped placement pending reservation sum",
                        })?;
                }
            }
            if aggregate.requires_reconciliation {
                account
                    .exclusive_pending_children
                    .checked_add(aggregate.child_count)
                    .ok_or(MemoryPlanningError::ArithmeticOverflow {
                        operation: "exclusive pending child sum",
                    })?;
            }
        }

        for (domain, aggregate) in &aggregates {
            let account = accounts.entry(domain.clone()).or_default();
            account.pending_bytes += aggregate.peak_bytes;
            if let Some(owner_scope_id) = owner_scope_id {
                account
                    .by_scope
                    .entry(owner_scope_id)
                    .or_default()
                    .pending_bytes += aggregate.peak_bytes;
            }
            *account.pending_bytes_by_cohort.entry(cohort).or_default() += aggregate.peak_bytes;
            if aggregate.requires_reconciliation {
                debug_assert!(
                    account
                        .exclusive_pending_cohort
                        .is_none_or(|exclusive| exclusive == cohort)
                );
                account.exclusive_pending_cohort = Some(cohort);
                account.exclusive_pending_children += aggregate.child_count;
            }
        }

        if let Some(owner_scope_id) = owner_scope_id {
            for ((domain, placement), bytes) in &placement_peaks {
                let scoped = accounts
                    .entry(domain.clone())
                    .or_default()
                    .by_scope
                    .entry(owner_scope_id)
                    .or_default();
                scoped
                    .by_placement
                    .entry(*placement)
                    .or_default()
                    .pending_bytes += *bytes;
            }
        }

        let mut batches = Vec::with_capacity(partitions.len());
        for (partition, owner_placement) in partitions.into_iter().zip(owner_placements) {
            let mut entries = Vec::with_capacity(partition.len());
            for request in partition {
                let holds_exclusive_gate = aggregates
                    .get(&request.domain)
                    .expect("partition domains were aggregated above")
                    .requires_reconciliation;
                entries.push(ReservationEntry {
                    domain: request.domain,
                    resource_id: request.resource_id,
                    reserved_peak_bytes: request.peak_bytes,
                    quoted_retained_bytes: request.retained_bytes,
                    committed_bytes: 0,
                    requires_reconciliation: request.requires_reconciliation,
                    holds_exclusive_gate,
                    cohort,
                    owner_scope_id,
                    owner_placement,
                    quarantine_bytes: request.peak_bytes,
                    receipt_resource: None,
                    receipt_descriptor: None,
                });
            }
            let state = if entries.is_empty() {
                ReservationState::Released
            } else {
                ReservationState::Pending
            };
            batches.push(DeviceMemoryReservationBatch {
                broker: Arc::clone(self),
                entries,
                state,
                receipt_owner: None,
            });
        }
        drop(accounts);
        Ok(batches)
    }

    pub fn usage(&self, domain: &MemoryDomainKey) -> DeviceMemoryUsage {
        let accounts = self.lock_accounts();
        let Some(account) = accounts.get(domain) else {
            return DeviceMemoryUsage::default();
        };
        DeviceMemoryUsage {
            pending_bytes: account.pending_bytes,
            committed_bytes: account.committed_bytes,
            unreclaimable_bytes: account.unreclaimable_bytes,
            exclusive_pending: account.exclusive_pending_children != 0,
            quarantined: account.quarantined,
        }
    }

    /// Diagnostic snapshot of every physical-domain ledger row. Receipts
    /// compare against this map; admission never reads it.
    #[allow(dead_code)]
    pub(crate) fn ledger_snapshot(&self) -> BTreeMap<MemoryDomainKey, DeviceMemoryUsage> {
        self.lock_accounts()
            .iter()
            .map(|(domain, account)| {
                (
                    domain.clone(),
                    DeviceMemoryUsage {
                        pending_bytes: account.pending_bytes,
                        committed_bytes: account.committed_bytes,
                        unreclaimable_bytes: account.unreclaimable_bytes,
                        exclusive_pending: account.exclusive_pending_children != 0,
                        quarantined: account.quarantined,
                    },
                )
            })
            .collect()
    }

    /// Diagnostic rows for one service root, split by the placement captured
    /// before admission. Receipt attestation uses this projection so equal
    /// byte totals in two execution lanes cannot cancel each other out.
    pub(crate) fn ledger_snapshot_for_scope_by_placement(
        &self,
        scope_id: NativeExecutionScopeId,
    ) -> HashMap<(MemoryDomainKey, RuntimeOwnerPlacement), DeviceMemoryUsage> {
        let accounts = self.lock_accounts();
        let mut rows = HashMap::new();
        for (domain, account) in accounts.iter() {
            let Some(scoped) = account.by_scope.get(&scope_id) else {
                continue;
            };
            for (placement, attributed) in &scoped.by_placement {
                rows.insert(
                    (domain.clone(), *placement),
                    DeviceMemoryUsage {
                        pending_bytes: attributed.pending_bytes,
                        committed_bytes: attributed.committed_bytes,
                        unreclaimable_bytes: attributed.unreclaimable_bytes,
                        exclusive_pending: account.exclusive_pending_children != 0,
                        quarantined: account.quarantined,
                    },
                );
            }
        }
        rows
    }

    fn lock_accounts(&self) -> MutexGuard<'_, HashMap<MemoryDomainKey, DomainAccount>> {
        self.accounts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceMemoryUsage {
    pub pending_bytes: u64,
    pub committed_bytes: u64,
    pub unreclaimable_bytes: u64,
    pub exclusive_pending: bool,
    pub quarantined: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationState {
    Pending,
    Committed,
    Quarantined,
    Released,
}

/// RAII ownership of an admitted allocation. Commit only after the native
/// allocation succeeds; keep the committed value inside the actual native
/// buffer/runtime owner so logical cache eviction cannot refund bytes before
/// the buffer's `Drop` really runs.
#[derive(Debug)]
struct ReservationEntry {
    domain: MemoryDomainKey,
    resource_id: String,
    reserved_peak_bytes: u64,
    quoted_retained_bytes: u64,
    committed_bytes: u64,
    requires_reconciliation: bool,
    holds_exclusive_gate: bool,
    cohort: ReservationCohortKey,
    owner_scope_id: Option<NativeExecutionScopeId>,
    /// Diagnostic attribution captured at admission. It never affects the
    /// physical-domain capacity decision.
    owner_placement: RuntimeOwnerPlacement,
    /// Conservative charge if native state becomes unreclaimable before the
    /// transaction can commit. Reconciliation evidence may raise this above
    /// the provider's provisional estimate.
    quarantine_bytes: u64,
    /// Diagnostic-only native ownership context. It mirrors this entry's
    /// lifecycle but never participates in admission or accounting.
    receipt_resource: Option<RuntimeResourceGuard>,
    receipt_descriptor: Option<RuntimeResourceDescriptor>,
}

/// RAII ownership for all physical domains retained by one concrete runtime
/// owner (weight cache, session arena, or runner high-water allocation).
#[derive(Debug)]
pub struct DeviceMemoryReservationBatch {
    broker: Arc<DeviceMemoryBrokerSet>,
    entries: Vec<ReservationEntry>,
    state: ReservationState,
    /// Diagnostic-only owner context. Declared after entries so resource guards
    /// drop before their owner guard and cannot leave dangling receipt rows.
    receipt_owner: Option<crate::models::runtime_receipts::RuntimeOwnerGuard>,
}

impl DeviceMemoryReservationBatch {
    /// Attaches bounded receipt evidence to an already-admitted batch. This is
    /// deliberately post-admission and one-way: receipt allocation cannot
    /// reject, resize, or otherwise alter the broker ledger.
    pub(crate) fn attach_receipt(
        &mut self,
        collector: RuntimeReceiptCollector,
        mut owner_descriptor: RuntimeOwnerDescriptor,
        resources: Vec<(MemoryDomainKey, RuntimeResourceDescriptor)>,
    ) {
        if !collector.is_available() || self.entries.is_empty() {
            return;
        }
        let owner_placement = self.entries[0].owner_placement;
        if self
            .entries
            .iter()
            .any(|entry| entry.owner_placement != owner_placement)
        {
            return;
        }
        // The pre-admission attribution is authoritative. Caller-built
        // receipt metadata may add labels/evidence, but cannot rebind a lease.
        owner_descriptor.placement = owner_placement;
        let mut resources = resources.into_iter().collect::<HashMap<_, _>>();
        let owner = collector.start_owner(
            owner_descriptor,
            crate::models::native_execution_services::current_execution_cache_attempt_id(),
        );
        let Some(owner_id) = owner.owner_id() else {
            return;
        };
        for entry in &mut self.entries {
            let Some(mut descriptor) = resources.remove(&entry.domain) else {
                continue;
            };
            descriptor.placement = entry.owner_placement;
            entry.receipt_resource = collector.acquire_resource(owner_id, descriptor.clone());
            entry.receipt_descriptor = entry.receipt_resource.as_ref().map(|_| descriptor);
        }
        self.receipt_owner = Some(owner);
    }

    pub(crate) fn record_receipt_reuse(&self) {
        if let Some(owner) = self.receipt_owner.as_ref() {
            owner.record_reuse(
                crate::models::native_execution_services::current_execution_cache_attempt_id(),
            );
        }
    }

    pub(crate) fn update_receipt_descriptors(
        &mut self,
        resources: Vec<(MemoryDomainKey, RuntimeResourceDescriptor)>,
    ) {
        let mut resources = resources.into_iter().collect::<HashMap<_, _>>();
        for entry in &mut self.entries {
            let (Some(resource), Some(mut descriptor)) = (
                entry.receipt_resource.as_ref(),
                resources.remove(&entry.domain),
            ) else {
                continue;
            };
            descriptor.placement = entry.owner_placement;
            resource.update_descriptor(descriptor.clone());
            entry.receipt_descriptor = Some(descriptor);
        }
    }
    fn refresh_receipt_broker_projection(&self) {
        for entry in &self.entries {
            let Some(resource) = entry.receipt_resource.as_ref() else {
                continue;
            };
            let Some(mut descriptor) = entry.receipt_descriptor.clone() else {
                continue;
            };
            let Some(native) = descriptor.native.as_mut() else {
                continue;
            };
            // Native evidence must describe the same scope + placement as the
            // attached resource. A process-wide row can be numerically valid
            // while attributing another lane's bytes to this owner.
            let usage = self.domain_usage(&entry.domain);
            native.broker_pending_bytes = RuntimeReceiptMetric::Known(usage.pending_bytes);
            native.broker_committed_bytes = RuntimeReceiptMetric::Known(usage.committed_bytes);
            native.broker_unreclaimable_bytes =
                RuntimeReceiptMetric::Known(usage.unreclaimable_bytes);
            resource.update_descriptor(descriptor.clone());
        }
    }

    pub(crate) fn domain_usage(&self, domain: &MemoryDomainKey) -> DeviceMemoryUsage {
        let Some(entry) = self.entries.iter().find(|entry| &entry.domain == domain) else {
            return DeviceMemoryUsage::default();
        };
        let Some(scope_id) = entry.owner_scope_id else {
            return self.broker.usage(domain);
        };
        let accounts = self.broker.lock_accounts();
        let Some(account) = accounts.get(domain) else {
            return DeviceMemoryUsage::default();
        };
        let Some(attributed) = account
            .by_scope
            .get(&scope_id)
            .and_then(|scope| scope.by_placement.get(&entry.owner_placement))
        else {
            return DeviceMemoryUsage {
                exclusive_pending: account.exclusive_pending_children != 0,
                quarantined: account.quarantined,
                ..DeviceMemoryUsage::default()
            };
        };
        DeviceMemoryUsage {
            pending_bytes: attributed.pending_bytes,
            committed_bytes: attributed.committed_bytes,
            unreclaimable_bytes: attributed.unreclaimable_bytes,
            exclusive_pending: account.exclusive_pending_children != 0,
            quarantined: account.quarantined,
        }
    }

    fn transition_receipt_state(&self, next_state: RuntimeResourceState) {
        for entry in &self.entries {
            if let Some(resource) = entry.receipt_resource.as_ref() {
                resource.set_state(next_state);
            }
        }
    }

    /// Remove receipt rows in resource-before-owner order once the broker has
    /// refunded the physical charge. No `Released` resource remains in the
    /// live table: the bounded event history records the release, while live
    /// reconciliation sees either the short retryable transition or no row.
    fn release_receipts(&mut self) {
        for entry in &mut self.entries {
            entry.receipt_resource.take();
            entry.receipt_descriptor = None;
        }
        self.receipt_owner.take();
    }

    /// A quarantined broker charge is intentionally permanent for this
    /// service root. Preserve its live diagnostic rows without leaking guard
    /// objects or making receipt lifetime influence admission.
    fn persist_quarantined_receipts(&mut self) {
        for entry in &mut self.entries {
            if let Some(resource) = entry.receipt_resource.take() {
                resource.persist_for_quarantine();
            }
        }
        if let Some(owner) = self.receipt_owner.take() {
            owner.persist_for_quarantine();
        }
    }

    fn prepare_receipt_descriptor(
        entry: &mut ReservationEntry,
        peak_bytes: u64,
        retained_bytes: u64,
    ) {
        if let Some(mut descriptor) = entry.receipt_descriptor.clone() {
            descriptor.peak = RuntimeReceiptMetric::Known(match descriptor.peak {
                RuntimeReceiptMetric::Known(previous) => previous.max(peak_bytes),
                RuntimeReceiptMetric::Unavailable | RuntimeReceiptMetric::Unknown => peak_bytes,
            });
            descriptor.retained = RuntimeReceiptMetric::Known(match descriptor.retained {
                RuntimeReceiptMetric::Known(previous) => previous.max(retained_bytes),
                RuntimeReceiptMetric::Unavailable | RuntimeReceiptMetric::Unknown => retained_bytes,
            });
            entry.receipt_descriptor = Some(descriptor);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_pending(&self) -> bool {
        self.state == ReservationState::Pending
    }

    pub fn requires_reconciliation(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.requires_reconciliation)
    }

    /// Commits proven upper-bound quotes. Provisional quotes must use
    /// [`Self::reconcile_and_commit`] so requested bytes are never relabelled
    /// as physical commitment merely because allocation happened to succeed.
    pub fn commit_quoted(&mut self) -> Result<(), MemoryPlanningError> {
        self.require_pending()?;
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.requires_reconciliation)
        {
            return Err(MemoryPlanningError::ReconciliationRequired {
                domain: entry.domain.clone(),
                resource_id: entry.resource_id.clone(),
            });
        }
        let committed: Vec<(MemoryDomainKey, u64)> = self
            .entries
            .iter()
            .map(|entry| (entry.domain.clone(), entry.quoted_retained_bytes))
            .collect();
        self.commit_entries(&committed)
    }

    /// Reconciles a provisional quote against live post-allocation physical
    /// statistics, then atomically commits every domain. On error the batch
    /// intentionally remains pending: the caller must destroy the candidate's
    /// native owner before dropping this reservation and trying fallback.
    pub fn reconcile_and_commit(
        &mut self,
        reconciliations: &[DomainMemoryReconciliation],
    ) -> Result<(), MemoryPlanningError> {
        self.require_pending()?;
        if reconciliations.len() != self.entries.len() {
            return Err(MemoryPlanningError::ReconciliationSetMismatch {
                expected: self.entries.len(),
                actual: reconciliations.len(),
            });
        }

        let mut by_domain = HashMap::with_capacity(reconciliations.len());
        for reconciliation in reconciliations {
            if by_domain
                .insert(reconciliation.domain.clone(), reconciliation)
                .is_some()
            {
                return Err(MemoryPlanningError::DuplicateMemoryDomain {
                    domain: reconciliation.domain.clone(),
                });
            }
        }

        let mut accounts = self.broker.lock_accounts();
        let mut committed = Vec::with_capacity(self.entries.len());
        for entry in &mut self.entries {
            let reconciliation = by_domain.get(&entry.domain).ok_or_else(|| {
                MemoryPlanningError::MissingDomainReconciliation {
                    domain: entry.domain.clone(),
                }
            })?;
            if reconciliation.actual_retained_bytes > reconciliation.actual_peak_bytes {
                return Err(MemoryPlanningError::InvalidReconciliation {
                    domain: entry.domain.clone(),
                    actual_peak_bytes: reconciliation.actual_peak_bytes,
                    actual_retained_bytes: reconciliation.actual_retained_bytes,
                });
            }
            entry.quarantine_bytes = entry
                .quarantine_bytes
                .max(reconciliation.actual_peak_bytes)
                .max(reconciliation.actual_retained_bytes);
            if entry.requires_reconciliation && !entry.holds_exclusive_gate {
                return Err(MemoryPlanningError::ProvisionalReservationNotExclusive {
                    domain: entry.domain.clone(),
                    resource_id: entry.resource_id.clone(),
                });
            }
            if !entry.requires_reconciliation
                && (reconciliation.actual_peak_bytes > entry.reserved_peak_bytes
                    || reconciliation.actual_retained_bytes > entry.quoted_retained_bytes)
            {
                return Err(MemoryPlanningError::BackendQuoteInvariantViolated {
                    domain: entry.domain.clone(),
                    quoted_peak_bytes: entry.reserved_peak_bytes,
                    quoted_retained_bytes: entry.quoted_retained_bytes,
                    actual_peak_bytes: reconciliation.actual_peak_bytes,
                    actual_retained_bytes: reconciliation.actual_retained_bytes,
                });
            }
            if reconciliation.snapshot_after.confidence == MemoryObservationConfidence::Unknown {
                return Err(MemoryPlanningError::MemoryObservationUnavailable {
                    domain: entry.domain.clone(),
                    resource_id: entry.resource_id.clone(),
                });
            }
            let snapshot = reconciliation.snapshot_after.normalized()?;
            let (policy_ceiling, observed_ceiling) = self.broker.policy.limits(snapshot)?;
            let account = accounts.get(&entry.domain).ok_or_else(|| {
                MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: entry.domain.clone(),
                }
            })?;
            if account.quarantined {
                return Err(MemoryPlanningError::DeviceQuarantined {
                    domain: entry.domain.clone(),
                });
            }
            if !pending_entry_is_consistent(account, entry)
                || !exclusive_entry_is_consistent(account, entry)
            {
                accounts
                    .get_mut(&entry.domain)
                    .expect("reservation account exists")
                    .quarantined = true;
                return Err(MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: entry.domain.clone(),
                });
            }
            let other_pending = account
                .pending_bytes
                .checked_sub(entry.reserved_peak_bytes)
                .ok_or_else(|| MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: entry.domain.clone(),
                })?;
            let available_owned = policy_ceiling.saturating_sub(
                account
                    .committed_bytes
                    .saturating_add(other_pending)
                    .saturating_add(account.unreclaimable_bytes),
            );
            // `snapshot_after.free_bytes` already reflects this candidate's
            // live allocation. Only *other* pending transactions still need
            // to be held back from the observed headroom.
            // A zero-byte residual marker proves that this candidate did not
            // grow the domain. Its exclusive gate still protected the
            // observation window, but an already-small heap must not fail the
            // transaction solely because its baseline capacity is below the
            // global headroom policy. Non-zero growth keeps the full live
            // headroom check.
            let observed_safe = reconciliation.actual_peak_bytes == 0
                || snapshot.free_bytes
                    >= self
                        .broker
                        .policy
                        .minimum_headroom_bytes
                        .saturating_add(other_pending);
            if reconciliation.actual_peak_bytes > available_owned || !observed_safe {
                return Err(MemoryPlanningError::PostAllocationBudgetExceeded {
                    domain: entry.domain.clone(),
                    resource_id: entry.resource_id.clone(),
                    actual_peak_bytes: reconciliation.actual_peak_bytes,
                    actual_retained_bytes: reconciliation.actual_retained_bytes,
                    available_owned_bytes: available_owned,
                    other_pending_bytes: other_pending,
                    observed_ceiling,
                });
            }
            account
                .committed_bytes
                .checked_add(reconciliation.actual_retained_bytes)
                .ok_or(MemoryPlanningError::ArithmeticOverflow {
                    operation: "reconciled committed reservation sum",
                })?;
            committed.push((entry.domain.clone(), reconciliation.actual_retained_bytes));
        }
        for entry in &mut self.entries {
            let (_, committed_bytes) = committed
                .iter()
                .find(|(domain, _)| domain == &entry.domain)
                .expect("reconciliation domain set validated above");
            let account = accounts
                .get_mut(&entry.domain)
                .expect("reservation ledger validated above");
            release_pending_bytes(account, entry);
            account.committed_bytes += *committed_bytes;
            add_scoped_committed_bytes(account, entry, *committed_bytes);
            release_exclusive_child(account, entry);
            entry.committed_bytes = *committed_bytes;
            let reconciliation = by_domain
                .get(&entry.domain)
                .expect("reconciliation domain set validated above");
            Self::prepare_receipt_descriptor(
                entry,
                reconciliation.actual_peak_bytes,
                reconciliation.actual_retained_bytes,
            );
        }
        drop(accounts);
        self.refresh_receipt_broker_projection();
        self.transition_receipt_state(RuntimeResourceState::Reconciled);
        self.transition_receipt_state(RuntimeResourceState::Committed);
        self.state = ReservationState::Committed;
        Ok(())
    }

    /// A lost/poisoned backend may intentionally leak its native owner to
    /// avoid an unsafe free. Preserve those bytes; pretending they were
    /// released would allow a guaranteed overcommit.
    ///
    /// A dedicated heap is also quarantined because every consumer of that
    /// domain addresses the failed physical device. `SystemMemory` is
    /// different: CPU and unified-memory accelerators share its capacity but
    /// not their health. A poisoned Metal/UMA backend must leave its
    /// unreclaimable charge in the ledger without disabling the independent
    /// CPU fallback. Backend health remains quarantined by the native backend
    /// owner itself. Ledger corruption still quarantines either domain via the
    /// consistency checks below.
    pub fn quarantine(&mut self) {
        if matches!(
            self.state,
            ReservationState::Released | ReservationState::Quarantined
        ) {
            return;
        }
        let mut accounts = self.broker.lock_accounts();
        for entry in &mut self.entries {
            let account = accounts.entry(entry.domain.clone()).or_default();
            let bytes = match self.state {
                ReservationState::Pending => {
                    release_pending_bytes(account, entry);
                    release_exclusive_child(account, entry);
                    entry.quarantine_bytes
                }
                ReservationState::Committed => {
                    release_committed_bytes(account, entry);
                    entry.committed_bytes
                }
                ReservationState::Quarantined | ReservationState::Released => 0,
            };
            if bytes > 0 {
                account.unreclaimable_bytes = account.unreclaimable_bytes.saturating_add(bytes);
                add_scoped_unreclaimable_bytes(account, entry, bytes);
                Self::prepare_receipt_descriptor(entry, bytes, bytes);
            }
            if matches!(entry.domain, MemoryDomainKey::DedicatedDevice { .. }) {
                account.quarantined = true;
            }
        }
        drop(accounts);
        self.refresh_receipt_broker_projection();
        self.transition_receipt_state(RuntimeResourceState::Quarantined);
        self.persist_quarantined_receipts();
        self.state = ReservationState::Quarantined;
    }

    pub fn reserved_peak_bytes(&self, domain: &MemoryDomainKey) -> Option<u64> {
        self.entries
            .iter()
            .find(|entry| &entry.domain == domain)
            .map(|entry| entry.reserved_peak_bytes)
    }

    pub fn committed_bytes(&self, domain: &MemoryDomainKey) -> Option<u64> {
        self.entries
            .iter()
            .find(|entry| &entry.domain == domain)
            .map(|entry| entry.committed_bytes)
    }

    /// Rebinds a still-pending child to a fresh native quote without changing
    /// the candidate-level bytes already reserved atomically. This is used when
    /// an earlier child intentionally mutates backend-private generation before
    /// the engine-owned child validates its token.
    pub fn rebind_quote(
        &mut self,
        requests: &[DomainReservationRequest],
    ) -> Result<(), MemoryPlanningError> {
        if self.entries.is_empty() && requests.is_empty() {
            return Ok(());
        }
        self.require_pending()?;
        if requests.len() != self.entries.len() {
            return Err(MemoryPlanningError::ReconciliationSetMismatch {
                expected: self.entries.len(),
                actual: requests.len(),
            });
        }
        let mut by_domain = HashMap::with_capacity(requests.len());
        for request in requests {
            if by_domain.insert(request.domain.clone(), request).is_some() {
                return Err(MemoryPlanningError::DuplicateMemoryDomain {
                    domain: request.domain.clone(),
                });
            }
        }
        for entry in &self.entries {
            let request = by_domain.get(&entry.domain).ok_or_else(|| {
                MemoryPlanningError::MissingDomainReconciliation {
                    domain: entry.domain.clone(),
                }
            })?;
            if request.peak_bytes > entry.reserved_peak_bytes
                || request.retained_bytes > entry.reserved_peak_bytes
            {
                return Err(MemoryPlanningError::ReboundQuoteExceedsReservation {
                    domain: entry.domain.clone(),
                    resource_id: request.resource_id.clone(),
                    reserved_peak_bytes: entry.reserved_peak_bytes,
                    rebound_peak_bytes: request.peak_bytes,
                    rebound_retained_bytes: request.retained_bytes,
                });
            }
            if request.requires_reconciliation && !entry.holds_exclusive_gate {
                return Err(MemoryPlanningError::ProvisionalReservationNotExclusive {
                    domain: entry.domain.clone(),
                    resource_id: request.resource_id.clone(),
                });
            }
        }
        for entry in &mut self.entries {
            let request = by_domain
                .get(&entry.domain)
                .expect("rebound domain set validated above");
            entry.resource_id = request.resource_id.clone();
            entry.quoted_retained_bytes = request.retained_bytes;
            entry.requires_reconciliation = request.requires_reconciliation;
        }
        Ok(())
    }

    fn require_pending(&self) -> Result<(), MemoryPlanningError> {
        if self.state == ReservationState::Pending {
            Ok(())
        } else {
            Err(MemoryPlanningError::InvalidReservationTransition)
        }
    }

    fn commit_entries(
        &mut self,
        committed: &[(MemoryDomainKey, u64)],
    ) -> Result<(), MemoryPlanningError> {
        self.require_pending()?;
        if committed.len() != self.entries.len() {
            return Err(MemoryPlanningError::ReconciliationSetMismatch {
                expected: self.entries.len(),
                actual: committed.len(),
            });
        }
        let mut accounts = self.broker.lock_accounts();
        for entry in &self.entries {
            let (_, committed_bytes) = committed
                .iter()
                .find(|(domain, _)| domain == &entry.domain)
                .ok_or_else(|| MemoryPlanningError::MissingDomainReconciliation {
                    domain: entry.domain.clone(),
                })?;
            let account = accounts.get(&entry.domain).ok_or_else(|| {
                MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: entry.domain.clone(),
                }
            })?;
            if !pending_entry_is_consistent(account, entry)
                || !exclusive_entry_is_consistent(account, entry)
            {
                accounts
                    .get_mut(&entry.domain)
                    .expect("reservation account exists")
                    .quarantined = true;
                return Err(MemoryPlanningError::ReservationLedgerCorrupted {
                    domain: entry.domain.clone(),
                });
            }
            account
                .committed_bytes
                .checked_add(*committed_bytes)
                .ok_or(MemoryPlanningError::ArithmeticOverflow {
                    operation: "committed reservation sum",
                })?;
        }
        for entry in &mut self.entries {
            let (_, committed_bytes) = committed
                .iter()
                .find(|(domain, _)| domain == &entry.domain)
                .expect("validated above");
            let account = accounts.get_mut(&entry.domain).expect("validated above");
            release_pending_bytes(account, entry);
            account.committed_bytes += *committed_bytes;
            add_scoped_committed_bytes(account, entry, *committed_bytes);
            release_exclusive_child(account, entry);
            entry.committed_bytes = *committed_bytes;
            Self::prepare_receipt_descriptor(entry, *committed_bytes, *committed_bytes);
        }
        drop(accounts);
        self.refresh_receipt_broker_projection();
        self.transition_receipt_state(RuntimeResourceState::Committed);
        self.state = ReservationState::Committed;
        Ok(())
    }

    fn release(&mut self) {
        if matches!(
            self.state,
            ReservationState::Released | ReservationState::Quarantined
        ) {
            return;
        }
        let mut accounts = self.broker.lock_accounts();
        for entry in &self.entries {
            let account = accounts.entry(entry.domain.clone()).or_default();
            match self.state {
                ReservationState::Pending => {
                    release_pending_bytes(account, entry);
                    release_exclusive_child(account, entry);
                }
                ReservationState::Committed => {
                    release_committed_bytes(account, entry);
                }
                ReservationState::Quarantined | ReservationState::Released => {}
            }
        }
        drop(accounts);
        self.release_receipts();
        self.state = ReservationState::Released;
    }
}

fn release_exclusive_child(account: &mut DomainAccount, entry: &ReservationEntry) {
    if entry.holds_exclusive_gate {
        if account.exclusive_pending_cohort != Some(entry.cohort)
            || account.exclusive_pending_children == 0
        {
            account.quarantined = true;
            return;
        }
        account.exclusive_pending_children -= 1;
        if account.exclusive_pending_children == 0 {
            account.exclusive_pending_cohort = None;
        }
    }
}

fn domain_account_is_consistent(account: &DomainAccount) -> bool {
    let cohort_sum = account
        .pending_bytes_by_cohort
        .values()
        .try_fold(0_u64, |total, bytes| total.checked_add(*bytes));
    if cohort_sum != Some(account.pending_bytes) {
        return false;
    }
    let scoped = account.by_scope.values().try_fold(
        (0_u64, 0_u64, 0_u64),
        |(pending, committed, unreclaimable), scope| {
            let placement_totals = scope.by_placement.values().try_fold(
                (0_u64, 0_u64, 0_u64),
                |(placement_pending, placement_committed, placement_unreclaimable), placement| {
                    Some((
                        placement_pending.checked_add(placement.pending_bytes)?,
                        placement_committed.checked_add(placement.committed_bytes)?,
                        placement_unreclaimable.checked_add(placement.unreclaimable_bytes)?,
                    ))
                },
            )?;
            if placement_totals
                != (
                    scope.pending_bytes,
                    scope.committed_bytes,
                    scope.unreclaimable_bytes,
                )
            {
                return None;
            }
            Some((
                pending.checked_add(scope.pending_bytes)?,
                committed.checked_add(scope.committed_bytes)?,
                unreclaimable.checked_add(scope.unreclaimable_bytes)?,
            ))
        },
    );
    if scoped.is_none_or(|(pending, committed, unreclaimable)| {
        pending > account.pending_bytes
            || committed > account.committed_bytes
            || unreclaimable > account.unreclaimable_bytes
    }) {
        return false;
    }

    match (
        account.exclusive_pending_children,
        account.exclusive_pending_cohort,
    ) {
        (0, None) => true,
        (0, Some(_)) | (_, None) => false,
        (_, Some(cohort)) => {
            account.pending_bytes == 0 || account.pending_bytes_by_cohort.contains_key(&cohort)
        }
    }
}

fn pending_entry_is_consistent(account: &DomainAccount, entry: &ReservationEntry) -> bool {
    let scope_consistent = entry.owner_scope_id.is_none_or(|scope_id| {
        account.by_scope.get(&scope_id).is_some_and(|scope| {
            scope.pending_bytes >= entry.reserved_peak_bytes
                && scope
                    .by_placement
                    .get(&entry.owner_placement)
                    .is_some_and(|placement| placement.pending_bytes >= entry.reserved_peak_bytes)
        })
    });
    scope_consistent
        && account.pending_bytes >= entry.reserved_peak_bytes
        && account
            .pending_bytes_by_cohort
            .get(&entry.cohort)
            .map_or(entry.reserved_peak_bytes == 0, |bytes| {
                *bytes >= entry.reserved_peak_bytes
            })
}

fn exclusive_entry_is_consistent(account: &DomainAccount, entry: &ReservationEntry) -> bool {
    !entry.holds_exclusive_gate
        || (account.exclusive_pending_cohort == Some(entry.cohort)
            && account.exclusive_pending_children > 0)
}

fn release_pending_bytes(account: &mut DomainAccount, entry: &ReservationEntry) {
    let Some(next_total) = account.pending_bytes.checked_sub(entry.reserved_peak_bytes) else {
        account.quarantined = true;
        return;
    };
    let Some(current_cohort) = account.pending_bytes_by_cohort.get(&entry.cohort).copied() else {
        if entry.reserved_peak_bytes == 0 {
            return;
        }
        account.quarantined = true;
        return;
    };
    let Some(next_cohort) = current_cohort.checked_sub(entry.reserved_peak_bytes) else {
        account.quarantined = true;
        return;
    };
    let scoped_next = if let Some(scope_id) = entry.owner_scope_id {
        let Some(scoped) = account.by_scope.get(&scope_id) else {
            account.quarantined = true;
            return;
        };
        let Some(next_scoped) = scoped.pending_bytes.checked_sub(entry.reserved_peak_bytes) else {
            account.quarantined = true;
            return;
        };
        let Some(placement) = scoped.by_placement.get(&entry.owner_placement) else {
            account.quarantined = true;
            return;
        };
        let Some(next_placement) = placement
            .pending_bytes
            .checked_sub(entry.reserved_peak_bytes)
        else {
            account.quarantined = true;
            return;
        };
        Some((scope_id, next_scoped, next_placement))
    } else {
        None
    };

    account.pending_bytes = next_total;
    if let Some((scope_id, next_scoped, next_placement)) = scoped_next {
        let scoped = account
            .by_scope
            .get_mut(&scope_id)
            .expect("scoped reservation validated above");
        scoped.pending_bytes = next_scoped;
        scoped
            .by_placement
            .get_mut(&entry.owner_placement)
            .expect("placement reservation validated above")
            .pending_bytes = next_placement;
    }
    if next_cohort == 0 {
        account.pending_bytes_by_cohort.remove(&entry.cohort);
    } else {
        *account
            .pending_bytes_by_cohort
            .get_mut(&entry.cohort)
            .expect("cohort reservation validated above") = next_cohort;
    }
}

fn release_committed_bytes(account: &mut DomainAccount, entry: &ReservationEntry) {
    let Some(next) = account.committed_bytes.checked_sub(entry.committed_bytes) else {
        account.quarantined = true;
        return;
    };
    let scoped_next = if let Some(scope_id) = entry.owner_scope_id {
        let Some(scoped) = account.by_scope.get(&scope_id) else {
            account.quarantined = true;
            return;
        };
        let Some(next_scoped) = scoped.committed_bytes.checked_sub(entry.committed_bytes) else {
            account.quarantined = true;
            return;
        };
        let Some(placement) = scoped.by_placement.get(&entry.owner_placement) else {
            account.quarantined = true;
            return;
        };
        let Some(next_placement) = placement.committed_bytes.checked_sub(entry.committed_bytes)
        else {
            account.quarantined = true;
            return;
        };
        Some((scope_id, next_scoped, next_placement))
    } else {
        None
    };

    account.committed_bytes = next;
    if let Some((scope_id, next_scoped, next_placement)) = scoped_next {
        let scoped = account
            .by_scope
            .get_mut(&scope_id)
            .expect("scoped reservation validated above");
        scoped.committed_bytes = next_scoped;
        scoped
            .by_placement
            .get_mut(&entry.owner_placement)
            .expect("placement reservation validated above")
            .committed_bytes = next_placement;
    }
}

fn add_scoped_committed_bytes(account: &mut DomainAccount, entry: &ReservationEntry, bytes: u64) {
    if let Some(scope_id) = entry.owner_scope_id {
        let scoped = account.by_scope.entry(scope_id).or_default();
        let placement = scoped
            .by_placement
            .entry(entry.owner_placement)
            .or_default();
        let (Some(next_scoped), Some(next_placement)) = (
            scoped.committed_bytes.checked_add(bytes),
            placement.committed_bytes.checked_add(bytes),
        ) else {
            account.quarantined = true;
            return;
        };
        scoped.committed_bytes = next_scoped;
        placement.committed_bytes = next_placement;
    }
}

fn add_scoped_unreclaimable_bytes(
    account: &mut DomainAccount,
    entry: &ReservationEntry,
    bytes: u64,
) {
    if let Some(scope_id) = entry.owner_scope_id {
        let scoped = account.by_scope.entry(scope_id).or_default();
        let placement = scoped
            .by_placement
            .entry(entry.owner_placement)
            .or_default();
        let (Some(next_scoped), Some(next_placement)) = (
            scoped.unreclaimable_bytes.checked_add(bytes),
            placement.unreclaimable_bytes.checked_add(bytes),
        ) else {
            account.quarantined = true;
            return;
        };
        scoped.unreclaimable_bytes = next_scoped;
        placement.unreclaimable_bytes = next_placement;
    }
}

impl Drop for DeviceMemoryReservationBatch {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemoryPlanningError {
    #[error("physical device key must not be empty")]
    EmptyPhysicalDeviceKey,
    #[error("memory resource id must not be empty")]
    EmptyResourceId,
    #[error("memory resource '{resource_id}' has no live execution phase")]
    EmptyPhaseSet { resource_id: String },
    #[error("memory snapshot is invalid: free_bytes={free_bytes}, total_bytes={total_bytes}")]
    InvalidMemorySnapshot { free_bytes: u64, total_bytes: u64 },
    #[error("device owned-memory fraction is invalid: basis_points={basis_points}")]
    InvalidOwnedFraction { basis_points: u16 },
    #[error(
        "memory resource '{resource_id}' has invalid incremental commitment: peak={incremental_peak_bytes:?}, retained={incremental_retained_bytes:?}"
    )]
    InvalidCommitmentBound {
        resource_id: String,
        incremental_peak_bytes: Option<u64>,
        incremental_retained_bytes: Option<u64>,
    },
    #[error(
        "memory domain {domain} has an invalid footprint: peak={peak_bytes}, retained={retained_bytes}"
    )]
    InvalidDomainFootprint {
        domain: MemoryDomainKey,
        peak_bytes: u64,
        retained_bytes: u64,
    },
    #[error("memory domain {domain} appears more than once in one atomic operation")]
    DuplicateMemoryDomain { domain: MemoryDomainKey },
    #[error("one atomic memory reservation mixed distinct execution cohorts")]
    MixedReservationCohorts,
    #[error("memory reservation placement set mismatch: expected={expected}, actual={actual}")]
    ReservationPlacementSetMismatch { expected: usize, actual: usize },
    #[error("a diagnostic memory placement requires an execution-service scope")]
    ReservationPlacementWithoutScope,
    #[error("memory capacity is unproven for resource '{resource_id}'")]
    CapacityUnproven { resource_id: String },
    #[error("memory observation is unavailable for {domain} while reserving '{resource_id}'")]
    MemoryObservationUnavailable {
        domain: MemoryDomainKey,
        resource_id: String,
    },
    #[error("memory domain {domain} is quarantined after a terminal device failure")]
    DeviceQuarantined { domain: MemoryDomainKey },
    #[error(
        "memory domain {domain} is held exclusively by another provisional candidate while reserving '{resource_id}': pending={pending_bytes}, exclusive_children={exclusive_pending_children}"
    )]
    DeviceDomainBusy {
        domain: MemoryDomainKey,
        resource_id: String,
        pending_bytes: u64,
        exclusive_pending_children: u32,
    },
    #[error(
        "device memory budget exceeded for {domain} while reserving '{resource_id}': requested={requested_bytes}, available={available_bytes}, pending={pending_bytes}, committed={committed_bytes}, unreclaimable={unreclaimable_bytes}, policy_ceiling={policy_ceiling}, observed_ceiling={observed_ceiling}"
    )]
    DeviceBudgetExceeded {
        domain: MemoryDomainKey,
        resource_id: String,
        requested_bytes: u64,
        pending_bytes: u64,
        committed_bytes: u64,
        unreclaimable_bytes: u64,
        policy_ceiling: u64,
        observed_ceiling: u64,
        available_bytes: u64,
    },
    #[error(
        "provisional memory quote for {domain} ('{resource_id}') requires post-allocation reconciliation"
    )]
    ReconciliationRequired {
        domain: MemoryDomainKey,
        resource_id: String,
    },
    #[error(
        "provisional memory reservation for {domain} ('{resource_id}') is missing its exclusive domain gate"
    )]
    ProvisionalReservationNotExclusive {
        domain: MemoryDomainKey,
        resource_id: String,
    },
    #[error("memory reconciliation domain set mismatch: expected={expected}, actual={actual}")]
    ReconciliationSetMismatch { expected: usize, actual: usize },
    #[error("memory reconciliation is missing domain {domain}")]
    MissingDomainReconciliation { domain: MemoryDomainKey },
    #[error(
        "memory reconciliation for {domain} is invalid: actual_peak={actual_peak_bytes}, actual_retained={actual_retained_bytes}"
    )]
    InvalidReconciliation {
        domain: MemoryDomainKey,
        actual_peak_bytes: u64,
        actual_retained_bytes: u64,
    },
    #[error(
        "backend memory quote invariant was violated for {domain}: quoted_peak={quoted_peak_bytes}, quoted_retained={quoted_retained_bytes}, actual_peak={actual_peak_bytes}, actual_retained={actual_retained_bytes}"
    )]
    BackendQuoteInvariantViolated {
        domain: MemoryDomainKey,
        quoted_peak_bytes: u64,
        quoted_retained_bytes: u64,
        actual_peak_bytes: u64,
        actual_retained_bytes: u64,
    },
    #[error(
        "fresh native quote for {domain} ('{resource_id}') exceeds its atomic child reservation: reserved_peak={reserved_peak_bytes}, rebound_peak={rebound_peak_bytes}, rebound_retained={rebound_retained_bytes}"
    )]
    ReboundQuoteExceedsReservation {
        domain: MemoryDomainKey,
        resource_id: String,
        reserved_peak_bytes: u64,
        rebound_peak_bytes: u64,
        rebound_retained_bytes: u64,
    },
    #[error(
        "post-allocation memory budget exceeded for {domain} ('{resource_id}'): peak={actual_peak_bytes}, retained={actual_retained_bytes}, owned_available={available_owned_bytes}, other_pending={other_pending_bytes}, observed_ceiling={observed_ceiling}"
    )]
    PostAllocationBudgetExceeded {
        domain: MemoryDomainKey,
        resource_id: String,
        actual_peak_bytes: u64,
        actual_retained_bytes: u64,
        available_owned_bytes: u64,
        other_pending_bytes: u64,
        observed_ceiling: u64,
    },
    #[error("memory reservation ledger is inconsistent for {domain}")]
    ReservationLedgerCorrupted { domain: MemoryDomainKey },
    #[error("memory reservation is not pending and cannot be committed")]
    InvalidReservationTransition,
    #[error("memory planning arithmetic overflowed during {operation}")]
    ArithmeticOverflow { operation: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn domain() -> MemoryDomainKey {
        MemoryDomainKey::DedicatedDevice {
            physical_device: PhysicalDeviceKey::new("0000:01:00.0").unwrap(),
            heap_index: 0,
        }
    }

    fn snapshot(free: u64) -> DeviceMemorySnapshot {
        DeviceMemorySnapshot {
            free_bytes: free,
            total_bytes: 8 * GIB,
            confidence: MemoryObservationConfidence::DeviceSnapshot,
        }
    }

    fn request(
        domain: MemoryDomainKey,
        free: u64,
        peak_bytes: u64,
        retained_bytes: u64,
        resource_id: &str,
    ) -> DomainReservationRequest {
        DomainReservationRequest {
            domain,
            snapshot: snapshot(free),
            peak_bytes,
            retained_bytes,
            observed_peak_bytes: None,
            requires_reconciliation: false,
            resource_id: resource_id.to_string(),
            cohort_id: None,
        }
    }

    #[test]
    fn phase_peak_uses_maximum_overlap_not_sum_of_all_workspaces() {
        let domain = domain();
        let footprint = AllocationFootprint::new(vec![
            MemoryClaim {
                resource_id: "weights".to_string(),
                domain: domain.clone(),
                requested_bytes: 4 * GIB,
                incremental_peak_bytes: Some(4 * GIB),
                incremental_retained_bytes: Some(4 * GIB),
                confidence: QuoteConfidence::ExactCommitted,
                lifetime: AllocationLifetime::PackShared,
                phases: PhaseSet::ALL,
            },
            MemoryClaim {
                resource_id: "kv".to_string(),
                domain: domain.clone(),
                requested_bytes: GIB / 4,
                incremental_peak_bytes: Some(GIB / 4),
                incremental_retained_bytes: Some(GIB / 4),
                confidence: QuoteConfidence::ExactCommitted,
                lifetime: AllocationLifetime::SessionResident,
                phases: PhaseSet::range(
                    ExecutionPhase::DecoderPrefill,
                    ExecutionPhase::DecoderStep,
                ),
            },
            MemoryClaim {
                resource_id: "encoder-workspace".to_string(),
                domain: domain.clone(),
                requested_bytes: GIB,
                incremental_peak_bytes: Some(GIB),
                incremental_retained_bytes: Some(0),
                confidence: QuoteConfidence::CommittedUpperBound,
                lifetime: AllocationLifetime::PhaseTransient,
                phases: PhaseSet::one(ExecutionPhase::Encoder),
            },
            MemoryClaim {
                resource_id: "decoder-workspace".to_string(),
                domain: domain.clone(),
                requested_bytes: GIB / 2,
                incremental_peak_bytes: Some(GIB / 2),
                incremental_retained_bytes: Some(GIB / 2),
                confidence: QuoteConfidence::CommittedUpperBound,
                lifetime: AllocationLifetime::RunnerRetainedHighWater,
                phases: PhaseSet::range(
                    ExecutionPhase::DecoderPrefill,
                    ExecutionPhase::DecoderStep,
                ),
            },
        ]);
        // Encoder peak = 5 GiB. Decoder peak = 4.75 GiB. A naive sum would
        // incorrectly report 5.75 GiB.
        assert_eq!(footprint.peak_bytes(&domain).unwrap(), 5 * GIB);
        assert_eq!(footprint.retained_bytes(&domain).unwrap(), 19 * GIB / 4);
    }

    #[test]
    fn two_concurrent_sessions_reserve_atomically() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let first = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                4 * GIB,
                4 * GIB,
                "session-a",
            )])
            .unwrap();
        let second = broker.try_reserve_batch(vec![request(
            domain(),
            7 * GIB,
            4 * GIB,
            4 * GIB,
            "session-b",
        )]);
        assert!(matches!(
            second,
            Err(MemoryPlanningError::DeviceBudgetExceeded { .. })
        ));
        assert_eq!(broker.usage(&domain()).pending_bytes, 4 * GIB);
        drop(first);
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
    }

    #[test]
    fn committed_lease_is_refunded_only_when_actual_owner_drops() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut lease = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                GIB,
                GIB / 2,
                "resident-kv",
            )])
            .unwrap();
        lease.commit_quoted().unwrap();
        assert_eq!(broker.usage(&domain()).committed_bytes, GIB / 2);
        drop(lease);
        assert_eq!(broker.usage(&domain()).committed_bytes, 0);
    }

    #[test]
    fn multi_domain_candidate_is_all_or_nothing() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let host = MemoryDomainKey::SystemMemory;
        let result = broker.try_reserve_batch(vec![
            request(domain(), 7 * GIB, GIB, GIB, "gpu"),
            request(host.clone(), GIB, 1, 1, "host"),
        ]);
        assert!(matches!(
            result,
            Err(MemoryPlanningError::DeviceBudgetExceeded { domain, .. }) if domain == host
        ));
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).pending_bytes,
            0
        );
    }

    #[test]
    fn provisional_quote_requires_live_reconciliation() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut provisional = request(domain(), 7 * GIB, GIB, GIB / 2, "vulkan-private");
        provisional.requires_reconciliation = true;
        let mut lease = broker.try_reserve_batch(vec![provisional]).unwrap();
        assert!(matches!(
            lease.commit_quoted(),
            Err(MemoryPlanningError::ReconciliationRequired { .. })
        ));
        lease
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: GIB + GIB / 4,
                actual_retained_bytes: 3 * GIB / 4,
                snapshot_after: snapshot(6 * GIB),
            }])
            .unwrap();
        assert_eq!(lease.committed_bytes(&domain()), Some(3 * GIB / 4));
        assert_eq!(broker.usage(&domain()).committed_bytes, 3 * GIB / 4);
    }

    #[test]
    fn zero_growth_residual_marker_does_not_require_new_domain_headroom() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let mut residual = request(domain(), GIB / 4, 0, 0, "vulkan-unused-small-heap");
        residual.requires_reconciliation = true;
        let mut lease = broker.try_reserve_batch(vec![residual]).unwrap();

        lease
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: 0,
                actual_retained_bytes: 0,
                snapshot_after: snapshot(GIB / 4),
            }])
            .unwrap();

        assert_eq!(lease.committed_bytes(&domain()), Some(0));
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
        assert_eq!(broker.usage(&domain()).committed_bytes, 0);
        assert!(!broker.usage(&domain()).exclusive_pending);
    }

    #[test]
    fn nonzero_growth_on_small_domain_still_requires_headroom() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let mut residual = request(domain(), GIB / 4, 0, 0, "vulkan-small-heap-growth");
        residual.requires_reconciliation = true;
        let mut lease = broker.try_reserve_batch(vec![residual]).unwrap();

        let error = lease
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: 1,
                actual_retained_bytes: 1,
                snapshot_after: snapshot(GIB / 4),
            }])
            .unwrap_err();

        assert!(matches!(
            error,
            MemoryPlanningError::PostAllocationBudgetExceeded { .. }
        ));
        assert!(lease.is_pending());
        assert!(broker.usage(&domain()).exclusive_pending);
    }

    #[test]
    fn zero_growth_still_rejects_unknown_post_allocation_observation() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let mut residual = request(domain(), GIB / 4, 0, 0, "vulkan-unknown-small-heap");
        residual.requires_reconciliation = true;
        let mut lease = broker.try_reserve_batch(vec![residual]).unwrap();

        let error = lease
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: 0,
                actual_retained_bytes: 0,
                snapshot_after: DeviceMemorySnapshot {
                    confidence: MemoryObservationConfidence::Unknown,
                    ..snapshot(GIB / 4)
                },
            }])
            .unwrap_err();

        assert!(matches!(
            error,
            MemoryPlanningError::MemoryObservationUnavailable { .. }
        ));
        assert!(lease.is_pending());
        assert!(broker.usage(&domain()).exclusive_pending);
    }

    #[test]
    fn provisional_candidate_holds_domain_exclusive_until_reconciliation() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut private = request(domain(), 7 * GIB, 0, 0, "cuda-graph-private");
        private.requires_reconciliation = true;
        let engine = request(domain(), 7 * GIB, GIB, GIB, "scheduler-arena");
        let mut children = broker
            .try_reserve_partitioned(vec![vec![private], vec![engine]])
            .unwrap();
        let mut private = children.remove(0);
        let mut engine = children.remove(0);

        assert_eq!(broker.usage(&domain()).pending_bytes, GIB);
        assert!(broker.usage(&domain()).exclusive_pending);
        engine.commit_quoted().unwrap();
        assert_eq!(broker.usage(&domain()).committed_bytes, GIB);
        assert!(broker.usage(&domain()).exclusive_pending);

        let blocked = broker.try_reserve_batch(vec![request(
            domain(),
            6 * GIB,
            0,
            0,
            "second-session-zero-byte",
        )]);
        assert!(matches!(
            blocked,
            Err(MemoryPlanningError::DeviceDomainBusy { .. })
        ));

        // The provider did not prove an upper bound: the live graph-specific
        // high-water may exceed the zero estimate only because this candidate
        // has held the physical domain exclusively since admission.
        private
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: 2 * GIB,
                actual_retained_bytes: 2 * GIB,
                snapshot_after: snapshot(4 * GIB),
            }])
            .unwrap();
        assert!(!broker.usage(&domain()).exclusive_pending);
        assert_eq!(broker.usage(&domain()).committed_bytes, 3 * GIB);
    }

    #[test]
    fn nested_provisional_reservations_share_only_their_attempts_exclusive_gate() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let cohort = MemoryReservationCohortId::new(41);
        let mut outer_request = request(domain(), 7 * GIB, GIB, GIB / 2, "outer-host");
        outer_request.requires_reconciliation = true;
        let outer_request = outer_request.with_cohort_id(Some(cohort));
        let mut outer = broker.try_reserve_batch(vec![outer_request]).unwrap();

        let mut nested_request = request(domain(), 7 * GIB, GIB / 2, GIB / 4, "nested-native");
        nested_request.requires_reconciliation = true;
        let nested_request = nested_request.with_cohort_id(Some(cohort));
        let nested = broker.try_reserve_batch(vec![nested_request]).unwrap();
        assert_eq!(broker.usage(&domain()).pending_bytes, 3 * GIB / 2);
        assert!(broker.usage(&domain()).exclusive_pending);

        let unrelated = request(domain(), 7 * GIB, 0, 0, "unrelated")
            .with_cohort_id(Some(MemoryReservationCohortId::new(42)));
        assert!(matches!(
            broker.try_reserve_batch(vec![unrelated]),
            Err(MemoryPlanningError::DeviceDomainBusy { .. })
        ));

        outer
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: GIB,
                actual_retained_bytes: GIB / 2,
                snapshot_after: snapshot(6 * GIB),
            }])
            .unwrap();
        // The nested provisional owner still holds the cohort gate.
        assert!(broker.usage(&domain()).exclusive_pending);
        drop(nested);
        assert!(!broker.usage(&domain()).exclusive_pending);
    }

    #[test]
    fn one_atomic_reservation_cannot_mix_execution_cohorts() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let first = request(domain(), 7 * GIB, 1, 1, "first")
            .with_cohort_id(Some(MemoryReservationCohortId::new(1)));
        let second = request(MemoryDomainKey::SystemMemory, 7 * GIB, 1, 1, "second")
            .with_cohort_id(Some(MemoryReservationCohortId::new(2)));
        assert!(matches!(
            broker.try_reserve_batch(vec![first, second]),
            Err(MemoryPlanningError::MixedReservationCohorts)
        ));
        assert_eq!(broker.usage(&domain()), DeviceMemoryUsage::default());
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory),
            DeviceMemoryUsage::default()
        );
    }

    #[test]
    fn provisional_candidate_cannot_enter_behind_existing_pending_work() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let exact = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                GIB,
                GIB,
                "already-pending",
            )])
            .unwrap();
        let mut provisional = request(domain(), 7 * GIB, 0, 0, "provisional");
        provisional.requires_reconciliation = true;
        assert!(matches!(
            broker.try_reserve_batch(vec![provisional]),
            Err(MemoryPlanningError::DeviceDomainBusy { .. })
        ));
        drop(exact);
    }

    #[test]
    fn concurrent_provisional_candidates_cannot_both_enter_one_domain() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let start = Arc::new(std::sync::Barrier::new(2));
        let finish = Arc::new(std::sync::Barrier::new(2));
        let mut workers = Vec::new();
        for index in 0..2 {
            let broker = Arc::clone(&broker);
            let start = Arc::clone(&start);
            let finish = Arc::clone(&finish);
            workers.push(std::thread::spawn(move || {
                let mut provisional =
                    request(domain(), 7 * GIB, 0, 0, &format!("provisional-{index}"));
                provisional.requires_reconciliation = true;
                start.wait();
                let result = broker.try_reserve_batch(vec![provisional]);
                // Keep the winning gate live until the losing attempt has
                // returned, so scheduling order cannot turn this into two
                // sequentially-successful admissions.
                finish.wait();
                match result {
                    Ok(_lease) => true,
                    Err(MemoryPlanningError::DeviceDomainBusy { .. }) => false,
                    Err(error) => panic!("unexpected concurrent admission error: {error}"),
                }
            }));
        }
        let outcomes = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(outcomes.iter().filter(|outcome| **outcome).count(), 1);
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
        assert!(!broker.usage(&domain()).exclusive_pending);
    }

    #[test]
    fn partitioned_candidate_admission_is_atomic_and_children_refund_only_their_owner() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let rejected = broker.try_reserve_partitioned(vec![
            vec![request(domain(), 7 * GIB, 4 * GIB, 4 * GIB, "private")],
            vec![request(domain(), 7 * GIB, 4 * GIB, 4 * GIB, "engine")],
        ]);
        assert!(matches!(
            rejected,
            Err(MemoryPlanningError::DeviceBudgetExceeded { .. })
        ));
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);

        let mut children = broker
            .try_reserve_partitioned(vec![
                vec![request(domain(), 7 * GIB, GIB, GIB, "private")],
                vec![request(domain(), 7 * GIB, 2 * GIB, 2 * GIB, "engine")],
            ])
            .unwrap();
        let mut private = children.remove(0);
        let engine = children.remove(0);
        assert_eq!(broker.usage(&domain()).pending_bytes, 3 * GIB);
        private.commit_quoted().unwrap();
        assert_eq!(broker.usage(&domain()).pending_bytes, 2 * GIB);
        assert_eq!(broker.usage(&domain()).committed_bytes, GIB);
        drop(private);
        // Dropping the private owner refunds only its child; the scheduler
        // child's independently-owned pending bytes remain reserved.
        assert_eq!(broker.usage(&domain()).pending_bytes, 2 * GIB);
        assert_eq!(broker.usage(&domain()).committed_bytes, 0);
        drop(engine);
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
    }

    #[test]
    fn fresh_quote_rebind_cannot_expand_an_atomically_admitted_child() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let mut child = broker
            .try_reserve_batch(vec![request(
                domain(),
                8 * GIB,
                2 * GIB,
                GIB,
                "scheduler-before-private",
            )])
            .unwrap();

        child
            .rebind_quote(&[request(
                domain(),
                7 * GIB,
                GIB,
                GIB / 2,
                "scheduler-after-private",
            )])
            .unwrap();
        // Rebinding replaces the quote token/shape but never refunds capacity
        // early: the original candidate-level peak remains pending until the
        // child commits or drops.
        assert_eq!(broker.usage(&domain()).pending_bytes, 2 * GIB);
        child.commit_quoted().unwrap();
        assert_eq!(broker.usage(&domain()).committed_bytes, GIB / 2);

        let mut expanding = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                GIB,
                GIB,
                "scheduler-original",
            )])
            .unwrap();
        let error = expanding
            .rebind_quote(&[request(
                domain(),
                7 * GIB,
                2 * GIB,
                GIB,
                "scheduler-expanded",
            )])
            .unwrap_err();
        assert!(matches!(
            error,
            MemoryPlanningError::ReboundQuoteExceedsReservation { .. }
        ));
        assert_eq!(expanding.reserved_peak_bytes(&domain()), Some(GIB));
    }

    #[test]
    fn over_budget_reconciliation_stays_pending_until_owner_teardown() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: GIB,
        }));
        let mut provisional = request(domain(), 7 * GIB, GIB, GIB / 2, "driver-private");
        provisional.requires_reconciliation = true;
        let mut lease = broker.try_reserve_batch(vec![provisional]).unwrap();
        let error = lease
            .reconcile_and_commit(&[DomainMemoryReconciliation {
                domain: domain(),
                actual_peak_bytes: 2 * GIB,
                actual_retained_bytes: 2 * GIB,
                snapshot_after: snapshot(GIB / 2),
            }])
            .unwrap_err();
        assert!(matches!(
            error,
            MemoryPlanningError::PostAllocationBudgetExceeded { .. }
        ));
        assert!(lease.is_pending());
        assert_eq!(broker.usage(&domain()).pending_bytes, GIB);
        drop(lease);
        assert_eq!(broker.usage(&domain()).pending_bytes, 0);
    }

    #[test]
    fn quarantine_never_refunds_an_unreclaimable_native_allocation() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut lease = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                GIB,
                GIB,
                "poisoned-backend",
            )])
            .unwrap();
        lease.commit_quoted().unwrap();
        lease.quarantine();
        drop(lease);
        let usage = broker.usage(&domain());
        assert_eq!(usage.committed_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, GIB);
        assert!(usage.quarantined);
        assert!(matches!(
            broker.try_reserve_batch(vec![request(domain(), 7 * GIB, 1, 1, "next")]),
            Err(MemoryPlanningError::DeviceQuarantined { .. })
        ));
    }

    #[test]
    fn unified_backend_quarantine_charges_bytes_without_disabling_cpu_memory() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut lease = broker
            .try_reserve_batch(vec![request(
                MemoryDomainKey::SystemMemory,
                7 * GIB,
                GIB,
                GIB,
                "poisoned-unified-backend",
            )])
            .unwrap();
        lease.commit_quoted().unwrap();
        lease.quarantine();
        drop(lease);

        let usage = broker.usage(&MemoryDomainKey::SystemMemory);
        assert_eq!(usage.committed_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, GIB);
        assert!(!usage.quarantined);

        let cpu = broker
            .try_reserve_batch(vec![request(
                MemoryDomainKey::SystemMemory,
                7 * GIB,
                GIB,
                GIB,
                "cpu-fallback",
            )])
            .expect("CPU may use the remaining system-memory budget");
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).pending_bytes,
            GIB
        );
        drop(cpu);
    }

    #[test]
    fn pending_release_corruption_quarantines_instead_of_saturating() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let lease = broker
            .try_reserve_batch(vec![request(domain(), 7 * GIB, GIB, GIB, "pending")])
            .unwrap();
        broker
            .lock_accounts()
            .get_mut(&domain())
            .expect("domain account")
            .pending_bytes = 0;

        drop(lease);
        let usage = broker.usage(&domain());
        assert!(usage.quarantined);
        assert!(matches!(
            broker.try_reserve_batch(vec![request(domain(), 7 * GIB, 1, 1, "next")]),
            Err(MemoryPlanningError::DeviceQuarantined { .. })
        ));
    }

    #[test]
    fn corrupt_cohort_ledger_cannot_be_committed_or_reused() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut lease = broker
            .try_reserve_batch(vec![request(domain(), 7 * GIB, GIB, GIB, "pending")])
            .unwrap();
        broker
            .lock_accounts()
            .get_mut(&domain())
            .expect("domain account")
            .pending_bytes_by_cohort
            .clear();

        assert!(matches!(
            lease.commit_quoted(),
            Err(MemoryPlanningError::ReservationLedgerCorrupted { .. })
        ));
        assert!(lease.is_pending());
        assert!(broker.usage(&domain()).quarantined);
        assert!(matches!(
            broker.try_reserve_batch(vec![request(domain(), 7 * GIB, 1, 1, "next")]),
            Err(MemoryPlanningError::DeviceQuarantined { .. })
        ));
    }

    #[test]
    fn committed_release_corruption_quarantines_instead_of_saturating() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default()));
        let mut lease = broker
            .try_reserve_batch(vec![request(domain(), 7 * GIB, GIB, GIB, "committed")])
            .unwrap();
        lease.commit_quoted().unwrap();
        broker
            .lock_accounts()
            .get_mut(&domain())
            .expect("domain account")
            .committed_bytes = 0;

        drop(lease);
        assert!(broker.usage(&domain()).quarantined);
    }

    #[test]
    fn receipt_lifecycle_mirrors_commit_quarantine_and_release_without_changing_ledger() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let collector = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            64,
        )
        .unwrap();
        let scope_id = collector.snapshot().scope_id;
        let lane = collector
            .lane_projection(
                crate::device::execution_route::ExecutionProvider::Cpu,
                "CPU",
                crate::device::execution_policy::ExecutionPlacement::CpuOnly,
                crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            )
            .unwrap();
        let placement = RuntimeOwnerPlacement::LaneBound(lane);
        let receipt_domain = MemoryDomainKey::SystemMemory;
        let mut lease = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![request(
                    receipt_domain.clone(),
                    7 * GIB,
                    GIB,
                    GIB / 2,
                    "native-hook",
                )],
                Some(scope_id),
                placement,
            )
            .unwrap();
        let owner = collector
            .owner_descriptor("native-memory-test", None, Some("native-hook"), Some(lane))
            .unwrap();
        let resource = collector
            .resource_descriptor(
                "native-memory-domain",
                &receipt_domain,
                GIB,
                GIB,
                GIB / 2,
                QuoteConfidence::CommittedUpperBound,
                Some(MemoryObservationConfidence::DeviceSnapshot),
            )
            .unwrap();
        lease.attach_receipt(
            collector.clone(),
            owner,
            vec![(receipt_domain.clone(), resource)],
        );
        assert_eq!(
            collector.snapshot().live_owners[0]
                .resources
                .values()
                .next()
                .unwrap()
                .state,
            RuntimeResourceState::Reserved
        );
        lease.commit_quoted().unwrap();
        assert_eq!(
            collector.snapshot().live_owners[0]
                .resources
                .values()
                .next()
                .unwrap()
                .state,
            RuntimeResourceState::Committed
        );
        assert_eq!(broker.usage(&receipt_domain).committed_bytes, GIB / 2);
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
        lease.quarantine();
        assert_eq!(
            collector.snapshot().live_owners[0]
                .resources
                .values()
                .next()
                .unwrap()
                .state,
            RuntimeResourceState::Quarantined
        );
        drop(lease);
        assert_eq!(broker.usage(&receipt_domain).unreclaimable_bytes, GIB / 2);
        assert_eq!(collector.summary().live_owner_count, 1);
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
    }

    #[test]
    fn unavailable_receipts_are_a_no_op_and_never_block_admission() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let unavailable = RuntimeReceiptCollector::new_with_entropy_failure_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
        );
        let available = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            8,
        )
        .unwrap();
        let owner = available
            .owner_descriptor("unavailable", None, None, None)
            .unwrap();
        let mut lease = broker
            .try_reserve_batch(vec![request(
                domain(),
                7 * GIB,
                1,
                1,
                "receipt-unavailable",
            )])
            .unwrap();
        lease.attach_receipt(unavailable.clone(), owner, Vec::new());
        assert_eq!(unavailable.summary().live_owner_count, 0);
        assert_eq!(broker.usage(&domain()).pending_bytes, 1);
        drop(lease);
        assert_eq!(broker.usage(&domain()), DeviceMemoryUsage::default());
    }

    #[test]
    fn committed_lease_receipts_cover_the_broker_ledger_including_incident_totals() {
        const INCIDENT_COMMITTED: u64 = 11_488_973_972;
        const INCIDENT_REQUESTED: u64 = 5_092_073_216;
        const INCIDENT_TOTAL: u64 = 15_406_611_046;
        const INCIDENT_OBSERVED_FREE: u64 = 4_461_342_720;
        const INCIDENT_POLICY_REMAINDER: u64 = 3_917_637_074;

        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let collector = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            64,
        )
        .unwrap();
        let system = MemoryDomainKey::SystemMemory;
        let scope_id = collector.snapshot().scope_id;
        let lane = collector
            .lane_projection(
                crate::device::execution_route::ExecutionProvider::Cpu,
                "cpu0",
                crate::device::execution_policy::ExecutionPlacement::CpuOnly,
                crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            )
            .unwrap();
        let mut lease = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![DomainReservationRequest {
                    domain: system.clone(),
                    snapshot: DeviceMemorySnapshot {
                        free_bytes: INCIDENT_TOTAL,
                        total_bytes: INCIDENT_TOTAL,
                        confidence: MemoryObservationConfidence::DeviceSnapshot,
                    },
                    peak_bytes: INCIDENT_COMMITTED,
                    retained_bytes: INCIDENT_COMMITTED,
                    observed_peak_bytes: None,
                    requires_reconciliation: false,
                    resource_id: "incident-committed-owners".to_string(),
                    cohort_id: None,
                }],
                Some(scope_id),
                RuntimeOwnerPlacement::LaneBound(lane),
            )
            .unwrap();
        let owner = collector
            .owner_descriptor(
                "firered-llm-encoder",
                Some("sha256:incident-pack"),
                Some("pack-weight-buffer"),
                Some(lane),
            )
            .unwrap();
        let resource = collector
            .resource_descriptor(
                "pack-weight-buffer",
                &system,
                INCIDENT_COMMITTED,
                INCIDENT_COMMITTED,
                INCIDENT_COMMITTED,
                QuoteConfidence::CommittedUpperBound,
                Some(MemoryObservationConfidence::DeviceSnapshot),
            )
            .unwrap();
        lease.attach_receipt(collector.clone(), owner, vec![(system.clone(), resource)]);
        lease.commit_quoted().unwrap();

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.live_owners.len(), 1);
        let live = &snapshot.live_owners[0];
        assert!(live.descriptor.content.is_some());
        assert!(matches!(
            live.descriptor.placement,
            crate::models::runtime_receipts::RuntimeOwnerPlacement::LaneBound(_)
        ));
        let resource = live.resources.values().next().unwrap();
        assert_eq!(resource.state, RuntimeResourceState::Committed);
        assert_eq!(
            resource.descriptor.retained,
            RuntimeReceiptMetric::Known(INCIDENT_COMMITTED)
        );
        assert_eq!(broker.usage(&system).committed_bytes, INCIDENT_COMMITTED);
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );

        let rejected = broker.try_reserve_batch(vec![DomainReservationRequest {
            domain: system.clone(),
            snapshot: DeviceMemorySnapshot {
                free_bytes: INCIDENT_OBSERVED_FREE,
                total_bytes: INCIDENT_TOTAL,
                confidence: MemoryObservationConfidence::DeviceSnapshot,
            },
            peak_bytes: INCIDENT_REQUESTED,
            retained_bytes: INCIDENT_REQUESTED,
            observed_peak_bytes: None,
            requires_reconciliation: false,
            resource_id: "pack-weight-buffer-chunk-0".to_string(),
            cohort_id: None,
        }]);
        match rejected {
            Err(MemoryPlanningError::DeviceBudgetExceeded {
                requested_bytes,
                committed_bytes,
                available_bytes,
                ..
            }) => {
                assert_eq!(requested_bytes, INCIDENT_REQUESTED);
                assert_eq!(committed_bytes, INCIDENT_COMMITTED);
                assert_eq!(available_bytes, INCIDENT_POLICY_REMAINDER);
                const {
                    assert!(INCIDENT_REQUESTED > INCIDENT_POLICY_REMAINDER);
                    assert!(INCIDENT_REQUESTED > INCIDENT_OBSERVED_FREE);
                }
            }
            other => panic!("incident arithmetic must fail closed, got {other:?}"),
        }
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
        drop(lease);
        assert_eq!(
            collector.reconcile_live_leases(&broker),
            crate::models::runtime_receipts::LeaseReceiptShadow::Matched
        );
    }

    #[test]
    fn scoped_placement_ledger_tracks_commit_refund_and_quarantine() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let collector = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            32,
        )
        .unwrap();
        let scope_id = collector.snapshot().scope_id;
        let lane = collector
            .lane_projection(
                crate::device::execution_route::ExecutionProvider::Cuda,
                "cuda:0",
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
                crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            )
            .unwrap();
        let placement = RuntimeOwnerPlacement::LaneBound(lane);
        let key = (MemoryDomainKey::SystemMemory, placement);

        let mut committed = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![request(
                    MemoryDomainKey::SystemMemory,
                    8 * GIB,
                    64,
                    32,
                    "placement-commit",
                )],
                Some(scope_id),
                placement,
            )
            .unwrap();
        assert_eq!(
            broker
                .ledger_snapshot_for_scope_by_placement(scope_id)
                .get(&key)
                .unwrap()
                .pending_bytes,
            64
        );
        committed.commit_quoted().unwrap();
        let usage = broker.ledger_snapshot_for_scope_by_placement(scope_id)[&key];
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.committed_bytes, 32);
        drop(committed);
        let usage = broker.ledger_snapshot_for_scope_by_placement(scope_id)[&key];
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.committed_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, 0);

        let mut quarantined = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![request(
                    MemoryDomainKey::SystemMemory,
                    8 * GIB,
                    64,
                    32,
                    "placement-quarantine",
                )],
                Some(scope_id),
                placement,
            )
            .unwrap();
        quarantined.commit_quoted().unwrap();
        quarantined.quarantine();
        let usage = broker.ledger_snapshot_for_scope_by_placement(scope_id)[&key];
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.committed_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, 32);
        drop(quarantined);
        assert_eq!(
            broker.ledger_snapshot_for_scope_by_placement(scope_id)[&key].unreclaimable_bytes,
            32
        );
    }

    #[test]
    fn reservation_diagnostics_never_borrow_another_scope_or_lanes_bytes() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let first = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            8,
        )
        .unwrap();
        let second = RuntimeReceiptCollector::new_for_test(
            crate::models::native_execution_services::NativeExecutionScopeId::next(),
            8,
        )
        .unwrap();
        let first_lane = first
            .lane_projection(
                crate::device::execution_route::ExecutionProvider::Cuda,
                "cuda:0",
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
                crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            )
            .unwrap();
        let second_lane = second
            .lane_projection(
                crate::device::execution_route::ExecutionProvider::Vulkan,
                "vulkan:0",
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
                crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            )
            .unwrap();
        let domain = MemoryDomainKey::SystemMemory;
        let mut first_lease = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![request(domain.clone(), 8 * GIB, 10, 10, "first")],
                Some(first.snapshot().scope_id),
                RuntimeOwnerPlacement::LaneBound(first_lane),
            )
            .unwrap();
        let mut second_lease = broker
            .try_reserve_batch_for_scope_and_placement(
                vec![request(domain.clone(), 8 * GIB, 20, 20, "second")],
                Some(second.snapshot().scope_id),
                RuntimeOwnerPlacement::LaneBound(second_lane),
            )
            .unwrap();
        first_lease.commit_quoted().unwrap();
        second_lease.commit_quoted().unwrap();

        assert_eq!(broker.usage(&domain).committed_bytes, 30);
        assert_eq!(first_lease.domain_usage(&domain).committed_bytes, 10);
        assert_eq!(second_lease.domain_usage(&domain).committed_bytes, 20);
    }

    #[test]
    fn impossible_free_snapshot_is_clamped_to_total() {
        let normalized = DeviceMemorySnapshot {
            free_bytes: u64::MAX,
            total_bytes: 8 * GIB,
            confidence: MemoryObservationConfidence::WorkingSetBudget,
        }
        .normalized()
        .unwrap();
        assert_eq!(normalized.free_bytes, normalized.total_bytes);
    }
}
