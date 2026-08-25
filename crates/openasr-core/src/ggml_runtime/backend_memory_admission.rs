//! Translation between ggml's native physical-memory ABI and OpenASR's
//! process-wide memory broker.
//!
//! This module is intentionally policy-free. It never reads an execution
//! route. Native domain identifiers and caller-supplied stable backend-device
//! identities are the only inputs used to establish physical accounting keys.

#![allow(dead_code)]

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    fmt::{self, Write as _},
    mem,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::models::runtime_receipts::{
    RuntimeBackendOwnedReliability, RuntimeNativeMemoryEvidence, RuntimeOwnerPlacement,
    RuntimeReceiptMetric,
};

use crate::device::execution_memory::{
    AllocationFootprint, AllocationLifetime, DeviceMemoryBrokerSet, DeviceMemoryReservationBatch,
    DeviceMemorySnapshot, DomainMemoryReconciliation, DomainReservationRequest, MemoryClaim,
    MemoryDomainKey, MemoryObservationConfidence, MemoryPlanningError, MemoryReservationCohortId,
    PhaseSet, PhysicalDeviceKey, QuoteConfidence,
};

use crate::models::native_execution_services::{
    current_execution_lane_key, current_native_execution_scope_id, current_runtime_receipts,
};

use super::{
    backend_memory::{
        BackendFailureDisposition, BackendMemoryAbi, BackendMemoryAbiError,
        BackendMemoryDomainKind, BackendMemoryLifecyclePoint, BackendMemoryQuote,
        BackendMemoryStatsSnapshot, BackendReleaseProof, BackendTerminalEvidence,
        BackendTerminalIdentity, BackendTerminalOutcome, BackendTerminalStage,
        backend_owned_unknown_reason,
    },
    ffi,
};

/// Phase/lifetime facts that ggml cannot know. The caller must bind these to
/// request IDs from the frozen scheduler plan; this adapter does not infer
/// them from a model family or execution route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeMemoryClaimSemantics {
    pub(crate) resource_id: String,
    pub(crate) lifetime: AllocationLifetime,
    pub(crate) phases: PhaseSet,
}

/// One backend-specific quote plus the live statistics fetched immediately
/// after it. The quote token is retained so the allocation layer can later
/// perform the backend-private transactional reserve against the same quote.
pub(crate) struct NativeQuotedBackendGroup {
    group_id: String,
    backend_device_identity: PhysicalDeviceKey,
    abi: BackendMemoryAbi,
    provider: crate::device::execution_route::ExecutionProvider,
    requests: Vec<ffi::GgmlBackendMemoryRequestV1>,
    quote: BackendMemoryQuote,
    fresh_stats: BackendMemoryStatsSnapshot,
    request_semantics: BTreeMap<u64, NativeMemoryClaimSemantics>,
    shared_semantics: NativeMemoryClaimSemantics,
}

impl NativeQuotedBackendGroup {
    /// Quotes one concrete backend and then obtains a fresh stats snapshot.
    /// No domain-enumeration snapshot is used for admission.
    pub(crate) fn quote(
        group_id: impl Into<String>,
        backend_device_identity: PhysicalDeviceKey,
        abi: BackendMemoryAbi,
        requests: Vec<ffi::GgmlBackendMemoryRequestV1>,
        request_semantics: BTreeMap<u64, NativeMemoryClaimSemantics>,
        shared_semantics: NativeMemoryClaimSemantics,
    ) -> Result<Self, NativeMemoryAdmissionError> {
        let group_id = group_id.into();
        if group_id.trim().is_empty() {
            return Err(NativeMemoryAdmissionError::EmptyGroupId);
        }
        let quote = abi.quote(&requests)?;
        let provider = abi.provider();
        let fresh_stats = abi.stats_at(BackendMemoryLifecyclePoint::AdmissionQuote)?;
        Ok(Self {
            group_id,
            backend_device_identity,
            abi,
            provider,
            requests,
            quote,
            fresh_stats,
            request_semantics,
            shared_semantics,
        })
    }

    pub(crate) fn group_id(&self) -> &str {
        &self.group_id
    }

    pub(crate) fn quote_value(&self) -> &BackendMemoryQuote {
        &self.quote
    }

    /// Allocates backend-private resources against the original quote token.
    /// The caller must already own the broker's pending reservation.
    ///
    /// Returned actual claims are diagnostic, not a new admission authority:
    /// exact/upper quotes are already bound to the validated token, while
    /// provisional commitment is established from live post-reserve stats.
    /// Consumers must therefore not relabel requested/actual claim bytes as
    /// committed physical memory merely because this call succeeded.
    pub(crate) fn reserve_private(
        &self,
    ) -> Result<Vec<ffi::GgmlBackendMemoryClaimV1>, BackendMemoryAbiError> {
        self.abi.reserve_private(&self.requests, &self.quote)
    }

    /// Re-quotes the same still-live backend requests after another
    /// provisional candidate releases the physical-domain gate. Quote tokens
    /// and stats generations are deliberately refreshed together; retrying a
    /// broker admission with the old snapshot would make the observed-capacity
    /// check optimistic after the other candidate commits.
    fn refresh(self) -> Result<Self, NativeMemoryAdmissionError> {
        Self::quote(
            self.group_id,
            self.backend_device_identity,
            self.abi,
            self.requests,
            self.request_semantics,
            self.shared_semantics,
        )
    }
}

/// Raw native evidence retained for diagnostics. `MemoryClaim::confidence`
/// carries the admission meaning, while this record preserves every native
/// flag and quote-level residual bit without lossy reinterpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeMemoryQuoteEvidence {
    pub(crate) group_id: String,
    pub(crate) quote_flags: u32,
    pub(crate) residual_flags: u32,
    pub(crate) residual_request_count: u32,
    pub(crate) provisional_requested_upper_bytes: u64,
    pub(crate) claim_flags: Vec<(u64, u32)>,
}

/// A complete candidate plan. All backend groups have already been joined to
/// fresh live statistics, but no broker state has been mutated yet.
pub(crate) struct NativeMemoryAdmissionPlan {
    groups: Vec<NativeQuotedBackendGroup>,
    claims: Vec<MemoryClaim>,
    requests: Vec<DomainReservationRequest>,
    evidence: Vec<NativeMemoryQuoteEvidence>,
    reconciliation_baseline: ReconciliationBaseline,
}

fn runtime_receipt_backend(
    provider: crate::device::execution_route::ExecutionProvider,
) -> crate::ggml_runtime::GgmlCpuGraphBackend {
    match provider {
        crate::device::execution_route::ExecutionProvider::Cpu => {
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu
        }
        crate::device::execution_route::ExecutionProvider::Metal => {
            crate::ggml_runtime::GgmlCpuGraphBackend::Metal
        }
        crate::device::execution_route::ExecutionProvider::Cuda
        | crate::device::execution_route::ExecutionProvider::Hip
        | crate::device::execution_route::ExecutionProvider::Vulkan
        | crate::device::execution_route::ExecutionProvider::Accelerator
        | crate::device::execution_route::ExecutionProvider::Unknown => {
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu
        }
    }
}

fn runtime_owner_placement(groups: &[NativeQuotedBackendGroup]) -> RuntimeOwnerPlacement {
    let Some(collector) = current_runtime_receipts().filter(|collector| collector.is_available())
    else {
        return RuntimeOwnerPlacement::Unknown;
    };
    let Some(first_group) = groups.first() else {
        return RuntimeOwnerPlacement::Unknown;
    };
    current_execution_lane_key(runtime_receipt_backend(first_group.provider))
        .receipt_projection(&collector)
        .map_or(
            RuntimeOwnerPlacement::Unknown,
            RuntimeOwnerPlacement::LaneBound,
        )
}

impl NativeMemoryAdmissionPlan {
    const DOMAIN_BUSY_WAIT: Duration = Duration::from_secs(30);
    const DOMAIN_BUSY_INITIAL_BACKOFF: Duration = Duration::from_millis(1);
    const DOMAIN_BUSY_MAX_BACKOFF: Duration = Duration::from_millis(32);

    pub(crate) fn from_groups(
        groups: Vec<NativeQuotedBackendGroup>,
    ) -> Result<Self, NativeMemoryAdmissionError> {
        let views = groups.iter().map(NativeGroupView::from).collect::<Vec<_>>();
        let built = build_from_views(&views)?;
        Ok(Self {
            groups,
            claims: built.claims,
            requests: built.requests,
            evidence: built.evidence,
            reconciliation_baseline: built.reconciliation_baseline,
        })
    }

    pub(crate) fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub(crate) fn reservation_requests(&self) -> &[DomainReservationRequest] {
        &self.requests
    }

    pub(crate) fn evidence(&self) -> &[NativeMemoryQuoteEvidence] {
        &self.evidence
    }

    pub(crate) fn quote_confidence_for_domain(&self, domain: &MemoryDomainKey) -> QuoteConfidence {
        quote_confidence_for_domain(&self.claims, domain)
    }

    /// The single quote-to-broker admission edge. Every native backend/group
    /// has already been merged by physical domain before this one atomic call.
    pub(crate) fn try_reserve(
        mut self,
        broker: &Arc<DeviceMemoryBrokerSet>,
        cohort_id: Option<MemoryReservationCohortId>,
    ) -> Result<NativeMemoryAllocationTransaction, NativeMemoryAdmissionError> {
        let deadline = Instant::now() + Self::DOMAIN_BUSY_WAIT;
        let mut retry_delay = Self::DOMAIN_BUSY_INITIAL_BACKOFF;
        loop {
            classify_request_kinds(self.groups.iter().flat_map(|group| group.requests.iter()))?;
            self.require_opaque_driver_headroom(broker)?;
            for request in &mut self.requests {
                request.cohort_id = cohort_id;
            }
            let owner_scope_id = current_native_execution_scope_id();
            let owner_placement = runtime_owner_placement(&self.groups);
            match broker.try_reserve_batch_for_scope_and_placement(
                self.requests.clone(),
                owner_scope_id,
                owner_placement,
            ) {
                Ok(reservation) => {
                    let mut transaction = NativeMemoryAllocationTransaction {
                        groups: self.groups,
                        claims: self.claims,
                        requests: self.requests,
                        evidence: self.evidence,
                        reconciliation_baseline: self.reconciliation_baseline,
                        reservation,
                    };
                    transaction.attach_runtime_receipt();
                    return Ok(transaction);
                }
                Err(error @ MemoryPlanningError::DeviceDomainBusy { .. }) => {
                    if !Self::wait_for_domain_busy_retry(deadline, &mut retry_delay)? {
                        return Err(error.into());
                    }
                    self = self.refresh()?;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Reserves several ownership partitions as one candidate-level atomic
    /// operation. Native request classes stay separate in each returned child
    /// transaction, while the broker admits the sum across shared physical
    /// domains under one lock.
    pub(crate) fn try_reserve_partitioned(
        mut plans: Vec<Self>,
        broker: &Arc<DeviceMemoryBrokerSet>,
        cohort_id: Option<MemoryReservationCohortId>,
    ) -> Result<Vec<NativeMemoryAllocationTransaction>, NativeMemoryAdmissionError> {
        let deadline = Instant::now() + Self::DOMAIN_BUSY_WAIT;
        let mut retry_delay = Self::DOMAIN_BUSY_INITIAL_BACKOFF;
        loop {
            for plan in &plans {
                classify_request_kinds(plan.groups.iter().flat_map(|group| group.requests.iter()))?;
                plan.require_opaque_driver_headroom(broker)?;
            }
            for request in plans.iter_mut().flat_map(|plan| &mut plan.requests) {
                request.cohort_id = cohort_id;
            }
            let owner_scope_id = current_native_execution_scope_id();
            let owner_placements = plans
                .iter()
                .map(|plan| runtime_owner_placement(&plan.groups))
                .collect();
            let reservations = match broker.try_reserve_partitioned_for_scope_and_placements(
                plans.iter().map(|plan| plan.requests.clone()).collect(),
                owner_scope_id,
                owner_placements,
            ) {
                Ok(reservations) => reservations,
                Err(error @ MemoryPlanningError::DeviceDomainBusy { .. }) => {
                    if !Self::wait_for_domain_busy_retry(deadline, &mut retry_delay)? {
                        return Err(error.into());
                    }
                    plans = plans
                        .into_iter()
                        .map(Self::refresh)
                        .collect::<Result<Vec<_>, _>>()?;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            if reservations.len() != plans.len() {
                return Err(
                    NativeMemoryAdmissionError::PartitionedReservationSetMismatch {
                        expected: plans.len(),
                        actual: reservations.len(),
                    },
                );
            }
            return Ok(plans
                .into_iter()
                .zip(reservations)
                .map(|(plan, reservation)| {
                    let mut transaction = NativeMemoryAllocationTransaction {
                        groups: plan.groups,
                        claims: plan.claims,
                        requests: plan.requests,
                        evidence: plan.evidence,
                        reconciliation_baseline: plan.reconciliation_baseline,
                        reservation,
                    };
                    transaction.attach_runtime_receipt();
                    transaction
                })
                .collect());
        }
    }

    fn refresh(self) -> Result<Self, NativeMemoryAdmissionError> {
        let groups = self
            .groups
            .into_iter()
            .map(NativeQuotedBackendGroup::refresh)
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_groups(groups)
    }

    fn wait_for_domain_busy_retry(
        deadline: Instant,
        retry_delay: &mut Duration,
    ) -> Result<bool, NativeMemoryAdmissionError> {
        if super::thread_job_cancel_flag()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(NativeMemoryAdmissionError::CanceledWhileWaitingForDomain);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(*retry_delay);
        *retry_delay = (*retry_delay * 2).min(Self::DOMAIN_BUSY_MAX_BACKOFF);
        if super::thread_job_cancel_flag()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        {
            return Err(NativeMemoryAdmissionError::CanceledWhileWaitingForDomain);
        }
        Ok(true)
    }

    fn require_opaque_driver_headroom(
        &self,
        broker: &DeviceMemoryBrokerSet,
    ) -> Result<(), NativeMemoryAdmissionError> {
        if broker.minimum_headroom_bytes() != 0 {
            return Ok(());
        }
        if let Some(evidence) = self.evidence.iter().find(|evidence| {
            evidence.quote_flags
                & ffi::GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM
                != 0
        }) {
            return Err(
                NativeMemoryAdmissionError::OpaqueDriverHeadroomUnavailable {
                    group_id: evidence.group_id.clone(),
                },
            );
        }
        Ok(())
    }
}

fn quote_confidence_for_domain(
    claims: &[MemoryClaim],
    domain: &MemoryDomainKey,
) -> QuoteConfidence {
    let mut confidence = None;
    for claim in claims {
        if &claim.domain != domain {
            continue;
        }
        confidence = Some(match (confidence, claim.confidence) {
            (Some(QuoteConfidence::Provisional), _) | (_, QuoteConfidence::Provisional) => {
                QuoteConfidence::Provisional
            }
            (Some(QuoteConfidence::Unknown), _) | (_, QuoteConfidence::Unknown) => {
                QuoteConfidence::Unknown
            }
            (Some(QuoteConfidence::CommittedUpperBound), _)
            | (_, QuoteConfidence::CommittedUpperBound) => QuoteConfidence::CommittedUpperBound,
            _ => QuoteConfidence::ExactCommitted,
        });
    }
    confidence.unwrap_or(QuoteConfidence::Provisional)
}

/// Pending broker ownership plus the native quote tokens and pre-allocation
/// observations required by the allocation transaction.
pub(crate) struct NativeMemoryAllocationTransaction {
    groups: Vec<NativeQuotedBackendGroup>,
    claims: Vec<MemoryClaim>,
    requests: Vec<DomainReservationRequest>,
    evidence: Vec<NativeMemoryQuoteEvidence>,
    reconciliation_baseline: ReconciliationBaseline,
    reservation: DeviceMemoryReservationBatch,
}

/// Typed failure Interface consumed by the owner-attached transaction. The
/// failure producer owns evidence interpretation; the transaction alone owns
/// the broker state transition, so callers cannot classify one error twice.
pub(crate) trait NativeOwnerAttachedCommitOutcome {
    fn requires_quarantine(&self) -> bool;
}

/// Engine-owned native resources must expose status-aware destruction. The
/// broker may refund their lease only after this owner proves that every
/// native release callback completed.
pub(crate) trait NativeMemoryOwner {
    fn release_native(&mut self) -> BackendReleaseProof;
}

/// Typed native commit failure consumed by the single broker transaction.
/// `may_have_mutated` is conservative: once a provider callback or allocator
/// binding phase was entered, the caller must set it even when the provider
/// returned an ordinary OOM status.
pub(crate) struct NativeEngineCommitFailure<E> {
    source: E,
    status: i32,
    may_have_mutated: bool,
    release_must_remain_unproven: bool,
}

impl<E> NativeEngineCommitFailure<E> {
    pub(crate) fn new(source: E, status: i32, may_have_mutated: bool) -> Self {
        Self {
            source,
            status,
            may_have_mutated,
            release_must_remain_unproven: false,
        }
    }

    pub(crate) fn with_unproven_release(mut self) -> Self {
        self.release_must_remain_unproven = true;
        self
    }

    pub(crate) fn into_source(self) -> E {
        self.source
    }
}

struct NativeMemoryReleaseAudit {
    backends: Vec<BackendMemoryAbi>,
    identity: BackendTerminalIdentity,
}

impl NativeMemoryReleaseAudit {
    fn outcome(
        &self,
        stage: BackendTerminalStage,
        status: i32,
        may_have_mutated: bool,
        release_proof: BackendReleaseProof,
    ) -> BackendTerminalOutcome {
        BackendTerminalOutcome::backend_operation(
            stage,
            status,
            may_have_mutated,
            self.identity.clone(),
            BackendTerminalEvidence::capture(&self.backends, 0),
        )
        .with_release_proof(release_proof)
    }

    fn release_outcome<T: NativeMemoryOwner>(
        &self,
        owner: &mut T,
        stage: BackendTerminalStage,
        status: i32,
        may_have_mutated: bool,
    ) -> BackendTerminalOutcome {
        let release_proof = owner.release_native();
        self.outcome(stage, status, may_have_mutated, release_proof)
    }
}

fn owner_attached_native_commit_error<E>(
    reservation: &mut DeviceMemoryReservationBatch,
    failure: E,
) -> NativeOwnerAttachedMemoryError<E>
where
    E: NativeOwnerAttachedCommitOutcome,
{
    let quarantined = failure.requires_quarantine();
    if quarantined {
        reservation.quarantine();
    }
    NativeOwnerAttachedMemoryError::NativeCommit {
        source: failure,
        quarantined,
    }
}

impl NativeMemoryAllocationTransaction {
    pub(crate) fn groups(&self) -> &[NativeQuotedBackendGroup] {
        &self.groups
    }

    pub(crate) fn claims(&self) -> &[MemoryClaim] {
        &self.claims
    }

    pub(crate) fn reservation_requests(&self) -> &[DomainReservationRequest] {
        &self.requests
    }

    fn terminal_evidence(&self) -> BackendTerminalEvidence {
        let mut seen = HashSet::new();
        let backends = self
            .groups
            .iter()
            .filter_map(|group| {
                seen.insert(group.abi.backend() as usize)
                    .then_some(group.abi)
            })
            .collect::<Vec<_>>();
        BackendTerminalEvidence::capture(&backends, 0)
    }

    fn private_reserve_outcome(
        &self,
        identity: BackendTerminalIdentity,
        source: &BackendMemoryAbiError,
        may_have_mutated: bool,
    ) -> BackendTerminalOutcome {
        BackendTerminalOutcome::backend_operation(
            BackendTerminalStage::BackendPrivateReserve,
            source.terminal_status(),
            may_have_mutated,
            identity,
            self.terminal_evidence(),
        )
    }

    fn private_reserve_identity(&self, group_index: usize) -> BackendTerminalIdentity {
        let group = &self.groups[group_index];
        let mut domains = self
            .requests
            .iter()
            .map(|request| request.domain.clone())
            .collect::<Vec<_>>();
        domains.sort();
        domains.dedup();
        BackendTerminalIdentity::exact(
            group.provider,
            group.backend_device_identity.as_str().to_owned(),
            domains,
        )
    }

    fn standalone_release_audit(
        &self,
    ) -> Result<NativeMemoryReleaseAudit, NativeMemoryRequestKindError> {
        if self.groups.len() != 1 {
            return Err(NativeMemoryRequestKindError::StandaloneOwnerSpansBackends {
                backend_count: self.groups.len(),
            });
        }
        Ok(NativeMemoryReleaseAudit {
            backends: vec![self.groups[0].abi],
            identity: self.private_reserve_identity(0),
        })
    }

    fn attach_runtime_receipt(&mut self) {
        let Some(collector) =
            current_runtime_receipts().filter(|collector| collector.is_available())
        else {
            return;
        };
        let Some(first_group) = self.groups.first() else {
            return;
        };
        let backend = runtime_receipt_backend(first_group.provider);
        let lane = current_execution_lane_key(backend).receipt_projection(&collector);
        let Some(owner_descriptor) = collector.owner_descriptor(
            "native-memory-owner",
            None,
            Some(first_group.group_id()),
            lane,
        ) else {
            return;
        };
        let resources = self.runtime_receipt_descriptors(&collector);
        self.reservation
            .attach_receipt(collector, owner_descriptor, resources);
    }

    fn runtime_receipt_descriptors(
        &self,
        collector: &crate::models::runtime_receipts::RuntimeReceiptCollector,
    ) -> Vec<(
        MemoryDomainKey,
        crate::models::runtime_receipts::RuntimeResourceDescriptor,
    )> {
        self.requests
            .iter()
            .filter_map(|request| {
                let descriptor = collector.resource_descriptor(
                    "native-memory-domain",
                    &request.domain,
                    request.peak_bytes,
                    request.peak_bytes,
                    request.retained_bytes,
                    quote_confidence_for_domain(&self.claims, &request.domain),
                    Some(request.snapshot.confidence),
                )?;
                let mut native = self
                    .reconciliation_baseline
                    .observations
                    .get(&request.domain)
                    .map(runtime_native_evidence)?;
                let usage = self.reservation.domain_usage(&request.domain);
                native.broker_pending_bytes = RuntimeReceiptMetric::Known(usage.pending_bytes);
                native.broker_committed_bytes = RuntimeReceiptMetric::Known(usage.committed_bytes);
                native.broker_unreclaimable_bytes =
                    RuntimeReceiptMetric::Known(usage.unreclaimable_bytes);
                Some((
                    request.domain.clone(),
                    crate::models::runtime_receipts::RuntimeReceiptCollector::with_native_evidence(
                        descriptor, native,
                    ),
                ))
            })
            .collect()
    }

    pub(crate) fn reservation(&self) -> &DeviceMemoryReservationBatch {
        &self.reservation
    }

    pub(crate) fn reservation_mut(&mut self) -> &mut DeviceMemoryReservationBatch {
        &mut self.reservation
    }

    /// Detaches the broker batch so a candidate-activation reservation can
    /// hold known-domain peak/retained without immediately allocating native
    /// owners. Quote tokens are dropped; later declared components still JIT
    /// against this pending batch's cohort.
    pub(crate) fn into_reservation(self) -> DeviceMemoryReservationBatch {
        self.reservation
    }

    pub(crate) fn requires_reconciliation(&self) -> bool {
        self.reservation.requires_reconciliation()
    }

    /// Replaces stale native quote tokens/evidence after another child in the
    /// same atomically-admitted candidate changed backend generation. The fresh
    /// physical requirements must fit entirely inside this child's original
    /// reservation; this method can never expand candidate admission.
    pub(crate) fn rebind_fresh_plan(
        mut self,
        fresh: NativeMemoryAdmissionPlan,
    ) -> Result<Self, NativeMemoryAdmissionError> {
        let expected =
            classify_request_kinds(self.groups.iter().flat_map(|group| group.requests.iter()))?;
        let actual =
            classify_request_kinds(fresh.groups.iter().flat_map(|group| group.requests.iter()))?;
        if actual != expected {
            return Err(NativeMemoryRequestKindError::WrongTransaction { expected, actual }.into());
        }
        self.reservation.rebind_quote(&fresh.requests)?;
        self.groups = fresh.groups;
        self.claims = fresh.claims;
        self.requests = fresh.requests;
        self.evidence = fresh.evidence;
        self.reconciliation_baseline = fresh.reconciliation_baseline;
        if let Some(collector) =
            current_runtime_receipts().filter(|collector| collector.is_available())
        {
            self.reservation
                .update_receipt_descriptors(self.runtime_receipt_descriptors(&collector));
        }
        Ok(self)
    }

    /// Moves a deferred private transaction's live-stat baseline. When
    /// `accumulate_private_growth` is true, growth since the prior baseline is
    /// retained as this transaction's own commitment. When false, the growth
    /// belongs to an atomically-admitted sibling owner (the scheduler arena)
    /// and is deliberately excluded before first-compute reconciliation.
    fn rebase_live_observations(
        &mut self,
        accumulate_private_growth: bool,
    ) -> Result<(), NativeMemoryAdmissionError> {
        let observations = fetch_live_observations(&self.groups)?;
        for request in &self.requests {
            let before = self
                .reconciliation_baseline
                .observations
                .get(&request.domain)
                .ok_or_else(|| NativeMemoryAdmissionError::MissingReconciliationStats {
                    domain: request.domain.clone(),
                })?;
            let after = observations.get(&request.domain).ok_or_else(|| {
                NativeMemoryAdmissionError::MissingReconciliationStats {
                    domain: request.domain.clone(),
                }
            })?;
            if accumulate_private_growth {
                let delta = observation_growth(before, after);
                let carried = self
                    .reconciliation_baseline
                    .carried_private_bytes
                    .entry(request.domain.clone())
                    .or_default();
                *carried = carried.checked_add(delta).ok_or(
                    NativeMemoryAdmissionError::ArithmeticOverflow {
                        operation: "deferred private reconciliation carry",
                    },
                )?;
            }
        }
        self.reconciliation_baseline.observations = observations;
        Ok(())
    }

    /// Fetches live post-allocation stats from every original backend group
    /// and produces one reconciliation row per merged broker domain.
    pub(crate) fn build_post_allocation_reconciliations(
        &self,
    ) -> Result<Vec<DomainMemoryReconciliation>, NativeMemoryAdmissionError> {
        build_live_reconciliations(&self.groups, &self.reconciliation_baseline, &self.requests)
    }

    /// Executes an engine-owned allocation transaction in the only valid
    /// order:
    ///
    /// 1. validate every quote token through `reserve_private` (these request
    ///    groups contain no GRAPH_PRIVATE item, so this cannot grow a backend
    ///    high-water allocation);
    /// 2. let the caller commit engine-controlled native allocations;
    /// 3. fetch live post-allocation stats and reconcile/commit the broker;
    /// 4. return one owner wrapper that drops native state before its lease.
    ///
    /// The backends referenced by the quote groups must outlive this call and
    /// the returned `T`. A closure failure must classify whether entering the
    /// native callback may have changed provider state; no untyped error may
    /// implicitly refund the reservation.
    pub(crate) fn commit_engine_owned_with<T, E, F>(
        mut self,
        native_commit: F,
    ) -> Result<NativeMemoryAllocation<T>, NativeMemoryAllocationError<E>>
    where
        T: NativeMemoryOwner,
        F: FnOnce() -> Result<T, NativeEngineCommitFailure<E>>,
    {
        self.require_request_class(NativeRequestClass::EngineOwned)
            .map_err(NativeMemoryAllocationError::RequestKinds)?;
        let release_audit = self
            .standalone_release_audit()
            .map_err(NativeMemoryAllocationError::RequestKinds)?;
        for index in 0..self.groups.len() {
            if let Err(source) = self.groups[index].reserve_private() {
                let outcome = self.private_reserve_outcome(
                    self.private_reserve_identity(index),
                    &source,
                    source.may_have_committed_private_state(),
                );
                let quarantined = outcome.disposition() == BackendFailureDisposition::Quarantine;
                if quarantined {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::PrivateReserve {
                    group_id: self.groups[index].group_id.clone(),
                    source,
                    outcome,
                });
            }
        }

        let mut owner = match native_commit() {
            Ok(owner) => owner,
            Err(failure) => {
                let outcome = release_audit.outcome(
                    BackendTerminalStage::EngineOwnedCommit,
                    failure.status,
                    failure.may_have_mutated,
                    BackendReleaseProof::Unproven,
                );
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::NativeCommit {
                    source: failure.source,
                    outcome,
                });
            }
        };

        let groups = &self.groups;
        let baseline = &self.reconciliation_baseline;
        let requests = &self.requests;
        match finalize_reservation_with(&mut self.reservation, || {
            build_live_reconciliations(groups, baseline, requests)
        }) {
            Ok(()) => {}
            Err(ReservationFinalizeError::Evidence(source)) => {
                let outcome = release_audit.release_outcome(
                    &mut owner,
                    BackendTerminalStage::EngineOwnedReconcile,
                    ffi::GGML_STATUS_FAILED,
                    true,
                );
                drop(owner);
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::PostAllocationStats { source, outcome });
            }
            Err(ReservationFinalizeError::Broker(source)) => {
                let outcome = release_audit.release_outcome(
                    &mut owner,
                    BackendTerminalStage::EngineOwnedReconcile,
                    ffi::GGML_STATUS_FAILED,
                    true,
                );
                drop(owner);
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::BrokerCommit { source, outcome });
            }
        }

        Ok(NativeMemoryAllocation {
            owner: Some(owner),
            reservation: Some(self.reservation),
        })
    }

    /// Commits a native owner that already exists before its fallible commit
    /// phase (the direct ggml graph allocator). Ownership moves into this
    /// transaction before any provider mutation, so every failure can destroy
    /// the owner synchronously and combine release proof with fresh exact-
    /// backend health evidence before deciding refund versus quarantine.
    pub(crate) fn commit_prepared_engine_owner_with<T, E, F>(
        mut self,
        mut owner: T,
        native_commit: F,
    ) -> Result<NativeMemoryAllocation<T>, NativeMemoryAllocationError<E>>
    where
        T: NativeMemoryOwner,
        F: FnOnce(&mut T) -> Result<(), NativeEngineCommitFailure<E>>,
    {
        self.require_request_class(NativeRequestClass::EngineOwned)
            .map_err(NativeMemoryAllocationError::RequestKinds)?;
        let release_audit = self
            .standalone_release_audit()
            .map_err(NativeMemoryAllocationError::RequestKinds)?;
        for index in 0..self.groups.len() {
            if let Err(source) = self.groups[index].reserve_private() {
                let outcome = release_audit.release_outcome(
                    &mut owner,
                    BackendTerminalStage::BackendPrivateReserve,
                    source.terminal_status(),
                    source.may_have_committed_private_state(),
                );
                drop(owner);
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::PrivateReserve {
                    group_id: self.groups[index].group_id.clone(),
                    source,
                    outcome,
                });
            }
        }

        if let Err(failure) = native_commit(&mut owner) {
            let mut outcome = release_audit.release_outcome(
                &mut owner,
                BackendTerminalStage::EngineOwnedCommit,
                failure.status,
                failure.may_have_mutated,
            );
            if failure.release_must_remain_unproven {
                outcome = outcome.with_release_proof(BackendReleaseProof::Unproven);
            }
            drop(owner);
            if outcome.disposition() == BackendFailureDisposition::Quarantine {
                self.reservation.quarantine();
            }
            return Err(NativeMemoryAllocationError::NativeCommit {
                source: failure.source,
                outcome,
            });
        }

        let groups = &self.groups;
        let baseline = &self.reconciliation_baseline;
        let requests = &self.requests;
        match finalize_reservation_with(&mut self.reservation, || {
            build_live_reconciliations(groups, baseline, requests)
        }) {
            Ok(()) => {}
            Err(ReservationFinalizeError::Evidence(source)) => {
                let outcome = release_audit.release_outcome(
                    &mut owner,
                    BackendTerminalStage::EngineOwnedReconcile,
                    ffi::GGML_STATUS_FAILED,
                    true,
                );
                drop(owner);
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::PostAllocationStats { source, outcome });
            }
            Err(ReservationFinalizeError::Broker(source)) => {
                let outcome = release_audit.release_outcome(
                    &mut owner,
                    BackendTerminalStage::EngineOwnedReconcile,
                    ffi::GGML_STATUS_FAILED,
                    true,
                );
                drop(owner);
                if outcome.disposition() == BackendFailureDisposition::Quarantine {
                    self.reservation.quarantine();
                }
                return Err(NativeMemoryAllocationError::BrokerCommit { source, outcome });
            }
        }

        Ok(NativeMemoryAllocation {
            owner: Some(owner),
            reservation: Some(self.reservation),
        })
    }

    /// Commits engine-requested memory whose native owner is an enclosing
    /// object (currently the ggml scheduler) rather than a standalone Rust
    /// buffer guard. The returned lease must be stored inside that owner and
    /// the owner must release its native allocation before dropping the
    /// lease.
    ///
    /// Unlike [`Self::commit_engine_owned_with`], a failure after native state
    /// may have changed cannot locally destroy the allocation: the enclosing
    /// owner may already have grown an internal high-water arena. The callback
    /// must therefore classify its failure precisely. Pre-mutation validation
    /// failures refund normally; potentially-mutating failures quarantine the
    /// broker reservation and require poisoning or destroying the owner.
    pub(crate) fn commit_owner_attached_with<E, F>(
        mut self,
        identity: BackendTerminalIdentity,
        native_commit: F,
    ) -> Result<NativeOwnerAttachedMemoryLease, NativeOwnerAttachedMemoryError<E>>
    where
        E: NativeOwnerAttachedCommitOutcome,
        F: FnOnce() -> Result<(), E>,
    {
        self.require_request_class(NativeRequestClass::EngineOwned)
            .map_err(NativeOwnerAttachedMemoryError::RequestKinds)?;
        for index in 0..self.groups.len() {
            if let Err(source) = self.groups[index].reserve_private() {
                let may_have_mutated = source.may_have_committed_private_state();
                let outcome =
                    self.private_reserve_outcome(identity.clone(), &source, may_have_mutated);
                let quarantined = outcome.disposition() == BackendFailureDisposition::Quarantine;
                if quarantined {
                    self.reservation.quarantine();
                }
                return Err(NativeOwnerAttachedMemoryError::PrivateReserve {
                    group_id: self.groups[index].group_id.clone(),
                    source,
                    outcome,
                });
            }
        }

        if let Err(failure) = native_commit() {
            return Err(owner_attached_native_commit_error(
                &mut self.reservation,
                failure,
            ));
        }

        let groups = &self.groups;
        let baseline = &self.reconciliation_baseline;
        let requests = &self.requests;
        match finalize_reservation_with(&mut self.reservation, || {
            build_live_reconciliations(groups, baseline, requests)
        }) {
            Ok(()) => {}
            Err(ReservationFinalizeError::Evidence(source)) => {
                self.reservation.quarantine();
                return Err(NativeOwnerAttachedMemoryError::PostAllocationStats { source });
            }
            Err(ReservationFinalizeError::Broker(source)) => {
                self.reservation.quarantine();
                return Err(NativeOwnerAttachedMemoryError::BrokerCommit { source });
            }
        }

        Ok(NativeOwnerAttachedMemoryLease {
            reservation: Some(self.reservation),
        })
    }

    /// Commits backend-owned GRAPH_PRIVATE allocations. Once any group's
    /// reserve succeeds, later failure cannot prove that shared/cached backend
    /// high-water memory was released. Such failure quarantines the broker
    /// reservation; this layer never calls broad `trim` on a shared backend.
    pub(crate) fn commit_backend_private(
        self,
        identity: BackendTerminalIdentity,
    ) -> Result<NativeBackendPrivateMemoryLease, NativeBackendPrivateMemoryError> {
        let lease = self.reserve_backend_private_deferred(identity)?;
        lease.finalize_pending()?;
        Ok(lease)
    }

    /// Validates/reserves GRAPH_PRIVATE requests while intentionally retaining
    /// the broker's pending (and, for provisional quotes, domain-exclusive)
    /// gate. The caller attaches the returned lease to every backend owner,
    /// performs the first graph compute that establishes dynamic pool high
    /// water, then calls [`NativeBackendPrivateMemoryLease::finalize_pending`].
    pub(crate) fn reserve_backend_private_deferred(
        mut self,
        identity: BackendTerminalIdentity,
    ) -> Result<NativeBackendPrivateMemoryLease, NativeBackendPrivateMemoryError> {
        self.require_request_class(NativeRequestClass::BackendPrivate)
            .map_err(NativeBackendPrivateMemoryError::RequestKinds)?;
        let mut any_reserved = false;
        for index in 0..self.groups.len() {
            if let Err(source) = self.groups[index].reserve_private() {
                let may_have_mutated = any_reserved || source.may_have_committed_private_state();
                let outcome =
                    self.private_reserve_outcome(identity.clone(), &source, may_have_mutated);
                let quarantined = outcome.disposition() == BackendFailureDisposition::Quarantine;
                if quarantined {
                    self.reservation.quarantine();
                }
                return Err(NativeBackendPrivateMemoryError::PrivateReserve {
                    group_id: self.groups[index].group_id.clone(),
                    source,
                    outcome,
                });
            }
            any_reserved = true;
        }
        Ok(NativeBackendPrivateMemoryLease {
            inner: Rc::new(RefCell::new(NativeBackendPrivateMemoryLeaseInner {
                transaction: Some(self),
                committed_reservation: None,
                committed: false,
                quarantined: false,
            })),
        })
    }

    fn finalize_backend_private_in_place(&mut self) -> Result<(), NativeBackendPrivateMemoryError> {
        let groups = &self.groups;
        let baseline = &self.reconciliation_baseline;
        let requests = &self.requests;
        match finalize_reservation_with(&mut self.reservation, || {
            build_live_reconciliations(groups, baseline, requests)
        }) {
            Ok(()) => Ok(()),
            Err(ReservationFinalizeError::Evidence(source)) => {
                self.reservation.quarantine();
                Err(NativeBackendPrivateMemoryError::PostAllocationStats { source })
            }
            Err(ReservationFinalizeError::Broker(source)) => {
                self.reservation.quarantine();
                Err(NativeBackendPrivateMemoryError::BrokerCommit { source })
            }
        }
    }

    fn require_request_class(
        &self,
        expected: NativeRequestClass,
    ) -> Result<(), NativeMemoryRequestKindError> {
        let actual =
            classify_request_kinds(self.groups.iter().flat_map(|group| group.requests.iter()))?;
        if actual != expected {
            return Err(NativeMemoryRequestKindError::WrongTransaction { expected, actual });
        }
        Ok(())
    }
}

fn build_live_reconciliations(
    groups: &[NativeQuotedBackendGroup],
    baseline: &ReconciliationBaseline,
    requests: &[DomainReservationRequest],
) -> Result<Vec<DomainMemoryReconciliation>, NativeMemoryAdmissionError> {
    let observations = fetch_live_observations(groups)?;
    build_reconciliations_from_observations(baseline, requests, &observations)
}

fn fetch_live_observations(
    groups: &[NativeQuotedBackendGroup],
) -> Result<BTreeMap<MemoryDomainKey, DomainObservation>, NativeMemoryAdmissionError> {
    let mut observations = BTreeMap::new();
    for group in groups {
        let stats = group
            .abi
            .stats_at(BackendMemoryLifecyclePoint::PostAllocationReconciliation)?;
        let mapped = map_group_stats(
            &group.group_id,
            &group.backend_device_identity,
            group.provider,
            stats.domains(),
            None,
        )?;
        merge_group_observations(&mut observations, mapped)?;
    }
    Ok(observations)
}

enum ReservationFinalizeError<E> {
    Evidence(E),
    Broker(MemoryPlanningError),
}

/// Exact/upper quotes bypass the evidence closure entirely. This is important:
/// a concurrent external allocation can make a post snapshot noisier without
/// invalidating a proven native commitment bound.
fn finalize_reservation_with<E, F>(
    reservation: &mut DeviceMemoryReservationBatch,
    build_reconciliations: F,
) -> Result<(), ReservationFinalizeError<E>>
where
    F: FnOnce() -> Result<Vec<DomainMemoryReconciliation>, E>,
{
    if reservation.is_empty() {
        return Ok(());
    }
    if !reservation.requires_reconciliation() {
        return reservation
            .commit_quoted()
            .map_err(ReservationFinalizeError::Broker);
    }
    let reconciliations = build_reconciliations().map_err(ReservationFinalizeError::Evidence)?;
    reservation
        .reconcile_and_commit(&reconciliations)
        .map_err(ReservationFinalizeError::Broker)
}

fn classify_request_kinds<'a>(
    requests: impl IntoIterator<Item = &'a ffi::GgmlBackendMemoryRequestV1>,
) -> Result<NativeRequestClass, NativeMemoryRequestKindError> {
    let mut observed = None;
    for request in requests {
        let class = match request.kind {
            ffi::GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE => NativeRequestClass::BackendPrivate,
            ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER
            | ffi::GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT
            | ffi::GGML_BACKEND_MEMORY_REQUEST_TRANSFER => NativeRequestClass::EngineOwned,
            kind => return Err(NativeMemoryRequestKindError::Unsupported { kind }),
        };
        if let Some(previous) = observed
            && previous != class
        {
            return Err(NativeMemoryRequestKindError::Mixed);
        }
        observed = Some(class);
    }
    observed.ok_or(NativeMemoryRequestKindError::Empty)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeRequestClass {
    EngineOwned,
    BackendPrivate,
}

impl fmt::Display for NativeRequestClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EngineOwned => formatter.write_str("engine-owned"),
            Self::BackendPrivate => formatter.write_str("backend-private"),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum NativeMemoryRequestKindError {
    #[error("native memory allocation transaction contains no requests")]
    Empty,
    #[error("native memory allocation transaction mixes engine-owned and GRAPH_PRIVATE requests")]
    Mixed,
    #[error("native memory allocation transaction contains unsupported request kind {kind}")]
    Unsupported { kind: u32 },
    #[error(
        "one standalone native owner cannot span {backend_count} backend quote groups; use an owner-attached transaction"
    )]
    StandaloneOwnerSpansBackends { backend_count: usize },
    #[error("native memory transaction requires {expected} requests but received {actual}")]
    WrongTransaction {
        expected: NativeRequestClass,
        actual: NativeRequestClass,
    },
}

/// The true native owner and its committed process-wide memory lease. This
/// wrapper intentionally has an explicit `Drop`: native buffers disappear
/// before the reservation can refund committed bytes.
pub(crate) struct NativeMemoryAllocation<T: NativeMemoryOwner> {
    owner: Option<T>,
    /// The broker batch owns the one receipt row for this physical owner. Its
    /// state moves Reserved -> Committed/Reconciled -> Released together with
    /// the ledger; a second committed receipt would double-count one lease.
    reservation: Option<DeviceMemoryReservationBatch>,
}

/// Committed accounting for memory retained by a backend rather than an
/// engine-owned buffer. The caller must store this lease in the backend owner
/// itself; dropping it earlier would refund bytes while the backend high-water
/// allocation may still be resident.
#[must_use = "backend-private memory leases must be attached to the backend owner"]
#[derive(Clone)]
pub(crate) struct NativeBackendPrivateMemoryLease {
    /// One GRAPH_PRIVATE transaction may span multiple scheduler backends.
    /// Every involved backend owner receives a clone of this handle; the
    /// broker reservation therefore remains live until the last native owner
    /// is gone instead of being refunded when any one backend is dropped.
    inner: Rc<RefCell<NativeBackendPrivateMemoryLeaseInner>>,
}

struct NativeBackendPrivateMemoryLeaseInner {
    /// Kept after commit so its broker batch follows the backend owner's true
    /// lifetime. Quote groups are also retained; they are small and preserve
    /// the evidence required when finalization is intentionally deferred.
    transaction: Option<NativeMemoryAllocationTransaction>,
    /// Used when a committed broker lease has no remaining quote transaction
    /// (and by the clone-lifetime regression). Normal deferred native leases
    /// keep their committed batch inside `transaction`.
    committed_reservation: Option<DeviceMemoryReservationBatch>,
    committed: bool,
    quarantined: bool,
}

/// Accounting attached to an enclosing native owner, such as a ggml
/// scheduler that owns its gallocr buffers internally. The enclosing owner is
/// responsible for dropping native state before this lease collection.
#[must_use = "owner-attached memory leases must be stored inside their native owner"]
pub(crate) struct NativeOwnerAttachedMemoryLease {
    reservation: Option<DeviceMemoryReservationBatch>,
}

impl NativeOwnerAttachedMemoryLease {
    pub(crate) fn reservation(&self) -> &DeviceMemoryReservationBatch {
        self.reservation
            .as_ref()
            .expect("owner-attached lease is present until Drop")
    }

    pub(crate) fn quarantine(&mut self) {
        self.reservation
            .as_mut()
            .expect("owner-attached lease is present until Drop")
            .quarantine();
    }
}

impl NativeBackendPrivateMemoryLease {
    pub(crate) fn shares_reservation_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn is_pending(&self) -> bool {
        let inner = self.inner.borrow();
        !inner.committed && !inner.quarantined && inner.transaction.is_some()
    }

    /// Records that a later graph compute reused this backend-owned high-water
    /// lease instead of admitting a second receipt row for the same native pool.
    pub(crate) fn record_receipt_reuse(&self) {
        let inner = self.inner.borrow();
        if let Some(transaction) = inner.transaction.as_ref() {
            transaction.reservation().record_receipt_reuse();
        } else if let Some(reservation) = inner.committed_reservation.as_ref() {
            reservation.record_receipt_reuse();
        }
    }

    /// Captures any private growth performed by `reserve_private` itself,
    /// before the scheduler sibling mutates the same physical domain.
    pub(crate) fn checkpoint_private_growth(&self) -> Result<(), NativeBackendPrivateMemoryError> {
        self.rebase_pending(true)
    }

    /// Excludes the scheduler sibling's already-accounted allocation from the
    /// private transaction, establishing the baseline for first-compute pool
    /// growth without double-charging the same physical bytes.
    pub(crate) fn rebase_after_sibling_allocation(
        &self,
    ) -> Result<(), NativeBackendPrivateMemoryError> {
        self.rebase_pending(false)
    }

    fn rebase_pending(
        &self,
        accumulate_private_growth: bool,
    ) -> Result<(), NativeBackendPrivateMemoryError> {
        let mut inner = self.inner.borrow_mut();
        if inner.committed {
            return Ok(());
        }
        if inner.quarantined {
            return Err(NativeBackendPrivateMemoryError::LeaseQuarantined);
        }
        let transaction = inner
            .transaction
            .as_mut()
            .ok_or(NativeBackendPrivateMemoryError::LeaseQuarantined)?;
        if let Err(source) = transaction.rebase_live_observations(accumulate_private_growth) {
            transaction.reservation.quarantine();
            inner.quarantined = true;
            return Err(NativeBackendPrivateMemoryError::PostAllocationStats { source });
        }
        Ok(())
    }

    /// Converts a deferred provisional gate into committed backend-owner
    /// accounting using live statistics after the first graph compute has had
    /// a chance to establish graph-specific private pool high water.
    pub(crate) fn finalize_pending(&self) -> Result<(), NativeBackendPrivateMemoryError> {
        let mut inner = self.inner.borrow_mut();
        if inner.committed {
            return Ok(());
        }
        if inner.quarantined {
            return Err(NativeBackendPrivateMemoryError::LeaseQuarantined);
        }
        let result = inner
            .transaction
            .as_mut()
            .ok_or(NativeBackendPrivateMemoryError::LeaseQuarantined)?
            .finalize_backend_private_in_place();
        match result {
            Ok(()) => {
                inner.committed = true;
                Ok(())
            }
            Err(source) => {
                inner.quarantined = true;
                Err(source)
            }
        }
    }

    /// Prevents a poisoned backend's retained high-water bytes from ever
    /// being returned to the available budget. The native backend owner is
    /// intentionally leaked by the caller after this transition because its
    /// allocator can no longer prove that the physical commitment is gone.
    pub(crate) fn quarantine(&self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(transaction) = inner.transaction.as_mut() {
            transaction.reservation.quarantine();
        } else if let Some(reservation) = inner.committed_reservation.as_mut() {
            reservation.quarantine();
        }
        inner.quarantined = true;
    }
}

impl<T: NativeMemoryOwner> NativeMemoryAllocation<T> {
    pub(crate) fn record_receipt_reuse(&self) {
        self.reservation().record_receipt_reuse();
    }

    pub(crate) fn owner(&self) -> &T {
        self.owner
            .as_ref()
            .expect("native memory allocation owner is present until Drop")
    }

    pub(crate) fn owner_mut(&mut self) -> &mut T {
        self.owner
            .as_mut()
            .expect("native memory allocation owner is present until Drop")
    }

    pub(crate) fn reservation(&self) -> &DeviceMemoryReservationBatch {
        self.reservation
            .as_ref()
            .expect("native memory allocation lease is present until Drop")
    }
}

impl<T: NativeMemoryOwner> Deref for NativeMemoryAllocation<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.owner()
    }
}

impl<T: NativeMemoryOwner> DerefMut for NativeMemoryAllocation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.owner_mut()
    }
}

impl<T: NativeMemoryOwner> Drop for NativeMemoryAllocation<T> {
    fn drop(&mut self) {
        if let Some(mut owner) = self.owner.take() {
            let release_proof = owner.release_native();
            drop(owner);
            if release_proof != BackendReleaseProof::Proven
                && let Some(reservation) = self.reservation.as_mut()
            {
                reservation.quarantine();
            }
        }
        drop(self.reservation.take());
    }
}

#[derive(Debug)]
pub(crate) enum NativeMemoryAllocationError<E> {
    RequestKinds(NativeMemoryRequestKindError),
    PrivateReserve {
        group_id: String,
        source: BackendMemoryAbiError,
        outcome: BackendTerminalOutcome,
    },
    NativeCommit {
        source: E,
        outcome: BackendTerminalOutcome,
    },
    PostAllocationStats {
        source: NativeMemoryAdmissionError,
        outcome: BackendTerminalOutcome,
    },
    BrokerCommit {
        source: MemoryPlanningError,
        outcome: BackendTerminalOutcome,
    },
}

#[derive(Debug)]
pub(crate) enum NativeOwnerAttachedMemoryError<E> {
    RequestKinds(NativeMemoryRequestKindError),
    PrivateReserve {
        group_id: String,
        source: BackendMemoryAbiError,
        outcome: BackendTerminalOutcome,
    },
    NativeCommit {
        source: E,
        quarantined: bool,
    },
    PostAllocationStats {
        source: NativeMemoryAdmissionError,
    },
    BrokerCommit {
        source: MemoryPlanningError,
    },
}

impl<E: fmt::Display> fmt::Display for NativeOwnerAttachedMemoryError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestKinds(source) => source.fmt(formatter),
            Self::PrivateReserve {
                group_id,
                source,
                outcome,
            } => write!(
                formatter,
                "native quote-token validation failed for owner group '{group_id}' (outcome={outcome:?}): {source}"
            ),
            Self::NativeCommit {
                source,
                quarantined,
            } => write!(
                formatter,
                "native owner-attached allocation commit failed (quarantined={quarantined}): {source}"
            ),
            Self::PostAllocationStats { source } => write!(
                formatter,
                "owner-attached post-allocation stats failed and were quarantined: {source}"
            ),
            Self::BrokerCommit { source } => write!(
                formatter,
                "owner-attached broker commit failed and was quarantined: {source}"
            ),
        }
    }
}

impl<E: fmt::Display> fmt::Display for NativeMemoryAllocationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestKinds(source) => source.fmt(formatter),
            Self::PrivateReserve {
                group_id,
                source,
                outcome,
            } => write!(
                formatter,
                "native quote-token validation failed for group '{group_id}' (outcome={outcome:?}): {source}"
            ),
            Self::NativeCommit { source, outcome } => {
                write!(
                    formatter,
                    "native allocation commit failed (outcome={outcome:?}): {source}"
                )
            }
            Self::PostAllocationStats { source, outcome } => {
                write!(
                    formatter,
                    "post-allocation native memory stats failed (outcome={outcome:?}): {source}"
                )
            }
            Self::BrokerCommit { source, outcome } => {
                write!(
                    formatter,
                    "native memory broker commit failed (outcome={outcome:?}): {source}"
                )
            }
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum NativeBackendPrivateMemoryError {
    #[error(transparent)]
    RequestKinds(#[from] NativeMemoryRequestKindError),
    #[error(
        "native backend-private reserve failed for group '{group_id}' (outcome={outcome:?}): {source}"
    )]
    PrivateReserve {
        group_id: String,
        source: BackendMemoryAbiError,
        outcome: BackendTerminalOutcome,
    },
    #[error(
        "post-allocation backend-private stats failed and the reservation was quarantined: {source}"
    )]
    PostAllocationStats { source: NativeMemoryAdmissionError },
    #[error(
        "backend-private broker reconciliation failed and the reservation was quarantined: {source}"
    )]
    BrokerCommit { source: MemoryPlanningError },
    #[error("backend-private memory lease is already quarantined")]
    LeaseQuarantined,
}

impl<E> std::error::Error for NativeMemoryAllocationError<E> where E: std::error::Error + 'static {}

struct NativeGroupView<'a> {
    group_id: &'a str,
    backend_device_identity: &'a PhysicalDeviceKey,
    provider: crate::device::execution_route::ExecutionProvider,
    quote: &'a ffi::GgmlBackendMemoryQuoteV1,
    claims: &'a [ffi::GgmlBackendMemoryClaimV1],
    stats: &'a [ffi::GgmlBackendMemoryStatsV1],
    request_semantics: &'a BTreeMap<u64, NativeMemoryClaimSemantics>,
    shared_semantics: &'a NativeMemoryClaimSemantics,
}

impl<'a> From<&'a NativeQuotedBackendGroup> for NativeGroupView<'a> {
    fn from(group: &'a NativeQuotedBackendGroup) -> Self {
        Self {
            group_id: &group.group_id,
            backend_device_identity: &group.backend_device_identity,
            provider: group.provider,
            quote: group.quote.raw(),
            claims: group.quote.claims(),
            stats: group.fresh_stats.domains(),
            request_semantics: &group.request_semantics,
            shared_semantics: &group.shared_semantics,
        }
    }
}

struct BuiltAdmission {
    claims: Vec<MemoryClaim>,
    requests: Vec<DomainReservationRequest>,
    evidence: Vec<NativeMemoryQuoteEvidence>,
    reconciliation_baseline: ReconciliationBaseline,
}

fn build_from_views(
    groups: &[NativeGroupView<'_>],
) -> Result<BuiltAdmission, NativeMemoryAdmissionError> {
    let mut group_ids = HashSet::with_capacity(groups.len());
    let mut observations = BTreeMap::<MemoryDomainKey, DomainObservation>::new();
    let mut group_observed_domains = BTreeMap::<String, Vec<MemoryDomainKey>>::new();

    for group in groups {
        if group.group_id.trim().is_empty() {
            return Err(NativeMemoryAdmissionError::EmptyGroupId);
        }
        if !group_ids.insert(group.group_id.to_owned()) {
            return Err(NativeMemoryAdmissionError::DuplicateGroupId {
                group_id: group.group_id.to_owned(),
            });
        }
        if group.stats.is_empty() {
            return Err(NativeMemoryAdmissionError::MissingFreshStats {
                group_id: group.group_id.to_owned(),
            });
        }
        let mapped = map_group_stats(
            group.group_id,
            group.backend_device_identity,
            group.provider,
            group.stats,
            Some(group.quote.stats_generation),
        )?;
        group_observed_domains.insert(group.group_id.to_owned(), mapped.keys().cloned().collect());
        merge_group_observations(&mut observations, mapped)?;
    }

    let mut claims = Vec::new();
    let mut evidence = Vec::with_capacity(groups.len());
    for group in groups {
        let quote_requires_reconciliation = quote_requires_reconciliation(group.quote);
        let has_residual_uncertainty = quote_has_residual_uncertainty(group.quote);
        let mut claimed_domains = HashSet::new();
        let mut provisional_domains = HashSet::new();
        let mut group_claim_peak_bytes = 0_u64;
        let mut claim_flags = Vec::with_capacity(group.claims.len());
        for native in group.claims {
            if native.struct_size < mem::size_of::<ffi::GgmlBackendMemoryClaimV1>() as u32 {
                return Err(NativeMemoryAdmissionError::IncompatibleClaimLayout {
                    group_id: group.group_id.to_owned(),
                });
            }
            let domain = map_native_domain(&native.domain, group.backend_device_identity)?;
            if let Some(observation) = observations.get_mut(&domain) {
                observation.claim_flags |= native.flags;
            }
            if !observations.contains_key(&domain) {
                return Err(NativeMemoryAdmissionError::MissingClaimStats {
                    group_id: group.group_id.to_owned(),
                    domain,
                });
            }
            let semantics = if native.request_id == 0 {
                group.shared_semantics
            } else {
                group
                    .request_semantics
                    .get(&native.request_id)
                    .ok_or_else(|| NativeMemoryAdmissionError::MissingRequestSemantics {
                        group_id: group.group_id.to_owned(),
                        request_id: native.request_id,
                    })?
            };
            let confidence = claim_confidence(native.flags, quote_requires_reconciliation)?;
            // FILE_BACKED host-import of a pack mapping is charged once per
            // open mapping through `device::pack_weight_residency`, not here.
            // Callers set `currently_allocated_bytes = requested_bytes` on the
            // HOST_IMPORT quote so the backend reports reuse
            // (`commit_peak_extra_upper_bytes = 0`). Blindly zeroing every
            // FILE_BACKED claim would under-count distinct concurrent packs;
            // blindly charging full size would double-count multi-stage binds
            // of one mapping. Trust the native incremental numbers.
            group_claim_peak_bytes = group_claim_peak_bytes
                .checked_add(native.commit_peak_extra_upper_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "native claim peak sum",
                })?;
            if confidence == QuoteConfidence::Provisional {
                provisional_domains.insert(domain.clone());
            }
            let incremental_retained = native
                .retained_after_use_upper_bytes
                .saturating_sub(native.committed_before_bytes);
            claims.push(MemoryClaim {
                resource_id: semantics.resource_id.clone(),
                domain: domain.clone(),
                requested_bytes: native.payload_requested_bytes,
                incremental_peak_bytes: Some(native.commit_peak_extra_upper_bytes),
                incremental_retained_bytes: Some(incremental_retained),
                confidence,
                lifetime: semantics.lifetime,
                phases: semantics.phases,
            });
            claimed_domains.insert(domain);
            claim_flags.push((native.request_id, native.flags));
        }

        // `provisional_requested_upper_bytes` is a group total, not an extra
        // charge on top of per-claim estimates. Add only an uncovered delta.
        // A group aggregate may only be attributed when native evidence leaves
        // exactly one possible physical domain; otherwise the provider must
        // return a domain-scoped claim rather than forcing this layer to guess.
        if group.quote.provisional_requested_upper_bytes > group_claim_peak_bytes {
            let delta = group.quote.provisional_requested_upper_bytes - group_claim_peak_bytes;
            let candidate_domains = if provisional_domains.len() == 1 {
                provisional_domains.iter().cloned().collect::<Vec<_>>()
            } else if claimed_domains.len() == 1 {
                claimed_domains.iter().cloned().collect::<Vec<_>>()
            } else {
                group_observed_domains
                    .get(group.group_id)
                    .expect("group observations were built above")
                    .to_vec()
            };
            if candidate_domains.len() != 1 {
                return Err(
                    NativeMemoryAdmissionError::AmbiguousProvisionalEstimateDomain {
                        group_id: group.group_id.to_owned(),
                        candidate_domains: candidate_domains.len(),
                    },
                );
            }
            let domain = candidate_domains
                .first()
                .expect("length checked above")
                .clone();
            claims.push(MemoryClaim {
                resource_id: format!(
                    "{}/native-provisional-estimate/{}",
                    group.shared_semantics.resource_id, group.group_id
                ),
                domain: domain.clone(),
                requested_bytes: delta,
                incremental_peak_bytes: Some(delta),
                incremental_retained_bytes: Some(0),
                confidence: QuoteConfidence::Provisional,
                lifetime: group.shared_semantics.lifetime,
                phases: group.shared_semantics.phases,
            });
            claimed_domains.insert(domain);
        }

        // A backend-private residual has no native claim row. Preserve it as a
        // zero-byte provisional marker on every live domain the backend
        // reports. The broker turns this marker into an exclusive physical-
        // domain gate, held through first compute and live reconciliation, so
        // a missing provider upper bound can never be concurrently oversold.
        if has_residual_uncertainty {
            let domains = group_observed_domains
                .get(group.group_id)
                .expect("group observations were built above");
            for domain in domains {
                if claimed_domains.insert(domain.clone()) {
                    claims.push(MemoryClaim {
                        resource_id: format!(
                            "{}/native-residual/{}",
                            group.shared_semantics.resource_id, group.group_id
                        ),
                        domain: domain.clone(),
                        requested_bytes: 0,
                        incremental_peak_bytes: Some(0),
                        incremental_retained_bytes: Some(0),
                        confidence: QuoteConfidence::Provisional,
                        lifetime: group.shared_semantics.lifetime,
                        phases: group.shared_semantics.phases,
                    });
                }
            }
        }

        evidence.push(NativeMemoryQuoteEvidence {
            group_id: group.group_id.to_owned(),
            quote_flags: group.quote.flags,
            residual_flags: group.quote.residual_flags,
            residual_request_count: group.quote.residual_request_count,
            provisional_requested_upper_bytes: group.quote.provisional_requested_upper_bytes,
            claim_flags,
        });
    }

    let footprint = AllocationFootprint::new(claims.clone());
    let mut requests = Vec::new();
    for domain_footprint in footprint.domain_footprints()? {
        let snapshot = observations
            .get(&domain_footprint.domain)
            .ok_or_else(|| NativeMemoryAdmissionError::MissingClaimStats {
                group_id: "merged-candidate".to_owned(),
                domain: domain_footprint.domain.clone(),
            })?
            .snapshot;
        requests.push(DomainReservationRequest::from_footprint(
            domain_footprint,
            snapshot,
        ));
    }

    Ok(BuiltAdmission {
        claims,
        requests,
        evidence,
        reconciliation_baseline: ReconciliationBaseline {
            observations,
            carried_private_bytes: BTreeMap::new(),
        },
    })
}

fn quote_requires_reconciliation(quote: &ffi::GgmlBackendMemoryQuoteV1) -> bool {
    quote.flags
        & (ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL
            | ffi::GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY)
        != 0
        || quote.residual_flags != 0
        || quote.residual_request_count != 0
}

fn quote_has_residual_uncertainty(quote: &ffi::GgmlBackendMemoryQuoteV1) -> bool {
    quote.flags & ffi::GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY != 0
        || quote.residual_flags != 0
        || quote.residual_request_count != 0
}

fn claim_confidence(
    flags: u32,
    quote_requires_reconciliation: bool,
) -> Result<QuoteConfidence, NativeMemoryAdmissionError> {
    let provisional = quote_requires_reconciliation
        || flags
            & (ffi::GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL
                | ffi::GGML_BACKEND_MEMORY_CLAIM_DRIVER_ESTIMATE)
            != 0;
    if provisional {
        return Ok(QuoteConfidence::Provisional);
    }
    let exact = flags & ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT != 0;
    let upper = flags & ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER != 0;
    match (exact, upper) {
        (true, false) => Ok(QuoteConfidence::ExactCommitted),
        (false, true) => Ok(QuoteConfidence::CommittedUpperBound),
        _ => Err(NativeMemoryAdmissionError::InvalidClaimConfidence { flags }),
    }
}

fn map_native_domain(
    domain: &ffi::GgmlBackendMemoryDomainIdV1,
    _backend_device_identity: &PhysicalDeviceKey,
) -> Result<MemoryDomainKey, NativeMemoryAdmissionError> {
    match domain.kind {
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE
        | ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED
        | ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED
        | ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED => Ok(MemoryDomainKey::SystemMemory),
        ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL => {
            if domain.physical_device_uuid.iter().all(|byte| *byte == 0) {
                Err(NativeMemoryAdmissionError::UnprovenPhysicalDeviceIdentity {
                    heap_index: domain.heap_index,
                })
            } else {
                let mut encoded = String::with_capacity(37);
                encoded.push_str("uuid:");
                for byte in domain.physical_device_uuid {
                    write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
                }
                Ok(MemoryDomainKey::DedicatedDevice {
                    physical_device: PhysicalDeviceKey::new(encoded)?,
                    heap_index: domain.heap_index,
                })
            }
        }
        kind => Err(NativeMemoryAdmissionError::UnsupportedDomainKind { kind }),
    }
}

#[derive(Debug, Clone)]
struct DomainObservation {
    snapshot: DeviceMemorySnapshot,
    domain_kind: BackendMemoryDomainKind,
    provider: Option<crate::device::execution_route::ExecutionProvider>,
    backend_owned_reliability: RuntimeBackendOwnedReliability,
    heap_index: u32,
    total_bytes: u64,
    budget_bytes: Option<u64>,
    stats_generation: u64,
    device_used_bytes: u64,
    /// Current provider-owned physical commitment. CUDA scratch is normally
    /// cached after an op, so `live_bytes` alone would incorrectly return to
    /// zero; workspace/live+cached expose the retained pool high-water.
    backend_owned_committed_bytes: u64,
    backend_owned_live_bytes: u64,
    backend_owned_cached_bytes: u64,
    backend_owned_workspace_bytes: u64,
    backend_owned_high_water_bytes: u64,
    claim_flags: u32,
    quote_generation: Option<u64>,
}

fn backend_domain_kind(kind: u32) -> BackendMemoryDomainKind {
    match kind {
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE => BackendMemoryDomainKind::HostPageable,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED => BackendMemoryDomainKind::HostPinned,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED => BackendMemoryDomainKind::Unified,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL => BackendMemoryDomainKind::DeviceLocal,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED => BackendMemoryDomainKind::FileBacked,
        kind => BackendMemoryDomainKind::Unknown(kind),
    }
}

fn map_group_stats(
    group_id: &str,
    backend_device_identity: &PhysicalDeviceKey,
    provider: crate::device::execution_route::ExecutionProvider,
    stats: &[ffi::GgmlBackendMemoryStatsV1],
    expected_generation: Option<u64>,
) -> Result<BTreeMap<MemoryDomainKey, DomainObservation>, NativeMemoryAdmissionError> {
    let mut mapped = BTreeMap::new();
    for raw in stats {
        if raw.struct_size < mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32 {
            return Err(NativeMemoryAdmissionError::IncompatibleStatsLayout {
                group_id: group_id.to_owned(),
            });
        }
        if let Some(expected) = expected_generation
            && raw.generation != expected
        {
            return Err(NativeMemoryAdmissionError::StaleStatsGeneration {
                group_id: group_id.to_owned(),
                quote_generation: expected,
                stats_generation: raw.generation,
            });
        }
        match raw.health {
            ffi::GGML_BACKEND_MEMORY_HEALTHY | ffi::GGML_BACKEND_MEMORY_DEGRADED => {}
            ffi::GGML_BACKEND_MEMORY_QUARANTINED | ffi::GGML_BACKEND_MEMORY_DEVICE_LOST => {
                return Err(NativeMemoryAdmissionError::UnhealthyBackend {
                    group_id: group_id.to_owned(),
                    health: raw.health,
                    status: raw.last_ggml_status,
                    native_error: raw.last_native_error,
                    quarantine_generation: raw.quarantine_generation,
                });
            }
            health => {
                return Err(NativeMemoryAdmissionError::UnknownBackendHealth {
                    group_id: group_id.to_owned(),
                    health,
                });
            }
        }
        if raw.flags & ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE != 0 {
            return Err(NativeMemoryAdmissionError::StatsBudgetUnavailable {
                group_id: group_id.to_owned(),
                heap_index: raw.domain.heap_index,
            });
        }
        let domain = map_native_domain(&raw.domain, backend_device_identity)?;
        let total_bytes = if raw.budget_bytes != 0 {
            raw.budget_bytes
        } else {
            raw.total_bytes
        };
        if total_bytes == 0 {
            return Err(NativeMemoryAdmissionError::InvalidStatsSnapshot {
                group_id: group_id.to_owned(),
                heap_index: raw.domain.heap_index,
            });
        }
        let confidence = if raw.domain.kind == ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED {
            MemoryObservationConfidence::WorkingSetBudget
        } else {
            MemoryObservationConfidence::DeviceSnapshot
        };
        let backend_owned_live_and_cached = raw
            .backend_owned_live_bytes
            .checked_add(raw.backend_owned_cached_bytes)
            .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                operation: "backend-owned live plus cached bytes",
            })?;
        let observation = DomainObservation {
            snapshot: DeviceMemorySnapshot {
                free_bytes: raw.device_free_bytes.min(total_bytes),
                total_bytes,
                confidence,
            },
            domain_kind: backend_domain_kind(raw.domain.kind),
            provider: Some(provider),
            backend_owned_reliability: if backend_owned_unknown_reason(provider).is_some() {
                RuntimeBackendOwnedReliability::Incomplete
            } else {
                RuntimeBackendOwnedReliability::Complete
            },
            heap_index: raw.domain.heap_index,
            total_bytes: raw.total_bytes,
            budget_bytes: (raw.budget_bytes != 0).then_some(raw.budget_bytes),
            stats_generation: raw.generation,
            device_used_bytes: raw.device_used_bytes,
            backend_owned_committed_bytes: raw
                .backend_owned_workspace_bytes
                .max(backend_owned_live_and_cached)
                .max(raw.backend_owned_live_bytes),
            backend_owned_live_bytes: raw.backend_owned_live_bytes,
            backend_owned_cached_bytes: raw.backend_owned_cached_bytes,
            backend_owned_workspace_bytes: raw.backend_owned_workspace_bytes,
            backend_owned_high_water_bytes: raw.backend_owned_high_water_bytes,
            claim_flags: 0,
            quote_generation: expected_generation,
        };
        merge_observation_within_group(&mut mapped, domain, observation);
    }
    Ok(mapped)
}

fn merge_observation_within_group(
    mapped: &mut BTreeMap<MemoryDomainKey, DomainObservation>,
    domain: MemoryDomainKey,
    observation: DomainObservation,
) {
    mapped
        .entry(domain)
        .and_modify(|existing| {
            existing.snapshot = merge_snapshots(existing.snapshot, observation.snapshot);
            // A backend may repeat process-wide counters on each native heap.
            // Max avoids counting the same backend pool more than once.
            existing.device_used_bytes = existing
                .device_used_bytes
                .max(observation.device_used_bytes);
            existing.backend_owned_committed_bytes = existing
                .backend_owned_committed_bytes
                .max(observation.backend_owned_committed_bytes);
            existing.backend_owned_live_bytes = existing
                .backend_owned_live_bytes
                .max(observation.backend_owned_live_bytes);
            existing.backend_owned_cached_bytes = existing
                .backend_owned_cached_bytes
                .max(observation.backend_owned_cached_bytes);
            existing.backend_owned_workspace_bytes = existing
                .backend_owned_workspace_bytes
                .max(observation.backend_owned_workspace_bytes);
            existing.backend_owned_high_water_bytes = existing
                .backend_owned_high_water_bytes
                .max(observation.backend_owned_high_water_bytes);
            existing.provider = match (existing.provider, observation.provider) {
                (Some(left), Some(right)) if left == right => Some(left),
                _ => None,
            };
            if observation.backend_owned_reliability == RuntimeBackendOwnedReliability::Incomplete {
                existing.backend_owned_reliability = RuntimeBackendOwnedReliability::Incomplete;
            }
            existing.claim_flags |= observation.claim_flags;
            existing.quote_generation =
                match (existing.quote_generation, observation.quote_generation) {
                    (Some(left), Some(right)) if left == right => Some(left),
                    _ => None,
                };
        })
        .or_insert(observation);
}

fn merge_group_observations(
    merged: &mut BTreeMap<MemoryDomainKey, DomainObservation>,
    group: BTreeMap<MemoryDomainKey, DomainObservation>,
) -> Result<(), NativeMemoryAdmissionError> {
    for (domain, observation) in group {
        if let Some(existing) = merged.get_mut(&domain) {
            existing.snapshot = merge_snapshots(existing.snapshot, observation.snapshot);
            // Multiple APIs may expose the same physical heap. Physical usage
            // is one observation, while backend-owned allocations are disjoint
            // and therefore additive across concrete backend groups.
            existing.device_used_bytes = existing
                .device_used_bytes
                .max(observation.device_used_bytes);
            existing.backend_owned_committed_bytes = existing
                .backend_owned_committed_bytes
                .checked_add(observation.backend_owned_committed_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "backend-owned live-byte merge",
                })?;
            existing.backend_owned_live_bytes = existing
                .backend_owned_live_bytes
                .checked_add(observation.backend_owned_live_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "backend-owned live-byte merge",
                })?;
            existing.backend_owned_cached_bytes = existing
                .backend_owned_cached_bytes
                .checked_add(observation.backend_owned_cached_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "backend-owned cached-byte merge",
                })?;
            existing.backend_owned_workspace_bytes = existing
                .backend_owned_workspace_bytes
                .checked_add(observation.backend_owned_workspace_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "backend-owned workspace-byte merge",
                })?;
            existing.backend_owned_high_water_bytes = existing
                .backend_owned_high_water_bytes
                .checked_add(observation.backend_owned_high_water_bytes)
                .ok_or(NativeMemoryAdmissionError::ArithmeticOverflow {
                    operation: "backend-owned high-water merge",
                })?;
            existing.provider = match (existing.provider, observation.provider) {
                (Some(left), Some(right)) if left == right => Some(left),
                _ => None,
            };
            if observation.backend_owned_reliability == RuntimeBackendOwnedReliability::Incomplete {
                existing.backend_owned_reliability = RuntimeBackendOwnedReliability::Incomplete;
            }
            existing.claim_flags |= observation.claim_flags;
            existing.quote_generation =
                match (existing.quote_generation, observation.quote_generation) {
                    (Some(left), Some(right)) if left == right => Some(left),
                    _ => None,
                };
        } else {
            merged.insert(domain, observation);
        }
    }
    Ok(())
}

fn merge_snapshots(
    left: DeviceMemorySnapshot,
    right: DeviceMemorySnapshot,
) -> DeviceMemorySnapshot {
    DeviceMemorySnapshot {
        free_bytes: left.free_bytes.min(right.free_bytes),
        total_bytes: left.total_bytes.min(right.total_bytes),
        confidence: weaker_confidence(left.confidence, right.confidence),
    }
}

fn weaker_confidence(
    left: MemoryObservationConfidence,
    right: MemoryObservationConfidence,
) -> MemoryObservationConfidence {
    use MemoryObservationConfidence::{DeviceSnapshot, Heuristic, Unknown, WorkingSetBudget};
    let rank = |value| match value {
        DeviceSnapshot => 3,
        WorkingSetBudget => 2,
        Heuristic => 1,
        Unknown => 0,
    };
    if rank(left) <= rank(right) {
        left
    } else {
        right
    }
}

#[derive(Debug, Clone)]
struct ReconciliationBaseline {
    observations: BTreeMap<MemoryDomainKey, DomainObservation>,
    /// Private commitment observed before a separately-owned sibling mutates
    /// the same domain. Reconciliation adds the later first-compute delta to
    /// this carry while excluding the sibling's already-accounted bytes.
    carried_private_bytes: BTreeMap<MemoryDomainKey, u64>,
}

struct PostStatsView<'a> {
    group_id: &'a str,
    backend_device_identity: &'a PhysicalDeviceKey,
    provider: crate::device::execution_route::ExecutionProvider,
    stats: &'a [ffi::GgmlBackendMemoryStatsV1],
}

fn build_reconciliations(
    baseline: &ReconciliationBaseline,
    requests: &[DomainReservationRequest],
    post: &[PostStatsView<'_>],
) -> Result<Vec<DomainMemoryReconciliation>, NativeMemoryAdmissionError> {
    let mut post_ids = HashSet::with_capacity(post.len());
    let mut observations = BTreeMap::new();
    for group in post {
        if !post_ids.insert(group.group_id) {
            return Err(NativeMemoryAdmissionError::DuplicateGroupId {
                group_id: group.group_id.to_owned(),
            });
        }
        let mapped = map_group_stats(
            group.group_id,
            group.backend_device_identity,
            group.provider,
            group.stats,
            None,
        )?;
        merge_group_observations(&mut observations, mapped)?;
    }

    build_reconciliations_from_observations(baseline, requests, &observations)
}

fn build_reconciliations_from_observations(
    baseline: &ReconciliationBaseline,
    requests: &[DomainReservationRequest],
    observations: &BTreeMap<MemoryDomainKey, DomainObservation>,
) -> Result<Vec<DomainMemoryReconciliation>, NativeMemoryAdmissionError> {
    let mut reconciliations = Vec::with_capacity(requests.len());
    for request in requests {
        let before = baseline.observations.get(&request.domain).ok_or_else(|| {
            NativeMemoryAdmissionError::MissingReconciliationStats {
                domain: request.domain.clone(),
            }
        })?;
        let after = observations.get(&request.domain).ok_or_else(|| {
            NativeMemoryAdmissionError::MissingReconciliationStats {
                domain: request.domain.clone(),
            }
        })?;
        let current_delta = observation_growth(before, after);
        let carried = baseline
            .carried_private_bytes
            .get(&request.domain)
            .copied()
            .unwrap_or(0);
        let private_delta = carried.checked_add(current_delta).ok_or(
            NativeMemoryAdmissionError::ArithmeticOverflow {
                operation: "private reconciliation delta sum",
            },
        )?;
        // Never discount quoted retained bytes merely because unrelated system
        // activity obscured the live delta. Conversely, any larger live delta
        // remains charged to this provisional transaction conservatively.
        let actual_retained_bytes = request.retained_bytes.max(private_delta);
        let actual_peak_bytes = request.peak_bytes.max(actual_retained_bytes);
        reconciliations.push(DomainMemoryReconciliation {
            domain: request.domain.clone(),
            actual_peak_bytes,
            actual_retained_bytes,
            snapshot_after: after.snapshot,
        });
    }
    Ok(reconciliations)
}

fn runtime_native_evidence(observation: &DomainObservation) -> RuntimeNativeMemoryEvidence {
    let domain_kind = Some(match observation.domain_kind {
        BackendMemoryDomainKind::HostPageable
        | BackendMemoryDomainKind::HostPinned
        | BackendMemoryDomainKind::Unified
        | BackendMemoryDomainKind::FileBacked => {
            crate::models::runtime_receipts::SafeMemoryDomainKind::SystemMemory
        }
        BackendMemoryDomainKind::DeviceLocal | BackendMemoryDomainKind::Unknown(_) => {
            crate::models::runtime_receipts::SafeMemoryDomainKind::DedicatedDevice
        }
    });
    let metric = |value: u64| {
        if value != 0 {
            RuntimeReceiptMetric::Known(value)
        } else {
            RuntimeReceiptMetric::Unavailable
        }
    };
    let backend_owned = |value| match observation.backend_owned_reliability {
        RuntimeBackendOwnedReliability::Complete => RuntimeReceiptMetric::Known(value),
        RuntimeBackendOwnedReliability::Incomplete => RuntimeReceiptMetric::Unavailable,
    };
    RuntimeNativeMemoryEvidence {
        domain_kind,
        provider: observation.provider,
        backend_owned_reliability: observation.backend_owned_reliability,
        heap_index: Some(observation.heap_index),
        total_bytes: metric(observation.total_bytes),
        budget_bytes: observation
            .budget_bytes
            .map(RuntimeReceiptMetric::Known)
            .unwrap_or(RuntimeReceiptMetric::Unavailable),
        free_bytes: RuntimeReceiptMetric::Known(observation.snapshot.free_bytes),
        used_bytes: RuntimeReceiptMetric::Known(observation.device_used_bytes),
        backend_owned_live_bytes: backend_owned(observation.backend_owned_live_bytes),
        backend_owned_cached_bytes: backend_owned(observation.backend_owned_cached_bytes),
        backend_owned_workspace_bytes: backend_owned(observation.backend_owned_workspace_bytes),
        backend_owned_high_water_bytes: backend_owned(observation.backend_owned_high_water_bytes),
        stats_generation: metric(observation.stats_generation),
        quote_generation: observation
            .quote_generation
            .map(RuntimeReceiptMetric::Known)
            .unwrap_or(RuntimeReceiptMetric::Unavailable),
        claim_flags: observation.claim_flags,
        observation_confidence: Some(observation.snapshot.confidence),
        broker_pending_bytes: RuntimeReceiptMetric::Unavailable,
        broker_committed_bytes: RuntimeReceiptMetric::Unavailable,
        broker_unreclaimable_bytes: RuntimeReceiptMetric::Unavailable,
    }
}
fn observation_growth(before: &DomainObservation, after: &DomainObservation) -> u64 {
    let physical_used_delta = after
        .device_used_bytes
        .saturating_sub(before.device_used_bytes);
    let observed_free_delta = before
        .snapshot
        .free_bytes
        .saturating_sub(after.snapshot.free_bytes);
    let backend_owned_delta = after
        .backend_owned_committed_bytes
        .saturating_sub(before.backend_owned_committed_bytes);
    physical_used_delta
        .max(observed_free_delta)
        .max(backend_owned_delta)
}

#[derive(Debug, Error)]
pub(crate) enum NativeMemoryAdmissionError {
    #[error(transparent)]
    Abi(#[from] BackendMemoryAbiError),
    #[error(transparent)]
    Planning(#[from] MemoryPlanningError),
    #[error(transparent)]
    RequestKinds(#[from] NativeMemoryRequestKindError),
    #[error("native memory admission was canceled while waiting for a provisional domain gate")]
    CanceledWhileWaitingForDomain,
    #[error("partitioned native reservation set mismatch: expected={expected}, actual={actual}")]
    PartitionedReservationSetMismatch { expected: usize, actual: usize },
    #[error(
        "native memory backend group '{group_id}' requires non-zero physical-domain headroom for opaque driver costs"
    )]
    OpaqueDriverHeadroomUnavailable { group_id: String },
    #[error("native memory backend group id must not be empty")]
    EmptyGroupId,
    #[error("native memory backend group id '{group_id}' appears more than once")]
    DuplicateGroupId { group_id: String },
    #[error("native memory backend group '{group_id}' returned no fresh stats")]
    MissingFreshStats { group_id: String },
    #[error(
        "native memory backend group '{group_id}' quote is stale: quote generation {quote_generation}, fresh stats generation {stats_generation}"
    )]
    StaleStatsGeneration {
        group_id: String,
        quote_generation: u64,
        stats_generation: u64,
    },
    #[error("native memory backend group '{group_id}' returned an incompatible claim layout")]
    IncompatibleClaimLayout { group_id: String },
    #[error("native memory backend group '{group_id}' returned an incompatible stats layout")]
    IncompatibleStatsLayout { group_id: String },
    #[error(
        "native memory backend group '{group_id}' has no phase/lifetime semantics for request {request_id}"
    )]
    MissingRequestSemantics { group_id: String, request_id: u64 },
    #[error("native memory claim has invalid confidence flags 0x{flags:08x}")]
    InvalidClaimConfidence { flags: u32 },
    #[error(
        "native memory backend group '{group_id}' has a provisional aggregate estimate spanning {candidate_domains} possible domains"
    )]
    AmbiguousProvisionalEstimateDomain {
        group_id: String,
        candidate_domains: usize,
    },
    #[error(
        "native memory backend group '{group_id}' has unpriced residual uncertainty: flags=0x{residual_flags:08x}, requests={residual_request_count}"
    )]
    UnpricedResidualUncertainty {
        group_id: String,
        residual_flags: u32,
        residual_request_count: u32,
    },
    #[error("native memory domain kind {kind} is unsupported")]
    UnsupportedDomainKind { kind: u32 },
    #[error(
        "device-local native memory heap {heap_index} has no canonical physical-device identity"
    )]
    UnprovenPhysicalDeviceIdentity { heap_index: u32 },
    #[error("native memory backend group '{group_id}' has no fresh stats for {domain}")]
    MissingClaimStats {
        group_id: String,
        domain: MemoryDomainKey,
    },
    #[error(
        "native memory backend group '{group_id}' cannot report a live budget for heap {heap_index}"
    )]
    StatsBudgetUnavailable { group_id: String, heap_index: u32 },
    #[error(
        "native memory backend group '{group_id}' returned invalid stats for heap {heap_index}"
    )]
    InvalidStatsSnapshot { group_id: String, heap_index: u32 },
    #[error(
        "native memory backend group '{group_id}' is unhealthy: health={health}, status={status}, native_error={native_error}, quarantine_generation={quarantine_generation}"
    )]
    UnhealthyBackend {
        group_id: String,
        health: u32,
        status: i32,
        native_error: i64,
        quarantine_generation: u64,
    },
    #[error("native memory backend group '{group_id}' returned unknown health {health}")]
    UnknownBackendHealth { group_id: String, health: u32 },
    #[error("post-allocation stats are missing physical domain {domain}")]
    MissingReconciliationStats { domain: MemoryDomainKey },
    #[error("native memory admission arithmetic overflowed during {operation}")]
    ArithmeticOverflow { operation: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use crate::device::execution_memory::{DeviceMemoryPolicy, ExecutionPhase};

    const GIB: u64 = 1024 * 1024 * 1024;

    #[cfg(target_os = "macos")]
    fn live_buffer_admission_plan(
        group_id: &str,
        request_id: u64,
    ) -> (ffi::GgmlBackendRaw, NativeMemoryAdmissionPlan) {
        crate::ggml_runtime::ensure_backends_loaded();
        let backend = unsafe { ffi::ggml_backend_init_best() };
        assert!(!backend.is_null(), "macOS must expose a ggml backend");
        let device = unsafe { ffi::ggml_backend_get_device(backend) };
        assert!(!device.is_null());
        let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device) };
        assert!(!buft.is_null());
        let request = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
            usage: ffi::GGML_BACKEND_BUFFER_USAGE_COMPUTE as u32,
            request_id,
            backend,
            buft,
            requested_bytes: 64 * 1024,
            ..Default::default()
        };
        let semantics = semantics(group_id, PhaseSet::ALL);
        let abi = unsafe { BackendMemoryAbi::from_backend(backend) }
            .expect("the selected backend must expose the memory ABI");
        let group = NativeQuotedBackendGroup::quote(
            group_id,
            identity(group_id),
            abi,
            vec![request],
            BTreeMap::from([(request_id, semantics.clone())]),
            semantics,
        )
        .unwrap();
        let plan = NativeMemoryAdmissionPlan::from_groups(vec![group]).unwrap();
        assert!(!plan.reservation_requests().is_empty());
        (backend, plan)
    }

    #[cfg(target_os = "macos")]
    fn provisional_gate_for_plan(
        broker: &Arc<DeviceMemoryBrokerSet>,
        plan: &NativeMemoryAdmissionPlan,
        resource_id: &str,
    ) -> DeviceMemoryReservationBatch {
        let quoted = &plan.reservation_requests()[0];
        broker
            .try_reserve_batch(vec![DomainReservationRequest {
                domain: quoted.domain.clone(),
                snapshot: quoted.snapshot,
                peak_bytes: 0,
                retained_bytes: 0,
                observed_peak_bytes: None,
                requires_reconciliation: true,
                resource_id: resource_id.to_owned(),
                cohort_id: None,
            }])
            .unwrap()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn provisional_domain_wait_requotes_after_the_competing_gate_releases() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let (backend, plan) = live_buffer_admission_plan("retry-single", 1);
        let blocker = provisional_gate_for_plan(&broker, &plan, "competing-single");
        let release = thread::spawn(move || {
            thread::sleep(Duration::from_millis(25));
            drop(blocker);
        });

        let started = Instant::now();
        let transaction = plan.try_reserve(&broker, None).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(10));
        drop(transaction);
        release.join().unwrap();
        unsafe { ffi::ggml_backend_free(backend) };
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn partitioned_provisional_domain_wait_requotes_every_child_plan() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let (backend_a, plan_a) = live_buffer_admission_plan("retry-partition-a", 1);
        let (backend_b, plan_b) = live_buffer_admission_plan("retry-partition-b", 2);
        let blocker = provisional_gate_for_plan(&broker, &plan_a, "competing-partition");
        let release = thread::spawn(move || {
            thread::sleep(Duration::from_millis(25));
            drop(blocker);
        });

        let started = Instant::now();
        let transactions =
            NativeMemoryAdmissionPlan::try_reserve_partitioned(vec![plan_a, plan_b], &broker, None)
                .unwrap();
        assert_eq!(transactions.len(), 2);
        assert!(started.elapsed() >= Duration::from_millis(10));
        drop(transactions);
        release.join().unwrap();
        unsafe {
            ffi::ggml_backend_free(backend_a);
            ffi::ggml_backend_free(backend_b);
        }
    }

    #[test]
    fn provisional_domain_wait_observes_job_cancellation_before_sleeping() {
        let canceled = Arc::new(AtomicBool::new(true));
        let _cancel = crate::ggml_runtime::InheritedJobCancelGuard::arm(&canceled);
        let mut retry_delay = Duration::from_millis(32);

        let error = NativeMemoryAdmissionPlan::wait_for_domain_busy_retry(
            Instant::now() + Duration::from_secs(30),
            &mut retry_delay,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            NativeMemoryAdmissionError::CanceledWhileWaitingForDomain
        ));
        assert_eq!(retry_delay, Duration::from_millis(32));
        assert!(canceled.load(Ordering::Acquire));
    }

    #[test]
    fn provisional_domain_wait_preserves_the_original_busy_error_after_deadline() {
        let mut retry_delay = Duration::from_millis(1);

        assert!(
            !NativeMemoryAdmissionPlan::wait_for_domain_busy_retry(
                Instant::now(),
                &mut retry_delay,
            )
            .unwrap()
        );
        assert_eq!(retry_delay, Duration::from_millis(1));
    }

    fn identity(value: &str) -> PhysicalDeviceKey {
        PhysicalDeviceKey::new(value).unwrap()
    }

    fn dedicated_request(
        domain: MemoryDomainKey,
        bytes: u64,
        resource_id: &str,
    ) -> DomainReservationRequest {
        DomainReservationRequest {
            domain,
            snapshot: DeviceMemorySnapshot {
                free_bytes: 8 * GIB,
                total_bytes: 8 * GIB,
                confidence: MemoryObservationConfidence::DeviceSnapshot,
            },
            peak_bytes: bytes,
            retained_bytes: bytes,
            observed_peak_bytes: None,
            requires_reconciliation: false,
            resource_id: resource_id.to_owned(),
            cohort_id: None,
        }
    }

    fn owner_attached_dedicated_reservation(
        physical_device: &str,
        resource_id: &str,
    ) -> (
        Arc<DeviceMemoryBrokerSet>,
        MemoryDomainKey,
        DeviceMemoryReservationBatch,
    ) {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let domain = MemoryDomainKey::DedicatedDevice {
            physical_device: identity(physical_device),
            heap_index: 0,
        };
        let reservation = broker
            .try_reserve_batch(vec![dedicated_request(domain.clone(), GIB, resource_id)])
            .unwrap();
        (broker, domain, reservation)
    }

    fn domain(kind: u32, heap_index: u32, uuid: [u8; 16]) -> ffi::GgmlBackendMemoryDomainIdV1 {
        ffi::GgmlBackendMemoryDomainIdV1 {
            physical_device_uuid: uuid,
            heap_index,
            kind,
        }
    }

    fn stats(
        domain: ffi::GgmlBackendMemoryDomainIdV1,
        generation: u64,
        free: u64,
        total: u64,
    ) -> ffi::GgmlBackendMemoryStatsV1 {
        ffi::GgmlBackendMemoryStatsV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32,
            domain,
            generation,
            total_bytes: total,
            budget_bytes: total,
            device_free_bytes: free,
            device_used_bytes: total - free,
            health: ffi::GGML_BACKEND_MEMORY_HEALTHY,
            ..Default::default()
        }
    }

    fn claim(
        request_id: u64,
        domain: ffi::GgmlBackendMemoryDomainIdV1,
        flags: u32,
        requested: u64,
        before: u64,
        peak: u64,
        retained_after: u64,
    ) -> ffi::GgmlBackendMemoryClaimV1 {
        ffi::GgmlBackendMemoryClaimV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryClaimV1>() as u32,
            flags,
            request_id,
            domain,
            payload_requested_bytes: requested,
            committed_before_bytes: before,
            committed_after_upper_bytes: retained_after,
            commit_peak_extra_upper_bytes: peak,
            resident_after_upper_bytes: retained_after,
            retained_after_use_upper_bytes: retained_after,
            ..Default::default()
        }
    }

    fn semantics(resource: &str, phases: PhaseSet) -> NativeMemoryClaimSemantics {
        NativeMemoryClaimSemantics {
            resource_id: resource.to_owned(),
            lifetime: AllocationLifetime::SessionResident,
            phases,
        }
    }

    fn view<'a>(
        group_id: &'a str,
        identity: &'a PhysicalDeviceKey,
        quote: &'a ffi::GgmlBackendMemoryQuoteV1,
        claims: &'a [ffi::GgmlBackendMemoryClaimV1],
        stats: &'a [ffi::GgmlBackendMemoryStatsV1],
        request_semantics: &'a BTreeMap<u64, NativeMemoryClaimSemantics>,
        shared_semantics: &'a NativeMemoryClaimSemantics,
    ) -> NativeGroupView<'a> {
        view_with_provider(
            group_id,
            identity,
            quote,
            claims,
            stats,
            request_semantics,
            shared_semantics,
            crate::device::execution_route::ExecutionProvider::Cpu,
        )
    }

    fn view_with_provider<'a>(
        group_id: &'a str,
        identity: &'a PhysicalDeviceKey,
        quote: &'a ffi::GgmlBackendMemoryQuoteV1,
        claims: &'a [ffi::GgmlBackendMemoryClaimV1],
        stats: &'a [ffi::GgmlBackendMemoryStatsV1],
        request_semantics: &'a BTreeMap<u64, NativeMemoryClaimSemantics>,
        shared_semantics: &'a NativeMemoryClaimSemantics,
        provider: crate::device::execution_route::ExecutionProvider,
    ) -> NativeGroupView<'a> {
        NativeGroupView {
            group_id,
            backend_device_identity: identity,
            provider,
            quote,
            claims,
            stats,
            request_semantics,
            shared_semantics,
        }
    }

    #[test]
    fn same_group_repeated_domain_observations_use_max_high_water() {
        let identity = identity("cpu:repeat");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE, 0, [0; 16]);
        let mut first = stats(native_domain, 1, 8 * GIB, 16 * GIB);
        first.backend_owned_high_water_bytes = GIB;
        let mut second = first;
        second.backend_owned_high_water_bytes = 2 * GIB;
        let claims = [claim(
            1,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
            1,
            0,
            1,
            1,
        )];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 1,
            ..Default::default()
        };
        let metadata = BTreeMap::from([(1, semantics("repeat", PhaseSet::ALL))]);
        let shared = semantics("shared", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "repeat",
            &identity,
            &quote,
            &claims,
            &[first, second],
            &metadata,
            &shared,
        )])
        .unwrap();
        assert_eq!(
            built
                .reconciliation_baseline
                .observations
                .get(&MemoryDomainKey::SystemMemory)
                .unwrap()
                .backend_owned_high_water_bytes,
            2 * GIB
        );
    }

    #[test]
    fn cross_backend_domain_high_water_is_checked_additive() {
        let first_identity = identity("cpu:first");
        let second_identity = identity("cpu:second");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE, 0, [0; 16]);
        let mut first_stats = stats(native_domain, 1, 8 * GIB, 16 * GIB);
        first_stats.backend_owned_high_water_bytes = GIB;
        let mut second_stats = first_stats;
        second_stats.backend_owned_high_water_bytes = 2 * GIB;
        let first_claim = [claim(
            1,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
            1,
            0,
            1,
            1,
        )];
        let second_claim = [claim(
            2,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
            2,
            0,
            2,
            2,
        )];
        let first_quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 1,
            ..Default::default()
        };
        let second_quote = first_quote;
        let first_metadata = BTreeMap::from([(1, semantics("first", PhaseSet::ALL))]);
        let second_metadata = BTreeMap::from([(2, semantics("second", PhaseSet::ALL))]);
        let first_shared = semantics("first-shared", PhaseSet::ALL);
        let second_shared = semantics("second-shared", PhaseSet::ALL);
        let built = build_from_views(&[
            view(
                "first",
                &first_identity,
                &first_quote,
                &first_claim,
                &[first_stats],
                &first_metadata,
                &first_shared,
            ),
            view(
                "second",
                &second_identity,
                &second_quote,
                &second_claim,
                &[second_stats],
                &second_metadata,
                &second_shared,
            ),
        ])
        .unwrap();
        assert_eq!(
            built
                .reconciliation_baseline
                .observations
                .get(&MemoryDomainKey::SystemMemory)
                .unwrap()
                .backend_owned_high_water_bytes,
            3 * GIB
        );
    }

    #[test]
    fn incomplete_provider_accounting_never_projects_partial_backend_owned_counters() {
        let identity = identity("cuda:incomplete");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL, 0, [0x44; 16]);
        let mut native_stats = stats(native_domain, 1, 8 * GIB, 16 * GIB);
        native_stats.backend_owned_live_bytes = GIB;
        native_stats.backend_owned_cached_bytes = GIB;
        native_stats.backend_owned_workspace_bytes = GIB;
        native_stats.backend_owned_high_water_bytes = 3 * GIB;
        let claims = [claim(
            1,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
            1,
            0,
            1,
            1,
        )];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 1,
            ..Default::default()
        };
        let metadata = BTreeMap::from([(1, semantics("cuda", PhaseSet::ALL))]);
        let shared = semantics("cuda-shared", PhaseSet::ALL);
        let built = build_from_views(&[view_with_provider(
            "cuda",
            &identity,
            &quote,
            &claims,
            &[native_stats],
            &metadata,
            &shared,
            crate::device::execution_route::ExecutionProvider::Cuda,
        )])
        .unwrap();
        let evidence = runtime_native_evidence(
            built
                .reconciliation_baseline
                .observations
                .values()
                .next()
                .unwrap(),
        );
        assert_eq!(
            evidence.provider,
            Some(crate::device::execution_route::ExecutionProvider::Cuda)
        );
        assert_eq!(
            evidence.backend_owned_reliability,
            RuntimeBackendOwnedReliability::Incomplete
        );
        assert_eq!(
            evidence.backend_owned_high_water_bytes,
            RuntimeReceiptMetric::Unavailable
        );
        assert_eq!(
            evidence.backend_owned_live_bytes,
            RuntimeReceiptMetric::Unavailable
        );
    }
    #[test]
    fn discrete_and_host_claims_merge_into_one_atomic_two_domain_batch() {
        let uuid = [0xabu8; 16];
        let device = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL, 2, uuid);
        let host = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED, 9, uuid);
        let native_claims = [
            claim(
                1,
                device,
                ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
                GIB,
                GIB / 4,
                GIB,
                3 * GIB / 4,
            ),
            claim(
                2,
                host,
                ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER,
                GIB / 8,
                0,
                GIB / 8,
                GIB / 8,
            ),
        ];
        let native_stats = [
            stats(device, 7, 6 * GIB, 8 * GIB),
            stats(host, 7, 24 * GIB, 32 * GIB),
        ];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 7,
            ..Default::default()
        };
        let metadata = BTreeMap::from([
            (1, semantics("gpu-arena", PhaseSet::ALL)),
            (
                2,
                semantics("host-transfer", PhaseSet::one(ExecutionPhase::ModelLoad)),
            ),
        ]);
        let shared = semantics("shared", PhaseSet::ALL);
        let backend_identity = identity("vulkan:device-0");
        let built = build_from_views(&[view(
            "vk0",
            &backend_identity,
            &quote,
            &native_claims,
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();

        assert_eq!(built.requests.len(), 2);
        let dedicated = MemoryDomainKey::DedicatedDevice {
            physical_device: identity("uuid:abababababababababababababababab"),
            heap_index: 2,
        };
        let gpu = built
            .requests
            .iter()
            .find(|request| request.domain == dedicated)
            .unwrap();
        assert_eq!(gpu.peak_bytes, GIB);
        assert_eq!(gpu.retained_bytes, GIB / 2);
        assert!(
            built
                .requests
                .iter()
                .any(|request| request.domain == MemoryDomainKey::SystemMemory)
        );
    }

    #[test]
    fn uma_and_all_host_kinds_share_system_memory() {
        let identity = identity("metal:registry-42");
        for kind in [
            ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE,
            ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED,
            ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED,
            ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED,
        ] {
            assert_eq!(
                map_native_domain(&domain(kind, 3, [0x11; 16]), &identity).unwrap(),
                MemoryDomainKey::SystemMemory
            );
        }

        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED, 0, [0; 16]);
        let native_claims = [claim(
            1,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER,
            GIB,
            0,
            GIB,
            GIB,
        )];
        let native_stats = [stats(native_domain, 4, 12 * GIB, 16 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 4,
            ..Default::default()
        };
        let metadata = BTreeMap::from([(1, semantics("metal-arena", PhaseSet::ALL))]);
        let shared = semantics("metal-private", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "metal",
            &identity,
            &quote,
            &native_claims,
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();
        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].domain, MemoryDomainKey::SystemMemory);
        assert_eq!(
            built.requests[0].snapshot.confidence,
            MemoryObservationConfidence::WorkingSetBudget
        );
    }

    #[test]
    fn file_backed_reuse_quote_contributes_zero_incremental_system_memory() {
        // After pack-weight residency acquires the mapping charge, the HOST_IMPORT
        // quote sets currently_allocated_bytes = requested so the backend reports
        // peak/retained incremental 0. This layer must honor those numbers rather
        // than invent a blanket FILE_BACKED zeroing rule (distinct packs still
        // charge via residency).
        let identity = identity("cpu:0");
        let file_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED, 0, [0; 16]);
        let host_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE, 0, [0; 16]);
        let pack_bytes = 5 * GIB;
        let native_claims = [
            // peak=0, retained_after=before => incremental retained 0 (reuse)
            claim(
                1,
                file_domain,
                ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER
                    | ffi::GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED,
                pack_bytes,
                pack_bytes,
                0,
                pack_bytes,
            ),
            claim(
                2,
                host_domain,
                ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
                GIB / 4,
                0,
                GIB / 4,
                GIB / 4,
            ),
        ];
        let native_stats = [
            stats(file_domain, 3, 8 * GIB, 16 * GIB),
            stats(host_domain, 3, 8 * GIB, 16 * GIB),
        ];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 3,
            ..Default::default()
        };
        let metadata = BTreeMap::from([
            (1, semantics("pack-weight-host-import", PhaseSet::ALL)),
            (2, semantics("direct-graph-buffer", PhaseSet::ALL)),
        ]);
        let shared = semantics("context-buffer", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "cpu-weight",
            &identity,
            &quote,
            &native_claims,
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();
        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].domain, MemoryDomainKey::SystemMemory);
        assert_eq!(
            built.requests[0].peak_bytes,
            GIB / 4,
            "reused file-backed import must not add to peak"
        );
        assert_eq!(
            built.requests[0].retained_bytes,
            GIB / 4,
            "reused file-backed import must not add to retained"
        );
        let file_claim = built
            .claims
            .iter()
            .find(|claim| claim.resource_id == "pack-weight-host-import")
            .expect("file-backed claim retained");
        assert_eq!(file_claim.requested_bytes, pack_bytes);
        assert_eq!(file_claim.incremental_peak_bytes, Some(0));
        assert_eq!(file_claim.incremental_retained_bytes, Some(0));
    }

    #[test]
    fn file_backed_first_bind_quote_still_charges_incremental_bytes() {
        // Without residency reuse markers a first-bind FILE_BACKED quote that
        // reports non-zero peak must still flow into SystemMemory. Distinct
        // packs rely on this path when residency is not yet held.
        let identity = identity("cpu:0");
        let file_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED, 0, [0; 16]);
        let pack_bytes = 3 * GIB;
        let native_claims = [claim(
            1,
            file_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER
                | ffi::GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED,
            pack_bytes,
            0,
            pack_bytes,
            pack_bytes,
        )];
        let native_stats = [stats(file_domain, 2, 12 * GIB, 16 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 2,
            ..Default::default()
        };
        let metadata = BTreeMap::from([(1, semantics("pack-weight-host-import", PhaseSet::ALL))]);
        let shared = semantics("context-buffer", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "cpu-weight",
            &identity,
            &quote,
            &native_claims,
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();
        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].peak_bytes, pack_bytes);
        assert_eq!(built.requests[0].retained_bytes, pack_bytes);
    }

    #[test]
    fn zero_device_local_identity_fails_closed_instead_of_splitting_one_physical_gpu() {
        let identity = identity("cuda:physical-device-7");
        let error = map_native_domain(
            &domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL, 5, [0; 16]),
            &identity,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            NativeMemoryAdmissionError::UnprovenPhysicalDeviceIdentity { heap_index: 5 }
        ));
    }

    #[test]
    fn multiple_backend_groups_collapse_before_the_single_broker_batch() {
        let cpu_identity = identity("cpu:0");
        let metal_identity = identity("metal:0");
        let cpu_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE, 0, [0; 16]);
        let metal_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED, 0, [0; 16]);
        let cpu_claims = [claim(
            1,
            cpu_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_EXACT,
            GIB,
            0,
            GIB,
            GIB,
        )];
        let metal_claims = [claim(
            2,
            metal_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER,
            GIB / 2,
            0,
            GIB / 2,
            GIB / 2,
        )];
        let cpu_stats = [stats(cpu_domain, 5, 24 * GIB, 32 * GIB)];
        let metal_stats = [stats(metal_domain, 8, 12 * GIB, 16 * GIB)];
        let cpu_quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 5,
            ..Default::default()
        };
        let metal_quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 8,
            ..Default::default()
        };
        let cpu_metadata = BTreeMap::from([(1, semantics("cpu-work", PhaseSet::ALL))]);
        let metal_metadata = BTreeMap::from([(2, semantics("metal-work", PhaseSet::ALL))]);
        let cpu_shared = semantics("cpu-shared", PhaseSet::ALL);
        let metal_shared = semantics("metal-shared", PhaseSet::ALL);
        let built = build_from_views(&[
            view(
                "cpu",
                &cpu_identity,
                &cpu_quote,
                &cpu_claims,
                &cpu_stats,
                &cpu_metadata,
                &cpu_shared,
            ),
            view(
                "metal",
                &metal_identity,
                &metal_quote,
                &metal_claims,
                &metal_stats,
                &metal_metadata,
                &metal_shared,
            ),
        ])
        .unwrap();

        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].domain, MemoryDomainKey::SystemMemory);
        assert_eq!(built.requests[0].peak_bytes, 3 * GIB / 2);
        assert_eq!(built.requests[0].snapshot.total_bytes, 16 * GIB);
        assert_eq!(built.requests[0].snapshot.free_bytes, 12 * GIB);
    }

    #[test]
    fn stale_quote_generation_is_a_typed_failure() {
        let identity = identity("cpu:0");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE, 0, [0; 16]);
        let native_stats = [stats(native_domain, 12, 8 * GIB, 16 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            stats_generation: 11,
            ..Default::default()
        };
        let metadata = BTreeMap::new();
        let shared = semantics("cpu-private", PhaseSet::ALL);
        let error = build_from_views(&[view(
            "cpu",
            &identity,
            &quote,
            &[],
            &native_stats,
            &metadata,
            &shared,
        )])
        .err()
        .expect("stale generation must fail");
        assert!(matches!(
            error,
            NativeMemoryAdmissionError::StaleStatsGeneration {
                quote_generation: 11,
                stats_generation: 12,
                ..
            }
        ));
    }

    #[test]
    fn provisional_residual_survives_as_reconciliation_required() {
        let identity = identity("vulkan:pci1");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL, 0, [0x22; 16]);
        let native_stats = [stats(native_domain, 9, 6 * GIB, 8 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            flags: ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL
                | ffi::GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY,
            residual_flags: ffi::GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE,
            residual_request_count: 1,
            provisional_requested_upper_bytes: GIB / 4,
            stats_generation: 9,
            ..Default::default()
        };
        let metadata = BTreeMap::new();
        let shared = semantics(
            "graph-private",
            PhaseSet::one(ExecutionPhase::DecoderPrefill),
        );
        let built = build_from_views(&[view(
            "vk-residual",
            &identity,
            &quote,
            &[],
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();

        assert_eq!(built.claims.len(), 1);
        assert_eq!(built.claims[0].confidence, QuoteConfidence::Provisional);
        assert_eq!(built.claims[0].incremental_peak_bytes, Some(GIB / 4));
        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].peak_bytes, GIB / 4);
        assert!(built.requests[0].requires_reconciliation);
        assert_eq!(built.evidence[0].residual_request_count, 1);
        assert_eq!(
            built.evidence[0].residual_flags,
            ffi::GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE
        );
    }

    #[test]
    fn entirely_unpriced_residual_becomes_zero_byte_exclusive_reconciliation_marker() {
        let identity = identity("metal:unpriced");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED, 0, [0; 16]);
        let native_stats = [stats(native_domain, 3, 12 * GIB, 16 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            flags: ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL
                | ffi::GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY,
            residual_flags: ffi::GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE,
            residual_request_count: 1,
            stats_generation: 3,
            ..Default::default()
        };
        let metadata = BTreeMap::new();
        let shared = semantics("metal-private", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "metal",
            &identity,
            &quote,
            &[],
            &native_stats,
            &metadata,
            &shared,
        )])
        .unwrap();
        assert_eq!(built.requests.len(), 1);
        assert_eq!(built.requests[0].peak_bytes, 0);
        assert_eq!(built.requests[0].retained_bytes, 0);
        assert!(built.requests[0].requires_reconciliation);
    }

    #[test]
    fn post_allocation_builder_uses_live_delta_and_never_discounts_quote() {
        let identity = identity("cuda:reconcile");
        let native_domain = domain(ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL, 0, [0x44; 16]);
        let native_claims = [claim(
            1,
            native_domain,
            ffi::GGML_BACKEND_MEMORY_CLAIM_DRIVER_ESTIMATE
                | ffi::GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL,
            GIB / 4,
            0,
            GIB / 4,
            GIB / 8,
        )];
        let before_stats = [stats(native_domain, 2 * GIB, 6 * GIB, 8 * GIB)];
        let quote = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            flags: ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL,
            provisional_requested_upper_bytes: GIB / 4,
            stats_generation: 2 * GIB,
            ..Default::default()
        };
        let metadata = BTreeMap::from([(1, semantics("cuda-arena", PhaseSet::ALL))]);
        let shared = semantics("cuda-private", PhaseSet::ALL);
        let built = build_from_views(&[view(
            "cuda",
            &identity,
            &quote,
            &native_claims,
            &before_stats,
            &metadata,
            &shared,
        )])
        .unwrap();
        let after_stats = [stats(native_domain, 5 * GIB / 2, 11 * GIB / 2, 8 * GIB)];
        let post = [PostStatsView {
            group_id: "cuda",
            backend_device_identity: &identity,
            provider: crate::device::execution_route::ExecutionProvider::Cuda,
            stats: &after_stats,
        }];
        let reconciled =
            build_reconciliations(&built.reconciliation_baseline, &built.requests, &post).unwrap();
        assert_eq!(reconciled.len(), 1);
        assert_eq!(reconciled[0].actual_retained_bytes, GIB / 2);
        assert_eq!(reconciled[0].actual_peak_bytes, GIB / 2);
        assert_eq!(reconciled[0].snapshot_after.free_bytes, 11 * GIB / 2);
    }

    #[test]
    fn deferred_private_rebase_excludes_already_accounted_scheduler_growth() {
        let domain = MemoryDomainKey::DedicatedDevice {
            physical_device: identity("uuid:44444444444444444444444444444444"),
            heap_index: 0,
        };
        // Private reserve grew its pool by 1/4 GiB before scheduler commit.
        // That delta was checkpointed, then the baseline was moved past a
        // separate 1 GiB scheduler allocation. First compute grows only the
        // private pool by another 1/2 GiB.
        let baseline = ReconciliationBaseline {
            observations: BTreeMap::from([(
                domain.clone(),
                DomainObservation {
                    domain_kind: BackendMemoryDomainKind::DeviceLocal,
                    provider: Some(crate::device::execution_route::ExecutionProvider::Cuda),
                    backend_owned_reliability: RuntimeBackendOwnedReliability::Incomplete,
                    heap_index: 0,
                    total_bytes: 8 * GIB,
                    budget_bytes: Some(8 * GIB),
                    stats_generation: 1,
                    snapshot: DeviceMemorySnapshot {
                        free_bytes: 19 * GIB / 4,
                        total_bytes: 8 * GIB,
                        confidence: MemoryObservationConfidence::DeviceSnapshot,
                    },
                    device_used_bytes: 13 * GIB / 4,
                    backend_owned_committed_bytes: GIB / 4,
                    backend_owned_live_bytes: GIB / 4,
                    backend_owned_cached_bytes: 0,
                    backend_owned_workspace_bytes: 0,
                    backend_owned_high_water_bytes: GIB / 4,
                    claim_flags: 0,
                    quote_generation: None,
                },
            )]),
            carried_private_bytes: BTreeMap::from([(domain.clone(), GIB / 4)]),
        };
        let after = BTreeMap::from([(
            domain.clone(),
            DomainObservation {
                domain_kind: BackendMemoryDomainKind::DeviceLocal,
                provider: Some(crate::device::execution_route::ExecutionProvider::Cuda),
                backend_owned_reliability: RuntimeBackendOwnedReliability::Incomplete,
                heap_index: 0,
                total_bytes: 8 * GIB,
                budget_bytes: Some(8 * GIB),
                stats_generation: 1,
                snapshot: DeviceMemorySnapshot {
                    free_bytes: 17 * GIB / 4,
                    total_bytes: 8 * GIB,
                    confidence: MemoryObservationConfidence::DeviceSnapshot,
                },
                device_used_bytes: 15 * GIB / 4,
                backend_owned_committed_bytes: 3 * GIB / 4,
                backend_owned_live_bytes: 3 * GIB / 4,
                backend_owned_cached_bytes: 0,
                backend_owned_workspace_bytes: 0,
                backend_owned_high_water_bytes: 3 * GIB / 4,
                claim_flags: 0,
                quote_generation: None,
            },
        )]);
        let request = DomainReservationRequest {
            domain: domain.clone(),
            snapshot: DeviceMemorySnapshot {
                free_bytes: 6 * GIB,
                total_bytes: 8 * GIB,
                confidence: MemoryObservationConfidence::DeviceSnapshot,
            },
            peak_bytes: 0,
            retained_bytes: 0,
            observed_peak_bytes: None,
            requires_reconciliation: true,
            resource_id: "cuda-private".to_owned(),
            cohort_id: None,
        };

        let reconciled =
            build_reconciliations_from_observations(&baseline, &[request], &after).unwrap();
        assert_eq!(reconciled[0].actual_retained_bytes, 3 * GIB / 4);
        assert_eq!(reconciled[0].actual_peak_bytes, 3 * GIB / 4);
    }

    #[test]
    fn mixed_engine_and_backend_private_requests_are_rejected_before_mutation() {
        let engine = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
            ..Default::default()
        };
        let private = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE,
            ..Default::default()
        };
        assert_eq!(
            classify_request_kinds([&engine, &private]).unwrap_err(),
            NativeMemoryRequestKindError::Mixed
        );
    }

    #[test]
    fn allocation_wrapper_drops_native_owner_before_refunding_lease() {
        struct OwnerDropProbe {
            broker: Arc<DeviceMemoryBrokerSet>,
            observed_committed: Arc<Mutex<Option<u64>>>,
        }

        impl NativeMemoryOwner for OwnerDropProbe {
            fn release_native(&mut self) -> BackendReleaseProof {
                let committed = self
                    .broker
                    .usage(&MemoryDomainKey::SystemMemory)
                    .committed_bytes;
                *self.observed_committed.lock().unwrap() = Some(committed);
                BackendReleaseProof::Proven
            }
        }

        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let mut lease = broker
            .try_reserve_batch(vec![DomainReservationRequest {
                domain: MemoryDomainKey::SystemMemory,
                snapshot: DeviceMemorySnapshot {
                    free_bytes: 8 * GIB,
                    total_bytes: 8 * GIB,
                    confidence: MemoryObservationConfidence::DeviceSnapshot,
                },
                peak_bytes: GIB,
                retained_bytes: GIB,
                observed_peak_bytes: None,
                requires_reconciliation: false,
                resource_id: "native-owner-order".to_owned(),
                cohort_id: None,
            }])
            .unwrap();
        lease.commit_quoted().unwrap();
        let observed_committed = Arc::new(Mutex::new(None));
        let allocation = NativeMemoryAllocation {
            owner: Some(OwnerDropProbe {
                broker: Arc::clone(&broker),
                observed_committed: Arc::clone(&observed_committed),
            }),
            reservation: Some(lease),
        };

        drop(allocation);
        assert_eq!(*observed_committed.lock().unwrap(), Some(GIB));
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    #[test]
    fn backend_private_lease_refunds_only_after_every_owner_clone_drops() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let mut reservation = broker
            .try_reserve_batch(vec![DomainReservationRequest {
                domain: MemoryDomainKey::SystemMemory,
                snapshot: DeviceMemorySnapshot {
                    free_bytes: 8 * GIB,
                    total_bytes: 8 * GIB,
                    confidence: MemoryObservationConfidence::DeviceSnapshot,
                },
                peak_bytes: GIB,
                retained_bytes: GIB,
                observed_peak_bytes: None,
                requires_reconciliation: false,
                resource_id: "cross-backend-private".to_owned(),
                cohort_id: None,
            }])
            .unwrap();
        reservation.commit_quoted().unwrap();
        let first_owner = NativeBackendPrivateMemoryLease {
            inner: Rc::new(RefCell::new(NativeBackendPrivateMemoryLeaseInner {
                transaction: None,
                committed_reservation: Some(reservation),
                committed: true,
                quarantined: false,
            })),
        };
        let second_owner = first_owner.clone();

        drop(first_owner);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            GIB
        );
        drop(second_owner);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    #[test]
    fn exact_quote_commits_without_calling_post_stats() {
        let broker = Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        }));
        let mut lease = broker
            .try_reserve_batch(vec![DomainReservationRequest {
                domain: MemoryDomainKey::SystemMemory,
                snapshot: DeviceMemorySnapshot {
                    free_bytes: 8 * GIB,
                    total_bytes: 8 * GIB,
                    confidence: MemoryObservationConfidence::DeviceSnapshot,
                },
                peak_bytes: GIB,
                retained_bytes: GIB / 2,
                observed_peak_bytes: None,
                requires_reconciliation: false,
                resource_id: "exact-no-post-stats".to_owned(),
                cohort_id: None,
            }])
            .unwrap();
        let stats_called = Cell::new(false);
        let result = finalize_reservation_with(&mut lease, || {
            stats_called.set(true);
            Err::<Vec<DomainMemoryReconciliation>, _>("must not be called")
        });

        assert!(result.is_ok());
        assert!(!stats_called.get());
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            GIB / 2
        );
    }

    #[test]
    fn opaque_driver_cost_flag_requires_nonzero_domain_headroom_policy() {
        let plan = NativeMemoryAdmissionPlan {
            groups: Vec::new(),
            claims: Vec::new(),
            requests: Vec::new(),
            evidence: vec![NativeMemoryQuoteEvidence {
                group_id: "metal-private".to_owned(),
                quote_flags:
                    ffi::GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM,
                residual_flags: 0,
                residual_request_count: 0,
                provisional_requested_upper_bytes: 0,
                claim_flags: Vec::new(),
            }],
            reconciliation_baseline: ReconciliationBaseline {
                observations: BTreeMap::new(),
                carried_private_bytes: BTreeMap::new(),
            },
        };
        let no_headroom = DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 0,
        });
        assert!(matches!(
            plan.require_opaque_driver_headroom(&no_headroom),
            Err(NativeMemoryAdmissionError::OpaqueDriverHeadroomUnavailable { group_id })
                if group_id == "metal-private"
        ));

        let protected = DeviceMemoryBrokerSet::new(DeviceMemoryPolicy {
            maximum_owned_basis_points: 10_000,
            minimum_headroom_bytes: 256 * 1024 * 1024,
        });
        assert!(plan.require_opaque_driver_headroom(&protected).is_ok());
    }

    #[test]
    fn owner_attached_failure_before_native_mutation_refunds_dedicated_domain() {
        let (broker, dedicated, mut reservation) = owner_attached_dedicated_reservation(
            "uuid:55555555555555555555555555555555",
            "scheduler-validation",
        );

        let error = owner_attached_native_commit_error(
            &mut reservation,
            TestCommitFailure {
                message: "stale graph",
                quarantine: false,
            },
        );
        assert!(matches!(
            error,
            NativeOwnerAttachedMemoryError::NativeCommit {
                source: TestCommitFailure {
                    message: "stale graph",
                    ..
                },
                quarantined: false,
            }
        ));
        drop(reservation);

        let usage = broker.usage(&dedicated);
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, 0);
        assert!(!usage.quarantined);
        assert!(
            broker
                .try_reserve_batch(vec![dedicated_request(dedicated, GIB, "scheduler-retry")])
                .is_ok()
        );
    }

    #[test]
    fn owner_attached_failure_after_possible_native_mutation_quarantines_dedicated_domain() {
        let (broker, dedicated, mut reservation) = owner_attached_dedicated_reservation(
            "uuid:66666666666666666666666666666666",
            "scheduler-mutation",
        );

        let error = owner_attached_native_commit_error(
            &mut reservation,
            TestCommitFailure {
                message: "allocation changed",
                quarantine: true,
            },
        );
        assert!(matches!(
            error,
            NativeOwnerAttachedMemoryError::NativeCommit {
                source: TestCommitFailure {
                    message: "allocation changed",
                    ..
                },
                quarantined: true,
            }
        ));
        drop(reservation);

        let usage = broker.usage(&dedicated);
        assert_eq!(usage.pending_bytes, 0);
        assert_eq!(usage.unreclaimable_bytes, GIB);
        assert!(usage.quarantined);
        assert!(matches!(
            broker.try_reserve_batch(vec![dedicated_request(dedicated, 1, "scheduler-blocked")]),
            Err(MemoryPlanningError::DeviceQuarantined { .. })
        ));
    }

    #[derive(Debug)]
    struct TestCommitFailure {
        message: &'static str,
        quarantine: bool,
    }

    impl NativeOwnerAttachedCommitOutcome for TestCommitFailure {
        fn requires_quarantine(&self) -> bool {
            self.quarantine
        }
    }
}
