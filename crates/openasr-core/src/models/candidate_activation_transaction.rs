//! Backend-neutral candidate activation transaction primitives.
//!
//! This module stops at the transaction boundary. It does not resolve a
//! backend, infer a device class, attach to an execution context, or publish
//! through an existing runtime service. Callers supply already-resolved facts,
//! reservations, staged owners, attestation contracts, and a journal factory.
//! Production NES attempts and default-model activation both enter here;
//! family modules must not construct a second transaction.
#![allow(dead_code, private_bounds, private_interfaces, clippy::type_complexity)]

use std::marker::PhantomData;

use crate::device::execution_policy::ExecutionCandidate;
use crate::ggml_runtime::{GgmlDecodeOutputPlan, GgmlDecodeReuseMode, ResolvedFamilyRuntimeInput};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DefaultModelResidentComponentPlan {
    pub(crate) component: &'static str,
    pub(crate) variant: &'static str,
    pub(crate) phase: crate::arch::runtime_footprint::ResidentPhase,
    pub(crate) lifetime: crate::arch::runtime_footprint::ResidentLifetime,
    pub(crate) dependencies: Vec<&'static str>,
    pub(crate) representations: Vec<crate::arch::runtime_footprint::ResidentRepresentation>,
    pub(crate) checkout: crate::arch::runtime_footprint::ResidentCheckout,
    pub(crate) placement: crate::arch::runtime_footprint::ResidentPlacementVariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DefaultModelResidentTopologyPlan {
    pub(crate) architecture: &'static str,
    pub(crate) components: Vec<DefaultModelResidentComponentPlan>,
    pub(crate) dependency_order: Vec<&'static str>,
}

/// The externally visible lifecycle of one candidate activation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationStage {
    Prepared,
    Reserved,
    Materialized,
    AttestationPending,
    Attested,
    Committed,
    RolledBack,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidTransition {
    pub from: ActivationStage,
    pub to: ActivationStage,
}

impl ActivationStage {
    /// Check an edge independently of a transaction. The typestate wrappers
    /// make the same invalid edges unrepresentable at call sites.
    pub const fn transition(self, to: Self) -> Result<(), InvalidTransition> {
        let valid = matches!(
            (self, to),
            (Self::Prepared, Self::Reserved | Self::RolledBack)
                | (
                    Self::Reserved,
                    Self::Materialized | Self::RolledBack | Self::Quarantined
                )
                | (
                    Self::Materialized,
                    Self::AttestationPending | Self::RolledBack | Self::Quarantined,
                )
                | (
                    Self::AttestationPending,
                    Self::Attested | Self::RolledBack | Self::Quarantined
                )
                | (
                    Self::Attested,
                    Self::Committed | Self::RolledBack | Self::Quarantined
                )
        );

        if valid {
            Ok(())
        } else {
            Err(InvalidTransition { from: self, to })
        }
    }
}

/// Immutable facts resolved by the caller.
///
/// `Plan`, `Lane`, and `Identity` are opaque to this module. No backend,
/// `is_gpu_class`, `Auto`, or output-plan conversion is inspected here. The
/// exact values supplied by the caller are retained and returned unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExecutionFacts<Plan, Lane, Identity> {
    plan: Plan,
    exact_lane: Lane,
    identity: Identity,
}

impl<Plan, Lane, Identity> ResolvedExecutionFacts<Plan, Lane, Identity> {
    pub const fn new(plan: Plan, exact_lane: Lane, identity: Identity) -> Self {
        Self {
            plan,
            exact_lane,
            identity,
        }
    }

    pub const fn plan(&self) -> &Plan {
        &self.plan
    }

    pub const fn exact_lane(&self) -> &Lane {
        &self.exact_lane
    }

    pub const fn identity(&self) -> &Identity {
        &self.identity
    }

    pub fn into_parts(self) -> (Plan, Lane, Identity) {
        (self.plan, self.exact_lane, self.identity)
    }
}

/// A staged owner that has not yet been published to a shared registry.
/// Transactions invoke the operations in reverse construction order.
pub trait StagedOwner {
    type Error;

    fn teardown(&mut self) -> Result<(), Self::Error>;
    fn quarantine(&mut self) -> Result<(), Self::Error>;
}

/// The reservation token supplied by the admission layer.
pub trait ActivationReservation {
    type Error;

    fn release(&mut self) -> Result<(), Self::Error>;
    fn quarantine(&mut self) -> Result<(), Self::Error>;
}

struct EmptyOwner;

impl StagedOwner for EmptyOwner {
    type Error = std::convert::Infallible;

    fn teardown(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A typed result from an attestation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationFailure<Error> {
    Rejected(Error),
    MustQuarantine(Error),
}

/// Evidence must carry the same opaque identity as the resolved facts.
pub trait AttestationEvidence<Identity> {
    fn identity(&self) -> &Identity;
}

/// The only way to produce an attested transaction.
///
/// There is deliberately no default implementation. A contract declares its
/// identity, and the transaction checks both that identity and the evidence
/// identity against the immutable resolved facts before producing Attested.
pub trait TypedAttestation<Plan, Lane> {
    type Identity: Eq;
    type Evidence: AttestationEvidence<Self::Identity>;
    type Error;

    fn identity(&self) -> &Self::Identity;

    fn attest(
        &self,
        facts: &ResolvedExecutionFacts<Plan, Lane, Self::Identity>,
    ) -> Result<Self::Evidence, AttestationFailure<Self::Error>>;
}

/// Private publication capability. Family modules cannot name or implement
/// this trait, and no transaction exposes the journal field or a mutating
/// journal adapter.
trait PublicationJournal<Candidate, Plan, Lane, Identity> {
    type Error;

    fn publish(
        &mut self,
        candidate: &Candidate,
        facts: &ResolvedExecutionFacts<Plan, Lane, Identity>,
    ) -> Result<(), PublicationFailure<Self::Error>>;

    fn rollback(
        &mut self,
        candidate: &Candidate,
        facts: &ResolvedExecutionFacts<Plan, Lane, Identity>,
    ) -> Result<(), Self::Error>;

    fn quarantine(&mut self) -> Result<(), Self::Error>;
}

/// The only journal surface visible outside this module is construction. Its
/// associated journal type has no externally callable publication methods.
pub trait PublicationJournalFactory<Candidate, Plan, Lane, Identity> {
    type Journal;

    fn build(
        self,
        candidate: &Candidate,
        facts: &ResolvedExecutionFacts<Plan, Lane, Identity>,
    ) -> Self::Journal;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationFailure<Error> {
    Rejected(Error),
    MustQuarantine(Error),
}

/// A read-only observer seam for a future GPU/runtime adapter. It has no
/// transaction handle and no mutating method, and is not wired to context.
pub trait ReadOnlyActivationObserver<Plan, Lane, Identity> {
    fn observe(&self, stage: ActivationStage, facts: &ResolvedExecutionFacts<Plan, Lane, Identity>);
}

#[allow(dead_code)]
pub struct ReadOnlyObserverAdapter<'a, Observer> {
    observer: &'a Observer,
}

impl<'a, Observer> ReadOnlyObserverAdapter<'a, Observer> {
    pub const fn new(observer: &'a Observer) -> Self {
        Self { observer }
    }

    pub fn notify<Plan, Lane, Identity>(
        &self,
        stage: ActivationStage,
        facts: &ResolvedExecutionFacts<Plan, Lane, Identity>,
    ) where
        Observer: ReadOnlyActivationObserver<Plan, Lane, Identity>,
    {
        self.observer.observe(stage, facts);
    }
}

#[derive(Debug)]
pub struct OwnerSetError<Error> {
    pub first: Error,
    pub failures: usize,
}

struct StagedOwnerSet<Owner> {
    owners: Vec<Owner>,
}

impl<Owner> StagedOwnerSet<Owner> {
    fn new(owners: impl IntoIterator<Item = Owner>) -> Self {
        Self {
            owners: owners.into_iter().collect(),
        }
    }

    fn teardown_reverse(&mut self) -> Result<(), OwnerSetError<Owner::Error>>
    where
        Owner: StagedOwner,
    {
        let mut first = None;
        let mut failures = 0;
        for owner in self.owners.iter_mut().rev() {
            if let Err(error) = owner.teardown() {
                first.get_or_insert(error);
                failures += 1;
            }
        }
        match first {
            Some(first) => Err(OwnerSetError { first, failures }),
            None => Ok(()),
        }
    }

    fn quarantine_reverse(&mut self) -> Result<(), OwnerSetError<Owner::Error>>
    where
        Owner: StagedOwner,
    {
        let mut first = None;
        let mut failures = 0;
        for owner in self.owners.iter_mut().rev() {
            if let Err(error) = owner.quarantine() {
                first.get_or_insert(error);
                failures += 1;
            }
        }
        match first {
            Some(first) => Err(OwnerSetError { first, failures }),
            None => Ok(()),
        }
    }
}

#[derive(Debug)]
pub struct CleanupError<ReservationError, OwnerError, JournalError> {
    pub reservation: Option<ReservationError>,
    pub owners: Option<OwnerSetError<OwnerError>>,
    pub journal: Option<JournalError>,
}

impl<ReservationError, OwnerError, JournalError>
    CleanupError<ReservationError, OwnerError, JournalError>
{
    fn is_empty(&self) -> bool {
        self.reservation.is_none() && self.owners.is_none() && self.journal.is_none()
    }
}

struct ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner> {
    candidate: Candidate,
    facts: ResolvedExecutionFacts<Plan, Lane, Identity>,
    journal: Journal,
    reservation: Reservation,
    owners: StagedOwnerSet<Owner>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
    ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    fn cleanup(
        &mut self,
        quarantine: bool,
    ) -> Result<(), CleanupError<Reservation::Error, Owner::Error, Journal::Error>> {
        let journal = if quarantine {
            self.journal.quarantine().err()
        } else {
            self.journal.rollback(&self.candidate, &self.facts).err()
        };
        let owners = if quarantine {
            self.owners.quarantine_reverse().err()
        } else {
            self.owners.teardown_reverse().err()
        };
        let reservation = if quarantine {
            self.reservation.quarantine().err()
        } else {
            self.reservation.release().err()
        };
        let error = CleanupError {
            reservation,
            owners,
            journal,
        };
        if error.is_empty() { Ok(()) } else { Err(error) }
    }
}

/// Every active stage owns this private guard. Dropping it performs safe
/// quarantine compensation; it never silently drops native state.
struct ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    parts: Option<ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
    ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    fn new(
        parts: ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
    ) -> Self {
        Self { parts: Some(parts) }
    }

    fn take(
        &mut self,
    ) -> ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner> {
        self.parts
            .take()
            .expect("active transaction guard already reached a terminal state")
    }

    fn facts(&self) -> &ResolvedExecutionFacts<Plan, Lane, Identity> {
        &self
            .parts
            .as_ref()
            .expect("active transaction guard already reached a terminal state")
            .facts
    }

    fn cleanup(
        &mut self,
        quarantine: bool,
    ) -> Result<(), CleanupError<Reservation::Error, Owner::Error, Journal::Error>> {
        self.parts
            .as_mut()
            .expect("active transaction guard already reached a terminal state")
            .cleanup(quarantine)
    }

    fn disarm(&mut self) {
        let _ = self.parts.take();
    }
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner> Drop
    for ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    fn drop(&mut self) {
        if let Some(mut parts) = self.parts.take() {
            let _ = parts.cleanup(true);
        }
    }
}

struct PreparedParts<Candidate, Plan, Lane, Identity, Journal> {
    candidate: Candidate,
    facts: ResolvedExecutionFacts<Plan, Lane, Identity>,
    journal: Journal,
}

impl<Candidate, Plan, Lane, Identity, Journal>
    PreparedParts<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    fn rollback(&mut self) -> Result<(), Journal::Error> {
        self.journal.rollback(&self.candidate, &self.facts)
    }

    fn quarantine(&mut self) -> Result<(), Journal::Error> {
        self.journal.quarantine()
    }
}

/// Prepared state owns a journal guard. A normal drop attempts rollback; an
/// unsuccessful rollback immediately continues with quarantine.
struct PreparedGuard<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    parts: Option<PreparedParts<Candidate, Plan, Lane, Identity, Journal>>,
    rollback_failed: bool,
}

impl<Candidate, Plan, Lane, Identity, Journal>
    PreparedGuard<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    fn new(parts: PreparedParts<Candidate, Plan, Lane, Identity, Journal>) -> Self {
        Self {
            parts: Some(parts),
            rollback_failed: false,
        }
    }

    fn facts(&self) -> &ResolvedExecutionFacts<Plan, Lane, Identity> {
        &self
            .parts
            .as_ref()
            .expect("prepared transaction guard already reached a terminal state")
            .facts
    }

    fn take(&mut self) -> PreparedParts<Candidate, Plan, Lane, Identity, Journal> {
        self.parts
            .take()
            .expect("prepared transaction guard already reached a terminal state")
    }

    fn rollback(&mut self) -> Result<(), Journal::Error> {
        let result = self
            .parts
            .as_mut()
            .expect("prepared transaction guard already reached a terminal state")
            .rollback();
        if result.is_err() {
            self.rollback_failed = true;
        }
        result
    }

    fn disarm(&mut self) {
        let _ = self.parts.take();
    }
}

impl<Candidate, Plan, Lane, Identity, Journal> Drop
    for PreparedGuard<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    fn drop(&mut self) {
        if let Some(mut parts) = self.parts.take() {
            if !self.rollback_failed && parts.rollback().is_err() {
                self.rollback_failed = true;
            }
            if self.rollback_failed {
                let _ = parts.quarantine();
            }
        }
    }
}

/// The prepared transaction entry point.
pub struct PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    guard: PreparedGuard<Candidate, Plan, Lane, Identity, Journal>,
}

/// The canonical name for the prepared transaction entry point.
pub type CandidateActivationTransaction<Candidate, Plan, Lane, Identity, Journal> =
    PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>;

impl<Candidate, Plan, Lane, Identity, Journal>
    PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    pub fn prepare(
        candidate: Candidate,
        facts: ResolvedExecutionFacts<Plan, Lane, Identity>,
        journal: Journal,
    ) -> Self {
        Self {
            guard: PreparedGuard::new(PreparedParts {
                candidate,
                facts,
                journal,
            }),
        }
    }

    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Prepared
    }

    pub fn facts(&self) -> &ResolvedExecutionFacts<Plan, Lane, Identity> {
        self.guard.facts()
    }

    pub fn reserve<Reservation>(
        mut self,
        reservation: Reservation,
    ) -> ReservedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation>
    where
        Reservation: ActivationReservation,
    {
        let parts = self.guard.take();
        ReservedTransaction {
            guard: ActiveGuard::new(ActiveParts {
                candidate: parts.candidate,
                facts: parts.facts,
                journal: parts.journal,
                reservation,
                owners: StagedOwnerSet::new([]),
            }),
        }
    }

    /// Factory construction is the only public(crate) family seam. The
    /// resulting journal still has a private mutation capability.
    pub fn prepare_from_factory<Factory>(
        candidate: Candidate,
        facts: ResolvedExecutionFacts<Plan, Lane, Identity>,
        factory: Factory,
    ) -> PreparedTransaction<Candidate, Plan, Lane, Identity, Factory::Journal>
    where
        Factory: PublicationJournalFactory<Candidate, Plan, Lane, Identity>,
        Factory::Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    {
        let journal = factory.build(&candidate, &facts);
        PreparedTransaction::prepare(candidate, facts, journal)
    }
}

impl<Candidate, Plan, Lane, Identity, Journal>
    PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    pub fn rollback(mut self) -> Result<PreparedRollback, CleanupError<(), (), Journal::Error>> {
        let result = self.guard.rollback();
        if let Err(journal) = result {
            // Keep the guard armed. Its Drop path performs quarantine, so a
            // failed explicit rollback cannot fall through to ordinary drop.
            Err(CleanupError {
                reservation: None,
                owners: None,
                journal: Some(journal),
            })
        } else {
            self.guard.disarm();
            Ok(PreparedRollback {
                _private: PhantomData,
            })
        }
    }
}

/// Transaction with an active reservation but no staged owners yet.
pub struct ReservedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
{
    guard: ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, EmptyOwner>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation>
    ReservedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
{
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Reserved
    }

    pub fn materialize<Owner>(
        mut self,
        owners: impl IntoIterator<Item = Owner>,
    ) -> MaterializedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
    where
        Owner: StagedOwner,
    {
        let parts = self.guard.take();
        MaterializedTransaction {
            guard: ActiveGuard::new(ActiveParts {
                candidate: parts.candidate,
                facts: parts.facts,
                journal: parts.journal,
                reservation: parts.reservation,
                owners: StagedOwnerSet::new(owners),
            }),
        }
    }

    pub fn rollback(
        mut self,
    ) -> Result<
        RollbackTerminal,
        CleanupError<Reservation::Error, std::convert::Infallible, Journal::Error>,
    > {
        let result = self.guard.cleanup(false);
        if result.is_ok() {
            self.guard.disarm();
            Ok(RollbackTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }

    pub fn quarantine(
        mut self,
    ) -> Result<
        QuarantineTerminal,
        CleanupError<Reservation::Error, std::convert::Infallible, Journal::Error>,
    > {
        let result = self.guard.cleanup(true);
        if result.is_ok() {
            self.guard.disarm();
            Ok(QuarantineTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }
}

/// Transaction after all candidate owners have been staged, but before
/// attestation.
pub struct MaterializedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    guard: ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
    MaterializedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Materialized
    }

    pub fn begin_attestation<Contract>(
        mut self,
        contract: Contract,
    ) -> AttestationPendingTransaction<
        Candidate,
        Plan,
        Lane,
        Identity,
        Journal,
        Reservation,
        Owner,
        Contract,
    > {
        AttestationPendingTransaction {
            guard: ActiveGuard::new(self.guard.take()),
            contract,
        }
    }

    pub fn rollback(
        mut self,
    ) -> Result<RollbackTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(false);
        if result.is_ok() {
            self.guard.disarm();
            Ok(RollbackTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }

    pub fn quarantine(
        mut self,
    ) -> Result<QuarantineTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(true);
        if result.is_ok() {
            self.guard.disarm();
            Ok(QuarantineTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }
}

/// A pending attestation retains the explicit contract. There is no operation
/// that can construct `AttestedTransaction` without invoking that contract.
pub struct AttestationPendingTransaction<
    Candidate,
    Plan,
    Lane,
    Identity,
    Journal,
    Reservation,
    Owner,
    Contract,
> where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    guard: ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
    contract: Contract,
}

#[derive(Debug)]
pub enum AttestationError<Error> {
    Contract(AttestationFailure<Error>),
    ContractIdentityMismatch,
    EvidenceIdentityMismatch,
}

#[derive(Debug)]
pub enum AttestationOutcome<Pending, Attested, Quarantine, Error> {
    Attested(Attested),
    Rejected {
        source: AttestationError<Error>,
        transaction: Pending,
    },
    MustQuarantine {
        source: AttestationError<Error>,
        transaction: Quarantine,
    },
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract>
    AttestationPendingTransaction<
        Candidate,
        Plan,
        Lane,
        Identity,
        Journal,
        Reservation,
        Owner,
        Contract,
    >
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::AttestationPending
    }

    pub fn rollback(
        mut self,
    ) -> Result<RollbackTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(false);
        if result.is_ok() {
            self.guard.disarm();
            Ok(RollbackTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }

    pub fn quarantine(
        mut self,
    ) -> Result<QuarantineTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(true);
        if result.is_ok() {
            self.guard.disarm();
            Ok(QuarantineTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract>
    AttestationPendingTransaction<
        Candidate,
        Plan,
        Lane,
        Identity,
        Journal,
        Reservation,
        Owner,
        Contract,
    >
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Identity: Eq,
    Contract: TypedAttestation<Plan, Lane, Identity = Identity>,
{
    pub fn attest(
        mut self,
    ) -> AttestationOutcome<
        Self,
        AttestedTransaction<
            Candidate,
            Plan,
            Lane,
            Identity,
            Journal,
            Reservation,
            Owner,
            Contract,
            Contract::Evidence,
        >,
        QuarantineRequired<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract>,
        Contract::Error,
    > {
        if self.contract.identity() != self.guard.facts().identity() {
            return AttestationOutcome::Rejected {
                source: AttestationError::ContractIdentityMismatch,
                transaction: self,
            };
        }

        match self.contract.attest(self.guard.facts()) {
            Ok(evidence) if evidence.identity() == self.guard.facts().identity() => {
                AttestationOutcome::Attested(AttestedTransaction {
                    guard: ActiveGuard::new(self.guard.take()),
                    proof: AttestationProof {
                        contract: self.contract,
                        evidence,
                    },
                })
            }
            Ok(_) => AttestationOutcome::Rejected {
                source: AttestationError::EvidenceIdentityMismatch,
                transaction: self,
            },
            Err(AttestationFailure::Rejected(error)) => AttestationOutcome::Rejected {
                source: AttestationError::Contract(AttestationFailure::Rejected(error)),
                transaction: self,
            },
            Err(AttestationFailure::MustQuarantine(error)) => AttestationOutcome::MustQuarantine {
                source: AttestationError::Contract(AttestationFailure::MustQuarantine(error)),
                transaction: QuarantineRequired {
                    guard: self.guard,
                    contract: self.contract,
                },
            },
        }
    }
}

/// A MustQuarantine result has a narrower capability than a rejected pending
/// transaction. It exposes only quarantine, and its guard drops into
/// quarantine; rollback is not a method on this type.
pub struct QuarantineRequired<
    Candidate,
    Plan,
    Lane,
    Identity,
    Journal,
    Reservation,
    Owner,
    Contract,
> where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    guard: ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
    contract: Contract,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract>
    QuarantineRequired<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    pub fn quarantine(
        mut self,
    ) -> Result<QuarantineTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(true);
        if result.is_ok() {
            self.guard.disarm();
            Ok(QuarantineTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }
}

/// The indivisible typed attestation proof retained until publication.
#[derive(Debug)]
struct AttestationProof<Contract, Evidence> {
    contract: Contract,
    evidence: Evidence,
}

/// Transaction with a contract-backed attestation proof.
pub struct AttestedTransaction<
    Candidate,
    Plan,
    Lane,
    Identity,
    Journal,
    Reservation,
    Owner,
    Contract,
    Evidence,
> where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    guard: ActiveGuard<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
    proof: AttestationProof<Contract, Evidence>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract, Evidence>
    AttestedTransaction<
        Candidate,
        Plan,
        Lane,
        Identity,
        Journal,
        Reservation,
        Owner,
        Contract,
        Evidence,
    >
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Attested
    }

    pub fn commit(
        mut self,
    ) -> Result<
        CommittedTransaction<
            Candidate,
            Plan,
            Lane,
            Identity,
            Journal,
            Reservation,
            Owner,
            Contract,
            Evidence,
        >,
        CommitError<Reservation::Error, Owner::Error, Journal::Error>,
    > {
        let parts = self
            .guard
            .parts
            .as_mut()
            .expect("active transaction guard already reached a terminal state");
        let publication = parts.journal.publish(&parts.candidate, &parts.facts);
        match publication {
            Ok(()) => Ok(CommittedTransaction {
                parts: self.guard.take(),
                proof: self.proof,
            }),
            Err(PublicationFailure::Rejected(source)) => {
                let cleanup = self.guard.cleanup(false).err();
                if cleanup.is_none() {
                    self.guard.disarm();
                }
                Err(CommitError::Rejected { source, cleanup })
            }
            Err(PublicationFailure::MustQuarantine(source)) => {
                let cleanup = self.guard.cleanup(true).err();
                if cleanup.is_none() {
                    self.guard.disarm();
                }
                Err(CommitError::MustQuarantine { source, cleanup })
            }
        }
    }

    pub fn rollback(
        mut self,
    ) -> Result<RollbackTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(false);
        if result.is_ok() {
            self.guard.disarm();
            Ok(RollbackTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }

    pub fn quarantine(
        mut self,
    ) -> Result<QuarantineTerminal, CleanupError<Reservation::Error, Owner::Error, Journal::Error>>
    {
        let result = self.guard.cleanup(true);
        if result.is_ok() {
            self.guard.disarm();
            Ok(QuarantineTerminal {
                _private: PhantomData,
            })
        } else {
            result.map(|_| unreachable!())
        }
    }
}

#[derive(Debug)]
pub enum CommitError<ReservationError, OwnerError, JournalError> {
    Rejected {
        source: JournalError,
        cleanup: Option<CleanupError<ReservationError, OwnerError, JournalError>>,
    },
    MustQuarantine {
        source: JournalError,
        cleanup: Option<CleanupError<ReservationError, OwnerError, JournalError>>,
    },
}

/// The only successful pre-publication rollback terminal.
pub struct RollbackTerminal {
    _private: PhantomData<()>,
}

impl RollbackTerminal {
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::RolledBack
    }
}

/// The terminal returned by explicit or contract-required quarantine.
pub struct QuarantineTerminal {
    _private: PhantomData<()>,
}

impl QuarantineTerminal {
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Quarantined
    }
}

/// A committed transaction has no rollback operation. Releasing the old
/// publication is a later transaction, not an authority retained by this one.
pub struct CommittedTransaction<
    Candidate,
    Plan,
    Lane,
    Identity,
    Journal,
    Reservation,
    Owner,
    Contract,
    Evidence,
> where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    parts: ActiveParts<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner>,
    proof: AttestationProof<Contract, Evidence>,
}

impl<Candidate, Plan, Lane, Identity, Journal, Reservation, Owner, Contract, Evidence>
    CommittedTransaction<
        Candidate,
        Plan,
        Lane,
        Identity,
        Journal,
        Reservation,
        Owner,
        Contract,
        Evidence,
    >
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
    Reservation: ActivationReservation,
    Owner: StagedOwner,
{
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::Committed
    }
}

/// A prepared transaction has no reservation or staged owner to compensate.
pub struct PreparedRollback {
    _private: PhantomData<()>,
}

impl PreparedRollback {
    pub const fn stage(&self) -> ActivationStage {
        ActivationStage::RolledBack
    }
}

/// Opaque identity for one NES candidate attempt. The selected candidate is
/// retained as facts; this module does not inspect provider or placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCandidateAttemptFacts {
    candidate: ExecutionCandidate,
}

impl NativeCandidateAttemptFacts {
    pub fn new(candidate: ExecutionCandidate) -> Self {
        Self { candidate }
    }

    #[allow(dead_code)]
    pub fn candidate(&self) -> &ExecutionCandidate {
        &self.candidate
    }
}

/// Evidence that one NES candidate attempt completed its attestation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCandidateAttemptEvidence {
    identity: NativeCandidateAttemptFacts,
}

impl ExecutionCandidateAttemptEvidence {
    pub fn new(identity: NativeCandidateAttemptFacts) -> Self {
        Self { identity }
    }
}

impl AttestationEvidence<NativeCandidateAttemptFacts> for ExecutionCandidateAttemptEvidence {
    fn identity(&self) -> &NativeCandidateAttemptFacts {
        &self.identity
    }
}

/// Token owner for an NES attempt. Resident owners are staged through the
/// bound cache journal during attestation, not as a second publication path.
#[derive(Debug, Default)]
pub struct ExecutionCandidateAttemptOwner;

impl StagedOwner for ExecutionCandidateAttemptOwner {
    type Error = std::convert::Infallible;

    fn teardown(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Binds the NES cache-journal finish callback. Publish commits staged owners;
/// rollback and quarantine destroy them. Quote is not a reservation: NES does
/// not treat a forecast as this journal's reservation token.
pub struct ExecutionCandidateAttemptJournalFactory {
    finish: Option<Box<dyn FnOnce(bool) + 'static>>,
}

pub struct ExecutionCandidateAttemptJournal {
    finish: Option<Box<dyn FnOnce(bool) + 'static>>,
}

impl ExecutionCandidateAttemptJournalFactory {
    pub fn bind(finish: impl FnOnce(bool) + 'static) -> Self {
        Self {
            finish: Some(Box::new(finish)),
        }
    }

    pub fn prepare(
        self,
        candidate: NativeCandidateAttemptFacts,
        facts: ResolvedExecutionFacts<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
        >,
    ) -> NativeCandidatePreparedActivation {
        CandidateActivationTransaction::<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            ExecutionCandidateAttemptJournal,
        >::prepare_from_factory(candidate, facts, self)
    }
}

impl
    PublicationJournalFactory<
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
    > for ExecutionCandidateAttemptJournalFactory
{
    type Journal = ExecutionCandidateAttemptJournal;

    fn build(
        self,
        _candidate: &NativeCandidateAttemptFacts,
        _facts: &ResolvedExecutionFacts<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
        >,
    ) -> Self::Journal {
        ExecutionCandidateAttemptJournal {
            finish: self.finish,
        }
    }
}

impl ExecutionCandidateAttemptJournal {
    fn finish(&mut self, commit: bool) -> Result<(), String> {
        if let Some(finish) = self.finish.take() {
            finish(commit);
        }
        Ok(())
    }
}

impl
    PublicationJournal<
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
    > for ExecutionCandidateAttemptJournal
{
    type Error = String;

    fn publish(
        &mut self,
        _candidate: &NativeCandidateAttemptFacts,
        _facts: &ResolvedExecutionFacts<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
        >,
    ) -> Result<(), PublicationFailure<Self::Error>> {
        self.finish(true).map_err(PublicationFailure::Rejected)
    }

    fn rollback(
        &mut self,
        _candidate: &NativeCandidateAttemptFacts,
        _facts: &ResolvedExecutionFacts<
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
            NativeCandidateAttemptFacts,
        >,
    ) -> Result<(), Self::Error> {
        self.finish(false)
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        self.finish(false)
    }
}

pub type NativeCandidatePreparedActivation = CandidateActivationTransaction<
    NativeCandidateAttemptFacts,
    NativeCandidateAttemptFacts,
    NativeCandidateAttemptFacts,
    NativeCandidateAttemptFacts,
    ExecutionCandidateAttemptJournal,
>;

/// Default-model activation candidate identity. The path is the installed pack
/// the host intends to publish after attestation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelActivationCandidate {
    pub pull: String,
    pub path: std::path::PathBuf,
    pub pack_content_id: String,
}

/// Immutable semantic execution plan resolved from the verified pack and the
/// exact policy candidate before any owner is materialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelActivationPlan {
    path: std::path::PathBuf,
    pack_content_id: String,
    architecture_id: String,
    execution_intent: crate::device::execution_policy::ExecutionIntent,
    resolved_runtime: ResolvedFamilyRuntimeInput,
    resident_topology: DefaultModelResidentTopologyPlan,
}

impl DefaultModelActivationPlan {
    pub(crate) fn new(
        path: std::path::PathBuf,
        pack_content_id: String,
        architecture_id: String,
        execution_intent: crate::device::execution_policy::ExecutionIntent,
        resolved_runtime: ResolvedFamilyRuntimeInput,
        resident_topology: DefaultModelResidentTopologyPlan,
    ) -> Self {
        Self {
            path,
            pack_content_id,
            architecture_id,
            execution_intent,
            resolved_runtime,
            resident_topology,
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn pack_content_id(&self) -> &str {
        &self.pack_content_id
    }

    pub fn architecture_id(&self) -> &str {
        &self.architecture_id
    }

    pub fn execution_intent(&self) -> &crate::device::execution_policy::ExecutionIntent {
        &self.execution_intent
    }

    pub fn resolved_runtime(&self) -> ResolvedFamilyRuntimeInput {
        self.resolved_runtime
    }

    pub fn output_plan(&self) -> GgmlDecodeOutputPlan {
        self.resolved_runtime.output_plan()
    }

    pub fn reuse_mode(&self) -> GgmlDecodeReuseMode {
        self.resolved_runtime.reuse_mode()
    }

    pub fn matches_identity(&self, identity: &DefaultModelActivationIdentity) -> bool {
        self.path == identity.path
            && self.pack_content_id == identity.pack_content_id
            && self.architecture_id == identity.architecture_id
            && self.execution_intent == identity.execution_intent
            && self.output_plan() == identity.output_plan
            && self.reuse_mode() == identity.reuse_mode
            && self.resident_topology == identity.resident_topology
    }

    pub(crate) fn resident_topology(&self) -> &DefaultModelResidentTopologyPlan {
        &self.resident_topology
    }
}

/// Exact physical execution candidate selected for this activation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelActivationLane {
    candidate: ExecutionCandidate,
}

impl DefaultModelActivationLane {
    pub fn new(candidate: ExecutionCandidate) -> Self {
        Self { candidate }
    }

    pub const fn candidate(&self) -> &ExecutionCandidate {
        &self.candidate
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelActivationIdentity {
    pull: String,
    path: std::path::PathBuf,
    pack_content_id: String,
    architecture_id: String,
    execution_intent: crate::device::execution_policy::ExecutionIntent,
    candidate: ExecutionCandidate,
    output_plan: GgmlDecodeOutputPlan,
    reuse_mode: GgmlDecodeReuseMode,
    resident_topology: DefaultModelResidentTopologyPlan,
}

impl DefaultModelActivationIdentity {
    pub(crate) fn new(
        pull: String,
        path: std::path::PathBuf,
        pack_content_id: String,
        architecture_id: String,
        execution_intent: crate::device::execution_policy::ExecutionIntent,
        candidate: ExecutionCandidate,
        output_plan: GgmlDecodeOutputPlan,
        reuse_mode: GgmlDecodeReuseMode,
        resident_topology: DefaultModelResidentTopologyPlan,
    ) -> Self {
        Self {
            pull,
            path,
            pack_content_id,
            architecture_id,
            execution_intent,
            candidate,
            output_plan,
            reuse_mode,
            resident_topology,
        }
    }

    pub fn pull(&self) -> &str {
        &self.pull
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn pack_content_id(&self) -> &str {
        &self.pack_content_id
    }

    pub fn architecture_id(&self) -> &str {
        &self.architecture_id
    }

    pub fn execution_intent(&self) -> &crate::device::execution_policy::ExecutionIntent {
        &self.execution_intent
    }

    pub const fn candidate(&self) -> &ExecutionCandidate {
        &self.candidate
    }

    pub const fn output_plan(&self) -> GgmlDecodeOutputPlan {
        self.output_plan
    }

    pub const fn reuse_mode(&self) -> GgmlDecodeReuseMode {
        self.reuse_mode
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelActivationEvidence {
    identity: DefaultModelActivationIdentity,
}

impl DefaultModelActivationEvidence {
    pub fn new(identity: DefaultModelActivationIdentity) -> Self {
        Self { identity }
    }
}

impl AttestationEvidence<DefaultModelActivationIdentity> for DefaultModelActivationEvidence {
    fn identity(&self) -> &DefaultModelActivationIdentity {
        &self.identity
    }
}

/// Reservation used when set-default does not hold a broker batch of its own.
/// Native warmup/probe admits its owners through the existing NES path.
#[derive(Debug, Default)]
pub struct NoopActivationReservation;

impl ActivationReservation for NoopActivationReservation {
    type Error = std::convert::Infallible;

    fn release(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Factory for the production default-model publication journal. Persist of
/// V2 happens only when an attested transaction commits.
pub struct DefaultModelActivationJournalFactory {
    home: std::path::PathBuf,
    pack: crate::InstalledPack,
    preference: crate::QuantPreference,
    publication: DefaultModelActivationPublication,
    write_fault: Option<crate::default_selection::DefaultSelectionWriteFault>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefaultModelActivationPublication {
    PersistSelection,
    ReactivateDurableSelection,
}

/// Opaque production journal. Families and the server cannot call publish.
pub struct DefaultModelActivationJournal {
    home: std::path::PathBuf,
    pack: crate::InstalledPack,
    preference: crate::QuantPreference,
    architecture_id: String,
    execution_intent: crate::device::execution_policy::ExecutionIntent,
    publication: DefaultModelActivationPublication,
    write_fault: Option<crate::default_selection::DefaultSelectionWriteFault>,
}

impl DefaultModelActivationJournalFactory {
    pub fn persist_selection(
        home: std::path::PathBuf,
        pack: crate::InstalledPack,
        preference: crate::QuantPreference,
    ) -> Self {
        Self {
            home,
            pack,
            preference,
            publication: DefaultModelActivationPublication::PersistSelection,
            write_fault: None,
        }
    }

    pub fn reactivate_durable_selection(
        home: std::path::PathBuf,
        pack: crate::InstalledPack,
        preference: crate::QuantPreference,
    ) -> Self {
        Self {
            home,
            pack,
            preference,
            publication: DefaultModelActivationPublication::ReactivateDurableSelection,
            write_fault: None,
        }
    }

    #[doc(hidden)]
    pub fn with_selection_write_fault_for_test(
        mut self,
        fault: crate::default_selection::DefaultSelectionWriteFault,
    ) -> Self {
        self.write_fault = Some(fault);
        self
    }

    pub fn prepare(
        self,
        candidate: DefaultModelActivationCandidate,
        facts: ResolvedExecutionFacts<
            DefaultModelActivationPlan,
            DefaultModelActivationLane,
            DefaultModelActivationIdentity,
        >,
    ) -> DefaultModelPreparedActivation {
        CandidateActivationTransaction::<
            DefaultModelActivationCandidate,
            DefaultModelActivationPlan,
            DefaultModelActivationLane,
            DefaultModelActivationIdentity,
            DefaultModelActivationJournal,
        >::prepare_from_factory(candidate, facts, self)
    }
}

impl
    PublicationJournalFactory<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
    > for DefaultModelActivationJournalFactory
{
    type Journal = DefaultModelActivationJournal;

    fn build(
        self,
        _candidate: &DefaultModelActivationCandidate,
        facts: &ResolvedExecutionFacts<
            DefaultModelActivationPlan,
            DefaultModelActivationLane,
            DefaultModelActivationIdentity,
        >,
    ) -> Self::Journal {
        DefaultModelActivationJournal {
            home: self.home,
            pack: self.pack,
            preference: self.preference,
            architecture_id: facts.plan().architecture_id().to_string(),
            execution_intent: facts.plan().execution_intent().clone(),
            publication: self.publication,
            write_fault: self.write_fault,
        }
    }
}

impl
    PublicationJournal<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
    > for DefaultModelActivationJournal
{
    type Error = String;

    fn publish(
        &mut self,
        _candidate: &DefaultModelActivationCandidate,
        _facts: &ResolvedExecutionFacts<
            DefaultModelActivationPlan,
            DefaultModelActivationLane,
            DefaultModelActivationIdentity,
        >,
    ) -> Result<(), PublicationFailure<Self::Error>> {
        match self.publication {
            DefaultModelActivationPublication::PersistSelection => {
                match crate::default_selection::persist_activation_detailed_with_fault(
                    &self.home,
                    &self.pack,
                    self.preference.clone(),
                    &self.architecture_id,
                    &self.execution_intent,
                    self.write_fault,
                ) {
                    Ok(crate::default_selection::DefaultSelectionCommitOutcome::NotCommitted {
                        reason,
                    }) => Err(PublicationFailure::Rejected(reason)),
                    Ok(_) => Ok(()),
                    Err(error) => Err(PublicationFailure::Rejected(error.to_string())),
                }
            }
            DefaultModelActivationPublication::ReactivateDurableSelection => {
                let durable = crate::default_selection::read_active_model_selection_v2(&self.home)
                    .map_err(|error| PublicationFailure::Rejected(error.to_string()))?;
                let durable_matches_plan = durable.as_ref().is_some_and(|record| {
                    record
                        .architecture_id
                        .as_deref()
                        .is_none_or(|architecture| architecture == self.architecture_id)
                        && crate::default_selection::execution_intent_from_v2_wire(
                            &record.execution_intent,
                        )
                        .is_ok_and(|intent| intent == self.execution_intent)
                });
                match crate::default_selection::resolve_with_catalog(&self.home, None) {
                    Ok(crate::default_selection::DefaultModelResolution::Installed(pack))
                        if pack.path == self.pack.path
                            && pack.sha256.eq_ignore_ascii_case(&self.pack.sha256)
                            && pack.size_bytes == self.pack.size_bytes
                            && durable_matches_plan =>
                    {
                        Ok(())
                    }
                    Ok(_) => Err(PublicationFailure::Rejected(
                        "durable default selection changed during startup reactivation".to_string(),
                    )),
                    Err(error) => Err(PublicationFailure::Rejected(error.to_string())),
                }
            }
        }
    }

    fn rollback(
        &mut self,
        _candidate: &DefaultModelActivationCandidate,
        _facts: &ResolvedExecutionFacts<
            DefaultModelActivationPlan,
            DefaultModelActivationLane,
            DefaultModelActivationIdentity,
        >,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub type DefaultModelPreparedActivation = CandidateActivationTransaction<
    DefaultModelActivationCandidate,
    DefaultModelActivationPlan,
    DefaultModelActivationLane,
    DefaultModelActivationIdentity,
    DefaultModelActivationJournal,
>;

pub type DefaultModelActivationFacts = ResolvedExecutionFacts<
    DefaultModelActivationPlan,
    DefaultModelActivationLane,
    DefaultModelActivationIdentity,
>;

fn format_cleanup<ReservationError, OwnerError, JournalError>(
    error: CleanupError<ReservationError, OwnerError, JournalError>,
) -> String
where
    ReservationError: std::fmt::Debug,
    OwnerError: std::fmt::Debug,
    JournalError: std::fmt::Debug,
{
    format!("{error:?}")
}

impl<Reservation>
    ReservedTransaction<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
        DefaultModelActivationJournal,
        Reservation,
    >
where
    Reservation: ActivationReservation,
    Reservation::Error: std::fmt::Debug,
{
    pub fn rollback_activation(self) -> Result<RollbackTerminal, String> {
        self.rollback().map_err(format_cleanup)
    }
}

impl<Reservation, Owner, Contract>
    AttestationPendingTransaction<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
        DefaultModelActivationJournal,
        Reservation,
        Owner,
        Contract,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn rollback_activation(self) -> Result<RollbackTerminal, String> {
        self.rollback().map_err(format_cleanup)
    }

    pub fn quarantine_activation(self) -> Result<QuarantineTerminal, String> {
        self.quarantine().map_err(format_cleanup)
    }
}

impl<Reservation, Owner, Contract>
    QuarantineRequired<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
        DefaultModelActivationJournal,
        Reservation,
        Owner,
        Contract,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn quarantine_activation(self) -> Result<QuarantineTerminal, String> {
        self.quarantine().map_err(format_cleanup)
    }
}

impl<Reservation, Owner, Contract, Evidence>
    AttestedTransaction<
        DefaultModelActivationCandidate,
        DefaultModelActivationPlan,
        DefaultModelActivationLane,
        DefaultModelActivationIdentity,
        DefaultModelActivationJournal,
        Reservation,
        Owner,
        Contract,
        Evidence,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn commit_activation(self) -> Result<(), String> {
        match self.commit() {
            Ok(_) => Ok(()),
            Err(CommitError::Rejected { source, cleanup }) => Err(match cleanup {
                Some(cleanup) => format!("{source}; cleanup={cleanup:?}"),
                None => source,
            }),
            Err(CommitError::MustQuarantine { source, cleanup }) => Err(match cleanup {
                Some(cleanup) => format!("{source}; cleanup={cleanup:?}"),
                None => source,
            }),
        }
    }
}

impl<Reservation, Owner, Contract>
    AttestationPendingTransaction<
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        ExecutionCandidateAttemptJournal,
        Reservation,
        Owner,
        Contract,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn rollback_attempt(self) -> Result<RollbackTerminal, String> {
        self.rollback().map_err(format_cleanup)
    }

    pub fn quarantine_attempt(self) -> Result<QuarantineTerminal, String> {
        self.quarantine().map_err(format_cleanup)
    }
}

impl<Reservation, Owner, Contract>
    QuarantineRequired<
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        ExecutionCandidateAttemptJournal,
        Reservation,
        Owner,
        Contract,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn quarantine_attempt(self) -> Result<QuarantineTerminal, String> {
        self.quarantine().map_err(format_cleanup)
    }
}

impl<Reservation, Owner, Contract, Evidence>
    AttestedTransaction<
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        NativeCandidateAttemptFacts,
        ExecutionCandidateAttemptJournal,
        Reservation,
        Owner,
        Contract,
        Evidence,
    >
where
    Reservation: ActivationReservation,
    Owner: StagedOwner,
    Reservation::Error: std::fmt::Debug,
    Owner::Error: std::fmt::Debug,
{
    pub fn commit_attempt(self) -> Result<(), String> {
        match self.commit() {
            Ok(_) => Ok(()),
            Err(CommitError::Rejected { source, cleanup }) => Err(match cleanup {
                Some(cleanup) => format!("{source}; cleanup={cleanup:?}"),
                None => source,
            }),
            Err(CommitError::MustQuarantine { source, cleanup }) => Err(match cleanup {
                Some(cleanup) => format!("{source}; cleanup={cleanup:?}"),
                None => source,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MockError(&'static str);

    #[derive(Debug)]
    struct MockReservation {
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ActivationReservation for MockReservation {
        type Error = MockError;

        fn release(&mut self) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push("release");
            Ok(())
        }

        fn quarantine(&mut self) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push("reservation-quarantine");
            Ok(())
        }
    }

    #[derive(Debug)]
    struct MockOwner {
        id: u8,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl StagedOwner for MockOwner {
        type Error = MockError;

        fn teardown(&mut self) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push(if self.id == 1 {
                "teardown-1"
            } else {
                "teardown-2"
            });
            Ok(())
        }

        fn quarantine(&mut self) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push(if self.id == 1 {
                "quarantine-1"
            } else {
                "quarantine-2"
            });
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct MockJournal {
        events: Arc<Mutex<Vec<&'static str>>>,
        publication: Result<(), PublicationFailure<MockError>>,
        rollback: Result<(), MockError>,
    }

    impl PublicationJournal<u8, u8, u8, u8> for MockJournal {
        type Error = MockError;

        fn publish(
            &mut self,
            _candidate: &u8,
            _facts: &ResolvedExecutionFacts<u8, u8, u8>,
        ) -> Result<(), PublicationFailure<Self::Error>> {
            self.events.lock().unwrap().push("publish");
            self.publication.clone()
        }

        fn rollback(
            &mut self,
            _candidate: &u8,
            _facts: &ResolvedExecutionFacts<u8, u8, u8>,
        ) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push("journal-rollback");
            self.rollback.clone()
        }

        fn quarantine(&mut self) -> Result<(), Self::Error> {
            self.events.lock().unwrap().push("journal-quarantine");
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct MockFactory(MockJournal);

    impl PublicationJournalFactory<u8, u8, u8, u8> for MockFactory {
        type Journal = MockJournal;

        fn build(
            self,
            _candidate: &u8,
            _facts: &ResolvedExecutionFacts<u8, u8, u8>,
        ) -> Self::Journal {
            self.0
        }
    }

    #[derive(Debug, Clone)]
    struct Evidence {
        identity: u8,
    }

    impl AttestationEvidence<u8> for Evidence {
        fn identity(&self) -> &u8 {
            &self.identity
        }
    }

    #[derive(Debug, Clone)]
    struct Contract {
        identity: u8,
        evidence_identity: u8,
        outcome: Result<(), AttestationFailure<MockError>>,
    }

    impl TypedAttestation<u8, u8> for Contract {
        type Identity = u8;
        type Evidence = Evidence;
        type Error = MockError;

        fn identity(&self) -> &Self::Identity {
            &self.identity
        }

        fn attest(
            &self,
            facts: &ResolvedExecutionFacts<u8, u8, u8>,
        ) -> Result<Self::Evidence, AttestationFailure<Self::Error>> {
            assert_eq!(*facts.plan(), 7);
            assert_eq!(*facts.exact_lane(), 9);
            self.outcome.clone().map(|_| Evidence {
                identity: self.evidence_identity,
            })
        }
    }

    fn journal(events: Arc<Mutex<Vec<&'static str>>>) -> MockJournal {
        MockJournal {
            events,
            publication: Ok(()),
            rollback: Ok(()),
        }
    }

    fn prepared(
        events: Arc<Mutex<Vec<&'static str>>>,
    ) -> PreparedTransaction<u8, u8, u8, u8, MockJournal> {
        PreparedTransaction::prepare(1, ResolvedExecutionFacts::new(7, 9, 3), journal(events))
    }

    fn materialized(
        events: Arc<Mutex<Vec<&'static str>>>,
    ) -> MaterializedTransaction<u8, u8, u8, u8, MockJournal, MockReservation, MockOwner> {
        prepared(events.clone())
            .reserve(MockReservation {
                events: events.clone(),
            })
            .materialize([
                MockOwner {
                    id: 1,
                    events: events.clone(),
                },
                MockOwner { id: 2, events },
            ])
    }

    fn contract() -> Contract {
        Contract {
            identity: 3,
            evidence_identity: 3,
            outcome: Ok(()),
        }
    }

    #[test]
    fn prepared_direct_drop_rolls_back_its_journal() {
        let events = Arc::new(Mutex::new(Vec::new()));
        drop(prepared(events.clone()));
        assert_eq!(*events.lock().unwrap(), vec!["journal-rollback"]);
    }

    #[test]
    fn prepared_rollback_failure_keeps_guard_armed_for_quarantine() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut failing = journal(events.clone());
        failing.rollback = Err(MockError("rollback failed"));
        let transaction =
            PreparedTransaction::prepare(1, ResolvedExecutionFacts::new(7, 9, 3), failing);
        assert!(transaction.rollback().is_err());
        assert_eq!(
            *events.lock().unwrap(),
            vec!["journal-rollback", "journal-quarantine"]
        );
    }

    #[test]
    fn reserve_transfers_the_prepared_guard_exactly_once() {
        let events = Arc::new(Mutex::new(Vec::new()));
        drop(prepared(events.clone()).reserve(MockReservation {
            events: events.clone(),
        }));
        assert_eq!(
            *events.lock().unwrap(),
            vec!["journal-quarantine", "reservation-quarantine"]
        );
    }
    #[test]
    fn legal_and_illegal_transitions_are_explicit() {
        assert!(
            ActivationStage::Prepared
                .transition(ActivationStage::Reserved)
                .is_ok()
        );
        assert!(
            ActivationStage::Attested
                .transition(ActivationStage::Committed)
                .is_ok()
        );
        assert!(
            ActivationStage::Prepared
                .transition(ActivationStage::Committed)
                .is_err()
        );
        assert!(
            ActivationStage::Committed
                .transition(ActivationStage::RolledBack)
                .is_err()
        );
    }

    #[test]
    fn commit_requires_attestation_and_attestation_requires_a_contract() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events).begin_attestation(contract());
        let outcome = pending.attest();
        let AttestationOutcome::Attested(attested) = outcome else {
            panic!("the explicit contract should attest");
        };
        let committed = attested.commit().expect("publication should succeed");
        assert_eq!(committed.stage(), ActivationStage::Committed);
    }

    #[test]
    fn rejected_publication_rolls_back_while_must_quarantine_stays_distinct() {
        for (publication, expected) in [
            (
                PublicationFailure::Rejected(MockError("durable write rejected")),
                vec![
                    "publish",
                    "journal-rollback",
                    "teardown-2",
                    "teardown-1",
                    "release",
                ],
            ),
            (
                PublicationFailure::MustQuarantine(MockError("publication may have mutated")),
                vec![
                    "publish",
                    "journal-quarantine",
                    "quarantine-2",
                    "quarantine-1",
                    "reservation-quarantine",
                ],
            ),
        ] {
            let events = Arc::new(Mutex::new(Vec::new()));
            let mut failing = journal(events.clone());
            failing.publication = Err(publication);
            let materialized =
                PreparedTransaction::prepare(1, ResolvedExecutionFacts::new(7, 9, 3), failing)
                    .reserve(MockReservation {
                        events: events.clone(),
                    })
                    .materialize([
                        MockOwner {
                            id: 1,
                            events: events.clone(),
                        },
                        MockOwner {
                            id: 2,
                            events: events.clone(),
                        },
                    ]);
            let AttestationOutcome::Attested(attested) =
                materialized.begin_attestation(contract()).attest()
            else {
                panic!("contract should attest before publication failure");
            };
            assert!(attested.commit().is_err());
            assert_eq!(*events.lock().unwrap(), expected);
        }
    }

    #[test]
    fn rejected_attestation_cannot_become_attested() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events.clone()).begin_attestation(Contract {
            outcome: Err(AttestationFailure::Rejected(MockError("bad proof"))),
            ..contract()
        });
        let AttestationOutcome::Rejected { transaction, .. } = pending.attest() else {
            panic!("the rejected contract must not produce a proof");
        };
        let terminal = transaction.rollback().expect("rollback should succeed");
        assert_eq!(terminal.stage(), ActivationStage::RolledBack);
    }

    #[test]
    fn must_quarantine_has_no_rollback_capability_and_drop_quarantines() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events.clone()).begin_attestation(Contract {
            outcome: Err(AttestationFailure::MustQuarantine(MockError("device lost"))),
            ..contract()
        });
        let AttestationOutcome::MustQuarantine { transaction, .. } = pending.attest() else {
            panic!("the contract must require quarantine");
        };
        drop(transaction);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "journal-quarantine",
                "quarantine-2",
                "quarantine-1",
                "reservation-quarantine"
            ]
        );
    }

    #[test]
    fn explicit_quarantine_is_a_distinct_terminal_path() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events.clone()).begin_attestation(Contract {
            outcome: Err(AttestationFailure::MustQuarantine(MockError("device lost"))),
            ..contract()
        });
        let AttestationOutcome::MustQuarantine { transaction, .. } = pending.attest() else {
            panic!("the contract must require quarantine");
        };
        let terminal = transaction.quarantine().expect("quarantine should succeed");
        assert_eq!(terminal.stage(), ActivationStage::Quarantined);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "journal-quarantine",
                "quarantine-2",
                "quarantine-1",
                "reservation-quarantine"
            ]
        );
    }

    #[test]
    fn rollback_tears_down_staged_owners_in_reverse_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let terminal = materialized(events.clone())
            .rollback()
            .expect("rollback should succeed");
        assert_eq!(terminal.stage(), ActivationStage::RolledBack);
        assert_eq!(
            *events.lock().unwrap(),
            vec!["journal-rollback", "teardown-2", "teardown-1", "release"]
        );
    }

    #[test]
    fn dropping_an_active_stage_runs_quarantine_compensation() {
        let events = Arc::new(Mutex::new(Vec::new()));
        drop(materialized(events.clone()));
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "journal-quarantine",
                "quarantine-2",
                "quarantine-1",
                "reservation-quarantine"
            ]
        );
    }

    #[test]
    fn wrong_contract_or_evidence_identity_is_rejected() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events.clone()).begin_attestation(Contract {
            identity: 4,
            ..contract()
        });
        let AttestationOutcome::Rejected {
            transaction,
            source,
        } = pending.attest()
        else {
            panic!("a wrong contract identity must be rejected");
        };
        assert!(matches!(source, AttestationError::ContractIdentityMismatch));
        transaction.rollback().expect("rollback should succeed");

        let events = Arc::new(Mutex::new(Vec::new()));
        let pending = materialized(events.clone()).begin_attestation(Contract {
            evidence_identity: 4,
            ..contract()
        });
        let AttestationOutcome::Rejected {
            transaction,
            source,
        } = pending.attest()
        else {
            panic!("a wrong evidence identity must be rejected");
        };
        assert!(matches!(source, AttestationError::EvidenceIdentityMismatch));
        transaction.rollback().expect("rollback should succeed");
    }

    #[test]
    fn observer_adapter_is_read_only_and_facts_keep_identity() {
        struct Observer(Arc<Mutex<Vec<ActivationStage>>>);
        impl ReadOnlyActivationObserver<Arc<u8>, Arc<u8>, Arc<u8>> for Observer {
            fn observe(
                &self,
                stage: ActivationStage,
                _facts: &ResolvedExecutionFacts<Arc<u8>, Arc<u8>, Arc<u8>>,
            ) {
                self.0.lock().unwrap().push(stage);
            }
        }

        let plan = Arc::new(11);
        let lane = Arc::new(13);
        let identity = Arc::new(17);
        let facts = ResolvedExecutionFacts::new(plan.clone(), lane.clone(), identity.clone());
        assert!(Arc::ptr_eq(facts.plan(), &plan));
        assert!(Arc::ptr_eq(facts.exact_lane(), &lane));
        assert!(Arc::ptr_eq(facts.identity(), &identity));

        let stages = Arc::new(Mutex::new(Vec::new()));
        let observer = Observer(stages.clone());
        ReadOnlyObserverAdapter::new(&observer).notify(ActivationStage::Prepared, &facts);
        assert_eq!(*stages.lock().unwrap(), vec![ActivationStage::Prepared]);
    }

    #[test]
    fn factory_is_the_only_public_journal_seam() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let transaction = PreparedTransaction::<u8, u8, u8, u8, MockJournal>::prepare_from_factory(
            1,
            ResolvedExecutionFacts::new(7, 9, 3),
            MockFactory(journal(events.clone())),
        );
        assert_eq!(transaction.stage(), ActivationStage::Prepared);

        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/models/candidate_activation_transaction.rs"),
        )
        .expect("module source should be readable");
        assert!(!source.contains(&["pub trait ", "PublicationJournal<"].concat()));
        assert!(!source.contains(&["pub(crate) trait ", "PublicationJournal<"].concat()));
        let capability_start = source
            .find("trait PublicationJournal<")
            .expect("private journal capability should exist");
        let capability_end = source[capability_start..]
            .find("/// The only journal surface")
            .expect("factory seam should follow the private capability");
        let capability = &source[capability_start..capability_start + capability_end];
        assert!(capability.contains("fn publish"));
        assert!(!capability.contains("pub(crate)"));
        let pending_impl = source
            .split("AttestationPendingTransaction<")
            .nth(2)
            .expect("AttestationPendingTransaction impl should exist");
        let pending_methods = pending_impl
            .split("pub struct QuarantineRequired")
            .next()
            .expect("pending impl should end before QuarantineRequired");
        assert!(
            !pending_methods.contains("fn commit("),
            "AttestationPending must not expose commit; persist is only after attest"
        );
    }

    fn native_attempt_facts() -> NativeCandidateAttemptFacts {
        NativeCandidateAttemptFacts::new(ExecutionCandidate {
            device: crate::device::execution_policy::ExecutionDeviceSnapshot {
                route: crate::device::execution_route::ResolvedExecutionRoute::cpu(),
                ggml_kind: crate::ggml_runtime::GgmlBackendKind::Cpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: crate::device::execution_policy::ExecutionPlacement::CpuOnly,
        })
    }

    #[test]
    fn execution_attempt_journal_publish_and_rollback_invoke_the_bound_finish() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let publish_events = Arc::clone(&events);
        let facts = native_attempt_facts();
        let prepared = ExecutionCandidateAttemptJournalFactory::bind(move |commit| {
            publish_events
                .lock()
                .unwrap()
                .push(if commit { "publish" } else { "rollback" });
        })
        .prepare(
            facts.clone(),
            ResolvedExecutionFacts::new(facts.clone(), facts.clone(), facts),
        );
        drop(prepared);
        assert_eq!(*events.lock().unwrap(), vec!["rollback"]);
    }
}
