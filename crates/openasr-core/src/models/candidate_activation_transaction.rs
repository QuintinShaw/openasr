//! Backend-neutral candidate activation transaction primitives.
//!
//! This module stops at the transaction boundary. It does not resolve a
//! backend, infer a device class, attach to an execution context, or publish
//! through an existing runtime service. Callers supply already-resolved facts,
//! reservations, staged owners, attestation contracts, and a journal factory.
#![allow(dead_code, private_bounds, private_interfaces)]

use std::marker::PhantomData;

/// The externally visible lifecycle of one candidate activation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivationStage {
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
pub(crate) struct InvalidTransition {
    pub(crate) from: ActivationStage,
    pub(crate) to: ActivationStage,
}

impl ActivationStage {
    /// Check an edge independently of a transaction. The typestate wrappers
    /// make the same invalid edges unrepresentable at call sites.
    pub(crate) const fn transition(self, to: Self) -> Result<(), InvalidTransition> {
        let valid = match (self, to) {
            (Self::Prepared, Self::Reserved | Self::RolledBack)
            | (Self::Reserved, Self::Materialized | Self::RolledBack | Self::Quarantined)
            | (
                Self::Materialized,
                Self::AttestationPending | Self::RolledBack | Self::Quarantined,
            )
            | (Self::AttestationPending, Self::Attested | Self::RolledBack | Self::Quarantined)
            | (Self::Attested, Self::Committed | Self::RolledBack | Self::Quarantined) => true,
            _ => false,
        };

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
pub(crate) struct ResolvedExecutionFacts<Plan, Lane, Identity> {
    plan: Plan,
    exact_lane: Lane,
    identity: Identity,
}

impl<Plan, Lane, Identity> ResolvedExecutionFacts<Plan, Lane, Identity> {
    pub(crate) const fn new(plan: Plan, exact_lane: Lane, identity: Identity) -> Self {
        Self {
            plan,
            exact_lane,
            identity,
        }
    }

    pub(crate) const fn plan(&self) -> &Plan {
        &self.plan
    }

    pub(crate) const fn exact_lane(&self) -> &Lane {
        &self.exact_lane
    }

    pub(crate) const fn identity(&self) -> &Identity {
        &self.identity
    }

    pub(crate) fn into_parts(self) -> (Plan, Lane, Identity) {
        (self.plan, self.exact_lane, self.identity)
    }
}

/// A staged owner that has not yet been published to a shared registry.
/// Transactions invoke the operations in reverse construction order.
pub(crate) trait StagedOwner {
    type Error;

    fn teardown(&mut self) -> Result<(), Self::Error>;
    fn quarantine(&mut self) -> Result<(), Self::Error>;
}

/// The reservation token supplied by the admission layer.
pub(crate) trait ActivationReservation {
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
pub(crate) enum AttestationFailure<Error> {
    Rejected(Error),
    MustQuarantine(Error),
}

/// Evidence must carry the same opaque identity as the resolved facts.
pub(crate) trait AttestationEvidence<Identity> {
    fn identity(&self) -> &Identity;
}

/// The only way to produce an attested transaction.
///
/// There is deliberately no default implementation. A contract declares its
/// identity, and the transaction checks both that identity and the evidence
/// identity against the immutable resolved facts before producing Attested.
pub(crate) trait TypedAttestation<Plan, Lane> {
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
pub(crate) trait PublicationJournalFactory<Candidate, Plan, Lane, Identity> {
    type Journal;

    fn build(
        self,
        candidate: &Candidate,
        facts: &ResolvedExecutionFacts<Plan, Lane, Identity>,
    ) -> Self::Journal;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PublicationFailure<Error> {
    Rejected(Error),
    MustQuarantine(Error),
}

/// A read-only observer seam for a future GPU/runtime adapter. It has no
/// transaction handle and no mutating method, and is not wired to context.
pub(crate) trait ReadOnlyActivationObserver<Plan, Lane, Identity> {
    fn observe(&self, stage: ActivationStage, facts: &ResolvedExecutionFacts<Plan, Lane, Identity>);
}

pub(crate) struct ReadOnlyObserverAdapter<'a, Observer> {
    observer: &'a Observer,
}

impl<'a, Observer> ReadOnlyObserverAdapter<'a, Observer> {
    pub(crate) const fn new(observer: &'a Observer) -> Self {
        Self { observer }
    }

    pub(crate) fn notify<Plan, Lane, Identity>(
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
pub(crate) struct OwnerSetError<Error> {
    pub(crate) first: Error,
    pub(crate) failures: usize,
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
pub(crate) struct CleanupError<ReservationError, OwnerError, JournalError> {
    pub(crate) reservation: Option<ReservationError>,
    pub(crate) owners: Option<OwnerSetError<OwnerError>>,
    pub(crate) journal: Option<JournalError>,
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
pub(crate) struct PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    guard: PreparedGuard<Candidate, Plan, Lane, Identity, Journal>,
}

/// The canonical name for the prepared transaction entry point.
pub(crate) type CandidateActivationTransaction<Candidate, Plan, Lane, Identity, Journal> =
    PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>;

impl<Candidate, Plan, Lane, Identity, Journal>
    PreparedTransaction<Candidate, Plan, Lane, Identity, Journal>
where
    Journal: PublicationJournal<Candidate, Plan, Lane, Identity>,
{
    pub(crate) fn prepare(
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

    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Prepared
    }

    pub(crate) fn facts(&self) -> &ResolvedExecutionFacts<Plan, Lane, Identity> {
        self.guard.facts()
    }

    pub(crate) fn reserve<Reservation>(
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
    pub(crate) fn prepare_from_factory<Factory>(
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
    pub(crate) fn rollback(
        mut self,
    ) -> Result<PreparedRollback, CleanupError<(), (), Journal::Error>> {
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
pub(crate) struct ReservedTransaction<Candidate, Plan, Lane, Identity, Journal, Reservation>
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
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Reserved
    }

    pub(crate) fn materialize<Owner>(
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

    pub(crate) fn rollback(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }

    pub(crate) fn quarantine(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }
}

/// Transaction after all candidate owners have been staged, but before
/// attestation.
pub(crate) struct MaterializedTransaction<
    Candidate,
    Plan,
    Lane,
    Identity,
    Journal,
    Reservation,
    Owner,
> where
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
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Materialized
    }

    pub(crate) fn begin_attestation<Contract>(
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

    pub(crate) fn rollback(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }

    pub(crate) fn quarantine(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }
}

/// A pending attestation retains the explicit contract. There is no operation
/// that can construct `AttestedTransaction` without invoking that contract.
pub(crate) struct AttestationPendingTransaction<
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
pub(crate) enum AttestationError<Error> {
    Contract(AttestationFailure<Error>),
    ContractIdentityMismatch,
    EvidenceIdentityMismatch,
}

#[derive(Debug)]
pub(crate) enum AttestationOutcome<Pending, Attested, Quarantine, Error> {
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
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::AttestationPending
    }

    pub(crate) fn rollback(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }

    pub(crate) fn quarantine(
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
            result.map(|_| unreachable!()).map_err(|error| error)
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
    pub(crate) fn attest(
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
pub(crate) struct QuarantineRequired<
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
    pub(crate) fn quarantine(
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
            result.map(|_| unreachable!()).map_err(|error| error)
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
pub(crate) struct AttestedTransaction<
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
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Attested
    }

    pub(crate) fn commit(
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
                let cleanup = self.guard.cleanup(true).err();
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

    pub(crate) fn rollback(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }

    pub(crate) fn quarantine(
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
            result.map(|_| unreachable!()).map_err(|error| error)
        }
    }
}

#[derive(Debug)]
pub(crate) enum CommitError<ReservationError, OwnerError, JournalError> {
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
pub(crate) struct RollbackTerminal {
    _private: PhantomData<()>,
}

impl RollbackTerminal {
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::RolledBack
    }
}

/// The terminal returned by explicit or contract-required quarantine.
pub(crate) struct QuarantineTerminal {
    _private: PhantomData<()>,
}

impl QuarantineTerminal {
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Quarantined
    }
}

/// A committed transaction has no rollback operation. Releasing the old
/// publication is a later transaction, not an authority retained by this one.
pub(crate) struct CommittedTransaction<
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
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::Committed
    }
}

/// A prepared transaction has no reservation or staged owner to compensate.
pub(crate) struct PreparedRollback {
    _private: PhantomData<()>,
}

impl PreparedRollback {
    pub(crate) const fn stage(&self) -> ActivationStage {
        ActivationStage::RolledBack
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
    }
}
