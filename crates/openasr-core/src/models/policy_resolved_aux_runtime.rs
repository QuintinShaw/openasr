//! Persistent execution-policy ownership for auxiliary model stages.
//!
//! Auxiliary stages have their own execution semantics and therefore their
//! own ordered candidate plan. They must never inherit whichever ASR
//! candidate happens to be installed on the current thread. This module is
//! the single seam that resolves an auxiliary architecture, constructs its
//! persistent runtime inside the selected candidate context, and rebinds that
//! context for every later replay-safe invocation.

use std::{
    any::Any,
    fmt,
    panic::{self, AssertUnwindSafe},
    sync::Arc,
};

use thiserror::Error;

use crate::device::{
    execution_policy::{
        ExecutionCandidate, ExecutionCandidateFailure, ExecutionIntent, ExecutionPlan,
        ExecutionPolicyError,
    },
    execution_route::enumerate_compute_devices_from_ggml,
};
use crate::ggml_runtime::{AutoGpuPolicy, GgmlCpuGraphBackend, RequestBackendPreference};

use super::{
    admitted_host_object_cache::{
        AdmittedHostObjectCacheLimits, SingleFlightWeightedCache, SingleFlightWeightedLookup,
    },
    aux_pack_registry::{
        AuxiliaryExecutionPolicy, auxiliary_execution_policy, auxiliary_runtime_ownership,
    },
    native_execution_services::{
        CandidateActivationQuoteSource, ExecutionLaneKey, NativeExecutionServices,
        current_execution_cache_attempt_id, current_execution_lane_key,
        drop_execution_candidate_value_without_cache_publication,
        install_candidate_activation_quote, run_execution_candidate_attempt,
        stage_execution_cache_commit,
    },
    system_memory_owner::{AdmittedHostObject, SystemMemoryOwner},
};

#[derive(Debug, Error)]
pub(crate) enum AuxiliaryExecutionPlanError {
    #[error("auxiliary runtime architecture '{architecture_id}' has no execution policy")]
    UnregisteredArchitecture { architecture_id: &'static str },
    #[error("could not resolve an auxiliary execution candidate: {0}")]
    Policy(#[from] ExecutionPolicyError),
}

/// Resolves one auxiliary architecture without inheriting an ASR placement.
///
/// Every registered stage preserves the request intent while applying the
/// auxiliary architecture's own capabilities and Auto policy.
pub(crate) fn resolve_auxiliary_execution_plan(
    execution_services: &NativeExecutionServices,
    architecture_id: &'static str,
    request_intent: &ExecutionIntent,
) -> Result<ExecutionPlan, AuxiliaryExecutionPlanError> {
    let policy = auxiliary_execution_policy(architecture_id)
        .ok_or(AuxiliaryExecutionPlanError::UnregisteredArchitecture { architecture_id })?;
    let ownership = auxiliary_runtime_ownership(architecture_id)
        .ok_or(AuxiliaryExecutionPlanError::UnregisteredArchitecture { architecture_id })?;
    crate::stage_timing::log_detail_event(
        "native_auxiliary_runtime",
        format_args!(
            "stage=plan event=resolve architecture={architecture_id} ownership={}",
            ownership.as_str()
        ),
    );
    let AuxiliaryExecutionPolicy::RequestScoped {
        capabilities,
        auto_gpu_policy,
    } = policy;
    let intent = request_intent.clone();
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(intent, auto_gpu_policy, capabilities, &inventory)
        .map_err(AuxiliaryExecutionPlanError::from)
}

pub(crate) fn resolved_runtime_for_auxiliary_candidate(
    candidate: &ExecutionCandidate,
) -> crate::ggml_runtime::ResolvedFamilyRuntimeInput {
    let preference = match candidate.placement {
        crate::device::execution_policy::ExecutionPlacement::CpuOnly => {
            Some(RequestBackendPreference::CpuOnly)
        }
        crate::device::execution_policy::ExecutionPlacement::FullDevice
        | crate::device::execution_policy::ExecutionPlacement::Hybrid => Some(
            RequestBackendPreference::Exact(candidate.device.route.clone()),
        ),
    };
    // The policy resolver has already selected this candidate. Mapping its
    // CpuOnly/Exact route into the runtime must not re-apply a family Auto
    // policy or force every caller to repeat descriptor-owned policy data.
    crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(preference, AutoGpuPolicy::AllBackends)
}

type AuxiliaryRuntimeBuilder<R, E> =
    Arc<dyn Fn(&ExecutionCandidate) -> Result<R, E> + Send + Sync + 'static>;

/// Failure at the policy seam. Ordinary model/input errors never authorize a
/// candidate change; only the typed failure side channel can produce
/// `CandidatesExhausted`.
#[derive(Debug)]
pub(crate) enum PolicyResolvedAuxRuntimeError<E> {
    Operation(E),
    CandidateFailed {
        stage: &'static str,
        failure: ExecutionCandidateFailure,
        source: Option<E>,
    },
    CandidatesExhausted {
        stage: &'static str,
        failure: ExecutionCandidateFailure,
        source: Option<E>,
    },
    EmptyPlan {
        stage: &'static str,
    },
}

impl<E: fmt::Display> fmt::Display for PolicyResolvedAuxRuntimeError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation(error) => error.fmt(formatter),
            Self::CandidateFailed {
                stage,
                failure,
                source,
            } => {
                write!(
                    formatter,
                    "pinned auxiliary stage '{stage}' failed with {:?} at {}: {}",
                    failure.kind, failure.operation, failure.detail
                )?;
                if let Some(source) = source {
                    write!(formatter, ": {source}")?;
                }
                Ok(())
            }
            Self::CandidatesExhausted {
                stage,
                failure,
                source,
            } => {
                write!(
                    formatter,
                    "auxiliary stage '{stage}' exhausted its execution plan after {:?} at {}: {}",
                    failure.kind, failure.operation, failure.detail
                )?;
                if let Some(source) = source {
                    write!(formatter, ": {source}")?;
                }
                Ok(())
            }
            Self::EmptyPlan { stage } => {
                write!(
                    formatter,
                    "auxiliary stage '{stage}' received an empty execution plan"
                )
            }
        }
    }
}

impl<E> std::error::Error for PolicyResolvedAuxRuntimeError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::CandidateFailed {
                source: Some(error),
                ..
            } => Some(error),
            Self::CandidatesExhausted {
                source: Some(error),
                ..
            } => Some(error),
            Self::CandidateFailed { source: None, .. } => None,
            Self::CandidatesExhausted { source: None, .. } | Self::EmptyPlan { .. } => None,
        }
    }
}

/// One persistent auxiliary runtime bound to its own execution plan.
///
/// The build closure is retained so a replay-safe invocation can discard a
/// failed candidate, construct the next one, and retry the same pure
/// operation. Stateful/non-replayable operations must not use
/// [`Self::invoke_replay_safe`].
pub(crate) struct PolicyResolvedAuxRuntime<R, E> {
    execution_services: Arc<NativeExecutionServices>,
    execution_plan: ExecutionPlan,
    candidate_index: usize,
    runtime: Option<R>,
    builder: AuxiliaryRuntimeBuilder<R, E>,
    stage: &'static str,
    activation_quote: CandidateActivationQuoteSource,
}

impl<R, E> PolicyResolvedAuxRuntime<R, E> {
    pub(crate) fn try_new(
        execution_services: Arc<NativeExecutionServices>,
        execution_plan: ExecutionPlan,
        stage: &'static str,
        builder: AuxiliaryRuntimeBuilder<R, E>,
        activation_quote: CandidateActivationQuoteSource,
    ) -> Result<Self, PolicyResolvedAuxRuntimeError<E>> {
        let (candidate_index, runtime) = Self::construct_from(
            execution_services.as_ref(),
            &execution_plan,
            stage,
            builder.as_ref(),
            &activation_quote,
            0,
        )?;
        Ok(Self {
            execution_services,
            execution_plan,
            candidate_index,
            runtime: Some(runtime),
            builder,
            stage,
            activation_quote,
        })
    }

    fn construct_from(
        execution_services: &NativeExecutionServices,
        execution_plan: &ExecutionPlan,
        stage: &'static str,
        builder: &(dyn Fn(&ExecutionCandidate) -> Result<R, E> + Send + Sync),
        activation_quote: &CandidateActivationQuoteSource,
        start_index: usize,
    ) -> Result<(usize, R), PolicyResolvedAuxRuntimeError<E>> {
        let candidates = execution_plan.candidates();
        for (candidate_index, candidate) in candidates.iter().enumerate().skip(start_index) {
            let _quote = install_candidate_activation_quote(activation_quote.clone());
            let attempt = run_execution_candidate_attempt(execution_services, candidate, || {
                builder(candidate)
            });
            match (attempt.result, attempt.candidate_failure) {
                (Ok(runtime), None) => {
                    log_auxiliary_candidate_selected(stage, candidate);
                    return Ok((candidate_index, runtime));
                }
                (Err(error), None) => {
                    return Err(PolicyResolvedAuxRuntimeError::Operation(error));
                }
                (result, Some(failure)) => {
                    if candidate_index + 1 == candidates.len() {
                        return Err(PolicyResolvedAuxRuntimeError::CandidatesExhausted {
                            stage,
                            failure,
                            source:
                                super::native_execution_services::execution_candidate_failure_source(
                                    result,
                                ),
                        });
                    }
                    log_auxiliary_candidate_retry(stage, "build", candidate, &failure);
                    let _ = super::native_execution_services::execution_candidate_failure_source(
                        result,
                    );
                }
            }
        }
        Err(PolicyResolvedAuxRuntimeError::EmptyPlan { stage })
    }

    /// Runs a pure/replay-safe operation in the active auxiliary lane. A typed
    /// resource/device failure drops that runtime before constructing the next
    /// candidate; ordinary errors return immediately and never change lanes.
    pub(crate) fn invoke_replay_safe<T>(
        &mut self,
        mut operation: impl FnMut(&R) -> Result<T, E>,
    ) -> Result<T, PolicyResolvedAuxRuntimeError<E>> {
        if self.runtime.is_none() {
            return Err(PolicyResolvedAuxRuntimeError::EmptyPlan { stage: self.stage });
        }
        loop {
            let candidate = self.execution_plan.candidates()[self.candidate_index].clone();
            let _quote = install_candidate_activation_quote(self.activation_quote.clone());
            let attempt = run_execution_candidate_attempt(
                self.execution_services.as_ref(),
                &candidate,
                || {
                    operation(
                        self.runtime
                            .as_ref()
                            .expect("an active auxiliary candidate owns a runtime"),
                    )
                },
            );
            match (attempt.result, attempt.candidate_failure) {
                (Ok(value), None) => return Ok(value),
                (Err(error), None) => {
                    return Err(PolicyResolvedAuxRuntimeError::Operation(error));
                }
                (result, Some(failure)) => {
                    let source =
                        super::native_execution_services::execution_candidate_failure_source(
                            result,
                        );
                    let failed_runtime = self
                        .runtime
                        .take()
                        .expect("a failed auxiliary candidate owns a runtime");
                    drop_execution_candidate_value_without_cache_publication(failed_runtime);
                    let next_index = self.candidate_index.saturating_add(1);
                    if next_index >= self.execution_plan.candidates().len() {
                        return Err(PolicyResolvedAuxRuntimeError::CandidatesExhausted {
                            stage: self.stage,
                            failure,
                            source,
                        });
                    }
                    log_auxiliary_candidate_retry(
                        self.stage,
                        "invoke-replay-safe",
                        &candidate,
                        &failure,
                    );
                    let (candidate_index, runtime) = Self::construct_from(
                        self.execution_services.as_ref(),
                        &self.execution_plan,
                        self.stage,
                        self.builder.as_ref(),
                        &self.activation_quote,
                        next_index,
                    )?;
                    self.candidate_index = candidate_index;
                    self.runtime = Some(runtime);
                }
            }
        }
    }

    /// Runs an operation in the active lane without ever advancing the plan.
    /// Stateful auxiliary stages switch to this mode after their first
    /// externally observable output: replaying a later request on a fresh
    /// candidate could violate session continuity even when the request itself
    /// looks syntactically pure.
    pub(crate) fn invoke_pinned<T>(
        &mut self,
        operation: impl FnOnce(&R) -> Result<T, E>,
    ) -> Result<T, PolicyResolvedAuxRuntimeError<E>> {
        if self.runtime.is_none() {
            return Err(PolicyResolvedAuxRuntimeError::EmptyPlan { stage: self.stage });
        }
        let candidate = self.execution_plan.candidates()[self.candidate_index].clone();
        let _quote = install_candidate_activation_quote(self.activation_quote.clone());
        let attempt =
            run_execution_candidate_attempt(self.execution_services.as_ref(), &candidate, || {
                operation(
                    self.runtime
                        .as_ref()
                        .expect("an active auxiliary candidate owns a runtime"),
                )
            });
        match (attempt.result, attempt.candidate_failure) {
            (Ok(value), None) => Ok(value),
            (Err(error), None) => Err(PolicyResolvedAuxRuntimeError::Operation(error)),
            (result, Some(failure)) => {
                let source =
                    super::native_execution_services::execution_candidate_failure_source(result);
                let failed_runtime = self
                    .runtime
                    .take()
                    .expect("a failed pinned auxiliary candidate owns a runtime");
                drop_execution_candidate_value_without_cache_publication(failed_runtime);
                Err(PolicyResolvedAuxRuntimeError::CandidateFailed {
                    stage: self.stage,
                    failure,
                    source,
                })
            }
        }
    }

    #[cfg(test)]
    fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    #[cfg(test)]
    fn runtime_for_test(&self) -> Option<&R> {
        self.runtime.as_ref()
    }
}

/// Stateful auxiliary lane whose replay frontier closes permanently after
/// the first successful operation. This is the reusable policy primitive for
/// stages such as incremental translation: before any result can escape, a
/// typed candidate failure may rebuild and replay; afterward, changing lanes
/// would lose hidden session state and is therefore forbidden.
pub(crate) struct PolicyResolvedStatefulAuxRuntime<R, E> {
    runtime: PolicyResolvedAuxRuntime<R, E>,
    output_committed: bool,
}

impl<R, E> PolicyResolvedStatefulAuxRuntime<R, E> {
    pub(crate) const fn new(runtime: PolicyResolvedAuxRuntime<R, E>) -> Self {
        Self {
            runtime,
            output_committed: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn invoke<T>(
        &mut self,
        mut operation: impl FnMut(&R) -> Result<T, E>,
    ) -> Result<T, PolicyResolvedAuxRuntimeError<E>> {
        self.invoke_with_commit(|runtime| operation(runtime).map(|value| (value, true)))
    }

    /// Stateful invocation whose result explicitly declares whether externally
    /// observable state was produced. Buffered streaming inputs can therefore
    /// remain replay-safe until the first real model decision instead of
    /// pinning a lane merely because an operation returned `Ok`.
    pub(crate) fn invoke_with_commit<T>(
        &mut self,
        mut operation: impl FnMut(&R) -> Result<(T, bool), E>,
    ) -> Result<T, PolicyResolvedAuxRuntimeError<E>> {
        let result = if self.output_committed {
            self.runtime.invoke_pinned(|runtime| operation(runtime))
        } else {
            self.runtime
                .invoke_replay_safe(|runtime| operation(runtime))
        };
        match result {
            Ok((value, commit)) => {
                self.output_committed |= commit;
                Ok(value)
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) const fn output_committed(&self) -> bool {
        self.output_committed
    }

    #[cfg(test)]
    fn candidate_index(&self) -> usize {
        self.runtime.candidate_index()
    }

    #[cfg(test)]
    pub(crate) fn runtime_for_test(&self) -> Option<&R> {
        self.runtime.runtime_for_test()
    }
}

fn log_auxiliary_candidate_selected(stage: &'static str, candidate: &ExecutionCandidate) {
    crate::stage_timing::log_detail_event(
        "native_auxiliary_runtime",
        format_args!(
            "stage=execution_candidate event=selected auxiliary_stage={stage} provider={} placement={:?} stable_id={} backend_kind={:?}",
            candidate.device.route.provider,
            candidate.placement,
            candidate.device.route.stable_id,
            candidate.device.ggml_kind,
        ),
    );
}

fn log_auxiliary_candidate_retry(
    stage: &'static str,
    operation: &'static str,
    candidate: &ExecutionCandidate,
    failure: &ExecutionCandidateFailure,
) {
    crate::stage_timing::log_detail_event(
        "native_auxiliary_runtime",
        format_args!(
            "stage=execution_candidate event=retry auxiliary_stage={stage} operation={operation} provider={} placement={:?} failure={:?} failure_operation={}",
            candidate.device.route.provider, candidate.placement, failure.kind, failure.operation,
        ),
    );
}

/// Host representation is part of auxiliary cache identity. The content id
/// alone cannot distinguish two legal materializations of the same pack (for
/// example a CPU-native and an uploaded representation), while a Rust type
/// alone cannot distinguish schema revisions that intentionally reuse a
/// wrapper. Both axes are therefore mandatory and checked before lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AuxiliaryHostRepresentationKey {
    representation_id: &'static str,
    owner_type: &'static str,
}

impl AuxiliaryHostRepresentationKey {
    pub(crate) fn admitted<T: Send + Sync + 'static>(representation_id: &'static str) -> Self {
        Self {
            representation_id,
            owner_type: std::any::type_name::<T>(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AuxiliaryRuntimeCacheKey {
    architecture_id: &'static str,
    pack_content_id: String,
    host_representation: AuxiliaryHostRepresentationKey,
    lane: Option<ExecutionLaneKey>,
}

/// Content/representation/physical-lane identity for a runtime that remains
/// on a dedicated owner thread. Unlike [`AuxiliaryRuntimeCacheKey`], this key
/// does not require `R: Send + Sync`; only its process-side actor handle crosses
/// threads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AuxiliaryPinnedRuntimeCacheKey {
    architecture_id: &'static str,
    pack_content_id: String,
    representation_id: &'static str,
    runtime_type: &'static str,
    lane: ExecutionLaneKey,
}

impl AuxiliaryPinnedRuntimeCacheKey {
    pub(crate) fn for_current_lane<R: 'static>(
        architecture_id: &'static str,
        pack_content_id: impl Into<String>,
        representation_id: &'static str,
        backend: GgmlCpuGraphBackend,
    ) -> Self {
        Self {
            architecture_id,
            pack_content_id: pack_content_id.into(),
            representation_id,
            runtime_type: std::any::type_name::<R>(),
            lane: current_execution_lane_key(backend),
        }
    }

    pub(crate) fn has_content_id(&self, pack_content_id: &str) -> bool {
        self.pack_content_id == pack_content_id
    }
}

impl AuxiliaryRuntimeCacheKey {
    /// Identifies an immutable host representation whose bytes and semantics
    /// do not depend on the execution backend. CPU and accelerator candidates
    /// deliberately share this owner; lane-specific device state belongs in a
    /// pinned runtime actor keyed by [`AuxiliaryPinnedRuntimeCacheKey`].
    pub(crate) fn host_neutral<T: Send + Sync + 'static>(
        architecture_id: &'static str,
        pack_content_id: impl Into<String>,
        representation_id: &'static str,
    ) -> Self {
        Self {
            architecture_id,
            pack_content_id: pack_content_id.into(),
            host_representation: AuxiliaryHostRepresentationKey::admitted::<T>(representation_id),
            lane: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_current_lane<T: Send + Sync + 'static>(
        architecture_id: &'static str,
        pack_content_id: impl Into<String>,
        representation_id: &'static str,
        backend: GgmlCpuGraphBackend,
    ) -> Self {
        Self {
            architecture_id,
            pack_content_id: pack_content_id.into(),
            host_representation: AuxiliaryHostRepresentationKey::admitted::<T>(representation_id),
            lane: Some(current_execution_lane_key(backend)),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum AuxiliaryRuntimeCacheError {
    #[error("auxiliary runtime cache lock is poisoned")]
    Poisoned,
    #[error(
        "auxiliary runtime key representation '{representation_id}' declares owner type '{declared}', requested '{requested}'"
    )]
    OwnerTypeMismatch {
        representation_id: &'static str,
        declared: &'static str,
        requested: &'static str,
    },
    #[error("auxiliary runtime build panicked: {0}")]
    BuildPanicked(String),
}

#[derive(Clone)]
struct ErasedAdmittedHostObject {
    owner: Arc<dyn Any + Send + Sync>,
    committed_requested_bytes: u64,
}

impl ErasedAdmittedHostObject {
    fn new<T: Send + Sync + 'static>(owner: AdmittedHostObject<T>) -> Self {
        let committed_requested_bytes = owner.committed_requested_bytes();
        let owner: Arc<dyn Any + Send + Sync> = owner;
        Self {
            owner,
            committed_requested_bytes,
        }
    }

    fn downcast<T: Send + Sync + 'static>(
        &self,
    ) -> Result<AdmittedHostObject<T>, AuxiliaryRuntimeCacheError> {
        Arc::clone(&self.owner)
            .downcast::<SystemMemoryOwner<T>>()
            .map_err(|_| AuxiliaryRuntimeCacheError::OwnerTypeMismatch {
                representation_id: "<erased-owner>",
                declared: "<erased-owner>",
                requested: std::any::type_name::<T>(),
            })
    }
}

/// Process-root-owned, content/representation/lane keyed cache for persistent
/// auxiliary owners. It is a byte-weighted single-flight LRU, and publication
/// participates in the candidate journal: another thread waits while an owner
/// is staged, then sees either the committed owner or a clean retryable slot.
pub(crate) struct AuxiliaryRuntimeOwnerCache {
    core: SingleFlightWeightedCache<AuxiliaryRuntimeCacheKey, ErasedAdmittedHostObject>,
}

impl Default for AuxiliaryRuntimeOwnerCache {
    fn default() -> Self {
        Self::new(AdmittedHostObjectCacheLimits::new(
            8,
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
        ))
    }
}

impl fmt::Debug for AuxiliaryRuntimeOwnerCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(test)]
        let usage = Some(self.core.usage_for_test());
        #[cfg(not(test))]
        let usage: Option<(usize, u64)> = None;
        formatter
            .debug_struct("AuxiliaryRuntimeOwnerCache")
            .field("usage", &usage)
            .finish()
    }
}

impl AuxiliaryRuntimeOwnerCache {
    pub(crate) fn new(limits: AdmittedHostObjectCacheLimits) -> Self {
        Self {
            core: SingleFlightWeightedCache::new(limits),
        }
    }

    pub(crate) fn get_or_try_insert_admitted_with<T, E>(
        &self,
        key: AuxiliaryRuntimeCacheKey,
        quoted_committed_requested_bytes: u64,
        build: impl FnOnce() -> Result<AdmittedHostObject<T>, E>,
        map_cache_error: impl Fn(AuxiliaryRuntimeCacheError) -> E,
    ) -> Result<AdmittedHostObject<T>, E>
    where
        T: Send + Sync + 'static,
    {
        let requested_type = std::any::type_name::<T>();
        if key.host_representation.owner_type != requested_type {
            return Err(map_cache_error(
                AuxiliaryRuntimeCacheError::OwnerTypeMismatch {
                    representation_id: key.host_representation.representation_id,
                    declared: key.host_representation.owner_type,
                    requested: requested_type,
                },
            ));
        }
        let attempt_id = current_execution_cache_attempt_id();
        match self
            .core
            .lookup_or_reserve(key, attempt_id)
            .map_err(|_| map_cache_error(AuxiliaryRuntimeCacheError::Poisoned))?
        {
            SingleFlightWeightedLookup::Ready(owner) => {
                owner.downcast::<T>().map_err(map_cache_error)
            }
            SingleFlightWeightedLookup::Build(permit) => {
                let retain = permit
                    .make_room_for(quoted_committed_requested_bytes)
                    .map_err(|_| map_cache_error(AuxiliaryRuntimeCacheError::Poisoned))?;
                let owner = match panic::catch_unwind(AssertUnwindSafe(build)) {
                    Ok(Ok(owner)) => owner,
                    Ok(Err(error)) => return Err(error),
                    Err(payload) => {
                        return Err(map_cache_error(AuxiliaryRuntimeCacheError::BuildPanicked(
                            describe_panic_payload(payload.as_ref()),
                        )));
                    }
                };
                let erased = ErasedAdmittedHostObject::new(Arc::clone(&owner));
                let actual_weight = erased.committed_requested_bytes;
                if let Some(attempt_id) = attempt_id {
                    let publication = permit
                        .stage(erased, actual_weight, retain, attempt_id)
                        .map_err(|_| map_cache_error(AuxiliaryRuntimeCacheError::Poisoned))?;
                    stage_execution_cache_commit(move || {
                        let _ = publication.commit();
                    });
                } else {
                    permit
                        .publish(erased, actual_weight, retain)
                        .map_err(|_| map_cache_error(AuxiliaryRuntimeCacheError::Poisoned))?;
                }
                Ok(owner)
            }
        }
    }

    pub(crate) fn clear(&self) {
        self.core.clear();
    }

    pub(crate) fn evict_content_id(&self, pack_content_id: &str) {
        self.core
            .evict_where(|key| key.pack_content_id == pack_content_id);
    }

    #[cfg(test)]
    fn usage_for_test(&self) -> (usize, u64) {
        self.core.usage_for_test()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.core.len_for_test()
    }
}

fn describe_panic_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use crate::device::{
        execution_memory::{DeviceMemoryBrokerSet, DeviceMemoryPolicy},
        execution_policy::{
            ExecutionCandidateFailure, ExecutionDeviceSnapshot, ExecutionPlacement,
        },
        execution_route::{
            DeviceAddressability, ExecutionProvider, ResolvedExecutionRoute, RouteDeviceKind,
        },
    };
    use crate::ggml_runtime::GgmlBackendKind;

    use super::*;
    use crate::models::native_execution_services::{
        CandidateActivationQuoteSource, install_candidate_activation_quote,
        record_current_execution_candidate_failure,
    };
    use crate::models::system_memory_owner::SystemMemoryAllocationQuote;

    fn test_aux_activation_quote(stage: &str) -> CandidateActivationQuoteSource {
        CandidateActivationQuoteSource::Declared(
            SystemMemoryAllocationQuote::new(
                format!("test-aux.{stage}.declared-resident"),
                64 * 1024,
                64 * 1024,
            )
            .expect("test aux declared resident"),
        )
    }

    fn candidate(provider: ExecutionProvider, stable_id: &str) -> ExecutionCandidate {
        ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route: ResolvedExecutionRoute {
                    provider,
                    stable_id: stable_id.to_string(),
                    registry_ordinal: 0,
                    kind: if provider == ExecutionProvider::Cpu {
                        RouteDeviceKind::Cpu
                    } else {
                        RouteDeviceKind::Accelerated
                    },
                    addressability: DeviceAddressability::NotExactlyAddressable {
                        reason: "synthetic auxiliary policy test route",
                    },
                },
                ggml_kind: if provider == ExecutionProvider::Cpu {
                    GgmlBackendKind::Cpu
                } else {
                    GgmlBackendKind::Gpu
                },
                memory: None,
                buffer_alignment: None,
            },
            placement: if provider == ExecutionProvider::Cpu {
                ExecutionPlacement::CpuOnly
            } else {
                ExecutionPlacement::FullDevice
            },
        }
    }

    fn services() -> Arc<NativeExecutionServices> {
        Arc::new(
            NativeExecutionServices::new_with_broker(
                Arc::new(crate::device::execution_policy::DefaultExecutionPolicyResolver),
                Arc::new(DeviceMemoryBrokerSet::new(DeviceMemoryPolicy::default())),
            )
            .unwrap(),
        )
    }

    #[test]
    fn replay_safe_invocation_rebuilds_only_after_typed_failure() {
        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let builds_for_closure = Arc::clone(&builds);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            builds_for_closure.fetch_add(1, Ordering::SeqCst);
            Ok::<_, &'static str>(candidate.device.route.provider)
        });
        let mut runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-aux",
            builder,
            test_aux_activation_quote("test-aux"),
        )
        .unwrap();

        let value = runtime
            .invoke_replay_safe(|provider| {
                if *provider == ExecutionProvider::Vulkan {
                    record_current_execution_candidate_failure(
                        ExecutionCandidateFailure::capacity("test-invoke", "full"),
                    );
                    return Err("gpu full");
                }
                Ok(*provider)
            })
            .unwrap();

        assert_eq!(value, ExecutionProvider::Cpu);
        assert_eq!(runtime.candidate_index(), 1);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn replay_safe_invocation_drops_failed_value_and_runtime_before_rebuild() {
        struct RuntimeDropProbe {
            provider: ExecutionProvider,
            dropped: Arc<AtomicBool>,
            published: Arc<Mutex<Vec<&'static str>>>,
            track_drop: bool,
        }

        impl Drop for RuntimeDropProbe {
            fn drop(&mut self) {
                if self.track_drop {
                    self.dropped.store(true, Ordering::SeqCst);
                    let published = Arc::clone(&self.published);
                    stage_execution_cache_commit(move || {
                        published.lock().unwrap().push("failed-runtime")
                    });
                }
            }
        }

        struct ResultDropProbe {
            dropped: Arc<AtomicBool>,
            track_drop: bool,
        }

        impl Drop for ResultDropProbe {
            fn drop(&mut self) {
                if self.track_drop {
                    self.dropped.store(true, Ordering::SeqCst);
                }
            }
        }

        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let runtime_dropped = Arc::new(AtomicBool::new(false));
        let result_dropped = Arc::new(AtomicBool::new(false));
        let published = Arc::new(Mutex::new(Vec::new()));
        let runtime_dropped_for_builder = Arc::clone(&runtime_dropped);
        let result_dropped_for_builder = Arc::clone(&result_dropped);
        let published_for_builder = Arc::clone(&published);
        let builder = Arc::new(move |candidate: &ExecutionCandidate| {
            if candidate.device.route.provider == ExecutionProvider::Cpu {
                assert!(
                    result_dropped_for_builder.load(Ordering::SeqCst),
                    "failed operation value must be destroyed before replacement admission"
                );
                assert!(
                    runtime_dropped_for_builder.load(Ordering::SeqCst),
                    "failed runtime must be destroyed before replacement admission"
                );
                assert!(
                    published_for_builder.lock().unwrap().is_empty(),
                    "failed runtime Drop must not republish candidate-local cache state"
                );
            }
            Ok::<_, &'static str>(RuntimeDropProbe {
                provider: candidate.device.route.provider,
                dropped: Arc::clone(&runtime_dropped_for_builder),
                published: Arc::clone(&published_for_builder),
                track_drop: candidate.device.route.provider == ExecutionProvider::Vulkan,
            })
        });
        let mut runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-drop-order",
            builder,
            test_aux_activation_quote("test-drop-order"),
        )
        .unwrap();

        let output = runtime
            .invoke_replay_safe(|runtime| {
                let track_drop = runtime.provider == ExecutionProvider::Vulkan;
                if track_drop {
                    record_current_execution_candidate_failure(
                        ExecutionCandidateFailure::capacity("test-invoke", "full"),
                    );
                }
                Ok(ResultDropProbe {
                    dropped: Arc::clone(&result_dropped),
                    track_drop,
                })
            })
            .unwrap();

        assert_eq!(runtime.candidate_index(), 1);
        assert!(runtime_dropped.load(Ordering::SeqCst));
        assert!(result_dropped.load(Ordering::SeqCst));
        assert!(published.lock().unwrap().is_empty());
        drop(output);
    }

    #[test]
    fn ordinary_error_never_advances_auxiliary_candidate() {
        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let mut runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-aux",
            Arc::new(|candidate: &ExecutionCandidate| {
                Ok::<_, &'static str>(candidate.device.route.provider)
            }),
            test_aux_activation_quote("test-aux"),
        )
        .unwrap();

        assert!(matches!(
            runtime.invoke_replay_safe::<()>(|_| Err("ordinary")),
            Err(PolicyResolvedAuxRuntimeError::Operation("ordinary"))
        ));
        assert_eq!(runtime.candidate_index(), 0);
    }

    #[test]
    fn pinned_invocation_never_rebuilds_after_typed_failure() {
        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let builds_for_closure = Arc::clone(&builds);
        let mut runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-pinned-aux",
            Arc::new(move |candidate: &ExecutionCandidate| {
                builds_for_closure.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>(candidate.device.route.provider)
            }),
            test_aux_activation_quote("test-pinned-aux"),
        )
        .unwrap();

        let result = runtime.invoke_pinned::<()>(|_| {
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "test-pinned",
                "lost",
            ));
            Err("lost")
        });

        assert!(matches!(
            result,
            Err(PolicyResolvedAuxRuntimeError::CandidateFailed { .. })
        ));
        assert_eq!(runtime.candidate_index(), 0);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(matches!(
            runtime.invoke_pinned(|_| Ok::<_, &'static str>(())),
            Err(PolicyResolvedAuxRuntimeError::EmptyPlan { .. })
        ));
    }

    #[test]
    fn stateful_runtime_replays_before_first_success_then_pins_the_lane() {
        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let builds_for_closure = Arc::clone(&builds);
        let runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-stateful-aux",
            Arc::new(move |candidate: &ExecutionCandidate| {
                builds_for_closure.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>(candidate.device.route.provider)
            }),
            test_aux_activation_quote("test-stateful-aux"),
        )
        .unwrap();
        let mut runtime = PolicyResolvedStatefulAuxRuntime::new(runtime);

        let first = runtime
            .invoke(|provider| {
                if *provider == ExecutionProvider::Vulkan {
                    record_current_execution_candidate_failure(
                        ExecutionCandidateFailure::capacity("test-first-output", "full"),
                    );
                    return Err("gpu full");
                }
                Ok(*provider)
            })
            .unwrap();
        assert_eq!(first, ExecutionProvider::Cpu);
        assert!(runtime.output_committed());
        assert_eq!(runtime.candidate_index(), 1);

        let later = runtime.invoke::<()>(|_| {
            record_current_execution_candidate_failure(ExecutionCandidateFailure::device_lost(
                "test-after-output",
                "lost",
            ));
            Err("lost")
        });
        assert!(matches!(
            later,
            Err(PolicyResolvedAuxRuntimeError::CandidateFailed { .. })
        ));
        assert_eq!(runtime.candidate_index(), 1);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert!(matches!(
            runtime.invoke(|_| Ok::<_, &'static str>(())),
            Err(PolicyResolvedAuxRuntimeError::EmptyPlan { .. })
        ));
    }

    #[test]
    fn buffered_success_keeps_stateful_lane_replay_safe_until_output() {
        let services = services();
        let plan = ExecutionPlan::for_test(
            ExecutionIntent::Auto,
            vec![
                candidate(ExecutionProvider::Vulkan, "gpu-0"),
                candidate(ExecutionProvider::Cpu, "cpu"),
            ],
        );
        let runtime = PolicyResolvedAuxRuntime::try_new(
            services,
            plan,
            "test-buffered-stateful-aux",
            Arc::new(|candidate: &ExecutionCandidate| {
                Ok::<_, &'static str>(candidate.device.route.provider)
            }),
            test_aux_activation_quote("test-buffered-stateful-aux"),
        )
        .unwrap();
        let mut runtime = PolicyResolvedStatefulAuxRuntime::new(runtime);

        let buffered = runtime
            .invoke_with_commit(|provider| Ok((*provider, false)))
            .unwrap();
        assert_eq!(buffered, ExecutionProvider::Vulkan);
        assert!(!runtime.output_committed());

        let recovered = runtime
            .invoke_with_commit(|provider| {
                if *provider == ExecutionProvider::Vulkan {
                    record_current_execution_candidate_failure(
                        ExecutionCandidateFailure::device_lost(
                            "test-first-decision",
                            "lost after buffered input",
                        ),
                    );
                    return Err("lost");
                }
                Ok((*provider, true))
            })
            .unwrap();
        assert_eq!(recovered, ExecutionProvider::Cpu);
        assert!(runtime.output_committed());
        assert_eq!(runtime.candidate_index(), 1);
    }

    #[test]
    fn owner_cache_reuses_content_and_lane() {
        let cache = AuxiliaryRuntimeOwnerCache::default();
        let builds = AtomicUsize::new(0);
        let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
            "test",
            "sha256:test",
            "test.usize.v1",
            GgmlCpuGraphBackend::Cpu,
        );
        let first = cache
            .get_or_try_insert_admitted_with(
                key.clone(),
                8,
                || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                        SystemMemoryOwner::with_committed_requested_bytes_for_test(7_usize, 8),
                    ))
                },
                |error| error,
            )
            .unwrap();
        let second = cache
            .get_or_try_insert_admitted_with(
                key,
                8,
                || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                        SystemMemoryOwner::with_committed_requested_bytes_for_test(9_usize, 8),
                    ))
                },
                |error| error,
            )
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(**second, 7);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn host_neutral_owner_is_shared_while_lane_bound_owners_remain_isolated() {
        let neutral_cpu = AuxiliaryRuntimeCacheKey::host_neutral::<usize>(
            "test",
            "sha256:neutral",
            "test.host-neutral.v1",
        );
        let neutral_metal = AuxiliaryRuntimeCacheKey::host_neutral::<usize>(
            "test",
            "sha256:neutral",
            "test.host-neutral.v1",
        );
        assert_eq!(neutral_cpu, neutral_metal);
        assert!(neutral_cpu.lane.is_none());

        let cpu = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
            "test",
            "sha256:lane-bound",
            "test.lane-bound.v1",
            GgmlCpuGraphBackend::Cpu,
        );
        let metal = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
            "test",
            "sha256:lane-bound",
            "test.lane-bound.v1",
            GgmlCpuGraphBackend::Metal,
        );
        assert_ne!(cpu, metal);
        assert!(cpu.lane.is_some());
        assert!(metal.lane.is_some());
    }

    #[test]
    fn owner_cache_rejects_a_type_that_disagrees_with_the_representation_key() {
        let cache = AuxiliaryRuntimeOwnerCache::default();
        let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
            "test",
            "sha256:type-check",
            "test.typed.v1",
            GgmlCpuGraphBackend::Cpu,
        );
        let error = cache
            .get_or_try_insert_admitted_with::<String, _>(
                key,
                0,
                || panic!("a mismatched type must be rejected before construction"),
                |error| error,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            AuxiliaryRuntimeCacheError::OwnerTypeMismatch { .. }
        ));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn owner_cache_uses_the_shared_weighted_lru_core() {
        let cache = AuxiliaryRuntimeOwnerCache::new(AdmittedHostObjectCacheLimits::new(1, 8));
        let builds = AtomicUsize::new(0);
        for content_id in ["sha256:a", "sha256:b", "sha256:a"] {
            let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
                "test",
                content_id,
                "test.usize.v1",
                GgmlCpuGraphBackend::Cpu,
            );
            drop(
                cache
                    .get_or_try_insert_admitted_with(
                        key,
                        8,
                        || {
                            let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                            Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                                SystemMemoryOwner::with_committed_requested_bytes_for_test(
                                    value, 8,
                                ),
                            ))
                        },
                        |error| error,
                    )
                    .unwrap(),
            );
        }
        assert_eq!(builds.load(Ordering::SeqCst), 3);
        assert_eq!(cache.usage_for_test(), (1, 8));
    }

    #[test]
    fn same_attempt_sees_staged_owner_and_commit_makes_it_a_global_hit() {
        let services = services();
        let cache = services.auxiliary_runtime_owners();
        let cpu = candidate(ExecutionProvider::Cpu, "cpu");
        let builds = AtomicUsize::new(0);
        let _quote = install_candidate_activation_quote(test_aux_activation_quote("staged-hit"));
        let outcome = run_execution_candidate_attempt(services.as_ref(), &cpu, || {
            let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
                "test",
                "sha256:staged-hit",
                "test.usize.v1",
                GgmlCpuGraphBackend::Cpu,
            );
            let first = cache.get_or_try_insert_admitted_with(
                key.clone(),
                8,
                || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                        SystemMemoryOwner::with_committed_requested_bytes_for_test(7_usize, 8),
                    ))
                },
                |error| error,
            )?;
            let second = cache.get_or_try_insert_admitted_with(
                key,
                8,
                || panic!("the building attempt must observe its staged owner"),
                |error| error,
            )?;
            assert!(Arc::ptr_eq(&first, &second));
            Ok::<_, AuxiliaryRuntimeCacheError>(())
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.candidate_failure.is_none());

        let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
            "test",
            "sha256:staged-hit",
            "test.usize.v1",
            GgmlCpuGraphBackend::Cpu,
        );
        let hit: AdmittedHostObject<usize> = cache
            .get_or_try_insert_admitted_with(
                key,
                8,
                || panic!("a committed staged owner must be ready"),
                |error| error,
            )
            .unwrap();
        assert_eq!(**hit, 7);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn other_attempt_waits_and_rebuilds_after_staged_owner_rolls_back() {
        let services = services();
        let cache = Arc::new(AuxiliaryRuntimeOwnerCache::new(
            AdmittedHostObjectCacheLimits::new(1, 8),
        ));
        let builds = Arc::new(AtomicUsize::new(0));
        let (staged_tx, staged_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let failed_services = Arc::clone(&services);
        let failed_cache = Arc::clone(&cache);
        let failed_builds = Arc::clone(&builds);
        let failed_candidate = candidate(ExecutionProvider::Cpu, "cpu");
        let failed = std::thread::spawn(move || {
            let _quote = install_candidate_activation_quote(test_aux_activation_quote("rollback"));
            run_execution_candidate_attempt(failed_services.as_ref(), &failed_candidate, || {
                let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
                    "test",
                    "sha256:rollback",
                    "test.usize.v1",
                    GgmlCpuGraphBackend::Cpu,
                );
                let owner = failed_cache.get_or_try_insert_admitted_with(
                    key,
                    8,
                    || {
                        let value = failed_builds.fetch_add(1, Ordering::SeqCst) + 1;
                        Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                            SystemMemoryOwner::with_committed_requested_bytes_for_test(value, 8),
                        ))
                    },
                    |error| error,
                )?;
                record_current_execution_candidate_failure(ExecutionCandidateFailure::capacity(
                    "test-rollback",
                    "fail after staging",
                ));
                staged_tx.send(()).expect("signal staged owner");
                release_rx.recv().expect("release failed attempt");
                drop(owner);
                Ok::<_, AuxiliaryRuntimeCacheError>(())
            })
        });

        staged_rx.recv().expect("failed attempt staged owner");
        let waiter_services = Arc::clone(&services);
        let waiter_cache = Arc::clone(&cache);
        let waiter_builds = Arc::clone(&builds);
        let waiter_candidate = candidate(ExecutionProvider::Cpu, "cpu");
        let waiter = std::thread::spawn(move || {
            let _quote = install_candidate_activation_quote(test_aux_activation_quote("waiter"));
            run_execution_candidate_attempt(waiter_services.as_ref(), &waiter_candidate, || {
                let key = AuxiliaryRuntimeCacheKey::for_current_lane::<usize>(
                    "test",
                    "sha256:rollback",
                    "test.usize.v1",
                    GgmlCpuGraphBackend::Cpu,
                );
                waiter_cache.get_or_try_insert_admitted_with(
                    key,
                    8,
                    || {
                        let value = waiter_builds.fetch_add(1, Ordering::SeqCst) + 1;
                        Ok::<_, AuxiliaryRuntimeCacheError>(Arc::new(
                            SystemMemoryOwner::with_committed_requested_bytes_for_test(value, 8),
                        ))
                    },
                    |error| error,
                )
            })
        });
        release_tx.send(()).expect("finish failed attempt");
        let failed_outcome = failed.join().expect("failed attempt joins");
        assert!(failed_outcome.candidate_failure.is_some());
        let waiter_outcome = waiter.join().expect("waiter joins");
        assert!(waiter_outcome.candidate_failure.is_none());
        let rebuilt = waiter_outcome.result.expect("waiter rebuilds");

        assert_eq!(**rebuilt, 2);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(cache.usage_for_test(), (1, 8));
    }
}
