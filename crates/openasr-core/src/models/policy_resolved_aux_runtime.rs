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
        ExecutionLaneKey, NativeExecutionServices, current_execution_cache_attempt_id,
        current_execution_lane_key, run_execution_candidate_attempt, stage_execution_cache_commit,
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
/// `FixedCpu` is an explicit stage topology, so it intentionally ignores the
/// request's accelerated intent. `RequestScoped` preserves that intent but
/// still uses the auxiliary architecture's own capabilities and Auto policy.
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
    let (intent, auto_gpu_policy, capabilities) = match policy {
        AuxiliaryExecutionPolicy::FixedCpu => (
            ExecutionIntent::CpuOnly,
            AutoGpuPolicy::Never,
            crate::device::execution_policy::ExecutionCapabilities::new(true),
        ),
        AuxiliaryExecutionPolicy::RequestScoped {
            capabilities,
            auto_gpu_policy,
        } => (request_intent.clone(), auto_gpu_policy, capabilities),
    };
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(intent, auto_gpu_policy, capabilities, &inventory)
        .map_err(AuxiliaryExecutionPlanError::from)
}

pub(crate) fn resolve_fixed_cpu_execution_plan(
    execution_services: &NativeExecutionServices,
) -> Result<ExecutionPlan, AuxiliaryExecutionPlanError> {
    let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
    execution_services
        .policy_resolver()
        .resolve(
            ExecutionIntent::CpuOnly,
            AutoGpuPolicy::Never,
            crate::device::execution_policy::ExecutionCapabilities::new(true),
            &inventory,
        )
        .map_err(AuxiliaryExecutionPlanError::from)
}

pub(crate) fn resolved_runtime_for_auxiliary_candidate(
    candidate: &ExecutionCandidate,
    auto_gpu_policy: AutoGpuPolicy,
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
    crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(preference, auto_gpu_policy)
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
            }
            | Self::CandidatesExhausted {
                source: Some(error),
                ..
            } => Some(error),
            Self::CandidateFailed { source: None, .. }
            | Self::CandidatesExhausted { source: None, .. }
            | Self::EmptyPlan { .. } => None,
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
}

impl<R, E> PolicyResolvedAuxRuntime<R, E> {
    pub(crate) fn try_new(
        execution_services: Arc<NativeExecutionServices>,
        execution_plan: ExecutionPlan,
        stage: &'static str,
        builder: AuxiliaryRuntimeBuilder<R, E>,
    ) -> Result<Self, PolicyResolvedAuxRuntimeError<E>> {
        let (candidate_index, runtime) = Self::construct_from(
            execution_services.as_ref(),
            &execution_plan,
            stage,
            builder.as_ref(),
            0,
        )?;
        Ok(Self {
            execution_services,
            execution_plan,
            candidate_index,
            runtime: Some(runtime),
            builder,
            stage,
        })
    }

    fn construct_from(
        execution_services: &NativeExecutionServices,
        execution_plan: &ExecutionPlan,
        stage: &'static str,
        builder: &(dyn Fn(&ExecutionCandidate) -> Result<R, E> + Send + Sync),
        start_index: usize,
    ) -> Result<(usize, R), PolicyResolvedAuxRuntimeError<E>> {
        let candidates = execution_plan.candidates();
        for (candidate_index, candidate) in candidates.iter().enumerate().skip(start_index) {
            let attempt = run_execution_candidate_attempt(execution_services, candidate, || {
                builder(candidate)
            });
            match (attempt.result, attempt.candidate_failure) {
                (Ok(runtime), None) => return Ok((candidate_index, runtime)),
                (Err(error), None) => {
                    return Err(PolicyResolvedAuxRuntimeError::Operation(error));
                }
                (result, Some(failure)) => {
                    if candidate_index + 1 == candidates.len() {
                        return Err(PolicyResolvedAuxRuntimeError::CandidatesExhausted {
                            stage,
                            failure,
                            source: result.err(),
                        });
                    }
                    log_auxiliary_candidate_retry(stage, "build", candidate, &failure);
                    // `result` drops here, before the next candidate builds,
                    // releasing every owner returned alongside typed success.
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
        loop {
            let candidate = self.execution_plan.candidates()[self.candidate_index].clone();
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
                    let next_index = self.candidate_index.saturating_add(1);
                    if next_index >= self.execution_plan.candidates().len() {
                        return Err(PolicyResolvedAuxRuntimeError::CandidatesExhausted {
                            stage: self.stage,
                            failure,
                            source: result.err(),
                        });
                    }
                    log_auxiliary_candidate_retry(
                        self.stage,
                        "invoke-replay-safe",
                        &candidate,
                        &failure,
                    );
                    // The operation may have returned a value while a lower
                    // layer recorded a typed failure. Destroy that value and
                    // the failed runtime before the next candidate quotes or
                    // allocates anything: both may retain candidate-local
                    // buffers or committed leases.
                    drop(result);
                    self.runtime.take();
                    let (candidate_index, runtime) = Self::construct_from(
                        self.execution_services.as_ref(),
                        &self.execution_plan,
                        self.stage,
                        self.builder.as_ref(),
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
        let candidate = self.execution_plan.candidates()[self.candidate_index].clone();
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
            (result, Some(failure)) => Err(PolicyResolvedAuxRuntimeError::CandidateFailed {
                stage: self.stage,
                failure,
                source: result.err(),
            }),
        }
    }

    #[cfg(test)]
    fn candidate_index(&self) -> usize {
        self.candidate_index
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

    pub(crate) fn invoke<T>(
        &mut self,
        mut operation: impl FnMut(&R) -> Result<T, E>,
    ) -> Result<T, PolicyResolvedAuxRuntimeError<E>> {
        let result = if self.output_committed {
            self.runtime.invoke_pinned(|runtime| operation(runtime))
        } else {
            self.runtime
                .invoke_replay_safe(|runtime| operation(runtime))
        };
        if result.is_ok() {
            self.output_committed = true;
        }
        result
    }

    pub(crate) const fn output_committed(&self) -> bool {
        self.output_committed
    }

    #[cfg(test)]
    fn candidate_index(&self) -> usize {
        self.runtime.candidate_index()
    }
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
    lane: ExecutionLaneKey,
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
    instance_id: Option<u64>,
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
            instance_id: None,
            lane: current_execution_lane_key(backend),
        }
    }

    pub(crate) fn for_current_session_lane<R: 'static>(
        architecture_id: &'static str,
        pack_content_id: impl Into<String>,
        representation_id: &'static str,
        instance_id: u64,
        backend: GgmlCpuGraphBackend,
    ) -> Self {
        let mut key = Self::for_current_lane::<R>(
            architecture_id,
            pack_content_id,
            representation_id,
            backend,
        );
        key.instance_id = Some(instance_id);
        key
    }

    pub(crate) fn has_content_id(&self, pack_content_id: &str) -> bool {
        self.pack_content_id == pack_content_id
    }

    pub(crate) fn has_instance_id(&self, instance_id: u64) -> bool {
        self.instance_id == Some(instance_id)
    }
}

impl AuxiliaryRuntimeCacheKey {
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
            lane: current_execution_lane_key(backend),
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    use crate::models::native_execution_services::record_current_execution_candidate_failure;

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
        let mut runtime =
            PolicyResolvedAuxRuntime::try_new(services, plan, "test-aux", builder).unwrap();

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
            track_drop: bool,
        }

        impl Drop for RuntimeDropProbe {
            fn drop(&mut self) {
                if self.track_drop {
                    self.dropped.store(true, Ordering::SeqCst);
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
        let runtime_dropped_for_builder = Arc::clone(&runtime_dropped);
        let result_dropped_for_builder = Arc::clone(&result_dropped);
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
            }
            Ok::<_, &'static str>(RuntimeDropProbe {
                provider: candidate.device.route.provider,
                dropped: Arc::clone(&runtime_dropped_for_builder),
                track_drop: candidate.device.route.provider == ExecutionProvider::Vulkan,
            })
        });
        let mut runtime =
            PolicyResolvedAuxRuntime::try_new(services, plan, "test-drop-order", builder).unwrap();

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
