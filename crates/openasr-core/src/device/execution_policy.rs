//! Product execution intent, ordered candidates, and shared planning services.
//!
//! This layer deliberately does not estimate model memory and does not change
//! model semantics. It turns one user intent plus family/backend capabilities
//! into an ordered list. Each candidate is then quoted and admitted by the
//! physical-footprint planner and [`super::execution_memory::DeviceMemoryBrokerSet`].
//! A rejected candidate never mutates the invocation window, model, or state
//! precision; those would be product decisions, not memory fallback.

use thiserror::Error;

use super::{
    execution_memory::{DeviceMemorySnapshot, MemoryObservationConfidence},
    execution_route::{
        EnumeratedComputeDevice, ExactDeviceSelector, ExecutionHardwareVendor, ExecutionProvider,
        ExecutionRouteError, ExecutionRouteRequest, ResolvedExecutionRoute, RouteDeviceKind,
        ranked_preferred_accelerated_devices, resolve_execution_route,
    },
};
use crate::ggml_runtime::{AutoGpuPolicy, GgmlBackendKind};

/// User/product intent after parsing the public request surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionIntent {
    Auto,
    CpuOnly,
    /// Accelerated execution may use a backend-supported CPU/GPU hybrid, but
    /// must not silently become pure CPU.
    AcceleratedOnly,
    /// Accelerated execution constrained by a stable provider or proven
    /// hardware-vendor fact. It never appends pure CPU or an unrelated device.
    ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint),
    /// Exact remains internal until public multi-device pinning is ready.
    Exact(ExactDeviceSelector),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratedDeviceConstraint {
    Provider(ExecutionProvider),
    HardwareVendor(ExecutionHardwareVendor),
}

impl std::fmt::Display for AcceleratedDeviceConstraint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(provider) => write!(formatter, "provider {provider}"),
            Self::HardwareVendor(vendor) => write!(formatter, "hardware vendor {vendor}"),
        }
    }
}

impl From<ExecutionRouteRequest> for ExecutionIntent {
    fn from(value: ExecutionRouteRequest) -> Self {
        match value {
            ExecutionRouteRequest::Auto => Self::Auto,
            ExecutionRouteRequest::Cpu => Self::CpuOnly,
            ExecutionRouteRequest::Accelerated => Self::AcceleratedOnly,
            ExecutionRouteRequest::Exact(selector) => Self::Exact(selector),
        }
    }
}

impl From<crate::ExecutionTarget> for ExecutionIntent {
    fn from(value: crate::ExecutionTarget) -> Self {
        Self::from(ExecutionRouteRequest::from_execution_target(value))
    }
}

/// Placement shapes implemented by a family/runtime pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPlacement {
    CpuOnly,
    FullDevice,
    /// Weights/operations may be split across CPU and the selected device.
    Hybrid,
}

/// Accelerated placement shapes supported by one provider implementation.
///
/// This is deliberately distinct from [`ExecutionCapabilities`]: two ggml
/// providers exposed by the same process need not implement the same split or
/// offload path for a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceleratedPlacementCapabilities(u8);

impl AcceleratedPlacementCapabilities {
    const FULL_DEVICE_BIT: u8 = 1 << 0;
    const HYBRID_BIT: u8 = 1 << 1;

    pub const NONE: Self = Self(0);
    pub const FULL_DEVICE: Self = Self(Self::FULL_DEVICE_BIT);
    pub const HYBRID: Self = Self(Self::HYBRID_BIT);
    pub const FULL_DEVICE_AND_HYBRID: Self = Self(Self::FULL_DEVICE_BIT | Self::HYBRID_BIT);

    pub const fn supports(self, placement: ExecutionPlacement) -> bool {
        let required = match placement {
            ExecutionPlacement::CpuOnly => return false,
            ExecutionPlacement::FullDevice => Self::FULL_DEVICE_BIT,
            ExecutionPlacement::Hybrid => Self::HYBRID_BIT,
        };
        self.0 & required != 0
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

const EXECUTION_PROVIDER_COUNT: usize = 7;

const fn provider_slot(provider: ExecutionProvider) -> usize {
    match provider {
        ExecutionProvider::Cpu => 0,
        ExecutionProvider::Metal => 1,
        ExecutionProvider::Cuda => 2,
        ExecutionProvider::Hip => 3,
        ExecutionProvider::Vulkan => 4,
        ExecutionProvider::Accelerator => 5,
        ExecutionProvider::Unknown => 6,
    }
}

/// Family/runtime support matrix indexed by provider and placement.
///
/// Providers must be declared individually. This prevents a family that, for
/// example, implements hybrid execution only on CUDA from accidentally
/// producing hybrid candidates for every GPU visible in the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    cpu: bool,
    accelerated: [AcceleratedPlacementCapabilities; EXECUTION_PROVIDER_COUNT],
}

impl ExecutionCapabilities {
    pub const fn new(cpu: bool) -> Self {
        Self {
            cpu,
            accelerated: [AcceleratedPlacementCapabilities::NONE; EXECUTION_PROVIDER_COUNT],
        }
    }

    /// Declare the placements implemented by one accelerated provider.
    /// Repeating the same provider replaces its previous row.
    pub const fn with_provider(
        mut self,
        provider: ExecutionProvider,
        placements: AcceleratedPlacementCapabilities,
    ) -> Self {
        assert!(
            !matches!(provider, ExecutionProvider::Cpu),
            "CPU support belongs in ExecutionCapabilities::new"
        );
        self.accelerated[provider_slot(provider)] = placements;
        self
    }

    pub const fn supports_cpu(self) -> bool {
        self.cpu
    }

    pub const fn accelerated_for(
        self,
        provider: ExecutionProvider,
    ) -> AcceleratedPlacementCapabilities {
        self.accelerated[provider_slot(provider)]
    }

    pub const fn supports(
        self,
        provider: ExecutionProvider,
        placement: ExecutionPlacement,
    ) -> bool {
        match placement {
            ExecutionPlacement::CpuOnly => matches!(provider, ExecutionProvider::Cpu) && self.cpu,
            ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => {
                !matches!(provider, ExecutionProvider::Cpu)
                    && self.accelerated_for(provider).supports(placement)
            }
        }
    }

    const fn has_accelerated_placement(self) -> bool {
        let mut index = 1;
        while index < EXECUTION_PROVIDER_COUNT {
            if !self.accelerated[index].is_empty() {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Immutable device facts carried from enumeration into quote/admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionDeviceSnapshot {
    pub route: ResolvedExecutionRoute,
    pub ggml_kind: GgmlBackendKind,
    /// Enumeration-time diagnostics only. Physical admission refreshes stats
    /// through the backend memory ABI and takes its domain UUID/kind from the
    /// resulting quote; a route can consume both host and device domains, so
    /// policy must not guess one singular budget key here.
    pub memory: Option<DeviceMemorySnapshot>,
    pub buffer_alignment: Option<usize>,
}

/// One independently-attemptable execution candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCandidate {
    pub device: ExecutionDeviceSnapshot,
    pub placement: ExecutionPlacement,
}

/// Typed reason why one otherwise-valid execution candidate could not be
/// completed. Only failures in this enum permit the policy layer to try the
/// next candidate; model/input/format/cancel/decode errors are never recorded
/// here and therefore always fail closed on the current attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionCandidateFailureKind {
    /// The quoted or actual physical footprint could not be admitted.
    CapacityUnavailable,
    /// The selected provider/device could not be initialized or disappeared.
    DeviceUnavailable,
    /// The device was initialized but became unusable while executing.
    DeviceLost,
    /// A graph completed on a backend other than the selected provider.
    /// This is a candidate-local routing failure: Auto may try another row,
    /// while an exact or accelerated-only plan remains fail-closed because it
    /// contains no pure-CPU candidate.
    PlacementViolation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionCandidateFailure {
    pub kind: ExecutionCandidateFailureKind,
    pub operation: &'static str,
    /// Diagnostic text only. Candidate selection keys exclusively on `kind`;
    /// no caller parses this string to decide whether fallback is allowed.
    pub detail: String,
}

impl ExecutionCandidateFailure {
    pub fn capacity(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind: ExecutionCandidateFailureKind::CapacityUnavailable,
            operation,
            detail: detail.into(),
        }
    }

    pub fn device_unavailable(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind: ExecutionCandidateFailureKind::DeviceUnavailable,
            operation,
            detail: detail.into(),
        }
    }

    pub fn device_lost(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind: ExecutionCandidateFailureKind::DeviceLost,
            operation,
            detail: detail.into(),
        }
    }

    pub fn placement(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind: ExecutionCandidateFailureKind::PlacementViolation,
            operation,
            detail: detail.into(),
        }
    }
}

/// Ordered, semantics-preserving candidates. Allocation is transactional:
/// failure releases the candidate's reservations before trying the next row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    intent: ExecutionIntent,
    candidates: Vec<ExecutionCandidate>,
}

impl ExecutionPlan {
    pub fn intent(&self) -> &ExecutionIntent {
        &self.intent
    }

    pub fn candidates(&self) -> &[ExecutionCandidate] {
        &self.candidates
    }

    #[cfg(test)]
    pub(crate) fn for_test(intent: ExecutionIntent, candidates: Vec<ExecutionCandidate>) -> Self {
        assert!(
            !candidates.is_empty(),
            "a test execution plan needs a candidate"
        );
        Self { intent, candidates }
    }
}

pub trait ExecutionPolicyResolver: Send + Sync {
    fn resolve(
        &self,
        intent: ExecutionIntent,
        family_auto_gpu_policy: AutoGpuPolicy,
        capabilities: ExecutionCapabilities,
        inventory: &[EnumeratedComputeDevice],
    ) -> Result<ExecutionPlan, ExecutionPolicyError>;
}

#[derive(Debug, Default)]
pub struct DefaultExecutionPolicyResolver;

impl ExecutionPolicyResolver for DefaultExecutionPolicyResolver {
    fn resolve(
        &self,
        intent: ExecutionIntent,
        family_auto_gpu_policy: AutoGpuPolicy,
        capabilities: ExecutionCapabilities,
        inventory: &[EnumeratedComputeDevice],
    ) -> Result<ExecutionPlan, ExecutionPolicyError> {
        let mut candidates = Vec::new();
        match &intent {
            ExecutionIntent::CpuOnly => {
                require_capability(capabilities.supports_cpu(), ExecutionPlacement::CpuOnly)?;
                candidates.push(cpu_candidate(inventory)?);
            }
            ExecutionIntent::Auto => {
                let accelerated: Vec<&EnumeratedComputeDevice> =
                    ranked_preferred_accelerated_devices(inventory)
                        .into_iter()
                        .filter(|device| {
                            auto_allows_device(family_auto_gpu_policy, device.provider)
                        })
                        .collect();
                append_accelerated_candidates(&mut candidates, &accelerated, capabilities)?;
                if capabilities.supports_cpu() {
                    candidates.push(cpu_candidate(inventory)?);
                }
            }
            ExecutionIntent::AcceleratedOnly => {
                let accelerated = ranked_preferred_accelerated_devices(inventory);
                if accelerated.is_empty() {
                    return Err(ExecutionPolicyError::Route(
                        ExecutionRouteError::AcceleratedUnavailable,
                    ));
                }
                if !capabilities.has_accelerated_placement() {
                    return Err(ExecutionPolicyError::NoAcceleratedPlacement);
                }
                append_accelerated_candidates(&mut candidates, &accelerated, capabilities)?;
            }
            ExecutionIntent::ConstrainedAcceleratedOnly(constraint) => {
                let accelerated: Vec<&EnumeratedComputeDevice> =
                    ranked_preferred_accelerated_devices(inventory)
                        .into_iter()
                        .filter(|device| constraint.matches(device))
                        .collect();
                if accelerated.is_empty() {
                    return Err(ExecutionPolicyError::ConstrainedAcceleratedUnavailable {
                        constraint: *constraint,
                    });
                }
                append_accelerated_candidates(&mut candidates, &accelerated, capabilities)?;
                if candidates.is_empty() {
                    return Err(ExecutionPolicyError::NoAcceleratedPlacementForConstraint {
                        constraint: *constraint,
                    });
                }
            }
            ExecutionIntent::Exact(selector) => {
                let route = resolve_execution_route(
                    &ExecutionRouteRequest::Exact(selector.clone()),
                    inventory,
                )?;
                let device = inventory
                    .iter()
                    .find(|device| {
                        device.registry_ordinal == route.registry_ordinal
                            && device.stable_id == route.stable_id
                            && device.provider == route.provider
                    })
                    .ok_or_else(|| ExecutionPolicyError::InventoryDrift {
                        route: route.isolation_key(),
                    })?;
                append_accelerated_candidates(&mut candidates, &[device], capabilities)?;
                if candidates.is_empty() {
                    return Err(ExecutionPolicyError::NoAcceleratedPlacementForProvider {
                        provider: device.provider,
                    });
                }
            }
        }
        if candidates.is_empty() {
            return Err(ExecutionPolicyError::NoSupportedCandidate { intent });
        }
        Ok(ExecutionPlan { intent, candidates })
    }
}

impl AcceleratedDeviceConstraint {
    fn matches(self, device: &EnumeratedComputeDevice) -> bool {
        match self {
            Self::Provider(provider) => device.provider == provider,
            Self::HardwareVendor(vendor) => device.hardware_vendor == Some(vendor),
        }
    }
}

fn append_accelerated_candidates(
    output: &mut Vec<ExecutionCandidate>,
    devices: &[&EnumeratedComputeDevice],
    capabilities: ExecutionCapabilities,
) -> Result<(), ExecutionPolicyError> {
    // Product order is placement-major: every full-device option is tried
    // before any hybrid option, then Auto may append pure CPU.
    for placement in [ExecutionPlacement::FullDevice, ExecutionPlacement::Hybrid] {
        output.extend(
            devices
                .iter()
                .filter(|device| capabilities.supports(device.provider, placement))
                .map(|device| candidate(device, placement))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(())
}

fn cpu_candidate(
    inventory: &[EnumeratedComputeDevice],
) -> Result<ExecutionCandidate, ExecutionPolicyError> {
    let route = resolve_execution_route(&ExecutionRouteRequest::Cpu, inventory)?;
    let enumerated = inventory.iter().find(|device| {
        device.kind == RouteDeviceKind::Cpu
            && device.stable_id == route.stable_id
            && device.registry_ordinal == route.registry_ordinal
    });
    match enumerated {
        Some(device) => candidate(device, ExecutionPlacement::CpuOnly),
        None => Ok(ExecutionCandidate {
            device: ExecutionDeviceSnapshot {
                route,
                ggml_kind: GgmlBackendKind::Cpu,
                memory: None,
                buffer_alignment: None,
            },
            placement: ExecutionPlacement::CpuOnly,
        }),
    }
}

fn candidate(
    device: &EnumeratedComputeDevice,
    placement: ExecutionPlacement,
) -> Result<ExecutionCandidate, ExecutionPolicyError> {
    let confidence = match device.provider {
        ExecutionProvider::Metal => MemoryObservationConfidence::WorkingSetBudget,
        ExecutionProvider::Cpu => MemoryObservationConfidence::Heuristic,
        _ => MemoryObservationConfidence::DeviceSnapshot,
    };
    let memory = device.memory.map(|memory| DeviceMemorySnapshot {
        free_bytes: u64::try_from(memory.free_bytes).unwrap_or(u64::MAX),
        total_bytes: u64::try_from(memory.total_bytes).unwrap_or(u64::MAX),
        confidence,
    });
    Ok(ExecutionCandidate {
        device: ExecutionDeviceSnapshot {
            route: device.to_resolved_route(),
            ggml_kind: device.ggml_kind,
            memory,
            buffer_alignment: device.buffer_alignment,
        },
        placement,
    })
}

const fn auto_allows_device(policy: AutoGpuPolicy, provider: ExecutionProvider) -> bool {
    match policy {
        AutoGpuPolicy::AllBackends => true,
        AutoGpuPolicy::ExceptMetal => !matches!(provider, ExecutionProvider::Metal),
        AutoGpuPolicy::Never => false,
    }
}

fn require_capability(
    available: bool,
    placement: ExecutionPlacement,
) -> Result<(), ExecutionPolicyError> {
    if available {
        Ok(())
    } else {
        Err(ExecutionPolicyError::UnsupportedPlacement { placement })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExecutionPolicyError {
    #[error(transparent)]
    Route(#[from] ExecutionRouteError),
    #[error(transparent)]
    Memory(#[from] super::execution_memory::MemoryPlanningError),
    #[error("execution placement {placement:?} is not supported by this model runtime")]
    UnsupportedPlacement { placement: ExecutionPlacement },
    #[error("model runtime supports no accelerated placement")]
    NoAcceleratedPlacement,
    #[error("model runtime supports no accelerated placement for provider {provider}")]
    NoAcceleratedPlacementForProvider { provider: ExecutionProvider },
    #[error("no accelerated execution device matches {constraint}")]
    ConstrainedAcceleratedUnavailable {
        constraint: AcceleratedDeviceConstraint,
    },
    #[error("model runtime supports no accelerated placement for {constraint}")]
    NoAcceleratedPlacementForConstraint {
        constraint: AcceleratedDeviceConstraint,
    },
    #[error("no supported execution candidate exists for intent {intent:?}")]
    NoSupportedCandidate { intent: ExecutionIntent },
    #[error("execution inventory changed while resolving route {route}")]
    InventoryDrift { route: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::execution_route::{DeviceAddressability, PhysicalResourceKey},
        ggml_runtime::GgmlDeviceMemory,
    };

    fn device(
        ordinal: usize,
        provider: ExecutionProvider,
        kind: GgmlBackendKind,
        stable_id: &str,
        physical_id: Option<&str>,
    ) -> EnumeratedComputeDevice {
        let addressability = match physical_id.and_then(PhysicalResourceKey::new) {
            Some(physical_key) => DeviceAddressability::ExactlyAddressable { physical_key },
            None => DeviceAddressability::NotExactlyAddressable {
                reason: "test device has no physical id",
            },
        };
        EnumeratedComputeDevice {
            provider,
            stable_id: stable_id.to_string(),
            description: stable_id.to_string(),
            registry_ordinal: ordinal,
            kind: if kind == GgmlBackendKind::Cpu {
                RouteDeviceKind::Cpu
            } else {
                RouteDeviceKind::Accelerated
            },
            ggml_kind: kind,
            memory: Some(GgmlDeviceMemory {
                free_bytes: 6 * 1024 * 1024 * 1024,
                total_bytes: 8 * 1024 * 1024 * 1024,
            }),
            buffer_alignment: Some(256),
            addressability,
            device_id: physical_id.map(str::to_string),
            hardware_vendor: match provider {
                ExecutionProvider::Cuda => Some(ExecutionHardwareVendor::Nvidia),
                ExecutionProvider::Hip => Some(ExecutionHardwareVendor::Amd),
                _ => None,
            },
        }
    }

    fn inventory() -> Vec<EnumeratedComputeDevice> {
        vec![
            device(0, ExecutionProvider::Cpu, GgmlBackendKind::Cpu, "CPU", None),
            device(
                1,
                ExecutionProvider::Vulkan,
                GgmlBackendKind::IntegratedGpu,
                "Vulkan0",
                Some("0000:00:02.0"),
            ),
            device(
                2,
                ExecutionProvider::Vulkan,
                GgmlBackendKind::Gpu,
                "Vulkan1",
                Some("0000:01:00.0"),
            ),
        ]
    }

    fn vulkan_capabilities(
        cpu: bool,
        accelerated: AcceleratedPlacementCapabilities,
    ) -> ExecutionCapabilities {
        ExecutionCapabilities::new(cpu).with_provider(ExecutionProvider::Vulkan, accelerated)
    }

    fn mixed_provider_inventory() -> Vec<EnumeratedComputeDevice> {
        vec![
            device(0, ExecutionProvider::Cpu, GgmlBackendKind::Cpu, "CPU", None),
            device(
                1,
                ExecutionProvider::Vulkan,
                GgmlBackendKind::IntegratedGpu,
                "Vulkan0",
                Some("0000:00:02.0"),
            ),
            device(
                2,
                ExecutionProvider::Cuda,
                GgmlBackendKind::Gpu,
                "CUDA0",
                Some("0000:01:00.0"),
            ),
            device(
                3,
                ExecutionProvider::Hip,
                GgmlBackendKind::Gpu,
                "HIP0",
                Some("0000:02:00.0"),
            ),
        ]
    }

    #[test]
    fn auto_orders_all_full_device_then_hybrid_then_cpu() {
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Auto,
                AutoGpuPolicy::AllBackends,
                vulkan_capabilities(
                    true,
                    AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
                ),
                &inventory(),
            )
            .unwrap();
        let actual: Vec<(&str, ExecutionPlacement)> = plan
            .candidates()
            .iter()
            .map(|candidate| {
                (
                    candidate.device.route.stable_id.as_str(),
                    candidate.placement,
                )
            })
            .collect();
        assert_eq!(
            actual,
            vec![
                ("Vulkan1", ExecutionPlacement::FullDevice),
                ("Vulkan0", ExecutionPlacement::FullDevice),
                ("Vulkan1", ExecutionPlacement::Hybrid),
                ("Vulkan0", ExecutionPlacement::Hybrid),
                ("CPU", ExecutionPlacement::CpuOnly),
            ]
        );
    }

    #[test]
    fn provider_matrix_filters_each_accelerated_placement_independently() {
        let capabilities = ExecutionCapabilities::new(true)
            .with_provider(
                ExecutionProvider::Cuda,
                AcceleratedPlacementCapabilities::FULL_DEVICE,
            )
            .with_provider(
                ExecutionProvider::Vulkan,
                AcceleratedPlacementCapabilities::HYBRID,
            );
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Auto,
                AutoGpuPolicy::AllBackends,
                capabilities,
                &mixed_provider_inventory(),
            )
            .unwrap();
        let actual: Vec<(&str, ExecutionPlacement)> = plan
            .candidates()
            .iter()
            .map(|candidate| {
                (
                    candidate.device.route.stable_id.as_str(),
                    candidate.placement,
                )
            })
            .collect();
        assert_eq!(
            actual,
            vec![
                ("CUDA0", ExecutionPlacement::FullDevice),
                ("Vulkan0", ExecutionPlacement::Hybrid),
                ("CPU", ExecutionPlacement::CpuOnly),
            ]
        );
    }

    #[test]
    fn accelerated_never_appends_pure_cpu() {
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::AcceleratedOnly,
                AutoGpuPolicy::Never,
                vulkan_capabilities(
                    true,
                    AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
                ),
                &inventory(),
            )
            .unwrap();
        assert!(
            plan.candidates()
                .iter()
                .all(|candidate| candidate.placement != ExecutionPlacement::CpuOnly)
        );
    }

    #[test]
    fn auto_family_gate_never_changes_explicit_accelerated() {
        let auto = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Auto,
                AutoGpuPolicy::Never,
                vulkan_capabilities(
                    true,
                    AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
                ),
                &inventory(),
            )
            .unwrap();
        assert_eq!(auto.candidates().len(), 1);
        assert_eq!(auto.candidates()[0].placement, ExecutionPlacement::CpuOnly);

        let accelerated = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::AcceleratedOnly,
                AutoGpuPolicy::Never,
                vulkan_capabilities(
                    true,
                    AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
                ),
                &inventory(),
            )
            .unwrap();
        assert!(!accelerated.candidates().is_empty());
    }

    #[test]
    fn policy_does_not_guess_physical_memory_domains_from_routes() {
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Auto,
                AutoGpuPolicy::AllBackends,
                vulkan_capabilities(true, AcceleratedPlacementCapabilities::FULL_DEVICE),
                &inventory()[..2],
            )
            .unwrap();
        assert_eq!(
            plan.candidates()[0].device.ggml_kind,
            GgmlBackendKind::IntegratedGpu
        );
        assert_eq!(plan.candidates()[1].device.ggml_kind, GgmlBackendKind::Cpu);
    }

    #[test]
    fn exact_plan_never_adds_a_different_card_or_cpu() {
        let selector =
            ExactDeviceSelector::PhysicalKey(PhysicalResourceKey::new("0000:00:02.0").unwrap());
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Exact(selector),
                AutoGpuPolicy::AllBackends,
                vulkan_capabilities(
                    true,
                    AcceleratedPlacementCapabilities::FULL_DEVICE_AND_HYBRID,
                ),
                &inventory(),
            )
            .unwrap();
        assert_eq!(plan.candidates().len(), 2);
        assert!(
            plan.candidates()
                .iter()
                .all(|candidate| candidate.device.route.stable_id == "Vulkan0")
        );
    }

    #[test]
    fn exact_accelerated_rejects_an_unsupported_provider_without_fallback() {
        let selector =
            ExactDeviceSelector::PhysicalKey(PhysicalResourceKey::new("0000:01:00.0").unwrap());
        let error = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Exact(selector),
                AutoGpuPolicy::AllBackends,
                ExecutionCapabilities::new(true).with_provider(
                    ExecutionProvider::Vulkan,
                    AcceleratedPlacementCapabilities::FULL_DEVICE,
                ),
                &mixed_provider_inventory(),
            )
            .expect_err("Exact CUDA must not switch to Vulkan or CPU");
        assert_eq!(
            error,
            ExecutionPolicyError::NoAcceleratedPlacementForProvider {
                provider: ExecutionProvider::Cuda,
            }
        );
    }

    #[test]
    fn exact_cpu_is_not_a_cpu_fallback_surface() {
        let selector = ExactDeviceSelector::StableId {
            provider: Some(ExecutionProvider::Cpu),
            stable_id: "CPU".to_string(),
        };
        let error = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::Exact(selector),
                AutoGpuPolicy::AllBackends,
                ExecutionCapabilities::new(true),
                &inventory(),
            )
            .expect_err("CPU must be selected through CpuOnly, never Exact");
        assert!(matches!(
            error,
            ExecutionPolicyError::Route(ExecutionRouteError::NotAddressable { .. })
        ));
    }

    #[test]
    fn constrained_provider_never_selects_another_provider_or_cpu() {
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
                    ExecutionProvider::Cuda,
                )),
                AutoGpuPolicy::AllBackends,
                ExecutionCapabilities::new(true)
                    .with_provider(
                        ExecutionProvider::Cuda,
                        AcceleratedPlacementCapabilities::FULL_DEVICE,
                    )
                    .with_provider(
                        ExecutionProvider::Hip,
                        AcceleratedPlacementCapabilities::FULL_DEVICE,
                    ),
                &mixed_provider_inventory(),
            )
            .unwrap();
        assert_eq!(plan.candidates().len(), 1);
        assert_eq!(
            plan.candidates()[0].device.route.provider,
            ExecutionProvider::Cuda
        );
    }

    #[test]
    fn constrained_vendor_uses_proven_vendor_fact_and_fails_closed_when_missing() {
        let capabilities = ExecutionCapabilities::new(true)
            .with_provider(
                ExecutionProvider::Hip,
                AcceleratedPlacementCapabilities::FULL_DEVICE,
            )
            .with_provider(
                ExecutionProvider::Vulkan,
                AcceleratedPlacementCapabilities::FULL_DEVICE,
            );
        let plan = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::ConstrainedAcceleratedOnly(
                    AcceleratedDeviceConstraint::HardwareVendor(ExecutionHardwareVendor::Amd),
                ),
                AutoGpuPolicy::AllBackends,
                capabilities,
                &mixed_provider_inventory(),
            )
            .unwrap();
        assert_eq!(plan.candidates().len(), 1);
        assert_eq!(
            plan.candidates()[0].device.route.provider,
            ExecutionProvider::Hip
        );

        let error = DefaultExecutionPolicyResolver
            .resolve(
                ExecutionIntent::ConstrainedAcceleratedOnly(
                    AcceleratedDeviceConstraint::HardwareVendor(ExecutionHardwareVendor::Intel),
                ),
                AutoGpuPolicy::AllBackends,
                capabilities,
                &mixed_provider_inventory(),
            )
            .expect_err("an unproven Intel device must not be guessed from another GPU");
        assert!(matches!(
            error,
            ExecutionPolicyError::ConstrainedAcceleratedUnavailable { .. }
        ));
    }
}
