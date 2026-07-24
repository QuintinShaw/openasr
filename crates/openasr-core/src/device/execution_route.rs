//! Request-level execution route foundation.
//!
//! Public request surfaces stay coarse (`ExecutionTarget::{Auto,Cpu,Accelerated}`).
//! This module adds the internal exact-route vocabulary that cache, worker, and
//! admission isolation must share before any product UI exposes GPU0/GPU1 picks.
//!
//! Correct abstraction:
//! - [`ResolvedExecutionRoute`] = logical `(provider, stable_id)` plus optional
//!   [`PhysicalResourceKey`] (PCI BDF when ggml supplies `device_id`)
//! - Exact resolution is typed fail-closed: no silent card swap, no CPU fallback
//! - Metal devices are enumerable but [`DeviceAddressability::NotExactlyAddressable`]
//!   because ggml Metal still initializes via `MTLCreateSystemDefaultDevice` only

use std::fmt;

use thiserror::Error;

use crate::ggml_runtime::{GgmlBackendDevice, GgmlBackendKind};

/// Backend provider family for route identity. Distinct from the public coarse
/// [`crate::ExecutionTarget`] surface (`auto` / `cpu` / `accelerated`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionProvider {
    Cpu,
    Metal,
    Cuda,
    Hip,
    Vulkan,
    Accelerator,
    Unknown,
}

impl ExecutionProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
            Self::Vulkan => "vulkan",
            Self::Accelerator => "accelerator",
            Self::Unknown => "unknown",
        }
    }

    /// Infer provider from a ggml backend/device name.
    pub fn from_backend_name(name: &str) -> Self {
        let lower = name.trim().to_ascii_lowercase();
        if lower.is_empty() {
            return Self::Unknown;
        }
        if lower == "cpu" || lower.starts_with("cpu") {
            return Self::Cpu;
        }
        if lower.contains("metal") || lower.starts_with("mtl") {
            return Self::Metal;
        }
        if lower.starts_with("cuda") {
            return Self::Cuda;
        }
        if lower.starts_with("hip") || lower.starts_with("rocm") {
            return Self::Hip;
        }
        if lower.starts_with("vulkan") || lower.starts_with("vk") {
            return Self::Vulkan;
        }
        if lower.contains("blas") || lower.contains("accel") {
            return Self::Accelerator;
        }
        Self::Unknown
    }

    pub const fn supports_exact_selection(self) -> bool {
        matches!(self, Self::Cuda | Self::Hip | Self::Vulkan)
    }
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable physical identity when the backend exposes one.
///
/// For PCI devices ggml documents `device_id` as lower-case
/// `domain:bus:device.function` (e.g. `0000:c1:00.0`). CUDA/HIP always aim to
/// provide this; Vulkan does when the instance exposes the PCI bus id; Metal
/// never does.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalResourceKey(String);

impl PhysicalResourceKey {
    pub fn new(raw: impl Into<String>) -> Option<Self> {
        let value = normalize_physical_key(&raw.into())?;
        Some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PhysicalResourceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether a visible device can be the target of an Exact request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceAddressability {
    ExactlyAddressable {
        physical_key: PhysicalResourceKey,
    },
    /// Device is usable via Auto/Accelerated, but Exact pin is refused.
    NotExactlyAddressable {
        reason: &'static str,
    },
}

impl DeviceAddressability {
    pub const fn is_exactly_addressable(&self) -> bool {
        matches!(self, Self::ExactlyAddressable { .. })
    }

    pub fn physical_key(&self) -> Option<&PhysicalResourceKey> {
        match self {
            Self::ExactlyAddressable { physical_key } => Some(physical_key),
            Self::NotExactlyAddressable { .. } => None,
        }
    }
}

/// Coarse class used by route ranking (CPU vs any accelerated device).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteDeviceKind {
    Cpu,
    Accelerated,
}

impl RouteDeviceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Accelerated => "accelerated",
        }
    }
}

/// Logical route identity used for cache / worker / admission isolation.
///
/// `(provider, stable_id)` is always present. [`PhysicalResourceKey`] is layered
/// on when ggml supplies a PCI-style `device_id`. Registry ordinal is retained
/// only as a fail-closed disambiguator when two same-provider devices would
/// otherwise collapse to one isolation key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResolvedExecutionRoute {
    pub provider: ExecutionProvider,
    /// Provider-local ggml device name (`CUDA0`, `Vulkan1`, `Metal`, `CPU`, ...).
    pub stable_id: String,
    pub registry_ordinal: usize,
    pub kind: RouteDeviceKind,
    pub addressability: DeviceAddressability,
}

impl ResolvedExecutionRoute {
    /// Isolation key shared by backend cache, streaming worker keys, and model
    /// admission slots. Exact and preferred-accelerated routes that resolve to
    /// the same device must produce the same key.
    pub fn isolation_key(&self) -> String {
        match self.addressability.physical_key() {
            Some(physical) => format!(
                "{}/{}/pci:{}",
                self.provider.as_str(),
                self.stable_id,
                physical.as_str()
            ),
            None => format!("{}/{}", self.provider.as_str(), self.stable_id),
        }
    }

    /// Backend-cache key: provider + stable_id, plus physical key when present.
    /// Ordinal is intentionally excluded so a stable device keeps its cache
    /// entry across harmless re-enumeration order shifts when identity is known.
    pub fn cache_key(&self) -> ExecutionRouteCacheKey {
        ExecutionRouteCacheKey {
            provider: self.provider,
            stable_id: self.stable_id.clone(),
            physical_key: self
                .addressability
                .physical_key()
                .map(|key| key.as_str().to_string()),
        }
    }

    pub fn cpu() -> Self {
        Self {
            provider: ExecutionProvider::Cpu,
            stable_id: "CPU".to_string(),
            registry_ordinal: 0,
            kind: RouteDeviceKind::Cpu,
            addressability: DeviceAddressability::NotExactlyAddressable {
                reason: "CPU is selected by the coarse cpu target, not by Exact device pin",
            },
        }
    }
}

/// Hash key for the thread-local ggml backend cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutionRouteCacheKey {
    pub provider: ExecutionProvider,
    pub stable_id: String,
    pub physical_key: Option<String>,
}

impl fmt::Display for ExecutionRouteCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.physical_key {
            Some(physical) => write!(
                f,
                "{}/{}/pci:{}",
                self.provider.as_str(),
                self.stable_id,
                physical
            ),
            None => write!(f, "{}/{}", self.provider.as_str(), self.stable_id),
        }
    }
}

/// Internal request intent. The public HTTP/CLI surface still only accepts
/// `auto` / `cpu` / `accelerated`; Exact exists so the runtime can grow a pin
/// path without inventing a second abstraction later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionRouteRequest {
    Auto,
    Cpu,
    /// Preferred accelerated device (discrete-over-integrated). Not Exact.
    Accelerated,
    /// Pin one device. Fail-closed on miss / not-addressable / init failure.
    Exact(ExactDeviceSelector),
}

impl ExecutionRouteRequest {
    pub fn from_execution_target(target: crate::ExecutionTarget) -> Self {
        match target {
            crate::ExecutionTarget::Auto => Self::Auto,
            crate::ExecutionTarget::Cpu => Self::Cpu,
            crate::ExecutionTarget::Accelerated => Self::Accelerated,
        }
    }
}

/// Exact device selector. Prefer physical PCI identity when the caller has it;
/// fall back to provider-scoped stable ggml name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactDeviceSelector {
    PhysicalKey(PhysicalResourceKey),
    StableId {
        provider: Option<ExecutionProvider>,
        stable_id: String,
    },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ExecutionRouteError {
    #[error("requested execution device was not found: {detail}")]
    DeviceNotFound { detail: String },
    #[error("requested execution device is not exactly addressable: {detail}")]
    NotAddressable { detail: String },
    #[error("requested execution device failed to initialize: {detail}")]
    InitFailed { detail: String },
    #[error("no accelerated execution device is available")]
    AcceleratedUnavailable,
}

impl ExecutionRouteError {
    pub fn device_not_found(detail: impl Into<String>) -> Self {
        Self::DeviceNotFound {
            detail: detail.into(),
        }
    }

    pub fn not_addressable(detail: impl Into<String>) -> Self {
        Self::NotAddressable {
            detail: detail.into(),
        }
    }

    pub fn init_failed(detail: impl Into<String>) -> Self {
        Self::InitFailed {
            detail: detail.into(),
        }
    }
}

/// One inventory row produced from ggml enumeration (or a fake test registry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumeratedComputeDevice {
    pub provider: ExecutionProvider,
    pub stable_id: String,
    pub description: String,
    pub registry_ordinal: usize,
    pub kind: RouteDeviceKind,
    pub ggml_kind: GgmlBackendKind,
    pub addressability: DeviceAddressability,
    /// Raw ggml `device_id` string when present (pre-normalization).
    pub device_id: Option<String>,
}

impl EnumeratedComputeDevice {
    pub fn to_resolved_route(&self) -> ResolvedExecutionRoute {
        ResolvedExecutionRoute {
            provider: self.provider,
            stable_id: self.stable_id.clone(),
            registry_ordinal: self.registry_ordinal,
            kind: self.kind,
            addressability: self.addressability.clone(),
        }
    }
}

/// Build route inventory from live (or test) ggml device rows.
pub fn enumerate_compute_devices_from_ggml(
    devices: &[GgmlBackendDevice],
) -> Vec<EnumeratedComputeDevice> {
    devices
        .iter()
        .enumerate()
        .map(|(registry_ordinal, device)| enumerated_from_ggml_device(registry_ordinal, device))
        .collect()
}

fn enumerated_from_ggml_device(
    registry_ordinal: usize,
    device: &GgmlBackendDevice,
) -> EnumeratedComputeDevice {
    let provider = ExecutionProvider::from_backend_name(&device.name);
    let kind = if device.kind == GgmlBackendKind::Cpu || provider == ExecutionProvider::Cpu {
        RouteDeviceKind::Cpu
    } else {
        RouteDeviceKind::Accelerated
    };
    let addressability = addressability_for_device(provider, device.device_id.as_deref());
    EnumeratedComputeDevice {
        provider,
        stable_id: device.name.clone(),
        description: device.description.clone(),
        registry_ordinal,
        kind,
        ggml_kind: device.kind,
        addressability,
        device_id: device.device_id.clone(),
    }
}

fn addressability_for_device(
    provider: ExecutionProvider,
    raw_device_id: Option<&str>,
) -> DeviceAddressability {
    match provider {
        ExecutionProvider::Metal => DeviceAddressability::NotExactlyAddressable {
            reason: "Metal initializes via MTLCreateSystemDefaultDevice only; \
                     exact multi-device selection is not available",
        },
        ExecutionProvider::Cpu => DeviceAddressability::NotExactlyAddressable {
            reason: "CPU is selected by the coarse cpu target, not by Exact device pin",
        },
        ExecutionProvider::Accelerator | ExecutionProvider::Unknown => {
            DeviceAddressability::NotExactlyAddressable {
                reason: "provider does not expose a stable Exact device identity",
            }
        }
        ExecutionProvider::Cuda | ExecutionProvider::Hip | ExecutionProvider::Vulkan => {
            match raw_device_id.and_then(PhysicalResourceKey::new) {
                Some(physical_key) => DeviceAddressability::ExactlyAddressable { physical_key },
                // Vulkan (and rare CUDA/HIP builds) may omit PCI ids. Stable ggml
                // names still isolate cache/worker slots; Exact by stable_id is
                // allowed, Exact by physical key is not.
                None => DeviceAddressability::NotExactlyAddressable {
                    reason: "backend did not report a PCI device_id; Exact by \
                             physical key is unavailable (stable_id Exact may still work)",
                },
            }
        }
    }
}

/// Resolve a request against an inventory. Exact never falls back to another
/// card or to CPU.
pub fn resolve_execution_route(
    request: &ExecutionRouteRequest,
    inventory: &[EnumeratedComputeDevice],
) -> Result<ResolvedExecutionRoute, ExecutionRouteError> {
    match request {
        ExecutionRouteRequest::Cpu => Ok(resolve_cpu_route(inventory)),
        ExecutionRouteRequest::Auto => resolve_auto_route(inventory),
        ExecutionRouteRequest::Accelerated => resolve_preferred_accelerated_route(inventory),
        ExecutionRouteRequest::Exact(selector) => resolve_exact_route(selector, inventory),
    }
}

fn resolve_cpu_route(inventory: &[EnumeratedComputeDevice]) -> ResolvedExecutionRoute {
    inventory
        .iter()
        .find(|device| device.kind == RouteDeviceKind::Cpu)
        .map(EnumeratedComputeDevice::to_resolved_route)
        .unwrap_or_else(ResolvedExecutionRoute::cpu)
}

fn resolve_auto_route(
    inventory: &[EnumeratedComputeDevice],
) -> Result<ResolvedExecutionRoute, ExecutionRouteError> {
    match resolve_preferred_accelerated_route(inventory) {
        Ok(route) => Ok(route),
        Err(ExecutionRouteError::AcceleratedUnavailable) => Ok(resolve_cpu_route(inventory)),
        Err(other) => Err(other),
    }
}

fn resolve_preferred_accelerated_route(
    inventory: &[EnumeratedComputeDevice],
) -> Result<ResolvedExecutionRoute, ExecutionRouteError> {
    inventory
        .iter()
        .filter(|device| device.kind == RouteDeviceKind::Accelerated)
        .min_by_key(|device| {
            (
                crate::ggml_runtime::accelerated_device_rank(device.ggml_kind),
                device.registry_ordinal,
            )
        })
        .map(EnumeratedComputeDevice::to_resolved_route)
        .ok_or(ExecutionRouteError::AcceleratedUnavailable)
}

fn resolve_exact_route(
    selector: &ExactDeviceSelector,
    inventory: &[EnumeratedComputeDevice],
) -> Result<ResolvedExecutionRoute, ExecutionRouteError> {
    let matches: Vec<&EnumeratedComputeDevice> = match selector {
        ExactDeviceSelector::PhysicalKey(wanted) => inventory
            .iter()
            .filter(|device| {
                device
                    .addressability
                    .physical_key()
                    .is_some_and(|key| key == wanted)
            })
            .collect(),
        ExactDeviceSelector::StableId {
            provider,
            stable_id,
        } => inventory
            .iter()
            .filter(|device| {
                provider.is_none_or(|wanted| device.provider == wanted)
                    && device.stable_id == *stable_id
            })
            .collect(),
    };

    match matches.as_slice() {
        [] => Err(ExecutionRouteError::device_not_found(format!(
            "selector={selector:?}; inventory_stable_ids=[{}]",
            inventory
                .iter()
                .map(|device| device.stable_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        [device] => {
            // Metal (and other not-exactly-addressable providers) may match by
            // stable_id in inventory, but Exact must still fail closed.
            if device.provider == ExecutionProvider::Metal
                || (matches!(selector, ExactDeviceSelector::PhysicalKey(_))
                    && !device.addressability.is_exactly_addressable())
            {
                return Err(ExecutionRouteError::not_addressable(format!(
                    "provider={} stable_id={} reason={}",
                    device.provider.as_str(),
                    device.stable_id,
                    match &device.addressability {
                        DeviceAddressability::NotExactlyAddressable { reason } => *reason,
                        DeviceAddressability::ExactlyAddressable { .. } => {
                            "device is not exactly addressable"
                        }
                    }
                )));
            }
            // Stable-id Exact on CUDA/HIP/Vulkan is allowed even when PCI id is
            // missing: the stable ggml name is still a concrete device pin and
            // must not silently retarget another card.
            if matches!(selector, ExactDeviceSelector::StableId { .. })
                && !device.provider.supports_exact_selection()
                && device.provider != ExecutionProvider::Cpu
            {
                return Err(ExecutionRouteError::not_addressable(format!(
                    "provider={} does not support Exact selection (stable_id={})",
                    device.provider.as_str(),
                    device.stable_id
                )));
            }
            Ok(device.to_resolved_route())
        }
        many => Err(ExecutionRouteError::device_not_found(format!(
            "selector={selector:?} matched {} devices (ordinals {}); Exact requires a unique target",
            many.len(),
            many.iter()
                .map(|device| device.registry_ordinal.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ))),
    }
}

/// Admission / capacity slot identity: model pack identity plus resolved route.
pub fn admission_identity_for_route(
    model_identity: &str,
    route: Option<&ResolvedExecutionRoute>,
) -> String {
    match route {
        Some(route) => format!("{model_identity}|route={}", route.isolation_key()),
        None => model_identity.to_string(),
    }
}

/// Worker-key route component. Coarse targets keep their public spelling when no
/// resolved route is available; once a route is resolved (preferred accelerated
/// or Exact), isolation uses the route key so two GPUs never share a worker.
pub fn worker_route_isolation_key(
    coarse_target: &str,
    route: Option<&ResolvedExecutionRoute>,
) -> String {
    match route {
        Some(route) => route.isolation_key(),
        None => coarse_target.to_string(),
    }
}

fn normalize_physical_key(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgmlDeviceMemory;

    fn fake_device(
        ordinal: usize,
        name: &str,
        kind: GgmlBackendKind,
        device_id: Option<&str>,
    ) -> EnumeratedComputeDevice {
        let provider = ExecutionProvider::from_backend_name(name);
        let route_kind = if kind == GgmlBackendKind::Cpu {
            RouteDeviceKind::Cpu
        } else {
            RouteDeviceKind::Accelerated
        };
        EnumeratedComputeDevice {
            provider,
            stable_id: name.to_string(),
            description: name.to_string(),
            registry_ordinal: ordinal,
            kind: route_kind,
            ggml_kind: kind,
            addressability: addressability_for_device(provider, device_id),
            device_id: device_id.map(str::to_string),
        }
    }

    fn hybrid_inventory() -> Vec<EnumeratedComputeDevice> {
        vec![
            fake_device(0, "CPU", GgmlBackendKind::Cpu, None),
            fake_device(
                1,
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some("0000:00:02.0"),
            ),
            fake_device(2, "Vulkan1", GgmlBackendKind::Gpu, Some("0000:01:00.0")),
        ]
    }

    #[test]
    fn auto_prefers_discrete_gpu_over_integrated() {
        let route = resolve_execution_route(&ExecutionRouteRequest::Auto, &hybrid_inventory())
            .expect("auto resolves");
        assert_eq!(route.stable_id, "Vulkan1");
        assert_eq!(route.provider, ExecutionProvider::Vulkan);
        assert!(route.addressability.is_exactly_addressable());
        assert_eq!(route.isolation_key(), "vulkan/Vulkan1/pci:0000:01:00.0");
    }

    #[test]
    fn accelerated_fail_closed_without_gpu() {
        let inventory = vec![fake_device(0, "CPU", GgmlBackendKind::Cpu, None)];
        let error = resolve_execution_route(&ExecutionRouteRequest::Accelerated, &inventory)
            .expect_err("no gpu");
        assert_eq!(error, ExecutionRouteError::AcceleratedUnavailable);
    }

    #[test]
    fn exact_by_physical_key_pins_one_card() {
        let inventory = hybrid_inventory();
        let key = PhysicalResourceKey::new("0000:00:02.0").unwrap();
        let route = resolve_execution_route(
            &ExecutionRouteRequest::Exact(ExactDeviceSelector::PhysicalKey(key)),
            &inventory,
        )
        .expect("exact physical");
        assert_eq!(route.stable_id, "Vulkan0");
        assert_ne!(
            route.isolation_key(),
            hybrid_inventory()[2].to_resolved_route().isolation_key()
        );
    }

    #[test]
    fn exact_missing_device_is_device_not_found() {
        let key = PhysicalResourceKey::new("0000:ff:00.0").unwrap();
        let error = resolve_execution_route(
            &ExecutionRouteRequest::Exact(ExactDeviceSelector::PhysicalKey(key)),
            &hybrid_inventory(),
        )
        .expect_err("missing");
        assert!(matches!(error, ExecutionRouteError::DeviceNotFound { .. }));
    }

    #[test]
    fn metal_exact_is_not_addressable() {
        let inventory = vec![
            fake_device(0, "CPU", GgmlBackendKind::Cpu, None),
            fake_device(1, "Metal", GgmlBackendKind::Gpu, None),
        ];
        let error = resolve_execution_route(
            &ExecutionRouteRequest::Exact(ExactDeviceSelector::StableId {
                provider: Some(ExecutionProvider::Metal),
                stable_id: "Metal".to_string(),
            }),
            &inventory,
        )
        .expect_err("metal exact");
        assert!(matches!(error, ExecutionRouteError::NotAddressable { .. }));
        assert!(!inventory[1].addressability.is_exactly_addressable());
    }

    #[test]
    fn cuda_without_pci_still_allows_stable_id_exact() {
        let inventory = vec![
            fake_device(0, "CPU", GgmlBackendKind::Cpu, None),
            fake_device(1, "CUDA0", GgmlBackendKind::Gpu, None),
            fake_device(2, "CUDA1", GgmlBackendKind::Gpu, None),
        ];
        let route = resolve_execution_route(
            &ExecutionRouteRequest::Exact(ExactDeviceSelector::StableId {
                provider: Some(ExecutionProvider::Cuda),
                stable_id: "CUDA1".to_string(),
            }),
            &inventory,
        )
        .expect("stable id exact");
        assert_eq!(route.stable_id, "CUDA1");
        assert_eq!(route.isolation_key(), "cuda/CUDA1");
        assert_eq!(route.cache_key().stable_id, "CUDA1");
    }

    #[test]
    fn admission_and_worker_keys_include_resolved_route() {
        let route =
            resolve_execution_route(&ExecutionRouteRequest::Accelerated, &hybrid_inventory())
                .unwrap();
        assert_eq!(
            admission_identity_for_route("native:whisper@pack", Some(&route)),
            "native:whisper@pack|route=vulkan/Vulkan1/pci:0000:01:00.0"
        );
        assert_eq!(
            worker_route_isolation_key("accelerated", Some(&route)),
            "vulkan/Vulkan1/pci:0000:01:00.0"
        );
        assert_eq!(worker_route_isolation_key("cpu", None), "cpu");
    }

    #[test]
    fn ggml_device_inventory_reads_device_id() {
        let devices = vec![
            GgmlBackendDevice::for_test("CPU", "CPU", GgmlBackendKind::Cpu, None),
            GgmlBackendDevice::for_test_with_device_id(
                "CUDA0",
                "NVIDIA A100",
                GgmlBackendKind::Gpu,
                Some(GgmlDeviceMemory {
                    free_bytes: 1,
                    total_bytes: 2,
                }),
                Some("0000:C1:00.0"),
            ),
        ];
        let inventory = enumerate_compute_devices_from_ggml(&devices);
        assert_eq!(inventory[1].provider, ExecutionProvider::Cuda);
        assert_eq!(
            inventory[1]
                .addressability
                .physical_key()
                .map(PhysicalResourceKey::as_str),
            Some("0000:c1:00.0")
        );
    }
}
