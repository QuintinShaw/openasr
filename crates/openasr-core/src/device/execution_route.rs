//! Request-level execution route foundation.
//!
//! Public request surfaces stay coarse (`ExecutionTarget::{Auto,Cpu,Accelerated}`).
//! This module adds the internal exact-route vocabulary that backend-handle cache
//! and streaming-worker isolation share before any product UI exposes GPU0/GPU1
//! picks.
//!
//! Correct abstraction:
//! - [`ResolvedExecutionRoute`] = logical `(provider, stable_id)` plus optional
//!   [`PhysicalResourceKey`] (PCI BDF when ggml supplies `device_id`)
//! - Exact resolution is typed fail-closed: no silent card swap, no CPU fallback
//! - Metal devices are enumerable but [`DeviceAddressability::NotExactlyAddressable`]
//!   because ggml Metal still initializes via `MTLCreateSystemDefaultDevice` only
//! - Admission capacity stays **per physical device** through the device-memory
//!   broker. Route identity also feeds the unified execution-lane key used by
//!   every resident backend owner and serve-batch engine. Content-only prepared
//!   caches are compile-time restricted to host-neutral values.

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
///
/// Normalization is intentionally weak today: trim + ASCII lower-case only.
/// A full PCI BDF grammar validator is a known follow-up (see
/// `docs/KNOWN_LIMITATIONS.md`); callers must not treat this type as proof of
/// a well-formed BDF string.
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

/// Stable hardware-vendor identity used only when a product target names a
/// vendor. Provider-specific backends are authoritative by construction;
/// shared providers such as Vulkan require an explicit PCI vendor id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionHardwareVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
}

impl fmt::Display for ExecutionHardwareVendor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Apple => "Apple",
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
        })
    }
}

impl RouteDeviceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Accelerated => "accelerated",
        }
    }
}

/// Logical route identity used for backend-handle cache and streaming-worker
/// isolation (not admission capacity -- see [`admission_identity_for_route`]).
///
/// `(provider, stable_id)` is always present. [`PhysicalResourceKey`] is layered
/// on when ggml supplies a PCI-style `device_id`. `registry_ordinal` is retained
/// so init-time device matching can disambiguate inventory rows when PCI ids are
/// absent; it is intentionally **not** part of [`Self::isolation_key`] /
/// [`Self::cache_key`] so a stable device keeps its key across harmless
/// re-enumeration order shifts when `stable_id` (and PCI when present) identify it.
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
    /// Isolation key shared by the thread-local ggml backend-handle cache and
    /// streaming worker keys. Exact and preferred-accelerated routes that resolve
    /// to the same device must produce the same key. Admission capacity does
    /// **not** use this key (see [`admission_identity_for_route`]).
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

    /// Backend-handle cache key: provider + stable_id, plus physical key when
    /// present. Ordinal is intentionally excluded (see struct docs).
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

    /// Recover a typed route error from a coarse string that embedded this
    /// error's Display text (family executors historically stringify
    /// `GgmlCpuGraphError` before it reaches `dispatch_error_to_backend`).
    ///
    /// Matches the stable thiserror messages below; keep those strings in sync
    /// if the `#[error(...)]` text on this enum changes.
    pub fn from_embedded_message(message: &str) -> Option<Self> {
        const NOT_FOUND: &str = "requested execution device was not found: ";
        const NOT_ADDRESSABLE: &str = "requested execution device is not exactly addressable: ";
        const INIT_FAILED: &str = "requested execution device failed to initialize: ";
        const ACCEL_UNAVAILABLE: &str = "no accelerated execution device is available";

        if let Some(detail) = message
            .rfind(NOT_FOUND)
            .map(|idx| &message[idx + NOT_FOUND.len()..])
        {
            return Some(Self::device_not_found(detail.trim()));
        }
        if let Some(detail) = message
            .rfind(NOT_ADDRESSABLE)
            .map(|idx| &message[idx + NOT_ADDRESSABLE.len()..])
        {
            return Some(Self::not_addressable(detail.trim()));
        }
        if let Some(detail) = message
            .rfind(INIT_FAILED)
            .map(|idx| &message[idx + INIT_FAILED.len()..])
        {
            return Some(Self::init_failed(detail.trim()));
        }
        if message.contains(ACCEL_UNAVAILABLE) {
            return Some(Self::AcceleratedUnavailable);
        }
        None
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
    /// Live memory observation reported by the backend at enumeration time.
    /// Selection policy must not reinterpret this as an allocation guarantee;
    /// the memory broker refreshes/validates it when admitting a candidate.
    pub memory: Option<crate::ggml_runtime::GgmlDeviceMemory>,
    /// Default backend-buffer alignment for physical-footprint rounding.
    pub buffer_alignment: Option<usize>,
    pub addressability: DeviceAddressability,
    /// Raw ggml `device_id` string when present (pre-normalization).
    pub device_id: Option<String>,
    /// Proven hardware vendor. Vulkan is populated only from the backend's
    /// numeric PCI vendor procedure; a missing fact remains `None`.
    pub hardware_vendor: Option<ExecutionHardwareVendor>,
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
        memory: device.memory,
        buffer_alignment: device.buffer_alignment,
        addressability,
        device_id: device.device_id.clone(),
        hardware_vendor: hardware_vendor_for_device(provider, device.pci_vendor_id),
    }
}

fn hardware_vendor_for_device(
    provider: ExecutionProvider,
    pci_vendor_id: Option<u32>,
) -> Option<ExecutionHardwareVendor> {
    match provider {
        ExecutionProvider::Metal => cfg!(all(target_vendor = "apple", target_arch = "aarch64"))
            .then_some(ExecutionHardwareVendor::Apple),
        ExecutionProvider::Cuda => Some(ExecutionHardwareVendor::Nvidia),
        ExecutionProvider::Hip => Some(ExecutionHardwareVendor::Amd),
        ExecutionProvider::Vulkan => match pci_vendor_id {
            Some(0x1002) | Some(0x1022) => Some(ExecutionHardwareVendor::Amd),
            Some(0x8086) => Some(ExecutionHardwareVendor::Intel),
            Some(0x10de) => Some(ExecutionHardwareVendor::Nvidia),
            _ => None,
        },
        ExecutionProvider::Cpu | ExecutionProvider::Accelerator | ExecutionProvider::Unknown => {
            None
        }
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
    ranked_preferred_accelerated_devices(inventory)
        .into_iter()
        .next()
        .map(EnumeratedComputeDevice::to_resolved_route)
        .ok_or(ExecutionRouteError::AcceleratedUnavailable)
}

/// Preferred accelerated candidates in Optimus-aware rank order
/// (discrete GPU before iGPU / accelerator), then registry ordinal.
///
/// Used by route resolve and by preferred/Auto GPU backend init so discrete
/// init failure can fall through to the next ranked device **under that
/// device's own cache key**.
pub fn ranked_preferred_accelerated_devices(
    inventory: &[EnumeratedComputeDevice],
) -> Vec<&EnumeratedComputeDevice> {
    ranked_preferred_accelerated_devices_for_provider(
        inventory,
        crate::ggml_runtime::activated_backend_execution_provider(),
    )
}

pub(crate) fn ranked_preferred_accelerated_devices_for_provider(
    inventory: &[EnumeratedComputeDevice],
    preferred_provider: Option<ExecutionProvider>,
) -> Vec<&EnumeratedComputeDevice> {
    let has_preferred_provider = preferred_provider.is_some_and(|provider| {
        inventory.iter().any(|device| {
            device.kind == RouteDeviceKind::Accelerated && device.provider == provider
        })
    });
    let mut devices: Vec<&EnumeratedComputeDevice> = inventory
        .iter()
        .filter(|device| device.kind == RouteDeviceKind::Accelerated)
        .collect();
    devices.sort_by_key(|device| {
        (
            u8::from(
                has_preferred_provider
                    && preferred_provider.is_some_and(|provider| device.provider != provider),
            ),
            crate::ggml_runtime::accelerated_device_rank(device.ggml_kind),
            device.registry_ordinal,
        )
    });
    devices
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
            // CPU is selected only by the coarse `cpu` target, never Exact pin.
            if device.provider == ExecutionProvider::Cpu {
                return Err(ExecutionRouteError::not_addressable(format!(
                    "provider=cpu stable_id={} reason={}",
                    device.stable_id,
                    match &device.addressability {
                        DeviceAddressability::NotExactlyAddressable { reason } => *reason,
                        DeviceAddressability::ExactlyAddressable { .. } => {
                            "CPU is selected by the coarse cpu target, not by Exact device pin"
                        }
                    }
                )));
            }
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
            // must not silently retarget another card. Non-exact providers
            // (Accelerator/Unknown/...) stay fail-closed here.
            if matches!(selector, ExactDeviceSelector::StableId { .. })
                && !device.provider.supports_exact_selection()
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

/// Admission / capacity slot identity for native model sessions.
///
/// Foundation invariant: capacity is **per model identity only**. The optional
/// `route` argument is accepted so call sites can pass the resolved route they
/// already have for worker/backend-handle isolation, but it must not split the
/// global per-model slot (CPU and accelerated/Exact requests for the same model
/// share one capacity unit). Route isolation belongs to backend-handle cache and
/// streaming workers via [`worker_route_isolation_key`] / [`ResolvedExecutionRoute::cache_key`].
pub fn admission_identity_for_route(
    model_identity: &str,
    route: Option<&ResolvedExecutionRoute>,
) -> String {
    let _ = route;
    model_identity.to_string()
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
            memory: Some(GgmlDeviceMemory {
                free_bytes: 8 * 1024 * 1024 * 1024,
                total_bytes: 8 * 1024 * 1024 * 1024,
            }),
            buffer_alignment: Some(256),
            addressability: addressability_for_device(provider, device_id),
            device_id: device_id.map(str::to_string),
            hardware_vendor: hardware_vendor_for_device(provider, None),
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
    fn admission_stays_per_model_while_worker_keys_include_route() {
        let accelerated =
            resolve_execution_route(&ExecutionRouteRequest::Accelerated, &hybrid_inventory())
                .unwrap();
        let cpu =
            resolve_execution_route(&ExecutionRouteRequest::Cpu, &hybrid_inventory()).unwrap();
        // Capacity is global per model: CPU and accelerated must share one slot.
        assert_eq!(
            admission_identity_for_route("native:whisper@pack", Some(&accelerated)),
            "native:whisper@pack"
        );
        assert_eq!(
            admission_identity_for_route("native:whisper@pack", Some(&cpu)),
            admission_identity_for_route("native:whisper@pack", Some(&accelerated))
        );
        assert_eq!(
            admission_identity_for_route("native:whisper@pack", None),
            "native:whisper@pack"
        );
        // Workers still isolate by concrete device.
        assert_eq!(
            worker_route_isolation_key("accelerated", Some(&accelerated)),
            "vulkan/Vulkan1/pci:0000:01:00.0"
        );
        assert_ne!(
            worker_route_isolation_key("accelerated", Some(&accelerated)),
            worker_route_isolation_key("cpu", Some(&cpu))
        );
        assert_eq!(worker_route_isolation_key("cpu", None), "cpu");
    }

    #[test]
    fn cpu_stable_id_exact_is_not_addressable() {
        let inventory = hybrid_inventory();
        let error = resolve_execution_route(
            &ExecutionRouteRequest::Exact(ExactDeviceSelector::StableId {
                provider: Some(ExecutionProvider::Cpu),
                stable_id: "CPU".to_string(),
            }),
            &inventory,
        )
        .expect_err("cpu exact");
        assert!(matches!(error, ExecutionRouteError::NotAddressable { .. }));
    }

    #[test]
    fn preferred_accelerated_rank_orders_discrete_before_integrated() {
        let inventory = hybrid_inventory();
        let ranked = ranked_preferred_accelerated_devices(&inventory);
        assert_eq!(
            ranked
                .iter()
                .map(|device| device.stable_id.as_str())
                .collect::<Vec<_>>(),
            vec!["Vulkan1", "Vulkan0"]
        );
        // Fallthrough cache keys must be the actual candidate's key, never the
        // preferred route's key reused for a lower-ranked device.
        assert_ne!(
            ranked[0].to_resolved_route().cache_key(),
            ranked[1].to_resolved_route().cache_key()
        );
    }

    #[test]
    fn preferred_accelerated_rank_puts_activated_provider_before_registry_order() {
        let inventory = vec![
            fake_device(0, "Vulkan0", GgmlBackendKind::Gpu, Some("0000:01:00.0")),
            fake_device(1, "CUDA0", GgmlBackendKind::Gpu, Some("0000:01:00.0")),
        ];
        let ranked = ranked_preferred_accelerated_devices_for_provider(
            &inventory,
            Some(ExecutionProvider::Cuda),
        );
        assert_eq!(ranked[0].provider, ExecutionProvider::Cuda);
        assert_eq!(ranked[1].provider, ExecutionProvider::Vulkan);
    }

    #[test]
    fn preferred_accelerated_rank_ignores_absent_activated_provider() {
        let inventory = vec![
            fake_device(
                0,
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some("0000:00:02.0"),
            ),
            fake_device(1, "Vulkan1", GgmlBackendKind::Gpu, Some("0000:01:00.0")),
        ];
        let ranked = ranked_preferred_accelerated_devices_for_provider(
            &inventory,
            Some(ExecutionProvider::Cuda),
        );
        assert_eq!(ranked[0].stable_id, "Vulkan1");
    }

    #[test]
    fn embedded_route_error_message_recovers_typed_variant() {
        let source =
            ExecutionRouteError::init_failed("provider=cuda stable_id=CUDA0 backend=CUDA0");
        let wrapped = format!("could not initialize ggml cpu graph runner: {source}");
        assert_eq!(
            ExecutionRouteError::from_embedded_message(&wrapped),
            Some(source)
        );
        assert_eq!(
            ExecutionRouteError::from_embedded_message(
                "ggml executor failed: no accelerated execution device is available"
            ),
            Some(ExecutionRouteError::AcceleratedUnavailable)
        );
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

    #[test]
    fn vulkan_vendor_is_accepted_only_from_numeric_backend_fact() {
        let proven = GgmlBackendDevice::for_test_with_hardware_facts(
            "Vulkan0",
            "arbitrary description",
            GgmlBackendKind::Gpu,
            None,
            Some("0000:01:00.0"),
            Some(0x1002),
        );
        let unknown = GgmlBackendDevice::for_test_with_hardware_facts(
            "Vulkan1",
            "AMD words are not evidence",
            GgmlBackendKind::Gpu,
            None,
            Some("0000:02:00.0"),
            None,
        );
        let inventory = enumerate_compute_devices_from_ggml(&[proven, unknown]);
        assert_eq!(
            inventory[0].hardware_vendor,
            Some(ExecutionHardwareVendor::Amd)
        );
        assert_eq!(inventory[1].hardware_vendor, None);
    }
}
