//! Safe ownership wrappers around ggml's optional physical-memory ABI.
//!
//! This layer deliberately does not infer policy or physical-domain aliases.
//! It preserves the backend's native UUID/heap/kind claims for the process
//! broker to map, reserve atomically, and reconcile after commit.

#![allow(dead_code)]

use std::{ffi::c_void, marker::PhantomData, mem, ptr};

use thiserror::Error;

use crate::device::execution_memory::{MemoryDomainKey, MemoryObservationConfidence};
use crate::device::execution_route::ExecutionProvider;

use super::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryLifecyclePoint {
    BackendInitialized,
    AdmissionQuote,
    PostAllocationReconciliation,
    TerminalFailure,
    AfterGraphCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryUnknownReason {
    AbiUnavailable,
    StatsUnavailable,
    IncompatibleStats,
    DeviceBudgetUnavailable,
    ProviderDoesNotReportBackendOwned,
    ProviderOwnedAccountingIncomplete,
    ProviderReliabilityUnspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryBytes {
    Known(u64),
    Unknown(BackendMemoryUnknownReason),
}

pub(crate) fn backend_owned_unknown_reason(
    provider: ExecutionProvider,
) -> Option<BackendMemoryUnknownReason> {
    match provider {
        ExecutionProvider::Vulkan => {
            Some(BackendMemoryUnknownReason::ProviderDoesNotReportBackendOwned)
        }
        ExecutionProvider::Cpu => None,
        ExecutionProvider::Cuda => {
            Some(BackendMemoryUnknownReason::ProviderOwnedAccountingIncomplete)
        }
        _ => Some(BackendMemoryUnknownReason::ProviderReliabilityUnspecified),
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMemoryDomainKind {
    HostPageable,
    HostPinned,
    Unified,
    DeviceLocal,
    FileBacked,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendDeviceHealth {
    Healthy,
    Degraded,
    Quarantined,
    DeviceLost,
    Unavailable,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendTerminalStatusClass {
    Success,
    Validation,
    Capacity,
    Cancelled,
    Execution,
    DeviceLost,
    BackendPoisoned,
    Unknown(i32),
}

impl BackendTerminalStatusClass {
    fn from_raw(status: i32) -> Self {
        match status {
            ffi::GGML_STATUS_SUCCESS => Self::Success,
            ffi::GGML_STATUS_FAILED => Self::Validation,
            ffi::GGML_STATUS_ALLOC_FAILED => Self::Capacity,
            ffi::GGML_STATUS_ABORTED => Self::Cancelled,
            ffi::GGML_STATUS_EXECUTION_FAILED => Self::Execution,
            ffi::GGML_STATUS_DEVICE_LOST => Self::DeviceLost,
            ffi::GGML_STATUS_BACKEND_POISONED => Self::BackendPoisoned,
            status => Self::Unknown(status),
        }
    }
}

/// Sanitized memory evidence attached to an Exact smoke observation. It never
/// exposes physical UUIDs, backend pointers, paths, or native error text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafeBackendMemoryReceipt {
    pub(crate) provider: ExecutionProvider,
    pub(crate) lifecycle: BackendMemoryLifecyclePoint,
    pub(crate) device_health: BackendDeviceHealth,
    pub(crate) last_status: BackendTerminalStatusClass,
    pub(crate) last_native_error: i64,
    pub(crate) quarantine_generation: u64,
    pub(crate) domain_kind: Option<BackendMemoryDomainKind>,
    pub(crate) heap_index: Option<u32>,
    pub(crate) total_bytes: BackendMemoryBytes,
    pub(crate) budget_bytes: BackendMemoryBytes,
    pub(crate) stats_generation: BackendMemoryBytes,
    pub(crate) quote_generation: BackendMemoryBytes,
    pub(crate) claim_flags: u32,
    pub(crate) observation_confidence: MemoryObservationConfidence,
    pub(crate) device_used_bytes: BackendMemoryBytes,
    pub(crate) device_free_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_live_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_cached_bytes: BackendMemoryBytes,
    pub(crate) backend_owned_workspace_bytes: BackendMemoryBytes,
    /// Greatest provider-reported high-water or current commitment proven at
    /// this sample. The observation sink carries the maximum across samples.
    pub(crate) backend_owned_observed_high_water_bytes: BackendMemoryBytes,
}

impl SafeBackendMemoryReceipt {
    pub(crate) fn unknown(
        lifecycle: BackendMemoryLifecyclePoint,
        reason: BackendMemoryUnknownReason,
    ) -> Self {
        Self::unknown_for_provider(ExecutionProvider::Unknown, lifecycle, reason)
    }

    fn unknown_for_provider(
        provider: ExecutionProvider,
        lifecycle: BackendMemoryLifecyclePoint,
        reason: BackendMemoryUnknownReason,
    ) -> Self {
        let value = BackendMemoryBytes::Unknown(reason);
        Self {
            provider,
            lifecycle,
            device_health: BackendDeviceHealth::Unavailable,
            last_status: BackendTerminalStatusClass::Unknown(i32::MIN),
            last_native_error: 0,
            quarantine_generation: 0,
            domain_kind: None,
            heap_index: None,
            total_bytes: value,
            budget_bytes: value,
            stats_generation: value,
            quote_generation: BackendMemoryBytes::Unknown(reason),
            claim_flags: 0,
            observation_confidence: MemoryObservationConfidence::Unknown,
            device_used_bytes: value,
            device_free_bytes: value,
            backend_owned_live_bytes: value,
            backend_owned_cached_bytes: value,
            backend_owned_workspace_bytes: value,
            backend_owned_observed_high_water_bytes: value,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BackendMemoryAbiError {
    #[error("backend memory ABI is unavailable")]
    Unavailable,
    #[error("backend memory ABI v1 has an incompatible layout or version")]
    Incompatible,
    #[error("backend memory operation '{operation}' failed with ggml status {status}")]
    Status {
        operation: &'static str,
        status: i32,
    },
    #[error("backend memory operation '{operation}' returned an unstable item count")]
    UnstableCount { operation: &'static str },
    #[error(
        "backend memory reserve_private committed but returned an unstable actual-claim count: sized={sized}, returned={returned}"
    )]
    ReservePrivatePostCommitCountMismatch { sized: u32, returned: u32 },
    #[error("backend memory quote mixed requests from different primary backends")]
    MixedPrimaryBackend,
    #[error("scheduler memory plan returned an item without a primary backend")]
    MissingPrimaryBackend,
}

impl BackendMemoryAbiError {
    /// `reserve_private` returned native success before Rust discovered an
    /// invalid result shape. Unlike a native non-success (failure-atomic by
    /// ABI contract), this state may already retain private allocations and
    /// therefore must be quarantined rather than refunded.
    pub(crate) fn may_have_committed_private_state(&self) -> bool {
        matches!(self, Self::ReservePrivatePostCommitCountMismatch { .. })
    }

    pub(crate) fn terminal_status(&self) -> i32 {
        match self {
            Self::Status { status, .. } => *status,
            _ => i32::MAX,
        }
    }
}

fn status(operation: &'static str, value: i32) -> Result<(), BackendMemoryAbiError> {
    if value == ffi::GGML_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(BackendMemoryAbiError::Status {
            operation,
            status: value,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BackendMemoryAbi {
    raw: &'static ffi::GgmlBackendMemoryApiV1,
    backend: ffi::GgmlBackendRaw,
    device: ffi::GgmlBackendDevRaw,
}

unsafe extern "C" {
    fn ggml_backend_memory_api_for_device_v1(
        device: ffi::GgmlBackendDevRaw,
    ) -> *const ffi::GgmlBackendMemoryApiV1;
}

impl BackendMemoryAbi {
    /// Resolves the optional v1 table from the concrete backend's registry.
    ///
    /// `backend` must remain a live ggml backend. Plugin registries are process
    /// resident after loading, so the returned function table has static
    /// lifetime even when the backend owner later drops.
    pub(crate) unsafe fn from_backend(
        backend: ffi::GgmlBackendRaw,
    ) -> Result<Self, BackendMemoryAbiError> {
        if backend.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: caller guarantees `backend` is live.
        let device = unsafe { ffi::ggml_backend_get_device(backend) };
        if device.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: the shared ggml trampoline owns registry lookup and catches
        // every provider exception before it can cross this Rust FFI seam.
        let raw = unsafe { ffi::ggml_backend_memory_api_for_backend_v1(backend) };
        let Some(raw) = (unsafe { raw.as_ref() }) else {
            return Err(BackendMemoryAbiError::Unavailable);
        };
        if raw.struct_size < mem::size_of::<ffi::GgmlBackendMemoryApiV1>() as u32
            || raw.abi_version != ffi::GGML_BACKEND_MEMORY_ABI_V1
            || raw.get_domains.is_none()
            || raw.quote.is_none()
            || raw.reserve_private.is_none()
            || raw.get_stats.is_none()
        {
            return Err(BackendMemoryAbiError::Incompatible);
        }
        Ok(Self {
            raw,
            backend,
            device,
        })
    }

    /// Resolves the optional v1 table from a registry device without constructing
    /// a live backend context. Vulkan BUFFER quotes accept a null `backend` when
    /// `buft` is set; AMD WDDM does not return the ~2.1 MiB host slab that
    /// `ggml_backend_dev_init` leaves behind after `ggml_backend_free`.
    ///
    /// `device` must remain a live registry device. Plugin registries are
    /// process resident after loading, so the returned function table has
    /// static lifetime.
    pub(crate) unsafe fn from_device(
        device: ffi::GgmlBackendDevRaw,
    ) -> Result<Self, BackendMemoryAbiError> {
        if device.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: the shared ggml trampoline owns registry lookup and catches
        // every provider exception before it can cross this Rust FFI seam.
        let raw = unsafe { ggml_backend_memory_api_for_device_v1(device) };
        let Some(raw) = (unsafe { raw.as_ref() }) else {
            return Err(BackendMemoryAbiError::Unavailable);
        };
        if raw.struct_size < mem::size_of::<ffi::GgmlBackendMemoryApiV1>() as u32
            || raw.abi_version != ffi::GGML_BACKEND_MEMORY_ABI_V1
            || raw.get_domains.is_none()
            || raw.quote.is_none()
            || raw.reserve_private.is_none()
            || raw.get_stats.is_none()
        {
            return Err(BackendMemoryAbiError::Incompatible);
        }
        Ok(Self {
            raw,
            backend: ptr::null_mut(),
            device,
        })
    }

    pub(crate) fn domains(
        &self,
    ) -> Result<Vec<ffi::GgmlBackendMemoryDomainV1>, BackendMemoryAbiError> {
        self.raw
            .get_domains
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        let mut count = 0_u32;
        status("domains/count", unsafe {
            ffi::ggml_backend_memory_api_get_domains_v1(
                self.raw,
                self.device,
                ptr::null_mut(),
                &mut count,
            )
        })?;
        let mut domains: Vec<_> = (0..count)
            .map(|_| ffi::GgmlBackendMemoryDomainV1 {
                struct_size: mem::size_of::<ffi::GgmlBackendMemoryDomainV1>() as u32,
                flags: 0,
                id: ffi::GgmlBackendMemoryDomainIdV1::default(),
                name: [0; 48],
            })
            .collect();
        let mut capacity = count;
        status("domains", unsafe {
            ffi::ggml_backend_memory_api_get_domains_v1(
                self.raw,
                self.device,
                domains.as_mut_ptr(),
                &mut capacity,
            )
        })?;
        if capacity > count {
            return Err(BackendMemoryAbiError::UnstableCount {
                operation: "domains",
            });
        }
        domains.truncate(capacity as usize);
        Ok(domains)
    }

    pub(crate) fn quote(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
    ) -> Result<BackendMemoryQuote, BackendMemoryAbiError> {
        self.validate_primary_backends(requests)?;
        self.raw.quote.ok_or(BackendMemoryAbiError::Incompatible)?;
        let count = u32::try_from(requests.len())
            .map_err(|_| BackendMemoryAbiError::UnstableCount { operation: "quote" })?;
        let request_ptr = if requests.is_empty() {
            ptr::null()
        } else {
            requests.as_ptr()
        };
        let mut raw = ffi::GgmlBackendMemoryQuoteV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>() as u32,
            ..Default::default()
        };
        let mut claim_count = 0_u32;
        // SAFETY: all pointers refer to initialized values for this call.
        status("quote/count", unsafe {
            ffi::ggml_backend_memory_api_quote_v1(
                self.raw,
                request_ptr,
                count,
                &mut raw,
                ptr::null_mut(),
                &mut claim_count,
            )
        })?;

        let mut claims = initialized_claims(claim_count as usize);
        let mut capacity = claim_count;
        // SAFETY: `claims` has initialized writable elements and capacity.
        status("quote", unsafe {
            ffi::ggml_backend_memory_api_quote_v1(
                self.raw,
                request_ptr,
                count,
                &mut raw,
                claims.as_mut_ptr(),
                &mut capacity,
            )
        })?;
        if capacity as usize > claims.len() {
            return Err(BackendMemoryAbiError::UnstableCount { operation: "quote" });
        }
        claims.truncate(capacity as usize);
        Ok(BackendMemoryQuote { raw, claims })
    }

    /// Performs backend-private transactional reservation against the exact
    /// quote token. Engine-controlled buffers are committed separately by the
    /// frozen scheduler plan, then callers fetch fresh stats for reconcile.
    pub(crate) fn reserve_private(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
        quote: &BackendMemoryQuote,
    ) -> Result<Vec<ffi::GgmlBackendMemoryClaimV1>, BackendMemoryAbiError> {
        self.validate_primary_backends(requests)?;
        self.raw
            .reserve_private
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        let count =
            u32::try_from(requests.len()).map_err(|_| BackendMemoryAbiError::UnstableCount {
                operation: "reserve_private",
            })?;
        let request_ptr = if requests.is_empty() {
            ptr::null()
        } else {
            requests.as_ptr()
        };
        let mut actual_count = 0_u32;
        // First call is a sizing query and must not mutate backend state.
        status("reserve_private/count", unsafe {
            ffi::ggml_backend_memory_api_reserve_private_v1(
                self.raw,
                request_ptr,
                count,
                &quote.raw,
                ptr::null_mut(),
                &mut actual_count,
            )
        })?;
        // A non-null pointer is intentional even for zero items: it
        // distinguishes the commit call from the preceding sizing query.
        let mut actual = initialized_claims(actual_count.max(1) as usize);
        let mut capacity = actual_count;
        status("reserve_private", unsafe {
            ffi::ggml_backend_memory_api_reserve_private_v1(
                self.raw,
                request_ptr,
                count,
                &quote.raw,
                actual.as_mut_ptr(),
                &mut capacity,
            )
        })?;
        if capacity > actual_count {
            return Err(
                BackendMemoryAbiError::ReservePrivatePostCommitCountMismatch {
                    sized: actual_count,
                    returned: capacity,
                },
            );
        }
        actual.truncate(capacity as usize);
        Ok(actual)
    }

    pub(crate) fn stats(&self) -> Result<BackendMemoryStatsSnapshot, BackendMemoryAbiError> {
        self.raw
            .get_stats
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        let mut count = 0_u32;
        status("stats/count", unsafe {
            ffi::ggml_backend_memory_api_get_stats_v1(
                self.raw,
                self.device,
                self.backend,
                ptr::null_mut(),
                &mut count,
            )
        })?;
        let mut domains = initialized_stats(count as usize);
        let mut capacity = count;
        status("stats", unsafe {
            ffi::ggml_backend_memory_api_get_stats_v1(
                self.raw,
                self.device,
                self.backend,
                domains.as_mut_ptr(),
                &mut capacity,
            )
        })?;
        if capacity > count {
            return Err(BackendMemoryAbiError::UnstableCount { operation: "stats" });
        }
        domains.truncate(capacity as usize);
        Ok(BackendMemoryStatsSnapshot { domains })
    }

    pub(crate) fn stats_at(
        &self,
        lifecycle: BackendMemoryLifecyclePoint,
    ) -> Result<BackendMemoryStatsSnapshot, BackendMemoryAbiError> {
        let snapshot = self.stats()?;
        crate::models::native_execution_services::record_current_execution_backend_memory_stats(
            self.backend as usize,
            lifecycle,
            &snapshot,
        );
        Ok(snapshot)
    }

    pub(crate) fn backend(&self) -> ffi::GgmlBackendRaw {
        self.backend
    }

    pub(crate) fn provider(&self) -> ExecutionProvider {
        if self.device.is_null() {
            return ExecutionProvider::Unknown;
        }
        let name = unsafe { ffi::ggml_backend_dev_name(self.device) };
        if name.is_null() {
            return ExecutionProvider::Unknown;
        }
        let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
        ExecutionProvider::from_backend_name(name.as_ref())
    }

    pub(crate) fn trim(&self, flags: u64) -> Result<(), BackendMemoryAbiError> {
        self.raw.trim.ok_or(BackendMemoryAbiError::Incompatible)?;
        status("trim", unsafe {
            ffi::ggml_backend_memory_api_trim_v1(self.raw, self.backend, flags)
        })
    }

    pub(crate) fn quarantine(
        &self,
        request: &ffi::GgmlBackendMemoryQuarantineV1,
    ) -> Result<(), BackendMemoryAbiError> {
        self.raw
            .quarantine
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        status("quarantine", unsafe {
            ffi::ggml_backend_memory_api_quarantine_v1(self.raw, self.backend, request)
        })
    }

    fn validate_primary_backends(
        &self,
        requests: &[ffi::GgmlBackendMemoryRequestV1],
    ) -> Result<(), BackendMemoryAbiError> {
        if requests
            .iter()
            .any(|request| !request.backend.is_null() && request.backend != self.backend)
        {
            return Err(BackendMemoryAbiError::MixedPrimaryBackend);
        }
        Ok(())
    }
}

fn initialized_claims(count: usize) -> Vec<ffi::GgmlBackendMemoryClaimV1> {
    (0..count)
        .map(|_| ffi::GgmlBackendMemoryClaimV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryClaimV1>() as u32,
            ..Default::default()
        })
        .collect()
}

fn initialized_stats(count: usize) -> Vec<ffi::GgmlBackendMemoryStatsV1> {
    (0..count)
        .map(|_| ffi::GgmlBackendMemoryStatsV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32,
            ..Default::default()
        })
        .collect()
}

#[derive(Debug)]
pub(crate) struct BackendMemoryQuote {
    raw: ffi::GgmlBackendMemoryQuoteV1,
    claims: Vec<ffi::GgmlBackendMemoryClaimV1>,
}

impl BackendMemoryQuote {
    pub(crate) fn raw(&self) -> &ffi::GgmlBackendMemoryQuoteV1 {
        &self.raw
    }

    pub(crate) fn claims(&self) -> &[ffi::GgmlBackendMemoryClaimV1] {
        &self.claims
    }

    pub(crate) fn is_provisional(&self) -> bool {
        self.raw.flags & ffi::GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL != 0
    }
}

#[derive(Debug)]
pub(crate) struct BackendMemoryStatsSnapshot {
    domains: Vec<ffi::GgmlBackendMemoryStatsV1>,
}

impl BackendMemoryStatsSnapshot {
    pub(crate) fn domains(&self) -> &[ffi::GgmlBackendMemoryStatsV1] {
        &self.domains
    }

    pub(crate) fn safe_receipts(
        &self,
        provider: ExecutionProvider,
        lifecycle: BackendMemoryLifecyclePoint,
    ) -> Vec<SafeBackendMemoryReceipt> {
        self.domains
            .iter()
            .map(|raw| safe_receipt(provider, lifecycle, raw))
            .collect()
    }
}

fn safe_receipt(
    provider: ExecutionProvider,
    lifecycle: BackendMemoryLifecyclePoint,
    raw: &ffi::GgmlBackendMemoryStatsV1,
) -> SafeBackendMemoryReceipt {
    if raw.struct_size < mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32 {
        return SafeBackendMemoryReceipt::unknown_for_provider(
            provider,
            lifecycle,
            BackendMemoryUnknownReason::IncompatibleStats,
        );
    }
    let domain_kind = match raw.domain.kind {
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE => BackendMemoryDomainKind::HostPageable,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED => BackendMemoryDomainKind::HostPinned,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED => BackendMemoryDomainKind::Unified,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL => BackendMemoryDomainKind::DeviceLocal,
        ffi::GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED => BackendMemoryDomainKind::FileBacked,
        kind => BackendMemoryDomainKind::Unknown(kind),
    };
    let observation_confidence = if raw.domain.kind == ffi::GGML_BACKEND_MEMORY_DOMAIN_UNIFIED {
        MemoryObservationConfidence::WorkingSetBudget
    } else {
        MemoryObservationConfidence::DeviceSnapshot
    };
    let total_bytes = if raw.total_bytes == 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::IncompatibleStats)
    } else {
        BackendMemoryBytes::Known(raw.total_bytes)
    };
    let budget_bytes = if raw.flags & ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE != 0
        || raw.budget_bytes == 0
    {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable)
    } else {
        BackendMemoryBytes::Known(raw.budget_bytes)
    };
    let stats_generation = if raw.generation == 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::IncompatibleStats)
    } else {
        BackendMemoryBytes::Known(raw.generation)
    };
    let quote_generation =
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::StatsUnavailable);
    let device = if raw.flags & ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE != 0 {
        BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable)
    } else {
        BackendMemoryBytes::Known(raw.device_used_bytes)
    };
    let device_free = if matches!(device, BackendMemoryBytes::Known(_)) {
        BackendMemoryBytes::Known(raw.device_free_bytes)
    } else {
        device
    };
    let current_owned = raw
        .backend_owned_live_bytes
        .saturating_add(raw.backend_owned_cached_bytes)
        .max(raw.backend_owned_workspace_bytes);
    let owned_unknown = backend_owned_unknown_reason(provider);
    let owned = |value| {
        owned_unknown.map_or(
            BackendMemoryBytes::Known(value),
            BackendMemoryBytes::Unknown,
        )
    };
    SafeBackendMemoryReceipt {
        provider,
        lifecycle,
        device_health: match raw.health {
            ffi::GGML_BACKEND_MEMORY_HEALTHY => BackendDeviceHealth::Healthy,
            ffi::GGML_BACKEND_MEMORY_DEGRADED => BackendDeviceHealth::Degraded,
            ffi::GGML_BACKEND_MEMORY_QUARANTINED => BackendDeviceHealth::Quarantined,
            ffi::GGML_BACKEND_MEMORY_DEVICE_LOST => BackendDeviceHealth::DeviceLost,
            health => BackendDeviceHealth::Unknown(health),
        },
        last_status: BackendTerminalStatusClass::from_raw(raw.last_ggml_status),
        last_native_error: raw.last_native_error,
        quarantine_generation: raw.quarantine_generation,
        domain_kind: Some(domain_kind),
        heap_index: Some(raw.domain.heap_index),
        total_bytes,
        budget_bytes,
        stats_generation,
        quote_generation,
        claim_flags: 0,
        observation_confidence,
        device_used_bytes: device,
        device_free_bytes: device_free,
        backend_owned_live_bytes: owned(raw.backend_owned_live_bytes),
        backend_owned_cached_bytes: owned(raw.backend_owned_cached_bytes),
        backend_owned_workspace_bytes: owned(raw.backend_owned_workspace_bytes),
        backend_owned_observed_high_water_bytes: owned(
            raw.backend_owned_high_water_bytes.max(current_owned),
        ),
    }
}

pub(crate) fn record_backend_memory_probe(
    backend: ffi::GgmlBackendRaw,
    lifecycle: BackendMemoryLifecyclePoint,
) {
    let backend_identity = backend as usize;
    let result = unsafe { BackendMemoryAbi::from_backend(backend) };
    match result {
        Ok(abi) => {
            if abi.stats_at(lifecycle).is_err() {
                crate::models::native_execution_services::record_current_execution_backend_memory_unavailable(
                    backend_identity,
                    lifecycle,
                    BackendMemoryUnknownReason::StatsUnavailable,
                );
            }
        }
        Err(_) => {
            crate::models::native_execution_services::record_current_execution_backend_memory_unavailable(
                backend_identity,
                lifecycle,
                BackendMemoryUnknownReason::AbiUnavailable,
            );
        }
    }
}

/// RAII ownership of a frozen scheduler measurement. Dropping before commit
/// restores the scheduler; successful commit consumes the plan handle.
pub(crate) struct SchedulerMemoryPlan<'scheduler> {
    raw: ffi::GgmlBackendSchedMemoryPlanRaw,
    _scheduler: PhantomData<&'scheduler mut c_void>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMutationEvidence {
    ProvenUnchanged,
    MayHaveMutated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendReleaseProof {
    NotRequired,
    Proven,
    Unproven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendFailureDisposition {
    Complete,
    Cancel,
    Refund,
    Quarantine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendTerminalStage {
    BackendPrivateReserve,
    EngineOwnedCommit,
    EngineOwnedReconcile,
    EngineOwnedRelease,
    SchedulerPlanCommit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendTerminalResourceDomains {
    Exact(Vec<MemoryDomainKey>),
    Unavailable,
}

/// Failure-time evidence sampled from every concrete backend named by the
/// frozen scheduler plan. Successful receipts are retained even when another
/// backend is unavailable, but policy treats any missing backend as unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalEvidence {
    pub(crate) receipts: Vec<SafeBackendMemoryReceipt>,
    pub(crate) unavailable_backend_count: usize,
}

impl BackendTerminalEvidence {
    fn unavailable() -> Self {
        Self {
            receipts: Vec::new(),
            unavailable_backend_count: 1,
        }
    }

    fn device_health(&self) -> BackendDeviceHealth {
        let mut health = if self.unavailable_backend_count == 0 && !self.receipts.is_empty() {
            BackendDeviceHealth::Healthy
        } else {
            BackendDeviceHealth::Unavailable
        };
        for receipt in &self.receipts {
            health = strongest_device_health(health, receipt.device_health);
        }
        health
    }

    pub(crate) fn capture(
        backends: &[BackendMemoryAbi],
        mut unavailable_backend_count: usize,
    ) -> Self {
        let mut receipts = Vec::new();
        for abi in backends {
            match abi.stats_at(BackendMemoryLifecyclePoint::TerminalFailure) {
                Ok(snapshot) => {
                    let mut backend_receipts = snapshot.safe_receipts(
                        abi.provider(),
                        BackendMemoryLifecyclePoint::TerminalFailure,
                    );
                    if backend_receipts.is_empty() {
                        unavailable_backend_count += 1;
                    } else {
                        receipts.append(&mut backend_receipts);
                    }
                }
                Err(_) => unavailable_backend_count += 1,
            }
        }
        if receipts.is_empty() && unavailable_backend_count == 0 {
            unavailable_backend_count = 1;
        }
        Self {
            receipts,
            unavailable_backend_count,
        }
    }
}

fn strongest_device_health(
    left: BackendDeviceHealth,
    right: BackendDeviceHealth,
) -> BackendDeviceHealth {
    let rank = |health| match health {
        BackendDeviceHealth::Healthy => 0,
        BackendDeviceHealth::Degraded => 1,
        BackendDeviceHealth::Unavailable | BackendDeviceHealth::Unknown(_) => 2,
        BackendDeviceHealth::Quarantined => 3,
        BackendDeviceHealth::DeviceLost => 4,
    };
    if rank(right) > rank(left) {
        right
    } else {
        left
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalIdentity {
    pub(crate) provider: ExecutionProvider,
    pub(crate) stable_device_id: String,
    pub(crate) resource_domains: BackendTerminalResourceDomains,
}

impl BackendTerminalIdentity {
    pub(crate) fn exact(
        provider: ExecutionProvider,
        stable_device_id: impl Into<String>,
        resource_domains: Vec<MemoryDomainKey>,
    ) -> Self {
        Self {
            provider,
            stable_device_id: stable_device_id.into(),
            resource_domains: BackendTerminalResourceDomains::Exact(resource_domains),
        }
    }

    pub(crate) fn unavailable(provider: ExecutionProvider, stable_device_id: &str) -> Self {
        Self {
            provider,
            stable_device_id: stable_device_id.to_owned(),
            resource_domains: BackendTerminalResourceDomains::Unavailable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendTerminalOutcome {
    pub(crate) stage: BackendTerminalStage,
    pub(crate) identity: BackendTerminalIdentity,
    pub(crate) status: BackendTerminalStatusClass,
    pub(crate) mutation: BackendMutationEvidence,
    pub(crate) device_health: BackendDeviceHealth,
    pub(crate) release_proof: BackendReleaseProof,
    pub(crate) evidence: BackendTerminalEvidence,
}

impl BackendTerminalOutcome {
    pub(crate) fn backend_operation(
        stage: BackendTerminalStage,
        status: i32,
        may_have_mutated: bool,
        identity: BackendTerminalIdentity,
        evidence: BackendTerminalEvidence,
    ) -> Self {
        let status = BackendTerminalStatusClass::from_raw(status);
        let device_health = evidence.device_health();
        Self {
            stage,
            identity,
            status,
            mutation: if may_have_mutated {
                BackendMutationEvidence::MayHaveMutated
            } else {
                BackendMutationEvidence::ProvenUnchanged
            },
            device_health,
            release_proof: if may_have_mutated {
                BackendReleaseProof::Unproven
            } else {
                BackendReleaseProof::NotRequired
            },
            evidence,
        }
    }

    fn scheduler_commit(
        status: i32,
        flags: u32,
        identity: BackendTerminalIdentity,
        evidence: BackendTerminalEvidence,
    ) -> Self {
        let status = BackendTerminalStatusClass::from_raw(status);
        let device_health = evidence.device_health();
        let mutation_flag = ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED;
        let release_flag = ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN;
        let unknown_flags = flags & !(mutation_flag | release_flag);
        let may_have_mutated = flags & (mutation_flag | release_flag) != 0 || unknown_flags != 0;
        let mutation = if may_have_mutated {
            BackendMutationEvidence::MayHaveMutated
        } else {
            BackendMutationEvidence::ProvenUnchanged
        };
        let release_proof = if !may_have_mutated {
            BackendReleaseProof::NotRequired
        } else if flags & release_flag != 0 && unknown_flags == 0 {
            BackendReleaseProof::Proven
        } else {
            BackendReleaseProof::Unproven
        };
        Self {
            stage: BackendTerminalStage::SchedulerPlanCommit,
            identity,
            status,
            mutation,
            device_health,
            release_proof,
            evidence,
        }
    }

    pub(crate) fn with_release_proof(mut self, release_proof: BackendReleaseProof) -> Self {
        if self.mutation == BackendMutationEvidence::MayHaveMutated {
            self.release_proof = release_proof;
        }
        self
    }

    pub(crate) fn disposition(&self) -> BackendFailureDisposition {
        if self.status == BackendTerminalStatusClass::Success {
            return BackendFailureDisposition::Complete;
        }
        if matches!(
            self.status,
            BackendTerminalStatusClass::DeviceLost
                | BackendTerminalStatusClass::BackendPoisoned
                | BackendTerminalStatusClass::Unknown(_)
        ) {
            return BackendFailureDisposition::Quarantine;
        }
        // Cancellation remains the user-visible request result, but it cannot
        // authorize a physical refund after native state may have changed.
        // The transaction quarantines that lease first; error projection can
        // still report Canceled to the caller.
        if self.mutation == BackendMutationEvidence::MayHaveMutated
            && self.release_proof != BackendReleaseProof::Proven
        {
            return BackendFailureDisposition::Quarantine;
        }
        if self.status == BackendTerminalStatusClass::Cancelled {
            return BackendFailureDisposition::Cancel;
        }
        if matches!(
            self.device_health,
            BackendDeviceHealth::DeviceLost
                | BackendDeviceHealth::Quarantined
                | BackendDeviceHealth::Unavailable
                | BackendDeviceHealth::Unknown(_)
        ) {
            return BackendFailureDisposition::Quarantine;
        }
        BackendFailureDisposition::Refund
    }
}

#[derive(Debug, Error)]
#[error("{source} (scheduler terminal outcome={outcome:?})")]
pub(crate) struct SchedulerMemoryPlanCommitError {
    source: BackendMemoryAbiError,
    outcome: BackendTerminalOutcome,
}

impl SchedulerMemoryPlanCommitError {
    pub(crate) fn requires_quarantine(&self) -> bool {
        self.outcome.disposition() == BackendFailureDisposition::Quarantine
    }

    pub(crate) fn outcome(&self) -> BackendTerminalOutcome {
        self.outcome.clone()
    }

    pub(crate) fn into_source(self) -> BackendMemoryAbiError {
        self.source
    }
}

impl super::backend_memory_admission::NativeOwnerAttachedCommitOutcome
    for SchedulerMemoryPlanCommitError
{
    fn requires_quarantine(&self) -> bool {
        SchedulerMemoryPlanCommitError::requires_quarantine(self)
    }
}

impl<'scheduler> SchedulerMemoryPlan<'scheduler> {
    /// `scheduler`, `graph`, and every tensor reachable from `graph` must stay
    /// live and immutable until this plan is committed or dropped.
    pub(crate) unsafe fn create(
        scheduler: ffi::GgmlBackendSchedRaw,
        graph: ffi::GgmlCgraphRaw,
    ) -> Result<Self, BackendMemoryAbiError> {
        let mut raw = ptr::null_mut();
        status("scheduler_plan/create", unsafe {
            ffi::ggml_backend_sched_memory_plan_create_v1(scheduler, graph, &mut raw)
        })?;
        if raw.is_null() {
            return Err(BackendMemoryAbiError::Status {
                operation: "scheduler_plan/create",
                status: ffi::GGML_STATUS_FAILED,
            });
        }
        Ok(Self {
            raw,
            _scheduler: PhantomData,
        })
    }

    pub(crate) fn requests(
        &self,
    ) -> Result<Vec<ffi::GgmlBackendMemoryRequestV1>, BackendMemoryAbiError> {
        let count = unsafe { ffi::ggml_backend_sched_memory_plan_get_item_count_v1(self.raw) };
        let mut requests = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut item = ffi::GgmlBackendMemoryRequestV1::default();
            if !unsafe {
                ffi::ggml_backend_sched_memory_plan_get_item_v1(self.raw, index, &mut item)
            } {
                return Err(BackendMemoryAbiError::UnstableCount {
                    operation: "scheduler_plan/items",
                });
            }
            requests.push(item);
        }
        Ok(requests)
    }

    /// Partitions one multi-backend scheduler plan without altering request
    /// order inside each backend batch. Each batch must be quoted by the proc
    /// table resolved from that exact primary backend.
    pub(crate) fn requests_by_backend(
        &self,
    ) -> Result<
        Vec<(ffi::GgmlBackendRaw, Vec<ffi::GgmlBackendMemoryRequestV1>)>,
        BackendMemoryAbiError,
    > {
        let requests = self.requests()?;
        let mut batches: Vec<(ffi::GgmlBackendRaw, Vec<ffi::GgmlBackendMemoryRequestV1>)> =
            Vec::new();
        for request in requests {
            if request.backend.is_null() {
                return Err(BackendMemoryAbiError::MissingPrimaryBackend);
            }
            if let Some((_, batch)) = batches
                .iter_mut()
                .find(|(backend, _)| *backend == request.backend)
            {
                batch.push(request);
            } else {
                batches.push((request.backend, vec![request]));
            }
        }
        Ok(batches)
    }

    pub(crate) fn commit(
        mut self,
        identity: BackendTerminalIdentity,
    ) -> Result<(), SchedulerMemoryPlanCommitError> {
        let mut flags = 0_u32;
        if let Err(source) = status("scheduler_plan/commit", unsafe {
            ffi::ggml_backend_sched_memory_plan_commit_v2(self.raw, &mut flags)
        }) {
            let raw_status = match &source {
                BackendMemoryAbiError::Status { status, .. } => *status,
                _ => i32::MAX,
            };
            let evidence = self.terminal_evidence();
            return Err(SchedulerMemoryPlanCommitError {
                source,
                outcome: BackendTerminalOutcome::scheduler_commit(
                    raw_status, flags, identity, evidence,
                ),
            });
        }
        unsafe { ffi::ggml_backend_sched_memory_plan_free_v1(self.raw) };
        self.raw = ptr::null_mut();
        Ok(())
    }

    fn terminal_evidence(&self) -> BackendTerminalEvidence {
        let batches = match self.requests_by_backend() {
            Ok(batches) => batches,
            Err(_) => return BackendTerminalEvidence::unavailable(),
        };
        let mut backends = Vec::new();
        let mut unavailable_backend_count = 0;
        for (backend, _) in batches {
            match unsafe { BackendMemoryAbi::from_backend(backend) } {
                Ok(abi) => backends.push(abi),
                Err(_) => unavailable_backend_count += 1,
            }
        }
        BackendTerminalEvidence::capture(&backends, unavailable_backend_count)
    }
}

impl Drop for SchedulerMemoryPlan<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::ggml_backend_sched_memory_plan_free_v1(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    use crate::ggml_runtime::ensure_backends_loaded;

    #[test]
    fn ffi_layouts_match_the_v1_fixed_width_contract() {
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryDomainIdV1>(), 24);
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryRequestV1>(), 88);
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryClaimV1>(), 96);
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryQuoteV1>(), 48);
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryStatsV1>(), 152);
        assert_eq!(mem::size_of::<ffi::GgmlBackendMemoryApiV1>(), 64);
    }

    fn test_stats() -> ffi::GgmlBackendMemoryStatsV1 {
        ffi::GgmlBackendMemoryStatsV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendMemoryStatsV1>() as u32,
            domain: ffi::GgmlBackendMemoryDomainIdV1 {
                kind: ffi::GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL,
                heap_index: 2,
                ..Default::default()
            },
            device_used_bytes: 900,
            device_free_bytes: 100,
            backend_owned_live_bytes: 20,
            backend_owned_cached_bytes: 30,
            backend_owned_workspace_bytes: 40,
            backend_owned_high_water_bytes: 45,
            ..Default::default()
        }
    }

    #[test]
    fn safe_cuda_receipt_keeps_incomplete_owned_accounting_typed_unknown() {
        let receipt = safe_receipt(
            ExecutionProvider::Cuda,
            BackendMemoryLifecyclePoint::PostAllocationReconciliation,
            &test_stats(),
        );
        assert_eq!(receipt.device_used_bytes, BackendMemoryBytes::Known(900));
        assert_eq!(receipt.device_free_bytes, BackendMemoryBytes::Known(100));
        let unknown = BackendMemoryBytes::Unknown(
            BackendMemoryUnknownReason::ProviderOwnedAccountingIncomplete,
        );
        assert_eq!(receipt.backend_owned_live_bytes, unknown);
        assert_eq!(receipt.backend_owned_cached_bytes, unknown);
        assert_eq!(receipt.backend_owned_workspace_bytes, unknown);
        assert_eq!(receipt.backend_owned_observed_high_water_bytes, unknown);
    }

    #[test]
    fn safe_vulkan_receipt_never_presents_unreported_owned_zero_as_known() {
        let mut raw = test_stats();
        raw.backend_owned_live_bytes = 0;
        raw.backend_owned_cached_bytes = 0;
        raw.backend_owned_workspace_bytes = 0;
        raw.backend_owned_high_water_bytes = 0;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::BackendInitialized,
            &raw,
        );
        let unknown = BackendMemoryBytes::Unknown(
            BackendMemoryUnknownReason::ProviderDoesNotReportBackendOwned,
        );
        assert_eq!(receipt.backend_owned_live_bytes, unknown);
        assert_eq!(receipt.backend_owned_cached_bytes, unknown);
        assert_eq!(receipt.backend_owned_workspace_bytes, unknown);
        assert_eq!(receipt.backend_owned_observed_high_water_bytes, unknown);
        assert_eq!(receipt.device_used_bytes, BackendMemoryBytes::Known(900));
    }

    #[test]
    fn safe_receipt_keeps_unavailable_device_budget_typed_unknown() {
        let mut raw = test_stats();
        raw.flags = ffi::GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::AdmissionQuote,
            &raw,
        );
        let unknown =
            BackendMemoryBytes::Unknown(BackendMemoryUnknownReason::DeviceBudgetUnavailable);
        assert_eq!(receipt.device_used_bytes, unknown);
        assert_eq!(receipt.device_free_bytes, unknown);
    }

    fn terminal_evidence(
        provider: ExecutionProvider,
        health: BackendDeviceHealth,
        last_status: i32,
        native_error: i64,
        quarantine_generation: u64,
    ) -> BackendTerminalEvidence {
        let raw_health = match health {
            BackendDeviceHealth::Healthy => ffi::GGML_BACKEND_MEMORY_HEALTHY,
            BackendDeviceHealth::Degraded => ffi::GGML_BACKEND_MEMORY_DEGRADED,
            BackendDeviceHealth::Quarantined => ffi::GGML_BACKEND_MEMORY_QUARANTINED,
            BackendDeviceHealth::DeviceLost => ffi::GGML_BACKEND_MEMORY_DEVICE_LOST,
            BackendDeviceHealth::Unavailable | BackendDeviceHealth::Unknown(_) => u32::MAX,
        };
        let mut raw = test_stats();
        raw.health = raw_health;
        raw.last_ggml_status = last_status;
        raw.last_native_error = native_error;
        raw.quarantine_generation = quarantine_generation;
        BackendTerminalEvidence {
            receipts: vec![safe_receipt(
                provider,
                BackendMemoryLifecyclePoint::TerminalFailure,
                &raw,
            )],
            unavailable_backend_count: 0,
        }
    }

    fn healthy_terminal_evidence(provider: ExecutionProvider) -> BackendTerminalEvidence {
        terminal_evidence(
            provider,
            BackendDeviceHealth::Healthy,
            ffi::GGML_STATUS_SUCCESS,
            0,
            0,
        )
    }

    #[test]
    fn scheduler_commit_quarantines_only_unrecoverable_failures() {
        let identity =
            || BackendTerminalIdentity::unavailable(ExecutionProvider::Vulkan, "Vulkan0");
        let error = |status, flags, evidence| SchedulerMemoryPlanCommitError {
            source: BackendMemoryAbiError::Status {
                operation: "scheduler_plan/commit",
                status,
            },
            outcome: BackendTerminalOutcome::scheduler_commit(status, flags, identity(), evidence),
        };
        let healthy = || healthy_terminal_evidence(ExecutionProvider::Vulkan);
        let mutated = ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED;
        let released = mutated | ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN;

        assert!(!error(ffi::GGML_STATUS_FAILED, 0, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_ALLOC_FAILED, 0, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_EXECUTION_FAILED, 0, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_FAILED, mutated, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_ALLOC_FAILED, mutated, healthy()).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_ALLOC_FAILED, released, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_DEVICE_LOST, 0, healthy()).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_BACKEND_POISONED, 0, healthy()).requires_quarantine());
        assert!(error(i32::MAX, 0, healthy()).requires_quarantine());
        assert_eq!(
            error(
                ffi::GGML_STATUS_ABORTED,
                mutated,
                BackendTerminalEvidence::unavailable()
            )
            .outcome()
            .disposition(),
            BackendFailureDisposition::Quarantine
        );
        assert_eq!(
            error(ffi::GGML_STATUS_ABORTED, 0, healthy())
                .outcome()
                .disposition(),
            BackendFailureDisposition::Cancel
        );
        assert_eq!(
            error(ffi::GGML_STATUS_ABORTED, released, healthy())
                .outcome()
                .disposition(),
            BackendFailureDisposition::Cancel
        );

        let released_outcome = BackendTerminalOutcome::scheduler_commit(
            ffi::GGML_STATUS_ALLOC_FAILED,
            released,
            identity(),
            healthy(),
        );
        assert_eq!(
            released_outcome.disposition(),
            BackendFailureDisposition::Refund
        );
        assert_eq!(
            released_outcome.identity.provider,
            ExecutionProvider::Vulkan
        );
        assert_eq!(released_outcome.identity.stable_device_id, "Vulkan0");

        let lost_evidence = terminal_evidence(
            ExecutionProvider::Vulkan,
            BackendDeviceHealth::DeviceLost,
            ffi::GGML_STATUS_DEVICE_LOST,
            -4,
            7,
        );
        assert!(error(ffi::GGML_STATUS_ALLOC_FAILED, 0, lost_evidence).requires_quarantine());
        assert!(
            error(
                ffi::GGML_STATUS_ALLOC_FAILED,
                0,
                BackendTerminalEvidence::unavailable(),
            )
            .requires_quarantine()
        );
    }

    #[test]
    fn terminal_outcome_preserves_exact_lane_and_resource_domains() {
        let identity = BackendTerminalIdentity::exact(
            ExecutionProvider::Hip,
            "HIP0",
            vec![MemoryDomainKey::SystemMemory],
        );
        let outcome = BackendTerminalOutcome::scheduler_commit(
            ffi::GGML_STATUS_ALLOC_FAILED,
            0,
            identity,
            healthy_terminal_evidence(ExecutionProvider::Hip),
        );
        assert_eq!(outcome.stage, BackendTerminalStage::SchedulerPlanCommit);
        assert_eq!(outcome.identity.provider, ExecutionProvider::Hip);
        assert_eq!(outcome.identity.stable_device_id, "HIP0");
        assert_eq!(
            outcome.identity.resource_domains,
            BackendTerminalResourceDomains::Exact(vec![MemoryDomainKey::SystemMemory])
        );
        assert_eq!(outcome.disposition(), BackendFailureDisposition::Refund);
    }

    #[test]
    fn safe_receipt_preserves_typed_terminal_health_evidence() {
        let mut raw = test_stats();
        raw.health = ffi::GGML_BACKEND_MEMORY_DEVICE_LOST;
        raw.last_ggml_status = ffi::GGML_STATUS_DEVICE_LOST;
        raw.last_native_error = -4;
        raw.quarantine_generation = 7;
        let receipt = safe_receipt(
            ExecutionProvider::Vulkan,
            BackendMemoryLifecyclePoint::AfterGraphCompute,
            &raw,
        );
        assert_eq!(receipt.device_health, BackendDeviceHealth::DeviceLost);
        assert_eq!(receipt.last_status, BackendTerminalStatusClass::DeviceLost);
        assert_eq!(receipt.last_native_error, -4);
        assert_eq!(receipt.quarantine_generation, 7);
    }

    #[test]
    fn memory_api_resolves_from_registry_device_without_a_backend_context() {
        crate::ggml_runtime::ensure_backends_loaded();
        let cpu = crate::ggml_available_devices()
            .into_iter()
            .find(|device| device.kind == crate::GgmlBackendKind::Cpu)
            .expect("cpu device");
        let abi =
            unsafe { BackendMemoryAbi::from_device(cpu.as_ptr()) }.expect("device-only memory ABI");
        assert!(abi.backend().is_null());
        assert_eq!(abi.provider(), ExecutionProvider::Cpu);
        abi.domains()
            .expect("cpu registry device must expose memory domains");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn repeated_metal_memory_quotes_reuse_one_device_context() {
        ensure_backends_loaded();
        let backend = unsafe { ffi::ggml_backend_init_best() };
        assert!(!backend.is_null(), "macOS must expose a ggml backend");

        let name = unsafe {
            std::ffi::CStr::from_ptr(ffi::ggml_backend_name(backend))
                .to_string_lossy()
                .to_ascii_lowercase()
        };
        if !name.contains("metal") && !name.starts_with("mtl") {
            unsafe { ffi::ggml_backend_free(backend) };
            return;
        }

        let abi = unsafe { BackendMemoryAbi::from_backend(backend) }
            .expect("Metal must expose the memory ABI");
        let device = unsafe { ffi::ggml_backend_get_device(backend) };
        assert!(!device.is_null());
        let buft = unsafe { ffi::ggml_backend_dev_buffer_type(device) };
        assert!(!buft.is_null());
        let request = ffi::GgmlBackendMemoryRequestV1 {
            kind: ffi::GGML_BACKEND_MEMORY_REQUEST_BUFFER,
            usage: ffi::GGML_BACKEND_BUFFER_USAGE_COMPUTE as u32,
            request_id: 1,
            backend,
            buft,
            requested_bytes: 64 * 1024,
            ..Default::default()
        };
        let before = unsafe { ffi::openasr_ggml_metal_cached_device_count() };
        for _ in 0..128 {
            abi.quote(&[request])
                .expect("repeated Metal memory quote must remain valid");
        }
        let after = unsafe { ffi::openasr_ggml_metal_cached_device_count() };

        unsafe { ffi::ggml_backend_free(backend) };
        assert_eq!(after, before, "memory quote created Metal device contexts");
    }
}
