//! Safe ownership wrappers around ggml's optional physical-memory ABI.
//!
//! This layer deliberately does not infer policy or physical-domain aliases.
//! It preserves the backend's native UUID/heap/kind claims for the process
//! broker to map, reserve atomically, and reconcile after commit.

#![allow(dead_code)]

use std::{ffi::c_void, marker::PhantomData, mem, ptr};

use thiserror::Error;

use super::ffi;

const MEMORY_API_PROC: &[u8] = b"ggml_backend_memory_get_api_v1\0";

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
        // SAFETY: `device` came from the live backend.
        let reg = unsafe { ffi::ggml_backend_dev_backend_reg(device) };
        if reg.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: NUL-terminated static proc name and live registry.
        let proc =
            unsafe { ffi::ggml_backend_reg_get_proc_address(reg, MEMORY_API_PROC.as_ptr().cast()) };
        if proc.is_null() {
            return Err(BackendMemoryAbiError::Unavailable);
        }
        // SAFETY: the versioned proc name's C contract fixes this signature.
        let get_api: ffi::GgmlBackendMemoryGetApiV1Fn = unsafe { mem::transmute(proc) };
        // SAFETY: function pointer was resolved from the backend registry.
        let raw = unsafe { get_api() };
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

    pub(crate) fn domains(
        &self,
    ) -> Result<Vec<ffi::GgmlBackendMemoryDomainV1>, BackendMemoryAbiError> {
        let get_domains = self
            .raw
            .get_domains
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        let mut count = 0_u32;
        status("domains/count", unsafe {
            get_domains(self.device, ptr::null_mut(), &mut count)
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
            get_domains(self.device, domains.as_mut_ptr(), &mut capacity)
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
        let quote_fn = self.raw.quote.ok_or(BackendMemoryAbiError::Incompatible)?;
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
            quote_fn(
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
            quote_fn(
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
        let reserve = self
            .raw
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
            reserve(
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
            reserve(
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
        let get_stats = self
            .raw
            .get_stats
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        let mut count = 0_u32;
        status("stats/count", unsafe {
            get_stats(self.device, self.backend, ptr::null_mut(), &mut count)
        })?;
        let mut domains = initialized_stats(count as usize);
        let mut capacity = count;
        status("stats", unsafe {
            get_stats(
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

    pub(crate) fn backend(&self) -> ffi::GgmlBackendRaw {
        self.backend
    }

    pub(crate) fn trim(&self, flags: u64) -> Result<(), BackendMemoryAbiError> {
        let trim = self.raw.trim.ok_or(BackendMemoryAbiError::Incompatible)?;
        status("trim", unsafe { trim(self.backend, flags) })
    }

    pub(crate) fn quarantine(
        &self,
        request: &ffi::GgmlBackendMemoryQuarantineV1,
    ) -> Result<(), BackendMemoryAbiError> {
        let quarantine = self
            .raw
            .quarantine
            .ok_or(BackendMemoryAbiError::Incompatible)?;
        status("quarantine", unsafe { quarantine(self.backend, request) })
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
}

/// RAII ownership of a frozen scheduler measurement. Dropping before commit
/// restores the scheduler; successful commit consumes the plan handle.
pub(crate) struct SchedulerMemoryPlan<'scheduler> {
    raw: ffi::GgmlBackendSchedMemoryPlanRaw,
    _scheduler: PhantomData<&'scheduler mut c_void>,
}

#[derive(Debug, Error)]
#[error("{source} (scheduler native allocation may_have_mutated={may_have_mutated})")]
pub(crate) struct SchedulerMemoryPlanCommitError {
    source: BackendMemoryAbiError,
    may_have_mutated: bool,
}

impl SchedulerMemoryPlanCommitError {
    pub(crate) fn requires_quarantine(&self) -> bool {
        if self.may_have_mutated {
            return true;
        }
        match &self.source {
            BackendMemoryAbiError::Status { status, .. } => {
                *status != ffi::GGML_STATUS_FAILED && *status != ffi::GGML_STATUS_ALLOC_FAILED
            }
            _ => true,
        }
    }

    pub(crate) fn into_source(self) -> BackendMemoryAbiError {
        self.source
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

    pub(crate) fn commit(mut self) -> Result<(), SchedulerMemoryPlanCommitError> {
        let mut flags = 0_u32;
        if let Err(source) = status("scheduler_plan/commit", unsafe {
            ffi::ggml_backend_sched_memory_plan_commit_v2(self.raw, &mut flags)
        }) {
            let known_mutation =
                flags & ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED != 0;
            let unknown_flags =
                flags & !ffi::GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED;
            return Err(SchedulerMemoryPlanCommitError {
                source,
                // Unknown future flags are conservative: only an exact zero
                // proves that native scheduler allocation did not change.
                may_have_mutated: known_mutation || unknown_flags != 0,
            });
        }
        unsafe { ffi::ggml_backend_sched_memory_plan_free_v1(self.raw) };
        self.raw = ptr::null_mut();
        Ok(())
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

    #[test]
    fn scheduler_commit_quarantines_only_unrecoverable_failures() {
        let error = |status, may_have_mutated| SchedulerMemoryPlanCommitError {
            source: BackendMemoryAbiError::Status {
                operation: "scheduler_plan/commit",
                status,
            },
            may_have_mutated,
        };

        assert!(!error(ffi::GGML_STATUS_FAILED, false).requires_quarantine());
        assert!(!error(ffi::GGML_STATUS_ALLOC_FAILED, false).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_FAILED, true).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_DEVICE_LOST, false).requires_quarantine());
        assert!(error(ffi::GGML_STATUS_BACKEND_POISONED, false).requires_quarantine());
        assert!(error(i32::MAX, false).requires_quarantine());
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
