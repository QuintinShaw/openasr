use std::ffi::{c_char, c_int, c_void};

pub(crate) type GgmlBackendRaw = *mut c_void;
pub(crate) type GgmlBackendDevRaw = *mut c_void;
pub(crate) type GgmlBackendBufferRaw = *mut c_void;
pub(crate) type GgmlBackendBufferTypeRaw = *mut c_void;
pub(crate) type GgmlBackendSchedRaw = *mut c_void;
pub(crate) type GgmlBackendSchedMemoryPlanRaw = *mut c_void;
pub(crate) type GgmlGallocrRaw = *mut c_void;
pub(crate) type GgmlBackendRegRaw = *mut c_void;
pub(crate) type GgmlContextRaw = *mut c_void;
pub(crate) type GgmlTensorRaw = *mut c_void;
pub(crate) type GgmlCgraphRaw = *mut c_void;
pub(crate) type GgufContextRaw = *mut c_void;
pub(crate) const GGML_MAX_DIMS: usize = 4;
pub(crate) const GGML_PREC_DEFAULT: c_int = 0;
pub(crate) const GGML_PREC_F32: c_int = 10;
pub(crate) const GGML_OP_POOL_MAX: c_int = 0;

pub(crate) type GgmlToFloatFn = unsafe extern "C" fn(x: *const c_void, y: *mut f32, k: i64);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTensorLayoutPrefix {
    pub type_: c_int,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
}

pub(crate) const GGML_MAX_SRC: usize = 10;
pub(crate) const GGML_MAX_OP_PARAMS_I32: usize = 16;

/// Mirrors `struct ggml_tensor` (ggml.h) far enough to reach the `view_src`
/// and `data` fields the CPU step-buffer grow-to-fit reuse needs to inspect
/// (see `bind_unallocated_context_tensors` in cpu_graph.rs), extending the
/// `type_`/`buffer`/`ne`/`nb` prefix `GgmlTensorLayoutPrefix` already reads.
/// `#[repr(C)]` with matching field types/order reproduces the C compiler's
/// layout, so this must be kept in lockstep with the vendored ggml.h if the
/// submodule pin ever changes tensor struct fields ahead of `data`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTensorAllocPrefix {
    pub type_: c_int,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: c_int,
    pub op_params: [i32; GGML_MAX_OP_PARAMS_I32],
    pub flags: i32,
    pub src: [*mut c_void; GGML_MAX_SRC],
    pub view_src: *mut c_void,
    pub view_offs: usize,
    pub data: *mut c_void,
}

/// Mirrors `struct ggml_tallocr` (ggml-alloc.h), a stable public ggml struct
/// (not an internal implementation detail) exposing the bump-allocator ggml
/// itself uses to bind tensors into an already-allocated backend buffer.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTallocr {
    pub buffer: GgmlBackendBufferRaw,
    pub base: *mut c_void,
    pub alignment: usize,
    pub offset: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgufInitParams {
    pub no_alloc: bool,
    pub ctx: *mut *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgufParseLimits {
    pub max_tensors: u64,
    pub max_kv: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_header_bytes: u64,
}

pub(crate) const GGUF_PARSE_ERROR_NONE: c_int = 0;
pub(crate) const GGUF_PARSE_ERROR_INVALID_DATA: c_int = 1;
pub(crate) const GGUF_PARSE_ERROR_ALLOCATION: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

pub(crate) const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_ACCEL: c_int = 3;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_META: c_int = 4;

pub(crate) const GGML_STATUS_ALLOC_FAILED: c_int = -2;
pub(crate) const GGML_STATUS_FAILED: c_int = -1;
pub(crate) const GGML_STATUS_SUCCESS: c_int = 0;
/// Mirrors `enum ggml_status` in ggml.h. `ggml_backend_graph_compute` and
/// `ggml_backend_sched_graph_compute` return the merged submit + completion
/// status, so this is the single terminal outcome for a compute call -- no
/// separate synchronize step is needed to observe a completion-phase failure.
/// `GGML_STATUS_ABORTED` is the cooperative cancellation result; Rust maps it
/// to the typed per-job cancellation path rather than a generic compute
/// failure. All other non-success values below are terminal failures.
pub(crate) const GGML_STATUS_ABORTED: c_int = 1;
pub(crate) const GGML_STATUS_EXECUTION_FAILED: c_int = 2;
pub(crate) const GGML_STATUS_DEVICE_LOST: c_int = 3;
pub(crate) const GGML_STATUS_BACKEND_POISONED: c_int = 4;
pub(crate) const GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN: u32 = 1 << 1;
pub(crate) const GGML_GALLOCR_MEASURE_COMMIT_MAY_HAVE_MUTATED: u32 = 1 << 0;
pub(crate) const GGML_GALLOCR_MEASURE_COMMIT_RELEASE_UNPROVEN: u32 = 1 << 1;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_DISABLED: c_int = 0;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_NATIVE: c_int = 1;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_SEGMENTED: c_int = 2;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE: c_int = 0;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT: c_int = 1;
pub(crate) const GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GgmlBackendGraphCancelCapability {
    pub mechanism: c_int,
    pub observation_granularity: c_int,
}

impl Default for GgmlBackendGraphCancelCapability {
    fn default() -> Self {
        Self {
            mechanism: GGML_BACKEND_GRAPH_CANCEL_DISABLED,
            observation_granularity: GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE,
        }
    }
}
pub(crate) const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub(crate) const GGML_BACKEND_BUFFER_USAGE_COMPUTE: c_int = 2;

pub(crate) const GGML_BACKEND_MEMORY_ABI_V1: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED: u32 = 3;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_UNIFIED: u32 = 4;
pub(crate) const GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED: u32 = 5;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_BUFFER: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE: u32 = 3;
pub(crate) const GGML_BACKEND_MEMORY_REQUEST_TRANSFER: u32 = 4;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY: u32 = 1 << 1;
pub(crate) const GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM: u32 =
    1 << 2;
#[cfg(test)]
pub(crate) const GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_EXACT: u32 = 1 << 0;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER: u32 = 1 << 1;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_DRIVER_ESTIMATE: u32 = 1 << 2;
#[cfg(test)]
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED: u32 = 1 << 5;
pub(crate) const GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL: u32 = 1 << 6;
pub(crate) const GGML_BACKEND_MEMORY_HEALTHY: u32 = 0;
pub(crate) const GGML_BACKEND_MEMORY_DEGRADED: u32 = 1;
pub(crate) const GGML_BACKEND_MEMORY_QUARANTINED: u32 = 2;
pub(crate) const GGML_BACKEND_MEMORY_DEVICE_LOST: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct GgmlBackendMemoryDomainIdV1 {
    pub physical_device_uuid: [u8; 16],
    pub heap_index: u32,
    pub kind: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GgmlBackendMemoryDomainV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub id: GgmlBackendMemoryDomainIdV1,
    pub name: [c_char; 48],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlBackendMemoryRequestV1 {
    pub struct_size: u32,
    pub kind: u32,
    pub flags: u32,
    pub usage: u32,
    pub request_id: u64,
    pub backend: GgmlBackendRaw,
    pub peer_backend: GgmlBackendRaw,
    pub buft: GgmlBackendBufferTypeRaw,
    pub graph: GgmlCgraphRaw,
    pub host_ptr: *const c_void,
    pub requested_bytes: u64,
    pub currently_allocated_bytes: u64,
    pub max_tensor_bytes: u64,
}

impl Default for GgmlBackendMemoryRequestV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            kind: 0,
            flags: 0,
            usage: 0,
            request_id: 0,
            backend: std::ptr::null_mut(),
            peer_backend: std::ptr::null_mut(),
            buft: std::ptr::null_mut(),
            graph: std::ptr::null_mut(),
            host_ptr: std::ptr::null(),
            requested_bytes: 0,
            currently_allocated_bytes: 0,
            max_tensor_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryClaimV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub request_id: u64,
    pub domain: GgmlBackendMemoryDomainIdV1,
    pub payload_requested_bytes: u64,
    pub committed_before_bytes: u64,
    pub committed_after_upper_bytes: u64,
    pub commit_peak_extra_upper_bytes: u64,
    pub resident_after_upper_bytes: u64,
    pub retained_after_use_upper_bytes: u64,
    pub releasable_after_use_upper_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryQuoteV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub residual_flags: u32,
    pub residual_request_count: u32,
    pub provisional_requested_upper_bytes: u64,
    pub stats_generation: u64,
    pub quote_token: u64,
    pub request_fingerprint: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct GgmlBackendMemoryStatsV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub domain: GgmlBackendMemoryDomainIdV1,
    pub generation: u64,
    pub timestamp_monotonic_ns: u64,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub device_used_bytes: u64,
    pub device_free_bytes: u64,
    pub backend_owned_live_bytes: u64,
    pub backend_owned_cached_bytes: u64,
    pub backend_owned_workspace_bytes: u64,
    pub backend_owned_high_water_bytes: u64,
    pub allocation_count: u64,
    pub allocation_failure_count: u64,
    pub health: u32,
    pub last_ggml_status: i32,
    pub last_native_error: i64,
    pub quarantine_generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GgmlBackendMemoryQuarantineV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub reason: u32,
    pub ggml_status: i32,
    pub native_error: i64,
    pub message: [c_char; 96],
}

pub(crate) type GgmlMemoryGetDomainsFn =
    unsafe extern "C" fn(GgmlBackendDevRaw, *mut GgmlBackendMemoryDomainV1, *mut u32) -> c_int;
pub(crate) type GgmlMemoryQuoteFn = unsafe extern "C" fn(
    *const GgmlBackendMemoryRequestV1,
    u32,
    *mut GgmlBackendMemoryQuoteV1,
    *mut GgmlBackendMemoryClaimV1,
    *mut u32,
) -> c_int;
pub(crate) type GgmlMemoryReservePrivateFn = unsafe extern "C" fn(
    *const GgmlBackendMemoryRequestV1,
    u32,
    *const GgmlBackendMemoryQuoteV1,
    *mut GgmlBackendMemoryClaimV1,
    *mut u32,
) -> c_int;
pub(crate) type GgmlMemoryGetStatsFn = unsafe extern "C" fn(
    GgmlBackendDevRaw,
    GgmlBackendRaw,
    *mut GgmlBackendMemoryStatsV1,
    *mut u32,
) -> c_int;
pub(crate) type GgmlMemoryTrimFn = unsafe extern "C" fn(GgmlBackendRaw, u64) -> c_int;
pub(crate) type GgmlMemoryQuarantineFn =
    unsafe extern "C" fn(GgmlBackendRaw, *const GgmlBackendMemoryQuarantineV1) -> c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlBackendMemoryApiV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub capabilities: u64,
    pub get_domains: Option<GgmlMemoryGetDomainsFn>,
    pub quote: Option<GgmlMemoryQuoteFn>,
    pub reserve_private: Option<GgmlMemoryReservePrivateFn>,
    pub get_stats: Option<GgmlMemoryGetStatsFn>,
    pub trim: Option<GgmlMemoryTrimFn>,
    pub quarantine: Option<GgmlMemoryQuarantineFn>,
}

/// ggml abort predicate: return true to abort the in-flight graph. Called from
/// ggml worker threads -- must stay panic-free and lock-light.
pub(crate) type GgmlAbortCallback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

pub(crate) const GGML_TYPE_F32: c_int = 0;
pub(crate) const GGML_TYPE_F16: c_int = 1;
pub(crate) const GGML_TYPE_Q4_0: c_int = 2;
pub(crate) const GGML_TYPE_Q8_0: c_int = 8;
pub(crate) const GGML_TYPE_Q3_K: c_int = 11;
pub(crate) const GGML_TYPE_Q4_K: c_int = 12;
pub(crate) const GGML_TYPE_Q5_K: c_int = 13;
pub(crate) const GGML_TYPE_Q6_K: c_int = 14;
pub(crate) const GGML_TYPE_I32: c_int = 26;

pub(crate) const GGML_LSTM_GATE_ORDER_IOFC: c_int = 0;
pub(crate) const GGML_LSTM_GATE_ORDER_IFGO: c_int = 1;

#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NEOX: c_int = 2;

/// GPT-J / interleaved RoPE layout (rotates adjacent pairs x[2i], x[2i+1]).
/// Matches HuggingFace `repeat_interleave(2)` rotary embedding (e.g. Moonshine).
#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NORMAL: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTypeTraits {
    pub type_name: *const c_char,
    pub blck_size: i64,
    pub blck_size_interleave: i64,
    pub type_size: usize,
    pub is_quantized: bool,
    pub to_float: Option<GgmlToFloatFn>,
    pub from_float_ref: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GgmlBackendDevCaps {
    pub async_: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlBackendDevProps {
    pub name: *const c_char,
    pub description: *const c_char,
    pub memory_free: usize,
    pub memory_total: usize,
    pub type_: c_int,
    pub device_id: *const c_char,
    pub caps: GgmlBackendDevCaps,
}

pub(crate) const GGUF_TYPE_UINT32: c_int = 4;
pub(crate) const GGUF_TYPE_FLOAT32: c_int = 6;
pub(crate) const GGUF_TYPE_BOOL: c_int = 7;
pub(crate) const GGUF_TYPE_STRING: c_int = 8;
pub(crate) const GGUF_TYPE_ARRAY: c_int = 9;
pub(crate) const GGUF_TYPE_UINT64: c_int = 10;

unsafe extern "C" {
    pub(crate) fn ggml_backend_name(backend: GgmlBackendRaw) -> *const c_char;
    pub(crate) fn ggml_backend_free(backend: GgmlBackendRaw);
    pub(crate) fn ggml_backend_free_status(backend: GgmlBackendRaw) -> c_int;
    pub(crate) fn ggml_backend_init_best() -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_init_by_type(
        dev_type: c_int,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    // GGML_BACKEND_DL: register backend plugin DLLs (ggml-<name>-*.dll / .so).
    // Must run before any registry query under DL, where the static GGML_USE_*
    // backend registration is compiled out (empty registry, init_best returns
    // null, otherwise). The verified bundled loader scans only the explicit
    // application directory and rejects modules before init when the OpenASR
    // ABI attestation does not match. Optional packs use the exact verified
    // path loader below; scanning every installed version is not a safe
    // activation policy.
    pub(crate) fn ggml_backend_load_verified_v3_utf8(
        path_utf8: *const c_char,
        dependency_dirs_utf8: *const *const c_char,
        dependency_dir_count: usize,
        expected_openasr_abi_v1: *const c_char,
        expected_provider_v1: *const c_char,
        expected_device_target: *const c_char,
        minimum_driver_version: *const c_char,
    ) -> GgmlBackendRegRaw;
    pub(crate) fn ggml_backend_probe_verified_v3_utf8(
        path_utf8: *const c_char,
        dependency_dirs_utf8: *const *const c_char,
        dependency_dir_count: usize,
        expected_openasr_abi_v1: *const c_char,
        expected_provider_v1: *const c_char,
        expected_device_target: *const c_char,
        minimum_driver_version: *const c_char,
        driver_out: *mut c_char,
        driver_out_capacity: usize,
    ) -> bool;
    pub(crate) fn ggml_backend_load_best_verified_utf8(
        paths_utf8: *const *const c_char,
        path_count: usize,
        expected_openasr_abi_v1: *const c_char,
        expected_provider_v1: *const c_char,
    ) -> GgmlBackendRegRaw;
    pub(crate) fn ggml_backend_unload(reg: GgmlBackendRegRaw);
    pub(crate) fn ggml_backend_reg_name(reg: GgmlBackendRegRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_pci_vendor_id(device: GgmlBackendDevRaw) -> u32;
    pub(crate) fn ggml_backend_set_n_threads_if_supported(
        backend: GgmlBackendRaw,
        n_threads: c_int,
    ) -> c_int;
    pub(crate) fn ggml_backend_buffer_free_status(buffer: GgmlBackendBufferRaw) -> c_int;
    pub(crate) fn ggml_backend_buffer_is_host(buffer: GgmlBackendBufferRaw) -> bool;
    pub(crate) fn ggml_backend_buffer_set_usage(buffer: GgmlBackendBufferRaw, usage: c_int);
    // Keep raw graph execution inside ggml_runtime. Model families must use
    // GgmlCpuGraphRunner so request cancellation and typed terminal-status
    // mapping cannot be bypassed as new executors are added.
    pub(super) fn ggml_backend_graph_compute(
        backend: GgmlBackendRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(super) fn ggml_backend_graph_compute_with_abort(
        backend: GgmlBackendRaw,
        cgraph: GgmlCgraphRaw,
        abort_callback: GgmlAbortCallback,
        abort_callback_data: *mut c_void,
        cancel_capability: *mut GgmlBackendGraphCancelCapability,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_new(
        backends: *mut GgmlBackendRaw,
        bufts: *mut GgmlBackendBufferTypeRaw,
        n_backends: c_int,
        graph_size: usize,
        parallel: bool,
        op_offload: bool,
    ) -> GgmlBackendSchedRaw;
    pub(crate) fn ggml_backend_sched_free_status(sched: GgmlBackendSchedRaw) -> c_int;
    pub(crate) fn ggml_backend_sched_reset(sched: GgmlBackendSchedRaw);
    pub(crate) fn ggml_backend_sched_get_tensor_backend(
        sched: GgmlBackendSchedRaw,
        node: GgmlTensorRaw,
    ) -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_sched_memory_plan_create_v1(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
        out_plan: *mut GgmlBackendSchedMemoryPlanRaw,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_memory_plan_get_item_count_v1(
        plan: GgmlBackendSchedMemoryPlanRaw,
    ) -> u32;
    pub(crate) fn ggml_backend_sched_memory_plan_get_item_v1(
        plan: GgmlBackendSchedMemoryPlanRaw,
        index: u32,
        out_item: *mut GgmlBackendMemoryRequestV1,
    ) -> bool;
    pub(crate) fn ggml_backend_sched_memory_plan_commit_v2(
        plan: GgmlBackendSchedMemoryPlanRaw,
        out_flags: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_memory_plan_free_v1(plan: GgmlBackendSchedMemoryPlanRaw);
    pub(crate) fn ggml_backend_memory_api_for_backend_v1(
        backend: GgmlBackendRaw,
    ) -> *const GgmlBackendMemoryApiV1;
    pub(crate) fn ggml_backend_memory_api_get_domains_v1(
        api: *const GgmlBackendMemoryApiV1,
        dev: GgmlBackendDevRaw,
        domains: *mut GgmlBackendMemoryDomainV1,
        inout_count: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_backend_memory_api_quote_v1(
        api: *const GgmlBackendMemoryApiV1,
        requests: *const GgmlBackendMemoryRequestV1,
        request_count: u32,
        quote: *mut GgmlBackendMemoryQuoteV1,
        claims: *mut GgmlBackendMemoryClaimV1,
        inout_claim_count: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_backend_memory_api_reserve_private_v1(
        api: *const GgmlBackendMemoryApiV1,
        requests: *const GgmlBackendMemoryRequestV1,
        request_count: u32,
        quote: *const GgmlBackendMemoryQuoteV1,
        actual: *mut GgmlBackendMemoryClaimV1,
        inout_actual_count: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_backend_memory_api_get_stats_v1(
        api: *const GgmlBackendMemoryApiV1,
        dev: GgmlBackendDevRaw,
        backend: GgmlBackendRaw,
        stats: *mut GgmlBackendMemoryStatsV1,
        inout_count: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_backend_memory_api_trim_v1(
        api: *const GgmlBackendMemoryApiV1,
        backend: GgmlBackendRaw,
        flags: u64,
    ) -> c_int;
    pub(crate) fn ggml_backend_memory_api_quarantine_v1(
        api: *const GgmlBackendMemoryApiV1,
        backend: GgmlBackendRaw,
        request: *const GgmlBackendMemoryQuarantineV1,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_graph_compute(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(super) fn ggml_backend_sched_graph_compute_with_abort(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
        abort_callback: GgmlAbortCallback,
        abort_callback_data: *mut c_void,
        cancel_capability: *mut GgmlBackendGraphCancelCapability,
    ) -> c_int;
    pub(crate) fn ggml_backend_tensor_set(
        tensor: GgmlTensorRaw,
        data: *const c_void,
        offset: usize,
        size: usize,
    ) -> c_int;
    pub(crate) fn ggml_backend_tensor_get(
        tensor: GgmlTensorRaw,
        data: *mut c_void,
        offset: usize,
        size: usize,
    ) -> c_int;
    pub(crate) fn ggml_backend_tensor_alloc(
        buffer: GgmlBackendBufferRaw,
        tensor: GgmlTensorRaw,
        addr: *mut c_void,
    ) -> c_int;
    pub(crate) fn ggml_backend_alloc_ctx_tensors(
        ctx: GgmlContextRaw,
        backend: GgmlBackendRaw,
    ) -> GgmlBackendBufferRaw;
    // Read-only sizing query for the CPU step-buffer pool's grow-to-fit check:
    // computes what `ggml_backend_alloc_ctx_tensors_from_buft` would allocate
    // for `ctx`'s currently-unallocated tensors without allocating anything.
    pub(crate) fn ggml_backend_alloc_ctx_tensors_from_buft_size(
        ctx: GgmlContextRaw,
        buft: GgmlBackendBufferTypeRaw,
    ) -> usize;
    pub(crate) fn ggml_backend_get_default_buffer_type(
        backend: GgmlBackendRaw,
    ) -> GgmlBackendBufferTypeRaw;
    pub(crate) fn ggml_backend_buft_alloc_buffer(
        buft: GgmlBackendBufferTypeRaw,
        size: usize,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_buffer_get_size(buffer: GgmlBackendBufferRaw) -> usize;
    pub(crate) fn ggml_gallocr_new(buft: GgmlBackendBufferTypeRaw) -> GgmlGallocrRaw;
    pub(crate) fn ggml_gallocr_free_status(galloc: GgmlGallocrRaw) -> c_int;
    pub(crate) fn ggml_gallocr_measure_n_v1(
        galloc: GgmlGallocrRaw,
        graph: GgmlCgraphRaw,
        node_buffer_ids: *const c_int,
        leaf_buffer_ids: *const c_int,
    ) -> bool;
    pub(crate) fn ggml_gallocr_measure_get_chunk_count_v1(galloc: GgmlGallocrRaw) -> u32;
    pub(crate) fn ggml_gallocr_measure_get_chunk_v1(
        galloc: GgmlGallocrRaw,
        index: u32,
        buft: *mut GgmlBackendBufferTypeRaw,
        requested_bytes: *mut u64,
        currently_allocated_bytes: *mut u64,
    ) -> bool;
    pub(crate) fn ggml_gallocr_measure_commit_v2(
        galloc: GgmlGallocrRaw,
        out_flags: *mut u32,
    ) -> c_int;
    pub(crate) fn ggml_gallocr_alloc_graph_v2(
        galloc: GgmlGallocrRaw,
        graph: GgmlCgraphRaw,
    ) -> c_int;
    // Tensor allocator (ggml-alloc.h): binds tensors into an already-allocated
    // buffer -- the primitive `ggml_backend_alloc_ctx_tensors` itself uses,
    // exposed here to bind the CPU step pool's *reused* buffer without
    // allocating a fresh one every step.
    pub(crate) fn ggml_tallocr_new(buffer: GgmlBackendBufferRaw) -> GgmlTallocr;
    pub(crate) fn ggml_tallocr_alloc(talloc: *mut GgmlTallocr, tensor: GgmlTensorRaw) -> c_int;
    pub(crate) fn ggml_backend_view_init(tensor: GgmlTensorRaw) -> c_int;
    pub(crate) fn ggml_backend_get_device(backend: GgmlBackendRaw) -> GgmlBackendDevRaw;

    pub(crate) fn ggml_backend_dev_count() -> usize;
    pub(crate) fn ggml_backend_dev_get(index: usize) -> GgmlBackendDevRaw;
    pub(crate) fn ggml_backend_dev_name(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_description(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_type(device: GgmlBackendDevRaw) -> c_int;
    pub(crate) fn ggml_backend_dev_memory(
        device: GgmlBackendDevRaw,
        free: *mut usize,
        total: *mut usize,
    );
    pub(crate) fn ggml_backend_dev_get_props(
        device: GgmlBackendDevRaw,
        props: *mut GgmlBackendDevProps,
    );
    pub(crate) fn ggml_backend_dev_buffer_type(
        device: GgmlBackendDevRaw,
    ) -> GgmlBackendBufferTypeRaw;
    pub(crate) fn ggml_backend_buft_get_alignment(buft: GgmlBackendBufferTypeRaw) -> usize;
    pub(crate) fn ggml_backend_dev_init(
        device: GgmlBackendDevRaw,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_dev_buffer_from_host_ptr(
        device: GgmlBackendDevRaw,
        ptr: *mut c_void,
        size: usize,
        max_tensor_size: usize,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_dev_supports_op(
        device: GgmlBackendDevRaw,
        op: GgmlTensorRaw,
    ) -> bool;

    // The host sets CPU threads through the registry proc-address table
    // (`backend_set_n_threads`), which works under GGML_BACKEND_DL where the
    // ggml-cpu plugin's symbols are not linked into the core. The macOS
    // BLAS-accelerator path calls ggml_backend_blas_set_n_threads directly.
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_init() -> GgmlBackendRaw;
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_set_n_threads(backend: GgmlBackendRaw, n_threads: c_int);
    #[cfg(all(target_os = "macos", test))]
    pub(crate) fn openasr_ggml_metal_cached_device_count() -> usize;
    pub(crate) fn ggml_init(params: GgmlInitParams) -> GgmlContextRaw;
    pub(crate) fn ggml_reset(ctx: GgmlContextRaw);
    pub(crate) fn ggml_free(ctx: GgmlContextRaw);
    pub(crate) fn ggml_blck_size(type_: c_int) -> i64;
    pub(crate) fn ggml_type_size(type_: c_int) -> usize;
    pub(crate) fn ggml_row_size(type_: c_int, ne: i64) -> usize;
    #[allow(dead_code)]
    pub(crate) fn ggml_is_quantized(type_: c_int) -> bool;
    pub(crate) fn ggml_get_type_traits(type_: c_int) -> *const GgmlTypeTraits;
    pub(crate) fn ggml_quantize_chunk(
        type_: c_int,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
    pub(crate) fn ggml_get_data(tensor: GgmlTensorRaw) -> *mut c_void;
    pub(crate) fn ggml_get_first_tensor(ctx: GgmlContextRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_get_next_tensor(ctx: GgmlContextRaw, tensor: GgmlTensorRaw)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_get_name(tensor: GgmlTensorRaw) -> *const c_char;
    pub(crate) fn ggml_op_desc(tensor: GgmlTensorRaw) -> *const c_char;
    pub(crate) fn ggml_nbytes(tensor: GgmlTensorRaw) -> usize;
    pub(crate) fn ggml_new_tensor_1d(ctx: GgmlContextRaw, type_: c_int, ne0: i64) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_2d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_3d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_4d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_set_name(tensor: GgmlTensorRaw, name: *const c_char) -> GgmlTensorRaw;
    pub(crate) fn ggml_add(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sub(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_div(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqr(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqrt(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_abs(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_log(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sin(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_cos(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_leaky_relu(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        negative_slope: f32,
        inplace: bool,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul_mat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul_mat_set_prec(a: GgmlTensorRaw, prec: c_int);
    pub(crate) fn ggml_get_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)] // last-max binding; unused until a native last-max lane is authorized
    pub(crate) fn ggml_argmax(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_argmax_first(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[cfg(test)]
    pub(crate) fn ggml_top_k(ctx: GgmlContextRaw, a: GgmlTensorRaw, k: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_scale(ctx: GgmlContextRaw, a: GgmlTensorRaw, s: f32) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum_rows(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_mean(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    // group normalize along ne0*ne1*n_groups (ggml.h:1382). wav2vec2 base uses
    // feat_extract_norm=="group" with n_groups == n_channels (per-channel norm).
    pub(crate) fn ggml_group_norm(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        n_groups: c_int,
        eps: f32,
    ) -> GgmlTensorRaw;
    // concat a and b along `dim` (ggml.h:1084). Used to stitch per-group conv_1d
    // outputs back into one [out_channels, T] tensor for the grouped pos-conv.
    pub(crate) fn ggml_concat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        dim: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rms_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_flash_attn_ext(
        ctx: GgmlContextRaw,
        q: GgmlTensorRaw,
        k: GgmlTensorRaw,
        v: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_flash_attn_ext_set_prec(a: GgmlTensorRaw, prec: c_int);
    /// OpenASR-local CPU fused Transformer-XL relative-position attention.
    /// Non-CPU backends do not implement this op.
    pub(crate) fn ggml_flash_attn_rel_pos(
        ctx: GgmlContextRaw,
        q_u: GgmlTensorRaw,
        q_v: GgmlTensorRaw,
        k: GgmlTensorRaw,
        r: GgmlTensorRaw,
        v: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_ssm_conv(
        ctx: GgmlContextRaw,
        sx: GgmlTensorRaw,
        c: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_lstm_seq(
        ctx: GgmlContextRaw,
        x: GgmlTensorRaw,
        w: GgmlTensorRaw,
        r: GgmlTensorRaw,
        b: GgmlTensorRaw,
        gate_order: c_int,
        reverse: bool,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_can_repeat(a: GgmlTensorRaw, b: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_gelu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_gelu_erf(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_tanh(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_relu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sigmoid(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_softplus(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_swoosh(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        offset: f32,
        shift: f32,
        linear_scale: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_exp(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_silu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_cast(ctx: GgmlContextRaw, a: GgmlTensorRaw, type_: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_cont(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_nelements(tensor: GgmlTensorRaw) -> i64;
    pub(crate) fn ggml_is_transposed(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_is_contiguous(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_is_contiguous_rows(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_reshape_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_1d(ctx: GgmlContextRaw, a: GgmlTensorRaw, ne0: i64)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_view_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_cpy(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_set_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_transpose(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_permute(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        p0: c_int,
        d0: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_pool_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        op: c_int,
        k0: c_int,
        s0: c_int,
        p0: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_direct(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_dw_direct(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        stride0: c_int,
        stride1: c_int,
        pad0: c_int,
        pad1: c_int,
        dilation0: c_int,
        dilation1: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rope_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_graph_custom(
        ctx: GgmlContextRaw,
        size: usize,
        grads: bool,
    ) -> GgmlCgraphRaw;
    pub(crate) fn ggml_build_forward_expand(cgraph: GgmlCgraphRaw, tensor: GgmlTensorRaw);
    pub(crate) fn ggml_graph_n_nodes(cgraph: GgmlCgraphRaw) -> c_int;
    pub(crate) fn ggml_graph_node(cgraph: GgmlCgraphRaw, index: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_set_input(tensor: GgmlTensorRaw);
    pub(crate) fn ggml_set_output(tensor: GgmlTensorRaw);
    // Sizing helpers for no_alloc metadata contexts. `ggml_tensor_overhead`
    // returns the per-tensor bookkeeping cost (ggml_object + ggml_tensor) and
    // `ggml_graph_overhead_custom` the bytes a cgraph of `size` nodes consumes
    // inside its context — together they give the EXACT capacity a metadata-only
    // context needs, mirroring llama.cpp's compute-meta buffer sizing.
    pub(crate) fn ggml_tensor_overhead() -> usize;
    pub(crate) fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    pub(crate) fn ggml_used_mem(ctx: GgmlContextRaw) -> usize;
    pub(crate) fn ggml_get_mem_size(ctx: GgmlContextRaw) -> usize;
    // ggml_cpu_has_* / ggml_cpu_get_* are not declared here: under
    // GGML_BACKEND_DL they live in the loaded ggml-cpu plugin, not the linked
    // core. GgmlCpuFeatures::detect() reads CPU features via the Rust stdlib
    // instead (build-mode-agnostic).

    pub(crate) fn gguf_init_from_buffer_with_limits(
        data: *const c_void,
        size: usize,
        params: GgufInitParams,
        limits: GgufParseLimits,
        error: *mut c_int,
    ) -> GgufContextRaw;
    pub(crate) fn gguf_bounded_parser_structural_bytes(
        n_kv: u64,
        n_tensors: u64,
        result: *mut usize,
    ) -> bool;
    pub(crate) fn gguf_bounded_parser_payload_wire_multiplier() -> usize;
    pub(crate) fn gguf_free(ctx: GgufContextRaw);
    pub(crate) fn gguf_get_n_kv(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_key(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_kv_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_arr_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_val_u32(ctx: *const c_void, key_id: i64) -> u32;
    pub(crate) fn gguf_get_val_u64(ctx: *const c_void, key_id: i64) -> u64;
    pub(crate) fn gguf_get_val_f32(ctx: *const c_void, key_id: i64) -> f32;
    pub(crate) fn gguf_get_val_bool(ctx: *const c_void, key_id: i64) -> bool;
    pub(crate) fn gguf_get_val_str(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_arr_n(ctx: *const c_void, key_id: i64) -> usize;
    pub(crate) fn gguf_get_arr_data(ctx: *const c_void, key_id: i64) -> *const c_void;
    pub(crate) fn gguf_get_arr_str(ctx: *const c_void, key_id: i64, i: usize) -> *const c_char;
    pub(crate) fn gguf_get_data_offset(ctx: *const c_void) -> usize;
    pub(crate) fn gguf_get_n_tensors(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_tensor_offset(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn gguf_get_tensor_name(ctx: *const c_void, tensor_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_tensor_ne(ctx: *const c_void, tensor_id: i64) -> *const i64;
    pub(crate) fn gguf_get_tensor_type(ctx: *const c_void, tensor_id: i64) -> c_int;
    pub(crate) fn gguf_get_tensor_size(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn ggml_type_name(type_: c_int) -> *const c_char;

    pub(crate) fn gguf_init_empty() -> GgufContextRaw;
    pub(crate) fn gguf_set_val_u32(ctx: GgufContextRaw, key: *const c_char, val: u32);
    pub(crate) fn gguf_set_val_u64(ctx: GgufContextRaw, key: *const c_char, val: u64);
    pub(crate) fn gguf_set_val_f32(ctx: GgufContextRaw, key: *const c_char, val: f32);
    pub(crate) fn gguf_set_val_bool(ctx: GgufContextRaw, key: *const c_char, val: bool);
    pub(crate) fn gguf_set_val_str(ctx: GgufContextRaw, key: *const c_char, val: *const c_char);
    pub(crate) fn gguf_set_arr_data(
        ctx: GgufContextRaw,
        key: *const c_char,
        type_: c_int,
        data: *const c_void,
        n: usize,
    );
    pub(crate) fn gguf_set_arr_str(
        ctx: GgufContextRaw,
        key: *const c_char,
        data: *const *const c_char,
        n: usize,
    );
    pub(crate) fn gguf_add_tensor(ctx: GgufContextRaw, tensor: GgmlTensorRaw);
    pub(crate) fn gguf_set_tensor_type(ctx: GgufContextRaw, name: *const c_char, type_: c_int);
    pub(crate) fn gguf_set_tensor_data(
        ctx: GgufContextRaw,
        name: *const c_char,
        data: *const c_void,
    );
    pub(crate) fn gguf_write_to_file(
        ctx: *const c_void,
        fname: *const c_char,
        only_meta: bool,
    ) -> bool;

    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_metal_init() -> GgmlBackendRaw;
}
