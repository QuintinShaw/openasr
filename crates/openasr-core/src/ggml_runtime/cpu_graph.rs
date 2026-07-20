//! Hands-off: single-responsibility ggml graph transcription runtime, guarded
//! by golden/parity tests. Do not split this module for "tidiness" -- the
//! tensor wiring is validated as a whole and refactoring here risks silent
//! numeric drift.

use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, CString, c_int, c_void},
    marker::PhantomData,
    path::Path,
    ptr::{self, NonNull},
    sync::OnceLock,
};

use memmap2::Mmap;
use thiserror::Error;

use super::ffi;
use super::{
    GgmlBackendKind, GgmlRuntimeError, GgufTensorDataReader, GgufWeightTensorPayload,
    ensure_backends_loaded, ggml_available_devices,
};

const F32_WIDTH_BYTES: usize = std::mem::size_of::<f32>();
const F16_WIDTH_BYTES: usize = std::mem::size_of::<u16>();
const I32_WIDTH_BYTES: usize = std::mem::size_of::<i32>();
const DEFAULT_CONTEXT_BYTES: usize = 1024 * 1024;
const DEFAULT_GRAPH_SIZE: usize = 4096;

/// `head_dim` (q/k/v ne0) values the Metal `GGML_OP_FLASH_ATTN_EXT` support
/// check accepts, mirrored from `ggml-metal-device.m`'s
/// `ggml_metal_device_supports_op` switch on `GGML_OP_FLASH_ATTN_EXT` ("for
/// new head sizes, add checks here"). Any other head_dim silently falls back
/// to a CPU-side (non-Metal) path inside ggml or asserts, depending on ggml
/// version, so `flash_attn_ext` fails closed here instead of trusting the
/// caller to have picked a supported head_dim.
const METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS: &[usize] = &[
    32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576,
];

/// Pure predicate behind `ensure_flash_attn_ext_metal_head_dim_supported`,
/// pulled out of the tensor-bearing method so the whitelist logic is testable
/// without standing up a live Metal backend. Only the Metal backend enforces
/// the whitelist; CPU/Gpu accept any head_dim `flash_attn_ext`'s other shape
/// checks already allow.
fn flash_attn_ext_head_dim_supported_on_backend(
    backend_kind: GgmlCpuGraphBackend,
    head_dim: usize,
) -> bool {
    backend_kind != GgmlCpuGraphBackend::Metal
        || METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS.contains(&head_dim)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlCpuGraphConfig {
    pub context_bytes: usize,
    pub graph_size: usize,
    pub n_threads: Option<usize>,
    pub backend: GgmlCpuGraphBackend,
    pub use_scheduler: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlCpuGraphBackend {
    Cpu,
    Metal,
    Gpu,
}

/// Per-family Auto-mode GPU policy: which GPU-class backend(s) Auto is
/// allowed to pick automatically for this family. `backend == Metal` is
/// exactly "Apple Silicon Metal" (see `default_gpu_backend_for_target`), so
/// `ExceptMetal` precisely targets the M-series measurements without ever
/// touching the discrete-GPU lane (CUDA/HIP/Vulkan), which no family here has
/// been measured to regress on.
///
/// Like the `bool` this replaces, a policy can only ever pin Auto to CPU; it
/// never overrides an explicit per-request `execution_target` preference
/// (see [`GgmlCpuGraphConfig::resolve_family_runtime_backend`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AutoGpuPolicy {
    /// Auto may use any available GPU-class backend (Metal, or the generic
    /// CUDA/HIP/Vulkan lane). Equivalent to the old `auto_gpu_enabled = true`.
    #[default]
    AllBackends,
    /// Auto may use the generic GPU lane (CUDA/HIP/Vulkan) but falls back to
    /// CPU on Apple Silicon Metal specifically.
    ExceptMetal,
    /// Auto never picks a GPU-class backend for this family. Equivalent to
    /// the old `auto_gpu_enabled = false`.
    Never,
}

impl GgmlCpuGraphBackend {
    /// GPU-class backends (Metal and the generic discrete-GPU lane:
    /// HIP/CUDA/Vulkan), as opposed to the CPU backend.
    ///
    /// These run the decode graph on a single GPU backend (no CPU-fallback
    /// scheduler) and can build a fixed-span-KV decode graph once and reuse it
    /// across tokens — each token re-encodes only the refreshed inputs. The CPU
    /// compute path mis-recomputes such an in-place-KV reused graph (and so does
    /// the multi-backend scheduler path, whose `sched_alloc_graph` drops the
    /// per-token inputs), so both must rebuild every token instead.
    pub fn is_gpu_class(self) -> bool {
        matches!(self, Self::Metal | Self::Gpu)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlCpuGraphThreadingWorkload {
    Default,
    EncoderPrelude,
    Decoder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgmlCpuGraphCpuAcceleratorPolicy {
    Auto,
    Disabled,
    Blas,
}

impl Default for GgmlCpuGraphConfig {
    fn default() -> Self {
        Self::runtime_default()
    }
}

/// Per-request execution-backend preference, installed thread-locally for the
/// duration of a decode (same idiom as the inference-threads override).
/// `GgmlCpuGraphConfig::resolve_runtime_backend` consults it BEFORE the env,
/// so every downstream backend decision — graph configs, runtime cache keys,
/// serve-batch job snapshots, telemetry labels — follows the request instead
/// of the process-global default. Values that cross threads (e.g. qwen
/// serve-batch jobs) materialize the backend on the submitting thread, so the
/// override propagates with the job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestBackendPreference {
    CpuOnly,
    Accelerated,
}

thread_local! {
    static REQUEST_BACKEND_PREFERENCE: std::cell::Cell<Option<RequestBackendPreference>> =
        const { std::cell::Cell::new(None) };
}

pub struct RequestBackendOverrideGuard {
    previous: Option<RequestBackendPreference>,
}

impl Drop for RequestBackendOverrideGuard {
    fn drop(&mut self) {
        REQUEST_BACKEND_PREFERENCE.with(|preference| preference.set(self.previous));
    }
}

#[must_use = "the override is uninstalled when the guard drops"]
pub fn install_request_backend_override(
    preference: Option<RequestBackendPreference>,
) -> RequestBackendOverrideGuard {
    let previous = REQUEST_BACKEND_PREFERENCE.with(|cell| {
        let previous = cell.get();
        cell.set(preference);
        previous
    });
    RequestBackendOverrideGuard { previous }
}

pub fn request_backend_override() -> Option<RequestBackendPreference> {
    REQUEST_BACKEND_PREFERENCE.with(std::cell::Cell::get)
}

impl GgmlCpuGraphConfig {
    pub const THREADS_ENV: &'static str = "OPENASR_GGML_CPU_THREADS";
    pub const BACKEND_ENV: &'static str = "OPENASR_GGML_BACKEND";
    pub const USE_SCHEDULER_ENV: &'static str = "OPENASR_GGML_USE_SCHEDULER";
    pub const CPU_ACCELERATOR_ENV: &'static str = "OPENASR_GGML_CPU_ACCELERATOR";

    pub const fn conservative_default() -> Self {
        Self {
            context_bytes: DEFAULT_CONTEXT_BYTES,
            graph_size: DEFAULT_GRAPH_SIZE,
            n_threads: None,
            backend: GgmlCpuGraphBackend::Cpu,
            use_scheduler: false,
        }
    }

    /// Exact byte capacity a `no_alloc` metadata context needs to hold a forward
    /// graph of up to `graph_size` nodes: the cgraph object itself plus
    /// per-tensor bookkeeping for every node (leafs never exceed the node budget
    /// in practice). This mirrors llama.cpp's compute-meta buffer sizing
    /// (`ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(...)`) and
    /// replaces hand-tuned byte counts that historically over-reserved by ~24x:
    /// `ggml_init` always mallocs the full `mem_size` even when `no_alloc=true`,
    /// so three coexisting 2 GiB encoder contexts committed 6 GiB and tripped
    /// `_aligned_malloc` -> NULL -> `GGML_ASSERT(ctx->mem_buffer != NULL)` on CPU,
    /// even though each context only ever used ~83 MB. Both `ggml_tensor_overhead`
    /// and `ggml_graph_overhead_custom` are pure size arithmetic and need no
    /// initialized context or loaded backend, so this is safe to call at config
    /// time.
    pub fn metadata_context_bytes(graph_size: usize) -> usize {
        let per_tensor = unsafe { ffi::ggml_tensor_overhead() };
        let graph = unsafe { ffi::ggml_graph_overhead_custom(graph_size, false) };
        graph_size.saturating_mul(per_tensor).saturating_add(graph)
    }

    pub fn runtime_default() -> Self {
        Self::runtime_default_for_backend(Self::resolve_runtime_backend())
    }

    /// Same defaults as [`Self::runtime_default`] but for a family whose Auto
    /// backend selection is gated by an [`AutoGpuPolicy`] (see
    /// [`Self::resolve_family_runtime_backend`]) rather than following
    /// `resolve_runtime_backend()` unconditionally. Families that build their
    /// base config from `GgmlCpuGraphConfig::default()`/`runtime_default()`
    /// bypass the family gate entirely (those always resolve the generic
    /// backend); a gated family must start from this instead.
    pub fn gated_runtime_default(policy: AutoGpuPolicy) -> Self {
        Self::runtime_default_for_backend(Self::resolve_family_runtime_backend(policy))
    }

    fn runtime_default_for_backend(backend: GgmlCpuGraphBackend) -> Self {
        Self {
            context_bytes: DEFAULT_CONTEXT_BYTES,
            graph_size: DEFAULT_GRAPH_SIZE,
            n_threads: Self::resolve_runtime_thread_count_for(
                backend,
                GgmlCpuGraphThreadingWorkload::Default,
            ),
            backend,
            use_scheduler: Self::resolve_runtime_scheduler_usage(),
        }
    }

    pub fn resolve_runtime_thread_count_for(
        backend: GgmlCpuGraphBackend,
        workload: GgmlCpuGraphThreadingWorkload,
    ) -> Option<usize> {
        Self::parse_thread_count_env(std::env::var(Self::THREADS_ENV).ok().as_deref())
            .or_else(|| Self::available_parallelism_thread_count(backend, workload))
    }

    fn available_parallelism_thread_count(
        backend: GgmlCpuGraphBackend,
        workload: GgmlCpuGraphThreadingWorkload,
    ) -> Option<usize> {
        std::thread::available_parallelism()
            .ok()
            .map(|value| value.get())
            .map(|value| Self::adaptive_thread_count_for_available(value, backend, workload))
            .and_then(Self::validate_thread_count)
    }

    fn adaptive_thread_count_for_available(
        available: usize,
        backend: GgmlCpuGraphBackend,
        workload: GgmlCpuGraphThreadingWorkload,
    ) -> usize {
        if available <= 1 {
            return 1;
        }

        // These fractions are workload-specific perf heuristics, not style
        // constants. Encoder prelude work benefits from wider CPU parallelism,
        // while autoregressive decoder steps are latency-bound and lose more to
        // oversubscription. GPU-class runs also leave CPU headroom for driver
        // work. Do not flatten this table without a multi-core thread sweep
        // (small and large hosts); the bench-suite's single-host RTF gate is not
        // enough evidence for thread policy changes.
        let (numerator, denominator, min_threads) = if backend.is_gpu_class() {
            match workload {
                GgmlCpuGraphThreadingWorkload::EncoderPrelude => (2, 3, 2),
                GgmlCpuGraphThreadingWorkload::Default | GgmlCpuGraphThreadingWorkload::Decoder => {
                    (1, 3, 2)
                }
            }
        } else {
            match workload {
                GgmlCpuGraphThreadingWorkload::Default => (3, 4, 1),
                GgmlCpuGraphThreadingWorkload::EncoderPrelude => (7, 8, 2),
                GgmlCpuGraphThreadingWorkload::Decoder => (1, 2, 2),
            }
        };
        let scaled = available.saturating_mul(numerator) / denominator;
        scaled.clamp(min_threads, available)
    }

    fn parse_thread_count_env(raw: Option<&str>) -> Option<usize> {
        let raw = raw?.trim();
        if raw.is_empty() {
            return None;
        }
        raw.parse::<usize>()
            .ok()
            .and_then(Self::validate_thread_count)
    }

    fn validate_thread_count(n_threads: usize) -> Option<usize> {
        (n_threads > 0 && c_int::try_from(n_threads).is_ok()).then_some(n_threads)
    }

    pub fn resolve_runtime_backend() -> GgmlCpuGraphBackend {
        match request_backend_override() {
            Some(RequestBackendPreference::CpuOnly) => GgmlCpuGraphBackend::Cpu,
            Some(RequestBackendPreference::Accelerated) => Self::default_gpu_backend(),
            None => Self::parse_backend_env(std::env::var(Self::BACKEND_ENV).ok().as_deref()),
        }
    }

    /// Single choke point for "should this family run on GPU": every family
    /// whose Auto default can be gated (rather than always following
    /// `resolve_runtime_backend()` unconditionally) must resolve its backend
    /// through this function instead of hand-rolling the override check
    /// (dolphin's `dolphin_runtime_backend`, xasr-zipformer's
    /// `encoder_gpu_enabled`, qwen's and moonshine's `*_runtime_graph_config`
    /// are the builtin cases that route through this gate -- see their doc
    /// comments for the per-family policy and its evidence). The
    /// [`AutoGpuPolicy`] is a pure Auto-mode default: it can only ever pin
    /// Auto to CPU, never override an explicit per-request preference. A
    /// request that explicitly asked for `CpuOnly` or
    /// `Accelerated` always gets what it asked for via
    /// `resolve_runtime_backend()`, matching the product rule that hardware
    /// selection is the engine's call only in Auto -- an explicit user choice
    /// always wins, even where auto-mode would have picked differently.
    ///
    /// This is also the function any provenance/telemetry label reporting
    /// "which backend actually ran" for such a family must call (with the
    /// same [`AutoGpuPolicy`] the family itself used) instead of
    /// `resolve_runtime_backend()` directly -- calling the generic resolver
    /// for a gated family's provenance label reports what Auto would
    /// generically pick, not what the family actually decided, which is
    /// exactly the kind of drift that produced a `core.native.backend:metal`
    /// label on a request that in fact ran entirely on CPU.
    ///
    /// `policy` is a pure Auto-mode default: an explicit per-request
    /// preference (`RequestBackendPreference::CpuOnly` /
    /// `::Accelerated`) always wins over it, matching the product rule that
    /// hardware selection is the engine's call only in Auto.
    pub fn resolve_family_runtime_backend(policy: AutoGpuPolicy) -> GgmlCpuGraphBackend {
        if request_backend_override().is_some() {
            return Self::resolve_runtime_backend();
        }
        let resolved = Self::resolve_runtime_backend();
        let gate_to_cpu = match policy {
            AutoGpuPolicy::AllBackends => false,
            AutoGpuPolicy::Never => resolved.is_gpu_class(),
            AutoGpuPolicy::ExceptMetal => matches!(resolved, GgmlCpuGraphBackend::Metal),
        };
        if gate_to_cpu {
            GgmlCpuGraphBackend::Cpu
        } else {
            resolved
        }
    }

    pub(crate) fn resolve_backend_name_for(
        backend: GgmlCpuGraphBackend,
    ) -> Result<String, GgmlCpuGraphError> {
        let guard = match backend {
            GgmlCpuGraphBackend::Cpu => GgmlBackendGuard::cpu()?,
            GgmlCpuGraphBackend::Metal => GgmlBackendGuard::metal()?,
            GgmlCpuGraphBackend::Gpu => GgmlBackendGuard::gpu()?,
        };
        Ok(guard.name())
    }

    fn parse_backend_env(raw: Option<&str>) -> GgmlCpuGraphBackend {
        match raw.map(str::trim).filter(|value| !value.is_empty()) {
            Some(value) if value.eq_ignore_ascii_case("metal") => GgmlCpuGraphBackend::Metal,
            Some(value) if value.eq_ignore_ascii_case("gpu") => Self::default_gpu_backend(),
            Some(value) if is_generic_gpu_backend_alias(value) => GgmlCpuGraphBackend::Gpu,
            Some(value) if value.eq_ignore_ascii_case("cpu") => GgmlCpuGraphBackend::Cpu,
            _ => Self::default_runtime_backend(),
        }
    }

    #[cfg(test)]
    fn parse_backend_env_with_default(
        raw: Option<&str>,
        default_backend: GgmlCpuGraphBackend,
    ) -> GgmlCpuGraphBackend {
        match raw.map(str::trim).filter(|value| !value.is_empty()) {
            Some(value) if value.eq_ignore_ascii_case("metal") => GgmlCpuGraphBackend::Metal,
            Some(value) if value.eq_ignore_ascii_case("gpu") => {
                Self::default_gpu_backend_for_target()
            }
            Some(value) if is_generic_gpu_backend_alias(value) => GgmlCpuGraphBackend::Gpu,
            Some(value) if value.eq_ignore_ascii_case("cpu") => GgmlCpuGraphBackend::Cpu,
            _ => default_backend,
        }
    }

    fn default_runtime_backend() -> GgmlCpuGraphBackend {
        if runtime_gpu_is_available() {
            Self::default_gpu_backend()
        } else {
            GgmlCpuGraphBackend::Cpu
        }
    }

    fn default_gpu_backend() -> GgmlCpuGraphBackend {
        Self::default_gpu_backend_for_target()
    }

    #[cfg(target_os = "macos")]
    fn default_gpu_backend_for_target() -> GgmlCpuGraphBackend {
        GgmlCpuGraphBackend::Metal
    }

    #[cfg(not(target_os = "macos"))]
    fn default_gpu_backend_for_target() -> GgmlCpuGraphBackend {
        GgmlCpuGraphBackend::Gpu
    }

    pub fn resolve_runtime_scheduler_usage() -> bool {
        match std::env::var(Self::USE_SCHEDULER_ENV).ok() {
            Some(raw) => Self::parse_bool_env(Some(raw.as_str())),
            None => true,
        }
    }

    fn parse_bool_env(raw: Option<&str>) -> bool {
        matches!(
            raw.map(str::trim).filter(|value| !value.is_empty()),
            Some(value)
                if value.eq_ignore_ascii_case("1")
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("yes")
                    || value.eq_ignore_ascii_case("on")
        )
    }

    fn resolve_runtime_cpu_accelerator_policy(
        backend: GgmlCpuGraphBackend,
    ) -> GgmlCpuGraphCpuAcceleratorPolicy {
        Self::resolve_runtime_cpu_accelerator_policy_with_env(
            std::env::var(Self::CPU_ACCELERATOR_ENV).ok().as_deref(),
            backend,
        )
    }

    fn resolve_runtime_cpu_accelerator_policy_with_env(
        raw: Option<&str>,
        backend: GgmlCpuGraphBackend,
    ) -> GgmlCpuGraphCpuAcceleratorPolicy {
        match raw.map(str::trim).filter(|value| !value.is_empty()) {
            Some(value) => Self::parse_cpu_accelerator_env(Some(value)),
            None => {
                if matches!(
                    backend,
                    GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu
                ) {
                    GgmlCpuGraphCpuAcceleratorPolicy::Disabled
                } else {
                    GgmlCpuGraphCpuAcceleratorPolicy::Auto
                }
            }
        }
    }

    fn parse_cpu_accelerator_env(raw: Option<&str>) -> GgmlCpuGraphCpuAcceleratorPolicy {
        let normalized = raw.map(str::trim).filter(|value| !value.is_empty());
        match normalized {
            Some(value)
                if value.eq_ignore_ascii_case("off")
                    || value.eq_ignore_ascii_case("none")
                    || value.eq_ignore_ascii_case("0") =>
            {
                GgmlCpuGraphCpuAcceleratorPolicy::Disabled
            }
            Some(value) if value.eq_ignore_ascii_case("blas") => {
                GgmlCpuGraphCpuAcceleratorPolicy::Blas
            }
            _ => GgmlCpuGraphCpuAcceleratorPolicy::Auto,
        }
    }

    pub fn cpu_accelerator_enabled_for_backend(backend: GgmlCpuGraphBackend) -> bool {
        !matches!(
            Self::resolve_runtime_cpu_accelerator_policy(backend),
            GgmlCpuGraphCpuAcceleratorPolicy::Disabled
        )
    }

    pub fn cpu_accelerator_enabled_with_env(
        raw: Option<&str>,
        backend: GgmlCpuGraphBackend,
    ) -> bool {
        !matches!(
            Self::resolve_runtime_cpu_accelerator_policy_with_env(raw, backend),
            GgmlCpuGraphCpuAcceleratorPolicy::Disabled
        )
    }
}

fn runtime_gpu_is_available() -> bool {
    static HAS_GPU: OnceLock<bool> = OnceLock::new();
    *HAS_GPU.get_or_init(|| {
        // Register plugin backends before the first registry query — under
        // GGML_BACKEND_DL the registry is otherwise empty and this OnceLock would
        // cache "no GPU" forever.
        ensure_backends_loaded();
        let best_backend_is_gpu = NonNull::new(unsafe { ffi::ggml_backend_init_best() })
            .map(|raw| {
                let name = backend_name(raw).to_ascii_lowercase();
                unsafe { ffi::ggml_backend_free(raw.as_ptr()) };
                backend_name_is_accelerated(&name)
            })
            .unwrap_or(false);
        if best_backend_is_gpu {
            return true;
        }
        ggml_available_devices()
            .into_iter()
            .any(|device| backend_kind_is_accelerated(device.kind))
    })
}

fn is_generic_gpu_backend_alias(value: &str) -> bool {
    value.eq_ignore_ascii_case("hip")
        || value.eq_ignore_ascii_case("rocm")
        || value.eq_ignore_ascii_case("cuda")
        || value.eq_ignore_ascii_case("vulkan")
}

fn backend_kind_is_accelerated(kind: GgmlBackendKind) -> bool {
    matches!(
        kind,
        GgmlBackendKind::Gpu | GgmlBackendKind::IntegratedGpu | GgmlBackendKind::Accelerator
    )
}

fn backend_name_is_accelerated(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("metal")
        || name.starts_with("mtl")
        || name.contains("hip")
        || name.contains("rocm")
        || name.contains("cuda")
        || name.contains("vulkan")
        || name.contains("gpu")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlCpuBinaryOp {
    Add,
    Mul,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GgmlRopeExtParams {
    pub n_dims: usize,
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[allow(dead_code)]
impl GgmlRopeExtParams {
    pub(crate) fn qwen_neox(
        n_dims: usize,
        n_ctx_orig: usize,
        rope_theta: f32,
    ) -> Result<Self, GgmlCpuGraphError> {
        let n_ctx_orig =
            i32::try_from(n_ctx_orig).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext n_ctx_orig exceeds ggml int boundary",
            })?;
        Ok(Self {
            n_dims,
            mode: ffi::GGML_ROPE_TYPE_NEOX,
            n_ctx_orig,
            freq_base: rope_theta,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        })
    }

    /// GPT-J / interleaved (HF `repeat_interleave(2)`) partial RoPE.
    ///
    /// `n_dims` is the rotary dimension count (`int(head_dim * partial_rotary_factor)`),
    /// which may be smaller than the head dimension; ggml rotates the leading `n_dims`
    /// of every head and passes the remaining tail through unrotated. This matches the
    /// Moonshine encoder/decoder partial rotary embedding.
    pub(crate) fn moonshine_gptj(
        n_dims: usize,
        n_ctx_orig: usize,
        rope_theta: f32,
    ) -> Result<Self, GgmlCpuGraphError> {
        let n_ctx_orig =
            i32::try_from(n_ctx_orig).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext n_ctx_orig exceeds ggml int boundary",
            })?;
        Ok(Self {
            n_dims,
            mode: ffi::GGML_ROPE_TYPE_NORMAL,
            n_ctx_orig,
            freq_base: rope_theta,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GgmlCpuGraphError {
    #[error("ggml cpu graph context bytes must be positive")]
    InvalidContextBytes,
    #[error("ggml cpu graph node capacity must be positive")]
    InvalidGraphSize,
    #[error("ggml cpu graph thread count must be positive")]
    InvalidThreadCount,
    #[error("ggml cpu graph thread count exceeds ggml int boundary: n_threads={n_threads}")]
    ThreadCountOutOfRange { n_threads: usize },
    #[error("ggml cpu graph context initialization failed (bytes={context_bytes})")]
    ContextInitFailed { context_bytes: usize },
    #[error("ggml cpu backend is unavailable")]
    CpuBackendUnavailable,
    #[error("ggml metal backend is unavailable (actual backend: {actual_backend})")]
    MetalBackendUnavailable { actual_backend: String },
    #[error("ggml gpu backend is unavailable (actual backend: {actual_backend})")]
    GpuBackendUnavailable { actual_backend: String },
    #[error("ggml cpu graph only supports add in this version, got {operation:?}")]
    UnsupportedOperation { operation: GgmlCpuBinaryOp },
    #[error("ggml cpu graph input tensors are unsupported: {reason}")]
    UnsupportedInputs { reason: &'static str },
    #[error("ggml cpu graph tensor allocation failed for '{tensor}'")]
    TensorAllocationFailed { tensor: &'static str },
    #[error("ggml cpu graph construction failed at '{step}'")]
    GraphBuildFailed { step: &'static str },
    #[error("ggml cpu graph backend buffer allocation failed")]
    BackendBufferAllocationFailed,
    #[error(
        "ggml cpu graph backend has no device (backend outlived the thread-local \
         backend that owns it); refusing to allocate against a dangling backend"
    )]
    BackendDeviceUnavailable,
    #[error("ggml cpu graph could not create loaded weight context: {reason}")]
    LoadedWeightContextFailed { reason: String },
    #[error("ggml cpu graph loaded tensor '{tensor}' is missing from context")]
    LoadedTensorMissing { tensor: String },
    #[error(
        "ggml cpu graph input byte width mismatch for '{tensor}': expected={expected}, actual={actual}"
    )]
    InputByteSizeMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("ggml cpu graph output byte width mismatch: expected={expected}, actual={actual}")]
    OutputByteSizeMismatch { expected: usize, actual: usize },
    #[error("ggml cpu graph compute failed with status={status}")]
    ComputeFailed { status: i32 },
    #[error("ggml cpu graph backend scheduler initialization failed")]
    BackendSchedulerInitFailed,
    #[error("ggml cpu graph backend scheduler graph allocation failed")]
    BackendSchedulerGraphAllocationFailed,
    #[error("ggml cpu graph tensor index out of bounds: index={index}, len={len}")]
    TensorIndexOutOfBounds { index: usize, len: usize },
    #[error("ggml cpu graph tensor bytes are not aligned to f32 width: bytes={bytes}")]
    TensorByteWidthMisaligned { bytes: usize },
    #[error(
        "ggml cpu graph tensor byte range out of bounds for '{tensor}': offset={offset}, len={len}, nbytes={nbytes}"
    )]
    TensorByteRangeOutOfBounds {
        tensor: &'static str,
        offset: usize,
        len: usize,
        nbytes: usize,
    },
    #[error(
        "ggml cpu graph cannot add new tensors or ops after backend allocation has started ({step})"
    )]
    GraphFrozenAfterAllocation { step: &'static str },
    #[error(
        "ggml cpu graph cannot materialize weight tensor '{tensor}' with unsupported rank {rank} (only 1d/2d/3d are supported)"
    )]
    UnsupportedWeightTensorRank { tensor: String, rank: usize },
    #[error("ggml cpu graph tensor upload requires contiguous layout for '{tensor}'")]
    TensorUploadRequiresContiguous { tensor: String },
    #[error(
        "ggml cpu graph tensor upload element-count mismatch for '{tensor}': expected={expected}, actual={actual}"
    )]
    TensorUploadElementCountMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "ggml cpu graph tensor upload shape mismatch for '{tensor}': expected={expected:?}, actual={actual:?}"
    )]
    TensorUploadShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error(
        "ggml cpu graph tensor upload byte-width mismatch for '{tensor}': expected={expected}, actual={actual}"
    )]
    TensorUploadByteSizeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    #[error("ggml cpu graph invalid permute axes: [{axis0}, {axis1}, {axis2}, {axis3}]")]
    InvalidPermuteAxes {
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    },
    #[error(
        "ggml_flash_attn_ext head_dim={head_dim} is not in the Metal backend's supported set {supported:?} (see ggml-metal-device.m FLASH_ATTN_EXT support check); use the naive attention fallback for this head_dim on Metal"
    )]
    FlashAttnExtUnsupportedMetalHeadDim {
        head_dim: usize,
        supported: &'static [usize],
    },
}

pub struct GgmlCpuGraphRunner {
    context: GgmlContextGuard,
    backend: GgmlBackendGuard,
    backend_kind: GgmlCpuGraphBackend,
    backend_name: String,
    graph_size: usize,
    _scheduler_accel_backends: Vec<GgmlBackendGuard>,
    _scheduler_cpu_fallback: Option<GgmlBackendGuard>,
    scheduler: Option<GgmlBackendSchedulerGuard>,
}

pub(crate) struct GgmlStaticTensorArena {
    context: GgmlContextGuard,
    backend: NonNull<c_void>,
    buffer: Option<GgmlBackendBufferGuard>,
    require_direct_matmul_weight_support: bool,
}

pub(crate) struct GgmlLoadedWeightContext {
    _context: GgmlContextGuard,
    _buffer: GgmlBackendBufferGuard,
    _mmap: Option<std::sync::Arc<Mmap>>,
    tensors: HashMap<String, GgmlLoadedTensor>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GgmlStaticTensor {
    raw: NonNull<c_void>,
}

impl GgmlStaticTensor {
    pub(crate) fn as_graph_tensor<'a>(self) -> GgmlCpuTensor<'a> {
        GgmlCpuTensor {
            raw: self.raw,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GgmlLoadedTensor {
    raw: NonNull<c_void>,
}

impl GgmlLoadedTensor {
    pub(crate) fn as_graph_tensor<'a>(self) -> GgmlCpuTensor<'a> {
        GgmlCpuTensor {
            raw: self.raw,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GgmlCpuTensor<'a> {
    raw: NonNull<c_void>,
    _marker: PhantomData<&'a ()>,
}

pub(crate) struct GgmlCpuGraphBuilder<'a> {
    context: NonNull<c_void>,
    backend: NonNull<c_void>,
    backend_kind: GgmlCpuGraphBackend,
    scheduler: Option<NonNull<c_void>>,
    graph_size: usize,
    buffer: Option<GgmlBackendBufferGuard>,
    prepared_graph: Option<NonNull<c_void>>,
    side_effect_roots: Vec<NonNull<c_void>>,
    _runner_borrow: PhantomData<&'a mut GgmlCpuGraphRunner>,
}

/// A graph builder whose ggml context outlives a single token, enabling a
/// built+allocated cgraph to be re-executed across decode steps with only its
/// input tensors refreshed. Field order matters for drop: `builder` (and its
/// backend buffer) is dropped before `_context` frees the ggml context the
/// builder's tensors live in. See `start_persistent_graph_session`.
///
/// Covered by the `persistent_graph_session_reuses_built_graph_*` tests and used
/// by the qwen whole-decoder for build-once/re-run decode (P9 graph reuse).
pub(crate) struct GgmlPersistentGraphSession {
    builder: GgmlCpuGraphBuilder<'static>,
    _context: GgmlContextGuard,
}

impl GgmlPersistentGraphSession {
    pub(crate) fn builder(&mut self) -> &mut GgmlCpuGraphBuilder<'static> {
        &mut self.builder
    }
}

impl GgmlCpuGraphRunner {
    pub fn new(config: GgmlCpuGraphConfig) -> Result<Self, GgmlCpuGraphError> {
        if config.context_bytes == 0 {
            return Err(GgmlCpuGraphError::InvalidContextBytes);
        }
        if config.graph_size == 0 {
            return Err(GgmlCpuGraphError::InvalidGraphSize);
        }

        let context = GgmlContextGuard::new(config.context_bytes)?;
        let mut backend = match config.backend {
            GgmlCpuGraphBackend::Cpu => GgmlBackendGuard::cpu()?,
            GgmlCpuGraphBackend::Metal => GgmlBackendGuard::metal()?,
            GgmlCpuGraphBackend::Gpu => GgmlBackendGuard::gpu()?,
        };
        if matches!(config.backend, GgmlCpuGraphBackend::Cpu)
            && let Some(n_threads) = config.n_threads
        {
            backend.set_n_threads(n_threads)?;
        }

        let cpu_accelerator_policy =
            GgmlCpuGraphConfig::resolve_runtime_cpu_accelerator_policy(config.backend);
        let scheduler_accel_backends = if config.use_scheduler {
            GgmlBackendGuard::accelerators(config.n_threads, cpu_accelerator_policy)
        } else {
            Vec::new()
        };

        let mut scheduler_cpu_fallback = None;
        if config.use_scheduler
            && matches!(
                config.backend,
                GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu
            )
        {
            let mut cpu = GgmlBackendGuard::cpu()?;
            if let Some(n_threads) = config.n_threads {
                cpu.set_n_threads(n_threads)?;
            }
            scheduler_cpu_fallback = Some(cpu);
        }
        let scheduler = if config.use_scheduler {
            let mut backends = Vec::new();
            if matches!(
                config.backend,
                GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu
            ) {
                backends.push(backend.raw.as_ptr());
            }
            backends.extend(
                scheduler_accel_backends
                    .iter()
                    .map(|accel| accel.raw.as_ptr()),
            );
            if matches!(config.backend, GgmlCpuGraphBackend::Cpu) {
                backends.push(backend.raw.as_ptr());
            }
            if let Some(cpu) = scheduler_cpu_fallback.as_ref() {
                backends.push(cpu.raw.as_ptr());
            }
            Some(GgmlBackendSchedulerGuard::new(
                &mut backends,
                config.graph_size,
            )?)
        } else {
            None
        };

        let backend_name = backend.name();
        Ok(Self {
            context,
            backend,
            backend_kind: config.backend,
            backend_name,
            graph_size: config.graph_size,
            _scheduler_accel_backends: scheduler_accel_backends,
            _scheduler_cpu_fallback: scheduler_cpu_fallback,
            scheduler,
        })
    }

    pub(crate) fn backend_kind(&self) -> GgmlCpuGraphBackend {
        self.backend_kind
    }

    pub(crate) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(crate) fn uses_scheduler(&self) -> bool {
        self.scheduler.is_some()
    }

    pub(crate) fn load_gguf_weight_context(
        &self,
        path: &Path,
    ) -> Result<GgmlLoadedWeightContext, GgmlCpuGraphError> {
        GgmlLoadedWeightContext::from_path_with_backend(
            path,
            self.backend.raw,
            self.backend_kind.is_gpu_class() && self.scheduler.is_none(),
        )
    }

    pub(crate) fn start_graph(&mut self) -> GgmlCpuGraphBuilder<'_> {
        unsafe { ffi::ggml_reset(self.context.raw.as_ptr()) };
        GgmlCpuGraphBuilder {
            context: self.context.raw,
            backend: self.backend.raw,
            backend_kind: self.backend_kind,
            scheduler: self.scheduler.as_ref().map(|scheduler| scheduler.raw),
            graph_size: self.graph_size,
            buffer: None,
            prepared_graph: None,
            side_effect_roots: Vec::new(),
            _runner_borrow: PhantomData,
        }
    }

    pub(crate) fn start_static_tensor_arena(
        &self,
        context_bytes: usize,
    ) -> Result<GgmlStaticTensorArena, GgmlCpuGraphError> {
        GgmlStaticTensorArena::new(
            context_bytes,
            self.backend.raw,
            self.backend_kind.is_gpu_class() && self.scheduler.is_none(),
        )
    }

    /// Open a graph builder whose ggml context is dedicated and SURVIVES across
    /// tokens (unlike `start_graph`, which `ggml_reset`s the runner's shared
    /// context every call and drops the builder). This is the foundation for
    /// "build once, re-run with new inputs" (the llama.cpp decode pattern):
    /// build the graph + `prepare_outputs_for_upload` once, then on later steps
    /// only refresh inputs via `set_*_slice` and call `compute_outputs_f32`
    /// again — the prepared cgraph is reused (no rebuild, no `sched_reset`).
    ///
    /// SAFETY: the returned session holds raw pointers into this runner's
    /// backend + scheduler, so the runner MUST outlive the session.
    pub(crate) fn start_persistent_graph_session(
        &mut self,
        context_bytes: usize,
    ) -> Result<GgmlPersistentGraphSession, GgmlCpuGraphError> {
        if context_bytes == 0 {
            return Err(GgmlCpuGraphError::InvalidContextBytes);
        }
        let context = GgmlContextGuard::new(context_bytes)?;
        let builder = GgmlCpuGraphBuilder {
            context: context.raw,
            backend: self.backend.raw,
            backend_kind: self.backend_kind,
            scheduler: self.scheduler.as_ref().map(|scheduler| scheduler.raw),
            graph_size: self.graph_size,
            buffer: None,
            prepared_graph: None,
            side_effect_roots: Vec::new(),
            _runner_borrow: PhantomData,
        };
        Ok(GgmlPersistentGraphSession {
            builder,
            _context: context,
        })
    }

    pub fn compute_add_f32(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<Vec<f32>, GgmlCpuGraphError> {
        self.compute_binary_f32(lhs, rhs, GgmlCpuBinaryOp::Add)
    }

    pub fn compute_binary_f32(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
        operation: GgmlCpuBinaryOp,
    ) -> Result<Vec<f32>, GgmlCpuGraphError> {
        if operation != GgmlCpuBinaryOp::Add {
            return Err(GgmlCpuGraphError::UnsupportedOperation { operation });
        }
        if lhs.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "empty tensors are not supported",
            });
        }
        if lhs.len() != rhs.len() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "lhs and rhs length mismatch",
            });
        }

        let mut graph = self.start_graph();
        let lhs_tensor = graph.new_tensor_1d_f32(lhs.len(), "lhs")?;
        let rhs_tensor = graph.new_tensor_1d_f32(rhs.len(), "rhs")?;

        graph.set_input(lhs_tensor)?;
        graph.set_input(rhs_tensor)?;

        let output_tensor = graph.add(lhs_tensor, rhs_tensor)?;
        graph.set_output(output_tensor)?;

        graph.set_f32_slice(lhs_tensor, lhs, "lhs")?;
        graph.set_f32_slice(rhs_tensor, rhs, "rhs")?;
        graph.compute_output_f32(output_tensor, lhs.len())
    }
}

impl GgmlLoadedWeightContext {
    pub(crate) fn tensor(&self, name: &str) -> Option<GgmlLoadedTensor> {
        self.tensors.get(name).copied()
    }

    fn from_path_with_backend(
        path: &Path,
        backend: NonNull<c_void>,
        require_direct_backend_matmul_support: bool,
    ) -> Result<Self, GgmlCpuGraphError> {
        let path_cstring = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
            GgmlCpuGraphError::LoadedWeightContextFailed {
                reason: format!("path contains interior NUL bytes: {}", path.display()),
            }
        })?;
        let mut ggml_ctx_raw: ffi::GgmlContextRaw = ptr::null_mut();
        let gguf_ctx_raw = unsafe {
            ffi::gguf_init_from_file(
                path_cstring.as_ptr(),
                ffi::GgufInitParams {
                    no_alloc: true,
                    ctx: &mut ggml_ctx_raw,
                },
            )
        };
        let Some(gguf_ctx_raw) = NonNull::new(gguf_ctx_raw) else {
            return Err(GgmlCpuGraphError::LoadedWeightContextFailed {
                reason: format!("gguf_init_from_file failed for {}", path.display()),
            });
        };
        let gguf_ctx = GgufContextGuard { raw: gguf_ctx_raw };
        let Some(ggml_ctx_raw) = NonNull::new(ggml_ctx_raw) else {
            return Err(GgmlCpuGraphError::LoadedWeightContextFailed {
                reason: format!(
                    "gguf_init_from_file did not return ggml context for {}",
                    path.display()
                ),
            });
        };
        let context = GgmlContextGuard::from_raw(ggml_ctx_raw);
        if require_direct_backend_matmul_support {
            validate_direct_backend_matmul_weight_support(context.raw, backend, path)?;
        }
        let reader = GgufTensorDataReader::from_path(path).map_err(|error| {
            GgmlCpuGraphError::LoadedWeightContextFailed {
                reason: error.to_string(),
            }
        })?;
        let (buffer, mmap) = match maybe_allocate_weight_buffer_from_host_ptr(backend, &reader)? {
            Some((buffer, mmap)) => (buffer, Some(mmap)),
            None => (
                GgmlBackendBufferGuard::allocate(context.raw, backend)?,
                None,
            ),
        };
        let mut tensors = HashMap::new();
        let mut tensor_raw = unsafe { ffi::ggml_get_first_tensor(context.raw.as_ptr()) };
        while let Some(raw) = NonNull::new(tensor_raw) {
            let name = unsafe { cstr_lossy(ffi::ggml_get_name(raw.as_ptr())) };
            let payload = reader.host_tensor_bytes_by_name(&name).map_err(|error| {
                GgmlCpuGraphError::LoadedWeightContextFailed {
                    reason: format!(
                        "could not read tensor '{name}' from {}: {error}",
                        path.display()
                    ),
                }
            })?;
            if let Some(mmap) = mmap.as_ref() {
                let addr = unsafe { mmap.as_ptr().add(payload.start).cast_mut().cast::<c_void>() };
                let status = unsafe {
                    ffi::ggml_backend_tensor_alloc(buffer.raw.as_ptr(), raw.as_ptr(), addr)
                };
                if status != ffi::GGML_STATUS_SUCCESS {
                    return Err(GgmlCpuGraphError::LoadedWeightContextFailed {
                        reason: format!(
                            "backend tensor alloc failed for '{name}' with status={status}"
                        ),
                    });
                }
            } else {
                unsafe {
                    ffi::ggml_backend_tensor_set(
                        raw.as_ptr(),
                        payload.bytes.as_ptr().cast::<c_void>(),
                        0,
                        payload.bytes.len(),
                    );
                }
            }
            tensors.insert(name, GgmlLoadedTensor { raw });
            tensor_raw = unsafe { ffi::ggml_get_next_tensor(context.raw.as_ptr(), raw.as_ptr()) };
        }
        drop(gguf_ctx);
        Ok(Self {
            _context: context,
            _buffer: buffer,
            _mmap: mmap,
            tensors,
        })
    }
}

impl GgmlStaticTensorArena {
    fn new(
        context_bytes: usize,
        backend: NonNull<c_void>,
        require_direct_matmul_weight_support: bool,
    ) -> Result<Self, GgmlCpuGraphError> {
        if context_bytes == 0 {
            return Err(GgmlCpuGraphError::InvalidContextBytes);
        }
        Ok(Self {
            context: GgmlContextGuard::new(context_bytes)?,
            backend,
            buffer: None,
            require_direct_matmul_weight_support,
        })
    }

    pub(crate) fn new_tensor_2d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_2d_typed(ne0, ne1, ffi::GGML_TYPE_F16, tensor_name)
    }

    pub(crate) fn new_tensor_2d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_2d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_2d(self.context.raw.as_ptr(), ggml_type, ne0_i64, ne1_i64)
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    pub(crate) fn new_matmul_weight_2d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        if self.require_direct_matmul_weight_support {
            validate_direct_matmul_weight_type(self.backend, ggml_type, tensor_name)?;
        }
        self.new_tensor_2d_typed(ne0, ne1, ggml_type, tensor_name)
    }

    pub(crate) fn new_tensor_1d_f32(
        &self,
        len: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_1d")?;
        let len_i64 = checked_dim_to_i64(len)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_1d(self.context.raw.as_ptr(), ffi::GGML_TYPE_F32, len_i64)
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    pub(crate) fn new_tensor_1d_i32(
        &self,
        len: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_1d")?;
        let len_i64 = checked_dim_to_i64(len)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_1d(self.context.raw.as_ptr(), ffi::GGML_TYPE_I32, len_i64)
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    pub(crate) fn new_tensor_2d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_2d_typed(ne0, ne1, ffi::GGML_TYPE_F32, tensor_name)
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_3d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_3d_typed(ne0, ne1, ne2, ffi::GGML_TYPE_F32, tensor_name)
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_3d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_3d_typed(ne0, ne1, ne2, ffi::GGML_TYPE_F16, tensor_name)
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_3d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_3d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_3d(
                self.context.raw.as_ptr(),
                ggml_type,
                ne0_i64,
                ne1_i64,
                ne2_i64,
            )
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_from_weight_payload(
        &self,
        payload: &GgufWeightTensorPayload<'_>,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_from_weight_payload")?;
        match payload.dims.as_slice() {
            [ne0] => {
                let ne0 = checked_dim_to_i64(*ne0)?;
                let raw = unsafe {
                    ffi::ggml_new_tensor_1d(
                        self.context.raw.as_ptr(),
                        payload.element_type.ggml_type(),
                        ne0,
                    )
                };
                NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
                    GgmlCpuGraphError::TensorAllocationFailed {
                        tensor: "weight_payload",
                    },
                )
            }
            [ne0, ne1] => self.new_tensor_2d_typed(
                *ne0,
                *ne1,
                payload.element_type.ggml_type(),
                "weight_payload",
            ),
            [ne0, ne1, ne2] => {
                let ne0 = checked_dim_to_i64(*ne0)?;
                let ne1 = checked_dim_to_i64(*ne1)?;
                let ne2 = checked_dim_to_i64(*ne2)?;
                let raw = unsafe {
                    ffi::ggml_new_tensor_3d(
                        self.context.raw.as_ptr(),
                        payload.element_type.ggml_type(),
                        ne0,
                        ne1,
                        ne2,
                    )
                };
                NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
                    GgmlCpuGraphError::TensorAllocationFailed {
                        tensor: "weight_payload",
                    },
                )
            }
            _ => Err(GgmlCpuGraphError::UnsupportedWeightTensorRank {
                tensor: payload.metadata.name.clone(),
                rank: payload.dims.len(),
            }),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_4d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_4d_typed(ne0, ne1, ne2, ne3, ffi::GGML_TYPE_F32, tensor_name)
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_4d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.new_tensor_4d_typed(ne0, ne1, ne2, ne3, ffi::GGML_TYPE_F16, tensor_name)
    }

    #[allow(dead_code)]
    pub(crate) fn new_tensor_4d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_new_tensor_4d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let ne3_i64 = checked_dim_to_i64(ne3)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_4d(
                self.context.raw.as_ptr(),
                ggml_type,
                ne0_i64,
                ne1_i64,
                ne2_i64,
                ne3_i64,
            )
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    pub(crate) fn graph_tensor<'a>(&self, tensor: GgmlStaticTensor) -> GgmlCpuTensor<'a> {
        GgmlCpuTensor {
            raw: tensor.raw,
            _marker: PhantomData,
        }
    }

    pub(crate) fn view_2d(
        &self,
        input: GgmlStaticTensor,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
        self.ensure_can_extend("ggml_view_2d")?;
        let element_size = self.static_element_size_bytes(self.static_tensor_type(input))?;
        if nb1
            != ne0
                .checked_mul(element_size)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_2d row stride overflows usize",
                })?
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d requires contiguous row stride",
            });
        }
        if !offset.is_multiple_of(element_size) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d offset must align to tensor element size",
            });
        }
        let start = offset / element_size;
        let count = ne0
            .checked_mul(ne1)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d shape overflows usize",
            })?;
        let span = count
            .checked_add(start)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d span overflows usize",
            })?;
        if self.static_tensor_nelements(input)? < span {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d span exceeds source tensor elements",
            });
        }
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let raw = unsafe {
            ffi::ggml_view_2d(
                self.context.raw.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                nb1,
                offset,
            )
        };
        NonNull::new(raw).map(|raw| GgmlStaticTensor { raw }).ok_or(
            GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            },
        )
    }

    pub(crate) fn allocate_backend_buffer(&mut self) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()
    }

    pub(crate) fn set_f16_bits_slice(
        &mut self,
        tensor: GgmlStaticTensor,
        values: &[u16],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let expected = values.len().checked_mul(F16_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
            false,
        )
    }

    pub(crate) fn set_bytes_slice(
        &mut self,
        tensor: GgmlStaticTensor,
        values: &[u8],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            values.len(),
            tensor_name,
            false,
        )
    }

    pub(crate) fn set_f32_slice(
        &mut self,
        tensor: GgmlStaticTensor,
        values: &[f32],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let expected = values.len().checked_mul(F32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
            false,
        )
    }

    pub(crate) fn set_i32_slice(
        &mut self,
        tensor: GgmlStaticTensor,
        values: &[i32],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let actual_type = self.static_tensor_type(tensor);
        if actual_type != ffi::GGML_TYPE_I32 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "static tensor upload requires i32 tensor",
            });
        }
        let expected = values.len().checked_mul(I32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
            false,
        )
    }

    pub(crate) fn set_f16_bits_slice_with_offset(
        &mut self,
        tensor: GgmlStaticTensor,
        offset_elements: usize,
        values: &[u16],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let offset_bytes = offset_elements.checked_mul(F16_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte offset overflow",
            },
        )?;
        let expected = values.len().checked_mul(F16_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            offset_bytes,
            expected,
            tensor_name,
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn read_f16_bits_slice(
        &self,
        tensor: GgmlStaticTensor,
        expected_len: usize,
    ) -> Result<Vec<u16>, GgmlCpuGraphError> {
        let actual_type = self.static_tensor_type(tensor);
        if actual_type != ffi::GGML_TYPE_F16 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "static tensor read requires f16 tensor",
            });
        }
        let expected_nbytes = expected_len.checked_mul(F16_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        let actual_nbytes = unsafe { ffi::ggml_nbytes(tensor.raw.as_ptr()) };
        if actual_nbytes != expected_nbytes {
            return Err(GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }
        let mut values = vec![0_u16; expected_len];
        unsafe {
            ffi::ggml_backend_tensor_get(
                tensor.raw.as_ptr(),
                values.as_mut_ptr().cast::<c_void>(),
                0,
                expected_nbytes,
            );
        }
        Ok(values)
    }

    #[allow(dead_code)]
    pub(crate) fn set_weight_tensor_from_payload(
        &mut self,
        tensor: GgmlStaticTensor,
        payload: &GgufWeightTensorPayload<'_>,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let actual_nbytes = unsafe { ffi::ggml_nbytes(tensor.raw.as_ptr()) };
        if actual_nbytes != payload.bytes.len() {
            return Err(GgmlCpuGraphError::TensorUploadByteSizeMismatch {
                tensor: payload.metadata.name.clone(),
                expected: payload.bytes.len(),
                actual: actual_nbytes,
            });
        }
        let actual_type =
            unsafe { *(tensor.raw.as_ptr() as *const ffi::GgmlTensorLayoutPrefix) }.type_;
        if actual_type != payload.element_type.ggml_type() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "weight tensor upload type does not match tensor type",
            });
        }
        let is_contiguous = unsafe { ffi::ggml_is_contiguous(tensor.raw.as_ptr()) };
        if !is_contiguous {
            return Err(GgmlCpuGraphError::TensorUploadRequiresContiguous {
                tensor: payload.metadata.name.clone(),
            });
        }
        unsafe {
            write_tensor_data(
                tensor.raw,
                payload.bytes.as_ptr().cast::<c_void>(),
                0,
                actual_nbytes,
            );
        }
        Ok(())
    }

    fn ensure_backend_buffer(&mut self) -> Result<(), GgmlCpuGraphError> {
        if self.buffer.is_none() {
            self.buffer = Some(GgmlBackendBufferGuard::allocate_with_usage(
                self.context.raw,
                self.backend,
                ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            )?);
        }
        Ok(())
    }

    fn ensure_can_extend(&self, step: &'static str) -> Result<(), GgmlCpuGraphError> {
        if self.buffer.is_some() {
            return Err(GgmlCpuGraphError::GraphFrozenAfterAllocation { step });
        }
        Ok(())
    }

    fn static_tensor_layout_prefix(&self, tensor: GgmlStaticTensor) -> ffi::GgmlTensorLayoutPrefix {
        unsafe { *(tensor.raw.as_ptr() as *const ffi::GgmlTensorLayoutPrefix) }
    }

    fn static_tensor_type(&self, tensor: GgmlStaticTensor) -> i32 {
        self.static_tensor_layout_prefix(tensor).type_
    }

    fn static_tensor_nelements(
        &self,
        tensor: GgmlStaticTensor,
    ) -> Result<usize, GgmlCpuGraphError> {
        let nelements = unsafe { ffi::ggml_nelements(tensor.raw.as_ptr()) };
        usize::try_from(nelements).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "tensor element count exceeds usize boundary",
        })
    }

    fn static_element_size_bytes(&self, type_: i32) -> Result<usize, GgmlCpuGraphError> {
        match type_ {
            ffi::GGML_TYPE_F16 => Ok(F16_WIDTH_BYTES),
            ffi::GGML_TYPE_F32 => Ok(F32_WIDTH_BYTES),
            ffi::GGML_TYPE_I32 => Ok(I32_WIDTH_BYTES),
            _ => Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor type is not supported for ggml_view_2d",
            }),
        }
    }

    fn write_tensor_bytes_checked(
        &self,
        tensor: GgmlStaticTensor,
        data_ptr: *const c_void,
        offset: usize,
        expected_nbytes: usize,
        tensor_name: &'static str,
        allow_partial: bool,
    ) -> Result<(), GgmlCpuGraphError> {
        let actual_nbytes = unsafe { ffi::ggml_nbytes(tensor.raw.as_ptr()) };
        let end =
            offset
                .checked_add(expected_nbytes)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte range overflow",
                })?;
        if !allow_partial && offset == 0 && actual_nbytes != expected_nbytes {
            return Err(GgmlCpuGraphError::InputByteSizeMismatch {
                tensor: tensor_name,
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }
        if end > actual_nbytes {
            return Err(GgmlCpuGraphError::TensorByteRangeOutOfBounds {
                tensor: tensor_name,
                offset,
                len: expected_nbytes,
                nbytes: actual_nbytes,
            });
        }
        let is_contiguous = unsafe { ffi::ggml_is_contiguous(tensor.raw.as_ptr()) };
        if !is_contiguous {
            return Err(GgmlCpuGraphError::TensorUploadRequiresContiguous {
                tensor: tensor_name.to_string(),
            });
        }
        unsafe {
            write_tensor_data(tensor.raw, data_ptr, offset, expected_nbytes);
        }
        Ok(())
    }
}

impl<'a> GgmlCpuGraphBuilder<'a> {
    pub(crate) fn backend_kind(&self) -> GgmlCpuGraphBackend {
        self.backend_kind
    }

    pub(crate) fn new_tensor_1d_f32(
        &self,
        len: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_1d")?;
        let len_i64 = checked_dim_to_i64(len)?;
        let raw =
            unsafe { ffi::ggml_new_tensor_1d(self.context.as_ptr(), ffi::GGML_TYPE_F32, len_i64) };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_tensor_2d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.new_tensor_2d_typed(ne0, ne1, ffi::GGML_TYPE_F32, tensor_name)
    }

    pub(crate) fn new_tensor_2d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_2d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let raw =
            unsafe { ffi::ggml_new_tensor_2d(self.context.as_ptr(), ggml_type, ne0_i64, ne1_i64) };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_matmul_weight_2d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        if self.scheduler.is_none() {
            validate_direct_matmul_weight_type(self.backend, ggml_type, tensor_name)?;
        }
        self.new_tensor_2d_typed(ne0, ne1, ggml_type, tensor_name)
    }

    pub(crate) fn new_tensor_4d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.new_tensor_4d_typed(ne0, ne1, ne2, ne3, ffi::GGML_TYPE_F32, tensor_name)
    }

    pub(crate) fn new_tensor_4d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.new_tensor_4d_typed(ne0, ne1, ne2, ne3, ffi::GGML_TYPE_F16, tensor_name)
    }

    pub(crate) fn new_tensor_4d_i32(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.new_tensor_4d_typed(ne0, ne1, ne2, ne3, ffi::GGML_TYPE_I32, tensor_name)
    }

    pub(crate) fn new_tensor_4d_typed(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        ggml_type: i32,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_4d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let ne3_i64 = checked_dim_to_i64(ne3)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_4d(
                self.context.as_ptr(),
                ggml_type,
                ne0_i64,
                ne1_i64,
                ne2_i64,
                ne3_i64,
            )
        };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_tensor_1d_i32(
        &self,
        len: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_1d")?;
        let len_i64 = checked_dim_to_i64(len)?;
        let raw =
            unsafe { ffi::ggml_new_tensor_1d(self.context.as_ptr(), ffi::GGML_TYPE_I32, len_i64) };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_tensor_3d_f32(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_3d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_3d(
                self.context.as_ptr(),
                ffi::GGML_TYPE_F32,
                ne0_i64,
                ne1_i64,
                ne2_i64,
            )
        };
        self.new_tensor_checked(raw, tensor_name)
    }

    #[cfg(test)]
    pub(crate) fn new_tensor_1d_f16(
        &self,
        len: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_1d")?;
        let len_i64 = checked_dim_to_i64(len)?;
        let raw =
            unsafe { ffi::ggml_new_tensor_1d(self.context.as_ptr(), ffi::GGML_TYPE_F16, len_i64) };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_tensor_3d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_new_tensor_3d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let raw = unsafe {
            ffi::ggml_new_tensor_3d(
                self.context.as_ptr(),
                ffi::GGML_TYPE_F16,
                ne0_i64,
                ne1_i64,
                ne2_i64,
            )
        };
        self.new_tensor_checked(raw, tensor_name)
    }

    pub(crate) fn new_tensor_2d_f16(
        &self,
        ne0: usize,
        ne1: usize,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.new_tensor_2d_typed(ne0, ne1, ffi::GGML_TYPE_F16, tensor_name)
    }

    #[cfg(test)]
    pub(crate) fn set_weight_tensor_from_payload(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        payload: &GgufWeightTensorPayload<'_>,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let tensor_name = payload.metadata.name.clone();
        let expected_type = payload.element_type.ggml_type();
        self.ensure_tensor_type(tensor, expected_type, "weight_upload")?;
        self.ensure_tensor_shape_prefix_matches(tensor, &payload.dims, &tensor_name)?;
        let actual_num_elements = self.tensor_nelements(tensor)?;
        if actual_num_elements != payload.num_elements {
            return Err(GgmlCpuGraphError::TensorUploadElementCountMismatch {
                tensor: tensor_name,
                expected: payload.num_elements,
                actual: actual_num_elements,
            });
        }
        let actual_nbytes = self.tensor_nbytes(tensor);
        let expected_nbytes = payload.bytes.len();
        if actual_nbytes != expected_nbytes {
            return Err(GgmlCpuGraphError::TensorUploadByteSizeMismatch {
                tensor: payload.metadata.name.clone(),
                expected: expected_nbytes,
                actual: actual_nbytes,
            });
        }
        let contiguous = unsafe { ffi::ggml_is_contiguous(tensor.raw.as_ptr()) };
        if !contiguous {
            return Err(GgmlCpuGraphError::TensorUploadRequiresContiguous {
                tensor: payload.metadata.name.clone(),
            });
        }
        unsafe {
            write_tensor_data(
                tensor.raw,
                payload.bytes.as_ptr().cast::<c_void>(),
                0,
                actual_nbytes,
            );
        }
        Ok(())
    }

    pub(crate) fn set_input(&mut self, tensor: GgmlCpuTensor<'a>) -> Result<(), GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_set_input")?;
        unsafe { ffi::ggml_set_input(tensor.raw.as_ptr()) };
        Ok(())
    }

    pub(crate) fn set_output(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_set_output")?;
        unsafe { ffi::ggml_set_output(tensor.raw.as_ptr()) };
        Ok(())
    }

    pub(crate) fn add(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_add")?;
        self.ensure_tensor_type(lhs, ffi::GGML_TYPE_F32, "ggml_add lhs")?;
        self.ensure_tensor_type(rhs, ffi::GGML_TYPE_F32, "ggml_add rhs")?;
        let can_repeat = unsafe { ffi::ggml_can_repeat(rhs.raw.as_ptr(), lhs.raw.as_ptr()) };
        if !can_repeat {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_add rhs cannot broadcast to lhs shape",
            });
        }
        let raw =
            unsafe { ffi::ggml_add(self.context.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_add")
    }

    #[allow(dead_code)]
    pub(crate) fn sub(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sub")?;
        self.ensure_tensor_type(lhs, ffi::GGML_TYPE_F32, "ggml_sub lhs")?;
        self.ensure_tensor_type(rhs, ffi::GGML_TYPE_F32, "ggml_sub rhs")?;
        let can_repeat = unsafe { ffi::ggml_can_repeat(rhs.raw.as_ptr(), lhs.raw.as_ptr()) };
        if !can_repeat {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_sub rhs cannot broadcast to lhs shape",
            });
        }
        let raw =
            unsafe { ffi::ggml_sub(self.context.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sub")
    }

    pub(crate) fn mul(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_mul")?;
        self.ensure_tensor_type(lhs, ffi::GGML_TYPE_F32, "ggml_mul lhs")?;
        self.ensure_tensor_type(rhs, ffi::GGML_TYPE_F32, "ggml_mul rhs")?;
        let can_repeat = unsafe { ffi::ggml_can_repeat(rhs.raw.as_ptr(), lhs.raw.as_ptr()) };
        if !can_repeat {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_mul rhs cannot broadcast to lhs shape",
            });
        }
        let raw =
            unsafe { ffi::ggml_mul(self.context.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_mul")
    }

    #[allow(dead_code)]
    pub(crate) fn div(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_div")?;
        self.ensure_tensor_type(lhs, ffi::GGML_TYPE_F32, "ggml_div lhs")?;
        self.ensure_tensor_type(rhs, ffi::GGML_TYPE_F32, "ggml_div rhs")?;
        let can_repeat = unsafe { ffi::ggml_can_repeat(rhs.raw.as_ptr(), lhs.raw.as_ptr()) };
        if !can_repeat {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_div rhs cannot broadcast to lhs shape",
            });
        }
        let raw =
            unsafe { ffi::ggml_div(self.context.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_div")
    }

    #[allow(dead_code)]
    pub(crate) fn sqr(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sqr")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_sqr input")?;
        let raw = unsafe { ffi::ggml_sqr(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sqr")
    }

    #[allow(dead_code)]
    pub(crate) fn sqrt(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sqrt")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_sqrt input")?;
        let raw = unsafe { ffi::ggml_sqrt(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sqrt")
    }

    #[allow(dead_code)]
    pub(crate) fn log(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_log")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_log input")?;
        let raw = unsafe { ffi::ggml_log(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_log")
    }

    pub(crate) fn mul_mat(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_mul_mat")?;
        self.ensure_tensor_type_in(
            lhs,
            &[
                ffi::GGML_TYPE_F16,
                ffi::GGML_TYPE_F32,
                ffi::GGML_TYPE_Q4_0,
                ffi::GGML_TYPE_Q8_0,
                ffi::GGML_TYPE_Q3_K,
                ffi::GGML_TYPE_Q4_K,
                ffi::GGML_TYPE_Q5_K,
                ffi::GGML_TYPE_Q6_K,
            ],
            "ggml_mul_mat lhs",
        )?;
        self.ensure_tensor_type_in(
            rhs,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_mul_mat rhs",
        )?;
        self.ensure_tensor_not_transposed(lhs, "ggml_mul_mat lhs")?;
        self.ensure_can_mul_mat(lhs, rhs)?;
        let raw =
            unsafe { ffi::ggml_mul_mat(self.context.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_mul_mat")
    }

    pub(crate) fn get_rows(
        &self,
        embeddings: GgmlCpuTensor<'a>,
        row_indices: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_get_rows")?;
        self.ensure_tensor_type_in(
            embeddings,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_get_rows embeddings",
        )?;
        self.ensure_tensor_type(row_indices, ffi::GGML_TYPE_I32, "ggml_get_rows indices")?;
        self.ensure_tensor_contiguous(row_indices, "ggml_get_rows indices")?;

        let embeddings_shape = self.tensor_shape_4d(embeddings)?;
        let indices_shape = self.tensor_shape_4d(row_indices)?;
        if indices_shape[3] != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_get_rows indices.ne3 must equal 1",
            });
        }
        if embeddings_shape[2] != indices_shape[1] || embeddings_shape[3] != indices_shape[2] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_get_rows batch dimensions are incompatible",
            });
        }

        let raw = unsafe {
            ffi::ggml_get_rows(
                self.context.as_ptr(),
                embeddings.raw.as_ptr(),
                row_indices.raw.as_ptr(),
            )
        };
        self.new_tensor_checked(raw, "ggml_get_rows")
    }

    pub(crate) fn top1_argmax(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        let shape = self.tensor_shape_4d(input)?;
        if shape[1] != 1 || shape[2] != 1 || shape[3] != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_top1_argmax input must have exactly one row group",
            });
        }
        self.ensure_can_extend_graph("ggml_argmax")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_argmax input")?;
        let raw = unsafe { ffi::ggml_argmax(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_argmax")
    }

    /// OpenASR greedy top-1 uses first-max tie semantics. Native ggml argmax
    /// returns the last exact max, so reverse the single logits column first and
    /// let the caller map the returned reversed index back to the original id.
    pub(crate) fn top1_argmax_first_max_reversed(
        &self,
        input: GgmlCpuTensor<'a>,
        reverse_indices: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        let shape = self.tensor_shape_4d(input)?;
        if shape[1] != 1 || shape[2] != 1 || shape[3] != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_first_max_argmax input must have exactly one row group",
            });
        }
        let index_shape = self.tensor_shape_4d(reverse_indices)?;
        if index_shape != [shape[0], 1, 1, 1] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_first_max_argmax reverse index shape mismatch",
            });
        }
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_first_max_argmax input")?;
        self.ensure_tensor_type(
            reverse_indices,
            ffi::GGML_TYPE_I32,
            "ggml_first_max_argmax reverse indices",
        )?;

        let logits_as_rows = self.reshape_2d(input, 1, shape[0])?;
        let reversed = self.get_rows(logits_as_rows, reverse_indices)?;
        let reversed = self.cont(reversed)?;
        let reversed = self.reshape_2d(reversed, shape[0], 1)?;
        self.top1_argmax(reversed)
    }

    #[cfg(test)]
    pub(crate) fn top_k(
        &self,
        input: GgmlCpuTensor<'a>,
        k: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_top_k")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_top_k input")?;
        let shape = self.tensor_shape_4d(input)?;
        if k == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_top_k requires k > 0",
            });
        }
        if k > shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_top_k requires input.ne0 >= k",
            });
        }
        let k = i32::try_from(k).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "ggml_top_k k exceeds ggml int boundary",
        })?;
        let raw = unsafe { ffi::ggml_top_k(self.context.as_ptr(), input.raw.as_ptr(), k) };
        self.new_tensor_checked(raw, "ggml_top_k")
    }

    pub(crate) fn scale(
        &self,
        input: GgmlCpuTensor<'a>,
        scalar: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_scale")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_scale input")?;
        self.ensure_tensor_contiguous(input, "ggml_scale")?;
        let raw = unsafe { ffi::ggml_scale(self.context.as_ptr(), input.raw.as_ptr(), scalar) };
        self.new_tensor_checked(raw, "ggml_scale")
    }

    #[allow(dead_code)]
    pub(crate) fn sum(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sum")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_sum input")?;
        let raw = unsafe { ffi::ggml_sum(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sum")
    }

    #[allow(dead_code)]
    pub(crate) fn sum_rows(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sum_rows")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_sum_rows input")?;
        let raw = unsafe { ffi::ggml_sum_rows(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sum_rows")
    }

    #[allow(dead_code)]
    pub(crate) fn mean_rows(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_mean")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_mean input")?;
        let raw = unsafe { ffi::ggml_mean(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_mean")
    }

    pub(crate) fn norm(
        &self,
        input: GgmlCpuTensor<'a>,
        eps: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_norm")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_norm input")?;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_norm epsilon must be finite and positive",
            });
        }
        let raw = unsafe { ffi::ggml_norm(self.context.as_ptr(), input.raw.as_ptr(), eps) };
        self.new_tensor_checked(raw, "ggml_norm")
    }

    /// GroupNorm over `ne0*ne1*n_groups` (ggml semantics): the channel axis is
    /// `ne2`, split into `n_groups` groups. wav2vec2 base's first feature-extractor
    /// conv layer uses `feat_extract_norm=="group"` with `n_groups == n_channels`
    /// (per-channel instance norm). The gamma/beta affine is applied separately by
    /// the caller. Input must be f32.
    pub(crate) fn group_norm(
        &self,
        input: GgmlCpuTensor<'a>,
        n_groups: usize,
        eps: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_group_norm")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_group_norm input")?;
        if n_groups == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_group_norm n_groups must be positive",
            });
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_group_norm epsilon must be finite and positive",
            });
        }
        let n_groups =
            i32::try_from(n_groups).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_group_norm n_groups exceeds ggml int boundary",
            })?;
        let raw = unsafe {
            ffi::ggml_group_norm(self.context.as_ptr(), input.raw.as_ptr(), n_groups, eps)
        };
        self.new_tensor_checked(raw, "ggml_group_norm")
    }

    /// Concatenate two tensors along `dim`. Used to stitch the per-group
    /// `conv_1d` outputs of the wav2vec2 grouped positional conv back into one
    /// `[out_channels, T]` tensor (concat along the channel axis, dim 1). Both
    /// inputs must be f32 and contiguous, and match in every axis except `dim`.
    pub(crate) fn concat(
        &self,
        a: GgmlCpuTensor<'a>,
        b: GgmlCpuTensor<'a>,
        dim: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_concat")?;
        self.ensure_tensor_type(a, ffi::GGML_TYPE_F32, "ggml_concat a")?;
        self.ensure_tensor_type(b, ffi::GGML_TYPE_F32, "ggml_concat b")?;
        self.ensure_tensor_contiguous(a, "ggml_concat a")?;
        self.ensure_tensor_contiguous(b, "ggml_concat b")?;
        if dim >= ffi::GGML_MAX_DIMS {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_concat dim out of range",
            });
        }
        let shape_a = self.tensor_shape_4d(a)?;
        let shape_b = self.tensor_shape_4d(b)?;
        for axis in 0..ffi::GGML_MAX_DIMS {
            if axis != dim && shape_a[axis] != shape_b[axis] {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_concat inputs must match in every axis except `dim`",
                });
            }
        }
        let dim = i32::try_from(dim).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "ggml_concat dim exceeds ggml int boundary",
        })?;
        let raw =
            unsafe { ffi::ggml_concat(self.context.as_ptr(), a.raw.as_ptr(), b.raw.as_ptr(), dim) };
        self.new_tensor_checked(raw, "ggml_concat")
    }

    #[allow(dead_code)]
    pub(crate) fn rms_norm(
        &self,
        input: GgmlCpuTensor<'a>,
        eps: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_rms_norm")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_rms_norm input")?;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rms_norm epsilon must be finite and positive",
            });
        }
        let raw = unsafe { ffi::ggml_rms_norm(self.context.as_ptr(), input.raw.as_ptr(), eps) };
        self.new_tensor_checked(raw, "ggml_rms_norm")
    }

    #[allow(dead_code)]
    pub(crate) fn repeat(
        &self,
        input: GgmlCpuTensor<'a>,
        target: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_repeat")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_repeat input")?;
        self.ensure_tensor_type(target, ffi::GGML_TYPE_F32, "ggml_repeat target")?;
        let can_repeat = unsafe { ffi::ggml_can_repeat(input.raw.as_ptr(), target.raw.as_ptr()) };
        if !can_repeat {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_repeat input cannot broadcast to target shape",
            });
        }
        let raw = unsafe {
            ffi::ggml_repeat(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                target.raw.as_ptr(),
            )
        };
        self.new_tensor_checked(raw, "ggml_repeat")
    }

    #[allow(dead_code)]
    pub(crate) fn repeat_4d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_repeat_4d")?;
        // F16 is exercised by f16 KV caches on the non-native-GQA expand path;
        // the vendored CPU op handles both (ggml_compute_forward_repeat_f16).
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_repeat_4d input",
        )?;
        let input_shape = self.tensor_shape_4d(input)?;
        let target_shape = [ne0, ne1, ne2, ne3];
        if target_shape
            .iter()
            .zip(input_shape.iter())
            .any(|(target, source)| *target % *source != 0)
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_repeat_4d target shape must be a multiple of input shape",
            });
        }
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let ne3_i64 = checked_dim_to_i64(ne3)?;
        let raw = unsafe {
            ffi::ggml_repeat_4d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                ne2_i64,
                ne3_i64,
            )
        };
        self.new_tensor_checked(raw, "ggml_repeat_4d")
    }

    pub(crate) fn soft_max(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_soft_max")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_soft_max input")?;
        self.ensure_tensor_contiguous(input, "ggml_soft_max")?;
        let raw = unsafe { ffi::ggml_soft_max(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_soft_max")
    }

    pub(crate) fn soft_max_ext(
        &self,
        input: GgmlCpuTensor<'a>,
        mask: Option<GgmlCpuTensor<'a>>,
        scale: f32,
        max_bias: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_soft_max_ext")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_soft_max_ext input")?;
        self.ensure_tensor_contiguous(input, "ggml_soft_max_ext")?;
        if !scale.is_finite() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext scale must be finite",
            });
        }
        if !max_bias.is_finite() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext max_bias must be finite",
            });
        }
        if max_bias > 0.0 && mask.is_none() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext max_bias > 0 requires a mask tensor",
            });
        }

        let mask_raw = if let Some(mask) = mask {
            self.ensure_tensor_contiguous(mask, "ggml_soft_max_ext mask")?;
            self.ensure_tensor_type_in(
                mask,
                &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
                "ggml_soft_max_ext mask",
            )?;
            self.ensure_soft_max_ext_mask_compatible(input, mask)?;
            mask.raw.as_ptr()
        } else {
            ptr::null_mut()
        };

        let raw = unsafe {
            ffi::ggml_soft_max_ext(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                mask_raw,
                scale,
                max_bias,
            )
        };
        self.new_tensor_checked(raw, "ggml_soft_max_ext")
    }

    pub(crate) fn flash_attn_ext(
        &self,
        q: GgmlCpuTensor<'a>,
        k: GgmlCpuTensor<'a>,
        v: GgmlCpuTensor<'a>,
        mask: Option<GgmlCpuTensor<'a>>,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_flash_attn_ext")?;
        self.ensure_tensor_type(q, ffi::GGML_TYPE_F32, "ggml_flash_attn_ext q")?;
        self.ensure_tensor_type_in(
            k,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_flash_attn_ext k",
        )?;
        self.ensure_tensor_type_in(
            v,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_flash_attn_ext v",
        )?;
        if self.tensor_type(k) != self.tensor_type(v) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext requires matching k/v tensor types",
            });
        }
        self.ensure_tensor_contiguous_rows(q, "ggml_flash_attn_ext q")?;
        self.ensure_tensor_contiguous_rows(k, "ggml_flash_attn_ext k")?;
        self.ensure_tensor_contiguous_rows(v, "ggml_flash_attn_ext v")?;

        if !(scale.is_finite() && max_bias.is_finite() && logit_softcap.is_finite()) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext params must be finite",
            });
        }
        if max_bias > 0.0 && mask.is_none() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext max_bias > 0 requires a mask tensor",
            });
        }
        self.ensure_flash_attn_ext_compatible(q, k, v)?;
        self.ensure_flash_attn_ext_metal_head_dim_supported(q)?;

        let mask_raw = if let Some(mask) = mask {
            self.ensure_tensor_type(mask, ffi::GGML_TYPE_F16, "ggml_flash_attn_ext mask")?;
            self.ensure_tensor_contiguous(mask, "ggml_flash_attn_ext mask")?;
            self.ensure_flash_attn_ext_mask_compatible(q, mask)?;
            mask.raw.as_ptr()
        } else {
            ptr::null_mut()
        };

        let raw = unsafe {
            ffi::ggml_flash_attn_ext(
                self.context.as_ptr(),
                q.raw.as_ptr(),
                k.raw.as_ptr(),
                v.raw.as_ptr(),
                mask_raw,
                scale,
                max_bias,
                logit_softcap,
            )
        };
        self.new_tensor_checked(raw, "ggml_flash_attn_ext")
    }

    pub(crate) fn gelu(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_gelu")?;
        let raw = unsafe { ffi::ggml_gelu(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_gelu")
    }

    pub(crate) fn gelu_erf(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_gelu_erf")?;
        let raw = unsafe { ffi::ggml_gelu_erf(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_gelu_erf")
    }

    pub(crate) fn tanh(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_tanh")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_tanh input")?;
        let raw = unsafe { ffi::ggml_tanh(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_tanh")
    }

    pub(crate) fn relu(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_relu")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_relu input")?;
        let raw = unsafe { ffi::ggml_relu(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_relu")
    }

    pub(crate) fn sigmoid(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_sigmoid")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_sigmoid input")?;
        let raw = unsafe { ffi::ggml_sigmoid(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_sigmoid")
    }

    #[allow(dead_code)]
    pub(crate) fn softplus(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_softplus")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_softplus input")?;
        let raw = unsafe { ffi::ggml_softplus(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_softplus")
    }

    #[allow(dead_code)]
    pub(crate) fn exp(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_exp")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_exp input")?;
        let raw = unsafe { ffi::ggml_exp(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_exp")
    }

    #[allow(dead_code)]
    pub(crate) fn silu(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_silu")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_silu input")?;
        let raw = unsafe { ffi::ggml_silu(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_silu")
    }

    #[allow(dead_code)]
    pub(crate) fn cast(
        &self,
        input: GgmlCpuTensor<'a>,
        target_type: i32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_cast")?;
        self.ensure_tensor_type_in(
            input,
            &[
                ffi::GGML_TYPE_F16,
                ffi::GGML_TYPE_F32,
                ffi::GGML_TYPE_Q8_0,
                ffi::GGML_TYPE_Q4_K,
            ],
            "ggml_cast input",
        )?;
        self.ensure_supported_storage_type(target_type, "ggml_cast target")?;
        let raw = unsafe { ffi::ggml_cast(self.context.as_ptr(), input.raw.as_ptr(), target_type) };
        self.new_tensor_checked(raw, "ggml_cast")
    }

    /// `cast` specialized to an F16 target, exposed so callers outside this
    /// module (`ffi::GGML_TYPE_F16` is private to `ggml_runtime`) can convert
    /// an additive f32 attention mask to the F16 type `flash_attn_ext`
    /// requires without reaching into the ffi layer themselves.
    pub(crate) fn cast_to_f16(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.cast(input, ffi::GGML_TYPE_F16)
    }

    pub(crate) fn cont(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_cont")?;
        let raw = unsafe { ffi::ggml_cont(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_cont")
    }

    pub(crate) fn reshape_2d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_reshape_2d")?;
        self.ensure_tensor_contiguous(input, "ggml_reshape_2d")?;
        let expected_nelements =
            ne0.checked_mul(ne1)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "reshape_2d target shape overflows usize",
                })?;
        self.ensure_tensor_nelements(input, expected_nelements, "ggml_reshape_2d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let raw = unsafe {
            ffi::ggml_reshape_2d(self.context.as_ptr(), input.raw.as_ptr(), ne0_i64, ne1_i64)
        };
        self.new_tensor_checked(raw, "ggml_reshape_2d")
    }

    pub(crate) fn reshape_1d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_reshape_1d")?;
        self.ensure_tensor_contiguous(input, "ggml_reshape_1d")?;
        self.ensure_tensor_nelements(input, ne0, "ggml_reshape_1d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let raw =
            unsafe { ffi::ggml_reshape_1d(self.context.as_ptr(), input.raw.as_ptr(), ne0_i64) };
        self.new_tensor_checked(raw, "ggml_reshape_1d")
    }

    pub(crate) fn reshape_3d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_reshape_3d")?;
        self.ensure_tensor_contiguous(input, "ggml_reshape_3d")?;
        let expected_nelements = ne0
            .checked_mul(ne1)
            .and_then(|value| value.checked_mul(ne2))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "reshape_3d target shape overflows usize",
            })?;
        self.ensure_tensor_nelements(input, expected_nelements, "ggml_reshape_3d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let raw = unsafe {
            ffi::ggml_reshape_3d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                ne2_i64,
            )
        };
        self.new_tensor_checked(raw, "ggml_reshape_3d")
    }

    pub(crate) fn reshape_4d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_reshape_4d")?;
        self.ensure_tensor_contiguous(input, "ggml_reshape_4d")?;
        let expected_nelements = ne0
            .checked_mul(ne1)
            .and_then(|value| value.checked_mul(ne2))
            .and_then(|value| value.checked_mul(ne3))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "reshape_4d target shape overflows usize",
            })?;
        self.ensure_tensor_nelements(input, expected_nelements, "ggml_reshape_4d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let ne3_i64 = checked_dim_to_i64(ne3)?;
        let raw = unsafe {
            ffi::ggml_reshape_4d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                ne2_i64,
                ne3_i64,
            )
        };
        self.new_tensor_checked(raw, "ggml_reshape_4d")
    }

    pub(crate) fn view_1d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        offset: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_view_1d")?;
        self.ensure_tensor_contiguous(input, "ggml_view_1d")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_view_1d input",
        )?;
        let element_size = self.element_size_bytes(self.tensor_type(input))?;
        if !offset.is_multiple_of(element_size) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_1d offset must align to tensor element size",
            });
        }
        let start = offset / element_size;
        let span = ne0
            .checked_add(start)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_1d span overflows usize",
            })?;
        self.ensure_tensor_nelements_at_least(input, span, "ggml_view_1d")?;
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let raw = unsafe {
            ffi::ggml_view_1d(self.context.as_ptr(), input.raw.as_ptr(), ne0_i64, offset)
        };
        self.new_tensor_checked(raw, "ggml_view_1d")
    }

    pub(crate) fn view_3d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_view_3d")?;
        self.ensure_tensor_contiguous(input, "ggml_view_3d")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_view_3d input",
        )?;
        let element_size = self.element_size_bytes(self.tensor_type(input))?;
        let min_nb1 =
            ne0.checked_mul(element_size)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_3d nb1 overflows usize",
                })?;
        if nb1 < min_nb1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d row stride overlaps rows",
            });
        }
        let min_nb2 = ne1
            .checked_mul(nb1)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d nb2 overflows usize",
            })?;
        if nb2 < min_nb2 && nb2 < min_nb1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d plane stride overlaps planes",
            });
        }
        if !offset.is_multiple_of(element_size) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d offset must align to tensor element size",
            });
        }
        let last_byte_offset = offset
            .checked_add((ne0.saturating_sub(1)).checked_mul(element_size).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_3d shape overflows usize",
                },
            )?)
            .and_then(|value| value.checked_add((ne1.saturating_sub(1)).checked_mul(nb1)?))
            .and_then(|value| value.checked_add((ne2.saturating_sub(1)).checked_mul(nb2)?))
            .and_then(|value| value.checked_add(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d span overflows usize",
            })?;
        if last_byte_offset > self.tensor_nbytes(input) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_3d span exceeds source tensor bytes",
            });
        }
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let raw = unsafe {
            ffi::ggml_view_3d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                ne2_i64,
                nb1,
                nb2,
                offset,
            )
        };
        self.new_tensor_checked(raw, "ggml_view_3d")
    }

    #[allow(dead_code)]
    pub(crate) fn view_4d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_view_4d")?;
        self.ensure_tensor_contiguous(input, "ggml_view_4d")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_view_4d input",
        )?;
        let element_size = self.element_size_bytes(self.tensor_type(input))?;
        let min_nb1 =
            ne0.checked_mul(element_size)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_4d nb1 overflows usize",
                })?;
        if nb1 < min_nb1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d row stride overlaps rows",
            });
        }
        let min_nb2 = ne1
            .checked_mul(nb1)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d nb2 overflows usize",
            })?;
        if nb2 < min_nb2 && nb2 < min_nb1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d plane stride overlaps planes",
            });
        }
        let min_nb3 = ne2
            .checked_mul(nb2)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d nb3 overflows usize",
            })?;
        if nb3 < min_nb3 && nb3 < min_nb2 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d hyperplane stride overlaps hyperplanes",
            });
        }
        if !offset.is_multiple_of(element_size) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d offset must align to tensor element size",
            });
        }
        let last_byte_offset = offset
            .checked_add((ne0.saturating_sub(1)).checked_mul(element_size).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_4d shape overflows usize",
                },
            )?)
            .and_then(|value| value.checked_add((ne1.saturating_sub(1)).checked_mul(nb1)?))
            .and_then(|value| value.checked_add((ne2.saturating_sub(1)).checked_mul(nb2)?))
            .and_then(|value| value.checked_add((ne3.saturating_sub(1)).checked_mul(nb3)?))
            .and_then(|value| value.checked_add(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d span overflows usize",
            })?;
        if last_byte_offset > self.tensor_nbytes(input) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_4d span exceeds source tensor bytes",
            });
        }
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let ne2_i64 = checked_dim_to_i64(ne2)?;
        let ne3_i64 = checked_dim_to_i64(ne3)?;
        let raw = unsafe {
            ffi::ggml_view_4d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                ne2_i64,
                ne3_i64,
                nb1,
                nb2,
                nb3,
                offset,
            )
        };
        self.new_tensor_checked(raw, "ggml_view_4d")
    }

    pub(crate) fn view_2d(
        &self,
        input: GgmlCpuTensor<'a>,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_view_2d")?;
        self.ensure_tensor_contiguous(input, "ggml_view_2d")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_view_2d input",
        )?;
        let element_size = self.element_size_bytes(self.tensor_type(input))?;
        let min_nb1 =
            ne0.checked_mul(element_size)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_2d row stride overflows usize",
                })?;
        if nb1 < min_nb1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d row stride overlaps rows",
            });
        }
        if !offset.is_multiple_of(element_size) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d offset must align to tensor element size",
            });
        }
        let last_byte_offset = offset
            .checked_add((ne0.saturating_sub(1)).checked_mul(element_size).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_view_2d shape overflows usize",
                },
            )?)
            .and_then(|value| value.checked_add((ne1.saturating_sub(1)).checked_mul(nb1)?))
            .and_then(|value| value.checked_add(element_size))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d span overflows usize",
            })?;
        if last_byte_offset > self.tensor_nbytes(input) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_view_2d span exceeds source tensor bytes",
            });
        }
        let ne0_i64 = checked_dim_to_i64(ne0)?;
        let ne1_i64 = checked_dim_to_i64(ne1)?;
        let raw = unsafe {
            ffi::ggml_view_2d(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                ne0_i64,
                ne1_i64,
                nb1,
                offset,
            )
        };
        self.new_tensor_checked(raw, "ggml_view_2d")
    }

    pub(crate) fn cpy(
        &self,
        src: GgmlCpuTensor<'a>,
        dst: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_cpy")?;
        let src_shape = self.tensor_shape_4d(src)?;
        let dst_shape = self.tensor_shape_4d(dst)?;
        if src_shape != dst_shape {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_cpy requires src and dst shape match",
            });
        }
        let src_type = self.tensor_type(src);
        let dst_type = self.tensor_type(dst);
        let supported_type_pair = src_type == dst_type
            || (src_type == ffi::GGML_TYPE_F32 && dst_type == ffi::GGML_TYPE_F16)
            || (src_type == ffi::GGML_TYPE_F16 && dst_type == ffi::GGML_TYPE_F32);
        if !supported_type_pair {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_cpy supports matching types or f32/f16 conversion",
            });
        }
        self.ensure_tensor_contiguous(src, "ggml_cpy src")?;
        self.ensure_tensor_contiguous(dst, "ggml_cpy dst")?;
        let raw =
            unsafe { ffi::ggml_cpy(self.context.as_ptr(), src.raw.as_ptr(), dst.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_cpy")
    }

    #[allow(dead_code)]
    pub(crate) fn cpy_into_view(
        &self,
        src: GgmlCpuTensor<'a>,
        dst: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_cpy")?;
        self.ensure_cpy_compatible(src, dst)?;
        self.ensure_tensor_contiguous(src, "ggml_cpy src")?;
        let raw =
            unsafe { ffi::ggml_cpy(self.context.as_ptr(), src.raw.as_ptr(), dst.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_cpy")
    }

    #[allow(dead_code)]
    pub(crate) fn set_rows(
        &self,
        dst: GgmlCpuTensor<'a>,
        src: GgmlCpuTensor<'a>,
        row_indices: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_set_rows")?;
        self.ensure_tensor_type_in(
            dst,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_set_rows dst",
        )?;
        self.ensure_tensor_type(src, ffi::GGML_TYPE_F32, "ggml_set_rows src")?;
        self.ensure_tensor_type(row_indices, ffi::GGML_TYPE_I32, "ggml_set_rows indices")?;
        self.ensure_tensor_contiguous(src, "ggml_set_rows src")?;
        self.ensure_tensor_contiguous(row_indices, "ggml_set_rows indices")?;
        self.ensure_set_rows_compatible(dst, src, row_indices)?;
        let raw = unsafe {
            ffi::ggml_set_rows(
                self.context.as_ptr(),
                dst.raw.as_ptr(),
                src.raw.as_ptr(),
                row_indices.raw.as_ptr(),
            )
        };
        self.new_tensor_checked(raw, "ggml_set_rows")
    }

    pub(crate) fn add_side_effect_root(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_build_forward_expand(side_effect)")?;
        self.side_effect_roots.push(tensor.raw);
        Ok(())
    }

    pub(crate) fn transpose(
        &self,
        input: GgmlCpuTensor<'a>,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_transpose")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_transpose input",
        )?;
        let raw = unsafe { ffi::ggml_transpose(self.context.as_ptr(), input.raw.as_ptr()) };
        self.new_tensor_checked(raw, "ggml_transpose")
    }

    pub(crate) fn permute(
        &self,
        input: GgmlCpuTensor<'a>,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_permute")?;
        self.ensure_tensor_type_in(
            input,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32, ffi::GGML_TYPE_I32],
            "ggml_permute input",
        )?;
        self.validate_permute_axes(axis0, axis1, axis2, axis3)?;
        let raw = unsafe {
            ffi::ggml_permute(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                axis0,
                axis1,
                axis2,
                axis3,
            )
        };
        self.new_tensor_checked(raw, "ggml_permute")
    }

    #[allow(dead_code)]
    pub(crate) fn rope_ext(
        &self,
        input: GgmlCpuTensor<'a>,
        positions: GgmlCpuTensor<'a>,
        params: GgmlRopeExtParams,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_rope_ext")?;
        self.ensure_tensor_type(input, ffi::GGML_TYPE_F32, "ggml_rope_ext input")?;
        self.ensure_tensor_type(positions, ffi::GGML_TYPE_I32, "ggml_rope_ext positions")?;
        self.ensure_tensor_contiguous(positions, "ggml_rope_ext positions")?;
        if params.n_dims == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext n_dims must be positive",
            });
        }
        let n_dims =
            i32::try_from(params.n_dims).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext n_dims exceeds ggml int boundary",
            })?;
        if ![
            params.freq_base,
            params.freq_scale,
            params.ext_factor,
            params.attn_factor,
            params.beta_fast,
            params.beta_slow,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext float params must be finite",
            });
        }
        if !(params.freq_base > 0.0 && params.freq_scale > 0.0 && params.attn_factor > 0.0) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext frequency and attention params must be positive",
            });
        }
        self.ensure_rope_ext_compatible(input, positions, params.n_dims)?;
        let raw = unsafe {
            ffi::ggml_rope_ext(
                self.context.as_ptr(),
                input.raw.as_ptr(),
                positions.raw.as_ptr(),
                ptr::null_mut(),
                n_dims,
                params.mode,
                params.n_ctx_orig,
                params.freq_base,
                params.freq_scale,
                params.ext_factor,
                params.attn_factor,
                params.beta_fast,
                params.beta_slow,
            )
        };
        self.new_tensor_checked(raw, "ggml_rope_ext")
    }

    pub(crate) fn conv_1d(
        &self,
        kernel: GgmlCpuTensor<'a>,
        data: GgmlCpuTensor<'a>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_conv_1d")?;
        let stride = i32::try_from(stride).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv stride exceeds ggml int boundary",
        })?;
        let padding = i32::try_from(padding).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv padding exceeds ggml int boundary",
        })?;
        let dilation =
            i32::try_from(dilation).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv dilation exceeds ggml int boundary",
            })?;
        let raw = unsafe {
            ffi::ggml_conv_1d(
                self.context.as_ptr(),
                kernel.raw.as_ptr(),
                data.raw.as_ptr(),
                stride,
                padding,
                dilation,
            )
        };
        self.new_tensor_checked(raw, "ggml_conv_1d")
    }

    pub(crate) fn conv_2d(
        &self,
        kernel: GgmlCpuTensor<'a>,
        data: GgmlCpuTensor<'a>,
        stride0: usize,
        stride1: usize,
        padding0: usize,
        padding1: usize,
        dilation0: usize,
        dilation1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_conv_2d")?;
        self.ensure_tensor_type_in(
            kernel,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_conv_2d kernel",
        )?;
        self.ensure_tensor_type(data, ffi::GGML_TYPE_F32, "ggml_conv_2d data")?;
        let stride0 = i32::try_from(stride0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv stride exceeds ggml int boundary",
        })?;
        let stride1 = i32::try_from(stride1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv stride exceeds ggml int boundary",
        })?;
        let padding0 =
            i32::try_from(padding0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv padding exceeds ggml int boundary",
            })?;
        let padding1 =
            i32::try_from(padding1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv padding exceeds ggml int boundary",
            })?;
        let dilation0 =
            i32::try_from(dilation0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv dilation exceeds ggml int boundary",
            })?;
        let dilation1 =
            i32::try_from(dilation1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv dilation exceeds ggml int boundary",
            })?;
        let raw = unsafe {
            ffi::ggml_conv_2d(
                self.context.as_ptr(),
                kernel.raw.as_ptr(),
                data.raw.as_ptr(),
                stride0,
                stride1,
                padding0,
                padding1,
                dilation0,
                dilation1,
            )
        };
        self.new_tensor_checked(raw, "ggml_conv_2d")
    }

    pub(crate) fn conv_2d_dw_direct(
        &self,
        kernel: GgmlCpuTensor<'a>,
        data: GgmlCpuTensor<'a>,
        stride0: usize,
        stride1: usize,
        padding0: usize,
        padding1: usize,
        dilation0: usize,
        dilation1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.ensure_can_extend_graph("ggml_conv_2d_dw_direct")?;
        self.ensure_tensor_type_in(
            kernel,
            &[ffi::GGML_TYPE_F16, ffi::GGML_TYPE_F32],
            "ggml_conv_2d_dw_direct kernel",
        )?;
        self.ensure_tensor_type(data, ffi::GGML_TYPE_F32, "ggml_conv_2d_dw_direct data")?;
        let stride0 = i32::try_from(stride0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv stride exceeds ggml int boundary",
        })?;
        let stride1 = i32::try_from(stride1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "conv stride exceeds ggml int boundary",
        })?;
        let padding0 =
            i32::try_from(padding0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv padding exceeds ggml int boundary",
            })?;
        let padding1 =
            i32::try_from(padding1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv padding exceeds ggml int boundary",
            })?;
        let dilation0 =
            i32::try_from(dilation0).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv dilation exceeds ggml int boundary",
            })?;
        let dilation1 =
            i32::try_from(dilation1).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "conv dilation exceeds ggml int boundary",
            })?;
        let raw = unsafe {
            ffi::ggml_conv_2d_dw_direct(
                self.context.as_ptr(),
                kernel.raw.as_ptr(),
                data.raw.as_ptr(),
                stride0,
                stride1,
                padding0,
                padding1,
                dilation0,
                dilation1,
            )
        };
        self.new_tensor_checked(raw, "ggml_conv_2d_dw_direct")
    }

    /// Depthwise 2D conv via the fused `GGML_OP_CONV_2D_DW` op
    /// (`ggml_conv_2d_dw_direct`). The Metal backend has a native CONV_2D_DW kernel
    /// (F16 or F32 kernel weights), so this is a single direct op on every backend
    /// -- no F32->F16 cast and no IM2COL + MUL_MAT expansion. Callers stay
    /// backend-agnostic.
    pub(crate) fn depthwise_conv_2d(
        &self,
        kernel: GgmlCpuTensor<'a>,
        data: GgmlCpuTensor<'a>,
        stride0: usize,
        stride1: usize,
        padding0: usize,
        padding1: usize,
        dilation0: usize,
        dilation1: usize,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        self.conv_2d_dw_direct(
            kernel, data, stride0, stride1, padding0, padding1, dilation0, dilation1,
        )
    }

    #[cfg(test)]
    pub(crate) fn set_f32_1d(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        index: usize,
        value: f32,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let len = self.tensor_len_f32(tensor)?;
        if index >= len {
            return Err(GgmlCpuGraphError::TensorIndexOutOfBounds { index, len });
        }
        let offset =
            index
                .checked_mul(F32_WIDTH_BYTES)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte offset overflow",
                })?;
        unsafe {
            write_tensor_data(
                tensor.raw,
                (&value as *const f32).cast::<c_void>(),
                offset,
                F32_WIDTH_BYTES,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn get_f32_1d(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        index: usize,
    ) -> Result<f32, GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let len = self.tensor_len_f32(tensor)?;
        if index >= len {
            return Err(GgmlCpuGraphError::TensorIndexOutOfBounds { index, len });
        }
        let offset =
            index
                .checked_mul(F32_WIDTH_BYTES)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte offset overflow",
                })?;
        let mut value = 0.0f32;
        unsafe {
            ffi::ggml_backend_tensor_get(
                tensor.raw.as_ptr(),
                (&mut value as *mut f32).cast::<c_void>(),
                offset,
                F32_WIDTH_BYTES,
            );
        }
        Ok(value)
    }

    pub(crate) fn set_f32_slice(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        values: &[f32],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let expected = values.len().checked_mul(F32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
        )
    }

    pub(crate) fn set_f32_slice_with_offset(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        offset_elements: usize,
        values: &[f32],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let offset_bytes = offset_elements.checked_mul(F32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte offset overflow",
            },
        )?;
        let expected = values.len().checked_mul(F32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            offset_bytes,
            expected,
            tensor_name,
        )
    }

    pub(crate) fn prepare_outputs_for_upload(
        &mut self,
        outputs: &[GgmlCpuTensor<'a>],
    ) -> Result<(), GgmlCpuGraphError> {
        if self.prepared_graph.is_some() {
            return Ok(());
        }
        let graph = self.build_forward_graph(outputs)?;
        if let Some(scheduler) = self.scheduler {
            unsafe { ffi::ggml_backend_sched_reset(scheduler.as_ptr()) };
            let allocated =
                unsafe { ffi::ggml_backend_sched_alloc_graph(scheduler.as_ptr(), graph.as_ptr()) };
            if !allocated {
                return Err(GgmlCpuGraphError::BackendSchedulerGraphAllocationFailed);
            }
        } else {
            self.ensure_backend_buffer()?;
        }
        self.prepared_graph = Some(graph);
        Ok(())
    }

    pub(crate) fn prepare_side_effects_for_upload(&mut self) -> Result<(), GgmlCpuGraphError> {
        if self.prepared_graph.is_some() {
            return Ok(());
        }
        if self.side_effect_roots.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "at least one side-effect root is required",
            });
        }
        let graph = self.build_forward_graph(&[])?;
        if let Some(scheduler) = self.scheduler {
            unsafe { ffi::ggml_backend_sched_reset(scheduler.as_ptr()) };
            let allocated =
                unsafe { ffi::ggml_backend_sched_alloc_graph(scheduler.as_ptr(), graph.as_ptr()) };
            if !allocated {
                return Err(GgmlCpuGraphError::BackendSchedulerGraphAllocationFailed);
            }
        } else {
            self.ensure_backend_buffer()?;
        }
        self.prepared_graph = Some(graph);
        Ok(())
    }

    pub(crate) fn set_f16_bits_slice(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        values: &[u16],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        let expected = values.len().checked_mul(F16_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
        )
    }

    pub(crate) fn set_bytes_slice(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        values: &[u8],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            values.len(),
            tensor_name,
        )
    }

    pub(crate) fn set_i32_slice(
        &mut self,
        tensor: GgmlCpuTensor<'a>,
        values: &[i32],
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_backend_buffer()?;
        self.ensure_tensor_type(tensor, ffi::GGML_TYPE_I32, "tensor_upload_i32")?;
        let expected = values.len().checked_mul(I32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        self.write_tensor_bytes_checked(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            expected,
            tensor_name,
        )
    }

    pub(crate) fn compute_output_f32(
        &mut self,
        output: GgmlCpuTensor<'a>,
        expected_len: usize,
    ) -> Result<Vec<f32>, GgmlCpuGraphError> {
        let mut outputs = self.compute_outputs_f32(&[(output, expected_len)])?;
        Ok(outputs.remove(0))
    }

    pub(crate) fn compute_outputs_f32(
        &mut self,
        outputs: &[(GgmlCpuTensor<'a>, usize)],
    ) -> Result<Vec<Vec<f32>>, GgmlCpuGraphError> {
        if outputs.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "at least one output tensor is required",
            });
        }
        let mut output_nbytes = Vec::with_capacity(outputs.len());
        for (output, expected_len) in outputs {
            self.ensure_tensor_type(*output, ffi::GGML_TYPE_F32, "compute_outputs_f32 output")?;
            let expected_nbytes = expected_len.checked_mul(F32_WIDTH_BYTES).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte width overflow",
                },
            )?;
            let actual_nbytes = self.tensor_nbytes(*output);
            if actual_nbytes != expected_nbytes {
                return Err(GgmlCpuGraphError::OutputByteSizeMismatch {
                    expected: expected_nbytes,
                    actual: actual_nbytes,
                });
            }
            output_nbytes.push(actual_nbytes);
        }

        let output_tensors = outputs
            .iter()
            .map(|(output, _)| *output)
            .collect::<Vec<_>>();
        let graph_prepared = self.prepared_graph.is_some();
        let graph = if let Some(graph) = self.prepared_graph {
            graph
        } else {
            self.ensure_backend_buffer()?;
            self.build_forward_graph(&output_tensors)?
        };

        let status = unsafe {
            if let Some(scheduler) = self.scheduler {
                if !graph_prepared {
                    ffi::ggml_backend_sched_reset(scheduler.as_ptr());
                }
                ffi::ggml_backend_sched_graph_compute(scheduler.as_ptr(), graph.as_ptr())
            } else {
                ffi::ggml_backend_graph_compute(self.backend.as_ptr(), graph.as_ptr())
            }
        };
        if status != ffi::GGML_STATUS_SUCCESS {
            return Err(GgmlCpuGraphError::ComputeFailed { status });
        }

        let mut results = Vec::with_capacity(outputs.len());
        for ((output, expected_len), output_nbytes) in outputs.iter().zip(output_nbytes) {
            let mut values = vec![0.0f32; *expected_len];
            unsafe {
                ffi::ggml_backend_tensor_get(
                    output.raw.as_ptr(),
                    values.as_mut_ptr().cast::<c_void>(),
                    0,
                    output_nbytes,
                );
            }
            results.push(values);
        }
        Ok(results)
    }

    pub(crate) fn compute_outputs_f32_i32(
        &mut self,
        f32_outputs: &[(GgmlCpuTensor<'a>, usize)],
        i32_outputs: &[(GgmlCpuTensor<'a>, usize)],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<i32>>), GgmlCpuGraphError> {
        if f32_outputs.is_empty() && i32_outputs.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "at least one output tensor is required",
            });
        }

        let mut f32_output_nbytes = Vec::with_capacity(f32_outputs.len());
        for (output, expected_len) in f32_outputs {
            self.ensure_tensor_type(
                *output,
                ffi::GGML_TYPE_F32,
                "compute_outputs_f32_i32 f32 output",
            )?;
            let expected_nbytes = expected_len.checked_mul(F32_WIDTH_BYTES).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte width overflow",
                },
            )?;
            let actual_nbytes = self.tensor_nbytes(*output);
            if actual_nbytes != expected_nbytes {
                return Err(GgmlCpuGraphError::OutputByteSizeMismatch {
                    expected: expected_nbytes,
                    actual: actual_nbytes,
                });
            }
            f32_output_nbytes.push(actual_nbytes);
        }

        let mut i32_output_nbytes = Vec::with_capacity(i32_outputs.len());
        for (output, expected_len) in i32_outputs {
            self.ensure_tensor_type(
                *output,
                ffi::GGML_TYPE_I32,
                "compute_outputs_f32_i32 i32 output",
            )?;
            let expected_nbytes = expected_len.checked_mul(I32_WIDTH_BYTES).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor byte width overflow",
                },
            )?;
            let actual_nbytes = self.tensor_nbytes(*output);
            if actual_nbytes != expected_nbytes {
                return Err(GgmlCpuGraphError::OutputByteSizeMismatch {
                    expected: expected_nbytes,
                    actual: actual_nbytes,
                });
            }
            i32_output_nbytes.push(actual_nbytes);
        }

        let mut output_tensors = Vec::with_capacity(f32_outputs.len() + i32_outputs.len());
        output_tensors.extend(f32_outputs.iter().map(|(output, _)| *output));
        output_tensors.extend(i32_outputs.iter().map(|(output, _)| *output));
        let graph_prepared = self.prepared_graph.is_some();
        let graph = if let Some(graph) = self.prepared_graph {
            graph
        } else {
            self.ensure_backend_buffer()?;
            self.build_forward_graph(&output_tensors)?
        };

        let status = unsafe {
            if let Some(scheduler) = self.scheduler {
                if !graph_prepared {
                    ffi::ggml_backend_sched_reset(scheduler.as_ptr());
                }
                ffi::ggml_backend_sched_graph_compute(scheduler.as_ptr(), graph.as_ptr())
            } else {
                ffi::ggml_backend_graph_compute(self.backend.as_ptr(), graph.as_ptr())
            }
        };
        if status != ffi::GGML_STATUS_SUCCESS {
            return Err(GgmlCpuGraphError::ComputeFailed { status });
        }

        let mut f32_results = Vec::with_capacity(f32_outputs.len());
        for ((output, expected_len), output_nbytes) in f32_outputs.iter().zip(f32_output_nbytes) {
            let mut values = vec![0.0f32; *expected_len];
            unsafe {
                ffi::ggml_backend_tensor_get(
                    output.raw.as_ptr(),
                    values.as_mut_ptr().cast::<c_void>(),
                    0,
                    output_nbytes,
                );
            }
            f32_results.push(values);
        }

        let mut i32_results = Vec::with_capacity(i32_outputs.len());
        for ((output, expected_len), output_nbytes) in i32_outputs.iter().zip(i32_output_nbytes) {
            let mut values = vec![0_i32; *expected_len];
            unsafe {
                ffi::ggml_backend_tensor_get(
                    output.raw.as_ptr(),
                    values.as_mut_ptr().cast::<c_void>(),
                    0,
                    output_nbytes,
                );
            }
            i32_results.push(values);
        }

        Ok((f32_results, i32_results))
    }

    pub(crate) fn compute_side_effects(&mut self) -> Result<(), GgmlCpuGraphError> {
        if self.side_effect_roots.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "at least one side-effect root is required",
            });
        }
        let graph_prepared = self.prepared_graph.is_some();
        let graph = if let Some(graph) = self.prepared_graph {
            graph
        } else {
            self.ensure_backend_buffer()?;
            self.build_forward_graph(&[])?
        };

        let status = unsafe {
            if let Some(scheduler) = self.scheduler {
                if !graph_prepared {
                    ffi::ggml_backend_sched_reset(scheduler.as_ptr());
                }
                ffi::ggml_backend_sched_graph_compute(scheduler.as_ptr(), graph.as_ptr())
            } else {
                ffi::ggml_backend_graph_compute(self.backend.as_ptr(), graph.as_ptr())
            }
        };
        if status != ffi::GGML_STATUS_SUCCESS {
            return Err(GgmlCpuGraphError::ComputeFailed { status });
        }
        Ok(())
    }

    pub(crate) fn compute_output_i32(
        &mut self,
        output: GgmlCpuTensor<'a>,
        expected_len: usize,
    ) -> Result<Vec<i32>, GgmlCpuGraphError> {
        self.ensure_tensor_type(output, ffi::GGML_TYPE_I32, "compute_output_i32 output")?;
        let expected_nbytes = expected_len.checked_mul(I32_WIDTH_BYTES).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte width overflow",
            },
        )?;
        let output_nbytes = self.tensor_nbytes(output);
        if output_nbytes != expected_nbytes {
            return Err(GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: expected_nbytes,
                actual: output_nbytes,
            });
        }

        let graph_prepared = self.prepared_graph.is_some();
        let graph = if let Some(graph) = self.prepared_graph {
            graph
        } else {
            self.ensure_backend_buffer()?;
            let graph = self.allocate_forward_graph("ggml_new_graph_custom")?;
            for root in &self.side_effect_roots {
                unsafe { ffi::ggml_build_forward_expand(graph.as_ptr(), root.as_ptr()) };
            }
            unsafe { ffi::ggml_build_forward_expand(graph.as_ptr(), output.raw.as_ptr()) };
            graph
        };

        let status = unsafe {
            if let Some(scheduler) = self.scheduler {
                if !graph_prepared {
                    ffi::ggml_backend_sched_reset(scheduler.as_ptr());
                }
                ffi::ggml_backend_sched_graph_compute(scheduler.as_ptr(), graph.as_ptr())
            } else {
                ffi::ggml_backend_graph_compute(self.backend.as_ptr(), graph.as_ptr())
            }
        };
        if status != ffi::GGML_STATUS_SUCCESS {
            return Err(GgmlCpuGraphError::ComputeFailed { status });
        }

        let mut values = vec![0_i32; expected_len];
        unsafe {
            ffi::ggml_backend_tensor_get(
                output.raw.as_ptr(),
                values.as_mut_ptr().cast::<c_void>(),
                0,
                output_nbytes,
            );
        }
        Ok(values)
    }

    fn ensure_backend_buffer(&mut self) -> Result<(), GgmlCpuGraphError> {
        if self.buffer.is_none() {
            if self.scheduler.is_some() && self.prepared_graph.is_some() {
                return Ok(());
            }
            let buffer = GgmlBackendBufferGuard::allocate(self.context, self.backend)?;
            self.buffer = Some(buffer);
        }
        Ok(())
    }

    fn build_forward_graph(
        &self,
        outputs: &[GgmlCpuTensor<'a>],
    ) -> Result<NonNull<c_void>, GgmlCpuGraphError> {
        let graph = self.allocate_forward_graph("ggml_new_graph_custom")?;
        for root in &self.side_effect_roots {
            unsafe { ffi::ggml_build_forward_expand(graph.as_ptr(), root.as_ptr()) };
        }
        for output in outputs {
            unsafe { ffi::ggml_build_forward_expand(graph.as_ptr(), output.raw.as_ptr()) };
        }
        Ok(graph)
    }

    fn allocate_forward_graph(
        &self,
        step: &'static str,
    ) -> Result<NonNull<c_void>, GgmlCpuGraphError> {
        NonNull::new(unsafe {
            ffi::ggml_new_graph_custom(self.context.as_ptr(), self.graph_size, false)
        })
        .ok_or(GgmlCpuGraphError::GraphBuildFailed { step })
    }

    fn ensure_can_extend_graph(&self, step: &'static str) -> Result<(), GgmlCpuGraphError> {
        if self.buffer.is_some() {
            return Err(GgmlCpuGraphError::GraphFrozenAfterAllocation { step });
        }
        Ok(())
    }

    fn new_tensor_checked(
        &self,
        raw: *mut c_void,
        tensor_name: &'static str,
    ) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
        NonNull::new(raw)
            .map(|raw| GgmlCpuTensor {
                raw,
                _marker: PhantomData,
            })
            .ok_or(GgmlCpuGraphError::TensorAllocationFailed {
                tensor: tensor_name,
            })
    }

    fn write_tensor_bytes_checked(
        &self,
        tensor: GgmlCpuTensor<'a>,
        data_ptr: *const c_void,
        offset_nbytes: usize,
        expected_nbytes: usize,
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let actual_nbytes = self.tensor_nbytes(tensor);
        let end_nbytes = offset_nbytes.checked_add(expected_nbytes).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor byte range overflow",
            },
        )?;
        if end_nbytes > actual_nbytes {
            return Err(GgmlCpuGraphError::TensorByteRangeOutOfBounds {
                tensor: tensor_name,
                offset: offset_nbytes,
                len: expected_nbytes,
                nbytes: actual_nbytes,
            });
        }
        self.ensure_tensor_contiguous(tensor, "tensor_upload")?;
        unsafe {
            write_tensor_data(tensor.raw, data_ptr, offset_nbytes, expected_nbytes);
        }
        Ok(())
    }

    fn tensor_nbytes(&self, tensor: GgmlCpuTensor<'a>) -> usize {
        unsafe { ffi::ggml_nbytes(tensor.raw.as_ptr()) }
    }

    #[cfg(test)]
    fn tensor_len_f32(&self, tensor: GgmlCpuTensor<'a>) -> Result<usize, GgmlCpuGraphError> {
        let nbytes = self.tensor_nbytes(tensor);
        if !nbytes.is_multiple_of(F32_WIDTH_BYTES) {
            return Err(GgmlCpuGraphError::TensorByteWidthMisaligned { bytes: nbytes });
        }
        Ok(nbytes / F32_WIDTH_BYTES)
    }

    fn tensor_nelements(&self, tensor: GgmlCpuTensor<'a>) -> Result<usize, GgmlCpuGraphError> {
        let nelements = unsafe { ffi::ggml_nelements(tensor.raw.as_ptr()) };
        usize::try_from(nelements).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "tensor element count exceeds usize boundary",
        })
    }

    fn ensure_tensor_contiguous(
        &self,
        tensor: GgmlCpuTensor<'a>,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let is_contiguous = unsafe { ffi::ggml_is_contiguous(tensor.raw.as_ptr()) };
        if !is_contiguous {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_reshape_2d" => "reshape_2d requires contiguous input tensor",
                    "ggml_reshape_3d" => "reshape_3d requires contiguous input tensor",
                    "ggml_reshape_4d" => "reshape_4d requires contiguous input tensor",
                    "ggml_view_1d" => "ggml_view_1d requires contiguous input tensor",
                    "ggml_view_2d" => "ggml_view_2d requires contiguous input tensor",
                    "ggml_view_3d" => "ggml_view_3d requires contiguous input tensor",
                    "ggml_view_4d" => "ggml_view_4d requires contiguous input tensor",
                    "ggml_set_rows src" => "ggml_set_rows src requires contiguous input tensor",
                    "ggml_set_rows indices" => {
                        "ggml_set_rows indices requires contiguous input tensor"
                    }
                    "ggml_rope_ext positions" => {
                        "ggml_rope_ext positions requires contiguous input tensor"
                    }
                    "ggml_flash_attn_ext mask" => {
                        "ggml_flash_attn_ext mask requires contiguous input tensor"
                    }
                    "ggml_cpy src" => "ggml_cpy src requires contiguous tensor",
                    "ggml_cpy dst" => "ggml_cpy dst requires contiguous tensor",
                    _ => "operation requires contiguous input tensor",
                },
            });
        }
        Ok(())
    }

    fn ensure_tensor_nelements(
        &self,
        tensor: GgmlCpuTensor<'a>,
        expected_nelements: usize,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let actual_nelements = self.tensor_nelements(tensor)?;
        if actual_nelements != expected_nelements {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_reshape_2d" => "reshape_2d target shape element count mismatch",
                    "ggml_reshape_3d" => "reshape_3d target shape element count mismatch",
                    _ => "tensor shape element count mismatch",
                },
            });
        }
        Ok(())
    }

    fn ensure_tensor_nelements_at_least(
        &self,
        tensor: GgmlCpuTensor<'a>,
        minimum_nelements: usize,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let actual_nelements = self.tensor_nelements(tensor)?;
        if actual_nelements < minimum_nelements {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_view_1d" => "ggml_view_1d span exceeds source tensor elements",
                    "ggml_view_2d" => "ggml_view_2d span exceeds source tensor elements",
                    "ggml_view_3d" => "ggml_view_3d span exceeds source tensor elements",
                    _ => "tensor view span exceeds source tensor elements",
                },
            });
        }
        Ok(())
    }

    fn tensor_layout_prefix(&self, tensor: GgmlCpuTensor<'a>) -> ffi::GgmlTensorLayoutPrefix {
        unsafe { *(tensor.raw.as_ptr() as *const ffi::GgmlTensorLayoutPrefix) }
    }

    fn tensor_type(&self, tensor: GgmlCpuTensor<'a>) -> i32 {
        self.tensor_layout_prefix(tensor).type_
    }

    fn tensor_shape_4d(
        &self,
        tensor: GgmlCpuTensor<'a>,
    ) -> Result<[usize; ffi::GGML_MAX_DIMS], GgmlCpuGraphError> {
        let layout = self.tensor_layout_prefix(tensor);
        let mut shape = [0usize; ffi::GGML_MAX_DIMS];
        for (index, dim) in layout.ne.iter().enumerate() {
            shape[index] =
                usize::try_from(*dim).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor shape dimensions exceed usize boundary",
                })?;
            if shape[index] == 0 {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "tensor shape dimensions must be positive",
                });
            }
        }
        Ok(shape)
    }

    #[cfg(test)]
    fn ensure_tensor_shape_prefix_matches(
        &self,
        tensor: GgmlCpuTensor<'a>,
        expected_prefix: &[usize],
        tensor_name: &str,
    ) -> Result<(), GgmlCpuGraphError> {
        let actual_shape = self.tensor_shape_4d(tensor)?;
        let mut expected_shape = vec![1_usize; ffi::GGML_MAX_DIMS];
        for (index, dim) in expected_prefix.iter().enumerate() {
            if index >= ffi::GGML_MAX_DIMS {
                return Err(GgmlCpuGraphError::TensorUploadShapeMismatch {
                    tensor: tensor_name.to_string(),
                    expected: expected_prefix.to_vec(),
                    actual: actual_shape.to_vec(),
                });
            }
            expected_shape[index] = *dim;
        }

        if actual_shape.as_slice() != expected_shape.as_slice() {
            return Err(GgmlCpuGraphError::TensorUploadShapeMismatch {
                tensor: tensor_name.to_string(),
                expected: expected_prefix.to_vec(),
                actual: actual_shape.to_vec(),
            });
        }
        Ok(())
    }

    fn ensure_tensor_type(
        &self,
        tensor: GgmlCpuTensor<'a>,
        expected_type: i32,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        if self.tensor_type(tensor) != expected_type {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_mul_mat lhs" => "ggml_mul_mat lhs must be f32",
                    "ggml_mul_mat rhs" => "ggml_mul_mat rhs must be f32",
                    "ggml_add lhs" => "ggml_add lhs must be f32",
                    "ggml_add rhs" => "ggml_add rhs must be f32",
                    "ggml_add_inplace lhs" => "ggml_add_inplace lhs must be f32",
                    "ggml_add_inplace rhs" => "ggml_add_inplace rhs must be f32",
                    "ggml_sub lhs" => "ggml_sub lhs must be f32",
                    "ggml_sub rhs" => "ggml_sub rhs must be f32",
                    "ggml_mul lhs" => "ggml_mul lhs must be f32",
                    "ggml_mul rhs" => "ggml_mul rhs must be f32",
                    "ggml_mul_inplace lhs" => "ggml_mul_inplace lhs must be f32",
                    "ggml_mul_inplace rhs" => "ggml_mul_inplace rhs must be f32",
                    "ggml_div lhs" => "ggml_div lhs must be f32",
                    "ggml_div rhs" => "ggml_div rhs must be f32",
                    "ggml_sqr input" => "ggml_sqr input must be f32",
                    "ggml_sqrt input" => "ggml_sqrt input must be f32",
                    "ggml_log input" => "ggml_log input must be f32",
                    "ggml_scale input" => "ggml_scale input must be f32",
                    "ggml_sum input" => "ggml_sum input must be f32",
                    "ggml_sum_rows input" => "ggml_sum_rows input must be f32",
                    "ggml_mean input" => "ggml_mean input must be f32",
                    "ggml_norm input" => "ggml_norm input must be f32",
                    "ggml_rms_norm input" => "ggml_rms_norm input must be f32",
                    "ggml_softplus input" => "ggml_softplus input must be f32",
                    "ggml_exp input" => "ggml_exp input must be f32",
                    "ggml_silu input" => "ggml_silu input must be f32",
                    "ggml_repeat input" => "ggml_repeat input must be f32",
                    "ggml_repeat target" => "ggml_repeat target must be f32",
                    "ggml_repeat_4d input" => "ggml_repeat_4d input must be f32",
                    "ggml_soft_max input" => "ggml_soft_max input must be f32",
                    "ggml_soft_max_ext input" => "ggml_soft_max_ext input must be f32",
                    "ggml_argmax input" => "ggml_argmax input must be f32",
                    "ggml_top_k input" => "ggml_top_k input must be f32",
                    "ggml_get_rows indices" => "ggml_get_rows indices must be i32",
                    "ggml_set_rows src" => "ggml_set_rows src must be f32",
                    "ggml_set_rows indices" => "ggml_set_rows indices must be i32",
                    "ggml_rope_ext input" => "ggml_rope_ext input must be f32",
                    "ggml_rope_ext positions" => "ggml_rope_ext positions must be i32",
                    "tensor_upload_i32" => "tensor upload expects i32 tensor element type",
                    "ggml_flash_attn_ext q" => "ggml_flash_attn_ext q must be f32",
                    "ggml_flash_attn_ext mask" => "ggml_flash_attn_ext mask must be f16",
                    "compute_output_i32 output" => "compute_output_i32 output must be i32",
                    "weight_upload" => "weight tensor upload type does not match tensor type",
                    _ => "operation requires a specific tensor element type",
                },
            });
        }
        Ok(())
    }

    fn ensure_tensor_type_in(
        &self,
        tensor: GgmlCpuTensor<'a>,
        allowed_types: &[i32],
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        if allowed_types.contains(&self.tensor_type(tensor)) {
            return Ok(());
        }
        Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: match step {
                "ggml_mul_mat lhs" => "ggml_mul_mat lhs must be f16/f32/q4_0/q8_0/q4_k",
                "ggml_mul_mat rhs" => "ggml_mul_mat rhs must be f16 or f32",
                "ggml_soft_max_ext mask" => "ggml_soft_max_ext mask must be f16 or f32",
                "ggml_flash_attn_ext k" => "ggml_flash_attn_ext k must be f16 or f32",
                "ggml_flash_attn_ext v" => "ggml_flash_attn_ext v must be f16 or f32",
                "ggml_get_rows embeddings" => "ggml_get_rows embeddings must be f16 or f32",
                "ggml_view_1d input" => "ggml_view_1d input type is unsupported",
                "ggml_view_2d input" => "ggml_view_2d input type is unsupported",
                "ggml_view_3d input" => "ggml_view_3d input type is unsupported",
                "ggml_view_4d input" => "ggml_view_4d input type is unsupported",
                "ggml_permute input" => "ggml_permute input type is unsupported",
                "ggml_cast input" => "ggml_cast input type is unsupported",
                "ggml_set_rows dst" => "ggml_set_rows dst must be f16 or f32",
                _ => "operation input type is unsupported",
            },
        })
    }

    #[allow(dead_code)]
    fn ensure_supported_storage_type(
        &self,
        type_: i32,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let supported_quant_type = matches!(
            type_,
            ffi::GGML_TYPE_Q4_0
                | ffi::GGML_TYPE_Q8_0
                | ffi::GGML_TYPE_Q3_K
                | ffi::GGML_TYPE_Q4_K
                | ffi::GGML_TYPE_Q5_K
                | ffi::GGML_TYPE_Q6_K
        ) && unsafe { ffi::ggml_is_quantized(type_) };
        if matches!(type_, ffi::GGML_TYPE_F16 | ffi::GGML_TYPE_F32) || supported_quant_type {
            return Ok(());
        }
        Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: match step {
                "ggml_cast target" => "ggml_cast target type is unsupported",
                _ => "tensor storage type is unsupported",
            },
        })
    }

    fn ensure_tensor_not_transposed(
        &self,
        tensor: GgmlCpuTensor<'a>,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let is_transposed = unsafe { ffi::ggml_is_transposed(tensor.raw.as_ptr()) };
        if is_transposed {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_mul_mat lhs" => "ggml_mul_mat lhs must not be transposed",
                    _ => "operation requires non-transposed input tensor",
                },
            });
        }
        Ok(())
    }

    fn ensure_tensor_contiguous_rows(
        &self,
        tensor: GgmlCpuTensor<'a>,
        step: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let layout = self.tensor_layout_prefix(tensor);
        let element_size = self.element_size_bytes(layout.type_)?;
        let ne = self.tensor_shape_4d(tensor)?;
        let block_size = self.type_block_size(layout.type_)?;
        let expected_nb0 = element_size;
        let expected_nb1 = expected_nb0.checked_mul(ne[0] / block_size).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor stride overflow",
            },
        )?;
        if layout.nb[0] == expected_nb0 && layout.nb[1] == expected_nb1 {
            return Ok(());
        }
        if matches!(
            step,
            "ggml_flash_attn_ext q" | "ggml_flash_attn_ext k" | "ggml_flash_attn_ext v"
        ) && layout.nb[0] == expected_nb0
            && layout.nb[1] >= expected_nb1
            && layout.nb[1].is_multiple_of(expected_nb1)
        {
            return Ok(());
        }
        {
            Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: match step {
                    "ggml_flash_attn_ext q" => "ggml_flash_attn_ext q requires contiguous rows",
                    "ggml_flash_attn_ext k" => "ggml_flash_attn_ext k requires contiguous rows",
                    "ggml_flash_attn_ext v" => "ggml_flash_attn_ext v requires contiguous rows",
                    "ggml_backend_tensor_set_2d" => {
                        "ggml_backend_tensor_set_2d requires contiguous tensor rows"
                    }
                    _ => "operation requires contiguous tensor rows",
                },
            })
        }
    }

    fn type_block_size(&self, type_: i32) -> Result<usize, GgmlCpuGraphError> {
        let block_size = usize::try_from(unsafe { ffi::ggml_blck_size(type_) }).map_err(|_| {
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor block size exceeds usize boundary",
            }
        })?;
        if block_size == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor block size must be positive",
            });
        }
        Ok(block_size)
    }

    fn element_size_bytes(&self, type_: i32) -> Result<usize, GgmlCpuGraphError> {
        let size = unsafe { ffi::ggml_type_size(type_) };
        if size == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "tensor element byte width must be positive",
            });
        }
        Ok(size)
    }

    fn ensure_can_mul_mat(
        &self,
        lhs: GgmlCpuTensor<'a>,
        rhs: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let lhs_shape = self.tensor_shape_4d(lhs)?;
        let rhs_shape = self.tensor_shape_4d(rhs)?;
        if lhs_shape[0] != rhs_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_mul_mat lhs.ne0 must equal rhs.ne0",
            });
        }
        if rhs_shape[2] % lhs_shape[2] != 0 || rhs_shape[3] % lhs_shape[3] != 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_mul_mat batch dims are not broadcast-compatible",
            });
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn ensure_cpy_compatible(
        &self,
        src: GgmlCpuTensor<'a>,
        dst: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let src_shape = self.tensor_shape_4d(src)?;
        let dst_shape = self.tensor_shape_4d(dst)?;
        if src_shape != dst_shape {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_cpy requires src and dst shape match",
            });
        }
        let src_type = self.tensor_type(src);
        let dst_type = self.tensor_type(dst);
        let supported_type_pair = src_type == dst_type
            || (src_type == ffi::GGML_TYPE_F32 && dst_type == ffi::GGML_TYPE_F16)
            || (src_type == ffi::GGML_TYPE_F16 && dst_type == ffi::GGML_TYPE_F32);
        if !supported_type_pair {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_cpy supports matching types or f32/f16 conversion",
            });
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn ensure_set_rows_compatible(
        &self,
        dst: GgmlCpuTensor<'a>,
        src: GgmlCpuTensor<'a>,
        row_indices: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let dst_shape = self.tensor_shape_4d(dst)?;
        let src_shape = self.tensor_shape_4d(src)?;
        let indices_shape = self.tensor_shape_4d(row_indices)?;
        if src_shape[0] != dst_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_set_rows src.ne0 must equal dst.ne0",
            });
        }
        if src_shape[1] != indices_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_set_rows indices.ne0 must equal src.ne1",
            });
        }
        if src_shape[2] != dst_shape[2] || src_shape[3] != dst_shape[3] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_set_rows source and destination batch dims must match",
            });
        }
        if indices_shape[3] != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_set_rows indices.ne3 must equal 1",
            });
        }
        if dst_shape[2] % indices_shape[1] != 0 || dst_shape[3] % indices_shape[2] != 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_set_rows indices batch dims are not broadcast-compatible",
            });
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn ensure_rope_ext_compatible(
        &self,
        input: GgmlCpuTensor<'a>,
        positions: GgmlCpuTensor<'a>,
        n_dims: usize,
    ) -> Result<(), GgmlCpuGraphError> {
        let input_shape = self.tensor_shape_4d(input)?;
        let position_shape = self.tensor_shape_4d(positions)?;
        if n_dims > input_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext n_dims must be <= input.ne0",
            });
        }
        if input_shape[2] != position_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext positions.ne0 must equal input.ne2",
            });
        }
        if position_shape[1] != 1 || position_shape[2] != 1 || position_shape[3] != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_rope_ext positions must be a 1d tensor",
            });
        }
        Ok(())
    }

    fn ensure_soft_max_ext_mask_compatible(
        &self,
        input: GgmlCpuTensor<'a>,
        mask: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let input_shape = self.tensor_shape_4d(input)?;
        let mask_shape = self.tensor_shape_4d(mask)?;
        if mask_shape[0] != input_shape[0] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext mask.ne0 must equal input.ne0",
            });
        }
        if mask_shape[1] < input_shape[1] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext mask.ne1 must be >= input.ne1",
            });
        }
        if input_shape[2] % mask_shape[2] != 0 || input_shape[3] % mask_shape[3] != 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_soft_max_ext mask batch dims are not broadcast-compatible",
            });
        }
        Ok(())
    }

    fn ensure_flash_attn_ext_compatible(
        &self,
        q: GgmlCpuTensor<'a>,
        k: GgmlCpuTensor<'a>,
        v: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        self.ensure_can_mul_mat(k, q)?;

        let q_shape = self.tensor_shape_4d(q)?;
        let k_shape = self.tensor_shape_4d(k)?;
        let v_shape = self.tensor_shape_4d(v)?;

        if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext q/k/v ne3 dimensions must match",
            });
        }
        if k_shape[1] != v_shape[1] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext k/v ne1 dimensions must match",
            });
        }
        if k_shape[2] != v_shape[2] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext k/v ne2 dimensions must match",
            });
        }
        Ok(())
    }

    fn ensure_flash_attn_ext_mask_compatible(
        &self,
        q: GgmlCpuTensor<'a>,
        mask: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let q_shape = self.tensor_shape_4d(q)?;
        let mask_shape = self.tensor_shape_4d(mask)?;
        if q_shape[2] % mask_shape[2] != 0 || q_shape[3] % mask_shape[3] != 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "ggml_flash_attn_ext mask batch dims are not broadcast-compatible",
            });
        }
        Ok(())
    }

    /// Metal's `GGML_OP_FLASH_ATTN_EXT` support check only accepts a fixed
    /// whitelist of head sizes (`METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS`).
    /// Building the op with an unsupported head_dim on Metal degrades into a
    /// backend-scheduler fallback or, on some ggml versions, an assert deep in
    /// the Metal kernel dispatch. Fail closed here with a typed error instead,
    /// so callers (e.g. a future encoder with head_dim 36/52 such as
    /// moonshine) can catch it and fall back to the naive attention path.
    fn ensure_flash_attn_ext_metal_head_dim_supported(
        &self,
        q: GgmlCpuTensor<'a>,
    ) -> Result<(), GgmlCpuGraphError> {
        let head_dim = self.tensor_shape_4d(q)?[0];
        if flash_attn_ext_head_dim_supported_on_backend(self.backend_kind, head_dim) {
            return Ok(());
        }
        Err(GgmlCpuGraphError::FlashAttnExtUnsupportedMetalHeadDim {
            head_dim,
            supported: METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS,
        })
    }

    fn validate_permute_axes(
        &self,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> Result<(), GgmlCpuGraphError> {
        let axes = [axis0, axis1, axis2, axis3];
        let mut seen = [false; 4];
        for axis in axes {
            if !(0..=3).contains(&axis) {
                return Err(GgmlCpuGraphError::InvalidPermuteAxes {
                    axis0,
                    axis1,
                    axis2,
                    axis3,
                });
            }
            let axis =
                usize::try_from(axis).map_err(|_| GgmlCpuGraphError::InvalidPermuteAxes {
                    axis0,
                    axis1,
                    axis2,
                    axis3,
                })?;
            if seen[axis] {
                return Err(GgmlCpuGraphError::InvalidPermuteAxes {
                    axis0,
                    axis1,
                    axis2,
                    axis3,
                });
            }
            seen[axis] = true;
        }
        Ok(())
    }
}

fn checked_dim_to_i64(value: usize) -> Result<i64, GgmlCpuGraphError> {
    if value == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "tensor shape dimensions must be positive",
        });
    }
    i64::try_from(value).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "tensor length exceeds ggml int64 shape boundary",
    })
}

struct GgmlContextGuard {
    raw: NonNull<c_void>,
}

impl GgmlContextGuard {
    fn new(context_bytes: usize) -> Result<Self, GgmlCpuGraphError> {
        let raw = unsafe {
            ffi::ggml_init(ffi::GgmlInitParams {
                mem_size: context_bytes,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or(GgmlCpuGraphError::ContextInitFailed { context_bytes })
    }

    fn from_raw(raw: NonNull<c_void>) -> Self {
        Self { raw }
    }
}

impl Drop for GgmlContextGuard {
    fn drop(&mut self) {
        unsafe { ffi::ggml_free(self.raw.as_ptr()) };
    }
}

struct GgufContextGuard {
    raw: NonNull<c_void>,
}

impl Drop for GgufContextGuard {
    fn drop(&mut self) {
        unsafe { ffi::gguf_free(self.raw.as_ptr()) };
    }
}

struct GgmlBackendGuard {
    raw: NonNull<c_void>,
    free_on_drop: bool,
}

struct GgmlCachedBackendGuard {
    raw: NonNull<c_void>,
}

struct GgmlBackendSchedulerGuard {
    raw: NonNull<c_void>,
}

/// GPU-class backends are expensive to initialize (device enumeration + driver
/// context creation) and are safe to keep resident for the thread's lifetime,
/// so they are cached per-thread per-kind and handed out with
/// `free_on_drop=false` (the cached entry owns the single instance). Metal had
/// this since inception; the discrete-GPU lane (HIP/CUDA/Vulkan) now shares the
/// exact same path so it stops paying `ggml_backend_dev_init`/free on every
/// `execute()`. The CPU backend is cheap and is intentionally not cached.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CachedBackendKind {
    Metal,
    Gpu,
}

thread_local! {
    static THREAD_BACKEND_CACHE_BY_KIND: RefCell<HashMap<CachedBackendKind, GgmlCachedBackendGuard>> =
        RefCell::new(HashMap::new());
}

impl GgmlBackendGuard {
    fn cpu() -> Result<Self, GgmlCpuGraphError> {
        // Registry path, not ggml_backend_cpu_init: under GGML_BACKEND_DL the CPU
        // backend is a loaded plugin (ggml-cpu-<variant>.dll) whose init symbol is
        // not linked into the host. init_by_type works for static builds too.
        ensure_backends_loaded();
        let raw = unsafe {
            ffi::ggml_backend_init_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU, std::ptr::null())
        };
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                free_on_drop: true,
            })
            .ok_or(GgmlCpuGraphError::CpuBackendUnavailable)
    }

    fn metal() -> Result<Self, GgmlCpuGraphError> {
        Self::cached_backend(CachedBackendKind::Metal, Self::init_metal_backend)
    }

    fn gpu() -> Result<Self, GgmlCpuGraphError> {
        Self::cached_backend(CachedBackendKind::Gpu, Self::init_gpu_backend)
    }

    /// Return a thread-local cached backend of `kind`, initializing it once via
    /// `init` on first use. The cached entry owns the backend for the thread's
    /// lifetime; handed-out guards never free it (`free_on_drop=false`).
    fn cached_backend(
        kind: CachedBackendKind,
        init: impl FnOnce() -> Result<NonNull<c_void>, GgmlCpuGraphError>,
    ) -> Result<Self, GgmlCpuGraphError> {
        if let Some(raw) =
            THREAD_BACKEND_CACHE_BY_KIND.with(|cache| cache.borrow().get(&kind).map(|g| g.raw))
        {
            return Ok(Self {
                raw,
                free_on_drop: false,
            });
        }
        let raw = init()?;
        THREAD_BACKEND_CACHE_BY_KIND.with(|cache| {
            cache
                .borrow_mut()
                .insert(kind, GgmlCachedBackendGuard { raw })
        });
        Ok(Self {
            raw,
            free_on_drop: false,
        })
    }

    fn accelerators(
        n_threads: Option<usize>,
        policy: GgmlCpuGraphCpuAcceleratorPolicy,
    ) -> Vec<Self> {
        if matches!(policy, GgmlCpuGraphCpuAcceleratorPolicy::Disabled) {
            return Vec::new();
        }
        #[cfg(target_os = "macos")]
        {
            let raw = unsafe { ffi::ggml_backend_blas_init() };
            let Some(raw) = NonNull::new(raw) else {
                return Vec::new();
            };
            let mut backend = Self {
                raw,
                free_on_drop: true,
            };
            if let Some(n_threads) = n_threads {
                let _ = backend.set_n_threads_if_supported(n_threads);
            }
            vec![backend]
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = n_threads;
            Vec::new()
        }
    }

    fn init_metal_backend() -> Result<NonNull<c_void>, GgmlCpuGraphError> {
        let raw = unsafe { ffi::ggml_backend_init_best() };
        let Some(raw) = NonNull::new(raw) else {
            return Err(GgmlCpuGraphError::MetalBackendUnavailable {
                actual_backend: "<none>".to_string(),
            });
        };
        let name = backend_name(raw);
        let name_lower = name.to_ascii_lowercase();
        if !(name_lower.contains("metal") || name_lower.starts_with("mtl")) {
            unsafe { ffi::ggml_backend_free(raw.as_ptr()) };
            return Err(GgmlCpuGraphError::MetalBackendUnavailable {
                actual_backend: name,
            });
        }
        Ok(raw)
    }

    fn init_gpu_backend() -> Result<NonNull<c_void>, GgmlCpuGraphError> {
        for device in ggml_available_devices()
            .into_iter()
            .filter(|device| backend_kind_is_accelerated(device.kind))
        {
            match device.initialize() {
                Ok(backend) => return Ok(backend.into_raw()),
                Err(GgmlRuntimeError::BackendUnavailable(_)) => continue,
            }
        }

        let raw = unsafe { ffi::ggml_backend_init_best() };
        let Some(raw) = NonNull::new(raw) else {
            return Err(GgmlCpuGraphError::GpuBackendUnavailable {
                actual_backend: "<none>".to_string(),
            });
        };
        let name = backend_name(raw);
        if !backend_name_is_accelerated(&name) {
            unsafe { ffi::ggml_backend_free(raw.as_ptr()) };
            return Err(GgmlCpuGraphError::GpuBackendUnavailable {
                actual_backend: name,
            });
        }
        Ok(raw)
    }

    fn set_n_threads(&mut self, n_threads: usize) -> Result<(), GgmlCpuGraphError> {
        if n_threads == 0 {
            return Err(GgmlCpuGraphError::InvalidThreadCount);
        }
        let n_threads = c_int::try_from(n_threads)
            .map_err(|_| GgmlCpuGraphError::ThreadCountOutOfRange { n_threads })?;
        // Set threads through the backend registry's proc-address table, not the
        // ggml_backend_cpu_set_n_threads symbol: under GGML_BACKEND_DL that symbol
        // lives in the loaded ggml-cpu plugin and is not linked into the host.
        // Works for static builds too; a no-op if the backend lacks the tunable.
        backend_set_n_threads(self.raw, n_threads);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn set_n_threads_if_supported(&mut self, n_threads: usize) -> Result<(), GgmlCpuGraphError> {
        if n_threads == 0 {
            return Err(GgmlCpuGraphError::InvalidThreadCount);
        }
        let n_threads = c_int::try_from(n_threads)
            .map_err(|_| GgmlCpuGraphError::ThreadCountOutOfRange { n_threads })?;
        let name = backend_name(self.raw).to_ascii_lowercase();
        if name.contains("blas") {
            unsafe { ffi::ggml_backend_blas_set_n_threads(self.raw.as_ptr(), n_threads) };
        }
        Ok(())
    }

    fn name(&self) -> String {
        backend_name(self.raw)
    }
}

/// Set a backend's thread count via the registry proc-address table — the only
/// mechanism that works under `GGML_BACKEND_DL`, where
/// `ggml_backend_cpu_set_n_threads` lives in the loaded ggml-cpu plugin rather
/// than the linked core. No-op if the backend's device/registry does not expose
/// the `ggml_backend_set_n_threads` tunable. Works for static builds too.
fn backend_set_n_threads(backend: NonNull<c_void>, n_threads: c_int) {
    type SetNThreadsFn = unsafe extern "C" fn(ffi::GgmlBackendRaw, c_int);
    unsafe {
        let device = ffi::ggml_backend_get_device(backend.as_ptr());
        if device.is_null() {
            return;
        }
        let reg = ffi::ggml_backend_dev_backend_reg(device);
        if reg.is_null() {
            return;
        }
        let proc =
            ffi::ggml_backend_reg_get_proc_address(reg, c"ggml_backend_set_n_threads".as_ptr());
        if proc.is_null() {
            return;
        }
        let set_fn: SetNThreadsFn = std::mem::transmute(proc);
        set_fn(backend.as_ptr(), n_threads);
    }
}

/// True when `backend` still resolves to a device. A live ggml backend always
/// has one; a null device means the backend is no longer usable -- e.g. a
/// cached (`free_on_drop=false`) Metal/GPU backend whose owning thread-local
/// `THREAD_BACKEND_CACHE_BY_KIND` entry was dropped when its thread exited,
/// leaving a non-owning guard elsewhere pointing at freed memory.
/// `ggml_backend_alloc_ctx_tensors` dereferences the device unconditionally and
/// `GGML_ASSERT(device)`-aborts the whole daemon on null, so buffer allocation
/// fails closed here (typed error, propagated up the graph builder) instead.
fn backend_device_present(backend: NonNull<c_void>) -> bool {
    !unsafe { ffi::ggml_backend_get_device(backend.as_ptr()) }.is_null()
}

fn ensure_backend_device_present(backend: NonNull<c_void>) -> Result<(), GgmlCpuGraphError> {
    if backend_device_present(backend) {
        Ok(())
    } else {
        Err(GgmlCpuGraphError::BackendDeviceUnavailable)
    }
}

fn backend_name(backend: NonNull<c_void>) -> String {
    let raw_name = unsafe { ffi::ggml_backend_name(backend.as_ptr()) };
    cstr_lossy(raw_name)
}

fn cstr_lossy(raw_name: *const std::ffi::c_char) -> String {
    if raw_name.is_null() {
        return "<unknown>".to_string();
    }
    unsafe { CStr::from_ptr(raw_name) }
        .to_string_lossy()
        .into_owned()
}

impl Drop for GgmlBackendGuard {
    fn drop(&mut self) {
        if self.free_on_drop {
            unsafe { ffi::ggml_backend_free(self.raw.as_ptr()) };
        }
    }
}

impl Drop for GgmlCachedBackendGuard {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_free(self.raw.as_ptr()) };
    }
}

impl GgmlBackendSchedulerGuard {
    fn new(
        backends: &mut [ffi::GgmlBackendRaw],
        graph_size: usize,
    ) -> Result<Self, GgmlCpuGraphError> {
        let n_backends =
            c_int::try_from(backends.len()).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "backend scheduler backend count exceeds ggml int boundary",
            })?;
        let raw = unsafe {
            ffi::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                n_backends,
                graph_size,
                false,
                true,
            )
        };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or(GgmlCpuGraphError::BackendSchedulerInitFailed)
    }
}

impl Drop for GgmlBackendSchedulerGuard {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_sched_free(self.raw.as_ptr()) };
    }
}

struct GgmlBackendBufferGuard {
    raw: NonNull<c_void>,
}

impl GgmlBackendBufferGuard {
    fn allocate(
        context: NonNull<c_void>,
        backend: NonNull<c_void>,
    ) -> Result<Self, GgmlCpuGraphError> {
        ensure_backend_device_present(backend)?;
        let raw =
            unsafe { ffi::ggml_backend_alloc_ctx_tensors(context.as_ptr(), backend.as_ptr()) };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or(GgmlCpuGraphError::BackendBufferAllocationFailed)
    }

    fn allocate_with_usage(
        context: NonNull<c_void>,
        backend: NonNull<c_void>,
        usage: c_int,
    ) -> Result<Self, GgmlCpuGraphError> {
        ensure_backend_device_present(backend)?;
        let raw =
            unsafe { ffi::ggml_backend_alloc_ctx_tensors(context.as_ptr(), backend.as_ptr()) };
        let Some(raw) = NonNull::new(raw) else {
            return Err(GgmlCpuGraphError::BackendBufferAllocationFailed);
        };
        unsafe { ffi::ggml_backend_buffer_set_usage(raw.as_ptr(), usage) };
        Ok(Self { raw })
    }

    fn from_raw(raw: NonNull<c_void>, usage: c_int) -> Self {
        unsafe { ffi::ggml_backend_buffer_set_usage(raw.as_ptr(), usage) };
        Self { raw }
    }
}

fn maybe_allocate_weight_buffer_from_host_ptr(
    backend: NonNull<c_void>,
    reader: &GgufTensorDataReader,
) -> Result<Option<(GgmlBackendBufferGuard, std::sync::Arc<Mmap>)>, GgmlCpuGraphError> {
    let device_raw = unsafe { ffi::ggml_backend_get_device(backend.as_ptr()) };
    let Some(device_raw) = NonNull::new(device_raw) else {
        return Ok(None);
    };
    let mut props = ffi::GgmlBackendDevProps {
        name: ptr::null(),
        description: ptr::null(),
        memory_free: 0,
        memory_total: 0,
        type_: 0,
        device_id: ptr::null(),
        caps: ffi::GgmlBackendDevCaps::default(),
    };
    unsafe { ffi::ggml_backend_dev_get_props(device_raw.as_ptr(), &mut props) };
    if !props.caps.buffer_from_host_ptr {
        return Ok(None);
    }
    let mmap = reader.backing_mmap();
    let buffer_raw = unsafe {
        ffi::ggml_backend_dev_buffer_from_host_ptr(
            device_raw.as_ptr(),
            mmap.as_ptr().cast_mut().cast::<c_void>(),
            mmap.len(),
            0,
        )
    };
    let Some(buffer_raw) = NonNull::new(buffer_raw) else {
        return Ok(None);
    };
    Ok(Some((
        GgmlBackendBufferGuard::from_raw(buffer_raw, ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS),
        mmap,
    )))
}

fn validate_direct_backend_matmul_weight_support(
    context: NonNull<c_void>,
    backend: NonNull<c_void>,
    path: &Path,
) -> Result<(), GgmlCpuGraphError> {
    let device_raw = unsafe { ffi::ggml_backend_get_device(backend.as_ptr()) };
    let Some(device_raw) = NonNull::new(device_raw) else {
        return Ok(());
    };
    let mut unsupported = Vec::new();
    let mut tensor_raw = unsafe { ffi::ggml_get_first_tensor(context.as_ptr()) };
    while let Some(raw) = NonNull::new(tensor_raw) {
        let layout = unsafe { *(raw.as_ptr() as *const ffi::GgmlTensorLayoutPrefix) };
        if is_loaded_matmul_weight_candidate(layout)
            && !super::backend::device_supports_matmul_for_type(device_raw, layout.type_)
        {
            let name = unsafe { cstr_lossy(ffi::ggml_get_name(raw.as_ptr())) };
            unsupported.push(format!("{name}:{}", ggml_type_name_lossy(layout.type_)));
        }
        tensor_raw = unsafe { ffi::ggml_get_next_tensor(context.as_ptr(), raw.as_ptr()) };
    }
    if unsupported.is_empty() {
        return Ok(());
    }
    Err(GgmlCpuGraphError::LoadedWeightContextFailed {
        reason: format!(
            "direct GPU backend cannot mul_mat one or more 2-D weight types in {}; \
             disable this model stage's GPU lane or enable a scheduler-backed fallback before loading: {}",
            path.display(),
            unsupported.join(", ")
        ),
    })
}

fn validate_direct_matmul_weight_type(
    backend: NonNull<c_void>,
    ggml_type: c_int,
    tensor_name: &'static str,
) -> Result<(), GgmlCpuGraphError> {
    let device_raw = unsafe { ffi::ggml_backend_get_device(backend.as_ptr()) };
    let Some(device_raw) = NonNull::new(device_raw) else {
        return Ok(());
    };
    if super::backend::device_supports_matmul_for_type(device_raw, ggml_type) {
        return Ok(());
    }
    Err(GgmlCpuGraphError::LoadedWeightContextFailed {
        reason: format!(
            "direct GPU backend cannot mul_mat tensor '{tensor_name}' with ggml type {}; \
             disable this model stage's GPU lane or use a supported quantization",
            ggml_type_name_lossy(ggml_type)
        ),
    })
}

fn is_loaded_matmul_weight_candidate(layout: ffi::GgmlTensorLayoutPrefix) -> bool {
    layout.ne[0] > 0
        && layout.ne[1] > 0
        && layout.ne[2] == 1
        && layout.ne[3] == 1
        && super::backend::is_known_matmul_weight_type(layout.type_)
}

fn ggml_type_name_lossy(type_: c_int) -> String {
    unsafe { cstr_lossy(ffi::ggml_type_name(type_)) }
}

impl Drop for GgmlBackendBufferGuard {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_buffer_free(self.raw.as_ptr()) };
    }
}

unsafe fn write_tensor_data(
    tensor: NonNull<c_void>,
    data_ptr: *const c_void,
    offset: usize,
    actual_nbytes: usize,
) {
    let layout = unsafe { *(tensor.as_ptr() as *const ffi::GgmlTensorLayoutPrefix) };
    if !layout.buffer.is_null() && unsafe { ffi::ggml_backend_buffer_is_host(layout.buffer) } {
        let dst = unsafe { ffi::ggml_get_data(tensor.as_ptr()) };
        if !dst.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(
                    data_ptr.cast::<u8>(),
                    dst.cast::<u8>().add(offset),
                    actual_nbytes,
                );
            }
            return;
        }
    }
    unsafe { ffi::ggml_backend_tensor_set(tensor.as_ptr(), data_ptr, offset, actual_nbytes) };
}

#[cfg(test)]
mod tests {
    use crate::ggml_runtime::{
        GgufTensorMetadata, GgufWeightTensorElementType, GgufWeightTensorPayload, ffi,
    };
    use crate::nn::half::f32_to_f16_bits as f32_to_f16_bits_for_test;

    use super::{
        AutoGpuPolicy, GgmlCpuBinaryOp, GgmlCpuGraphBackend, GgmlCpuGraphConfig,
        GgmlCpuGraphCpuAcceleratorPolicy, GgmlCpuGraphError, GgmlCpuGraphRunner,
        GgmlCpuGraphThreadingWorkload, GgmlRopeExtParams, METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS,
        flash_attn_ext_head_dim_supported_on_backend, runtime_gpu_is_available,
    };

    fn softplus_reference(value: f32) -> f32 {
        value.max(0.0) + (-(value.abs())).exp().ln_1p()
    }

    #[test]
    fn backend_device_present_holds_for_a_live_backend() {
        // A live ggml backend always resolves to a device, so the fail-closed
        // buffer-allocation guard must accept it. The null case it defends
        // against -- a cached GPU-class backend whose owning thread exited,
        // leaving a dangling non-owning guard -- is a use-after-free that cannot
        // be forged safely here; this pins the positive path so the guard stays a
        // safety net and never rejects a valid backend.
        let backend = super::GgmlBackendGuard::cpu().expect("cpu backend");
        assert!(super::backend_device_present(backend.raw));
        assert!(super::ensure_backend_device_present(backend.raw).is_ok());
    }

    #[test]
    #[ignore = "manual probe: prints runtime backend resolution pieces on this host"]
    fn probe_runtime_backend_resolution() {
        let best = std::ptr::NonNull::new(unsafe { ffi::ggml_backend_init_best() });
        let best_name = best.map(|raw| {
            let name = super::backend_name(raw);
            unsafe { ffi::ggml_backend_free(raw.as_ptr()) };
            name
        });
        eprintln!("probe: ggml_backend_init_best name={best_name:?}");
        let devices = crate::ggml_runtime::ggml_available_devices()
            .into_iter()
            .map(|device| (device.name.clone(), device.kind))
            .collect::<Vec<_>>();
        eprintln!("probe: ggml_available_devices={devices:?}");
        eprintln!(
            "probe: runtime_gpu_is_available={}",
            runtime_gpu_is_available()
        );
        eprintln!(
            "probe: request_backend_override={:?}",
            super::request_backend_override()
        );
        eprintln!(
            "probe: OPENASR_GGML_BACKEND={:?}",
            std::env::var(GgmlCpuGraphConfig::BACKEND_ENV).ok()
        );
        eprintln!(
            "probe: resolve_runtime_backend={:?}",
            GgmlCpuGraphConfig::resolve_runtime_backend()
        );
    }

    #[test]
    fn resolve_family_runtime_backend_gates_auto_but_never_explicit() {
        use super::{RequestBackendPreference, install_request_backend_override};

        // No per-request preference installed (Auto): a family that declares
        // `AutoGpuPolicy::Never` must stay pinned to CPU...
        assert_eq!(
            GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::Never),
            GgmlCpuGraphBackend::Cpu
        );
        // ...while a family that allows Auto to pick any GPU backend just
        // gets whatever the generic resolver would have picked anyway.
        assert_eq!(
            GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::AllBackends),
            GgmlCpuGraphConfig::resolve_runtime_backend()
        );

        // An explicit CpuOnly preference is honored regardless of the gate.
        {
            let _guard = install_request_backend_override(Some(RequestBackendPreference::CpuOnly));
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::Never),
                GgmlCpuGraphBackend::Cpu
            );
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::AllBackends),
                GgmlCpuGraphBackend::Cpu
            );
        }

        // An explicit Accelerated preference always wins, even for a family
        // whose Auto default is gated to CPU -- the gate can only ever pin
        // Auto, never override an explicit per-request choice.
        {
            let _guard =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            let expected = GgmlCpuGraphConfig::resolve_runtime_backend();
            assert!(expected.is_gpu_class());
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::Never),
                expected
            );
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::AllBackends),
                expected
            );
        }
    }

    #[test]
    fn resolve_family_runtime_backend_except_metal_gates_only_metal() {
        use super::{RequestBackendPreference, install_request_backend_override};

        // Auto (no explicit preference): ExceptMetal only pins Metal to CPU;
        // it must never touch a resolved CPU or the generic Gpu (CUDA/HIP/
        // Vulkan) lane, since only Metal has been measured to regress.
        match GgmlCpuGraphConfig::resolve_runtime_backend() {
            GgmlCpuGraphBackend::Metal => assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::ExceptMetal),
                GgmlCpuGraphBackend::Cpu
            ),
            other => assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::ExceptMetal),
                other
            ),
        }

        // An explicit Accelerated preference always wins over ExceptMetal too
        // -- the gate can only ever pin Auto, never an explicit request.
        {
            let _guard =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            let expected = GgmlCpuGraphConfig::resolve_runtime_backend();
            assert!(expected.is_gpu_class());
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::ExceptMetal),
                expected
            );
        }

        // An explicit CpuOnly preference is honored regardless of the gate.
        {
            let _guard = install_request_backend_override(Some(RequestBackendPreference::CpuOnly));
            assert_eq!(
                GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::ExceptMetal),
                GgmlCpuGraphBackend::Cpu
            );
        }
    }

    #[test]
    fn extended_unary_math_ops_compute_expected_values() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_1d_f32(4, "input")
            .expect("input allocation should succeed");
        graph
            .set_input(input)
            .expect("input set_input should succeed");

        let sqr = graph.sqr(input).expect("sqr should build");
        let sqrt = graph.sqrt(input).expect("sqrt should build");
        let log = graph.log(input).expect("log should build");
        let exp = graph.exp(input).expect("exp should build");
        let softplus = graph.softplus(input).expect("softplus should build");
        for output in [sqr, sqrt, log, exp, softplus] {
            graph
                .set_output(output)
                .expect("set_output should succeed before allocation");
        }

        let values = [0.25_f32, 1.0, 2.0, 4.0];
        graph
            .set_f32_slice(input, &values, "input")
            .expect("input upload should succeed");
        let outputs = graph
            .compute_outputs_f32(&[
                (sqr, values.len()),
                (sqrt, values.len()),
                (log, values.len()),
                (exp, values.len()),
                (softplus, values.len()),
            ])
            .expect("extended unary graph should compute");

        assert_f32_close(&outputs[0], &values.map(|value| value * value), 1.0e-6);
        assert_f32_close(&outputs[1], &values.map(f32::sqrt), 1.0e-6);
        assert_f32_close(&outputs[2], &values.map(f32::ln), 1.0e-6);
        assert_f32_close(&outputs[3], &values.map(f32::exp), 1.0e-5);
        assert_f32_close(&outputs[4], &values.map(softplus_reference), 1.0e-6);
    }

    #[test]
    fn extended_reduction_and_binary_ops_compute_expected_values() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(3, 2, "input")
            .expect("input allocation should succeed");
        let scalar = graph
            .new_tensor_1d_f32(1, "scalar")
            .expect("scalar allocation should succeed");
        graph
            .set_input(input)
            .expect("input set_input should succeed");
        graph
            .set_input(scalar)
            .expect("scalar set_input should succeed");

        let sub = graph.sub(input, scalar).expect("sub should build");
        let div = graph.div(input, scalar).expect("div should build");
        let sum = graph.sum(input).expect("sum should build");
        let sum_rows = graph.sum_rows(input).expect("sum_rows should build");
        let mean_rows = graph.mean_rows(input).expect("mean_rows should build");
        for output in [sub, div, sum, sum_rows, mean_rows] {
            graph
                .set_output(output)
                .expect("set_output should succeed before allocation");
        }

        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        graph
            .set_f32_slice(input, &values, "input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(scalar, &[2.0], "scalar")
            .expect("scalar upload should succeed");
        let outputs = graph
            .compute_outputs_f32(&[(sub, 6), (div, 6), (sum, 1), (sum_rows, 2), (mean_rows, 2)])
            .expect("extended reduction graph should compute");

        assert_f32_close(&outputs[0], &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 1.0e-6);
        assert_f32_close(&outputs[1], &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 1.0e-6);
        assert_f32_close(&outputs[2], &[21.0], 1.0e-6);
        assert_f32_close(&outputs[3], &[6.0, 15.0], 1.0e-6);
        assert_f32_close(&outputs[4], &[2.0, 5.0], 1.0e-6);
    }

    #[test]
    fn moonshine_gptj_rope_uses_interleaved_mode_zero() {
        let params =
            GgmlRopeExtParams::moonshine_gptj(32, 194, 10_000.0).expect("rope params should build");
        // GGML_ROPE_TYPE_NORMAL (0) is the interleaved / GPT-J pairing used by Moonshine.
        assert_eq!(params.mode, 0);
        assert_eq!(params.n_dims, 32);
        assert_eq!(params.freq_base, 10_000.0);
        assert_eq!(params.freq_scale, 1.0);
    }

    #[test]
    fn tanh_and_partial_gptj_rope_build_and_compute_finite() {
        let runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("runner should build");
        let mut runner = runner;
        let mut graph = runner.start_graph();
        // [head_dim=4, heads=2, tokens=3]; rotate only the first 2 dims (partial rotary).
        let states = graph
            .new_tensor_3d_f32(4, 2, 3, "states")
            .expect("states alloc");
        let positions = graph.new_tensor_1d_i32(3, "positions").expect("pos alloc");
        graph.set_input(states).expect("states input");
        graph.set_input(positions).expect("positions input");
        let activated = graph.tanh(states).expect("tanh should build");
        let params =
            GgmlRopeExtParams::moonshine_gptj(2, 194, 10_000.0).expect("rope params should build");
        let roped = graph
            .rope_ext(activated, positions, params)
            .expect("rope should build");
        graph.set_output(roped).expect("output");
        let values: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 - 1.0).collect();
        graph
            .set_f32_slice(states, &values, "states")
            .expect("set states");
        graph
            .set_i32_slice(positions, &[0, 1, 2], "positions")
            .expect("set positions");
        let out = graph.compute_output_f32(roped, 24).expect("compute");
        assert_eq!(out.len(), 24);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    fn test_tensor_metadata(
        name: &str,
        dims: &[u64],
        ggml_type: i32,
        type_name: &str,
        size_bytes: u64,
    ) -> GgufTensorMetadata {
        GgufTensorMetadata {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            type_name: type_name.to_string(),
            size_bytes,
            offset_bytes: 0,
        }
    }

    fn reference_softmax(scores: &[f32]) -> Vec<f32> {
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = Vec::with_capacity(scores.len());
        let mut denom = 0.0_f32;
        for score in scores {
            let weight = (*score - max_score).exp();
            denom += weight;
            weights.push(weight);
        }
        weights.into_iter().map(|weight| weight / denom).collect()
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "slice length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: {actual} != {expected} within {tolerance}"
            );
        }
    }

    #[test]
    fn tiny_graph_compute_add_success() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let output = runner
            .compute_add_f32(&[1.0, 2.5, -3.0], &[4.0, -0.5, 7.0])
            .expect("tiny add graph should compute");
        assert_eq!(output, vec![5.0, 2.0, 4.0]);
    }

    #[test]
    fn persistent_graph_session_reuses_built_graph_across_runs() {
        // Build `out = input * input` (element-wise square) ONCE into a
        // persistent session, then re-run it twice with different input data
        // WITHOUT rebuilding — the prepared cgraph is reused. This is the
        // build-once/re-run mechanism qwen decode graph reuse will stand on.
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut session = runner
            .start_persistent_graph_session(1024 * 1024)
            .expect("persistent session should open");
        let graph = session.builder();
        let input = graph
            .new_tensor_2d_f32(4, 1, "input")
            .expect("input tensor");
        graph.set_input(input).expect("set_input");
        let out = graph.mul(input, input).expect("element-wise square");
        graph.set_output(out).expect("set_output");
        // Allocate the cgraph ONCE (stores prepared_graph).
        graph
            .prepare_outputs_for_upload(&[out])
            .expect("prepare allocates the graph once");

        // First run.
        graph
            .set_f32_slice(input, &[1.0, 2.0, 3.0, 4.0], "input")
            .expect("set input run 1");
        let r1 = graph
            .compute_outputs_f32(&[(out, 4)])
            .expect("reuse compute run 1");
        assert_eq!(r1, vec![vec![1.0, 4.0, 9.0, 16.0]]);

        // Second run: refresh ONLY the input data, recompute the SAME graph.
        graph
            .set_f32_slice(input, &[2.0, 2.0, 5.0, 0.5], "input")
            .expect("set input run 2");
        let r2 = graph
            .compute_outputs_f32(&[(out, 4)])
            .expect("reuse compute run 2");
        assert_eq!(r2, vec![vec![4.0, 4.0, 25.0, 0.25]]);
    }

    #[test]
    fn persistent_graph_session_reuses_built_graph_for_i32_output() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut session = runner
            .start_persistent_graph_session(1024 * 1024)
            .expect("persistent session should open");
        let graph = session.builder();
        let logits = graph.new_tensor_2d_f32(5, 1, "logits").expect("logits");
        graph.set_input(logits).expect("set_input");
        let top1 = graph.top1_argmax(logits).expect("top1");
        graph.set_output(top1).expect("set_output");
        graph
            .prepare_outputs_for_upload(&[top1])
            .expect("prepare allocates the graph once");

        graph
            .set_f32_slice(logits, &[-2.0, 0.5, 7.0, 9.25, 1.0], "logits")
            .expect("set logits run 1");
        let r1 = graph.compute_output_i32(top1, 1).expect("reuse run 1");
        assert_eq!(r1, vec![3]);

        graph
            .set_f32_slice(logits, &[12.0, 0.5, 7.0, 9.25, 1.0], "logits")
            .expect("set logits run 2");
        let r2 = graph.compute_output_i32(top1, 1).expect("reuse run 2");
        assert_eq!(r2, vec![0]);
    }

    /// goals 7+8 Step 1 BLOCKER de-risk: prove a `GgmlCpuGraphBuilder` graph can
    /// `mul_mat` a zero-copy `GgmlLoadedWeightContext` leaf (a tensor that lives in
    /// a SEPARATE ggml context, bound via `ggml_backend_tensor_alloc`, never
    /// `set_input`/`set_f32_slice`'d) and compute the correct result. cohere binds
    /// such leaves through `GgmlStaticTensorArena`; the qwen audio encoder uses the
    /// builder instead. If the builder's allocator clobbered or re-allocated the
    /// cross-context leaf this would return wrong values or crash — it must leave
    /// an already-buffered leaf alone. Passing unblocks binding the audio encoder's
    /// 2D weights zero-copy (the ~2.4 GB f32 dequant win) without an arena rewrite.
    #[test]
    fn builder_graph_mul_mats_a_zero_copy_loaded_weight_leaf() {
        use crate::ggml_runtime::gguf_write::{
            GgufWriteTensor, GgufWriteTensorType, write_gguf_file_v0,
        };
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir().expect("temp dir");
        let pack = dir.path().join("loaded_leaf.gguf");
        // W is ggml [in=3, out=2]: out0 = [1,2,3], out1 = [4,5,6] (row-major, ne0=in
        // contiguous). mul_mat(W, x) with x=[1,1,1] -> [1+2+3, 4+5+6] = [6, 15].
        let w: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = vec![GgufWriteTensor {
            name: "loaded.weight".to_string(),
            dims: vec![3, 2],
            tensor_type: GgufWriteTensorType::F32,
            data,
        }];
        write_gguf_file_v0(&pack, &BTreeMap::new(), &tensors).expect("write tiny gguf");

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let loaded = runner
            .load_gguf_weight_context(&pack)
            .expect("load zero-copy weight context");
        let weight = loaded
            .tensor("loaded.weight")
            .expect("loaded tensor present")
            .as_graph_tensor();

        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(3, 1, "input")
            .expect("input tensor");
        // Define ops BEFORE uploading data (set_f32_slice freezes the graph).
        let out = graph.mul_mat(weight, input).expect("mul_mat loaded leaf");
        graph
            .set_f32_slice(input, &[1.0, 1.0, 1.0], "input")
            .expect("upload input");
        let result = graph.compute_output_f32(out, 2).expect("compute");
        assert_eq!(result, vec![6.0, 15.0]);
    }

    #[test]
    fn persistent_graph_session_reuses_built_graph_on_scheduler_path() {
        // Same build-once/re-run proof, but with the backend scheduler enabled
        // (the path Metal uses): prepare allocates via ggml_backend_sched_alloc_
        // graph once, then each compute calls sched_graph_compute WITHOUT a
        // sched_reset. This de-risks reusing a graph across decode steps on the
        // scheduler/Metal path.
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.use_scheduler = true;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("scheduler cpu graph runner should initialize");
        let mut session = runner
            .start_persistent_graph_session(1024 * 1024)
            .expect("persistent session should open");
        let graph = session.builder();
        let input = graph
            .new_tensor_2d_f32(4, 1, "input")
            .expect("input tensor");
        graph.set_input(input).expect("set_input");
        let out = graph.mul(input, input).expect("element-wise square");
        graph.set_output(out).expect("set_output");
        graph
            .prepare_outputs_for_upload(&[out])
            .expect("prepare allocates the graph once");

        graph
            .set_f32_slice(input, &[1.0, 2.0, 3.0, 4.0], "input")
            .expect("set input run 1");
        let r1 = graph
            .compute_outputs_f32(&[(out, 4)])
            .expect("reuse compute run 1");
        assert_eq!(r1, vec![vec![1.0, 4.0, 9.0, 16.0]]);

        graph
            .set_f32_slice(input, &[2.0, 2.0, 5.0, 0.5], "input")
            .expect("set input run 2");
        let r2 = graph
            .compute_outputs_f32(&[(out, 4)])
            .expect("reuse compute run 2");
        assert_eq!(r2, vec![vec![4.0, 4.0, 25.0, 0.25]]);
    }

    #[test]
    fn scheduler_graph_compute_add_success() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.use_scheduler = true;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("scheduler cpu graph runner should initialize");
        let output = runner
            .compute_add_f32(&[1.0, 2.5, -3.0], &[4.0, -0.5, 7.0])
            .expect("scheduler tiny add graph should compute");
        assert_eq!(output, vec![5.0, 2.0, 4.0]);
    }

    #[test]
    fn runner_reports_scheduler_usage() {
        let mut direct = GgmlCpuGraphConfig::conservative_default();
        direct.use_scheduler = false;
        let direct_runner =
            GgmlCpuGraphRunner::new(direct).expect("direct runner should initialize");
        assert!(!direct_runner.uses_scheduler());

        let mut scheduled = GgmlCpuGraphConfig::conservative_default();
        scheduled.use_scheduler = true;
        let scheduled_runner =
            GgmlCpuGraphRunner::new(scheduled).expect("scheduler runner should initialize");
        assert!(scheduled_runner.uses_scheduler());
    }

    #[test]
    fn tensor_set_get_f32_on_2d_tensor() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_2d_f32(2, 2, "tensor")
            .expect("2d tensor allocation should succeed");
        graph
            .set_input(tensor)
            .expect("set_input should succeed before allocation");
        graph
            .set_output(tensor)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_1d(tensor, 0, 1.25)
            .expect("set_f32_1d for index 0 should succeed");
        graph
            .set_f32_1d(tensor, 1, -2.0)
            .expect("set_f32_1d for index 1 should succeed");
        graph
            .set_f32_1d(tensor, 2, 3.5)
            .expect("set_f32_1d for index 2 should succeed");
        graph
            .set_f32_1d(tensor, 3, 8.0)
            .expect("set_f32_1d for index 3 should succeed");

        let value = graph
            .get_f32_1d(tensor, 2)
            .expect("get_f32_1d should return the written value");
        assert!((value - 3.5).abs() < 1.0e-6);

        let output = graph
            .compute_output_f32(tensor, 4)
            .expect("single-tensor forward graph should compute");
        assert_eq!(output, vec![1.25, -2.0, 3.5, 8.0]);
    }

    #[test]
    fn graph_can_compute_multiple_f32_outputs() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let lhs = graph
            .new_tensor_1d_f32(3, "lhs")
            .expect("lhs allocation should succeed");
        let rhs = graph
            .new_tensor_1d_f32(3, "rhs")
            .expect("rhs allocation should succeed");
        graph.set_input(lhs).expect("lhs input should be accepted");
        graph.set_input(rhs).expect("rhs input should be accepted");
        let sum = graph.add(lhs, rhs).expect("add should build");
        let product = graph.mul(lhs, rhs).expect("mul should build");
        graph
            .set_f32_slice(lhs, &[1.0, 2.0, 3.0], "lhs")
            .expect("lhs upload should succeed");
        graph
            .set_f32_slice(rhs, &[4.0, 5.0, 6.0], "rhs")
            .expect("rhs upload should succeed");

        let outputs = graph
            .compute_outputs_f32(&[(sum, 3), (product, 3)])
            .expect("multi-output graph should compute");
        assert_eq!(outputs, vec![vec![5.0, 7.0, 9.0], vec![4.0, 10.0, 18.0]]);
    }

    #[test]
    fn prelude_like_tiny_graph_with_conv_add_gelu_reshape_permute_and_view() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();

        let mel = graph
            .new_tensor_2d_f32(4, 2, "mel")
            .expect("mel allocation should succeed");
        let conv1_w = graph
            .new_tensor_3d_f32(3, 2, 2, "conv1_w")
            .expect("conv1_w allocation should succeed");
        let conv1_b = graph
            .new_tensor_2d_f32(1, 2, "conv1_b")
            .expect("conv1_b allocation should succeed");
        let conv2_w = graph
            .new_tensor_3d_f32(3, 2, 3, "conv2_w")
            .expect("conv2_w allocation should succeed");
        let conv2_b = graph
            .new_tensor_2d_f32(1, 3, "conv2_b")
            .expect("conv2_b allocation should succeed");

        graph.set_input(mel).expect("mel set_input should succeed");
        graph
            .set_input(conv1_w)
            .expect("conv1_w set_input should succeed");
        graph
            .set_input(conv1_b)
            .expect("conv1_b set_input should succeed");
        graph
            .set_input(conv2_w)
            .expect("conv2_w set_input should succeed");
        graph
            .set_input(conv2_b)
            .expect("conv2_b set_input should succeed");

        let conv1 = graph
            .conv_1d(conv1_w, mel, 1, 1, 1)
            .expect("conv1 op should build");
        let conv1 = graph
            .add(conv1, conv1_b)
            .expect("conv1 bias add should build");
        let conv1 = graph.gelu(conv1).expect("conv1 gelu should build");

        let conv2 = graph
            .conv_1d(conv2_w, conv1, 2, 1, 1)
            .expect("conv2 op should build");
        let conv2 = graph
            .add(conv2, conv2_b)
            .expect("conv2 bias add should build");
        let conv2 = graph.gelu(conv2).expect("conv2 gelu should build");

        let reshaped = graph
            .reshape_3d(conv2, 2, 3, 1)
            .expect("reshape_3d should build");
        let permuted = graph
            .permute(reshaped, 1, 0, 2, 3)
            .expect("permute should build");
        let contiguous = graph
            .cont(permuted)
            .expect("contiguous projection should build");
        let flattened = graph
            .reshape_2d(contiguous, 3, 2)
            .expect("reshape_2d should build");
        let output_tensor = graph
            .view_2d(flattened, 3, 2, 3 * std::mem::size_of::<f32>(), 0)
            .expect("view_2d should build");
        graph
            .set_output(output_tensor)
            .expect("set_output should succeed before allocation");

        let _ = output_tensor;
    }

    #[test]
    fn tiny_graph_mul_mat_success() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let lhs = graph
            .new_tensor_2d_f32(2, 2, "lhs")
            .expect("lhs allocation should succeed");
        let rhs = graph
            .new_tensor_2d_f32(2, 2, "rhs")
            .expect("rhs allocation should succeed");

        graph.set_input(lhs).expect("lhs set_input should succeed");
        graph.set_input(rhs).expect("rhs set_input should succeed");

        let product = graph.mul_mat(lhs, rhs).expect("mul_mat should build");
        graph
            .set_output(product)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_slice(lhs, &[1.0, 2.0, 3.0, 4.0], "lhs")
            .expect("lhs upload should succeed");
        graph
            .set_f32_slice(rhs, &[5.0, 6.0, 7.0, 8.0], "rhs")
            .expect("rhs upload should succeed");

        let output = graph
            .compute_output_f32(product, 4)
            .expect("mul_mat graph should compute");
        assert_eq!(output, vec![17.0, 39.0, 23.0, 53.0]);
    }

    #[test]
    fn graph_can_use_static_f16_weight_tensor() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static tensor arena should initialize");
        let weight = arena
            .new_tensor_2d_f16(2, 2, "static_weight")
            .expect("static f16 tensor should allocate");
        arena
            .set_f16_bits_slice(weight, &[0x3c00, 0x0000, 0x0000, 0x3c00], "static_weight")
            .expect("static weight upload should succeed");

        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 1, "input")
            .expect("input should allocate");
        graph.set_input(input).expect("input should be marked");
        let output = graph
            .mul_mat(arena.graph_tensor(weight), input)
            .expect("static weight should be usable in graph");
        graph.set_output(output).expect("output should be marked");
        graph
            .set_f32_slice(input, &[2.0, 3.0], "input")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(output, 2)
            .expect("graph should compute with static f16 tensor");
        assert_eq!(output, vec![2.0, 3.0]);
    }

    #[test]
    fn graph_can_use_static_f16_view_tensor() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static tensor arena should initialize");
        let base = arena
            .new_tensor_2d_f16(2, 4, "static_weight_base")
            .expect("static base tensor should allocate");
        let view = arena
            .view_2d(
                base,
                2,
                2,
                2 * std::mem::size_of::<u16>(),
                4 * std::mem::size_of::<u16>(),
                "static_weight_view",
            )
            .expect("static view should allocate");
        arena
            .set_f16_bits_slice(
                base,
                &[
                    0x3c00, 0x0000, 0x0000, 0x3c00, // identity
                    0x4000, 0x0000, 0x0000, 0x4000, // 2 * identity
                ],
                "static_weight_base",
            )
            .expect("static base upload should succeed");

        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 1, "input")
            .expect("input should allocate");
        graph.set_input(input).expect("input should be marked");
        let output = graph
            .mul_mat(arena.graph_tensor(view), input)
            .expect("static view should be usable in graph");
        graph.set_output(output).expect("output should be marked");
        graph
            .set_f32_slice(input, &[3.0, 4.0], "input")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(output, 2)
            .expect("graph should compute with static f16 view tensor");
        assert_eq!(output, vec![6.0, 8.0]);
    }

    #[test]
    fn graph_mul_mat_accepts_f16_rhs_input() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static tensor arena should initialize");
        let weight = arena
            .new_tensor_2d_f16(2, 2, "static_weight")
            .expect("static f16 tensor should allocate");
        arena
            .set_f16_bits_slice(weight, &[0x3c00, 0x0000, 0x0000, 0x3c00], "static_weight")
            .expect("static weight upload should succeed");

        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f16(2, 1, "input_f16")
            .expect("input should allocate");
        graph.set_input(input).expect("input should be marked");
        let output = graph
            .mul_mat(arena.graph_tensor(weight), input)
            .expect("mul_mat should accept f16 rhs");
        let output_f32 = graph
            .cast(output, ffi::GGML_TYPE_F32)
            .expect("output cast should build");
        graph
            .set_output(output_f32)
            .expect("output should be marked");
        graph
            .set_f16_bits_slice(input, &[0x4000, 0x4200], "input_f16")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(output_f32, 2)
            .expect("graph should compute with f16 rhs input");
        assert_eq!(output, vec![2.0, 3.0]);
    }

    #[test]
    fn graph_side_effect_root_can_copy_f32_into_static_f16_tensor() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static tensor arena should initialize");
        let target = arena
            .new_tensor_2d_f16(2, 1, "static_target")
            .expect("static f16 target should allocate");
        arena
            .set_f16_bits_slice(target, &[0x0000, 0x0000], "static_target")
            .expect("static target should initialize");

        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 1, "input")
            .expect("input should allocate");
        graph.set_input(input).expect("input should be settable");
        let input_flat = graph
            .reshape_1d(input, 2)
            .expect("input should reshape to one dimension");
        let target_flat = graph
            .view_1d(arena.graph_tensor(target), 2, 0)
            .expect("target view should build");
        let write = graph
            .cpy(input_flat, target_flat)
            .expect("f32 to f16 cpy should build");
        graph
            .add_side_effect_root(write)
            .expect("side effect root should register before allocation");
        let output_buffer = graph
            .new_tensor_1d_f32(2, "output_buffer")
            .expect("output buffer should allocate");
        let output = graph
            .cpy(target_flat, output_buffer)
            .expect("f16 to f32 cpy should build");
        graph
            .set_output(output)
            .expect("output should be settable before allocation");

        graph
            .set_f32_slice(input, &[1.5, -2.0], "input")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(output, 2)
            .expect("graph should compute side effect before output");
        assert_eq!(output, vec![1.5, -2.0]);
    }

    #[test]
    fn norm_scale_add_with_repeat_smoke() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 2, "input")
            .expect("input allocation should succeed");
        let bias_base = graph
            .new_tensor_1d_f32(2, "bias_base")
            .expect("bias_base allocation should succeed");
        graph
            .set_input(input)
            .expect("input set_input should succeed");
        graph
            .set_input(bias_base)
            .expect("bias_base set_input should succeed");

        let normed = graph.norm(input, 1.0e-5).expect("norm should build");
        let _ = graph
            .rms_norm(input, 1.0e-5)
            .expect("rms_norm should build for encoder seam completeness");
        let scaled = graph.scale(normed, 0.5).expect("scale should build");
        let bias = graph
            .repeat(bias_base, scaled)
            .expect("repeat should build for broadcast bias");
        let shifted = graph.add(scaled, bias).expect("add should build");
        graph
            .set_output(shifted)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_slice(input, &[1.0, 2.0, -1.0, 3.0], "input")
            .expect("input upload should succeed");
        graph
            .set_f32_slice(bias_base, &[0.1, -0.1], "bias_base")
            .expect("bias upload should succeed");

        let output = graph
            .compute_output_f32(shifted, 4)
            .expect("norm/scale/add graph should compute");
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn qwen_llm_primitive_ops_smoke() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let states = graph
            .new_tensor_3d_f32(4, 2, 2, "states")
            .expect("states allocation should succeed");
        let positions = graph
            .new_tensor_1d_i32(2, "positions")
            .expect("positions allocation should succeed");
        graph.set_input(states).expect("states should be input");
        graph
            .set_input(positions)
            .expect("positions should be input");

        let params =
            GgmlRopeExtParams::qwen_neox(4, 64, 1_000_000.0).expect("rope params should build");
        let rope = graph
            .rope_ext(states, positions, params)
            .expect("rope_ext should build");
        let rope = graph.cont(rope).expect("rope output should materialize");
        let silu = graph.silu(rope).expect("silu should build");
        let expanded = graph
            .repeat_4d(silu, 4, 2, 2, 2)
            .expect("repeat_4d should build");
        let cast_f16 = graph
            .cast(expanded, ffi::GGML_TYPE_F16)
            .expect("cast to f16 should build");
        let cast_f32 = graph
            .cast(cast_f16, ffi::GGML_TYPE_F32)
            .expect("cast back to f32 should build");
        graph
            .set_output(cast_f32)
            .expect("output should be settable");

        let values: Vec<f32> = (0..16).map(|index| index as f32 * 0.125 - 1.0).collect();
        graph
            .set_f32_slice(states, &values, "states")
            .expect("states upload should succeed");
        graph
            .set_i32_slice(positions, &[0, 1], "positions")
            .expect("positions upload should succeed");
        let output = graph
            .compute_output_f32(cast_f32, 32)
            .expect("primitive graph should compute");
        assert_eq!(output.len(), 32);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn batched_rope_ext_sequence_axis_matches_serial_positions() {
        fn compute_rope(values: &[f32], positions: &[i32], n_seq: usize) -> Vec<f32> {
            let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
                .expect("cpu graph runner should initialize");
            let mut graph = runner.start_graph();
            let states = graph
                .new_tensor_3d_f32(4, 1, n_seq, "states")
                .expect("states should allocate");
            let positions_tensor = graph
                .new_tensor_1d_i32(n_seq, "positions")
                .expect("positions should allocate");
            graph.set_input(states).expect("states should be input");
            graph
                .set_input(positions_tensor)
                .expect("positions should be input");
            let params =
                GgmlRopeExtParams::qwen_neox(4, 64, 1_000_000.0).expect("rope params should build");
            let output = graph
                .rope_ext(states, positions_tensor, params)
                .expect("rope_ext should build");
            graph.set_output(output).expect("output should be settable");
            graph
                .set_f32_slice(states, values, "states")
                .expect("states upload should succeed");
            graph
                .set_i32_slice(positions_tensor, positions, "positions")
                .expect("positions upload should succeed");
            graph
                .compute_output_f32(output, values.len())
                .expect("rope_ext should compute")
        }

        let seq0 = [1.0, 0.5, -0.25, 0.75];
        let seq1 = [-0.3, 0.8, 1.25, -1.0];
        let batched_values = [
            seq0[0], seq0[1], seq0[2], seq0[3], seq1[0], seq1[1], seq1[2], seq1[3],
        ];
        let batched = compute_rope(&batched_values, &[0, 7], 2);
        let serial0 = compute_rope(&seq0, &[0], 1);
        let serial1 = compute_rope(&seq1, &[7], 1);

        assert_f32_close(&batched[0..4], &serial0, 1.0e-5);
        assert_f32_close(&batched[4..8], &serial1, 1.0e-5);
    }

    #[test]
    fn batched_set_rows_sequence_indices_match_serial_planes() {
        fn compute_set_rows(n_seq: usize, src_values: &[f32], rows: &[i32]) -> Vec<f32> {
            const HEAD_DIM: usize = 2;
            const MAX_POSITIONS: usize = 4;
            const KV_HEADS: usize = 2;

            let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
                .expect("cpu graph runner should initialize");
            let mut graph = runner.start_graph();
            let dst = graph
                .new_tensor_4d_f32(HEAD_DIM, MAX_POSITIONS, KV_HEADS, n_seq, "dst")
                .expect("dst should allocate");
            let src = graph
                .new_tensor_4d_f32(HEAD_DIM, 1, KV_HEADS, n_seq, "src")
                .expect("src should allocate");
            let row_indices = graph
                .new_tensor_4d_typed(1, 1, n_seq, 1, ffi::GGML_TYPE_I32, "row_indices")
                .expect("row indices should allocate");
            graph.set_input(dst).expect("dst should be input");
            graph.set_input(src).expect("src should be input");
            graph
                .set_input(row_indices)
                .expect("row indices should be input");
            let output = graph
                .set_rows(dst, src, row_indices)
                .expect("set_rows should build for [1,1,N,1] indices");
            graph.set_output(output).expect("output should be settable");

            let dst_values = vec![0.0; HEAD_DIM * MAX_POSITIONS * KV_HEADS * n_seq];
            graph
                .set_f32_slice(dst, &dst_values, "dst")
                .expect("dst upload should succeed");
            graph
                .set_f32_slice(src, src_values, "src")
                .expect("src upload should succeed");
            graph
                .set_i32_slice(row_indices, rows, "row_indices")
                .expect("row indices upload should succeed");
            graph
                .compute_output_f32(output, dst_values.len())
                .expect("set_rows should compute")
        }

        let batched_src = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];
        let batched = compute_set_rows(2, &batched_src, &[1, 3]);
        let serial0 = compute_set_rows(1, &batched_src[0..4], &[1]);
        let serial1 = compute_set_rows(1, &batched_src[4..8], &[3]);
        let plane_len = serial0.len();

        assert_f32_close(&batched[0..plane_len], &serial0, 0.0);
        assert_f32_close(&batched[plane_len..plane_len * 2], &serial1, 0.0);
    }

    #[test]
    fn batched_set_rows_multiple_query_rows_match_serial_planes() {
        fn compute_set_rows(
            n_seq: usize,
            token_count: usize,
            src_values: &[f32],
            rows: &[i32],
        ) -> Vec<f32> {
            const HEAD_DIM: usize = 2;
            const MAX_POSITIONS: usize = 4;
            const KV_HEADS: usize = 2;

            let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
                .expect("cpu graph runner should initialize");
            let mut graph = runner.start_graph();
            let dst = graph
                .new_tensor_4d_f32(HEAD_DIM, MAX_POSITIONS, KV_HEADS, n_seq, "dst")
                .expect("dst should allocate");
            let src = graph
                .new_tensor_4d_f32(HEAD_DIM, token_count, KV_HEADS, n_seq, "src")
                .expect("src should allocate");
            let row_indices = graph
                .new_tensor_4d_typed(token_count, 1, n_seq, 1, ffi::GGML_TYPE_I32, "row_indices")
                .expect("row indices should allocate");
            graph.set_input(dst).expect("dst should be input");
            graph.set_input(src).expect("src should be input");
            graph
                .set_input(row_indices)
                .expect("row indices should be input");
            let output = graph
                .set_rows(dst, src, row_indices)
                .expect("set_rows should build for [T,1,N,1] indices");
            graph.set_output(output).expect("output should be settable");

            let dst_values = vec![0.0; HEAD_DIM * MAX_POSITIONS * KV_HEADS * n_seq];
            graph
                .set_f32_slice(dst, &dst_values, "dst")
                .expect("dst upload should succeed");
            graph
                .set_f32_slice(src, src_values, "src")
                .expect("src upload should succeed");
            graph
                .set_i32_slice(row_indices, rows, "row_indices")
                .expect("row indices upload should succeed");
            graph
                .compute_output_f32(output, dst_values.len())
                .expect("set_rows should compute")
        }

        let seq0_src = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let seq1_src = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0];
        let batched_src = [
            seq0_src[0],
            seq0_src[1],
            seq0_src[2],
            seq0_src[3],
            seq0_src[4],
            seq0_src[5],
            seq0_src[6],
            seq0_src[7],
            seq1_src[0],
            seq1_src[1],
            seq1_src[2],
            seq1_src[3],
            seq1_src[4],
            seq1_src[5],
            seq1_src[6],
            seq1_src[7],
        ];
        let batched = compute_set_rows(2, 2, &batched_src, &[0, 1, 2, 3]);
        let serial0 = compute_set_rows(1, 2, &seq0_src, &[0, 1]);
        let serial1 = compute_set_rows(1, 2, &seq1_src, &[2, 3]);
        let plane_len = serial0.len();

        assert_f32_close(&batched[0..plane_len], &serial0, 0.0);
        assert_f32_close(&batched[plane_len..plane_len * 2], &serial1, 0.0);
    }

    #[test]
    fn set_rows_rejects_indices_with_dim3_batching() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let graph = runner.start_graph();
        let dst = graph
            .new_tensor_4d_f32(2, 4, 2, 2, "dst")
            .expect("dst should allocate");
        let src = graph
            .new_tensor_4d_f32(2, 1, 2, 2, "src")
            .expect("src should allocate");
        let row_indices = graph
            .new_tensor_4d_typed(1, 1, 1, 2, ffi::GGML_TYPE_I32, "row_indices")
            .expect("row indices should allocate");

        match graph.set_rows(dst, src, row_indices) {
            Ok(_) => panic!("set_rows should reject indices.ne3 > 1"),
            Err(GgmlCpuGraphError::UnsupportedInputs { reason }) => {
                assert_eq!(reason, "ggml_set_rows indices.ne3 must equal 1");
            }
            Err(err) => panic!("unexpected set_rows error: {err}"),
        }
    }

    #[test]
    fn batched_flash_attn_ext_mask_planes_match_serial_runs() {
        fn compute_flash_attn(
            n_seq: usize,
            q_values: &[f32],
            k_values: &[f32],
            v_values: &[f32],
            mask_bits: &[u16],
        ) -> Vec<f32> {
            let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
                .expect("cpu graph runner should initialize");
            let mut graph = runner.start_graph();
            let q = graph
                .new_tensor_4d_f32(2, 1, 1, n_seq, "q")
                .expect("q should allocate");
            let k = graph
                .new_tensor_4d_f32(2, 2, 1, n_seq, "k")
                .expect("k should allocate");
            let v = graph
                .new_tensor_4d_f32(2, 2, 1, n_seq, "v")
                .expect("v should allocate");
            let mask = graph
                .new_tensor_4d_typed(2, 1, 1, n_seq, ffi::GGML_TYPE_F16, "mask")
                .expect("mask should allocate");
            graph.set_input(q).expect("q should be input");
            graph.set_input(k).expect("k should be input");
            graph.set_input(v).expect("v should be input");
            graph.set_input(mask).expect("mask should be input");
            let output = graph
                .flash_attn_ext(q, k, v, Some(mask), (2.0_f32).sqrt().recip(), 0.0, 0.0)
                .expect("flash_attn_ext should build");
            graph.set_output(output).expect("output should be settable");
            graph
                .set_f32_slice(q, q_values, "q")
                .expect("q upload should succeed");
            graph
                .set_f32_slice(k, k_values, "k")
                .expect("k upload should succeed");
            graph
                .set_f32_slice(v, v_values, "v")
                .expect("v upload should succeed");
            graph
                .set_f16_bits_slice(mask, mask_bits, "mask")
                .expect("mask upload should succeed");
            graph
                .compute_output_f32(output, 2 * n_seq)
                .expect("flash_attn_ext should compute")
        }

        let neg_inf = f32_to_f16_bits_for_test(f32::NEG_INFINITY);
        let zero = f32_to_f16_bits_for_test(0.0);
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let v = [10.0, 20.0, 30.0, 40.0, -5.0, -6.0, 50.0, 60.0];
        let mask = [zero, neg_inf, neg_inf, zero];

        let batched = compute_flash_attn(2, &q, &k, &v, &mask);
        let serial0 = compute_flash_attn(1, &q[0..2], &k[0..4], &v[0..4], &mask[0..2]);
        let serial1 = compute_flash_attn(1, &q[2..4], &k[4..8], &v[4..8], &mask[2..4]);

        assert_f32_close(&batched[0..2], &serial0, 1.0e-4);
        assert_f32_close(&batched[2..4], &serial1, 1.0e-4);
    }

    #[test]
    fn unary_ops_and_transpose_smoke() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let input = graph
            .new_tensor_2d_f32(2, 3, "input")
            .expect("input allocation should succeed");
        graph.set_input(input).expect("input should be input");

        let relu = graph.relu(input).expect("relu should build");
        let sigmoid = graph.sigmoid(relu).expect("sigmoid should build");
        let transposed = graph.transpose(sigmoid).expect("transpose should build");
        let restored = graph
            .transpose(transposed)
            .expect("second transpose should build");
        let restored = graph.cont(restored).expect("restored cont should build");
        graph
            .set_output(restored)
            .expect("output should be settable");

        graph
            .set_f32_slice(input, &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], "input")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(restored, 6)
            .expect("unary ops graph should compute");
        let expected: Vec<f32> = [0.0_f32, 0.0, 0.0, 1.0, 2.0, 3.0]
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect();
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn depthwise_conv_2d_variants_smoke() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.backend = GgmlCpuGraphBackend::Cpu;
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let kernel = graph
            .new_tensor_4d_typed(1, 1, 1, 1, ffi::GGML_TYPE_F16, "kernel")
            .expect("kernel allocation should succeed");
        let input = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "input")
            .expect("input allocation should succeed");
        graph.set_input(kernel).expect("kernel should be input");
        graph.set_input(input).expect("input should be input");

        let dw = graph
            .depthwise_conv_2d(kernel, input, 1, 1, 0, 0, 1, 1)
            .expect("depthwise conv should build");
        let dw_direct = graph
            .conv_2d_dw_direct(kernel, input, 1, 1, 0, 0, 1, 1)
            .expect("direct depthwise conv should build");
        let dw_output = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "dw_output")
            .expect("dw output buffer should allocate");
        let dw_direct_output = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "dw_direct_output")
            .expect("direct dw output buffer should allocate");
        let dw = graph
            .cpy(dw, dw_output)
            .expect("dw copy to f32 output should build");
        let dw_direct = graph
            .cpy(dw_direct, dw_direct_output)
            .expect("direct dw copy to f32 output should build");

        graph
            .set_f16_bits_slice(kernel, &[0x4000], "kernel")
            .expect("kernel upload should succeed");
        graph
            .set_f32_slice(input, &[1.0, 2.0, 3.0, 4.0], "input")
            .expect("input upload should succeed");
        let outputs = graph
            .compute_outputs_f32(&[(dw, 4), (dw_direct, 4)])
            .expect("depthwise conv variants should compute");
        assert_eq!(outputs[0], vec![2.0, 4.0, 6.0, 8.0]);
        assert!(outputs[0].iter().all(|value| value.is_finite()));
        assert!(outputs[1].iter().all(|value| value.is_finite()));
    }

    /// `depthwise_conv_2d` routes to the fused `GGML_OP_CONV_2D_DW` op on every
    /// backend, including Metal (which gained a native CONV_2D_DW kernel upstream).
    /// This runs the same tiny depthwise conv on a real Metal runner and checks it
    /// matches the CPU reference, exercising the native Metal kernel path rather
    /// than the retired IM2COL + MUL_MAT detour. Skips on hosts without a Metal GPU.
    #[test]
    fn depthwise_conv_2d_executes_on_metal_backend() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.backend = GgmlCpuGraphBackend::Metal;
        let mut runner = match GgmlCpuGraphRunner::new(config) {
            Ok(runner) => runner,
            Err(error) => {
                eprintln!(
                    "depthwise_conv_2d_executes_on_metal_backend: Metal unavailable ({error}) - skipping"
                );
                return;
            }
        };
        let mut graph = runner.start_graph();
        let kernel = graph
            .new_tensor_4d_typed(1, 1, 1, 1, ffi::GGML_TYPE_F16, "kernel")
            .expect("kernel allocation should succeed");
        let input = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "input")
            .expect("input allocation should succeed");
        graph.set_input(kernel).expect("kernel should be input");
        graph.set_input(input).expect("input should be input");

        let dw = graph
            .depthwise_conv_2d(kernel, input, 1, 1, 0, 0, 1, 1)
            .expect("depthwise conv should build");
        let dw_output = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "dw_output")
            .expect("dw output buffer should allocate");
        let dw = graph
            .cpy(dw, dw_output)
            .expect("dw copy to f32 output should build");

        graph
            .set_f16_bits_slice(kernel, &[0x4000], "kernel")
            .expect("kernel upload should succeed");
        graph
            .set_f32_slice(input, &[1.0, 2.0, 3.0, 4.0], "input")
            .expect("input upload should succeed");
        let outputs = graph
            .compute_outputs_f32(&[(dw, 4)])
            .expect("depthwise conv should compute on the metal backend");
        assert_eq!(outputs[0], vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// `group_norm` with `n_groups == n_channels` (per-channel instance norm),
    /// the wav2vec2 `feat_extract_norm=="group"` case. Input is `[ne0=2, ne1=1,
    /// ne2=2 channels]`; each channel's 2 elements are normalized to mean 0 /
    /// var 1. Reference computed in numpy (eps 1e-5).
    #[test]
    fn group_norm_per_channel_matches_numpy_reference() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        // 2 channels of 2 elements each, laid out ne0-fastest: ch0=[1,3], ch1=[-2,2].
        let input = graph
            .new_tensor_3d_f32(2, 1, 2, "gn_input")
            .expect("group_norm input should allocate");
        graph.set_input(input).expect("input should be marked");
        let normed = graph
            .group_norm(input, 2, 1.0e-5)
            .expect("group_norm should build with n_groups == channels");
        graph.set_output(normed).expect("output should be marked");
        graph
            .set_f32_slice(input, &[1.0, 3.0, -2.0, 2.0], "gn_input")
            .expect("input upload should succeed");
        let output = graph
            .compute_output_f32(normed, 4)
            .expect("group_norm graph should compute");
        let expected = [-0.999_995, 0.999_995, -0.999_998_75, 0.999_998_75];
        for (actual, expected) in output.iter().copied().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "group_norm {actual} != {expected}"
            );
        }
    }

    /// `concat` two `[d, n]` tensors along the channel axis (dim 1): the building
    /// block that stitches the per-group conv outputs of the wav2vec2 grouped
    /// pos-conv back into one tensor.
    #[test]
    fn concat_along_dim1_stacks_rows() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let a = graph
            .new_tensor_2d_f32(2, 1, "concat_a")
            .expect("a should allocate");
        let b = graph
            .new_tensor_2d_f32(2, 2, "concat_b")
            .expect("b should allocate");
        graph.set_input(a).expect("a input");
        graph.set_input(b).expect("b input");
        let cat = graph
            .concat(a, b, 1)
            .expect("concat should build along dim 1");
        graph.set_output(cat).expect("output should be marked");
        graph
            .set_f32_slice(a, &[1.0, 2.0], "concat_a")
            .expect("a upload");
        graph
            .set_f32_slice(b, &[3.0, 4.0, 5.0, 6.0], "concat_b")
            .expect("b upload");
        let output = graph.compute_output_f32(cat, 6).expect("concat compute");
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Grouped Conv1d emulated by a per-group loop of `conv_1d` over channel-sliced
    /// views + `concat` — the exact construction the wav2vec2 grouped positional
    /// conv uses (groups != 1 and != channels). Tiny case: in=4, out=4, groups=2,
    /// kernel=2, T=3, no padding/bias. Reference computed with a PyTorch grouped
    /// `nn.Conv1d` (values embedded below in ggml kernel layout `[K, in/g, out/g]`).
    #[test]
    fn grouped_conv_1d_via_per_group_loop_matches_pytorch_reference() {
        const G: usize = 2;
        const IN: usize = 4;
        const OUT: usize = 4;
        const K: usize = 2;
        const T: usize = 3;
        const T_OUT: usize = T - K + 1;
        const IN_G: usize = IN / G;
        const OUT_G: usize = OUT / G;

        // ggml conv_1d kernel layout per group: [K, in/g, out/g], K-fastest.
        let kernel_g: [[f32; K * IN_G * OUT_G]; G] = [
            [
                0.03492, -0.3012, 0.15921, 0.15689, -0.26724, -0.07494, -0.29291, 0.12974,
            ],
            [
                -0.13468, 0.35127, 0.35494, 0.05094, -0.21316, -0.29368, -0.05491, -0.14071,
            ],
        ];
        // ggml data layout [T, IN] (T-fastest within each channel row).
        let data: [f32; T * IN] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        // PyTorch reference output [OUT, T_OUT] flat.
        let expected: [f32; OUT * T_OUT] = [
            0.804, 0.85383, -0.43474, -0.94009, 5.35462, 5.97708, -5.23606, -5.93852,
        ];

        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::conservative_default().context_bytes)
            .expect("static arena should initialize");
        // Stage the per-group kernels as f16 arena tensors (the layout the
        // wav2vec2 graph uses for conv kernels). Allocate ALL tensors first (the
        // arena freezes on the first upload), then upload.
        let kernel_handles: Vec<_> = (0..G)
            .map(|_| {
                arena
                    .new_tensor_3d_typed(K, IN_G, OUT_G, ffi::GGML_TYPE_F16, "gconv_k")
                    .expect("group kernel should allocate")
            })
            .collect();
        for g in 0..G {
            let bits: Vec<u16> = kernel_g[g]
                .iter()
                .copied()
                .map(f32_to_f16_bits_for_test)
                .collect();
            arena
                .set_f16_bits_slice(kernel_handles[g], &bits, "gconv_k")
                .expect("group kernel upload");
        }

        let mut graph = runner.start_graph();
        let data_t = graph
            .new_tensor_2d_f32(T, IN, "gconv_data")
            .expect("data should allocate");
        graph.set_input(data_t).expect("data input");

        // Per-group: slice in/g input channels (a view along ne1), conv_1d, concat.
        let element = std::mem::size_of::<f32>();
        let mut group_outputs = Vec::with_capacity(G);
        #[allow(clippy::needless_range_loop)]
        for g in 0..G {
            let in_view = graph
                .view_2d(data_t, T, IN_G, T * element, g * IN_G * T * element)
                .expect("input channel slice view");
            let in_view = graph.cont(in_view).expect("input slice cont");
            let conv = graph
                .conv_1d(arena.graph_tensor(kernel_handles[g]), in_view, 1, 0, 1)
                .expect("per-group conv_1d should build");
            group_outputs.push(graph.cont(conv).expect("group conv cont"));
        }
        let mut grouped = group_outputs[0];
        for &next in &group_outputs[1..] {
            grouped = graph
                .concat(grouped, next, 1)
                .expect("concat group outputs");
        }
        graph.set_output(grouped).expect("output should be marked");
        graph
            .set_f32_slice(data_t, &data, "gconv_data")
            .expect("data upload");
        let output = graph
            .compute_output_f32(grouped, OUT * T_OUT)
            .expect("grouped conv graph should compute");
        for (i, (actual, expected)) in output.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 5.0e-3,
                "grouped_conv elem {i}: {actual} != {expected}"
            );
        }
    }

    /// LayerNorm OVER CHANNELS — the wav2vec2 `feat_extract_norm=="layer"` case.
    /// Conv output is `[T, C]` (T-fastest); HF normalizes over the channel dim C
    /// for each time step, then applies an affine gamma/beta `[C]`. The graph
    /// does this by transposing to `[C, T]`, `norm` over ne0=C, scale+shift,
    /// transpose back. Reference computed in numpy: for each column t, normalize
    /// the C values to mean 0 / var 1 (eps 1e-5), then `y = gamma*xhat + beta`.
    #[test]
    fn layer_norm_over_channels_matches_numpy_reference() {
        const T: usize = 2;
        const C: usize = 3;
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        // [T, C] layout, T-fastest within each channel row:
        //   ch0 = [1, 4], ch1 = [2, 0], ch2 = [3, -4]. So column t=0 = [1,2,3],
        //   column t=1 = [4,0,-4].
        let conv = graph
            .new_tensor_2d_f32(T, C, "ln_conv")
            .expect("conv input should allocate");
        let gamma = graph
            .new_tensor_1d_f32(C, "ln_gamma")
            .expect("gamma should allocate");
        let beta = graph
            .new_tensor_1d_f32(C, "ln_beta")
            .expect("beta should allocate");
        graph.set_input(conv).expect("conv input");
        graph.set_input(gamma).expect("gamma input");
        graph.set_input(beta).expect("beta input");
        // transpose [T, C] -> [C, T] so norm runs over ne0 = channels.
        let feat_major = graph.transpose(conv).expect("transpose to [C, T]");
        let feat_major = graph.cont(feat_major).expect("transpose cont");
        let normed = graph.norm(feat_major, 1.0e-5).expect("norm over channels");
        let scaled = graph.mul(normed, gamma).expect("affine scale");
        let shifted = graph.add(scaled, beta).expect("affine shift");
        // transpose back to [T, C].
        let back = graph.transpose(shifted).expect("transpose back to [T, C]");
        let back = graph.cont(back).expect("transpose back cont");
        graph.set_output(back).expect("output should be marked");
        graph
            .set_f32_slice(conv, &[1.0, 4.0, 2.0, 0.0, 3.0, -4.0], "ln_conv")
            .expect("conv upload");
        graph
            .set_f32_slice(gamma, &[2.0, 1.0, 0.5], "ln_gamma")
            .expect("gamma upload");
        graph
            .set_f32_slice(beta, &[0.1, -0.2, 0.3], "ln_beta")
            .expect("beta upload");
        let output = graph
            .compute_output_f32(back, T * C)
            .expect("layer-norm-over-channels graph should compute");
        // numpy reference, output in [T, C] layout (T-fastest within each channel):
        //   col0 [1,2,3] -> xhat=[-1.2247,0,1.2247]; y=[2*xhat0+.1, xhat1-.2, .5*xhat2+.3]
        //                  = [-2.3494, -0.2, 0.9124]
        //   col1 [4,0,-4] -> xhat=[1.2247,0,-1.2247]; y=[2.5494, -0.2, -0.3124]
        // [T,C]-flat (ch-major, t-fastest): ch0=[y00,y10], ch1=[y01,y11], ch2=[y02,y12]
        let expected = [
            -2.349_5, 2.549_5, // ch0
            -0.2, -0.2, // ch1
            0.912_4, -0.312_4, // ch2
        ];
        for (i, (actual, expected)) in output.iter().copied().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-3,
                "layer_norm_over_channels elem {i}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn qwen_decode_flash_attention_matches_reference() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let q = graph
            .new_tensor_3d_f32(2, 1, 1, "q")
            .expect("q allocation should succeed");
        let k = graph
            .new_tensor_3d_f32(2, 2, 1, "k")
            .expect("k allocation should succeed");
        let v = graph
            .new_tensor_3d_f32(2, 2, 1, "v")
            .expect("v allocation should succeed");
        graph.set_input(q).expect("q should be input");
        graph.set_input(k).expect("k should be input");
        graph.set_input(v).expect("v should be input");

        let scale = (2.0_f32).sqrt().recip();
        let output = graph
            .flash_attn_ext(q, k, v, None, scale, 0.0, 0.0)
            .expect("flash attention should build");
        graph.set_output(output).expect("output should be settable");

        graph
            .set_f32_slice(q, &[1.0, 0.0], "q")
            .expect("q upload should succeed");
        graph
            .set_f32_slice(k, &[1.0, 0.0, 0.0, 1.0], "k")
            .expect("k upload should succeed");
        graph
            .set_f32_slice(v, &[10.0, 20.0, 30.0, 40.0], "v")
            .expect("v upload should succeed");

        let actual = graph
            .compute_output_f32(output, 2)
            .expect("flash attention should compute");
        let weights = reference_softmax(&[scale, 0.0]);
        let expected = [
            weights[0] * 10.0 + weights[1] * 30.0,
            weights[0] * 20.0 + weights[1] * 40.0,
        ];
        for (actual, expected) in actual.iter().copied().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn kv_cache_view_copy_and_set_rows_smoke() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static arena should initialize");
        let cache = arena
            .new_tensor_4d_typed(2, 4, 1, 1, ffi::GGML_TYPE_F16, "kv_cache")
            .expect("kv cache should allocate");
        arena
            .set_f16_bits_slice(cache, &[0; 8], "kv_cache")
            .expect("kv cache should initialize");

        let mut graph = runner.start_graph();
        let src = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "src")
            .expect("src should allocate");
        let rows = graph
            .new_tensor_1d_i32(2, "rows")
            .expect("rows should allocate");
        let output_buffer = graph
            .new_tensor_4d_f32(2, 4, 1, 1, "output")
            .expect("output should allocate");
        graph.set_input(src).expect("src should be input");
        graph.set_input(rows).expect("rows should be input");

        let cache_tensor = arena.graph_tensor(cache);
        let write_rows = graph
            .set_rows(cache_tensor, src, rows)
            .expect("set_rows should build");
        graph
            .add_side_effect_root(write_rows)
            .expect("set_rows should be a side effect root");

        let row_stride = 2 * std::mem::size_of::<u16>();
        let cache_window = graph
            .view_4d(
                cache_tensor,
                2,
                2,
                1,
                1,
                row_stride,
                row_stride * 4,
                row_stride * 4,
                row_stride,
            )
            .expect("cache view should build");
        let window_src = graph
            .new_tensor_4d_f32(2, 2, 1, 1, "window_src")
            .expect("window source should allocate");
        graph
            .set_input(window_src)
            .expect("window source should be input");
        let write_window = graph
            .cpy_into_view(window_src, cache_window)
            .expect("copy into view should build");
        graph
            .add_side_effect_root(write_window)
            .expect("copy into view should be a side effect root");

        let output = graph
            .cpy(cache_tensor, output_buffer)
            .expect("cache readback should build");
        graph.set_output(output).expect("output should be settable");

        graph
            .set_f32_slice(src, &[1.0, 2.0, 3.0, 4.0], "src")
            .expect("src upload should succeed");
        graph
            .set_i32_slice(rows, &[0, 3], "rows")
            .expect("rows upload should succeed");
        graph
            .set_f32_slice(window_src, &[5.0, 6.0, 7.0, 8.0], "window_src")
            .expect("window source upload should succeed");
        let output = graph
            .compute_output_f32(output, 8)
            .expect("kv cache graph should compute");
        assert_eq!(output, vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0]);
    }

    #[test]
    fn attention_like_qk_softmax_v_smoke() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let q = graph
            .new_tensor_2d_f32(2, 2, "q")
            .expect("q allocation should succeed");
        let k = graph
            .new_tensor_2d_f32(2, 2, "k")
            .expect("k allocation should succeed");
        let v = graph
            .new_tensor_2d_f32(2, 2, "v")
            .expect("v allocation should succeed");
        graph.set_input(q).expect("q set_input should succeed");
        graph.set_input(k).expect("k set_input should succeed");
        graph.set_input(v).expect("v set_input should succeed");

        let kq = graph.mul_mat(k, q).expect("kq matmul should build");
        let mask = graph
            .new_tensor_3d_f32(2, 2, 1, "kq_mask")
            .expect("kq_mask allocation should succeed");
        graph
            .set_input(mask)
            .expect("kq_mask set_input should succeed");
        let probs = graph
            .soft_max_ext(kq, Some(mask), 1.0, 0.0)
            .expect("soft_max_ext should build");
        let attended = graph.mul_mat(v, probs).expect("kqv matmul should build");
        let _ = graph.soft_max(kq).expect("soft_max wrapper should build");
        graph
            .set_output(attended)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_slice(q, &[0.5, -0.25, 1.0, 0.75], "q")
            .expect("q upload should succeed");
        graph
            .set_f32_slice(k, &[1.0, 0.0, 0.5, 1.0], "k")
            .expect("k upload should succeed");
        graph
            .set_f32_slice(v, &[0.2, 0.8, -0.4, 1.2], "v")
            .expect("v upload should succeed");
        graph
            .set_f32_slice(mask, &[0.0, -f32::INFINITY, 0.0, 0.0], "kq_mask")
            .expect("kq_mask upload should succeed");

        let output = graph
            .compute_output_f32(attended, 4)
            .expect("attention-like graph should compute");
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn embedding_lookup_get_rows_is_finite() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let embeddings = graph
            .new_tensor_2d_f32(3, 4, "embeddings")
            .expect("embeddings allocation should succeed");
        let token_ids = graph
            .new_tensor_1d_i32(2, "token_ids")
            .expect("token id allocation should succeed");
        graph
            .set_input(embeddings)
            .expect("embeddings set_input should succeed");
        graph
            .set_input(token_ids)
            .expect("token ids set_input should succeed");

        let gathered = graph
            .get_rows(embeddings, token_ids)
            .expect("get_rows should build");
        graph
            .set_output(gathered)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_slice(
                embeddings,
                &[
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0, //
                    7.0, 8.0, 9.0, //
                    10.0, 11.0, 12.0,
                ],
                "embeddings",
            )
            .expect("embeddings upload should succeed");
        graph
            .set_i32_slice(token_ids, &[1, 3], "token_ids")
            .expect("token ids upload should succeed");

        let output = graph
            .compute_output_f32(gathered, 6)
            .expect("embedding lookup graph should compute");
        assert_eq!(output, vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn top1_argmax_matches_top_k_single_column() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let logits = graph
            .new_tensor_2d_f32(5, 1, "logits")
            .expect("logits allocation should succeed");
        graph
            .set_input(logits)
            .expect("logits set_input should succeed");

        let top1 = graph.top1_argmax(logits).expect("top1 argmax should build");
        let topk = graph.top_k(logits, 1).expect("top_k(1) should build");
        graph
            .set_output(top1)
            .expect("set_output should succeed before allocation");

        graph
            .set_f32_slice(logits, &[-2.0, 0.5, 7.0, 9.25, 1.0], "logits")
            .expect("logits upload should succeed");

        let top1_index = graph
            .compute_output_i32(top1, 1)
            .expect("top1 argmax should compute");
        let topk_index = graph
            .compute_output_i32(topk, 1)
            .expect("top_k(1) should compute");
        assert_eq!(top1_index, vec![3]);
        assert_eq!(topk_index, vec![3]);
    }

    #[test]
    fn decoder_attention_like_smoke_with_embedding_and_top1() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();

        let token_embedding = graph
            .new_tensor_2d_f32(4, 8, "token_embedding")
            .expect("token embedding allocation should succeed");
        let token_ids = graph
            .new_tensor_1d_i32(2, "token_ids")
            .expect("token ids allocation should succeed");
        let position_embedding = graph
            .new_tensor_2d_f32(4, 2, "position_embedding")
            .expect("position embedding allocation should succeed");
        let gain = graph
            .new_tensor_1d_f32(4, "gain")
            .expect("gain allocation should succeed");
        let output_proj = graph
            .new_tensor_2d_f32(4, 8, "output_proj")
            .expect("output projection allocation should succeed");

        graph
            .set_input(token_embedding)
            .expect("token embedding set_input should succeed");
        graph
            .set_input(token_ids)
            .expect("token ids set_input should succeed");
        graph
            .set_input(position_embedding)
            .expect("position embedding set_input should succeed");
        graph
            .set_input(gain)
            .expect("gain set_input should succeed");
        graph
            .set_input(output_proj)
            .expect("output projection set_input should succeed");

        let state = graph
            .get_rows(token_embedding, token_ids)
            .expect("token embedding lookup should build");
        let state = graph
            .add(state, position_embedding)
            .expect("token + position add should build");

        let q = graph
            .reshape_3d(state, 2, 2, 2)
            .expect("reshape for attention q should build");
        let k = graph
            .reshape_3d(state, 2, 2, 2)
            .expect("reshape for attention k should build");
        let v = graph
            .reshape_3d(state, 2, 2, 2)
            .expect("reshape for attention v should build");

        let scores = graph.mul_mat(k, q).expect("qk matmul should build");
        let scores = graph.cont(scores).expect("qk cont should build");
        let probs = graph.soft_max(scores).expect("qk softmax should build");
        let v_t = graph
            .permute(v, 1, 0, 2, 3)
            .expect("v transpose should build");
        let v_t = graph.cont(v_t).expect("v cont should build");
        let context = graph.mul_mat(v_t, probs).expect("av matmul should build");

        let context = graph
            .permute(context, 0, 2, 1, 3)
            .expect("context permute should build");
        let context = graph.cont(context).expect("context cont should build");
        let context = graph
            .reshape_2d(context, 4, 2)
            .expect("context reshape should build");
        let context = graph
            .mul(context, gain)
            .expect("elementwise gain should build");
        let state = graph
            .add(context, state)
            .expect("residual add should build");

        let logits = graph
            .mul_mat(output_proj, state)
            .expect("output projection should build");
        let last_token_logits = graph
            .view_2d(
                logits,
                8,
                1,
                8 * std::mem::size_of::<f32>(),
                8 * std::mem::size_of::<f32>(),
            )
            .expect("last-token logits view should build");
        let top1 = graph
            .top1_argmax(last_token_logits)
            .expect("top1 argmax should build");
        graph
            .set_output(top1)
            .expect("set_output should succeed before allocation");

        let token_embedding_values: Vec<f32> = (0..32)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.125)
            .collect();
        graph
            .set_f32_slice(token_embedding, &token_embedding_values, "token_embedding")
            .expect("token embedding upload should succeed");
        graph
            .set_i32_slice(token_ids, &[1, 4], "token_ids")
            .expect("token ids upload should succeed");
        graph
            .set_f32_slice(
                position_embedding,
                &[
                    0.01, 0.02, //
                    0.03, 0.04, //
                    0.05, 0.06, //
                    0.07, 0.08,
                ],
                "position_embedding",
            )
            .expect("position embedding upload should succeed");
        graph
            .set_f32_slice(gain, &[1.0, 0.9, 1.1, 1.0], "gain")
            .expect("gain upload should succeed");
        let output_proj_values: Vec<f32> = (0..32)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.1)
            .collect();
        graph
            .set_f32_slice(output_proj, &output_proj_values, "output_proj")
            .expect("output projection upload should succeed");

        let top1_index = graph
            .compute_output_i32(top1, 1)
            .expect("decoder-like graph top1 should compute");
        assert_eq!(top1_index.len(), 1);
        assert!(
            (0..8).contains(&top1_index[0]),
            "top1 token id should be in vocabulary range, got {:?}",
            top1_index
        );
    }

    #[test]
    fn fail_closed_for_add_shape_mismatch() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let graph = runner.start_graph();
        let lhs = graph
            .new_tensor_1d_f32(3, "lhs")
            .expect("lhs allocation should succeed");
        let rhs = graph
            .new_tensor_1d_f32(2, "rhs")
            .expect("rhs allocation should succeed");
        match graph.add(lhs, rhs) {
            Ok(_) => panic!("shape mismatch must fail closed"),
            Err(error) => assert_eq!(
                error,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_add rhs cannot broadcast to lhs shape",
                }
            ),
        }
    }

    #[test]
    fn fail_closed_for_flash_attn_ext_requires_mask_for_max_bias() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let graph = runner.start_graph();
        let q = graph
            .new_tensor_2d_f32(2, 2, "q")
            .expect("q allocation should succeed");
        let k = graph
            .new_tensor_2d_f32(2, 2, "k")
            .expect("k allocation should succeed");
        let v = graph
            .new_tensor_2d_f32(2, 2, "v")
            .expect("v allocation should succeed");
        match graph.flash_attn_ext(q, k, v, None, 1.0, 0.5, 0.0) {
            Ok(_) => panic!("flash_attn_ext max_bias without mask must fail closed"),
            Err(error) => assert_eq!(
                error,
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "ggml_flash_attn_ext max_bias > 0 requires a mask tensor",
                }
            ),
        }
    }

    #[test]
    fn flash_attn_ext_metal_head_dim_whitelist_matches_ggml_metal_device_support_check() {
        // Every whitelisted head_dim must be accepted on Metal...
        for &head_dim in METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS {
            assert!(
                flash_attn_ext_head_dim_supported_on_backend(GgmlCpuGraphBackend::Metal, head_dim),
                "head_dim={head_dim} should be Metal-supported"
            );
        }
        // ...and a handful of unsupported values (including moonshine's
        // relative-position head sizes 36/52, the concrete future-caller risk
        // this guard exists for) must be rejected on Metal.
        for head_dim in [1, 16, 36, 52, 60, 100, 200, 577] {
            assert!(
                !flash_attn_ext_head_dim_supported_on_backend(GgmlCpuGraphBackend::Metal, head_dim),
                "head_dim={head_dim} should not be Metal-supported"
            );
        }
        // Cpu/Gpu never enforce the whitelist -- only Metal's kernel dispatch
        // is fixed to this set of head sizes.
        for head_dim in [1, 36, 52, 64, 577] {
            assert!(flash_attn_ext_head_dim_supported_on_backend(
                GgmlCpuGraphBackend::Cpu,
                head_dim
            ));
            assert!(flash_attn_ext_head_dim_supported_on_backend(
                GgmlCpuGraphBackend::Gpu,
                head_dim
            ));
        }
    }

    #[test]
    fn flash_attn_ext_rejects_unsupported_metal_head_dim_end_to_end() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default())
            .expect("cpu graph runner should initialize");
        let graph = runner.start_graph();
        // head_dim=3 is not in the Metal whitelist; the guard must fire even
        // though this graph never touches a real Metal backend (backend_kind
        // is a config value the builder carries, independent of which
        // physical device the test happens to run compute on).
        let mut metal_graph = graph;
        metal_graph.backend_kind = GgmlCpuGraphBackend::Metal;
        let q = metal_graph
            .new_tensor_4d_f32(3, 1, 1, 1, "q")
            .expect("q should allocate");
        let k = metal_graph
            .new_tensor_4d_f32(3, 2, 1, 1, "k")
            .expect("k should allocate");
        let v = metal_graph
            .new_tensor_4d_f32(3, 2, 1, 1, "v")
            .expect("v should allocate");
        match metal_graph.flash_attn_ext(q, k, v, None, 1.0, 0.0, 0.0) {
            Ok(_) => panic!("flash_attn_ext must reject head_dim=3 on the Metal backend"),
            Err(GgmlCpuGraphError::FlashAttnExtUnsupportedMetalHeadDim {
                head_dim,
                supported,
            }) => {
                assert_eq!(head_dim, 3);
                assert_eq!(supported, METAL_FLASH_ATTN_EXT_SUPPORTED_HEAD_DIMS);
            }
            Err(err) => panic!("unexpected flash_attn_ext error: {err}"),
        }
    }

    #[test]
    fn uploads_weight_tensor_from_f32_payload() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_2d_f32(2, 2, "weight_f32")
            .expect("f32 tensor allocation should succeed");
        let values = [1.0_f32, -2.5_f32, 3.25_f32, 0.5_f32];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let metadata = test_tensor_metadata(
            "encoder.weight.f32",
            &[2, 2],
            super::ffi::GGML_TYPE_F32,
            "f32",
            bytes.len() as u64,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &bytes,
            dims: vec![2, 2],
            num_elements: 4,
            element_type: GgufWeightTensorElementType::F32,
        };

        graph
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect("f32 payload upload should succeed");
        assert_eq!(
            graph
                .get_f32_1d(tensor, 2)
                .expect("f32 upload should be readable"),
            3.25
        );
    }

    #[test]
    fn uploads_weight_tensor_from_f16_payload() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_3d_f16(2, 2, 1, "weight_f16")
            .expect("f16 tensor allocation should succeed");
        let bits = [0x3c00_u16, 0x3800_u16, 0x4000_u16, 0x0000_u16];
        let mut bytes = Vec::new();
        for value in bits {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let metadata = test_tensor_metadata(
            "encoder.weight.f16",
            &[2, 2, 1],
            super::ffi::GGML_TYPE_F16,
            "f16",
            bytes.len() as u64,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &bytes,
            dims: vec![2, 2, 1],
            num_elements: 4,
            element_type: GgufWeightTensorElementType::F16,
        };

        graph
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect("f16 payload upload should succeed");
        assert_eq!(
            graph
                .tensor_nelements(tensor)
                .expect("f16 tensor should expose nelements"),
            4
        );
    }

    #[test]
    fn static_arena_uploads_raw_quantized_weight_payload() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::default().context_bytes)
            .expect("static arena should initialize");
        let bytes = vec![0_u8; 34];
        let metadata = test_tensor_metadata(
            "llm.weight.q8",
            &[32, 1],
            super::ffi::GGML_TYPE_Q8_0,
            "q8_0",
            bytes.len() as u64,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &bytes,
            dims: vec![32, 1],
            num_elements: 32,
            element_type: GgufWeightTensorElementType::RawGgml {
                ggml_type: super::ffi::GGML_TYPE_Q8_0,
            },
        };

        let tensor = arena
            .new_tensor_from_weight_payload(&payload)
            .expect("raw q8 tensor should allocate");
        arena
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect("raw q8 payload upload should succeed");

        let graph = runner.start_graph();
        let tensor = arena.graph_tensor(tensor);
        assert_eq!(graph.tensor_nbytes(tensor), bytes.len());
        assert_eq!(graph.tensor_type(tensor), super::ffi::GGML_TYPE_Q8_0);
        assert_eq!(
            graph
                .tensor_nelements(tensor)
                .expect("q8 tensor logical element count"),
            32
        );
    }

    #[test]
    fn fail_closed_for_weight_upload_shape_mismatch() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_2d_f32(2, 2, "shape_target")
            .expect("f32 tensor allocation should succeed");
        let values = [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let metadata = test_tensor_metadata(
            "encoder.weight.mismatch",
            &[4],
            super::ffi::GGML_TYPE_F32,
            "f32",
            bytes.len() as u64,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &bytes,
            dims: vec![4],
            num_elements: 4,
            element_type: GgufWeightTensorElementType::F32,
        };

        let error = graph
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect_err("shape mismatch must fail");
        assert!(matches!(
            error,
            GgmlCpuGraphError::TensorUploadShapeMismatch { .. }
        ));
    }

    #[test]
    fn fail_closed_for_weight_upload_byte_mismatch() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_1d_f16(2, "byte_target")
            .expect("f16 tensor allocation should succeed");
        let metadata = test_tensor_metadata(
            "encoder.weight.byte_mismatch",
            &[2],
            super::ffi::GGML_TYPE_F16,
            "f16",
            2,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &[0_u8, 0_u8],
            dims: vec![2],
            num_elements: 2,
            element_type: GgufWeightTensorElementType::F16,
        };

        let error = graph
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect_err("byte mismatch must fail");
        assert!(matches!(
            error,
            GgmlCpuGraphError::TensorUploadByteSizeMismatch { .. }
        ));
    }

    #[test]
    fn fail_closed_after_allocation_for_graph_mutation_paths() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut graph = runner.start_graph();
        let tensor = graph
            .new_tensor_1d_f32(2, "frozen_base")
            .expect("base tensor allocation should succeed");
        let values = [1.0_f32, 2.0_f32];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let metadata = test_tensor_metadata(
            "encoder.weight.freeze",
            &[2],
            super::ffi::GGML_TYPE_F32,
            "f32",
            bytes.len() as u64,
        );
        let payload = GgufWeightTensorPayload {
            metadata: &metadata,
            bytes: &bytes,
            dims: vec![2],
            num_elements: 2,
            element_type: GgufWeightTensorElementType::F32,
        };
        graph
            .set_weight_tensor_from_payload(tensor, &payload)
            .expect("initial upload should allocate backend buffer");

        let new_tensor_error = graph
            .new_tensor_1d_f32(1, "post_alloc")
            .expect_err("new tensor after allocation must fail");
        assert!(matches!(
            new_tensor_error,
            GgmlCpuGraphError::GraphFrozenAfterAllocation { .. }
        ));

        let set_input_error = graph
            .set_input(tensor)
            .expect_err("set_input after allocation must fail");
        assert!(matches!(
            set_input_error,
            GgmlCpuGraphError::GraphFrozenAfterAllocation { .. }
        ));
    }

    #[test]
    fn fail_closed_for_unsupported_operation() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let error = runner
            .compute_binary_f32(&[1.0], &[2.0], GgmlCpuBinaryOp::Mul)
            .expect_err("unsupported op must fail closed");

        assert_eq!(
            error,
            GgmlCpuGraphError::UnsupportedOperation {
                operation: GgmlCpuBinaryOp::Mul
            }
        );
    }

    #[test]
    fn fail_closed_for_invalid_context_bytes() {
        match GgmlCpuGraphRunner::new(GgmlCpuGraphConfig {
            context_bytes: 0,
            ..GgmlCpuGraphConfig::default()
        }) {
            Ok(_) => panic!("zero context bytes must fail closed"),
            Err(error) => assert_eq!(error, GgmlCpuGraphError::InvalidContextBytes),
        }
    }

    #[test]
    fn fail_closed_for_invalid_thread_count() {
        match GgmlCpuGraphRunner::new(GgmlCpuGraphConfig {
            n_threads: Some(0),
            ..GgmlCpuGraphConfig::default()
        }) {
            Ok(_) => panic!("zero thread count must fail closed"),
            Err(error) => assert_eq!(error, GgmlCpuGraphError::InvalidThreadCount),
        }
    }

    #[test]
    fn thread_count_env_parser_accepts_positive_integer() {
        assert_eq!(
            GgmlCpuGraphConfig::parse_thread_count_env(Some(" 8 ")),
            Some(8)
        );
    }

    #[test]
    fn default_thread_count_scales_with_workload_profile() {
        assert_eq!(
            GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                1,
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::Default
            ),
            1
        );
        assert_eq!(
            GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                8,
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::Default
            ),
            6
        );
        assert_eq!(
            GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                16,
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::Default
            ),
            12
        );
        assert_eq!(
            GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                8,
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::EncoderPrelude
            ),
            7
        );
        assert_eq!(
            GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                8,
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::Decoder
            ),
            4
        );
        for backend in [GgmlCpuGraphBackend::Metal, GgmlCpuGraphBackend::Gpu] {
            assert_eq!(
                GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                    8,
                    backend,
                    GgmlCpuGraphThreadingWorkload::Default
                ),
                2
            );
            assert_eq!(
                GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                    8,
                    backend,
                    GgmlCpuGraphThreadingWorkload::EncoderPrelude
                ),
                5
            );
            assert_eq!(
                GgmlCpuGraphConfig::adaptive_thread_count_for_available(
                    8,
                    backend,
                    GgmlCpuGraphThreadingWorkload::Decoder
                ),
                2
            );
        }
    }

    #[test]
    fn thread_count_env_parser_ignores_invalid_values() {
        for value in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some("-1"),
            Some("abc"),
        ] {
            assert_eq!(GgmlCpuGraphConfig::parse_thread_count_env(value), None);
        }
    }

    #[test]
    fn backend_env_parser_accepts_metal_aliases() {
        assert_eq!(
            GgmlCpuGraphConfig::parse_backend_env_with_default(
                Some(" metal "),
                GgmlCpuGraphBackend::Cpu
            ),
            GgmlCpuGraphBackend::Metal
        );
    }

    #[test]
    fn backend_env_parser_accepts_generic_gpu_aliases() {
        let default_gpu_backend = GgmlCpuGraphConfig::default_gpu_backend_for_target();
        assert_eq!(
            GgmlCpuGraphConfig::parse_backend_env_with_default(
                Some("GPU"),
                GgmlCpuGraphBackend::Cpu
            ),
            default_gpu_backend
        );
        for value in [Some("hip"), Some("ROCM"), Some(" cuda "), Some("vulkan")] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_backend_env_with_default(value, GgmlCpuGraphBackend::Cpu),
                GgmlCpuGraphBackend::Gpu
            );
        }
    }

    #[test]
    fn backend_env_parser_uses_default_backend_when_unset_or_unknown() {
        for value in [None, Some(""), Some(" "), Some("unknown")] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_backend_env_with_default(
                    value,
                    GgmlCpuGraphBackend::Metal
                ),
                GgmlCpuGraphBackend::Metal
            );
        }
        assert_eq!(
            GgmlCpuGraphConfig::parse_backend_env_with_default(
                Some("cpu"),
                GgmlCpuGraphBackend::Metal
            ),
            GgmlCpuGraphBackend::Cpu
        );
        assert_eq!(
            GgmlCpuGraphConfig::parse_backend_env_with_default(
                Some("cpu"),
                GgmlCpuGraphBackend::Cpu
            ),
            GgmlCpuGraphBackend::Cpu
        );
    }

    #[test]
    fn loaded_matmul_weight_candidate_covers_direct_gpu_quantized_2d_weights() {
        for ggml_type in [
            super::ffi::GGML_TYPE_F32,
            super::ffi::GGML_TYPE_F16,
            super::ffi::GGML_TYPE_Q8_0,
            super::ffi::GGML_TYPE_Q4_0,
            super::ffi::GGML_TYPE_Q3_K,
            super::ffi::GGML_TYPE_Q4_K,
            super::ffi::GGML_TYPE_Q5_K,
            super::ffi::GGML_TYPE_Q6_K,
        ] {
            assert!(
                super::is_loaded_matmul_weight_candidate(test_layout(ggml_type, [256, 32, 1, 1])),
                "ggml_type={ggml_type} should be probed before direct GPU load"
            );
        }
    }

    #[test]
    fn loaded_matmul_weight_candidate_ignores_non_matmul_weight_shapes() {
        assert!(!super::is_loaded_matmul_weight_candidate(test_layout(
            super::ffi::GGML_TYPE_F32,
            [256, 0, 1, 1]
        )));
        assert!(!super::is_loaded_matmul_weight_candidate(test_layout(
            super::ffi::GGML_TYPE_F32,
            [3, 3, 1, 16]
        )));
        assert!(!super::is_loaded_matmul_weight_candidate(test_layout(
            26,
            [256, 32, 1, 1]
        )));
    }

    fn test_layout(
        ggml_type: std::ffi::c_int,
        ne: [i64; super::ffi::GGML_MAX_DIMS],
    ) -> super::ffi::GgmlTensorLayoutPrefix {
        super::ffi::GgmlTensorLayoutPrefix {
            type_: ggml_type,
            buffer: std::ptr::null_mut(),
            ne,
            nb: [0; super::ffi::GGML_MAX_DIMS],
        }
    }

    #[test]
    // Probes the LIVE GPU (`runtime_gpu_is_available()` initializes the Metal
    // backend → ~20s/process on a Metal host) and is Metal-host-specific, so it is
    // ignored by default. Validate the GPU-preferring default on a Metal host with
    // `cargo nextest run --run-ignored all -E 'test(backend_runtime_default_prefers_gpu)'`.
    // CI runs on Linux (no Metal) where the CPU branch is the meaningful one.
    #[ignore = "probes live Metal (~20s); Metal-host-only — run via --run-ignored"]
    fn backend_runtime_default_prefers_gpu_when_available() {
        // The pure no-env default-selection logic (NOT env-honoring
        // `resolve_runtime_backend()`, which would also be flaky under the parallel
        // suite's set_var/remove_var races + the committed test default cpu).
        let backend = GgmlCpuGraphConfig::default_runtime_backend();
        if runtime_gpu_is_available() {
            assert_eq!(
                backend,
                GgmlCpuGraphConfig::default_gpu_backend_for_target()
            );
        } else {
            assert_eq!(backend, GgmlCpuGraphBackend::Cpu);
        }
    }

    #[test]
    fn backend_env_cpu_forces_cpu_even_when_gpu_exists() {
        assert_eq!(
            GgmlCpuGraphConfig::parse_backend_env_with_default(
                Some("cpu"),
                GgmlCpuGraphBackend::Metal
            ),
            GgmlCpuGraphBackend::Cpu
        );
    }

    #[test]
    fn backend_env_unknown_defaults_to_runtime_default() {
        // This validates the PARSE logic (unknown/empty/None -> the passed default),
        // which is independent of the live default, so use a fixed representative
        // default rather than probing the GPU (`runtime_gpu_is_available()` inits
        // Metal ~20s). The Cpu-default case is covered by
        // `backend_env_parser_can_default_to_cpu`.
        let default_backend = GgmlCpuGraphBackend::Metal;
        for value in [None, Some(""), Some(" "), Some("unknown")] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_backend_env_with_default(value, default_backend),
                default_backend
            );
        }
    }

    #[test]
    fn backend_env_parser_can_default_to_cpu() {
        for value in [None, Some(""), Some(" "), Some("unknown")] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_backend_env_with_default(value, GgmlCpuGraphBackend::Cpu),
                GgmlCpuGraphBackend::Cpu
            );
        }
    }

    #[test]
    fn thread_count_env_parser_ignores_out_of_range_values() {
        let out_of_range = ((std::ffi::c_int::MAX as usize) + 1).to_string();
        assert_eq!(
            GgmlCpuGraphConfig::parse_thread_count_env(Some(&out_of_range)),
            None
        );
    }

    #[test]
    fn cpu_accelerator_env_parser_supports_off_and_none() {
        for value in [Some("off"), Some("none"), Some("0"), Some(" OFF ")] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_cpu_accelerator_env(value),
                GgmlCpuGraphCpuAcceleratorPolicy::Disabled
            );
        }
    }

    #[test]
    fn cpu_accelerator_env_parser_supports_blas_and_auto_default() {
        assert_eq!(
            GgmlCpuGraphConfig::parse_cpu_accelerator_env(Some("blas")),
            GgmlCpuGraphCpuAcceleratorPolicy::Blas
        );
        for value in [
            None,
            Some(""),
            Some("auto"),
            Some("default"),
            Some("unknown"),
        ] {
            assert_eq!(
                GgmlCpuGraphConfig::parse_cpu_accelerator_env(value),
                GgmlCpuGraphCpuAcceleratorPolicy::Auto
            );
        }
    }

    #[test]
    fn cpu_accelerator_default_is_disabled_on_metal_when_env_unset() {
        for backend in [GgmlCpuGraphBackend::Metal, GgmlCpuGraphBackend::Gpu] {
            assert_eq!(
                GgmlCpuGraphConfig::resolve_runtime_cpu_accelerator_policy_with_env(None, backend),
                GgmlCpuGraphCpuAcceleratorPolicy::Disabled
            );
            assert_eq!(
                GgmlCpuGraphConfig::resolve_runtime_cpu_accelerator_policy_with_env(
                    Some(" "),
                    backend,
                ),
                GgmlCpuGraphCpuAcceleratorPolicy::Disabled
            );
        }
    }

    #[test]
    fn cpu_accelerator_default_is_auto_on_cpu_when_env_unset() {
        assert_eq!(
            GgmlCpuGraphConfig::resolve_runtime_cpu_accelerator_policy_with_env(
                None,
                GgmlCpuGraphBackend::Cpu,
            ),
            GgmlCpuGraphCpuAcceleratorPolicy::Auto
        );
    }

    #[test]
    fn cpu_accelerator_enabled_with_env_matches_policy_resolution() {
        for value in [Some("off"), Some("none"), Some("0"), Some(" OFF ")] {
            assert!(!GgmlCpuGraphConfig::cpu_accelerator_enabled_with_env(
                value,
                GgmlCpuGraphBackend::Cpu
            ));
        }
        for value in [None, Some(""), Some("auto"), Some("blas"), Some("default")] {
            assert!(GgmlCpuGraphConfig::cpu_accelerator_enabled_with_env(
                value,
                GgmlCpuGraphBackend::Cpu
            ));
        }
        assert!(!GgmlCpuGraphConfig::cpu_accelerator_enabled_with_env(
            None,
            GgmlCpuGraphBackend::Metal
        ));
        assert!(!GgmlCpuGraphConfig::cpu_accelerator_enabled_with_env(
            None,
            GgmlCpuGraphBackend::Gpu
        ));
    }
}
