use std::{
    collections::HashSet,
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::{Component, Path, PathBuf},
    ptr::{self, NonNull},
    time::Instant,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::ffi;
use crate::registry::{live_backend_driver_floor, load_model_catalog_from_verified_cache};
use crate::{
    CatalogBackendVendor, ExecutionProvider,
    backend_distribution::{
        ActivatedBackendPack, BackendHostAbi, QualificationBackendPack,
        catalog_backend_accepts_device_target, qualification_backend_from_environment,
        read_activated_backend, require_catalog_backend_activated,
    },
    load_local_catalog_file_with_identity,
    pe_image_identity::{
        BackendBundleContractEntry, backend_bundle_contract_sha256, pe_image_identity,
    },
    pull::{
        backend_artifact_fingerprint, backend_pack_install_dir,
        read_and_verify_installed_backend_for_activation,
    },
    resolve_catalog_backend_pull, resolve_compatible_catalog_backend_pull_for_driver,
    resolve_local_catalog_env_override,
};

pub const OPTIONAL_BACKEND_PACK_ENV: &str = "OPENASR_BACKEND_PLUGIN_ID";
#[cfg(all(feature = "backend-plugin-development", debug_assertions))]
const OPTIONAL_BACKEND_TARGET_ENV: &str = "OPENASR_BACKEND_PLUGIN_TARGET";

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BackendPluginActivationError {
    #[error("an optional backend plugin was selected in a host without GGML_BACKEND_DL")]
    DynamicLoadingUnavailable,
    #[error("OpenASR home could not be resolved: {0}")]
    Home(String),
    #[error("the signed backend catalog could not be loaded: {0}")]
    Catalog(String),
    #[error("backend activation state is invalid: {0}")]
    ActivationState(String),
    #[error("backend plugin environment overrides are available only in debug builds")]
    DevelopmentOverrideUnavailable,
    #[error("the bundled backend directory could not be resolved safely: {0}")]
    BundledDirectory(String),
    #[error("backend pack '{backend_id}' could not be resolved: {reason}")]
    Resolution { backend_id: String, reason: String },
    #[error("backend pack '{backend_id}' is not compatible with this host ABI")]
    HostAbiMismatch { backend_id: String },
    #[error(
        "backend pack '{backend_id}' is not installed or failed integrity verification: {reason}"
    )]
    InstalledPackInvalid { backend_id: String, reason: String },
    #[error("backend pack '{backend_id}' resolved outside its verified install directory")]
    EscapedInstallDirectory { backend_id: String },
    #[error("backend pack '{backend_id}' has a path that cannot be represented as UTF-8")]
    NonUtf8Path { backend_id: String },
    #[error("backend pack '{backend_id}' could not be loaded by ggml")]
    LoadFailed { backend_id: String },
    #[error("backend pack '{backend_id}' failed live device/driver attestation")]
    LiveProbeFailed { backend_id: String },
    #[error("backend pack '{backend_id}' registered '{actual}', expected provider '{expected}'")]
    ProviderMismatch {
        backend_id: String,
        expected: &'static str,
        actual: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlRuntimeInfo {
    pub cpu_backend_name: String,
    pub best_backend_name: Option<String>,
    pub metal_backend_name: Option<String>,
    pub devices: Vec<GgmlBackendDevice>,
    pub cpu_features: GgmlCpuFeatures,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgmlBackendDevice {
    raw: NonNull<c_void>,
    pub name: String,
    pub description: String,
    pub kind: GgmlBackendKind,
    pub memory: Option<GgmlDeviceMemory>,
    /// Buffer-type-reported allocation alignment in bytes for this device
    /// (`ggml_backend_buft_get_alignment` on the device's default buffer
    /// type). Backend-agnostic: this is the same generic ggml call for every
    /// backend, it just happens to surface each backend's own notion of
    /// alignment -- for Vulkan specifically it is
    /// `VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment`. Surfaced so
    /// a misaligned-buffer crash report (see issues #153/#154/#155) can be
    /// diagnosed from the boot log alone instead of asking the reporter to
    /// re-run with a debug build.
    pub buffer_alignment: Option<usize>,
    /// Optional stable physical identity from `ggml_backend_dev_props.device_id`.
    /// For PCI devices ggml documents this as lower-case
    /// `domain:bus:device.function` (e.g. `0000:c1:00.0`). CUDA/HIP populate it;
    /// Vulkan does when the instance exposes a PCI bus id; Metal leaves it
    /// null (system-default device only). Consumed by execution-route Exact
    /// resolution and cache/worker isolation keys.
    pub device_id: Option<String>,
    /// PCI vendor id reported by an optional backend registry procedure.
    /// `None` is deliberately preserved as unknown; routing code must never
    /// substitute a device-description/name heuristic for this fact.
    pub pci_vendor_id: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlDeviceMemory {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlBackendKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Meta,
    Unknown(i32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GgmlCpuFeatures {
    pub sse3: bool,
    pub ssse3: bool,
    pub avx: bool,
    pub avx_vnni: bool,
    pub avx2: bool,
    pub bmi2: bool,
    pub f16c: bool,
    pub fma: bool,
    pub avx512: bool,
    pub avx512_vbmi: bool,
    pub avx512_vnni: bool,
    pub avx512_bf16: bool,
    pub amx_int8: bool,
    pub neon: bool,
    pub arm_fma: bool,
    pub fp16_va: bool,
    pub dotprod: bool,
    pub matmul_int8: bool,
    pub sve: bool,
    pub sve_vector_bytes: i32,
    pub sme: bool,
    pub riscv_v: bool,
    pub rvv_vector_bytes: i32,
    pub vsx: bool,
    pub vxe: bool,
    pub wasm_simd: bool,
    pub llamafile: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GgmlRuntimeError {
    #[error("ggml backend is unavailable: {0}")]
    BackendUnavailable(&'static str),
    #[error(transparent)]
    BackendPluginActivation(#[from] BackendPluginActivationError),
}

pub struct GgmlBackend {
    raw: NonNull<c_void>,
}

impl GgmlRuntimeInfo {
    pub fn detect() -> Self {
        ggml_runtime_info()
    }
}

impl GgmlBackendDevice {
    pub fn initialize(&self) -> Result<GgmlBackend, GgmlRuntimeError> {
        let raw = unsafe { ffi::ggml_backend_dev_init(self.raw.as_ptr(), ptr::null()) };
        GgmlBackend::from_raw(raw, "device")
    }

    /// Whether this device can execute a `mul_mat` whose weight operand has
    /// `weight_ggml_type` (f32 activations). This is the load-time correctness
    /// probe for the single-backend direct-run lane (design S3): that lane drops
    /// the multi-backend scheduler's `op_offload` fallback, so a weight whose
    /// matmul the device cannot run must be materialized as a supported GPU type
    /// or the stage must fail closed / divert to CPU before execution. CPU-buffer
    /// fallback is only safe on scheduler-backed paths. Mirrors whisper.cpp
    /// `weight_buft_supported`: build a throwaway `no_alloc` op tensor and ask the
    /// device. `k` is a multiple of 256 so the probe is valid for every
    /// quantization (Q*_K superblocks as well as Q8_0/Q4_0 blocks).
    pub fn supports_matmul_for_type(&self, weight_ggml_type: c_int) -> bool {
        device_supports_matmul_for_type(self.raw, weight_ggml_type)
    }

    /// Whether this device reports support for native `GGML_OP_ARGMAX_FIRST`.
    ///
    /// This is the `ggml_backend_dev_supports_op` declaration only. It is not
    /// production compact-output authorization; the shared planner still
    /// requires the proven evidence dimensions / three-layer receipts.
    pub(crate) fn supports_argmax_first(&self) -> bool {
        device_supports_argmax_first(self.raw)
    }

    /// Whether this device reports support for parameterized `GGML_UNARY_OP_SWOOSH`.
    #[cfg(test)]
    pub(crate) fn supports_swoosh(&self) -> bool {
        device_supports_swoosh(self.raw)
    }

    /// Probe the representative weight types and report which the device can run
    /// `mul_mat` for. Surface for `doctor` diagnostics and load-time weight
    /// placement; on a discrete GPU with narrow quant coverage (e.g. Vulkan)
    /// some entries report `false`, which is exactly the signal to materialize
    /// those weights as a supported type or avoid the direct GPU lane.
    pub fn supported_matmul_weight_types(&self) -> Vec<(&'static str, bool)> {
        MATMUL_WEIGHT_TYPES
            .iter()
            .map(|(name, ggml_type)| (*name, self.supports_matmul_for_type(*ggml_type)))
            .collect()
    }

    /// Build a metadata-only device for tests that exercise pure enumeration
    /// shaping (name/description/kind/memory) without a live ggml backend. The
    /// `raw` handle is dangling and must never be initialized or probed; only
    /// the shaping consumers that read the public fields are valid callers.
    #[cfg(test)]
    pub(crate) fn for_test(
        name: &str,
        description: &str,
        kind: GgmlBackendKind,
        memory: Option<GgmlDeviceMemory>,
    ) -> Self {
        Self::for_test_with_device_id(name, description, kind, memory, None)
    }

    /// Test helper that also sets ggml `device_id` (PCI BDF when simulating
    /// CUDA/HIP/Vulkan identity).
    #[cfg(test)]
    pub(crate) fn for_test_with_device_id(
        name: &str,
        description: &str,
        kind: GgmlBackendKind,
        memory: Option<GgmlDeviceMemory>,
        device_id: Option<&str>,
    ) -> Self {
        Self::for_test_with_hardware_facts(name, description, kind, memory, device_id, None)
    }

    #[cfg(test)]
    pub(crate) fn for_test_with_hardware_facts(
        name: &str,
        description: &str,
        kind: GgmlBackendKind,
        memory: Option<GgmlDeviceMemory>,
        device_id: Option<&str>,
        pci_vendor_id: Option<u32>,
    ) -> Self {
        Self {
            raw: NonNull::dangling(),
            name: name.to_string(),
            description: description.to_string(),
            kind,
            memory,
            buffer_alignment: None,
            device_id: device_id.map(str::to_string),
            pci_vendor_id,
        }
    }
}

/// The float + quantized weight types OpenASR materializes as direct-GPU
/// `mul_mat` operands. Single source of truth for the load-time placement probe
/// ([`device_supports_matmul_for_type`]), the loaded-context candidate filter
/// ([`is_known_matmul_weight_type`]), and the `doctor` diagnostics surface
/// ([`GgmlBackendDevice::supported_matmul_weight_types`]).
pub(crate) const MATMUL_WEIGHT_TYPES: &[(&str, c_int)] = &[
    ("f32", ffi::GGML_TYPE_F32),
    ("f16", ffi::GGML_TYPE_F16),
    ("q8_0", ffi::GGML_TYPE_Q8_0),
    ("q4_0", ffi::GGML_TYPE_Q4_0),
    ("q4_k", ffi::GGML_TYPE_Q4_K),
    ("q5_k", ffi::GGML_TYPE_Q5_K),
    ("q6_k", ffi::GGML_TYPE_Q6_K),
    ("q3_k", ffi::GGML_TYPE_Q3_K),
];

/// Whether `weight_ggml_type` is one of the direct-GPU matmul weight types
/// OpenASR knows how to place (see [`MATMUL_WEIGHT_TYPES`]).
pub(crate) fn is_known_matmul_weight_type(weight_ggml_type: c_int) -> bool {
    MATMUL_WEIGHT_TYPES
        .iter()
        .any(|(_, ggml_type)| *ggml_type == weight_ggml_type)
}

/// Load-time `mul_mat` weight-type probe shared by
/// [`GgmlBackendDevice::supports_matmul_for_type`] and the cpu_graph
/// direct-placement validator. The single-backend direct-run lane (design S3)
/// drops the multi-backend scheduler's `op_offload` fallback, so a weight whose
/// matmul the device cannot run must be materialized as a supported GPU type or
/// the stage must fail closed / divert to CPU before execution. Mirrors
/// whisper.cpp `weight_buft_supported`: build a throwaway `no_alloc` op tensor
/// and ask the device. `k` is a multiple of 256 so the probe is valid for every
/// quantization (Q*_K superblocks as well as Q8_0/Q4_0 blocks).
pub(crate) fn device_supports_matmul_for_type(
    device: NonNull<c_void>,
    weight_ggml_type: c_int,
) -> bool {
    const K: i64 = 256;
    const M: i64 = 32;
    const N: i64 = 8;
    let params = ffi::GgmlInitParams {
        mem_size: 16 * 1024,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    unsafe {
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return false;
        }
        let weight = ffi::ggml_new_tensor_2d(ctx, weight_ggml_type, K, M);
        let activation = ffi::ggml_new_tensor_2d(ctx, ffi::GGML_TYPE_F32, K, N);
        let supported = if weight.is_null() || activation.is_null() {
            false
        } else {
            let op = ffi::ggml_mul_mat(ctx, weight, activation);
            !op.is_null() && ffi::ggml_backend_dev_supports_op(device.as_ptr(), op)
        };
        ffi::ggml_free(ctx);
        supported
    }
}

/// Load-time `GGML_OP_ARGMAX_FIRST` probe. Builds a throwaway contiguous F32
/// row matrix through the existing `ggml_argmax_first` FFI symbol and asks
/// `ggml_backend_dev_supports_op`. This is the native-operator declaration
/// seam; it cannot authorize compact token output by itself.
pub(crate) fn device_supports_argmax_first(device: NonNull<c_void>) -> bool {
    const COLS: i64 = 4;
    const ROWS: i64 = 1;
    let params = ffi::GgmlInitParams {
        mem_size: 16 * 1024,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    unsafe {
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return false;
        }
        let logits = ffi::ggml_new_tensor_2d(ctx, ffi::GGML_TYPE_F32, COLS, ROWS);
        let supported = if logits.is_null() {
            false
        } else {
            let op = ffi::ggml_argmax_first(ctx, logits);
            !op.is_null() && ffi::ggml_backend_dev_supports_op(device.as_ptr(), op)
        };
        ffi::ggml_free(ctx);
        supported
    }
}

/// Load-time `GGML_UNARY_OP_SWOOSH` probe. Builds a throwaway contiguous F32
/// vector through `ggml_swoosh` and asks `ggml_backend_dev_supports_op`.
#[cfg(test)]
pub(crate) fn device_supports_swoosh(device: NonNull<c_void>) -> bool {
    const N: i64 = 8;
    let params = ffi::GgmlInitParams {
        mem_size: 16 * 1024,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    unsafe {
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return false;
        }
        let input = ffi::ggml_new_tensor_1d(ctx, ffi::GGML_TYPE_F32, N);
        let supported = if input.is_null() {
            false
        } else {
            let op = ffi::ggml_swoosh(ctx, input, 1.0, -0.08, 0.08);
            !op.is_null() && ffi::ggml_backend_dev_supports_op(device.as_ptr(), op)
        };
        ffi::ggml_free(ctx);
        supported
    }
}

impl GgmlBackendKind {
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu | Self::IntegratedGpu)
    }
}

/// Priority for picking among several simultaneously-available accelerated
/// devices (the Optimus/hybrid-graphics case: a Vulkan-capable laptop
/// enumerates both its Intel integrated GPU and its NVIDIA/AMD discrete GPU
/// as separate devices through the same backend). Lower ranks first.
///
/// A discrete GPU (`Gpu`, `ggml`'s `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` on
/// the Vulkan backend) has its own VRAM and is essentially always the
/// faster, less contended choice, so it ranks first. An `Accelerator`
/// (NPU-class device) also has dedicated silicon and ranks second. An
/// integrated GPU (`IntegratedGpu`) shares system RAM and memory bandwidth
/// with the CPU and ranks last among the "accelerated" kinds -- it should
/// only be picked when nothing better is present. Devices that tie on rank
/// (e.g. two discrete GPUs) keep the registry's original enumeration order:
/// we have no other signal to break that tie, and
/// [`preferred_accelerated_device`] relies on `Iterator::min_by_key`
/// returning the first minimal element to preserve it.
pub(crate) fn accelerated_device_rank(kind: GgmlBackendKind) -> u8 {
    match kind {
        GgmlBackendKind::Gpu => 0,
        GgmlBackendKind::Accelerator => 1,
        GgmlBackendKind::IntegratedGpu => 2,
        _ => 3,
    }
}

/// Minimum free memory a device must report to be considered a viable
/// model-load target. Conservative floor: most OpenASR models need roughly
/// 500 MB - 1 GB of weights, so a device that cannot even guarantee 512 MiB
/// free is too loaded to be useful -- a hybrid-graphics laptop whose discrete
/// VRAM is consumed by other workloads (browser, game) is better served by
/// its integrated GPU (shared system RAM, often 16 GB+) than by a
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY` on the discrete device and the resulting
/// fall-through to CPU.
const MIN_VIABLE_FREE_VRAM_BYTES: usize = 512 * 1024 * 1024;

/// Whether `device` reports enough free memory to be a viable model-load
/// target (see [`MIN_VIABLE_FREE_VRAM_BYTES`]). Devices that report no memory
/// information at all are treated as viable: absence of a report must not
/// penalize a backend that does not surface `ggml_backend_dev_memory`.
fn device_reports_viable_free_vram(device: &GgmlBackendDevice) -> bool {
    device
        .memory
        .is_none_or(|memory| memory.free_bytes >= MIN_VIABLE_FREE_VRAM_BYTES)
}

/// Why [`select_accelerated_device`] picked the device it did. Surfaced in the
/// daemon boot log (`gpu_selection=... rule=...`) so a "wrong GPU used" report
/// is diagnosable from `daemon.log` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AcceleratedDeviceSelectionRule {
    /// The best-ranked accelerated device reports viable free memory (or none
    /// at all); kind ranking alone decided.
    KindRanking,
    /// A better-ranked accelerated device was skipped because it reported less
    /// than [`MIN_VIABLE_FREE_VRAM_BYTES`] free; the pick is the best-ranked
    /// device among those with viable free memory.
    LowVramSkipped,
    /// No accelerated device reported viable free memory; selection fell back
    /// to kind-only ranking (the historical behavior) so ggml still tries the
    /// best-ranked device and drives its own OOM fallback.
    LowVramFallback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActivatedBackendRuntime {
    backend_id: String,
    device_target: String,
    driver_version: String,
    artifact_fingerprint: String,
    provider: ExecutionProvider,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActivatedBackendExecutionIdentity {
    pub backend_id: String,
    pub device_target: String,
    pub driver_version: String,
    pub artifact_fingerprint: String,
    pub provider: ExecutionProvider,
}

impl AcceleratedDeviceSelectionRule {
    /// The `rule=` value used in the boot summary's `gpu_selection` field.
    fn boot_log_label(self) -> &'static str {
        match self {
            Self::KindRanking => "discrete_over_integrated",
            Self::LowVramSkipped => "discrete_low_vram_skipped",
            Self::LowVramFallback => "all_low_vram_kind_fallback",
        }
    }
}

/// Pick the device to prefer when more than one accelerated device is
/// available and record why it won (see [`AcceleratedDeviceSelectionRule`]).
/// Kind ranking ([`accelerated_device_rank`]) decides, except that a device
/// reporting less than [`MIN_VIABLE_FREE_VRAM_BYTES`] of free memory is
/// skipped in favor of a viable lower-ranked one: on a hybrid-graphics laptop
/// whose discrete GPU's VRAM is largely consumed by other workloads, loading
/// model weights onto it fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` and the
/// run falls back to CPU, even though the integrated GPU (shared system RAM)
/// has ample space. If NO accelerated device reports viable free memory the
/// selection falls back to kind-only ranking (the historical behavior), so
/// ggml still attempts the best-ranked device and drives its own OOM
/// fallback. Devices that report no memory information are never penalized.
fn select_accelerated_device(
    devices: &[GgmlBackendDevice],
    is_accelerated: impl Fn(GgmlBackendKind) -> bool,
) -> Option<(&GgmlBackendDevice, AcceleratedDeviceSelectionRule)> {
    select_accelerated_device_for_provider(
        devices,
        is_accelerated,
        activated_backend_execution_provider(),
        cfg!(target_os = "windows"),
    )
}

fn select_accelerated_device_for_provider(
    devices: &[GgmlBackendDevice],
    is_accelerated: impl Fn(GgmlBackendKind) -> bool,
    preferred_provider: Option<ExecutionProvider>,
    require_activated_provider: bool,
) -> Option<(&GgmlBackendDevice, AcceleratedDeviceSelectionRule)> {
    // An activated optional pack is an explicit process-wide provider choice,
    // not merely another device appended to the registry. The CPU-neutral host
    // has no bundled GPU recovery rail, so registry ordinal must never make an
    // optional provider appear selected. Constrain ranking to the signed,
    // Activated-only provider when it has a live accelerated device; if its
    // inventory drifts, return no accelerated device rather than manufacturing
    // a device or selecting another installed provider.
    if require_activated_provider && preferred_provider.is_none() {
        return None;
    }
    let has_preferred_provider = preferred_provider.is_some_and(|provider| {
        devices.iter().any(|device| {
            is_accelerated(device.kind)
                && ExecutionProvider::from_backend_name(&device.name) == provider
        })
    });
    if preferred_provider.is_some() && !has_preferred_provider {
        return None;
    }
    let eligible = |device: &&GgmlBackendDevice| {
        is_accelerated(device.kind)
            && (!has_preferred_provider
                || preferred_provider.is_some_and(|provider| {
                    ExecutionProvider::from_backend_name(&device.name) == provider
                }))
    };
    // Host-only discovery resolves a target-scoped optional pack from the
    // provider's primary (ordinal-zero) device. Preserve that same registry
    // order after activation; a free-VRAM heuristic must not move an sm/gfx
    // specific module onto another, incompatible adapter.
    if has_preferred_provider {
        return devices
            .iter()
            .find(eligible)
            .map(|device| (device, AcceleratedDeviceSelectionRule::KindRanking));
    }
    let kind_pick = devices
        .iter()
        .filter(eligible)
        .min_by_key(|device| accelerated_device_rank(device.kind))?;
    if device_reports_viable_free_vram(kind_pick) {
        return Some((kind_pick, AcceleratedDeviceSelectionRule::KindRanking));
    }
    match devices
        .iter()
        .filter(eligible)
        .filter(|device| device_reports_viable_free_vram(device))
        .min_by_key(|device| accelerated_device_rank(device.kind))
    {
        Some(viable_pick) => Some((viable_pick, AcceleratedDeviceSelectionRule::LowVramSkipped)),
        None => Some((kind_pick, AcceleratedDeviceSelectionRule::LowVramFallback)),
    }
}

/// Pick the device to prefer when more than one accelerated device is
/// available: the free-VRAM-aware kind ranking of
/// [`select_accelerated_device`]. `is_accelerated` lets each caller keep its
/// own notion of "counts as an accelerated device" (today's callers want
/// `Gpu`/`IntegratedGpu`; an NPU-aware initializer could fold in
/// `Accelerator`), while the tie-break/ranking logic stays in one place so
/// device selection can never drift from the diagnostics that describe it.
pub(crate) fn preferred_accelerated_device(
    devices: &[GgmlBackendDevice],
    is_accelerated: impl Fn(GgmlBackendKind) -> bool,
) -> Option<&GgmlBackendDevice> {
    select_accelerated_device(devices, is_accelerated).map(|(device, _rule)| device)
}

/// Register backend modules once per process before the first registry query.
/// A dynamic host loads only the bundled CPU rescue module and at most one
/// exact, signed, integrity-verified optional GPU pack. Static hosts never load
/// modules, avoiding the double-ggml global-state collision seen when a static
/// GPU host loads a dynamic module.
pub(crate) fn ensure_backends_loaded() {
    let _ = bundled_cpu_activation_cell().get_or_init(load_bundled_cpu_module);
    backend_plugin_activation_cell().get_or_init(|| {
        bundled_cpu_activation_cell()
            .get()
            .expect("bundled CPU activation cell initialized")
            .clone()?;
        activate_selected_backend_plugin()
    });
}

fn bundled_cpu_activation_cell()
-> &'static std::sync::OnceLock<Result<(), BackendPluginActivationError>> {
    static CPU: std::sync::OnceLock<Result<(), BackendPluginActivationError>> =
        std::sync::OnceLock::new();
    &CPU
}

struct BundledProviderPaths {
    paths: Vec<CString>,
    host_abi_fingerprint: String,
}

fn discover_bundled_cpu_modules()
-> Result<Option<BundledProviderPaths>, BackendPluginActivationError> {
    if !ggml_backend_dl_build_enabled() {
        return Ok(None);
    }
    let executable = std::env::current_exe()
        .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
    let executable_directory = executable.parent().ok_or_else(|| {
        BackendPluginActivationError::BundledDirectory(
            "current executable has no parent directory".to_string(),
        )
    })?;
    let executable_directory = std::fs::canonicalize(executable_directory)
        .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
    if !executable_directory.is_absolute() {
        return Err(BackendPluginActivationError::BundledDirectory(
            "resolved directory is not absolute".to_string(),
        ));
    }
    let current_abi = BackendHostAbi::current();
    let directory = bundled_backend_directory_for_host(&executable_directory, &current_abi);
    let manifest_path = directory.join("openasr-backend-bundle-v1.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
    let manifest: BundledBackendManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
    validate_bundled_backend_manifest(&manifest)
        .map_err(BackendPluginActivationError::BundledDirectory)?;
    if manifest.host_abi_fingerprint != current_abi.fingerprint {
        return Err(BackendPluginActivationError::BundledDirectory(
            "bundled backend manifest does not match this neutral host ABI".to_string(),
        ));
    }
    let expected_provider_contract = option_env!("OPENASR_BUNDLED_CPU_CONTRACT_SHA256")
        .ok_or_else(|| {
            BackendPluginActivationError::BundledDirectory(
                "neutral host has no embedded bundled-CPU contract".to_string(),
            )
        })?;
    let manifest_provider_contract = &manifest.cpu_contract_sha256;
    let mut contract_entries = Vec::new();
    let mut provider_paths = Vec::new();
    for file in manifest
        .files
        .iter()
        .filter(|file| file.provider == "host" || file.provider == "cpu")
    {
        let relative = Path::new(&file.filename);
        if relative.components().count() != 1
            || !matches!(relative.components().next(), Some(Component::Normal(_)))
        {
            return Err(BackendPluginActivationError::BundledDirectory(
                "bundled backend manifest contains an unsafe filename".to_string(),
            ));
        }
        let path = std::fs::canonicalize(directory.join(relative))
            .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
        if !path.starts_with(&directory) {
            return Err(BackendPluginActivationError::BundledDirectory(
                "bundled backend file escaped the application directory".to_string(),
            ));
        }
        let bytes = std::fs::read(&path)
            .map_err(|error| BackendPluginActivationError::BundledDirectory(error.to_string()))?;
        if bytes.len() as u64 != file.size_bytes
            || format!("{:x}", Sha256::digest(&bytes)) != file.sha256
        {
            return Err(BackendPluginActivationError::BundledDirectory(format!(
                "bundled backend file '{}' failed integrity verification",
                file.filename
            )));
        }
        let image = pe_image_identity(&bytes).map_err(|error| {
            BackendPluginActivationError::BundledDirectory(format!(
                "bundled backend file '{}' is not a valid PE image: {error}",
                file.filename
            ))
        })?;
        if image.sha256 != file.image_sha256 || image.size_bytes != file.image_size_bytes {
            return Err(BackendPluginActivationError::BundledDirectory(format!(
                "bundled backend file '{}' failed stable PE identity verification",
                file.filename
            )));
        }
        contract_entries.push(BackendBundleContractEntry {
            filename: file.filename.clone(),
            provider: file.provider.clone(),
            image_sha256: image.sha256,
            image_size_bytes: image.size_bytes,
        });
        match file.provider.as_str() {
            "cpu" => provider_paths.push(path_to_utf8_cstring("bundled-cpu", &path)?),
            "host" => {}
            _ => {
                return Err(BackendPluginActivationError::BundledDirectory(
                    "bundled backend manifest contains an unknown provider".to_string(),
                ));
            }
        }
    }
    let actual_provider_contract =
        backend_bundle_contract_sha256(&manifest.host_abi_fingerprint, &contract_entries);
    if actual_provider_contract != manifest_provider_contract.as_str()
        || actual_provider_contract != expected_provider_contract
    {
        return Err(BackendPluginActivationError::BundledDirectory(format!(
            "bundled CPU modules do not match the provider contract embedded in this neutral host \
             (actual={actual_provider_contract}, manifest={manifest_provider_contract}, host={expected_provider_contract})"
        )));
    }
    if provider_paths.is_empty() {
        return Err(BackendPluginActivationError::BundledDirectory(
            "neutral Windows bundle has no CPU rescue module".to_string(),
        ));
    }
    Ok(Some(BundledProviderPaths {
        paths: provider_paths,
        host_abi_fingerprint: current_abi.fingerprint,
    }))
}

fn load_bundled_cpu_module() -> Result<(), BackendPluginActivationError> {
    let paths = discover_bundled_cpu_modules()?;
    let Some(paths) = paths else {
        return Ok(());
    };
    load_best_bundled_backend(&paths.paths, "cpu", &paths.host_abi_fingerprint)
}

fn bundled_backend_directory_for_host(
    executable_directory: &Path,
    abi: &BackendHostAbi,
) -> PathBuf {
    let exact = executable_directory
        .join("openasr-backend-bundles")
        .join(&abi.fingerprint);
    if exact.join("openasr-backend-bundle-v1.json").is_file() {
        exact
    } else {
        // Release bundles intentionally keep the public, flat layout. The
        // ABI-scoped directory exists in Cargo target trees so simultaneous
        // feature combinations cannot overwrite one another's local runtime.
        executable_directory.to_path_buf()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundledBackendManifest {
    schema_version: u32,
    host_abi_fingerprint: String,
    bundle_contract_sha256: String,
    cpu_contract_sha256: String,
    files: Vec<BundledBackendManifestFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundledBackendManifestFile {
    filename: String,
    provider: String,
    sha256: String,
    size_bytes: u64,
    image_sha256: String,
    image_size_bytes: u64,
}

fn validate_bundled_backend_manifest(manifest: &BundledBackendManifest) -> Result<(), String> {
    if manifest.schema_version != 4 {
        return Err("bundled backend manifest has an unsupported schema".to_string());
    }
    if manifest.host_abi_fingerprint.len() != 64
        || !manifest
            .host_abi_fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("bundled backend manifest has an invalid host ABI fingerprint".to_string());
    }
    for (label, contract) in [
        ("bundle", &manifest.bundle_contract_sha256),
        ("CPU", &manifest.cpu_contract_sha256),
    ] {
        if contract.len() != 64
            || !contract
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(format!(
                "bundled backend manifest has an invalid {label} contract"
            ));
        }
    }
    if !(3..=64).contains(&manifest.files.len()) {
        return Err("bundled backend manifest has an invalid file count".to_string());
    }

    let mut filenames = HashSet::new();
    let mut host = HashSet::new();
    let mut cpu_count = 0_usize;
    for file in &manifest.files {
        let name = file.filename.to_ascii_lowercase();
        if !filenames.insert(name.clone()) {
            return Err("bundled backend manifest contains duplicate filenames".to_string());
        }
        if file.size_bytes == 0
            || file.sha256.len() != 64
            || !file
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || file.image_size_bytes == 0
            || file.image_sha256.len() != 64
            || !file
                .image_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(format!(
                "bundled backend file '{}' has an invalid byte identity",
                file.filename
            ));
        }
        match file.provider.as_str() {
            "host" if matches!(name.as_str(), "ggml.dll" | "ggml-base.dll") => {
                host.insert(name);
            }
            "cpu" if name.starts_with("ggml-cpu") && name.ends_with(".dll") => {
                cpu_count += 1;
            }
            _ => {
                return Err(format!(
                    "bundled backend file '{}' has an invalid provider role",
                    file.filename
                ));
            }
        }
    }
    if host != HashSet::from(["ggml.dll".to_string(), "ggml-base.dll".to_string()])
        || cpu_count == 0
    {
        return Err(
            "neutral bundle must contain exactly one host pair and CPU candidates".to_string(),
        );
    }
    Ok(())
}

fn load_best_bundled_backend(
    paths: &[CString],
    provider: &'static str,
    host_abi_fingerprint: &str,
) -> Result<(), BackendPluginActivationError> {
    let path_ptrs = paths.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
    let abi = CString::new(host_abi_fingerprint).expect("backend ABI fingerprint is hexadecimal");
    let provider_c = CString::new(provider).expect("provider is static ASCII");
    let reg = unsafe {
        ffi::ggml_backend_load_best_verified_utf8(
            path_ptrs.as_ptr(),
            path_ptrs.len(),
            abi.as_ptr(),
            provider_c.as_ptr(),
        )
    };
    NonNull::new(reg)
        .map(|_| ())
        .ok_or_else(|| BackendPluginActivationError::LoadFailed {
            backend_id: format!("bundled-{provider}"),
        })
}

fn path_to_utf8_cstring(
    backend_id: &str,
    path: &Path,
) -> Result<CString, BackendPluginActivationError> {
    // hipBLASLt / rocBLAS resolve Tensile data next to the loaded module.
    // A `\\?\` extended path from canonicalize makes that join crash at
    // first GEMM; pass the unextended absolute path to the Windows loader.
    let path = crate::backend_distribution::path_for_vendor_env(path);
    let path = path
        .to_str()
        .ok_or_else(|| BackendPluginActivationError::NonUtf8Path {
            backend_id: backend_id.to_string(),
        })?;
    CString::new(path).map_err(|_| BackendPluginActivationError::NonUtf8Path {
        backend_id: backend_id.to_string(),
    })
}

/// Process-global result of the optional backend candidate transaction. Rescue
/// modules stay usable after an optional failure, while attestation can inspect
/// this result and refuse to claim CUDA/HIP activation.
fn backend_plugin_activation_cell() -> &'static std::sync::OnceLock<
    Result<Option<ActivatedBackendRuntime>, BackendPluginActivationError>,
> {
    static ACTIVATION: std::sync::OnceLock<
        Result<Option<ActivatedBackendRuntime>, BackendPluginActivationError>,
    > = std::sync::OnceLock::new();
    &ACTIVATION
}

pub fn backend_plugin_activation_status() -> Result<Option<String>, BackendPluginActivationError> {
    ensure_backends_loaded();
    bundled_cpu_activation_cell()
        .get()
        .expect("bundled CPU activation cell initialized")
        .clone()?;
    backend_plugin_activation_cell()
        .get()
        .expect("backend activation cell initialized")
        .clone()
        .map(|activated| activated.map(|activated| activated.backend_id))
}

/// Provider selected by the successfully completed optional-pack transaction.
///
/// This intentionally does not initialize the activation cell: callers use it
/// only after backend enumeration has completed.  During the live probe inside
/// activation the cell is still being initialized, and recursively waiting on
/// it would deadlock.
pub(crate) fn activated_backend_execution_provider() -> Option<ExecutionProvider> {
    backend_plugin_activation_cell()
        .get()
        .and_then(|result| result.as_ref().ok())
        .and_then(Option::as_ref)
        .map(|activated| activated.provider)
}

/// Exact signed optional-backend identity selected by the completed process
/// activation transaction. A provider label alone is never qualification
/// evidence.
pub(crate) fn activated_backend_execution_identity() -> Option<ActivatedBackendExecutionIdentity> {
    backend_plugin_activation_cell()
        .get()
        .and_then(|result| result.as_ref().ok())
        .and_then(Option::as_ref)
        .map(|activated| ActivatedBackendExecutionIdentity {
            backend_id: activated.backend_id.clone(),
            device_target: activated.device_target.clone(),
            driver_version: activated.driver_version.clone(),
            artifact_fingerprint: activated.artifact_fingerprint.clone(),
            provider: activated.provider,
        })
}

/// Whether this binary is the neutral dynamic-backend host used by the
/// plugin distribution. Static CUDA/HIP/Vulkan sidecars deliberately report
/// false even though they expose the same CLI surface during the one-release
/// migration window.
pub fn backend_plugin_host_available() -> bool {
    ggml_backend_dl_build_enabled()
}

/// Verifies and loads only the bundled CPU rescue module. Optional GPU
/// activation is intentionally excluded so a broken selected plugin cannot
/// make host-topology discovery unavailable to the Desktop recovery path.
pub fn bundled_backend_activation_status() -> Result<(), BackendPluginActivationError> {
    bundled_cpu_activation_cell()
        .get_or_init(load_bundled_cpu_module)
        .clone()
}

#[cfg(windows)]
fn lock_verified_backend_load_files(
    backend_id: &str,
    plugin_path: &Path,
    dependency_dirs: &[PathBuf],
) -> Result<Vec<std::fs::File>, BackendPluginActivationError> {
    use std::{fs::OpenOptions, os::windows::fs::OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;

    fn collect_dlls(directory: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "backend dependency tree contains a symlink",
                ));
            }
            if file_type.is_dir() {
                collect_dlls(&entry.path(), files)?;
            } else if file_type.is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
            {
                files.push(entry.path());
            }
        }
        Ok(())
    }

    let mut paths = vec![plugin_path.to_path_buf()];
    for directory in dependency_dirs {
        collect_dlls(directory, &mut paths).map_err(|error| {
            BackendPluginActivationError::InstalledPackInvalid {
                backend_id: backend_id.to_string(),
                reason: format!("could not enumerate verified dependency DLLs: {error}"),
            }
        })?;
    }
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .map(|path| {
            OpenOptions::new()
                .read(true)
                // Deny write and delete until live probe + verified load have
                // completed. This closes the rehash-to-LoadLibrary race while
                // still allowing the Windows loader its own read handle.
                .share_mode(FILE_SHARE_READ)
                .open(&path)
                .map_err(|error| BackendPluginActivationError::InstalledPackInvalid {
                    backend_id: backend_id.to_string(),
                    reason: format!(
                        "could not lock verified backend file '{}' for activation: {error}",
                        path.display()
                    ),
                })
        })
        .collect()
}

#[cfg(not(windows))]
fn lock_verified_backend_load_files(
    _backend_id: &str,
    _plugin_path: &Path,
    _dependency_dirs: &[PathBuf],
) -> Result<Vec<std::fs::File>, BackendPluginActivationError> {
    Ok(Vec::new())
}

fn activate_selected_backend_plugin()
-> Result<Option<ActivatedBackendRuntime>, BackendPluginActivationError> {
    if !ggml_backend_dl_build_enabled() {
        return Err(BackendPluginActivationError::DynamicLoadingUnavailable);
    }

    let home = crate::home::openasr_home()
        .map_err(|error| BackendPluginActivationError::Home(error.to_string()))?;
    let qualification = qualification_backend_from_environment(&home)
        .map_err(|error| BackendPluginActivationError::ActivationState(error.to_string()))?;
    let activated = if qualification.is_none() {
        read_activated_backend(&home)
            .map_err(|error| BackendPluginActivationError::ActivationState(error.to_string()))?
    } else {
        None
    };
    let development = development_backend_override()?;
    let (backend_id, device_target, activated_record, qualification_record) =
        if let Some((backend_id, target)) = development {
            (backend_id, target, None, None)
        } else if let Some(record) = qualification {
            (
                record.backend_id.clone(),
                record.device_target.clone(),
                None,
                Some(record),
            )
        } else if let Some(record) = activated {
            (
                record.backend_id.clone(),
                record.device_target.clone(),
                Some(record),
                None,
            )
        } else {
            return Ok(None);
        };
    // Runtime discovery is commonly called from an async server handler. The
    // activation catalog path below is strictly filesystem-only: install or
    // explicit catalog refresh owns all network I/O.
    let catalog = load_backend_activation_catalog(&home)?;
    let requested = resolve_catalog_backend_pull(&catalog, &backend_id).map_err(|error| {
        BackendPluginActivationError::Resolution {
            backend_id: backend_id.clone(),
            reason: error.to_string(),
        }
    })?;
    if qualification_record.is_some() {
        if matches!(
            requested.activation.state,
            crate::CatalogBackendActivationState::Revoked
                | crate::CatalogBackendActivationState::Unknown
        ) {
            return Err(BackendPluginActivationError::ActivationState(
                "revoked or unknown backend cannot enter qualification".to_string(),
            ));
        }
    } else {
        require_catalog_backend_activated(&requested)
            .map_err(|error| BackendPluginActivationError::ActivationState(error.to_string()))?;
    }
    if !BackendHostAbi::current().is_compatible_with(&requested.host_abi) {
        return Err(BackendPluginActivationError::HostAbiMismatch { backend_id });
    }
    if let Some(record) = activated_record.as_ref() {
        verify_activated_backend_record(record, &requested)?;
    }
    if let Some(record) = qualification_record.as_ref() {
        verify_qualification_backend_record(record, &requested)?;
    }

    let install_dir = backend_pack_install_dir(&home, &requested).map_err(|error| {
        BackendPluginActivationError::Resolution {
            backend_id: backend_id.clone(),
            reason: error.to_string(),
        }
    })?;
    let stage_started = Instant::now();
    let installed = read_and_verify_installed_backend_for_activation(&install_dir, &requested)
        .map_err(|error| BackendPluginActivationError::InstalledPackInvalid {
            backend_id: backend_id.clone(),
            reason: error.to_string(),
        })?;
    crate::stage_timing::log_stage(
        "server_boot",
        "ggml_backend_verify1",
        stage_started.elapsed(),
    );
    let canonical_dir = std::fs::canonicalize(&install_dir).map_err(|error| {
        BackendPluginActivationError::InstalledPackInvalid {
            backend_id: backend_id.clone(),
            reason: error.to_string(),
        }
    })?;
    let plugin_path =
        std::fs::canonicalize(install_dir.join(&installed.plugin_filename)).map_err(|error| {
            BackendPluginActivationError::InstalledPackInvalid {
                backend_id: backend_id.clone(),
                reason: error.to_string(),
            }
        })?;
    if !plugin_path.starts_with(&canonical_dir) {
        return Err(BackendPluginActivationError::EscapedInstallDirectory { backend_id });
    }
    let dependency_dirs = crate::backend_distribution::verified_backend_dependency_dirs(
        &backend_id,
        &canonical_dir,
        &installed,
    )
    .map_err(|error| BackendPluginActivationError::InstalledPackInvalid {
        backend_id: backend_id.clone(),
        reason: error.to_string(),
    })?;
    if requested.vendor == CatalogBackendVendor::Hip {
        crate::backend_distribution::bind_verified_hip_kernel_libpaths(&dependency_dirs);
    }
    let stage_started = Instant::now();
    let _load_guards =
        lock_verified_backend_load_files(&backend_id, &plugin_path, &dependency_dirs)?;
    crate::stage_timing::log_stage("server_boot", "ggml_backend_lock", stage_started.elapsed());
    // Rehash mapped images while write/delete sharing is denied. Both
    // passes use LoadImages: the first establishes identity and image
    // hashes, the second proves the locked DLL bytes still match the
    // signed catalog immediately before any DllMain can run.
    let stage_started = Instant::now();
    read_and_verify_installed_backend_for_activation(&install_dir, &requested).map_err(
        |error| BackendPluginActivationError::InstalledPackInvalid {
            backend_id: backend_id.clone(),
            reason: error.to_string(),
        },
    )?;
    crate::stage_timing::log_stage(
        "server_boot",
        "ggml_backend_verify2",
        stage_started.elapsed(),
    );
    let stage_started = Instant::now();
    let live_driver = probe_exact_backend_plugin_candidate(
        &backend_id,
        requested.vendor,
        &plugin_path,
        &dependency_dirs,
        &device_target,
        live_backend_driver_floor(requested.vendor, requested.min_driver_api.as_deref()),
    )?;
    crate::stage_timing::log_stage("server_boot", "ggml_backend_probe", stage_started.elapsed());
    let expected_driver = activated_record
        .as_ref()
        .map(|record| record.driver_version.as_str())
        .or_else(|| {
            qualification_record
                .as_ref()
                .map(|record| record.driver_version.as_str())
        });
    if expected_driver.is_some_and(|expected| expected != live_driver) {
        return Err(BackendPluginActivationError::ActivationState(
            "live backend driver changed from the exact signed/scoped binding".to_string(),
        ));
    }
    let resolved = resolve_compatible_catalog_backend_pull_for_driver(
        &catalog,
        requested.vendor,
        &BackendHostAbi::current(),
        Some(&device_target),
        Some(&live_driver),
    )
    .map_err(|error| BackendPluginActivationError::Resolution {
        backend_id: backend_id.clone(),
        reason: error.to_string(),
    })?;
    if resolved.backend_id != backend_id {
        return Err(BackendPluginActivationError::Resolution {
            backend_id,
            reason: format!(
                "live device/driver proof resolves to '{}' instead of the selected pack",
                resolved.backend_id
            ),
        });
    }
    let stage_started = Instant::now();
    load_exact_backend_plugin(
        &backend_id,
        resolved.vendor,
        &plugin_path,
        &dependency_dirs,
        &device_target,
        live_backend_driver_floor(resolved.vendor, resolved.min_driver_api.as_deref()),
    )?;
    crate::stage_timing::log_stage("server_boot", "ggml_backend_load", stage_started.elapsed());
    Ok(Some(ActivatedBackendRuntime {
        backend_id,
        device_target,
        driver_version: live_driver,
        artifact_fingerprint: backend_artifact_fingerprint(&resolved),
        provider: execution_provider_for_catalog_vendor(resolved.vendor),
    }))
}

const fn execution_provider_for_catalog_vendor(vendor: CatalogBackendVendor) -> ExecutionProvider {
    match vendor {
        CatalogBackendVendor::Cpu => ExecutionProvider::Cpu,
        CatalogBackendVendor::Cuda => ExecutionProvider::Cuda,
        CatalogBackendVendor::Hip => ExecutionProvider::Hip,
        CatalogBackendVendor::Vulkan => ExecutionProvider::Vulkan,
        CatalogBackendVendor::Unknown => ExecutionProvider::Unknown,
    }
}

/// Resolves the exact same explicitly configured catalog identity used by the
/// CLI/server before falling back to the production catalog.  The optional
/// backend activation pointer names an entry in that catalog, so resolving a
/// different source here can otherwise make a correctly installed plugin
/// disappear (or validate it against the wrong release) in the first runtime
/// process after installation.
///
/// A half-configured local file/identity pair is fail-closed for optional GPU
/// activation. The bundled CPU rescue module was already loaded before
/// this function runs, so recovery remains available without guessing which
/// trust identity the local bytes were meant to use.
fn load_backend_activation_catalog(
    home: &Path,
) -> Result<crate::ModelCatalog, BackendPluginActivationError> {
    let (local, warning) = resolve_local_catalog_env_override();
    if let Some(warning) = warning {
        return Err(BackendPluginActivationError::Catalog(warning));
    }
    if let Some(local) = local {
        return load_local_catalog_file_with_identity(&local.path, &local.identity, home)
            .map_err(|error| BackendPluginActivationError::Catalog(error.to_string()));
    }

    let catalog_url = std::env::var("OPENASR_CATALOG_URL")
        .ok()
        .filter(|value| !value.trim().is_empty());
    load_model_catalog_from_verified_cache(catalog_url.as_deref(), home)
        .map_err(|error| BackendPluginActivationError::Catalog(error.to_string()))
}

#[cfg(all(feature = "backend-plugin-development", debug_assertions))]
fn development_backend_override() -> Result<Option<(String, String)>, BackendPluginActivationError>
{
    let Some(backend_id) = std::env::var_os(OPTIONAL_BACKEND_PACK_ENV)
        .map(|value| value.to_string_lossy().trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let target = std::env::var(OPTIONAL_BACKEND_TARGET_ENV).map_err(|_| {
        BackendPluginActivationError::ActivationState(format!(
            "{OPTIONAL_BACKEND_TARGET_ENV} is required with {OPTIONAL_BACKEND_PACK_ENV}"
        ))
    })?;
    Ok(Some((backend_id, target)))
}

#[cfg(not(all(feature = "backend-plugin-development", debug_assertions)))]
fn development_backend_override() -> Result<Option<(String, String)>, BackendPluginActivationError>
{
    Ok(None)
}

fn verify_activated_backend_record(
    record: &ActivatedBackendPack,
    resolved: &crate::ResolvedCatalogBackendPull,
) -> Result<(), BackendPluginActivationError> {
    let valid = record.vendor == resolved.vendor
        && record.version == resolved.version
        && record.host_abi_fingerprint == resolved.host_abi.fingerprint
        && record.artifact_fingerprint == backend_artifact_fingerprint(resolved)
        && resolved.activation.qualified_device_target.as_deref()
            == Some(record.device_target.as_str())
        && resolved.activation.qualified_driver_version.as_deref()
            == Some(record.driver_version.as_str())
        && record.qualification_source_catalog_sha256
            == resolved
                .activation
                .qualification_source_catalog_sha256
                .as_deref()
                .unwrap_or_default()
        && record.hardware_evidence_sha256
            == resolved
                .activation
                .hardware_evidence_sha256
                .as_deref()
                .unwrap_or_default()
        && record.correctness_matrix_sha256
            == resolved
                .activation
                .correctness_matrix_sha256
                .as_deref()
                .unwrap_or_default()
        && record.correctness_receipts_sha256
            == resolved
                .activation
                .correctness_receipts_sha256
                .as_deref()
                .unwrap_or_default();
    if valid {
        Ok(())
    } else {
        Err(BackendPluginActivationError::ActivationState(format!(
            "activated backend '{}' no longer matches the signed catalog identity",
            record.backend_id
        )))
    }
}

fn verify_qualification_backend_record(
    record: &QualificationBackendPack,
    resolved: &crate::ResolvedCatalogBackendPull,
) -> Result<(), BackendPluginActivationError> {
    let valid = record.vendor == resolved.vendor
        && record.version == resolved.version
        && record.host_abi_fingerprint == resolved.host_abi.fingerprint
        && record.artifact_fingerprint == backend_artifact_fingerprint(resolved)
        && record.device_target.len() > 1
        && catalog_backend_accepts_device_target(resolved, &record.device_target);
    if valid {
        Ok(())
    } else {
        Err(BackendPluginActivationError::ActivationState(
            "qualification selector does not match the exact signed backend entry".to_string(),
        ))
    }
}

fn load_exact_backend_plugin(
    backend_id: &str,
    vendor: CatalogBackendVendor,
    plugin_path: &Path,
    dependency_dirs: &[PathBuf],
    device_target: &str,
    minimum_driver: Option<&str>,
) -> Result<(), BackendPluginActivationError> {
    let minimum_driver = live_backend_driver_floor(vendor, minimum_driver);
    let path = path_to_utf8_cstring(backend_id, plugin_path)?;
    let abi = CString::new(BackendHostAbi::current().fingerprint)
        .expect("backend ABI fingerprint is hexadecimal");
    let provider_c = CString::new(backend_provider_label(vendor, backend_id)?)
        .expect("provider is static ASCII");
    let target_c = CString::new(device_target).map_err(|_| {
        BackendPluginActivationError::ActivationState(
            "backend device target contains an interior NUL".to_string(),
        )
    })?;
    let minimum_driver_c = CString::new(minimum_driver.unwrap_or("")).map_err(|_| {
        BackendPluginActivationError::ActivationState(
            "backend minimum driver contains an interior NUL".to_string(),
        )
    })?;
    let (dependency_dir_cstrings, dependency_dir_ptrs) =
        dependency_dirs_to_ffi(backend_id, dependency_dirs)?;
    let dependency_dir_ptr = if dependency_dir_ptrs.is_empty() {
        std::ptr::null()
    } else {
        dependency_dir_ptrs.as_ptr()
    };
    let reg = unsafe {
        ffi::ggml_backend_load_verified_v3_utf8(
            path.as_ptr(),
            dependency_dir_ptr,
            dependency_dir_cstrings.len(),
            abi.as_ptr(),
            provider_c.as_ptr(),
            target_c.as_ptr(),
            minimum_driver_c.as_ptr(),
        )
    };
    let Some(reg) = NonNull::new(reg) else {
        return Err(BackendPluginActivationError::LoadFailed {
            backend_id: backend_id.to_string(),
        });
    };
    let name = unsafe { ffi::ggml_backend_reg_name(reg.as_ptr()) };
    let actual = unsafe { cstr_lossy(name) };
    let actual = if actual.is_empty() {
        "<unknown>".to_string()
    } else {
        actual
    };
    let (expected, matches) = match vendor {
        CatalogBackendVendor::Cpu => ("cpu", actual.to_ascii_lowercase().contains("cpu")),
        CatalogBackendVendor::Cuda => ("cuda", actual.to_ascii_lowercase().contains("cuda")),
        CatalogBackendVendor::Hip => {
            let lower = actual.to_ascii_lowercase();
            ("hip", lower.contains("hip") || lower.contains("rocm"))
        }
        CatalogBackendVendor::Vulkan => ("vulkan", actual.to_ascii_lowercase().contains("vulkan")),
        CatalogBackendVendor::Unknown => ("known backend", false),
    };
    if !matches {
        unsafe { ffi::ggml_backend_unload(reg.as_ptr()) };
        return Err(BackendPluginActivationError::ProviderMismatch {
            backend_id: backend_id.to_string(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn backend_provider_label(
    vendor: CatalogBackendVendor,
    backend_id: &str,
) -> Result<&'static str, BackendPluginActivationError> {
    match vendor {
        CatalogBackendVendor::Cpu => Ok("cpu"),
        CatalogBackendVendor::Cuda => Ok("cuda"),
        CatalogBackendVendor::Hip => Ok("hip"),
        CatalogBackendVendor::Vulkan => Ok("vulkan"),
        CatalogBackendVendor::Unknown => Err(BackendPluginActivationError::ProviderMismatch {
            backend_id: backend_id.to_string(),
            expected: "known backend",
            actual: "unknown catalog vendor".to_string(),
        }),
    }
}

pub(crate) fn probe_exact_backend_plugin_candidate(
    backend_id: &str,
    vendor: CatalogBackendVendor,
    plugin_path: &Path,
    dependency_dirs: &[PathBuf],
    device_target: &str,
    minimum_driver: Option<&str>,
) -> Result<String, BackendPluginActivationError> {
    let minimum_driver = live_backend_driver_floor(vendor, minimum_driver);
    let path = path_to_utf8_cstring(backend_id, plugin_path)?;
    let abi = CString::new(BackendHostAbi::current().fingerprint)
        .expect("backend ABI fingerprint is hexadecimal");
    let provider = CString::new(backend_provider_label(vendor, backend_id)?)
        .expect("provider is static ASCII");
    let target = CString::new(device_target).map_err(|_| {
        BackendPluginActivationError::ActivationState(
            "backend device target contains an interior NUL".to_string(),
        )
    })?;
    let minimum_driver = CString::new(minimum_driver.unwrap_or("")).map_err(|_| {
        BackendPluginActivationError::ActivationState(
            "backend minimum driver contains an interior NUL".to_string(),
        )
    })?;
    let (dependency_dir_cstrings, dependency_dir_ptrs) =
        dependency_dirs_to_ffi(backend_id, dependency_dirs)?;
    let dependency_dir_ptr = if dependency_dir_ptrs.is_empty() {
        std::ptr::null()
    } else {
        dependency_dir_ptrs.as_ptr()
    };
    let mut driver = [0 as std::ffi::c_char; 64];
    let ok = unsafe {
        ffi::ggml_backend_probe_verified_v3_utf8(
            path.as_ptr(),
            dependency_dir_ptr,
            dependency_dir_cstrings.len(),
            abi.as_ptr(),
            provider.as_ptr(),
            target.as_ptr(),
            minimum_driver.as_ptr(),
            driver.as_mut_ptr(),
            driver.len(),
        )
    };
    if !ok {
        return Err(BackendPluginActivationError::LiveProbeFailed {
            backend_id: backend_id.to_string(),
        });
    }
    let driver = unsafe { CStr::from_ptr(driver.as_ptr()) }
        .to_str()
        .map_err(|_| BackendPluginActivationError::LiveProbeFailed {
            backend_id: backend_id.to_string(),
        })?
        .to_string();
    if driver.is_empty() {
        return Err(BackendPluginActivationError::LiveProbeFailed {
            backend_id: backend_id.to_string(),
        });
    }
    Ok(driver)
}

fn dependency_dirs_to_ffi(
    backend_id: &str,
    dependency_dirs: &[PathBuf],
) -> Result<(Vec<CString>, Vec<*const std::ffi::c_char>), BackendPluginActivationError> {
    let mut cstrings = Vec::with_capacity(dependency_dirs.len());
    for dependency_dir in dependency_dirs {
        if !dependency_dir.is_absolute() {
            return Err(BackendPluginActivationError::InstalledPackInvalid {
                backend_id: backend_id.to_string(),
                reason: "backend dependency directory is not absolute".to_string(),
            });
        }
        cstrings.push(path_to_utf8_cstring(backend_id, dependency_dir)?);
    }
    let pointers = cstrings.iter().map(|value| value.as_ptr()).collect();
    Ok((cstrings, pointers))
}

impl GgmlBackend {
    pub fn cpu() -> Result<Self, GgmlRuntimeError> {
        bundled_cpu_activation_cell()
            .get_or_init(load_bundled_cpu_module)
            .clone()?;
        // Go through the registry (not ggml_backend_cpu_init): under
        // GGML_BACKEND_DL that symbol lives in the loaded ggml-cpu plugin and is
        // not linked into the host. init_by_type works for static builds too.
        let raw = unsafe {
            ffi::ggml_backend_init_by_type(ffi::GGML_BACKEND_DEVICE_TYPE_CPU, std::ptr::null())
        };
        Self::from_raw(raw, "cpu")
    }

    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, GgmlRuntimeError> {
        let raw = unsafe { ffi::ggml_backend_metal_init() };
        Self::from_raw(raw, "metal")
    }

    pub fn best() -> Result<Self, GgmlRuntimeError> {
        backend_plugin_activation_status()?;
        let devices = ggml_available_devices();
        preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .or_else(|| {
                devices
                    .iter()
                    .find(|device| device.kind == GgmlBackendKind::Cpu)
            })
            .ok_or(GgmlRuntimeError::BackendUnavailable("best"))?
            .initialize()
    }

    pub fn name(&self) -> String {
        unsafe { cstr_lossy(ffi::ggml_backend_name(self.raw.as_ptr())) }
    }

    pub(crate) fn as_ptr(&self) -> ffi::GgmlBackendRaw {
        self.raw.as_ptr()
    }

    pub(crate) fn into_raw(self) -> NonNull<c_void> {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }

    fn from_raw(raw: ffi::GgmlBackendRaw, name: &'static str) -> Result<Self, GgmlRuntimeError> {
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or(GgmlRuntimeError::BackendUnavailable(name))
    }
}

impl Drop for GgmlBackend {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_free(self.raw.as_ptr()) };
    }
}

pub fn ggml_runtime_info() -> GgmlRuntimeInfo {
    let devices = ggml_available_devices();
    let cpu_backend_name = GgmlBackend::cpu()
        .map(|backend| backend.name())
        .unwrap_or_else(|_| "unavailable".to_string());
    let best_backend_name = best_device_name(&devices).or_else(|| {
        (!cpu_backend_name.is_empty() && cpu_backend_name != "unavailable")
            .then(|| cpu_backend_name.clone())
    });
    let metal_backend_name = metal_device_name(&devices);

    GgmlRuntimeInfo {
        cpu_backend_name,
        best_backend_name,
        metal_backend_name,
        devices,
        cpu_features: GgmlCpuFeatures::detect(),
    }
}

/// One-line, no-user-data summary of the detected ggml backend(s)/device(s)
/// for daemon boot logs (see `server_boot` stage in
/// `crates/openasr-server/src/lib.rs`). Only backend/device metadata -- name,
/// kind, memory, buffer alignment -- never model, audio, or file-path data,
/// matching the no-telemetry / local-log-only posture (written to stderr,
/// never transmitted). Added after issues #153/#154/#155 (Vulkan crash
/// reports) where the reporter's misfiled "backend used" field left the
/// actually-selected backend and device unknown; this makes it recoverable
/// from `daemon.log` alone.
///
/// When more than one accelerated-kind device is present (an Optimus/hybrid-
/// graphics host exposing both an integrated and a discrete GPU through the
/// same Vulkan backend), the summary also appends a `gpu_selection=` field
/// naming the device [`preferred_accelerated_device`] picked and the `rule`
/// that decided it: `discrete_over_integrated` (kind ranking alone decided),
/// `discrete_low_vram_skipped` (a better-ranked device was skipped because it
/// reported too little free VRAM to load a model), or
/// `all_low_vram_kind_fallback` (no device reported viable free VRAM, so kind
/// ranking alone decided and ggml's own OOM fallback stays in charge) -- so a
/// "wrong GPU used" report is diagnosable from the boot log alone (see the
/// discrete-GPU preference fix that added this).
pub fn ggml_runtime_boot_summary(info: &GgmlRuntimeInfo) -> String {
    let best = info.best_backend_name.as_deref().unwrap_or("unavailable");
    let mut summary = format!("best_backend={best} cpu_backend={}", info.cpu_backend_name);
    if info.devices.is_empty() {
        summary.push_str(" devices=none");
        return summary;
    }
    let devices = info
        .devices
        .iter()
        .map(|device| {
            let mem = device
                .memory
                .map(|memory| format!(" mem_total_mib={}", memory.total_bytes / (1024 * 1024)))
                .unwrap_or_default();
            let alignment = device
                .buffer_alignment
                .map(|alignment| format!(" alignment_bytes={alignment}"))
                .unwrap_or_default();
            format!(
                "{{name={:?} kind={:?}{mem}{alignment}}}",
                device.name, device.kind
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    summary.push_str(" devices=[");
    summary.push_str(&devices);
    summary.push(']');

    let accelerated_kinds = info
        .devices
        .iter()
        .filter(|device| device.kind.is_gpu())
        .count();
    if accelerated_kinds > 1
        && let Some((preferred, rule)) =
            select_accelerated_device(&info.devices, GgmlBackendKind::is_gpu)
    {
        summary.push_str(&format!(
            " gpu_selection={{picked={:?} kind={:?} rule={} activated_provider={}}}",
            preferred.name,
            preferred.kind,
            rule.boot_log_label(),
            activated_backend_execution_provider()
                .map(ExecutionProvider::as_str)
                .unwrap_or("none")
        ));
    }
    summary
}

pub fn ggml_available_devices() -> Vec<GgmlBackendDevice> {
    ensure_backends_loaded();
    let count = unsafe { ffi::ggml_backend_dev_count() };
    let mut devices = Vec::with_capacity(count);

    for index in 0..count {
        let raw = unsafe { ffi::ggml_backend_dev_get(index) };
        let Some(raw) = NonNull::new(raw) else {
            continue;
        };

        let kind = unsafe { backend_kind(ffi::ggml_backend_dev_type(raw.as_ptr())) };
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;
        unsafe {
            ffi::ggml_backend_dev_memory(raw.as_ptr(), &mut free_bytes, &mut total_bytes);
        }
        let memory = (total_bytes > 0).then_some(GgmlDeviceMemory {
            free_bytes,
            total_bytes,
        });
        // Generic across every backend (CPU/Metal/Vulkan/...): each backend's
        // buffer type reports its own alignment through this one ggml call,
        // so no backend-specific branching is needed here. On Vulkan this is
        // `minStorageBufferOffsetAlignment`.
        let buffer_alignment = unsafe {
            let buft = ffi::ggml_backend_dev_buffer_type(raw.as_ptr());
            (!buft.is_null()).then(|| ffi::ggml_backend_buft_get_alignment(buft))
        };

        let device_id = {
            let mut props = ffi::GgmlBackendDevProps {
                name: ptr::null(),
                description: ptr::null(),
                memory_free: 0,
                memory_total: 0,
                type_: 0,
                device_id: ptr::null(),
                caps: ffi::GgmlBackendDevCaps::default(),
            };
            unsafe { ffi::ggml_backend_dev_get_props(raw.as_ptr(), &mut props) };
            let raw_id = unsafe { cstr_lossy(props.device_id) };
            let trimmed = raw_id.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        };
        let pci_vendor_id = unsafe { device_pci_vendor_id(raw) };

        devices.push(GgmlBackendDevice {
            raw,
            name: unsafe { cstr_lossy(ffi::ggml_backend_dev_name(raw.as_ptr())) },
            description: unsafe { cstr_lossy(ffi::ggml_backend_dev_description(raw.as_ptr())) },
            kind,
            memory,
            buffer_alignment,
            device_id,
            pci_vendor_id,
        });
    }

    devices
}

/// Optional backend hardware fact queried through ggml's shared no-throw
/// adapter. Older plugins remain ABI-safe and report zero/unknown.
unsafe fn device_pci_vendor_id(device: NonNull<c_void>) -> Option<u32> {
    let vendor_id = unsafe { ffi::ggml_backend_dev_pci_vendor_id(device.as_ptr()) };
    (vendor_id != 0).then_some(vendor_id)
}

impl GgmlCpuFeatures {
    pub fn detect() -> Self {
        // Detect via the Rust stdlib, not ggml_cpu_has_*: under GGML_BACKEND_DL
        // those symbols live in the loaded ggml-cpu plugin and are not linked
        // into the host. This is build-mode-agnostic. Fields not detectable on
        // the current architecture stay at their Default (false / 0); `llamafile`
        // is a ggml build option (not a CPU feature) and is reported false here.
        #[allow(unused_mut)]
        let mut features = Self::default();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            features.sse3 = is_x86_feature_detected!("sse3");
            features.ssse3 = is_x86_feature_detected!("ssse3");
            features.avx = is_x86_feature_detected!("avx");
            features.avx_vnni = is_x86_feature_detected!("avxvnni");
            features.avx2 = is_x86_feature_detected!("avx2");
            features.bmi2 = is_x86_feature_detected!("bmi2");
            features.f16c = is_x86_feature_detected!("f16c");
            features.fma = is_x86_feature_detected!("fma");
            features.avx512 = is_x86_feature_detected!("avx512f");
            features.avx512_vbmi = is_x86_feature_detected!("avx512vbmi");
            features.avx512_vnni = is_x86_feature_detected!("avx512vnni");
            features.avx512_bf16 = is_x86_feature_detected!("avx512bf16");
            // amx_int8 stays Default(false): is_x86_feature_detected!("amx-int8")
            // requires the unstable `x86_amx_intrinsics` feature, and AMX is a
            // server-only ISA irrelevant to the consumer/desktop targets.
        }
        #[cfg(target_arch = "aarch64")]
        {
            features.neon = std::arch::is_aarch64_feature_detected!("neon");
            features.arm_fma = features.neon; // aarch64 NEON implies FMA
            features.fp16_va = std::arch::is_aarch64_feature_detected!("fp16");
            features.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            features.matmul_int8 = std::arch::is_aarch64_feature_detected!("i8mm");
            features.sve = std::arch::is_aarch64_feature_detected!("sve");
            // sme stays Default(false): is_aarch64_feature_detected!("sme")
            // requires the unstable `stdarch_aarch64_feature_detection` feature
            // (rust-lang/rust#127764) and does not compile on the pinned stable
            // toolchain — same hazard the amx_int8 path above avoids. Leaving it
            // false is lossless: `sme` is purely diagnostic (doctor CPU report).
        }
        features
    }
}

pub fn ggml_native_build_enabled() -> bool {
    option_env!("OPENASR_GGML_NATIVE_ENABLED") == Some("1")
}

/// Whether this build compiled ggml with `GGML_BACKEND_DL` (build.rs
/// `use_backend_dl`): the CPU/GPU compute backends are runtime-loaded plugin
/// DLLs rather than statically linked. See [`ensure_backends_loaded`] for why
/// this gates the `ggml_backend_load_all` directory scan.
fn ggml_backend_dl_build_enabled() -> bool {
    option_env!("OPENASR_GGML_BACKEND_DL_ENABLED") == Some("1")
}

pub fn ggml_hip_tuning_summary() -> Option<&'static str> {
    match option_env!("OPENASR_HIP_TUNING") {
        Some("disabled") | None => None,
        Some(summary) => Some(summary),
    }
}

fn best_device_name(devices: &[GgmlBackendDevice]) -> Option<String> {
    preferred_accelerated_device(devices, GgmlBackendKind::is_gpu)
        .or_else(|| {
            devices
                .iter()
                .find(|device| device.kind == GgmlBackendKind::Cpu)
        })
        .map(|device| device.name.clone())
}

#[cfg(target_os = "macos")]
fn metal_device_name(devices: &[GgmlBackendDevice]) -> Option<String> {
    preferred_accelerated_device(devices, GgmlBackendKind::is_gpu).map(|device| device.name.clone())
}

#[cfg(not(target_os = "macos"))]
fn metal_device_name(_devices: &[GgmlBackendDevice]) -> Option<String> {
    None
}

fn backend_kind(kind: c_int) -> GgmlBackendKind {
    match kind {
        ffi::GGML_BACKEND_DEVICE_TYPE_CPU => GgmlBackendKind::Cpu,
        ffi::GGML_BACKEND_DEVICE_TYPE_GPU => GgmlBackendKind::Gpu,
        ffi::GGML_BACKEND_DEVICE_TYPE_IGPU => GgmlBackendKind::IntegratedGpu,
        ffi::GGML_BACKEND_DEVICE_TYPE_ACCEL => GgmlBackendKind::Accelerator,
        ffi::GGML_BACKEND_DEVICE_TYPE_META => GgmlBackendKind::Meta,
        unknown => GgmlBackendKind::Unknown(unknown),
    }
}

unsafe fn cstr_lossy(value: *const c_char) -> String {
    if value.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_backend_directory_prefers_exact_abi_scoped_bundle() {
        let temp = tempfile::tempdir().expect("temporary executable directory");
        let abi = BackendHostAbi::current();

        assert_eq!(
            bundled_backend_directory_for_host(temp.path(), &abi),
            temp.path()
        );

        let exact = temp
            .path()
            .join("openasr-backend-bundles")
            .join(&abi.fingerprint);
        std::fs::create_dir_all(&exact).expect("create exact ABI bundle directory");
        std::fs::write(exact.join("openasr-backend-bundle-v1.json"), b"{}")
            .expect("write exact ABI bundle marker");

        assert_eq!(bundled_backend_directory_for_host(temp.path(), &abi), exact);
    }

    fn bundled_file(filename: &str, provider: &str) -> BundledBackendManifestFile {
        BundledBackendManifestFile {
            filename: filename.to_string(),
            provider: provider.to_string(),
            sha256: "a".repeat(64),
            size_bytes: 1,
            image_sha256: "c".repeat(64),
            image_size_bytes: 1,
        }
    }

    fn valid_bundled_manifest() -> BundledBackendManifest {
        BundledBackendManifest {
            schema_version: 4,
            host_abi_fingerprint: "b".repeat(64),
            bundle_contract_sha256: "d".repeat(64),
            cpu_contract_sha256: "e".repeat(64),
            files: vec![
                bundled_file("ggml.dll", "host"),
                bundled_file("ggml-base.dll", "host"),
                bundled_file("ggml-cpu-avx2.dll", "cpu"),
            ],
        }
    }

    #[test]
    fn bundled_manifest_contract_is_strict_and_case_insensitive() {
        let valid = valid_bundled_manifest();
        assert_eq!(validate_bundled_backend_manifest(&valid), Ok(()));

        let mut duplicate = valid_bundled_manifest();
        duplicate.files.push(bundled_file("GGML.DLL", "host"));
        assert!(validate_bundled_backend_manifest(&duplicate).is_err());

        let mut wrong_role = valid_bundled_manifest();
        wrong_role.files[2].provider = "vulkan".to_string();
        assert!(validate_bundled_backend_manifest(&wrong_role).is_err());

        let mut leaked_vulkan = valid_bundled_manifest();
        leaked_vulkan
            .files
            .push(bundled_file("ggml-vulkan.dll", "vulkan"));
        assert!(validate_bundled_backend_manifest(&leaked_vulkan).is_err());

        let mut malformed_hash = valid_bundled_manifest();
        malformed_hash.files[0].sha256 = "A".repeat(64);
        assert!(validate_bundled_backend_manifest(&malformed_hash).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn bundled_cpu_candidate_loading_skips_a_broken_candidate() {
        let Some(discovered) =
            discover_bundled_cpu_modules().expect("discover bundled CPU candidates")
        else {
            return;
        };
        let temp = tempfile::tempdir().expect("temporary missing candidate directory");
        let missing = path_to_utf8_cstring(
            "missing-cpu-candidate",
            &temp.path().join("ggml-cpu-broken.dll"),
        )
        .unwrap();
        let mut candidates = vec![missing];
        candidates.extend(discovered.paths);
        load_best_bundled_backend(&candidates, "cpu", &discovered.host_abi_fingerprint)
            .expect("a later verified CPU candidate must survive one broken candidate");
    }

    #[test]
    fn cpu_backend_initializes() {
        let backend = GgmlBackend::cpu().expect("cpu backend");
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn boot_summary_includes_device_name_kind_and_alignment_when_present() {
        let info = GgmlRuntimeInfo {
            cpu_backend_name: "CPU".to_string(),
            best_backend_name: Some("Vulkan0".to_string()),
            metal_backend_name: None,
            devices: vec![GgmlBackendDevice {
                raw: NonNull::dangling(),
                name: "NVIDIA GeForce RTX 4070".to_string(),
                description: "NVIDIA GeForce RTX 4070 (Vulkan)".to_string(),
                kind: GgmlBackendKind::Gpu,
                memory: Some(GgmlDeviceMemory {
                    free_bytes: 8 * 1024 * 1024 * 1024,
                    total_bytes: 12 * 1024 * 1024 * 1024,
                }),
                buffer_alignment: Some(16),
                device_id: Some("0000:01:00.0".to_string()),
                pci_vendor_id: Some(0x10de),
            }],
            cpu_features: GgmlCpuFeatures::default(),
        };

        let summary = ggml_runtime_boot_summary(&info);

        assert!(summary.contains("best_backend=Vulkan0"));
        assert!(summary.contains("cpu_backend=CPU"));
        assert!(summary.contains("NVIDIA GeForce RTX 4070"));
        assert!(summary.contains("kind=Gpu"));
        assert!(summary.contains("mem_total_mib=12288"));
        assert!(summary.contains("alignment_bytes=16"));
    }

    #[test]
    fn boot_summary_reports_no_devices_without_panicking() {
        let info = GgmlRuntimeInfo {
            cpu_backend_name: "CPU".to_string(),
            best_backend_name: None,
            metal_backend_name: None,
            devices: vec![],
            cpu_features: GgmlCpuFeatures::default(),
        };

        let summary = ggml_runtime_boot_summary(&info);

        assert_eq!(
            summary,
            "best_backend=unavailable cpu_backend=CPU devices=none"
        );
    }

    fn test_device(name: &str, kind: GgmlBackendKind) -> GgmlBackendDevice {
        GgmlBackendDevice::for_test(name, name, kind, None)
    }

    fn test_device_with_memory(
        name: &str,
        kind: GgmlBackendKind,
        memory: Option<GgmlDeviceMemory>,
    ) -> GgmlBackendDevice {
        GgmlBackendDevice::for_test(name, name, kind, memory)
    }

    fn memory_mib(free_mib: usize, total_mib: usize) -> GgmlDeviceMemory {
        GgmlDeviceMemory {
            free_bytes: free_mib * 1024 * 1024,
            total_bytes: total_mib * 1024 * 1024,
        }
    }

    #[test]
    fn preferred_accelerated_device_picks_discrete_over_integrated() {
        // The Optimus/hybrid-graphics case (issue: "double GPU picks the
        // wrong one"): an Intel integrated GPU enumerated ahead of the
        // NVIDIA discrete GPU must still yield the discrete GPU when it
        // reports enough free VRAM to load a model.
        let devices = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(16 * 1024, 16 * 1024)),
            ),
            test_device_with_memory(
                "Vulkan1",
                GgmlBackendKind::Gpu,
                Some(memory_mib(8 * 1024, 12 * 1024)),
            ),
        ];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("a device is picked");
        assert_eq!(picked.name, "Vulkan1");
        assert_eq!(picked.kind, GgmlBackendKind::Gpu);
    }

    #[test]
    fn preferred_accelerated_device_skips_discrete_with_low_free_vram() {
        // The failure this guards: on a hybrid-graphics laptop whose discrete
        // GPU's VRAM is consumed by other workloads (browser, game), a ~1 GB
        // weight allocation on the discrete GPU fails with
        // VK_ERROR_OUT_OF_DEVICE_MEMORY and the run falls back to CPU -- even
        // though the integrated GPU (shared system RAM) has ample space. The
        // low-VRAM discrete device must be skipped, not merely ranked first.
        let devices = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(16 * 1024, 16 * 1024)),
            ),
            test_device_with_memory(
                "Vulkan1",
                GgmlBackendKind::Gpu,
                Some(memory_mib(128, 8 * 1024)),
            ),
        ];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("a device is picked");
        assert_eq!(picked.name, "Vulkan0");
        assert_eq!(picked.kind, GgmlBackendKind::IntegratedGpu);
    }

    #[test]
    fn preferred_accelerated_device_falls_back_to_kind_ranking_when_every_device_is_low() {
        // No device can guarantee the minimum free memory: keep the historical
        // kind-only pick (discrete first) so ggml still attempts it and drives
        // its own OOM fallback, rather than the selector diverting to an
        // equally-loaded integrated GPU.
        let devices = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(64, 16 * 1024)),
            ),
            test_device_with_memory(
                "Vulkan1",
                GgmlBackendKind::Gpu,
                Some(memory_mib(128, 8 * 1024)),
            ),
        ];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("a device is picked");
        assert_eq!(picked.name, "Vulkan1");
        assert_eq!(picked.kind, GgmlBackendKind::Gpu);
    }

    #[test]
    fn preferred_accelerated_device_does_not_penalize_missing_memory_reports() {
        // A backend that does not surface memory info (memory: None) must not
        // be treated as low-VRAM and skipped: the discrete GPU with no report
        // still beats an integrated GPU that reports plenty.
        let devices = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(16 * 1024, 16 * 1024)),
            ),
            test_device("Vulkan1", GgmlBackendKind::Gpu), // memory: None
        ];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("a device is picked");
        assert_eq!(picked.name, "Vulkan1");
        assert_eq!(picked.kind, GgmlBackendKind::Gpu);
    }

    #[test]
    fn select_accelerated_device_reports_why_the_pick_won() {
        let kind_only = vec![
            test_device("Vulkan0", GgmlBackendKind::IntegratedGpu),
            test_device("Vulkan1", GgmlBackendKind::Gpu),
        ];
        assert_eq!(
            select_accelerated_device(&kind_only, GgmlBackendKind::is_gpu),
            Some((&kind_only[1], AcceleratedDeviceSelectionRule::KindRanking))
        );

        let skipped = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(16 * 1024, 16 * 1024)),
            ),
            test_device_with_memory(
                "Vulkan1",
                GgmlBackendKind::Gpu,
                Some(memory_mib(128, 8 * 1024)),
            ),
        ];
        assert_eq!(
            select_accelerated_device(&skipped, GgmlBackendKind::is_gpu),
            Some((&skipped[0], AcceleratedDeviceSelectionRule::LowVramSkipped))
        );

        let all_low = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::IntegratedGpu,
                Some(memory_mib(64, 16 * 1024)),
            ),
            test_device_with_memory(
                "Vulkan1",
                GgmlBackendKind::Gpu,
                Some(memory_mib(128, 8 * 1024)),
            ),
        ];
        assert_eq!(
            select_accelerated_device(&all_low, GgmlBackendKind::is_gpu),
            Some((&all_low[1], AcceleratedDeviceSelectionRule::LowVramFallback))
        );

        assert_eq!(
            select_accelerated_device(&[], GgmlBackendKind::is_gpu),
            None
        );
    }

    #[test]
    fn accelerated_device_selection_rule_boot_log_labels_are_stable() {
        // The labels are a boot-log contract ("wrong GPU used" reports are
        // triaged off daemon.log); pin them so a rename is a deliberate diff.
        assert_eq!(
            AcceleratedDeviceSelectionRule::KindRanking.boot_log_label(),
            "discrete_over_integrated"
        );
        assert_eq!(
            AcceleratedDeviceSelectionRule::LowVramSkipped.boot_log_label(),
            "discrete_low_vram_skipped"
        );
        assert_eq!(
            AcceleratedDeviceSelectionRule::LowVramFallback.boot_log_label(),
            "all_low_vram_kind_fallback"
        );
    }

    #[test]
    fn preferred_accelerated_device_falls_back_to_integrated_when_no_discrete_present() {
        let devices = vec![test_device("Vulkan0", GgmlBackendKind::IntegratedGpu)];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("integrated GPU is picked");
        assert_eq!(picked.name, "Vulkan0");
    }

    #[test]
    fn preferred_accelerated_device_picks_discrete_when_only_discrete_present() {
        let devices = vec![test_device("Vulkan0", GgmlBackendKind::Gpu)];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("discrete GPU is picked");
        assert_eq!(picked.name, "Vulkan0");
    }

    #[test]
    fn preferred_accelerated_device_keeps_enumeration_order_between_two_discrete_gpus() {
        // No further signal to break the tie between two discrete GPUs; the
        // first one the registry enumerated is kept (matches
        // `Iterator::min_by_key`'s documented first-minimum behavior).
        let devices = vec![
            test_device("Vulkan0", GgmlBackendKind::Gpu),
            test_device("Vulkan1", GgmlBackendKind::Gpu),
        ];
        let picked = preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu)
            .expect("a discrete GPU is picked");
        assert_eq!(picked.name, "Vulkan0");
    }

    #[test]
    fn activated_provider_precedes_earlier_unbound_vulkan_device() {
        let devices = vec![
            test_device("Vulkan0", GgmlBackendKind::Gpu),
            test_device("CUDA0", GgmlBackendKind::Gpu),
        ];
        let picked = select_accelerated_device_for_provider(
            &devices,
            GgmlBackendKind::is_gpu,
            Some(ExecutionProvider::Cuda),
            true,
        )
        .expect("activated CUDA device is picked")
        .0;
        assert_eq!(picked.name, "CUDA0");
    }

    #[test]
    fn activated_provider_choice_is_not_overridden_by_other_provider_free_vram() {
        let devices = vec![
            test_device_with_memory(
                "Vulkan0",
                GgmlBackendKind::Gpu,
                Some(memory_mib(8 * 1024, 12 * 1024)),
            ),
            test_device_with_memory(
                "ROCm0",
                GgmlBackendKind::Gpu,
                Some(memory_mib(128, 12 * 1024)),
            ),
        ];
        let picked = select_accelerated_device_for_provider(
            &devices,
            GgmlBackendKind::is_gpu,
            Some(ExecutionProvider::Hip),
            true,
        )
        .expect("activated HIP device is picked")
        .0;
        assert_eq!(picked.name, "ROCm0");
    }

    #[test]
    fn missing_activated_provider_fails_closed_instead_of_borrowing_vulkan() {
        let devices = vec![test_device("Vulkan0", GgmlBackendKind::Gpu)];
        assert!(
            select_accelerated_device_for_provider(
                &devices,
                GgmlBackendKind::is_gpu,
                Some(ExecutionProvider::Cuda),
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn windows_policy_rejects_unbound_vulkan_without_catalog_activation() {
        let devices = vec![test_device("Vulkan0", GgmlBackendKind::Gpu)];
        assert!(
            select_accelerated_device_for_provider(&devices, GgmlBackendKind::is_gpu, None, true,)
                .is_none()
        );
    }

    #[test]
    fn preferred_accelerated_device_none_when_no_accelerated_device_present() {
        let devices = vec![test_device("CPU", GgmlBackendKind::Cpu)];
        assert!(preferred_accelerated_device(&devices, GgmlBackendKind::is_gpu).is_none());
    }

    #[test]
    fn best_device_name_prefers_discrete_gpu_on_hybrid_graphics_host() {
        let devices = vec![
            test_device("Vulkan0", GgmlBackendKind::IntegratedGpu),
            test_device("Vulkan1", GgmlBackendKind::Gpu),
        ];
        assert_eq!(best_device_name(&devices).as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn boot_summary_reports_gpu_selection_only_with_multiple_accelerated_devices() {
        let single = GgmlRuntimeInfo {
            cpu_backend_name: "CPU".to_string(),
            best_backend_name: Some("Vulkan0".to_string()),
            metal_backend_name: None,
            devices: vec![test_device("Vulkan0", GgmlBackendKind::Gpu)],
            cpu_features: GgmlCpuFeatures::default(),
        };
        assert!(!ggml_runtime_boot_summary(&single).contains("gpu_selection="));

        let hybrid = GgmlRuntimeInfo {
            cpu_backend_name: "CPU".to_string(),
            best_backend_name: Some("Vulkan1".to_string()),
            metal_backend_name: None,
            devices: vec![
                test_device("Vulkan0", GgmlBackendKind::IntegratedGpu),
                test_device("Vulkan1", GgmlBackendKind::Gpu),
            ],
            cpu_features: GgmlCpuFeatures::default(),
        };
        let summary = ggml_runtime_boot_summary(&hybrid);
        assert!(summary.contains("gpu_selection={picked=\"Vulkan1\" kind=Gpu"));
        assert!(summary.contains("rule=discrete_over_integrated"));
    }

    #[test]
    fn boot_summary_reports_discrete_low_vram_skipped_rule() {
        let hybrid = GgmlRuntimeInfo {
            cpu_backend_name: "CPU".to_string(),
            best_backend_name: Some("Vulkan0".to_string()),
            metal_backend_name: None,
            devices: vec![
                test_device_with_memory(
                    "Vulkan0",
                    GgmlBackendKind::IntegratedGpu,
                    Some(memory_mib(16 * 1024, 16 * 1024)),
                ),
                test_device_with_memory(
                    "Vulkan1",
                    GgmlBackendKind::Gpu,
                    Some(memory_mib(128, 8 * 1024)),
                ),
            ],
            cpu_features: GgmlCpuFeatures::default(),
        };
        let summary = ggml_runtime_boot_summary(&hybrid);
        assert!(summary.contains("gpu_selection={picked=\"Vulkan0\" kind=IntegratedGpu"));
        assert!(summary.contains("rule=discrete_low_vram_skipped"));
    }

    #[test]
    fn cpu_device_supports_core_matmul_types() {
        // The CPU device is always present and runs every ggml type, so this is
        // the always-on CI coverage of the weight-buft probe (S3).
        let devices = ggml_available_devices();
        let Some(cpu) = devices
            .iter()
            .find(|device| device.kind == GgmlBackendKind::Cpu)
        else {
            return;
        };
        assert!(cpu.supports_matmul_for_type(ffi::GGML_TYPE_F32));
        assert!(cpu.supports_matmul_for_type(ffi::GGML_TYPE_F16));
        assert!(cpu.supports_matmul_for_type(ffi::GGML_TYPE_Q8_0));
        assert!(
            cpu.supported_matmul_weight_types()
                .iter()
                .all(|(_, supported)| *supported)
        );
        let names = cpu
            .supported_matmul_weight_types()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"q5_k"));
    }

    #[test]
    fn cpu_device_supports_argmax_first() {
        let devices = ggml_available_devices();
        let Some(cpu) = devices
            .iter()
            .find(|device| device.kind == GgmlBackendKind::Cpu)
        else {
            return;
        };
        assert!(
            cpu.supports_argmax_first(),
            "CPU must declare GGML_OP_ARGMAX_FIRST through supports_op"
        );
    }

    #[test]
    fn cpu_device_supports_swoosh() {
        let devices = ggml_available_devices();
        let Some(cpu) = devices
            .iter()
            .find(|device| device.kind == GgmlBackendKind::Cpu)
        else {
            return;
        };
        assert!(
            cpu.supports_swoosh(),
            "CPU must declare GGML_UNARY_OP_SWOOSH through supports_op"
        );
    }

    #[test]
    fn metal_device_does_not_support_argmax_first_when_present() {
        let devices = ggml_available_devices();
        let Some(metal) = devices.iter().find(|device| {
            ExecutionProvider::from_backend_name(&device.name) == ExecutionProvider::Metal
        }) else {
            return;
        };
        assert!(
            !metal.supports_argmax_first(),
            "Metal must not declare GGML_OP_ARGMAX_FIRST"
        );
    }

    #[test]
    fn metal_device_supports_swoosh_when_present() {
        let devices = ggml_available_devices();
        let Some(metal) = devices.iter().find(|device| {
            ExecutionProvider::from_backend_name(&device.name) == ExecutionProvider::Metal
        }) else {
            return;
        };
        assert!(
            metal.supports_swoosh(),
            "Metal must declare GGML_UNARY_OP_SWOOSH through supports_op"
        );
    }

    #[test]
    fn accelerated_device_supports_f32_matmul_when_present() {
        // Runs only when a GPU/accelerator backend is linked + present (e.g.
        // `cargo test --features hip` on a ROCm host); skipped on CPU-only CI.
        let devices = ggml_available_devices();
        let Some(gpu) = devices
            .iter()
            .find(|device| device.kind.is_gpu() && !device.name.trim().is_empty())
        else {
            return;
        };
        // Any GPU backend must be able to run an f32 mul_mat; if even this is
        // unsupported the probe (or the device) is broken.
        assert!(
            gpu.supports_matmul_for_type(ffi::GGML_TYPE_F32),
            "device {} reported no f32 mul_mat support",
            gpu.name
        );
    }

    #[test]
    fn registry_exposes_devices() {
        let devices = ggml_available_devices();
        assert!(
            devices
                .iter()
                .any(|device| device.kind == GgmlBackendKind::Cpu)
        );
    }

    #[test]
    fn runtime_info_reports_cpu_features() {
        let info = ggml_runtime_info();
        assert!(!info.cpu_backend_name.is_empty());
        assert!(!info.devices.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_backend_initializes_when_available() {
        let has_gpu = ggml_available_devices()
            .iter()
            .any(|device| device.kind.is_gpu() && !device.name.trim().is_empty());
        if has_gpu {
            let backend = GgmlBackend::metal().expect("metal backend");
            assert!(!backend.name().is_empty());
        }
    }
}
