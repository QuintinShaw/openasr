use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr::{self, NonNull},
};

use thiserror::Error;

use super::ffi;

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
        Self {
            raw: NonNull::dangling(),
            name: name.to_string(),
            description: description.to_string(),
            kind,
            memory,
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

impl GgmlBackendKind {
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu | Self::IntegratedGpu)
    }
}

/// Register backend plugin DLLs once per process before the first registry
/// query. Under `GGML_BACKEND_DL` the static `GGML_USE_*` backend registration
/// is compiled out, so without this the registry is empty and every
/// init/enumeration call returns nothing.
///
/// In a statically-linked build (macOS, Linux, GPU-feature Windows builds —
/// `ggml_backend_dl_build_enabled() == false`) the compute backend is already
/// registered at static-init time, so the directory scan is NOT run: it is not
/// a harmless no-op. `ggml_backend_load_all` dlopens every `ggml-*.dll`/`.so`
/// sitting next to the exe, and a desktop bundle can legitimately ship the CPU
/// `BACKEND_DL` plugin DLLs next to a statically-linked GPU exe (the shell app
/// needs them for other components). Loading that plugin pulls a second copy
/// of ggml core into the process, and the two copies' global state collide at
/// `ggml.cpp:22 GGML_ASSERT(prev != ggml_uncaught_exception)`, which fastfails
/// the whole process (0xc0000409) rather than returning an error. Skipping the
/// scan for static builds avoids that mixed-build crash while keeping the
/// scan for genuine `BACKEND_DL` builds, which have no static backend and
/// would otherwise register nothing at all.
/// Idempotent and process-wide.
pub(crate) fn ensure_backends_loaded() {
    use std::sync::OnceLock;
    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        if ggml_backend_dl_build_enabled() {
            // Base-installer plugins next to the exe + GGML_BACKEND_PATH (the
            // CPU variants on Windows). Only safe/needed under GGML_BACKEND_DL,
            // where no backend is statically registered — see the doc comment.
            unsafe { ffi::ggml_backend_load_all() };
        }
        // Downloaded GPU packs under OPENASR_HOME/backends/<vendor>/<version>/.
        load_installed_backend_plugins();
    });
}

/// Register every downloaded GPU backend pack under
/// `OPENASR_HOME/backends/<vendor>/<version>/` with the ggml registry. The base
/// installer's CPU variants load from next to the executable (handled by
/// `ggml_backend_load_all`); this adds the packs a user pulled on demand
/// (HIP/Vulkan/CUDA). Best-effort and fail-open: a missing tree, an unreadable
/// entry, or a plugin that fails to load is skipped — the engine always keeps
/// the CPU backend, so a broken pack degrades to CPU rather than failing.
fn load_installed_backend_plugins() {
    let Ok(home) = crate::home::openasr_home() else {
        return;
    };
    for dir in installed_backend_plugin_dirs(&home) {
        let Ok(dir) = std::ffi::CString::new(dir.to_string_lossy().as_bytes()) else {
            continue;
        };
        // On Windows each plugin's satellite runtime DLLs (amdhip64/rocblas/
        // vulkan-1/cudart/...) are staged in this same pack dir, which is not on
        // the default DLL search path. They resolve because the loader
        // (`dl_load_library`) opens each absolute-path plugin with
        // LOAD_WITH_ALTERED_SEARCH_PATH, so the plugin's own directory is searched
        // for its dependencies — see third_party/openasr-ggml/src/ggml-backend-dl.cpp.
        unsafe { ffi::ggml_backend_load_all_from_path(dir.as_ptr()) };
    }
}

/// The `OPENASR_HOME/backends/<vendor>/<version>/` directories that hold
/// downloaded backend packs, sorted for a deterministic registration order.
/// Pure (no FFI, no process-global state) so the discovery is unit-testable;
/// [`load_installed_backend_plugins`] feeds each directory to the registry. A
/// missing or unreadable tree yields an empty list (a CPU-only install).
fn installed_backend_plugin_dirs(home: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    let Ok(vendors) = std::fs::read_dir(home.join("backends")) else {
        return dirs;
    };
    for vendor in vendors.flatten() {
        let Ok(versions) = std::fs::read_dir(vendor.path()) else {
            continue;
        };
        for version in versions.flatten() {
            let path = version.path();
            if path.is_dir() {
                dirs.push(path);
            }
        }
    }
    dirs.sort();
    dirs
}

impl GgmlBackend {
    pub fn cpu() -> Result<Self, GgmlRuntimeError> {
        ensure_backends_loaded();
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
        ensure_backends_loaded();
        let raw = unsafe { ffi::ggml_backend_init_best() };
        Self::from_raw(raw, "best")
    }

    pub fn name(&self) -> String {
        unsafe { cstr_lossy(ffi::ggml_backend_name(self.raw.as_ptr())) }
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

        devices.push(GgmlBackendDevice {
            raw,
            name: unsafe { cstr_lossy(ffi::ggml_backend_dev_name(raw.as_ptr())) },
            description: unsafe { cstr_lossy(ffi::ggml_backend_dev_description(raw.as_ptr())) },
            kind,
            memory,
        });
    }

    devices
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
    devices
        .iter()
        .find(|device| device.kind.is_gpu())
        .or_else(|| {
            devices
                .iter()
                .find(|device| device.kind == GgmlBackendKind::Cpu)
        })
        .map(|device| device.name.clone())
}

#[cfg(target_os = "macos")]
fn metal_device_name(devices: &[GgmlBackendDevice]) -> Option<String> {
    devices
        .iter()
        .find(|device| device.kind.is_gpu())
        .map(|device| device.name.clone())
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
    fn cpu_backend_initializes() {
        let backend = GgmlBackend::cpu().expect("cpu backend");
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn installed_backend_plugin_dirs_finds_vendor_version_dirs_sorted() {
        let home = tempfile::tempdir().unwrap();
        let backends = home.path().join("backends");
        std::fs::create_dir_all(backends.join("vulkan").join("0.13.1")).unwrap();
        std::fs::create_dir_all(backends.join("hip").join("0.13.1")).unwrap();
        // A stray file under a vendor dir must be ignored (only version DIRS load).
        std::fs::write(backends.join("hip").join("README.txt"), b"x").unwrap();

        let dirs = installed_backend_plugin_dirs(home.path());

        assert_eq!(
            dirs,
            vec![
                backends.join("hip").join("0.13.1"),
                backends.join("vulkan").join("0.13.1"),
            ]
        );
    }

    #[test]
    fn installed_backend_plugin_dirs_empty_without_backends_tree() {
        let home = tempfile::tempdir().unwrap();
        assert!(installed_backend_plugin_dirs(home.path()).is_empty());
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
