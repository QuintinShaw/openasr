//! Side-effect-light Windows GPU architecture discovery before a target pack
//! is downloaded.
//!
//! CUDA is queried through the OS-provided driver DLL. HIP uses only the
//! signed, content-addressed ROCr/HIP runtime objects prepared from the public
//! backend catalog. Neither path links a GPU runtime into the neutral host.

use crate::{CatalogBackendVendor, pull::PreparedBackendRuntimeObjects};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendDeviceProof {
    pub target: String,
    pub driver_api_version: String,
}

#[derive(Debug, Error)]
#[cfg_attr(
    not(windows),
    allow(
        dead_code,
        reason = "Windows discovery failure codes remain a stable machine protocol on every host"
    )
)]
pub(crate) enum BackendDeviceProbeError {
    #[error("target-scoped discovery is unsupported for this provider")]
    UnsupportedProvider,
    #[error("the {provider} driver is unavailable: {message}")]
    DriverUnavailable {
        provider: &'static str,
        message: String,
    },
    #[error("the signed provider discovery runtime is unavailable: {0}")]
    DiscoveryRuntimeUnavailable(String),
    #[error("{0} reported no supported GPU")]
    NoDevice(&'static str),
    #[error("the provider reported an invalid target: {0}")]
    UnknownTarget(String),
    #[error("the {provider} driver query failed: {message}")]
    DriverQueryFailed {
        provider: &'static str,
        message: String,
    },
    #[error("target-scoped provider discovery is currently supported only on Windows")]
    #[cfg_attr(windows, allow(dead_code))]
    UnsupportedPlatform,
}

impl BackendDeviceProbeError {
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedProvider => "provider_discovery_unsupported",
            Self::DriverUnavailable { .. } => "driver_unavailable",
            Self::DiscoveryRuntimeUnavailable(_) => "discovery_runtime_unavailable",
            Self::NoDevice(_) => "no_device",
            Self::UnknownTarget(_) => "unknown_target",
            Self::DriverQueryFailed { .. } => "driver_query_failed",
            Self::UnsupportedPlatform => "platform_unsupported",
        }
    }
}

pub(crate) fn probe_provider_device(
    vendor: CatalogBackendVendor,
    runtime: &PreparedBackendRuntimeObjects,
) -> Result<BackendDeviceProof, BackendDeviceProbeError> {
    #[cfg(windows)]
    {
        match vendor {
            CatalogBackendVendor::Cuda => windows::probe_cuda(),
            CatalogBackendVendor::Hip => windows::probe_hip(runtime),
            _ => Err(BackendDeviceProbeError::UnsupportedProvider),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (vendor, runtime);
        Err(BackendDeviceProbeError::UnsupportedPlatform)
    }
}

#[cfg(any(windows, test))]
fn normalize_cuda_driver_version(raw: i32) -> Result<String, String> {
    if raw <= 0 {
        return Err("CUDA driver API returned a non-positive version".to_string());
    }
    Ok(format!("{}.{}.{}", raw / 1000, (raw % 1000) / 10, raw % 10))
}

#[cfg(any(windows, test))]
fn normalize_hip_driver_version(raw: i32) -> Result<String, String> {
    if raw <= 0 {
        return Err("HIP driver API returned a non-positive version".to_string());
    }
    if raw >= 1_000_000 {
        return Ok(format!(
            "{}.{}.{}",
            raw / 10_000_000,
            (raw / 100_000) % 100,
            (raw / 1_000) % 100
        ));
    }
    Ok(raw.to_string())
}

#[cfg(any(windows, test))]
fn canonical_hip_target(name: &str) -> Option<String> {
    let name = name.trim().split(':').next()?.to_ascii_lowercase();
    let digits = name.strip_prefix("gfx")?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| format!("gfx{digits}"))
}

#[cfg(windows)]
mod windows {
    use std::{
        ffi::c_void,
        fs::File,
        io::Read as _,
        os::windows::ffi::OsStrExt,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    use sha2::{Digest as _, Sha256};

    use windows_sys::Win32::{
        Foundation::FreeLibrary,
        System::{
            LibraryLoader::{
                AddDllDirectory, GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
                LOAD_LIBRARY_SEARCH_SYSTEM32, LOAD_LIBRARY_SEARCH_USER_DIRS, LoadLibraryExW,
                RemoveDllDirectory,
            },
            SystemInformation::GetSystemDirectoryW,
        },
    };

    use super::{
        BackendDeviceProbeError, BackendDeviceProof, canonical_hip_target,
        normalize_cuda_driver_version, normalize_hip_driver_version,
    };
    use crate::pull::PreparedBackendRuntimeObjects;

    const CUDA_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    const CUDA_COMPUTE_CAPABILITY_MINOR: i32 = 76;
    const HSA_STATUS_SUCCESS: u32 = 0;
    const HSA_AGENT_INFO_NAME: u32 = 0;
    const HSA_AGENT_INFO_DEVICE: u32 = 17;
    const HSA_DEVICE_TYPE_GPU: u32 = 1;
    static HIP_DISCOVERY_LOAD_LOCK: Mutex<()> = Mutex::new(());

    struct DynamicLibrary(windows_sys::Win32::Foundation::HMODULE);

    struct DllSearchDirectories(Vec<*mut c_void>);

    impl DllSearchDirectories {
        fn add(paths: &[PathBuf]) -> Result<Self, String> {
            let mut directories = Self(Vec::new());
            for path in paths {
                let canonical = path.canonicalize().map_err(|error| {
                    format!(
                        "could not resolve verified provider directory '{}': {error}",
                        path.display()
                    )
                })?;
                let wide = canonical
                    .as_os_str()
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>();
                // SAFETY: `wide` is a live NUL-terminated absolute directory
                // path. The returned cookie is removed by Drop.
                let cookie = unsafe { AddDllDirectory(wide.as_ptr()) };
                if cookie.is_null() {
                    return Err(format!(
                        "could not add verified provider directory '{}': {}",
                        canonical.display(),
                        std::io::Error::last_os_error()
                    ));
                }
                directories.0.push(cookie);
            }
            Ok(directories)
        }
    }

    impl Drop for DllSearchDirectories {
        fn drop(&mut self) {
            for cookie in self.0.drain(..).rev() {
                // SAFETY: every cookie came from AddDllDirectory in Self::add
                // and is removed exactly once.
                unsafe { RemoveDllDirectory(cookie) };
            }
        }
    }

    impl DynamicLibrary {
        fn load(path: &Path, flags: u32) -> Result<Self, String> {
            let mut wide = path
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect::<Vec<_>>();
            // SAFETY: `wide` is a live NUL-terminated UTF-16 path for the
            // duration of the call. The returned handle is owned by Self.
            let handle = unsafe { LoadLibraryExW(wide.as_mut_ptr(), std::ptr::null_mut(), flags) };
            if handle.is_null() {
                return Err(format!(
                    "could not load provider discovery runtime '{}': {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
            Ok(Self(handle))
        }

        fn symbol(&self, name: &'static [u8]) -> Result<*const c_void, String> {
            debug_assert_eq!(name.last(), Some(&0));
            // SAFETY: `self.0` is live and `name` is NUL-terminated.
            let function = unsafe { GetProcAddress(self.0, name.as_ptr()) }
                .ok_or_else(|| format!("provider runtime omitted symbol {}", symbol_name(name)))?;
            Ok(function as *const () as *const c_void)
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            // SAFETY: Self owns this live library handle and drops it once.
            unsafe { FreeLibrary(self.0) };
        }
    }

    fn symbol_name(name: &[u8]) -> String {
        String::from_utf8_lossy(name.strip_suffix(&[0]).unwrap_or(name)).into_owned()
    }

    fn system_library_path(filename: &str) -> Result<PathBuf, String> {
        let mut buffer = vec![0_u16; 32_768];
        // SAFETY: `buffer` is writable for the supplied capacity.
        let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
        if length == 0 || length as usize >= buffer.len() {
            return Err(format!(
                "could not resolve the Windows system directory: {}",
                std::io::Error::last_os_error()
            ));
        }
        buffer.truncate(length as usize);
        Ok(PathBuf::from(
            String::from_utf16(&buffer)
                .map_err(|_| "Windows system directory was not valid UTF-16".to_string())?,
        )
        .join(filename))
    }

    pub(super) fn probe_cuda() -> Result<BackendDeviceProof, BackendDeviceProbeError> {
        type CuInit = unsafe extern "system" fn(u32) -> i32;
        type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> i32;
        type CuDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
        type CuDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
        type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;

        let driver_path = system_library_path("nvcuda.dll").map_err(|message| {
            BackendDeviceProbeError::DriverUnavailable {
                provider: "CUDA",
                message,
            }
        })?;
        let library = DynamicLibrary::load(&driver_path, LOAD_LIBRARY_SEARCH_SYSTEM32).map_err(
            |message| BackendDeviceProbeError::DriverUnavailable {
                provider: "CUDA",
                message,
            },
        )?;
        // SAFETY: each symbol name and signature is defined by the CUDA Driver
        // API. The handle remains live through all calls below.
        let symbol = |name| {
            library
                .symbol(name)
                .map_err(|message| BackendDeviceProbeError::DriverQueryFailed {
                    provider: "CUDA",
                    message,
                })
        };
        let init: CuInit = unsafe { std::mem::transmute(symbol(b"cuInit\0")?) };
        let get_count: CuDeviceGetCount =
            unsafe { std::mem::transmute(symbol(b"cuDeviceGetCount\0")?) };
        let get_device: CuDeviceGet = unsafe { std::mem::transmute(symbol(b"cuDeviceGet\0")?) };
        let get_attribute: CuDeviceGetAttribute =
            unsafe { std::mem::transmute(symbol(b"cuDeviceGetAttribute\0")?) };
        let get_driver: CuDriverGetVersion =
            unsafe { std::mem::transmute(symbol(b"cuDriverGetVersion\0")?) };
        // SAFETY: CUDA accepts these primitive pointers and device ordinals.
        if unsafe { init(0) } != 0 {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "CUDA",
                message: "driver initialization failed".to_string(),
            });
        }
        let mut count = 0;
        if unsafe { get_count(&mut count) } != 0 || count <= 0 {
            return Err(BackendDeviceProbeError::NoDevice("CUDA"));
        }
        // Target-scoped packs follow the provider's primary device (ordinal
        // zero). A mixed-GPU workstation is valid; rejecting it merely because
        // another adapter has a different SM makes the CUDA lifecycle unusable.
        let mut device = 0;
        let mut major = 0;
        let mut minor = 0;
        if unsafe { get_device(&mut device, 0) } != 0
            || unsafe { get_attribute(&mut major, CUDA_COMPUTE_CAPABILITY_MAJOR, device) } != 0
            || unsafe { get_attribute(&mut minor, CUDA_COMPUTE_CAPABILITY_MINOR, device) } != 0
            || major <= 0
            || minor < 0
        {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "CUDA",
                message: "could not inspect primary device ordinal 0".to_string(),
            });
        }
        let mut driver = 0;
        if unsafe { get_driver(&mut driver) } != 0 {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "CUDA",
                message: "driver API version query failed".to_string(),
            });
        }
        Ok(BackendDeviceProof {
            target: format!("sm_{major}{minor}"),
            driver_api_version: normalize_cuda_driver_version(driver)
                .map_err(BackendDeviceProbeError::UnknownTarget)?,
        })
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct HsaAgent {
        handle: u64,
    }

    type HsaAgentGetInfo = unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> u32;

    struct HsaCallbackState {
        get_info: HsaAgentGetInfo,
        primary_target: Option<String>,
        gpu_agent_count: usize,
    }

    unsafe extern "C" fn collect_hsa_agent(agent: HsaAgent, data: *mut c_void) -> u32 {
        // SAFETY: `data` points to HsaCallbackState for the synchronous
        // hsa_iterate_agents call; HSA supplies a valid agent handle.
        let state = unsafe { &mut *(data.cast::<HsaCallbackState>()) };
        let mut device_type = 0_u32;
        if unsafe {
            (state.get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                (&mut device_type as *mut u32).cast(),
            )
        } != HSA_STATUS_SUCCESS
            || device_type != HSA_DEVICE_TYPE_GPU
        {
            return HSA_STATUS_SUCCESS;
        }
        state.gpu_agent_count += 1;
        let mut name = [0_u8; 64];
        if unsafe { (state.get_info)(agent, HSA_AGENT_INFO_NAME, name.as_mut_ptr().cast()) }
            != HSA_STATUS_SUCCESS
        {
            return HSA_STATUS_SUCCESS;
        }
        let length = name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(name.len());
        if state.gpu_agent_count == 1
            && let Ok(name) = std::str::from_utf8(&name[..length])
            && let Some(target) = canonical_hip_target(name)
        {
            state.primary_target = Some(target);
        }
        HSA_STATUS_SUCCESS
    }

    fn find_verified_runtime_library(
        runtime: &PreparedBackendRuntimeObjects,
        filename: &str,
    ) -> Result<PathBuf, String> {
        let mut matches = runtime.files.iter().filter(|file| {
            file.path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(filename))
        });
        let proof = matches
            .next()
            .ok_or_else(|| format!("verified HIP runtime omitted {filename}"))?;
        if matches.next().is_some() {
            return Err(format!(
                "verified HIP runtime declared more than one {filename}"
            ));
        }
        let path = proof
            .path
            .canonicalize()
            .map_err(|error| format!("verified HIP runtime file could not be resolved: {error}"))?;
        let inside_verified_root = runtime.dependency_dirs.iter().any(|directory| {
            directory
                .canonicalize()
                .is_ok_and(|directory| path.starts_with(directory))
        });
        if !inside_verified_root {
            return Err("verified HIP runtime file escaped its content object".to_string());
        }
        let metadata = path
            .metadata()
            .map_err(|error| format!("verified HIP runtime metadata failed: {error}"))?;
        if !metadata.is_file() || metadata.len() != proof.size_bytes {
            return Err("verified HIP runtime size changed before loading".to_string());
        }
        let mut file = File::open(&path)
            .map_err(|error| format!("verified HIP runtime could not be reopened: {error}"))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 128 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("verified HIP runtime rehash failed: {error}"))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if !actual.eq_ignore_ascii_case(&proof.sha256) {
            return Err("verified HIP runtime hash changed before loading".to_string());
        }
        Ok(path)
    }

    pub(super) fn probe_hip(
        runtime: &PreparedBackendRuntimeObjects,
    ) -> Result<BackendDeviceProof, BackendDeviceProbeError> {
        type HsaInit = unsafe extern "C" fn() -> u32;
        type HsaShutdown = unsafe extern "C" fn() -> u32;
        type HsaIterate = unsafe extern "C" fn(
            unsafe extern "C" fn(HsaAgent, *mut c_void) -> u32,
            *mut c_void,
        ) -> u32;
        type HipDriverGetVersion = unsafe extern "C" fn(*mut i32) -> i32;

        let runtime_error = BackendDeviceProbeError::DiscoveryRuntimeUnavailable;
        let _load_guard = HIP_DISCOVERY_LOAD_LOCK
            .lock()
            .map_err(|_| runtime_error("HIP discovery loader lock was poisoned".to_string()))?;
        let hsa_path =
            find_verified_runtime_library(runtime, "hsa-runtime64.dll").map_err(runtime_error)?;
        let hip_path =
            find_verified_runtime_library(runtime, "amdhip64.dll").map_err(runtime_error)?;
        let _search_directories =
            DllSearchDirectories::add(&runtime.dependency_dirs).map_err(runtime_error)?;
        let search_flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
            | LOAD_LIBRARY_SEARCH_USER_DIRS
            | LOAD_LIBRARY_SEARCH_SYSTEM32;
        let hsa = DynamicLibrary::load(&hsa_path, search_flags).map_err(runtime_error)?;
        let hip = DynamicLibrary::load(&hip_path, search_flags).map_err(runtime_error)?;
        // SAFETY: signatures are the stable HSA/HIP C ABIs; both handles stay
        // live until discovery and shutdown complete.
        let hsa_init: HsaInit =
            unsafe { std::mem::transmute(hsa.symbol(b"hsa_init\0").map_err(runtime_error)?) };
        let hsa_shutdown: HsaShutdown =
            unsafe { std::mem::transmute(hsa.symbol(b"hsa_shut_down\0").map_err(runtime_error)?) };
        let hsa_iterate: HsaIterate = unsafe {
            std::mem::transmute(hsa.symbol(b"hsa_iterate_agents\0").map_err(runtime_error)?)
        };
        let hsa_get_info: HsaAgentGetInfo = unsafe {
            std::mem::transmute(hsa.symbol(b"hsa_agent_get_info\0").map_err(runtime_error)?)
        };
        let hip_driver: HipDriverGetVersion = unsafe {
            std::mem::transmute(
                hip.symbol(b"hipDriverGetVersion\0")
                    .map_err(runtime_error)?,
            )
        };
        if unsafe { hsa_init() } != HSA_STATUS_SUCCESS {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "HIP",
                message: "HSA runtime initialization failed".to_string(),
            });
        }
        let mut state = HsaCallbackState {
            get_info: hsa_get_info,
            primary_target: None,
            gpu_agent_count: 0,
        };
        let iterate_status = unsafe {
            hsa_iterate(
                collect_hsa_agent,
                (&mut state as *mut HsaCallbackState).cast(),
            )
        };
        let shutdown_status = unsafe { hsa_shutdown() };
        if iterate_status != HSA_STATUS_SUCCESS || shutdown_status != HSA_STATUS_SUCCESS {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "HIP",
                message: "HSA device enumeration failed".to_string(),
            });
        }
        let mut driver = 0;
        if unsafe { hip_driver(&mut driver) } != 0 {
            return Err(BackendDeviceProbeError::DriverQueryFailed {
                provider: "HIP",
                message: "driver API version query failed".to_string(),
            });
        }
        let target = state.primary_target.ok_or_else(|| {
            if state.gpu_agent_count > 0 {
                BackendDeviceProbeError::UnknownTarget(
                    "HIP primary HSA GPU agent did not expose a canonical gfx target".to_string(),
                )
            } else {
                BackendDeviceProbeError::NoDevice("HIP")
            }
        })?;
        Ok(BackendDeviceProof {
            target,
            driver_api_version: normalize_hip_driver_version(driver)
                .map_err(BackendDeviceProbeError::UnknownTarget)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_driver_versions_match_plugin_probe_normalization() {
        assert_eq!(normalize_cuda_driver_version(12_070).unwrap(), "12.7.0");
        assert_eq!(normalize_hip_driver_version(60_504_000).unwrap(), "6.5.4");
        assert!(normalize_cuda_driver_version(0).is_err());
        assert!(normalize_hip_driver_version(-1).is_err());
    }

    #[test]
    fn hip_target_is_strict_and_drops_only_feature_suffixes() {
        assert_eq!(
            canonical_hip_target("gfx1201:sramecc+"),
            Some("gfx1201".to_string())
        );
        assert_eq!(
            canonical_hip_target(" GFX1100 "),
            Some("gfx1100".to_string())
        );
        assert_eq!(canonical_hip_target("Radeon 7900 XTX"), None);
        assert_eq!(canonical_hip_target("gfx90a"), None);
    }

    #[test]
    fn probe_failure_codes_are_stable_machine_contracts() {
        assert_eq!(
            BackendDeviceProbeError::NoDevice("CUDA").code(),
            "no_device"
        );
        assert_eq!(
            BackendDeviceProbeError::UnknownTarget("unknown".to_string()).code(),
            "unknown_target"
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a physical NVIDIA GPU and installed Windows driver"]
    fn live_cuda_probe_reports_one_canonical_target() {
        let proof = probe_provider_device(
            CatalogBackendVendor::Cuda,
            &PreparedBackendRuntimeObjects::default(),
        )
        .unwrap();
        eprintln!(
            "OPENASR_BACKEND_DEVICE_PROOF target={} driver_api_version={}",
            proof.target, proof.driver_api_version
        );
        assert!(proof.target.starts_with("sm_"));
        assert!(proof.target[3..].bytes().all(|byte| byte.is_ascii_digit()));
        assert!(proof.driver_api_version.split('.').count() >= 2);
        if let Ok(expected) = std::env::var("OPENASR_TEST_EXPECTED_CUDA_TARGET") {
            assert_eq!(proof.target, expected);
        }
    }
}
