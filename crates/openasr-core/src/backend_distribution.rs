//! Windows backend-host compatibility identity.
//!
//! The release catalog, installed-pack marker, and runtime loader all compare
//! this exact identity. It deliberately describes only the neutral host ABI;
//! backend-specific GPU targets and vendor-runtime requirements belong to the
//! backend pack identity and do not make otherwise-compatible hosts diverge.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CatalogBackendFileRole, CatalogBackendVendor, ModelCatalog,
    atomic_file::write_file_atomically,
    backend_device_probe::probe_provider_device,
    ggml_runtime::probe_exact_backend_plugin_candidate,
    pull::{
        BackendStoreMutationLock, InstalledBackend, PreparedBackendRuntimeObjects, PullProgress,
        backend_artifact_fingerprint, backend_pack_download_plan, backend_pack_install_dir,
        install_backend_pack, install_backend_pack_locked, installed_backend_protected_bytes,
        prepare_backend_runtime_objects_locked, read_and_verify_installed_backend,
    },
    registry::{resolve_catalog_backend_pull, resolve_compatible_catalog_backend_pull_for_driver},
};

pub const BACKEND_HOST_ABI_SCHEMA_VERSION: u32 = 2;
pub const ACTIVATED_BACKEND_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendHostAbi {
    pub schema_version: u32,
    pub fingerprint: String,
    pub target: String,
    pub crt: String,
    #[serde(default)]
    pub toolchain: String,
    #[serde(default)]
    pub compile_flags_sha256: String,
    pub ggml_backend_api_version: u32,
    pub ggml_revision: String,
    pub ggml_headers_sha256: String,
    pub openasr_ffi_sha256: String,
    #[serde(default)]
    pub openasr_extension_sha256: String,
}

/// The one optional backend pack selected for the next process. This pointer
/// contains no executable path: runtime re-resolves the id against the signed
/// catalog, checks the exact host/device/driver contract, and rehashes the
/// installed pack before loading its plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivatedBackendPack {
    pub schema_version: u32,
    pub backend_id: String,
    pub vendor: CatalogBackendVendor,
    pub version: String,
    pub artifact_fingerprint: String,
    pub host_abi_fingerprint: String,
    pub device_target: String,
    pub driver_version: String,
    pub activated_at_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPluginStatus {
    pub schema_version: u32,
    /// `neutral_dynamic` is the only topology that may consume optional
    /// backend packs. `legacy_static` keeps old whole-sidecar clients
    /// diagnosable during the migration window without treating them as
    /// plugin hosts.
    pub host_mode: String,
    pub host_abi: BackendHostAbi,
    pub activated: Option<ActivatedBackendPack>,
}

/// One provider pack prepared for the exact GPU target reported by the live
/// driver. Preparation installs and verifies bytes but deliberately does not
/// mutate the activation selector, so a product shell can defer the process
/// restart until cold start or an explicitly proven idle boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedBackendPack {
    pub schema_version: u32,
    pub backend_id: String,
    pub vendor: CatalogBackendVendor,
    pub version: String,
    pub artifact_fingerprint: String,
    pub host_abi_fingerprint: String,
    pub device_target: String,
    pub driver_version: String,
    pub size_bytes: u64,
    pub plugin_size_bytes: u64,
    pub vendor_size_bytes: u64,
    /// Conservative logical bytes protected by this installed pack and its
    /// shared content objects. Product shells use this proof for retention
    /// budgets without inspecting open-core's private store layout.
    pub protected_bytes: u64,
}

/// Download sizing for a provider before consent. Target-specific plugin
/// bytes are reported as a conservative maximum; the live-device preparation
/// transaction later selects exactly one target pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProviderDescription {
    pub schema_version: u32,
    pub vendor: CatalogBackendVendor,
    pub host_abi_fingerprint: String,
    pub target_pack_count: usize,
    pub size_bytes: u64,
    pub plugin_size_bytes: u64,
    pub vendor_size_bytes: u64,
    pub required_download_size_bytes: u64,
    pub required_plugin_download_size_bytes: u64,
    pub required_vendor_download_size_bytes: u64,
}

#[derive(Debug, Error)]
pub enum BackendActivationError {
    #[error("backend activation state could not be read from '{path}': {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("backend activation state at '{path}' is invalid: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("backend activation state has unsupported schema {0}")]
    UnsupportedSchema(u32),
    #[error("no unique compatible backend pack is available: {0}")]
    Resolution(String),
    #[error("compatible backend pack is not installed or failed verification: {0}")]
    InstalledPack(String),
    #[error("backend pack installation failed: {0}")]
    Install(String),
    #[error("backend plugin store is busy or unavailable: {0}")]
    Store(String),
    #[error("backend activation state could not be written to '{path}': {source}")]
    Write { path: PathBuf, source: io::Error },
    #[error("backend activation device target and live driver proof must be non-empty")]
    MissingDeviceProof,
    #[error("the requested provider is unsupported by the live device or driver: {0}")]
    UnsupportedDevice(String),
    #[error("the requested provider failed live device discovery ({code}): {message}")]
    DeviceProbe { code: &'static str, message: String },
    #[error("no signed backend pack matches live target '{target}': {message}")]
    NoCatalogMatch { target: String, message: String },
}

impl BackendActivationError {
    pub fn machine_failure_class(&self) -> &'static str {
        match self {
            Self::UnsupportedDevice(_) | Self::DeviceProbe { .. } | Self::NoCatalogMatch { .. } => {
                "unsupported_device"
            }
            Self::Install(_) | Self::Store(_) => "download",
            Self::InstalledPack(_)
            | Self::Resolution(_)
            | Self::MissingDeviceProof
            | Self::UnsupportedSchema(_)
            | Self::Parse { .. } => "verification",
            Self::Read { .. } | Self::Write { .. } => "io",
        }
    }

    pub fn machine_failure_code(&self) -> &'static str {
        match self {
            Self::UnsupportedDevice(_) => "unsupported_device",
            Self::DeviceProbe { code, .. } => code,
            Self::NoCatalogMatch { .. } => "no_catalog_match",
            Self::Install(_) => "install_failed",
            Self::Store(_) => "store_unavailable",
            Self::InstalledPack(_) => "installed_pack_invalid",
            Self::Resolution(_) => "catalog_resolution_failed",
            Self::MissingDeviceProof => "device_proof_missing",
            Self::UnsupportedSchema(_) => "state_schema_unsupported",
            Self::Parse { .. } => "state_parse_failed",
            Self::Read { .. } => "state_read_failed",
            Self::Write { .. } => "state_write_failed",
        }
    }
}

/// The production transaction for an optional backend pack. The caller names
/// one signed-catalog backend id; core owns resolution, installation, complete
/// file re-verification, live target/driver proof, and the final atomic
/// activation pointer. Callers must not synthesize an `active.json` record or
/// infer a target from an OS adapter label.
pub fn install_and_activate_backend_pack(
    catalog: &ModelCatalog,
    backend_id: &str,
    home: &Path,
    progress: impl FnMut(PullProgress),
) -> Result<ActivatedBackendPack, BackendActivationError> {
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    if !BackendHostAbi::current().is_compatible_with(&requested.host_abi) {
        return Err(BackendActivationError::Resolution(
            "selected backend does not match the current neutral-host ABI".to_string(),
        ));
    }
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    install_backend_pack_locked(&requested, home, progress)
        .map_err(|error| BackendActivationError::Install(error.to_string()))?;
    activate_installed_backend_pack_auto_locked(catalog, &requested, home)
}

pub fn install_and_activate_backend_provider(
    catalog: &ModelCatalog,
    vendor: CatalogBackendVendor,
    home: &Path,
    mut progress: impl FnMut(PullProgress),
) -> Result<ActivatedBackendPack, BackendActivationError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    let prepared =
        prepare_backend_provider_for_live_device_locked(catalog, vendor, home, &mut progress)?;
    activate_installed_backend_pack_locked(
        catalog,
        &prepared.backend_id,
        &prepared.device_target,
        home,
    )
}

/// Discover the exact live GPU architecture and install only its signed pack.
///
/// CUDA discovery uses the Windows driver DLL and performs no download. HIP
/// first prepares the runtime/archive objects that are byte-identical across
/// every host-compatible target pack, then queries the signed HSA/HIP runtime
/// for the canonical `gfx` target. The global store lock covers bootstrap,
/// target resolution, and installation so concurrent clients cannot observe a
/// half-prepared provider generation.
pub fn prepare_backend_provider_for_live_device(
    catalog: &ModelCatalog,
    vendor: CatalogBackendVendor,
    home: &Path,
    mut progress: impl FnMut(PullProgress),
) -> Result<PreparedBackendPack, BackendActivationError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    prepare_backend_provider_for_live_device_locked(catalog, vendor, home, &mut progress)
}

fn prepare_backend_provider_for_live_device_locked(
    catalog: &ModelCatalog,
    vendor: CatalogBackendVendor,
    home: &Path,
    progress: &mut impl FnMut(PullProgress),
) -> Result<PreparedBackendPack, BackendActivationError> {
    if !matches!(
        vendor,
        CatalogBackendVendor::Cuda | CatalogBackendVendor::Hip
    ) {
        return Err(BackendActivationError::Resolution(
            "only CUDA and HIP use target-scoped provider preparation".to_string(),
        ));
    }
    let host_abi = BackendHostAbi::current();
    let runtime = if vendor == CatalogBackendVendor::Hip {
        let bootstrap = shared_provider_runtime_bootstrap(catalog, vendor, &host_abi)?;
        prepare_backend_runtime_objects_locked(&bootstrap, home, &mut *progress)
            .map_err(|error| BackendActivationError::Install(error.to_string()))?
    } else {
        PreparedBackendRuntimeObjects::default()
    };
    let device = probe_provider_device(vendor, &runtime).map_err(|error| {
        BackendActivationError::DeviceProbe {
            code: error.code(),
            message: error.to_string(),
        }
    })?;
    let resolved = resolve_compatible_catalog_backend_pull_for_driver(
        catalog,
        vendor,
        &host_abi,
        Some(&device.target),
        Some(&device.driver_api_version),
    )
    .map_err(|error| BackendActivationError::NoCatalogMatch {
        target: device.target.clone(),
        message: error.to_string(),
    })?;
    install_backend_pack_locked(&resolved, home, &mut *progress)
        .map_err(|error| BackendActivationError::Install(error.to_string()))?;
    let plugin_size_bytes = resolved
        .files
        .iter()
        .filter(|file| file.role == CatalogBackendFileRole::Plugin)
        .try_fold(0_u64, |total, file| total.checked_add(file.size_bytes))
        .ok_or_else(|| BackendActivationError::Resolution("backend size overflow".to_string()))?;
    let vendor_size_bytes = resolved
        .files
        .iter()
        .filter(|file| file.role != CatalogBackendFileRole::Plugin)
        .try_fold(0_u64, |total, file| total.checked_add(file.size_bytes))
        .ok_or_else(|| BackendActivationError::Resolution("backend size overflow".to_string()))?;
    let size_bytes = plugin_size_bytes
        .checked_add(vendor_size_bytes)
        .ok_or_else(|| BackendActivationError::Resolution("backend size overflow".to_string()))?;
    let protected_bytes = installed_backend_protected_bytes(&resolved, home)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    Ok(PreparedBackendPack {
        schema_version: 1,
        backend_id: resolved.backend_id.clone(),
        vendor,
        version: resolved.version.clone(),
        artifact_fingerprint: backend_artifact_fingerprint(&resolved),
        host_abi_fingerprint: resolved.host_abi.fingerprint.clone(),
        device_target: device.target,
        driver_version: device.driver_api_version,
        size_bytes,
        plugin_size_bytes,
        vendor_size_bytes,
        protected_bytes,
    })
}

pub fn describe_backend_provider(
    catalog: &ModelCatalog,
    vendor: CatalogBackendVendor,
    home: &Path,
) -> Result<BackendProviderDescription, BackendActivationError> {
    let host_abi = BackendHostAbi::current();
    let mut variants = catalog
        .backends
        .iter()
        .filter(|backend| backend.vendor == vendor)
        .filter(|backend| host_abi.is_compatible_with(&backend.host_abi))
        .map(|backend| resolve_catalog_backend_pull(catalog, &backend.id))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    variants.sort_by(|left, right| left.backend_id.cmp(&right.backend_id));
    if variants.is_empty() {
        return Err(BackendActivationError::Resolution(
            "no host-compatible provider pack is available".to_string(),
        ));
    }
    if vendor == CatalogBackendVendor::Hip {
        shared_provider_runtime_bootstrap(catalog, vendor, &host_abi)?;
    }
    let mut result = BackendProviderDescription {
        schema_version: 1,
        vendor,
        host_abi_fingerprint: host_abi.fingerprint,
        target_pack_count: variants.len(),
        size_bytes: 0,
        plugin_size_bytes: 0,
        vendor_size_bytes: 0,
        required_download_size_bytes: 0,
        required_plugin_download_size_bytes: 0,
        required_vendor_download_size_bytes: 0,
    };
    for variant in &variants {
        let plan = backend_pack_download_plan(home, variant)
            .map_err(|error| BackendActivationError::Install(error.to_string()))?;
        result.size_bytes = result.size_bytes.max(plan.total_bytes);
        result.plugin_size_bytes = result.plugin_size_bytes.max(plan.plugin_bytes);
        result.vendor_size_bytes = result.vendor_size_bytes.max(plan.vendor_bytes);
        result.required_download_size_bytes = result
            .required_download_size_bytes
            .max(plan.required_download_bytes);
        result.required_plugin_download_size_bytes = result
            .required_plugin_download_size_bytes
            .max(plan.required_plugin_bytes);
        result.required_vendor_download_size_bytes = result
            .required_vendor_download_size_bytes
            .max(plan.required_vendor_bytes);
    }
    Ok(result)
}

fn shared_provider_runtime_bootstrap(
    catalog: &ModelCatalog,
    vendor: CatalogBackendVendor,
    host_abi: &BackendHostAbi,
) -> Result<crate::ResolvedCatalogBackendPull, BackendActivationError> {
    let mut variants = catalog
        .backends
        .iter()
        .filter(|backend| backend.vendor == vendor)
        .filter(|backend| host_abi.is_compatible_with(&backend.host_abi))
        .map(|backend| resolve_catalog_backend_pull(catalog, &backend.id))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    variants.sort_by(|left, right| left.backend_id.cmp(&right.backend_id));
    let Some(first) = variants.first() else {
        return Err(BackendActivationError::Resolution(
            "no host-compatible provider pack is available".to_string(),
        ));
    };
    let expected = shared_runtime_identity(first);
    if expected.is_empty() {
        return Err(BackendActivationError::Resolution(
            "provider packs omit the signed shared discovery runtime".to_string(),
        ));
    }
    if variants
        .iter()
        .skip(1)
        .any(|candidate| shared_runtime_identity(candidate) != expected)
    {
        return Err(BackendActivationError::Resolution(
            "provider target packs disagree on their shared runtime identity".to_string(),
        ));
    }
    let mut bootstrap = first.clone();
    bootstrap
        .files
        .retain(|file| file.role != CatalogBackendFileRole::Plugin);
    Ok(bootstrap)
}

fn shared_runtime_identity(
    resolved: &crate::ResolvedCatalogBackendPull,
) -> Vec<(String, String, u64, String, Option<String>, Option<String>)> {
    let mut identity = resolved
        .files
        .iter()
        .filter(|file| file.role != CatalogBackendFileRole::Plugin)
        .map(|file| {
            (
                file.filename.clone(),
                file.sha256.clone(),
                file.size_bytes,
                format!("{:?}", file.role),
                file.extract_subdir.clone(),
                file.extracted_tree_sha256.clone(),
            )
        })
        .collect::<Vec<_>>();
    identity.sort();
    identity
}

pub fn install_backend_pack_from_catalog(
    catalog: &ModelCatalog,
    backend_id: &str,
    home: &Path,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, BackendActivationError> {
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    // Install-only may target a future signed host ABI. Activation still
    // checks the current host exactly, so prefetched native code cannot enter
    // the old process during an NSIS hand-off.
    install_backend_pack(&requested, home, progress)
        .map_err(|error| BackendActivationError::Install(error.to_string()))
}

pub fn activate_installed_backend_pack_auto(
    catalog: &ModelCatalog,
    backend_id: &str,
    home: &Path,
) -> Result<ActivatedBackendPack, BackendActivationError> {
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    if !BackendHostAbi::current().is_compatible_with(&requested.host_abi) {
        return Err(BackendActivationError::Resolution(
            "selected backend does not match the current neutral-host ABI".to_string(),
        ));
    }
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    activate_installed_backend_pack_auto_locked(catalog, &requested, home)
}

fn activate_installed_backend_pack_auto_locked(
    catalog: &ModelCatalog,
    requested: &crate::ResolvedCatalogBackendPull,
    home: &Path,
) -> Result<ActivatedBackendPack, BackendActivationError> {
    let backend_id = requested.backend_id.as_str();
    let install_dir = backend_pack_install_dir(home, requested)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let installed = read_and_verify_installed_backend(&install_dir, requested)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let canonical_dir = fs::canonicalize(&installed.dir)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let plugin_path = fs::canonicalize(installed.dir.join(&installed.plugin_filename))
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    if !plugin_path.starts_with(&canonical_dir) {
        return Err(BackendActivationError::InstalledPack(
            "verified plugin path escaped its install directory".to_string(),
        ));
    }
    let dependency_dirs = verified_backend_dependency_dirs(backend_id, &canonical_dir, &installed)?;

    // A fat pack may declare several signed SM/gfx targets. Probe them in the
    // catalog's signed order and freeze the first live proof. The target is an
    // artifact-compatibility proof, not a Desktop device-selection policy;
    // exact device routing remains a separate runtime contract.
    let mut selected_target = None;
    for target in &requested.targets {
        if probe_exact_backend_plugin_candidate(
            backend_id,
            requested.vendor,
            &plugin_path,
            &dependency_dirs,
            target,
            requested.min_driver_api.as_deref(),
        )
        .is_ok()
        {
            selected_target = Some(target.as_str());
            break;
        }
    }
    let selected_target = selected_target.ok_or_else(|| {
        BackendActivationError::Resolution(
            "installed backend did not attest any signed device target".to_string(),
        )
    })?;
    activate_installed_backend_pack_locked(catalog, backend_id, selected_target, home)
}

pub fn backend_plugin_status(home: &Path) -> Result<BackendPluginStatus, BackendActivationError> {
    let dynamic = crate::ggml_runtime::backend_plugin_host_available();
    Ok(BackendPluginStatus {
        schema_version: 1,
        host_mode: if dynamic {
            "neutral_dynamic"
        } else {
            "legacy_static"
        }
        .to_string(),
        host_abi: BackendHostAbi::current(),
        activated: read_activated_backend(home)?,
    })
}

pub fn deactivate_backend_pack(home: &Path) -> Result<(), BackendActivationError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    let path = activated_backend_path(home);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(BackendActivationError::Write { path, source }),
    }
}

pub fn activated_backend_path(home: &Path) -> PathBuf {
    home.join("backends").join("active.json")
}

pub fn read_activated_backend(
    home: &Path,
) -> Result<Option<ActivatedBackendPack>, BackendActivationError> {
    let path = activated_backend_path(home);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(BackendActivationError::Read { path, source }),
    };
    let record: ActivatedBackendPack = serde_json::from_str(&text)
        .map_err(|source| BackendActivationError::Parse { path, source })?;
    if record.schema_version != ACTIVATED_BACKEND_SCHEMA_VERSION {
        return Err(BackendActivationError::UnsupportedSchema(
            record.schema_version,
        ));
    }
    if record.device_target.trim().is_empty() || record.driver_version.trim().is_empty() {
        return Err(BackendActivationError::MissingDeviceProof);
    }
    Ok(Some(record))
}

/// Verifies and atomically activates one installed pack. Catalog resolution is
/// repeated at runtime, so this record is an activation pointer, never a
/// substitute for the signed catalog or installed-file hashes.
pub fn activate_installed_backend_pack(
    catalog: &ModelCatalog,
    backend_id: &str,
    device_target: &str,
    home: &Path,
) -> Result<ActivatedBackendPack, BackendActivationError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    activate_installed_backend_pack_locked(catalog, backend_id, device_target, home)
}

fn activate_installed_backend_pack_locked(
    catalog: &ModelCatalog,
    backend_id: &str,
    device_target: &str,
    home: &Path,
) -> Result<ActivatedBackendPack, BackendActivationError> {
    if device_target.trim().is_empty() {
        return Err(BackendActivationError::MissingDeviceProof);
    }
    let host_abi = BackendHostAbi::current();
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    if !host_abi.is_compatible_with(&requested.host_abi) {
        return Err(BackendActivationError::Resolution(
            "selected backend does not match the current neutral-host ABI".to_string(),
        ));
    }
    let install_dir = backend_pack_install_dir(home, &requested)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let installed = read_and_verify_installed_backend(&install_dir, &requested)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let canonical_dir = fs::canonicalize(&install_dir)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let plugin_path = fs::canonicalize(install_dir.join(&installed.plugin_filename))
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    if !plugin_path.starts_with(&canonical_dir) {
        return Err(BackendActivationError::InstalledPack(
            "verified plugin path escaped its install directory".to_string(),
        ));
    }
    let dependency_dirs = verified_backend_dependency_dirs(backend_id, &canonical_dir, &installed)?;
    let driver_version = probe_exact_backend_plugin_candidate(
        backend_id,
        requested.vendor,
        &plugin_path,
        &dependency_dirs,
        device_target,
        requested.min_driver_api.as_deref(),
    )
    .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    let resolved = resolve_compatible_catalog_backend_pull_for_driver(
        catalog,
        requested.vendor,
        &host_abi,
        Some(device_target),
        Some(&driver_version),
    )
    .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    if resolved.backend_id != requested.backend_id {
        return Err(BackendActivationError::Resolution(format!(
            "live device/driver proof resolves to '{}' instead of the selected pack",
            resolved.backend_id
        )));
    }
    let record = ActivatedBackendPack {
        schema_version: ACTIVATED_BACKEND_SCHEMA_VERSION,
        backend_id: resolved.backend_id.clone(),
        vendor: resolved.vendor,
        version: resolved.version.clone(),
        artifact_fingerprint: backend_artifact_fingerprint(&resolved),
        host_abi_fingerprint: host_abi.fingerprint,
        device_target: device_target.to_ascii_lowercase(),
        driver_version,
        activated_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    };
    let path = activated_backend_path(home);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| BackendActivationError::Write {
            path: path.clone(),
            source,
        })?;
    }
    let mut json = serde_json::to_vec_pretty(&record).expect("activation record serializes");
    json.push(b'\n');
    write_file_atomically(&path, &json)
        .map_err(|source| BackendActivationError::Write { path, source })?;
    Ok(record)
}

pub(crate) fn verified_backend_dependency_dirs(
    backend_id: &str,
    canonical_install_dir: &Path,
    installed: &InstalledBackend,
) -> Result<Vec<PathBuf>, BackendActivationError> {
    let mut dependency_dirs = std::collections::BTreeSet::new();
    for file in &installed.files {
        if file.role == crate::CatalogBackendFileRole::Plugin {
            continue;
        }
        for materialized in &file.materialized_files {
            let Some(parent) = Path::new(&materialized.relative_path).parent() else {
                continue;
            };
            let candidate = fs::canonicalize(canonical_install_dir.join(parent))
                .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
            if !candidate.starts_with(canonical_install_dir) {
                return Err(BackendActivationError::InstalledPack(format!(
                    "backend '{backend_id}' dependency directory escaped its install directory"
                )));
            }
            dependency_dirs.insert(candidate);
        }
    }
    Ok(dependency_dirs.into_iter().collect())
}

impl BackendHostAbi {
    pub fn current() -> Self {
        Self {
            schema_version: env!("OPENASR_BACKEND_ABI_SCHEMA_VERSION")
                .parse()
                .expect("build.rs emitted an invalid backend ABI schema version"),
            fingerprint: env!("OPENASR_BACKEND_HOST_ABI_FINGERPRINT").to_string(),
            target: env!("OPENASR_BACKEND_TARGET").to_string(),
            crt: env!("OPENASR_BACKEND_CRT").to_string(),
            toolchain: env!("OPENASR_BACKEND_TOOLCHAIN").to_string(),
            compile_flags_sha256: env!("OPENASR_BACKEND_COMPILE_FLAGS_SHA256").to_string(),
            ggml_backend_api_version: env!("OPENASR_GGML_BACKEND_API_VERSION")
                .parse()
                .expect("build.rs emitted an invalid ggml backend API version"),
            ggml_revision: env!("OPENASR_GGML_REVISION").to_string(),
            ggml_headers_sha256: env!("OPENASR_GGML_HEADERS_SHA256").to_string(),
            openasr_ffi_sha256: env!("OPENASR_GGML_FFI_SHA256").to_string(),
            openasr_extension_sha256: env!("OPENASR_GGML_EXTENSION_SHA256").to_string(),
        }
    }

    pub fn is_compatible_with(&self, candidate: &Self) -> bool {
        self.schema_version == candidate.schema_version && self.fingerprint == candidate.fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved_with_files(
        files: Vec<crate::CatalogBackendFile>,
    ) -> crate::ResolvedCatalogBackendPull {
        crate::ResolvedCatalogBackendPull {
            backend_id: "hip-windows-gfx1100".to_string(),
            vendor: CatalogBackendVendor::Hip,
            version: "test".to_string(),
            display_name: "HIP".to_string(),
            host_abi: BackendHostAbi::current(),
            targets: vec!["gfx1100".to_string()],
            min_driver_api: Some("6.0.0".to_string()),
            files,
        }
    }

    fn backend_file(
        filename: &str,
        sha256: char,
        role: CatalogBackendFileRole,
    ) -> crate::CatalogBackendFile {
        crate::CatalogBackendFile {
            filename: filename.to_string(),
            url: format!("https://example.invalid/{filename}"),
            mirrors: Vec::new(),
            sha256: sha256.to_string().repeat(64),
            size_bytes: 42,
            role,
            extract_subdir: None,
            extracted_tree_sha256: None,
        }
    }

    fn is_lower_hex(value: &str, len: usize) -> bool {
        value.len() == len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    #[test]
    fn current_backend_host_abi_is_complete_and_self_compatible() {
        let current = BackendHostAbi::current();
        assert_eq!(current.schema_version, BACKEND_HOST_ABI_SCHEMA_VERSION);
        assert!(is_lower_hex(&current.fingerprint, 64));
        assert!(!current.target.is_empty());
        assert!(!current.crt.is_empty());
        assert!(!current.toolchain.is_empty());
        assert!(is_lower_hex(&current.compile_flags_sha256, 64));
        assert!(current.ggml_backend_api_version > 0);
        assert!(!current.ggml_revision.is_empty());
        assert!(is_lower_hex(&current.ggml_headers_sha256, 64));
        assert!(is_lower_hex(&current.openasr_ffi_sha256, 64));
        assert!(is_lower_hex(&current.openasr_extension_sha256, 64));
        assert!(current.is_compatible_with(&current));
    }

    #[test]
    fn compatibility_is_exact_and_schema_scoped() {
        let current = BackendHostAbi::current();
        let mut different_fingerprint = current.clone();
        different_fingerprint.fingerprint = "0".repeat(64);
        assert!(!current.is_compatible_with(&different_fingerprint));

        let mut different_schema = current.clone();
        different_schema.schema_version += 1;
        assert!(!current.is_compatible_with(&different_schema));
    }

    #[test]
    fn shared_runtime_identity_excludes_target_plugin_but_binds_runtime_bytes() {
        let first = resolved_with_files(vec![
            backend_file("ggml-hip.dll", 'a', CatalogBackendFileRole::Plugin),
            backend_file("hip-runtime.zip", 'b', CatalogBackendFileRole::Archive),
        ]);
        let mut second = first.clone();
        second.files[0].sha256 = "c".repeat(64);
        second.files[1].url = "https://mirror.invalid/runtime.zip".to_string();
        assert_eq!(
            shared_runtime_identity(&first),
            shared_runtime_identity(&second)
        );

        second.files[1].sha256 = "d".repeat(64);
        assert_ne!(
            shared_runtime_identity(&first),
            shared_runtime_identity(&second)
        );
    }

    #[test]
    fn machine_failure_contract_is_stable_and_actionable() {
        let cases = [
            (
                BackendActivationError::DeviceProbe {
                    code: "driver_unavailable",
                    message: "redacted".to_string(),
                },
                "unsupported_device",
                "driver_unavailable",
            ),
            (
                BackendActivationError::NoCatalogMatch {
                    target: "sm_86".to_string(),
                    message: "redacted".to_string(),
                },
                "unsupported_device",
                "no_catalog_match",
            ),
            (
                BackendActivationError::Install("redacted".to_string()),
                "download",
                "install_failed",
            ),
            (
                BackendActivationError::InstalledPack("redacted".to_string()),
                "verification",
                "installed_pack_invalid",
            ),
        ];
        for (error, class, code) in cases {
            assert_eq!(error.machine_failure_class(), class);
            assert_eq!(error.machine_failure_code(), code);
        }
    }
}
