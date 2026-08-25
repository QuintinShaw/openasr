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
    CatalogBackendActivationState, CatalogBackendFileRole, CatalogBackendVendor, ModelCatalog,
    atomic_file::write_file_atomically,
    backend_device_probe::probe_provider_device,
    ggml_runtime::probe_exact_backend_plugin_candidate,
    pull::{
        BackendStoreMutationLock, InstalledBackend, PreparedBackendRuntimeObjects, PullProgress,
        backend_artifact_fingerprint, backend_pack_download_plan, backend_pack_install_dir,
        install_backend_pack, install_backend_pack_locked, installed_backend_protected_bytes,
        prepare_backend_runtime_objects_locked, read_and_verify_installed_backend,
    },
    registry::{
        is_canonical_vulkan_qualification_target, live_backend_driver_floor,
        resolve_catalog_backend_pull, resolve_compatible_catalog_backend_pull_for_driver,
    },
    short_audio_receipt::{sha256_file, sha256_hex_bytes},
};

// Schema 3 makes catalog activation state part of the neutral-host contract.
// Older hosts ignore unknown catalog fields, so reusing schema 2 would let a
// pre-qualification binary activate a newly published-inert pack.
pub const BACKEND_HOST_ABI_SCHEMA_VERSION: u32 = 3;
pub const ACTIVATED_BACKEND_SCHEMA_VERSION: u32 = 2;
pub const QUALIFICATION_BACKEND_SCHEMA_VERSION: u32 = 1;
pub const BACKEND_QUALIFICATION_SCOPE_ENV: &str = "OPENASR_BACKEND_QUALIFICATION_SCOPE";

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
    pub qualification_source_catalog_sha256: String,
    pub hardware_evidence_sha256: String,
    pub correctness_matrix_sha256: String,
    pub correctness_receipts_sha256: String,
    pub activated_at_unix_seconds: u64,
}

/// Exact optional backend selected only for an explicitly scoped qualification
/// process.  It is stored separately from `active.json`; ordinary Auto and
/// explicit runtime selection never read this record.  The caller must present
/// the original scope and exact signed-catalog digest in every child process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationBackendPack {
    pub schema_version: u32,
    pub scope_sha256: String,
    pub catalog_sha256: String,
    pub backend_id: String,
    pub vendor: CatalogBackendVendor,
    pub version: String,
    pub artifact_fingerprint: String,
    pub host_abi_fingerprint: String,
    pub device_target: String,
    pub driver_version: String,
    pub prepared_at_unix_seconds: u64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualification: Option<QualificationBackendPack>,
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
    #[error("backend pack '{backend_id}' is {state} in the signed catalog and cannot be activated")]
    NotActivated { backend_id: String, state: String },
    #[error("backend qualification scope is invalid: {0}")]
    Qualification(String),
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
            | Self::NotActivated { .. }
            | Self::Qualification(_)
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
            Self::NotActivated { .. } => "qualification_required",
            Self::Qualification(_) => "qualification_scope_invalid",
            Self::MissingDeviceProof => "device_proof_missing",
            Self::UnsupportedSchema(_) => "state_schema_unsupported",
            Self::Parse { .. } => "state_parse_failed",
            Self::Read { .. } => "state_read_failed",
            Self::Write { .. } => "state_write_failed",
        }
    }
}

pub(crate) fn require_catalog_backend_activated(
    requested: &crate::ResolvedCatalogBackendPull,
) -> Result<(), BackendActivationError> {
    if requested.activation.is_activated()
        && requested
            .activation
            .qualification_source_catalog_sha256
            .is_some()
        && requested.activation.hardware_evidence_sha256.is_some()
        && requested.activation.qualified_device_target.is_some()
        && requested.activation.qualified_driver_version.is_some()
        && requested.activation.correctness_matrix_sha256.is_some()
        && requested.activation.correctness_receipts_sha256.is_some()
    {
        return Ok(());
    }
    let state = match requested.activation.state {
        CatalogBackendActivationState::PublishedInert => "published-inert",
        CatalogBackendActivationState::Qualified => "qualified",
        CatalogBackendActivationState::Activated => "activated",
        CatalogBackendActivationState::Revoked => "revoked",
        CatalogBackendActivationState::Unknown => "unknown",
    };
    Err(BackendActivationError::NotActivated {
        backend_id: requested.backend_id.clone(),
        state: state.to_string(),
    })
}

fn qualification_scope_sha256(scope: &str) -> Result<String, BackendActivationError> {
    let scope = scope.trim();
    if !(16..=192).contains(&scope.len())
        || !scope.is_ascii()
        || !scope
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-'))
    {
        return Err(BackendActivationError::Qualification(
            "scope must be 16-192 safe ASCII characters".to_string(),
        ));
    }
    Ok(sha256_hex_bytes(scope.as_bytes()))
}

fn require_sha256(field: &str, value: &str) -> Result<(), BackendActivationError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(BackendActivationError::Qualification(format!(
            "{field} must be lowercase SHA-256"
        )))
    }
}

fn is_dotted_numeric_driver_version(value: &str) -> bool {
    (1..=64).contains(&value.len())
        && value
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

pub(crate) fn catalog_backend_accepts_device_target(
    requested: &crate::ResolvedCatalogBackendPull,
    device_target: &str,
) -> bool {
    match requested.vendor {
        CatalogBackendVendor::Cuda | CatalogBackendVendor::Hip => {
            requested.targets.as_slice() == [device_target]
        }
        CatalogBackendVendor::Vulkan => {
            requested.targets.is_empty() && is_canonical_vulkan_qualification_target(device_target)
        }
        CatalogBackendVendor::Cpu | CatalogBackendVendor::Unknown => false,
    }
}

fn qualification_device_target(
    requested: &crate::ResolvedCatalogBackendPull,
    supplied_target: Option<&str>,
) -> Result<String, BackendActivationError> {
    match requested.vendor {
        CatalogBackendVendor::Cuda | CatalogBackendVendor::Hip => {
            let [catalog_target] = requested.targets.as_slice() else {
                return Err(BackendActivationError::Qualification(
                    "CUDA/HIP qualification requires one exact target-scoped backend".to_string(),
                ));
            };
            if supplied_target.is_some_and(|target| target != catalog_target) {
                return Err(BackendActivationError::Qualification(
                    "qualification target does not match the signed backend entry".to_string(),
                ));
            }
            Ok(catalog_target.clone())
        }
        CatalogBackendVendor::Vulkan => {
            let target = supplied_target.ok_or_else(|| {
                BackendActivationError::Qualification(
                    "Vulkan qualification requires --device-target vk_caps_<vendor>_<device>_<pipeline UUID>"
                        .to_string(),
                )
            })?;
            if !catalog_backend_accepts_device_target(requested, target) {
                return Err(BackendActivationError::Qualification(
                    "Vulkan qualification target must be a canonical vk_caps capability class"
                        .to_string(),
                ));
            }
            Ok(target.to_string())
        }
        CatalogBackendVendor::Cpu | CatalogBackendVendor::Unknown => {
            Err(BackendActivationError::Qualification(
                "only optional GPU backend packs can enter qualification".to_string(),
            ))
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
    require_catalog_backend_activated(&requested)?;
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
/// every host-compatible target pack, then queries the signed HIP runtime
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
    let host_abi = BackendHostAbi::current();
    let (resolved, device_target, driver_version) = match vendor {
        CatalogBackendVendor::Cuda | CatalogBackendVendor::Hip => {
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
            (resolved, device.target, device.driver_api_version)
        }
        CatalogBackendVendor::Vulkan => {
            let requested = resolve_compatible_catalog_backend_pull_for_driver(
                catalog, vendor, &host_abi, None, None,
            )
            .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
            // A generic Vulkan artifact has no baked physical target. Product
            // preparation can therefore obtain the target only from the
            // signed Activated binding; hardware qualification supplies its
            // UUID explicitly through the isolated qualification command.
            require_catalog_backend_activated(&requested)?;
            let target = requested
                .activation
                .qualified_device_target
                .clone()
                .expect("activated catalog entry was checked above");
            install_backend_pack_locked(&requested, home, &mut *progress)
                .map_err(|error| BackendActivationError::Install(error.to_string()))?;
            let proven = prove_installed_backend_pack_locked(catalog, &requested, &target, home)?;
            if requested.activation.qualified_driver_version.as_deref()
                != Some(proven.driver_version.as_str())
            {
                return Err(BackendActivationError::Resolution(
                    "live Vulkan driver does not match the signed qualification binding"
                        .to_string(),
                ));
            }
            (proven.resolved, proven.device_target, proven.driver_version)
        }
        CatalogBackendVendor::Cpu | CatalogBackendVendor::Unknown => {
            return Err(BackendActivationError::Resolution(
                "only optional GPU providers support provider preparation".to_string(),
            ));
        }
    };
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
        device_target,
        driver_version,
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
    require_catalog_backend_activated(&requested)?;
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
    require_catalog_backend_activated(requested)?;
    let device_target = requested
        .activation
        .qualified_device_target
        .as_deref()
        .expect("activated catalog entry was checked above");
    let proven = prove_installed_backend_pack_locked(catalog, requested, device_target, home)?;
    write_activated_backend_record(home, proven)
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
        qualification: qualification_backend_from_environment(home)?,
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
    let record: ActivatedBackendPack =
        serde_json::from_str(&text).map_err(|source| BackendActivationError::Parse {
            path: path.clone(),
            source,
        })?;
    if record.schema_version != ACTIVATED_BACKEND_SCHEMA_VERSION {
        return Err(BackendActivationError::UnsupportedSchema(
            record.schema_version,
        ));
    }
    if record.device_target.trim().is_empty()
        || !is_dotted_numeric_driver_version(&record.driver_version)
    {
        return Err(BackendActivationError::MissingDeviceProof);
    }
    if [
        &record.qualification_source_catalog_sha256,
        &record.hardware_evidence_sha256,
        &record.correctness_matrix_sha256,
        &record.correctness_receipts_sha256,
    ]
    .into_iter()
    .any(|value| {
        value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err(BackendActivationError::Parse {
            path,
            source: serde_json::Error::io(io::Error::new(
                io::ErrorKind::InvalidData,
                "activation qualification bindings must be lowercase SHA-256",
            )),
        });
    }
    Ok(Some(record))
}

pub fn qualification_backend_path(
    home: &Path,
    scope: &str,
) -> Result<PathBuf, BackendActivationError> {
    Ok(home
        .join("backends")
        .join("qualification")
        .join(format!("{}.json", qualification_scope_sha256(scope)?)))
}

pub fn read_qualification_backend(
    home: &Path,
    scope: &str,
) -> Result<Option<QualificationBackendPack>, BackendActivationError> {
    let expected_scope_sha256 = qualification_scope_sha256(scope)?;
    let path = qualification_backend_path(home, scope)?;
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(BackendActivationError::Read { path, source }),
    };
    let record: QualificationBackendPack =
        serde_json::from_str(&text).map_err(|source| BackendActivationError::Parse {
            path: path.clone(),
            source,
        })?;
    if record.schema_version != QUALIFICATION_BACKEND_SCHEMA_VERSION {
        return Err(BackendActivationError::UnsupportedSchema(
            record.schema_version,
        ));
    }
    require_sha256("scope_sha256", &record.scope_sha256)?;
    require_sha256("catalog_sha256", &record.catalog_sha256)?;
    require_sha256("artifact_fingerprint", &record.artifact_fingerprint)?;
    require_sha256("host_abi_fingerprint", &record.host_abi_fingerprint)?;
    if record.scope_sha256 != expected_scope_sha256 {
        return Err(BackendActivationError::Qualification(
            "record is bound to another scope".to_string(),
        ));
    }
    if record.backend_id.trim().is_empty()
        || record.version.trim().is_empty()
        || record.device_target.trim().is_empty()
        || !is_dotted_numeric_driver_version(&record.driver_version)
    {
        return Err(BackendActivationError::Qualification(
            "record identity is incomplete".to_string(),
        ));
    }
    Ok(Some(record))
}

/// Resolve the qualification selector only when an explicit scope is present.
/// Normal product processes never inspect the qualification directory.
pub fn qualification_backend_from_environment(
    home: &Path,
) -> Result<Option<QualificationBackendPack>, BackendActivationError> {
    let Some(scope) = std::env::var(BACKEND_QUALIFICATION_SCOPE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };
    if std::env::var("OPENASR_OFFLINE").as_deref() != Ok("1") {
        return Err(BackendActivationError::Qualification(
            "qualification children must set OPENASR_OFFLINE=1".to_string(),
        ));
    }
    if read_activated_backend(home)?.is_some() {
        return Err(BackendActivationError::Qualification(
            "qualification home also contains a product activation selector".to_string(),
        ));
    }
    let record = read_qualification_backend(home, &scope)?.ok_or_else(|| {
        BackendActivationError::Qualification(
            "scoped qualification selector is missing".to_string(),
        )
    })?;
    let cached_catalog = home.join("catalog.json");
    let (_, cached_catalog_sha256) = sha256_file(&cached_catalog).map_err(|error| {
        BackendActivationError::Qualification(format!(
            "could not hash cached qualification catalog: {error}"
        ))
    })?;
    if cached_catalog_sha256 != record.catalog_sha256 {
        return Err(BackendActivationError::Qualification(
            "qualification selector is bound to another cached catalog".to_string(),
        ));
    }
    Ok(Some(record))
}

/// Install, live-probe, and write a non-product qualification selector.  The
/// record is immutable for its unique scope and cannot become `active.json`.
pub fn prepare_backend_pack_for_qualification(
    catalog: &ModelCatalog,
    backend_id: &str,
    device_target: Option<&str>,
    catalog_sha256: &str,
    scope: &str,
    home: &Path,
    progress: impl FnMut(PullProgress),
) -> Result<QualificationBackendPack, BackendActivationError> {
    require_sha256("catalog_sha256", catalog_sha256)?;
    let scope_sha256 = qualification_scope_sha256(scope)?;
    if read_activated_backend(home)?.is_some() {
        return Err(BackendActivationError::Qualification(
            "qualification requires a home without active.json".to_string(),
        ));
    }
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    if matches!(
        requested.activation.state,
        CatalogBackendActivationState::Revoked | CatalogBackendActivationState::Unknown
    ) {
        return Err(BackendActivationError::Qualification(
            "revoked or unknown backend cannot enter qualification".to_string(),
        ));
    }
    if !BackendHostAbi::current().is_compatible_with(&requested.host_abi) {
        return Err(BackendActivationError::Resolution(
            "selected backend does not match the current neutral-host ABI".to_string(),
        ));
    }
    let device_target = qualification_device_target(&requested, device_target)?;
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    install_backend_pack_locked(&requested, home, progress)
        .map_err(|error| BackendActivationError::Install(error.to_string()))?;
    let proven = prove_installed_backend_pack_locked(catalog, &requested, &device_target, home)?;
    let record = QualificationBackendPack {
        schema_version: QUALIFICATION_BACKEND_SCHEMA_VERSION,
        scope_sha256,
        catalog_sha256: catalog_sha256.to_string(),
        backend_id: proven.resolved.backend_id.clone(),
        vendor: proven.resolved.vendor,
        version: proven.resolved.version.clone(),
        artifact_fingerprint: backend_artifact_fingerprint(&proven.resolved),
        host_abi_fingerprint: proven.resolved.host_abi.fingerprint.clone(),
        device_target: proven.device_target,
        driver_version: proven.driver_version,
        prepared_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    };
    let path = qualification_backend_path(home, scope)?;
    if path.exists() {
        return Err(BackendActivationError::Qualification(
            "refusing to overwrite an existing qualification selector".to_string(),
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| BackendActivationError::Write {
            path: path.clone(),
            source,
        })?;
    }
    let mut json = serde_json::to_vec_pretty(&record).expect("qualification record serializes");
    json.push(b'\n');
    write_file_atomically(&path, &json)
        .map_err(|source| BackendActivationError::Write { path, source })?;
    Ok(record)
}

pub fn clear_backend_qualification(home: &Path, scope: &str) -> Result<(), BackendActivationError> {
    let _store_lock = BackendStoreMutationLock::acquire(home)
        .map_err(|error| BackendActivationError::Store(error.to_string()))?;
    let path = qualification_backend_path(home, scope)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(BackendActivationError::Write { path, source }),
    }
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
    let requested = resolve_catalog_backend_pull(catalog, backend_id)
        .map_err(|error| BackendActivationError::Resolution(error.to_string()))?;
    require_catalog_backend_activated(&requested)?;
    let proven = prove_installed_backend_pack_locked(catalog, &requested, device_target, home)?;
    write_activated_backend_record(home, proven)
}

struct ProvenBackendPack {
    resolved: crate::ResolvedCatalogBackendPull,
    device_target: String,
    driver_version: String,
}

fn prove_installed_backend_pack_locked(
    catalog: &ModelCatalog,
    requested: &crate::ResolvedCatalogBackendPull,
    device_target: &str,
    home: &Path,
) -> Result<ProvenBackendPack, BackendActivationError> {
    if device_target.trim().is_empty()
        || !catalog_backend_accepts_device_target(requested, device_target)
    {
        return Err(BackendActivationError::MissingDeviceProof);
    }
    let host_abi = BackendHostAbi::current();
    if !host_abi.is_compatible_with(&requested.host_abi) {
        return Err(BackendActivationError::Resolution(
            "selected backend does not match the current neutral-host ABI".to_string(),
        ));
    }
    let install_dir = backend_pack_install_dir(home, requested)
        .map_err(|error| BackendActivationError::InstalledPack(error.to_string()))?;
    let installed = read_and_verify_installed_backend(&install_dir, requested)
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
    let dependency_dirs =
        verified_backend_dependency_dirs(&requested.backend_id, &canonical_dir, &installed)?;
    let driver_version = probe_exact_backend_plugin_candidate(
        &requested.backend_id,
        requested.vendor,
        &plugin_path,
        &dependency_dirs,
        device_target,
        live_backend_driver_floor(requested.vendor, requested.min_driver_api.as_deref()),
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
    Ok(ProvenBackendPack {
        resolved,
        device_target: device_target.to_ascii_lowercase(),
        driver_version,
    })
}

fn write_activated_backend_record(
    home: &Path,
    proven: ProvenBackendPack,
) -> Result<ActivatedBackendPack, BackendActivationError> {
    require_catalog_backend_activated(&proven.resolved)?;
    if proven
        .resolved
        .activation
        .qualified_device_target
        .as_deref()
        != Some(proven.device_target.as_str())
        || proven
            .resolved
            .activation
            .qualified_driver_version
            .as_deref()
            != Some(proven.driver_version.as_str())
    {
        return Err(BackendActivationError::Resolution(
            "live target/driver do not match the signed qualification bindings".to_string(),
        ));
    }
    let resolved = proven.resolved;
    let record = ActivatedBackendPack {
        schema_version: ACTIVATED_BACKEND_SCHEMA_VERSION,
        backend_id: resolved.backend_id.clone(),
        vendor: resolved.vendor,
        version: resolved.version.clone(),
        artifact_fingerprint: backend_artifact_fingerprint(&resolved),
        host_abi_fingerprint: resolved.host_abi.fingerprint.clone(),
        device_target: proven.device_target,
        driver_version: proven.driver_version,
        qualification_source_catalog_sha256: resolved
            .activation
            .qualification_source_catalog_sha256
            .clone()
            .expect("activated catalog entry was checked above"),
        hardware_evidence_sha256: resolved
            .activation
            .hardware_evidence_sha256
            .clone()
            .expect("activated catalog entry was checked above"),
        correctness_matrix_sha256: resolved
            .activation
            .correctness_matrix_sha256
            .clone()
            .expect("activated catalog entry was checked above"),
        correctness_receipts_sha256: resolved
            .activation
            .correctness_receipts_sha256
            .clone()
            .expect("activated catalog entry was checked above"),
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

/// hipBLASLt / rocBLAS look up Tensile `.dat`/`.co` via these process env
/// vars, not via the Windows DLL search path. Leaving them unset makes
/// `libhipblaslt.dll` walk from its own module path; a `\\?\` extended
/// path from `canonicalize` then crashes at first GEMM (`0xC0000005`).
const HIPBLASLT_TENSILE_LIBPATH: &str = "HIPBLASLT_TENSILE_LIBPATH";
const ROCBLAS_TENSILE_LIBPATH: &str = "ROCBLAS_TENSILE_LIBPATH";

pub(crate) fn tensile_env_name_for_library_dir(dir: &Path) -> Option<&'static str> {
    let name = dir.file_name()?;
    let parent = dir.parent()?.file_name()?;
    if !name.eq_ignore_ascii_case("library") {
        return None;
    }
    if parent.eq_ignore_ascii_case("hipblaslt") {
        Some(HIPBLASLT_TENSILE_LIBPATH)
    } else if parent.eq_ignore_ascii_case("rocblas") {
        Some(ROCBLAS_TENSILE_LIBPATH)
    } else {
        None
    }
}

pub(crate) fn path_for_vendor_env(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = raw.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}

pub(crate) fn bind_verified_hip_kernel_libpaths(dependency_dirs: &[PathBuf]) {
    for dir in dependency_dirs {
        let Some(key) = tensile_env_name_for_library_dir(dir) else {
            continue;
        };
        set_verified_kernel_libpath(key, &path_for_vendor_env(dir));
    }
}

fn set_verified_kernel_libpath(key: &str, value: &Path) {
    #[expect(
        unsafe_code,
        reason = "bind hipBLASLt/rocBLAS Tensile search to the verified vendor tree"
    )]
    unsafe {
        std::env::set_var(key, value);
    }
    // MSVC CRT caches getenv() at startup. SetEnvironmentVariableW alone is
    // invisible to libhipblaslt / rocblas if they call getenv/_wgetenv.
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let key_wide: Vec<u16> = std::ffi::OsStr::new(key)
            .encode_wide()
            .chain(Some(0))
            .collect();
        let value_wide: Vec<u16> = value.as_os_str().encode_wide().chain(Some(0)).collect();
        unsafe extern "C" {
            fn _wputenv_s(varname: *const u16, value_string: *const u16) -> i32;
        }
        #[expect(
            unsafe_code,
            reason = "refresh MSVC CRT getenv cache so hipBLASLt sees Tensile libpath"
        )]
        let _ = unsafe { _wputenv_s(key_wide.as_ptr(), value_wide.as_ptr()) };
    }
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
    use crate::test_process_env::with_test_process_env;
    use std::ffi::OsString;

    fn resolved_with_files(
        files: Vec<crate::CatalogBackendFile>,
    ) -> crate::ResolvedCatalogBackendPull {
        crate::ResolvedCatalogBackendPull {
            backend_id: "hip-windows-gfx1100".to_string(),
            vendor: CatalogBackendVendor::Hip,
            version: "test".to_string(),
            display_name: "HIP".to_string(),
            min_cli_version: crate::current_cli_version().to_string(),
            host_abi: BackendHostAbi::current(),
            targets: vec!["gfx1100".to_string()],
            min_driver_api: Some("6.0.0".to_string()),
            activation: crate::CatalogBackendActivation::default(),
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
    fn published_inert_qualified_and_revoked_backends_cannot_activate() {
        let mut resolved = resolved_with_files(vec![backend_file(
            "ggml-hip.dll",
            'a',
            CatalogBackendFileRole::Plugin,
        )]);
        assert!(matches!(
            require_catalog_backend_activated(&resolved),
            Err(BackendActivationError::NotActivated { .. })
        ));
        resolved.activation = crate::CatalogBackendActivation {
            state: CatalogBackendActivationState::Qualified,
            qualification_source_catalog_sha256: Some("1".repeat(64)),
            hardware_evidence_sha256: Some("2".repeat(64)),
            qualified_device_target: Some("gfx1100".to_string()),
            qualified_driver_version: Some("6.0.0".to_string()),
            correctness_matrix_sha256: None,
            correctness_receipts_sha256: None,
        };
        assert!(matches!(
            require_catalog_backend_activated(&resolved),
            Err(BackendActivationError::NotActivated { .. })
        ));
        resolved.activation.state = CatalogBackendActivationState::Revoked;
        assert!(matches!(
            require_catalog_backend_activated(&resolved),
            Err(BackendActivationError::NotActivated { state, .. }) if state == "revoked"
        ));
    }

    #[test]
    fn activated_backend_requires_all_four_signed_bindings() {
        let mut resolved = resolved_with_files(vec![backend_file(
            "ggml-hip.dll",
            'a',
            CatalogBackendFileRole::Plugin,
        )]);
        resolved.activation.state = CatalogBackendActivationState::Activated;
        assert!(require_catalog_backend_activated(&resolved).is_err());
        resolved.activation.qualification_source_catalog_sha256 = Some("1".repeat(64));
        resolved.activation.hardware_evidence_sha256 = Some("2".repeat(64));
        resolved.activation.qualified_device_target = Some("gfx1100".to_string());
        resolved.activation.qualified_driver_version = Some("6.0.0".to_string());
        resolved.activation.correctness_matrix_sha256 = Some("3".repeat(64));
        resolved.activation.correctness_receipts_sha256 = Some("4".repeat(64));
        require_catalog_backend_activated(&resolved).expect("complete activated binding");
    }

    #[test]
    fn generic_vulkan_pack_accepts_only_one_canonical_capability_class() {
        let target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef";
        let mut resolved = resolved_with_files(vec![backend_file(
            "ggml-vulkan.dll",
            'a',
            CatalogBackendFileRole::Plugin,
        )]);
        resolved.backend_id = "vulkan-windows-generic".to_string();
        resolved.vendor = CatalogBackendVendor::Vulkan;
        resolved.targets.clear();
        resolved.min_driver_api = None;

        assert!(catalog_backend_accepts_device_target(&resolved, target));
        assert!(!catalog_backend_accepts_device_target(
            &resolved,
            "vk_caps_invalid"
        ));
        assert!(qualification_device_target(&resolved, None).is_err());
        assert_eq!(
            qualification_device_target(&resolved, Some(target)).unwrap(),
            target
        );
    }

    #[test]
    fn qualification_selector_is_scope_bound_and_invisible_to_ordinary_runtime() {
        let home = tempfile::tempdir().unwrap();
        let scope = "openasr-qualification/v1/0123456789abcdef";
        let catalog_bytes = b"{\"schema_version\":1,\"models\":[],\"backends\":[]}";
        fs::write(home.path().join("catalog.json"), catalog_bytes).unwrap();
        let record = QualificationBackendPack {
            schema_version: QUALIFICATION_BACKEND_SCHEMA_VERSION,
            scope_sha256: qualification_scope_sha256(scope).unwrap(),
            catalog_sha256: sha256_hex_bytes(catalog_bytes),
            backend_id: "hip-windows-gfx1100".to_string(),
            vendor: CatalogBackendVendor::Hip,
            version: "test".to_string(),
            artifact_fingerprint: "9".repeat(64),
            host_abi_fingerprint: BackendHostAbi::current().fingerprint,
            device_target: "gfx1100".to_string(),
            driver_version: "6.0.0".to_string(),
            prepared_at_unix_seconds: 1,
        };
        let path = qualification_backend_path(home.path(), scope).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();

        with_test_process_env([(BACKEND_QUALIFICATION_SCOPE_ENV, None)], || {
            assert!(
                qualification_backend_from_environment(home.path())
                    .unwrap()
                    .is_none(),
                "a persisted qualification selector is not a product activation"
            );
        });
        with_test_process_env(
            [
                (BACKEND_QUALIFICATION_SCOPE_ENV, Some(OsString::from(scope))),
                ("OPENASR_OFFLINE", Some(OsString::from("1"))),
            ],
            || {
                assert_eq!(
                    qualification_backend_from_environment(home.path()).unwrap(),
                    Some(record.clone())
                );
                fs::write(home.path().join("catalog.json"), b"tampered").unwrap();
                assert!(qualification_backend_from_environment(home.path()).is_err());
            },
        );
    }

    #[test]
    fn qualification_scope_without_offline_confinement_fails_closed() {
        let home = tempfile::tempdir().unwrap();
        let scope = "openasr-qualification/v1/0123456789abcdef";
        with_test_process_env(
            [
                (BACKEND_QUALIFICATION_SCOPE_ENV, Some(OsString::from(scope))),
                ("OPENASR_OFFLINE", None),
            ],
            || {
                assert!(matches!(
                    qualification_backend_from_environment(home.path()),
                    Err(BackendActivationError::Qualification(_))
                ));
            },
        );
    }

    #[test]
    fn legacy_active_pointer_schema_cannot_load() {
        let home = tempfile::tempdir().unwrap();
        let path = activated_backend_path(home.path());
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            r#"{"schema_version":1,"backend_id":"old","vendor":"hip","version":"old","artifact_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","host_abi_fingerprint":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","device_target":"gfx1100","driver_version":"old","activated_at_unix_seconds":1}"#,
        )
        .unwrap();
        assert!(read_activated_backend(home.path()).is_err());
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

    #[test]
    fn tensile_env_name_binds_only_hip_vendor_library_dirs() {
        let hipblaslt = PathBuf::from("pack")
            .join("vendor")
            .join("hipblaslt")
            .join("library");
        let rocblas = PathBuf::from("pack")
            .join("vendor")
            .join("rocblas")
            .join("library");
        let vendor = PathBuf::from("pack").join("vendor");
        let amdhip64 = vendor.join("amdhip64");
        assert_eq!(
            tensile_env_name_for_library_dir(&hipblaslt),
            Some(HIPBLASLT_TENSILE_LIBPATH)
        );
        assert_eq!(
            tensile_env_name_for_library_dir(&rocblas),
            Some(ROCBLAS_TENSILE_LIBPATH)
        );
        assert_eq!(tensile_env_name_for_library_dir(&vendor), None);
        assert_eq!(tensile_env_name_for_library_dir(&amdhip64), None);
    }

    #[test]
    fn vendor_env_path_strips_windows_extended_prefix() {
        assert_eq!(
            path_for_vendor_env(Path::new(r"\\?\E:\hip\vendor\hipblaslt\library")),
            PathBuf::from(r"E:\hip\vendor\hipblaslt\library")
        );
        assert_eq!(
            path_for_vendor_env(Path::new(r"\\?\UNC\server\share\library")),
            PathBuf::from(r"\\server\share\library")
        );
        assert_eq!(
            path_for_vendor_env(Path::new(r"E:\hip\vendor\hipblaslt\library")),
            PathBuf::from(r"E:\hip\vendor\hipblaslt\library")
        );
    }
}
