//! Explicit, isolated runtime for inert GPU qualification artifacts.
//!
//! This module is intentionally separate from the ordinary backend catalog and
//! activation pointer. Its only artifact authority is a
//! [`VerifiedQualificationManifest`]; it never accepts a plugin path, never
//! writes `backends/active.json`, and never produces an Auto/Explicit runtime
//! candidate.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File, OpenOptions},
    io,
    path::{Component, Path, PathBuf},
    process::Command,
};

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt as _;
#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    BackendHostAbi, DiagnosticDecodeConformanceSuite, ExecutionProvider, GgmlBackendKind,
    NativeExecutionServices, PullError, PullProgress, QualificationBinaryArtifact,
    QualificationProvider, VerifiedQualificationManifest, atomic_file,
    pull::{
        PreparedQualificationArchive, PreparedQualificationFile, file_size_and_sha256,
        prepare_qualification_release_artifacts, reject_qualification_file_links,
    },
};

pub const QUALIFICATION_ARTIFACT_PREPARATION_SCHEMA: &str =
    "openasr.qualification-artifact-preparation.v2";
pub const QUALIFICATION_BACKEND_RUNTIME_SCHEMA: &str = "openasr.qualification-backend-runtime.v2";
const QUALIFICATION_ROOT_MARKER_SCHEMA_VERSION: u32 = 1;
const QUALIFICATION_ROOT_MARKER_FILE: &str = "qualification-root.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationArtifactPreparation {
    pub schema: String,
    pub manifest_sha256: String,
    pub release_subject: String,
    pub provider: String,
    /// Target of the immutable release artifact. Vulkan uses `generic`; this
    /// field never carries a live `vk_caps_*` physical capability identity.
    pub artifact_target: String,
    pub host_abi_fingerprint: String,
    pub binary_sha256: String,
    pub binary_bundle_sha256: String,
    pub plugin_sha256: Option<String>,
    pub vendor_sha256: Vec<String>,
    pub attestation_bundle_sha256: String,
    pub attestation_verifier_version: String,
    pub attestation_verifier_sha256: String,
    pub attestation_verifications: Vec<QualificationAttestationVerification>,
    pub host_bundle_file_count: usize,
    pub vendor_file_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationAttestationVerification {
    pub file_name: String,
    pub sha256: String,
    pub verification_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationBackendRuntimeEvidence {
    pub schema: String,
    pub preparation: QualificationArtifactPreparation,
    pub backend_id: String,
    pub provider: String,
    pub artifact_target: String,
    /// Exact target observed by the verified provider module. For CUDA/HIP it
    /// equals the compiled artifact target; Vulkan derives one `vk_caps_*`
    /// identity from the signed generic plugin before exact probe/load.
    pub device_target: String,
    pub driver_api_version: Option<String>,
    pub device_name: String,
    pub device_description: String,
    pub device_kind: String,
    pub device_id: String,
    pub pci_vendor_id: u32,
    pub provider_device_count: usize,
    pub provider_device_index: usize,
    /// Shared exact-route Layer-1/Layer-2/production-shape operator evidence.
    /// It remains distinct from real-family token/transcript correctness.
    pub decode_conformance: DiagnosticDecodeConformanceSuite,
    pub ordinary_activation_pointer_written: bool,
}

impl QualificationBackendRuntimeEvidence {
    /// Validate the fresh child stdout against the exact artifact preparation
    /// completed by its parent. A JSON parse alone is not evidence.
    pub fn validate_child_result(
        &self,
        expected_preparation: &QualificationArtifactPreparation,
    ) -> Result<(), QualificationRuntimeError> {
        let bounded = |value: &str, max: usize| {
            !value.trim().is_empty() && value.len() <= max && !value.contains(['\n', '\r'])
        };
        if self.schema != QUALIFICATION_BACKEND_RUNTIME_SCHEMA
            || &self.preparation != expected_preparation
            || self.preparation.schema != QUALIFICATION_ARTIFACT_PREPARATION_SCHEMA
            || self.provider != self.preparation.provider
            || self.artifact_target != self.preparation.artifact_target
            || !bounded(&self.backend_id, 256)
            || !bounded(&self.provider, 32)
            || !bounded(&self.artifact_target, 128)
            || !bounded(&self.device_target, 128)
            || !bounded(&self.device_name, 128)
            || !bounded(&self.device_description, 256)
            || !bounded(&self.device_kind, 32)
            || !bounded(&self.device_id, 128)
            || !matches!(self.device_kind.as_str(), "discrete_gpu" | "integrated_gpu")
            || self.pci_vendor_id == 0
            || self.provider_device_count == 0
            || self.provider_device_index >= self.provider_device_count
            || self.ordinary_activation_pointer_written
            || self.decode_conformance.provider.as_str() != self.provider
            || self.decode_conformance.stable_device_id != self.device_name
        {
            return Err(QualificationRuntimeError::InvalidChildEvidence(
                "runtime identity does not match the parent-bound preparation".to_string(),
            ));
        }
        if !qualification_target_binding_is_valid(
            &self.provider,
            &self.artifact_target,
            &self.device_target,
        ) {
            return Err(QualificationRuntimeError::InvalidChildEvidence(
                "runtime device target does not match the prepared provider artifact".to_string(),
            ));
        }
        let Some(driver) = self.driver_api_version.as_deref() else {
            return Err(QualificationRuntimeError::InvalidChildEvidence(
                "qualified dynamic provider did not report a driver API version".to_string(),
            ));
        };
        if !bounded(driver, 128) {
            return Err(QualificationRuntimeError::InvalidChildEvidence(
                "driver API version is empty or unbounded".to_string(),
            ));
        }
        self.decode_conformance.validate().map_err(|error| {
            QualificationRuntimeError::InvalidChildEvidence(format!(
                "decode conformance result is invalid: {error}"
            ))
        })
    }
}

fn qualification_target_binding_is_valid(
    provider: &str,
    artifact_target: &str,
    device_target: &str,
) -> bool {
    match provider {
        "cuda" => {
            device_target == artifact_target
                && device_target.strip_prefix("sm_").is_some_and(|suffix| {
                    matches!(suffix.len(), 2 | 3)
                        && suffix.bytes().all(|byte| byte.is_ascii_digit())
                })
        }
        "hip" => {
            device_target == artifact_target
                && device_target.strip_prefix("gfx").is_some_and(|suffix| {
                    (3..=8).contains(&suffix.len())
                        && suffix
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
        }
        "vulkan" => {
            artifact_target == "generic"
                && crate::registry::is_canonical_vulkan_qualification_target(device_target)
        }
        _ => false,
    }
}

#[derive(Debug, Error)]
pub enum QualificationRuntimeError {
    #[error("backend qualification is available only on Windows release hosts")]
    UnsupportedPlatform,
    #[error("qualification home must be an absolute path without '.' or '..': '{0}'")]
    UnsafeHome(PathBuf),
    #[error("qualification home must not be the ordinary user OpenASR home: '{0}'")]
    OrdinaryHome(PathBuf),
    #[error("qualification home contains a symlink, junction, or reparse point: '{0}'")]
    LinkedHome(PathBuf),
    #[error("qualification home contains state outside its signed qualification root: '{0}'")]
    UnexpectedHomeEntry(PathBuf),
    #[error("qualification home is missing or has a mismatched root marker: {0}")]
    RootMarker(String),
    #[error("qualification host ABI does not exactly match the running executable")]
    HostAbiMismatch,
    #[error("qualification artifact preparation failed: {0}")]
    Pull(#[from] PullError),
    #[error("could not access qualification path '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not serialize qualification root marker: {0}")]
    SerializeMarker(#[source] serde_json::Error),
    #[error("could not parse qualification root marker: {0}")]
    ParseMarker(#[source] serde_json::Error),
    #[error("running executable is not the exact signed release bundle member: {0}")]
    BinaryBundle(String),
    #[error("qualification artifact integrity verification failed: {0}")]
    ArtifactIntegrity(String),
    #[error("GitHub CLI is required to verify release attestations: {0}")]
    AttestationTool(#[source] io::Error),
    #[error("GitHub attestation rejected '{subject}' (exit {status}): {message}")]
    AttestationRejected {
        subject: String,
        status: i32,
        message: String,
    },
    #[error("GitHub attestation output for '{subject}' was not a non-empty JSON array")]
    AttestationOutput { subject: String },
    #[error("qualification provider could not be loaded: {0}")]
    BackendActivation(#[source] crate::ggml_runtime::BackendPluginActivationError),
    #[error("qualification provider/device placement failed closed: {0}")]
    DevicePlacement(String),
    #[error("qualification shared decode conformance failed closed: {0}")]
    DecodeConformance(String),
    #[error("qualification child evidence failed closed: {0}")]
    InvalidChildEvidence(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationRootMarker {
    schema_version: u32,
    manifest_sha256: String,
    release_subject: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundHostFile {
    path: PathBuf,
    relative_path: String,
    sha256: String,
    size_bytes: u64,
}

struct QualificationAttestationVerifier {
    path: PathBuf,
    version: String,
    sha256: String,
    _guard: File,
}

pub(crate) struct AttestedQualificationBackend {
    manifest_sha256: String,
    provider: QualificationProvider,
    artifact_target: String,
    qualification_home: PathBuf,
    current_executable: PathBuf,
    binary: QualificationBinaryArtifact,
    prepared: crate::pull::PreparedQualificationArtifacts,
    host_files: Vec<BoundHostFile>,
    _artifact_guards: Vec<File>,
}

struct PreparedAttestedQualification {
    receipt: QualificationArtifactPreparation,
    backend: AttestedQualificationBackend,
}

/// Download, unpack, and attest all immutable release subjects without
/// loading a provider. The returned JSON-shaped value is artifact preparation
/// evidence only; it is not GPU correctness or capability approval.
pub fn prepare_backend_qualification_artifacts(
    verified: &VerifiedQualificationManifest,
    qualification_home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<QualificationArtifactPreparation, QualificationRuntimeError> {
    prepare_attested_backend_qualification(verified, qualification_home.as_ref(), progress)
        .map(|prepared| prepared.receipt)
}

/// Prepare, attest, load the provider, and run shared exact-route Layer-1/2
/// conformance in the current qualification child. This is still not real-model
/// token/transcript correctness; that requires the shared ShortAudioReceipt
/// evidence producer and exact pack/fixture/matrix bindings.
pub fn execute_backend_qualification(
    verified: &VerifiedQualificationManifest,
    qualification_home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<QualificationBackendRuntimeEvidence, QualificationRuntimeError> {
    let prepared =
        prepare_attested_backend_qualification(verified, qualification_home.as_ref(), progress)?;
    let activation =
        crate::ggml_runtime::activate_attested_qualification_backend(&prepared.backend)
            .map_err(QualificationRuntimeError::BackendActivation)?;
    let expected_provider = match prepared.backend.provider() {
        QualificationProvider::Cuda => ExecutionProvider::Cuda,
        QualificationProvider::Hip => ExecutionProvider::Hip,
        QualificationProvider::Vulkan => ExecutionProvider::Vulkan,
        QualificationProvider::Unknown => {
            return Err(QualificationRuntimeError::DevicePlacement(
                "unknown provider reached qualification execution".to_string(),
            ));
        }
    };
    let runtime = crate::ggml_runtime::ggml_runtime_info();
    let provider_devices = runtime
        .devices
        .iter()
        .filter(|device| ExecutionProvider::from_backend_name(&device.name) == expected_provider)
        .collect::<Vec<_>>();
    let device = provider_devices
        .get(activation.provider_device_index)
        .ok_or_else(|| {
            QualificationRuntimeError::DevicePlacement(format!(
                "loaded provider '{}' did not enumerate attested device index {}",
                expected_provider.as_str(),
                activation.provider_device_index
            ))
        })?;
    if !matches!(
        device.kind,
        GgmlBackendKind::Gpu | GgmlBackendKind::IntegratedGpu
    ) {
        return Err(QualificationRuntimeError::DevicePlacement(format!(
            "provider '{}' enumerated non-GPU device kind {:?}",
            expected_provider.as_str(),
            device.kind
        )));
    }
    let software_label = format!("{} {}", device.name, device.description).to_ascii_lowercase();
    if expected_provider == ExecutionProvider::Vulkan
        && ["lavapipe", "llvmpipe", "swiftshader", "software", "cpu"]
            .iter()
            .any(|marker| software_label.contains(marker))
    {
        return Err(QualificationRuntimeError::DevicePlacement(
            "software Vulkan cannot populate a physical-device qualification cell".to_string(),
        ));
    }
    let device_id = device
        .device_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            QualificationRuntimeError::DevicePlacement(
                "qualified GPU did not expose a stable physical device identity".to_string(),
            )
        })?
        .to_string();
    let pci_vendor_id = device.pci_vendor_id.ok_or_else(|| {
        QualificationRuntimeError::DevicePlacement(
            "qualified GPU did not expose a PCI vendor identity".to_string(),
        )
    })?;
    let route = crate::enumerate_compute_devices_from_ggml(&runtime.devices)
        .into_iter()
        .find(|candidate| {
            candidate.provider == expected_provider && candidate.stable_id == device.name
        })
        .map(|candidate| candidate.to_resolved_route())
        .ok_or_else(|| {
            QualificationRuntimeError::DevicePlacement(
                "qualified live device could not be resolved back to one exact route".to_string(),
            )
        })?;
    let services = NativeExecutionServices::for_local_process().map_err(|error| {
        QualificationRuntimeError::DecodeConformance(format!(
            "could not construct an isolated native service root: {error}"
        ))
    })?;
    let decode_conformance = {
        let _services =
            crate::models::native_execution_services::install_native_execution_services(&services);
        crate::run_diagnostic_decode_conformance_suite(route)
            .map_err(|error| QualificationRuntimeError::DecodeConformance(error.to_string()))?
    };
    prepared.backend.reverify_for_load()?;
    let active_pointer = qualification_home
        .as_ref()
        .join("backends")
        .join("active.json");
    if active_pointer.exists() {
        return Err(QualificationRuntimeError::UnexpectedHomeEntry(
            active_pointer,
        ));
    }
    let evidence = QualificationBackendRuntimeEvidence {
        schema: QUALIFICATION_BACKEND_RUNTIME_SCHEMA.to_string(),
        preparation: prepared.receipt,
        backend_id: activation.backend_id,
        provider: expected_provider.as_str().to_string(),
        artifact_target: prepared.backend.artifact_target().to_string(),
        device_target: activation.device_target,
        driver_api_version: activation.driver_api_version,
        device_name: device.name.clone(),
        device_description: device.description.clone(),
        device_kind: match device.kind {
            GgmlBackendKind::Gpu => "discrete_gpu",
            GgmlBackendKind::IntegratedGpu => "integrated_gpu",
            _ => unreachable!("rejected above"),
        }
        .to_string(),
        device_id,
        pci_vendor_id,
        provider_device_count: provider_devices.len(),
        provider_device_index: activation.provider_device_index,
        decode_conformance,
        ordinary_activation_pointer_written: false,
    };
    evidence.validate_child_result(&evidence.preparation)?;
    Ok(evidence)
}

fn prepare_attested_backend_qualification(
    verified: &VerifiedQualificationManifest,
    qualification_home: &Path,
    progress: impl FnMut(PullProgress),
) -> Result<PreparedAttestedQualification, QualificationRuntimeError> {
    if !cfg!(windows) {
        return Err(QualificationRuntimeError::UnsupportedPlatform);
    }
    require_exact_host_abi(verified)?;
    let qualification_home = initialize_qualification_home(verified, qualification_home)?;
    let active_pointer = qualification_home.join("backends").join("active.json");
    if active_pointer.exists() {
        return Err(QualificationRuntimeError::UnexpectedHomeEntry(
            active_pointer,
        ));
    }
    let prepared = prepare_qualification_release_artifacts(
        verified,
        &qualification_home.join("artifacts"),
        progress,
    )?;
    if prepared.artifact_root
        != qualification_home
            .join("artifacts")
            .join(verified.manifest_sha256())
    {
        return Err(QualificationRuntimeError::ArtifactIntegrity(
            "qualification artifact root escaped its manifest identity".to_string(),
        ));
    }
    verify_prepared_archive_tree(&prepared.binary_bundle)?;
    verify_prepared_file(&prepared.attestation_bundle)?;
    if let Some(plugin) = &prepared.plugin {
        verify_prepared_file(plugin)?;
    }
    for vendor in &prepared.vendor {
        verify_prepared_archive_tree(vendor)?;
    }
    let current_executable =
        env::current_exe().map_err(|source| QualificationRuntimeError::Io {
            path: PathBuf::from("<current executable>"),
            source,
        })?;
    let host_files = verify_binary_bundle_at_path(
        &prepared.binary_bundle,
        &verified.manifest().artifacts.binary,
        &current_executable,
    )?;
    let artifact_guards = lock_qualification_artifact_files(&prepared, &host_files)?;
    reject_linked_path_components(&qualification_home)?;
    verify_prepared_archive_tree(&prepared.binary_bundle)?;
    verify_prepared_file(&prepared.attestation_bundle)?;
    if let Some(plugin) = &prepared.plugin {
        verify_prepared_file(plugin)?;
    }
    for vendor in &prepared.vendor {
        verify_prepared_archive_tree(vendor)?;
    }
    for file in &host_files {
        reject_qualification_file_links(&file.path)?;
        let (size_bytes, sha256) = file_size_and_sha256(&file.path)?;
        if size_bytes != file.size_bytes || sha256 != file.sha256 {
            return Err(QualificationRuntimeError::BinaryBundle(format!(
                "bound host file '{}' changed after bundle verification",
                file.relative_path
            )));
        }
    }
    let attestation_verifier = resolve_attestation_verifier()?;
    let mut attestation_verifications = Vec::new();
    attestation_verifications.push(verify_github_attestation(
        &attestation_verifier,
        verified,
        &prepared.binary_bundle.source,
        &prepared.attestation_bundle,
    )?);
    if let Some(plugin) = &prepared.plugin {
        attestation_verifications.push(verify_github_attestation(
            &attestation_verifier,
            verified,
            plugin,
            &prepared.attestation_bundle,
        )?);
    }
    for vendor in &prepared.vendor {
        attestation_verifications.push(verify_github_attestation(
            &attestation_verifier,
            verified,
            &vendor.source,
            &prepared.attestation_bundle,
        )?);
    }
    attestation_verifications.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    reject_linked_path_components(&qualification_home)?;
    if active_pointer.exists() {
        return Err(QualificationRuntimeError::UnexpectedHomeEntry(
            active_pointer,
        ));
    }
    verify_prepared_archive_tree(&prepared.binary_bundle)?;
    verify_prepared_file(&prepared.attestation_bundle)?;
    if let Some(plugin) = &prepared.plugin {
        verify_prepared_file(plugin)?;
    }
    for vendor in &prepared.vendor {
        verify_prepared_archive_tree(vendor)?;
    }
    let final_host_files = verify_binary_bundle_at_path(
        &prepared.binary_bundle,
        &verified.manifest().artifacts.binary,
        &current_executable,
    )?;
    if final_host_files != host_files {
        return Err(QualificationRuntimeError::BinaryBundle(
            "bound host file identity changed during attestation".to_string(),
        ));
    }

    let manifest = verified.manifest();
    let receipt = QualificationArtifactPreparation {
        schema: QUALIFICATION_ARTIFACT_PREPARATION_SCHEMA.to_string(),
        manifest_sha256: verified.manifest_sha256().to_string(),
        release_subject: manifest.release_subject.clone(),
        provider: manifest.provider_target.provider.as_str().to_string(),
        artifact_target: manifest.provider_target.target.clone(),
        host_abi_fingerprint: manifest.host_abi.fingerprint.clone(),
        binary_sha256: manifest.artifacts.binary.sha256.clone(),
        binary_bundle_sha256: prepared.binary_bundle.source.sha256.clone(),
        plugin_sha256: prepared.plugin.as_ref().map(|file| file.sha256.clone()),
        vendor_sha256: prepared
            .vendor
            .iter()
            .map(|archive| archive.source.sha256.clone())
            .collect(),
        attestation_bundle_sha256: prepared.attestation_bundle.sha256.clone(),
        attestation_verifier_version: attestation_verifier.version,
        attestation_verifier_sha256: attestation_verifier.sha256,
        attestation_verifications,
        host_bundle_file_count: host_files.len(),
        vendor_file_count: prepared
            .vendor
            .iter()
            .map(|archive| archive.materialized_files.len())
            .sum(),
    };
    let backend = AttestedQualificationBackend {
        manifest_sha256: verified.manifest_sha256().to_string(),
        provider: manifest.provider_target.provider,
        artifact_target: manifest.provider_target.target.clone(),
        qualification_home,
        current_executable,
        binary: manifest.artifacts.binary.clone(),
        prepared,
        host_files,
        _artifact_guards: artifact_guards,
    };
    Ok(PreparedAttestedQualification { receipt, backend })
}

fn lock_qualification_artifact_files(
    prepared: &crate::pull::PreparedQualificationArtifacts,
    host_files: &[BoundHostFile],
) -> Result<Vec<File>, QualificationRuntimeError> {
    let mut paths = BTreeSet::new();
    paths.insert(prepared.binary_bundle.source.path.clone());
    paths.insert(prepared.attestation_bundle.path.clone());
    for file in host_files {
        paths.insert(file.path.clone());
    }
    for materialized in &prepared.binary_bundle.materialized_files {
        paths.insert(
            prepared
                .binary_bundle
                .payload_root
                .join(&materialized.relative_path),
        );
    }
    if let Some(plugin) = &prepared.plugin {
        paths.insert(plugin.path.clone());
    }
    for archive in &prepared.vendor {
        paths.insert(archive.source.path.clone());
        for materialized in &archive.materialized_files {
            paths.insert(archive.payload_root.join(&materialized.relative_path));
        }
    }
    paths
        .into_iter()
        .map(|path| {
            reject_qualification_file_links(&path)?;
            open_read_locked(&path).map_err(|source| QualificationRuntimeError::Io { path, source })
        })
        .collect()
}

fn open_read_locked(path: &Path) -> io::Result<File> {
    #[cfg(windows)]
    {
        OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(path)
    }
    #[cfg(not(windows))]
    {
        OpenOptions::new().read(true).open(path)
    }
}

impl AttestedQualificationBackend {
    pub(crate) fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub(crate) const fn provider(&self) -> QualificationProvider {
        self.provider
    }

    pub(crate) fn artifact_target(&self) -> &str {
        &self.artifact_target
    }

    pub(crate) fn plugin_path(&self) -> Option<&Path> {
        self.prepared
            .plugin
            .as_ref()
            .map(|plugin| plugin.path.as_path())
    }

    /// Exact digest of the signed optional provider image. The qualification
    /// activation cell records this identity so request receipts cannot fall
    /// back to a provider label after the attested DLL has been loaded.
    pub(crate) fn plugin_sha256(&self) -> Option<&str> {
        self.prepared
            .plugin
            .as_ref()
            .map(|plugin| plugin.sha256.as_str())
    }

    pub(crate) fn dependency_dirs(&self) -> Vec<PathBuf> {
        let mut directories = BTreeSet::new();
        for archive in &self.prepared.vendor {
            for file in &archive.materialized_files {
                let relative = Path::new(&file.relative_path);
                directories.insert(
                    archive
                        .payload_root
                        .join(relative.parent().unwrap_or_else(|| Path::new(""))),
                );
            }
        }
        directories.into_iter().collect()
    }

    pub(crate) fn reverify_for_load(&self) -> Result<(), QualificationRuntimeError> {
        reject_linked_path_components(&self.qualification_home)?;
        let active_pointer = self.qualification_home.join("backends").join("active.json");
        if active_pointer.exists() {
            return Err(QualificationRuntimeError::UnexpectedHomeEntry(
                active_pointer,
            ));
        }
        verify_prepared_archive_tree(&self.prepared.binary_bundle)?;
        verify_prepared_file(&self.prepared.attestation_bundle)?;
        if let Some(plugin) = &self.prepared.plugin {
            verify_prepared_file(plugin)?;
        }
        for vendor in &self.prepared.vendor {
            verify_prepared_archive_tree(vendor)?;
        }
        let host_files = verify_binary_bundle_at_path(
            &self.prepared.binary_bundle,
            &self.binary,
            &self.current_executable,
        )?;
        if host_files != self.host_files {
            return Err(QualificationRuntimeError::BinaryBundle(
                "bound host file identity changed before provider load".to_string(),
            ));
        }
        Ok(())
    }
}

fn require_exact_host_abi(
    verified: &VerifiedQualificationManifest,
) -> Result<(), QualificationRuntimeError> {
    let declared = &verified.manifest().host_abi;
    let current = BackendHostAbi::current();
    let exact = declared.schema_version == current.schema_version
        && declared.fingerprint == current.fingerprint
        && declared.target == current.target
        && declared.crt == current.crt
        && declared.toolchain == current.toolchain
        && declared.compile_flags_sha256 == current.compile_flags_sha256
        && declared.ggml_backend_api_version == current.ggml_backend_api_version
        && declared.ggml_revision == current.ggml_revision
        && declared.ggml_headers_sha256 == current.ggml_headers_sha256
        && declared.openasr_ffi_sha256 == current.openasr_ffi_sha256
        && declared.openasr_extension_sha256 == current.openasr_extension_sha256;
    exact
        .then_some(())
        .ok_or(QualificationRuntimeError::HostAbiMismatch)
}

fn verify_prepared_file(file: &PreparedQualificationFile) -> Result<(), QualificationRuntimeError> {
    reject_qualification_file_links(&file.path)?;
    let (size_bytes, sha256) = file_size_and_sha256(&file.path)?;
    if size_bytes != file.size_bytes {
        return Err(PullError::SizeMismatch {
            path: file.path.clone(),
            expected: file.size_bytes,
            actual: size_bytes,
        }
        .into());
    }
    if sha256 != file.sha256 {
        return Err(PullError::ShaMismatch {
            path: file.path.clone(),
            expected: file.sha256.clone(),
            actual: sha256,
        }
        .into());
    }
    Ok(())
}

fn verify_prepared_archive_tree(
    archive: &PreparedQualificationArchive,
) -> Result<(), QualificationRuntimeError> {
    verify_prepared_file(&archive.source)?;
    let mut actual = BTreeMap::new();
    collect_bound_host_files(&archive.payload_root, &archive.payload_root, &mut actual)?;
    if actual.len() != archive.materialized_files.len() {
        return Err(QualificationRuntimeError::ArtifactIntegrity(
            "qualification archive materialized file set changed after extraction".to_string(),
        ));
    }
    for expected in &archive.materialized_files {
        let key = expected.relative_path.to_lowercase();
        let Some((_path, size_bytes, sha256)) = actual.remove(&key) else {
            return Err(QualificationRuntimeError::ArtifactIntegrity(format!(
                "qualification archive is missing '{}'",
                expected.relative_path
            )));
        };
        if size_bytes != expected.size_bytes || sha256 != expected.sha256 {
            return Err(QualificationRuntimeError::ArtifactIntegrity(format!(
                "qualification archive file '{}' changed after extraction",
                expected.relative_path
            )));
        }
    }
    Ok(())
}

fn initialize_qualification_home(
    verified: &VerifiedQualificationManifest,
    home: &Path,
) -> Result<PathBuf, QualificationRuntimeError> {
    if !home.is_absolute()
        || home
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(QualificationRuntimeError::UnsafeHome(home.to_path_buf()));
    }
    reject_ordinary_home(home)?;
    fs::create_dir_all(home).map_err(|source| QualificationRuntimeError::Io {
        path: home.to_path_buf(),
        source,
    })?;
    reject_linked_path_components(home)?;
    let canonical = fs::canonicalize(home).map_err(|source| QualificationRuntimeError::Io {
        path: home.to_path_buf(),
        source,
    })?;
    let marker_path = canonical.join(QUALIFICATION_ROOT_MARKER_FILE);
    let entries = fs::read_dir(&canonical)
        .map_err(|source| QualificationRuntimeError::Io {
            path: canonical.clone(),
            source,
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| QualificationRuntimeError::Io {
            path: canonical.clone(),
            source,
        })?;
    if marker_path.is_file() {
        reject_linked_path_components(&marker_path)?;
        reject_qualification_file_links(&marker_path)?;
        let bytes = fs::read(&marker_path).map_err(|source| QualificationRuntimeError::Io {
            path: marker_path.clone(),
            source,
        })?;
        let marker: QualificationRootMarker =
            serde_json::from_slice(&bytes).map_err(QualificationRuntimeError::ParseMarker)?;
        if marker.schema_version != QUALIFICATION_ROOT_MARKER_SCHEMA_VERSION
            || marker.manifest_sha256 != verified.manifest_sha256()
            || marker.release_subject != verified.manifest().release_subject
        {
            return Err(QualificationRuntimeError::RootMarker(
                "marker identity does not match the verified manifest".to_string(),
            ));
        }
        for entry in entries {
            let name = entry.file_name();
            if name != QUALIFICATION_ROOT_MARKER_FILE && name != "artifacts" {
                return Err(QualificationRuntimeError::UnexpectedHomeEntry(entry.path()));
            }
        }
    } else {
        if !entries.is_empty() {
            return Err(QualificationRuntimeError::RootMarker(
                "a non-empty directory has no qualification root marker".to_string(),
            ));
        }
        let marker = QualificationRootMarker {
            schema_version: QUALIFICATION_ROOT_MARKER_SCHEMA_VERSION,
            manifest_sha256: verified.manifest_sha256().to_string(),
            release_subject: verified.manifest().release_subject.clone(),
        };
        let json = serde_json::to_vec_pretty(&marker)
            .map_err(QualificationRuntimeError::SerializeMarker)?;
        let mut contents = json;
        contents.push(b'\n');
        atomic_file::write_file_atomically(&marker_path, &contents).map_err(|source| {
            QualificationRuntimeError::Io {
                path: marker_path.clone(),
                source,
            }
        })?;
        reject_qualification_file_links(&marker_path)?;
    }
    Ok(canonical)
}

fn reject_ordinary_home(home: &Path) -> Result<(), QualificationRuntimeError> {
    for user_home in [env::var_os("USERPROFILE"), env::var_os("HOME")]
        .into_iter()
        .flatten()
        .filter(|value| !value.is_empty())
    {
        let ordinary = PathBuf::from(user_home).join(".openasr");
        if path_is_same_or_descendant(home, &ordinary) {
            return Err(QualificationRuntimeError::OrdinaryHome(home.to_path_buf()));
        }
    }
    Ok(())
}

fn path_is_same_or_descendant(candidate: &Path, root: &Path) -> bool {
    let candidate = fs::canonicalize(candidate).unwrap_or_else(|_| candidate.to_path_buf());
    let root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !cfg!(windows) {
        return candidate.starts_with(root);
    }
    let candidate = candidate
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let root = root
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_ascii_lowercase())
        .collect::<Vec<_>>();
    candidate.starts_with(&root)
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = fs::canonicalize(left).unwrap_or_else(|_| left.to_path_buf());
    let right = fs::canonicalize(right).unwrap_or_else(|_| right.to_path_buf());
    if cfg!(windows) {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    } else {
        left == right
    }
}

fn reject_linked_path_components(path: &Path) -> Result<(), QualificationRuntimeError> {
    for component in path.ancestors().collect::<Vec<_>>().into_iter().rev() {
        let metadata = match fs::symlink_metadata(component) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(QualificationRuntimeError::Io {
                    path: component.to_path_buf(),
                    source,
                });
            }
        };
        if metadata.file_type().is_symlink() || is_windows_reparse_point(&metadata) {
            return Err(QualificationRuntimeError::LinkedHome(
                component.to_path_buf(),
            ));
        }
    }
    Ok(())
}

#[cfg(windows)]
fn is_windows_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_windows_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

fn verify_binary_bundle_at_path(
    bundle: &PreparedQualificationArchive,
    binary: &QualificationBinaryArtifact,
    current_executable: &Path,
) -> Result<Vec<BoundHostFile>, QualificationRuntimeError> {
    let current_executable =
        fs::canonicalize(current_executable).map_err(|source| QualificationRuntimeError::Io {
            path: current_executable.to_path_buf(),
            source,
        })?;
    reject_qualification_file_links(&current_executable)?;
    let current_root = current_executable.parent().ok_or_else(|| {
        QualificationRuntimeError::BinaryBundle(
            "running executable has no containing directory".to_string(),
        )
    })?;
    let binary_members = bundle
        .materialized_files
        .iter()
        .filter(|file| {
            Path::new(&file.relative_path)
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case(&binary.file_name))
        })
        .collect::<Vec<_>>();
    let [binary_member] = binary_members.as_slice() else {
        return Err(QualificationRuntimeError::BinaryBundle(
            "release archive must contain exactly one signed executable member".to_string(),
        ));
    };
    if binary_member.size_bytes != binary.size_bytes || binary_member.sha256 != binary.sha256 {
        return Err(QualificationRuntimeError::BinaryBundle(
            "executable member identity differs from artifacts.binary".to_string(),
        ));
    }
    let archive_prefix = Path::new(&binary_member.relative_path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let mut expected = BTreeMap::new();
    for file in &bundle.materialized_files {
        let relative = Path::new(&file.relative_path)
            .strip_prefix(archive_prefix)
            .map_err(|_| {
                QualificationRuntimeError::BinaryBundle(
                    "release archive contains files outside the executable bundle root".to_string(),
                )
            })?;
        if relative.as_os_str().is_empty() {
            return Err(QualificationRuntimeError::BinaryBundle(
                "release archive contains an empty bundle-relative path".to_string(),
            ));
        }
        let relative = relative.to_str().ok_or_else(|| {
            QualificationRuntimeError::BinaryBundle(
                "release archive contains a non-UTF-8 path".to_string(),
            )
        })?;
        if expected
            .insert(
                relative.to_lowercase(),
                (relative.to_string(), file.size_bytes, file.sha256.clone()),
            )
            .is_some()
        {
            return Err(QualificationRuntimeError::BinaryBundle(
                "release archive contains case-colliding paths".to_string(),
            ));
        }
    }
    let mut actual = BTreeMap::new();
    collect_bound_host_files(current_root, current_root, &mut actual)?;
    if actual.len() != expected.len() || actual.keys().ne(expected.keys()) {
        return Err(QualificationRuntimeError::BinaryBundle(
            "extracted host file set differs from the signed release archive".to_string(),
        ));
    }
    let mut bound = Vec::with_capacity(expected.len());
    for (key, (relative_path, expected_size, expected_sha256)) in expected {
        let (path, actual_size, actual_sha256) = actual.remove(&key).expect("key sets matched");
        if actual_size != expected_size || actual_sha256 != expected_sha256 {
            return Err(QualificationRuntimeError::BinaryBundle(format!(
                "extracted host file '{relative_path}' differs from the signed release archive"
            )));
        }
        bound.push(BoundHostFile {
            path,
            relative_path,
            sha256: actual_sha256,
            size_bytes: actual_size,
        });
    }
    let expected_executable = current_root.join(&binary.file_name);
    if !paths_equivalent(&current_executable, &expected_executable) {
        return Err(QualificationRuntimeError::BinaryBundle(
            "running executable is not the bundle-root executable".to_string(),
        ));
    }
    Ok(bound)
}

fn collect_bound_host_files(
    root: &Path,
    directory: &Path,
    files: &mut BTreeMap<String, (PathBuf, u64, String)>,
) -> Result<(), QualificationRuntimeError> {
    let entries = fs::read_dir(directory).map_err(|source| QualificationRuntimeError::Io {
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| QualificationRuntimeError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| QualificationRuntimeError::Io {
                path: path.clone(),
                source,
            })?;
        if metadata.file_type().is_symlink() || is_windows_reparse_point(&metadata) {
            return Err(QualificationRuntimeError::BinaryBundle(format!(
                "bundle path '{}' is a link or reparse point",
                path.display()
            )));
        }
        if metadata.is_dir() {
            collect_bound_host_files(root, &path, files)?;
            continue;
        }
        if !metadata.is_file() {
            return Err(QualificationRuntimeError::BinaryBundle(format!(
                "bundle path '{}' is not a regular file",
                path.display()
            )));
        }
        reject_qualification_file_links(&path)?;
        let relative = path.strip_prefix(root).map_err(|_| {
            QualificationRuntimeError::BinaryBundle(
                "bundle file escaped the running executable directory".to_string(),
            )
        })?;
        let relative = relative.to_str().ok_or_else(|| {
            QualificationRuntimeError::BinaryBundle(
                "running bundle contains a non-UTF-8 path".to_string(),
            )
        })?;
        let (size_bytes, sha256) = file_size_and_sha256(&path)?;
        if files
            .insert(relative.to_lowercase(), (path, size_bytes, sha256))
            .is_some()
        {
            return Err(QualificationRuntimeError::BinaryBundle(
                "running bundle contains case-colliding paths".to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_attestation_verifier()
-> Result<QualificationAttestationVerifier, QualificationRuntimeError> {
    let executable_name = if cfg!(windows) { "gh.exe" } else { "gh" };
    let path = env::var_os("PATH")
        .into_iter()
        .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .map(|directory| directory.join(executable_name))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            QualificationRuntimeError::AttestationTool(io::Error::new(
                io::ErrorKind::NotFound,
                "gh executable was not found on PATH",
            ))
        })?;
    let path = fs::canonicalize(&path).map_err(QualificationRuntimeError::AttestationTool)?;
    reject_qualification_file_links(&path)?;
    let guard = open_read_locked(&path).map_err(QualificationRuntimeError::AttestationTool)?;
    let (_size_bytes, sha256) = file_size_and_sha256(&path)?;
    let output = Command::new(&path)
        .arg("version")
        .output()
        .map_err(QualificationRuntimeError::AttestationTool)?;
    if !output.status.success() {
        return Err(QualificationRuntimeError::AttestationRejected {
            subject: "gh version".to_string(),
            status: output.status.code().unwrap_or(-1),
            message: String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(2_048)
                .collect(),
        });
    }
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| line.starts_with("gh version ") && line.len() <= 256)
        .ok_or_else(|| QualificationRuntimeError::AttestationOutput {
            subject: "gh version".to_string(),
        })?
        .to_string();
    Ok(QualificationAttestationVerifier {
        path,
        version,
        sha256,
        _guard: guard,
    })
}

fn verify_github_attestation(
    verifier: &QualificationAttestationVerifier,
    verified: &VerifiedQualificationManifest,
    subject: &PreparedQualificationFile,
    bundle: &PreparedQualificationFile,
) -> Result<QualificationAttestationVerification, QualificationRuntimeError> {
    let manifest = verified.manifest();
    let output = Command::new(&verifier.path)
        .args(["attestation", "verify"])
        .arg(&subject.path)
        .args(["--repo", &manifest.attestation.repository])
        .args(["--signer-workflow", &manifest.attestation.signer_workflow])
        .args(["--source-digest", &manifest.attestation.source_digest])
        .args(["--predicate-type", &manifest.attestation.predicate_type])
        .arg("--deny-self-hosted-runners")
        .arg("--bundle")
        .arg(&bundle.path)
        .args(["--format", "json"])
        .output()
        .map_err(QualificationRuntimeError::AttestationTool)?;
    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(2_048)
            .collect::<String>();
        return Err(QualificationRuntimeError::AttestationRejected {
            subject: subject
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<subject>")
                .to_string(),
            status: output.status.code().unwrap_or(-1),
            message,
        });
    }
    let parsed = serde_json::from_slice::<serde_json::Value>(&output.stdout).map_err(|_| {
        QualificationRuntimeError::AttestationOutput {
            subject: subject
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<subject>")
                .to_string(),
        }
    })?;
    if parsed.as_array().is_none_or(|records| records.is_empty()) {
        return Err(QualificationRuntimeError::AttestationOutput {
            subject: subject
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<subject>")
                .to_string(),
        });
    }
    Ok(QualificationAttestationVerification {
        file_name: subject
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<subject>")
            .to_string(),
        sha256: subject.sha256.clone(),
        verification_sha256: format!("{:x}", Sha256::digest(&output.stdout)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pull::InstalledBackendMaterializedFile;

    fn prepared_bundle(root: &Path, files: &[(&str, &[u8])]) -> PreparedQualificationArchive {
        let payload_root = root.join("payload");
        fs::create_dir_all(&payload_root).unwrap();
        let materialized_files = files
            .iter()
            .map(|(relative, bytes)| {
                let path = payload_root.join(relative);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, bytes).unwrap();
                InstalledBackendMaterializedFile {
                    relative_path: relative.to_string(),
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                    size_bytes: bytes.len() as u64,
                }
            })
            .collect();
        let source_path = root.join("host.zip");
        fs::write(&source_path, b"zip").unwrap();
        PreparedQualificationArchive {
            source: PreparedQualificationFile {
                path: source_path,
                size_bytes: 3,
                sha256: format!("{:x}", Sha256::digest(b"zip")),
            },
            payload_root,
            materialized_files,
        }
    }

    #[test]
    fn binary_bundle_comparison_binds_executable_and_every_companion_file() {
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("archive");
        fs::create_dir(&archive).unwrap();
        let bundle = prepared_bundle(
            &archive,
            &[
                ("release/openasr.exe", b"host"),
                ("release/ggml.dll", b"runtime"),
            ],
        );
        let extracted = temp.path().join("extracted");
        fs::create_dir(&extracted).unwrap();
        fs::write(extracted.join("openasr.exe"), b"host").unwrap();
        fs::write(extracted.join("ggml.dll"), b"runtime").unwrap();
        let binary = QualificationBinaryArtifact {
            file_name: "openasr.exe".to_string(),
            sha256: format!("{:x}", Sha256::digest(b"host")),
            size_bytes: 4,
            bundle: crate::QualificationArtifact {
                file_name: "host.zip".to_string(),
                format: crate::QualificationArtifactFormat::ZipArchive,
                sha256: "0".repeat(64),
                size_bytes: 1,
                unpacked_size_bytes: Some(11),
                unpacked_tree_sha256: Some("1".repeat(64)),
                urls: vec!["https://example.invalid/host.zip".to_string()],
            },
        };

        let bound =
            verify_binary_bundle_at_path(&bundle, &binary, &extracted.join("openasr.exe")).unwrap();
        assert_eq!(bound.len(), 2);

        fs::write(extracted.join("ggml.dll"), b"changed").unwrap();
        assert!(matches!(
            verify_binary_bundle_at_path(&bundle, &binary, &extracted.join("openasr.exe"),),
            Err(QualificationRuntimeError::BinaryBundle(_))
        ));
    }

    fn sample_preparation() -> QualificationArtifactPreparation {
        QualificationArtifactPreparation {
            schema: QUALIFICATION_ARTIFACT_PREPARATION_SCHEMA.to_string(),
            manifest_sha256: "1".repeat(64),
            release_subject: "v0.1.36-test".to_string(),
            provider: "cuda".to_string(),
            artifact_target: "sm_89".to_string(),
            host_abi_fingerprint: "2".repeat(64),
            binary_sha256: "3".repeat(64),
            binary_bundle_sha256: "4".repeat(64),
            plugin_sha256: None,
            vendor_sha256: Vec::new(),
            attestation_bundle_sha256: "5".repeat(64),
            attestation_verifier_version: "gh version test".to_string(),
            attestation_verifier_sha256: "6".repeat(64),
            attestation_verifications: vec![QualificationAttestationVerification {
                file_name: "openasr-test".to_string(),
                sha256: "7".repeat(64),
                verification_sha256: "8".repeat(64),
            }],
            host_bundle_file_count: 1,
            vendor_file_count: 0,
        }
    }

    #[test]
    fn qualification_child_evidence_is_typed_parent_bound_and_revalidated() {
        let preparation = sample_preparation();
        let evidence = QualificationBackendRuntimeEvidence {
            schema: QUALIFICATION_BACKEND_RUNTIME_SCHEMA.to_string(),
            preparation: preparation.clone(),
            backend_id: "builtin-cpu-test".to_string(),
            provider: "cuda".to_string(),
            artifact_target: "sm_89".to_string(),
            device_target: "sm_89".to_string(),
            driver_api_version: Some("12.8".to_string()),
            device_name: "CPU".to_string(),
            device_description: "test CPU".to_string(),
            device_kind: "discrete_gpu".to_string(),
            device_id: "test-device".to_string(),
            pci_vendor_id: 0x8086,
            provider_device_count: 1,
            provider_device_index: 0,
            decode_conformance: crate::run_diagnostic_decode_conformance_suite(
                crate::ResolvedExecutionRoute::cpu(),
            )
            .expect("CPU conformance fixture"),
            ordinary_activation_pointer_written: false,
        };
        assert!(qualification_target_binding_is_valid(
            "cuda", "sm_89", "sm_89"
        ));
        let json = serde_json::to_string(&evidence).expect("serialize child evidence");
        let decoded: QualificationBackendRuntimeEvidence =
            serde_json::from_str(&json).expect("strict child evidence round-trip");
        assert_eq!(decoded, evidence);

        let mut unknown = serde_json::to_value(&evidence).expect("evidence value");
        unknown
            .as_object_mut()
            .expect("evidence object")
            .insert("activation_mode".to_string(), serde_json::json!("auto"));
        assert!(serde_json::from_value::<QualificationBackendRuntimeEvidence>(unknown).is_err());

        let mut tampered = evidence.clone();
        tampered.decode_conformance.result = "pass-ish".to_string();
        assert!(tampered.validate_child_result(&preparation).is_err());
        let mut wrong_parent = preparation.clone();
        wrong_parent.binary_sha256 = "9".repeat(64);
        assert!(evidence.validate_child_result(&wrong_parent).is_err());
    }

    #[test]
    fn qualification_target_binding_separates_generic_vulkan_artifact_from_live_device() {
        let vk_caps = "vk_caps_00001002_0000744c_00112233445566778899aabbccddeeff";
        assert!(qualification_target_binding_is_valid(
            "vulkan", "generic", vk_caps
        ));
        assert!(!qualification_target_binding_is_valid(
            "vulkan", vk_caps, vk_caps
        ));
        assert!(!qualification_target_binding_is_valid(
            "vulkan", "generic", "generic"
        ));
        assert!(!qualification_target_binding_is_valid(
            "cuda", "sm_89", "sm_90"
        ));
        assert!(!qualification_target_binding_is_valid(
            "hip", "gfx1200", "gfx1100"
        ));
    }
}
