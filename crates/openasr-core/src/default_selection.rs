//! Single authority for resolving and persisting the "default model" state,
//! which spans two files under the OpenASR home: `config.json`'s
//! `default_model` field (the user's explicit choice) and `default.json` (a
//! pointer recording the last pack a default write installed). Before this
//! module existed, the server, the CLI, and the config layer each carried
//! their own reading of these two files, and only the server's read the
//! `default.json` pointer as a fallback -- see `docs/default-model-resolution.md`
//! for the contract this module now owns for every caller (server routes,
//! CLI serve/transcribe pack lookup, and eventually the desktop shell).
//!
//! Fail-closed by design: `resolve` never invents a default when nothing is
//! configured, and never substitutes a different installed pack when the
//! configured one is missing. Silently picking "some" installed model would
//! defeat the point of a "default" (the user chose *this* model) and could
//! route audio through an unexpected model/quant.

use std::{
    cell::Cell,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, OnceLock},
};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    CatalogError, CatalogPullRequest, ConfigError, InstalledPack, LaunchPackRequest, ModelCatalog,
    PullError, QuantPreference, canonical_quant_tag, default_pack_pointer_path,
    host_quant_recommendation_profile, list_installed_packs, load_config_document,
    load_embedded_signed_catalog, load_model_catalog, parse_model_ref,
    persist_default_pack_pointer, read_default_pack_pointer, resolve_catalog_pull_with_profile,
    resolve_launch_pack,
};

/// The schema version for the durable active-model selection record.
pub const ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION: u32 = 2;
/// Compatibility spelling for callers that refer to the record as default selection.
pub const DEFAULT_SELECTION_V2_SCHEMA_VERSION: u32 = ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION;
const ACTIVE_MODEL_SELECTION_V2_FILE_NAME: &str = "default-selection.json";
static DEFAULT_SELECTION_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
const SELECTION_LOCK_FILE_NAME: &str = ".openasr-default-selection.lock";

fn selection_write_lock() -> MutexGuard<'static, ()> {
    DEFAULT_SELECTION_WRITE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

thread_local! {
    static PERSIST_COMMIT_FAILPOINT: Cell<bool> = const { Cell::new(false) };
}

/// Test-only failpoint for activation persist. Production never enables this.
#[doc(hidden)]
pub fn set_persist_commit_failpoint_for_test(enabled: bool) {
    PERSIST_COMMIT_FAILPOINT.with(|flag| flag.set(enabled));
}

fn persist_commit_failpoint_enabled() -> bool {
    PERSIST_COMMIT_FAILPOINT.with(Cell::get)
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultSelectionWriteFault {
    BeforeStagingWrite,
    BeforeStagingSync,
    BeforeAtomicReplace,
    AfterAtomicReplace,
}

#[cfg(test)]
static RECOVERY_FAIL_AFTER_EVIDENCE: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn set_recovery_fail_after_evidence(enabled: bool) {
    RECOVERY_FAIL_AFTER_EVIDENCE.store(enabled, Ordering::SeqCst);
}

fn recovery_fail_after_evidence() -> bool {
    #[cfg(test)]
    return RECOVERY_FAIL_AFTER_EVIDENCE.load(Ordering::SeqCst);
    #[cfg(not(test))]
    false
}

struct SelectionFileLock {
    file: File,
}

impl SelectionFileLock {
    fn acquire(home: &Path) -> Result<Self, DefaultSelectionError> {
        fs::create_dir_all(home).map_err(|source| {
            DefaultSelectionError::Pull(PullError::Io {
                path: home.to_path_buf(),
                source,
            })
        })?;
        let path = home.join(SELECTION_LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|source| DefaultSelectionError::Pull(PullError::Io { path, source }))?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
                return Err(DefaultSelectionError::Pull(PullError::Io {
                    path: home.join(SELECTION_LOCK_FILE_NAME),
                    source: std::io::Error::last_os_error(),
                }));
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LockFileEx};
            let mut overlapped = unsafe { std::mem::zeroed() };
            if unsafe {
                LockFileEx(
                    file.as_raw_handle() as _,
                    LOCKFILE_EXCLUSIVE_LOCK,
                    0,
                    u32::MAX,
                    u32::MAX,
                    &mut overlapped,
                )
            } == 0
            {
                return Err(DefaultSelectionError::Pull(PullError::Io {
                    path: home.join(SELECTION_LOCK_FILE_NAME),
                    source: std::io::Error::last_os_error(),
                }));
            }
        }
        Ok(Self { file })
    }
}

impl Drop for SelectionFileLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;
            let mut overlapped = unsafe { std::mem::zeroed() };
            let _ = unsafe {
                UnlockFileEx(
                    self.file.as_raw_handle() as _,
                    0,
                    u32::MAX,
                    u32::MAX,
                    &mut overlapped,
                )
            };
        }
    }
}

/// The semantic state persisted by [`ActiveModelSelectionV2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveModelSelectionStatus {
    Installed,
    NotInstalled,
    Unset,
}

/// The content identity used by V2. It deliberately contains no local path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedPackIdentityV2 {
    pub sha256: String,
    pub size_bytes: u64,
}

/// Versioned, self-checksummed durable model selection.
///
/// This record is the sole V2 authority. `config.json` and `default.json` are
/// written only as compatibility projections for older clients. All strings
/// describe logical identity; no local filesystem path is part of the record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveModelSelectionV2 {
    pub schema_version: u32,
    pub selection_generation: u64,
    pub status: ActiveModelSelectionStatus,
    #[serde(default)]
    pub pull: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub architecture_id: Option<String>,
    #[serde(default)]
    pub expected_pack: Option<ExpectedPackIdentityV2>,
    #[serde(default)]
    pub quant_preference: QuantPreference,
    /// Stable execution-intent wire value (for example `auto` or `cpu_only`).
    pub execution_intent: String,
    /// SHA-256 of the record with this field set to the empty string.
    pub checksum: String,
}

fn is_logical_atom(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains("..")
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn is_logical_pull(value: &str) -> bool {
    let mut parts = value.split(':');
    parts.next().is_some_and(is_logical_atom)
        && parts.next().is_some_and(is_logical_atom)
        && parts.next().is_none()
}

fn is_logical_intent(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('/')
        && !value.contains('\\')
        && value.split(':').all(is_logical_atom)
}

fn encode_intent_atom(value: &str) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn decode_intent_atom(value: &str) -> Result<String, String> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return Err("encoded execution-intent atom has an invalid length".to_string());
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digits = std::str::from_utf8(pair)
            .map_err(|_| "encoded execution-intent atom is not ASCII".to_string())?;
        bytes.push(
            u8::from_str_radix(digits, 16)
                .map_err(|_| "encoded execution-intent atom is not hexadecimal".to_string())?,
        );
    }
    String::from_utf8(bytes)
        .map_err(|_| "encoded execution-intent atom is not valid UTF-8".to_string())
}

fn execution_provider_from_wire(
    value: &str,
) -> Result<crate::device::execution_route::ExecutionProvider, String> {
    use crate::device::execution_route::ExecutionProvider;
    match value {
        "cpu" => Ok(ExecutionProvider::Cpu),
        "metal" => Ok(ExecutionProvider::Metal),
        "cuda" => Ok(ExecutionProvider::Cuda),
        "hip" => Ok(ExecutionProvider::Hip),
        "vulkan" => Ok(ExecutionProvider::Vulkan),
        "accelerator" => Ok(ExecutionProvider::Accelerator),
        "unknown" => Ok(ExecutionProvider::Unknown),
        _ => Err(format!("unknown persisted execution provider {value}")),
    }
}

/// Stable, path-free wire form stored in [`ActiveModelSelectionV2`]. Exact
/// selectors hex-encode provider-local identifiers so arbitrary device names
/// cannot violate the record's logical-value grammar.
pub fn execution_intent_to_v2_wire(
    intent: &crate::device::execution_policy::ExecutionIntent,
) -> String {
    use crate::device::{
        execution_policy::{AcceleratedDeviceConstraint, ExecutionIntent},
        execution_route::{ExactDeviceSelector, ExecutionHardwareVendor},
    };
    match intent {
        ExecutionIntent::Auto => "auto".to_string(),
        ExecutionIntent::CpuOnly => "cpu_only".to_string(),
        ExecutionIntent::AcceleratedOnly => "accelerated_only".to_string(),
        ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
            provider,
        )) => format!("provider:{}", provider.as_str()),
        ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::HardwareVendor(vendor),
        ) => format!(
            "vendor:{}",
            match vendor {
                ExecutionHardwareVendor::Apple => "apple",
                ExecutionHardwareVendor::Nvidia => "nvidia",
                ExecutionHardwareVendor::Amd => "amd",
                ExecutionHardwareVendor::Intel => "intel",
            }
        ),
        ExecutionIntent::Exact(ExactDeviceSelector::PhysicalKey(key)) => {
            format!("exact_physical:{}", encode_intent_atom(key.as_str()))
        }
        ExecutionIntent::Exact(ExactDeviceSelector::StableId {
            provider,
            stable_id,
        }) => format!(
            "exact_stable:{}:{}",
            provider.map_or("any", |provider| provider.as_str()),
            encode_intent_atom(stable_id)
        ),
    }
}

pub fn execution_intent_from_v2_wire(
    value: &str,
) -> Result<crate::device::execution_policy::ExecutionIntent, String> {
    use crate::device::{
        execution_policy::{AcceleratedDeviceConstraint, ExecutionIntent},
        execution_route::{ExactDeviceSelector, ExecutionHardwareVendor, PhysicalResourceKey},
    };
    match value {
        "auto" => return Ok(ExecutionIntent::Auto),
        "cpu_only" => return Ok(ExecutionIntent::CpuOnly),
        "accelerated_only" => return Ok(ExecutionIntent::AcceleratedOnly),
        _ => {}
    }
    if let Some(provider) = value.strip_prefix("provider:") {
        return Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::Provider(execution_provider_from_wire(provider)?),
        ));
    }
    if let Some(vendor) = value.strip_prefix("vendor:") {
        let vendor = match vendor {
            "apple" => ExecutionHardwareVendor::Apple,
            "nvidia" => ExecutionHardwareVendor::Nvidia,
            "amd" => ExecutionHardwareVendor::Amd,
            "intel" => ExecutionHardwareVendor::Intel,
            _ => return Err(format!("unknown persisted execution vendor {vendor}")),
        };
        return Ok(ExecutionIntent::ConstrainedAcceleratedOnly(
            AcceleratedDeviceConstraint::HardwareVendor(vendor),
        ));
    }
    if let Some(encoded) = value.strip_prefix("exact_physical:") {
        let decoded = decode_intent_atom(encoded)?;
        let key = PhysicalResourceKey::new(decoded)
            .ok_or_else(|| "persisted exact physical key is empty".to_string())?;
        return Ok(ExecutionIntent::Exact(ExactDeviceSelector::PhysicalKey(
            key,
        )));
    }
    if let Some(rest) = value.strip_prefix("exact_stable:") {
        let (provider, encoded) = rest
            .split_once(':')
            .ok_or_else(|| "persisted exact stable selector is malformed".to_string())?;
        let provider = if provider == "any" {
            None
        } else {
            Some(execution_provider_from_wire(provider)?)
        };
        return Ok(ExecutionIntent::Exact(ExactDeviceSelector::StableId {
            provider,
            stable_id: decode_intent_atom(encoded)?,
        }));
    }
    Err(format!("unknown persisted execution intent {value}"))
}

impl ActiveModelSelectionV2 {
    fn checksum_payload(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut unsigned = self.clone();
        unsigned.checksum.clear();
        serde_json::to_vec(&unsigned)
    }

    pub fn checksum_for_record(&self) -> Result<String, serde_json::Error> {
        let bytes = self.checksum_payload()?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn with_checksum(mut self) -> Result<Self, serde_json::Error> {
        self.checksum = self.checksum_for_record()?;
        Ok(self)
    }

    fn validate(&self, path: &Path) -> Result<(), DefaultSelectionError> {
        if self.schema_version != ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION {
            return Err(DefaultSelectionError::Corrupt {
                path: path.to_path_buf(),
                reason: format!("unsupported schema_version {}", self.schema_version),
            });
        }
        if !is_logical_intent(&self.execution_intent) {
            return Err(DefaultSelectionError::Corrupt {
                path: path.to_path_buf(),
                reason: "execution_intent must be a non-empty logical value".to_string(),
            });
        }
        let expected =
            self.checksum_for_record()
                .map_err(|source| DefaultSelectionError::Corrupt {
                    path: path.to_path_buf(),
                    reason: format!("cannot calculate checksum: {source}"),
                })?;
        if self.checksum != expected {
            return Err(DefaultSelectionError::Corrupt {
                path: path.to_path_buf(),
                reason: "checksum mismatch".to_string(),
            });
        }
        match self.status {
            ActiveModelSelectionStatus::Unset => {
                if self.pull.is_some()
                    || self.model_id.is_some()
                    || self.quant.is_some()
                    || self.architecture_id.is_some()
                    || self.expected_pack.is_some()
                {
                    return Err(DefaultSelectionError::Corrupt {
                        path: path.to_path_buf(),
                        reason: "unset record contains a logical selection".to_string(),
                    });
                }
            }
            ActiveModelSelectionStatus::Installed | ActiveModelSelectionStatus::NotInstalled => {
                let valid_model = self
                    .model_id
                    .as_deref()
                    .is_some_and(|value| !value.is_empty());
                let valid_installed_identity =
                    self.quant.as_deref().is_some_and(|value| !value.is_empty())
                        && self.pull.as_deref().is_some_and(|value| !value.is_empty())
                        && self.expected_pack.as_ref().is_some_and(|identity| {
                            identity.sha256.len() == 64
                                && identity.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                        });
                if !valid_model
                    || (self.status == ActiveModelSelectionStatus::Installed
                        && !valid_installed_identity)
                {
                    return Err(DefaultSelectionError::Corrupt {
                        path: path.to_path_buf(),
                        reason: "selected record is missing logical pack identity".to_string(),
                    });
                }
                if let Some(architecture_id) = self.architecture_id.as_deref()
                    && !is_logical_atom(architecture_id)
                {
                    return Err(DefaultSelectionError::Corrupt {
                        path: path.to_path_buf(),
                        reason: "architecture_id must be a non-empty logical value".to_string(),
                    });
                }
                for (name, value, valid) in [
                    (
                        "model_id",
                        self.model_id.as_deref().unwrap_or_default(),
                        is_logical_atom(self.model_id.as_deref().unwrap_or_default()),
                    ),
                    (
                        "quant",
                        self.quant.as_deref().unwrap_or_default(),
                        is_logical_atom(self.quant.as_deref().unwrap_or_default()),
                    ),
                    (
                        "pull",
                        self.pull.as_deref().unwrap_or_default(),
                        is_logical_pull(self.pull.as_deref().unwrap_or_default())
                            || (self.status == ActiveModelSelectionStatus::NotInstalled
                                && is_logical_atom(self.pull.as_deref().unwrap_or_default())),
                    ),
                ] {
                    if !(valid
                        || (self.status == ActiveModelSelectionStatus::NotInstalled
                            && value.is_empty()))
                    {
                        return Err(DefaultSelectionError::Corrupt {
                            path: path.to_path_buf(),
                            reason: format!("{name} must not contain an absolute path"),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn active_model_selection_v2_path(home: impl AsRef<Path>) -> PathBuf {
    home.as_ref().join(ACTIVE_MODEL_SELECTION_V2_FILE_NAME)
}

pub fn default_selection_v2_path(home: impl AsRef<Path>) -> PathBuf {
    active_model_selection_v2_path(home)
}

#[derive(Debug, Error)]
pub enum DefaultSelectionError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Pull(#[from] PullError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("corrupt active model selection at {path}: {reason}")]
    Corrupt { path: PathBuf, reason: String },
    #[error("default selection was not committed: {reason}")]
    NotCommitted { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultSelectionCommitOutcome {
    NotCommitted { reason: String },
    V2Committed,
    V2CommittedProjectionFailed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultSelectionRecoveryOutcome {
    Committed {
        record: ActiveModelSelectionV2,
        evidence_path: PathBuf,
    },
    ProjectionFailed {
        record: ActiveModelSelectionV2,
        evidence_path: PathBuf,
        reason: String,
    },
}

impl DefaultSelectionRecoveryOutcome {
    pub fn record(&self) -> &ActiveModelSelectionV2 {
        match self {
            Self::Committed { record, .. } | Self::ProjectionFailed { record, .. } => record,
        }
    }

    pub fn into_record(self) -> ActiveModelSelectionV2 {
        match self {
            Self::Committed { record, .. } | Self::ProjectionFailed { record, .. } => record,
        }
    }
}

/// actually installed on disk. A bare `Option<InstalledPack>` cannot tell
/// "nothing configured" apart from "configured but not installed" -- both
/// collapse to `None` -- yet callers (the desktop default-model banner, the
/// `GET /v1/models/default` status field) need to tell those apart to show
/// the right prompt ("choose a model" vs. "reinstall your default").
// `resolve` returns this by value at most once per call (never in a hot loop
// or a large collection), so the `Installed(InstalledPack)` / `NotInstalled
// (String)` size delta doesn't warrant boxing every caller's match arm --
// server routes and the CLI both want to destructure it directly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultModelResolution {
    /// `config.default_model` (or the `default.json` pointer as a fallback)
    /// names a model that has a matching pack installed.
    Installed(InstalledPack),
    /// A default is configured, but no installed pack matches it (removed,
    /// never installed, or the wrong quant with no fallback available).
    NotInstalled(String),
    /// Neither `config.default_model` nor the `default.json` pointer is set.
    Unset,
}

impl DefaultModelResolution {
    pub fn installed_pack(&self) -> Option<&InstalledPack> {
        match self {
            Self::Installed(pack) => Some(pack),
            Self::NotInstalled(_) | Self::Unset => None,
        }
    }

    pub fn into_installed_pack(self) -> Option<InstalledPack> {
        match self {
            Self::Installed(pack) => Some(pack),
            Self::NotInstalled(_) | Self::Unset => None,
        }
    }
}

/// Returns the currently configured model identity using the V2 record when it
/// exists. An explicit V2 `Unset` therefore returns `None` and never falls
/// through to stale compatibility projections. Only a missing V2 file falls
/// back to the legacy config/pointer precedence.
pub fn current_default_model(home: &Path) -> Result<Option<String>, DefaultSelectionError> {
    if let Some(record) = read_v2(home)? {
        return Ok(record.model_id);
    }

    let document = load_config_document(home)?;
    if let Some(model_id) = document.config.default_model {
        return Ok(Some(model_id));
    }
    Ok(read_default_pack_pointer(home)?.map(|pointer| pointer.model_id))
}

/// Resolves the persisted default model against installed packs, loading the
/// catalog from `catalog_url` first (a bare `None` skips catalog loading
/// entirely rather than falling back to the bundled catalog -- see
/// `resolve_with_catalog` for the shared logic and why callers that already
/// hold a loaded catalog, like the CLI, should call that instead).
///
/// Priority (ported verbatim from the original server-only resolver):
/// `config.default_model` wins when set; the `default.json` pointer's model
/// id is a fallback only when `config.default_model` is unset. When
/// `preferences.quant_preference` is `Pinned` and a pointer exists, the
/// pointer's quant is tried first (falling back to the best installed quant
/// if that exact quant was removed) -- this keeps `openasr pull <id>:<quant>`
/// sticky across quant changes without re-filtering the candidate list to a
/// single quant (which would break the Pinned-missing fallback ladder).
pub fn resolve(
    home: &Path,
    catalog_url: Option<&str>,
) -> Result<DefaultModelResolution, DefaultSelectionError> {
    let catalog = catalog_url
        .map(|catalog_url| load_model_catalog(Some(catalog_url), home))
        .transpose()?;
    resolve_with_catalog(home, catalog.as_ref())
}

/// Same resolution as `resolve`, but against an already-loaded catalog
/// (or `None` to resolve without catalog-assisted alias matching). Lets a
/// caller that owns its own catalog-loading policy -- the CLI's
/// `OPENASR_CATALOG_URL`/local-file override in `load_cli_model_catalog`,
/// distinct from the server's `catalog_url` override -- share this resolver
/// without loading the catalog twice or adopting the server's policy.
pub fn resolve_with_catalog(
    home: &Path,
    catalog: Option<&ModelCatalog>,
) -> Result<DefaultModelResolution, DefaultSelectionError> {
    if let Some(record) = read_v2(home)? {
        return resolve_v2(&record, home, catalog);
    }

    let packs = list_installed_packs(home)?;
    let document = load_config_document(home)?;
    let pointer = read_default_pack_pointer(home)?;

    if matches!(
        document.preferences.quant_preference,
        QuantPreference::Pinned { .. }
    ) && let Some(pointer) = pointer.as_ref()
    {
        let pointer_preference = QuantPreference::pinned(&pointer.quant);
        let reference = document
            .config
            .default_model
            .as_deref()
            .unwrap_or(pointer.model_id.as_str());
        return Ok(select(&packs, reference, &pointer_preference, catalog));
    }

    let Some(default_model) = document
        .config
        .default_model
        .as_deref()
        .or_else(|| pointer.as_ref().map(|pointer| pointer.model_id.as_str()))
    else {
        return Ok(DefaultModelResolution::Unset);
    };
    Ok(select(
        &packs,
        default_model,
        &document.preferences.quant_preference,
        catalog,
    ))
}

pub fn read_active_model_selection_v2(
    home: impl AsRef<Path>,
) -> Result<Option<ActiveModelSelectionV2>, DefaultSelectionError> {
    read_v2(home.as_ref())
}

fn read_v2(home: &Path) -> Result<Option<ActiveModelSelectionV2>, DefaultSelectionError> {
    let path = active_model_selection_v2_path(home);
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(DefaultSelectionError::Pull(PullError::Io { path, source })),
    };
    let record = serde_json::from_str::<ActiveModelSelectionV2>(&contents).map_err(|source| {
        DefaultSelectionError::Corrupt {
            path: path.clone(),
            reason: format!("invalid JSON: {source}"),
        }
    })?;
    record.validate(&path)?;
    Ok(Some(record))
}

fn resolve_v2(
    record: &ActiveModelSelectionV2,
    home: &Path,
    catalog: Option<&ModelCatalog>,
) -> Result<DefaultModelResolution, DefaultSelectionError> {
    if record.status == ActiveModelSelectionStatus::Unset {
        return Ok(DefaultModelResolution::Unset);
    }
    let packs = list_installed_packs(home)?;
    let model_id = record.model_id.as_deref().expect("validated V2 model id");
    if record.status == ActiveModelSelectionStatus::NotInstalled && record.expected_pack.is_none() {
        let reference = record.pull.as_deref().unwrap_or(model_id);
        let request = LaunchPackRequest {
            model_ref: reference,
            preference: &record.quant_preference,
            catalog,
            host_profile: host_quant_recommendation_profile(),
        };
        if let Ok(selection) = resolve_launch_pack(&packs, &request) {
            return Ok(DefaultModelResolution::Installed(selection.pack));
        }
    }
    let Some(quant) = record.quant.as_deref() else {
        return Ok(DefaultModelResolution::NotInstalled(model_id.to_string()));
    };
    let Some(expected) = record.expected_pack.as_ref() else {
        return Ok(DefaultModelResolution::NotInstalled(model_id.to_string()));
    };
    let matching = packs.iter().find(|pack| {
        pack.model_id == model_id
            && crate::canonical_quant_tag(&pack.quant) == crate::canonical_quant_tag(quant)
            && pack.sha256.eq_ignore_ascii_case(&expected.sha256)
            && pack.size_bytes == expected.size_bytes
    });
    let _ = catalog;
    Ok(match matching {
        Some(pack) => DefaultModelResolution::Installed(pack.clone()),
        None => DefaultModelResolution::NotInstalled(model_id.to_string()),
    })
}

fn select(
    packs: &[InstalledPack],
    reference: &str,
    preference: &QuantPreference,
    catalog: Option<&ModelCatalog>,
) -> DefaultModelResolution {
    let request = LaunchPackRequest {
        model_ref: reference,
        preference,
        catalog,
        host_profile: host_quant_recommendation_profile(),
    };
    match resolve_launch_pack(packs, &request) {
        Ok(selection) => DefaultModelResolution::Installed(selection.pack),
        Err(_) => DefaultModelResolution::NotInstalled(reference.to_string()),
    }
}

/// Persists `pack` as the default model: writes `config.json`'s
/// `default_model` (bare model id) and the `default.json` pointer, in that
/// order. Callers must go through this single function rather than calling
/// the two underlying writes separately, so the two files never drift.
/// Source-compatible legacy writer. It never rolls back a committed V2 record:
/// projection failure returns success with an explicit warning and can be
/// repaired through [`repair_compat_projection`].
pub fn persist(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<(), DefaultSelectionError> {
    match persist_detailed(home, pack, quant_preference)? {
        DefaultSelectionCommitOutcome::NotCommitted { reason } => {
            Err(DefaultSelectionError::NotCommitted { reason })
        }
        DefaultSelectionCommitOutcome::V2Committed => Ok(()),
        DefaultSelectionCommitOutcome::V2CommittedProjectionFailed { reason } => {
            eprintln!(
                "openasr-core: default V2 committed; legacy projection repair is pending: {reason}"
            );
            Ok(())
        }
    }
}

pub fn persist_detailed(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    persist_detailed_with_activation_metadata(home, pack, quant_preference, None, "auto", None)
}

/// Activation-only writer. The ordinary legacy projection API above keeps its
/// historical `auto`/unknown-architecture metadata; the attested server path
/// supplies the immutable architecture and user execution intent resolved by
/// the activation plan.
pub fn persist_activation_detailed(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
    architecture_id: &str,
    execution_intent: &crate::device::execution_policy::ExecutionIntent,
) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    persist_activation_detailed_with_fault(
        home,
        pack,
        quant_preference,
        architecture_id,
        execution_intent,
        None,
    )
}

#[doc(hidden)]
pub fn persist_activation_detailed_with_fault(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
    architecture_id: &str,
    execution_intent: &crate::device::execution_policy::ExecutionIntent,
    fault: Option<DefaultSelectionWriteFault>,
) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    let execution_intent = execution_intent_to_v2_wire(execution_intent);
    persist_detailed_with_activation_metadata(
        home,
        pack,
        quant_preference,
        Some(architecture_id),
        &execution_intent,
        fault,
    )
}

fn persist_detailed_with_activation_metadata(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
    architecture_id: Option<&str>,
    execution_intent: &str,
    fault: Option<DefaultSelectionWriteFault>,
) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    if persist_commit_failpoint_enabled() {
        return Ok(DefaultSelectionCommitOutcome::NotCommitted {
            reason: "injected persist failure".to_string(),
        });
    }
    let generation = match next_generation(home) {
        Ok(generation) => generation,
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::NotCommitted {
                reason: error.to_string(),
            });
        }
    };
    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: generation,
        status: ActiveModelSelectionStatus::Installed,
        pull: Some(pack.pull.clone()),
        model_id: Some(pack.model_id.clone()),
        quant: Some(pack.quant.clone()),
        architecture_id: architecture_id.map(str::to_string),
        expected_pack: Some(ExpectedPackIdentityV2 {
            sha256: pack.sha256.clone(),
            size_bytes: pack.size_bytes,
        }),
        quant_preference: quant_preference.clone(),
        execution_intent: execution_intent.to_string(),
        checksum: String::new(),
    };
    let record = match record.with_checksum() {
        Ok(record) => record,
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::NotCommitted {
                reason: error.to_string(),
            });
        }
    };
    if fault == Some(DefaultSelectionWriteFault::BeforeStagingWrite) {
        return Ok(DefaultSelectionCommitOutcome::NotCommitted {
            reason: "injected failure before V2 staging write".to_string(),
        });
    }
    match persist_v2_record_unlocked_with_fault(home, record, fault) {
        Ok(_) => {}
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::NotCommitted {
                reason: error.to_string(),
            });
        }
    }
    // Compatibility projections are deliberately after the authoritative V2
    // commit. A crash here leaves a complete V2 record for the next reader.
    let mut projection = match load_config_document(home) {
        Ok(document) => document,
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
                reason: error.to_string(),
            });
        }
    };
    projection.config.default_model = Some(pack.model_id.clone());
    projection.preferences.quant_preference = quant_preference;
    if let Err(error) = crate::config::save_config_document_unlocked(home, &projection) {
        return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
            reason: error.to_string(),
        });
    }
    if let Err(error) = persist_default_pack_pointer(home, pack) {
        return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
            reason: error.to_string(),
        });
    }
    Ok(DefaultSelectionCommitOutcome::V2Committed)
}

/// Preserve a corrupt or unknown V2 record as evidence, then atomically reset
/// the active record to a checksummed `Unset`. This is an explicit operator
/// action; normal resolution never downgrades to legacy or repairs silently.
pub fn recover_corrupt_v2_detailed(
    home: &Path,
) -> Result<DefaultSelectionRecoveryOutcome, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    let path = active_model_selection_v2_path(home);
    match read_v2(home) {
        Ok(Some(record)) => {
            return Ok(DefaultSelectionRecoveryOutcome::Committed {
                record,
                evidence_path: PathBuf::new(),
            });
        }
        Ok(None) => {
            return Err(DefaultSelectionError::Corrupt {
                path,
                reason: "no V2 record exists".to_string(),
            });
        }
        Err(DefaultSelectionError::Corrupt { .. }) => {}
        Err(error) => return Err(error),
    }
    let original = fs::read(&path).map_err(|source| {
        DefaultSelectionError::Pull(PullError::Io {
            path: path.clone(),
            source,
        })
    })?;
    let evidence = home.join(format!(
        "default-selection.corrupt.{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or_default()
    ));
    let mut evidence_options = OpenOptions::new();
    evidence_options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        evidence_options.mode(0o600);
    }
    let mut evidence_file = evidence_options.open(&evidence).map_err(|source| {
        DefaultSelectionError::Pull(PullError::Io {
            path: evidence.clone(),
            source,
        })
    })?;
    evidence_file
        .write_all(&original)
        .and_then(|_| evidence_file.flush())
        .and_then(|_| evidence_file.sync_all())
        .map_err(|source| {
            DefaultSelectionError::Pull(PullError::Io {
                path: evidence.clone(),
                source,
            })
        })?;
    crate::atomic_file::sync_parent_dir_best_effort(&evidence);
    if recovery_fail_after_evidence() {
        return Err(DefaultSelectionError::Corrupt {
            path,
            reason: "injected recovery failure after evidence copy".to_string(),
        });
    }
    let mut record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 1,
        status: ActiveModelSelectionStatus::Unset,
        pull: None,
        model_id: None,
        quant: None,
        architecture_id: None,
        expected_pack: None,
        quant_preference: QuantPreference::Auto,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    record = record
        .with_checksum()
        .map_err(|source| DefaultSelectionError::Corrupt {
            path: active_model_selection_v2_path(home),
            reason: source.to_string(),
        })?;
    persist_v2_record_without_generation(home, &record)?;
    match repair_compat_projection_unlocked(home, &record) {
        Ok(()) => Ok(DefaultSelectionRecoveryOutcome::Committed {
            record,
            evidence_path: evidence,
        }),
        Err(reason) => Ok(DefaultSelectionRecoveryOutcome::ProjectionFailed {
            record,
            evidence_path: evidence,
            reason,
        }),
    }
}

pub fn recover_corrupt_v2(home: &Path) -> Result<ActiveModelSelectionV2, DefaultSelectionError> {
    let outcome = recover_corrupt_v2_detailed(home)?;
    if let DefaultSelectionRecoveryOutcome::ProjectionFailed { reason, .. } = &outcome {
        eprintln!(
            "openasr-core: corrupt V2 recovery committed; legacy projection repair is pending: {reason}"
        );
    }
    Ok(outcome.into_record())
}

pub fn reset_corrupt_v2(home: &Path) -> Result<ActiveModelSelectionV2, DefaultSelectionError> {
    recover_corrupt_v2(home)
}

/// without changing its generation. This is idempotent and is safe to call
/// after a projection-only failure or on server startup.
pub fn repair_compat_projection(
    home: &Path,
) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    let Some(record) = read_v2(home)? else {
        return Ok(DefaultSelectionCommitOutcome::NotCommitted {
            reason: "no V2 record exists".to_string(),
        });
    };
    match repair_compat_projection_unlocked(home, &record) {
        Ok(()) => Ok(DefaultSelectionCommitOutcome::V2Committed),
        Err(reason) => Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed { reason }),
    }
}

fn repair_compat_projection_unlocked(
    home: &Path,
    record: &ActiveModelSelectionV2,
) -> Result<(), String> {
    match record.status {
        ActiveModelSelectionStatus::Unset => {
            let mut document = load_config_document(home).map_err(|error| error.to_string())?;
            document.config.default_model = None;
            document.preferences.quant_preference = QuantPreference::Auto;
            crate::config::save_config_document_unlocked(home, &document)
                .map_err(|error| error.to_string())?;
            remove_legacy_pointer(home).map_err(|error| error.to_string())?;
        }
        ActiveModelSelectionStatus::Installed | ActiveModelSelectionStatus::NotInstalled => {
            let model_id = record.model_id.as_deref().expect("validated V2 model id");
            let mut document = load_config_document(home).map_err(|error| error.to_string())?;
            document.config.default_model = Some(model_id.to_string());
            document.preferences.quant_preference = record.quant_preference.clone();
            crate::config::save_config_document_unlocked(home, &document)
                .map_err(|error| error.to_string())?;
            let packs = list_installed_packs(home).map_err(|error| error.to_string())?;
            if let Some(expected) = record.expected_pack.as_ref()
                && let Some(pack) = packs.iter().find(|pack| {
                    pack.model_id == model_id
                        && pack.sha256.eq_ignore_ascii_case(&expected.sha256)
                        && pack.size_bytes == expected.size_bytes
                })
            {
                persist_default_pack_pointer(home, pack).map_err(|error| error.to_string())?;
            } else {
                remove_legacy_pointer(home).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn remove_legacy_pointer(home: &Path) -> Result<(), DefaultSelectionError> {
    let path = default_pack_pointer_path(home);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(DefaultSelectionError::Pull(PullError::Io { path, source })),
    }
}

/// Save a generic config update without allowing it to overwrite the V2
/// selection projection observed before the write. The process mutex and
/// cross-process lock cover the final read/modify/write boundary; when V2 is
/// present its model and quant preference remain authoritative.
pub fn save_config_document_preserving_v2_selection(
    home: &Path,
    document: &crate::OpenAsrConfigDocument,
) -> Result<(), DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    let mut document = document.clone();
    if let Some(record) = read_v2(home)? {
        document.config.default_model = record.model_id;
        document.preferences.quant_preference = record.quant_preference;
    }
    crate::config::save_config_document_unlocked(home, &document)
        .map_err(DefaultSelectionError::Config)
}

/// The supplied generation is ignored: writers cannot move the generation
/// backwards or overwrite an existing malformed record.
pub fn persist_v2_record(
    home: &Path,
    record: ActiveModelSelectionV2,
) -> Result<ActiveModelSelectionV2, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    persist_v2_record_unlocked(home, record)
}

fn persist_v2_record_unlocked(
    home: &Path,
    record: ActiveModelSelectionV2,
) -> Result<ActiveModelSelectionV2, DefaultSelectionError> {
    persist_v2_record_unlocked_with_fault(home, record, None)
}

fn persist_v2_record_unlocked_with_fault(
    home: &Path,
    mut record: ActiveModelSelectionV2,
    fault: Option<DefaultSelectionWriteFault>,
) -> Result<ActiveModelSelectionV2, DefaultSelectionError> {
    record.schema_version = ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION;
    record.selection_generation = next_generation(home)?;
    record.checksum = String::new();
    record = record
        .with_checksum()
        .map_err(|source| DefaultSelectionError::Corrupt {
            path: active_model_selection_v2_path(home),
            reason: format!("cannot serialize selection: {source}"),
        })?;
    record.validate(&active_model_selection_v2_path(home))?;
    persist_v2_record_without_generation_with_fault(home, &record, fault)?;
    Ok(record)
}

fn persist_v2_record_without_generation(
    home: &Path,
    record: &ActiveModelSelectionV2,
) -> Result<(), DefaultSelectionError> {
    persist_v2_record_without_generation_with_fault(home, record, None)
}

fn persist_v2_record_without_generation_with_fault(
    home: &Path,
    record: &ActiveModelSelectionV2,
    fault: Option<DefaultSelectionWriteFault>,
) -> Result<(), DefaultSelectionError> {
    std::fs::create_dir_all(home).map_err(|source| {
        DefaultSelectionError::Pull(PullError::Io {
            path: home.to_path_buf(),
            source,
        })
    })?;
    let path = active_model_selection_v2_path(home);
    let contents =
        serde_json::to_vec_pretty(record).map_err(|source| DefaultSelectionError::Corrupt {
            path: path.clone(),
            reason: format!("cannot serialize selection: {source}"),
        })?;
    let write = match fault {
        Some(DefaultSelectionWriteFault::BeforeStagingSync) => {
            crate::atomic_file::write_file_atomically_detailed_with_failpoint(
                &path,
                &contents,
                crate::atomic_file::AtomicFileMode::Default,
                crate::atomic_file::AtomicFileFailpoint::BeforeSync,
            )
        }
        Some(DefaultSelectionWriteFault::BeforeAtomicReplace) => {
            crate::atomic_file::write_file_atomically_detailed_with_failpoint(
                &path,
                &contents,
                crate::atomic_file::AtomicFileMode::Default,
                crate::atomic_file::AtomicFileFailpoint::BeforeReplace,
            )
        }
        Some(DefaultSelectionWriteFault::AfterAtomicReplace) => {
            crate::atomic_file::write_file_atomically_detailed_with_failpoint(
                &path,
                &contents,
                crate::atomic_file::AtomicFileMode::Default,
                crate::atomic_file::AtomicFileFailpoint::AfterReplace,
            )
        }
        Some(DefaultSelectionWriteFault::BeforeStagingWrite) | None => {
            crate::atomic_file::write_file_atomically_detailed(
                &path,
                &contents,
                crate::atomic_file::AtomicFileMode::Default,
            )
        }
    };
    match write {
        Ok(crate::atomic_file::AtomicWriteOutcome::Written) => Ok(()),
        Ok(crate::atomic_file::AtomicWriteOutcome::CommittedWithSyncWarning { source }) => {
            eprintln!("openasr-core: default V2 record committed but parent sync failed: {source}");
            Ok(())
        }
        Err(crate::atomic_file::AtomicWriteError::NotCommitted(source)) => {
            Err(DefaultSelectionError::Pull(PullError::Io { path, source }))
        }
    }
}

fn next_generation(home: &Path) -> Result<u64, DefaultSelectionError> {
    match read_v2(home)? {
        Some(record) => record.selection_generation.checked_add(1).ok_or_else(|| {
            DefaultSelectionError::Corrupt {
                path: active_model_selection_v2_path(home),
                reason: "selection generation overflow".to_string(),
            }
        }),
        None => Ok(1),
    }
}

/// Clears the persisted default: resets `config.json`'s `default_model` and
/// `quant_preference` to their unset states and removes the `default.json`
/// pointer file (a missing pointer file is not an error).
/// Source-compatible legacy clear wrapper. It reports pre-commit failure, but
/// never pretends a committed V2 record was rolled back after projection trouble.
pub fn clear(home: &Path) -> Result<(), DefaultSelectionError> {
    match clear_detailed(home)? {
        DefaultSelectionCommitOutcome::NotCommitted { reason } => {
            Err(DefaultSelectionError::NotCommitted { reason })
        }
        DefaultSelectionCommitOutcome::V2Committed => Ok(()),
        DefaultSelectionCommitOutcome::V2CommittedProjectionFailed { reason } => {
            eprintln!(
                "openasr-core: default V2 clear committed; legacy projection repair is pending: {reason}"
            );
            Ok(())
        }
    }
}

pub fn clear_detailed(home: &Path) -> Result<DefaultSelectionCommitOutcome, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 0,
        status: ActiveModelSelectionStatus::Unset,
        pull: None,
        model_id: None,
        quant: None,
        architecture_id: None,
        expected_pack: None,
        quant_preference: QuantPreference::Auto,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    match persist_v2_record_unlocked(home, record) {
        Ok(_) => {}
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::NotCommitted {
                reason: error.to_string(),
            });
        }
    }

    let mut document = match load_config_document(home) {
        Ok(document) => document,
        Err(error) => {
            return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
                reason: error.to_string(),
            });
        }
    };
    document.config.default_model = None;
    document.preferences.quant_preference = QuantPreference::Auto;
    if let Err(error) = crate::config::save_config_document_unlocked(home, &document) {
        return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
            reason: error.to_string(),
        });
    }

    if let Err(error) = remove_legacy_pointer(home) {
        return Ok(DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
            reason: error.to_string(),
        });
    }
    Ok(DefaultSelectionCommitOutcome::V2Committed)
}

fn canonicalize_legacy_reference(
    reference: &str,
    catalog: Option<&ModelCatalog>,
) -> (String, String, Option<String>) {
    let parsed = parse_model_ref(reference).ok();
    let bare = parsed
        .as_ref()
        .map(|parsed| parsed.family.as_str())
        .unwrap_or(reference.trim());
    let explicit_quant = parsed
        .as_ref()
        .and_then(|parsed| parsed.tag.as_deref())
        .map(canonical_quant_tag)
        .map(str::to_string);
    let canonical_model_id = catalog
        .and_then(|catalog| {
            resolve_catalog_pull_with_profile(
                catalog,
                &CatalogPullRequest {
                    reference: reference.to_string(),
                    quant: None,
                    size: None,
                },
                None,
            )
            .ok()
        })
        .map(|resolved| resolved.model_id)
        .unwrap_or_else(|| bare.to_string());
    let canonical_reference = explicit_quant
        .as_ref()
        .map(|quant| format!("{canonical_model_id}:{quant}"))
        .unwrap_or_else(|| canonical_model_id.clone());
    (canonical_model_id, canonical_reference, explicit_quant)
}

/// Explicitly migrate the legacy two-file state into one V2 record. Reading a
/// legacy state never writes as a side effect; callers choose when migration is
/// appropriate.
pub fn migrate_legacy_to_v2(
    home: &Path,
) -> Result<Option<ActiveModelSelectionV2>, DefaultSelectionError> {
    let catalog = load_embedded_signed_catalog(home)?;
    migrate_legacy_to_v2_with_catalog(home, Some(&catalog))
}

/// Migrate legacy selection state using the same catalog-aware model-reference
/// and quant-selection path as runtime resolution. Callers that already own a
/// verified catalog should pass it here; `None` retains the legacy no-catalog
/// behavior for offline callers.
pub fn migrate_legacy_to_v2_with_catalog(
    home: &Path,
    catalog: Option<&ModelCatalog>,
) -> Result<Option<ActiveModelSelectionV2>, DefaultSelectionError> {
    let _lock = selection_write_lock();
    let _file_lock = SelectionFileLock::acquire(home)?;
    if let Some(record) = read_v2(home)? {
        return Ok(Some(record));
    }
    let packs = list_installed_packs(home)?;
    let document = load_config_document(home)?;
    let pointer = read_default_pack_pointer(home)?;
    let Some(model_ref) = document
        .config
        .default_model
        .clone()
        .or_else(|| pointer.as_ref().map(|value| value.model_id.clone()))
    else {
        let record = ActiveModelSelectionV2 {
            schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
            selection_generation: 0,
            status: ActiveModelSelectionStatus::Unset,
            pull: None,
            model_id: None,
            quant: None,
            architecture_id: None,
            expected_pack: None,
            quant_preference: document.preferences.quant_preference,
            execution_intent: "auto".to_string(),
            checksum: String::new(),
        };
        return persist_v2_record_unlocked(home, record).map(Some);
    };
    let (canonical_model_id, canonical_reference, explicit_quant) =
        canonicalize_legacy_reference(&model_ref, catalog);
    let pointer = pointer.filter(|value| value.model_id == canonical_model_id);
    let preference = if matches!(
        document.preferences.quant_preference,
        QuantPreference::Pinned { .. }
    ) {
        pointer
            .as_ref()
            .map(|value| QuantPreference::pinned(&value.quant))
            .unwrap_or_else(|| document.preferences.quant_preference.clone())
    } else {
        document.preferences.quant_preference.clone()
    };
    let request = LaunchPackRequest {
        model_ref: &canonical_reference,
        preference: &preference,
        catalog,
        host_profile: host_quant_recommendation_profile(),
    };
    let selected = resolve_launch_pack(&packs, &request)
        .ok()
        .map(|selection| selection.pack);
    let selected_model_id = selected
        .as_ref()
        .map(|pack| pack.model_id.clone())
        .or_else(|| pointer.as_ref().map(|value| value.model_id.clone()))
        .unwrap_or_else(|| canonical_model_id.clone());
    let (pull, quant, expected, status) = if let Some(pack) = selected {
        (
            Some(pack.pull.clone()),
            Some(pack.quant.clone()),
            Some(ExpectedPackIdentityV2 {
                sha256: pack.sha256.clone(),
                size_bytes: pack.size_bytes,
            }),
            ActiveModelSelectionStatus::Installed,
        )
    } else if let Some(pointer) = pointer {
        (
            Some(pointer.pull),
            Some(pointer.quant),
            Some(ExpectedPackIdentityV2 {
                sha256: pointer.sha256,
                size_bytes: pointer.size_bytes,
            }),
            ActiveModelSelectionStatus::NotInstalled,
        )
    } else {
        (
            Some(canonical_reference),
            explicit_quant,
            None,
            ActiveModelSelectionStatus::NotInstalled,
        )
    };
    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 0,
        status,
        pull,
        model_id: Some(selected_model_id),
        quant,
        architecture_id: None,
        expected_pack: expected,
        quant_preference: document.preferences.quant_preference,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    persist_v2_record_unlocked(home, record).map(Some)
}

#[cfg(test)]
#[path = "default_selection_tests.rs"]
mod default_selection_tests;
