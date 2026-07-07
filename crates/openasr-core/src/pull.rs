use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    CatalogBackendFile, CatalogBackendFileRole, CatalogBackendVendor, CatalogModel,
    CatalogPullRequest, CatalogQuant, ModelCatalog, OPENASR_RUNTIME_PACK_EXTENSION,
    ResolvedCatalogBackendPull, ResolvedCatalogPull, atomic_file, canonical_quant_tag,
    catalog_series::family_aliases_match,
    download_source::{self, DownloadSource},
    has_openasr_runtime_pack_extension, http, parse_model_ref, resolve_catalog_pull,
    safety::{validate_safe_relative_path, validate_sha256},
    validate_ggml_runtime_source_path, validate_native_runtime_model_pack_contract,
};

const LOCK_STALE_AFTER: Duration = Duration::from_secs(6 * 60 * 60);
const LOCK_STALE_RECOVERY_ATTEMPTS: usize = 4;
const METADATA_WRITE_INTERVAL_BYTES: u64 = 8 * 1024 * 1024;
const DOWNLOAD_BUFFER_BYTES: usize = 64 * 1024;
const DOWNLOAD_MAX_RETRIES: usize = 6;
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_STALL_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_LOW_SPEED_TIMEOUT: Duration = Duration::from_secs(60);
const DOWNLOAD_LOW_SPEED_MIN_BYTES: u64 = 64 * 1024;
const DOWNLOAD_USER_AGENT: &str = concat!("OpenASR/", env!("CARGO_PKG_VERSION"));
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_GGUF_METADATA_ENTRIES: u64 = 100_000;
const MAX_GGUF_TENSORS: u64 = 1_000_000;
const MAX_GGUF_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_DIMS: u32 = 8;
const MAX_GGUF_ARRAY_VALUES: u64 = 16 * 1024 * 1024;
const OASR_PACKAGE_VERSION_KEY: &str = "openasr.package.version";
const OASR_PACKAGE_VERSION_V1: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPack {
    pub model_id: String,
    pub display_name: String,
    pub quant: String,
    pub suffix: String,
    pub pull: String,
    pub filename: String,
    pub path: PathBuf,
    pub url: String,
    pub hf_revision: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub installed_at_unix_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefaultPackPointer {
    pub model_id: String,
    pub quant: String,
    pub suffix: String,
    pub pull: String,
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    pub updated_at_unix_seconds: u64,
}

impl DefaultPackPointer {
    pub fn from_pack(pack: &InstalledPack) -> Self {
        Self {
            model_id: pack.model_id.clone(),
            quant: pack.quant.clone(),
            suffix: pack.suffix.clone(),
            pull: pack.pull.clone(),
            path: pack.path.clone(),
            sha256: pack.sha256.clone(),
            size_bytes: pack.size_bytes,
            updated_at_unix_seconds: unix_seconds_now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullProgress {
    UsingInstalled { path: PathBuf },
    DownloadStarted { bytes_total: u64, resume_from: u64 },
    Downloading { bytes_done: u64, bytes_total: u64 },
    Verifying { bytes_done: u64 },
    Installed { path: PathBuf },
}

#[derive(Debug, Error)]
pub enum PullError {
    #[error("Model pack URL must use https://: {url}")]
    NonHttpsUrl { url: String },
    #[error("Invalid catalog pull target '{field}': {reason}")]
    InvalidTarget { field: &'static str, reason: String },
    #[error("Could not create OpenASR model directory '{path}': {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Another pull is already writing '{path}'.")]
    LockHeld { path: PathBuf },
    #[error("Could not acquire pull lock '{path}': {source}")]
    LockIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "Insufficient free disk space under '{path}': need {needed_bytes} bytes, available {available_bytes} bytes"
    )]
    InsufficientSpace {
        path: PathBuf,
        needed_bytes: u64,
        available_bytes: u64,
    },
    #[error("Could not read or write model pack file '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "Model pack '{path}' is in use and cannot be replaced. Close OpenASR (and any app using this model), then try again."
    )]
    ModelInUse {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Unsafe OpenASR model storage path rejected: {path}")]
    UnsafeStoragePath { path: PathBuf },
    #[error("Could not serialize pull metadata for '{path}': {source}")]
    SerializeMeta {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("Could not parse pull metadata '{path}': {source}")]
    ParseMeta {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("HTTP request failed for '{url}': {message}")]
    Http { url: String, message: String },
    #[error("HTTP response for '{url}' returned status {status}; expected 200 or 206")]
    UnexpectedStatus { url: String, status: u16 },
    #[error(
        "HTTP resume for '{url}' could not safely append, so the partial download was restarted"
    )]
    RestartedPartial { url: String },
    #[error(
        "Downloaded pack size mismatch for '{path}': expected {expected} bytes, got {actual} bytes"
    )]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("Downloaded pack sha256 mismatch for '{path}': expected {expected}, got {actual}")]
    ShaMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("Downloaded pack failed Rust-only GGUF preflight for '{path}': {reason}")]
    GgufPreflight { path: PathBuf, reason: String },
    #[error("Downloaded backend file failed binary preflight for '{path}': {reason}")]
    BackendFilePreflight { path: PathBuf, reason: String },
    #[error("Downloaded pack failed runtime path validation for '{path}': {reason}")]
    RuntimeValidation { path: PathBuf, reason: String },
    #[error("Installed model pack not found: {reference}")]
    NotInstalled { reference: String },
    #[error("Model pack pull was canceled: {reference}")]
    Canceled { reference: String },
    #[error("Model pack pull was paused: {reference}")]
    Paused { reference: String },
}

#[derive(Clone, Debug)]
struct PullTarget {
    model_id: String,
    display_name: String,
    quant: String,
    suffix: String,
    pull: String,
    filename: String,
    url: String,
    hf_revision: String,
    sha256: String,
    size_bytes: u64,
    source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PartialMeta {
    model_id: String,
    quant: String,
    filename: String,
    url: String,
    hf_revision: String,
    sha256: String,
    size_bytes: u64,
    etag: Option<String>,
    bytes_done: u64,
    updated_at_unix_seconds: u64,
}

#[derive(Debug, Clone)]
struct PullPaths {
    dir: PathBuf,
    final_path: PathBuf,
    partial_path: PathBuf,
    partial_meta_path: PathBuf,
    installed_meta_path: PathBuf,
    lock_path: PathBuf,
}

#[derive(Debug, Clone)]
struct PullOptions {
    available_space_override: Option<u64>,
    low_speed_timeout: Duration,
    low_speed_min_bytes: u64,
}

impl PullOptions {
    fn default() -> Self {
        Self {
            available_space_override: None,
            low_speed_timeout: DOWNLOAD_LOW_SPEED_TIMEOUT,
            low_speed_min_bytes: DOWNLOAD_LOW_SPEED_MIN_BYTES,
        }
    }
}

trait DownloadClient {
    fn open(&mut self, url: &str, range_start: Option<u64>) -> Result<DownloadResponse, PullError>;
}

struct DownloadResponse {
    status: u16,
    content_length: Option<u64>,
    content_range: Option<String>,
    etag: Option<String>,
    reader: Box<dyn Read>,
}

struct DownloadedPartial {
    bytes_done: u64,
    sha256: String,
}

struct HttpDownloadClient {
    client: reqwest::blocking::Client,
    /// Optional Hugging Face access token (`OPENASR_HF_TOKEN`). Attached only to
    /// requests whose host is `huggingface.co`, never to the CDN/mirror redirect
    /// targets — the same origin-scoping rule applied to redirect cookies.
    hf_token: Option<String>,
}

/// A download-and-install request for a resolved catalog model pack.
///
/// Build with [`PullModelPackRequest::new`], optionally override the download
/// source chain with [`sources`](Self::sources) and attach cancel/pause
/// controls with [`cancel`](Self::cancel) / [`pause`](Self::pause), then run it
/// with [`execute`](Self::execute). Without an explicit source chain the request
/// uses the environment-configured chain; without controls it never cancels or
/// pauses. For the common no-control, environment-source case use the
/// [`pull_model_pack`] convenience wrapper.
pub struct PullModelPackRequest<'a> {
    resolved: &'a ResolvedCatalogPull,
    home: &'a Path,
    sources: Option<&'a [DownloadSource]>,
    should_cancel: Option<Box<dyn Fn() -> bool + 'a>>,
    should_pause: Option<Box<dyn Fn() -> bool + 'a>>,
}

impl<'a> PullModelPackRequest<'a> {
    /// Start a request for `resolved`, installing under `home`.
    pub fn new(resolved: &'a ResolvedCatalogPull, home: &'a Path) -> Self {
        Self {
            resolved,
            home,
            sources: None,
            should_cancel: None,
            should_pause: None,
        }
    }

    /// Override the download source chain. Defaults to the environment chain.
    pub fn sources(mut self, sources: &'a [DownloadSource]) -> Self {
        self.sources = Some(sources);
        self
    }

    /// Attach a cancellation predicate polled during the download.
    pub fn cancel(mut self, should_cancel: impl Fn() -> bool + 'a) -> Self {
        self.should_cancel = Some(Box::new(should_cancel));
        self
    }

    /// Attach a pause predicate polled during the download.
    pub fn pause(mut self, should_pause: impl Fn() -> bool + 'a) -> Self {
        self.should_pause = Some(Box::new(should_pause));
        self
    }

    /// Run the request, reporting progress to `progress`.
    pub fn execute(self, progress: impl FnMut(PullProgress)) -> Result<InstalledPack, PullError> {
        let PullModelPackRequest {
            resolved,
            home,
            sources,
            should_cancel,
            should_pause,
        } = self;
        let mut client = HttpDownloadClient::new()?;
        let env_sources;
        let sources = match sources {
            Some(sources) => sources,
            None => {
                env_sources = download_source::source_chain_from_env();
                &env_sources
            }
        };
        pull_model_pack_with_client_sources_and_cancel(
            resolved,
            home,
            &mut client,
            PullOptions::default(),
            sources,
            progress,
            || should_cancel.as_ref().is_some_and(|f| f()),
            || should_pause.as_ref().is_some_and(|f| f()),
        )
    }
}

/// Convenience wrapper over [`PullModelPackRequest`] using the environment
/// download source chain and no cancel/pause controls.
pub fn pull_model_pack(
    resolved: &ResolvedCatalogPull,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    PullModelPackRequest::new(resolved, home.as_ref()).execute(progress)
}

pub fn install_model_pack_from_path(
    resolved: &ResolvedCatalogPull,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    let target = PullTarget::from_resolved(resolved)?.with_source("local");
    install_model_pack_from_path_with_target(&target, source_path, home, progress)
}

pub fn install_catalog_model_pack_from_path(
    catalog: &ModelCatalog,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    let source_path = source_path.as_ref();
    if !has_openasr_runtime_pack_extension(source_path) {
        return Err(PullError::InvalidTarget {
            field: "path",
            reason: format!("local imports must use .{OPENASR_RUNTIME_PACK_EXTENSION} model packs"),
        });
    }
    let (size_bytes, sha256) = file_size_and_sha256(source_path)?;
    let resolved = resolve_catalog_pull_by_file_digest(catalog, size_bytes, &sha256)?;
    install_model_pack_from_path(&resolved, source_path, home, progress)
}

fn install_model_pack_from_path_with_target(
    target: &PullTarget,
    source_path: impl AsRef<Path>,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    let paths = pull_paths(home.as_ref(), target)?;
    ensure_storage_dir_within_root(home.as_ref(), &paths)?;
    let _lock = PullLock::acquire(&paths.lock_path)?;
    let source_path = source_path.as_ref();
    if !has_openasr_runtime_pack_extension(source_path) {
        return Err(PullError::InvalidTarget {
            field: "from",
            reason: format!("local imports must use .{OPENASR_RUNTIME_PACK_EXTENSION} model packs"),
        });
    }

    fs::copy(source_path, &paths.partial_path).map_err(|source| PullError::Io {
        path: paths.partial_path.clone(),
        source,
    })?;
    verify_partial_and_install(target, &paths, None, &|| false, progress)
}

fn resolve_catalog_pull_by_file_digest(
    catalog: &ModelCatalog,
    size_bytes: u64,
    sha256: &str,
) -> Result<ResolvedCatalogPull, PullError> {
    let matches = catalog
        .models
        .iter()
        .filter(|model| model.public)
        .flat_map(|model| {
            model
                .quants
                .iter()
                .filter(move |quant| quant.size_bytes == size_bytes && quant.sha256 == sha256)
                .map(move |quant| resolved_catalog_pull_from_quant(model, quant))
        })
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [resolved] => Ok(resolved.clone()),
        [] => Err(PullError::InvalidTarget {
            field: "sha256",
            reason: "local OASR pack sha256/size is not present in the signed model catalog"
                .to_string(),
        }),
        _ => Err(PullError::InvalidTarget {
            field: "sha256",
            reason: "local OASR pack sha256/size matches multiple catalog entries".to_string(),
        }),
    }
}

fn resolved_catalog_pull_from_quant(
    model: &CatalogModel,
    quant: &CatalogQuant,
) -> ResolvedCatalogPull {
    ResolvedCatalogPull {
        requested: quant.pull.clone(),
        model_id: model.id.clone(),
        display_name: model.display_name.clone(),
        quant: quant.quant.clone(),
        suffix: quant.suffix.clone(),
        pull: quant.pull.clone(),
        filename: quant.filename.clone(),
        url: quant.url.clone(),
        mirrors: quant.mirrors.clone(),
        hf_revision: model.hf_revision.clone(),
        sha256: quant.sha256.clone(),
        size_bytes: quant.size_bytes,
        license: model.license.clone(),
        license_url: model.license_url.clone(),
        license_class: model.license_class.clone(),
    }
}

pub fn list_installed_packs(home: impl AsRef<Path>) -> Result<Vec<InstalledPack>, PullError> {
    let root = models_root(home.as_ref());
    let mut packs = Vec::new();
    let Ok(model_dirs) = fs::read_dir(&root) else {
        return Ok(packs);
    };
    for model_dir in model_dirs {
        let model_dir = model_dir.map_err(|source| PullError::Io {
            path: root.clone(),
            source,
        })?;
        let Ok(quant_dirs) = fs::read_dir(model_dir.path()) else {
            continue;
        };
        for quant_dir in quant_dirs {
            let quant_dir = quant_dir.map_err(|source| PullError::Io {
                path: model_dir.path(),
                source,
            })?;
            let path = quant_dir.path().join("installed.json");
            if !path.exists() {
                continue;
            }
            let contents = fs::read_to_string(&path).map_err(|source| PullError::Io {
                path: path.clone(),
                source,
            })?;
            let pack: InstalledPack =
                serde_json::from_str(&contents).map_err(|source| PullError::ParseMeta {
                    path: path.clone(),
                    source,
                })?;
            if installed_pack_matches_quant_dir(&pack, &quant_dir.path()) {
                packs.push(pack);
            }
        }
    }
    packs.sort_by(|left, right| left.pull.cmp(&right.pull));
    Ok(packs)
}

pub fn default_pack_pointer_path(home: impl AsRef<Path>) -> PathBuf {
    home.as_ref().join("default.json")
}

pub fn read_default_pack_pointer(
    home: impl AsRef<Path>,
) -> Result<Option<DefaultPackPointer>, PullError> {
    let path = default_pack_pointer_path(home);
    match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents)
            .map(Some)
            .map_err(|source| PullError::ParseMeta { path, source }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(PullError::Io { path, source }),
    }
}

pub fn persist_default_pack_pointer(
    home: impl AsRef<Path>,
    pack: &InstalledPack,
) -> Result<(), PullError> {
    let home = home.as_ref();
    fs::create_dir_all(home).map_err(|source| PullError::CreateDir {
        path: home.to_path_buf(),
        source,
    })?;
    let path = default_pack_pointer_path(home);
    let pointer = DefaultPackPointer::from_pack(pack);
    let contents =
        serde_json::to_string_pretty(&pointer).map_err(|source| PullError::SerializeMeta {
            path: path.clone(),
            source,
        })?;
    atomic_file::write_file_atomically(&path, format!("{contents}\n").as_bytes())
        .map_err(|source| PullError::Io { path, source })
}

fn installed_pack_matches_quant_dir(pack: &InstalledPack, quant_dir: &Path) -> bool {
    if validate_safe_relative_path("model id", &pack.model_id).is_err()
        || validate_safe_relative_path("quant", &pack.quant).is_err()
        || validate_safe_relative_path("filename", &pack.filename).is_err()
        || pack.filename.contains('/')
        || pack.filename.contains('\\')
        || !has_openasr_runtime_pack_extension(&pack.filename)
    {
        return false;
    }
    let Some(model_dir) = quant_dir.parent() else {
        return false;
    };
    let Some(model_dir_name) = model_dir.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(quant_dir_name) = quant_dir.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if model_dir_name != pack.model_id || quant_dir_name != pack.quant {
        return false;
    }
    if pack.path != quant_dir.join(&pack.filename) {
        return false;
    }
    let Ok(metadata) = fs::symlink_metadata(&pack.path) else {
        return false;
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return false;
    }
    metadata.len() == pack.size_bytes
        && validate_native_runtime_model_pack_contract(&pack.path).is_ok()
}

pub fn remove_model_pack(
    home: impl AsRef<Path>,
    reference: &str,
) -> Result<Option<InstalledPack>, PullError> {
    let Some(pack) = find_installed_pack(home.as_ref(), reference)? else {
        return Ok(None);
    };
    if let Some(parent) = pack.path.parent() {
        fs::remove_dir_all(parent).map_err(|source| PullError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    Ok(Some(pack))
}

pub fn resolve_installed_pack_path(
    home: impl AsRef<Path>,
    reference: &str,
) -> Result<Option<PathBuf>, PullError> {
    Ok(find_installed_pack(home.as_ref(), reference)?.map(|pack| pack.path))
}

pub fn resolve_installed_pack_reference(
    packs: &[InstalledPack],
    reference: &str,
) -> Result<Option<InstalledPack>, PullError> {
    let reference = reference.trim();
    if reference.is_empty() {
        return Ok(None);
    }
    let reference_ref = parse_model_ref(reference).map_err(|error| PullError::InvalidTarget {
        field: "reference",
        reason: error.to_string(),
    })?;
    let quant = reference_ref.tag.as_deref().map(canonical_quant_tag);
    let matches = packs
        .iter()
        .filter(|pack| {
            pack.pull == reference
                || (family_aliases_match(&pack.model_id, &reference_ref.family)
                    && quant.is_none_or(|quant| {
                        canonical_quant_tag(&pack.quant) == quant
                            || canonical_quant_tag(&pack.suffix) == quant
                    }))
        })
        .cloned()
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(PullError::InvalidTarget {
            field: "reference",
            reason: format!("'{reference}' matches multiple installed quants; use <id>:<quant>"),
        });
    }
    Ok(matches.into_iter().next())
}

pub fn resolve_installed_pack_reference_with_catalog(
    packs: &[InstalledPack],
    catalog: &ModelCatalog,
    reference: &str,
) -> Result<Option<InstalledPack>, PullError> {
    if let Some(pack) = resolve_installed_pack_reference(packs, reference)? {
        return Ok(Some(pack));
    }
    let Ok(resolved) = resolve_catalog_pull(
        catalog,
        &CatalogPullRequest {
            reference: reference.trim().to_string(),
            quant: None,
            size: None,
        },
    ) else {
        return Ok(None);
    };
    resolve_installed_pack_reference(packs, &resolved.pull)
}

fn find_installed_pack(home: &Path, reference: &str) -> Result<Option<InstalledPack>, PullError> {
    let packs = list_installed_packs(home)?;
    resolve_installed_pack_reference(&packs, reference)
}

#[cfg(test)]
fn pull_model_pack_with_client<C: DownloadClient>(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    client: &mut C,
    options: PullOptions,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    pull_model_pack_with_client_and_cancel(
        resolved,
        home,
        client,
        options,
        progress,
        || false,
        || false,
    )
}

#[cfg(test)]
fn pull_model_pack_with_client_and_cancel<C: DownloadClient>(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    client: &mut C,
    options: PullOptions,
    progress: impl FnMut(PullProgress),
    should_cancel: impl Fn() -> bool,
    should_pause: impl Fn() -> bool,
) -> Result<InstalledPack, PullError> {
    pull_model_pack_with_client_sources_and_cancel(
        resolved,
        home,
        client,
        options,
        &[DownloadSource::Hf],
        progress,
        should_cancel,
        should_pause,
    )
}

fn pull_model_pack_with_client_sources_and_cancel<C: DownloadClient>(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    client: &mut C,
    options: PullOptions,
    sources: &[DownloadSource],
    mut progress: impl FnMut(PullProgress),
    should_cancel: impl Fn() -> bool,
    should_pause: impl Fn() -> bool,
) -> Result<InstalledPack, PullError> {
    let base_target = PullTarget::from_resolved(resolved)?;
    let source_targets = source_targets(resolved, &base_target, sources)?;
    let paths = pull_paths(home, &base_target)?;
    ensure_storage_dir_within_root(home, &paths)?;
    let _lock = PullLock::acquire(&paths.lock_path)?;

    if installed_matches(&base_target, &paths)? {
        let pack = write_installed_record(&base_target, &paths)?;
        progress(PullProgress::UsingInstalled {
            path: pack.path.clone(),
        });
        return Ok(pack);
    }

    let last_index = source_targets.len().saturating_sub(1);
    for (index, target) in source_targets.iter().enumerate() {
        let result = download_with_retries(
            target,
            &paths,
            client,
            options.clone(),
            &mut progress,
            &should_cancel,
            &should_pause,
        )
        .and_then(|downloaded| {
            if should_cancel() {
                cleanup_partial(&paths);
                return Err(PullError::Canceled {
                    reference: target.pull.clone(),
                });
            }
            verify_partial_and_install(
                target,
                &paths,
                Some(downloaded),
                &should_cancel,
                &mut progress,
            )
        });
        match result {
            Ok(pack) => return Ok(pack),
            Err(error) if index < last_index && is_source_fallback_error(&error) => {
                cleanup_partial(&paths);
            }
            Err(error) => return Err(error),
        }
    }
    Err(PullError::InvalidTarget {
        field: "sources",
        reason: "no usable download sources were available".to_string(),
    })
}

fn source_targets(
    resolved: &ResolvedCatalogPull,
    base_target: &PullTarget,
    sources: &[DownloadSource],
) -> Result<Vec<PullTarget>, PullError> {
    let default_sources = [DownloadSource::Hf];
    let sources = if sources.is_empty() {
        default_sources.as_slice()
    } else {
        sources
    };
    let mut targets = Vec::new();
    for source in sources {
        let Some(url) = source.url_for(resolved) else {
            continue;
        };
        ensure_https_url(&url)?;
        targets.push(base_target.with_url(url));
    }
    if targets.is_empty() {
        return Err(PullError::InvalidTarget {
            field: "sources",
            reason: "no usable download source URL was available for this catalog entry"
                .to_string(),
        });
    }
    Ok(targets)
}

fn download_with_retries<C: DownloadClient>(
    target: &PullTarget,
    paths: &PullPaths,
    client: &mut C,
    options: PullOptions,
    progress: &mut impl FnMut(PullProgress),
    should_cancel: &impl Fn() -> bool,
    should_pause: &impl Fn() -> bool,
) -> Result<DownloadedPartial, PullError> {
    let mut attempt = 0_usize;
    loop {
        if should_cancel() {
            cleanup_partial(paths);
            return Err(PullError::Canceled {
                reference: target.pull.clone(),
            });
        }
        if should_pause() {
            return Err(PullError::Paused {
                reference: target.pull.clone(),
            });
        }
        let resume_from = prepare_partial_for_resume(target, paths)?;
        if resume_from == target.size_bytes {
            let (_, sha256) = file_size_and_sha256(&paths.partial_path)?;
            return Ok(DownloadedPartial {
                bytes_done: resume_from,
                sha256,
            });
        }
        let needed = reserve_space_bytes(target.size_bytes.saturating_sub(resume_from));
        ensure_available_space(&paths.dir, needed, options.clone())?;
        let result = client
            .open(&target.url, (resume_from > 0).then_some(resume_from))
            .and_then(|response| {
                download_response(
                    target,
                    paths,
                    resume_from,
                    response,
                    &options,
                    progress,
                    should_cancel,
                    should_pause,
                )
            });
        match result {
            Ok(downloaded) => return Ok(downloaded),
            Err(error) if attempt < DOWNLOAD_MAX_RETRIES && is_retryable_download_error(&error) => {
                attempt += 1;
                std::thread::sleep(retry_backoff(attempt));
            }
            Err(error) => return Err(error),
        }
    }
}

fn download_response(
    target: &PullTarget,
    paths: &PullPaths,
    resume_from: u64,
    response: DownloadResponse,
    options: &PullOptions,
    progress: &mut impl FnMut(PullProgress),
    should_cancel: &impl Fn() -> bool,
    should_pause: &impl Fn() -> bool,
) -> Result<DownloadedPartial, PullError> {
    let append = match (resume_from, response.status) {
        (0, 200 | 206) => false,
        (_, 206) => true,
        (_, 200) => false,
        (_, status) => {
            return Err(PullError::UnexpectedStatus {
                url: target.url.clone(),
                status,
            });
        }
    };
    if append && !resume_content_range_matches(target, &response, resume_from) {
        let _ = fs::remove_file(&paths.partial_path);
        let _ = fs::remove_file(&paths.partial_meta_path);
        return Err(PullError::RestartedPartial {
            url: target.url.clone(),
        });
    }
    let actual_resume = if append { resume_from } else { 0 };
    if resume_from > 0 && !append {
        cleanup_partial(paths);
    }
    if let Some(content_length) = response.content_length {
        let expected_body = target.size_bytes.saturating_sub(actual_resume);
        if content_length != expected_body {
            cleanup_partial(paths);
            return Err(PullError::SizeMismatch {
                path: paths.partial_path.clone(),
                expected: expected_body,
                actual: content_length,
            });
        }
    }

    let mut hasher = Sha256::new();
    if append {
        hash_existing_partial(&paths.partial_path, &mut hasher)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&paths.partial_path)
        .map_err(|source| PullError::Io {
            path: paths.partial_path.clone(),
            source,
        })?;
    let mut bytes_done = actual_resume;
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(target, response.etag.clone(), bytes_done),
    )?;
    progress(PullProgress::DownloadStarted {
        bytes_total: target.size_bytes,
        resume_from: actual_resume,
    });

    let mut reader = response.reader;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    let mut next_meta_write = bytes_done.saturating_add(METADATA_WRITE_INTERVAL_BYTES);
    let mut low_speed = LowSpeedWindow::new();
    loop {
        if should_cancel() {
            cleanup_partial(paths);
            return Err(PullError::Canceled {
                reference: target.pull.clone(),
            });
        }
        if should_pause() {
            file.sync_all().map_err(|source| PullError::Io {
                path: paths.partial_path.clone(),
                source,
            })?;
            write_partial_meta(
                &paths.partial_meta_path,
                &PartialMeta::for_target(target, response.etag.clone(), bytes_done),
            )?;
            return Err(PullError::Paused {
                reference: target.pull.clone(),
            });
        }
        let read = reader
            .read(&mut buffer)
            .map_err(|source| map_download_read_error(target, &paths.partial_path, source))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|source| PullError::Io {
                path: paths.partial_path.clone(),
                source,
            })?;
        hasher.update(&buffer[..read]);
        bytes_done = bytes_done.saturating_add(read as u64);
        progress(PullProgress::Downloading {
            bytes_done,
            bytes_total: target.size_bytes,
        });
        low_speed.observe(target, bytes_done, read as u64, options)?;
        if bytes_done >= next_meta_write {
            write_partial_meta(
                &paths.partial_meta_path,
                &PartialMeta::for_target(target, response.etag.clone(), bytes_done),
            )?;
            next_meta_write = bytes_done.saturating_add(METADATA_WRITE_INTERVAL_BYTES);
        }
    }
    file.sync_all().map_err(|source| PullError::Io {
        path: paths.partial_path.clone(),
        source,
    })?;

    let digest = format!("{:x}", hasher.finalize());
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(target, response.etag, bytes_done),
    )?;
    Ok(DownloadedPartial {
        bytes_done,
        sha256: digest,
    })
}

fn cleanup_partial(paths: &PullPaths) {
    let _ = fs::remove_file(&paths.partial_path);
    let _ = fs::remove_file(&paths.partial_meta_path);
}

fn verify_partial_and_install(
    target: &PullTarget,
    paths: &PullPaths,
    downloaded: Option<DownloadedPartial>,
    should_cancel: &impl Fn() -> bool,
    mut progress: impl FnMut(PullProgress),
) -> Result<InstalledPack, PullError> {
    cancel_before_commit(target, paths, should_cancel)?;
    progress(PullProgress::Verifying {
        bytes_done: target.size_bytes,
    });
    let (actual_size, actual_sha) = match downloaded {
        Some(downloaded) => (downloaded.bytes_done, downloaded.sha256),
        None => file_size_and_sha256(&paths.partial_path)?,
    };
    cancel_before_commit(target, paths, should_cancel)?;
    if actual_size != target.size_bytes {
        cleanup_partial(paths);
        return Err(PullError::SizeMismatch {
            path: paths.partial_path.clone(),
            expected: target.size_bytes,
            actual: actual_size,
        });
    }
    if actual_sha != target.sha256 {
        cleanup_partial(paths);
        return Err(PullError::ShaMismatch {
            path: paths.partial_path.clone(),
            expected: target.sha256.clone(),
            actual: actual_sha,
        });
    }
    if let Err(error) = preflight_model_pack_for_install(&paths.partial_path) {
        cleanup_partial(paths);
        return Err(error);
    }
    cancel_before_commit(target, paths, should_cancel)?;
    remove_existing_final_pack(paths)?;
    fs::rename(&paths.partial_path, &paths.final_path).map_err(|source| PullError::Io {
        path: paths.final_path.clone(),
        source,
    })?;
    atomic_file::sync_parent_dir_best_effort(&paths.final_path);
    let _ = fs::remove_file(&paths.partial_meta_path);
    let pack = write_installed_record(target, paths)?;
    progress(PullProgress::Installed {
        path: pack.path.clone(),
    });
    Ok(pack)
}

fn remove_existing_final_pack(paths: &PullPaths) -> Result<(), PullError> {
    match fs::remove_file(&paths.final_path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        // On Windows, deleting a model that is currently mmap'd for inference
        // fails instead of succeeding lazily (as POSIX unlink does). Surface a
        // clear "model in use" message so re-pulling a changed version tells the
        // user to close OpenASR, rather than leaking a raw OS error code.
        Err(source) if is_file_in_use_error(&source) => Err(PullError::ModelInUse {
            path: paths.final_path.clone(),
            source,
        }),
        Err(source) => Err(PullError::Io {
            path: paths.final_path.clone(),
            source,
        }),
    }
}

/// True when an I/O error means the file cannot be replaced because it is still
/// open or memory-mapped by this or another process.
///
/// On Windows, replacing a model that is currently mmap'd for inference fails
/// with ERROR_USER_MAPPED_FILE (1224) or, for an open handle, with
/// ERROR_SHARING_VIOLATION (32). POSIX has no equivalent: `unlink`/`rename`
/// succeed even while a file is mapped (the inode lives until the last handle
/// closes), so this failure mode is Windows-only.
#[cfg(windows)]
fn is_file_in_use_error(source: &io::Error) -> bool {
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ERROR_USER_MAPPED_FILE: i32 = 1224;
    matches!(
        source.raw_os_error(),
        Some(ERROR_SHARING_VIOLATION | ERROR_USER_MAPPED_FILE)
    )
}

#[cfg(not(windows))]
fn is_file_in_use_error(_source: &io::Error) -> bool {
    false
}

fn cancel_before_commit(
    target: &PullTarget,
    paths: &PullPaths,
    should_cancel: &impl Fn() -> bool,
) -> Result<(), PullError> {
    if should_cancel() {
        cleanup_partial(paths);
        return Err(PullError::Canceled {
            reference: target.pull.clone(),
        });
    }
    Ok(())
}

fn installed_matches(target: &PullTarget, paths: &PullPaths) -> Result<bool, PullError> {
    if !paths.final_path.exists() {
        return Ok(false);
    }
    let (size, sha) = file_size_and_sha256(&paths.final_path)?;
    if size == target.size_bytes && sha == target.sha256 {
        preflight_gguf_package_contract(&paths.final_path)?;
        return Ok(true);
    }
    Ok(false)
}

fn prepare_partial_for_resume(target: &PullTarget, paths: &PullPaths) -> Result<u64, PullError> {
    if !paths.partial_path.exists() {
        return Ok(0);
    }
    let Ok(contents) = fs::read_to_string(&paths.partial_meta_path) else {
        let _ = fs::remove_file(&paths.partial_path);
        return Ok(0);
    };
    let meta: PartialMeta =
        serde_json::from_str(&contents).map_err(|source| PullError::ParseMeta {
            path: paths.partial_meta_path.clone(),
            source,
        })?;
    let partial_len = fs::metadata(&paths.partial_path)
        .map_err(|source| PullError::Io {
            path: paths.partial_path.clone(),
            source,
        })?
        .len();
    if !meta.matches_target(target)
        || meta.bytes_done != partial_len
        || partial_len > target.size_bytes
    {
        let _ = fs::remove_file(&paths.partial_path);
        let _ = fs::remove_file(&paths.partial_meta_path);
        return Ok(0);
    }
    Ok(partial_len)
}

fn write_installed_record(
    target: &PullTarget,
    paths: &PullPaths,
) -> Result<InstalledPack, PullError> {
    let pack = InstalledPack {
        model_id: target.model_id.clone(),
        display_name: target.display_name.clone(),
        quant: target.quant.clone(),
        suffix: target.suffix.clone(),
        pull: target.pull.clone(),
        filename: target.filename.clone(),
        path: paths.final_path.clone(),
        url: target.url.clone(),
        hf_revision: target.hf_revision.clone(),
        sha256: target.sha256.clone(),
        size_bytes: target.size_bytes,
        installed_at_unix_seconds: unix_seconds_now(),
        source: target.source.clone(),
    };
    let json = serde_json::to_string_pretty(&pack).map_err(|source| PullError::SerializeMeta {
        path: paths.installed_meta_path.clone(),
        source,
    })?;
    write_json_atomic(&paths.installed_meta_path, &format!("{json}\n"))?;
    Ok(pack)
}

fn ensure_storage_dir_within_root(home: &Path, paths: &PullPaths) -> Result<(), PullError> {
    let root = models_root(home);
    let Some(model_dir) = paths.dir.parent() else {
        return Err(PullError::UnsafeStoragePath {
            path: paths.dir.clone(),
        });
    };
    for path in [&root, model_dir, paths.dir.as_path()] {
        reject_symlink(path)?;
    }
    fs::create_dir_all(&paths.dir).map_err(|source| PullError::CreateDir {
        path: paths.dir.clone(),
        source,
    })?;
    for path in [&root, model_dir, paths.dir.as_path()] {
        reject_symlink(path)?;
    }
    let canonical_root = root.canonicalize().map_err(|source| PullError::Io {
        path: root.clone(),
        source,
    })?;
    let canonical_dir = paths.dir.canonicalize().map_err(|source| PullError::Io {
        path: paths.dir.clone(),
        source,
    })?;
    if !canonical_dir.starts_with(&canonical_root) {
        return Err(PullError::UnsafeStoragePath {
            path: paths.dir.clone(),
        });
    }
    Ok(())
}

fn reject_symlink(path: &Path) -> Result<(), PullError> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.file_type().is_symlink() {
        return Err(PullError::UnsafeStoragePath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn pull_paths(home: &Path, target: &PullTarget) -> Result<PullPaths, PullError> {
    validate_safe_relative_path("model id", &target.model_id).map_err(|reason| {
        PullError::InvalidTarget {
            field: "model_id",
            reason,
        }
    })?;
    validate_safe_relative_path("quant", &target.quant).map_err(|reason| {
        PullError::InvalidTarget {
            field: "quant",
            reason,
        }
    })?;
    validate_safe_relative_path("filename", &target.filename).map_err(|reason| {
        PullError::InvalidTarget {
            field: "filename",
            reason,
        }
    })?;
    let dir = models_root(home).join(&target.model_id).join(&target.quant);
    let final_path = dir.join(&target.filename);
    Ok(PullPaths {
        partial_path: dir.join(format!("{}.partial", target.filename)),
        partial_meta_path: dir.join(format!("{}.partial.meta.json", target.filename)),
        installed_meta_path: dir.join("installed.json"),
        lock_path: dir.join(format!("{}.lock", target.filename)),
        dir,
        final_path,
    })
}

fn models_root(home: &Path) -> PathBuf {
    home.join("models")
}

fn ensure_https_url(url: &str) -> Result<(), PullError> {
    if url.starts_with("https://") {
        Ok(())
    } else {
        Err(PullError::NonHttpsUrl {
            url: url.to_string(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedContentRange {
    start: u64,
    end: u64,
    total: Option<u64>,
}

fn resume_content_range_matches(
    target: &PullTarget,
    response: &DownloadResponse,
    resume_from: u64,
) -> bool {
    let Some(content_range) = response.content_range.as_deref() else {
        return false;
    };
    let Some(parsed) = parse_content_range(content_range) else {
        return false;
    };
    let Some(expected_end) = target.size_bytes.checked_sub(1) else {
        return false;
    };
    parsed.start == resume_from
        && parsed.end == expected_end
        && parsed
            .total
            .map(|total| total == target.size_bytes)
            .unwrap_or(true)
}

fn parse_content_range(value: &str) -> Option<ParsedContentRange> {
    let value = value.trim();
    let value = value.strip_prefix("bytes ")?;
    let (span, total) = value.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    let start = start.trim().parse().ok()?;
    let end = end.trim().parse().ok()?;
    if end < start {
        return None;
    }
    let total = match total.trim() {
        "*" => None,
        value => Some(value.parse().ok()?),
    };
    Some(ParsedContentRange { start, end, total })
}

struct LowSpeedWindow {
    started_at: Instant,
    bytes_read: u64,
}

impl LowSpeedWindow {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            bytes_read: 0,
        }
    }

    fn observe(
        &mut self,
        target: &PullTarget,
        bytes_done: u64,
        bytes_read: u64,
        options: &PullOptions,
    ) -> Result<(), PullError> {
        if options.low_speed_min_bytes == 0 || bytes_done >= target.size_bytes {
            return Ok(());
        }
        self.bytes_read = self.bytes_read.saturating_add(bytes_read);
        let elapsed = self.started_at.elapsed();
        if elapsed < options.low_speed_timeout {
            return Ok(());
        }
        if self.bytes_read < options.low_speed_min_bytes {
            return Err(PullError::Http {
                url: target.url.clone(),
                message: format!(
                    "download stalled: received {} bytes in {:.1}s, below the {} byte minimum",
                    self.bytes_read,
                    elapsed.as_secs_f64(),
                    options.low_speed_min_bytes
                ),
            });
        }
        self.started_at = Instant::now();
        self.bytes_read = 0;
        Ok(())
    }
}

fn map_download_read_error(target: &PullTarget, path: &Path, source: io::Error) -> PullError {
    if source.kind() == io::ErrorKind::TimedOut {
        return PullError::Http {
            url: target.url.clone(),
            message: format!("download stalled while reading response body: {source}"),
        };
    }
    PullError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn ensure_available_space(
    path: &Path,
    needed_bytes: u64,
    options: PullOptions,
) -> Result<(), PullError> {
    let available = options
        .available_space_override
        .or_else(|| available_space_bytes(path));
    if let Some(available_bytes) = available
        && available_bytes < needed_bytes
    {
        return Err(PullError::InsufficientSpace {
            path: path.to_path_buf(),
            needed_bytes,
            available_bytes,
        });
    }
    Ok(())
}

/// Best-effort free space (in bytes) on the filesystem containing `path`.
/// `None` means the platform/probe could not determine it -- callers should
/// treat that as "unknown" and stay permissive, matching how
/// [`ensure_available_space`] treats a `None` probe for model-pack pulls.
/// Exposed for other crates (e.g. `openasr-server`'s streaming upload path)
/// that need the same disk-headroom check pulls already rely on.
pub fn available_disk_space_bytes(path: &Path) -> Option<u64> {
    available_space_bytes(path)
}

fn file_size_and_sha256(path: &Path) -> Result<(u64, String), PullError> {
    let mut file = File::open(path).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let total = hash_file_range(&mut file, &mut hasher, None).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok((total, format!("{:x}", hasher.finalize())))
}

fn hash_file_range(file: &mut File, hasher: &mut Sha256, max: Option<u64>) -> io::Result<u64> {
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        let read_limit = max
            .map(|max| max.saturating_sub(total).min(buffer.len() as u64) as usize)
            .unwrap_or(buffer.len());
        if read_limit == 0 {
            break;
        }
        let read = file.read(&mut buffer[..read_limit])?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total = total.saturating_add(read as u64);
    }
    Ok(total)
}

fn hash_existing_partial(path: &Path, hasher: &mut Sha256) -> Result<(), PullError> {
    let mut file = File::open(path).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    hash_file_range(&mut file, hasher, None)
        .map(|_| ())
        .map_err(|source| PullError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn write_partial_meta(path: &Path, meta: &PartialMeta) -> Result<(), PullError> {
    let json = serde_json::to_string_pretty(meta).map_err(|source| PullError::SerializeMeta {
        path: path.to_path_buf(),
        source,
    })?;
    write_json_atomic(path, &format!("{json}\n"))
}

fn write_json_atomic(path: &Path, contents: &str) -> Result<(), PullError> {
    atomic_file::write_file_atomically(path, contents.as_bytes()).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })
}

impl PullTarget {
    fn from_resolved(resolved: &ResolvedCatalogPull) -> Result<Self, PullError> {
        validate_sha256("sha256", &resolved.sha256).map_err(|reason| PullError::InvalidTarget {
            field: "sha256",
            reason,
        })?;
        if resolved.size_bytes == 0 {
            return Err(PullError::InvalidTarget {
                field: "size_bytes",
                reason: "size_bytes must be greater than zero".to_string(),
            });
        }
        if !resolved
            .filename
            .ends_with(&format!(".{OPENASR_RUNTIME_PACK_EXTENSION}"))
            || resolved.filename.contains('/')
            || resolved.filename.contains('\\')
        {
            return Err(PullError::InvalidTarget {
                field: "filename",
                reason: format!(
                    "filename must be a local basename ending with .{OPENASR_RUNTIME_PACK_EXTENSION}"
                ),
            });
        }
        Ok(Self {
            model_id: resolved.model_id.clone(),
            display_name: resolved.display_name.clone(),
            quant: resolved.quant.clone(),
            suffix: resolved.suffix.clone(),
            pull: resolved.pull.clone(),
            filename: resolved.filename.clone(),
            url: resolved.url.clone(),
            hf_revision: resolved.hf_revision.clone(),
            sha256: resolved.sha256.clone(),
            size_bytes: resolved.size_bytes,
            source: None,
        })
    }

    fn with_url(&self, url: String) -> Self {
        Self {
            url,
            ..self.clone()
        }
    }

    fn with_source(&self, source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            ..self.clone()
        }
    }
}

impl PartialMeta {
    fn for_target(target: &PullTarget, etag: Option<String>, bytes_done: u64) -> Self {
        Self {
            model_id: target.model_id.clone(),
            quant: target.quant.clone(),
            filename: target.filename.clone(),
            url: target.url.clone(),
            hf_revision: target.hf_revision.clone(),
            sha256: target.sha256.clone(),
            size_bytes: target.size_bytes,
            etag,
            bytes_done,
            updated_at_unix_seconds: unix_seconds_now(),
        }
    }

    /// Partial identity is the content identity (pack + revision + digest),
    /// never the transport URL: mirror sources serve the same bytes under
    /// different hosts, and the source order can change between runs (locale,
    /// pinned source), so matching on URL would throw away resumable bytes.
    /// Content integrity is still enforced by the final sha256 verification.
    fn matches_target(&self, target: &PullTarget) -> bool {
        self.model_id == target.model_id
            && self.quant == target.quant
            && self.filename == target.filename
            && self.hf_revision == target.hf_revision
            && self.sha256 == target.sha256
            && self.size_bytes == target.size_bytes
    }
}

struct PullLock {
    path: PathBuf,
}

impl PullLock {
    fn acquire(path: &Path) -> Result<Self, PullError> {
        let mut stale_recoveries = 0_usize;
        let mut last_stale_error = None;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    writeln!(file, "pid={}", std::process::id()).map_err(|source| {
                        PullError::LockIo {
                            path: path.to_path_buf(),
                            source,
                        }
                    })?;
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                    if !lock_is_stale(path) {
                        return Err(PullError::LockHeld {
                            path: path.to_path_buf(),
                        });
                    }
                    if stale_recoveries >= LOCK_STALE_RECOVERY_ATTEMPTS {
                        let source = last_stale_error.unwrap_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::AlreadyExists,
                                format!(
                                    "stale pull lock persisted after {LOCK_STALE_RECOVERY_ATTEMPTS} recovery attempts"
                                ),
                            )
                        });
                        return Err(PullError::LockIo {
                            path: path.to_path_buf(),
                            source,
                        });
                    }
                    stale_recoveries += 1;
                    match fs::remove_file(path) {
                        Ok(()) => {}
                        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
                        Err(source) => {
                            last_stale_error = Some(source);
                        }
                    }
                }
                Err(source) => {
                    return Err(PullError::LockIo {
                        path: path.to_path_buf(),
                        source,
                    });
                }
            }
        }
    }
}

impl Drop for PullLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn lock_is_stale(path: &Path) -> bool {
    if lock_owner_is_gone(path) {
        return true;
    }
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return false;
    };
    modified
        .elapsed()
        .is_ok_and(|elapsed| elapsed > LOCK_STALE_AFTER)
}

#[cfg(unix)]
fn lock_owner_is_gone(path: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = contents
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|value| value.trim().parse::<libc::pid_t>().ok())
    else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    let result = unsafe { libc::kill(pid, 0) };
    if result == 0 {
        return false;
    }
    io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(windows)]
fn lock_owner_is_gone(path: &Path) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // STILL_ACTIVE (STATUS_PENDING): a process that has not exited reports this as
    // its "exit code". Any other value means it has terminated.
    const STILL_ACTIVE: u32 = 259;

    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = contents
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|value| value.trim().parse::<u32>().ok())
    else {
        return false;
    };
    if pid == 0 {
        return false;
    }
    // SAFETY: OpenProcess with a query-only access right is a read-only probe.
    //
    // A null handle means the pid no longer maps to any process object, so the
    // owner is gone. But a non-null handle is NOT proof of life: a process that
    // has exited keeps its pid reserved as long as anyone still holds an open
    // handle to it (in production the desktop's DaemonSupervisor holds the
    // sidecar's child handle, so a crashed sidecar lingers as such a zombie).
    // Decide liveness by the exit code, not by OpenProcess succeeding: only
    // STILL_ACTIVE means the owner is truly running and the lock must be honored.
    // Matches the spirit of the unix `libc::kill(pid, 0)` path, including its
    // accepted pid-reuse window.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return true;
        }
        let mut exit_code: u32 = 0;
        let queried = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        // queried == 0 → status unreadable; be conservative and treat as live.
        queried != 0 && exit_code != STILL_ACTIVE
    }
}

#[cfg(not(any(unix, windows)))]
fn lock_owner_is_gone(_path: &Path) -> bool {
    false
}

const DOWNLOAD_MAX_REDIRECTS: usize = 10;

impl HttpDownloadClient {
    fn new() -> Result<Self, PullError> {
        // The downloader follows redirects manually (see `open`) so it can route
        // the Hugging Face CDN hop through the mirror; hence a no-redirect client.
        let client = http::blocking_client_no_redirect(HTTP_CONNECT_TIMEOUT, HTTP_STALL_TIMEOUT)
            .map_err(|source| PullError::Http {
                url: "<client>".to_string(),
                message: http::error_message(&source),
            })?;
        Ok(Self {
            client,
            hf_token: hf_token_from_env(),
        })
    }
}

/// Optional Hugging Face access token from `OPENASR_HF_TOKEN`, trimmed; `None`
/// when unset or empty. The desktop app injects it at daemon launch so model
/// pulls can authenticate under shared-IP rate limits. Never read on any
/// fail-closed local path: an unset var simply means anonymous downloads.
fn hf_token_from_env() -> Option<String> {
    std::env::var("OPENASR_HF_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Whether the optional HF bearer token may be attached to a request to `host`.
/// Restricted to `huggingface.co` so the credential never reaches a CDN, mirror,
/// or attacker-controlled redirect target.
fn hf_token_allowed_for_host(host: Option<&str>) -> bool {
    host == Some("huggingface.co")
}

impl DownloadClient for HttpDownloadClient {
    fn open(&mut self, url: &str, range_start: Option<u64>) -> Result<DownloadResponse, PullError> {
        let mut current = url.to_string();
        let mut redirect_cookies: Vec<RedirectCookie> = Vec::new();
        for _ in 0..=DOWNLOAD_MAX_REDIRECTS {
            let current_host = redirect_url_host(&current);
            let mut request = self
                .client
                .get(current.as_str())
                .header(reqwest::header::USER_AGENT, DOWNLOAD_USER_AGENT);
            // Attach the optional HF token ONLY to huggingface.co -- never to the
            // CDN or mirror host a redirect points at, so the bearer credential
            // can't leak across origins (same scoping as redirect cookies below).
            if let Some(token) = self.hf_token.as_deref()
                && hf_token_allowed_for_host(current_host.as_deref())
            {
                request = request.bearer_auth(token);
            }
            // Only replay cookies the same host set: a cookie from huggingface.co
            // must not follow a redirect to a CDN or attacker host.
            if let Some(host) = current_host.as_deref() {
                let scoped = cookies_for_host(&redirect_cookies, host);
                if !scoped.is_empty() {
                    request = request.header(reqwest::header::COOKIE, scoped.join("; "));
                }
            }
            if let Some(start) = range_start {
                request = request.header(reqwest::header::RANGE, format!("bytes={start}-"));
            }
            let response = request.send().map_err(|source| PullError::Http {
                url: url.to_string(),
                message: http::error_message(&source),
            })?;
            let status = response.status();
            if status.is_redirection()
                && let Some(location) = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|value| value.to_str().ok())
            {
                if let Some(host) = current_host.as_deref() {
                    capture_redirect_cookies(response.headers(), host, &mut redirect_cookies);
                }
                current = resolve_redirect_location(&current, location)?;
                continue;
            }

            let status = status.as_u16();
            let content_length = response.content_length();
            let etag = response
                .headers()
                .get(reqwest::header::ETAG)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            let content_range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            return Ok(DownloadResponse {
                status,
                content_length,
                content_range,
                etag,
                reader: Box::new(response),
            });
        }
        Err(PullError::Http {
            url: url.to_string(),
            message: format!("exceeded {DOWNLOAD_MAX_REDIRECTS} redirects while downloading"),
        })
    }
}

/// Resolve a (possibly relative) `Location` header against the URL it came from.
/// If the selected source is an HF mirror, keep known HF CDN hops on that same
/// mirror endpoint.
fn resolve_redirect_location(current: &str, location: &str) -> Result<String, PullError> {
    let resolved = reqwest::Url::parse(current)
        .and_then(|base| base.join(location))
        .map_err(|source| PullError::Http {
            url: current.to_string(),
            message: format!("invalid redirect location '{location}': {source}"),
        })?;
    let endpoint = mirror_endpoint_for_current_url(current);
    let target =
        http::apply_hf_mirror_redirect_with_endpoint(resolved.as_str(), endpoint.as_deref());
    // The initial URL is https-checked before download; redirect targets were
    // not, so a 30x to http:// would silently downgrade the transfer to
    // cleartext. Enforce https on every hop.
    ensure_https_url(&target)?;
    Ok(target)
}

/// A redirect-set cookie scoped to the host that set it. Cookies are host
/// specific (RFC 6265): replaying a cookie set by `huggingface.co` to a CDN or
/// an attacker-controlled redirect host would leak session/auth state across
/// origins, so the jar records the setting host and only the matching host's
/// cookies are sent back (see `cookies_for_host`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct RedirectCookie {
    host: String,
    cookie: String,
}

/// Host of a URL, lowercased, for cookie scoping. `None` when the URL does not
/// parse or carries no host (such URLs never receive cookies).
fn redirect_url_host(url: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_ascii_lowercase))
}

/// The `name=value` cookies previously set by `host`, in jar order — the only
/// cookies allowed onto a request to that host.
fn cookies_for_host<'a>(jar: &'a [RedirectCookie], host: &str) -> Vec<&'a str> {
    jar.iter()
        .filter(|entry| entry.host == host)
        .map(|entry| entry.cookie.as_str())
        .collect()
}

fn capture_redirect_cookies(
    headers: &reqwest::header::HeaderMap,
    host: &str,
    jar: &mut Vec<RedirectCookie>,
) {
    for value in headers.get_all(reqwest::header::SET_COOKIE) {
        let Ok(raw) = value.to_str() else {
            continue;
        };
        let Some(cookie) = raw.split(';').next().map(str::trim).filter(|value| {
            let Some((name, cookie_value)) = value.split_once('=') else {
                return false;
            };
            !name.trim().is_empty() && !cookie_value.trim().is_empty()
        }) else {
            continue;
        };
        let name = cookie
            .split_once('=')
            .map(|(name, _)| name)
            .unwrap_or(cookie);
        // Dedup by (host, name): a later Set-Cookie for the same name on the
        // same host replaces the earlier value.
        if let Some(existing) = jar.iter_mut().find(|entry| {
            entry.host == host && entry.cookie.split_once('=').map(|(n, _)| n) == Some(name)
        }) {
            existing.cookie.clear();
            existing.cookie.push_str(cookie);
        } else {
            jar.push(RedirectCookie {
                host: host.to_string(),
                cookie: cookie.to_string(),
            });
        }
    }
}

fn is_retryable_download_error(error: &PullError) -> bool {
    match error {
        PullError::Http { .. }
        | PullError::Io { .. }
        | PullError::RestartedPartial { .. }
        | PullError::SizeMismatch { .. } => true,
        PullError::UnexpectedStatus { status, .. } => *status >= 500,
        _ => false,
    }
}

fn is_source_fallback_error(error: &PullError) -> bool {
    match error {
        PullError::Http { .. }
        | PullError::Io { .. }
        | PullError::RestartedPartial { .. }
        | PullError::SizeMismatch { .. }
        | PullError::ShaMismatch { .. }
        | PullError::GgufPreflight { .. }
        | PullError::BackendFilePreflight { .. }
        | PullError::RuntimeValidation { .. } => true,
        PullError::UnexpectedStatus { status, .. } => *status >= 500,
        _ => false,
    }
}

fn mirror_endpoint_for_current_url(current: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(current).ok()?;
    let host = parsed.host_str()?;
    if matches!(
        host,
        "huggingface.co" | "modelscope.cn" | "www.modelscope.cn"
    ) {
        return None;
    }
    Some(format!("{}://{}", parsed.scheme(), host))
}

fn retry_backoff(attempt: usize) -> Duration {
    let millis = 250_u64.saturating_mul(1_u64 << attempt.min(5));
    Duration::from_millis(millis.min(5_000))
}

fn reserve_space_bytes(bytes: u64) -> u64 {
    let reserved = (u128::from(bytes) * 11).div_ceil(10);
    u64::try_from(reserved).unwrap_or(u64::MAX)
}

/// Full pre-install validation every model pack must pass after download (or
/// local import) and before it is committed into the local model store:
///
/// 1. [`preflight_gguf_package_contract`] — structural GGUF scan plus the
///    `.oasr` v1 required-metadata gate (`openasr.package.version = "1"`).
/// 2. Runtime-source path validation (magic probe, fail-closed sandbox rules).
/// 3. The native runtime model-pack contract (family adapter selection or a
///    registered non-ASR pack contract such as diarization/translation packs).
///
/// Importer tests reuse this exact function so a pack a family importer can
/// build but `openasr pull` would reject can never ship.
pub fn preflight_model_pack_for_install(path: &Path) -> Result<(), PullError> {
    preflight_gguf_package_contract(path)?;
    validate_ggml_runtime_source_path(path).map_err(|source| PullError::RuntimeValidation {
        path: path.to_path_buf(),
        reason: source.to_string(),
    })?;
    validate_native_runtime_model_pack_contract(path).map_err(|reason| {
        PullError::RuntimeValidation {
            path: path.to_path_buf(),
            reason,
        }
    })?;
    Ok(())
}

/// The binary shape a downloaded backend-pack file must have, for the magic-byte
/// preflight ([`preflight_backend_file`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFileFormat {
    /// A native shared library — the `ggml-<vendor>` plugin or a runtime
    /// satellite. Accepts PE (Windows), ELF (Linux), or Mach-O (macOS).
    NativeLibrary,
    /// A zip archive extracted post-verify (e.g. the rocBLAS Tensile set).
    ZipArchive,
}

/// Bytes read from the file head for the magic check. PE places its `PE\0\0`
/// signature at `e_lfanew`, comfortably inside the first 4 KiB for any real DLL.
const BACKEND_PREFLIGHT_HEAD_BYTES: u64 = 4096;

/// Preflight a downloaded backend-pack file by its magic bytes BEFORE it is
/// installed or loaded — the backend analogue of [`preflight_gguf_package_contract`]
/// for the model path. sha256 is the integrity boundary; this gate fails closed
/// on the common corruption mode a hash alone still accepts only after the fact:
/// a mirror that returns a 404 HTML page, a captive-portal redirect, or a
/// truncated/garbage body instead of the binary. Library files must be a
/// recognized native shared-library format (PE/ELF/Mach-O); archives must be a
/// PKZIP container. Only the file head is read.
pub fn preflight_backend_file(path: &Path, format: BackendFileFormat) -> Result<(), PullError> {
    let file = File::open(path).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut head = Vec::with_capacity(BACKEND_PREFLIGHT_HEAD_BYTES as usize);
    file.take(BACKEND_PREFLIGHT_HEAD_BYTES)
        .read_to_end(&mut head)
        .map_err(|source| PullError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let recognized = match format {
        BackendFileFormat::NativeLibrary => is_native_shared_library(&head),
        BackendFileFormat::ZipArchive => is_zip_archive(&head),
    };
    if recognized {
        return Ok(());
    }
    let preview: String = head
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    Err(PullError::BackendFilePreflight {
        path: path.to_path_buf(),
        reason: format!(
            "expected {format:?} magic bytes but the file head was [{preview}] ({} bytes read)",
            head.len()
        ),
    })
}

/// PE (`MZ` + `PE\0\0` at `e_lfanew`), ELF, or Mach-O (thin/fat, either endian).
fn is_native_shared_library(head: &[u8]) -> bool {
    is_pe(head) || is_elf(head) || is_mach_o(head)
}

fn is_pe(head: &[u8]) -> bool {
    if head.len() < 0x40 || &head[..2] != b"MZ" {
        return false;
    }
    let e_lfanew = u32::from_le_bytes([head[0x3C], head[0x3D], head[0x3E], head[0x3F]]) as usize;
    matches!(head.get(e_lfanew..e_lfanew + 4), Some(b"PE\0\0"))
}

fn is_elf(head: &[u8]) -> bool {
    head.starts_with(&[0x7F, b'E', b'L', b'F'])
}

fn is_mach_o(head: &[u8]) -> bool {
    // MH_MAGIC/MH_CIGAM (32), MH_MAGIC_64/MH_CIGAM_64, FAT_MAGIC/FAT_CIGAM.
    const MACH_O_MAGICS: [[u8; 4]; 6] = [
        [0xFE, 0xED, 0xFA, 0xCE],
        [0xCE, 0xFA, 0xED, 0xFE],
        [0xFE, 0xED, 0xFA, 0xCF],
        [0xCF, 0xFA, 0xED, 0xFE],
        [0xCA, 0xFE, 0xBA, 0xBE],
        [0xBE, 0xBA, 0xFE, 0xCA],
    ];
    MACH_O_MAGICS.iter().any(|magic| head.starts_with(magic))
}

fn is_zip_archive(head: &[u8]) -> bool {
    // Local file header, empty archive (EOCD), or spanned-archive marker.
    head.starts_with(b"PK\x03\x04")
        || head.starts_with(b"PK\x05\x06")
        || head.starts_with(b"PK\x07\x08")
}

/// Record of an installed backend plugin pack, persisted as `backend.json` in
/// the pack directory so a re-pull short-circuits and doctor can report it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledBackend {
    pub backend_id: String,
    pub vendor: String,
    pub version: String,
    pub dir: PathBuf,
    pub plugin_filename: String,
    pub files: Vec<String>,
    pub installed_at_unix_seconds: u64,
}

fn backend_vendor_dirname(vendor: CatalogBackendVendor) -> &'static str {
    match vendor {
        CatalogBackendVendor::Cpu => "cpu",
        CatalogBackendVendor::Vulkan => "vulkan",
        CatalogBackendVendor::Hip => "hip",
        CatalogBackendVendor::Cuda => "cuda",
    }
}

fn backend_file_format(role: CatalogBackendFileRole) -> BackendFileFormat {
    match role {
        CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Runtime => {
            BackendFileFormat::NativeLibrary
        }
        CatalogBackendFileRole::Archive => BackendFileFormat::ZipArchive,
    }
}

/// Download, verify, and install a resolved backend plugin pack into
/// `OPENASR_HOME/backends/<vendor>/<version>/`, where [`crate::ggml_runtime`]'s
/// `ensure_backends_loaded` later registers it with the ggml registry. Each file
/// is streamed to a `.partial`, sha256-verified, magic-preflighted by role
/// ([`preflight_backend_file`]), and atomically placed; archive files extract
/// into their `extract_subdir` (zip-slip-safe). Idempotent: a complete prior
/// install (marker + plugin present) short-circuits. The pack dir is locked for
/// the duration so concurrent pulls of the same pack serialize.
///
/// Index-agnostic: it consumes a [`ResolvedCatalogBackendPull`] regardless of
/// whether the feeder is the signed catalog or a GitHub-Releases manifest.
pub fn install_backend_pack(
    resolved: &ResolvedCatalogBackendPull,
    home: impl AsRef<Path>,
    progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, PullError> {
    let mut client = HttpDownloadClient::new()?;
    install_backend_pack_with_client(resolved, home.as_ref(), &mut client, progress)
}

fn install_backend_pack_with_client<C: DownloadClient>(
    resolved: &ResolvedCatalogBackendPull,
    home: &Path,
    client: &mut C,
    mut progress: impl FnMut(PullProgress),
) -> Result<InstalledBackend, PullError> {
    let vendor = backend_vendor_dirname(resolved.vendor);
    // Defense in depth (the catalog is signed, but never join an unvalidated
    // component): the version must be a single safe path segment.
    if resolved.version.is_empty()
        || resolved
            .version
            .split(['/', '\\'])
            .any(|component| component.is_empty() || component == "..")
        || resolved.version.contains(':')
    {
        return Err(PullError::InvalidTarget {
            field: "backend.version",
            reason: format!("'{}' is not a safe path segment", resolved.version),
        });
    }
    let plugin_filename = resolved
        .files
        .iter()
        .find(|file| file.role == CatalogBackendFileRole::Plugin)
        .map(|file| file.filename.clone())
        .ok_or(PullError::InvalidTarget {
            field: "backend.files",
            reason: "pack declares no plugin file".to_string(),
        })?;

    let dir = home.join("backends").join(vendor).join(&resolved.version);
    fs::create_dir_all(&dir).map_err(|source| PullError::Io {
        path: dir.clone(),
        source,
    })?;
    let _lock = PullLock::acquire(&dir.join("pull.lock"))?;

    let marker_path = dir.join("backend.json");
    if marker_path.is_file()
        && dir.join(&plugin_filename).is_file()
        && let Ok(text) = fs::read_to_string(&marker_path)
        && let Ok(existing) = serde_json::from_str::<InstalledBackend>(&text)
    {
        progress(PullProgress::UsingInstalled { path: dir.clone() });
        return Ok(existing);
    }

    let mut installed_files = Vec::new();
    for file in &resolved.files {
        let dest = dir.join(&file.filename);
        download_backend_file(client, file, &dest, &mut progress)?;
        preflight_backend_file(&dest, backend_file_format(file.role))?;
        if file.role == CatalogBackendFileRole::Archive {
            let subdir = file.extract_subdir.as_deref().unwrap_or("");
            extract_backend_archive(&dest, &dir, subdir)?;
        }
        installed_files.push(file.filename.clone());
    }

    let record = InstalledBackend {
        backend_id: resolved.backend_id.clone(),
        vendor: vendor.to_string(),
        version: resolved.version.clone(),
        dir: dir.clone(),
        plugin_filename,
        files: installed_files,
        installed_at_unix_seconds: unix_seconds_now(),
    };
    let json =
        serde_json::to_string_pretty(&record).map_err(|source| PullError::SerializeMeta {
            path: marker_path.clone(),
            source,
        })?;
    write_json_atomic(&marker_path, &format!("{json}\n"))?;
    Ok(record)
}

fn download_backend_file<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    dest: &Path,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    let response = client.open(&file.url, None)?;
    if response.status != 200 {
        return Err(PullError::UnexpectedStatus {
            url: file.url.clone(),
            status: response.status,
        });
    }
    if let Some(content_length) = response.content_length
        && content_length != file.size_bytes
    {
        return Err(PullError::SizeMismatch {
            path: dest.to_path_buf(),
            expected: file.size_bytes,
            actual: content_length,
        });
    }
    let partial = dest.with_extension("partial");
    let mut hasher = Sha256::new();
    let mut out = File::create(&partial).map_err(|source| PullError::Io {
        path: partial.clone(),
        source,
    })?;
    let mut reader = response.reader;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    let mut bytes_done = 0_u64;
    loop {
        let read = reader.read(&mut buffer).map_err(|source| PullError::Io {
            path: partial.clone(),
            source,
        })?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])
            .map_err(|source| PullError::Io {
                path: partial.clone(),
                source,
            })?;
        hasher.update(&buffer[..read]);
        bytes_done = bytes_done.saturating_add(read as u64);
        progress(PullProgress::Downloading {
            bytes_done,
            bytes_total: file.size_bytes,
        });
    }
    out.sync_all().map_err(|source| PullError::Io {
        path: partial.clone(),
        source,
    })?;
    drop(out);
    let actual = format!("{:x}", hasher.finalize());
    if actual != file.sha256 {
        let _ = fs::remove_file(&partial);
        return Err(PullError::ShaMismatch {
            path: dest.to_path_buf(),
            expected: file.sha256.clone(),
            actual,
        });
    }
    fs::rename(&partial, dest).map_err(|source| PullError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Extract a verified zip archive into `<pack_dir>/<subdir>`, rejecting any entry
/// whose path escapes the destination (zip-slip). The archive's own sha256 was
/// already checked by the caller; this guards only the per-entry paths.
fn extract_backend_archive(
    zip_path: &Path,
    pack_dir: &Path,
    subdir: &str,
) -> Result<(), PullError> {
    let dest_root = pack_dir.join(subdir);
    let file = File::open(zip_path).map_err(|source| PullError::Io {
        path: zip_path.to_path_buf(),
        source,
    })?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|error| PullError::BackendFilePreflight {
            path: zip_path.to_path_buf(),
            reason: format!("could not open zip archive: {error}"),
        })?;
    for index in 0..archive.len() {
        let mut entry =
            archive
                .by_index(index)
                .map_err(|error| PullError::BackendFilePreflight {
                    path: zip_path.to_path_buf(),
                    reason: format!("could not read zip entry {index}: {error}"),
                })?;
        // enclosed_name() is None for any absolute / `..`-traversal path.
        let Some(relative) = entry.enclosed_name().map(|path| path.to_path_buf()) else {
            return Err(PullError::BackendFilePreflight {
                path: zip_path.to_path_buf(),
                reason: format!("zip entry '{}' escapes the extraction dir", entry.name()),
            });
        };
        let out_path = dest_root.join(&relative);
        if entry.is_dir() {
            fs::create_dir_all(&out_path).map_err(|source| PullError::Io {
                path: out_path.clone(),
                source,
            })?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|source| PullError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut out = File::create(&out_path).map_err(|source| PullError::Io {
            path: out_path.clone(),
            source,
        })?;
        io::copy(&mut entry, &mut out).map_err(|source| PullError::Io {
            path: out_path.clone(),
            source,
        })?;
    }
    Ok(())
}

/// Structural GGUF preflight plus the `.oasr` v1 required-metadata gate: the
/// pack must carry `openasr.package.version = "1"` or the pull fails closed.
pub fn preflight_gguf_package_contract(path: &Path) -> Result<(), PullError> {
    let file = File::open(path).map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| PullError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut reader = GgufPreflightReader::new(file, file_len, path);
    reader.scan().map_err(|reason| PullError::GgufPreflight {
        path: path.to_path_buf(),
        reason,
    })
}

struct GgufPreflightReader<'a> {
    file: BufReader<File>,
    file_len: u64,
    cursor: u64,
    path: &'a Path,
    alignment: u64,
    package_version: Option<String>,
}

impl<'a> GgufPreflightReader<'a> {
    fn new(file: File, file_len: u64, path: &'a Path) -> Self {
        Self {
            file: BufReader::new(file),
            file_len,
            cursor: 0,
            path,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            package_version: None,
        }
    }

    fn scan(&mut self) -> Result<(), String> {
        let mut magic = [0_u8; 4];
        self.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(format!("expected GGUF magic in '{}'", self.path.display()));
        }
        let version = self.read_u32()?;
        if version != 3 {
            return Err(format!("unsupported GGUF version {version}; expected 3"));
        }
        let tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;
        if tensor_count == 0 || tensor_count > MAX_GGUF_TENSORS {
            return Err(format!(
                "tensor count {tensor_count} is outside supported bounds"
            ));
        }
        if kv_count > MAX_GGUF_METADATA_ENTRIES {
            return Err(format!(
                "metadata entry count {kv_count} is outside supported bounds"
            ));
        }
        for _ in 0..kv_count {
            self.read_metadata_entry()?;
        }
        let mut tensor_spans = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        for _ in 0..tensor_count {
            let _name = self.read_string()?;
            let n_dims = self.read_u32()?;
            if n_dims == 0 || n_dims > MAX_GGUF_DIMS {
                return Err(format!(
                    "tensor dim count {n_dims} is outside supported bounds"
                ));
            }
            let mut elements = 1_u64;
            for _ in 0..n_dims {
                let dim = self.read_u64()?;
                if dim == 0 {
                    return Err("tensor dimensions must be greater than zero".to_string());
                }
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| "tensor element count overflowed u64".to_string())?;
            }
            let ggml_type = self.read_u32()?;
            let offset = self.read_u64()?;
            let size = ggml_tensor_payload_size(ggml_type, elements)?;
            tensor_spans.push((offset, size));
        }
        let data_start = align_up_u64(self.cursor, self.alignment)?;
        if data_start > self.file_len {
            return Err("GGUF data section starts past end of file".to_string());
        }
        for (offset, size) in tensor_spans {
            let start = data_start
                .checked_add(offset)
                .ok_or_else(|| "tensor absolute offset overflowed u64".to_string())?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| "tensor end offset overflowed u64".to_string())?;
            if end > self.file_len {
                return Err(format!(
                    "tensor payload range [{start}, {end}) exceeds file size {}",
                    self.file_len
                ));
            }
        }
        match self.package_version.as_deref() {
            Some(OASR_PACKAGE_VERSION_V1) => Ok(()),
            Some(value) => Err(format!(
                "unsupported OpenASR package version '{value}'; expected {OASR_PACKAGE_VERSION_V1}"
            )),
            None => Err(format!(
                "missing required metadata '{OASR_PACKAGE_VERSION_KEY}'"
            )),
        }
    }

    fn read_metadata_entry(&mut self) -> Result<(), String> {
        let key = self.read_string()?;
        let value_type = self.read_u32()?;
        match value_type {
            0 | 1 | 7 => self.skip(1)?,
            2 | 3 => self.skip(2)?,
            4..=6 => {
                if key == "general.alignment" {
                    let value = self.read_u32()?;
                    self.set_alignment(u64::from(value))?;
                } else {
                    self.skip(4)?;
                }
            }
            8 => {
                let value = self.read_string()?;
                if key == OASR_PACKAGE_VERSION_KEY {
                    self.package_version = Some(value);
                }
            }
            9 => self.skip_array_value()?,
            10..=12 => {
                if key == "general.alignment" && value_type == 10 {
                    let value = self.read_u64()?;
                    self.set_alignment(value)?;
                } else {
                    self.skip(8)?;
                }
            }
            other => return Err(format!("unsupported GGUF metadata value type {other}")),
        }
        Ok(())
    }

    fn skip_array_value(&mut self) -> Result<(), String> {
        let item_type = self.read_u32()?;
        let item_count = self.read_u64()?;
        if item_count > MAX_GGUF_ARRAY_VALUES {
            return Err(format!(
                "GGUF array length {item_count} exceeds supported bounds"
            ));
        }
        match item_type {
            0 | 1 | 7 => self.skip(item_count)?,
            2 | 3 => self.skip(item_count.saturating_mul(2))?,
            4..=6 => self.skip(item_count.saturating_mul(4))?,
            8 => {
                for _ in 0..item_count {
                    let _ = self.read_string()?;
                }
            }
            10..=12 => self.skip(item_count.saturating_mul(8))?,
            other => return Err(format!("unsupported GGUF array item type {other}")),
        }
        Ok(())
    }

    fn set_alignment(&mut self, value: u64) -> Result<(), String> {
        if value == 0 || !value.is_power_of_two() || value > 4096 {
            return Err(format!("unsupported GGUF alignment {value}"));
        }
        self.alignment = value;
        Ok(())
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()?;
        if len > MAX_GGUF_STRING_BYTES {
            return Err(format!("GGUF string length {len} exceeds supported bounds"));
        }
        let len_usize = usize::try_from(len).map_err(|_| "string length overflow".to_string())?;
        let mut bytes = vec![0_u8; len_usize];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|source| format!("GGUF string is not UTF-8: {source}"))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut bytes = [0_u8; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut bytes = [0_u8; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), String> {
        self.file
            .read_exact(bytes)
            .map_err(|source| format!("unexpected EOF while scanning GGUF: {source}"))?;
        self.cursor = self.cursor.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn skip(&mut self, bytes: u64) -> Result<(), String> {
        let next = self
            .cursor
            .checked_add(bytes)
            .ok_or_else(|| "GGUF cursor overflowed while skipping".to_string())?;
        if next > self.file_len {
            return Err("GGUF metadata extends past end of file".to_string());
        }
        self.file
            .seek(SeekFrom::Start(next))
            .map_err(|source| format!("could not seek while scanning GGUF: {source}"))?;
        self.cursor = next;
        Ok(())
    }
}

fn ggml_tensor_payload_size(ggml_type: u32, elements: u64) -> Result<u64, String> {
    match ggml_type {
        0 => elements
            .checked_mul(4)
            .ok_or_else(|| "f32 tensor size overflow".to_string()),
        1 | 30 => elements
            .checked_mul(2)
            .ok_or_else(|| "f16/bf16 tensor size overflow".to_string()),
        2 => block_payload(elements, 32, 18),
        8 => block_payload(elements, 32, 34),
        11 => block_payload(elements, 256, 110),
        12 => block_payload(elements, 256, 144),
        13 => block_payload(elements, 256, 176),
        14 => block_payload(elements, 256, 210),
        24 => Ok(elements),
        25 => elements
            .checked_mul(2)
            .ok_or_else(|| "i16 tensor size overflow".to_string()),
        26 => elements
            .checked_mul(4)
            .ok_or_else(|| "i32 tensor size overflow".to_string()),
        27 | 28 => elements
            .checked_mul(8)
            .ok_or_else(|| "i64/f64 tensor size overflow".to_string()),
        other => Err(format!("unsupported GGML tensor type {other}")),
    }
}

fn block_payload(elements: u64, block_elements: u64, block_bytes: u64) -> Result<u64, String> {
    let blocks = elements
        .checked_add(block_elements - 1)
        .ok_or_else(|| "quantized tensor block count overflow".to_string())?
        / block_elements;
    blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| "quantized tensor size overflow".to_string())
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "alignment overflowed u64".to_string())
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(unix)]
fn available_space_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    let result = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let stats = unsafe { stats.assume_init() };
    // Apple's `statvfs` (macOS and iOS share the same struct layout in libc's
    // `unix/bsd/apple` module) reports f_bavail/f_frsize as narrower integer
    // types than the POSIX-typical u64 used elsewhere (e.g. Linux); widen
    // before multiplying so the byte-count math cannot silently truncate.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(u64::from(stats.f_bavail).saturating_mul(stats.f_frsize))
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Some(stats.f_bavail.saturating_mul(stats.f_frsize))
    }
}

#[cfg(windows)]
fn available_space_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    // GetDiskFreeSpaceExW takes a directory path as a NUL-terminated UTF-16
    // string. `path` is the (already-created) storage dir on the target volume.
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut free_to_caller: u64 = 0;
    // SAFETY: lpDirectoryName points at a valid NUL-terminated UTF-16 buffer that
    // outlives the call; lpFreeBytesAvailableToCaller is a valid out-pointer for a
    // ULARGE_INTEGER (u64); the two totals we don't need are passed as null, which
    // the API explicitly permits. A zero return means failure (e.g. the path's
    // volume is unavailable), in which case we report "unknown" like the no-op
    // fallback so the preflight stays permissive rather than blocking a pull.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_to_caller,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some(free_to_caller)
}

#[cfg(not(any(unix, windows)))]
fn available_space_bytes(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
#[path = "pull_tests.rs"]
mod tests;
