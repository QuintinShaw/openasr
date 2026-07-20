use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
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
/// Fixed segment size for concurrent chunked downloads. 64 MiB amortizes
/// per-segment overhead (redirect resolution, TLS/connection setup) over a
/// large body while still giving the default connection count real
/// parallelism on typical multi-hundred-MB to multi-GB model packs (e.g. a
/// 300 MB pack still splits into 5 segments at 4 connections). This is a
/// fixed build-time constant, not an env knob: resumable segment bitmaps
/// (`SegmentedPartialMeta`) are keyed on it, so changing it is a format
/// change, not a runtime tuning parameter (see `PARALLEL_META_FORMAT`).
const DOWNLOAD_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;
/// Default number of concurrent Range connections for chunked downloads.
const DEFAULT_PULL_CONNECTIONS: usize = 4;
/// Hard upper clamp on `OPENASR_PULL_CONNECTIONS` so a misconfigured
/// environment can't open an unbounded number of sockets against a download
/// source.
const MAX_PULL_CONNECTIONS: usize = 8;
/// Environment override for the concurrent chunked-download connection
/// count; clamped to `[1, MAX_PULL_CONNECTIONS]`. Setting it to `1` disables
/// concurrent chunking entirely (the single-stream path is always used when
/// `connections <= 1`), which doubles as the escape hatch for a source that
/// misbehaves under concurrent Range requests.
const PULL_CONNECTIONS_ENV_VAR: &str = "OPENASR_PULL_CONNECTIONS";
/// Bounded per-segment retry attempts before the whole chunked attempt fails
/// and control returns to the outer `download_with_retries` retry loop
/// (which retries the whole attempt, resuming from the on-disk segment
/// bitmap). Deliberately smaller than `DOWNLOAD_MAX_RETRIES`: a segment this
/// persistently broken likely reflects a source-wide problem the outer loop
/// is already positioned to retry or fall back away from.
const SEGMENT_MAX_RETRIES: usize = 3;
/// Discriminator stamped into the segmented-download partial-meta file so a
/// resume never misreads a legacy (pre-chunking) `PartialMeta` -- or a future
/// incompatible format -- as a valid segment bitmap. Bumping the segment size
/// or the bitmap's shape must also bump this string.
const PARALLEL_META_FORMAT: &str = "segmented-v1";
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
    #[error(
        "Concurrent chunk fetch for '{url}' returned a different ETag than the first segment; the download was restarted"
    )]
    EtagChanged { url: String },
    #[error(
        "Downloaded segment [{start}-{end}] size mismatch for '{path}': expected {expected} bytes, got {actual}"
    )]
    SegmentSizeMismatch {
        path: PathBuf,
        start: u64,
        end: u64,
        expected: u64,
        actual: u64,
    },
    #[error(
        "Concurrent chunk fetch for '{url}' returned a Content-Range starting at a different offset than requested"
    )]
    SegmentRangeMismatch { url: String },
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
    /// Segment-completion bitmap for the chunked-download path. Deliberately
    /// a separate file from `partial_meta_path` (rather than a new variant
    /// of the same file) so the existing single-stream `PartialMeta` format
    /// and its resume logic are untouched by this feature: a resume only
    /// ever reads the meta file matching the mode it is about to use, and
    /// `cleanup_partial` removes both unconditionally.
    partial_segments_meta_path: PathBuf,
    installed_meta_path: PathBuf,
    lock_path: PathBuf,
}

#[derive(Debug, Clone)]
struct PullOptions {
    available_space_override: Option<u64>,
    low_speed_timeout: Duration,
    low_speed_min_bytes: u64,
    /// Test-only override for `DOWNLOAD_SEGMENT_BYTES`, so unit tests can
    /// exercise multi-segment concurrent download logic (splitting, resume
    /// bitmap, ETag invalidation, ...) against small in-memory fixtures
    /// instead of needing real multi-hundred-MB bodies. `None` in
    /// production, always -- the real segment size is the fixed constant.
    parallel_segment_bytes_override: Option<u64>,
}

impl PullOptions {
    fn default() -> Self {
        Self {
            available_space_override: None,
            low_speed_timeout: DOWNLOAD_LOW_SPEED_TIMEOUT,
            low_speed_min_bytes: DOWNLOAD_LOW_SPEED_MIN_BYTES,
            parallel_segment_bytes_override: None,
        }
    }
}

/// An HTTP byte-range request bound: open-ended (`bytes=start-`) when `end`
/// is `None` -- used by the single-stream resume path exactly as before --
/// or the inclusive `bytes=start-end` window a concurrent chunk fetch asks
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteRange {
    start: u64,
    end: Option<u64>,
}

impl ByteRange {
    fn from_start(start: u64) -> Self {
        Self { start, end: None }
    }

    fn bounded(start: u64, end_inclusive: u64) -> Self {
        Self {
            start,
            end: Some(end_inclusive),
        }
    }

    fn header_value(self) -> String {
        match self.end {
            Some(end) => format!("bytes={}-{end}", self.start),
            None => format!("bytes={}-", self.start),
        }
    }
}

trait DownloadClient {
    fn open(&mut self, url: &str, range: Option<ByteRange>) -> Result<DownloadResponse, PullError>;
}

struct DownloadResponse {
    status: u16,
    content_length: Option<u64>,
    content_range: Option<String>,
    etag: Option<String>,
    reader: Box<dyn Read>,
}

/// A `DownloadClient` boxed for use across worker threads: each concurrent
/// segment worker owns one, produced fresh by a `ParallelDownloadConfig`
/// factory (see its doc comment for why -- `DownloadClient::open` takes
/// `&mut self`, so a single client instance can't be shared between threads).
type BoxedDownloadClient = Box<dyn DownloadClient + Send>;

/// Concurrency knobs for the chunked-download path, threaded through from
/// `PullModelPackRequest::execute` (production, env-configured) or
/// constructed directly by tests. Absent (`None`) anywhere upstream simply
/// means "never chunk" -- the single-stream path is unconditionally correct
/// and is what every caller falls back to.
struct ParallelDownloadConfig<'a> {
    /// Upper bound on simultaneous Range connections; the actual worker
    /// count is `min(connections, remaining segment count)`.
    connections: usize,
    /// Produces one fresh, independently usable `DownloadClient` per worker
    /// thread. For `HttpDownloadClient` this clones the underlying
    /// `reqwest::blocking::Client`, which is an `Arc`-backed connection pool
    /// designed to be shared across threads -- so cloning it (rather than
    /// building a brand new pool per worker) lets concurrent segment
    /// requests to the same host reuse keep-alive connections.
    factory: &'a dyn Fn() -> Result<BoxedDownloadClient, PullError>,
}

struct DownloadedPartial {
    bytes_done: u64,
    sha256: String,
}

#[derive(Clone)]
struct HttpDownloadClient {
    /// `reqwest::blocking::Client` wraps an `Arc`-backed connection pool, so
    /// cloning is cheap and shares keep-alive connections across threads --
    /// exactly what the chunked-download worker threads want (see
    /// `ParallelDownloadConfig::factory`).
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
        // Each worker thread gets its own clone of `client` (a cheap,
        // `Arc`-backed connection pool -- see `HttpDownloadClient`'s doc
        // comment) rather than an independently constructed client, so
        // concurrent segment requests to the same host can share keep-alive
        // connections instead of each opening a fresh one.
        let worker_client = client.clone();
        let factory = move || -> Result<BoxedDownloadClient, PullError> {
            Ok(Box::new(worker_client.clone()))
        };
        let parallel = ParallelDownloadConfig {
            connections: pull_connections_from_env(),
            factory: &factory,
        };
        pull_model_pack_with_client_sources_and_cancel(
            resolved,
            home,
            &mut client,
            PullOptions::default(),
            sources,
            Some(parallel),
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
    ResolvedCatalogPull::from_model_and_quant(model, quant, quant.pull.clone())
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
    if let Some(quant_dir) = pack.path.parent() {
        fs::remove_dir_all(quant_dir).map_err(|source| PullError::Io {
            path: quant_dir.to_path_buf(),
            source,
        })?;
        // The quant dir just removed lives at <models>/<model_id>/<quant>/. If
        // that was the only installed quant, <models>/<model_id>/ is now an
        // empty leftover; clean it up too. `remove_dir` only ever deletes an
        // *empty* directory, so a sibling quant (or any other file a caller
        // left behind) is never touched -- we just swallow the "not empty"
        // and "already gone" outcomes as expected, non-error states.
        if let Some(model_dir) = quant_dir.parent() {
            match fs::remove_dir(model_dir) {
                Ok(()) => {}
                Err(source)
                    if matches!(
                        source.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::DirectoryNotEmpty
                    ) => {}
                Err(source) => {
                    return Err(PullError::Io {
                        path: model_dir.to_path_buf(),
                        source,
                    });
                }
            }
        }
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
        None,
        progress,
        should_cancel,
        should_pause,
    )
}

/// Test-only entry point that exercises the concurrent chunked-download path
/// (`pull_model_pack_with_client_and_cancel` / the production
/// `PullModelPackRequest::execute` never pass `parallel: None` in production,
/// but tests need to opt in explicitly with a mock client factory).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn pull_model_pack_with_client_parallel<C: DownloadClient>(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    client: &mut C,
    options: PullOptions,
    parallel: ParallelDownloadConfig,
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
        Some(parallel),
        progress,
        should_cancel,
        should_pause,
    )
}

#[allow(clippy::too_many_arguments)]
fn pull_model_pack_with_client_sources_and_cancel<C: DownloadClient>(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    client: &mut C,
    options: PullOptions,
    sources: &[DownloadSource],
    parallel: Option<ParallelDownloadConfig>,
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
            parallel.as_ref(),
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

#[allow(clippy::too_many_arguments)]
fn download_with_retries<C: DownloadClient>(
    target: &PullTarget,
    paths: &PullPaths,
    client: &mut C,
    options: PullOptions,
    parallel: Option<&ParallelDownloadConfig>,
    progress: &mut impl FnMut(PullProgress),
    should_cancel: &impl Fn() -> bool,
    should_pause: &impl Fn() -> bool,
) -> Result<DownloadedPartial, PullError> {
    let segment_bytes = options
        .parallel_segment_bytes_override
        .unwrap_or(DOWNLOAD_SEGMENT_BYTES);
    // Sticky within this call: once a probe (or a mid-download segment
    // response) shows the source ignores Range, don't keep re-probing on
    // every retry -- fall back to the single-stream path for the rest of
    // this pull attempt.
    let mut range_supported = true;
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

        if range_supported
            && let Some(parallel) = parallel
            && parallel_download_eligible(target, parallel.connections, segment_bytes)
        {
            match download_parallel_attempt(
                target,
                paths,
                client,
                parallel,
                segment_bytes,
                &options,
                progress,
                should_cancel,
                should_pause,
            ) {
                Ok(ParallelAttemptOutcome::Completed(downloaded)) => return Ok(downloaded),
                Ok(ParallelAttemptOutcome::RangeNotSupported) => {
                    range_supported = false;
                    cleanup_partial(paths);
                    // Fall through to the single-stream path below for this
                    // same loop iteration -- no wasted attempt/backoff.
                }
                Err(error)
                    if attempt < DOWNLOAD_MAX_RETRIES && is_retryable_download_error(&error) =>
                {
                    attempt += 1;
                    std::thread::sleep(retry_backoff(attempt));
                    continue;
                }
                Err(error) => return Err(error),
            }
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
            .open(
                &target.url,
                (resume_from > 0).then(|| ByteRange::from_start(resume_from)),
            )
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
    if append && !resume_content_range_matches(target.size_bytes, &response, resume_from) {
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
            .map_err(|source| map_download_read_error(&target.url, &paths.partial_path, source))?;
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
        low_speed.observe(
            &target.url,
            target.size_bytes,
            bytes_done,
            read as u64,
            options,
        )?;
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
    let _ = fs::remove_file(&paths.partial_segments_meta_path);
}

/// The env override for the chunked-download connection count, clamped to
/// `[1, MAX_PULL_CONNECTIONS]`. `connections <= 1` makes the download
/// unconditionally single-stream (see `parallel_download_eligible`).
fn pull_connections_from_env() -> usize {
    std::env::var(PULL_CONNECTIONS_ENV_VAR)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|connections| *connections > 0)
        .unwrap_or(DEFAULT_PULL_CONNECTIONS)
        .min(MAX_PULL_CONNECTIONS)
}

/// A pack only benefits from chunking once it splits into at least 2
/// segments -- otherwise a single Range request already reads the whole
/// body, and going concurrent would only add a wasted probe request.
fn parallel_download_eligible(target: &PullTarget, connections: usize, segment_bytes: u64) -> bool {
    connections > 1 && target.size_bytes >= segment_bytes.saturating_mul(2)
}

fn segment_count(size_bytes: u64, segment_bytes: u64) -> usize {
    size_bytes.div_ceil(segment_bytes) as usize
}

/// The inclusive `[start, end]` byte range for segment `index`, clamped to
/// `size_bytes` for the final (possibly short) segment.
fn segment_range(index: usize, size_bytes: u64, segment_bytes: u64) -> (u64, u64) {
    let start = index as u64 * segment_bytes;
    let end = start
        .saturating_add(segment_bytes)
        .saturating_sub(1)
        .min(size_bytes.saturating_sub(1));
    (start, end)
}

/// Per-segment completion bitmap for a chunked download, persisted next to
/// the (preallocated, full-size) `.partial` file as a distinct file from the
/// single-stream `PartialMeta` -- see `PullPaths::partial_segments_meta_path`.
///
/// Deliberately carries **no per-segment hash**: segments can complete out of
/// order across worker threads, so an incrementally-advancing hash cursor
/// (like the single-stream path's inline `Sha256`) would need to buffer or
/// stall on out-of-order bytes to hash them in file order, which defeats the
/// point of concurrency. Instead, integrity is checked once, the same way an
/// already-fully-resumed single-stream download is
/// (`download_with_retries`' `resume_from == target.size_bytes` shortcut):
/// after every segment is marked done, `download_parallel_attempt` rereads
/// the whole file and compares its sha256 against the catalog-pinned digest
/// in `verify_partial_and_install`, exactly like every other pull path. The
/// cost is one extra full-file sequential read, which is fast relative to
/// network transfer (see the PR description for measured overhead).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SegmentedPartialMeta {
    /// Discriminator against the single-stream `PartialMeta` format and
    /// against any future incompatible bitmap shape; see
    /// `PARALLEL_META_FORMAT`.
    format: String,
    model_id: String,
    quant: String,
    filename: String,
    hf_revision: String,
    sha256: String,
    size_bytes: u64,
    /// The fixed segment size this bitmap was built against. A resume whose
    /// current `DOWNLOAD_SEGMENT_BYTES` no longer matches is invalidated
    /// (see `load_segmented_meta`) rather than reinterpreted, since segment
    /// boundaries -- and therefore bitmap indices -- would no longer align.
    segment_bytes: u64,
    /// The ETag every segment's response is checked against
    /// (`fetch_segment_once`); `None` when the source sent no ETag at all,
    /// in which case cross-segment consistency can't be checked and the
    /// final sha256 comparison is the only integrity gate (same gap the
    /// single-stream path already has today).
    etag: Option<String>,
    segments_done: Vec<bool>,
    updated_at_unix_seconds: u64,
}

impl SegmentedPartialMeta {
    fn new(
        target: &PullTarget,
        segment_bytes: u64,
        etag: Option<String>,
        total_segments: usize,
    ) -> Self {
        Self {
            format: PARALLEL_META_FORMAT.to_string(),
            model_id: target.model_id.clone(),
            quant: target.quant.clone(),
            filename: target.filename.clone(),
            hf_revision: target.hf_revision.clone(),
            sha256: target.sha256.clone(),
            size_bytes: target.size_bytes,
            segment_bytes,
            etag,
            segments_done: vec![false; total_segments],
            updated_at_unix_seconds: unix_seconds_now(),
        }
    }

    /// Same content-identity comparison as `PartialMeta::matches_target` --
    /// see its doc comment for why the transport URL is intentionally
    /// excluded (mirrors serve the same bytes under different hosts).
    fn matches_target(&self, target: &PullTarget) -> bool {
        self.format == PARALLEL_META_FORMAT
            && self.model_id == target.model_id
            && self.quant == target.quant
            && self.filename == target.filename
            && self.hf_revision == target.hf_revision
            && self.sha256 == target.sha256
            && self.size_bytes == target.size_bytes
    }

    fn bytes_done(&self, size_bytes: u64, segment_bytes: u64) -> u64 {
        self.segments_done
            .iter()
            .enumerate()
            .filter(|(_, done)| **done)
            .map(|(index, _)| {
                let (start, end) = segment_range(index, size_bytes, segment_bytes);
                end - start + 1
            })
            .sum()
    }
}

fn write_partial_segments_meta(path: &Path, meta: &SegmentedPartialMeta) -> Result<(), PullError> {
    let json = serde_json::to_string_pretty(meta).map_err(|source| PullError::SerializeMeta {
        path: path.to_path_buf(),
        source,
    })?;
    write_json_atomic(path, &format!("{json}\n"))
}

/// Load a usable segment bitmap for `target`/`segment_bytes`, or start fresh.
///
/// Backward/forward compatibility choice: a bitmap is reused only if it
/// parses as the current `SegmentedPartialMeta` shape, matches the target's
/// content identity, was built with the *same* `segment_bytes`, has exactly
/// `total_segments` entries, and the on-disk `.partial` file is already at
/// full size (this path always preallocates to `size_bytes` up front, so
/// anything else means the file predates this feature or is otherwise
/// inconsistent). Anything else -- including a legacy single-stream
/// `.partial` left by a version of OpenASR before chunked downloads existed
/// -- is **not** reinterpreted: its bytes were never segment-aligned, so
/// `cleanup_partial` wipes both partial files and the download restarts from
/// segment 0. This trades a possible redundant re-download of an
/// in-progress legacy partial for never having to guess at a foreign file's
/// layout.
fn load_segmented_meta(
    target: &PullTarget,
    paths: &PullPaths,
    segment_bytes: u64,
    total_segments: usize,
) -> Result<SegmentedPartialMeta, PullError> {
    let partial_len = fs::metadata(&paths.partial_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let parsed = if paths.partial_path.exists() {
        fs::read_to_string(&paths.partial_segments_meta_path)
            .ok()
            .and_then(|contents| serde_json::from_str::<SegmentedPartialMeta>(&contents).ok())
    } else {
        None
    };
    if let Some(meta) = parsed
        && meta.matches_target(target)
        && meta.segment_bytes == segment_bytes
        && meta.segments_done.len() == total_segments
        && partial_len == target.size_bytes
    {
        return Ok(meta);
    }
    cleanup_partial(paths);
    Ok(SegmentedPartialMeta::new(
        target,
        segment_bytes,
        None,
        total_segments,
    ))
}

/// Outcome of one concurrent chunked-download attempt.
enum ParallelAttemptOutcome {
    /// Every segment verified complete; the caller feeds this straight into
    /// `verify_partial_and_install` exactly like the single-stream path.
    Completed(DownloadedPartial),
    /// The probe request got a `200` instead of `206`: the source is
    /// ignoring the `Range` header entirely. Concurrent chunking cannot work
    /// against this URL (writing a `200`'s from-byte-0 body into a
    /// mid-file segment window would corrupt the file), so the caller wipes
    /// the preallocated partial state and falls back to the existing
    /// single-stream path, which already handles a `200` response correctly.
    RangeNotSupported,
}

/// Run one attempt of the concurrent chunked-download path for `target`.
///
/// Sequence: reuse or initialize the segment bitmap: `load_segmented_meta`
/// -> if every segment is already done, skip the network entirely and
/// re-verify the file (same shortcut `download_with_retries` uses for a
/// fully-resumed single-stream download) -> otherwise probe the first
/// missing segment synchronously (confirms Range support and establishes the
/// reference ETag before any concurrency starts) -> spawn up to
/// `parallel.connections` worker threads pulling remaining segment indices
/// off a shared queue, each writing directly into its slice of the
/// preallocated file at the matching offset -> aggregate progress and
/// segment-done events over an `mpsc` channel on this (the caller's) thread,
/// which is also the only thread that polls `should_cancel`/`should_pause`
/// (matching how the single-stream path already confines those predicates to
/// one thread) -> once all workers finish, fsync and reread the whole file's
/// sha256 for the final integrity gate.
#[allow(clippy::too_many_arguments)]
fn download_parallel_attempt<C: DownloadClient + ?Sized>(
    target: &PullTarget,
    paths: &PullPaths,
    probe_client: &mut C,
    parallel: &ParallelDownloadConfig,
    segment_bytes: u64,
    options: &PullOptions,
    progress: &mut impl FnMut(PullProgress),
    should_cancel: &impl Fn() -> bool,
    should_pause: &impl Fn() -> bool,
) -> Result<ParallelAttemptOutcome, PullError> {
    let total_segments = segment_count(target.size_bytes, segment_bytes);
    let mut meta = load_segmented_meta(target, paths, segment_bytes, total_segments)?;

    let missing: Vec<usize> = meta
        .segments_done
        .iter()
        .enumerate()
        .filter(|(_, done)| !**done)
        .map(|(index, _)| index)
        .collect();

    if missing.is_empty() {
        // A prior run already fetched every segment; this attempt only verifies.
        // Signal `Verifying` before the full-file hash for the same reason as the
        // completion paths below.
        progress(PullProgress::Verifying {
            bytes_done: target.size_bytes,
        });
        let (actual_size, sha256) = file_size_and_sha256(&paths.partial_path)?;
        return Ok(ParallelAttemptOutcome::Completed(DownloadedPartial {
            bytes_done: actual_size,
            sha256,
        }));
    }

    let resume_from = meta.bytes_done(target.size_bytes, segment_bytes);
    progress(PullProgress::DownloadStarted {
        bytes_total: target.size_bytes,
        resume_from,
    });

    ensure_available_space(
        &paths.dir,
        reserve_space_bytes((missing.len() as u64).saturating_mul(segment_bytes)),
        options.clone(),
    )?;

    // Probe the first still-missing segment with a real, bounded Range
    // request before committing to concurrency. A single request both (a)
    // confirms the source honors Range (206) rather than ignoring it (200 --
    // handled by the caller falling back to the single-stream path) and (b)
    // establishes the reference ETag every other segment's response is
    // checked against, so this is not a wasted request: its bytes become the
    // probed segment's real data below.
    let probe_index = missing[0];
    let (probe_start, probe_end) = segment_range(probe_index, target.size_bytes, segment_bytes);
    let probe_response = probe_client.open(
        &target.url,
        Some(ByteRange::bounded(probe_start, probe_end)),
    )?;
    if probe_response.status == 200 {
        return Ok(ParallelAttemptOutcome::RangeNotSupported);
    }
    if probe_response.status != 206 {
        return Err(PullError::UnexpectedStatus {
            url: target.url.clone(),
            status: probe_response.status,
        });
    }
    if let Some(reference) = meta.etag.as_deref()
        && let Some(probe_etag) = probe_response.etag.as_deref()
        && reference != probe_etag
    {
        // A prior run's reference ETag no longer matches: the object behind
        // this URL changed since the segment bitmap was last written. Wipe
        // the whole partial state (bytes from two versions of the file can't
        // be selectively resumed) so the retry in `download_with_retries`
        // restarts clean from segment 0 with a fresh reference ETag.
        cleanup_partial(paths);
        return Err(PullError::EtagChanged {
            url: target.url.clone(),
        });
    }
    let reference_etag = meta.etag.clone().or_else(|| probe_response.etag.clone());
    meta.etag = reference_etag.clone();

    {
        // `truncate(false)`: a resumed download's `.partial` already holds
        // previously-written segment bytes at their correct offsets (see
        // `load_segmented_meta`) that must survive this open -- only a fresh
        // download hits `create(true)` for real, and `set_len` below is what
        // establishes the full preallocated size either way.
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&paths.partial_path)
            .map_err(|source| PullError::Io {
                path: paths.partial_path.clone(),
                source,
            })?;
        file.set_len(target.size_bytes)
            .map_err(|source| PullError::Io {
                path: paths.partial_path.clone(),
                source,
            })?;
    }
    write_partial_segments_meta(&paths.partial_segments_meta_path, &meta)?;

    let mut bytes_done = resume_from;
    {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&paths.partial_path)
            .map_err(|source| PullError::Io {
                path: paths.partial_path.clone(),
                source,
            })?;
        let written = write_segment_body(
            &mut file,
            &paths.partial_path,
            probe_start,
            probe_end,
            probe_response.reader,
            |delta| {
                bytes_done = bytes_done.saturating_add(delta);
                progress(PullProgress::Downloading {
                    bytes_done,
                    bytes_total: target.size_bytes,
                });
            },
            &|| should_cancel() || should_pause(),
        )?;
        // The probe segment downloads synchronously on this orchestrating
        // thread, before any worker thread exists to poll the controls, so it
        // must honor cancel/pause itself -- otherwise a cancel issued while the
        // probe is in flight would not take effect until the whole (up to
        // 64 MiB) probe segment finished, stranding the pull in `Downloading`
        // for seconds. `write_segment_body` stops early on the predicate above,
        // leaving a short segment, so these checks must come before the
        // size-mismatch check below (an intentional stop is not a mismatch).
        if should_cancel() {
            cleanup_partial(paths);
            return Err(PullError::Canceled {
                reference: target.pull.clone(),
            });
        }
        if should_pause() {
            // Keep the partial file and segment bitmap (segment not marked
            // done) so a later resume re-probes and refetches this segment
            // cleanly, exactly like a pause caught by the worker loop below.
            return Err(PullError::Paused {
                reference: target.pull.clone(),
            });
        }
        let expected = probe_end - probe_start + 1;
        if written != expected {
            cleanup_partial(paths);
            return Err(PullError::SegmentSizeMismatch {
                path: paths.partial_path.clone(),
                start: probe_start,
                end: probe_end,
                expected,
                actual: written,
            });
        }
    }
    meta.segments_done[probe_index] = true;
    write_partial_segments_meta(&paths.partial_segments_meta_path, &meta)?;

    let remaining: VecDeque<usize> = missing
        .into_iter()
        .filter(|index| *index != probe_index)
        .collect();
    if remaining.is_empty() {
        // Only the probe segment was missing and it just finished: the download
        // is complete. Same rationale as the main-loop completion below -- signal
        // `Verifying` before the full-file hash so the UI leaves the download
        // phase rather than freezing at 100%.
        progress(PullProgress::Verifying {
            bytes_done: target.size_bytes,
        });
        sync_partial_file(&paths.partial_path)?;
        let (actual_size, sha256) = file_size_and_sha256(&paths.partial_path)?;
        let _ = fs::remove_file(&paths.partial_segments_meta_path);
        return Ok(ParallelAttemptOutcome::Completed(DownloadedPartial {
            bytes_done: actual_size,
            sha256,
        }));
    }

    let remaining_count = remaining.len();
    let queue = Arc::new(Mutex::new(remaining));
    let abort = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel::<SegmentEvent>();
    // No lock needed yet: no worker thread exists before the spawn loop
    // below, so `remaining_count` (captured before `remaining` moved into
    // the mutex) is exact, not just a snapshot.
    let worker_count = parallel.connections.min(remaining_count).max(1);
    let size_bytes = target.size_bytes;
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let worker_client = (parallel.factory)()?;
        let worker_queue = queue.clone();
        let worker_abort = abort.clone();
        let worker_sender = sender.clone();
        let worker_path = paths.partial_path.clone();
        let worker_url = target.url.clone();
        let worker_reference_etag = reference_etag.clone();
        handles.push(std::thread::spawn(move || {
            run_segment_worker(
                worker_client,
                worker_queue,
                worker_abort,
                worker_sender,
                worker_path,
                worker_url,
                size_bytes,
                segment_bytes,
                worker_reference_etag,
            );
        }));
    }
    drop(sender);

    let mut first_error: Option<PullError> = None;
    let mut canceled = false;
    let mut paused = false;
    loop {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(SegmentEvent::Progress(delta)) => {
                bytes_done = bytes_done.saturating_add(delta);
                progress(PullProgress::Downloading {
                    bytes_done,
                    bytes_total: target.size_bytes,
                });
            }
            Ok(SegmentEvent::Done(index)) => {
                meta.segments_done[index] = true;
                write_partial_segments_meta(&paths.partial_segments_meta_path, &meta)?;
            }
            Ok(SegmentEvent::Failed(error)) => {
                let is_etag_change = matches!(error, PullError::EtagChanged { .. });
                if first_error.is_none() {
                    first_error = Some(error);
                }
                abort.store(true, Ordering::SeqCst);
                if is_etag_change {
                    // Same "can't selectively resume across an object swap"
                    // reasoning as the probe-time ETag check above.
                    cleanup_partial(paths);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if !canceled && !paused {
            if should_cancel() {
                canceled = true;
                abort.store(true, Ordering::SeqCst);
            } else if should_pause() {
                paused = true;
                abort.store(true, Ordering::SeqCst);
            }
        }
    }
    for handle in handles {
        let _ = handle.join();
    }

    if canceled {
        cleanup_partial(paths);
        return Err(PullError::Canceled {
            reference: target.pull.clone(),
        });
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    if paused {
        return Err(PullError::Paused {
            reference: target.pull.clone(),
        });
    }
    if meta.segments_done.iter().any(|done| !done) {
        // Unreachable in practice: every other exit path above requires the
        // work queue to have drained without cancellation, pause, or error.
        // Kept as a defensive fail-closed check so a future bug in the
        // orchestration above can never mistake a partially-filled file for
        // a complete one.
        return Err(PullError::Io {
            path: paths.partial_path.clone(),
            source: io::Error::other(
                "chunked download loop exited without completing every segment",
            ),
        });
    }

    // Every segment is on disk; the download is done. The integrity gate below
    // rereads and hashes the whole (up to multi-GB) file, which on a large pack
    // takes seconds with no byte progress to report. Signal `Verifying` first so
    // consumers leave the "downloading" phase instead of appearing frozen at
    // 100% while the hash runs (the single-stream path hashes incrementally as
    // it downloads and so never needs this). `verify_partial_and_install` emits
    // `Verifying` again after this returns; the repeat is idempotent.
    progress(PullProgress::Verifying {
        bytes_done: target.size_bytes,
    });
    sync_partial_file(&paths.partial_path)?;
    let (actual_size, sha256) = file_size_and_sha256(&paths.partial_path)?;
    let _ = fs::remove_file(&paths.partial_segments_meta_path);
    Ok(ParallelAttemptOutcome::Completed(DownloadedPartial {
        bytes_done: actual_size,
        sha256,
    }))
}

fn sync_partial_file(path: &Path) -> Result<(), PullError> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|source| PullError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.sync_all().map_err(|source| PullError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Stream `reader` (already capped to the segment's expected length by the
/// caller's use of `.take`, applied inside this function) into `file` at
/// `[start, end_inclusive]`, calling `on_progress` with each chunk's byte
/// count as it's written. Stops early (without error) as soon as `should_abort`
/// returns true, leaving the segment's on-disk bytes incomplete but never
/// marked done by the caller. Checked once per buffer read so a stop request
/// is honored within a single `DOWNLOAD_BUFFER_BYTES` chunk rather than only
/// after the whole (up to 64 MiB) segment finishes. Shared by the synchronous
/// probe-segment write (main thread, whose predicate polls the pull's
/// cancel/pause controls directly) and every worker's per-segment fetch
/// (`fetch_segment_once`, whose predicate reads the shared `abort` flag), so
/// both paths write segments identically and stop identically.
fn write_segment_body(
    file: &mut File,
    path_for_errors: &Path,
    start: u64,
    end_inclusive: u64,
    reader: Box<dyn Read>,
    mut on_progress: impl FnMut(u64),
    should_abort: &dyn Fn() -> bool,
) -> Result<u64, PullError> {
    file.seek(SeekFrom::Start(start))
        .map_err(|source| PullError::Io {
            path: path_for_errors.to_path_buf(),
            source,
        })?;
    let expected_len = end_inclusive - start + 1;
    let mut reader = reader.take(expected_len);
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    let mut written = 0_u64;
    loop {
        if should_abort() {
            break;
        }
        let read = reader.read(&mut buffer).map_err(|source| PullError::Io {
            path: path_for_errors.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|source| PullError::Io {
                path: path_for_errors.to_path_buf(),
                source,
            })?;
        written = written.saturating_add(read as u64);
        on_progress(read as u64);
    }
    Ok(written)
}

/// Events a segment worker thread reports back to the orchestrating thread
/// over the `mpsc` channel. Kept intentionally minimal: only the
/// orchestrating thread touches `should_cancel`/`should_pause`, the segment
/// bitmap, and the `progress` callback, so workers never need anything more
/// than "here is a byte delta" / "this segment index is done" / "this
/// segment failed".
enum SegmentEvent {
    Progress(u64),
    Done(usize),
    Failed(PullError),
}

/// One worker thread's loop: pop segment indices off the shared `queue`
/// until it's empty or `abort` is set, fetching and writing each with
/// `fetch_segment_with_retries`. Never panics on I/O failure -- every error
/// path reports a `SegmentEvent::Failed` and returns instead.
#[allow(clippy::too_many_arguments)]
fn run_segment_worker(
    mut client: BoxedDownloadClient,
    queue: Arc<Mutex<VecDeque<usize>>>,
    abort: Arc<AtomicBool>,
    sender: mpsc::Sender<SegmentEvent>,
    path: PathBuf,
    url: String,
    size_bytes: u64,
    segment_bytes: u64,
    reference_etag: Option<String>,
) {
    let mut file = match OpenOptions::new().write(true).open(&path) {
        Ok(file) => file,
        Err(source) => {
            let _ = sender.send(SegmentEvent::Failed(PullError::Io {
                path: path.clone(),
                source,
            }));
            return;
        }
    };
    loop {
        if abort.load(Ordering::SeqCst) {
            return;
        }
        let index = {
            let mut queue = queue.lock().unwrap();
            match queue.pop_front() {
                Some(index) => index,
                None => return,
            }
        };
        let (start, end) = segment_range(index, size_bytes, segment_bytes);
        match fetch_segment_with_retries(
            client.as_mut(),
            &mut file,
            &path,
            &url,
            start,
            end,
            reference_etag.as_deref(),
            &abort,
            &sender,
        ) {
            Ok(true) => {
                if sender.send(SegmentEvent::Done(index)).is_err() {
                    return;
                }
            }
            Ok(false) => return, // aborted mid-segment; no event, orchestrator already knows
            Err(error) => {
                let _ = sender.send(SegmentEvent::Failed(error));
                return;
            }
        }
    }
}

/// Retry one segment fetch up to `SEGMENT_MAX_RETRIES` times, backing off
/// between attempts exactly like the single-stream path's outer retry loop.
#[allow(clippy::too_many_arguments)]
fn fetch_segment_with_retries(
    client: &mut dyn DownloadClient,
    file: &mut File,
    path: &Path,
    url: &str,
    start: u64,
    end: u64,
    reference_etag: Option<&str>,
    abort: &AtomicBool,
    sender: &mpsc::Sender<SegmentEvent>,
) -> Result<bool, PullError> {
    let mut attempt = 0_usize;
    loop {
        if abort.load(Ordering::SeqCst) {
            return Ok(false);
        }
        match fetch_segment_once(
            client,
            file,
            path,
            url,
            start,
            end,
            reference_etag,
            abort,
            sender,
        ) {
            Ok(outcome) => return Ok(outcome),
            Err(error) if attempt < SEGMENT_MAX_RETRIES && is_retryable_download_error(&error) => {
                attempt += 1;
                std::thread::sleep(retry_backoff(attempt));
            }
            Err(error) => return Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fetch_segment_once(
    client: &mut dyn DownloadClient,
    file: &mut File,
    path: &Path,
    url: &str,
    start: u64,
    end: u64,
    reference_etag: Option<&str>,
    abort: &AtomicBool,
    sender: &mpsc::Sender<SegmentEvent>,
) -> Result<bool, PullError> {
    let response = client.open(url, Some(ByteRange::bounded(start, end)))?;
    if response.status != 206 {
        return Err(PullError::UnexpectedStatus {
            url: url.to_string(),
            status: response.status,
        });
    }
    if let (Some(reference), Some(etag)) = (reference_etag, response.etag.as_deref())
        && reference != etag
    {
        return Err(PullError::EtagChanged {
            url: url.to_string(),
        });
    }
    if let Some(content_range) = response.content_range.as_deref()
        && let Some(parsed) = parse_content_range(content_range)
        && parsed.start != start
    {
        // A 206 whose Content-Range doesn't start where we asked: a
        // misbehaving proxy/CDN, not a normal condition. Treated the same
        // way the single-stream path treats a resume Content-Range mismatch
        // -- restart rather than trust a response at the wrong offset.
        return Err(PullError::SegmentRangeMismatch {
            url: url.to_string(),
        });
    }
    let written = write_segment_body(
        file,
        path,
        start,
        end,
        response.reader,
        |delta| {
            let _ = sender.send(SegmentEvent::Progress(delta));
        },
        &|| abort.load(Ordering::SeqCst),
    )?;
    if abort.load(Ordering::SeqCst) {
        return Ok(false);
    }
    let expected = end - start + 1;
    if written != expected {
        return Err(PullError::SegmentSizeMismatch {
            path: path.to_path_buf(),
            start,
            end,
            expected,
            actual: written,
        });
    }
    Ok(true)
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
    // A resume can switch from the chunked/parallel path (which persists
    // `partial_segments_meta_path`) to this single-stream success path once
    // the remaining bytes drop below the parallel-eligibility threshold; clean
    // it up here too so it cannot outlive the `.partial` file it describes.
    let _ = fs::remove_file(&paths.partial_segments_meta_path);
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
        partial_segments_meta_path: dir.join(format!("{}.partial.segments.json", target.filename)),
        installed_meta_path: dir.join("installed.json"),
        lock_path: dir.join(format!("{}.lock", target.filename)),
        dir,
        final_path,
    })
}

/// The single resolution point every model-pack read/write path in this file
/// funnels through -- see `crate::config::models_dir`'s doc comment for the
/// full env/config/default priority. Loads `config.json` fresh on each call
/// (a small local file) rather than threading a loaded `OpenAsrConfig`
/// through every `home`-taking function in this module's public API; a
/// missing or unreadable config just falls back to the default `<home>/models`
/// root, matching this function's pre-override behavior.
fn models_root(home: &Path) -> PathBuf {
    let config = crate::config::load_config(home).unwrap_or_default();
    crate::config::models_dir(home, &config)
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

/// Shared by the model-pack single-stream resume path and the backend-pack
/// downloader ([`download_backend_file`]): only needs the expected total
/// size, so it takes that directly rather than a whole `&PullTarget`.
fn resume_content_range_matches(
    expected_size_bytes: u64,
    response: &DownloadResponse,
    resume_from: u64,
) -> bool {
    let Some(content_range) = response.content_range.as_deref() else {
        return false;
    };
    let Some(parsed) = parse_content_range(content_range) else {
        return false;
    };
    let Some(expected_end) = expected_size_bytes.checked_sub(1) else {
        return false;
    };
    parsed.start == resume_from
        && parsed.end == expected_end
        && parsed
            .total
            .map(|total| total == expected_size_bytes)
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

/// Bounds how long a single underlying `Read::read` call may hang before
/// it's treated as a stall, filling the gap left by
/// `http::blocking_client_no_redirect` deliberately setting no total request
/// timeout (see its doc comment): without any bound at all, a connection
/// that goes silently dead (no error, no EOF, no more bytes) could hang a
/// `read` call forever, and the app-level `LowSpeedWindow` below can't catch
/// that either -- it only measures elapsed time *between* successful reads,
/// so it never gets a chance to run while a single `read` is stuck.
///
/// Runs the real `Read::read` calls on a dedicated background thread and
/// relays each chunk (or EOF, or the underlying I/O error) over a bounded
/// channel; `Read::read` below waits on that channel with
/// `recv_timeout(stall_timeout)` and turns an elapsed wait into an
/// `io::ErrorKind::TimedOut` error -- the same kind
/// `map_download_read_error` already recognizes and reports as a stall, and
/// the same kind `is_retryable_download_error` already retries.
///
/// A caveat this doesn't (and structurally can't) fully close: if the
/// background thread's own `read` call is the one that's stuck, the thread
/// itself is never reclaimed (its `Sender` just sits there, the channel
/// recv on the foreground side keeps timing out every `stall_timeout` and
/// reports the stall each time, and this download attempt is abandoned by
/// the retry/fallback logic above it -- see `is_retryable_download_error`).
/// The leaked thread is bounded in number by the download concurrency limit
/// (`MAX_PULL_CONNECTIONS` for the chunked path, one for the single-stream
/// path) and is reclaimed by the OS once the underlying connection is
/// eventually torn down, so this trades an unbounded hang for a small,
/// bounded resource cost -- an acceptable trade for a downloader.
struct StallGuardedReader {
    receiver: mpsc::Receiver<io::Result<Vec<u8>>>,
    stall_timeout: Duration,
    /// Bytes already received from the background thread but not yet
    /// returned to the caller, because the caller's `buf` was smaller than
    /// the chunk that arrived. `Read::read` is allowed to return fewer bytes
    /// than `buf.len()`, but must never drop bytes it already has.
    pending: VecDeque<u8>,
    /// Set once EOF, an error, or a disconnect has been observed and
    /// reported, so a `Read` contract-following caller that calls `read`
    /// again afterward gets a clean `Ok(0)` instead of hanging on a closed
    /// channel.
    finished: bool,
}

impl StallGuardedReader {
    fn new(mut reader: Box<dyn Read + Send>, stall_timeout: Duration) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<io::Result<Vec<u8>>>(1);
        std::thread::spawn(move || {
            let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
            loop {
                let (message, stop) = match reader.read(&mut buffer) {
                    Ok(0) => (Ok(Vec::new()), true),
                    Ok(read) => (Ok(buffer[..read].to_vec()), false),
                    Err(error) => (Err(error), true),
                };
                if sender.send(message).is_err() || stop {
                    // Either the foreground gave up (dropped the receiver --
                    // this attempt was abandoned) or this was the last
                    // message (EOF/error); either way, stop reading.
                    return;
                }
            }
        });
        Self {
            receiver,
            stall_timeout,
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl Read for StallGuardedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.pending.is_empty() {
            let len = self.pending.len().min(buf.len());
            for slot in &mut buf[..len] {
                *slot = self.pending.pop_front().expect("checked non-empty above");
            }
            return Ok(len);
        }
        if self.finished {
            return Ok(0);
        }
        match self.receiver.recv_timeout(self.stall_timeout) {
            Ok(Ok(chunk)) if chunk.is_empty() => {
                self.finished = true;
                Ok(0)
            }
            Ok(Ok(chunk)) => {
                let len = chunk.len().min(buf.len());
                buf[..len].copy_from_slice(&chunk[..len]);
                self.pending.extend(&chunk[len..]);
                Ok(len)
            }
            Ok(Err(error)) => {
                self.finished = true;
                Err(error)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "no data received from the download source within {stall_timeout:?}",
                    stall_timeout = self.stall_timeout
                ),
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.finished = true;
                Ok(0)
            }
        }
    }
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

    /// Shared by the model-pack and backend-pack downloaders; only needs the
    /// URL (for the error message) and the expected total size, not a whole
    /// `&PullTarget`.
    fn observe(
        &mut self,
        url: &str,
        size_bytes: u64,
        bytes_done: u64,
        bytes_read: u64,
        options: &PullOptions,
    ) -> Result<(), PullError> {
        if options.low_speed_min_bytes == 0 || bytes_done >= size_bytes {
            return Ok(());
        }
        self.bytes_read = self.bytes_read.saturating_add(bytes_read);
        let elapsed = self.started_at.elapsed();
        if elapsed < options.low_speed_timeout {
            return Ok(());
        }
        if self.bytes_read < options.low_speed_min_bytes {
            return Err(PullError::Http {
                url: url.to_string(),
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

/// Shared by the model-pack and backend-pack ([`download_backend_file`])
/// stream-to-file loops: turns the stall-guard's `TimedOut` read error into
/// the retryable `PullError::Http` variant `is_retryable_download_error`
/// recognizes, everything else into a plain `Io` error. Takes the URL
/// directly rather than a whole `&PullTarget` so both callers can share it.
fn map_download_read_error(url: &str, path: &Path, source: io::Error) -> PullError {
    if source.kind() == io::ErrorKind::TimedOut {
        return PullError::Http {
            url: url.to_string(),
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
        let client = http::blocking_client_no_redirect(HTTP_CONNECT_TIMEOUT).map_err(|source| {
            PullError::Http {
                url: "<client>".to_string(),
                message: http::error_message(&source),
            }
        })?;
        Ok(Self {
            client,
            hf_token: hf_token_from_env(),
        })
    }
}

/// Env var names carrying an optional Hugging Face access token, in precedence
/// order: the OpenASR-specific var the desktop app injects at daemon launch first,
/// then the two standard HF client vars so a token already in the user's
/// environment is picked up. First non-empty wins.
const HF_TOKEN_ENV_VARS: &[&str] = &["OPENASR_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// Optional Hugging Face access token from the environment (see
/// [`HF_TOKEN_ENV_VARS`]), trimmed; `None` when unset or empty. The desktop app
/// injects it so model pulls can authenticate under shared-IP rate limits. Never
/// read on any fail-closed local path, and only ever attached to a direct
/// huggingface.co request (see [`hf_token_allowed_for_host`]): an unset var simply
/// means anonymous downloads, and the worker/mirror sources are always anonymous.
fn hf_token_from_env() -> Option<String> {
    HF_TOKEN_ENV_VARS
        .iter()
        .find_map(|var| normalize_hf_token(std::env::var(var).ok()))
}

/// Trim a raw token value and drop it if empty. Extracted so the selection logic is
/// unit-testable without mutating process-global environment variables.
fn normalize_hf_token(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Whether the optional HF bearer token may be attached to a request to `host`.
/// Restricted to `huggingface.co` (the direct source) so the credential never
/// reaches a CDN, mirror, the weights.openasr.org worker, or an
/// attacker-controlled redirect target.
fn hf_token_allowed_for_host(host: Option<&str>) -> bool {
    host == Some("huggingface.co")
}

impl DownloadClient for HttpDownloadClient {
    fn open(&mut self, url: &str, range: Option<ByteRange>) -> Result<DownloadResponse, PullError> {
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
            if let Some(range) = range {
                request = request.header(reqwest::header::RANGE, range.header_value());
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
                // `blocking_client_no_redirect` deliberately sets no total
                // request timeout (see its doc comment), so a single `read`
                // on this response body could otherwise hang indefinitely on
                // a connection that goes silently dead without an error or
                // EOF. `StallGuardedReader` bounds that per-read wait.
                reader: Box::new(StallGuardedReader::new(
                    Box::new(response),
                    HTTP_STALL_TIMEOUT,
                )),
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
        | PullError::SizeMismatch { .. }
        | PullError::EtagChanged { .. }
        | PullError::SegmentSizeMismatch { .. }
        | PullError::SegmentRangeMismatch { .. } => true,
        // Only 5xx here: a 4xx from the currently open source is not a
        // transient fault of *this* request, so retrying the same source
        // again would just repeat it (see `is_source_fallback_error`, which
        // moves to the *next* source instead for the 403/404 case).
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
        | PullError::RuntimeValidation { .. }
        | PullError::EtagChanged { .. }
        | PullError::SegmentSizeMismatch { .. }
        | PullError::SegmentRangeMismatch { .. } => true,
        // 5xx: this source's own infra failed -- try the next one. 403/404:
        // this source does not have (or will not serve) the requested object,
        // which is a per-source availability gap, not a global failure -- e.g.
        // weights.openasr.org only proxies the `OpenASR/*` org and 404s for
        // anything outside it, so the chain must be able to fall through to
        // hf-mirror/hf instead of hard-failing the whole pull. 400/401 are
        // deliberately NOT included: 400 is a malformed request that would
        // recur identically against every source in the chain, and 401 means
        // the underlying (possibly gated) resource requires credentials this
        // pull does not have -- switching mirrors cannot supply the missing
        // bearer token, so falling through would just fail three times instead
        // of once.
        PullError::UnexpectedStatus { status, .. } => {
            *status >= 500 || *status == 403 || *status == 404
        }
        _ => false,
    }
}

fn mirror_endpoint_for_current_url(current: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(current).ok()?;
    let host = parsed.host_str()?;
    // Sources whose downstream CDN 302 must be followed VERBATIM (no host swap):
    // - huggingface.co / modelscope.cn: the direct sources, whose Xet redirect is
    //   already on a reachable host.
    // - weights.openasr.org: the first-party worker transparently passes the 302
    //   through to Xet (`us.aws.cdn.hf.co`), which the worker does NOT re-serve;
    //   rewriting the redirect back onto the worker would 404. Behaves exactly like
    //   the direct Hf source here.
    if matches!(
        host,
        "huggingface.co" | "modelscope.cn" | "www.modelscope.cn" | "weights.openasr.org"
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

/// Fail-closed, not a panic: `filter_forward_compatible_catalog` already drops
/// a backend pack whose `vendor` this build does not recognize before it can
/// reach pull resolution, but this is trust-boundary code parsing signed-yet-
/// external data, so an `Unknown` reaching here (a filtering bug, or a caller
/// that built a `ResolvedCatalogBackendPull` some other way) must return a
/// typed error rather than panic or silently guess a directory.
fn backend_vendor_dirname(vendor: CatalogBackendVendor) -> Result<&'static str, PullError> {
    Ok(match vendor {
        CatalogBackendVendor::Cpu => "cpu",
        CatalogBackendVendor::Vulkan => "vulkan",
        CatalogBackendVendor::Hip => "hip",
        CatalogBackendVendor::Cuda => "cuda",
        CatalogBackendVendor::Unknown => {
            return Err(PullError::InvalidTarget {
                field: "backend.vendor",
                reason: "backend pack vendor is not recognized by this build".to_string(),
            });
        }
    })
}

/// See [`backend_vendor_dirname`]'s doc comment: same fail-closed contract for
/// an unrecognized backend file `role`.
fn backend_file_format(role: CatalogBackendFileRole) -> Result<BackendFileFormat, PullError> {
    Ok(match role {
        CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Runtime => {
            BackendFileFormat::NativeLibrary
        }
        CatalogBackendFileRole::Archive => BackendFileFormat::ZipArchive,
        CatalogBackendFileRole::Unknown => {
            return Err(PullError::InvalidTarget {
                field: "backend.files[].role",
                reason: "backend pack file role is not recognized by this build".to_string(),
            });
        }
    })
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
    let vendor = backend_vendor_dirname(resolved.vendor)?;
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
        preflight_backend_file(&dest, backend_file_format(file.role)?)?;
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

/// Download a single backend-pack file (plugin binary or archive) to `dest`,
/// streamed to a `.partial` file and sha256-verified before the atomic
/// rename -- the backend-pack analogue of the model-pack single-stream path
/// (`download_with_retries` / `download_response`), sharing its retry
/// backoff, resume, and stall/low-speed detection rather than a second,
/// weaker re-derivation of that machinery (previously: any dropped
/// connection failed the whole ~150 MB backend pack permanently). Every
/// `DownloadClient::open` response already comes back wrapped in
/// `StallGuardedReader`, so `map_download_read_error` promotes a stalled
/// read into the retryable `PullError::Http` variant here exactly as it does
/// for model packs.
///
/// Resume spans retries within this one call only (an in-memory
/// `expected_etag`, not a persisted meta file like the model-pack path
/// uses): a `.partial` left over from a previous process invocation has no
/// such provenance and is discarded up front, so only an in-process retry
/// after a transient failure resumes from a `.partial` offset.
fn download_backend_file<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    dest: &Path,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    let partial = dest.with_extension("partial");
    let _ = fs::remove_file(&partial);

    let mut attempt = 0_usize;
    let mut expected_etag: Option<String> = None;
    loop {
        match download_backend_file_attempt(
            client,
            file,
            dest,
            &partial,
            &mut expected_etag,
            progress,
        ) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < DOWNLOAD_MAX_RETRIES && is_retryable_download_error(&error) => {
                attempt += 1;
                std::thread::sleep(retry_backoff(attempt));
            }
            Err(error) => return Err(error),
        }
    }
}

fn prepare_backend_partial_for_resume(partial: &Path, size_bytes: u64) -> Result<u64, PullError> {
    if !partial.exists() {
        return Ok(0);
    }
    let partial_len = fs::metadata(partial)
        .map_err(|source| PullError::Io {
            path: partial.to_path_buf(),
            source,
        })?
        .len();
    if partial_len > size_bytes {
        let _ = fs::remove_file(partial);
        return Ok(0);
    }
    Ok(partial_len)
}

fn download_backend_file_attempt<C: DownloadClient>(
    client: &mut C,
    file: &CatalogBackendFile,
    dest: &Path,
    partial: &Path,
    expected_etag: &mut Option<String>,
    progress: &mut impl FnMut(PullProgress),
) -> Result<(), PullError> {
    let resume_from = prepare_backend_partial_for_resume(partial, file.size_bytes)?;
    let response = client.open(
        &file.url,
        (resume_from > 0).then(|| ByteRange::from_start(resume_from)),
    )?;

    if resume_from > 0
        && let (Some(expected), Some(actual)) = (expected_etag.as_deref(), response.etag.as_deref())
        && expected != actual
    {
        let _ = fs::remove_file(partial);
        return Err(PullError::RestartedPartial {
            url: file.url.clone(),
        });
    }
    if expected_etag.is_none() {
        *expected_etag = response.etag.clone();
    }

    let append = match (resume_from, response.status) {
        (0, 200 | 206) => false,
        (_, 206) => true,
        (_, 200) => false,
        (_, status) => {
            return Err(PullError::UnexpectedStatus {
                url: file.url.clone(),
                status,
            });
        }
    };
    if append && !resume_content_range_matches(file.size_bytes, &response, resume_from) {
        let _ = fs::remove_file(partial);
        return Err(PullError::RestartedPartial {
            url: file.url.clone(),
        });
    }
    let actual_resume = if append { resume_from } else { 0 };
    if resume_from > 0 && !append {
        let _ = fs::remove_file(partial);
    }
    if let Some(content_length) = response.content_length {
        let expected_body = file.size_bytes.saturating_sub(actual_resume);
        if content_length != expected_body {
            let _ = fs::remove_file(partial);
            return Err(PullError::SizeMismatch {
                path: dest.to_path_buf(),
                expected: expected_body,
                actual: content_length,
            });
        }
    }

    let mut hasher = Sha256::new();
    if append {
        hash_existing_partial(partial, &mut hasher)?;
    }
    let mut out = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(partial)
        .map_err(|source| PullError::Io {
            path: partial.to_path_buf(),
            source,
        })?;
    let mut reader = response.reader;
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    let mut bytes_done = actual_resume;
    let mut low_speed = LowSpeedWindow::new();
    let low_speed_options = PullOptions::default();
    progress(PullProgress::DownloadStarted {
        bytes_total: file.size_bytes,
        resume_from: actual_resume,
    });
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| map_download_read_error(&file.url, partial, source))?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])
            .map_err(|source| PullError::Io {
                path: partial.to_path_buf(),
                source,
            })?;
        hasher.update(&buffer[..read]);
        bytes_done = bytes_done.saturating_add(read as u64);
        progress(PullProgress::Downloading {
            bytes_done,
            bytes_total: file.size_bytes,
        });
        low_speed.observe(
            &file.url,
            file.size_bytes,
            bytes_done,
            read as u64,
            &low_speed_options,
        )?;
    }
    out.sync_all().map_err(|source| PullError::Io {
        path: partial.to_path_buf(),
        source,
    })?;
    drop(out);
    let actual = format!("{:x}", hasher.finalize());
    if actual != file.sha256 {
        let _ = fs::remove_file(partial);
        return Err(PullError::ShaMismatch {
            path: dest.to_path_buf(),
            expected: file.sha256.clone(),
            actual,
        });
    }
    fs::rename(partial, dest).map_err(|source| PullError::Io {
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
